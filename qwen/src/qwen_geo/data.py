from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qwen_geo.config import DatasetConfig, ExperimentConfig
from qwen_geo.geo import encode_geohash
from qwen_geo.prompts import build_training_messages


class SimpleDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def select(self, indices: range | list[int]) -> "SimpleDataset":
        return SimpleDataset([self.rows[index] for index in indices])


def _slugify_dataset_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "__")


def _pick_column(candidates: list[str], columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def detect_columns(dataset: Any, config: DatasetConfig) -> dict[str, str | None]:
    columns = list(dataset.column_names)
    image_column = config.image_column or _pick_column(["image", "img", "photo"], columns)
    latitude_column = config.latitude_column or _pick_column(["latitude", "lat", "y"], columns)
    longitude_column = config.longitude_column or _pick_column(["longitude", "lon", "lng", "long", "x"], columns)
    country_column = config.country_column or _pick_column(["country_code", "country", "cc"], columns)
    id_column = config.id_column or _pick_column(["id", "image_id", "photo_id", "uuid"], columns)

    if not image_column or not latitude_column or not longitude_column:
        raise ValueError(
            "Could not infer required columns. "
            f"Available columns: {columns}. Set image_column, latitude_column, and longitude_column explicitly."
        )

    return {
        "image": image_column,
        "latitude": latitude_column,
        "longitude": longitude_column,
        "country": country_column,
        "id": id_column,
    }


def _maybe_limit(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(limit))
    return SimpleDataset([dataset[index] for index in range(limit)])


def _normalize_country_codes(raw_values: list[Any], coords: list[tuple[float, float]]) -> list[str]:
    import reverse_geocoder as rg

    lookups = rg.search(coords)
    normalized: list[str] = []
    for value, fallback in zip(raw_values, lookups):
        if isinstance(value, str) and value.strip():
            stripped = value.strip().upper()
            normalized.append(stripped[:2] if len(stripped) >= 2 else stripped)
        else:
            normalized.append(fallback.get("cc", "UNK").upper())
    return normalized


def _normalize_split(dataset: Any, columns: dict[str, str | None]) -> Any:
    def transform(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        latitudes = [float(value) for value in batch[columns["latitude"]]]
        longitudes = [float(value) for value in batch[columns["longitude"]]]
        coords = list(zip(latitudes, longitudes))
        raw_countries = batch[columns["country"]] if columns["country"] else [None] * len(latitudes)
        countries = _normalize_country_codes(raw_countries, coords)

        if columns["id"]:
            ids = [str(value) for value in batch[columns["id"]]]
        else:
            ids = [f"sample-{index}" for index in indices]

        return {
            "id": ids,
            "latitude": latitudes,
            "longitude": longitudes,
            "country_code": countries,
            "geohash_3": [encode_geohash(lat, lon, 3) for lat, lon in coords],
            "geohash_4": [encode_geohash(lat, lon, 4) for lat, lon in coords],
            "geohash_5": [encode_geohash(lat, lon, 5) for lat, lon in coords],
        }

    normalized = dataset.map(transform, batched=True, with_indices=True, desc="Normalizing geolocation metadata")
    rename_targets = {
        columns["image"]: "image",
        columns["latitude"]: "latitude",
        columns["longitude"]: "longitude",
    }
    for source, target in rename_targets.items():
        if source != target:
            normalized = normalized.rename_column(source, target)

    keep_columns = ["id", "image", "latitude", "longitude", "country_code", "geohash_3", "geohash_4", "geohash_5"]
    return normalized.select_columns(keep_columns)


def _load_raw_bundle(config: DatasetConfig) -> Any:
    from datasets import DatasetDict, load_dataset

    kwargs: dict[str, Any] = {}
    if config.config_name:
        kwargs["name"] = config.config_name

    bundle = load_dataset(config.name, **kwargs)
    if isinstance(bundle, DatasetDict):
        return bundle
    return DatasetDict({config.train_split: bundle})


def _split_bundle(bundle: Any, config: DatasetConfig) -> Any:
    from datasets import DatasetDict

    train_split_name = config.train_split if config.train_split in bundle else next(iter(bundle.keys()))
    train_dataset = _maybe_limit(bundle[train_split_name], config.sample_limit)

    if config.eval_split and config.eval_split in bundle:
        eval_dataset = bundle[config.eval_split]
        test_dataset = bundle[config.test_split] if config.test_split and config.test_split in bundle else eval_dataset
        return DatasetDict(train=train_dataset, eval=eval_dataset, test=test_dataset)

    if config.test_split and config.test_split in bundle:
        test_dataset = bundle[config.test_split]
        split = train_dataset.train_test_split(test_size=config.eval_fraction, seed=config.split_seed)
        return DatasetDict(train=split["train"], eval=split["test"], test=test_dataset)

    held_out_fraction = config.eval_fraction + config.test_fraction
    split = train_dataset.train_test_split(test_size=held_out_fraction, seed=config.split_seed)
    held_out = split["test"]
    if config.test_fraction > 0:
        ratio = config.test_fraction / max(held_out_fraction, 1e-6)
        held_split = held_out.train_test_split(test_size=ratio, seed=config.split_seed + 1)
        eval_dataset = held_split["train"]
        test_dataset = held_split["test"]
    else:
        eval_dataset = held_out
        test_dataset = held_out
    return DatasetDict(train=split["train"], eval=eval_dataset, test=test_dataset)


def _make_smoke_record(
    *,
    split: str,
    region_index: int,
    sample_index: int,
    country_code: str,
    latitude: float,
    longitude: float,
    rgb: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "id": f"{split}-{country_code.lower()}-{sample_index:03d}",
        "image": {"rgb": [rgb[0], rgb[1], rgb[2]], "region_index": region_index},
        "latitude": latitude,
        "longitude": longitude,
        "country_code": country_code,
        "geohash_3": encode_geohash(latitude, longitude, 3),
        "geohash_4": encode_geohash(latitude, longitude, 4),
        "geohash_5": encode_geohash(latitude, longitude, 5),
    }


def _build_smoke_bundle(seed: int) -> dict[str, SimpleDataset]:
    rng = random.Random(seed)
    regions = [
        {"country_code": "US", "latitude": 40.7128, "longitude": -74.0060, "rgb": (225, 60, 60)},
        {"country_code": "BR", "latitude": -22.9068, "longitude": -43.1729, "rgb": (60, 200, 70)},
        {"country_code": "FR", "latitude": 48.8566, "longitude": 2.3522, "rgb": (70, 90, 225)},
        {"country_code": "JP", "latitude": 35.6764, "longitude": 139.6500, "rgb": (225, 215, 70)},
    ]
    split_sizes = {"train": 12, "eval": 4, "test": 4}
    bundle: dict[str, SimpleDataset] = {}

    for split_name, per_region in split_sizes.items():
        rows: list[dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            for sample_index in range(per_region):
                latitude = region["latitude"] + rng.uniform(-0.18, 0.18)
                longitude = region["longitude"] + rng.uniform(-0.18, 0.18)
                rgb = tuple(
                    max(0, min(255, base + rng.randint(-12, 12)))
                    for base in region["rgb"]
                )
                rows.append(
                    _make_smoke_record(
                        split=split_name,
                        region_index=region_index,
                        sample_index=sample_index,
                        country_code=region["country_code"],
                        latitude=latitude,
                        longitude=longitude,
                        rgb=rgb,
                    )
                )
        bundle[split_name] = SimpleDataset(rows)
    return bundle


def _write_simple_bundle(bundle: dict[str, SimpleDataset], prepared_path: Path, config: DatasetConfig) -> None:
    if prepared_path.exists():
        shutil.rmtree(prepared_path)
    prepared_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "simple_dataset",
        "dataset": config.name,
        "config": asdict(config),
        "sizes": {split_name: len(dataset) for split_name, dataset in bundle.items()},
    }
    (prepared_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for split_name, dataset in bundle.items():
        (prepared_path / f"{split_name}.json").write_text(
            json.dumps(dataset.rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def prepare_dataset(config: ExperimentConfig, prepared_root: Path, *, force: bool = False) -> Path:
    prepared_root.mkdir(parents=True, exist_ok=True)
    slug = _slugify_dataset_name(config.dataset.name)
    prepared_path = prepared_root / slug
    manifest_path = prepared_path / "manifest.json"

    if prepared_path.exists() and manifest_path.exists() and not force:
        return prepared_path

    if config.dataset.name.startswith("synthetic://"):
        bundle = _build_smoke_bundle(config.project.seed)
        _write_simple_bundle(bundle, prepared_path, config.dataset)
        return prepared_path

    from datasets import DatasetDict

    bundle = _load_raw_bundle(config.dataset)
    split_bundle = _split_bundle(bundle, config.dataset)
    columns = detect_columns(split_bundle["train"], config.dataset)

    normalized = DatasetDict(
        train=_normalize_split(split_bundle["train"], columns),
        eval=_normalize_split(split_bundle["eval"], columns),
        test=_normalize_split(split_bundle["test"], columns),
    )

    if prepared_path.exists():
        shutil.rmtree(prepared_path)
    prepared_path.mkdir(parents=True, exist_ok=True)
    normalized.save_to_disk(str(prepared_path))
    manifest_path.write_text(
        json.dumps(
            {
                "format": "hf_dataset",
                "dataset": config.dataset.name,
                "config": asdict(config.dataset),
                "columns": columns,
                "sizes": {split_name: len(split_dataset) for split_name, split_dataset in normalized.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared_path


def load_prepared_dataset(prepared_path: Path) -> dict[str, Any]:
    manifest_path = prepared_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") == "simple_dataset":
            return {
                split_name: SimpleDataset(
                    json.loads((prepared_path / f"{split_name}.json").read_text(encoding="utf-8"))
                )
                for split_name in ("train", "eval", "test")
            }

    from datasets import load_from_disk

    return load_from_disk(str(prepared_path))


def limit_split(dataset: Any, limit: int | None) -> Any:
    return _maybe_limit(dataset, limit)


class GeoChatDataset:
    def __init__(
        self,
        dataset: Any,
        *,
        prompt_style: str,
        label_style: str,
        geohash_precision: int,
    ) -> None:
        self.dataset = dataset
        self.prompt_style = prompt_style
        self.label_style = label_style
        self.geohash_precision = geohash_precision

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.dataset[index]
        return {
            "messages": build_training_messages(
                record,
                prompt_style=self.prompt_style,
                label_style=self.label_style,
                geohash_precision=self.geohash_precision,
            )
        }
