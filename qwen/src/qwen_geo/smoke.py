from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_geo.config import ExperimentConfig, dump_json
from qwen_geo.data import limit_split, load_prepared_dataset
from qwen_geo.evaluation import compute_metrics, parse_prediction, prediction_to_point
from qwen_geo.geo import almost_same_cell, decode_geohash, haversine_km


@dataclass
class SmokeModel:
    exemplars: list[dict[str, Any]]
    config: dict[str, Any]


def _rgb_feature(record: dict[str, Any]) -> tuple[float, float, float]:
    rgb = record["image"]["rgb"]
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def train_smoke_experiment(config: ExperimentConfig, prepared_path: Path, run_dir: Path) -> tuple[SmokeModel, None, dict[str, Any]]:
    started = time.time()
    dataset_dict = load_prepared_dataset(prepared_path)
    train_split = limit_split(dataset_dict["train"], config.dataset.train_limit)

    exemplars = []
    for index in range(len(train_split)):
        record = train_split[index]
        exemplars.append(
            {
                "id": record["id"],
                "feature": _rgb_feature(record),
                "country_code": record["country_code"],
                "latitude": record["latitude"],
                "longitude": record["longitude"],
                "geohash_3": record["geohash_3"],
                "geohash_4": record["geohash_4"],
                "geohash_5": record["geohash_5"],
            }
        )

    model = SmokeModel(exemplars=exemplars, config=config.to_dict())
    artifact = {
        "backend": "smoke",
        "num_exemplars": len(exemplars),
        "label_style": config.label_style,
        "prompt_style": config.prompt_style,
        "geohash_precision": config.dataset.geohash_precision,
    }
    dump_json(run_dir / "adapter" / "smoke_model.json", artifact)

    metrics = {
        "backend": "smoke",
        "train_seconds": time.time() - started,
        "train_examples": len(train_split),
        "eval_examples": min(len(dataset_dict["eval"]), config.evaluation.eval_examples),
    }
    dump_json(run_dir / "train_metrics.json", metrics)
    return model, None, metrics


def _predict_record(model: SmokeModel, record: dict[str, Any], config: ExperimentConfig) -> dict[str, Any]:
    feature = _rgb_feature(record)
    best = min(model.exemplars, key=lambda exemplar: _distance(feature, exemplar["feature"]))
    precision = max(3, min(5, config.dataset.geohash_precision))
    geohash = best[f"geohash_{precision}"]

    if config.label_style == "coords_json":
        payload: dict[str, Any] = {
            "latitude": round(best["latitude"], 4),
            "longitude": round(best["longitude"], 4),
        }
    elif config.label_style == "cell_json":
        payload = {"country_code": best["country_code"], "geohash": geohash}
    else:
        payload = {
            "country_code": best["country_code"],
            "geohash": geohash,
            "latitude": round(best["latitude"], 4),
            "longitude": round(best["longitude"], 4),
        }

    if config.prompt_style == "strict_json":
        return payload
    if config.prompt_style == "geohash_first":
        ordered = {}
        if "geohash" in payload:
            ordered["geohash"] = payload["geohash"]
        for key, value in payload.items():
            if key != "geohash":
                ordered[key] = value
        return ordered
    ordered = {}
    for key in ("latitude", "longitude", "country_code", "geohash"):
        if key in payload:
            ordered[key] = payload[key]
    return ordered


def _prediction_text(prediction: dict[str, Any], prompt_style: str) -> str:
    text = json.dumps(prediction, separators=(",", ":"))
    if prompt_style == "strict_json":
        return text
    if prompt_style == "geohash_first":
        return f"Best guess: {text}"
    return f"Coordinates estimate {text}"


def evaluate_smoke_model(model: SmokeModel, eval_dataset: Any, config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    predictions_path = run_dir / "predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as sink:
        for index in range(min(len(eval_dataset), config.evaluation.eval_examples)):
            record = eval_dataset[index]
            prediction = _predict_record(model, record, config)
            text = _prediction_text(prediction, config.prompt_style)
            parsed = parse_prediction(text)
            point = prediction_to_point(parsed)
            if point is None and parsed.get("geohash"):
                center = decode_geohash(parsed["geohash"]).center
                point = (center.latitude, center.longitude)
            distance = None
            if point is not None:
                distance = haversine_km(record["latitude"], record["longitude"], point[0], point[1])

            row = {
                "id": record["id"],
                "prediction_text": text,
                "parsed_ok": bool(parsed),
                "country_match": parsed.get("country_code") == record.get("country_code") if parsed else False,
                "geohash3_match": almost_same_cell(parsed.get("geohash"), record.get("geohash_3"), 3),
                "geohash4_match": almost_same_cell(parsed.get("geohash"), record.get("geohash_4"), 4),
                "distance_km": distance,
                "target_country_code": record.get("country_code"),
                "target_geohash_4": record.get("geohash_4"),
                "target_latitude": record["latitude"],
                "target_longitude": record["longitude"],
            }
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)

    metrics = compute_metrics(rows, config.evaluation.thresholds_km)
    metrics["backend"] = "smoke"
    dump_json(run_dir / "metrics.json", metrics)
    return metrics
