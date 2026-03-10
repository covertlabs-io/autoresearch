from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class ProjectConfig:
    name: str = "qwen-geo"
    output_root: str = "runs"
    tracker_db: str = "runs/experiments.sqlite"
    prepared_data_dir: str = "data/prepared"
    reports_dir: str = "runs/reports"
    seed: int = 17


@dataclass
class DatasetConfig:
    name: str = "marcelomoreno26/geoguessr"
    config_name: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    test_split: str | None = None
    image_column: str | None = None
    latitude_column: str | None = None
    longitude_column: str | None = None
    country_column: str | None = None
    id_column: str | None = None
    sample_limit: int | None = None
    train_limit: int = 4096
    eval_limit: int = 256
    test_limit: int = 256
    split_seed: int = 17
    eval_fraction: float = 0.08
    test_fraction: float = 0.02
    geohash_precision: int = 4


@dataclass
class ModelConfig:
    backend: str = "unsloth"
    name: str = "unsloth/Qwen3.5-2B-Base"
    max_seq_length: int = 1536
    load_in_4bit: bool = False
    use_gradient_checkpointing: bool = True
    max_image_size: int = 672
    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: str = "none"
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_steps: int = 200
    num_train_epochs: float = 1.0
    time_budget_minutes: int = 20
    logging_steps: int = 10
    save_steps: int = 200
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    num_proc: int = 1
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2


@dataclass
class EvaluationConfig:
    eval_examples: int = 128
    max_new_tokens: int = 80
    temperature: float = 0.0
    top_p: float = 1.0
    thresholds_km: list[int] = field(default_factory=lambda: [1, 25, 200, 750])


@dataclass
class AutonomyConfig:
    hours: float = 12.0
    max_trials: int = 48
    min_score_delta: float = 0.002
    promote_threshold: float = 0.01
    planner_seed: int = 17


@dataclass
class SearchConfig:
    learning_rates: list[float] = field(default_factory=lambda: [1e-4, 2e-4, 3e-4])
    lora_ranks: list[int] = field(default_factory=lambda: [8, 16, 32])
    geohash_precisions: list[int] = field(default_factory=lambda: [3, 4, 5])
    train_limits: list[int] = field(default_factory=lambda: [2048, 4096, 8192])
    image_sizes: list[int] = field(default_factory=lambda: [448, 576, 672])
    label_styles: list[str] = field(default_factory=lambda: ["full_json", "cell_json", "coords_json"])
    prompt_styles: list[str] = field(default_factory=lambda: ["strict_json", "geohash_first", "coords_first"])
    finetune_vision_layers: list[bool] = field(default_factory=lambda: [True, False])
    finetune_language_layers: list[bool] = field(default_factory=lambda: [True, False])


@dataclass
class ExperimentConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    label_style: str = "full_json"
    prompt_style: str = "strict_json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def output_root(self, config_path: Path) -> Path:
        return resolve_path(project_root(config_path), self.project.output_root)

    def tracker_path(self, config_path: Path) -> Path:
        return resolve_path(project_root(config_path), self.project.tracker_db)

    def prepared_data_path(self, config_path: Path) -> Path:
        return resolve_path(project_root(config_path), self.project.prepared_data_dir)

    def reports_path(self, config_path: Path) -> Path:
        return resolve_path(project_root(config_path), self.project.reports_dir)


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def project_root(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _construct(section_cls: type[Any], payload: dict[str, Any] | None) -> Any:
    return section_cls(**(payload or {}))


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(config_path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if overrides:
        raw = _deep_merge(raw, overrides)
    search_payload = dict(raw.get("search") or {})
    label_style = raw.get("label_style", search_payload.pop("label_style", "full_json"))
    prompt_style = raw.get("prompt_style", search_payload.pop("prompt_style", "strict_json"))
    return ExperimentConfig(
        project=_construct(ProjectConfig, raw.get("project")),
        dataset=_construct(DatasetConfig, raw.get("dataset")),
        model=_construct(ModelConfig, raw.get("model")),
        lora=_construct(LoraConfig, raw.get("lora")),
        training=_construct(TrainingConfig, raw.get("training")),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation")),
        autonomy=_construct(AutonomyConfig, raw.get("autonomy")),
        search=_construct(SearchConfig, search_payload),
        label_style=label_style,
        prompt_style=prompt_style,
    )


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
