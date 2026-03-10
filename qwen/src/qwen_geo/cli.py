from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_geo.config import dump_json, load_config


def _parse_overrides(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def command_prepare_data(args: argparse.Namespace) -> None:
    from qwen_geo.data import prepare_dataset

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    prepared_path = prepare_dataset(config, config.prepared_data_path(config_path), force=args.force)
    print(prepared_path)


def command_run_experiment(args: argparse.Namespace) -> None:
    from qwen_geo.data import limit_split, load_prepared_dataset, prepare_dataset
    from qwen_geo.evaluation import evaluate_model
    from qwen_geo.smoke import evaluate_smoke_model, train_smoke_experiment
    from qwen_geo.training import train_experiment

    config_path = Path(args.config).resolve()
    overrides = _parse_overrides(args.overrides_json)
    config = load_config(config_path, overrides)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "resolved_config.json", config.to_dict())

    prepared_path = prepare_dataset(config, config.prepared_data_path(config_path))
    dataset_dict = load_prepared_dataset(prepared_path)
    eval_split = limit_split(dataset_dict["eval"], config.evaluation.eval_examples)

    if config.model.backend == "smoke":
        model, processor, train_summary = train_smoke_experiment(config, prepared_path, run_dir)
        metrics = evaluate_smoke_model(model, eval_split, config, run_dir)
    else:
        model, processor, train_summary = train_experiment(config, prepared_path, run_dir)
        from unsloth import FastVisionModel

        FastVisionModel.for_inference(model)
        metrics = evaluate_model(
            model,
            processor,
            eval_split,
            config.evaluation,
            prompt_style=config.prompt_style,
            run_dir=run_dir,
        )
    combined = {**train_summary, **metrics}
    dump_json(run_dir / "experiment_summary.json", combined)
    print(json.dumps(combined, indent=2, sort_keys=True))


def command_run_loop(args: argparse.Namespace) -> None:
    from qwen_geo.agents import run_autonomous_loop

    config_path = Path(args.config).resolve()
    overrides: dict[str, Any] = {}
    if args.hours is not None:
        overrides["autonomy"] = {"hours": args.hours}
    if args.max_trials is not None:
        overrides.setdefault("autonomy", {})["max_trials"] = args.max_trials
    config = load_config(config_path, overrides if overrides else None)
    run_autonomous_loop(config_path, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous Qwen geolocation training loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Download and normalize the dataset.")
    prepare_parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    prepare_parser.add_argument("--force", action="store_true", help="Rebuild the prepared dataset cache.")
    prepare_parser.set_defaults(func=command_prepare_data)

    experiment_parser = subparsers.add_parser("run-experiment", help="Run one fine-tune + evaluation trial.")
    experiment_parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    experiment_parser.add_argument("--run-dir", required=True, help="Directory where outputs should be written.")
    experiment_parser.add_argument(
        "--overrides-json",
        default="{}",
        help="JSON object with nested config overrides for the current experiment.",
    )
    experiment_parser.set_defaults(func=command_run_experiment)

    loop_parser = subparsers.add_parser("run-loop", help="Run the autonomous planner/executor/judge loop.")
    loop_parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    loop_parser.add_argument("--hours", type=float, default=None, help="Optional wall-clock budget override.")
    loop_parser.add_argument("--max-trials", type=int, default=None, help="Optional trial-count override.")
    loop_parser.set_defaults(func=command_run_loop)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
