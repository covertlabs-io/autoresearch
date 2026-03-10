from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_geo.config import ExperimentConfig, dump_json, load_config
from qwen_geo.data import prepare_dataset
from qwen_geo.tracker import ExperimentTracker, RunRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentProposal:
    description: str
    overrides: dict[str, Any]
    parent_run_id: int | None


class PlannerAgent:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = random.Random(config.autonomy.planner_seed)

    def propose(self, tracker: ExperimentTracker) -> ExperimentProposal:
        completed = tracker.completed_runs()
        if not completed:
            return ExperimentProposal(description="baseline geolocation run", overrides={}, parent_run_id=None)

        champion = completed[0]
        champion_metrics = champion.metrics or {}
        champion_config = champion.config
        trial_index = len(tracker.list_runs()) + 1

        if champion_metrics.get("parse_rate", 0.0) < 0.7:
            return ExperimentProposal(
                description="tighten output schema and coarsen labels",
                overrides={
                    "prompt_style": "strict_json",
                    "label_style": "cell_json",
                    "dataset": {"geohash_precision": max(3, int(champion_config["dataset"]["geohash_precision"]) - 1)},
                },
                parent_run_id=champion.run_id,
            )

        mutation_axis = (trial_index - 1) % 8
        if mutation_axis == 0:
            value = self._alternate(self.config.search.learning_rates, champion_config["training"]["learning_rate"])
            return ExperimentProposal(
                description=f"adjust learning rate to {value}",
                overrides={"training": {"learning_rate": value}},
                parent_run_id=champion.run_id,
            )
        if mutation_axis == 1:
            value = self._alternate(self.config.search.lora_ranks, champion_config["lora"]["r"])
            return ExperimentProposal(
                description=f"change LoRA rank to {value}",
                overrides={"lora": {"r": value, "alpha": value}},
                parent_run_id=champion.run_id,
            )
        if mutation_axis == 2:
            value = self._alternate(self.config.search.geohash_precisions, champion_config["dataset"]["geohash_precision"])
            return ExperimentProposal(
                description=f"change target geohash precision to {value}",
                overrides={"dataset": {"geohash_precision": value}},
                parent_run_id=champion.run_id,
            )
        if mutation_axis == 3:
            value = self._alternate(self.config.search.label_styles, champion_config["label_style"])
            return ExperimentProposal(
                description=f"change label style to {value}",
                overrides={"label_style": value},
                parent_run_id=champion.run_id,
            )
        if mutation_axis == 4:
            value = self._alternate(self.config.search.prompt_styles, champion_config["prompt_style"])
            return ExperimentProposal(
                description=f"change prompt style to {value}",
                overrides={"prompt_style": value},
                parent_run_id=champion.run_id,
            )
        if mutation_axis == 5:
            value = self._alternate(self.config.search.image_sizes, champion_config["model"]["max_image_size"])
            return ExperimentProposal(
                description=f"change image resize to {value}",
                overrides={"model": {"max_image_size": value}},
                parent_run_id=champion.run_id,
            )
        if mutation_axis == 6:
            value = self._alternate(
                self.config.search.finetune_vision_layers,
                champion_config["lora"]["finetune_vision_layers"],
            )
            return ExperimentProposal(
                description=f"toggle vision-layer fine-tuning to {value}",
                overrides={"lora": {"finetune_vision_layers": value}},
                parent_run_id=champion.run_id,
            )

        value = self._alternate(self.config.search.train_limits, champion_config["dataset"]["train_limit"])
        language_value = self._alternate(
            self.config.search.finetune_language_layers,
            champion_config["lora"]["finetune_language_layers"],
        )
        return ExperimentProposal(
            description=f"change train subset to {value} and language layers to {language_value}",
            overrides={"dataset": {"train_limit": value}, "lora": {"finetune_language_layers": language_value}},
            parent_run_id=champion.run_id,
        )

    def _alternate(self, candidates: list[Any], current: Any) -> Any:
        filtered = [candidate for candidate in candidates if candidate != current]
        if not filtered:
            return current
        return filtered[self.rng.randrange(len(filtered))]


class ExecutorAgent:
    def __init__(self, package_root: Path) -> None:
        self.package_root = package_root

    def run(
        self,
        *,
        config_path: Path,
        run_dir: Path,
        overrides: dict[str, Any],
        timeout_minutes: int,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        env = os.environ.copy()
        src_path = str((self.package_root / "src").resolve())
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

        command = [
            sys.executable,
            "-m",
            "qwen_geo.cli",
            "run-experiment",
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--overrides-json",
            json.dumps(overrides),
        ]
        log_path = run_dir / "run.log"
        timeout_seconds = max(timeout_minutes * 60 + 900, 1800)

        try:
            with log_path.open("w", encoding="utf-8") as sink:
                completed = subprocess.run(
                    command,
                    cwd=str(self.package_root),
                    env=env,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
            if completed.returncode != 0:
                return False, None, f"experiment subprocess exited with code {completed.returncode}"
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                return False, None, "metrics.json was not produced"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return True, metrics, None
        except subprocess.TimeoutExpired:
            return False, None, f"experiment exceeded {timeout_seconds} seconds"


class JudgeAgent:
    def __init__(self, min_score_delta: float) -> None:
        self.min_score_delta = min_score_delta

    def promoted(self, tracker: ExperimentTracker, metrics: dict[str, Any]) -> bool:
        champion = tracker.best_run()
        if champion is None or champion.score is None:
            return True
        return metrics["judge_score"] >= champion.score + self.min_score_delta


def write_leaderboard(tracker: ExperimentTracker, report_path: Path) -> None:
    rows = tracker.list_runs()
    lines = [
        "# Qwen Geo Leaderboard",
        "",
        "| run | status | score | median_km | acc_25km | acc_200km | country_acc | description |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        metrics = row.metrics or {}
        lines.append(
            "| {run} | {status} | {score:.4f} | {median} | {acc25:.3f} | {acc200:.3f} | {country:.3f} | {desc} |".format(
                run=row.run_name,
                status=row.status,
                score=row.score or 0.0,
                median=f"{metrics.get('median_distance_km', float('inf')):.1f}"
                if metrics.get("median_distance_km") is not None
                else "n/a",
                acc25=metrics.get("acc_25km", 0.0),
                acc200=metrics.get("acc_200km", 0.0),
                country=metrics.get("country_accuracy", 0.0),
                desc=row.description,
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_autonomous_loop(config_path: Path, config: ExperimentConfig) -> None:
    package_root = config_path.parent.parent.resolve()
    output_root = config.output_root(config_path)
    tracker = ExperimentTracker(config.tracker_path(config_path))
    prepared_path = prepare_dataset(config, config.prepared_data_path(config_path))
    planner = PlannerAgent(config)
    executor = ExecutorAgent(package_root)
    judge = JudgeAgent(config.autonomy.min_score_delta)
    report_path = config.reports_path(config_path) / "leaderboard.md"

    output_root.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + config.autonomy.hours * 3600

    try:
        while time.time() < deadline and len(tracker.list_runs()) < config.autonomy.max_trials:
            proposal = planner.propose(tracker)
            run_index = len(tracker.list_runs()) + 1
            run_name = f"run_{run_index:03d}"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            merged_config = load_config(config_path, proposal.overrides)
            dump_json(run_dir / "resolved_config.json", merged_config.to_dict())
            dump_json(run_dir / "proposal.json", {"description": proposal.description, "overrides": proposal.overrides})

            run_id = tracker.start_run(
                run_name=run_name,
                started_at=utc_now(),
                description=proposal.description,
                config=merged_config.to_dict(),
                parent_run_id=proposal.parent_run_id,
            )

            success, metrics, error_message = executor.run(
                config_path=config_path,
                run_dir=run_dir,
                overrides=proposal.overrides,
                timeout_minutes=merged_config.training.time_budget_minutes,
            )

            if success and metrics is not None:
                promoted = judge.promoted(tracker, metrics)
                tracker.finish_run(
                    run_id,
                    status="promoted" if promoted else "completed",
                    finished_at=utc_now(),
                    score=metrics["judge_score"],
                    promoted=promoted,
                    metrics=metrics,
                    artifact_dir=str(run_dir),
                )
            else:
                tracker.finish_run(
                    run_id,
                    status="failed",
                    finished_at=utc_now(),
                    score=None,
                    promoted=False,
                    metrics=None,
                    artifact_dir=str(run_dir),
                    error_message=error_message,
                )

            write_leaderboard(tracker, report_path)
            dump_json(output_root / "state.json", {"prepared_data": str(prepared_path), "updated_at": utc_now()})
    finally:
        tracker.close()
