# Qwen Geo

`qwen/` is a clean-room adaptation of the autonomous experimentation pattern in `autoresearch`, retargeted at **photo geolocation** instead of language-model pretraining.

The core idea is the same:

- keep the loop small and repeatable
- run many short experiments with a fixed budget
- evaluate automatically
- promote only clearly better candidates
- let the system keep iterating for hours

Unlike the reference repo, this prototype does **not** self-edit Python code. For geolocation work, the better search surface is the training recipe itself: dataset settings, prompt/label structure, LoRA settings, image resolution, and which parts of the multimodal model are adapted.

## What this system does

This folder provides an end-to-end workflow for:

- selecting and preparing a geotagged image dataset
- fine-tuning **`unsloth/Qwen3.5-2B-Base`** with Unsloth vision LoRA
- evaluating location predictions on held-out images
- tracking experiment lineage and metrics in SQLite
- automatically proposing the next experiment
- running a long autonomous loop for many hours

## Default task framing

The prototype is configured for **single-image geolocation**:

- **Input:** one outdoor photo
- **Output:** a structured prediction containing a geohash cell and/or coordinates
- **Primary default dataset:** `marcelomoreno26/geoguessr`
- **Optional scale-up dataset config:** `osv5m/osv5m`

The model is trained to emit compact JSON such as:

```json
{"country_code":"US","geohash":"dr5r","latitude":40.7128,"longitude":-74.0060}
```

The planner can mutate the output schema across runs, for example:

- `full_json`
- `cell_json`
- `coords_json`

That makes the loop explore whether coarse geocells or direct coordinates are easier for the model to learn.

## Autonomous roles

The outer loop is implemented with explicit agent roles:

- **PlannerAgent**: chooses the next small mutation from the current best run
- **ExecutorAgent**: launches a subprocess for one train+eval trial
- **JudgeAgent**: scores the run and decides whether it becomes the new champion
- **ExperimentTracker**: stores configs, scores, metrics, and artifacts in SQLite

This mirrors the *plan -> execute -> evaluate -> refine* rhythm from `autoresearch`, but uses run configs and adapters as the experiment state instead of code diffs.

## Metrics and promotion logic

Each run is judged on held-out geolocation examples with:

- parse success rate
- country accuracy
- geohash prefix accuracy
- median distance in km
- accuracy within 1 km / 25 km / 200 km / 750 km

The loop computes a composite `judge_score` and promotes only runs that beat the current champion by a configurable margin.

## Project layout

```text
qwen/
├── configs/
│   ├── geoguessr_baseline.toml
│   ├── geoguessr_h200.toml
│   ├── osv5m_large.toml
│   └── smoke_local.toml
├── src/qwen_geo/
│   ├── agents.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── geo.py
│   ├── prompts.py
│   ├── smoke.py
│   └── training.py
├── tests/
│   ├── test_config.py
│   ├── test_geo.py
│   └── test_parse.py
└── pyproject.toml
```

## Install

From the `qwen/` directory:

```bash
uv sync
```

## Typical usage

### 0) Local smoke validation

If you want to validate the full control loop on a machine without GPU/Unsloth, use the smoke config first:

```bash
PYTHONPATH=src python3 -m qwen_geo.cli prepare-data --config configs/smoke_local.toml
PYTHONPATH=src python3 -m qwen_geo.cli run-experiment --config configs/smoke_local.toml --run-dir runs_smoke/manual_trial
PYTHONPATH=src python3 -m qwen_geo.cli run-loop --config configs/smoke_local.toml --hours 0.02 --max-trials 2
```

This runs the same planner/executor/judge/tracker workflow on a tiny synthetic geolocation dataset and a standard-library smoke backend, which is useful for validating the orchestration before moving to a real GPU.

### 1) Prepare the dataset cache

```bash
uv run qwen-geo prepare-data --config configs/geoguessr_baseline.toml
```

This downloads the raw dataset, normalizes it into a stable schema, derives country codes/geohashes, and writes a reusable prepared dataset cache.

### 2) Run one experiment

```bash
uv run qwen-geo run-experiment \
  --config configs/geoguessr_baseline.toml \
  --run-dir runs/manual_trial
```

Outputs include:

- `resolved_config.json`
- `train_metrics.json`
- `predictions.jsonl`
- `metrics.json`
- `experiment_summary.json`
- adapter weights under `adapter/`

### 3) Run the autonomous loop

```bash
uv run qwen-geo run-loop \
  --config configs/geoguessr_h200.toml \
  --hours 12 \
  --max-trials 24
```

The loop will:

1. prepare data if needed
2. launch a baseline run
3. score it
4. mutate one or two knobs
5. launch the next run
6. update SQLite + markdown leaderboard
7. continue until the time or trial budget is exhausted

## Practical notes

- `configs/smoke_local.toml` is the fully local validation path. It does not require CUDA and is intended for end-to-end orchestration testing.
- Unsloth vision fine-tuning APIs change quickly; this implementation uses the current `FastVisionModel` + `UnslothVisionDataCollator` pattern.
- For Qwen 3.5, **bf16 LoRA is preferred**. The config leaves `load_in_4bit = false` by default.
- `configs/geoguessr_h200.toml` is the recommended starting point for the target **H200** environment. It uses a larger train subset, larger image size, bigger LoRA rank, and more aggressive batch accumulation than the baseline config.
- If the default dataset is too small or noisy for your GPU budget, switch to the `osv5m_large.toml` config and shorten per-trial budgets during exploration.
- The planner is deliberately simple and local-search oriented. It is meant to run unattended and accumulate evidence, not to perform deep global optimization in one shot.

## Why this is a good fit for geolocation

Photo geolocation rewards exactly the kind of loop that `autoresearch` encourages:

- many cheap iterations
- narrow changes per run
- stable evaluation
- artifact tracking
- autonomous continuation for long periods

What changes is the experiment surface:

- image resolution
- schema choice: geocell vs coordinates
- prompt wording
- LoRA capacity
- which vision/language blocks are adapted
- dataset size and split strategy

That gives the system meaningful levers to improve over time without needing to rewrite model internals between runs.
