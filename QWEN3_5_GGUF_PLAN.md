# Qwen 3.5 GGUF fork plan

## Goal

Adapt this fork so it preserves the **operating principles** of `autoresearch`, but targets **Qwen 3.5** using the GGUF release at:

- https://huggingface.co/unsloth/Qwen3.5-2B-GGUF

This document is intentionally repo-specific. It explains what this project is today, what "the same principles" really means here, where Qwen 3.5 conflicts with the current design, and the best phased plan to make the fork coherent.

---

## What this project is today

This repository is a **minimal autonomous pretraining harness**:

- `prepare.py`
  - downloads parquet text shards from `karpathy/climbmix-400b-shuffle`
  - trains a custom BPE tokenizer with `rustbpe`
  - saves tokenizer artifacts under `~/.cache/autoresearch/`
  - provides the packed dataloader and fixed `evaluate_bpb`
- `train.py`
  - defines a custom GPT-like model from scratch
  - trains for a fixed 5-minute budget
  - reports `val_bpb`
- `program.md`
  - tells the agent to only edit `train.py`
  - forbids changing `prepare.py`, dependencies, and the evaluation harness
- `README.md`
  - documents the repo as a self-contained, single-GPU, single-file autonomous research loop

Important consequence: this fork is **not currently a model wrapper**, **not a Hugging Face fine-tuning project**, and **not a GGUF inference project**. It is a custom, from-scratch language-model training setup.

---

## The principles that should be preserved

The current repo does **not** encode a named model family. The real principles are operational:

1. **Tiny surface area**
   - Very few important files.
   - Easy for an agent to understand.

2. **One clear experiment surface**
   - Today that surface is `train.py`.
   - In the Qwen fork, there should still be one primary file the agent changes.

3. **Fixed benchmark**
   - Runs are comparable because the evaluation setup is stable.
   - The Qwen fork must keep a fixed dataset/task/metric.

4. **Autonomous loop**
   - Change something, run one command, measure, keep or discard.

5. **Simple local runtime**
   - Prefer a single-machine workflow.
   - Avoid a large framework/config sprawl if possible.

6. **Results are easy to log**
   - One summary block per run.
   - `results.tsv` stays human-readable.

7. **Simplicity wins ties**
   - Small gains do not justify large complexity.

These are the principles to carry forward. The current tokenizer, architecture, and `val_bpb` metric are **implementation details**, not principles.

---

## Why a direct swap to Qwen 3.5 GGUF will not work

### 1) Current repo assumptions are from-scratch pretraining assumptions

`prepare.py` currently assumes:

- text dataset shards
- local tokenizer training
- a tiny 8192-token custom vocab
- synthetic BOS insertion for every document
- bits-per-byte validation on token cross-entropy

`train.py` currently assumes:

- a custom GPT-like architecture
- PyTorch training on CUDA
- mutable model weights
- optimizer and batching tuned for 5-minute pretraining experiments

That is fundamentally different from using a quantized GGUF checkpoint.

### 2) The target model is not just "a text tokenizer + transformer"

Public metadata for `Qwen/Qwen3.5-2B` shows:

- architecture: `Qwen3_5ForConditionalGeneration`
- model type: `qwen3_5`
- text vocab size: `248320`
- context length: `262144`
- special tokens such as:
  - `<|endoftext|>`
  - `<|im_start|>`
  - `<|im_end|>`
  - `<|vision_start|>`
  - `<|vision_end|>`
  - `<|image_pad|>`
  - `<|video_pad|>`
- an upstream `chat_template.jinja`
- multimodal config entries and `mmproj` files in the GGUF repo

So this is a much larger and more structured model family than the current repo's custom text-only GPT.

### 3) GGUF implies an inference runtime, not this PyTorch training runtime

`unsloth/Qwen3.5-2B-GGUF` contains quantized `.gguf` files plus `mmproj` files. That means the natural execution model is something like:

- llama.cpp
- `llama-cpp-python`
- or another GGUF-compatible runtime

That is a different stack from:

- `torch`
- `kernels`
- flash attention
- from-scratch backprop training

### 4) The current "only edit train.py" rule is too strict

For a Qwen fork, model identity moves into:

- runtime/model loading
- prompt formatting
- evaluation dataset/task
- maybe download logic
- maybe multimodal handling

So `program.md` will need to change. A realistic Qwen fork cannot keep the "edit only train.py, never touch prepare.py" rule exactly as written.

---

## Recommended product definition

If the user wants **the same principles, but for Qwen 3.5 GGUF**, the fork should become:

> A minimal autonomous **Qwen 3.5 inference/evaluation research harness** where the agent iterates on prompts, decoding settings, and a small amount of runtime logic against a fixed benchmark.

That keeps the original spirit:

- autonomous
- benchmark-driven
- small
- comparable runs
- single-machine

But it stops pretending this is still a from-scratch pretraining project.

---

## Scope decision that should be made first

Before editing code, decide this explicitly:

### Preferred scope: text-only first

Even though Qwen 3.5 is multimodal, the first fork should treat it as **text-only**:

- only text messages in the benchmark
- ignore image/video inputs initially
- keep `mmproj` support out of the first milestone unless required

Reason:

- much smaller implementation surface
- easier to keep the repo agent-friendly
- aligns better with the current project's simplicity principle

### Defer multimodal support

Only add image/video support after the text-only path is stable. The upstream chat template already contains vision tokens, so multimodal expansion can come later without redesigning the repo twice.

---

## Best migration strategy

## Phase 0 - lock the new contract

Rewrite the project around a new contract before touching implementation details.

Decide and document:

1. **Primary benchmark**
   - Example options:
     - exact-match eval on a fixed task set
     - judge-based scoring on a fixed prompt dataset
     - code task pass rate
     - logprob/perplexity if the chosen GGUF runtime exposes token logprobs reliably
   - Recommendation for v1:
     - use a **fixed prompt-response benchmark with deterministic scoring**
     - avoid LLM-as-judge for the first version if possible

2. **Primary optimization knobs**
   - system prompt
   - prompt template additions
   - generation settings
   - reasoning/thinking mode toggles if supported
   - max tokens / stop conditions
   - tool prompt formatting if tool use is part of the benchmark

3. **Single editable file**
   - keep one main experiment file for the agent
   - recommendation: `experiment.py`

4. **Runtime choice**
   - recommendation: `llama-cpp-python` or a subprocess wrapper around llama.cpp
   - prefer the smallest reliable integration

5. **Quantization baseline**
   - pick one quant and hold it fixed for benchmark comparability
   - recommendation: start with `Qwen3.5-2B-Q4_K_M.gguf`
   - do not let the agent change quants during normal experiments

Deliverable of Phase 0:

- updated design note
- updated README direction
- agreement on benchmark and quant

---

## Phase 1 - reshape the repo without making it big

The current repo is flat and easy to scan. Preserve that.

Recommended target layout:

- `prepare.py`
  - repurposed or replaced to prepare the benchmark dataset/cache
  - no tokenizer training
- `runtime.py`
  - GGUF loading and model invocation
- `experiment.py`
  - the single file the agent edits
  - prompt construction, generation params, light orchestration
- `eval.py`
  - fixed metric and benchmark harness
- `program.md`
  - updated agent instructions
- `README.md`
  - updated fork docs

If minimizing file count is more important than separation, combine `runtime.py` and `eval.py` into `prepare.py`-style utilities. But keep **one obvious editable file** for the agent.

Deliverable of Phase 1:

- new file structure decided
- old pretraining-specific assumptions marked for removal

---

## Phase 2 - replace the current fixed assumptions

### Remove assumptions that must go away

These should not survive into the Qwen fork:

- custom tokenizer training with `rustbpe`
- `VOCAB_SIZE = 8192`
- synthetic `<|reserved_0|>` BOS token
- `encode_ordinary()`-based plain-text tokenization assumptions
- parquet shard download from `climbmix-400b-shuffle`
- `evaluate_bpb`
- custom GPT model construction in `train.py`
- Muon/AdamW pretraining loop as the experiment core

### Replace them with Qwen-aware fixed pieces

Add fixed components for:

1. **Model artifact selection**
   - choose the GGUF file
   - optionally choose `mmproj` if multimodal support is enabled later

2. **Prompt formatting**
   - preserve upstream Qwen control tokens
   - track the prompt contract in the repo
   - do not synthesize fake BOS behavior

3. **Benchmark data**
   - local cache under `~/.cache/autoresearch/` is still a good idea
   - benchmark set should be deterministic and versioned

4. **Evaluation**
   - fixed scoring function
   - fixed decoding rules for benchmark mode
   - stable summary output

Deliverable of Phase 2:

- the repo no longer pretends to be a pretraining harness
- fixed Qwen runtime assumptions are defined cleanly

---

## Phase 3 - build the minimum viable Qwen fork

This is the first real implementation target.

### v1 behavior

The fork should be able to:

1. install dependencies
2. download or reference the selected GGUF
3. run one benchmark command
4. print a summary block
5. let the agent edit one file and iterate

### Proposed commands

Example shape:

```bash
uv sync
uv run prepare.py
uv run experiment.py
```

Where:

- `prepare.py`
  - downloads benchmark data
  - optionally downloads or verifies the chosen GGUF artifact
- `experiment.py`
  - loads Qwen 3.5 GGUF through the fixed runtime
  - runs the benchmark
  - prints summary metrics

### Summary block recommendation

Keep the current spirit of the output format:

```text
---
score:            0.742000
eval_seconds:     118.2
total_seconds:    125.7
peak_vram_mb:     0.0
peak_ram_mb:      3821.4
tokens_per_sec:   87.4
quant:            Q4_K_M
benchmark:        text_v1
```

Use names that match the new world. Do not reuse `val_bpb` unless the metric truly is BPB.

Deliverable of Phase 3:

- runnable baseline Qwen benchmark
- stable output format
- first benchmark score recorded

---

## Phase 4 - rewrite the agent instructions around the new experiment surface

`program.md` should be rewritten so the autonomous loop matches the new project.

Recommended new rules:

### What the agent CAN change

- `experiment.py`
- maybe one prompt file, if prompt text is split out

### What the agent CANNOT change

- evaluation dataset
- metric implementation
- selected base quant
- runtime wrapper
- dependency list, unless explicitly allowed by the human

### New optimization target

The goal is no longer "lowest `val_bpb`". It should become:

- highest task score
- or lowest error rate
- or best deterministic benchmark metric

### Keep the keep/discard loop

The existing experiment loop still works well:

1. edit
2. run
3. score
4. commit
5. keep or reset

Deliverable of Phase 4:

- `program.md` matches the actual fork
- the agent has a clear, bounded experiment surface

---

## Phase 5 - only then consider multimodal support

Once the text-only benchmark is stable, optionally add:

- image inputs
- vision token handling
- `mmproj` model selection
- multimodal benchmark cases

This should be a second milestone, not part of the first port.

Reason:

- the target model metadata indicates multimodal capability
- the current repo has zero multimodal infrastructure
- adding this too early will make the fork much larger and harder for agents to navigate

Deliverable of Phase 5:

- optional multimodal expansion without breaking text-only simplicity

---

## Concrete file-by-file edit plan

## 1. `README.md`

Update it from:

- autonomous pretraining harness

to:

- autonomous Qwen 3.5 GGUF benchmark harness

Must change:

- quick start commands
- project structure section
- design choices section
- benchmark/metric description
- model/runtime assumptions

## 2. `program.md`

Rewrite:

- remove "only edit `train.py`"
- remove "do not modify `prepare.py`" if `prepare.py` changes meaning
- replace `val_bpb` optimization target
- define the new experiment file and benchmark loop

## 3. `prepare.py`

Replace or heavily rewrite:

- remove tokenizer training
- remove parquet training-data logic
- add benchmark cache/download logic
- optionally add model artifact verification

## 4. `train.py`

Do not try to mutate the current file into Qwen support.

Recommendation:

- delete or retire `train.py`
- replace it with `experiment.py` or another Qwen-specific entrypoint

Reason:

- current file is a custom PyTorch model trainer
- preserving it will create confusion and dead concepts

## 5. `pyproject.toml`

Update dependencies to match the chosen runtime.

Most likely:

- add GGUF runtime dependency
- possibly add `huggingface_hub`
- possibly remove training-only packages later if they are no longer needed

Do this conservatively to keep the repo small.

## 6. `analysis.ipynb`

Optional follow-up:

- update or replace it only if `results.tsv` schema changes enough to make it obsolete

---

## The biggest implementation risks

1. **Choosing GGUF when the original repo is training-centric**
   - this is a project reshape, not a small port

2. **Accidentally preserving the wrong constraints**
   - especially `val_bpb`, tokenizer training, and BOS insertion

3. **Underestimating Qwen 3.5 complexity**
   - upstream metadata indicates a multimodal model family

4. **Letting the editable surface sprawl**
   - if the agent can change too many files, the fork loses the original repo's strength

5. **Using a non-deterministic benchmark**
   - makes keep/discard decisions noisy

---

## Recommended first implementation milestone

If editing starts now, the first milestone should be:

> Build a text-only Qwen 3.5 GGUF benchmark harness with one editable experiment file and one fixed evaluation command.

That means:

- no multimodal support yet
- one fixed quant
- one fixed benchmark dataset
- one clear score
- one command to run
- one main file for the agent to edit

This milestone best preserves the original project's principles.

---

## Optional alternative: use trainable Qwen instead of GGUF

If the real goal is to preserve not only the workflow but also the idea of **changing model behavior through training**, then the better target is not `unsloth/Qwen3.5-2B-GGUF`.

A better fit would be:

- `Qwen/Qwen3.5-2B`
- or the underlying base model/checkpoint in Transformers format

Why:

- trainable weights
- native tokenizer/config files
- better fit for fine-tuning or continued training
- more compatible with the current repo's "modify model behavior through repeated runs" spirit

If the user insists on GGUF, the fork should be framed as an **inference-time optimization harness**, not a training harness.

---

## Bottom line

The right way to "apply the same principles" is:

- keep the repo small
- keep one main editable file
- keep a fixed benchmark
- keep an autonomous keep/discard loop
- keep simple output and logging

But the wrong way would be:

- trying to shoehorn Qwen 3.5 GGUF into the existing tokenizer/training/BPB machinery

This fork should become a **minimal autonomous Qwen benchmark repo**, not a patched version of the current from-scratch pretraining code.
