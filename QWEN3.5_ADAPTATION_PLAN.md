# Qwen 3.5 Adaptation Plan for Autoresearch

This document outlines the plan to adapt this autoresearch fork to apply the same autonomous research principles but for **Qwen 3.5 2B** ([unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)) instead of the custom GPT-from-scratch setup.

---

## 1. Project Summary (Current State)

### What This Project Does
- **Autoresearch**: An AI agent autonomously experiments with LLM training
- **Loop**: Agent edits `train.py` → runs 5 min training → evaluates `val_bpb` → keeps/discards based on improvement → repeats
- **Model**: Custom GPT (~50M params) trained from scratch on ClimbMix-400B
- **Metric**: `val_bpb` (bits per byte) — vocab-size-independent
- **Fixed constraints**: `prepare.py` is read-only; 5-minute time budget; single file (`train.py`) for agent edits

### Core Principles to Preserve
1. **Fixed 5-minute time budget** — experiments are comparable
2. **val_bpb metric** — vocab-size-independent evaluation
3. **Single file to modify** — agent only touches `train.py`
4. **Autonomous loop** — keep/discard based on improvement, no human in the loop
5. **Simplicity criterion** — simpler is better when results are equal

---

## 2. Qwen 3.5 2B Model Overview

| Spec | Value |
|------|-------|
| Parameters | 2B |
| Layers | 24 |
| Hidden dim | 2,048 |
| FFN intermediate | 6,144 |
| Architecture | Gated DeltaNet (3:1 linear:softmax attention) |
| Vocab size | 248,320 |
| Context | 262K tokens (native) |
| VRAM (LoRA bf16) | ~5 GB |
| License | Apache 2.0 |

**Note**: The GGUF model is for **inference** (llama.cpp, Ollama). For **fine-tuning**, we load the base model (e.g. `Qwen/Qwen3.5-2B` or `unsloth/Qwen3.5-2B`) and optionally export to GGUF after training.

---

## 3. Adaptation Strategy: Fine-Tuning vs From-Scratch

| Approach | Feasibility | Effort |
|----------|-------------|--------|
| **Fine-tune Qwen 3.5 2B** | ✅ High | Moderate |
| Train Qwen 3.5 architecture from scratch | ❌ Very high (2B params, Gated DeltaNet, MoE) | Massive |
| Incorporate Qwen 3.5 ideas into current GPT | ⚠️ Partial | Medium |

**Recommended**: **Fine-tune Qwen 3.5 2B** with LoRA, preserving the autoresearch loop. This keeps the principles intact while switching the base model.

---

## 4. Detailed Implementation Plan

### Phase 1: Dependencies & Tokenizer

#### 4.1 Update `pyproject.toml`
Add:
```toml
"transformers>=4.46.0",  # v5 for Qwen3.5
"unsloth",
"accelerate",
"huggingface-hub",
```

Unsloth provides optimized LoRA fine-tuning for Qwen 3.5 (1.5× faster, 50% less VRAM).

#### 4.2 Tokenizer Adaptation
**Problem**: `prepare.py` trains a custom BPE (vocab 8192). Qwen 3.5 uses its own tokenizer (vocab 248,320).

**Options**:
- **A (recommended)**: Add a `prepare_qwen.py` or extend `prepare.py` with a `--use-qwen-tokenizer` flag that:
  - Skips BPE training
  - Downloads Qwen tokenizer from HuggingFace
  - Builds `token_bytes.pt` for Qwen vocab (same logic: `len(token_str.encode("utf-8"))` per token)
  - Saves a Qwen-specific tokenizer artifact
- **B**: Keep `prepare.py` read-only and load Qwen tokenizer entirely from `train.py` — but then `make_dataloader` and `evaluate_bpb` need a tokenizer with `encode()`, `decode()`, `get_vocab_size()`, `get_bos_token_id()`.

**Tokenizer interface** (must match for `make_dataloader` and `evaluate_bpb`):
- `encode(text, prepend=None, num_threads=8)` → list of ids
- `decode(ids)` → str
- `get_vocab_size()` → int
- `get_bos_token_id()` → int

Qwen tokenizer (via `AutoTokenizer`) can be wrapped to match this interface.

#### 4.3 Data Format
ClimbMix-400B documents are BOS-aligned. Qwen uses `<|endoftext|>` or similar as BOS. Ensure:
- `BOS_TOKEN` maps to Qwen's appropriate token
- `make_dataloader` prepends BOS — works with any tokenizer that has `get_bos_token_id()`

---

### Phase 2: Model Loading & Training Loop

#### 4.4 Replace GPT with Qwen 3.5 in `train.py`

**Model loading** (Unsloth pattern):
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-2B",  # or Qwen/Qwen3.5-2B
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,  # auto
    load_in_4bit=False,  # bf16 LoRA recommended for Qwen3.5
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, ...)
```

**Forward signature**: The current `evaluate_bpb` expects:
```python
loss_flat = model(x, y, reduction='none')
```
So the model must accept `(idx, targets=None, reduction='mean')` and return loss or logits. Wrap the HuggingFace model to match:
```python
def forward(self, idx, targets=None, reduction='mean'):
    outputs = self.model(input_ids=idx, labels=targets)
    if targets is not None:
        return outputs.loss  # or handle reduction
    return outputs.logits
```

#### 4.5 Optimizer & Hyperparameters
- Replace MuonAdamW with standard AdamW (or keep for non-LoRA params if applicable)
- Unsloth typically uses AdamW with 2e-5 or similar for LoRA
- Adjust `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE` for 2B model + LoRA (~5GB VRAM)
- Keep LR schedules (warmup, warmdown) conceptually similar

#### 4.6 Time Budget & Loop
- Keep `TIME_BUDGET = 300` (5 min)
- Keep the same `while True` loop with `total_training_time >= TIME_BUDGET`
- Remove FLOPs estimation (or approximate for Qwen 3.5) — MFU reporting may differ

---

### Phase 3: prepare.py Modifications

**Constraint**: `program.md` says "Do not modify prepare.py". For this fork, we have two choices:

1. **Relax the constraint** for the Qwen fork: Allow modifications to `prepare.py` to support Qwen tokenizer and token_bytes. Update `program.md` accordingly.
2. **Minimal prepare.py changes**: Add optional Qwen path that is triggered by an env var or config file, so the "default" behavior stays the same.

**Required changes to prepare.py** (if we modify it):
- Add `USE_QWEN_TOKENIZER` or `--qwen` flag
- When enabled: download Qwen tokenizer, build token_bytes for Qwen vocab, save to `~/.cache/autoresearch/tokenizer_qwen/`
- `Tokenizer.from_directory()` accepts `tokenizer_dir` — add `from_qwen()` class method that loads HF tokenizer and builds token_bytes on first run

---

### Phase 4: program.md Updates

Update agent instructions:
- Model is now Qwen 3.5 2B (fine-tuning, not from scratch)
- What the agent CAN change: LoRA rank, alpha, learning rate, batch size, max_seq_length, optimizer settings, etc.
- What the agent CANNOT change: Base model architecture, tokenizer, evaluation metric, time budget
- Add note about VRAM (~5GB for 2B LoRA) and that OOM may occur with large batch sizes

---

## 5. File-by-File Change Summary

| File | Changes |
|------|---------|
| `pyproject.toml` | Add transformers, unsloth, accelerate, huggingface-hub |
| `prepare.py` | Add Qwen tokenizer path: download HF tokenizer, build token_bytes, optional `--qwen` flag |
| `train.py` | Replace GPT with Qwen 3.5 + LoRA; wrap forward for `(idx, targets, reduction)`; adjust optimizer, batch sizes, hyperparams |
| `program.md` | Update for Qwen 3.5 fine-tuning context; clarify agent scope |
| `README.md` | Document Qwen 3.5 fork; link to base model; note VRAM requirements |

---

## 6. Implementation Order

1. **Dependencies**: Update `pyproject.toml`, run `uv sync`
2. **Tokenizer**: Implement Qwen tokenizer support in prepare.py (or train.py if keeping prepare.py untouched)
3. **Model**: Replace GPT in train.py with Unsloth Qwen 3.5 + LoRA
4. **Compatibility**: Ensure `evaluate_bpb` works (model forward, token_bytes)
5. **Validation**: Run one full experiment (5 min) and verify val_bpb is reported
6. **program.md**: Update agent instructions
7. **README**: Document the fork

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Unsloth/Qwen 3.5 API changes | Pin versions; check Unsloth docs |
| OOM with 2B on smaller GPUs | Lower DEVICE_BATCH_SIZE, grad accumulation; document min VRAM |
| token_bytes for 248k vocab | Build once in prepare; ~1MB tensor |
| Data format mismatch | ClimbMix text should work; verify BOS handling |
| First-run model download | Cache to `~/.cache/huggingface`; document network requirement |

---

## 8. Success Criteria

- [ ] `uv run prepare.py --qwen` (or equivalent) prepares Qwen tokenizer + token_bytes
- [ ] `uv run train.py` loads Qwen 3.5 2B, fine-tunes for 5 min, prints val_bpb
- [ ] Agent can modify train.py (LoRA config, LR, batch size) and iterate
- [ ] val_bpb is comparable across runs (same tokenizer, same eval data)
- [ ] Peak VRAM stays within reasonable bounds (~5–10 GB for 2B LoRA)

---

## 9. References

- [Unsloth Qwen 3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Qwen 3.5 Model Card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Unsloth Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) (inference; base model for training)
- [Qwen 3.5 Architecture (Gated DeltaNet)](https://github.com/QwenLM/Qwen3.5)
