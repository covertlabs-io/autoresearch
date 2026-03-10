from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from qwen_geo.config import ExperimentConfig, dump_json
from qwen_geo.data import GeoChatDataset, limit_split, load_prepared_dataset


class WallClockBudgetCallback(TrainerCallback):
    def __init__(self, budget_minutes: int) -> None:
        self.budget_seconds = budget_minutes * 60
        self.start_time: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self.start_time = time.time()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.start_time is not None and (time.time() - self.start_time) >= self.budget_seconds:
            control.should_training_stop = True
        return control


def _build_trainer_args(config: ExperimentConfig, output_dir: Path) -> Any:
    import torch
    from trl import SFTConfig

    use_bf16 = config.training.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_steps=config.training.max_steps,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_grad_norm=config.training.max_grad_norm,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=config.training.dataloader_num_workers,
        seed=config.project.seed,
    )


def train_experiment(config: ExperimentConfig, prepared_path: Path, run_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    from trl import SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    dataset_dict = load_prepared_dataset(prepared_path)
    train_split = limit_split(dataset_dict["train"], config.dataset.train_limit)
    eval_split = limit_split(dataset_dict["eval"], config.dataset.eval_limit)

    train_dataset = GeoChatDataset(
        train_split,
        prompt_style=config.prompt_style,
        label_style=config.label_style,
        geohash_precision=config.dataset.geohash_precision,
    )

    model, processor = FastVisionModel.from_pretrained(
        config.model.name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        use_gradient_checkpointing="unsloth" if config.model.use_gradient_checkpointing else None,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.lora.finetune_vision_layers,
        finetune_language_layers=config.lora.finetune_language_layers,
        finetune_attention_modules=config.lora.finetune_attention_modules,
        finetune_mlp_modules=config.lora.finetune_mlp_modules,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
    )
    FastVisionModel.for_training(model)

    trainer_args = _build_trainer_args(config, run_dir / "trainer")
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": trainer_args,
        "data_collator": UnslothVisionDataCollator(
            model,
            processor,
            resize=(config.model.max_image_size, config.model.max_image_size),
        ),
        "callbacks": [WallClockBudgetCallback(config.training.time_budget_minutes)],
    }

    signature = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = processor
    elif "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = processor

    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    metrics = dict(train_result.metrics)

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(adapter_dir))

    dump_json(run_dir / "train_metrics.json", metrics)
    return model, processor, {"train_metrics": metrics, "train_examples": len(train_split), "eval_examples": len(eval_split)}
