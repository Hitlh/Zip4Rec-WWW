#!/usr/bin/env python3
"""Train a binary restricted-LM-head LoRA for profile-conditioned DensePages."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, TrainingArguments, set_seed

from .conversation import DensepageConversationDataset, process_conversation, read_jsonl, record_messages
from .sft_support import load_model, verify_verbalizers, DeltaTrainer


LABEL_TO_CLASS = {"A": 0, "B": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-acc-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6061)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data and processor inputs, write the report, and exit before loading the model.",
    )
    parser.add_argument(
        "--preflight-scan-all",
        action="store_true",
        help="Process every training row during preflight to verify the max-length bound.",
    )
    return parser.parse_args()


class BinaryCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed = []
        for row in features:
            prompt = process_conversation(
                self.processor, record_messages(row, include_assistant=False), add_generation_prompt=True
            )
            if int(prompt["attention_mask"].sum()) > self.max_length:
                raise ValueError(f"prompt too long: {row['sample_id']}")
            processed.append(prompt)
        maximum = max(int(row["input_ids"].shape[1]) for row in processed)
        target_length = math.ceil(maximum / 8) * 8
        pad_id = int(self.processor.tokenizer.pad_token_id)

        def pad(value: torch.Tensor, fill: int) -> torch.Tensor:
            sequence = value[0]
            amount = target_length - len(sequence)
            return sequence if amount <= 0 else torch.cat([sequence, torch.full((amount,), fill, dtype=sequence.dtype)])

        batch = {
            "input_ids": torch.stack([pad(row["input_ids"], pad_id) for row in processed]),
            "attention_mask": torch.stack([pad(row["attention_mask"], 0) for row in processed]),
            "native10_target": torch.tensor([LABEL_TO_CLASS[row["label"]] for row in features]),
            "native10_index": torch.tensor([row["_index"] for row in features]),
            "sample_weight": torch.tensor([float(row.get("sample_weight", 1.0)) for row in features]),
        }
        if all("pixel_values" in row and "image_grid_thw" in row for row in processed):
            batch["pixel_values"] = torch.cat([row["pixel_values"] for row in processed])
            batch["image_grid_thw"] = torch.cat([row["image_grid_thw"] for row in processed])
        if all("mm_token_type_ids" in row for row in processed):
            batch["mm_token_type_ids"] = torch.stack([pad(row["mm_token_type_ids"], 0) for row in processed])
        return batch


class WeightedDeltaTrainer(DeltaTrainer):
    """Adapter-minus-base CE with audited per-example confidence weights."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        targets = inputs.pop("native10_target")
        sample_indices = inputs.pop("native10_index")
        sample_weights = inputs.pop("sample_weight").to(dtype=torch.float32)
        keys = [int(value) for value in sample_indices.detach().cpu().tolist()]
        missing = [key for key in keys if key not in self.reference_logits]
        if missing:
            if len(missing) != len(keys):
                raise RuntimeError("mixed cached/uncached batches are unsupported")
            with torch.no_grad(), model.disable_adapter():
                base_logits = self._restricted_logits(model, inputs)
            for key, logits in zip(keys, base_logits.detach().float().cpu()):
                self.reference_logits[key] = logits
        adapter_logits = self._restricted_logits(model, inputs)
        base_logits = torch.stack([self.reference_logits[key] for key in keys]).to(
            device=adapter_logits.device, dtype=adapter_logits.dtype
        )
        calibrated_logits = adapter_logits - base_logits.detach()
        losses = F.cross_entropy(calibrated_logits, targets, reduction="none")
        weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
        loss = (losses * weights).mean()
        if return_outputs:
            return loss, {"logits": calibrated_logits, "adapter_logits": adapter_logits.detach(), "base_logits": base_logits.detach()}
        return loss


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume_from_checkpoint is None:
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("CD restricted-logit cache requires batch size 1")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    token_ids = verify_verbalizers(processor.tokenizer)[:2]
    rows = read_jsonl(args.train_data)
    for index, row in enumerate(rows):
        if row.get("label") not in LABEL_TO_CLASS:
            raise ValueError(row.get("label"))
        row["_index"] = index
    counts = Counter(row["label"] for row in rows)
    if counts["A"] != counts["B"]:
        raise ValueError(f"training data must be 1:1, got {counts}")
    weights = [float(row.get("sample_weight", 1.0)) for row in rows]
    collator = BinaryCollator(processor, args.max_length)
    first = collator([rows[0]])
    preflight = {
        "rows": len(rows), "class_counts": dict(counts),
        "first_prompt_tokens": int(first["attention_mask"].sum()),
        "first_image_patch_tokens": int(first["image_grid_thw"].prod(dim=1).sum() // 4) if "image_grid_thw" in first else 0,
        "expected_initial_loss": math.log(2),
        "objective": "CE over adapter-minus-base A/B restricted LM-head logits",
        "sample_weight_mean": sum(weights) / len(weights),
        "sample_weight_min": min(weights),
        "sample_weight_max": max(weights),
    }
    if args.preflight_scan_all:
        prompt_tokens: list[int] = []
        image_patch_tokens: list[int] = []
        for offset, row in enumerate(rows, start=1):
            batch = collator([row])
            prompt_tokens.append(int(batch["attention_mask"].sum()))
            image_patch_tokens.append(
                int(batch["image_grid_thw"].prod(dim=1).sum() // 4) if "image_grid_thw" in batch else 0
            )
            if offset % 500 == 0:
                print(f"preflight scanned {offset}/{len(rows)}", flush=True)
        preflight.update(
            {
                "scanned_rows": len(rows),
                "prompt_tokens_min": min(prompt_tokens),
                "prompt_tokens_max": max(prompt_tokens),
                "image_patch_tokens_min": min(image_patch_tokens),
                "image_patch_tokens_max": max(image_patch_tokens),
                "all_rows_within_max_length": max(prompt_tokens) <= args.max_length,
            }
        )
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "preflight": preflight}, indent=2) + "\n")
    if args.preflight_only:
        (args.output_dir / "preflight_complete.json").write_text(
            json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    model = load_model(args, processor)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, bf16=True, tf32=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps, logging_first_step=True, report_to="none",
        remove_unused_columns=False, dataloader_num_workers=0, seed=args.seed, data_seed=args.seed,
    )
    trainer = WeightedDeltaTrainer(
        model=model, args=training_args, train_dataset=DensepageConversationDataset(rows),
        data_collator=collator, verbalizer_token_ids=token_ids,
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    trainer.save_state()
    print(json.dumps({"adapter_saved": str(args.output_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
