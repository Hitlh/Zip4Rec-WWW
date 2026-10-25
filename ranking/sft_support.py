"""Language-only LoRA and FP32 restricted output head, extracted from SFT helpers."""
from __future__ import annotations
import os, json
from typing import Any
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3VLForConditionalGeneration, Trainer
from torch.utils.data import SequentialSampler
LANGUAGE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def verify_verbalizers(tokenizer: Any) -> list[int]:
    base = [
        {"role": "system", "content": "You are a music recommendation relevance model. Output A or B."},
        {"role": "user", "content": "Output one letter."},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        base, tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    token_ids: list[int] = []
    details: dict[str, Any] = {}
    for verbalizer in ("A", "B"):
        full_ids = tokenizer.apply_chat_template(
            base + [{"role": "assistant", "content": verbalizer}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"assistant template prefix mismatch for {verbalizer}")
        completion = full_ids[len(prompt_ids) :]
        if not completion:
            raise ValueError(f"empty assistant completion for {verbalizer}")
        raw = tokenizer.encode(verbalizer, add_special_tokens=False)
        if len(raw) != 1 or completion[0] != raw[0]:
            raise ValueError(
                f"verbalizer {verbalizer} is not one stable assistant token: "
                f"raw={raw}, completion_prefix={completion[:3]}"
            )
        token_ids.append(int(raw[0]))
        details[verbalizer] = {
            "token_id": int(raw[0]),
            "token": tokenizer.convert_ids_to_tokens(int(raw[0])),
            "completion_tokens": completion,
        }
    if len(set(token_ids)) != 2:
        raise ValueError(f"verbalizer token IDs are not unique: {token_ids}")
    print(json.dumps({"binary_verbalizers": details}, indent=2), flush=True)
    return token_ids


def discover_language_targets(model: torch.nn.Module) -> list[str]:
    suffixes = LANGUAGE_TARGETS
    targets = [
        name
        for name, module in model.named_modules()
        if "language_model.layers." in name
        and name.rsplit(".", 1)[-1] in suffixes
        and hasattr(module, "weight")
    ]
    counts = {
        suffix: sum(name.endswith(f".{suffix}") for name in targets) for suffix in suffixes
    }
    if not targets or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"incomplete language LoRA target discovery: {counts}")
    if any("visual" in name or "vision" in name for name in targets):
        raise RuntimeError("vision module accidentally selected as a LoRA target")
    print(json.dumps({"lora_target_counts": counts}, indent=2), flush=True)
    return targets


def load_model(args: argparse.Namespace, processor: Any) -> torch.nn.Module:
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(args.model), **load_kwargs)
    model.config.use_cache = False
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    targets = discover_language_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=targets,
        ),
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [
        name
        for name in trainable
        if "language_model.layers." not in name or "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected[:10]}")
    model.print_trainable_parameters()
    return model


class DeltaTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        verbalizer_token_ids: list[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.verbalizer_token_ids = verbalizer_token_ids
        self.reference_logits: dict[int, torch.Tensor] = {}

    def _get_train_sampler(self, train_dataset: Any | None = None) -> SequentialSampler:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        return SequentialSampler(dataset)

    def _restricted_logits(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        outputs = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        positions = inputs["attention_mask"].sum(dim=1).long() - 1
        batch_positions = torch.arange(positions.shape[0], device=positions.device)
        hidden = outputs.hidden_states[-1][batch_positions, positions].float()
        head = model.get_output_embeddings()
        ids = torch.tensor(
            self.verbalizer_token_ids, dtype=torch.long, device=head.weight.device
        )
        weight = head.weight.index_select(0, ids).float()
        logits = hidden @ weight.transpose(0, 1)
        bias = getattr(head, "bias", None)
        if bias is not None:
            logits = logits + bias.index_select(0, ids).float()
        return logits
