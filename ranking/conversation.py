from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def content_text(item: dict[str, Any]) -> str:
    value = item.get("text", item.get("content", ""))
    return str(value)


def record_messages(
    record: dict[str, Any],
    *,
    include_assistant: bool,
) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record {record.get('sample_id')} has no messages")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role {role!r} in {record.get('sample_id')}")
        if role == "assistant" and not include_assistant:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_content: str | list[dict[str, Any]] = content
        elif isinstance(content, list):
            normalized_items: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    raise TypeError(
                        f"non-object content item in record {record.get('sample_id')}"
                    )
                item_type = item.get("type", "text")
                if item_type == "text":
                    normalized_items.append({"type": "text", "text": content_text(item)})
                elif item_type == "image":
                    image_path = Path(str(item.get("image") or ""))
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"missing Densepage image for {record.get('sample_id')}: {image_path}"
                        )
                    normalized_items.append({"type": "image", "image": str(image_path)})
                else:
                    raise ValueError(
                        f"unsupported content type {item_type!r} in {record.get('sample_id')}"
                    )
            normalized_content = normalized_items
        else:
            raise TypeError(
                f"unsupported message content in record {record.get('sample_id')}: "
                f"{type(content).__name__}"
            )
        normalized.append({"role": role, "content": normalized_content})
    if include_assistant:
        assistant_positions = [
            index for index, message in enumerate(normalized) if message["role"] == "assistant"
        ]
        if assistant_positions != [len(normalized) - 1]:
            raise ValueError(
                f"training record {record.get('sample_id')} must end with one assistant message"
            )
    return normalized


def process_conversation(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> dict[str, torch.Tensor]:
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
        padding=False,
    )
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def image_token_id(processor: Any) -> int:
    value = getattr(processor, "image_token_id", None)
    if isinstance(value, int):
        return value
    value = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if not isinstance(value, int) or value < 0:
        raise ValueError("Qwen processor does not expose a valid image token id")
    return value


def token_counts(processor: Any, encoded: dict[str, torch.Tensor]) -> dict[str, int]:
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    total = int(attention_mask.sum().item())
    image_patches = int((input_ids == image_token_id(processor)).sum().item())
    return {
        "total_input_tokens": total,
        "image_patch_tokens": image_patches,
        "non_image_patch_tokens": total - image_patches,
    }


class DensepageConversationDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]
