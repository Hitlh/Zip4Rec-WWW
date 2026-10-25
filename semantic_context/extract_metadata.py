#!/usr/bin/env python3
"""Extract the small k-core catalog from Amazon Music 2023 metadata JSONL."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    elif isinstance(value, dict):
        value = " ".join(f"{key}: {part}" for key, part in value.items())
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    items: set[str] = set()
    with open_text(args.data) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{args.data}:{line_number}: expected five fields")
            items.add(fields[2])
    item_map = {item: index for index, item in enumerate(sorted(items))}
    found: dict[str, dict[str, Any]] = {}
    opener = gzip.open if args.metadata_jsonl.suffix == ".gz" else open
    with opener(args.metadata_jsonl, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.metadata_jsonl}:{line_number}") from exc
            asin = str(source.get("parent_asin", ""))
            if asin not in item_map:
                continue
            title = clean_text(source.get("title")) or f"Music item {asin}"
            raw_categories = source.get("categories", [])
            if isinstance(raw_categories, list):
                categories = " > ".join(
                    text for value in raw_categories if (text := clean_text(value))
                )
            else:
                categories = clean_text(raw_categories)
            description = clean_text(source.get("description"))
            brand = clean_text(source.get("store"))
            details = source.get("details")
            if not brand and isinstance(details, dict):
                brand = clean_text(details.get("Brand") or details.get("Manufacturer"))
            found[asin] = {
                "item_id": item_map[asin],
                "original_item_id": asin,
                "title": title,
                "item_text": title,
                "categories": categories,
                "brand": brand,
                "description_clean": description,
                "description_status": "usable" if description else "missing",
            }
            if len(found) == len(item_map):
                break
    missing = sorted(set(item_map) - set(found))
    if missing:
        raise ValueError(f"Missing metadata for {len(missing)} items: {missing[:20]}")
    rows = [found[item] for item in sorted(found, key=item_map.__getitem__)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"items": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
