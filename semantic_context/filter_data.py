#!/usr/bin/env python3
"""Prepare deduplicated positive temporal k-core data from Amazon Reviews 2023."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from .raw_filters import load_filtered_interactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-k", type=int, default=10)
    parser.add_argument("--item-k", type=int, default=15)
    return parser.parse_args()


def iterative_kcore(
    rows: list[tuple[str, str, int, int]], user_k: int, item_k: int
) -> tuple[list[tuple[str, str, int, int]], int]:
    current = rows
    rounds = 0
    while True:
        user_counts = Counter(row[0] for row in current)
        item_counts = Counter(row[1] for row in current)
        valid_users = {key for key, count in user_counts.items() if count >= user_k}
        valid_items = {key for key, count in item_counts.items() if count >= item_k}
        filtered = [
            row for row in current if row[0] in valid_users and row[1] in valid_items
        ]
        rounds += 1
        if len(filtered) == len(current):
            return filtered, rounds
        current = filtered


def main() -> int:
    args = parse_args()
    if args.user_k < 3 or args.item_k < 2:
        raise ValueError("user-k must be >=3 and item-k must be >=2")

    if args.output.exists():
        raise FileExistsError(args.output)
    rows, source_filter = load_filtered_interactions(
        args.reviews,
        args.metadata,
        minimum_user_interactions=args.user_k,
    )

    # Parent-ASIN variants can create repeated user-item rows. Keep the latest
    # event before temporal splitting so held-out targets cannot also be seen in
    # the same user's training history.
    latest: dict[tuple[str, str], tuple[int, int]] = {}
    for user_id, item_id, rating, timestamp in rows:
        key = (user_id, item_id)
        previous = latest.get(key)
        if previous is None or timestamp >= previous[1]:
            latest[key] = (rating, timestamp)

    positives = [
        (user_id, item_id, rating, timestamp)
        for (user_id, item_id), (rating, timestamp) in latest.items()
        if rating >= 4
    ]
    core_rows, rounds = iterative_kcore(positives, args.user_k, args.item_k)

    if not core_rows:
        raise ValueError('filters produced an empty k-core')
    grouped: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for row in core_rows:
        grouped[row[0]].append(row)

    splits: dict[str, list[tuple[str, str, int, int]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    for user_id, user_rows in grouped.items():
        user_rows.sort(key=lambda row: (row[3], row[1]))
        splits["train"].extend(user_rows[:-2])
        splits["valid"].append(user_rows[-2])
        splits["test"].append(user_rows[-1])

    train_item_counts = Counter(row[1] for row in splits["train"])
    valid_cold = sum(train_item_counts[row[1]] == 0 for row in splits["valid"])
    test_cold = sum(train_item_counts[row[1]] == 0 for row in splits["test"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "wt", encoding="utf-8") as handle:
        for split in ("train", "valid", "test"):
            for user_id, item_id, rating, timestamp in splits[split]:
                handle.write(
                    f"{split}\t{user_id}\t{item_id}\t{rating}\t{timestamp}\n"
                )

    report = {
        "source_filter": source_filter,
        "deduplication": "latest interaction per (user_id, parent_asin)",
        "positive_definition": "rating >= 4",
        "user_k": args.user_k,
        "item_k": args.item_k,
        "kcore_rounds": rounds,
        "users": len(grouped),
        "items": len({row[1] for row in core_rows}),
        "positive_interactions": len(core_rows),
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["valid"]),
        "test_rows": len(splits["test"]),
        "validation_cold_targets": valid_cold,
        "test_cold_targets": test_cold,
        "suggested_reranker_targets_per_user": 3,
        "suggested_reranker_training_rows": len(grouped) * 3,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
