#!/usr/bin/env python3
"""Build natural-hit conditional training groups from three temporal cutoffs."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .lightgcn import (
    Interaction,
    cohort_order,
    deterministic_shuffle,
    group_rows,
    read_items,
    score_all,
    train_lightgcn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--item-metadata", type=Path, required=True)
    parser.add_argument("--source-train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=603759240)
    parser.add_argument("--validation-users", type=int, default=1000)
    parser.add_argument("--test-users", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_interactions(path: Path) -> list[Interaction]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[Interaction] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            split, user, item, rating, timestamp = line.rstrip("\n").split("\t")
            rows.append(Interaction(split, user, item, float(rating), int(timestamp)))
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    interactions = read_interactions(args.interactions)
    grouped = group_rows(interactions)
    catalog, _ = read_items(args.item_metadata)
    order = cohort_order(grouped, args.seed)
    validation = set(order[: args.validation_users])
    test = set(order[args.validation_users : args.validation_users + args.test_users])
    external_eval = validation | test

    sources = read_jsonl(args.source_train_data)
    source_by_key = {(str(row["user_id"]), str(row["target_item_id"])): row for row in sources}
    if len(source_by_key) != len(sources):
        raise ValueError("duplicate user/target rows in source training data")

    selected: list[dict[str, Any]] = []
    round_reports: list[dict[str, Any]] = []
    for cutoff in range(1, args.cutoffs + 1):
        graph: list[Interaction] = []
        specs: list[tuple[str, Interaction, list[Interaction], set[str]]] = []
        for user_id, all_rows in grouped.items():
            eligible = all_rows[:-1] if user_id in external_eval else all_rows
            if len(eligible) < cutoff + 1:
                continue
            target = eligible[-cutoff]
            history = eligible[:-cutoff]
            # Include the final evaluation holdout among future positives.
            future_other = {row.item_id for row in all_rows[len(eligible) - cutoff + 1 :]}
            graph.extend(history)
            specs.append((user_id, target, history, future_other))

        model, user_index, item_index, train_report = train_lightgcn(
            graph, catalog, dim=64, layers=2, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=1e-3, weight_decay=1e-4,
            dns_k=32, seed=7 + cutoff, device_name=args.device,
            label=f"natural_hit_cutoff_{cutoff}",
        )
        scores, score_catalog = score_all(model, user_index, item_index)
        hits = 0
        retriever_ranks: list[int] = []
        for user_id, target, history, future_other in specs:
            excluded = {row.item_id for row in history} | future_other
            order_indices = scores[user_id].argsort(descending=True).tolist()
            ranking = [score_catalog[index] for index in order_indices if score_catalog[index] not in excluded]
            rank = ranking.index(target.item_id) + 1
            if rank > 10:
                continue
            source = source_by_key.get((user_id, target.item_id))
            if source is None:
                raise KeyError(f"missing source row for cutoff={cutoff} user={user_id}")
            ranked_top10 = ranking[:10]
            if set(ranked_top10) & ({row.item_id for row in history} | future_other):
                raise ValueError(f"seen/future item in Top-10: {user_id}")
            display = deterministic_shuffle(ranked_top10, args.seed, f"conditional:{source['sample_id']}")
            score_map = {item: float(scores[user_id][item_index[item]].item()) for item in display}
            row = dict(source)
            row.update({
                "split": "train",
                "candidate_item_ids": display,
                "candidate_scores": [score_map[item] for item in display],
                "lightgcn_ranked_top10_item_ids": ranked_top10,
                "lightgcn_top5_item_ids": ranked_top10[:5],
                "target_retriever_rank": rank,
                "target_in_lightgcn_top10": True,
                "target_injected": False,
                "natural_hit_cutoff": cutoff,
                "variant": "natural_hit_multicut_conditional_v1",
            })
            selected.append(row)
            hits += 1
            retriever_ranks.append(rank)
        round_reports.append({
            "cutoff": cutoff,
            "eligible_events": len(specs),
            "natural_hits": hits,
            "recall_at_10": hits / len(specs),
            "mean_hit_rank": sum(retriever_ranks) / len(retriever_ranks) if retriever_ranks else None,
            "lightgcn": train_report,
        })
        del model, scores

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    users = Counter(str(row["user_id"]) for row in selected)
    report = {
        "variant": "natural_hit_multicut_conditional_strict_future_mask_v1",
        "future_mask_policy": "all known positive interactions after each training target, including final evaluation holdout",
        "cutoffs": args.cutoffs,
        "groups": len(selected),
        "pointwise_rows_at_1_to_1": len(selected) * 2,
        "unique_users": len(users),
        "groups_per_user_distribution": dict(sorted(Counter(users.values()).items())),
        "rounds": round_reports,
        "checks": {
            "all_targets_naturally_retrieved": all(not row["target_injected"] for row in selected),
            "all_groups_have_ten_unique_candidates": all(len(row["candidate_item_ids"]) == len(set(row["candidate_item_ids"])) == 10 for row in selected),
            "all_targets_in_candidates": all(row["target_item_id"] in row["candidate_item_ids"] for row in selected),
            "all_sources_found": len(selected) > 0,
        },
    }
    write_jsonl(args.output_dir / "natural_hit_groups.jsonl", selected)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
