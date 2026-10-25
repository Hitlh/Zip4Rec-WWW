#!/usr/bin/env python3
"""Build the frozen Amazon Reviews 2023 CD user-disjoint temporal protocol.

This stage is deliberately independent of GPT assets and DensePage rendering. It
freezes cohorts, temporal targets, LightGCN graphs/rankings, candidate sets, and
profile-history contexts so later asset generation cannot influence selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


PROTOCOL = "amazon2023_cds_vinyl_user_disjoint_temporal_leave_one_out_v1"


@dataclass(frozen=True)
class Interaction:
    source_split: str
    user_id: str
    item_id: str
    rating: float
    timestamp: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--item-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--validation-users", type=int, default=1000)
    parser.add_argument("--test-users", type=int, default=1000)
    parser.add_argument("--sft-targets-per-user", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dns-k", type=int, default=32)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--evaluation-splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
        help="Use validation-only during retriever selection; include test only after freezing config.",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def exclusive_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_interactions(path: Path) -> list[Interaction]:
    rows: list[Interaction] = []
    seen: set[tuple[str, str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                raise ValueError(f"expected five TSV columns at {path}:{line_number}")
            split, user_id, item_id, rating, timestamp = parts
            row = Interaction(split, user_id, item_id, float(rating), int(timestamp))
            identity = (user_id, item_id, row.timestamp)
            if identity in seen:
                raise ValueError(f"duplicate user/item/timestamp at {path}:{line_number}")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError("interaction file is empty")
    return rows


def group_rows(rows: Iterable[Interaction]) -> dict[str, list[Interaction]]:
    grouped: dict[str, list[Interaction]] = defaultdict(list)
    for row in rows:
        grouped[row.user_id].append(row)
    for user_rows in grouped.values():
        user_rows.sort(key=lambda value: (value.timestamp, value.item_id))
    return dict(grouped)


def read_items(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"item_id", "original_item_id", "title"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"item metadata must contain {sorted(required)}")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        item_id = row["original_item_id"]
        if item_id in by_id:
            raise ValueError(f"duplicate metadata item: {item_id}")
        by_id[item_id] = row
    return sorted(by_id), by_id


def cohort_order(user_ids: Iterable[str], seed: int) -> list[str]:
    def key(user_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{user_id}".encode("utf-8")).hexdigest()
        return digest, user_id
    return sorted(user_ids, key=key)


def set_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return sha256_bytes(payload)


def normalized_adjacency(
    edge_users: torch.Tensor,
    edge_items: torch.Tensor,
    user_count: int,
    item_count: int,
    device: torch.device,
) -> torch.Tensor:
    item_nodes = edge_items + user_count
    row = torch.cat([edge_users, item_nodes])
    col = torch.cat([item_nodes, edge_users])
    degree = torch.bincount(row, minlength=user_count + item_count).float()
    inv = degree.clamp_min(1).pow(-0.5)
    values = inv[row] * inv[col]
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), values, (user_count + item_count,) * 2, device=device
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(self, users: int, items: int, dim: int, layers: int, adjacency: torch.Tensor):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        self.layers = layers
        self.adjacency = adjacency
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        layers = [value]
        for _ in range(self.layers):
            value = torch.sparse.mm(self.adjacency, value)
            layers.append(value)
        final = torch.stack(layers, dim=1).mean(dim=1)
        users = self.user_embedding.num_embeddings
        return final[:users], final[users:]


def train_lightgcn(
    rows: list[Interaction],
    catalog: list[str],
    *,
    dim: int,
    layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dns_k: int,
    seed: int,
    device_name: str,
    label: str,
) -> tuple[LightGCN, dict[str, int], dict[str, int], dict[str, Any]]:
    if not rows:
        raise ValueError(f"{label} LightGCN graph is empty")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    users = sorted({row.user_id for row in rows})
    user_index = {value: index for index, value in enumerate(users)}
    item_index = {value: index for index, value in enumerate(catalog)}
    edge_users = torch.tensor([user_index[row.user_id] for row in rows], dtype=torch.long)
    edge_items = torch.tensor([item_index[row.item_id] for row in rows], dtype=torch.long)
    positives: list[set[int]] = [set() for _ in users]
    for user, item in zip(edge_users.tolist(), edge_items.tolist()):
        positives[user].add(item)
    adjacency = normalized_adjacency(
        edge_users.to(device), edge_items.to(device), len(users), len(catalog), device
    )
    model = LightGCN(len(users), len(catalog), dim, layers, adjacency).to(device)
    # ``nn.Embedding`` performs its own initialization before our Xavier call,
    # consuming RNG state. Reset here so the final weights exactly match the
    # legacy Parameter-based DNS trainer for a fixed model seed.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    nn.init.xavier_uniform_(model.user_embedding.weight)
    nn.init.xavier_uniform_(model.item_embedding.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)
    edge_count = len(rows)
    edge_users_device = edge_users.to(device)
    edge_items_device = edge_items.to(device)
    positive_keys = torch.unique(
        edge_users_device * len(catalog) + edge_items_device, sorted=True
    )

    def collision_mask(users_tensor: torch.Tensor, items_tensor: torch.Tensor) -> torch.Tensor:
        keys = users_tensor * len(catalog) + items_tensor
        positions = torch.searchsorted(positive_keys, keys)
        safe = positions.clamp_max(positive_keys.numel() - 1)
        return (positions < positive_keys.numel()) & (positive_keys[safe] == keys)

    def dns_negatives(
        users_tensor: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.randint(
            len(catalog),
            (users_tensor.numel(), dns_k),
            device=device,
            generator=generator,
        )
        expanded_users = users_tensor[:, None].expand_as(candidates)
        bad = collision_mask(
            expanded_users.reshape(-1), candidates.reshape(-1)
        ).reshape_as(candidates)
        scores = (
            user_embeddings[users_tensor, None] * item_embeddings[candidates]
        ).sum(dim=2)
        scores[bad] = -torch.inf
        all_bad = bad.all(dim=1)
        while all_bad.any():
            count = int(all_bad.sum())
            replacement = torch.randint(
                len(catalog), (count,), device=device, generator=generator
            )
            replacement_users = users_tensor[all_bad]
            valid = ~collision_mask(replacement_users, replacement)
            row_indices = torch.where(all_bad)[0][valid]
            candidates[row_indices, 0] = replacement[valid]
            scores[row_indices, 0] = (
                user_embeddings[users_tensor[row_indices]]
                * item_embeddings[replacement[valid]]
            ).sum(dim=1)
            all_bad = torch.isneginf(scores).all(dim=1)
        return candidates.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)

    last_loss = math.nan
    for epoch in range(epochs):
        permutation = torch.randperm(edge_count, device=device, generator=generator)
        total_loss = 0.0
        steps = 0
        for start in range(0, edge_count, batch_size):
            selection = permutation[start : start + batch_size]
            batch_users = edge_users_device[selection]
            batch_items = edge_items_device[selection]
            user_final, item_final = model.propagate()
            negatives = dns_negatives(
                batch_users, user_final.detach(), item_final.detach()
            )
            positive_score = (user_final[batch_users] * item_final[batch_items]).sum(dim=1)
            negative_score = (user_final[batch_users] * item_final[negatives]).sum(dim=1)
            ranking_loss = -F.logsigmoid(positive_score - negative_score).mean()
            regularization = (
                model.user_embedding(batch_users).square().sum()
                + model.item_embedding(batch_items).square().sum()
                + model.item_embedding(negatives).square().sum()
            ) / batch_users.numel()
            loss = ranking_loss + weight_decay * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        last_loss = total_loss / max(1, steps)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(json.dumps({"lightgcn": label, "epoch": epoch + 1, "loss": last_loss}), flush=True)
    model.eval()
    report = {
        "label": label,
        "users": len(users),
        "items": len(catalog),
        "edges": len(rows),
        "embedding_dim": dim,
        "layers": layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dns_k": dns_k,
        "seed": seed,
        "final_epoch_loss": last_loss,
    }
    return model, user_index, item_index, report


@torch.no_grad()
def score_all(
    model: LightGCN, user_index: dict[str, int], item_index: dict[str, int]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    users, items = model.propagate()
    users = users.detach().cpu()
    items = items.detach().cpu()
    catalog = [""] * len(item_index)
    for item_id, index in item_index.items():
        catalog[index] = item_id
    return {user_id: users[index] @ items.T for user_id, index in user_index.items()}, catalog


def ranked_catalog(scores: torch.Tensor, catalog: list[str], excluded: set[str]) -> list[str]:
    order = torch.argsort(scores, descending=True).tolist()
    return [catalog[index] for index in order if catalog[index] not in excluded]


def deterministic_candidates(target: str, negatives: list[str], seed: int, sample_id: str) -> list[str]:
    candidates = [target, *negatives]
    return deterministic_shuffle(candidates, seed, sample_id)


def deterministic_shuffle(items: list[str], seed: int, identity: str) -> list[str]:
    result = list(items)
    rng = random.Random(int(sha256_bytes(f"{seed}\0{identity}".encode())[:16], 16))
    rng.shuffle(result)
    # A no-hint display must never accidentally preserve the retriever order.
    if len(result) > 1 and result == items:
        result = result[1:] + result[:1]
    return result


def history_payload(rows: list[Interaction]) -> dict[str, Any]:
    item_ids = [row.item_id for row in rows]
    recent = item_ids[-3:]
    long_history = item_ids[:-3]
    key = sha256_json({"ordered_item_ids": long_history})
    return {
        "history_item_ids": item_ids,
        "long_history_item_ids": long_history,
        "recent_history_item_ids": recent,
        "profile_history_key": key,
    }


def checkpoint_payload(
    model: LightGCN,
    user_index: dict[str, int],
    item_index: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        final_users, final_items = model.propagate()
    return {
        "state_dict": model.state_dict(),
        "final_user_embeddings": final_users.detach().cpu(),
        "final_item_embeddings": final_items.detach().cpu(),
        "user_index": user_index,
        "item_index": item_index,
        "config": config,
    }


def main() -> int:
    args = parse_args()
    if args.candidate_count != 10:
        raise ValueError("the frozen protocol requires candidate_count=10")
    rows = read_interactions(args.interactions)
    grouped = group_rows(rows)
    catalog, metadata = read_items(args.item_metadata)
    observed_items = {row.item_id for row in rows}
    if observed_items != set(catalog):
        raise ValueError(
            f"interaction/catalog mismatch: missing={len(observed_items-set(catalog))}, "
            f"unused={len(set(catalog)-observed_items)}"
        )
    user_order = cohort_order(grouped, args.seed)
    required_users = args.validation_users + args.test_users
    if len(user_order) <= required_users:
        raise ValueError("not enough users for validation, test, and train-only cohorts")
    validation = set(user_order[: args.validation_users])
    test = set(user_order[args.validation_users : required_users])
    train_only = set(user_order[required_users:])
    cohorts = {**{u: "validation" for u in validation}, **{u: "test" for u in test}, **{u: "train_only" for u in train_only}}
    exclusive_dir(args.output_dir)

    manifest_rows = []
    for user_id in user_order:
        manifest_rows.append({
            "protocol": PROTOCOL,
            "user_id": user_id,
            "cohort": cohorts[user_id],
            "stable_hash": sha256_bytes(f"{args.seed}\0{user_id}".encode("utf-8")),
            "interaction_count": len(grouped[user_id]),
        })
    write_jsonl(args.output_dir / "user_cohort_manifest.jsonl", manifest_rows)
    cohort_report = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "hash_definition": "SHA256(UTF8(str(seed) + NUL + raw_anonymous_user_id))",
        "assignment": "ascending stable_hash; first validation, next test, remainder train_only",
        "counts": {"validation": len(validation), "test": len(test), "train_only": len(train_only)},
        "user_set_sha256": {
            "validation": set_sha256(validation),
            "test": set_sha256(test),
            "train_only": set_sha256(train_only),
            "all": set_sha256(grouped),
        },
        "checks": {
            "validation_test_disjoint": validation.isdisjoint(test),
            "validation_train_only_disjoint": validation.isdisjoint(train_only),
            "test_train_only_disjoint": test.isdisjoint(train_only),
            "union_is_all_users": validation | test | train_only == set(grouped),
            "cohorts_frozen_before_retrieval": True,
        },
        "sources": {
            "interactions": str(args.interactions.resolve()),
            "interactions_sha256": sha256_file(args.interactions),
            "item_metadata": str(args.item_metadata.resolve()),
            "item_metadata_sha256": sha256_file(args.item_metadata),
        },
    }
    write_json(args.output_dir / "user_cohort_report.json", cohort_report)

    evaluation_targets: dict[str, Interaction] = {}
    visible_history: dict[str, list[Interaction]] = {}
    evaluation_graph: list[Interaction] = []
    temporary_graph: list[Interaction] = []
    sft_specs: list[tuple[str, int, Interaction, list[Interaction]]] = []
    for user_id, user_rows in grouped.items():
        cohort = cohorts[user_id]
        eligible = user_rows
        if cohort in {"validation", "test"}:
            evaluation_targets[user_id] = user_rows[-1]
            eligible = user_rows[:-1]
            evaluation_graph.extend(eligible)
        else:
            evaluation_graph.extend(user_rows)
        visible_history[user_id] = eligible
        holdout = min(args.sft_targets_per_user, len(eligible))
        prefix = eligible[:-holdout]
        temporary_graph.extend(prefix)
        for offset, target in enumerate(eligible[-holdout:]):
            history = eligible[: len(eligible) - holdout + offset]
            sft_specs.append((user_id, offset, target, history))
    if len(sft_specs) != len(grouped) * args.sft_targets_per_user:
        raise ValueError("unexpected SFT row count; each user must contribute exactly three targets")

    eval_model, eval_users, eval_items, eval_train_report = train_lightgcn(
        evaluation_graph, catalog, dim=args.embedding_dim, layers=args.layers,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, dns_k=args.dns_k, seed=args.model_seed,
        device_name=args.device, label="evaluation",
    )
    eval_scores, eval_catalog = score_all(eval_model, eval_users, eval_items)
    eval_checkpoint = args.output_dir / "lightgcn_eval_checkpoint.pt"
    torch.save(checkpoint_payload(eval_model, eval_users, eval_items, eval_train_report), eval_checkpoint)

    temp_model, temp_users, temp_items, temp_train_report = train_lightgcn(
        temporary_graph, catalog, dim=args.embedding_dim, layers=args.layers,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, dns_k=args.dns_k, seed=args.model_seed + 1,
        device_name=args.device,
        label="sft_hard_negative",
    )
    temp_scores, temp_catalog = score_all(temp_model, temp_users, temp_items)
    temp_checkpoint = args.output_dir / "lightgcn_sft_hard_negative_checkpoint.pt"
    torch.save(checkpoint_payload(temp_model, temp_users, temp_items, temp_train_report), temp_checkpoint)

    all_known_positive = {
        user_id: {row.item_id for row in user_rows} for user_id, user_rows in grouped.items()
    }
    eval_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    requested_evaluation_splits = tuple(dict.fromkeys(args.evaluation_splits))
    recall = {split: 0 for split in requested_evaluation_splits}
    profile_contexts: dict[str, dict[str, Any]] = {}
    split_users = {"validation": validation, "test": test}
    # Generate private profile histories for both frozen evaluation cohorts
    # without scoring or constructing Test candidates before final evaluation.
    for user_id in sorted(validation | test):
        context = history_payload(visible_history[user_id])
        profile_contexts.setdefault(context["profile_history_key"], {
            "profile_history_key": context["profile_history_key"],
            "long_history_item_ids": context["long_history_item_ids"],
        })
    for cohort_name in requested_evaluation_splits:
        users = split_users[cohort_name]
        for user_id in sorted(users):
            target = evaluation_targets[user_id]
            history = visible_history[user_id]
            history_items = {row.item_id for row in history}
            ranking = ranked_catalog(eval_scores[user_id], eval_catalog, history_items)
            ranked_top10 = ranking[: args.candidate_count]
            sample_id = f"{cohort_name}-{sha256_bytes(user_id.encode())[:16]}"
            candidates = deterministic_shuffle(
                ranked_top10, args.seed, f"display:{sample_id}"
            )
            target_rank = ranking.index(target.item_id) + 1
            target_in_top10 = target.item_id in ranked_top10
            recall[cohort_name] += int(target_in_top10)
            context = history_payload(history)
            score_by_item = {item: float(eval_scores[user_id][eval_items[item]].item()) for item in candidates}
            ranked_score_by_item = {
                item: float(eval_scores[user_id][eval_items[item]].item())
                for item in ranked_top10
            }
            record = {
                "protocol": PROTOCOL,
                "split": cohort_name,
                "sample_id": sample_id,
                "user_id": user_id,
                "target_item_id": target.item_id,
                "target_timestamp": target.timestamp,
                "candidate_item_ids": candidates,
                "candidate_scores": [score_by_item[item] for item in candidates],
                "lightgcn_ranked_top10_item_ids": ranked_top10,
                "lightgcn_top5_item_ids": ranked_top10[:5],
                "target_retriever_rank": target_rank,
                "target_in_lightgcn_top10": target_in_top10,
                "target_injected": False,
                **context,
            }
            eval_rows.append(record)
            topk_rows.append({
                "protocol": PROTOCOL,
                "split": cohort_name,
                "sample_id": sample_id,
                "user_id": user_id,
                "target_item_id": target.item_id,
                "target_retriever_rank": target_rank,
                "target_in_lightgcn_top10": target_in_top10,
                "candidate_item_ids": ranked_top10,
                "candidate_scores": [ranked_score_by_item[item] for item in ranked_top10],
                "display_candidate_item_ids": candidates,
                "target_injected": False,
            })
    write_jsonl(args.output_dir / "evaluation_records.jsonl", eval_rows)
    write_jsonl(args.output_dir / "lightgcn_eval_topk.jsonl", topk_rows)

    sft_rows: list[dict[str, Any]] = []
    for sequence_number, (user_id, offset, target, history) in enumerate(sft_specs):
        known = all_known_positive[user_id]
        ranking = ranked_catalog(temp_scores[user_id], temp_catalog, known)
        negatives = ranking[: args.candidate_count - 1]
        if len(negatives) != args.candidate_count - 1:
            raise ValueError(f"too few negatives for user {user_id}")
        sample_id = f"train-{sequence_number:05d}-{sha256_bytes(user_id.encode())[:10]}-{offset + 1}"
        candidates = deterministic_candidates(target.item_id, negatives, args.seed, sample_id)
        score_by_item = {item: float(temp_scores[user_id][temp_items[item]].item()) for item in candidates}
        residual = sorted(
            (item for item in candidates if item != target.item_id),
            key=lambda item: (-score_by_item[item], item),
        )
        answer_items = [target.item_id, *residual[:4]]
        labels = {item: f"C{index + 1}" for index, item in enumerate(candidates)}
        context = history_payload(history)
        profile_contexts.setdefault(context["profile_history_key"], {
            "profile_history_key": context["profile_history_key"],
            "long_history_item_ids": context["long_history_item_ids"],
        })
        sft_rows.append({
            "protocol": PROTOCOL,
            "split": "train",
            "sample_id": sample_id,
            "user_id": user_id,
            "cohort": cohorts[user_id],
            "target_item_id": target.item_id,
            "target_timestamp": target.timestamp,
            "candidate_item_ids": candidates,
            "candidate_scores": [score_by_item[item] for item in candidates],
            "lightgcn_top5_item_ids": [item for item in sorted(candidates, key=lambda item: (-score_by_item[item], item))[:5]],
            "answer_item_ids": answer_items,
            "answer_labels": [labels[item] for item in answer_items],
            "answer": " > ".join(labels[item] for item in answer_items),
            "target_injected": True,
            **context,
        })
    write_jsonl(args.output_dir / "sft_records.jsonl", sft_rows)
    write_jsonl(
        args.output_dir / "profile_contexts.jsonl",
        [profile_contexts[key] for key in sorted(profile_contexts)],
    )

    eval_target_pairs = {(user, row.item_id) for user, row in evaluation_targets.items()}
    eval_graph_pairs = {(row.user_id, row.item_id) for row in evaluation_graph}
    sft_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sft_rows:
        sft_by_user[row["user_id"]].append(row)
    checks = {
        "validation_users_match_configuration": len(validation) == args.validation_users,
        "test_users_match_configuration": len(test) == args.test_users,
        "validation_test_user_disjoint": validation.isdisjoint(test),
        "evaluation_targets_are_last_interactions": all(
            target == grouped[user_id][-1] for user_id, target in evaluation_targets.items()
        ),
        "evaluation_targets_absent_from_lightgcn_graph": eval_target_pairs.isdisjoint(eval_graph_pairs),
        "sft_targets_precede_external_evaluation_targets": all(
            all(row["target_timestamp"] < evaluation_targets[user_id].timestamp for row in sft_by_user[user_id])
            for user_id in validation | test
        ),
        "sft_rows_match_configured_targets": len(sft_rows) == len(grouped) * args.sft_targets_per_user,
        "ten_unique_candidates_everywhere": all(
            len(row["candidate_item_ids"]) == 10 and len(set(row["candidate_item_ids"])) == 10
            for row in [*sft_rows, *eval_rows]
        ),
        "sft_target_in_candidates": all(row["target_item_id"] in row["candidate_item_ids"] for row in sft_rows),
        "sft_target_ranked_first": all(row["answer_item_ids"][0] == row["target_item_id"] for row in sft_rows),
        "sft_negatives_not_known_positives": all(
            all(item == row["target_item_id"] or item not in all_known_positive[row["user_id"]]
                for item in row["candidate_item_ids"])
            for row in sft_rows
        ),
        "eval_candidates_history_disjoint": all(
            set(row["candidate_item_ids"]).isdisjoint(row["history_item_ids"]) for row in eval_rows
        ),
        "eval_target_not_in_history": all(row["target_item_id"] not in row["history_item_ids"] for row in eval_rows),
        "eval_target_not_in_profile_or_recent": all(
            row["target_item_id"] not in row["long_history_item_ids"]
            and row["target_item_id"] not in row["recent_history_item_ids"] for row in eval_rows
        ),
        "eval_target_never_injected": all(not row["target_injected"] for row in eval_rows),
        "eval_display_order_hides_lightgcn_rank": all(
            row["candidate_item_ids"] != row["lightgcn_ranked_top10_item_ids"]
            and set(row["candidate_item_ids"]) == set(row["lightgcn_ranked_top10_item_ids"])
            for row in eval_rows
        ),
        "eval_lightgcn_top5_is_ranked_and_displayed": all(
            row["lightgcn_top5_item_ids"] == row["lightgcn_ranked_top10_item_ids"][:5]
            and set(row["lightgcn_top5_item_ids"]).issubset(row["candidate_item_ids"])
            for row in eval_rows
        ),
        "profile_contexts_cover_frozen_test_without_scoring": all(
            history_payload(visible_history[user])["profile_history_key"] in profile_contexts
            for user in validation | test),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol audit failed: {[key for key, value in checks.items() if not value]}")
    lightgcn_report = {
        "protocol": PROTOCOL,
        "evaluation_graph": eval_train_report,
        "sft_hard_negative_graph": temp_train_report,
        "evaluated_splits": list(requested_evaluation_splits),
        "recall_at_10": {
            split: recall[split] / len(split_users[split])
            for split in requested_evaluation_splits
        },
        "target_in_top10": recall,
        "artifacts": {
            "evaluation_checkpoint": str(eval_checkpoint.resolve()),
            "evaluation_checkpoint_sha256": sha256_file(eval_checkpoint),
            "sft_checkpoint": str(temp_checkpoint.resolve()),
            "sft_checkpoint_sha256": sha256_file(temp_checkpoint),
        },
    }
    write_json(args.output_dir / "lightgcn_eval_report.json", lightgcn_report)
    audit_report = {
        "protocol": PROTOCOL,
        "counts": {
            "users": len(grouped),
            "items": len(catalog),
            "interactions": len(rows),
            "validation_records": sum(row["split"] == "validation" for row in eval_rows),
            "test_records": sum(row["split"] == "test" for row in eval_rows),
            "sft_records": len(sft_rows),
            "profile_history_keys": len(profile_contexts),
        },
        "checks": checks,
        "hashes": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "user_cohort_manifest.jsonl", "user_cohort_report.json",
                "evaluation_records.jsonl", "sft_records.jsonl",
                "profile_contexts.jsonl", "lightgcn_eval_topk.jsonl",
                "lightgcn_eval_report.json",
            )
        },
    }
    write_json(args.output_dir / "protocol_audit_report.json", audit_report)
    print(json.dumps(audit_report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
