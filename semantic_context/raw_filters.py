"""Original sequential Amazon CD title/description and rating-mix filtering."""
from __future__ import annotations
import gzip, json, time
from collections import Counter, defaultdict
from pathlib import Path

def description_nonempty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(part).strip() for part in value if part is not None)
    return False


def progress(phase: str, count: int, started: float) -> None:
    if count % 1_000_000 == 0:
        print(f"{phase}: {count:,} rows in {time.time() - started:.1f}s", flush=True)


def load_filtered_interactions(
    reviews_path: Path,
    metadata_path: Path,
    minimum_user_interactions: int = 10,
) -> tuple[list[tuple[str, str, int, int]], dict]:
    """Apply the exact sequential filters and return final string-ID rows."""
    if minimum_user_interactions < 3:
        raise ValueError("minimum_user_interactions must be >= 3")
    started = time.time()
    metadata_ids: set[str] = set()
    eligible_metadata_ids: set[str] = set()
    metadata_rows = 0
    metadata_opener = gzip.open if metadata_path.suffix == ".gz" else open
    with metadata_opener(metadata_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            metadata_rows += 1
            item_id = row.get("parent_asin")
            if not item_id:
                continue
            metadata_ids.add(item_id)
            if str(row.get("title") or "").strip() and description_nonempty(
                row.get("description")
            ):
                eligible_metadata_ids.add(item_id)
    print(
        f"metadata: rows={metadata_rows:,} all={len(metadata_ids):,} "
        f"title+description={len(eligible_metadata_ids):,} "
        f"seconds={time.time() - started:.1f}",
        flush=True,
    )

    # Old stage 1/2: metadata join, then the configured initial user minimum.
    user_counts: Counter[str] = Counter()
    seen = 0
    started = time.time()
    reviews_opener = gzip.open if reviews_path.suffix == ".gz" else open
    with reviews_opener(reviews_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            user_id = row.get("user_id")
            item_id = row.get("parent_asin")
            if user_id and item_id in metadata_ids and row.get("rating") is not None:
                user_counts[user_id] += 1
            progress("filter-pass1", seen, started)
    initial_users = {
        user_id
        for user_id, count in user_counts.items()
        if count >= minimum_user_interactions
    }
    del user_counts

    # Old stage 3/4 plus the new description rule: title+description, item > 1.
    item_counts: Counter[str] = Counter()
    title_description_rows = 0
    seen = 0
    started = time.time()
    with reviews_opener(reviews_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            if (
                row.get("user_id") in initial_users
                and row.get("parent_asin") in eligible_metadata_ids
                and row.get("rating") is not None
            ):
                item_counts[row["parent_asin"]] += 1
                title_description_rows += 1
            progress("filter-pass2", seen, started)
    repeated_items = {item_id for item_id, count in item_counts.items() if count > 1}
    del item_counts

    # Old stage 5: at least three five-star and two four-star interactions.
    user_rating_counts: dict[str, list[int]] = defaultdict(lambda: [0] * 6)
    repeated_item_rows = 0
    seen = 0
    started = time.time()
    with reviews_opener(reviews_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            user_id = row.get("user_id")
            item_id = row.get("parent_asin")
            rating_value = row.get("rating")
            if (
                user_id in initial_users
                and item_id in repeated_items
                and item_id in eligible_metadata_ids
                and rating_value is not None
            ):
                rating = int(float(rating_value))
                user_rating_counts[user_id][rating] += 1
                repeated_item_rows += 1
            progress("filter-pass3", seen, started)
    rating_users = {
        user_id
        for user_id, counts in user_rating_counts.items()
        if counts[5] >= 3 and counts[4] >= 2
    }
    del user_rating_counts

    # Old stage 6: users must still satisfy the configured user minimum.
    remaining_user_counts: Counter[str] = Counter()
    rating_mix_rows = 0
    seen = 0
    started = time.time()
    with reviews_opener(reviews_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            user_id = row.get("user_id")
            item_id = row.get("parent_asin")
            if (
                user_id in rating_users
                and item_id in repeated_items
                and item_id in eligible_metadata_ids
                and row.get("rating") is not None
            ):
                remaining_user_counts[user_id] += 1
                rating_mix_rows += 1
            progress("filter-pass4", seen, started)
    final_users = {
        user_id
        for user_id, count in remaining_user_counts.items()
        if count >= minimum_user_interactions
    }
    del remaining_user_counts

    final_rows: list[tuple[str, str, int, int]] = []
    rating_distribution: Counter[int] = Counter()
    final_items: set[str] = set()
    seen = 0
    started = time.time()
    with reviews_opener(reviews_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seen += 1
            user_id = row.get("user_id")
            item_id = row.get("parent_asin")
            rating_value = row.get("rating")
            if (
                user_id in final_users
                and item_id in repeated_items
                and item_id in eligible_metadata_ids
                and rating_value is not None
            ):
                rating = int(float(rating_value))
                timestamp = int(row.get("timestamp") or 0)
                final_rows.append((user_id, item_id, rating, timestamp))
                final_items.add(item_id)
                rating_distribution[rating] += 1
            progress("filter-pass5", seen, started)

    filter_report = {
        "metadata_rows": metadata_rows,
        "metadata_title_description_items": len(eligible_metadata_ids),
        "minimum_user_interactions": minimum_user_interactions,
        "initial_users_meeting_minimum": len(initial_users),
        "initial_users_ge_10": len(initial_users) if minimum_user_interactions == 10 else None,
        "title_description_rows": title_description_rows,
        "repeated_items": len(repeated_items),
        "repeated_item_rows": repeated_item_rows,
        "rating_mix_users": len(rating_users),
        "rating_mix_rows": rating_mix_rows,
        "final_users": len(final_users),
        "final_items": len(final_items),
        "final_rows": len(final_rows),
        "rating_distribution": dict(sorted(rating_distribution.items())),
    }
    print("filter report:", json.dumps(filter_report, ensure_ascii=False), flush=True)
    return final_rows, filter_report
