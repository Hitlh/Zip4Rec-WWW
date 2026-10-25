"""Normalize validated offline item summaries and history-only user profiles."""
from pathlib import Path
from typing import Any
import json

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def format_preferences(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none established"
    return ", ".join(
        f"{row['name']} ({row['strength']}, evidence={row['evidence_count']})"
        for row in rows
    )


def format_profile(profile: dict[str, Any]) -> str:
    lines = [
        f"Summary: {profile['preference_summary']}",
        f"Preferred genres: {format_preferences(profile['preferred_genres'])}",
        "Preferred artists/scenes: "
        + format_preferences(profile["preferred_artists_or_scenes"]),
        f"Preferred eras: {', '.join(profile['preferred_eras']) or 'unknown'}",
        "Moods/styles: "
        + (", ".join(profile["preferred_moods_or_styles"]) or "unknown"),
        f"Diversity: {profile['diversity']}",
        f"Evidence strength: {profile['evidence_strength']}",
    ]
    if profile["uncertainties"]:
        lines.append(f"Uncertainties: {'; '.join(profile['uncertainties'])}")
    return "\n".join(lines)


def load_profiles(path: Path) -> dict[str, str]:
    profiles: dict[str, str] = {}
    for row in read_jsonl(path):
        key = str(row["profile_history_key"])
        value = format_profile(row["profile"])
        previous = profiles.setdefault(key, value)
        if previous != value:
            raise ValueError(f"conflicting validated user profile for {key}")
    return profiles


def load_validated_items(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        item_id = str(row["original_item_id"])
        summary = row["summary"]
        value = {
            "title": summary.get("canonical_title") or row["title"],
            "artists": summary.get("artists") or [],
            "genres": summary.get("genres") or [],
            "era": summary.get("era") or "",
            "moods": summary.get("styles_or_moods") or [],
            "summary": summary["compact_summary"],
        }
        previous = items.setdefault(item_id, value)
        if previous != value:
            raise ValueError(f"conflicting validated item summary for {item_id}")
    return items
