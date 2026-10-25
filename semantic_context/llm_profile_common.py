#!/usr/bin/env python3
"""Shared, dependency-light helpers for Amazon-Music LLM profile experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator


SPLITS = ("train", "valid", "test")
ITEM_PROMPT_VERSION = "amazonmusic-item-knowledge-v3"
PROFILE_PROMPT_VERSION = "amazonmusic-user-profile-knowledge-v3"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    if not records and not allow_empty:
        raise ValueError(f"No records found in {path}")
    return records


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, value


def write_jsonl_exclusive(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def distribution(values: Iterable[int]) -> dict[str, int | float]:
    rows = list(values)
    if not rows:
        return {"min": 0, "mean": 0.0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": min(rows),
        "mean": round(sum(rows) / len(rows), 2),
        "p50": percentile(rows, 0.50),
        "p90": percentile(rows, 0.90),
        "p95": percentile(rows, 0.95),
        "p99": percentile(rows, 0.99),
        "max": max(rows),
    }


def content_from_chat_completion_body(body: dict[str, Any]) -> str:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Response body has no choices[0].message") from exc
    refusal = message.get("refusal") if isinstance(message, dict) else None
    if refusal:
        raise ValueError(f"Model refused the request: {refusal}")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ]
        if text_parts:
            return "".join(text_parts)
    raise ValueError("Response message has no textual content")


def response_body_from_result(row: dict[str, Any]) -> dict[str, Any]:
    """Read a synchronous request result or a direct chat-completion body."""
    response = row.get("response")
    if isinstance(response, dict):
        status_code = response.get("status_code")
        if status_code is not None and int(status_code) != 200:
            raise ValueError(f"Request returned HTTP {status_code}")
        body = response.get("body")
        if isinstance(body, dict):
            return body
    if "choices" in row:
        return row
    raise ValueError("Result has neither response.body nor a chat-completion body")


def parse_json_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model content is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Model JSON output must be an object")
    return value


def require_exact_keys(value: dict[str, Any], keys: set[str], *, context: str) -> None:
    missing = keys.difference(value)
    extra = set(value).difference(keys)
    if missing or extra:
        raise ValueError(f"{context} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


def require_string(value: Any, *, context: str, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    result = " ".join(value.split())
    if not allow_empty and not result:
        raise ValueError(f"{context} must not be empty")
    return result


def require_string_list(
    value: Any, *, context: str, maximum: int, deduplicate: bool = False
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{context} must be a list with at most {maximum} entries")
    result = [require_string(item, context=f"{context}[]", allow_empty=False) for item in value]
    if len(set(result)) != len(result):
        if not deduplicate:
            raise ValueError(f"{context} contains duplicate entries")
        result = list(dict.fromkeys(result))
    return result


ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical_title": {"type": "string"},
        "item_type": {
            "type": "string",
            "enum": ["album", "track", "artist_collection", "other", "uncertain"],
        },
        "artists": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "genres": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "styles_or_moods": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "era": {"type": "string"},
        "grounded_summary": {"type": "string"},
        "knowledge_enrichment": {"type": "string"},
        "compact_summary": {"type": "string"},
        "knowledge_used": {"type": "boolean"},
        "uncertainties": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
    "required": [
        "canonical_title", "item_type", "artists", "genres", "styles_or_moods", "era",
        "grounded_summary", "knowledge_enrichment", "compact_summary", "knowledge_used",
        "uncertainties",
    ],
}


PROFILE_PREFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
        "evidence_count": {"type": "integer"},
    },
    "required": ["name", "strength", "evidence_count"],
}


PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "preference_summary": {"type": "string"},
        "preferred_genres": {
            "type": "array", "items": PROFILE_PREFERENCE_SCHEMA, "maxItems": 6,
        },
        "preferred_artists_or_scenes": {
            "type": "array", "items": PROFILE_PREFERENCE_SCHEMA, "maxItems": 6,
        },
        "preferred_eras": {
            "type": "array", "items": {"type": "string"}, "maxItems": 5,
        },
        "preferred_moods_or_styles": {
            "type": "array", "items": {"type": "string"}, "maxItems": 6,
        },
        "diversity": {"type": "string", "enum": ["narrow", "moderate", "broad", "unknown"]},
        "evidence_strength": {
            "type": "string", "enum": ["strong", "moderate", "sparse"],
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": [
        "preference_summary", "preferred_genres", "preferred_artists_or_scenes",
        "preferred_eras", "preferred_moods_or_styles", "diversity", "evidence_strength",
        "uncertainties",
    ],
}


def structured_response_format(name: str, schema: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "json-schema":
        return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    if mode == "json-object":
        return {"type": "json_object"}
    raise ValueError(f"Unsupported response format mode: {mode}")


def validate_item_summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = set(ITEM_SCHEMA["required"])
    require_exact_keys(value, keys, context="item summary")
    item_type = require_string(value["item_type"], context="item_type", allow_empty=False)
    if item_type not in {"album", "track", "artist_collection", "other", "uncertain"}:
        raise ValueError(f"Invalid item_type: {item_type}")
    result = {
        "canonical_title": require_string(value["canonical_title"], context="canonical_title", allow_empty=False),
        "item_type": item_type,
        "artists": require_string_list(value["artists"], context="artists", maximum=5),
        "genres": require_string_list(value["genres"], context="genres", maximum=6),
        "styles_or_moods": require_string_list(value["styles_or_moods"], context="styles_or_moods", maximum=6),
        "era": require_string(value["era"], context="era"),
        "grounded_summary": require_string(value["grounded_summary"], context="grounded_summary", allow_empty=False),
        "knowledge_enrichment": require_string(value["knowledge_enrichment"], context="knowledge_enrichment"),
        "compact_summary": require_string(value["compact_summary"], context="compact_summary", allow_empty=False),
        "knowledge_used": value["knowledge_used"],
        "uncertainties": require_string_list(value["uncertainties"], context="uncertainties", maximum=5),
    }
    if not isinstance(result["knowledge_used"], bool):
        raise ValueError("knowledge_used must be boolean")
    if len(result["compact_summary"].split()) > 60:
        raise ValueError("compact_summary exceeds 60 whitespace-delimited words")
    return result


def _validate_preferences(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 6:
        raise ValueError(f"{context} must be a list with at most 6 entries")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{context}[{index}] must be an object")
        require_exact_keys(row, {"name", "strength", "evidence_count"}, context=f"{context}[{index}]")
        name = require_string(row["name"], context=f"{context}[{index}].name", allow_empty=False)
        strength = require_string(row["strength"], context=f"{context}[{index}].strength", allow_empty=False)
        count = row["evidence_count"]
        if strength not in {"strong", "moderate", "weak"}:
            raise ValueError(f"Invalid strength in {context}[{index}]")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"Invalid evidence_count in {context}[{index}]")
        result.append({"name": name, "strength": strength, "evidence_count": count})
    return result


def validate_user_profile(value: dict[str, Any], *, long_history_count: int) -> dict[str, Any]:
    keys = set(PROFILE_SCHEMA["required"])
    require_exact_keys(value, keys, context="user profile")
    diversity = require_string(value["diversity"], context="diversity", allow_empty=False)
    evidence_strength = require_string(
        value["evidence_strength"], context="evidence_strength", allow_empty=False
    )
    if diversity not in {"narrow", "moderate", "broad", "unknown"}:
        raise ValueError(f"Invalid diversity: {diversity}")
    if evidence_strength not in {"strong", "moderate", "sparse"}:
        raise ValueError(f"Invalid evidence_strength: {evidence_strength}")
    result = {
        "preference_summary": require_string(value["preference_summary"], context="preference_summary", allow_empty=False),
        "preferred_genres": _validate_preferences(value["preferred_genres"], context="preferred_genres"),
        "preferred_artists_or_scenes": _validate_preferences(
            value["preferred_artists_or_scenes"], context="preferred_artists_or_scenes"
        ),
        "preferred_eras": require_string_list(value["preferred_eras"], context="preferred_eras", maximum=5),
        "preferred_moods_or_styles": require_string_list(
            value["preferred_moods_or_styles"], context="preferred_moods_or_styles", maximum=6
        ),
        "diversity": diversity,
        "evidence_strength": evidence_strength,
        "uncertainties": require_string_list(
            value["uncertainties"], context="uncertainties", maximum=6, deduplicate=True
        ),
    }
    for section in ("preferred_genres", "preferred_artists_or_scenes"):
        if any(row["evidence_count"] > long_history_count for row in result[section]):
            raise ValueError(f"{section} evidence_count exceeds long history length")
        if any(
            row["evidence_count"] == 1 and row["strength"] != "weak"
            for row in result[section]
        ):
            raise ValueError(
                f"{section} entries with one supporting history item must use weak strength"
            )
    if long_history_count <= 2 and result["evidence_strength"] != "sparse":
        raise ValueError("Histories with at most two long-term items must use sparse evidence_strength")
    if len(result["preference_summary"].split()) > 90:
        raise ValueError("preference_summary exceeds 90 whitespace-delimited words")
    return result


def item_system_prompt() -> str:
    return (
        "You create compact music-item profiles for a recommendation experiment. Return JSON only. "
        "Use the supplied title, categories, brand, and cleaned description as primary evidence. You may "
        "use reliable prior knowledge to identify artists, genres, styles, eras, or musical context; this "
        "knowledge enrichment is intentional. Never add ratings, popularity, sales, reviews, or claims about "
        "a user's preference. Treat ambiguous titles conservatively, state uncertainty, and do not guess an "
        "artist or release identity when multiple works plausibly match. grounded_summary must only describe "
        "input-supported facts and is a closed-book compression: never use memory to identify an artist, year, "
        "release type, track, scene, influence, or musical fact in grounded_summary unless that fact is literally "
        "present in the supplied fields. When cleaned_description is empty, grounded_summary may only restate the "
        "literal title, categories, and brand. Put every fact added from prior knowledge exclusively in "
        "knowledge_enrichment, or leave it empty. Before returning, verify that grounded_summary remains true if "
        "all prior knowledge is removed. "
        "compact_summary may combine both but must be self-contained and at most 60 words. Use empty arrays or "
        "empty strings for unknown fields."
    )


def item_user_prompt(source: dict[str, Any]) -> str:
    payload = {
        "title": source.get("title", ""),
        "categories": source.get("categories", ""),
        "brand": source.get("brand", ""),
        "cleaned_description": source.get("description_clean", ""),
    }
    return "Create a JSON music-item profile for this input:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def profile_system_prompt() -> str:
    return (
        "You infer a long-term music preference profile from an ordered sequence of positively rated history "
        "items. Return JSON only. The item summaries may contain explicitly allowed prior-knowledge enrichment. "
        "Use that musical knowledge to interpret recurring artists, genres, eras, scenes, styles, and moods. "
        "Infer only preferences supported by the supplied history; do not invent demographics, location, age, "
        "personality, dislikes, or unseen interests. evidence_count is the number of distinct supplied history "
        "items supporting that preference. Every entry with evidence_count=1 MUST use strength=weak; it must "
        "never be moderate or strong. Repeated evidence is required for moderate or strong strength. Evidence "
        "counts must not exceed the number of supplied items. Preserve uncertainty "
        "for ambiguous items. The output is a descriptive preference profile, not downstream ranking guidance; "
        "do not mention recommenders, collaborative filtering, candidates, or ranking hints. Keep "
        "preference_summary within 80 words. For one or two history items, evidence_strength must be sparse."
    )


def profile_user_prompt(items: list[dict[str, Any]]) -> str:
    rows = [
        {
            "position": index,
            "title": item["title"],
            "compact_summary": item["compact_summary"],
            "uncertainties": item.get("uncertainties", []),
        }
        for index, item in enumerate(items, start=1)
    ]
    return (
        "The following positive-history items are ordered from oldest to newest. Create the JSON long-term "
        "preference profile. Do not mention record IDs or positions in the output.\n"
        + json.dumps(rows, ensure_ascii=False, indent=2)
    )


def make_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    response_format_mode: str,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": structured_response_format(schema_name, schema, response_format_mode),
    }
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
