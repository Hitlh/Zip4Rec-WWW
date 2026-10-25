#!/usr/bin/env python3
"""Run portable Chat Completions JSONL against an OpenAI-compatible relay.

Supports resumable synchronous execution and retry request preparation.
No API key is accepted on the command line; credentials are read from an
environment variable to avoid leaking them through process listings or logs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .llm_profile_common import iter_jsonl, read_jsonl, sha256_file


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Run resumable synchronous requests")
    sync.add_argument("--requests", type=Path, required=True)
    sync.add_argument("--output", type=Path, required=True)
    sync.add_argument("--limit", type=int)
    sync.add_argument("--max-attempts", type=int, default=5)
    sync.add_argument("--min-interval", type=float, default=0.0)
    sync.add_argument("--workers", type=int, default=1)

    retry = subparsers.add_parser(
        "retry-input", help="Create a request JSONL containing missing or failed result IDs"
    )
    retry.add_argument("--requests", type=Path, required=True)
    retry.add_argument("--results", type=Path, required=True)
    retry.add_argument(
        "--validation-failures",
        type=Path,
        help="Optional failures.jsonl from parse-items or parse-profiles",
    )
    retry.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def credentials(args: argparse.Namespace) -> tuple[str, str]:
    base_url = os.environ.get(args.base_url_env, "").strip().rstrip("/")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not base_url:
        raise RuntimeError(f"Environment variable {args.base_url_env} is not set")
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")
    return base_url, api_key


def endpoint_url(base_url: str, path: str) -> str:
    normalized_path = "/" + path.lstrip("/")
    if base_url.endswith("/v1") and normalized_path.startswith("/v1/"):
        normalized_path = normalized_path[3:]
    return base_url + normalized_path


def client_for(args: argparse.Namespace) -> tuple[httpx.Client, str]:
    base_url, api_key = credentials(args)
    client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(args.timeout),
        follow_redirects=True,
    )
    return client, base_url


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for _line_number, row in iter_jsonl(path):
        custom_id = row.get("custom_id")
        if isinstance(custom_id, str):
            if custom_id in completed:
                raise ValueError(f"Duplicate custom_id in existing output: {custom_id}")
            completed.add(custom_id)
    return completed


def response_row(custom_id: str, response: httpx.Response) -> dict[str, Any]:
    try:
        body: Any = response.json()
    except ValueError:
        body = {"raw_text": response.text[:10000]}
    error = None
    if response.status_code != 200:
        error = {
            "code": "http_error",
            "status_code": response.status_code,
            "message": body,
        }
    return {
        "id": response.headers.get("x-request-id"),
        "custom_id": custom_id,
        "response": {
            "status_code": response.status_code,
            "request_id": response.headers.get("x-request-id"),
            "body": body,
        },
        "error": error,
    }


def exception_row(custom_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "id": None,
        "custom_id": custom_id,
        "response": None,
        "error": {"code": "client_exception", "message": f"{type(exc).__name__}: {exc}"},
    }


def run_sync(args: argparse.Namespace) -> int:
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    requests = read_jsonl(args.requests)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        requests = requests[: args.limit]
    completed = read_completed_ids(args.output)
    request_ids = {row["custom_id"] for row in requests}
    unexpected_completed = completed - request_ids
    if unexpected_completed:
        raise ValueError(
            "Existing output contains IDs outside the selected request file: "
            f"{sorted(unexpected_completed)[:20]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "x"
    client, base_url = client_for(args)
    written = 0
    failures = 0
    started = time.monotonic()

    def execute_request(index: int, request: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"Request {index} has no custom_id")
        method = request.get("method", "POST")
        if method != "POST":
            raise ValueError(f"Request {custom_id} uses unsupported method {method}")
        url = endpoint_url(base_url, request["url"])
        final_row: dict[str, Any] | None = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = client.post(url, json=request["body"])
                if response.status_code in RETRYABLE_STATUS and attempt < args.max_attempts:
                    retry_after = response.headers.get("retry-after")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else min(30.0, 2 ** (attempt - 1) + random.random())
                    )
                    time.sleep(delay)
                    continue
                final_row = response_row(custom_id, response)
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == args.max_attempts:
                    final_row = exception_row(custom_id, exc)
                    break
                time.sleep(min(30.0, 2 ** (attempt - 1) + random.random()))
        assert final_row is not None
        final_row["provider_metadata"] = {
            "base_url_host": urlsplit(base_url).netloc,
            "execution_mode": "synchronous",
            "workers": args.workers,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return index, custom_id, final_row

    pending = [
        (index, request)
        for index, request in enumerate(requests, start=1)
        if request.get("custom_id") not in completed
    ]
    try:
        with args.output.open(mode, encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures: dict[Future[tuple[int, str, dict[str, Any]]], int] = {
                    pool.submit(execute_request, index, request): index
                    for index, request in pending
                }
                for future in as_completed(futures):
                    index, custom_id, final_row = future.result()
                    handle.write(json.dumps(final_row, ensure_ascii=False) + "\n")
                    handle.flush()
                    written += 1
                    failures += int(bool(final_row.get("error")))
                    print(
                        json.dumps(
                            {
                                "progress": f"{len(completed) + written}/{len(requests)}",
                                "request_index": index,
                                "custom_id": custom_id,
                                "status": "failed" if final_row.get("error") else "ok",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if args.min_interval > 0:
                        time.sleep(args.min_interval)
    finally:
        client.close()
    summary = {
        "requests_file": str(args.requests.resolve()),
        "requests_sha256": sha256_file(args.requests),
        "selected": len(requests),
        "already_completed": len(completed.intersection({row["custom_id"] for row in requests})),
        "written": written,
        "failures_written": failures,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 2


def require_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def retry_input(args: argparse.Namespace) -> int:
    require_new_file(args.output)
    requests = read_jsonl(args.requests)
    results = read_jsonl(args.results, allow_empty=True)
    result_by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError("Every result row must contain custom_id")
        if custom_id in result_by_id:
            raise ValueError(f"Duplicate result custom_id: {custom_id}")
        result_by_id[custom_id] = row
    validation_failure_ids: set[str] = set()
    if args.validation_failures:
        for row in read_jsonl(args.validation_failures, allow_empty=True):
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError("Every validation failure row must contain custom_id")
            validation_failure_ids.add(custom_id)
    request_ids = {row["custom_id"] for row in requests}
    unexpected_validation_ids = validation_failure_ids - request_ids
    if unexpected_validation_ids:
        raise ValueError(
            "Validation failures contain IDs outside the request file: "
            f"{sorted(unexpected_validation_ids)[:20]}"
        )

    retry_rows: list[dict[str, Any]] = []
    reason_counts = {
        "missing": 0,
        "provider_error": 0,
        "http_error": 0,
        "validation_failure": 0,
    }
    for request in requests:
        custom_id = request["custom_id"]
        if custom_id in validation_failure_ids:
            retry_rows.append(request)
            reason_counts["validation_failure"] += 1
            continue
        result = result_by_id.get(custom_id)
        if result is None:
            retry_rows.append(request)
            reason_counts["missing"] += 1
            continue
        if result.get("error"):
            retry_rows.append(request)
            reason_counts["provider_error"] += 1
            continue
        response = result.get("response")
        if not isinstance(response, dict) or int(response.get("status_code", 0)) != 200:
            retry_rows.append(request)
            reason_counts["http_error"] += 1
    with args.output.open("x", encoding="utf-8") as handle:
        for row in retry_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "original_requests": len(requests),
        "result_rows": len(results),
        "validation_failure_rows": len(validation_failure_ids),
        "retry_requests": len(retry_rows),
        "reasons": reason_counts,
        "validation_failures": (
            str(args.validation_failures.resolve()) if args.validation_failures else None
        ),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "sync":
        return run_sync(args)
    if args.command == "retry-input":
        return retry_input(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
