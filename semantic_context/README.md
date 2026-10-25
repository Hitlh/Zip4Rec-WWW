# Module 1: Data and Semantic Context

This module starts from raw reviews and item metadata and produces item descriptions, user profiles, and training, validation, and test inputs for rendering and training.

## Workflow

1. Filter by item title/description, interaction counts, and rating mix. Keep the latest event for each user-item pair, define ratings >=4 as positive feedback, and apply k-core filtering.
2. Organize histories chronologically and fix disjoint validation/test user cohorts. Train LightGCN and retrieve natural Top-10 candidates without injecting targets. Use three earlier cutoffs for training and mask future positive feedback.
3. Generate structured item descriptions from cleaned metadata. Generate user profiles only from long-term positive history; retain the three most recent positive interactions separately.
4. Validate generated semantics and export the three splits, SFT positive/negative pairs, and ten-candidate GRPO groups. Retain all validation/test users, including retrieval misses, for overall evaluation.

## Input format

Both files use UTF-8 JSONL or JSONL.GZ, with one object per line. Join item IDs through `parent_asin`.

| File | Required fields | Optional fields |
| --- | --- | --- |
| Reviews | `user_id`, `parent_asin`, `rating`, `timestamp` | Other fields are not used |
| Metadata | `parent_asin`, nonempty `title`, nonempty `description` (string or list) | `categories`, `store`, `details` |

Example records illustrating the schema:

```json
{"user_id":"u1","parent_asin":"CD01","rating":5.0,"timestamp":1609459200000}
```

```json
{"parent_asin":"CD01","title":"Example Album","description":["A jazz recording."],"categories":["CDs & Vinyl","Jazz"],"store":"Example Artist"}
```

Ratings use a 1-5 scale. Timestamps must be convertible to integers and use a consistent unit. The CD filters require at least three five-star and two four-star interactions per user, followed by positive-feedback k-core filtering (user-10 / item-15 by default). Supply enough users for the evaluation cohorts and enough unseen items for ten-candidate retrieval. Each split needs natural-hit groups to continue to training.

## Code reference

| File | Purpose |
| --- | --- |
| `pipeline.py` | Run preparation, item validation, profile validation, and dataset export |
| `raw_filters.py`, `filter_data.py` | CD filtering, latest-event deduplication, positive feedback, and k-core |
| `extract_metadata.py` | Extract metadata, clean HTML/whitespace, and map item IDs |
| `lightgcn.py` | Temporal targets, user cohorts, LightGCN training, and natural Top-10 retrieval |
| `build_natural_train.py` | Build earlier-cutoff training groups with future-positive masking |
| `datasets.py` | Collect contexts, audit temporal boundaries, and export splits and complete evaluation cohorts |
| `generate_assets.py` | Prepare item/profile requests and parse and validate responses |
| `llm_profile_common.py` | Domain prompts, JSON schemas, semantic validation, and file utilities |
| `run_requests.py` | Execute concurrent API requests, resume recorded requests, and prepare retries |
| `assets.py` | Load validated item descriptions and format user profiles |
| `prepare.py` | Assemble histories, profiles, and candidate contexts |
| `samples.py` | Build pointwise records, deterministic SFT pairs, and replay weights |

## Run

Run from the repository root after configuring the environment and API variables in the [root README](../README.md).

```bash
# 1. Filter data, train retrieval models, and prepare item requests.
python -m semantic_context.pipeline prepare \
  --reviews data/raw/CDs_and_Vinyl.jsonl.gz \
  --metadata data/raw/meta_CDs_and_Vinyl.jsonl.gz \
  --output artifacts/cd_prepare --device cuda --asset-model gpt-4o

# 2. Generate item descriptions and prepare user-profile requests.
python -m semantic_context.run_requests sync \
  --requests artifacts/cd_prepare/item_requests/requests.jsonl \
  --output artifacts/cd_prepare/item_results.jsonl
python -m semantic_context.pipeline finish-items \
  --work artifacts/cd_prepare --results artifacts/cd_prepare/item_results.jsonl

# 3. Generate profiles and export all three splits.
python -m semantic_context.run_requests sync \
  --requests artifacts/cd_prepare/profile_requests/requests.jsonl \
  --output artifacts/cd_prepare/profile_results.jsonl
python -m semantic_context.pipeline finish-profiles \
  --work artifacts/cd_prepare --results artifacts/cd_prepare/profile_results.jsonl
```

Use `prepare --help` for filtering thresholds, seed, retrieval epochs, batch size, and cohort sizes. Validation and test default to 1,000 users each; reduce `--validation-users` and `--test-users` for smaller datasets. Start preparation in a new output directory.

If every long-term history is empty, the profile request file is empty. Create an empty `profile_results.jsonl` and run `finish-profiles`; the complete entry point handles this automatically. For invalid responses, inspect reports under `items/` or `profiles/`. Run `python -m semantic_context.run_requests --help` for request and retry commands.

## Outputs and next step

Under `artifacts/cd_prepare/`:

| Path | Contents |
| --- | --- |
| `filtered/` | Filtered interactions, item metadata, and statistics |
| `core/`, `natural_train/`, `sources/` | Retrieval models, temporal targets, candidates, histories, and profile contexts |
| `item_requests/`, `profile_requests/` | Requests and provenance; API responses use `*_results.jsonl` |
| `items/`, `profiles/` | Validated item descriptions and user profiles |
| `datasets/train.jsonl`, `valid.jsonl`, `test.jsonl` | Natural-hit groups and semantic contexts |
| `datasets/*_candidates.jsonl` | Pointwise candidates with `user_context` and `candidate_payload`, consumed by module 2 |
| `datasets/train_pairs.jsonl` | Positive/negative SFT pairs and weights |
| `datasets/valid_all.jsonl`, `test_all.jsonl` | Complete evaluation cohorts, including retrieval misses |
| `datasets/manifest.json` | Counts, retrieval recall, and file hashes |

Pass the complete `datasets/` directory to [module 2](../optical_compression/README.md). See the [root README](../README.md#use-another-dataset) for domain adaptation.
