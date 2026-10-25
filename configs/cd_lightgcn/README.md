# CD / LightGCN Configuration

This directory configures the complete Amazon Reviews 2023 CDs & Vinyl, natural LightGCN Top-10, and Qwen3-VL-8B-Instruct workflow.

## Files

| File | Purpose |
| --- | --- |
| `pipeline_agent.json` | Start from raw data, select layout code with the agent, then render, train, and evaluate |
| `pipeline.json` | Start from raw data and use the selected `optical_compression/selected/compact.py` layout |
| `layout_optimizer.json` | Initial layout, vision agent, fixed typography, search budget, token limits, and readback threshold |
| `grpo.json` | GRPO learning rate, update size, permutation sampling, KL, SFT replay, and fusion coefficient |

## Complete-pipeline parameters

Paths are relative to the repository root when running commands, or may be absolute.

| Field | Meaning |
| --- | --- |
| `reviews`, `metadata` | Raw review and item-metadata files, JSONL or JSONL.GZ |
| `base_model` | Local Qwen3-VL-8B-Instruct weights, tokenizer, and processor |
| `output` | Output directory for this run |
| `asset_model`, `asset_workers` | Item/profile generation model and request concurrency |
| `filtering.user_k`, `filtering.item_k` | Positive-feedback k-core thresholds, default 10 / 15 |
| `retrieval.device`, `retrieval.epochs` | LightGCN device (`cpu` / `cuda`) and training epochs |
| `retrieval.validation_users`, `retrieval.test_users` | Disjoint validation/test cohort sizes, default 1,000 each; reduce for smaller datasets |
| `layout_optimizer_config` | One-time agent search configuration |
| `layout_strategy` | Selected-code directory: `optical_compression/selected` or an exported `frozen/` |
| `sft_learning_rate` | SFT learning rate, default `1e-5` |
| `grpo_config` | GRPO configuration path |

Set only one of `layout_optimizer_config` and `layout_strategy`. Use a new `output` when changing configuration or data.

## Run

Install dependencies and prepare datasets and model files as described in the [root README](../../README.md). Update local paths and API settings, then choose one command from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_BASE_URL='https://api.openai.com/v1'
export OPENAI_API_KEY='<your API key>'

# Search once, then use fixed code for the remaining workflow.
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json

# Alternatively, reuse the selected compact layout.
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

Outputs are `semantic/`, `layout_search/` (search mode), `densepage/`, and `ranking/`. Final results are written to `<output>/ranking/results.json`.

For individual commands, inputs, outputs, and code references, see [module 1](../../semantic_context/README.md), [module 2](../../optical_compression/README.md), and [module 3](../../ranking/README.md). Dataset adaptation is described in the [root README](../../README.md#use-another-dataset).
