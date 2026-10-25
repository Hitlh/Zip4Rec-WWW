# Zip4Rec

Zip4Rec compresses user profiles, interaction history, and candidate item descriptions into DensePage images. A vision-language model is trained with supervised fine-tuning (SFT) and group relative policy optimization (GRPO) for recommendation reranking.

This repository provides an end-to-end example using **Amazon Reviews 2023 CDs & Vinyl, LightGCN, and Qwen3-VL-8B-Instruct**. It starts from raw data, generates training, validation, and test inputs, and trains and evaluates reranking models. Download the datasets and model weights separately; all processed data and experiment outputs are generated locally.

## Three modules

| Module | Workflow and code reference |
| --- | --- |
| [semantic_context](semantic_context/README.md) | Filter raw data, retrieve candidates with LightGCN, extract item descriptions and user profiles, and construct the three data splits |
| [optical_compression](optical_compression/README.md) | Search for a layout with a vision agent, freeze the selected code, and render DensePages and multimodal inputs |
| [ranking](ranking/README.md) | Train SFT and GRPO adapters, select checkpoints on validation data, and evaluate test reranking |

```text
Raw reviews + item metadata
  -> Filtering and temporal splits -> Natural LightGCN Top-10 candidates
  -> Item descriptions and user profiles -> Train / validation / test inputs
  -> Agent layout selection -> Batch rendering with fixed layout code
  -> SFT -> Validation selection -> GRPO -> Validation selection -> Test evaluation
```

## Environment and inputs

Use Python 3.10 and one CUDA GPU with BF16 support. For CUDA 12.4:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Prepare:

1. The CDs & Vinyl review and metadata files from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), in JSONL or JSONL.GZ format. See the [input schema](semantic_context/README.md#input-format).
2. Local [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) weights, tokenizer, and processor.
3. A Chat Completions-compatible API for item descriptions, user profiles, and the layout agent. The agent model must support both image inputs and tool calling.

## Run from raw data

Run every command from the repository root. Set the dataset, model, and output paths in [pipeline_agent.json](configs/cd_lightgcn/pipeline_agent.json), and set the vision agent model in [layout_optimizer.json](configs/cd_lightgcn/layout_optimizer.json). See the [configuration reference](configs/cd_lightgcn/README.md).

```bash
export CUDA_VISIBLE_DEVICES=0
export OPENAI_BASE_URL='https://api.openai.com/v1'
export OPENAI_API_KEY='<your API key>'
python run_pipeline.py --config configs/cd_lightgcn/pipeline_agent.json
```

The entry point runs all three modules. It searches for a layout once using training examples, exports `layout_search/frozen/`, and uses that fixed code to render all three splits.

The selected CD layout is stored in [optical_compression/selected/compact.py](optical_compression/selected/compact.py). To use it directly, set local paths in [pipeline.json](configs/cd_lightgcn/pipeline.json) and run:

```bash
python run_pipeline.py --config configs/cd_lightgcn/pipeline.json
```

Main outputs under the configured `output` directory:

| Path | Contents |
| --- | --- |
| `semantic/datasets/` | Semantic splits, SFT pairs, ten-candidate groups, and complete evaluation cohorts |
| `layout_search/frozen/` | Agent-selected layout code, fonts, and configuration |
| `densepage/` | Images, multimodal records, and actual token measurements for all splits |
| `ranking/` | SFT/GRPO adapters, validation selections, test scores, and rankings |
| `ranking/results.json` | LightGCN, SFT, and GRPO evaluation results |

Rerunning the same configuration reuses completed stages whose outputs pass verification. Use a new output directory when changing inputs or configuration. Stage records are stored in `workflow_state.json`.

## Use another dataset

The CD example requires both interactions and item text. Convert other datasets to the [input schema](semantic_context/README.md#input-format), then adapt the following locations:

| Change | Location |
| --- | --- |
| Paths, k-core thresholds, retrieval epochs, and validation/test cohort sizes | `configs/cd_lightgcn/pipeline*.json` |
| Rating filters and positive-feedback definition | `semantic_context/raw_filters.py`, `filter_data.py` |
| Metadata cleaning, domain prompts, semantic fields, and validation | `semantic_context/extract_metadata.py`, `llm_profile_common.py`, `assets.py` |
| User/candidate text and domain terminology | `semantic_context/prepare.py`, `samples.py` |
| Page fields and recommendation prompts | `optical_compression/document.py`, `build.py`, and `ranking/sft_support.py`; also `optical_compression/selected/compact.py` if reusing compact |
| Initial fonts, sizes, search budget, and readback threshold | `configs/cd_lightgcn/layout_optimizer.json` |

The supplied rating filters and semantic fields describe music. Other domains require corresponding code changes; datasets with the same field and semantic conventions can use the existing code with updated configuration. The training pipeline uses Qwen3-VL-8B and ten candidates per group.

## Supporting files and checks

`run_pipeline.py` runs the complete workflow. `workflow.py` records stages and verifies artifacts. `configs/` stores parameters, `tests/` checks data boundaries, rendering, training recovery, and ranking, and `optical_compression/fonts/` contains fixed fonts and their license.

```bash
python -m unittest discover -s tests -v
```
