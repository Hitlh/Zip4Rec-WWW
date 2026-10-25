# Module 2: Optical Compression

Use a vision agent to select layout code once, then freeze it and render DensePages in batches for module 3.

## Workflow and code

```text
Module 1 train_candidates.jsonl
  -> Ten real training examples -> Agent code edits, image review, token measurement, and readback
  -> frozen/layout.py + fonts + configuration
  -> Fixed-code rendering -> Train / validation / test images and multimodal records
```

Pages use a single column with user information above item information. Fonts, sizes, the main title, and blue/green section bars stay fixed. Search may adjust width, line height, margins, wrapping, and field grouping. These images are SFT/GRPO training inputs, so the search considers recognizability and field structure as well as token cost.

| File or directory | Purpose |
| --- | --- |
| `optimize.py` | One-time layout search entry point |
| `optimizer/agent.py` | Vision-model HTTP client, prompt, and tool-calling loop |
| `optimizer/tools.py` | Four tools, candidate evaluation, selection, and export |
| `layouts/initial.py` | Simple initial top/bottom layout |
| `document.py` | Label-free page text, pure-text baselines, and training-example selection |
| `layout_engine.py`, `layout_worker.py` | Execute layout code, validate text/geometry, and draw fixed decorations and pages |
| `presentation.py` | Field grouping, labels, and separators |
| `measurement.py` | Actual processor token counts and local VLM readback |
| `selected/compact.py`, `selected/selection.json` | Selected CD compact layout and selection marker |
| `strategy.py` | Load selected code for batch rendering |
| `build.py` | Generate images, candidate records, and SFT pairs for one split |
| `measure_tokens.py` | Validate actual multimodal input tokens and length |
| `pipeline.py` | Render and validate all three splits with fixed code |
| `fonts/` | Fixed fonts and license |

## 1. Search once

Configure the API as described in the [root README](../README.md). Set the vision model and budget in [layout_optimizer.json](../configs/cd_lightgcn/layout_optimizer.json), then run from the repository root:

```bash
python -m optical_compression.optimize \
  --config configs/cd_lightgcn/layout_optimizer.json \
  --samples artifacts/cd_prepare/datasets/train_candidates.jsonl \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/layout_cd_search
```

Supply at least ten distinct real training pages. By default, both regions of every example must reach readback similarity >=0.90. Each image input must have strictly fewer total tokens than its corresponding pure-text input and satisfy the 2,048-token limit. Search requires a vision API; local readback requires Qwen3-VL weights and a GPU.

Successful search exports `artifacts/layout_cd_search/frozen/` containing `layout.py`, `layout.json`, and `fonts/`. Search stops if no candidate qualifies. Tool implementations are in [optimizer/tools.py](optimizer/tools.py), and search parameters are in [layout_optimizer.json](../configs/cd_lightgcn/layout_optimizer.json).

## 2. Render with fixed code

```bash
python -m optical_compression.pipeline \
  --datasets artifacts/cd_prepare/datasets \
  --layout artifacts/layout_cd_search/frozen \
  --model data/models/Qwen3-VL-8B-Instruct \
  --output artifacts/cd_densepage
```

This step calls only the selected layout code and uses the processor to check actual tokens. It does not search or perform readback. The same code and fonts are used for training, validation, and test data without truncating content.

The selected CD code is [selected/compact.py](selected/compact.py). To reuse it, replace `--layout` above with `optical_compression/selected`; see the [usage reference](selected/README.md).

## Outputs and next step

Each split contains `images/`, `candidates.jsonl`, `sft_pairs.jsonl`, `actual_tokens.json`, and `manifest.json`. The root contains `complete.json`. Pass the complete `artifacts/cd_densepage/` directory and module 1's `datasets/` to [module 3](../ranking/README.md).

Multimodal records contain absolute image paths. Rebuild inputs after moving images. Training requires this step to include `--model` and pass actual processor validation.
