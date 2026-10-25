# Module 3: Recommendation Reranking

Train SFT and GRPO adapters on module 2's multimodal inputs, select checkpoints on validation data, and score and rerank the test candidates.

## Workflow

1. Initialize language LoRA adapters on Qwen3-VL-8B-Instruct and run SFT on A/B positive/negative pairs.
2. Score SFT checkpoints on validation data and select by NDCG@10, MRR@10, H@1, and earlier step.
3. Initialize GRPO from the selected SFT adapter, compute base-model and frozen-SFT reference logits, and train the permutation policy.
4. Select a GRPO checkpoint on validation data and lock the test protocol.
5. Evaluate selected SFT/GRPO adapters and report semantic scores, fused rankings, and the original LightGCN ranking.

The semantic score is the adapter-minus-base increment of the A/B logit difference. Fusion uses `z(LightGCN) + 1.5 * z(semantic)` within each user's ten candidates. Only naturally retrieved candidates are reranked; targets are not injected. Metrics cover both retrieval-hit users and the complete evaluation cohort.

## Code reference

| File | Purpose |
| --- | --- |
| `pipeline.py` | Run SFT, GRPO, validation selection, and locked-model testing |
| `train_sft.py`, `sft_support.py` | Weighted A/B delta cross-entropy, language LoRA, and SFT input processing |
| `make_run.py` | Freeze GRPO inputs, initial adapter, and configuration |
| `prepare_reference.py` | Compute base-model and frozen-SFT reference logits |
| `train_grpo.py` | Old-policy windows, training updates, checkpoints, and recovery |
| `policy.py` | Plackett-Luce sampling, relative advantages, clipping, KL, and SFT replay |
| `exact_policy.py` | Exact short-list expectations for policy diagnostics and gradient checks |
| `model_io.py` | Model/adapter loading, numerical precision, and forward passes |
| `conversation.py` | Multimodal conversations and processor inputs |
| `score.py` | Compute semantic scores using a fixed adapter |
| `select.py` | Select checkpoints from validation scores |
| `rerank.py`, `evaluate.py` | Score fusion, deterministic ranking, and conditional/overall metrics |
| `common.py`, `runtime_fingerprint.py` | Shared validation, file utilities, and runtime records |

## Inputs and execution

Use one BF16-capable CUDA GPU and the complete outputs of modules 1 and 2:

- `--data`: module 2 output root, containing multimodal candidates for train/valid/test, training pairs, and `complete.json`. Actual processor validation must have passed.
- `--datasets`: module 1's `datasets/`, providing split identities and complete evaluation cohort sizes.
- `--model`: local Qwen3-VL-8B-Instruct weights, tokenizer, and processor.
- `--grpo-config`: GRPO parameters; see [grpo.json](../configs/cd_lightgcn/grpo.json).

Run from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m ranking.pipeline \
  --data artifacts/cd_densepage \
  --datasets artifacts/cd_prepare/datasets \
  --model data/models/Qwen3-VL-8B-Instruct \
  --grpo-config configs/cd_lightgcn/grpo.json \
  --output artifacts/cd_ranking
```

Set `--sft-learning-rate` to change the SFT learning rate (default `1e-5`). SFT uses three epochs and gradient accumulation of 16; checkpoint steps follow the actual pair count. GRPO defaults to eight permutations per user and four user groups per update. See its configuration for other parameters. Each group has ten candidates and one positive. Preserve the replay flags and weights produced by modules 1 and 2.

## Outputs

| Path | Contents |
| --- | --- |
| `sft/` | SFT adapters and checkpoints |
| `grpo/` | Frozen protocol, reference cache, training state, and checkpoints |
| `validation/` | Validation scores for candidate checkpoints |
| `sft_selection.json`, `grpo_selection.json` | Checkpoint selections |
| `test_plan.json` | Model and fusion settings fixed before testing |
| `test/` | Test scores, user rankings, and metrics |
| `results.json` | SFT, GRPO, and original LightGCN results |

Rerunning the same configuration reuses complete stages. Resume interrupted training with `ranking.train_sft --resume-from-checkpoint` or `ranking.train_grpo --resume`; see each entry point's `--help`. For the full raw-data workflow, see the [root README](../README.md).
