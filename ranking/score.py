"""Score fixed adapters with the original FP32 A/B delta and per-input token counts."""
import argparse
import json
from pathlib import Path
from .common import digest, rows, signature, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoProcessor
    from .model_io import setup
    from .conversation import token_counts
    records = rows(a.data)
    if not records or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('empty input or duplicate sample IDs')
    model, encode, logits, fingerprint = setup(a.model, a.adapter)
    processor = AutoProcessor.from_pretrained(str(a.model), local_files_only=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream, torch.inference_mode():
        for row in records:
            encoded = encode(row)
            with model.disable_adapter():
                base = logits(encoded).float()
            adapted = logits(encoded).float()
            delta = adapted - base
            if not torch.isfinite(delta).all():
                raise ValueError('nonfinite delta')
            result = {k: row[k] for k in ['sample_id', 'group_id', 'user_id', 'candidate_item_id', 'candidate_score']}
            for key in ['label', 'target_item_id', 'split']:
                if key in row:
                    result[key] = row[key]
            result.update(base_logits=base.cpu().tolist(), adapted_logits=adapted.cpu().tolist(),
                          delta_logit_A=float(delta[0]), delta_logit_B=float(delta[1]),
                          condition_score=float(delta[0] - delta[1]), input_sha256=signature(row),
                          **token_counts(processor, encoded))
            stream.write(json.dumps(result) + '\n')
    write_json(a.output.with_suffix('.metadata.json'), dict(complete=True, fingerprint=fingerprint,
        data_sha256=digest(a.data), adapter_sha256=digest(a.adapter / 'adapter_model.safetensors'),
        output_sha256=digest(a.output), rows=len(records)))


if __name__ == '__main__':
    main()
