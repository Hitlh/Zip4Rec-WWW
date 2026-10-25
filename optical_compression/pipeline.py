"""Render all freshly prepared splits and verify actual model input lengths."""
import argparse
import json
import sys
from pathlib import Path
from workflow import Workflow, digest
from .strategy import RenderStrategy, SELECTED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, help='Actual processor for complete input-length validation')
    p.add_argument('--layout', type=Path, default=SELECTED,
                   help='Already-selected code directory; does not run the agent')
    a = p.parse_args()
    strategy = RenderStrategy(a.layout)
    if a.model and strategy.requires_compression_check:
        from .measurement import TokenMeter
        if TokenMeter(a.model).identity != strategy.metadata['processor']:
            raise ValueError('frozen strategy was optimized for a different processor or prompt protocol')
    manifest = json.loads((a.datasets / 'manifest.json').read_text())
    for name, expected in manifest['output_sha256'].items():
        if digest(a.datasets / name) != expected:
            raise ValueError('semantic dataset changed: ' + name)
    workflow = Workflow(a.output, dict(datasets=str(a.datasets.resolve()),
        dataset_manifest_sha256=digest(a.datasets / 'manifest.json'),
        model=str(a.model.resolve()) if a.model else None,
        layout_signature=strategy.signature))
    for split in ['train', 'valid', 'test']:
        if manifest['splits'][split]['natural_hit_groups'] == 0:
            raise ValueError('no natural-hit samples in ' + split)
        workflow.run('render_' + split,
            [sys.executable, '-m', 'optical_compression.build', '--samples',
             a.datasets / f'{split}_candidates.jsonl', '--output', a.output / split,
             '--layout', strategy.directory],
            [a.output / split / 'manifest.json', a.output / split / 'sft_pairs.jsonl',
             a.output / split / 'candidates.jsonl'])
        if a.model:
            workflow.run('encode_' + split,
                [sys.executable, '-m', 'optical_compression.measure_tokens', '--data',
                 a.output / split / 'candidates.jsonl', '--model', a.model,
                 '--output', a.output / split / 'actual_tokens.json',
                 *(['--text-source', a.datasets / f'{split}_candidates.jsonl',
                    '--max-length', strategy.metadata['evaluation_limits']['max_length']]
                   if strategy.requires_compression_check else [])],
                [a.output / split / 'actual_tokens.json'])
    summary = dict(dataset_manifest_sha256=digest(a.datasets / 'manifest.json'),
                   splits=manifest['splits'], processor_validated=bool(a.model),
                   layout_signature=strategy.signature, layout_kind=strategy.kind,
                   image_format='PNG fixed-font top/bottom; dynamic geometry, no text truncation'
                   if strategy.requires_compression_check else 'PNG compact640 font14; dynamic height, no text truncation')
    (a.output / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
