"""Encode every rendered page using the actual VLM processor without truncation."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--text-source', type=Path, help='Original semantic samples; require every image input to beat pure text')
    a = p.parse_args()
    import torch
    from .measurement import TokenMeter
    from .document import document
    from ranking.conversation import process_conversation, record_messages, token_counts, read_jsonl
    meter = TokenMeter(a.model)
    processor = meter.processor
    records = read_jsonl(a.data)
    text_sources = None
    if a.text_source:
        source_rows = read_jsonl(a.text_source)
        text_sources = {r['sample_id']: r for r in source_rows}
        if (len(text_sources) != len(source_rows) or len(records) != len(text_sources)
                or {r['sample_id'] for r in records} != set(text_sources)):
            raise ValueError('image and pure-text inputs must have identical unique sample IDs')
    counts = []
    with torch.inference_mode():
        for row in records:
            encoded = process_conversation(processor, record_messages(row, include_assistant=False),
                                           add_generation_prompt=True)
            value = dict(sample_id=row['sample_id'], group_id=row['group_id'], **token_counts(processor, encoded))
            if value['total_input_tokens'] > a.max_length:
                raise ValueError('input exceeds model guard without truncation: ' + row['sample_id'])
            if text_sources is not None:
                source = text_sources[row['sample_id']]
                baseline = meter.text(document(source['user_context'], source['candidate_payload']))
                value['pure_text_tokens'] = baseline['total_input_tokens']
                value['strictly_below_text'] = value['total_input_tokens'] < value['pure_text_tokens']
                if not value['strictly_below_text']:
                    raise ValueError('frozen layout does not beat pure text for sample: ' + row['sample_id'])
            counts.append(value)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(dict(records=counts, samples=len(counts),
            total_input_tokens=sum(c['total_input_tokens'] for c in counts),
            mean_input_tokens=statistics.fmean(c['total_input_tokens'] for c in counts),
            max_input_tokens=max(c['total_input_tokens'] for c in counts),
            max_length=a.max_length, all_inputs_within_guard=True,
            compression_checked=text_sources is not None, processor=meter.identity), stream, indent=2)


if __name__ == '__main__':
    main()
