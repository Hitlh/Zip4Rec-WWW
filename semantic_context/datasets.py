"""Freeze source splits and history-only contexts, then export semantic datasets."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from .llm_profile_common import (read_jsonl, sha256_file, sha256_json,
    validate_item_summary, validate_user_profile, write_json_exclusive, write_jsonl_exclusive)
from .assets import load_profiles, load_validated_items
from .prepare import organize
from .samples import pointwise


def metadata_mapping(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapping = {r['original_item_id']: int(r['item_id']) for r in rows}
    if len(mapping) != len(rows) or len(set(mapping.values())) != len(rows):
        raise ValueError('metadata IDs must map one-to-one')
    return mapping


def read_events(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    events = defaultdict(list)
    pairs = set()
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            _, user, item, rating, timestamp = line.rstrip('\n').split('\t')
            if float(rating) < 4 or (user, item) in pairs:
                raise ValueError('expected deduplicated positive user-item interactions')
            pairs.add((user, item))
            events[user].append((int(timestamp), item))
    return {u: sorted(rows) for u, rows in events.items()}


def audit_source(row, events):
    user, target = row['user_id'], str(row['target_item_id'])
    candidates = list(map(str, row['candidate_item_ids']))
    if row.get('target_injected') is not False:
        raise ValueError('only explicitly natural candidates may enter the final dataset')
    if len(candidates) != 10 or len(set(candidates)) != 10:
        raise ValueError('expect ten unique candidates')
    sequence = events[user]
    event = (int(row['target_timestamp']), target)
    if event not in sequence:
        raise ValueError('target event absent from source interactions')
    position = sequence.index(event)
    expected_history = [i for _, i in sequence[:position]]
    history = list(map(str, row['history_item_ids']))
    long = list(map(str, row['long_history_item_ids']))
    recent = list(map(str, row['recent_history_item_ids']))
    if history != expected_history or long != history[:-3] or recent != history[-3:]:
        raise ValueError('history does not match the exact pre-target prefix')
    if row['profile_history_key'] != sha256_json({'ordered_item_ids': long}):
        raise ValueError('profile key does not match ordered long-term history')
    if set(candidates) & set(history) or target in history:
        raise ValueError('candidate/history leakage')
    if row['split'] == 'train':
        future = {item for _, item in sequence[position + 1:]}
        if target not in candidates or set(candidates) & future:
            raise ValueError('training candidates contain a future positive or lack the target')
    elif position != len(sequence) - 1:
        raise ValueError('evaluation target must be the final positive event')


def collect(core_dir, natural_train, interactions, item_metadata, output_dir):
    train = read_jsonl(natural_train)
    evaluation = read_jsonl(core_dir / 'evaluation_records.jsonl')
    sources = {'train': train,
               'valid': [dict(r, split='valid') for r in evaluation if r['split'] in ['validation', 'valid']],
               'test': [r for r in evaluation if r['split'] == 'test']}
    cohorts = read_jsonl(core_dir / 'user_cohort_manifest.jsonl')
    expected = {s: {r['user_id'] for r in cohorts if r['cohort'] == c}
                for s, c in [('valid', 'validation'), ('test', 'test')]}
    if not all(sources.values()) or expected['valid'] & expected['test']:
        raise ValueError('missing split or overlapping validation/test cohorts')
    for split in ['valid', 'test']:
        users = [r['user_id'] for r in sources[split]]
        if len(set(users)) != len(users) or set(users) != expected[split]:
            raise ValueError('evaluation records must cover every frozen user exactly once')
    records = [r for split in ['train', 'valid', 'test'] for r in sources[split]]
    if any(r['split'] != 'train' for r in train) or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('invalid training split or duplicate source IDs')
    events, mapping = read_events(interactions), metadata_mapping(item_metadata)
    contexts = {}
    for row in records:
        audit_source(row, events)
        if not set(row['candidate_item_ids']) <= set(mapping):
            raise ValueError('candidate missing from metadata')
        long = list(map(str, row['long_history_item_ids']))
        context = dict(profile_history_key=row['profile_history_key'],
                       long_history_item_ids=[mapping[i] for i in long],
                       long_history_original_item_ids=long, long_history_count=len(long))
        old = contexts.setdefault(row['profile_history_key'], context)
        if old != context:
            raise ValueError('conflicting profile history key')
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, data in sources.items():
        write_jsonl_exclusive(output_dir / f'{split}_sources.jsonl', data)
    write_jsonl_exclusive(output_dir / 'profile_contexts_numeric.jsonl',
                          [contexts[k] for k in sorted(contexts)])
    write_json_exclusive(output_dir / 'manifest.json', dict(
        schema='zip4rec_semantic_sources_v1', splits={s: len(r) for s, r in sources.items()},
        profile_contexts=len(contexts),
        validation_test_users_disjoint=True, all_histories_exact_pre_target_prefixes=True,
        all_training_future_positives_masked=True,
        temporal_training_users_may_overlap_evaluation_users=True,
        sources={str(p.resolve()): sha256_file(p) for p in [interactions, item_metadata,
            natural_train, core_dir / 'evaluation_records.jsonl', core_dir / 'user_cohort_manifest.jsonl']},
        output_sha256={f.name: sha256_file(f) for f in output_dir.glob('*.jsonl')}))


def export(source_dir, item_summaries, user_profiles, output_dir):
    manifest = json.loads((source_dir / 'manifest.json').read_text())
    for name, expected in manifest['output_sha256'].items():
        if sha256_file(source_dir / name) != expected:
            raise ValueError('frozen source file changed: ' + name)
    items = read_jsonl(item_summaries)
    by_numeric = {int(r['item_id']): r for r in items}
    if len(by_numeric) != len(items) or len({r['original_item_id'] for r in items}) != len(items):
        raise ValueError('duplicate item summary IDs')
    for row in items:
        validate_item_summary(row['summary'])
        if row.get('summary_hash') != sha256_json(row['summary']):
            raise ValueError('item summary content/hash mismatch')
    contexts = {r['profile_history_key']: r for r in read_jsonl(source_dir / 'profile_contexts_numeric.jsonl')}
    profiles = read_jsonl(user_profiles)
    if len({r['profile_history_key'] for r in profiles}) != len(profiles):
        raise ValueError('duplicate profile keys')
    if {r['profile_history_key'] for r in profiles} != set(contexts):
        raise ValueError('missing or unexpected generated profiles')
    for row in profiles:
        context = contexts[row['profile_history_key']]
        ids = context['long_history_item_ids']
        if (row.get('long_history_item_ids') != ids
                or [by_numeric[i]['original_item_id'] for i in ids] != context['long_history_original_item_ids']):
            raise ValueError('profile history mapping mismatch')
        if row.get('ordered_item_summary_hashes') != [by_numeric[i]['summary_hash'] for i in ids]:
            raise ValueError('profile refers to different item summaries')
        validate_user_profile(row['profile'], long_history_count=len(ids))
        if row.get('profile_hash') != sha256_json(row['profile']):
            raise ValueError('profile content/hash mismatch')
    profiles_text, items_view = load_profiles(user_profiles), load_validated_items(item_summaries)
    built, counts = {}, {}
    for split in ['train', 'valid', 'test']:
        records = read_jsonl(source_dir / f'{split}_sources.jsonl')
        # Drop obsolete listwise supervision before creating pointwise-ready contexts.
        records = [{k: v for k, v in r.items() if k not in
                    ['messages', 'inst', 'answer', 'answer_item_ids', 'answer_labels']} for r in records]
        all_rows = [organize(r, profiles_text, items_view) for r in records]
        hit = [r for r in all_rows if r['target_item_id'] in r['candidate_item_ids']]
        built[split] = hit
        pairs, candidates = pointwise(hit, items_view)
        built[split + '_candidates'] = candidates
        if split == 'train':
            built['train_pairs'] = pairs
        if split != 'train':
            built[split + '_all'] = all_rows
        counts[split] = dict(all_groups=len(all_rows), natural_hit_groups=len(hit),
                             recall_at_10=len(hit) / len(all_rows),
                             downstream_pointwise_rows=len(hit) * (2 if split == 'train' else 10))
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in built.items():
        write_jsonl_exclusive(output_dir / (name + '.jsonl'), data)
    write_json_exclusive(output_dir / 'manifest.json', dict(schema='zip4rec_semantic_dataset_v1',
        splits=counts, training_user_reuse='earlier temporal events allowed; external holdouts excluded',
        assets={str(p.resolve()): sha256_file(p) for p in [source_dir / 'manifest.json', item_summaries, user_profiles]},
        output_sha256={f.name: sha256_file(f) for f in output_dir.glob('*.jsonl')}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('collect')
    for key in ['core-dir', 'natural-train', 'interactions', 'item-metadata', 'output-dir']:
        c.add_argument('--' + key, type=Path, required=True)
    e = commands.add_parser('export')
    for key in ['source-dir', 'item-summaries', 'user-profiles', 'output-dir']:
        e.add_argument('--' + key, type=Path, required=True)
    a = vars(p.parse_args())
    command = a.pop('command')
    (collect if command == 'collect' else export)(**a)


if __name__ == '__main__':
    main()
