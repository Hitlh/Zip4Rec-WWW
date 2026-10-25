"""Run the dataset-to-training-input pipeline in three inspectable stages."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .llm_profile_common import sha256_file, write_json_exclusive


def run(module, *arguments):
    subprocess.run([sys.executable, '-m', 'semantic_context.' + module,
                    *map(str, arguments)], check=True)


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    raw = a.output / 'filtered'
    raw.mkdir()
    interactions, metadata = raw / 'positive_kcore.tsv', raw / 'item_metadata.csv'
    write_json_exclusive(a.output / 'pipeline.json', dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        raw_sources={str(p.resolve()): sha256_file(p) for p in [a.reviews, a.metadata]},
        scope='Fixed retriever configuration; test targets never tune preprocessing or retriever parameters'))
    run('filter_data', '--reviews', a.reviews, '--metadata', a.metadata, '--output', interactions,
        '--user-k', a.user_k, '--item-k', a.item_k)
    run('extract_metadata', '--data', interactions, '--metadata-jsonl', a.metadata, '--output', metadata)
    run('lightgcn', '--interactions', interactions, '--item-metadata', metadata,
        '--output-dir', a.output / 'core', '--seed', a.seed,
        '--validation-users', a.validation_users, '--test-users', a.test_users,
        '--epochs', a.epochs, '--batch-size', a.batch_size, '--device', a.device,
        '--evaluation-splits', 'validation', 'test')
    run('build_natural_train', '--interactions', interactions, '--item-metadata', metadata,
        '--source-train-data', a.output / 'core/sft_records.jsonl', '--output-dir', a.output / 'natural_train',
        '--seed', a.seed, '--validation-users', a.validation_users, '--test-users', a.test_users,
        '--cutoffs', 3, '--epochs', a.epochs, '--batch-size', a.batch_size, '--device', a.device)
    run('datasets', 'collect', '--core-dir', a.output / 'core',
        '--natural-train', a.output / 'natural_train/natural_hit_groups.jsonl',
        '--interactions', interactions, '--item-metadata', metadata, '--output-dir', a.output / 'sources')
    run('generate_assets', 'prepare-items', '--item-metadata', metadata,
        '--model', a.asset_model, '--output-dir', a.output / 'item_requests')
    print('Prepared item_requests/requests.jsonl. Run semantic_context.run_requests, then finish-items.', flush=True)


def finish_items(a):
    config = json.loads((a.work / 'pipeline.json').read_text())['configuration']
    run('generate_assets', 'parse-items', '--request-dir', a.work / 'item_requests',
        '--results', a.results, '--output-dir', a.work / 'items')
    run('generate_assets', 'prepare-profiles', '--contexts', a.work / 'sources/profile_contexts_numeric.jsonl',
        '--item-summaries', a.work / 'items/item_summaries.jsonl', '--model', config['asset_model'],
        '--output-dir', a.work / 'profile_requests')
    print('Prepared profile_requests/requests.jsonl. Run semantic_context.run_requests, then finish-profiles.', flush=True)


def finish_profiles(a):
    run('generate_assets', 'parse-profiles', '--request-dir', a.work / 'profile_requests',
        '--results', a.results, '--output-dir', a.work / 'profiles')
    run('datasets', 'export', '--source-dir', a.work / 'sources',
        '--item-summaries', a.work / 'items/item_summaries.jsonl',
        '--user-profiles', a.work / 'profiles/user_profiles.jsonl', '--output-dir', a.work / 'datasets')
    print('Ready: datasets/train.jsonl, valid.jsonl, test.jsonl and complete valid_all/test_all cohorts.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    for key in ['reviews', 'metadata', 'output']:
        prep.add_argument('--' + key, type=Path, required=True)
    prep.add_argument('--user-k', type=int, default=10)
    prep.add_argument('--item-k', type=int, default=15)
    prep.add_argument('--seed', type=int, default=603759240)
    prep.add_argument('--validation-users', type=int, default=1000)
    prep.add_argument('--test-users', type=int, default=1000)
    prep.add_argument('--epochs', type=int, default=35)
    prep.add_argument('--batch-size', type=int, default=2048)
    prep.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    prep.add_argument('--asset-model', default='gpt-4o')
    for name in ['finish-items', 'finish-profiles']:
        step = commands.add_parser(name)
        step.add_argument('--work', type=Path, required=True)
        step.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    {'prepare': prepare, 'finish-items': finish_items, 'finish-profiles': finish_profiles}[a.command](a)


if __name__ == '__main__':
    main()
