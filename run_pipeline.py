"""Run all three Zip4Rec modules starting from raw Amazon CD dataset files."""
import argparse
import json
import sys
from pathlib import Path
from workflow import Workflow, digest
from optical_compression.strategy import RenderStrategy


def select_layout(config, workflow, datasets, model):
    """Search once from training samples, or reuse explicitly selected code."""
    optimizer = config.get('layout_optimizer_config')
    if optimizer and config.get('layout_strategy'):
        raise ValueError('choose layout_optimizer_config or layout_strategy, not both')
    if not optimizer:
        return RenderStrategy(config.get('layout_strategy'))
    output = workflow.directory / 'layout_search'
    workflow.run('select_layout', [sys.executable, '-m', 'optical_compression.optimize',
        '--config', optimizer, '--samples', datasets / 'train_candidates.jsonl',
        '--model', model, '--output', output], [output / 'frozen/layout.json'])
    return RenderStrategy(output / 'frozen')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if config.get('layout_optimizer_config') and config.get('layout_strategy'):
        p.error('choose layout_optimizer_config or layout_strategy, not both')
    root = Path(config['output']).resolve()
    semantic, dense, ranking = root / 'semantic', root / 'densepage', root / 'ranking'
    model = Path(config['base_model']).resolve()
    if not (model / 'config.json').is_file():
        raise FileNotFoundError('prepare the pretrained base model first: ' + str(model))
    grpo = Path(config['grpo_config']).resolve()
    optimizer = config.get('layout_optimizer_config')
    strategy = None if optimizer else RenderStrategy(config.get('layout_strategy'))
    workflow = Workflow(root, dict(config=config, config_sha256=digest(a.config),
        raw_sha256={key: digest(config[key]) for key in ['reviews', 'metadata']}, grpo_sha256=digest(grpo),
        **(dict(layout_optimizer_config_sha256=digest(optimizer)) if optimizer else {}),
        **(dict(layout_signature=strategy.signature) if strategy else {})))

    def run(name, module, arguments, outputs):
        workflow.run(name, [sys.executable, '-m', module, *arguments], outputs)

    filtering = config.get('filtering', {})
    retrieval = config.get('retrieval', {})
    run('prepare_data', 'semantic_context.pipeline', ['prepare', '--reviews', config['reviews'],
        '--metadata', config['metadata'], '--output', semantic, '--asset-model', config['asset_model'],
        '--user-k', filtering.get('user_k', 10), '--item-k', filtering.get('item_k', 15),
        '--device', retrieval.get('device', 'cuda'), '--epochs', retrieval.get('epochs', 35),
        '--validation-users', retrieval.get('validation_users', 1000),
        '--test-users', retrieval.get('test_users', 1000)], [semantic / 'item_requests/manifest.json'])
    for kind, finish in [('item', 'finish-items'), ('profile', 'finish-profiles')]:
        requests = semantic / f'{kind}_requests/requests.jsonl'
        results = semantic / f'{kind}_results.jsonl'
        if requests.stat().st_size:
            run('generate_' + kind, 'semantic_context.run_requests', ['sync', '--requests', requests,
                '--output', results, '--workers', config.get('asset_workers', 1)], [results])
        elif not results.exists():
            results.write_text('')
        outputs = ([semantic / 'profile_requests/manifest.json'] if kind == 'item'
                   else [semantic / 'datasets/manifest.json'])
        run('validate_' + kind, 'semantic_context.pipeline', [finish, '--work', semantic,
            '--results', results], outputs)
    strategy = select_layout(config, workflow, semantic / 'datasets', model)
    run('optical_compression', 'optical_compression.pipeline', ['--datasets', semantic / 'datasets',
        '--model', model, '--output', dense,
        '--layout', strategy.directory], [dense / 'complete.json'])
    run('train_and_evaluate', 'ranking.pipeline', ['--data', dense, '--datasets', semantic / 'datasets',
        '--model', model, '--grpo-config', grpo, '--output', ranking,
        '--sft-learning-rate', config.get('sft_learning_rate', 1e-5)], [ranking / 'results.json'])
    print('Completed project: ' + str(ranking / 'results.json'), flush=True)


if __name__ == '__main__':
    main()
