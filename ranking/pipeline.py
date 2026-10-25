"""Train fresh SFT and GRPO adapters, select on validation, then evaluate locked models."""
import argparse
import math
import sys
from pathlib import Path
from workflow import Workflow, digest
from .common import read_json, rows, write_json
from .evaluate import evaluate


def checkpoint_schedule(pair_count, epochs=3, accumulation=16):
    if pair_count < 2 or epochs != 3 or accumulation < 1:
        raise ValueError('SFT workflow requires paired samples, three epochs, positive accumulation')
    per_epoch = math.ceil(pair_count / accumulation)
    return per_epoch, [per_epoch, 2 * per_epoch, 3 * per_epoch]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Module-two output root')
    p.add_argument('--datasets', type=Path, required=True, help='Module-one final dataset root')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--grpo-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sft-learning-rate', type=float, default=1e-5)
    a = p.parse_args()
    a.output = a.output.resolve()
    manifest = read_json(a.datasets / 'manifest.json')
    dense_manifest = read_json(a.data / 'complete.json')
    if dense_manifest['dataset_manifest_sha256'] != digest(a.datasets / 'manifest.json'):
        raise ValueError('module-two inputs refer to a different semantic dataset')
    if not dense_manifest.get('processor_validated'):
        raise ValueError('run module two with --model to validate all actual input lengths')
    inputs = {s: a.data / s / 'candidates.jsonl' for s in ['train', 'valid', 'test']}
    pair_file = a.data / 'train/sft_pairs.jsonl'
    save_steps, sft_steps = checkpoint_schedule(len(rows(pair_file)))
    config = read_json(a.grpo_config)
    if config['fusion_alpha'] != 1.5:
        raise ValueError('the reference CD workflow uses fixed alpha=1.5')
    workflow = Workflow(a.output, dict(model=str(a.model.resolve()),
        **(dict(dense_manifest_sha256=digest(a.data / 'complete.json'))
           if dense_manifest.get('layout_signature') else {}),
        grpo_config_sha256=digest(a.grpo_config), sft_learning_rate=a.sft_learning_rate,
        dataset_manifest_sha256=digest(a.datasets / 'manifest.json'),
        input_sha256={s: digest(f) for s, f in inputs.items()}, sft_pairs_sha256=digest(pair_file)))

    def run(name, module, arguments, outputs):
        workflow.run(name, [sys.executable, '-m', 'ranking.' + module, *arguments], outputs)

    sft = a.output / 'sft'
    run('train_sft', 'train_sft', ['--model', a.model, '--train-data', pair_file,
        '--output-dir', sft, '--epochs', 3, '--learning-rate', a.sft_learning_rate,
        '--batch-size', 1, '--grad-acc-steps', 16, '--save-steps', save_steps,
        '--save-total-limit', 4, '--seed', config['seed']],
        [sft / f'checkpoint-{step}/adapter_model.safetensors' for step in sft_steps])
    sft_scores = []
    for step in sft_steps:
        output = a.output / f'validation/sft_step{step}.jsonl'
        run('validate_sft_' + str(step), 'score', ['--model', a.model, '--adapter',
            sft / f'checkpoint-{step}', '--data', inputs['valid'], '--output', output],
            [output, output.with_suffix('.metadata.json')])
        sft_scores.append(output)
    sft_selection = a.output / 'sft_selection.json'
    run('select_sft', 'select', ['--scores', *sft_scores, '--steps', *sft_steps,
        '--alpha', 1.5, '--output', sft_selection], [sft_selection])
    selected_sft = read_json(sft_selection)['selected']
    adapter = sft / f"checkpoint-{selected_sft['step']}"
    grpo = a.output / 'grpo'
    run('freeze_grpo', 'make_run', ['--config', a.grpo_config, '--base-model', a.model,
        '--initial-adapter', adapter, '--train', inputs['train'], '--valid', inputs['valid'],
        '--historical-valid-scores', selected_sft['scores'], '--output', grpo], [grpo / 'protocol.json'])
    run('prepare_references', 'prepare_reference', ['--run', grpo], [grpo / 'prepared/complete.json'])
    run('train_grpo', 'train_grpo', ['--run', grpo, '--resume'], [grpo / 'model/complete.json'])
    grpo_steps = read_json(grpo / 'protocol.json')['planned_checkpoints']
    grpo_scores = []
    for step in grpo_steps:
        output = a.output / f'validation/grpo_step{step}.jsonl'
        run('validate_grpo_' + str(step), 'score', ['--model', a.model, '--adapter',
            grpo / f'model/checkpoint-{step}', '--data', inputs['valid'], '--output', output],
            [output, output.with_suffix('.metadata.json')])
        grpo_scores.append(output)
    grpo_selection = a.output / 'grpo_selection.json'
    run('select_grpo', 'select', ['--scores', *grpo_scores, '--steps', *grpo_steps,
        '--alpha', 1.5, '--output', grpo_selection], [grpo_selection])
    selected_grpo = read_json(grpo_selection)['selected']
    adapters = dict(sft=adapter, grpo=grpo / f"model/checkpoint-{selected_grpo['step']}")
    lock = dict(alpha=1.5, test_used_for_selection=False,
        selection_sha256=dict(sft=digest(sft_selection), grpo=digest(grpo_selection)),
        adapter_sha256={name: digest(path / 'adapter_model.safetensors') for name, path in adapters.items()},
        test_input_sha256=digest(inputs['test']), all_test_users=manifest['splits']['test']['all_groups'])
    lock_path = a.output / 'test_plan.json'
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise ValueError('locked test protocol changed')
    else:
        write_json(lock_path, lock)
    summary = dict(sft_selection=selected_sft, grpo_selection=selected_grpo, test={})
    for name, path in adapters.items():
        scores = a.output / f'test/{name}_scores.jsonl'
        metrics = a.output / f'test/{name}_metrics.json'
        run('test_' + name, 'score', ['--model', a.model, '--adapter', path,
            '--data', inputs['test'], '--output', scores], [scores, scores.with_suffix('.metadata.json')])
        run('rank_' + name, 'rerank', ['--scores', scores, '--alpha', 1.5,
            '--output', a.output / f'test/{name}_rankings.jsonl'], [a.output / f'test/{name}_rankings.jsonl'])
        run('evaluate_' + name, 'evaluate', ['--scores', scores, '--alpha', 1.5,
            '--all-users', lock['all_test_users'], '--output', metrics], [metrics])
        result = read_json(metrics)
        summary['test'][name] = {k: v for k, v in result.items() if k != 'rankings'}
    prior = evaluate(rows(a.output / 'test/grpo_scores.jsonl'), alpha=0., all_users=lock['all_test_users'])
    summary['test']['lightgcn'] = {k: v for k, v in prior.items() if k in ['fused', 'end_to_end_fused', 'recall_at_10']}
    summary['test_plan_sha256'] = digest(lock_path)
    write_json(a.output / 'results.json', summary)


if __name__ == '__main__':
    main()
