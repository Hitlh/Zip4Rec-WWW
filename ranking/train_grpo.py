"""Independent SFT initialization; full-group graphs and resumable old-policy windows."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from .common import digest, grouped, now, read_json, rows, signature, write_json
from .model_io import setup
from .policy import objective, policy_logits, make_rollout


def lr_multiplier(step, updates, warmup_ratio):
    warmup = max(1, math.ceil(updates * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, updates - warmup)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output-name')
    a = p.parse_args()
    plan = read_json(a.run / 'protocol.json')
    for name, expected in plan['code_sha256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError('code changed after protocol freeze: ' + name)
    for name, path in [('train_ten', 'train_ten.jsonl'), ('valid', 'valid.jsonl')]:
        if digest(a.run / 'inputs' / path) != plan['input_sha256'][name]:
            raise ValueError('input changed after protocol freeze: ' + name)
    config = plan['training']
    if config['objective_kind'] != 'grpo':
        raise ValueError('Zip4Rec example supports the GRPO protocol')
    if a.output_name:
        assert a.pilot and a.output_name.isidentifier()
    output = a.run / (a.output_name or ('pilot' if a.pilot else 'model'))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        return
    groups = list(grouped(rows(a.run / 'inputs/train_ten.jsonl')).values())
    if a.pilot:
        groups = groups[:plan['pilot_groups']]
    random.Random(plan['seed']).shuffle(groups)
    assert len(groups) == (plan['pilot_groups'] if a.pilot else plan['train_groups'])
    updates = math.ceil(len(groups) / config['groups_per_update'])
    checkpoints = sorted({min(3, updates), updates}) if a.pilot else plan['planned_checkpoints']
    prepared = Path(plan['prepared'])
    preparation = read_json(prepared / 'complete.json')
    if preparation['protocol_sha256'] != digest(a.run / 'protocol.json'):
        raise ValueError('reference protocol mismatch')
    for split, expected in preparation['reference_sha256'].items():
        if digest(prepared / f'{split}.jsonl') != expected:
            raise ValueError('frozen reference changed: ' + split)
    if digest(prepared / 'base.json') != preparation['base_cache_sha256']:
        raise ValueError('base cache changed after reference preparation')
    if digest(Path(plan['initial_adapter']) / 'adapter_model.safetensors') != plan['initial_adapter_sha256']:
        raise ValueError('initial SFT adapter changed')
    references = {r['sample_id']: r for r in rows(prepared / 'train.jsonl')}
    cache = read_json(prepared / 'base.json')
    present = sorted((cp for cp in output.glob('checkpoint-*')
        if (cp / 'adapter_model.safetensors').exists() and (cp / 'training_state.pt').exists()
        and (cp / 'trainer_state.json').exists()), key=lambda cp: int(cp.name.split('-')[-1]))
    resumed = present[-1] if a.resume and present else None
    if (output / 'updates.jsonl').exists() and resumed is None:
        raise FileExistsError('unsaved training log exists; inspect before restarting')
    model, encode, logits, fingerprint = setup(plan['base_model'], resumed or plan['initial_adapter'],
        True, plan['seed'], config['gradient_checkpointing'])
    assert cache['base_fingerprint'] == fingerprint
    for group in groups:
        for row in group:
            key = 'dense:' + row['sample_id']
            assert cache['records'][key]['input_sha256'] == references[row['sample_id']]['input_sha256'] == signature(row)
    params = [(n, param) for n, param in model.named_parameters() if param.requires_grad]
    initial = {n: param.detach().cpu().clone() for n, param in params}
    optimizer = torch.optim.AdamW([param for _, param in params], lr=config['learning_rate'],
        betas=(.9, .999), eps=1e-8, weight_decay=0.)
    generator = torch.Generator(device='cuda:0').manual_seed(plan['seed'])
    step, cursor, elapsed_before, window, counts_before = 0, 0, 0., None, {}
    if resumed:
        saved = torch.load(resumed / 'training_state.pt', map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == digest(a.run / 'protocol.json')
        assert saved['group_order'] == [g[0]['group_id'] for g in groups]
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampling_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step, cursor, elapsed_before = saved['step'], saved['groups'], saved['elapsed_seconds']
        window, counts_before = saved['old_rollout_window'], saved['forward_counts']
        log_rows = rows(output / 'updates.jsonl')
        if any(r['step'] > step for r in log_rows):
            suffix = str(time.time_ns())
            (output / f'unsaved_updates_before_resume_{suffix}.jsonl').write_text(
                ''.join(json.dumps(r) + '\n' for r in log_rows if r['step'] > step))
        (output / 'updates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in log_rows if r['step'] <= step))
        if (output / 'rollout_audit.jsonl').exists():
            audit_rows = rows(output / 'rollout_audit.jsonl')
            if any(r['step'] > step for r in audit_rows):
                (output / f'unsaved_rollouts_before_resume_{time.time_ns()}.jsonl').write_text(
                    ''.join(json.dumps(r) + '\n' for r in audit_rows if r['step'] > step))
            (output / 'rollout_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit_rows if r['step'] <= step))
    model.train()
    errors = []
    for row in groups[0]:
        encoded = encode(row)
        current = logits(encoded).float()
        with torch.no_grad():
            repeated = logits(encoded).float()
        error = float((current.detach() - repeated).abs().max())
        assert error < 1e-4
        if not resumed:
            base = current.new_tensor(cache['records']['dense:' + row['sample_id']]['base_logits'])
            reference = references[row['sample_id']]
            ref_delta = current.new_tensor([reference['delta_logit_A'], reference['delta_logit_B']])
            error = max(error, float((current.detach() - base - ref_delta).abs().max()))
            assert error < 1e-4
        errors.append(error)
        del encoded, current, repeated
    write_json(output / 'initialization_gate.json', {'maximum_logit_error': max(errors),
        'initial_adapter_sha256': plan['initial_adapter_sha256'], 'resumed': bool(resumed),
        'reference_sha256': digest(prepared / 'train.jsonl'), 'physical_gpu_uuid': fingerprint['inference_runtime']['gpu_uuid']})
    write_json(output / 'run_config.json', {'created_utc': now(), 'training': config,
        'protocol_sha256': digest(a.run / 'protocol.json'), 'groups': len(groups), 'updates': updates,
        'checkpoints': checkpoints, 'trainable_parameters': sum(param.numel() for _, param in params),
        'reference_fingerprint': fingerprint, 'resumed_from': str(resumed) if resumed else None})
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    forward_counts = {k: counts_before.get(k, 0) + v for k, v in logits.counts.items()}
    def group_delta(group):
        values = []
        for row in group:
            raw = logits(encode(row)).float()
            values.append(raw - raw.new_tensor(cache['records']['dense:' + row['sample_id']]['base_logits']))
        return torch.stack(values)
    with (output / 'updates.jsonl').open('a') as stream, (output / 'rollout_audit.jsonl').open('a') as audit_stream:
        while cursor < len(groups):
            if window is None or cursor >= window['end']:
                window_start, window_end = cursor, min(cursor + config['rollout_window_groups'], len(groups))
                old_rollouts = []
                with torch.no_grad():
                    for group in groups[window_start:window_end]:
                        delta = group_delta(group)
                        old_logits = policy_logits(delta[:, 0] - delta[:, 1], delta.new_tensor([r['candidate_score'] for r in group]), config['policy_mode'])
                        positive = next(i for i, r in enumerate(group) if r['label'] == 'A')
                        old_rollouts.append(make_rollout(old_logits, positive, config, generator))
                window = {'start': window_start, 'end': window_end, 'actor_step': step,
                    'group_ids': [g[0]['group_id'] for g in groups[window_start:window_end]], 'rollouts': old_rollouts}
            end = min(cursor + config['groups_per_update'], len(groups))
            batch = groups[cursor:end]
            optimizer.zero_grad(set_to_none=True)
            infos, audit_groups = [], []
            for offset, group in enumerate(batch):
                delta = group_delta(group)
                positive = next(i for i, r in enumerate(group) if r['label'] == 'A')
                negative = next(i for i, r in enumerate(group) if r['label'] == 'B' and r['_replay'])
                rollout = window['rollouts'][cursor + offset - window['start']]
                loss, info = objective(delta, delta.new_tensor([references[r['sample_id']]['condition_score'] for r in group]),
                    delta.new_tensor([r['candidate_score'] for r in group]), positive, negative,
                    group[negative]['_replay_weight'], config, [r['candidate_item_id'] for r in group], rollout)
                assert torch.isfinite(loss)
                audit_group = {'group_id': group[0]['group_id'], 'delta': delta.detach().cpu().tolist(),
                    'positive': positive, 'current_logits': policy_logits(delta[:, 0] - delta[:, 1],
                        delta.new_tensor([r['candidate_score'] for r in group]), config['policy_mode']).detach().cpu().tolist(),
                    'task_loss': info['task_loss']}
                audit_group['rollout'] = {k: rollout[k].tolist() for k in ['permutations', 'reward', 'old_logp', 'advantages', 'old_logits']}
                audit_groups.append(audit_group)
                (loss / len(batch)).backward()
                infos.append(info)
                del delta, loss
                if torch.cuda.max_memory_allocated() > config['memory_ceiling_gib'] * 1024 ** 3:
                    raise MemoryError('training memory ceiling exceeded')
            assert all(param.grad is not None and torch.isfinite(param.grad).all() for _, param in params)
            norm = float(torch.nn.utils.clip_grad_norm_([param for _, param in params], config['max_grad_norm']))
            assert math.isfinite(norm) and norm > 0
            for pg in optimizer.param_groups:
                pg['lr'] = config['learning_rate'] * lr_multiplier(step, updates, config['warmup_ratio'])
            optimizer.step()
            metrics = {k: sum(info[k] for info in infos) / len(infos) for k in infos[0]}
            metrics['ratio_max_abs_deviation'] = max(info.get('ratio_max_abs_deviation', 0.) for info in infos)
            step, cursor = step + 1, end
            forward_counts = {k: counts_before.get(k, 0) + v for k, v in logits.counts.items()}
            metrics.update(step=step, epoch=1, groups=cursor, batch_groups=len(batch), grad_norm=norm,
                clipped=float(norm > config['max_grad_norm']), clip_multiplier=min(1., config['max_grad_norm'] / norm),
                learning_rate=optimizer.param_groups[0]['lr'], elapsed_seconds=elapsed_before + time.monotonic() - started,
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), **forward_counts,
                rollout_window_start=window['start'],
                rollout_old_actor_step=window['actor_step'],
                total_sampled_permutations=window['end'] * config['samples_per_group'])
            if step == 1 and not resumed:
                assert abs(metrics['kl']) < 1e-9
                assert metrics['ratio_max_abs_deviation'] < 1e-7
            stream.write(json.dumps(metrics) + '\n'); stream.flush()
            audit_stream.write(json.dumps({'step': step, 'old_actor_step': window['actor_step'],
                                           'window_start': window['start'], 'groups': audit_groups}) + '\n')
            audit_stream.flush()
            write_json(output / 'live_state.json', metrics)
            if step == 1 or step % 10 == 0 or a.pilot:
                print(json.dumps(metrics), flush=True)
            if step in checkpoints:
                cp = output / f'checkpoint-{step}'
                pending = output / f'.checkpoint-{step}.pending'
                pending.mkdir(exist_ok=True)
                model.save_pretrained(pending)
                state = {'optimizer': optimizer.state_dict(), 'sampling_rng': generator.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'step': step, 'groups': cursor, 'elapsed_seconds': metrics['elapsed_seconds'],
                    'protocol_sha256': digest(a.run / 'protocol.json'), 'group_order': [g[0]['group_id'] for g in groups],
                    'old_rollout_window': window, 'forward_counts': forward_counts}
                torch.save(state, pending / 'training_state.pt')
                write_json(pending / 'trainer_state.json', {'global_step': step, 'max_steps': updates,
                    'groups': cursor, 'epoch': cursor / len(groups), 'old_rollout_window_preserved': True,
                    'protocol_sha256': digest(a.run / 'protocol.json')})
                if cp.exists():
                    cp.rename(output / f'incomplete-checkpoint-{step}-{time.time_ns()}')
                pending.rename(cp)
    change = math.sqrt(sum(float((param.detach().cpu() - initial[n]).double().square().sum()) for n, param in params))
    assert math.isfinite(change) and (change > 0 or (resumed and step == updates))
    write_json(output / 'complete.json', {'completed_utc': now(), 'global_step': step, 'groups': cursor,
        'epochs': 1, 'parameter_change_l2_since_loaded_adapter': change,
        'seconds': elapsed_before + time.monotonic() - started,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(), 'forward_counts': forward_counts,
        'sampled_permutations': len(groups) * config['samples_per_group'],
        'checkpoint_sha256': {str(s): digest(output / f'checkpoint-{s}/adapter_model.safetensors') for s in checkpoints}})
    print('training complete', step, 'updates', flush=True)


if __name__ == '__main__':
    main()
