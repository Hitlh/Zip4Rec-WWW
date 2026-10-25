"""Create a local immutable GRPO run from portable paths and configuration."""
import argparse
import shutil
import math
from pathlib import Path
from .common import digest, grouped, read_json, rows, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--initial-adapter', type=Path, required=True)
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--valid', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-valid-scores', type=Path)
    a = p.parse_args()
    cfg = read_json(a.config)
    if (cfg['training']['objective_kind'] != 'grpo' or cfg['fusion_alpha'] != 1.5
            or cfg['training']['training_alpha'] != 1.5):
        raise ValueError('this CD implementation requires GRPO with fixed alpha=1.5')
    if cfg['training']['rollout_window_groups'] % cfg['training']['groups_per_update']:
        raise ValueError('old-policy windows must align with optimizer batches')
    if cfg['training']['epochs'] != 1 or cfg['training']['optimization_epochs'] != 1:
        raise ValueError('the original CD GRPO loop uses one epoch and one optimization pass')
    train, valid = rows(a.train), rows(a.valid)
    tg, vg = grouped(train), grouped(valid)
    if not tg or not vg:
        raise ValueError('empty train/validation cohort')
    for group in tg.values():
        replay = [r for r in group if r.get('_replay')]
        if len(replay) != 2 or sorted(r['label'] for r in replay) != ['A', 'B']:
            raise ValueError('each training group needs its original SFT replay pair')
    if any(r.get('split') != 'train' for r in train):
        raise ValueError('training input must declare split=train')
    if any(r.get('split') not in ['valid', 'validation'] for r in valid):
        raise ValueError('validation input must declare split=valid or validation')
    if set(tg) & set(vg):
        raise ValueError('training and validation request IDs overlap')
    # The original CD protocol reuses earlier events from evaluation users for SFT.
    # Its disjoint cohorts are validation vs. test; the held-out events must stay unseen.
    heldout = {}
    for row in valid:
        heldout.setdefault(row['user_id'], set()).add(row['target_item_id'])
    for row in train:
        if (set(row.get('history_item_ids', [])) | {row['candidate_item_id'], row['target_item_id']}) & heldout.get(row['user_id'], set()):
            raise ValueError('held-out validation positive appears in training')
    adapter = a.initial_adapter.resolve()
    adapter_sha = digest(adapter / 'adapter_model.safetensors')
    if not (a.base_model / 'config.json').is_file():
        raise FileNotFoundError(a.base_model / 'config.json')
    a.output.mkdir(parents=True, exist_ok=False)
    inputs = a.output / 'inputs'
    inputs.mkdir()
    shutil.copyfile(a.train, inputs / 'train_ten.jsonl')
    shutil.copyfile(a.valid, inputs / 'valid.jsonl')
    updates = math.ceil(len(tg) / cfg['training']['groups_per_update'])
    checkpoints = sorted({math.ceil(updates * f) for f in cfg['training']['checkpoint_fractions']})
    plan = dict(experiment='Zip4Rec_CD_LightGCN', seed=cfg['seed'], training=cfg['training'],
                planned_checkpoints=checkpoints, train_groups=len(tg),
                validation_natural_hit_users=len(vg), pilot_groups=min(12, len(tg)),
                base_model=str(a.base_model.resolve()), initial_adapter=str(adapter),
                initial_adapter_sha256=adapter_sha, prepared=str((a.output / 'prepared').resolve()),
                input_sha256=dict(train_ten=digest(inputs / 'train_ten.jsonl'), valid=digest(inputs / 'valid.jsonl')),
                code_sha256={f.name: digest(f) for f in Path(__file__).parent.glob('*.py')},
                historical_main_alpha=cfg['fusion_alpha'])
    if a.historical_valid_scores:
        plan['historical_valid_scores'] = str(a.historical_valid_scores.resolve())
    write_json(a.output / 'protocol.json', plan)


if __name__ == '__main__':
    main()
