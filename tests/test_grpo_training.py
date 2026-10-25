"""Exercise the GRPO update/checkpoint loop with a tiny CPU model."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from ranking import train_grpo
    from ranking.common import digest, rows, signature, write_json, write_rows


@unittest.skipUnless(torch is not None, 'GRPO loop test requires PyTorch')
class GrpoTrainingTests(unittest.TestCase):
    def test_interrupted_checkpoint_resume_matches_uninterrupted_training(self):
        initial = torch.tensor([[i * .02 - .1, i * .01 - .3] for i in range(10)])

        class TinyModel(torch.nn.Module):
            def __init__(self, state=None):
                super().__init__()
                self.weights = torch.nn.Parameter(initial.clone())
                if state is not None:
                    self.load_state_dict(state)

            def save_pretrained(self, path):
                if interrupt[0] and path.name == '.checkpoint-2.pending':
                    raise RuntimeError('simulated interrupted checkpoint')
                torch.save(self.state_dict(), path / 'adapter_model.safetensors')

        class CpuRuntime:
            # Only replace device-specific services; use real tensors, AdamW,
            # objectives, checkpoint writes, log recovery and random states.
            cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0, get_rng_state_all=lambda: [],
                set_rng_state_all=lambda state: None)

            def Generator(self, **kwargs):
                return torch.Generator()

            def __getattr__(self, name):
                return getattr(torch, name)

        def setup(model_path, adapter, *args):
            state_path = Path(adapter) / 'adapter_model.safetensors'
            state = torch.load(state_path, weights_only=False) if state_path.exists() else None
            model = TinyModel(state)
            def logits(row):
                logits.counts['candidate_forward_calls'] += 1
                return model.weights[int(row['candidate_item_id'])]
            logits.counts = dict(candidate_forward_calls=0)
            return model, lambda row: row, logits, fingerprint

        def prepare(root):
            (root / 'inputs').mkdir(parents=True)
            records = [dict(sample_id=f'g{g}-i{i}', group_id=f'g{g}', user_id=f'u{g}',
                candidate_item_id=str(i), target_item_id=str(g), label='A' if i == g else 'B',
                candidate_score=10 - i, _replay=i in [g, g + 1], _replay_weight=1.,
                messages=[dict(role='user', content=[dict(type='text', text=f'g{g}-i{i}')])])
                for g in range(5) for i in range(10)]
            for split in ['train_ten', 'valid']:
                write_rows(root / 'inputs' / f'{split}.jsonl', records)
            adapter = root / 'initial'
            adapter.mkdir()
            torch.save(TinyModel().state_dict(), adapter / 'adapter_model.safetensors')
            config = json.loads((Path(__file__).parents[1] / 'configs/cd_lightgcn/grpo.json').read_text())
            config['training'].update(groups_per_update=2, rollout_window_groups=4, learning_rate=1e-3)
            prepared = root / 'prepared'
            prepared.mkdir()
            plan = dict(seed=6062, training=config['training'], code_sha256={
                'train_grpo.py': digest(Path(train_grpo.__file__))},
                input_sha256={key: digest(root / 'inputs' / f'{key}.jsonl') for key in ['train_ten', 'valid']},
                base_model=str(root / 'base'), initial_adapter=str(adapter),
                initial_adapter_sha256=digest(adapter / 'adapter_model.safetensors'),
                prepared=str(prepared), train_groups=5, planned_checkpoints=[1, 2, 3])
            write_json(root / 'protocol.json', plan)
            reference, cache = [], {}
            for row in records:
                delta = initial[int(row['candidate_item_id'])]
                reference.append(dict(sample_id=row['sample_id'], input_sha256=signature(row),
                    condition_score=float(delta[0] - delta[1]),
                    delta_logit_A=float(delta[0]), delta_logit_B=float(delta[1])))
                cache['dense:' + row['sample_id']] = dict(input_sha256=signature(row), base_logits=[0., 0.])
            write_rows(prepared / 'train.jsonl', reference)
            write_json(prepared / 'base.json', dict(base_fingerprint=fingerprint, records=cache))
            write_json(prepared / 'complete.json', dict(protocol_sha256=digest(root / 'protocol.json'),
                reference_sha256=dict(train=digest(prepared / 'train.jsonl')),
                base_cache_sha256=digest(prepared / 'base.json')))

        def run(root):
            with patch.object(train_grpo, 'torch', CpuRuntime()), patch.object(train_grpo, 'setup', setup), \
                    patch('sys.argv', ['train_grpo', '--run', str(root), '--resume']), \
                    contextlib.redirect_stdout(io.StringIO()):
                train_grpo.main()

        fingerprint = dict(inference_runtime=dict(gpu_uuid='cpu-test'))
        interrupt = [False]
        with tempfile.TemporaryDirectory() as directory:
            full, resumed = Path(directory) / 'full', Path(directory) / 'resumed'
            for root in [full, resumed]:
                prepare(root)
            run(full)
            interrupt[0] = True
            with self.assertRaisesRegex(RuntimeError, 'simulated interrupted checkpoint'):
                run(resumed)
            self.assertEqual(len(rows(resumed / 'model/updates.jsonl')), 2)
            interrupt[0] = False
            run(resumed)
            self.assertTrue(list((resumed / 'model').glob('unsaved_updates_before_resume_*.jsonl')))
            self.assertEqual(rows(full / 'model/rollout_audit.jsonl'), rows(resumed / 'model/rollout_audit.jsonl'))
            for step in [1, 2, 3]:
                before = torch.load(full / f'model/checkpoint-{step}/adapter_model.safetensors', weights_only=False)
                after = torch.load(resumed / f'model/checkpoint-{step}/adapter_model.safetensors', weights_only=False)
                self.assertTrue(torch.equal(before['weights'], after['weights']))
            saved = torch.load(resumed / 'model/checkpoint-3/training_state.pt', weights_only=False)
            self.assertEqual(saved['step'], 3)
            self.assertEqual(saved['groups'], 5)


if __name__ == '__main__':
    unittest.main()
