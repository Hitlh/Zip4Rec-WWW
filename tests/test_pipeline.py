import copy
import json
import math
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from semantic_context.assets import load_profiles, load_validated_items, read_jsonl
from semantic_context.prepare import organize
from optical_compression.build import materialize
from optical_compression.selected.compact import render_compact
from ranking.evaluate import evaluate
from ranking.rerank import rerank
from workflow import digest
from semantic_context.samples import pointwise

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/cd_lightgcn'


def write_temporary_fixtures(directory):
    """Build tiny synthetic inputs during tests; no dataset files are distributed."""
    items = [dict(original_item_id=f'CD{i:02}', title=f'Synthetic Album {i}',
                  summary=dict(canonical_title=f'Synthetic Album {i}', artists=['Example Artist'],
                               genres=['Jazz' if i % 2 else 'Rock'], era='2000s',
                               styles_or_moods=['melodic'], summary='',
                               compact_summary=f'A fictional album, record {i}.')) for i in range(14)]
    profile = dict(preference_summary='Prefers melodic rock and jazz.',
                   preferred_genres=[dict(name='Rock', strength='strong', evidence_count=3)],
                   preferred_artists_or_scenes=[], preferred_eras=['2000s'],
                   preferred_moods_or_styles=['melodic'], diversity='moderate',
                   evidence_strength='moderate', uncertainties=[])
    profiles = []
    def write(name, records):
        (directory / name).write_text(''.join(json.dumps(r) + '\n' for r in records))
    for split in ['train', 'valid', 'test']:
        user = 'demo-' + split
        group = dict(sample_id=f'{split}-synthetic', split=split, user_id=user,
                     target_item_id='CD07', target_injected=False,
                     candidate_item_ids=[f'CD{i:02}' for i in range(10)],
                     candidate_scores=[1-i*.07 for i in range(10)],
                     history_item_ids=['CD10', 'CD11', 'CD12', 'CD13'],
                     long_history_item_ids=['CD10'], recent_history_item_ids=['CD11', 'CD12', 'CD13'],
                     profile_history_key=user)
        write(f'{split}_groups.jsonl', [group])
        profiles.append(dict(profile_history_key=user, profile=profile))
    write('item_summaries.jsonl', items)
    write('user_profiles.jsonl', profiles)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture_dir = Path(self.temporary.name)
        write_temporary_fixtures(self.fixture_dir)
        self.items = load_validated_items(self.fixture_dir / 'item_summaries.jsonl')
        self.profiles = load_profiles(self.fixture_dir / 'user_profiles.jsonl')
        self.group = read_jsonl(self.fixture_dir / 'train_groups.jsonl')[0]

    def test_reject_history_and_candidate_leakage(self):
        for field, value in [('target_injected', True), ('target_item_id', 'CD10'),
                             ('recent_history_item_ids', ['CD11'])]:
            bad = dict(self.group, **{field: value})
            with self.assertRaises(ValueError):
                organize(bad, self.profiles, self.items)
        bad = copy.deepcopy(self.group)
        bad['candidate_item_ids'][0] = 'CD10'
        with self.assertRaises(ValueError):
            organize(bad, self.profiles, self.items)

    def test_sft_pair_matches_grpo_replay_and_page_is_label_free(self):
        context = organize(self.group, self.profiles, self.items)
        with tempfile.TemporaryDirectory() as folder:
            _, samples = pointwise([context], self.items)
            pairs, candidates, tasks = materialize(samples, Path(folder))
            self.assertEqual([r['label'] for r in pairs], ['A', 'B'])
            self.assertEqual(len(candidates), 10)
            self.assertEqual({r['sample_id'] for r in pairs},
                             {r['sample_id'] for r in candidates if r['_replay']})
            negative = pairs[1]
            self.assertEqual(negative['sample_weight'], 1.)
            for row, task in zip(candidates, tasks):
                self.assertEqual(set(task), {'sample_id', 'profile', 'item', 'path'})
                self.assertNotIn('Candidate music items:', task['profile'])
                self.assertEqual(len([m for m in row['messages'] if m['role'] == 'assistant']), 1)
            page = render_compact(tasks[0])
            self.assertEqual(page['width'], 640)
            self.assertEqual(page['height'] % 32, 0)
            self.assertTrue(Path(tasks[0]['path']).is_file())

    def test_ranking_without_labels_and_metric_denominators(self):
        scores = [dict(sample_id=str(i), group_id='one', user_id='u', candidate_item_id=f'{i:02}',
                       target_item_id='02', label='A' if i == 2 else 'B',
                       candidate_score=10-i, condition_score=0.) for i in range(10)]
        result = evaluate(scores, alpha=0., all_users=100)
        self.assertAlmostEqual(result['fused']['NDCG@10'], 1 / math.log2(4))
        self.assertAlmostEqual(result['fused']['MRR@10'], 1 / 3)
        self.assertAlmostEqual(result['end_to_end_fused']['NDCG@10'], .005)
        unlabeled = [{k: v for k, v in row.items() if k not in ['label', 'target_item_id']} for row in scores]
        self.assertEqual(rerank(unlabeled, 0.)[0]['ranking'], result['rankings'][0]['ranking'])
        with self.assertRaises(ValueError):
            evaluate(scores, all_users=0)
        with self.assertRaises(ValueError):
            rerank([dict(r, condition_score=float('nan')) for r in unlabeled])

    def test_deterministic_ties_and_invalid_groups(self):
        scores = [dict(sample_id=str(i), group_id='g', user_id='u', candidate_item_id=f'{i:02}',
                       target_item_id='02', label='A' if i == 2 else 'B',
                       candidate_score=0., condition_score=0.) for i in reversed(range(10))]
        self.assertEqual(rerank(scores)[0]['ranking'], [f'{i:02}' for i in range(10)])
        with self.assertRaises(ValueError):
            evaluate(scores[:-1])
        scores[0]['sample_id'] = scores[1]['sample_id']
        with self.assertRaises(ValueError):
            evaluate(scores)

    def test_freeze_accepts_earlier_same_user_events_but_rejects_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / 'base').mkdir()
            (p / 'base/config.json').write_text('{}')
            (p / 'adapter').mkdir()
            (p / 'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            def group(split, offset, positive):
                return [dict(sample_id=f'{split}-{i}', group_id=split, user_id='shared-user',
                    split=split, candidate_item_id=f'item{offset+i}', target_item_id=f'item{positive}',
                    candidate_score=10-i, label='A' if offset+i == positive else 'B',
                    _replay=(offset+i in [positive, positive+1]), _replay_weight=1.) for i in range(10)]
            train, valid = group('train', 0, 2), group('valid', 20, 20)
            def write(name, rows):
                (p / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            write('train.jsonl', train)
            write('valid.jsonl', valid)
            command = [sys.executable, '-m', 'ranking.make_run', '--config', str(CONFIGS / 'grpo.json'),
                       '--base-model', str(p / 'base'), '--initial-adapter', str(p / 'adapter'),
                       '--train', str(p / 'train.jsonl'), '--valid', str(p / 'valid.jsonl'), '--output']
            subprocess.run(command + [str(p / 'ok')], cwd=ROOT, check=True, capture_output=True, text=True)
            train[0]['candidate_item_id'] = 'item20'
            write('train.jsonl', train)
            failed = subprocess.run(command + [str(p / 'bad')], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('held-out validation positive', failed.stderr)

    def test_all_splits_render_from_semantic_samples_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            datasets = p / 'datasets'
            datasets.mkdir()
            manifest = dict(output_sha256={}, splits={})
            for split in ['train', 'valid', 'test']:
                group = read_jsonl(self.fixture_dir / f'{split}_groups.jsonl')[0]
                context = organize(group, self.profiles, self.items)
                _, samples = pointwise([context], self.items)
                path = datasets / f'{split}_candidates.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in samples))
                manifest['output_sha256'][path.name] = digest(path)
                manifest['splits'][split] = dict(all_groups=1, natural_hit_groups=1)
            (datasets / 'manifest.json').write_text(json.dumps(manifest))
            command = [sys.executable, '-m', 'optical_compression.pipeline', '--datasets',
                       str(datasets), '--output', str(p / 'dense')]
            for _ in range(2):
                subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            for split in ['train', 'valid', 'test']:
                records = read_jsonl(p / f'dense/{split}/candidates.jsonl')
                self.assertEqual(len(records), 10)
                self.assertTrue(all(Path(r['densepage_path']).is_file() for r in records))
            complete = json.loads((p / 'dense/complete.json').read_text())
            self.assertFalse(complete['processor_validated'])
            self.assertEqual(complete['layout_kind'], 'selected_compact')
            self.assertIn('layout_signature', complete)
            for split in ['train', 'valid', 'test']:
                rendered = json.loads((p / f'dense/{split}/manifest.json').read_text())
                self.assertEqual(rendered['layout']['kind'], 'selected_compact')
                self.assertEqual(rendered['layout']['signature'], complete['layout_signature'])


if __name__ == '__main__':
    unittest.main()
