import argparse
import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from semantic_context.filter_data import iterative_kcore
from semantic_context.datasets import audit_source, collect, export
from semantic_context.generate_assets import prepare_items, parse_items, prepare_profiles, parse_profiles
from semantic_context.llm_profile_common import read_jsonl, sha256_json

ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def fake_summary(title):
    return dict(canonical_title=title, item_type='album', artists=[], genres=['Jazz'],
                styles_or_moods=['calm'], era='', grounded_summary=title,
                knowledge_enrichment='', compact_summary='A fictional jazz album.',
                knowledge_used=False, uncertainties=[])


def fake_profile(count):
    return dict(preference_summary='Sparse evidence of interest in jazz.',
                preferred_genres=([dict(name='Jazz', strength='weak', evidence_count=1)] if count else []),
                preferred_artists_or_scenes=[], preferred_eras=[], preferred_moods_or_styles=[],
                diversity='unknown', evidence_strength='sparse', uncertainties=[])


def mock_results(sources, path, kind):
    records = []
    for row in sources:
        value = fake_summary(row['title']) if kind == 'item' else fake_profile(row['long_history_count'])
        records.append(dict(custom_id=row['custom_id'], error=None, response=dict(status_code=200,
            body=dict(id='offline-fixture', model='offline-fixture',
                      choices=[dict(message=dict(content=json.dumps(value)))]))))
    write_rows(path, records)


class DataPreparationTests(unittest.TestCase):
    def test_iterative_kcore_cascade(self):
        rows = [('u1', 'i1', 5, 1), ('u1', 'i2', 5, 2), ('u2', 'i1', 5, 3)]
        kept, rounds = iterative_kcore(rows, 2, 2)
        self.assertEqual(kept, [])
        self.assertGreater(rounds, 1)

    def test_item_and_profile_generation_validation_and_request_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            metadata = p / 'metadata.csv'
            with metadata.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['item_id', 'original_item_id', 'title',
                    'categories', 'brand', 'description_clean', 'description_status'])
                writer.writeheader()
                writer.writerow(dict(item_id=0, original_item_id='ASIN0', title='Fixture Album',
                    categories='Music', brand='', description_clean='A jazz album.', description_status='usable'))
            requests = p / 'items_request'
            prepare_items(argparse.Namespace(item_metadata=metadata, output_dir=requests, model='fixture',
                response_format='json-schema', description_max_chars=4000, item_ids=None,
                sample_size=None, seed=42))
            result = p / 'item_results.jsonl'
            mock_results(read_jsonl(requests / 'item_sources.jsonl'), result, 'item')
            items = p / 'items'
            self.assertEqual(parse_items(argparse.Namespace(request_dir=requests, results=result,
                output_dir=items, allow_partial_results=False)), 0)
            contexts = p / 'contexts.jsonl'
            write_rows(contexts, [dict(profile_history_key='k', long_history_item_ids=[0])])
            prof_req = p / 'profile_requests'
            prepare_profiles(argparse.Namespace(contexts=contexts,
                item_summaries=items / 'item_summaries.jsonl', output_dir=prof_req,
                model='fixture', response_format='json-schema'))
            request = read_jsonl(prof_req / 'requests.jsonl')[0]
            content = request['body']['messages'][1]['content']
            self.assertIn('Fixture Album', content)
            for forbidden in ['target_item_id', 'candidate_item_ids', 'user_id', 'profile_history_key']:
                self.assertNotIn(forbidden, content)
            source = read_jsonl(prof_req / 'profile_request_sources.jsonl')
            prof_result = p / 'profile_results.jsonl'
            mock_results(source, prof_result, 'profile')
            self.assertEqual(parse_profiles(argparse.Namespace(request_dir=prof_req, results=prof_result,
                output_dir=p / 'profiles', allow_partial_results=False)), 0)
            # An unsupported strong preference from one history record must fail validation.
            response = read_jsonl(prof_result)[0]
            value = fake_profile(1)
            value['preferred_genres'][0]['strength'] = 'strong'
            response['response']['body']['choices'][0]['message']['content'] = json.dumps(value)
            write_rows(prof_result, [response])
            self.assertEqual(parse_profiles(argparse.Namespace(request_dir=prof_req, results=prof_result,
                output_dir=p / 'bad_profiles', allow_partial_results=False)), 2)

    def test_exact_prefix_and_all_future_positive_masking(self):
        sequence = [(i, f'I{i}') for i in range(6)]
        long, recent = [], ['I0', 'I1', 'I2']
        row = dict(user_id='u', split='train', target_item_id='I3', target_timestamp=3,
                   target_injected=False, candidate_item_ids=['I3'] + [f'N{i}' for i in range(9)],
                   history_item_ids=recent, long_history_item_ids=long, recent_history_item_ids=recent,
                   profile_history_key=sha256_json({'ordered_item_ids': long}))
        audit_source(row, {'u': sequence})
        bad = copy.deepcopy(row)
        bad['candidate_item_ids'][-1] = 'I5'
        with self.assertRaises(ValueError):
            audit_source(bad, {'u': sequence})
        bad = copy.deepcopy(row)
        bad['history_item_ids'] = ['I1', 'I0', 'I2']
        with self.assertRaises(ValueError):
            audit_source(bad, {'u': sequence})

    @unittest.skipUnless(importlib.util.find_spec('httpx'), 'request runner requires httpx')
    def test_compatible_endpoint_v1_not_duplicated(self):
        from semantic_context.run_requests import endpoint_url
        self.assertEqual(endpoint_url('https://provider.example/v1', '/v1/chat/completions'),
                         'https://provider.example/v1/chat/completions')

    @unittest.skipUnless(importlib.util.find_spec('httpx'), 'request runner requires httpx')
    def test_synchronous_requests_resume_and_retry_failed_or_invalid_results(self):
        import httpx
        from semantic_context.run_requests import run_sync, retry_input
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            requests, results = p / 'requests.jsonl', p / 'results.jsonl'
            write_rows(requests, [dict(custom_id=str(i), method='POST', url='/v1/chat/completions',
                body=dict(model='fixture', messages=[], metadata=dict(id=i))) for i in range(3)])
            calls = []
            def handle(request):
                self.assertEqual(str(request.url), 'https://provider.example/v1/chat/completions')
                index = json.loads(request.content)['metadata']['id']
                calls.append(index)
                return httpx.Response(200 if index != 1 else 400,
                    json=dict(choices=[dict(message=dict(content='{}'))]))
            args = argparse.Namespace(requests=requests, output=results, max_attempts=1,
                workers=2, limit=None, min_interval=0.)
            for _ in range(2):
                with patch('semantic_context.run_requests.client_for', return_value=(
                        httpx.Client(transport=httpx.MockTransport(handle)), 'https://provider.example/v1')):
                    status = run_sync(args)
            self.assertEqual(sorted(calls), [0, 1, 2])
            self.assertEqual(status, 0)  # Second run has no pending requests.
            self.assertEqual(len(read_jsonl(results)), 3)
            failures = p / 'failures.jsonl'
            write_rows(failures, [dict(custom_id='2')])
            retry = p / 'retry.jsonl'
            retry_input(argparse.Namespace(requests=requests, results=results,
                validation_failures=failures, output=retry))
            self.assertEqual([r['custom_id'] for r in read_jsonl(retry)], ['1', '2'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU LightGCN integration requires PyTorch')
    def test_raw_to_semantics_densepages_and_metrics_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            reviews, metadata = p / 'reviews.jsonl', p / 'metadata.jsonl'
            meta = [dict(parent_asin=f'I{i:02}', title=f'Fixture Album {i}',
                         description=['<p>A jazz album.</p>'], categories=['Music'], store='Fixture') for i in range(24)]
            records = []
            for u in range(8):
                for j in range(12):
                    records.append(dict(user_id=f'u{u}', parent_asin=f'I{(u * 3 + j) % 24:02}',
                                        rating=4 if j in [3, 4] else 5, timestamp=1000 + j))
            records.append(dict(user_id='u0', parent_asin='I00', rating=1, timestamp=2000))
            write_rows(reviews, records)
            write_rows(metadata, meta)
            work = p / 'work'
            subprocess.run([sys.executable, '-m', 'semantic_context.pipeline', 'prepare',
                '--reviews', str(reviews), '--metadata', str(metadata), '--output', str(work),
                '--validation-users', '2', '--test-users', '2', '--user-k', '10', '--item-k', '2',
                '--epochs', '1', '--batch-size', '128', '--device', 'cpu', '--asset-model', 'fixture'],
                cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            filtered = (work / 'filtered/positive_kcore.tsv').read_text()
            self.assertFalse(any(line.split('\t')[1:3] == ['u0', 'I00'] for line in filtered.splitlines()))
            with (work / 'filtered/item_metadata.csv').open() as stream:
                cleaned = list(csv.DictReader(stream))
            self.assertTrue(all(r['description_clean'] == 'A jazz album.' for r in cleaned))
            result = p / 'item_results.jsonl'
            mock_results(read_jsonl(work / 'item_requests/item_sources.jsonl'), result, 'item')
            subprocess.run([sys.executable, '-m', 'semantic_context.pipeline', 'finish-items',
                '--work', str(work), '--results', str(result)], cwd=ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            prof_result = p / 'profile_results.jsonl'
            mock_results(read_jsonl(work / 'profile_requests/profile_request_sources.jsonl'), prof_result, 'profile')
            subprocess.run([sys.executable, '-m', 'semantic_context.pipeline', 'finish-profiles',
                '--work', str(work), '--results', str(prof_result)], cwd=ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            manifest = json.loads((work / 'datasets/manifest.json').read_text())
            for split in ['train', 'valid', 'test']:
                self.assertTrue((work / f'datasets/{split}.jsonl').is_file())
            self.assertEqual(manifest['splits']['valid']['all_groups'], 2)
            self.assertEqual(manifest['splits']['test']['all_groups'], 2)
            self.assertEqual(len(read_jsonl(work / 'datasets/valid_all.jsonl')), 2)
            pairs = read_jsonl(work / 'datasets/train_pairs.jsonl')
            self.assertEqual(len(pairs), manifest['splits']['train']['natural_hit_groups'] * 2)
            self.assertEqual(sum(r['label'] == 'A' for r in pairs), len(pairs) // 2)
            self.assertTrue(all('user_context' in r and 'candidate_payload' in r for r in pairs))
            # Verify the module boundary using actual freshly prepared records
            # and Pillow images, without an API, processor, or model weights.
            dense = p / 'densepage'
            subprocess.run([sys.executable, '-m', 'optical_compression.pipeline',
                '--datasets', str(work / 'datasets'), '--output', str(dense)],
                cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            from ranking.evaluate import evaluate
            for split in ['train', 'valid', 'test']:
                candidates = read_jsonl(dense / split / 'candidates.jsonl')
                self.assertEqual(len(candidates), manifest['splits'][split]['natural_hit_groups'] * 10)
                self.assertTrue(all(Path(row['densepage_path']).is_file() for row in candidates))
                rendered_pairs = read_jsonl(dense / split / 'sft_pairs.jsonl')
                self.assertEqual({row['sample_id'] for row in rendered_pairs},
                                 {row['sample_id'] for row in candidates if row['_replay']})
                if split != 'train':
                    scores = [dict(row, condition_score=0.) for row in candidates]
                    result = evaluate(scores, alpha=0., all_users=manifest['splits'][split]['all_groups'])
                    recall = manifest['splits'][split]['natural_hit_groups'] / manifest['splits'][split]['all_groups']
                    self.assertAlmostEqual(result['end_to_end_fused']['NDCG@10'],
                                           result['fused']['NDCG@10'] * recall)
            source = work / 'sources/train_sources.jsonl'
            source.write_text(source.read_text() + '\n')
            with self.assertRaises(ValueError):
                export(work / 'sources', work / 'items/item_summaries.jsonl',
                       work / 'profiles/user_profiles.jsonl', p / 'corrupt_export')


if __name__ == '__main__':
    unittest.main()
