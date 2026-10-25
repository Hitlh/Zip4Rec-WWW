"""Offline checks for agent execution, immutable content, qualification, and integration."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

from optical_compression.document import document, fingerprint, plain_text, select_examples
from optical_compression.layout_engine import (Constraints, FrozenLayout, execute_layout,
    render_plan, validate_source, FIXED_DECORATIONS)
from optical_compression.selected.compact import render_compact
from optical_compression.presentation import prepare_document
from optical_compression.measurement import reading_result
from optical_compression.optimizer.agent import ChatClient, TOOLS, run_agent
from optical_compression.optimizer.tools import LayoutSession
from optical_compression.strategy import RenderStrategy, SELECTED
from run_pipeline import select_layout
from workflow import Workflow
from workflow import digest

ROOT = Path(__file__).resolve().parents[1]
INITIAL = (ROOT / 'optical_compression/layouts/initial.py').read_text()
PACKED = INITIAL.replace('max(640,', 'max(480,').replace('pack=False', 'pack=True').replace("'line_gap': 4", "'line_gap': 1")
COMPACT_EQUIVALENT = (ROOT / 'tests/fixtures/compact_equivalent.py').read_text()


def examples():
    output = []
    for i in range(10):
        doc = document(f'Interests: acoustic music {i}.\nEvidence: recent listening.',
                       dict(title=f'Album {i}', artists=['A', 'B'], summary='Detailed music description. ' * (i + 1)))
        output.append(dict(id=f'example_{i+1:02d}', document=doc, document_sha256=fingerprint(doc)))
    return output


class FakeMeter:
    identity = dict(processor_files={'processor_config.json': 'offline'}, protocol_sha256='fixed')

    def text(self, doc):
        return dict(total_input_tokens=5000)

    def image(self, path):
        with Image.open(path) as image:
            visual = (image.width * image.height + 1023) // 1024
        return dict(total_input_tokens=visual + 50, image_patch_tokens=visual,
                    non_image_patch_tokens=50, image_grid_thw=[[1, 10, 20]])


class FakeReader:
    def __init__(self):
        self.results = {}
        self.calls = 0

    def read(self, path):
        self.calls += 1
        return self.results[str(path)]

    def identity(self):
        return dict(model_weights_sha256={'model.safetensors': 'offline'})


def provide_readings(session, cid):
    for row in session.candidates[cid]['report']['records']:
        session.reader.results[row['image']] = json.dumps({k: ' '.join(v) for k, v in row['sections'].items()})


class LayoutOptimizerTests(unittest.TestCase):
    def session(self, directory, **kwargs):
        return LayoutSession(examples(), Constraints(), FakeMeter(), FakeReader(), directory, **kwargs)

    def test_bundled_selected_compact_is_explicit_and_renders_without_agent_imports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = '''import sys
from optical_compression.strategy import RenderStrategy
strategy = RenderStrategy()
assert strategy.kind == 'selected_compact'
assert strategy.metadata['status'] == 'selected'
assert strategy.metadata['source'] == 'compact.py'
assert strategy.metadata['entrypoint'] == 'render_compact'
strategy.render_tasks([dict(sample_id='one', profile='Interests: Jazz',
    item=dict(title='Album', artists=['A'], genres=['Jazz'], era='1990s', moods=['Calm'],
              summary='A description.'), path=sys.argv[1])])
assert 'optical_compression.optimizer.agent' not in sys.modules
assert 'optical_compression.measurement' not in sys.modules
assert 'httpx' not in sys.modules
'''
            with mock.patch.dict('os.environ', {}, clear=True):
                subprocess.run([sys.executable, '-c', script, str(root / 'image.png')],
                               cwd=ROOT, check=True, capture_output=True, text=True)
            with Image.open(root / 'image.png') as image:
                self.assertEqual(image.width, 640)
                self.assertEqual(image.height % 32, 0)
            workflow = mock.Mock()
            strategy = select_layout(dict(layout_strategy=str(SELECTED)), workflow, root, root / 'model')
            self.assertEqual(strategy.kind, 'selected_compact')
            workflow.run.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'not both'):
                select_layout(dict(layout_strategy=str(SELECTED), layout_optimizer_config='config.json'),
                              workflow, root, root / 'model')

    def test_full_pipeline_selects_from_train_once_then_reuses_frozen_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = self.session(root / 'seed', readability_threshold=None)
            cid = session.revise_layout(PACKED, 'offline strategy')['candidate_id']
            session.render_and_measure(cid)
            frozen = session.freeze({})
            workflow = Workflow(root / 'full', dict(mode='agent'))
            datasets, model = root / 'datasets', root / 'model'
            config = dict(layout_optimizer_config='configs/cd_lightgcn/layout_optimizer.json')
            commands = []
            def launch(command, check):
                commands.append(command)
                self.assertTrue(check)
                self.assertEqual(command[1:3], ['-m', 'optical_compression.optimize'])
                self.assertEqual(command[command.index('--samples') + 1],
                                 str(datasets / 'train_candidates.jsonl'))
                self.assertEqual(command[command.index('--model') + 1], str(model))
                target = Path(command[command.index('--output') + 1]) / 'frozen'
                shutil.copytree(frozen, target)
                return subprocess.CompletedProcess(command, 0)
            with mock.patch('workflow.subprocess.run', side_effect=launch):
                first = select_layout(config, workflow, datasets, model)
                second = select_layout(config, workflow, datasets, model)
            self.assertEqual(len(commands), 1)
            self.assertEqual(first.kind, 'agent_top_bottom')
            self.assertEqual(first.signature, second.signature)
            self.assertTrue(first.requires_compression_check)
            self.assertEqual(first.directory, root / 'full/layout_search/frozen')
            # Completed search output cannot be swapped for unverified source.
            (first.directory / 'layout.py').write_text(INITIAL)
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                select_layout(config, workflow, datasets, model)

    def test_compact_is_expressible_with_byte_identical_images(self):
        profiles = [
            'Preference summary: Guitar music.\nPreferred eras: 1990s\nMoods/styles: Calm\n'
            'Diversity: varied\nEvidence strength: strong\nUncertainties: None\nR1: Album.',
            'Preference summary: ' + 'Jazz and acoustic music. ' * 35,
            'Interests: Music\nPreferred eras: 1980s\nUnrelated: Other evidence\nMoods/styles: Gentle',
        ]
        cfg = Constraints()
        with tempfile.TemporaryDirectory() as folder:
            for index, profile in enumerate(profiles):
                task = dict(sample_id=str(index), profile=profile,
                    item=dict(title='Title: an album', artists=['A', 'B'] if index != 2 else [],
                              genres=['Folk', 'Pop'], era='1990s' if index != 2 else '',
                              moods=['Calm', 'Warm'] if index != 2 else [],
                              summary='A detailed summary. ' * (index + 1)))
                doc = document(task['profile'], task['item'])
                plan = execute_layout(COMPACT_EQUIVALENT, [doc], cfg)[0]
                original, expressed = Path(folder) / 'old.png', Path(folder) / 'new.png'
                render_compact(dict(task, path=original))
                render_plan(doc, plan, cfg, expressed)
                with self.subTest(index=index):
                    self.assertEqual(original.read_bytes(), expressed.read_bytes())
                    with Image.open(expressed) as image:
                        self.assertEqual(image.height % 32, 0)

    def test_presentation_is_generic_and_cannot_remove_duplicate_or_replace_values(self):
        doc = document('First line\nSecond line',
                       dict(authors=['Ada', 'Lin'], tags=['Science', 'Travel'], year=2025))
        untouched = json.loads(json.dumps(doc))
        rules = dict(user=[dict(ids=['u0', 'u1'], separator=' • ')], item=[
            dict(groups=[dict(ids=['i0'], label='Written by', list_separator=', '),
                         dict(ids=['i1', 'i2'], label='Topics / year', separator=' / ',
                              list_separator=', ')], separator=' • ')])
        prepared = prepare_document(doc, rules)
        self.assertEqual(prepared['user'][0]['text'], 'First line • Second line')
        self.assertEqual(prepared['item'][0]['text'],
                         'Written by: Ada, Lin • Topics / year: Science, Travel / 2025')
        self.assertEqual(doc, untouched)
        for bad in [dict(item=[dict(ids=['i0'])]),
                    dict(item=[dict(ids=['i0', 'i0', 'i1', 'i2'], label='All')]),
                    dict(item=[dict(ids=['i0', 'i1', 'i2'], label='All', text='Changed')]),
                    dict(item=[dict(ids=['i0', 'i1', 'i2'], label='')]),
                    dict(user=[dict(ids=['u0', 'u1'], separator='\u200b')]),
                    dict(item=[dict(ids=['i0', 'i1', 'i2'], label='All', empty_value='Replacement')])]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare_document(doc, bad)
        source = document('Context', dict(notes=['', 'Must remain'], extra=[]))
        formatted = prepare_document(source, dict(item=[dict(ids=['i0', 'i1'], label='Notes',
                                     skip_empty=True, empty_value='')]))
        self.assertIn('Must remain', formatted['item'][0]['text'])

    def test_independent_margins_line_height_and_rounding_enforce_geometry(self):
        doc = document('Upper: ' + 'gjpqXYZ words ' * 35, dict(notes='Lower text. ' * 12))
        cfg = Constraints()
        plan = execute_layout(INITIAL, [doc], cfg)[0]
        plan.update(margins=dict(left=10, right=20, top=12, bottom=25), line_height=16,
                    line_gap=0, text_anchor='la', text_origin='font', height_multiple=32)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'image.png'
            result = render_plan(doc, plan, cfg, output)
            self.assertEqual(result['height'] % 32, 0)
            for bad in [dict(line_height=1), dict(text_anchor='invalid'), dict(height_multiple=0),
                        dict(margins=dict(left=1, right=20, top=12, bottom=25)),
                        dict(margins=dict(left=10, right=20, top=12))]:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    render_plan(doc, dict(plan, **bad), cfg, output)
            # Validation applies after upward rounding, not just before it.
            unrounded = render_plan(doc, dict(plan, height_multiple=1), cfg, output)
            limited = Constraints(max_height=unrounded['height'])
            with self.assertRaisesRegex(ValueError, 'canvas limits'):
                render_plan(doc, dict(plan, height_multiple=256), limited, output)

    def test_formatted_strategy_freezes_and_replays_with_displayed_readback_truth(self):
        profile = 'Preferred eras: 1980s\nMoods/styles: Calm\nEvidence strength: strong'
        item = dict(title='A title', artists=['One', 'Two'], genres=['Pop'], era='1980s',
                    moods=['Calm'], summary='All values must survive.')
        doc = document(profile, item)
        inputs = [dict(id='example_01', document=doc, document_sha256=fingerprint(doc))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = LayoutSession(inputs, Constraints(), FakeMeter(), FakeReader(), root / 'search')
            self.assertEqual(session.candidates, {})
            self.assertNotEqual(session.initial_source, COMPACT_EQUIVALENT)
            cid = session.revise_layout(COMPACT_EQUIVALENT, 'coverage proof')['candidate_id']
            report = session.render_and_measure(cid)
            self.assertTrue(report['geometry_valid'])
            provide_readings(session, cid)
            self.assertTrue(session.check_readability(cid)['qualified'])
            frozen = FrozenLayout(session.freeze({}))
            replay = root / 'replay.png'
            frozen.render_tasks([dict(sample_id='new', profile=profile, item=item, path=replay)])
            self.assertEqual(digest(replay), report['records'][0]['image_sha256'])
            original = root / 'original.png'
            render_compact(dict(sample_id='new', profile=profile, item=item, path=original))
            self.assertEqual(original.read_bytes(), replay.read_bytes())
            # Even if a candidate computes spans from forged metadata, the trusted
            # renderer reconstructs the original values and rejects their removal.
            plan = execute_layout(COMPACT_EQUIVALENT, [doc], Constraints())[0]
            plan['presentation']['item'][1]['groups'].pop()
            with self.assertRaisesRegex(ValueError, 'every source field'):
                render_plan(doc, plan, Constraints(), replay)

    def test_fixed_title_and_colored_bars_match_original_compact(self):
        cfg = Constraints()
        task = dict(sample_id='style', profile='Music', item=dict(title='Album', artists=['A'],
                    genres=['Pop'], era='1990s', moods=['Calm'], summary='Description'))
        doc = document(task['profile'], task['item'])
        plan = execute_layout(INITIAL, [doc], cfg)[0]
        plan['padding'] = 10
        with tempfile.TemporaryDirectory() as folder:
            baseline = Path(folder) / 'compact.png'
            actual = Path(folder) / 'candidate.png'
            render_compact(dict(task, path=baseline))
            render_plan(doc, plan, cfg, actual)
            with Image.open(baseline) as old, Image.open(actual) as new:
                # Same title, typeface, placement, blue bar and bold heading.
                self.assertEqual(old.crop((4, 4, 637, 56)).tobytes(),
                                 new.crop((4, 4, 637, 56)).tobytes())
                colors = {color for count, color in new.getcolors(new.width * new.height)}
                self.assertIn((220, 252, 231), colors)
                self.assertIn((22, 101, 52), colors)
            for override in [dict(page_title=''), dict(user_bar=None), dict(title_size=1)]:
                with self.subTest(override=override), self.assertRaisesRegex(ValueError, 'plan keys'):
                    render_plan(doc, dict(plan, **override), cfg, actual)
            plan['width'] = 256
            with self.assertRaisesRegex(ValueError, 'fixed title or heading exceeds'):
                render_plan(doc, plan, cfg, actual)

    def test_general_code_preserves_long_words_and_top_bottom_sections(self):
        doc = document('Interests: ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 15,
                       dict(title='Title', notes=['Short note', 'Longer note']))
        cfg = Constraints()
        plan = execute_layout(PACKED, [doc], cfg)[0]
        with tempfile.TemporaryDirectory() as folder:
            result = render_plan(doc, plan, cfg, Path(folder) / 'image.png')
        self.assertTrue(result['geometry_valid'])
        for section in ['user', 'item']:
            actual = ''.join(''.join(result['sections'][section]).split())
            expected = ''.join(''.join(f['text'] for f in doc[section]).split())
            self.assertEqual(actual, expected)
        self.assertLess(plan['width'], 640)

    def test_code_can_choose_width_by_input_length(self):
        source = PACKED.replace('width = min(', "size = sum(len(f['text']) for f in document['user'])\n    width = min(")
        source = source.replace('max(480,', 'max(480 if size < 200 else 800,')
        docs = [document('Short', {'title': 'T'}), document('Word ' * 100, {'title': 'T'})]
        plans = execute_layout(source, docs, Constraints())
        self.assertEqual([p['width'] for p in plans], [480, 800])

    def test_removed_duplicated_reordered_text_and_wrong_section_are_rejected(self):
        doc = examples()[0]['document']
        plan = execute_layout(INITIAL, [doc], Constraints())[0]
        with tempfile.TemporaryDirectory() as folder:
            for change in ['remove', 'duplicate', 'swap', 'truncate']:
                modified = json.loads(json.dumps(plan))
                if change == 'remove':
                    modified['user_rows'] = modified['user_rows'][1:]
                elif change == 'duplicate':
                    modified['user_rows'] += modified['user_rows'][:1]
                elif change == 'swap':
                    modified['user_rows'], modified['item_rows'] = modified['item_rows'], modified['user_rows']
                else:
                    modified['user_rows'][0][0][2] -= 1
                with self.subTest(change=change), self.assertRaises(ValueError):
                    render_plan(doc, modified, Constraints(), Path(folder) / 'bad.png')

    def test_font_geometry_and_restricted_source_rules(self):
        for source in ['import os\n' + INITIAL,
                       INITIAL.replace('metrics.flow', 'metrics.__class__'),
                       INITIAL.replace('padding = 12', "padding = 12\n    open('/tmp/oops', 'w')"),
                       INITIAL.replace('constraints):', 'constraints=1):')]:
            with self.subTest(source=source[:60]):
                with self.assertRaises((ValueError, NameError)):
                    validate_source(source)
                    execute_layout(source, [examples()[0]['document']], Constraints())
        doc = examples()[0]['document']
        plan = execute_layout(INITIAL, [doc], Constraints())[0]
        plan['line_gap'] = -1
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            render_plan(doc, plan, Constraints(), Path(folder) / 'bad.png')
        doc = document('\U0010ffff', {'title': 'T'})
        plan = execute_layout(INITIAL, [doc], Constraints())[0]
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'no glyph'):
            render_plan(doc, plan, Constraints(), Path(folder) / 'bad.png')

    def test_worker_timeout_is_contained(self):
        source = 'def layout(document, metrics, constraints):\n    while True:\n        pass\n'
        with self.assertRaises(subprocess.TimeoutExpired):
            execute_layout(source, [examples()[0]['document']], Constraints(), timeout=.3)

    def test_training_selection_strips_labels_and_selects_length_extremes(self):
        rows = [dict(split='train', sample_id=f'{i}-A-target-secret', label='A', target_item_id='secret',
                     user_context=f'Context {i}', candidate_payload={'summary': 'word ' * (i + 1)}) for i in range(30)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            selected = select_examples(path)
            self.assertEqual(len(selected), 10)
            self.assertNotIn('secret', json.dumps(selected))
            self.assertIn('Context 0', json.dumps(selected))
            self.assertIn('Context 29', json.dumps(selected))
            rows[0]['split'] = 'test'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'split=train'):
                select_examples(path)

    def test_token_equality_and_one_failing_sample_block_export(self):
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.baselines['example_01'] = 1
            report = session.render_and_measure(cid)
            provide_readings(session, cid)
            self.assertFalse(report['token_qualified'])
            self.assertFalse(session.check_readability(cid)['qualified'])
            self.assertIsNone(session.freeze({}))
            self.assertFalse((session.output / 'frozen').exists())
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'equality')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            original = session.meter.image
            session.meter.image = lambda path: dict(original(path), total_input_tokens=5000)
            self.assertFalse(session.render_and_measure(cid)['token_qualified'])

    def test_readback_is_diagnostic_and_unverified_token_baselines_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'search', readability_threshold=None)
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            provide_readings(session, cid)
            session.reader.results[next(iter(session.reader.results))] = '{"user":"missing","item":"wrong"}'
            feedback = session.check_readability(cid)
            self.assertTrue(feedback['qualified'])
            self.assertLess(feedback['readability'][0]['mean_section_similarity'], .98)
            self.assertIsNotNone(session.freeze({}))
            with self.assertRaisesRegex(ValueError, 'pure-text counts'):
                self.session(Path(folder) / 'wrong_baseline', text_token_limits={'example_01': 1_000_000})
        self.assertFalse(reading_result('not json', {'user': ['x'], 'item': ['y']})['parsed'])

    def test_export_without_readback_does_not_require_model_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            session = self.session(Path(folder) / 'search', readability_threshold=None)
            session.reader = None
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            self.assertFalse(session.check_readability(cid)['enabled'])
            frozen = session.freeze({'model': 'offline'})
            metadata = json.loads((frozen / 'layout.json').read_text())
            self.assertIsNone(metadata['reader'])
            self.assertIsNone(metadata['evaluation_limits']['readability_threshold'])

    def test_default_90_percent_threshold_and_stricter_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.session(Path(folder) / 'default').threshold, .90)
            for threshold, expected in [(.90, True), (.95, True), (.98, False)]:
                session = self.session(Path(folder) / str(threshold), readability_threshold=threshold)
                cid = session.revise_layout(PACKED, 'compact')['candidate_id']
                self.assertFalse(session.render_and_measure(cid)['qualified'])
                provide_readings(session, cid)
                path = next(iter(session.reader.results))
                raw = json.loads(session.reader.results[path])
                raw['user'] = raw['user'][:int(len(raw['user']) * .94)]
                session.reader.results[path] = json.dumps(raw)
                self.assertEqual(session.check_readability(cid)['qualified'], expected)

    def test_frozen_bundle_is_portable_and_checks_source_and_font_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = self.session(root / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            provide_readings(session, cid)
            self.assertTrue(session.check_readability(cid)['qualified'])
            frozen = session.freeze({'model': 'offline'})
            portable = root / 'portable'
            shutil.copytree(frozen, portable)
            strategy = FrozenLayout(portable)
            self.assertEqual(strategy.metadata['fixed_decorations'], FIXED_DECORATIONS)
            self.assertEqual(Path(strategy.constraints.bold_font_path).parent, portable / 'fonts')
            task = dict(sample_id='new', profile='New user context', item={'title': 'A new album'}, path=root / 'new.png')
            self.assertEqual(len(strategy.render_tasks([task])), 1)
            self.assertTrue((root / 'new.png').is_file())
            bold = portable / 'fonts/bold.ttf'
            saved_bold = bold.read_bytes()
            bold.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                FrozenLayout(portable)
            bold.write_bytes(saved_bold)
            (portable / 'layout.py').write_text(INITIAL)
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                FrozenLayout(portable)
            shutil.copyfile(frozen / 'layout.py', portable / 'layout.py')
            (portable / 'fonts/body.ttf').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                FrozenLayout(portable)

    def test_visual_tool_loop_exports_actual_program_without_comparison_or_inspection_tool(self):
        with tempfile.TemporaryDirectory() as folder:
            supplied_initial = INITIAL.replace("'section_gap': 16", "'section_gap': 12")
            session = self.session(Path(folder) / 'search', initial_source=supplied_initial)
            self.assertEqual(session.inspect_inputs()['initial_source'], supplied_initial)
            cid = fingerprint(PACKED)[:16]
            class Client:
                turn = 0
                saw_initial_images = False
                saw_candidate_images = False
                def complete(client, messages):
                    image_messages = [m for m in messages if isinstance(m.get('content'), list)
                                      and any(p.get('type') == 'image_url' for p in m['content'])]
                    if client.turn == 0:
                        client.saw_initial_images = bool(image_messages)
                        self.assertIn(supplied_initial, messages[1]['content'])
                    if client.turn == 3:
                        client.saw_candidate_images = bool(image_messages)
                        provide_readings(session, cid)
                    actions = [('inspect_inputs', {}), ('revise_layout', dict(source=PACKED, reason='pack fields')),
                               ('render_and_measure', dict(candidate_id=cid)), ('check_readability', dict(candidate_id=cid))]
                    if client.turn == len(actions):
                        return dict(role='assistant', content='Done'), {}
                    name, args = actions[client.turn]
                    client.turn += 1
                    return dict(role='assistant', content=None, tool_calls=[dict(id=str(client.turn), type='function',
                        function=dict(name=name, arguments=json.dumps(args)))]), dict(model='offline', usage={})
            client = Client()
            path = run_agent(session, client, model='offline', max_turns=8)
            self.assertTrue(client.saw_initial_images and client.saw_candidate_images)
            self.assertEqual((path / 'layout.py').read_text(), PACKED)
            self.assertEqual({t['function']['name'] for t in TOOLS},
                {'inspect_inputs', 'revise_layout', 'render_and_measure', 'check_readability'})
            self.assertNotIn('data:image/png;base64,', (session.output / 'agent_trace.json').read_text())

    def test_chat_transport_uses_real_image_and_tool_message_protocol(self):
        import httpx
        real_client = httpx.Client
        def handler(request):
            self.assertEqual(str(request.url), 'https://example.test/v1/chat/completions')
            self.assertEqual(request.headers['Authorization'], 'Bearer offline-test')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'configured-vision-model')
            self.assertEqual(body['messages'][0]['content'][0]['image_url']['detail'], 'high')
            self.assertFalse(body['parallel_tool_calls'])
            self.assertEqual(len(body['tools']), 4)
            return httpx.Response(200, json=dict(model='resolved-model', usage={'total_tokens': 12},
                choices=[dict(finish_reason='tool_calls', message=dict(role='assistant', content=None,
                    tool_calls=[dict(id='call1', type='function', function=dict(name='inspect_inputs', arguments='{}'))]))]))
        def factory(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)
        with mock.patch.dict('os.environ', {'OPENAI_API_KEY': 'offline-test',
                                           'OPENAI_BASE_URL': 'https://example.test/v1'}, clear=True), \
             mock.patch('httpx.Client', side_effect=factory):
            client = ChatClient(dict(model='configured-vision-model'))
            try:
                message, usage = client.complete([dict(role='user', content=[dict(type='image_url',
                    image_url=dict(url='data:image/png;base64,AAAA', detail='high'))])])
            finally:
                client.close()
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'inspect_inputs')
        self.assertEqual(usage['model'], 'resolved-model')

    def test_module_two_renders_all_splits_with_frozen_strategy_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = self.session(root / 'search')
            cid = session.revise_layout(PACKED, 'compact')['candidate_id']
            session.render_and_measure(cid)
            provide_readings(session, cid)
            session.check_readability(cid)
            frozen = session.freeze({})
            data = root / 'data'
            data.mkdir()
            manifest = dict(output_sha256={}, splits={})
            for split in ['train', 'valid', 'test']:
                rows = [dict(sample_id=f'{split}-{i}', group_id=split, candidate_item_id=str(i),
                             user_context='Interest: music', candidate_payload={'title': f'New item {i}'},
                             label='A' if i == 0 else 'B', _replay=i < 2) for i in range(10)]
                path = data / f'{split}_candidates.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                manifest['output_sha256'][path.name] = digest(path)
                manifest['splits'][split] = dict(natural_hit_groups=1, all_groups=1)
            (data / 'manifest.json').write_text(json.dumps(manifest))
            args = [sys.executable, '-m', 'optical_compression.pipeline', '--datasets', str(data),
                    '--output', str(root / 'dense'), '--layout', str(frozen)]
            for _ in range(2):
                subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
            complete = json.loads((root / 'dense/complete.json').read_text())
            self.assertEqual(complete['layout_signature'], digest(frozen / 'layout.json'))
            self.assertFalse(complete['processor_validated'])
            for split in ['train', 'valid', 'test']:
                report = json.loads((root / f'dense/{split}/manifest.json').read_text())
                self.assertEqual(report['layout']['kind'], 'agent_top_bottom')
                self.assertEqual(len(report['pages']), 10)


if __name__ == '__main__':
    unittest.main()
