"""Four agent tools with independent checks and deterministic best-strategy selection."""
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import platform

from workflow import digest
from ..document import fingerprint
from ..layout_engine import (execute_layout, render_plan, validate_source, engine_hashes,
                            supported_glyph, FIXED_DECORATIONS, LAYOUT_API_VERSION)
from ..measurement import reading_result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def runtime_versions():
    result = dict(python=platform.python_version())
    for package in ['Pillow', 'torch', 'transformers', 'httpx']:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


class LayoutSession:
    def __init__(self, examples, constraints, meter, reader, output, *, max_length=2048,
                 max_candidates=20, text_token_limits=None, initial_source=None, readability_threshold=.90):
        if not (type(max_candidates) is int and max_candidates >= 1
                and type(max_length) is int and max_length >= 1):
            raise ValueError('invalid evaluation limits')
        if readability_threshold is not None:
            if (type(readability_threshold) not in (int, float) or not 0 < readability_threshold <= 1):
                raise ValueError('readability_threshold must be in (0,1] or null')
            if reader is None:
                raise ValueError('enable the local reader or set readability_threshold=null')
        self.examples, self.constraints = examples, constraints
        self.initial_source = initial_source if initial_source is not None else (
            Path(__file__).parents[1] / 'layouts/initial.py').read_text()
        validate_source(self.initial_source)
        for example in examples:
            chars = set(''.join(f['text'] for section in example['document'].values() for f in section))
            if any(not supported_glyph(constraints.font_path, constraints.font_size, char) for char in chars):
                raise ValueError('initialization font lacks dataset glyphs; choose a font covering the input language')
        self.meter, self.reader = meter, reader
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.max_length = max_length
        self.threshold = readability_threshold
        self.max_candidates, self.candidates, self.best = max_candidates, {}, None
        self.baselines = {e['id']: meter.text(e['document'])['total_input_tokens'] for e in examples}
        if text_token_limits is not None and text_token_limits != self.baselines:
            raise ValueError('provided pure-text counts differ from the actual same-protocol processor counts')
        self.initialization = dict(examples=examples, constraints=asdict(constraints),
            layout_api_version=LAYOUT_API_VERSION,
            initial_source_sha256=fingerprint(self.initial_source),
            font_sha256=digest(constraints.font_path), bold_font_sha256=digest(constraints.bold_font_path),
            fixed_decorations=FIXED_DECORATIONS.copy(), processor=meter.identity,
            text_token_limits=self.baselines, max_length=max_length,
            readability_mode='threshold' if self.threshold is not None else (
                'diagnostic_only' if reader is not None else 'disabled'),
            readability_threshold=self.threshold, max_candidates=max_candidates,
            input_scope='training only; labels, targets and source IDs removed')
        write_json(self.output / 'initialization.json', self.initialization)

    def inspect_inputs(self):
        return dict(**self.initialization, candidate_ids=list(self.candidates), best=self.best,
            initial_source=self.initial_source)

    def revise_layout(self, source, reason):
        validate_source(source)
        cid = fingerprint(source)[:16]
        if cid in self.candidates:
            return dict(candidate_id=cid, reused=True)
        if len(self.candidates) >= self.max_candidates:
            raise ValueError('candidate budget exhausted')
        directory = self.output / 'candidates' / cid
        directory.mkdir(parents=True)
        (directory / 'layout.py').write_text(source)
        self.candidates[cid] = dict(source=source, reason=reason, directory=directory, report=None)
        return dict(candidate_id=cid, source_sha256=digest(directory / 'layout.py'))

    def candidate(self, candidate_id):
        if candidate_id not in self.candidates:
            raise ValueError('unknown candidate ID')
        return self.candidates[candidate_id]

    def render_and_measure(self, candidate_id):
        candidate = self.candidate(candidate_id)
        if candidate['report'] is not None:
            return candidate['report']
        docs = [e['document'] for e in self.examples]
        report = dict(candidate_id=candidate_id, qualified=False, records=[])
        try:
            plans = execute_layout(candidate['source'], docs, self.constraints)
            # A second evaluation rejects stateful rules that change the layout for identical inputs.
            if plans != execute_layout(candidate['source'], docs, self.constraints):
                raise ValueError('layout must be deterministic')
            for example, plan in zip(self.examples, plans):
                path = candidate['directory'] / (example['id'] + '.png')
                page = render_plan(example['document'], plan, self.constraints, path)
                counts = self.meter.image(path)
                if counts['image_patch_tokens'] <= 0 or counts['total_input_tokens'] <= 0:
                    raise ValueError('processor did not produce valid visual tokens')
                total, baseline = counts['total_input_tokens'], self.baselines[example['id']]
                report['records'].append(dict(id=example['id'], image=str(path), **page, **counts,
                    pure_text_tokens=baseline, strictly_below_text=total < baseline,
                    within_length_guard=total <= self.max_length))
            report['geometry_valid'] = True
            report['token_qualified'] = all(r['strictly_below_text'] and r['within_length_guard']
                                            for r in report['records'])
            report['qualified'] = report['token_qualified'] and self.threshold is None
            report['mean_image_tokens'] = statistics.fmean(r['image_patch_tokens'] for r in report['records'])
            report['max_image_tokens'] = max(r['image_patch_tokens'] for r in report['records'])
            report['mean_total_tokens'] = statistics.fmean(r['total_input_tokens'] for r in report['records'])
        except (ValueError, KeyError, TypeError, TimeoutError, RuntimeError, subprocess.SubprocessError) as exc:
            report.update(qualified=False, geometry_valid=False, token_qualified=False, error=str(exc))
        candidate['report'] = report
        write_json(candidate['directory'] / 'evaluation.json', report)
        self.update_best(candidate_id)
        return report

    def update_best(self, candidate_id):
        report = self.candidate(candidate_id)['report']
        if report['qualified']:
            score = (report['mean_image_tokens'], report['max_image_tokens'], candidate_id)
            if self.best is None or score < tuple(self.best['score']):
                self.best = dict(candidate_id=candidate_id, score=list(score))
                write_json(self.output / 'best.json', self.best)

    def check_readability(self, candidate_id):
        candidate = self.candidate(candidate_id)
        report = candidate['report']
        if report is None or not report.get('geometry_valid'):
            raise ValueError('render a geometrically valid candidate first')
        if self.reader is None:
            return dict(candidate_id=candidate_id, enabled=False, qualified=report['qualified'],
                        note='Readback is disabled; agent visual review and token/content checks remain active.')
        if 'readability' not in report:
            readings = []
            for row in report['records']:
                raw = self.reader.read(Path(row['image']))
                readings.append(dict(id=row['id'], **reading_result(raw, row['sections'])))
            report['readability'] = readings
            report['readability_summary'] = dict(parsed_samples=sum(r['parsed'] for r in readings),
                samples=len(readings), section_mean_similarity={
                    section: statistics.fmean(r['section_similarity'][section] for r in readings if r['parsed'])
                    if any(r['parsed'] for r in readings) else None for section in ['user', 'item']})
            if self.threshold is not None:
                for reading in readings:
                    reading['passed'] = reading['parsed'] and all(
                        score >= self.threshold for score in reading['section_similarity'].values())
                report['readability_threshold'] = self.threshold
                report['readability_passed'] = all(r['passed'] for r in readings)
                report['qualified'] = report['token_qualified'] and report['readability_passed']
            write_json(candidate['directory'] / 'evaluation.json', report)
        self.update_best(candidate_id)
        return dict(candidate_id=candidate_id, qualified=report['qualified'],
                    readability=report['readability'], summary=report['readability_summary'], best=self.best)

    def dispatch(self, name, arguments):
        if name not in {'inspect_inputs', 'revise_layout', 'render_and_measure', 'check_readability'}:
            raise ValueError('unknown tool')
        return getattr(self, name)(**arguments)

    def freeze(self, agent_identity):
        write_json(self.output / 'best.json', self.best)
        summary = dict(status='qualified' if self.best else 'no_qualified_strategy', best=self.best,
            candidates=[dict(candidate_id=key, reason=value['reason'], report=value['report'])
                        for key, value in self.candidates.items()])
        write_json(self.output / 'search_summary.json', summary)
        if self.best is None:
            return None
        candidate = self.candidate(self.best['candidate_id'])
        target = self.output / 'frozen.pending'
        target.mkdir()
        (target / 'layout.py').write_text(candidate['source'])
        (target / 'fonts').mkdir()
        shutil.copyfile(self.constraints.font_path, target / 'fonts/body.ttf')
        shutil.copyfile(self.constraints.bold_font_path, target / 'fonts/bold.ttf')
        for font_path, name in [(self.constraints.font_path, 'LICENSE.txt'),
                                (self.constraints.bold_font_path, 'bold-LICENSE.txt')]:
            license_path = Path(font_path).parent / 'LICENSE.txt'
            if license_path.is_file():
                shutil.copyfile(license_path, target / 'fonts' / name)
        cfg = asdict(self.constraints)
        cfg['font_path'] = 'fonts/body.ttf'
        cfg['bold_font_path'] = 'fonts/bold.ttf'
        files = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob('*')) if p.is_file()}
        write_json(target / 'layout.json', dict(format='zip4rec-top-bottom-v1', qualified=True,
            layout_api_version=LAYOUT_API_VERSION,
            constraints=cfg, fixed_decorations=FIXED_DECORATIONS.copy(),
            files_sha256=files, engine_sha256=engine_hashes(),
            evaluation_limits=dict(max_length=self.max_length,
                                   readability_mode='threshold' if self.threshold is not None else 'diagnostic_only',
                                   readability_threshold=self.threshold,
                                   compression_rule='total_input_tokens < same-protocol pure_text_tokens, per example'),
            processor=self.meter.identity,
            reader=self.reader.identity() if candidate['report'].get('readability') else None, agent=agent_identity,
            runtime=runtime_versions(), evaluation_code_sha256={
                'measurement.py': digest(Path(__file__).parents[1] / 'measurement.py'),
                'optimizer/tools.py': digest(Path(__file__))},
            initialization_sha256=digest(self.output / 'initialization.json'),
            evaluation_sha256=digest(candidate['directory'] / 'evaluation.json'),
            example_document_sha256=[e['document_sha256'] for e in self.examples],
            metrics={k: v for k, v in candidate['report'].items() if k != 'records' and k != 'readability'}))
        frozen = self.output / 'frozen'
        target.rename(frozen)
        return frozen
