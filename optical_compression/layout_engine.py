"""Trusted single-column renderer. Generated code supplies rows of original-text spans."""
import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont
from workflow import digest
from .presentation import prepare_document

ROOT = Path(__file__).resolve().parents[1]
FONT = Path(__file__).parent / 'fonts/DejaVuSans.ttf'
BOLD_FONT = Path(__file__).parent / 'fonts/DejaVuSans-Bold.ttf'
PAGE_TITLE = 'PROFILE-CONDITIONED RELEVANCE'
FIXED_DECORATIONS = dict(page_title=PAGE_TITLE, title_color='#172033',
    user_bar='#dbeafe', user_heading='#153e75',
    item_bar='#dcfce7', item_heading='#166534', body_color='#202938',
    background='#f7f8fa', border='#c8d0dc', section_bar_height=20)
ENGINE_FILES = ['document.py', 'presentation.py', 'layout_engine.py', 'layout_worker.py']
LAYOUT_API_VERSION = 2


@lru_cache(maxsize=32)
def load_font(path, size):
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=8192)
def supported_glyph(path, size, char):
    fnt = load_font(path, size)
    return char.isspace() or bytes(fnt.getmask(char)) != bytes(fnt.getmask('\U0010ffff'))


@dataclass(frozen=True)
class Constraints:
    font_path: str = str(FONT)
    bold_font_path: str = str(BOLD_FONT)
    font_size: int = 14
    heading_size: int = 15
    title_size: int = 18
    min_width: int = 256
    max_width: int = 1536
    max_height: int = 16384
    max_pixels: int = 8_000_000

    def __post_init__(self):
        for path in [self.font_path, self.bold_font_path]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if any(type(getattr(self, name)) is not int for name in
               ['font_size', 'heading_size', 'title_size', 'min_width', 'max_width', 'max_height', 'max_pixels']):
            raise ValueError('fixed typography and canvas limits must be integers')
        if not (6 <= self.font_size <= 96 and 6 <= self.heading_size <= 96
                and 6 <= self.title_size <= 96
                and 32 <= self.min_width <= self.max_width <= 4096
                and 32 <= self.max_height <= 32768 and 1 <= self.max_pixels <= 32_000_000):
            raise ValueError('invalid fixed typography or canvas limits')


def engine_hashes():
    return {name: digest(Path(__file__).parent / name) for name in ENGINE_FILES}


def validate_source(source):
    """A restricted Python subset: layout arithmetic/control flow and font metrics only.

    This restriction plus process limits contains accidental generated-code failures;
    it is not an OS security sandbox for adversarial Python.
    """
    if not isinstance(source, str) or len(source) > 30_000:
        raise ValueError('layout source must be at most 30000 characters')
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != 'layout' or tree.body[0].decorator_list):
        raise ValueError('source must contain exactly def layout(document, metrics, constraints)')
    fn = tree.body[0]
    if ([a.arg for a in fn.args.args] != ['document', 'metrics', 'constraints']
            or fn.args.defaults or fn.args.kw_defaults or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs or fn.returns
            or any(a.annotation for a in fn.args.args)):
        raise ValueError('layout must have three plain arguments without defaults or annotations')
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda,
                 ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
                 ast.Delete, ast.Yield, ast.YieldFrom, ast.Await)
    for node in ast.walk(tree):
        if isinstance(node, forbidden) or (isinstance(node, ast.FunctionDef) and node is not fn):
            raise ValueError('unsupported layout Python statement: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('private names are not allowed')
        if isinstance(node, ast.Attribute):
            if (not isinstance(node.value, ast.Name) or node.value.id != 'metrics'
                    or node.attr not in {'flow', 'width', 'prepare'} or not isinstance(node.ctx, ast.Load)):
                raise ValueError('only metrics.flow, metrics.width and metrics.prepare attributes are available')
    return tree


def row_text(row, fields):
    parts, previous = [], None
    for field_id, start, end in row:
        if previous is not None and previous != field_id:
            parts.append(' ')
        parts.append(fields[field_id][start:end])
        previous = field_id
    return ''.join(parts)


class FontMetrics:
    def __init__(self, constraints):
        self.font = load_font(constraints.font_path, constraints.font_size)

    def width(self, text):
        box = self.font.getbbox(text, anchor='lt')
        return max(self.font.getlength(text), box[2]) - min(0, box[0])

    def prepare(self, document, presentation):
        return prepare_document(document, presentation)

    def flow(self, fields, width, pack=False, wrap='split'):
        """Return exact [field_id,start,end] spans; long words also wrap without loss."""
        if not isinstance(width, (int, float)) or not math.isfinite(width) or width < 32:
            raise ValueError('flow width must be finite and >=32')
        if not fields:
            return []
        if wrap not in ('split', 'words'):
            raise ValueError('wrap must be split or words')
        if wrap == 'words':
            return self.word_flow(fields, width, pack)
        texts = {f['id']: f['text'] for f in fields}
        rows, current = [], []
        for field in fields:
            if not pack and current:
                rows.append(current)
                current = []
            text, start = field['text'], 0
            while start < len(text):
                low, high = start, len(text)
                while low < high:
                    mid = (low + high + 1) // 2
                    proposal = current + [[field['id'], start, mid]]
                    if self.width(row_text(proposal, texts)) <= width:
                        low = mid
                    else:
                        high = mid - 1
                end = low
                if end == start:
                    if not current:
                        raise ValueError('a glyph is wider than the available width')
                    rows.append(current)
                    current = []
                    continue
                if end < len(text):
                    boundary = text.rfind(' ', start, end + 1)
                    if boundary > start:
                        end = boundary + 1
                        # A trailing space can itself exceed the line width.
                        if self.width(row_text(current + [[field['id'], start, end]], texts)) > width:
                            end = boundary
                current.append([field['id'], start, end])
                start = end
                if start < len(text):
                    rows.append(current)
                    current = []
        if current:
            rows.append(current)
        return rows

    def word_flow(self, fields, width, pack):
        """Word wrapping retains whitespace in spans; draw with trim_rows=True.

        No partial word is squeezed into a preceding line. Oversized words fail
        geometry validation rather than overflowing the page as the legacy renderer can.
        """
        texts = {f['id']: f['text'] for f in fields}
        rows, current = [], []
        for field in fields:
            if not pack and current:
                rows.append(current)
                current = []
            text = field['text']
            for match in re.finditer(r'\s*\S+\s*|\s+', text):
                span = [field['id'], match.start(), match.end()]
                if current and self.width(row_text(current + [span], texts).strip()) > width:
                    rows.append(current)
                    current = []
                if self.width(text[match.start():match.end()].strip()) > width:
                    raise ValueError('a word is wider than the available width; use wrap=split')
                current.append(span)
                if len(current) > 1 and current[-2][0] == span[0]:
                    current[-2][2] = span[2]
                    current.pop()
        if current:
            rows.append(current)
        return rows


def execute_layout(source, documents, constraints, timeout=20):
    """Execute candidates in a separate CPU-limited process without inherited API keys."""
    validate_source(source)
    payload = dict(source=source, documents=documents, constraints=asdict(constraints))
    env = {'PYTHONPATH': str(ROOT), 'PYTHONIOENCODING': 'utf-8'}
    with tempfile.TemporaryFile(mode='w+') as output, tempfile.TemporaryFile(mode='w+') as error:
        result = subprocess.run([sys.executable, '-m', 'optical_compression.layout_worker'],
            input=json.dumps(payload), text=True, stdout=output, stderr=error,
            cwd=ROOT, env=env, timeout=timeout)
        output.seek(0)
        error.seek(0)
        content = output.read(8_000_001)
        if result.returncode or len(content) > 8_000_000:
            raise ValueError('layout worker failed: ' + error.read(2000))
    plans = json.loads(content)
    if not isinstance(plans, list) or len(plans) != len(documents):
        raise ValueError('layout worker returned the wrong number of plans')
    return plans


def validate_rows(rows, fields):
    texts = {f['id']: f['text'] for f in fields}
    coverage = {key: 0 for key in texts}
    completed, active = set(), None
    if not isinstance(rows, list) or not rows or len(rows) > 20_000:
        raise ValueError('expect a nonempty bounded row list')
    for row in rows:
        if not isinstance(row, list) or not row or len(row) > len(fields) + 1:
            raise ValueError('invalid row')
        for span in row:
            if not isinstance(span, list) or len(span) != 3:
                raise ValueError('each text span is [field_id,start,end]')
            key, start, end = span
            if (key not in texts or type(start) is not int or type(end) is not int
                    or start != coverage[key] or not start < end <= len(texts[key])):
                raise ValueError('text lost, changed, duplicated or reordered within a field')
            if key != active:
                if active is not None:
                    completed.add(active)
                if key in completed:
                    raise ValueError('fields may be reordered but not interleaved')
                active = key
            coverage[key] = end
    if coverage != {key: len(value) for key, value in texts.items()}:
        raise ValueError('all original text characters must appear exactly once')
    return [row_text(row, texts) for row in rows]


def render_plan(doc, plan, constraints, path):
    required = {'width', 'padding', 'line_gap', 'section_gap', 'user_rows', 'item_rows'}
    optional = {'margins', 'line_height', 'text_anchor', 'text_origin', 'height_multiple', 'min_height',
                'trim_rows', 'presentation'}
    if not isinstance(plan, dict) or not required <= set(plan) or set(plan) - required - optional:
        raise ValueError('unexpected layout plan keys')
    for key in ['width', 'padding', 'line_gap', 'section_gap']:
        if type(plan[key]) is not int:
            raise ValueError('layout dimensions must be integers')
    width, pad = plan['width'], plan['padding']
    if not (constraints.min_width <= width <= constraints.max_width and 8 <= pad <= width // 4
            and 0 <= plan['line_gap'] <= 64 and 0 <= plan['section_gap'] <= 256):
        raise ValueError('layout dimensions violate constraints')
    margins = plan.get('margins', dict(left=pad, right=pad, top=pad + 2, bottom=pad))
    if (not isinstance(margins, dict) or set(margins) != {'left', 'right', 'top', 'bottom'}
            or any(type(value) is not int for value in margins.values())
            or not 8 <= margins['left'] <= width // 4
            or not 8 <= margins['right'] <= width // 4
            or not 4 <= margins['top'] <= 512 or not 4 <= margins['bottom'] <= 512):
        raise ValueError('invalid independent margins')
    left, right = margins['left'], margins['right']
    line_height = plan.get('line_height')
    if line_height is not None and (type(line_height) is not int or not 1 <= line_height <= 256):
        raise ValueError('line_height must be null or an integer in 1..256')
    anchor = plan.get('text_anchor', 'lt')
    origin = plan.get('text_origin', 'ink')
    multiple, minimum = plan.get('height_multiple', 1), plan.get('min_height', 32)
    trim = plan.get('trim_rows', False)
    if (anchor not in ('lt', 'la') or origin not in ('ink', 'font') or type(trim) is not bool
            or type(multiple) is not int or not 1 <= multiple <= 256
            or type(minimum) is not int or not 32 <= minimum <= constraints.max_height):
        raise ValueError('invalid text anchor or height alignment')
    presented = prepare_document(doc, plan.get('presentation'))
    texts = {section: validate_rows(plan[section + '_rows'], presented[section]) for section in ['user', 'item']}
    if trim:
        texts = {section: [text.strip() for text in rows] for section, rows in texts.items()}
    body = load_font(constraints.font_path, constraints.font_size)
    heading = load_font(constraints.bold_font_path, constraints.heading_size)
    title_font = load_font(constraints.bold_font_path, constraints.title_size)
    for char in set(''.join(text for rows in texts.values() for text in rows)):
        if not supported_glyph(constraints.font_path, constraints.font_size, char):
            raise ValueError('fixed font has no glyph for ' + repr(char))
    if not all(supported_glyph(constraints.bold_font_path, constraints.heading_size, c)
               for c in 'USER PROFILECANDIDATE ITEM'):
        raise ValueError('fixed font cannot render the section headings')
    if not all(supported_glyph(constraints.bold_font_path, constraints.title_size, c) for c in PAGE_TITLE):
        raise ValueError('fixed font cannot render the page title')
    # Decorations are owned by this engine, never by candidate layout code.
    for text, fnt in [(PAGE_TITLE, title_font), ('USER PROFILE', heading), ('CANDIDATE ITEM', heading)]:
        box = fnt.getbbox(text)
        if max(fnt.getlength(text), box[2]) - min(0, box[0]) > width - left - right:
            raise ValueError('fixed title or heading exceeds page width')
    title_box = title_font.getbbox(PAGE_TITLE)
    commands, bars = [], []
    title_y = margins['top']
    y = title_y + max(24, title_box[3] + 2)
    ink_bottom = y
    for section, title in [('user', 'USER PROFILE'), ('item', 'CANDIDATE ITEM')]:
        if section == 'item':
            y += plan['section_gap']
        if y < ink_bottom:
            raise ValueError('body text overlaps a fixed section bar')
        heading_box = heading.getbbox(title)
        bar_height = max(FIXED_DECORATIONS['section_bar_height'], heading_box[3] + 2)
        bars.append((y, bar_height, section, title))
        y += bar_height + 4
        ink_bottom = y - 4
        for text in texts[section]:
            fnt = body
            box = fnt.getbbox(text, anchor=anchor)
            ink_width = max(fnt.getlength(text), box[2]) - min(0, box[0])
            if ink_width > width - left - right:
                raise ValueError('text exceeds page width')
            if text and y + box[1] < ink_bottom:
                raise ValueError('body lines overlap; increase line_height or line_gap')
            commands.append((left - min(0, box[0]) if origin == 'ink' else left, y, text, fnt))
            if text:
                ink_bottom = y + box[3]
            y += (line_height if line_height is not None else max(1, box[3] - box[1])) + plan['line_gap']
    height = max(minimum, y + margins['bottom'], ink_bottom + 4)
    height = ((height + multiple - 1) // multiple) * multiple
    if height > constraints.max_height or height * width > constraints.max_pixels:
        raise ValueError('page exceeds canvas limits')
    image = Image.new('RGB', (width, height), FIXED_DECORATIONS['background'])
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, width - 4, height - 4), fill='white', outline=FIXED_DECORATIONS['border'], width=1)
    draw.text((left, title_y), PAGE_TITLE, font=title_font, fill=FIXED_DECORATIONS['title_color'])
    for bar_y, bar_height, section, title in bars:
        draw.rectangle((7, bar_y, width - 7, bar_y + bar_height - 1), fill=FIXED_DECORATIONS[section + '_bar'])
        draw.text((left, bar_y + 1), title, font=heading, fill=FIXED_DECORATIONS[section + '_heading'])
    for x, y, text, fnt in commands:
        draw.text((x, y), text, font=fnt, fill=FIXED_DECORATIONS['body_color'], anchor=anchor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    return dict(width=width, height=height, image_sha256=digest(path),
                content_preserved=True, geometry_valid=True, sections=texts)


class FrozenLayout:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        metadata_path = self.directory / 'layout.json'
        self.metadata = json.loads(metadata_path.read_text())
        if self.metadata['format'] != 'zip4rec-top-bottom-v1' or not self.metadata['qualified']:
            raise ValueError('layout is not a qualified frozen strategy')
        if self.metadata['engine_sha256'] != engine_hashes():
            raise ValueError('layout engine changed; revalidate and freeze the strategy again')
        if self.metadata.get('layout_api_version') != LAYOUT_API_VERSION:
            raise ValueError('layout API version changed; revalidate and freeze the strategy again')
        if not {'layout.py', 'fonts/body.ttf', 'fonts/bold.ttf'} <= set(self.metadata['files_sha256']):
            raise ValueError('frozen layout must fingerprint source and both fonts')
        if self.metadata.get('fixed_decorations') != FIXED_DECORATIONS:
            raise ValueError('frozen layout must preserve the fixed title and colored bars')
        for name, expected in self.metadata['files_sha256'].items():
            target = (self.directory / name).resolve()
            if not target.is_relative_to(self.directory) or digest(target) != expected:
                raise ValueError('frozen layout asset changed: ' + name)
        cfg = dict(self.metadata['constraints'])
        if cfg['font_path'] != 'fonts/body.ttf' or cfg.get('bold_font_path') != 'fonts/bold.ttf':
            raise ValueError('frozen typography must reference its bundled fonts')
        cfg['font_path'] = str(self.directory / cfg['font_path'])
        cfg['bold_font_path'] = str(self.directory / cfg['bold_font_path'])
        self.constraints = Constraints(**cfg)
        self.source = (self.directory / 'layout.py').read_text()
        validate_source(self.source)
        self.signature = digest(metadata_path)

    def render_tasks(self, tasks):
        from .document import document
        output = []
        for start in range(0, len(tasks), 32):
            batch = tasks[start:start + 32]
            docs = [document(t['profile'], t['item']) for t in batch]
            plans = execute_layout(self.source, docs, self.constraints)
            for task, doc, plan in zip(batch, docs, plans):
                result = render_plan(doc, plan, self.constraints, task['path'])
                result.pop('sections')
                output.append(dict(sample_id=task['sample_id'], **result))
        return output
