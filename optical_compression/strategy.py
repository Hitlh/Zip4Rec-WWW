"""Load already-selected layout code for rendering, without importing the agent."""
import json
from pathlib import Path

from workflow import digest
from .document import fingerprint

SELECTED = Path(__file__).parent / 'selected'


class RenderStrategy:
    def __init__(self, directory=None):
        self.directory = Path(directory or SELECTED).resolve()
        self.frozen = None
        if self.directory != SELECTED.resolve():
            from .layout_engine import FrozenLayout
            self.frozen = FrozenLayout(self.directory)
            self.metadata = self.frozen.metadata
            self.signature = self.frozen.signature
            self.kind = 'agent_top_bottom'
            return
        self.metadata = json.loads((self.directory / 'selection.json').read_text())
        if (self.metadata['strategy_id'] != 'cd_compact' or self.metadata['status'] != 'selected'
                or self.metadata['source'] != 'compact.py' or self.metadata['entrypoint'] != 'render_compact'):
            raise ValueError('unsupported project-selected layout')
        assets = [self.directory / 'compact.py', self.directory / 'selection.json',
                  SELECTED.parent / 'fonts/DejaVuSans.ttf', SELECTED.parent / 'fonts/DejaVuSans-Bold.ttf']
        self.signature = fingerprint({str(p.relative_to(SELECTED.parent)): digest(p) for p in assets})
        self.kind = 'selected_compact'

    @property
    def requires_compression_check(self):
        return self.frozen is not None

    def render_tasks(self, tasks):
        if self.frozen is not None:
            return self.frozen.render_tasks(tasks)
        from .selected.compact import render_compact
        return [render_compact(task) for task in tasks]
