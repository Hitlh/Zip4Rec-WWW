"""Small resumable command runner with frozen configuration and artifact hashes."""
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class Workflow:
    def __init__(self, directory, configuration):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'workflow_state.json'
        signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['configuration_sha256'] != signature:
                raise ValueError('configuration changed; use a new workflow directory')
        else:
            self.state = dict(configuration=configuration, configuration_sha256=signature, stages={})
            self.save()

    def save(self):
        pending = self.path.with_suffix('.tmp')
        pending.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + '\n')
        pending.replace(self.path)

    def run(self, name, command, outputs):
        command = list(map(str, command))
        existing = self.state['stages'].get(name)
        if existing and existing['status'] == 'complete':
            if existing['command'] != command:
                raise ValueError('completed stage command changed: ' + name)
            for path, expected in existing['outputs'].items():
                if not Path(path).is_file() or digest(path) != expected:
                    raise ValueError('completed artifact changed: ' + path)
            return
        self.state['stages'][name] = dict(status='running', command=command)
        self.save()
        print('Running stage: ' + name, flush=True)
        try:
            subprocess.run(command, check=True)
            hashes = {str(Path(path).resolve()): digest(path) for path in outputs}
        except Exception:
            self.state['stages'][name]['status'] = 'failed'
            self.save()
            raise
        self.state['stages'][name].update(status='complete', outputs=hashes)
        self.save()
