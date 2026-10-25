"""Frozen provenance, grouped inputs and deterministic ranking evaluation."""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, records):
    with Path(path).open('w') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def grouped(records):
    output = defaultdict(list)
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('duplicate sample IDs')
    for row in records:
        output[row['group_id']].append(row)
    for name, group in output.items():
        if len({r['user_id'] for r in group}) != 1:
            raise ValueError('mixed users in group: ' + name)
        if len(group) != 10 or len({r['candidate_item_id'] for r in group}) != 10:
            raise ValueError('invalid ten-candidate group: ' + name)
        if len({r['target_item_id'] for r in group}) != 1 or sum(r['label'] == 'A' for r in group) != 1:
            raise ValueError('invalid positive labels: ' + name)
        if any((r['label'] == 'A') != (r['candidate_item_id'] == r['target_item_id']) for r in group):
            raise ValueError('target label mismatch: ' + name)
    return dict(output)


def signature(row):
    messages = [m for m in row['messages'] if m['role'] != 'assistant']
    images = {p['image']: digest(p['image']) for m in messages for p in m['content'] if p.get('type') == 'image'}
    return hashlib.sha256(json.dumps({'messages': messages, 'images': images}, sort_keys=True,
                                    ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def z(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('scores must be finite and nonempty')
    mean, std = statistics.fmean(values), statistics.pstdev(values)
    return [(x - mean) / std for x in values] if std > 1e-12 else [0.] * len(values)


def rank_metrics(scored, alpha=None):
    users = []
    for name, group in sorted(grouped(scored).items()):
        scores = [r['condition_score'] for r in group]
        if not all(math.isfinite(x) for x in scores):
            raise ValueError('nonfinite scores')
        if alpha is not None:
            scores = [a + alpha * b for a, b in zip(z([r['candidate_score'] for r in group]), z(scores))]
        order = sorted(range(10), key=lambda i: (-scores[i], group[i]['candidate_item_id']))
        r = next(k + 1 for k, i in enumerate(order) if group[i]['label'] == 'A')
        users.append({'group_id': name, 'user_id': group[0]['user_id'], 'target_rank': r,
                      'NDCG@10': 1 / math.log2(r + 1), 'MRR@10': 1 / r,
                      'H@1': int(r == 1), 'H@5': int(r <= 5),
                      'ranking': [group[i]['candidate_item_id'] for i in order]})
    if not users:
        raise ValueError('empty ranking evaluation')
    return {'groups': len(users), **{key: statistics.fmean(u[key] for u in users)
                                   for key in ['NDCG@10', 'MRR@10', 'H@1', 'H@5']}}, users


def base_fingerprint(model):
    from .runtime_fingerprint import runtime_fingerprint
    model = Path(model)
    return {'model_path': str(model.resolve()), 'config_sha256': digest(model / 'config.json'),
            'weights': [{'name': f.name, 'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns}
                        for f in sorted(model.glob('*.safetensors'))],
            'processor_files': {f.name: digest(f) for f in model.glob('*.json') if 'safetensors' not in f.name},
            'inference_runtime': runtime_fingerprint()}
