"""Label-free, dataset-independent input to the fixed top/bottom layout engine."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(value):
    return ' '.join(str(value).split())


def value_text(value):
    if isinstance(value, list):
        return '; '.join(value_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value) if value is not None else 'Unknown'


def document(profile, item):
    if not isinstance(profile, str) or not isinstance(item, dict) or not item:
        raise ValueError('expect a user-context string and a nonempty item object')
    user = [dict(id=f'u{i}', text=normalize(line))
            for i, line in enumerate(profile.splitlines()) if normalize(line)]
    fields = [dict(id=f'i{i}', key=key, label=key.replace('_', ' ').title(),
                   values=[normalize(value_text(v)) for v in value] if isinstance(value, list)
                          else [normalize(value_text(value))],
                   text=normalize(key.replace('_', ' ').title() + ': ' + value_text(value)))
              for i, (key, value) in enumerate(item.items())]
    if not user:
        raise ValueError('empty user context')
    return dict(user=user, item=fields)


def plain_text(doc):
    return ('USER PROFILE\n' + '\n'.join(f['text'] for f in doc['user'])
            + '\n\nCANDIDATE ITEM\n' + '\n'.join(f['text'] for f in doc['item']))


def select_examples(path, count=10):
    """Select length-stratified training examples and expose no source IDs or labels."""
    unique = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != 'train':
                raise ValueError('layout initialization accepts split=train only')
            doc = document(row['user_context'], row['candidate_payload'])
            unique.setdefault(fingerprint(doc), doc)
    ordered = sorted(unique.items(), key=lambda p: (len(plain_text(p[1])), p[0]))
    if len(ordered) < count:
        raise ValueError(f'need {count} distinct training examples; found {len(ordered)}')
    indexes = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    return [dict(id=f'example_{i+1:02d}', document=ordered[k][1], document_sha256=ordered[k][0])
            for i, k in enumerate(indexes)]
