"""Fuse and rank unlabeled candidate scores; no evaluation target is required."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from .common import rows, z


def rerank(records, alpha=1.5):
    if not math.isfinite(alpha):
        raise ValueError('alpha must be finite')
    groups = defaultdict(list)
    for row in records:
        groups[row['group_id']].append(row)
    if not groups:
        raise ValueError('empty scores')
    output = []
    for gid, group in sorted(groups.items()):
        if len(group) != 10 or len({r['candidate_item_id'] for r in group}) != 10:
            raise ValueError('expect ten unique candidates per group')
        if len({r['user_id'] for r in group}) != 1:
            raise ValueError('mixed users in group')
        base = z([r['candidate_score'] for r in group])
        semantic = z([r['condition_score'] for r in group])
        fused = [b + alpha * s for b, s in zip(base, semantic)]
        order = sorted(range(len(group)), key=lambda i: (-fused[i], group[i]['candidate_item_id']))
        output.append(dict(group_id=gid, user_id=group[0]['user_id'], alpha=alpha,
                           ranking=[group[i]['candidate_item_id'] for i in order],
                           fused_scores=[fused[i] for i in order]))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=1.5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = rerank(rows(a.scores), a.alpha)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        for row in result:
            stream.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
