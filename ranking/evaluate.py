"""Evaluate natural-hit groups and scale to a declared complete evaluation cohort."""
import argparse
import math
from pathlib import Path
from .common import digest, rank_metrics, rows, write_json


def evaluate(records, alpha=1.5, all_users=None):
    if not math.isfinite(alpha):
        raise ValueError('alpha must be finite')
    raw, _ = rank_metrics(records)
    fused, rankings = rank_metrics(records, alpha)
    result = dict(raw=raw, fused=fused, alpha=alpha, rankings=rankings)
    if all_users is not None:
        if all_users < fused['groups']:
            raise ValueError('all_users cannot be smaller than natural-hit groups')
        recall = fused['groups'] / all_users
        result.update(all_users=all_users, recall_at_10=recall,
                      end_to_end_fused={k: v * recall for k, v in fused.items() if k != 'groups'})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=1.5)
    p.add_argument('--all-users', type=int)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = evaluate(rows(a.scores), a.alpha, a.all_users)
    result['scores_sha256'] = digest(a.scores)
    write_json(a.output, result)


if __name__ == '__main__':
    main()
