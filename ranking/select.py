"""Lock trained checkpoints using validation metrics and a fixed fusion coefficient."""
import argparse
from pathlib import Path
from .common import digest, rows, write_json
from .evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, nargs='+', required=True)
    p.add_argument('--steps', type=int, nargs='+', required=True)
    p.add_argument('--alpha', type=float, default=1.5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.scores) != len(a.steps) or len(set(a.steps)) != len(a.steps) or any(s <= 0 for s in a.steps):
        raise ValueError('provide distinct positive trained steps matching score files')
    if a.output.exists():
        raise FileExistsError(a.output)
    candidates, cohort = [], None
    for path, step in zip(a.scores, a.steps):
        records = rows(path)
        if any(r.get('split') not in ['valid', 'validation'] for r in records):
            raise ValueError('selection requires validation records with explicit split')
        identities = {(r['group_id'], r['candidate_item_id'], r['target_item_id'], r['candidate_score'],
                       r.get('input_sha256')) for r in records}
        if cohort is not None and identities != cohort:
            raise ValueError('checkpoint validation cohorts differ')
        cohort = identities
        metrics = evaluate(records, a.alpha)['fused']
        candidates.append(dict(step=step, fused=metrics, scores=str(path.resolve()), scores_sha256=digest(path)))
    best = max(candidates, key=lambda c: (c['fused']['NDCG@10'], c['fused']['MRR@10'],
                                        c['fused']['H@1'], -c['step']))
    write_json(a.output, dict(selected=best, alpha=a.alpha, candidates=candidates,
                             rule='validation NDCG, MRR, H1 descending; earlier trained step',
                             test_used_for_selection=False))


if __name__ == '__main__':
    main()
