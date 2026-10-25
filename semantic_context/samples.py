"""Create semantic SFT pairs and complete candidate lists before visual rendering."""
import hashlib
import math


def pointwise(groups, items):
    pairs, complete = [], []
    if len({g['sample_id'] for g in groups}) != len(groups):
        raise ValueError('duplicate group IDs')
    for group in groups:
        candidates = list(map(str, group['candidate_item_ids']))
        target = str(group['target_item_id'])
        if (len(candidates) != 10 or len(set(candidates)) != 10 or target not in candidates
                or len(group['candidate_scores']) != 10
                or not all(math.isfinite(float(s)) for s in group['candidate_scores'])):
            raise ValueError('expect ten-candidate natural-hit groups with finite scores')
        if group.get('target_injected', False) or set(candidates) & set(map(str, group.get('history_item_ids', []))):
            raise ValueError('invalid natural candidates')
        ranked = sorted(zip(candidates, group['candidate_scores']), key=lambda x: (-float(x[1]), x[0]))
        negatives = [item for item, _ in ranked if item != target]
        offset = int(hashlib.sha256(group['sample_id'].encode()).hexdigest()[:16], 16) % len(negatives)
        negative = negatives[offset]
        text = next(c['text'] for m in group['messages'] if m['role'] == 'user'
                    for c in m['content'] if c['type'] == 'text')
        context = text.split('\n\nCandidate music items:', 1)[0].strip()
        current = []
        for position, (item, score) in enumerate(zip(candidates, group['candidate_scores']), 1):
            label = 'A' if item == target else 'B'
            rank = next(i for i, (candidate, _) in enumerate(ranked, 1) if candidate == item)
            raw_weight = 1. if label == 'A' else .2 if rank <= 3 else .5 if rank <= 6 else 1.
            row = dict(sample_id=f"{group['sample_id']}-{label}-{item}", group_id=group['sample_id'],
                       split=group['split'], user_id=group['user_id'], target_item_id=target,
                       candidate_item_id=item, candidate_position=position, candidate_score=score,
                       label=label, _replay=(item in [target, negative]), raw_sample_weight=raw_weight,
                       user_context=context, candidate_payload=items[item])
            for key in ['target_timestamp', 'profile_history_key', 'history_item_ids']:
                if key in group:
                    row[key] = group[key]
            current.append(row)
        complete.extend(current)
        pairs.extend(sorted((r for r in current if r['_replay']), key=lambda r: r['label']))
    negative_weights = [r['raw_sample_weight'] for r in pairs if r['label'] == 'B']
    mean = sum(negative_weights) / len(negative_weights) if negative_weights else 1.
    for row in complete:
        row['sample_weight'] = row['raw_sample_weight'] / mean if row['label'] == 'B' else 1.
        row['_replay_weight'] = row['sample_weight'] if row['_replay'] else 0.
    return pairs, complete
