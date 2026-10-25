"""Exact short-list PL expectation for training diagnostics and gradient tests."""
from functools import lru_cache
import math

import torch


@lru_cache(maxsize=16)
def _subsets(negatives):
    layers = [[] for _ in range(negatives + 1)]
    bits = []
    predecessors = []
    for mask in range(1 << negatives):
        selected = [(mask >> j) & 1 for j in range(negatives)]
        layers[sum(selected)].append(mask)
        bits.append(selected)
        predecessors.append([mask ^ (1 << j) if selected[j] else 0
                             for j in range(negatives)])
    return layers, bits, predecessors


def positive_rank_distribution(scores, positive, temperature=1.0):
    """Exact differentiable positive-rank probabilities, for 1..12 items.

    Cost is O(n * 2**(n-1)); use a different method for larger candidate sets.
    Float64 and explicit remaining-weight sums reduce cancellation. The
    returned tensor keeps the gradient to the original score tensor.
    """
    if scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError('scores must be a floating-point vector')
    n = scores.numel()
    if not 1 <= n <= 12 or not 0 <= positive < n:
        raise ValueError('require 1..12 candidates and a valid positive index')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('scores must be finite')
    scaled = scores.double() / temperature
    weights = (scaled - scaled.max().detach()).exp()
    if not bool((weights > 0).all()):
        raise ValueError('score range is too wide for stable exponentiation')
    positive_weight = weights[positive]
    negative_weights = torch.cat((weights[:positive], weights[positive + 1:]))
    layers, bits, predecessors = _subsets(n - 1)
    selected = torch.tensor(bits, device=scores.device, dtype=weights.dtype)
    previous = torch.tensor(predecessors, device=scores.device, dtype=torch.long)
    remaining = positive_weight + ((1 - selected) * negative_weights).sum(-1)
    probability = weights.new_zeros(1 << (n - 1))
    probability[0] = 1.0
    ranks = []
    for level, masks in enumerate(layers):
        indices = torch.tensor(masks, device=scores.device, dtype=torch.long)
        if level:
            prev = previous[indices]
            arrivals = (probability[prev] * negative_weights / remaining[prev]
                        * selected[indices]).sum(-1)
            probability = probability.scatter(0, indices, arrivals)
        ranks.append((probability[indices] * positive_weight / remaining[indices]).sum())
    # Connect the n=1 constant result to scores as well.
    return torch.stack(ranks) + scores.sum() * 0


def expected_ndcg(scores, positive, temperature=1.0, cutoff=10):
    """Expected binary NDCG@cutoff for a list containing one positive."""
    if cutoff < 1:
        raise ValueError('cutoff must be positive')
    probabilities = positive_rank_distribution(scores, positive, temperature)
    rank = torch.arange(1, len(scores) + 1, device=scores.device, dtype=probabilities.dtype)
    reward = torch.where(rank <= cutoff, 1 / torch.log2(rank + 1), 0.0)
    return (probabilities * reward).sum()
