"""Candidate PL GRPO with per-action old probabilities and frozen advantages."""
import math
import torch
import torch.nn.functional as F
from .exact_policy import expected_ndcg


def standardize(scores):
    scores = scores.double()
    centered = scores - scores.mean()
    scale = centered.square().mean().sqrt()
    if float(scale.detach()) <= 1e-12:
        raise ValueError('flat DensePage scores: undefined differentiable z normalization')
    return centered / scale


def policy_logits(scores, retriever_scores, mode):
    dense = standardize(scores)
    if mode == 'dense':
        return dense
    if mode != 'fused':
        raise ValueError(mode)
    prior = retriever_scores.detach().to(dense)
    prior = torch.zeros_like(prior) if float(prior.std(unbiased=False)) <= 1e-12 else standardize(prior)
    return prior + 1.5 * dense


def sample_permutations(logits, number, generator, temperature=1.):
    u = torch.rand((number, len(logits)), dtype=torch.float64, device=logits.device,
                   generator=generator).clamp(1e-12, 1 - 1e-12)
    return torch.argsort(logits.detach().double()[None] / temperature - torch.log(-torch.log(u)),
                        dim=-1, descending=True)


def action_log_probs(logits, permutations, temperature=1.):
    ordered = (logits / temperature)[permutations]
    suffix = torch.logcumsumexp(ordered.flip(-1), dim=-1).flip(-1)
    return ordered - suffix


def permutation_rewards(permutations, positive):
    ranks = (permutations == positive).long().argmax(-1) + 1
    return 1 / torch.log2(ranks.double() + 1)


def make_rollout(old_logits, positive, config, generator):
    if config['objective_kind'] != 'grpo':
        raise ValueError('only the GRPO objective is supported')
    with torch.no_grad():
        permutations = sample_permutations(old_logits, config['samples_per_group'], generator,
                                           config['temperature'])
        reward = permutation_rewards(permutations, positive)
        old_logp = action_log_probs(old_logits, permutations, config['temperature'])
        std = reward.std(unbiased=False)
        relative = (torch.zeros_like(reward) if float(std) == 0.
                    else (reward - reward.mean()) / (std + 1e-6))
        advantage = relative[:, None].expand_as(old_logp)
        result = {'permutations': permutations, 'reward': reward, 'old_logp': old_logp,
                  'advantages': advantage, 'old_logits': old_logits}
        return {name: value.detach().cpu() for name, value in result.items()}


def clipped_task(logits, rollout, temperature=1., epsilon=.2):
    p = rollout['permutations'].to(logits.device)
    current = action_log_probs(logits, p, temperature)[:, :-1]
    old = rollout['old_logp'].detach().to(current)[:, :-1]
    advantage = rollout['advantages'].detach().to(current)[:, :-1]
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    # Sum effective actions, mean trajectories; train_grpo.py averages over users.
    task = -surrogate.sum(-1).mean()
    info = {'ratio_mean': float(ratio.detach().mean()),
            'ratio_max_abs_deviation': float((ratio.detach() - 1).abs().max()),
            'ratio_mean_abs_deviation': float((ratio.detach() - 1).abs().mean()),
            'clip_fraction': float(((ratio.detach() < 1 - epsilon) | (ratio.detach() > 1 + epsilon)).double().mean()),
            'surrogate_clipped_fraction': float((surrogate.detach() < (ratio * advantage).detach()).double().mean()),
            'advantage_mean': float(advantage.mean()), 'advantage_std': float(advantage.std(unbiased=False))}
    return task, info


def objective(delta, reference_scores, retriever_scores, positive, replay_negative,
              replay_weight, config, candidate_ids, rollout):
    if config['objective_kind'] != 'grpo':
        raise ValueError('only the GRPO objective is supported')
    scores = delta[:, 0] - delta[:, 1]
    logits = policy_logits(scores, retriever_scores, config['policy_mode'])
    ref = policy_logits(reference_scores.detach().to(scores), retriever_scores, config['policy_mode']).detach()
    task, info = clipped_task(logits, rollout, config['temperature'], config['clip_epsilon'])
    reward = rollout['reward']
    info.update(sample_reward=float(reward.mean()), sample_reward_std=float(reward.std(unbiased=False)),
                zero_variance_group=float(float(reward.std(unbiased=False)) == 0.))
    with torch.no_grad():
        expected = expected_ndcg(logits, positive, config['temperature'])
    logp = F.log_softmax(logits / config['temperature'], 0)
    logref = F.log_softmax(ref / config['temperature'], 0)
    kl = (logp.exp() * (logp - logref)).sum()
    targets = torch.tensor([0, 1], device=delta.device)
    replay = (F.cross_entropy(delta[[positive, replay_negative]], targets, reduction='none')
              * delta.new_tensor([1., replay_weight])).mean()
    components = [task, config['kl_coefficient'] * kl, config['sft_coefficient'] * replay]
    grads = [torch.autograd.grad(term, delta, retain_graph=True)[0].detach() for term in components]
    def cosine(a, b):
        den = float(a.norm() * b.norm())
        return float((a * b).sum()) / den if den > 1e-15 else 0.
    total = sum(components)
    order = sorted(range(len(scores)), key=lambda i: (-float(logits[i].detach()), candidate_ids[i]))
    rank = order.index(positive) + 1
    info.update(task_loss=float(task.detach()), expected_ndcg=float(expected.detach()),
        kl=float(kl.detach()), replay_ce=float(replay.detach()), total_loss=float(total.detach()),
        positive_rank=rank, greedy_ndcg=1 / math.log2(rank + 1),
        task_score_grad_norm=float(grads[0].norm()), weighted_kl_score_grad_norm=float(grads[1].norm()),
        weighted_replay_score_grad_norm=float(grads[2].norm()),
        task_kl_gradient_cosine=cosine(grads[0], grads[1]), task_replay_gradient_cosine=cosine(grads[0], grads[2]),
        raw_score_std=float(scores.detach().std(unbiased=False)),
        policy_entropy=float(-(logp.exp() * logp).sum().detach()))
    return total, info
