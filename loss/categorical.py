import torch

__all__ = [
    'categorical_kl_divergence',
    'categorical_flat_kl_divergence',
]


def categorical_kl_divergence(log_probs, log_probs_target, temperature_scale_num, mask = None, reduce = True):

    # Mask probabilities
    log_probs = log_probs + mask.log()
    log_probs_target = log_probs_target + mask.log()

    # Temperature scale probabilities
    log_probs = torch.log_softmax(log_probs/temperature_scale_num, dim = -1)
    log_probs_target = torch.log_softmax(log_probs_target / temperature_scale_num, dim = -1)

    # KL-divergence loss
    loss = torch.exp(log_probs_target) * (log_probs_target - log_probs)
    loss = (loss * mask).sum(-1)

    if reduce: return loss.mean()
    return loss


def categorical_flat_kl_divergence(log_probs, mask = None, reduce = True):

    # Define a flat distribution
    log_probs_target = torch.ones_like(log_probs)

    return categorical_kl_divergence(
        log_probs,
        log_probs_target,
        temperature_scale_num = 1.0,
        mask = mask,
        reduce = reduce
    )