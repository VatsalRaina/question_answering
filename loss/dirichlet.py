import numpy as np

import torch
import torch.nn.functional as F

__all__ = [
    'dirichlet_kl_divergence',
    'DirichletEstimation',
    'DirichletEstimationLoss',
]


def dirichlet_kl_divergence(log_alphas, log_alphas_target, temperature_scale_num, mask = None, reduce = True):

    # Get target scaled distributions
    alphas_target = torch.exp(log_alphas_target / temperature_scale_num)
    alphas0_target = (alphas_target * mask).sum(-1)

    # Get prediction scaled distribution
    alphas = torch.exp(log_alphas / temperature_scale_num)
    alphas0 = (alphas * mask).sum(-1)

    # Use similar to built in kl divergence (batch)
    t1 = alphas0_target.lgamma() - alphas0.lgamma()
    t2 = ((alphas_target.lgamma() - alphas.lgamma()) * mask).sum(-1)
    t3 = alphas_target - alphas
    t4 = alphas_target.digamma() - alphas0_target.digamma().unsqueeze(-1)

    # Compute masked loss
    loss = t1 - t2 + ((t3 * t4) * mask).sum(-1)

    if reduce: loss = loss.mean()
    return loss


class DirichletEstimation(object):
    def __init__(
            self,
            logprobs: torch.Tensor,
            temperature_scale: float = 1.0,
            estimation_iter: int = 1,
            mask: torch.Tensor = None
    ):
        self.logprobs = logprobs.clone().detach()/temperature_scale
        self.estimation_iter = estimation_iter
        self.eps_init = 1e-3
        self.eps_step = 1e-6

        # We need a mask to remove logits outside the context
        self.mask = mask
        self.additive_mask = (mask + 1e-12).log()

        # Logprobs should have size (batch, models, seqlen)
        assert logprobs.dim() == 3

    @torch.no_grad()
    def estimation_init(self):
        """
        Initialises the mean and scale of the estimated dirichlet.
        """
        # Normalise log probabilities
        self.logprobs = torch.log_softmax(self.logprobs + self.additive_mask.unsqueeze(1), dim = -1)

        # Extract size
        b, m, v = self.logprobs.size()

        # Compute all necessary quantities
        log_e_prob = torch.logsumexp(self.logprobs, dim=1) - np.log(m)
        log_e_sq_prob = torch.logsumexp(2 * self.logprobs, dim=1) - np.log(m)
        log_e_prob_sq = 2 * log_e_prob

        # Used to initialise alpha0
        alpha0 = torch.exp(log_e_prob - log_e_sq_prob) - 1
        alpha0 = alpha0 / (1 - torch.exp(log_e_prob_sq - log_e_sq_prob) + self.eps_init)

        # Average over all estimates in log space
        alpha0 = torch.log(alpha0) * self.mask
        alpha0 = alpha0.sum(dim = -1, keepdim = True) / self.mask.sum(-1, keepdim = True)
        alpha0 = torch.exp(alpha0)

        info = {
            'expected_log_prob': self.logprobs.mean(dim = 1),
            'log_expected_prob': log_e_prob,
            'expected_prob': torch.exp(log_e_prob),
            'alpha0': alpha0
        }
        return info

    @torch.no_grad()
    def estimation_step(self, info):
        """
        Performs an update step of the scale, alpha0
        """
        expected_log_prob = info['expected_log_prob']
        log_expected_prob = info['log_expected_prob']
        expected_prob  = info['expected_prob']
        initial_alpha0 = info['alpha0']

        # Number of classes size
        v = expected_prob.size(-1)

        # Sequence of steps to update alpha0
        new_alpha0 = torch.digamma(initial_alpha0 * expected_prob) - expected_log_prob
        new_alpha0 = (new_alpha0 * expected_prob * self.mask).sum(dim=-1, keepdim=True)

        # Following steps are numerically instable
        new_alpha0 += (v - 1) / (initial_alpha0 + self.eps_step) - torch.digamma(initial_alpha0 + self.eps_step)
        new_alpha0 = (v - 1) / (new_alpha0 + self.eps_step)  # Adding errors to ensure it works

        info['alpha0'] = F.softplus(new_alpha0, beta=5.0)
        return info

    @torch.no_grad()
    def estimation(self):
        # Initialise quantities
        info = self.estimation_init()

        # Perform estimation
        for i in range(self.estimation_iter):
            info = self.estimation_step(info)

        alpha0 = info['alpha0']

        # Due to highly confident predictions, overflows need to be removed
        mask = torch.isnan(alpha0)

        # Replace all nans with the mean of the rest
        alpha0[mask] = alpha0[~mask].mean()

        # Simply enforce all log alphas to be positive to avoid inverted dirichlet target
        estimated_log_alphas = F.softplus(info['log_expected_prob'] + torch.log(alpha0), beta=5.0)
        return estimated_log_alphas


class DirichletEstimationLoss(object):
    def __init__(self):
        pass

    def __call__(self, args, logits, noisy_logits, context_mask, reduce = True):

        # Arguments for estimation and numerical stability
        temperature_scale_est = getattr(args, 'temperature_scale_est', 1.0)
        temperature_scale_num = getattr(args, 'temperature_scale_num', 1.0)
        estimation_iter = getattr(args, 'estimation_iter', 1)
        reverse = getattr(args, 'reverse', False)

        # Estimator class for proxy training
        estimator = DirichletEstimation(
            logprobs = noisy_logits.clone().detach(),
            temperature_scale = temperature_scale_est,
            estimation_iter = estimation_iter,
            mask = context_mask
        )

        # Fist estimate target
        log_alphas_target = estimator.estimation()

        # Reverse loss
        if reverse:
            logits, log_alphas_target = log_alphas_target, logits

        # Get loss
        loss = dirichlet_kl_divergence(
            log_alphas = logits,
            log_alphas_target = log_alphas_target,
            temperature_scale_num = temperature_scale_num,
            mask = context_mask,
            reduce = reduce
        )
        return loss


