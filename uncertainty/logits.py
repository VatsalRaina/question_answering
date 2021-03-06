import numpy as np
import scipy as sp
import scipy.special

from typing import List, Dict

__all__ = [
    'ensemblelogits',
    'dirichletensemblelogits'
]


class BaseClass(object):
    def __init__(self):
        pass

    def __call__(self, args, start_logits: np.ndarray, end_logits: np.ndarray, **kwargs) -> Dict:
        """
        Computes all uncertainties and return a dict
        """
        raise NotImplementedError()


class EnsembleLogits(BaseClass):
    def __init__(self):
        super(EnsembleLogits, self).__init__()

    @staticmethod
    def compute_unanswerability(log_probs1: np.ndarray, log_probs2: np.ndarray):
        """
        Computes the probability mass of unanswerability
        """

        # Get the joint log probability matrix
        joint_log_prob = np.add.outer(log_probs1, log_probs2)

        # Get the probability matrix
        joint_prob = np.exp(joint_log_prob)

        # Get the unanswerability
        lower_prob = np.tril(joint_prob, -1)

        return lower_prob.sum()

    @staticmethod
    def compute_log_confidence(log_probs: np.ndarray):
        """
        Computes the negative log-confidence over the last axis
        """
        return -log_probs.max(-1)

    @staticmethod
    def compute_logit_confidence(logits: np.ndarray):
        """
        Computes the logit-confidence over the last axis
        """
        return -(logits.max(-1) - logits.min(-1))

    @staticmethod
    def compute_logit_margin(logits: np.ndarray):
        """
        Computes the logit-margin (largest - second largest) over the last axis
        """
        return -(logits.max(-1) - np.transpose(np.transpose(np.sort(logits))[-2]) )

    @staticmethod
    def compute_logit_margin_av(logits: np.ndarray):
        """
        Computes the max minus average logit over the last axis
        """
        return -(logits.max(-1) - logits.mean(-1))

    @staticmethod
    def compute_entropy(log_probs: np.ndarray):
        """
        Computes the entropy over the last axis
        """
        return -(np.exp(log_probs) * log_probs).sum(-1)

    @staticmethod
    def compute_renyi_entropy(log_probs: np.ndarray, alpha=0.1):
        """
        Computes Renyi entropy over the last axis
        """
        scale = 1. / (1. - alpha)
        return scale * sp.special.logsumexp(alpha * log_probs, axis = -1)

    # Methods for computing the average confidence
    def compute_expected_log_confidence(self, log_probs: np.ndarray):
        lconf = self.compute_log_confidence(log_probs)
        return lconf.mean(0)

    def compute_log_confidence_expected(self, log_probs: np.ndarray):
        # Number of models in ensemble
        n = log_probs.shape[0]

        # Average ensemble prediction
        log_probs = sp.special.logsumexp(log_probs, axis=0) - np.log(n)

        # Entropy of average prediction
        return self.compute_log_confidence(log_probs)

    def compute_expected_logit_confidence(self, log_probs: np.ndarray):
        lconf = self.compute_logit_confidence(log_probs)
        return lconf.mean(0)

    def compute_logit_confidence_expected(self, log_probs: np.ndarray):
        # Number of models in ensemble
        n = log_probs.shape[0]

        # Average ensemble prediction
        log_probs = sp.special.logsumexp(log_probs, axis=0) - np.log(n)

        # Entropy of average prediction
        return self.compute_logit_confidence(log_probs)

    def compute_expected_logit_margin(self, log_probs: np.ndarray):
        lconf = self.compute_logit_margin(log_probs)
        return lconf.mean(0)

    def compute_logit_margin_expected(self, log_probs: np.ndarray):
        # Number of models in ensemble
        n = log_probs.shape[0]

        # Average ensemble prediction
        log_probs = sp.special.logsumexp(log_probs, axis=0) - np.log(n)

        # Entropy of average prediction
        return self.compute_logit_margin(log_probs)

    def compute_expected_logit_margin_av(self, log_probs: np.ndarray):
        lconf = self.compute_logit_margin_av(log_probs)
        return lconf.mean(0)

    def compute_logit_margin_av_expected(self, log_probs: np.ndarray):
        # Number of models in ensemble
        n = log_probs.shape[0]

        # Average ensemble prediction
        log_probs = sp.special.logsumexp(log_probs, axis=0) - np.log(n)

        # Entropy of average prediction
        return self.compute_logit_margin_av(log_probs)

    # Methods for computing the expected_of_ and _of_expected
    def compute_expected_entropy(self, log_probs: np.ndarray):
        """
        Computes the entropy over each model prediction and averages over all models.
        The input is assumed to have shape (num models, *, seqlen)
        """
        entropies = self.compute_entropy(log_probs)
        return entropies.mean(0)

    def compute_entropy_expected(self, log_probs: np.ndarray):
        """
        Computes the entropy over the average model prediction
        The input is assumed to have shape (num models, *, seqlen)
        """
        # Number of models in ensemble
        n = log_probs.shape[0]

        # Average ensemble prediction
        log_probs = sp.special.logsumexp(log_probs, axis = 0) - np.log(n)

        # Entropy of average prediction
        return self.compute_entropy(log_probs)

    def compute_epkl(self, log_probs: np.ndarray):
        """
        Computes the expected pairwise KL divergence over the average model prediction
        The input is assumed to have shape (num models, *, seqlen)
        """
        exe = self.compute_expected_entropy(log_probs=log_probs)
        mean_logprobs = log_probs.mean(0)
        mean_probs = np.exp(log_probs).mean(0)
        return -1*(mean_probs * mean_logprobs).sum(-1) - exe

    def compute_expected_renyi_entropy(self, log_probs: np.ndarray):
        """
        Computes the Renyi entropy over each model prediction and averages over all models.
        The input is assumed to have shape (num models, *, seqlen)
        """
        entropies = self.compute_renyi_entropy(log_probs)
        return entropies.mean(0)

    def compute_renyi_entropy_expected(self, log_probs: np.ndarray):
        """
        Computes the Renyi entropy over the average model prediction
        The input is assumed to have shape (num models, *, seqlen)
        """
        # Number of models in ensemble
        n = log_probs.shape[0]

        # Average ensemble prediction
        log_probs = sp.special.logsumexp(log_probs, axis = 0) - np.log(n)

        # Entropy of average prediction
        return self.compute_renyi_entropy(log_probs)

    def __call__(self, args, start_logits: np.ndarray, end_logits: np.ndarray, **kwargs) -> Dict:
        """
        Computes all uncertainties of interest...
        """
        # Store all uncertainties here mapping name to value
        uncertainties = {}

        # Map logits to log-probabilities
        start_log_probs = sp.special.log_softmax(start_logits, axis = -1)
        end_log_probs = sp.special.log_softmax(end_logits, axis=-1)

        # Get the number of models and context length
        num_models, context_len = start_logits.shape

        # Compute unanswerability probability
        uncertainties['unc_unanswerability_conf'] = self.compute_unanswerability(start_log_probs, end_log_probs)

        # Compute all non-normalised confidence based measures
        uncertainties['unc_logit_conf_expected'] = self.compute_logit_confidence_expected(log_probs=start_log_probs)
        uncertainties['unc_logit_conf_expected'] += self.compute_logit_confidence_expected(log_probs=end_log_probs)

        uncertainties['unc_logit_margin_expected'] = self.compute_logit_margin_expected(log_probs=start_log_probs)
        uncertainties['unc_logit_margin_expected'] += self.compute_logit_margin_expected(log_probs=end_log_probs)

        uncertainties['unc_logit_margin_av_expected'] = self.compute_logit_margin_av_expected(log_probs=start_log_probs)
        uncertainties['unc_logit_margin_av_expected'] += self.compute_logit_margin_av_expected(log_probs=end_log_probs)

        uncertainties['unc_log_conf_expected'] = self.compute_log_confidence_expected(log_probs=start_log_probs)
        uncertainties['unc_log_conf_expected'] += self.compute_log_confidence_expected(log_probs=end_log_probs)

        # Compute all non-normalised uncertainties
        uncertainties['unc_entropy_expected'] = self.compute_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_entropy_expected'] += self.compute_entropy_expected(log_probs=end_log_probs)
        """
        uncertainties['unc_renyi_entropy_expected'] = self.compute_renyi_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_renyi_entropy_expected'] += self.compute_renyi_entropy_expected(log_probs=end_log_probs)
        """
        if num_models > 1:
            uncertainties['unc_expected_logit_conf'] = self.compute_expected_logit_confidence(log_probs=start_log_probs)
            uncertainties['unc_expected_logit_conf'] += self.compute_expected_logit_confidence(log_probs=end_log_probs)

            uncertainties['unc_expected_logit_margin'] = self.compute_expected_logit_margin(log_probs=start_log_probs)
            uncertainties['unc_expected_logit_margin'] += self.compute_expected_logit_margin(log_probs=end_log_probs)

            uncertainties['unc_expected_logit_margin_av'] = self.compute_expected_logit_margin_av(log_probs=start_log_probs)
            uncertainties['unc_expected_logit_margin_av'] += self.compute_expected_logit_margin_av(log_probs=end_log_probs)

            uncertainties['unc_expected_log_conf'] = self.compute_expected_log_confidence(log_probs=start_log_probs)
            uncertainties['unc_expected_log_conf'] += self.compute_expected_log_confidence(log_probs=end_log_probs)

            uncertainties['unc_expected_entropy'] = self.compute_expected_entropy(log_probs=start_log_probs)
            uncertainties['unc_expected_entropy'] += self.compute_expected_entropy(log_probs=end_log_probs)

            uncertainties['unc_mutual_information'] = uncertainties['unc_entropy_expected'] - uncertainties['unc_expected_entropy']

            uncertainties['unc_epkl'] = self.compute_epkl(log_probs=start_log_probs)
            uncertainties['unc_epkl'] += self.compute_epkl(log_probs=end_log_probs)
            """  
            uncertainties['unc_expected_renyi_entropy'] = self.compute_expected_renyi_entropy(log_probs=start_log_probs)
            uncertainties['unc_expected_renyi_entropy'] += self.compute_expected_renyi_entropy(log_probs=end_log_probs)

            uncertainties['unc_renyi_mutual_information'] = uncertainties['unc_renyi_entropy_expected'] - uncertainties['unc_expected_renyi_entropy']
            """
        # Now get various length normalised uncertainties
        names = list(uncertainties.keys())

        for name in names:

            if "_conf" in name: continue

            # Standard length normalisation
            uncertainties[name + "_len_norm"] = uncertainties[name]/context_len

            # Standard log-length normalisation
            uncertainties[name + "_log_len_norm"] = uncertainties[name]/np.log(context_len)

        return uncertainties


class DirichletEnsembleLogits(EnsembleLogits):
    def __init__(self, num_samples = 100):
        super(DirichletEnsembleLogits, self).__init__()

        # Numerical stability
        self.eps = 1e-8

        # Sampling based uncertainty estimation
        self.samples = num_samples

    @staticmethod
    def softplus(x: np.ndarray, beta = 5.0):
        x *= beta
        sx = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        return sx/beta

    def compute_expected_logit_confidence(self, log_alphas: np.ndarray):
        # Get the number of models and context length
        num_models, context_len = log_alphas.shape

        # Obtain samples from the dirichlet distribution
        probs = np.array([np.random.dirichlet(np.exp(la), self.samples) for la in log_alphas])
        probs = probs.reshape(num_models * self.samples, context_len)

        # Now map to log probabilities and compute entropy
        return super(DirichletEnsembleLogits, self).compute_expected_logit_confidence(
            log_probs = np.log(probs + self.eps)
        )

    def compute_expected_log_confidence(self, log_alphas: np.ndarray):
        # Get the number of models and context length
        num_models, context_len = log_alphas.shape

        # Obtain samples from the dirichlet distribution
        probs = np.array([np.random.dirichlet(np.exp(la), self.samples) for la in log_alphas])
        probs = probs.reshape(num_models * self.samples, context_len)

        # Now map to log probabilities and compute entropy
        return super(DirichletEnsembleLogits, self).compute_expected_log_confidence(
            log_probs = np.log(probs + self.eps)
        )

    def compute_expected_entropy(self, log_alphas: np.ndarray):
        """
        Computes the entropy over each model prediction and averages over all models.
        The input is assumed to have shape (num models, *, seqlen)
        """
        alphas = np.exp(log_alphas)
        alpha0 = alphas.sum(axis = -1)

        entropy = sp.special.digamma(alpha0 + 1)
        entropy -= np.sum(alphas * sp.special.digamma(alphas + 1), axis = -1) / alpha0

        return entropy.mean(0)

    def compute_expected_renyi_entropy(self, log_alphas: np.ndarray):
        """
        Computes the Renyi entropy over each model prediction and averages over all models.
        The input is assumed to have shape (num models, *, seqlen)
        """
        # Get the number of models and context length
        num_models, context_len = log_alphas.shape

        # Obtain samples from the dirichlet distribution
        probs = np.array([np.random.dirichlet(np.exp(la), self.samples) for la in log_alphas])
        probs = probs.reshape(num_models * self.samples, context_len)

        # Now map to log probabilities and compute entropy
        return super(DirichletEnsembleLogits, self).compute_expected_renyi_entropy(
            log_probs = np.log(probs + self.eps)
        )

    def compute_expected_exp_renyi_entropy(self, log_alphas: np.array):
        """
        Computes the Exponentiation of Renyi entropy over each model prediction and averages over all models.
        The input is assumed to have shape (num models, *, seqlen)
        """
        alphas = np.exp(log_alphas)
        alpha0 = alphas.sum(axis = -1)

        entropy = np.exp(sp.special.gammaln(alpha0) - sp.special.gammaln(alpha0 + 0.50))
        entropy *= (np.exp(sp.special.gammaln(alphas + 0.50) - sp.special.gammaln(alphas))).sum(axis = -1)

        return entropy.mean(0)

    def __call__(self, args, start_logits: np.ndarray, end_logits: np.ndarray, **kwargs) -> Dict:
        """
        Computes all uncertainties of interest...
        """
        # Store all uncertainties here mapping name to value
        uncertainties = {}

        if args.use_softplus:
            # Ensure alphas > 0 similar to training
            start_logits = self.softplus(start_logits)
            end_logits = self.softplus(end_logits)

        # Map logits to log-probabilities
        start_log_probs = sp.special.log_softmax(start_logits, axis = -1)
        end_log_probs = sp.special.log_softmax(end_logits, axis=-1)

        # Get the number of models and context length
        num_models, context_len = start_logits.shape

        # Compute all non-normalised confidence based measures
        uncertainties['unc_logit_conf_expected'] = self.compute_logit_confidence_expected(log_probs=start_log_probs)
        uncertainties['unc_logit_conf_expected'] += self.compute_logit_confidence_expected(log_probs=end_log_probs)

        uncertainties['unc_log_conf_expected'] = self.compute_log_confidence_expected(log_probs=start_log_probs)
        uncertainties['unc_log_conf_expected'] += self.compute_log_confidence_expected(log_probs=end_log_probs)

        uncertainties['unc_expected_logit_conf'] = self.compute_expected_logit_confidence(log_alphas=start_logits)
        uncertainties['unc_expected_logit_conf'] += self.compute_expected_logit_confidence(log_alphas=end_logits)

        uncertainties['unc_expected_log_conf'] = self.compute_expected_log_confidence(log_alphas=start_logits)
        uncertainties['unc_expected_log_conf'] += self.compute_expected_log_confidence(log_alphas=end_logits)

        # Compute all non-normalised uncertainties
        uncertainties['unc_entropy_expected'] = self.compute_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_entropy_expected'] += self.compute_entropy_expected(log_probs=end_log_probs)

        uncertainties['unc_renyi_entropy_expected'] = self.compute_renyi_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_renyi_entropy_expected'] += self.compute_renyi_entropy_expected(log_probs=end_log_probs)

        uncertainties['unc_expected_entropy'] = self.compute_expected_entropy(log_alphas=start_logits)
        uncertainties['unc_expected_entropy'] += self.compute_expected_entropy(log_alphas=end_logits)
        uncertainties['unc_mutual_information'] = uncertainties['unc_entropy_expected'] - uncertainties['unc_expected_entropy']

        uncertainties['unc_expected_renyi_entropy'] = self.compute_expected_renyi_entropy(log_alphas=start_logits)
        uncertainties['unc_expected_renyi_entropy'] += self.compute_expected_renyi_entropy(log_alphas=end_logits)
        uncertainties['unc_renyi_mutual_information'] = uncertainties['unc_renyi_entropy_expected'] - uncertainties['unc_expected_renyi_entropy']

        uncertainties['unc_expected_exp_renyi_entropy'] = self.compute_expected_exp_renyi_entropy(log_alphas = start_logits)
        uncertainties['unc_expected_exp_renyi_entropy'] += self.compute_expected_exp_renyi_entropy(log_alphas = end_logits)

        # Now get various length normalised uncertainties
        names = list(uncertainties.keys())

        for name in names:

            if "_conf" in name: continue

            # Standard length normalisation
            uncertainties[name + "_len_norm"] = uncertainties[name]/context_len

            # Standard log-length normalisation
            uncertainties[name + "_log_len_norm"] = uncertainties[name]/np.log(context_len)

        return uncertainties


def ensemblelogits():
    return EnsembleLogits()


def dirichletensemblelogits(num_samples = 100):
    return DirichletEnsembleLogits(num_samples = num_samples)


def load_class(args):
    if args.class_uncertainty == 'ensemblelogits':
        return ensemblelogits()
    return dirichletensemblelogits(num_samples = args.num_samples_uncertainty)