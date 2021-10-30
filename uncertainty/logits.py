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
    def compute_log_confidence(log_probs: np.ndarray):
        """
        Computes the log-confidence over the last axis
        """
        return sp.special.logsumexp(log_probs, axis = -1)

    @staticmethod
    def compute_entropy(log_probs: np.ndarray):
        """
        Computes the entropy over the last axis
        """
        return -(np.exp(log_probs) * log_probs).sum(-1)

    @staticmethod
    def compute_renyi_entropy(log_probs: np.ndarray, alpha=0.5):
        """
        Computes Renyi entropy over the last axis
        """
        scale = 1. / (1. - alpha)
        return scale * sp.special.logsumexp(alpha * log_probs, axis = -1)

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

        # Compute all non-normalised uncertainties
        uncertainties['unc_entropy_expected'] = self.compute_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_entropy_expected'] += self.compute_entropy_expected(log_probs=end_log_probs)

        uncertainties['unc_renyi_entropy_expected'] = self.compute_renyi_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_renyi_entropy_expected'] += self.compute_renyi_entropy_expected(log_probs=end_log_probs)

        if num_models > 1:
            uncertainties['unc_expected_entropy'] = self.compute_expected_entropy(log_probs=start_log_probs)
            uncertainties['unc_expected_entropy'] += self.compute_expected_entropy(log_probs=end_log_probs)

            uncertainties['unc_mutual_information'] = uncertainties['unc_entropy_expected'] - uncertainties['unc_expected_entropy']

            uncertainties['unc_expected_renyi_entropy'] = self.compute_expected_renyi_entropy(log_probs=start_log_probs)
            uncertainties['unc_expected_renyi_entropy'] += self.compute_expected_renyi_entropy(log_probs=end_log_probs)

            uncertainties['unc_renyi_mutual_information'] = uncertainties['unc_renyi_entropy_expected'] - uncertainties['unc_expected_renyi_entropy']

        # Now get various length normalised uncertainties
        names = list(uncertainties.keys())

        for name in names:

            # Standard length normalisation
            uncertainties[name + "_len_norm"] = uncertainties[name]/context_len

            # Standard log-length normalisation
            uncertainties[name + "_log_len_norm"] = uncertainties[name]/np.log(context_len)

        return uncertainties


class DirichletEnsembleLogits(EnsembleLogits):
    def __init__(self, num_samples = 100):
        super(DirichletEnsembleLogits, self).__init__()

        # Sampling based uncertainty estimation
        self.samples = num_samples

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
            log_probs = np.log(probs)
        )

    def compute_expected_exp_renyi_entropy(self, log_alphas: np.array):
        """
        Computes the Exponentiation of Renyi entropy over each model prediction and averages over all models.
        The input is assumed to have shape (num models, *, seqlen)
        """
        alphas = np.exp(log_alphas)
        alpha0 = alphas.sum(axis = -1)

        entropy = sp.special.gamma(alpha0)/sp.special.gamma(alpha0 + 0.50)
        entropy *= (sp.special.gamma(alphas + 0.50)/sp.special.gamma(alphas)).sum(axis = -1)

        return entropy.mean(0)

    def __call__(self, args, start_logits: np.ndarray, end_logits: np.ndarray, **kwargs) -> Dict:
        """
        Computes all uncertainties of interest...
        """
        # Store all uncertainties here mapping name to value
        uncertainties = {}

        # Map logits to log-probabilities
        start_log_probs = sp.special.log_softmax(start_logits, axis = -1)
        end_log_probs = sp.special.log_softmax(end_logits, axis=-1)

        start_log_alphas = np.exp(start_logits.astype('float128'))
        end_log_alphas = np.exp(end_logits.astype('float128'))

        import pdb; pdb.set_trace()

        # Get the number of models and context length
        num_models, context_len = start_logits.shape

        # Compute all non-normalised uncertainties
        uncertainties['unc_entropy_expected'] = self.compute_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_entropy_expected'] += self.compute_entropy_expected(log_probs=end_log_probs)

        uncertainties['unc_renyi_entropy_expected'] = self.compute_renyi_entropy_expected(log_probs=start_log_probs)
        uncertainties['unc_renyi_entropy_expected'] += self.compute_renyi_entropy_expected(log_probs=end_log_probs)

        uncertainties['unc_expected_entropy'] = self.compute_expected_entropy(log_alphas=start_log_alphas)
        uncertainties['unc_expected_entropy'] += self.compute_expected_entropy(log_alphas=end_log_alphas)
        uncertainties['unc_mutual_information'] = uncertainties['unc_entropy_expected'] - uncertainties['unc_expected_entropy']

        uncertainties['unc_expected_renyi_entropy'] = self.compute_expected_renyi_entropy(log_alphas=start_log_alphas)
        uncertainties['unc_expected_renyi_entropy'] += self.compute_expected_renyi_entropy(log_alphas=end_log_alphas)
        uncertainties['unc_renyi_mutual_information'] = uncertainties['unc_renyi_entropy_expected'] - uncertainties['unc_expected_renyi_entropy']

        uncertainties['unc_expected_exp_renyi_entropy'] = self.compute_expected_exp_renyi_entropy(log_alphas = start_log_alphas)
        uncertainties['unc_expected_exp_renyi_entropy'] += self.compute_expected_exp_renyi_entropy(log_alphas = end_log_alphas)

        # Now get various length normalised uncertainties
        names = list(uncertainties.keys())

        for name in names:

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