import numpy as np
import scipy as sp
import scipy.special

from typing import List, Dict

__all__ = [
    'ensemblelogits',
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

        if num_models > 1:
            uncertainties['unc_expected_entropy'] = self.compute_expected_entropy(log_probs=start_log_probs)
            uncertainties['unc_expected_entropy'] += self.compute_expected_entropy(log_probs=end_log_probs)

            uncertainties['unc_mutual_information'] = uncertainties['unc_entropy_expected'] - uncertainties['unc_expected_entropy']

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