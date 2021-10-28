import torch

from typing import List, Dict

__all__ = [
    'multiheadattention',
]


class BaseClass(object):
    def __init__(self):
        pass

    def __call__(self, args, attention: torch.Tensor,  input_ids: torch.Tensor, **kwargs) -> Dict:
        """
        Computes all uncertainties and return a dict
        """
        raise NotImplementedError()


class MultiHeadAttention(BaseClass):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.eps = 1e-30

    def get_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != 0).float()

    def compute_entropy(self, attention):
        """
        Computes the entropy of an attention matrix over the last index
        """
        return -(attention * (attention + self.eps).log()).sum(-1)

    def compute_expected_entropy(self, attention: torch.Tensor, mask: torch.Tensor):
        """
        Calculates per token expectation of entropy over a multi-head attention.
        This can be seen as data uncertainty.

        The input should have shape (batch, num heads, seqlen, seqlen)
        The mask should have shape (batch, seqlen)
        """
        # This has a shape (batch, num heads, seqlen)
        entropies = self.compute_entropy(attention)

        # Multiply by mask to remove influence from padding
        entropies = entropies * mask.unsqueeze(1)

        # Return the average entropy over each head, size (batch)
        return entropies.sum(-1).mean(1)

    def compute_entropy_expected(self, attention: torch.Tensor, mask: torch.Tensor):
        """
        Calculates per token entropy of expected over a multi-head attention.
        This can be seen as total uncertainty.

        The input should have shape (batch, num heads, seqlen, seqlen)
        The mask should have shape (batch, seqlen)
        """
        # This has a shape (batch, seqlen, seqlen)
        attention = attention.mean(1)

        # Compute entropy over final axis
        entropies = self.compute_entropy(attention)

        # Multiply by mask to remove influence from padding
        entropies = entropies * mask

        # Return the average entropy over each head, size (batch)
        return entropies.sum(-1)

    def __call__(self, args, attention: torch.Tensor,  input_ids: torch.Tensor, **kwargs) -> Dict:
        """
        Computes all uncertainties of interest.
        This class assumes that a single attention
        """
        # Store all uncertainties here mapping name to value
        uncertainties = {}

        # Get the padding mask needed for subsequent computations
        mask = self.get_padding_mask(input_ids)

        # Get the input lengths
        lengths = mask.sum(-1)

        # Compute the expected entropy from multi-head attention
        uncertainties['unc_expected_entropy'] = self.compute_expected_entropy(attention, mask)
        uncertainties['unc_entropy_expected'] = self.compute_entropy_expected(attention, mask)
        uncertainties['unc_mutual_information'] = uncertainties['unc_entropy_expected'] - uncertainties['unc_expected_entropy']

        # Now get various length normalised uncertainties
        names = list(uncertainties.keys())

        for name in names:
            # Standard length normalisation
            uncertainties[name + "_len_norm"] = uncertainties[name] / lengths

            # Standard log-length normalisation
            uncertainties[name + "_len_log_len_norm"] = uncertainties[name] / (lengths * lengths.log())

        return uncertainties


def multiheadattention():
    return MultiHeadAttention()