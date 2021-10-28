import numpy as np

import torch
import torch.nn as nn

__all__ = [
    'MultiplicativeGaussianLayer',
    'multiplicativegaussianlayer'
]

def get_default_device(use_cuda = True):
    """
    Returns cuda/cpu device
    """
    return torch.device('cuda') if (use_cuda and torch.cuda.is_available()) else torch.device('cpu')


class MultiplicativeGaussianLayer(nn.Module):
    def __init__(
            self,
            noise_a: float = 0.0,
            noise_b: float = 0.0,
            use_gpu: bool = True,
            **kwargs
    ):
        super(MultiplicativeGaussianLayer, self).__init__()
        self.device = get_default_device(use_gpu)
        self.a, self.b = noise_a, noise_b

        # Fixed standard deviation
        self.use_uniform: bool = np.abs(self.b - self.a) > 1e-3

    def get_uniform(self, x: torch.Tensor, *args, **kwargs):
        """
        Sample uniform standard deviation:
            x: (batch, classes)
            u: (batch, 1)
        """

        if not self.use_uniform: return self.a

        # Sample noise of data type:
        dtype = x.dtype

        # Get size of noise
        noise_size = x.size()[:-1]

        # Get uniform noise
        noise = torch.rand(*noise_size, dtype = dtype, device = self.device)

        # Scale the return
        return self.a +  (self.b - self.a) * noise.unsqueeze(-1)

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        if not self.training: return x

        # Sample noise of data type:
        dtype = x.dtype

        # Get noise of chosen data type
        noise = torch.randn(*x.size(), dtype = dtype, device = self.device)

        # Scale noise by a uniform random variable
        noise = noise * self.get_uniform(x)

        # One mean gaussian noise with random std deviation
        return x * (noise + 1.0)


def multiplicativegaussianlayer(**kwargs):
    return MultiplicativeGaussianLayer(**kwargs)