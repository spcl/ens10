# TODO:  WCRPS metric (Weighted CRPS with EFI)

import numpy as np
from scipy import special, integrate, stats
from torch import nn
from torch.autograd import Function
import torch
from typing import Union, Tuple, Callable


_normal_dist = torch.distributions.Normal(0., 1.)
_frac_sqrt_pi = 1 / np.sqrt(np.pi)


class CrpsGaussianLoss(nn.Module):
    """
      This is a CRPS loss function assuming a gaussian distribution. We use
      the following link:
      https://github.com/tobifinn/ensemble_transformer/blob/9b31f193048a31efd6aacb759e8a8b4a28734e6c/ens_transformer/measures.py

      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15):
        super(CrpsGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor):

        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)

        if self.mode == 'mean':
            return torch.mean(crps)
        return crps

class WeightedCrpsGaussianLoss(nn.Module):
    """
      This is a WCRPS loss function assuming a gaussian distribution with EFI indeces.
      """

    def __init__(self,
                 mode = 'mean',
                 eps: Union[int, float] = 1E-15):
        super(WeightedCrpsGaussianLoss, self).__init__()

        assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'

        self.mode = mode
        self.eps = eps

    def forward(self,
                pred_mean: torch.Tensor,
                pred_stddev: torch.Tensor,
                target: torch.Tensor,
                efi_tensor: torch.Tensor):

        normed_diff = (pred_mean - target + self.eps) / (pred_stddev + self.eps)
        try:
            cdf = _normal_dist.cdf(normed_diff)
            pdf = _normal_dist.log_prob(normed_diff).exp()
        except ValueError:
            print(normed_diff)
            raise ValueError
        crps = torch.abs(efi_tensor) * pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)

        if self.mode == 'mean':
            return torch.mean(crps)
        return crps