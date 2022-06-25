# Implementation of the EMOS baseline
# We will have two emos models: one for mean and one for std
# The input of each model is a one dimensional vector (mean of the variable) and the model has two parameters
# The output of the model is one also!

import torch
from torch import nn
import torch.nn.functional as F

#FIXME: This model does mean and std in one (also for mean model has 11 parameters)
class emos(nn.Module):
    # EMOS (Ensemble Model Output Statistics Calibrated Probabilistic Forecasting Using Ensemble Model Output Statistics ans Minimum CRPS Estimation (Gneiting et al. 2005))
    def __init__(self, args=None):
        super(emos, self).__init__()

        self.mu = nn.Linear(args.ens_num, 1)
        self.sigma = nn.Linear(1, 1)

    def forward(self, inp):
        # inp has shape [batch_size, 11] (ensembles X = {X_1, X_2 ... X_m} m = 10 and ensvar) for one variable only
        x1 = self.mu(inp)
        x2 = self.sigma(torch.std(inp, dim=-1, unbiased=False, keepdim=True))
        x = torch.cat((x1, x2), dim=-1)
        return x


def EMOS_prepare(args):
    return emos(args)