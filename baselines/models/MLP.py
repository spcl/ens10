
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LocationEmbedding(nn.Module):
    def __init__(self, infeature, outfeature, scale, offset):
        super().__init__()
        self.embed = nn.Embedding(infeature, outfeature, max_norm = 1.0)
        self.scale = scale
        self.offset = offset
    def forward(self, idx):
        if idx.dtype != torch.long:
            idx = torch.floor(idx*self.scale+self.offset).long()
        embed = self.embed(idx)
        return embed

class mlp(nn.Module):
    # Multilayer perceptron
    def __init__(self, nr_nodes, nr_variables, args=None):
        super(mlp, self).__init__()

        # Number of nodes in hidden layer
        self.l0_nr_nodes= nr_nodes
        self.n = nr_variables
        
        self.channels = 2*self.n + 3
        
        self.layers = nn.Sequential(
            nn.Linear(self.channels, self.l0_nr_nodes),
            nn.ReLU(),
            nn.Linear(self.l0_nr_nodes, 2),
        )

    def forward(self, inp):
        # inp has shape [batch_size, 16] (mean and std for each variable at pressure level to be predicted and for T=48h, and spatial information (latitude, longitude))
        x = self.layers(inp)
        return x

def MLP_prepare(args):
    if args.target_var in ['t2m']:
        return mlp(128, 11, args)
    return mlp(128, 7, args)