import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, feature_length):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_length))
        self.offset = nn.Parameter(torch.zeros(feature_length))

    def forward(self, activations):
        mean = torch.mean(activations, -1, keepdim=True)
        variance = torch.var(activations, -1, keepdim=True, unbiased=False)
        normalized_activations = (activations - mean) / torch.sqrt(variance + 1e-6)
        return (normalized_activations * self.scale) + self.offset
