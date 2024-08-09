import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hiddenLayerWidth, d_e, p_dropout):
        super().__init__()
        self.mlp1 = nn.Parameter(torch.rand(d_e, hiddenLayerWidth))
        self.mlp2 = nn.Parameter(torch.rand(hiddenLayerWidth, d_e))
        self.mlp1_bias = nn.Parameter(torch.zeros(hiddenLayerWidth))
        self.mlp2_bias = nn.Parameter(torch.zeros(d_e))
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, activations):
        activations = torch.matmul(activations, self.mlp1) + self.mlp1_bias
        activations = activations.relu()
        activations = torch.matmul(activations, self.mlp2) + self.mlp2_bias
        activations = self.dropout(activations)
        return activations
