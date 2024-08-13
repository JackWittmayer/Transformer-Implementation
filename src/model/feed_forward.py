import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, hidden_layer_width, d_e, p_dropout):
        super().__init__()
        self.mlp1 = nn.Linear(d_e, hidden_layer_width)
        self.mlp2 = nn.Linear(hidden_layer_width, d_e)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, activations):
        activations = self.mlp1(activations)
        activations = activations.relu()
        activations = self.mlp2(activations)
        activations = self.dropout(activations)
        return activations
