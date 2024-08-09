import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_sequence_length, device):
        super().__init__()
        self.table = nn.Embedding(max_sequence_length, embedding_size)
        self.device = device

    def forward(self, sequence):
        positions = torch.zeros(sequence.shape, dtype=torch.int32).to(self.device)
        positions[:, ::] = torch.arange(0, sequence.shape[-1])
        positional_embeddings = self.table(positions)
        return positional_embeddings
