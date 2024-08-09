import torch.nn as nn


class Unembedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.weight = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        return self.weight(x)
