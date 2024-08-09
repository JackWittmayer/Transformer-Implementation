import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, embedding_size)

    def forward(self, sequence):
        embeddings = self.table(sequence)
        return embeddings
