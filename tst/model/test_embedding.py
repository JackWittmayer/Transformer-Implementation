from model.embedding import Embedding
import torch
from torch import nn


def test_embedding():
    x = torch.tensor([[3, 2, 0, 1], [1, 2, 3, 0]], dtype=torch.int32)
    vocab_size = 4
    d_embedding = 4
    embedding = Embedding(vocab_size, d_embedding)
    embedding.table.weight = nn.Parameter(
        torch.tensor(
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float
        )
    )
    output = embedding(x)
    expected_output = torch.tensor(
        [
            [[3, 3, 3, 3], [2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [0, 0, 0, 0]],
        ],
        dtype=torch.float,
    )
    torch.testing.assert_close(output, expected_output)
