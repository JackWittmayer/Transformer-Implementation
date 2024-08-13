import torch
from torch import nn
from model.positional_embedding import PositionalEmbedding


def test_positional_embedding():
    x = torch.tensor([[3, 2, 0, 1], [1, 2, 3, 0]], dtype=torch.int32)
    d_embedding = 4
    positional_embedding = PositionalEmbedding(d_embedding, 10, torch.device("cpu"))
    positional_embedding.table.weight = nn.Parameter(
        torch.tensor(
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float
        )
    )
    output = positional_embedding(x)
    expected_output = torch.tensor(
        [
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        ],
        dtype=torch.float,
    )
    torch.testing.assert_close(output, expected_output)
