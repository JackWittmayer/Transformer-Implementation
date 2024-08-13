import torch
from torch import nn
from model.unembedding import Unembedding


def test_unembedding():
    embeddings = torch.tensor(
        [
            [[3, 3, 3, 3], [2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [0, 0, 0, 0]],
        ],
        dtype=torch.float,
    )
    vocab_size = 4
    d_embedding = 4
    unembedding = Unembedding(vocab_size, d_embedding)
    unembedding.weight.weight = nn.Parameter(
        torch.tensor(
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float
        )
    )
    unembedding.weight.bias = nn.Parameter(torch.zeros(vocab_size))
    output = unembedding(embeddings)
    expected_output = torch.tensor(
        [
            [
                [0.0, 12.0, 24.0, 36.0],
                [0.0, 8.0, 16.0, 24.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 8.0, 12.0],
            ],
            [
                [0.0, 4.0, 8.0, 12.0],
                [0.0, 8.0, 16.0, 24.0],
                [0.0, 12.0, 24.0, 36.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float,
    )
    torch.testing.assert_close(output, expected_output)
