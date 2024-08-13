import torch
from torch import nn
from model.layer_norm import LayerNorm


def test_layer_norm():
    feature_length = 4
    layer_norm = LayerNorm(feature_length)
    layer_norm.scale = nn.Parameter(torch.tensor([2, 2, 2, 2], dtype=torch.float))
    layer_norm.offset = nn.Parameter(torch.tensor([1, 1, 1, 1], dtype=torch.float))

    activations = torch.tensor(
        [
            [[2, 0, 0, 2], [3, 2, 1, 0], [2, 1, 0, -1], [1, 0, -1, -2]],
            [[0, 0, 0, 0], [-1, -1, -1, -1], [1, 1, 1, 1], [2, 2, 2, 2]],
        ],
        dtype=torch.float,
    )
    output = layer_norm(activations)
    expected_output = torch.tensor(
        [
            [
                [3.0000, -1.0000, -1.0000, 3.0000],
                [3.6833, 1.8944, 0.1056, -1.6833],
                [3.6833, 1.8944, 0.1056, -1.6833],
                [3.6833, 1.8944, 0.1056, -1.6833],
            ],
            [
                [1.0000, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.0000],
            ],
        ],
        dtype=torch.float,
    )

    torch.testing.assert_close(output, expected_output, rtol=10e-5, atol=10e-5)
