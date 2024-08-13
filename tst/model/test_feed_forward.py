import torch
from torch import nn
from model.feed_forward import FeedForward


def test_feed_forward():
    hidden_layer_width = 4
    d_embedding = 2
    feed_forward = FeedForward(hidden_layer_width, d_embedding, 0.0)
    feed_forward.mlp1.weight.data.fill_(1)
    feed_forward.mlp2.weight.data.fill_(2)
    feed_forward.mlp1.bias.data.fill_(0)
    feed_forward.mlp2.bias.data.fill_(0)
    activations = torch.tensor([[10, -3], [5, -1]], dtype=torch.float)
    output = feed_forward(activations)
    expected_output = torch.tensor([[56, 56], [32, 32]], dtype=torch.float)
    torch.testing.assert_close(output, expected_output)
