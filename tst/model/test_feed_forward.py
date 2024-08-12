import torch
from model.feed_forward import FeedForward


def test_feed_forward():
    hiddenLayerWidth = 3
    d_e = 4
    feed_forward = FeedForward(hiddenLayerWidth, d_e, 0.1)
    activations = torch.rand(10, 5, d_e)

    print("activations:", activations)
    output = feed_forward(activations)
    print("feed forward:", output)
    assert output.shape == activations.shape
