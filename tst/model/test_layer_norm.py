import torch
from model.layer_norm import LayerNorm


def test_layer_norm():
    feature_length = 4
    length_x = 3
    batch_size = 5
    layer_norm = LayerNorm(feature_length)

    activations = torch.rand(batch_size, length_x, feature_length)

    print("activations:", activations)
    print("layer_normed:", layer_norm(activations))
    assert layer_norm(activations).shape == activations.shape
