import torch
from torch import nn
from src.model.attention import MultiHeadedAttention, MaskStrategy

def test_attention():
    d_attn = 4
    length_x = 4
    length_z = 3
    batch_size = 2
    d_out = 2

    queries = torch.rand(batch_size, length_x, d_attn)
    keys = torch.rand(batch_size, length_z, d_attn)
    values = torch.rand(batch_size, length_z, d_out)
    mask = torch.tril(torch.ones(length_x, length_z) == 1)

    v_out = MultiHeadedAttention.attention(
        queries, keys, values, mask, nn.Dropout(0.1)
    )
    assert v_out.shape == (batch_size, length_x, d_out)


def test_multi_headed_attention_encoder_fixed():
    num_heads = 1
    d_attn = 4
    d_x = 4
    d_z = 4
    d_out = 1
    d_mid = 3
    length_z = 3
    batch_size = 1
    padding_mask = torch.tensor([[1, 1, 0]], dtype=torch.int32)

    multi_headed_attention = MultiHeadedAttention(
        num_heads, d_attn, d_x, d_z, d_out, d_mid, MaskStrategy["UNMASKED"], 0.0
    )
    z = torch.tensor([[[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]]], dtype=torch.float32)
    output = multi_headed_attention(z, z, padding_mask)
    assert output.shape == (batch_size, length_z, d_out)


def test_multi_headed_attention_encoder():
    num_heads = 4
    d_attn = 4
    d_x = 4
    d_z = 4
    d_out = 1
    d_mid = 4
    length_z = 3
    batch_size = 3
    padding_mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.int32)

    multi_headed_attention = MultiHeadedAttention(
        num_heads, d_attn, d_x, d_z, d_out, d_mid, MaskStrategy["UNMASKED"], 0.0
    )
    z = torch.tensor(
        [
            [[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]],
            [[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]],
            [[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]],
        ],
        dtype=torch.float32,
    )
    output = multi_headed_attention(z, z, padding_mask)
    assert output.shape == (batch_size, length_z, d_out)


def test_multi_headed_attention_encoder_decoder():
    num_heads = 4
    d_attn = 4
    d_x = 4
    d_z = 4
    d_out = 4
    d_mid = 4
    length_x = 3
    length_z = 3
    batch_size = 4
    padding_mask = torch.tensor([[1, 1, 0]], dtype=torch.int32)

    multi_headed_attention = MultiHeadedAttention(
        num_heads, d_attn, d_x, d_z, d_out, d_mid, MaskStrategy["UNMASKED"], 0.0
    )
    x = torch.rand(batch_size, length_x, d_x)
    z = torch.rand(batch_size, length_z, d_z)
    output = multi_headed_attention(z, x, padding_mask)
    assert output.shape == (batch_size, length_x, d_out)


def test_multi_headed_attention_decoder_self():
    num_heads = 8
    d_attn = 8
    d_x = 8
    d_out = 8
    d_mid = 8
    length_x = 3
    batch_size = 4
    padding_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.int32
    )

    multi_headed_attention = MultiHeadedAttention(
        num_heads, d_attn, d_x, d_x, d_out, d_mid, MaskStrategy["UNMASKED"], 0.0
    )
    multi_headed_attention.enable_subsequent_mask()
    x = torch.rand(batch_size, length_x, d_x)
    output = multi_headed_attention(x, x, padding_mask)
    assert output.shape == (batch_size, length_x, d_out)
