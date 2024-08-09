import torch.nn as nn
from .attention import MultiHeadedAttention, MaskStrategy
from .layer_norm import LayerNorm
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self, num_heads, d_attn, d_x, d_z, d_out, d_mid, d_mlp, p_dropout
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadedAttention(
            num_heads,
            d_attn,
            d_x,
            d_z,
            d_out,
            d_mid,
            MaskStrategy["UNMASKED"],
            p_dropout
        )
        self.layer_norm1 = LayerNorm(d_z)
        self.feed_forward = FeedForward(d_mlp, d_z, p_dropout)
        self.layer_norm2 = LayerNorm(d_z)

    def forward(self, z, padding_mask):
        z = self.layer_norm1(z)
        z = z + self.multi_head_attention(z, z, padding_mask)
        z = self.layer_norm2(z)
        z = z + self.feed_forward(z)
        return z


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_attn,
        d_x,
        d_z,
        d_out,
        d_mid,
        d_mlp,
        p_dropout
    ):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            encoder_layer = EncoderLayer(
                num_heads, d_attn, d_x, d_z, d_out, d_mid, d_mlp, p_dropout
            )
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.final_norm = LayerNorm(d_z)

    def forward(self, z, padding_mask):
        for layer in self.layers:
            z = layer(z, padding_mask)
        return self.final_norm(z)
