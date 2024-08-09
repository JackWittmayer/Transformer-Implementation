import torch.nn as nn
from .attention import MultiHeadedAttention, MaskStrategy
from .layer_norm import LayerNorm
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self, num_heads, d_attn, d_x, d_z, d_out, d_mid, d_mlp, p_dropout
    ):
        super().__init__()
        self.multi_head_self_attention = MultiHeadedAttention(
            num_heads,
            d_attn,
            d_x,
            d_z,
            d_out,
            d_mid,
            MaskStrategy["MASKED"],
            p_dropout
        )
        self.layer_norm1 = LayerNorm(d_x)
        self.multi_head_global_attention = MultiHeadedAttention(
            num_heads,
            d_attn,
            d_x,
            d_z,
            d_out,
            d_mid,
            MaskStrategy["UNMASKED"],
            p_dropout
        )
        self.layer_norm2 = LayerNorm(d_x)
        self.feed_forward = FeedForward(d_mlp, d_x, p_dropout)
        self.layer_norm3 = LayerNorm(d_x)

    def forward(self, z, x, src_mask, tgt_mask):
        x = self.layer_norm1(x)
        x = x + self.multi_head_self_attention(x, x, tgt_mask)
        x = self.layer_norm2(x)
        x = x + self.multi_head_global_attention(z, x, src_mask)
        x = self.layer_norm3(x)
        x = x + self.feed_forward(x)
        return x

    def disable_subsequent_mask(self):
        self.multi_head_self_attention.disable_subsequent_mask()


class Decoder(nn.Module):
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
            decoder_layer = DecoderLayer(
                num_heads, d_attn, d_x, d_z, d_out, d_mid, d_mlp, p_dropout
            )
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.final_norm = LayerNorm(d_x)

    def forward(self, z, x, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(z, x, src_mask, tgt_mask)
        return self.final_norm(x)

    def disable_subsequent_mask(self):
        for layer in self.layers:
            layer.multi_head_self_attention.disable_subsequent_mask()
