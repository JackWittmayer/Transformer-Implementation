from enum import Enum
from torch import nn
import torch
import math


class MaskStrategy(Enum):
    UNMASKED = 1
    MASKED = 2


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        d_attn,
        d_x,
        d_z,
        d_out,
        d_mid,
        maskStrategy,
        p_dropout
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_attn = d_attn
        self.d_x = d_x
        self.d_z = d_z
        self.d_out = d_out
        self.d_mid = d_mid
        self.maskStrategy = maskStrategy
        self.weight_query = nn.Linear(d_x, d_attn)
        self.weight_key = nn.Linear(d_z, d_attn)
        self.weight_value = nn.Linear(d_z, d_mid)
        self.weight_out = nn.Linear(d_mid, d_out)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, z, x, padding_mask):
        length_z = z.shape[-2]
        length_x = x.shape[-2]
        batch_size = x.shape[0]

        queries = (
            self.weight_query(x)
            .view(batch_size, length_x, self.num_heads, -1)
            .transpose(1, 2)
        )
        keys = (
            self.weight_key(z)
            .view(batch_size, length_z, self.num_heads, -1)
            .transpose(1, 2)
        )
        values = (
            self.weight_value(z)
            .view(batch_size, length_z, self.num_heads, -1)
            .transpose(1, 2)
        )

        assert queries.shape == (
            batch_size,
            self.num_heads,
            length_x,
            self.d_attn / self.num_heads,
        )
        assert keys.shape == (
            batch_size,
            self.num_heads,
            length_z,
            self.d_attn / self.num_heads,
        )
        assert values.shape == (
            batch_size,
            self.num_heads,
            length_z,
            self.d_mid / self.num_heads,
        )

        if self.maskStrategy == MaskStrategy["UNMASKED"]:
            mask = padding_mask.unsqueeze(-2)
        elif self.maskStrategy == MaskStrategy["MASKED"]:
            padding_mask = padding_mask.unsqueeze(-2)
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            mask = torch.tril(torch.ones(length_x, length_z) == 1).to(device)
            mask = mask & padding_mask
        mask = mask.unsqueeze(1)
        v_out = self.attention(queries, keys, values, mask, self.dropout)
        assert v_out.shape == (
            batch_size,
            self.num_heads,
            length_x,
            self.d_mid / self.num_heads,
        )
        v_out = v_out.transpose(1, 2).reshape(batch_size, length_x, -1)
        output = self.weight_out(v_out)
        assert output.shape == (batch_size, length_x, self.d_out)
        return output

    @staticmethod
    def attention(queries, keys, values, mask, dropout):
        keys_transposed = torch.transpose(keys, -2, -1)
        scores = torch.matmul(queries, keys_transposed)
        # assert scores.shape == (keys.shape[0], keys.shape[-1], queries.shape[-1])
        scores = scores.masked_fill(mask == 0, -1e9)
        d_attn = keys.shape[-1]
        scaled_scores = scores / math.sqrt(d_attn)
        softmax_scores = torch.softmax(scaled_scores, -1)
        softmax_scores = dropout(softmax_scores)
        v_out = torch.matmul(softmax_scores, values)
        return v_out

    def disable_subsequent_mask(self):
        self.maskStrategy = MaskStrategy["UNMASKED"]

    def enable_subsequent_mask(self):
        self.maskStrategy = MaskStrategy["MASKED"]
