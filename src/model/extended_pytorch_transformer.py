import torch
import torch.nn as nn

from embedding import Embedding
from unembedding import Unembedding
from positional_embedding import PositionalEmbedding


# For some reason the Pytorch transformer doesn't have its own embedding layers. Adding them here.
class ExtendedPytorchTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_mlp,
        p_dropout,
        vocab_size,
        max_sequence_length,
        batch_first=True,
        norm_first=True,
    ):
        super().__init__()
        self.src_embedding = Embedding(vocab_size, d_model)
        self.tgt_embedding = Embedding(vocab_size, d_model)
        self.unembedding = Unembedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            d_mlp,
            p_dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.embedding_dropout = nn.Dropout(p_dropout)
        self.positionalEmbedding = PositionalEmbedding(d_model, max_sequence_length)

    def forward(self, src_sequence, tgt_sequence, src_mask, tgt_key_padding_mask, device):
        src_sequence = self.src_embedding(src_sequence) + self.positionalEmbedding(
            src_sequence
        )
        src_sequence = self.embedding_dropout(src_sequence)
        tgt_sequence = self.tgt_embedding(tgt_sequence) + self.positionalEmbedding(
            tgt_sequence
        )
        tgt_sequence = self.embedding_dropout(tgt_sequence)
        tgt_mask = self.get_tgt_mask(tgt_sequence.shape[1], tgt_sequence.shape[1], device)
        src_mask = ~src_mask.bool()
        tgt_mask = ~tgt_mask.bool()
        tgt_key_padding_mask = ~tgt_key_padding_mask.bool()
        transformer_out = self.transformer(
            src_sequence,
            tgt_sequence,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        return self.unembedding(transformer_out)

    def get_tgt_mask(self, length_x, length_z, device):
        mask = torch.tril(torch.ones(length_x, length_z) == 1).to(device)
        return mask
