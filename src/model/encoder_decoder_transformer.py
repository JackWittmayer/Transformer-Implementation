from .decoder import Decoder
from .encoder import Encoder
from .positional_embedding import PositionalEmbedding
from .embedding import Embedding
from .unembedding import Unembedding
from torch import nn


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        num_heads,
        d_attn,
        d_x,
        d_z,
        d_out,
        d_mid,
        d_mlp,
        d_e,
        vocab_size,
        max_sequence_length,
        p_dropout,
        device
    ):
        super().__init__()
        self.src_embedding = Embedding(vocab_size, d_e)
        self.tgt_embedding = Embedding(vocab_size, d_e)
        self.unembedding = Unembedding(vocab_size, d_e)
        self.embedding_dropout = nn.Dropout(p_dropout)
        self.positionalEmbedding = PositionalEmbedding(d_e, max_sequence_length, device)
        self.encoder = Encoder(
            num_encoder_layers,
            num_heads,
            d_attn,
            d_x,
            d_z,
            d_out,
            d_mid,
            d_mlp,
            p_dropout
        )
        self.decoder = Decoder(
            num_decoder_layers,
            num_heads,
            d_attn,
            d_x,
            d_z,
            d_out,
            d_mid,
            d_mlp,
            p_dropout
        )

    def forward(self, z, x, src_mask, tgt_mask):
        z = self.src_embedding(z) + self.positionalEmbedding(z)
        z = self.embedding_dropout(z)
        z = self.encoder(z, src_mask)
        x = self.tgt_embedding(x) + self.positionalEmbedding(x)
        x = self.embedding_dropout(x)
        x = self.decoder(z, x, src_mask, tgt_mask)
        x = self.unembedding(x)
        return x

    def disable_subsequent_mask(self):
        self.decoder.disable_subsequent_mask()
