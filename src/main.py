import torch
from torch.utils.data import DataLoader

from model.encoder_decoder_transformer import EncoderDecoderTransformer
from training.trainer import train_model
from dataset.train_and_validation_sequence_datasets import (
    TrainAndValidationSequenceDatasets,
)
from dataset.pad_collate import PadCollate
from datetime import datetime


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    num_encoder_layers = 4
    num_decoder_layers = 4
    num_heads = 8
    d_attn = 256
    d_x = 256
    d_z = 256
    d_out = 256
    d_mid = 256
    d_mlp = 512
    d_e = 256
    max_sequence_length = 100
    p_dropout = 0.1
    enRawName = "../multi30kEnTrain.txt"
    deRawName = "../multi30kDeTrain.txt"
    saveDirectory = "./"
    nameSuffix = ""
    state_dict_filename = (
        saveDirectory
        + "encoder_decoder_transformer_state_dict_"
        + datetime.today().strftime("%Y-%m-%d %H")
        + nameSuffix
    )
    tensor = torch.tensor([1, 2, 3])
    tensor.float()
    vocab_size = 10000
    train_and_validation_sequence_datasets = TrainAndValidationSequenceDatasets(
        enRawName, deRawName, vocab_size, vocab_size, 0, 28250, 28250, 29000
    )
    custom_encoder_decoder_transformer = EncoderDecoderTransformer(
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
    ).to(device)
    custom_encoder_decoder_transformer.src_embedding.table = custom_encoder_decoder_transformer.src_embedding.table.to(device)
    custom_encoder_decoder_transformer.tgt_embedding.table = custom_encoder_decoder_transformer.tgt_embedding.table.to(device)
    custom_encoder_decoder_transformer.positionalEmbedding.table = custom_encoder_decoder_transformer.positionalEmbedding.table.to(device)
    train_dataset = train_and_validation_sequence_datasets.train_dataset
    val_dataset = train_and_validation_sequence_datasets.val_dataset
    pad_collate = PadCollate(enRawName, deRawName, vocab_size, vocab_size)
    train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=128, collate_fn=pad_collate)
    train_model(
        custom_encoder_decoder_transformer,
        train_dataloader,
        val_dataloader,
        pad_collate.src_tokenizer,
        pad_collate.tgt_tokenizer,
        device,
        state_dict_filename
    )


if __name__ == "__main__":
    main()
