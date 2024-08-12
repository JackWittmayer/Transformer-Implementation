import torch
from model.positional_embedding import PositionalEmbedding


def test_positional_embedding():
    SAMPLE_X = torch.tensor([[3, 2, 0, 1], [1, 2, 3, 0]], dtype=torch.int32)
    embedding_size = 8
    max_sequence_length = 10
    batch_size = 2
    positional_embedding = PositionalEmbedding(embedding_size, max_sequence_length, torch.device("cpu"))
    output = positional_embedding(SAMPLE_X)
    print("output:", output)
    assert output.shape == (batch_size, SAMPLE_X.shape[-1], embedding_size)


test_positional_embedding()
