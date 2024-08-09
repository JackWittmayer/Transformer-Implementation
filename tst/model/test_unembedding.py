import torch
from src.model.unembedding import Unembedding


def test_unembedding():
    torch.manual_seed(25)
    vocab_size = 10
    embedding_size = 4
    sequence_length = 4
    batch_size = 2
    input = torch.rand(batch_size, sequence_length, embedding_size)
    unembedding = Unembedding(vocab_size, embedding_size)

    print("weight:", unembedding.weight)
    print("input: ", input)
    output = unembedding(input)
    print("output:", output)
    assert output.shape == (batch_size, sequence_length, vocab_size)


test_unembedding()
