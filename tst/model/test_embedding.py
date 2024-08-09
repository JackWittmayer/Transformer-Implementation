from src.model.embedding import Embedding
import torch


def test_embedding():
    SAMPLE_X = torch.tensor([[3, 2, 0, 1], [1, 2, 3, 0]], dtype=torch.int32)
    torch.manual_seed(25)
    vocab_size = 4
    embedding = Embedding(vocab_size, 4)
    print("weight:", embedding.table.weight)
    print("SAMPLE_X: ", SAMPLE_X)
    output = embedding(SAMPLE_X)
    print("output:", output)
    for j in range(len(output)):
        for i in range(vocab_size):
            assert output[j, i, :].eq(embedding.table.weight[SAMPLE_X[j, i]]).all()
