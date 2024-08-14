from torch.utils.data import Dataset


class SequencePairDataset(Dataset):
    BOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    PAD_ID = 2

    def __init__(self, src_text, tgt_text, start_index, end_index):
        src_sequences = self.to_sequences(src_text, start_index, end_index)
        tgt_sequences = self.to_sequences(tgt_text, start_index, end_index)
        self.pairs = self.pair_sequences(src_sequences, tgt_sequences)

    def pair_sequences(self, src_sequences, tgt_sequences):
        paired_sequences = list(zip(src_sequences, tgt_sequences))
        sorted_pairs = sorted(paired_sequences, key=lambda x: len(x[0]))
        return sorted_pairs

    # split a loaded document into sequences
    def to_sequences(self, doc, sequence_start_index, sequence_end_index):
        sequences = doc.strip().split("\n")
        return sequences[sequence_start_index:sequence_end_index]

    def add_special_tokens(self, sequence):
        sequence = self.BOS_TOKEN + " " + sequence + " " + self.EOS_TOKEN
        return sequence

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src_seq, tgt_seq = self.pairs[index]
        return src_seq, tgt_seq
