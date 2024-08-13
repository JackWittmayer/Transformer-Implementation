from .sequence_pair_dataset import SequencePairDataset


class TrainAndValidationSequenceDatasets:
    def __init__(
        self,
        src_filename,
        tgt_filename,
        train_start_index,
        train_end_index,
        val_start_index,
        val_end_index,
    ):
        src_text = self.load_doc(src_filename)
        tgt_text = self.load_doc(tgt_filename)
        self.train_dataset = SequencePairDataset(
            src_text, tgt_text, train_start_index, train_end_index
        )
        self.val_dataset = SequencePairDataset(
            src_text, tgt_text, val_start_index, val_end_index
        )

    def load_doc(self, filename):
        file = open(filename, mode="rt")
        text = file.read()
        file.close()
        return text
