from .sequence_pair_dataset import SequencePairDataset
from tokenizers import Tokenizer
import pickle
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


class PadCollate:
    TOKENIZER_SUFFIX = "_tokenizer"

    def __init__(self, src_filename, tgt_filename, src_vocab_size, tgt_vocab_size):
        self.src_tokenizer, self.tgt_tokenizer = self.setup_tokenizers(
            src_filename,
            tgt_filename,
            src_vocab_size,
            tgt_vocab_size,
            src_filename + self.TOKENIZER_SUFFIX,
            tgt_filename + self.TOKENIZER_SUFFIX,
        )

    def setup_tokenizers(
        self,
        src_filename,
        tgt_filename,
        src_vocab_size,
        tgt_vocab_size,
        src_tokenizer_name,
        tgt_tokenizer_name,
    ):
        print("creating tokenizer for " + src_filename)
        src_tokenizer = Tokenizer(BPE(unk_token=SequencePairDataset.UNK_TOKEN))
        src_tokenizer.pre_tokenizer = Whitespace()
        # src_tokenizer.post_processor = TemplateProcessing(
        #     single="[BOS] $A [EOS]",
        #     special_tokens=[("[BOS]", 0), ("[EOS]", 1)],
        # )
        trainer = BpeTrainer(
            vocab_size=src_vocab_size,
            special_tokens=[
                SequencePairDataset.BOS_TOKEN,
                SequencePairDataset.EOS_TOKEN,
                SequencePairDataset.PAD_TOKEN,
                SequencePairDataset.UNK_TOKEN,
            ],
        )
        src_tokenizer.train([src_filename], trainer=trainer)
        pickle.dump(src_tokenizer, open(src_tokenizer_name, "wb"))

        print("creating tokenizer for " + tgt_filename)
        tgt_tokenizer = Tokenizer(BPE(unk_token=SequencePairDataset.UNK_TOKEN))
        tgt_tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=tgt_vocab_size,
            special_tokens=[
                SequencePairDataset.BOS_TOKEN,
                SequencePairDataset.EOS_TOKEN,
                SequencePairDataset.PAD_TOKEN,
                SequencePairDataset.UNK_TOKEN,
            ],
        )
        tgt_tokenizer.train([tgt_filename], trainer=trainer)
        tgt_tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 0), ("[EOS]", 1)],
        )
        pickle.dump(tgt_tokenizer, open(tgt_tokenizer_name, "wb"))
        return src_tokenizer, tgt_tokenizer

    def __call__(self, batch):
        # max_len_src = max([len(pair[0].split()) for pair in batch])
        # max_len_tgt = max([len(pair[1].split()) for pair in batch])

        # tgt_sequence_lengths

        self.src_tokenizer.no_padding()
        self.tgt_tokenizer.no_padding()

        self.src_tokenizer.no_truncation()
        self.tgt_tokenizer.no_truncation()

        src_tokenized = self.src_tokenizer.encode_batch([pair[0] for pair in batch])
        tgt_tokenized = self.tgt_tokenizer.encode_batch([pair[1] for pair in batch])

        max_len_src = max([len(sequence) for sequence in src_tokenized])
        max_len_tgt = max([len(sequence) for sequence in tgt_tokenized])

        # print("max len src:", max_len_src)
        # print("max len tgt:", max_len_tgt)

        self.src_tokenizer.enable_padding(
            pad_id=SequencePairDataset.PAD_ID, pad_token=SequencePairDataset.PAD_TOKEN
        )
        self.src_tokenizer.enable_truncation(max_length=max_len_src)
        self.tgt_tokenizer.enable_padding(
            pad_id=SequencePairDataset.PAD_ID, pad_token=SequencePairDataset.PAD_TOKEN
        )
        self.tgt_tokenizer.enable_truncation(max_length=max_len_tgt)

        # print("src batch:", [pair[0] for pair in batch])
        # print("tgt batch:", [pair[1] for pair in batch])

        src_tokenized = self.src_tokenizer.encode_batch([pair[0] for pair in batch])
        tgt_tokenized = self.tgt_tokenizer.encode_batch([pair[1] for pair in batch])
        # src_tokenized = [sequence.ids for sequence in src_tokenized]
        # tgt_tokenized = [sequence.ids for sequence in tgt_tokenized]
        # src_tensors = torch.IntTensor(src_tokenized)
        # tgt_tensor = torch.IntTensor(tgt_tokenized)

        return src_tokenized, tgt_tokenized
