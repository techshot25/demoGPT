from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import Vocab, build_vocab_from_iterator

from gptmodel.model import GPTModel


class SequenceDataset(Dataset):
    """A Dataset subclass for handling shifted sequence data"""

    def __init__(
        self, data_iter: Iterable, vocab: Vocab, tokenizer: Callable, block_size: int
    ):
        self.data_iter = data_iter
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.tokens = self.build_tokens()
        self.num_blocks = len(self.tokens) // block_size

    def build_tokens(self) -> Tensor:
        return self.transform_text("\n".join(self.data_iter))

    def transform_text(self, data: str) -> Tensor:
        return torch.tensor([self.vocab[token] for token in self.tokenizer(data)])

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx + 1) * self.block_size
        input_block = self.tokens[start_idx:end_idx]
        target_block = self.tokens[
            start_idx + 1:end_idx + 1
        ]  # shift target block by one
        return input_block, target_block


def get_datasets(
    tokenizer_name: str, block_size: int
) -> Tuple[SequenceDataset, SequenceDataset, Vocab]:
    tokenizer = get_tokenizer(tokenizer_name)
    train_iter, val_iter, vocab = get_text_data(tokenizer)
    train_dataset = SequenceDataset(train_iter, vocab, tokenizer, block_size)
    val_dataset = SequenceDataset(val_iter, vocab, tokenizer, block_size)
    return train_dataset, val_dataset, vocab


def get_train_val_iterators() -> Tuple[Iterable]:
    """Gets the iterators to build the data"""
    train_iter = WikiText2(root="data", split="train")
    val_iter = WikiText2(root="data", split="valid")
    return train_iter, val_iter


def yield_tokens(tokenizer: Callable, data_iter):
    """Should be modified if the iterator yields multiple outputs"""
    for text in data_iter:
        yield tokenizer(text)


def get_text_data(tokenizer: Callable):
    train_iter, val_iter = get_train_val_iterators()
    vocab = build_vocab_from_iterator(
        yield_tokens(tokenizer, train_iter),
        min_freq=3,
        specials=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    vocab.set_default_index(vocab["<unk>"])
    return train_iter, val_iter, vocab


def encode(inputs: str, vocab: Vocab) -> torch.LongTensor:
    """Encodes an input string into a LongTensor"""
    seq = inputs.split()
    res = vocab.lookup_indices(seq)
    return torch.tensor(res, dtype=torch.long)


def decode(inputs: torch.LongTensor, vocab: Vocab) -> str:
    """Decodes the output back into text"""
    seq = inputs.cpu().detach().tolist()
    res = vocab.lookup_tokens(seq)
    return " ".join(res)


def sample_tokens(
    model: GPTModel, vocab: Vocab, device: str, prompt: str, max_new_tokens: int
):
    encoded = encode(prompt).to(device)
    tokens = model.generate(encoded, max_new_tokens=max_new_tokens)
    return decode(tokens, vocab)
