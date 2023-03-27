import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_len: int):
        super().__init__()

        # Compute the positional encoding once in log space
        pe = torch.zeros(max_seq_len, embed_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        x = x + self.pe[:, : x.size(1)]
        return x


class Attention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.multi_head = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            bias=False,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        key = self.keys(x)
        query = self.queries(x)
        value = self.values(x)
        mask = mask_future_tokens(x.shape[1]).to(x.device)
        output, *_ = self.multi_head(
            query=query, value=value, key=key, attn_mask=mask, need_weights=False
        )
        return output


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        block_size: int,
        dropout: float,
        *args
    ):
        super().__init__()
        self.block_size = block_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.positional_enc = PositionalEncoding(
            embed_size=embed_size, max_seq_len=block_size
        )
        self.blocks = Attention(
            embed_size=embed_size, num_heads=num_heads, dropout=dropout
        )
        # self.blocks = nn.ModuleList([attention for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_size), nn.Linear(embed_size, vocab_size)
        )

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        # Get input sequence length
        embeds = self.embeddings(input_ids)
        embeds = self.positional_enc(embeds)
        blocks = self.blocks(embeds)
        logits = self.fc(blocks)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.block_size:]
            logits, _ = self(ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, sample), dim=-1)

        return input_ids


@torch.jit.script
def mask_future_tokens(tgt_seq_len: int) -> Tensor:
    """
    Generate a mask to hide future tokens in the target sequence.
    Parameters
    ----------
    embed_dim: int
        Size of the batch
    tgt_seq_len: int
        Length of the target sequence

    Returns
    -------
    Tensor
        A tensor of shape (embed_dim, tgt_seq_len, tgt_seq_len) with zeros in
        the lower triangle and ones elsewhere
    """
    # Create a square matrix of shape (tgt_seq_len, tgt_seq_len)
    future_mask = torch.ones((tgt_seq_len, tgt_seq_len))
    # Set the upper triangle of the matrix to 0
    future_mask = torch.tril(future_mask, diagonal=-1)
    return future_mask
