from __future__ import annotations
import math
from typing import Callable

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


class DecoderLayer(nn.Module):
    """A modified clone of torch.nn.TransformerDecoderLayer with the memory parameter turned optional
    in the forward method. This also omits the computations of memory multi-head attentions making this
    purely a decoder ONLY layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor, Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = nn._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """This method is usually not called directly but by the __call__ method

        Parameters
        ----------
        tgt:
            The output from the embedding (possibly with positional encoding).
        memory: Any
            Omitted
        tgt_mask: Tensor[torch.bool]
            Boolean mask with True denoting which values are hidden from the SA block.
        memory_mask: Any
            Omitted
        """

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class GPTModel(nn.Module):
    """Generalized GPT model that operates in a decoder-only sequence. For the neural network
    infrastructure, this depends on pytorch engine, preferably with CUDA in use."""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        dropout: float,
        device: str | None = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.positional_enc = PositionalEncoding(
            embed_size=embed_size, max_seq_len=block_size
        )
        decoder = DecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            device=device,
        )
        self.blocks = nn.TransformerDecoder(
            decoder_layer=decoder, num_layers=num_layers
        )

        self.register_buffer("mask", self._mask_future_tokens(block_size).bool())

        self.fc = nn.Linear(embed_size, vocab_size)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Returns number of parameters in the model

        Parameters
        ----------
        trainable_only: bool, optional
            Only use trainable parameters. True by default

        Returns
        -------
        int
        """
        if trainable_only:
            gen = (param.numel() for param in self.parameters() if param.requires_grad)
        else:
            gen = (param.numel() for param in self.parameters())

        return sum(gen)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        # Get input sequence length
        embeds = self.embeddings(input_ids)
        embeds = self.positional_enc(embeds)
        if self.training:
            blocks = self.blocks(embeds, memory=None, tgt_mask=self.mask)
        else:
            blocks = self.blocks(embeds, memory=None)
        logits = self.fc(blocks)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @staticmethod
    def _mask_future_tokens(tgt_seq_len: int) -> Tensor:
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
        future_mask = torch.ones((tgt_seq_len, tgt_seq_len))
        future_mask = torch.triu(future_mask, diagonal=1)
        return future_mask

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate tokens from a starting up to a maximum number of tokens

        Parameters
        ----------
        input_ids: Tensor
            The input tensor of shape (0, seq_len) of integer (long)
            encoded words.

        max_new_tokens: int
            Maximum number of tokens to generate.

        Returns
        -------
        Tensor
            A longer tensor of size (0, seq_len + max_new_tokens)
        """
        self.eval()
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.block_size :]
            logits, _ = self(ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, sample), dim=-1)

        return input_ids
