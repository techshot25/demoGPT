import torch
from torch import Tensor
from torch import nn


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Transformer Decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            embed_size, num_heads, hidden_size, dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Final linear layer
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids, memory=None):
        # Get input sequence length
        batch_size, seq_len = input_ids.shape

        # Embedding layer
        embed = self.embedding(input_ids)

        # Transformer Decoder layer
        if memory is None:
            # If no memory is given, use input sequence as memory
            memory = embed

        # Generate mask
        mask = mask_future_tokens(batch_size=batch_size, tgt_seq_len=embed.shape[-1])

        # Apply transformer to embeddings
        output = self.transformer_decoder(tgt=embed, memory=memory, tgt_mask=mask)

        # Reshape output to (batch_size * seq_len, hidden_size)
        output = output.reshape(-1, output.shape[2])

        # Final linear layer
        logits = self.fc(output)

        # Reshape logits back to (batch_size, seq_len, vocab_size)
        logits = logits.reshape(-1, seq_len, logits.shape[1])

        return logits


@torch.jit.script
def mask_future_tokens(batch_size: int, tgt_seq_len: int) -> Tensor:
    """
    Generate a mask to hide future tokens in the target sequence.
    Parameters
    ----------
    batch_size: int
        Size of the batch
    tgt_seq_len: int
        Length of the target sequence

    Returns
    -------
    Tensor
        A tensor of shape (batch_size, tgt_seq_len, tgt_seq_len) with zeros in
        the lower triangle and ones elsewhere
    """
    # Create a square matrix of shape (tgt_seq_len, tgt_seq_len)
    future_mask = torch.ones((tgt_seq_len, tgt_seq_len))
    # Set the upper triangle of the matrix to 0
    future_mask = torch.tril(future_mask, diagonal=-1)
    # Create a batched mask by unsqueezing the matrix along the batch dimension
    future_mask = future_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return future_mask
