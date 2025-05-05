import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
 

class InputEmbedding(nn.Module):
    """Embedding layer that scales token embeddings for transformer models.

    This layer maps token indices to dense vectors using `nn.Embedding` and
    scales them by the square root of `d_model`, following the approach from
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        d_model (int): Dimensionality of the dense vector embeddings.
        embedding (nn.Embedding): Embedding layer that converts token indices to vectors.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initialize the InputEmbedding layer.

        Args:
            vocab_size (int): Size of the input vocabulary.
            d_model (int): Dimensionality of the dense vector embeddings.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input token indices and scale by sqrt(d_model).

        Args:
            x (torch.Tensor): Tensor of token indices with shape (batch_size, seq_length).
        
        Returns:
            torch.Tensor: Scaled embedding vectors of shape 
                (batch_size, seq_length, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)




class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to token embeddings.

    Implements the fixed positional encoding from "Attention Is All You Need"
    (Vaswani et al., 2017). Encodes position information into the embedding
    space using a mix of sine and cosine functions.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied after adding positional encoding.
        pe (torch.Tensor): Positional encoding matrix of shape (1, seq_len, d_model).
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """Initialize the PositionalEncoding module.

        Args:
            d_model (int): Dimensionality of embeddings.
            seq_len (int): Maximum sequence length expected.
            dropout (float): Dropout probability applied after adding positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros((seq_len, d_model), dtype=torch.float)

        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        i_2 = torch.arange(0, d_model, 2, dtype=torch.float)  # (d_model // 2)
        div_term = torch.pow(10000, i_2 / d_model)  # (d_model // 2)

        pe[:, 0::2] = torch.sin(positions / div_term)
        pe[:, 1::2] = torch.cos(positions / div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Positionally encoded tensor of shape (batch_size, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # We need a vector of size (features), which is trainable.
        # So we wrap it in nn.Parameter
        self.alpha = nn.Parameter(torch.ones(features)) 
        self.bias = nn.Parameter(torch.zeros(features))


    def forward(self, x) -> torch.Tensor:
        # x := (batch, seq_len, d_model)
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        # Note that std could always be 0, so ensure division by 0 error does not happen
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    """Residual connection for Transformers, as described in 
    "Attention is all you need" (Vaswani et al. 2017). 

    Along with adding the input to the sublayer output, it also 
    adds LayerNormalization. This extra step is in compliance 
    with how the original Transformers was designed.

    Attributes:
        dropout (nn.Dropout): Dropout layer to prevent overfitting
        norm (LayerNormalization): Layer normalization applied after residual connection
    """
    def __init__(self, features: int, dropout: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer) -> torch.Tensor:
        # There are a few possibilites to this implementation
        # First, self.norm(self.dropout(x + sublayer(x)))
        # Second, x + self.norm(self.dropout(sublayer(x)))
        # Third, x + self.dropout(sublayer(self.norm(x)))
        # I get that the last option sort of prepares the input to be easily trainable by the sublayer, in this case attention head.
        # But still this design choice remains a mystery
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, dropout: nn.Dropout):
        # (batch, h, seq_len, d_k)
        d_k = q.size(-1)
        attention_scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k)
        return (attention_scores @ v), attention_scores 

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, -1)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, features: int, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x.to(next(self.parameters()).device)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        x.to(next(self.norm.parameters()).device)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, features: int, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        

    def encode(self, x, src_mask):
        x = self.src_embed(x)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def transformer_factory(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, d_ff: int = 2048, N: int = 6, h: int = 8, dropout: float = 0.1, tied_embeddings: bool= True) -> Transformer:

    # src_embed = None
    # tgt_embed = None
    # projection = None

    # if tied_embeddings == True:
    #     embedding = InputEmbedding(src_vocab_size, d_model)
    #     src_embed = embedding
    #     tgt_embed = embedding
    #     projection = lambda x: x @ embedding.embedding.weight.T
    # else:
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_layer = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_layer, d_model, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_layer = FeedForward(d_model, d_ff, dropout)

        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_layer, d_model, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
