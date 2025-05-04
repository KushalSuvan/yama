import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_ import encoder_factory
from transformer_ import Encoder, EncoderBlock, MultiHeadAttention, InputEmbedding, ProjectionLayer, PositionalEncoding, FeedForward
        

class ExtractiveHead(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, 1)
        
    def forward(self, x):
        return F.sigmoid(self.linear1(x))
    
class ExtractiveSummarizer(nn.Module):

    def __init__(self, encoder: Encoder, embed: InputEmbedding, pos: PositionalEncoding):
        super().__init__()
        self.encoder = encoder
        self.embed = embed
        self.pos = pos

    def forward(self, x):
        pass


def get_extractive_summarizer(d_model: int, h: int, N: int, d_ff: int, vocab_size: int, src_seq_len: int, dropout: float=0.1):
    encoder = encoder_factory(N, d_model, h, d_ff, dropout)

    src_embed = InputEmbedding(vocab_size, d_model)
    # tgt_embed = InputEmbedding(vocab_size, d_model)
    # projection = ProjectionLayer(d_model, vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    # tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_layer = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_layer, d_model, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # decoder_blocks = []
    # for _ in range(N):
    #     decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    #     decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
    #     feed_forward_layer = FeedForward(d_model, d_ff, dropout)

    #     decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_layer, d_model, dropout)
    #     decoder_blocks.append(decoder_block)
    
    # decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection)

    esummarizer = ExtractiveSummarizer(encoder, src_embed, src_pos)

    for p in esummarizer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return esummarizer