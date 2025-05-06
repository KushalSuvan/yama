from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from transformer_ import Encoder, EncoderBlock, MultiHeadAttention, InputEmbedding, ProjectionLayer, PositionalEncoding, FeedForward
        

class ExtractiveHead(nn.Module, ABC):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    

class NaiveExtractiveHead(ExtractiveHead):

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    

class DeepExtractiveHead(ExtractiveHead):

    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    
class ExtractiveSummarizer(nn.Module):

    def __init__(self, encoder: Encoder, embedding: InputEmbedding, pos: PositionalEncoding, score_head: ExtractiveHead):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.pos = pos

        self.score_head = score_head

    def encode(self, x: torch.Tensor):
        input_ids = x['input_ids']      # B (1), num_of_sentences, seq_len
        masks = x['attention_mask']     # B (1), num_of_sentences, seq_len

        B, N, S = input_ids.shape

        input_ids = input_ids.view(B*N, S).contiguous()     # B (1) * num_of_sentences, seq_len
        masks = masks.view(B*N, S).contiguous().unsqueeze(1).unsqueeze(1)  # B (1) * num_of_sentences, 1, 1, seq_len
        # Note that the reason for the two extra dimensions in between lies in MHA mechanism. Mask is used directly there without any changes.
        # So this change might seem unreasonable, because the reason lies at a bit lower level
        # Alternatively we can make this change in MHA class, but I do not want to fiddle around too much...gotta submit the assignment on time

        embed = self.embedding(input_ids)   # B (1) * num_of_sentences, seq_len, d_model
        pos_embed = self.pos(embed)         # B (1) * num_of_sentences, seq_len, d_model
        enc_output = self.encoder(pos_embed, masks)    # B (1) * num_of_sentences, seq_len, d_model

        enc_output = enc_output.view(B, N, S, -1).contiguous() # B (1), num_of_sentences, seq_len, d_model

        cls_enc_output = enc_output[:, :, 0, :] # B (1), num_of_sentences, d_model
        return cls_enc_output
    
    def score_sentences(self, encoder_output):
        return self.score_head(encoder_output)  # B (1), num_of_sentences, 1


    def forward(self, x):
        encoder_output = self.encode(x)         # B, num_sentences, d_model
        sentence_scores = self.score_sentences(encoder_output)  # B, num_sentences, 1
        return sentence_scores.squeeze(-1)      # B, num_sentences


def get_extractive_summarizer(d_model: int, h: int, N: int, d_ff: int, vocab_size: int, seq_len: int, complexity: str, dropout: float=0.1):
    # encoder = encoder_factory(N, d_model, h, d_ff, dropout)

    src_embed = InputEmbedding(vocab_size, d_model)
    # tgt_embed = InputEmbedding(vocab_size, d_model)
    # projection = ProjectionLayer(d_model, vocab_size)

    src_pos = PositionalEncoding(d_model, seq_len, dropout)
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

    score_head = None

    match complexity:
        case "naive":
            score_head = NaiveExtractiveHead(d_model)
        case "deep":
            score_head = DeepExtractiveHead(d_model)
        case "attentive":
            raise ValueError("Attentive Head is not implemented yet")
        
    esummarizer = ExtractiveSummarizer(encoder, src_embed, src_pos, score_head)

    for p in esummarizer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return esummarizer