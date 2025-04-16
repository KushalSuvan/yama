from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_.dataset import causal_mask
from transformer_.model import Transformer

def greedy_decode(model: Transformer, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id('[SOS]')
    eos_idx = tokenizer_src.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill(sos_idx).type_as(source).to(device)

    # NOTE: Finding which token to predict next is convoluted but correct. Take your time to understand it
    while True:
        if decoder_input.size(1) == max_len:
            break

        target_mask = causal_mask(decoder_input.size(0)).type_as(source_mask).to(device)
        out = model.decode(decoder_input, encoder_output, source_mask, target_mask)     # (B, seq_len, tgt_vocab_size)
        projection = model.project(out)         # (B, seq_len, tgt_vocab_size)

        # ASIDE NOTE: We can safely ignore F.softmax as we will be using max
        # Ignoring it is preferred than what we have done here
        # This will save compute time
        prob = F.softmax(out[:, -1], dim=-1)

        # Greedily choose the index with max probability
        value, index = torch.max(prob, dim=-1)

        # Add this to the decoder input for further decoding
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).fill(index.item()).type_as(decoder_input).to(device)
        ], dim=1) 

        if index == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model: Transformer, val_dataloader: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt:Tokenizer, max_len, device, print_msg, num_of_examples = 2):
    # Go into validation mode
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1

            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation dataloader"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print_msg('-' * 80)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_of_examples:
                break


