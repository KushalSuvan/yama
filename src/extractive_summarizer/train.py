import torch
from extractive_summarizer import ExtractiveSummarizationDataset, get_config
from extractive_summarizer import ExtractiveSummarizer

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from tqdm import tqdm

import warnings

import json

from extractive_summarizer.model import get_extractive_summarizer
from extractive_summarizer.config import get_config, get_weights_file_path

from transformers import AutoTokenizer, PreTrainedTokenizer

import gc

torch.cuda.empty_cache()
gc.collect()

def get_ds(config):
    raw_ds = None

    with open(config['datasource'], 'r') as f:
        raw_ds = json.load(f)

    assert raw_ds is not None, f"ERROR: Failed to load datasource at {config['datasource']}"

    dataloader = DataLoader(ExtractiveSummarizationDataset(raw_ds), batch_size=config['batch_size'])

    return dataloader



def get_all_sentences(ds):
    for pair in ds:
        text = pair['judgement']
        yield text


def get_or_build_tokenizer(config, ds_raw) -> PreTrainedTokenizer:
    # tokenizer_path = Path(config['tokenizer_file'])
    # if not Path.exists(tokenizer_path):
    #     tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    #     tokenizer.pre_tokenizer = Whitespace()
    #     trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
    #     tokenizer.train_from_iterator(get_all_sentences(ds_raw), trainer=trainer)
    #     tokenizer.save(str(tokenizer_path))
    # else:
    #     tokenizer = Tokenizer.from_file(str(tokenizer_path))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.model_max_length = config['seq_len']

    return tokenizer



def get_ds(config):
    """
    
    """
    # ds_raw = load_dataset(config['corpus'], split="train")['translation']

    ds_raw = None

    with open(config['datasource'], 'r') as f:
        ds_raw = json.load(f)

    assert ds_raw is not None, f"ERROR: Failed to load datasource at {config['datasource']}"

    # Build tokenizer
    tokenizer = get_or_build_tokenizer(config, ds_raw)
    # tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Get train-test split in raito 9:1
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, (train_ds_size, val_ds_size))

    train_ds = ExtractiveSummarizationDataset(ds_raw, tokenizer)
    # val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # max_len_src = 0
    # max_len_tgt = 0

    # for item in ds_raw:
    #     # src_ids = tokenizer.encode(item['judgement']).ids
    #     src_ids = tokenizer(item['judgement'])['input_ids']
    #     # tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

    #     max_len_src = max(max_len_src, len(src_ids))
    #     # max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f'Max tokenized length of sentences: {max_len_src}')
    # # print(f'Max tokenized length of target sentences: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)


    return train_dataloader, tokenizer


def get_model(config, vocab_size) -> ExtractiveSummarizer:
    model = get_extractive_summarizer(config['d_model'], config['h'], config['N'], config['d_ff'], vocab_size, config['seq_len'])
    return model


def train_model(config):
    # Setup device

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Deivce memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GiB")
    elif (device == "mps"):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")

    device = torch.device(device)

    # Setup dataset/dataloader

    train_dataloader, tokenizer = get_ds(config)

    # Setup model

    # model = get_model(config, tokenizer.get_vocab_size()).to(device)
    model = get_model(config, tokenizer.vocab_size).to(device)

    # Setup optimizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)

    # Preload model state if configured to do so

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model: {model_filename}')

        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # Create the loss function as cross entropy loss. Ignore [PAD] tokens as they are not meaningful and we do not want to adapt to them
    # the label smoothing is "being unsure" even when correct prediction is made. This helps the training loss manifold smooth a little
    # ----------------------------------------------------------------------------------------------------
    # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    loss_fn = nn.BCEWithLogitsLoss().to(device)


    # Get the tensor board ready
    writer = SummaryWriter()

    # Create model_folder to save states in
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Training loop

    for epoch in range(initial_epoch, config['num_epochs']):
        # Clear torch.cuda cache. Learn what it means and does
        torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')

        # Always check whether the tensor I am working with is in GPU memory
        # If it needs to be there and is not there by something like model.to(device), push it to device
        for batch in batch_iterator:

            # # Run the tensors through the transformer
            # encoder_output = model.encode(encoder_input, encoder_mask)
            # decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            # proj_output = model.project(decoder_output)     # (B, seq_len, tgt_vocab_size)

            # label = batch['label'].to(device) # (B, seq_len)

            # # Calculate loss
            # loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
            # # Log the loss to tensorboard
            # writer.add_scalar('train loss', loss.item(), global_step)
            # writer.flush()

            # # Backpropagate
            # loss.backward()

            # # Optimize a step
            # optimizer.step()
            # optimizer.zero_grad(set_to_none=True)

            # # Increment the global step (number of optimization steps)
            # global_step += 1

            # if (global_step % 2500 == 0):
            #     model_filename = get_weights_file_path(config, f'{epoch:02d}-{global_step:06d}')
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'global_step': global_step
            #     }, model_filename)

        # At end of each epoch, save state to disk
        

        # model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # torch.save({
        #     "model_state_dict": model.state_dict(),
        #     "epoch": epoch,
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "global_step": global_step,
        # }, model_filename)
            torch.cuda.empty_cache()
            
            judgement_tokens = batch['jtokens'].to(device)
            logits = model(judgement_tokens)


            target = batch['target'].to(device)

            loss = loss_fn(logits, target)

            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
            # Log the loss to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=None)

            # Increment the global step (number of optimization steps)
            global_step += 1

            if (global_step % 2500 == 0):
                model_filename = get_weights_file_path(config, f'{epoch:02d}-{global_step:06d}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }, model_filename)

        # At end of each epoch, save state to disk
        

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)
            



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)

