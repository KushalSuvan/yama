import torch
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

from transformer_.config import get_config, get_weights_file_path
from transformer_.dataset import BilingualDataset
from transformer_.model import transformer_factory, Transformer


def get_or_build_tokenizer(config, ds_raw, lang) -> Tokenizer:
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds_raw, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_all_sentences(ds, lang):
    for text_pair in ds:
        text = text_pair['translation'][lang]
        yield text

def get_ds(config):
    """
    
    """
    ds_raw = load_dataset(config['corpus'], f'{config["lang_src"]}-{config["lang_tgt"]}', split="train")

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Get train-test split in raito 9:1
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, (train_ds_size, val_ds_size))

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max tokenized length of source sentences: {max_len_src}')
    print(f'Max tokenized length of target sentences: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size) -> Transformer:
    model = transformer_factory(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
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

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Setup model

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


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
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            proj_output = model.project(decoder_output)     # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # Calculate loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
            # Log the loss to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate
            loss.backward()

            # Optimize a step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Increment the global step (number of optimization steps)
            global_step += 1

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
