import torch
from extractive_summarizer import ExtractiveSummarizationDataset, get_config
from extractive_summarizer import ExtractiveSummarizer

from torch.utils.data import DataLoader

import json


def get_ds(config):
    raw_ds = None

    with open(config['datasource'], 'r') as f:
        raw_ds = json.load(f)

    assert raw_ds is not None, f"ERROR: Failed to load datasource at {config['datasource']}"

    dataloader = DataLoader(ExtractiveSummarizationDataset(raw_ds), batch_size=config['batch_size'])

    return dataloader


def get_model(config, src_vocab_size, tgt_vocab_size) -> ExtractiveSummarizer:
    model = ExtractiveSummarizer(config['d_model'], config['h'], config['N'], config['d_ff'])
    return model


def train_model(config):
    dataloader = get_ds(config)

    # Send data pair to device
    for i in range(1):
        iterator = iter(dataloader)
        n = next(iterator)

    

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



if __name__=="__main__":
    config = get_config()
    train_model(config)

