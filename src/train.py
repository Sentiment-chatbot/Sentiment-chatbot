import os
import os.path as p

import wandb
import numpy as np
import torch
import torch.nn as nn

from .utils.metric import perplexity_score
from .utils.generate import generate_with_user_input, generate_with_data_loader


def save_ckpt(ckpt_path, model, epoch, train_loss, best_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "train_loss" : train_loss,
        "best_loss" : best_loss
    }, ckpt_path)
    
def validate(model, valid_loader, device):
    """ Validate with teacher forcing """

    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        epoch_loss = 0.0
        for step, (_, input_ids, attention_ids, *_) in enumerate(valid_loader):
            # input_ids: (batched) <s> <sp1> {sentence1} <sp2> {sentence2} </s>
            # attention_ids: [1, 1, 1, 1, 1, ..., 0, 0]
            # Other(*_): emotion labels
            input_ids, attention_ids = input_ids.to(device), attention_ids.to(device) # (N, max_seq_len)
            labels = input_ids[:, 1:].contiguous() # labels: (N, seq_len - 1)

            logits = model(input_ids=input_ids, attention_ids=attention_ids) # (N, seq_len, vocab_size)
            logits = logits[:, :-1, :].contiguous() # (N, seq_len - 1, vocab_size)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
            epoch_loss += loss.item()

        valid_loss = epoch_loss / len(valid_loader)

    return valid_loss, perplexity_score(valid_loss)

def train(
    model,
    tokenizer,
    train_loader,
    valid_loader,
    n_epochs,
    gen_max_seq_len,
    gen_policy,
    gen_ex_input,
    device, 
    opt='adamw',
    learning_rate=3e-4,
    lr_scheduler='SGDR',
    ckpt_path='./ckpt',
    logging_step=300 
):
    """ Train with next word prediction (Teacher forcing)
    
    train_loader => (input_ids, attention_ids, emotion_1, emotion_2)
    - q_ids: input ids of only sp1, used for LSTM classifier (not batched)
    - input_ids: (batched) <s> <sp1> {sentence1} <sp2> {sentence2} </s>
    - attention_ids: [1, 1, 1, 1, 1, ..., 0, 0]
    - emotion_1, emotion_2: emotion labels
    
    In this train flow, we will only use x for simplicity. (This may be changed after)
    """ 

    # Make a directory to save the checkpoint weight(s)
    os.makedirs(ckpt_path, exist_ok=True)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = None
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    # TODO: Learning scheduler
    # scheduler = None
    # if scheduler == 'SGDR':
    #     scheduler = CustomCosineAnnealingWarmRestart(optimizer)

    losses = []
    train_loss = []
    best_pp = float('inf')
    
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for step, (_, input_ids, attention_ids, *_) in enumerate(train_loader):
            # input_ids: (batched) <s> <sp1> {sentence1} <sp2> {sentence2} </s>
            # attention_ids: [1, 1, 1, 1, 1, ..., 0, 0]
            # Other(*_): emotion labels
            input_ids, attention_ids = input_ids.to(device), attention_ids.to(device) # (N, max_seq_len)
            labels = input_ids[:, 1:].contiguous() # labels: (N, seq_len - 1)

            optimizer.zero_grad()

            logits = model(input_ids=input_ids, attention_ids=attention_ids) # (N, seq_len, vocab_size)
            logits = logits[:, :-1, :].contiguous() # (N, seq_len - 1, vocab_size)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            # Wandb logging
            wandb.log({
                "train_loss": epoch_loss / (step + 1),
                "train_pp": perplexity_score(epoch_loss / (step + 1))
            })

            if (step + 1) % logging_step == 0:
                print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {epoch_loss/(step + 1): .3f}")

        train_loss.append(epoch_loss / len(train_loader))
        valid_loss, valid_pp = validate(model, valid_loader, device)
    
        if valid_pp < best_pp:
            best_pp = valid_pp
            save_ckpt(
                ckpt_path=p.join(ckpt_path, f"checkpoint_epoch_{epoch + 1}.pt"), 
                model=model, epoch=epoch + 1, 
                train_loss=train_loss, best_loss=valid_loss
            )
            print(f"Success to save checkpoint. Best perplexity so far: {best_pp: .3f}")
        else:
            print("No improvement detected. Skipping save")

        print(f"[Epoch {epoch + 1}/{n_epochs}] Test generation")
        print(f"Input: {gen_ex_input}")
        response_sentence = generate_with_user_input(
            user_input=gen_ex_input,
            max_seq_len=gen_max_seq_len,
            model=model,
            tokenizer=tokenizer,
            gen_policy=gen_policy,
            device=device
        )
        print(f"Output: {response_sentence}")

        # Wandb logging
        wandb.log({
            "valid_loss": valid_loss,
            "valid_pp": valid_pp,
            "generated": f"Input: {gen_ex_input} / Output: {response_sentence}"
        })

def test(
    model,
    test_loader,
    tokenizer,
    gen_policy,
    device
):
    return generate_with_data_loader(
        model,
        test_loader,
        tokenizer,
        gen_policy,
        device
    )