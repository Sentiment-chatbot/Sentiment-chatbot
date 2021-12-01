import os
import os.path as p

import numpy as np
import torch
import torch.nn as nn

from .utils.generate import generate


def save_ckpt(ckpt_path, model, epoch, train_loss, best_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "train_loss" : train_loss,
        "best_loss" : best_loss
    }, ckpt_path)
    
def validate(model, valid_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        epoch_loss = 0.0
        for step, (input_ids, attention_ids, *_) in enumerate(valid_loader):
            # input_ids: (batched) <s> <sp1> {sentence1} <sp2> {sentence2} </s>
            # attention_ids: [1, 1, 1, 1, 1, ..., 0, 0]
            # Other(*_): emotion labels
            input_ids, attention_ids = input_ids.to(device), attention_ids.to(device) # (N, max_seq_len)
            labels = input_ids[:, 1:].contiguous() # labels: (N, seq_len - 1)

            logits = model(input_ids=input_ids, attention_ids=attention_ids) # (N, seq_len, vocab_size)
            logits = logits[:, :-1, :].contiguous() # (N, seq_len - 1, vocab_size)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
            epoch_loss += loss.item()

        total_loss = epoch_loss / len(valid_loader)

    return total_loss

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
    """ Train with next word prediction
    
    train_loader => (input_ids, attention_ids, emotion_1, emotion_2)
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
    best_loss = float('inf')
    
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for step, (input_ids, attention_ids, *_) in enumerate(train_loader):
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
            if (step + 1) % logging_step == 0:
                print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {epoch_loss/(step + 1): .3f}")
        
        train_loss.append(epoch_loss / len(train_loader))
        valid_loss = validate(model, valid_loader, device)

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_ckpt(
                ckpt_path=p.join(ckpt_path, f"checkpoint_epoch_{epoch + 1}.pt"), 
                model=model, epoch=epoch + 1, 
                train_loss=train_loss, best_loss=best_loss
            )
            print(f"Success to save checkpoint. Best loss so far: {best_loss: .3f}")

        print(f"[Epoch {epoch + 1}/{n_epochs}] Test generation")
        print(f"Input: {gen_ex_input}")
        response_sentence = generate(
            user_input=gen_ex_input,
            max_seq_len=gen_max_seq_len,
            model=model,
            tokenizer=tokenizer,
            gen_policy=gen_policy,
            device=device
        )
        print(f"Output: {response_sentence}")
