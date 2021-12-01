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

    losses = []
    with torch.no_grad():
        epoch_loss = []
        for step, inputs in enumerate(valid_loader):
            x, _, _ = inputs # x: (N, max_seq_len)
            y = x[:, 1:].contiguous()
            x, y = x.to(device), y.to(device)
            # x: (batched) <s> <sp1> {sentence1} <sp2> {sentence2} </s>
            # _: emotion label

            logits = model(x) # (N, seq_len, vocab_size)
            logits = logits[:, :-1, :].contiguous() # (N, seq_len - 1, vocab_size)
            target = x[:, 1:].contiguous() # (N, seq_len - 1)
            
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            losses.append(loss.item())

    total_loss = np.mean(losses)
    
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
    ckpt_path='./ckpt',
    logging_step=300 
):
    """ Train with next word prediction
    
    train_loader => (x, emotion_1, emotion_2)
    x: <s> <sp1> {sentence1} <sp2> {sentence2} </s>
    emotion_1, emotion_2: emotion label
    
    In this train flow, we will only use x for simplicity. (This may be changed after)
    """ 

    # Make a directory to save the checkpoint weight(s)
    os.makedirs(ckpt_path, exist_ok=True)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    losses = []
    train_loss = []
    best_loss = float('inf')
    
    model.to(device)
    for epoch in range(n_epochs):
        epoch_loss = []
        for step, inputs in enumerate(train_loader):
            model.train()
            x, _, _ = inputs # x: (N, max_seq_len)
            y = x[:, 1:].contiguous()
            x, y = x.to(device), y.to(device)
            # x: (batched) <s> <sp1> {sentence1} <sp2> {sentence2} </s>
            # _: emotion labels

            optimizer.zero_grad()
            logits = model(x) # (N, seq_len, vocab_size)
            logits = logits[:, :-1, :].contiguous() # (N, seq_len - 1, vocab_size)
            target = x[:, 1:].contiguous() # (N, seq_len - 1)
            
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            if (step + 1) % logging_step == 0:
                print(f"[Epoch {epoch + 1}/{n_epochs}] Step {step  + 1}/{len(train_loader)} | loss: {np.mean(epoch_loss):.3f}")
                
        train_loss.append(np.mean(epoch_loss))
        valid_loss = validate(model, valid_loader, device)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_ckpt(
                ckpt_path=p.join(ckpt_path, f"checkpoint_epoch_{epoch + 1}.pt"), 
                model=model, epoch=epoch + 1, 
                train_loss=train_loss, best_loss=best_loss
            )

        print(f"[Epoch {epoch + 1}/{n_epochs}] Test generation")
        print(f"Input: {gen_example}")
        response_sentence = generate(
            user_input=gen_ex_input,
            max_seq_len=gen_max_seq_len,
            model=model,
            tokenizer=tokenizer,
            gen_policy=gen_policy,
            device=device
        )
        print(f"Output: {response_sentence}")
