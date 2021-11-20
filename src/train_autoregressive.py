import torch
import option
import argparse
import numpy as np
from model import GPT2Model

parser = option.get_option_arg_parser()
args = parser.parse_args()

model = GPT2Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def saveModel(epoch, train_loss, valid_loss):
    path = "../model/gpt2.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "train_loss" : train_loss,
        "valid_loss" : valid_loss} , path)
            
    
def valLoss(valid_loader):
    
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs in valid_loader:
            inputs = torch.stack(inputs)
            inputs = inputs.transpose(1,0)
            # run model on the valid set to print out logits
            logits = model(inputs)
            
            all_logits = logits[:,:-1].contiguous()
            target = inputs[:,1:].contiguous()
            
            loss = criterion(all_logits, target[:,:])
            losses.append(loss.item())
            
    total_loss = np.mean(losses)
    
    return total_loss

def train(n_epochs, train_loader, valid_loader):
    '''
    n_epochs -> args.epoch
    train_loader -> train_loader
    valid_loader -> valid_loader
    '''
    
    losses = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    
    model.to(device)
    train_loss = []
    best_loss = 1000000
    
    for epoch in range(n_epochs):
        epoch_loss = []
        for turn_num, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            # if inputs is list of Tensor, list of Tensor --> Tensor
            inputs = torch.stack(inputs)
            inputs = inputs.transpose(1,0)
            inputs = inputs.to(device)
            
            logits = model(inputs)
            
            all_logits = logits[:,:-1].contiguous()
            target = inputs[:,1:].contiguous()
            
            loss = criterion(all_logits, target[:,:])
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            if turn_num % 10 == 0:
                print("[%d, %d] loss: %.3f"
                      %(epoch + 1, turn_num +1, np.mean(epoch_loss)))
                
        train_loss.append(np.mean(epoch_loss))
        valid_loss = valLoss(valid_loader)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            saveModel(epoch, train_loss, best_loss) 