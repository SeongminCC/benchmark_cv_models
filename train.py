import wandb
import time
import os
import json
import torch
from collections import OrderedDict
from tqdm import tqdm

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, criterion, optimizer):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()  
    losses_m = AverageMeter() 
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())
        
        # accuracy
        pred = outputs.argmax(dim=1)
        acc_m.update(targets.eq(pred).sum().item()/targets.size(0), n=targets.size(0))
        
        batch_time_m.update(time.time() - end)
        end = time.time()
        
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])


def test(model, dataloader, criterion):
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # predict
            outputs = model(inputs)
            
            # loss
            loss = criterion(outputs, targets)
            
            # total loss & acc
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += targets.eq(pred).sum().item()
            total += targets.size(0)
            
        return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
    
    
def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler,
    epochs: int, use_wandb: bool, savedir: str, early_stopping: bool = False
) -> None:
    
    best_acc = 0
    step = 0
    
    for epoch in range(epochs):
        print(f"epoch : {epoch}")
        train_metrics = train(model = model,
                              dataloader = trainloader,
                              criterion = criterion,
                              optimizer = optimizer
                             )
        
        eval_metrics = test(model = model,
                            dataloader = testloader,
                            criterion = criterion
                           )
        print(f"acc : {eval_metrics['acc']}, loss : {eval_metrics['loss']}")
        
        if scheduler:
            scheduler.step()
            
        # wandb
        if use_wandb:
            metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
            metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            wandb.log(metrics, step=step)
        
        step += 1
        
        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent=4)
            
            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            best_acc = eval_metrics['acc']
            print(f"best_acc : {eval_metrics['acc']}")
        else:    
            print(f"best_acc : {best_acc}")
            
        # early stopping
        if early_stopping:
            keep = 0
            stop = 15
            if best_acc > eval_metrics['acc']:
                keep += 1
                print(f"Accuracy has not improved for {keep} epoch.")
                if keep > stop:
                    print("Early Stopping is working")
                    print(f"The best Accuracy is {best_acc}")