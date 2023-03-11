import numpy as np
import os
import random
import wandb

import torch
import argparse
import logging

from train import fit

from datasets import create_dataset, create_dataloader
from models import *



def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
    
    
    
def run(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    models = {
        'my_model': my_model,
        
        'ResNet18': ResNet18,
        'ResNet34': ResNet34,
        'ResNet50': ResNet50,
        
        'VGG11' : VGG11,
        'VGG13' : VGG13,
        'VGG16' : VGG16,
        'VGG19' : VGG19,
        
        'GoogLeNetV3' : GoogLeNetV3
        
    }
    
    model = models[args.model]().to(DEVICE)

    # load dataset
    trainset, testset = create_dataset(datadir=args.datadir, aug_name=args.aug_name) # arg

    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)  # arg
    testloader = create_dataloader(dataset=testset, batch_size=256, shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr) # arg

    
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, eta_min=0.00001) # arg
    
    
    # make save directory
    savedir = os.path.join(args.savedir, args.exp_name)  # arg
    os.makedirs(savedir, exist_ok=True)
    
#     # load checkpoints
#     if args.ckpdir:   # arg : ckpdir (ex : './saved_model/resnet50'
#         torch.load_state(args.ckpdir)
        
    # initialize wandb
    if args.use_wandb:  # arg : use_wandb
        wandb.init(name=args.exp_name, project='CIFAR100 Test', config=args) # arg
        
    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = args.epochs,   # arg
        savedir      = savedir,
        use_wandb    = args.use_wandb)  # arg
    
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="CIFAR100 test")

    # experiment setting
    parser.add_argument('--exp-name', type = str, help = 'experiment')
    parser.add_argument('--datadir',type=str,default='./datasets',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')
    parser.add_argument('--model',type=str,default='my_model',help='saved model directory')

    # optimizer
    parser.add_argument('--opt-name',type=str,choices=['SGD','Adam'],help='optimizer name')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')

    # augmentation
    parser.add_argument('--aug-name',type=str,choices=['default', 'weak'],help='augmentation type')

    # train
    parser.add_argument('--epochs',type=int,default=50,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--ckpdir',type=str,default=None,help='checkpoint directory')

    # seed
    parser.add_argument('--seed',type=int,default=1201,help='1201 is my birthday')

    # wandb
    parser.add_argument('--use-wandb',action='store_true',help='use wandb')

    args = parser.parse_args()

    run(args)