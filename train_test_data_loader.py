# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:26:47 2020

@author: H343380
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd

#### return back train_loader and test loader ####

dataset_type=datasets.MNIST
batch_size = 64
use_cuda = torch.cuda.is_available()

def data_loader(dataset_type=datasets.MNIST,batch_size = 64,kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}):
    torch.manual_seed(1)
    if dataset_type==datasets.MNIST:
        normalized_value_a=(0.1307,)
        normalized_value_b=(0.3081,)
    elif dataset_type==datasets.CIFAR10:
        normalized_value_a=(0.5, 0.5, 0.5)
        normalized_value_b=(0.5, 0.5, 0.5)
        
    train_loader = torch.utils.data.DataLoader(
    dataset_type('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(normalized_value_a,normalized_value_b)
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
    dataset_type('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(normalized_value_a,normalized_value_b)
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
    
    train_test_loader=(train_loader,test_loader)
    
    return train_test_loader
    
        
        
        
