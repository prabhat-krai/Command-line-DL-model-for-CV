import torch 
from torch import optim, nn
from torchvision import transforms, datasets
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json
from collections import OrderedDict

import helper 

arg = argparse.ArgumentParser(description = 'train.py')

arg.add_argument('get_data', nargs='*',
                action="store", default="./flowers")
arg.add_argument('--save_model', dest="save_model", 
                action="store", default="./checkpoint.pth")
arg.add_argument('--learning_rate', dest="learning_rate", 
                action="store", default=0.001)
arg.add_argument('--dropout', dest = "dropout", 
                action = "store", default = 0.4)
arg.add_argument('--epochs', dest="epochs", 
                action="store", type=int, default=1)
arg.add_argument('--arch', dest="arch", 
                action="store", default="resnet50", type = str)
arg.add_argument('--hidden_unit', type=int, dest="hidden_units", 
                action="store", default=768)
arg.add_argument('--compute', dest="compute", 
                action="store", default="cuda")

parser = arg.parse_args()
loc = parser.get_data
filepath = parser.save_model
lr = parser.learning_rate
architecture = parser.arch
dropout = parser.dropout
first_hidden_layer = parser.hidden_units
epochs = parser.epochs
compute = parser.compute

trainloader, validloader, testloader, class_to_idx = helper.get_data(loc)
model, criterion, optimizer = helper.model_setup(architecture, dropout, first_hidden_layer, lr, compute)
helper.train_model(model, criterion, optimizer, epochs, compute, trainloader, validloader)
helper.save_checkpoint(filepath, architecture, first_hidden_layer, lr, dropout, model, class_to_idx)
print("Done")



