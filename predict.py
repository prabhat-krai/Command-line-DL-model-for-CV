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

arg = argparse.ArgumentParser(description='predict.py')
arg.add_argument('input_img', default='./flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
arg.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store",type = str)
arg.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
arg.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
arg.add_argument('--compute', default="cuda", action="store", dest="compute")
arg.add_argument('--cat_to_name', default="cat_to_name.json", action="store", type = str) 

parser = arg.parse_args()
filepath = parser.input_img
k = parser.top_k
compute = parser.compute
path_checkpoint = parser.checkpoint
file_json = parser.cat_to_name

with open( file_json , 'r') as json_file:
    cat_to_name = json.load(json_file)

trainloader, validloader, testloader, _ = helper.get_data()
model= helper.load_checkpoint(path_checkpoint)
probs, classes = helper.predict(filepath, model, k, compute)
classes=classes.cpu().detach().numpy().tolist()[0]
idx_to_class = {val : key for key, val in model.class_to_idx.items()}
name_flowers = [cat_to_name[idx_to_class[int(a_class)]] for a_class in classes]
 
print(name_flowers, probs)
