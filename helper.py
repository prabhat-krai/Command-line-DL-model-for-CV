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

#archtecures that can be trained using this script
compute='cuda'
architectures={'resnet50': 2048,
               'densenet201':1920}

device = compute

def get_data(dir_for_model = './flowers'):
    data_dir=dir_for_model
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #apply transformations to images for training, testing and validation
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_validation_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    #extracting images and labels and performing transformations 
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_validation_transforms)
    
    
    #make data loaders to send data into the model
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size= 128, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 128,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 128, shuffle = True)
    
    return trainloader, validloader, testloader, train_data.class_to_idx

def model_setup(arch , dropout=0.4, hidden_layer= 768, lr=0.001, compute='cuda'):
    if (arch == 'resnet50'):
        model = models.resnet50(pretrained=True)
    elif (arch == 'densenet201'):
        model = models.densenet201(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier=nn.Sequential(OrderedDict([
                    ('dropout', nn.Dropout(dropout)),
                    ('fc1', nn.Linear(architectures[arch], hidden_layer)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(hidden_layer, 256)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(256, 102)),
                    ('out', nn.LogSoftmax(dim=1))  
                                        ]))
    criterion=nn.NLLLoss()
    
    if( arch == 'resnet50'):
        model.fc= classifier
        optimizer=optim.Adam(model.fc.parameters(), lr)
    else: 
        model.classifier = classifier
        optimizer=optim.Adam(model.classifier.parameters(), lr)
        
    if (torch.cuda.is_available() and compute=='cuda'):
        model.to('cuda')
    else:
        model.to('cpu')
    
    return model, criterion, optimizer 

def train_model(model, criterion, optimizer, epochs, compute, data_from, validloader):
    print_info=40
    steps=0
    running_loss=0
    if (torch.cuda.is_available() and compute=='cuda'):
        device = 'cuda'
    else:
        device = 'cpu'
    print("Training is being done on ", device)    
    for epoch in range(epochs):
        print ("This is epoch number", epoch + 1)
        
        for inputs, labels in data_from:
            
            model.train()
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            output_log_ps = model.forward(inputs)
            loss = criterion(output_log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (steps % print_info == 0):
                
                valid_loss=0
                accuracy=0
                model.eval()
                
                for v_inputs, v_labels in validloader: 
                    
                    v_inputs, v_labels = v_inputs.to(device) , v_labels.to(device)
                    
                    with torch.no_grad():
                        v_outputs = model.forward(v_inputs)
                        valid_loss = criterion(v_outputs, v_labels)
                        probs = torch.exp(v_outputs)
                        top_p, top_class = probs.topk(1, dim=1)
                        equals= top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                       
                accuracy /= len(validloader)
                
                print( "Loss during training per batch: {:.2f}".format(running_loss/print_info),
                       "Loss during validation per batch {:.2f}".format(valid_loss/ len(validloader)),
                       "Accuracy on the validation set: {:.2f}".format(accuracy))
            
                
def save_checkpoint(filepath, architecture, hidden_layer, lr, dropout, model, class_to_idx):
    
    model.class_to_idx = class_to_idx
    
    torch.save({'architecture' : architecture,
                'first_hidden_layer':hidden_layer,
                'lr':lr,
                'dropout':dropout,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                filepath)
    
    
def load_checkpoint(filepath='checkpoint.pth'):
    
    checkpoint = torch.load(filepath)
    architecture = checkpoint['architecture']
    hidden_layer = checkpoint['first_hidden_layer']
    lr=checkpoint['lr']
    dropout = checkpoint['dropout']
    
    model,_,_ = model_setup(architecture , dropout , hidden_layer , lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    
    img = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                             ])

    img= transform(img)

    return img


def predict(filepath, model, topk=5, compute='gpu'):
    
    if (torch.cuda.is_available() and compute=='gpu'):
        model.to('cuda')
    else:
        model.to('cpu')
    model.eval()
    img = process_image(filepath)
    img = img.unsqueeze_(0)

    if (compute == 'gpu'):
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img.cpu())

    probs = torch.exp(output)
    top_probs, top_classes= probs.topk(topk, dim=1)
    
    return top_probs, top_classes
            
        
    
    
    
    
