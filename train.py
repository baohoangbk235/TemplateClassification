import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import  models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from utils.dataset import TemplateDataset
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import argparse
from utils.losses import TripletLoss

if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str, help="path to root dir")

args = parser.parse_args()

def get_model(num_classes):
    vgg_16 = models.vgg16(pretrained=True)
    in_features = vgg_16.classifier[-1].in_features
    out_features = num_classes
    vgg_16.classifier[-1] = nn.Linear(in_features, out_features)
    return vgg_16

def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256), 
        A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]) 

def get_valid_transforms():
    pass 

def get_test_transforms():
    pass 

def collate_fn(batch):
    images_list = []
    labels_list = []
    for images, labels in batch:
        images_list.append(images)
        labels_list.append(labels)

    return images_list, labels_list

traindataset = TemplateDataset("/mnt/disk2/baohg/data", transforms=get_train_transforms())  
trainloader = DataLoader(traindataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# with open("BK_table_data/001/labels.txt", "r") as f:
#     lines = f.readlines()
#     classes = []
#     for line in lines:
#         classes.append(line.rstrip("\n"))

num_classes = len(os.listdir("/mnt/disk2/baohg/data"))

model = get_model(512)
model.to(device)
num_epochs = 1

triplet_loss = TripletLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

running_loss = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        anchors = torch.stack([imgs[0] for imgs in images], dim=0)
        positives = torch.stack([imgs[1] for imgs in images], dim=0)
        negatives = torch.stack([imgs[2] for imgs in images], dim=0)
        
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        anchor_emb = model(anchors)        
        pos_emb = model(positives)
        neg_emb = model(negatives)
        loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            print(f'Step {i} : {epoch+1}/{num_epochs} : Loss: {running_loss/100}')
            running_loss = 0

        del anchor_emb
        del pos_emb
        del neg_emb
    


