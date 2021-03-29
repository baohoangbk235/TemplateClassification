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
    device = "cuda"
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
        A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        A.Resize(256, 256),
        ToTensorV2(p=1.0)
    ], p=1.) 

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

traindataset = TemplateDataset("BK_table_data/labeld_001/", transforms=get_train_transforms())  
trainloader = DataLoader(traindataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# with open("BK_table_data/001/labels.txt", "r") as f:
#     lines = f.readlines()
#     classes = []
#     for line in lines:
#         classes.append(line.rstrip("\n"))

model = get_model(6)

num_epochs = 1

loss = TripletLoss()

for batch, (images, labels) in enumerate(trainloader):
    # anchor_emb = model(anchor)
    # print(torch.shape(anchor_emb))
    break 
    


