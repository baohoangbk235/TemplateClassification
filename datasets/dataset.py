from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import torch
import numpy as np 

import cv2
import matplotlib.pyplot as plt


class TemplateDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir 
        self.transforms = transforms 
        self.list_images = []

        if not os.path.exists(root_dir):
            exit()
        
        self.image_filenames = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, index: int):
        filename = self.image_filenames[index]
        img_path = os.path.join(self.root_dir, filename)
        print(img_path)
        img = cv2.imread(img_path)
        if self.transforms:
            img = self.transforms({"image": img})["image"]
        return img

if __name__ == "__main__":
    dataset = TemplateDataset("D:\TemplateClassification\BK_table_data\table")