import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
import os
import torch
import numpy as np 

import cv2
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str, help="path to root dir")

args = parser.parse_args()

def show_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

class TemplateDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir 
        self.transforms = transforms 
        self.list_images = []

        if not os.path.exists(root_dir):
            exit()
        
        self.labels = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, index: int):
        label = self.labels[index]
        self.image_filenames = os.listdir(os.path.join(self.root_dir, label))
        filename = random.choice(self.image_filenames)
        img_path = os.path.join(self.root_dir, self.label, filename)
        print(img_path)
        # img = cv2.imread(img_path)
        # show_image(img)
        # if self.transforms:
        #     img = self.transforms({"image": img})["image"]
        return img

if __name__ == "__main__":
    if args.path is None:
        dataset = TemplateDataset("BK_table_data/table")  
    else:
        dataset = TemplateDataset(args.path)
    sample = dataset[1]
