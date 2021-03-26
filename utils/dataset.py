import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
import os
import torch
import numpy as np 
import cv2
import matplotlib.pyplot as plt

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
    
    def get_triplet(self, anchor_label, neg_label):
        image_filenames = os.listdir(os.path.join(self.root_dir, anchor_label))
        anchor_filename = random.choice(image_filenames)
        anchor_path = os.path.join(self.root_dir, anchor_label, anchor_filename)
        anchor_img = cv2.imread(anchor_path.replace(os.sep, '/'))

        pos_filename = random.choice([pos for pos in image_filenames if pos != anchor_filename])
        pos_path = os.path.join(self.root_dir, anchor_label, pos_filename)
        pos_img = cv2.imread(pos_path.replace(os.sep, '/'))

        neg_filename = random.choice(os.listdir(os.path.join(self.root_dir, neg_label)))
        neg_path = os.path.join(self.root_dir, neg_label, neg_filename)
        neg_img = cv2.imread(neg_path.replace(os.sep, '/'))

        return anchor_img, pos_img, neg_img
    
    def show_triplet(self, images):
        titles = ["Anchor", "Positive", "Negative"]
        plt.figure(figsize=(12,4))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(images[i].astype("uint8"))
            plt.title(titles[i])
        plt.show()
    
    def __getitem__(self, index: int):
        label = self.labels[index]
        neg_label = random.choice([neg_ for neg_ in self.labels if neg_ != label])

        anchor, pos, neg = self.get_triplet(label, neg_label)
        self.show_triplet([anchor, pos, neg])
        if self.transforms:
            anchor = self.transforms({"image": anchor})["image"]
            pos = self.transforms({"image": pos})["image"]
            neg = self.transforms({"image": neg})["image"]
        return [anchor, pos, neg], [label, label, neg_label]

if __name__ == "__main__":
    if args.path is None:
        dataset = TemplateDataset("BK_table_data/table")  
    else:
        dataset = TemplateDataset(args.path)
    sample = dataset[2]
