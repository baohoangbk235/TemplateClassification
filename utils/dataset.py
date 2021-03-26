import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
import os
import torch
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import glob

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
        self.triplets = []
        self.triplets_labels = []
        self.label2int = {str(k):v for v, k in enumerate(self.labels)}
        for label in self.labels: 
            print("[INFO] Processing {} ...".format(label))
            list_paths = glob.glob(os.path.join(self.root_dir, label, "*.jpg"))
            neg_label = random.choice([neg_ for neg_ in self.labels if neg_ != label])
            for anchor in list_paths:
                if len(list_paths) < 2:
                    break
                positive = random.choice([p for p in list_paths if p != anchor])
                negative = random.choice(glob.glob(os.path.join(self.root_dir, neg_label, "*.jpg")))
                self.triplets.append([anchor, positive, negative])
                self.triplets_labels.append([self.label2int[label], self.label2int[label], self.label2int[neg_label]])    

    def __len__(self):
        return len(self.triplets)
    
    def get_img(self, path):
        path = path.replace(os.sep, '/')
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        return img 
    
    def show_triplet(self, images, titles):
        # titles = ["Anchor", "Positive", "Negative"]
        plt.figure(figsize=(12,4))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(images[i].astype("uint8"))
            plt.title(titles[i])
        plt.show()
    
    def __getitem__(self, index: int):
        triplets = self.triplets[index]
        labels = self.triplets_labels[index]
        anchor = self.get_img(triplets[0])
        pos = self.get_img(triplets[1])
        neg = self.get_img(triplets[2])
        images = [anchor, pos, neg]
        if self.transforms:
            anchor = self.transforms({"image": anchor})["image"]
            pos = self.transforms({"image": pos})["image"]
            neg = self.transforms({"image": neg})["image"]
        # self.show_triplet(images, labels)
        return images, labels
        

if __name__ == "__main__":
    if args.path is None:
        dataset = TemplateDataset("BK_table_data/table")  
    else:
        dataset = TemplateDataset(args.path)
    images  = dataset[2]
