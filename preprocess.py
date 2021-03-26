import cv2
import matplotlib.pyplot as plt
import numpy as np 
import random
import argparse
import os
from utils import thresholding, bilateral_filter

parser = argparse.ArgumentParser()

parser.add_argument('--path', "-p", type=str, help='path to image') 

args = parser.parse_args()

DATA_DIR = "BK_table_data/table"

def bilateral_filter(img, mode="average"):
    if mode == "average":
        blur = cv2.blur(img, (5,5))
    if mode == "gauss":
        blur = cv2.GaussianBlur(img,(5,5),0)
    if mode == "median":
        median = cv2.medianBlur(img,5)
    if mode == "bilateral":
        blur = cv.bilateralFilter(img,9,75,75)
    return blur

def thresholding(img, mode="global"):
    if mode == "mean":
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    elif mode == "gauss":
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    elif mode == "global":
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return th


def extract_dotline():
    pass 

def extract_line(img_path):
    ori_img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

def extract_line(ori_img):
    gray = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)

    gray = bilateral_filter(gray)

    th = thresholding(gray, "gauss")
    th = (255-th)

    vertical_kernel = np.ones((15,1),np.uint8)
    horizontal_kernel = np.ones((1,40),np.uint8)

    vertical_erosion = cv2.erode(th, vertical_kernel, iterations = 1)
    horizontal_erosion = cv2.erode(th, horizontal_kernel, iterations = 1)

    erosion = cv2.bitwise_or(vertical_erosion, horizontal_erosion, mask = None)   

    contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    filter_contours = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 10 or h > 15:
            filter_contours.append(cnt)

    img = ori_img.copy()

    cv2.drawContours(img, filter_contours, -1, (255,0,0), 3)

    mask= np.reshape(np.repeat([erosion], 3), (img.shape[0], img.shape[1], 3))

    final_img = cv2.bitwise_or(ori_img, mask)

    images = [ori_img, th, img, final_img]
    return  images


def visualize(titles, images):
    plt.figure(figsize=(12,12))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        if titles[i] == 'Erosion':
            plt.imshow(images[i].astype("uint8"), "gray")
        else:   
            plt.imshow(images[i].astype("uint8"))
        plt.title(titles[i])
    plt.show()

if __name__ == "__main__":
    titles = ['Original Image', 'Erosion', 'Contours', 'Final']
    if args.path is None:
        img_path = os.path.join(DATA_DIR, random.choice(os.listdir(DATA_DIR)))
    else:
        img_path = args.path
    
    img_path = img_path.replace(os.sep, '/')
    ori_img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    images = extract_line(ori_img)
    visualize(titles, images)