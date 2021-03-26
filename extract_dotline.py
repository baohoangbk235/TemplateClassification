import os
import cv2 
import numpy as np
from utils import bilateral_filter, thresholding
from preprocess import extract_line
import random
import matplotlib.pyplot as plt 

def extract_dotline(ori_img, text_boxes):
    pass


DATA_DIR = "BK_table_data/table"

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
    img_path = os.path.join(DATA_DIR, random.choice(os.listdir(DATA_DIR)))
    img_path = img_path.replace(os.sep, '/')
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    ori_img = extract_line(img)[-1]

    annot_path = "BK_table_data/result/res_" + os.path.splitext(os.path.basename(img_path))[0] + ".txt"

    with open(annot_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            pts = []
            coords = line.rstrip("\n").split(",")
            coords = [int(coord) for coord in coords]
            for c1, c2 in zip(coords[0:-1:2], coords[1::2]):
                pts.append([c1, c2])

            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(ori_img, pts =[pts], color=(255,255,255))

            gray = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
            gray = bilateral_filter(gray)

            th = thresholding(gray, "gauss")
            th = (255-th)

            vertical_kernel = np.ones((15,1),np.uint8)
            horizontal_kernel = np.ones((1,40),np.uint8)

            vertical_erosion = cv2.erode(th, vertical_kernel, iterations = 1)
            horizontal_erosion = cv2.erode(th, horizontal_kernel, iterations = 1)

            erosion = cv2.bitwise_or(vertical_erosion, horizontal_erosion, mask = None)   
            # cv2.polylines(ori_img,[pts],True,(255,0,0)) 
            
    titles = ['Original Image', 'Erosion', 'Contours', 'Final']
    visualize(titles, [ori_img, gray, th, erosion])
