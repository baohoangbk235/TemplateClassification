import os
import cv2 
import numpy as np

def extract_dotline(ori_img, text_boxes):
    pass

if __name__ == "__main__":
    ori_img = cv2.imread("BK_table_data/2020-12-15T16_26_15.747833_01859.jpg", cv2.COLOR_BGR2RGB)
    with open("BK_table_data/res_2020-12-15T16_26_15.747833_01859.txt", "r") as f:
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
            # cv2.polylines(ori_img,[pts],True,(255,0,0))
            
    cv2.imshow("img", ori_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()