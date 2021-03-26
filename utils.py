import cv2 

def bilateral_filter(img, mode="average"):
    if mode == "average":
        blur = cv2.blur(img, (5,5))
    if mode == "gauss":
        blur = cv2.GaussianBlur(img,(5,5),0)
    if mode == "median":
        median = cv2.medianBlur(img,5)
    if mode == "bilateral":
        blur = cv2.bilateralFilter(img,9,75,75)
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