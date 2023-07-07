import numpy as np
import cv2
import os
from generate_dset import preprocess

def detect_plate(img):
    img = preprocess(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        circles = np.round(circles[0,:]).astype('int')
        for (x,y,r) in circles:
            if r>100:
                cv2.circle(img, (x,y), r, (0,255,0), 4)
                cv2.rectangle(img, (x-5,y-5), (x+5,y+5), (0,128,255), -1)

    print(circles)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    raw_data_dir = 'raw_data'
    subfolders = os.listdir(raw_data_dir)
    for subfolder in subfolders:
        img_fns = sorted(os.listdir(os.path.join(raw_data_dir, subfolder)))
        for img_fn in img_fns:
            img = cv2.imread(os.path.join(raw_data_dir, subfolder, img_fn))
            detect_plate(img)
