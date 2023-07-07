import numpy as np
import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import random

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
COLOR_MASK_AUGS = [ 
    iaa.flip.Fliplr(0.5),
    sometimes(iaa.Affine(
                scale={"x": (1.0, 1.1), "y": (1.0, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-8, 8), # rotate by -45 to +45 degrees
                shear=(-10, 10), # shear by -16 to +16 degrees
                order=[0], # use nearest neighbour or bilinear interpolation (fast)
                mode=['constant']
            )),
    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05)))
    ]
COLOR_ONLY_AUGS = [
    sometimes(iaa.LinearContrast((0.65, 1.35), per_channel=0.25)), 
    sometimes(iaa.Add((-20, 20), per_channel=False)),
    sometimes(iaa.ChangeColorTemperature((5000, 10000))),
    sometimes(iaa.GammaContrast((0.75, 1.25))),
    sometimes(iaa.MultiplySaturation((0.85, 1.15))),
    sometimes(iaa.MultiplyHue((0.6,1.4))),
    sometimes(iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)))]

seq_color_mask = iaa.Sequential(COLOR_MASK_AUGS, random_order=True)
seq_color_only = iaa.Sequential(COLOR_ONLY_AUGS, random_order=True)

def refine_mask(mask):
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    
    for i in range(0, nlabels - 1):
        if areas[i] >= 200:   #keep
            result[labels == i + 1] = 255
    return result

def preprocess(img, crop_size=(640,480)):
    H,W,C = img.shape
    H_new = 550
    dim_diff = W-H_new
    img = img[:H_new, int(0.6*dim_diff):W-int(0.4*dim_diff)]
    img = cv2.resize(img, (480,480))
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    return img

def process(img, bg_img):
    img = preprocess(img)
    bg_img = preprocess(bg_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg_img_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(bg_img_gray.astype("uint8"), img_gray)
    threshold = 40
    mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    mask = refine_mask(mask)
    result = np.vstack((np.hstack((img_gray, bg_img_gray)), np.hstack((diff, mask))))
    #cv2.imshow('img', result)
    #cv2.waitKey(0)
    return img, mask

def random_transform(img, mask, plot=False):
    seq = seq_color_mask
    seq = seq.to_deterministic()
    img_aug = seq(image=img)
    img_aug = seq_color_only(image=img_aug)
    mask_aug = seq(image=mask)
    _, mask_aug = cv2.threshold(mask_aug, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = np.stack((mask_aug,)*3, axis=-1)
    vis = np.hstack((img_aug, mask_3ch))
    if plot:
        cv2.imshow('vis', vis)
        cv2.waitKey(0)
    return img_aug, mask_aug
    
if __name__ == '__main__':
    raw_data_dir = 'raw_data'
    output_dir = 'dset'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    augs_per_img = 10
    ctr = 0

    subfolders = os.listdir(raw_data_dir)
    #subfolders = ['large_blue_plate0', 'large_blue_plate1', 'large_blue_plate2', 'large_blue_plate4', 'small_3plates', 'small_green0', 'small_green1', 'small_green2']
    for subfolder in subfolders:
        img_fns = sorted(os.listdir(os.path.join(raw_data_dir, subfolder)))
        bg_img = cv2.imread(os.path.join(raw_data_dir, subfolder, img_fns[0]))
        for img_fn in img_fns[1:]:
            img = cv2.imread(os.path.join(raw_data_dir, subfolder, img_fn))
            img, mask = process(img, bg_img)
            cv2.imwrite(os.path.join(output_dir, '%05d.jpg'%ctr), img)
            cv2.imwrite(os.path.join(output_dir, '%05d_mask.jpg'%ctr), mask)
            ctr += 1
            for _ in range(augs_per_img):
                #img_aug, mask_aug = random_transform(img, mask, plot=True)
                img_aug, mask_aug = random_transform(img, mask, plot=False)
                cv2.imwrite(os.path.join(output_dir, '%05d.jpg'%ctr), img_aug)
                cv2.imwrite(os.path.join(output_dir, '%05d_mask.jpg'%ctr), mask_aug)
                ctr += 1
            print(ctr)
