import numpy as np
import cv2 
import os

def split_train_val(dset):
    train_dset_name = dset + '_train'
    val_dset_name = dset + '_val'
    if not os.path.exists(train_dset_name):
        os.mkdir(train_dset_name)
    if not os.path.exists(val_dset_name):
        os.mkdir(val_dset_name)
   
    n_images = len(os.listdir(dset))//2

    train_idxs = np.random.choice(n_images, int(0.8*n_images), replace=False)
    val_idxs = np.setdiff1d(np.arange(n_images), train_idxs)
    
    for idx, train_idx in enumerate(train_idxs):
        img = cv2.imread(os.path.join(dset, '%05d.jpg'%(train_idx)))
        mask = cv2.imread(os.path.join(dset, '%05d_mask.jpg'%(train_idx)), 0)
        cv2.imwrite(os.path.join(train_dset_name, '%05d.jpg'%(idx)), img)
        cv2.imwrite(os.path.join(train_dset_name, '%05d_mask.jpg'%(idx)), mask)

    for idx, val_idx in enumerate(val_idxs):
        img = cv2.imread(os.path.join(dset, '%05d.jpg'%(val_idx)))
        mask = cv2.imread(os.path.join(dset, '%05d_mask.jpg'%(val_idx)), 0)
        cv2.imwrite(os.path.join(val_dset_name, '%05d.jpg'%(idx)), img)
        cv2.imwrite(os.path.join(val_dset_name, '%05d_mask.jpg'%(idx)), mask)


if __name__ == '__main__':
    #dset_name = 'spaghetti_aug'
    dset_name = 'dset'
    split_train_val(dset_name)
