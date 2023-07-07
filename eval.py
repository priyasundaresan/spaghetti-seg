import os
import imageio
import torch
import numpy as np
import cv2

from torch.utils.data import DataLoader
from model import SegModel
from torchvision import transforms
from dataset import PlateMaskDataset

TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])

DATASET_HOME = "data"
DATASET = "spaghetti_seg_ood"
TEST_DATASET_DIR = '%s/%s/test'%(DATASET_HOME, DATASET)
SAVE_DIR = "preds"

def main():
    n_cpu = os.cpu_count()
    test_dataset = PlateMaskDataset(TEST_DATASET_DIR, transform=TRANSFORM)
    #test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=n_cpu)

    model = SegModel("FPN", "resnet34", in_channels=3, out_classes=1)
    model.load_state_dict(torch.load('checkpoints/%s/model_50.pth'%DATASET))

    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    if os.path.exists(SAVE_DIR):
        os.system("rm -rf "+ SAVE_DIR)
    os.mkdir(SAVE_DIR)

    for i in range(len(pr_masks)):
        rgb = np.transpose(batch["image"][i].numpy(), (1,2,0))
#        rgb = np.transpose(batch["image"][i].numpy(), (2, 1, 0))
        #gt = np.array(batch["mask"][i].numpy().reshape((480,640,1)))
        #gt = cv2.cvtColor(gt,cv2.COLOR_GRAY2BGR)
        _, H, W = pr_masks[i].shape
        pred = np.array(pr_masks[i].numpy().reshape((H,W,1)))
        pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)

        #combined = cv2.vconcat([rgb, gt, pred])
#        print("rgb.shape", rgb.shape, "pred", pred.shape)
        combined = cv2.vconcat([rgb, pred])
        imageio.imwrite(os.path.join(SAVE_DIR, "%05d.jpg"%i), combined)

if __name__ == '__main__':
    main()
