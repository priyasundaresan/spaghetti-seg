import os
import copy
import time
import torch
import torchvision
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from model import SegModel
from torchvision import transforms
from dataset import PlateMaskDataset

TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])

DATASET_HOME = "data"
#DATASET = "spaghetti_seg_v1"
DATASET = "spaghetti_seg_ood"
TRAIN_DATASET_DIR = '%s/%s/train'%(DATASET_HOME, DATASET)
TEST_DATASET_DIR = '%s/%s/test'%(DATASET_HOME, DATASET)

EPOCHS = 50

def main():
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists('checkpoints/%s'%DATASET):
        os.mkdir('checkpoints/%s'%DATASET)

    n_cpu = os.cpu_count()
    #n_cpu = 0
    train_dataset = PlateMaskDataset(TRAIN_DATASET_DIR, transform=TRANSFORM)
    valid_dataset = PlateMaskDataset(TEST_DATASET_DIR, transform=TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    model = SegModel("FPN", "resnet34", in_channels=3, out_classes=1)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=EPOCHS,
    )

    print("Begin training")
    start = time.time()
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
    )
    print("Finished training in", time.time() - start)

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    print(valid_metrics)

    torch.save(model.state_dict(), ('checkpoints/%s/model_'%DATASET)+str(EPOCHS)+'.pth')
    print("Saved model")

if __name__ == '__main__':
    main()
