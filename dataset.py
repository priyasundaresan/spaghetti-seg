import os
import imageio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PlateMaskDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([a for a in os.listdir(self.root_dir) if "jpg" in a])//2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_name = os.path.join(self.root_dir, "%05d.jpg"%idx)
        rgb_image = imageio.imread(rgb_name)
        mask_name = os.path.join(self.root_dir, "%05d_mask.jpg"%idx)
        mask_image = imageio.imread(mask_name)

        sample = {'image': rgb_image, 'mask': mask_image}

        if self.transform:
            for key in sample.keys():
                sample[key] = self.transform(sample[key])

        return sample

if __name__ == '__main__':
    TRANSFORM = transforms.Compose([
        transforms.ToTensor()
    ])
    dset = PlateMaskDataset('data/spaghetti_seg_v0/train', TRANSFORM) 
