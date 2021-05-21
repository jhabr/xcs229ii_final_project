import os

import torch
from torch.utils.data import Dataset

from constants import TRAIN_DIR
from utils.dataloader import SimpleDataLoader


class ISICDataset(Dataset):
    def __init__(self, resize_to, size=None, image_dir=TRAIN_DIR):
        self.train_images_masks = SimpleDataLoader(
            images_folder_path=os.path.join(image_dir, "images"),
            masks_folder_path=os.path.join(image_dir, "masks"),
            resize_to=resize_to,
            size=size
        ).get_images_masks()

    def __len__(self):
        return len(self.train_images_masks["images"])

    def __getitem__(self, idx):
        image = self.train_images_masks["images"][idx]
        label = self.train_images_masks["masks"][idx]
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)
        return image, label
