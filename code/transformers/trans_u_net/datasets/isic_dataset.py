import os

from torch.utils.data import Dataset

from constants import TRAIN_DIR
from utils.dataloader import SimpleDataLoader


class ISICDataset(Dataset):
    def __init__(self, resize_to):
        self.train_images_masks = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=resize_to
        ).get_images_masks()

    def __len__(self):
        return len(self.train_images_masks["images"])

    def __getitem__(self, idx):
        image = self.train_images_masks["images"][idx]
        label = self.train_images_masks["masks"][idx]
        return {'image': image, 'label': label}
