import cv2
import os
import numpy as np
import glob


class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None
    ):
        self.image_ids = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.mask_ids = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
        self.images = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        # self.augmentation = augmentation
        # self.preprocessing = preprocessing

    def __getitem__(self, i) -> tuple:
        image = cv2.imread(self.images[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)

        # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self) -> int:
        assert len(self.image_ids) == len(self.mask_ids)
        return len(self.image_ids)
