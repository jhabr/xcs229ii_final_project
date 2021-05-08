import glob
import os.path
from typing import List

import numpy as np
import tensorflow as tf
import segmentation_models as sm

from utils.augmentation import DataAugmentation
from utils.preprocessing import ImagePreprocessor


class DataLoader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i) -> List[np.ndarray]:
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class SimpleDataLoader:

    def __init__(self, images_path, backbone=None, mask_path=None, normalize=True, resize=True, size=None):
        self.images_path = images_path
        self.backbone = backbone
        self.mask_path = mask_path
        self.images = None
        self.masks = None
        self.normalize = normalize
        self.resize = resize
        self.size = size
        self.image_preprocessor = ImagePreprocessor()
        self.data_augmentation = DataAugmentation()
        
    def get_images(self) -> object:
        """
        Reads and returns the images as a list or np.array of np.arrays.

        :return: list or np.array of images
        """
        if self.images is not None:
            return self.images

        image_paths = sorted(glob.glob(os.path.join(self.images_path, "*.jpg")))
        if self.size is not None:
            image_paths = image_paths[:self.size]
            
        images = []

        for image_path in image_paths:
            image = self.image_preprocessor.apply_image_default(
                image_path=image_path,
                normalize=self.normalize,
                resize=self.resize
            )
            if self.backbone:
                image = self.data_augmentation.apply_default(
                    image=image,
                    default_augmentation=sm.get_preprocessing(self.backbone)
                )
            images.append(image)

        # if we don't resize, we cannot stack image as the dimensions of all the images must be the same
        if self.resize:
            self.images = np.array(images, dtype=np.float32)
        else:
            self.images = images

        return self.images

    def get_masks(self) -> object:
        """
        Reads and returns the masks as a list or np.array of np.arrays.

        :return: list or np.array of masks
        """
        if self.masks is not None:
            return self.masks

        if self.mask_path is None:
            return None

        mask_paths = sorted(glob.glob(os.path.join(self.mask_path, "*.png")))
        if self.size is not None:
            mask_paths = mask_paths[:self.size]

        masks = []

        for mask_path in mask_paths:
            mask = self.image_preprocessor.apply_mask_default(
                mask_path=mask_path,
                normalize=self.normalize,
                resize=self.resize
            )
            masks.append(mask)

        # if we don't resize, we cannot stack image as the dimensions of all the images must be the same
        if self.resize:
            self.masks = np.array(masks, dtype=np.float32)
        else:
            self.masks = masks

        return self.masks

    def get_images_masks(self) -> dict:
        return {
            "images": self.get_images(),
            "masks": self.get_masks()
        }
