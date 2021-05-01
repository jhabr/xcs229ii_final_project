import glob
import os.path

import cv2
import numpy as np
import tensorflow as tf


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

    def __getitem__(self, i) -> np.ndarray:
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

    def __init__(self, images_path, mask_path=None, preprocessing=None, size=None):
        self.images_path = images_path
        self.mask_path = mask_path
        self.preprocessing = preprocessing
        self.images = []
        self.masks = []
        self.size = size
        
    def get_images(self) -> list:
        image_paths = sorted(glob.glob(os.path.join(self.images_path, "*.jpg")))
        if self.size is not None:
            image_paths = image_paths[:self.size]
            
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.preprocessing:
                image = self.preprocessing(image=image)['image']
            self.images.append(image)

        return self.images

    def get_masks(self) -> object:
        if self.mask_path is None:
            return None

        mask_paths = sorted(glob.glob(os.path.join(self.mask_path, "*.png")))
        if self.size is not None:
            mask_paths = mask_paths[:self.size]

        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = np.expand_dims(mask, axis=2)
            self.masks.append(mask)

        return self.masks

    def get_images_masks(self) -> tuple:
        return self.get_images(), self.get_masks()
