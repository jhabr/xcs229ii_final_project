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

    IMAGE_SIZE = (768, 1024)

    def __init__(self, images_path, mask_path=None, preprocessing=None, size=None):
        self.images_path = images_path
        self.mask_path = mask_path
        self.preprocessing = preprocessing
        self.images = None
        self.masks = None
        self.size = size
        
    def get_images(self) -> np.array:
        image_paths = sorted(glob.glob(os.path.join(self.images_path, "*.jpg")))
        if self.size is not None:
            image_paths = image_paths[:self.size]
            
        images = []

        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
            image = cv2.resize(image, SimpleDataLoader.IMAGE_SIZE)
            if self.preprocessing:
                image = self.preprocessing(image=image)['image']
            images.append(image)

        self.images = np.array(images, dtype=np.float32)

        return self.images

    def get_masks(self) -> object:
        if self.mask_path is None:
            return None

        mask_paths = sorted(glob.glob(os.path.join(self.mask_path, "*.png")))
        if self.size is not None:
            mask_paths = mask_paths[:self.size]

        masks = []

        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
            mask = cv2.resize(mask, SimpleDataLoader.IMAGE_SIZE)
            mask = np.expand_dims(mask, axis=2)
            masks.append(mask)

        self.masks = np.array(masks, dtype=np.float32)

        return self.masks

    def get_images_masks(self) -> dict:
        return {
            "images": self.get_images(),
            "masks": self.get_masks()
        }
