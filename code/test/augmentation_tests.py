import os
import unittest

import numpy as np

from constants import TRAIN_DIR
from utils.augmentation import DataAugmentation, AdvancedHairAugmentation
from utils.dataloader import SimpleDataLoader
from utils.helper import Visualisation


class DataAugmentationTest(unittest.TestCase):

    def setUp(self):
        self.training_data = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            normalize=False,
            size=1
        ).get_images_masks()
        self.image = (self.training_data["images"][0] * 255.0).astype(np.uint64)
        self.mask = (self.training_data["masks"][0] * 255.0).astype(np.uint64)

    def tearDown(self):
        self.training_data = None
        self.image = None
        self.mask = None

    def test_advanced_augmentation(self):
        advanced_augmentations = DataAugmentation().apply_advanced(image=self.image, mask=self.mask)
        augmented_image = advanced_augmentations["image"]
        augmented_mask = advanced_augmentations["mask"]
        self.assertEqual(augmented_image.shape, self.image.shape)
        self.assertEqual(augmented_mask.shape, self.mask.shape)
        Visualisation().plot_images(image=augmented_image.squeeze(), mask=augmented_mask.squeeze())

    def test_hair_augmentation(self):
        augmented_image = AdvancedHairAugmentation().apply(self.image)
        self.assertEqual(augmented_image.shape, self.image.shape)
        Visualisation().plot_images(image=augmented_image.squeeze())


if __name__ == '__main__':
    unittest.main()
