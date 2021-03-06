import os
import unittest

from constants import TRAIN_DIR
from utils.dataloader import SimpleDataLoader
from utils.helper import Visualisation
import numpy as np


class DataLoaderTests(unittest.TestCase):

    def setUp(self):
        self.simple_data_loader = SimpleDataLoader(
            backbone='resnet34',
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(512, 512),
            size=10
        )

    def tearDown(self):
        self.simple_data_loader = None

    def test_load_images(self):
        images = self.simple_data_loader.get_images()
        self.assertEqual(len(images), 10)

    def test_load_masks(self):
        masks = self.simple_data_loader.get_masks()
        self.assertEqual(len(masks), 10)

    def test_load_data(self):
        data = self.simple_data_loader.get_images_masks()
        self.assertEqual(len(data.keys()), 2)
        self.assertEqual(len(data["images"]), 10)
        self.assertEqual(len(data["masks"]), 10)
        self.assertEqual(data["images"][0].shape, (512, 512, 3))
        self.assertEqual(data["masks"][0].shape, (512, 512, 1))
        Visualisation().plot_images(
            image=data["images"][0].squeeze(),
            mask=data["masks"][0].squeeze()
        )
        Visualisation().plot_images(
            image=data["images"][3].squeeze(),
            mask=data["masks"][3].squeeze()
        )
        Visualisation().plot_images(
            image=data["images"][9].squeeze(),
            mask=data["masks"][9].squeeze()
        )

    def test_resize_to(self):
        simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(128, 128),
            size=10
        )

        data = simple_data_loader.get_images_masks()
        self.assertEqual(data["images"][0].shape, (128, 128, 3))
        self.assertEqual(data["masks"][0].shape, (128, 128, 1))
        # make sure that mask has only two colors
        self.assertEqual(len(np.unique(data["masks"][0])), 2)

    def test_load_original_size_data(self):
        simple_data_loader = SimpleDataLoader(
            backbone='resnet34',
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            size=10
        )

        train_data = simple_data_loader.get_images_masks()
        train_image = train_data["images"][0]
        train_mask = train_data["masks"][0]

        self.assertEqual(train_image.shape, (767, 1022, 3))
        self.assertEqual(train_mask.shape, (767, 1022, 1))
        self.assertEqual(train_image.shape[:-1], train_mask.shape[:-1])

        train_image = simple_data_loader.get_images()[0]
        train_mask = simple_data_loader.get_masks()[0]

        self.assertEqual(train_image.shape, (767, 1022, 3))
        self.assertEqual(train_mask.shape, (767, 1022, 1))
        self.assertEqual(train_image.shape[:-1], train_mask.shape[:-1])

    def test_random_indexes(self):
        simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            size=3,
            random_selection=True
        )

        indexes = simple_data_loader.get_random_indexes(3)
        self.assertEqual(len(indexes), 3)
        self.assertNotEqual(indexes[0], indexes[1])
        self.assertNotEqual(indexes[0], indexes[2])
        self.assertNotEqual(indexes[1], indexes[2])

    def test_full_train_dataset(self):
        simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(256, 256)
        )

        data = simple_data_loader.get_images_masks()

        self.assertEqual(len(data["images"]), 2700)
        self.assertEqual(len(data["masks"]), 2700)

    def test_full_masks_train_dataset(self):
        simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(256, 256)
        )

        masks = simple_data_loader.get_masks()

        self.assertEqual(len(masks), 2700)

    def test_full_masks_validation_dataset(self):
        simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(256, 256)
        )

        masks = simple_data_loader.get_masks()

        self.assertEqual(len(masks), 2700)


if __name__ == '__main__':
    unittest.main()
