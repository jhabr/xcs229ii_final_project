import unittest

from baseline.trainer import Trainer
from utils.augmentation import DataAugmentation, AdvancedHairAugmentation
from utils.helper import Visualisation


class DataAugmentationTest(unittest.TestCase):

    def setUp(self):
        self.image = Trainer().get_training_data(dataset_size=1)["images"][0]

    def tearDown(self):
        self.image = None

    def test_advanced_augmentation(self):
        augmented_image = DataAugmentation().apply_advanced(self.image)
        self.assertEqual(augmented_image.shape, self.image.shape)
        Visualisation().plot_images(image=augmented_image.squeeze())

    def test_hair_augmentation(self):
        augmented_image = AdvancedHairAugmentation().apply(self.image)
        self.assertEqual(augmented_image.shape, self.image.shape)
        Visualisation().plot_images(image=augmented_image.squeeze())


if __name__ == '__main__':
    unittest.main()
