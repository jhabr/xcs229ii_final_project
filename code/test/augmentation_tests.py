import unittest

from baseline.trainer import Trainer
from utils.augmentation import DataAugmentation, AdvancedHairAugmentation
from utils.helper import Visualisation


class DataAugmentationTest(unittest.TestCase):

    def test_advanced_augementation(self):
        image = Trainer().get_training_data(dataset_size=1)["images"][0]
        augmented_image = DataAugmentation().apply_advanced(image)
        assert augmented_image.shape == image.shape
        Visualisation().show(image=augmented_image.squeeze())

    def test_hair_augementation(self):
        image = Trainer().get_training_data(dataset_size=1)["images"][0]
        augmented_image = AdvancedHairAugmentation().apply(image)
        assert augmented_image.shape == image.shape
        Visualisation().show(image=augmented_image.squeeze())
