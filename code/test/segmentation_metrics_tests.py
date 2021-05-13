import os
import unittest

from constants import TRAIN_DIR
from utils.dataloader import SimpleDataLoader
from utils.metrics import Metrics
import segmentation_models as sm
import numpy as np


class SegmentationMetricsTests(unittest.TestCase):

    def setUp(self):
        self.simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(256, 256),
            size=3
        )
        self.model = sm.Unet(activation='sigmoid')
        self.mask = self.__create_mask()
        self.predicted_mask = self.__create_predicted_mask()

    def __create_mask(self):
        true_mask = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]
        )
        true_mask = np.expand_dims(true_mask, axis=2)
        return true_mask

    def __create_predicted_mask(self):
        predicted_mask = np.array(
            [[1., 1., 1., 1., 1.],  # 5 true positives
             [1., 0., 0., 0., 0.],  # 1 true positive, 4 false negatives
             [0., 0., 0., 0., 0.],  # 5 true negatives
             [0., 0., 0., 0., 0.],  # 5 true negatives
             [1., 1., 1., 0., 0.]]  # 3 false positives, 2 true negatives
        )
        predicted_mask = np.expand_dims(predicted_mask, axis=2)
        return predicted_mask

    def tearDown(self):
        self.simple_data_loader = None
        self.model = None

    def test_true_positives(self):
        """TP: pixels correctly segmented as foreground"""
        results = Metrics().calculate(self.mask, self.predicted_mask)
        self.assertEqual(results.n_true_positives, 6)

    def test_true_negatives(self):
        """TN: pixels correctly detected as background"""
        results = Metrics().calculate(self.mask, self.predicted_mask)
        self.assertEqual(results.n_true_negatives, 15)

    def test_false_positives(self):
        """FP: pixels falsely segmented as foreground"""
        results = Metrics().calculate(self.mask, self.predicted_mask)
        self.assertEqual(results.n_false_positives, 3)

    def test_false_negatives(self):
        """FN: pixels falsely detected as background"""
        results = Metrics().calculate(self.mask, self.predicted_mask)
        self.assertEqual(results.n_false_negatives, 4)

    def test_calculate_segmentation_metrics(self):
        data = self.simple_data_loader.get_images_masks()
        image = data["images"][0]
        mask = data["masks"][0]

        test_image = np.expand_dims(image, axis=0)
        predicted_mask = self.model.predict(test_image).round().squeeze(axis=0)

        metrics = Metrics().calculate(
            mask=mask,
            predicted_mask=predicted_mask
        )
        self.assertEqual(metrics.n_images, 1)
        self.assertEqual(metrics.jaccard, 0.0)

    def test_calculate_batch_segmentation_metrics(self):
        data = self.simple_data_loader.get_images_masks()
        images = data["images"]
        masks = data["masks"]
        self.assertEqual(len(images), 3)
        self.assertEqual(len(masks), 3)

        predicted_masks_list = []

        for test_image in images:
            test_image = np.expand_dims(test_image, axis=0)
            predicted_mask = self.model.predict(test_image).round().squeeze(axis=0)
            predicted_masks_list.append(predicted_mask)

        predicted_masks = np.stack(predicted_masks_list)

        metrics = Metrics().calculate_batch(
            masks=masks,
            predicted_masks=predicted_masks
        )
        self.assertEqual(metrics.n_images, 3)
        self.assertEqual(metrics.jaccard, 0.0)
        # self.assertEqual(metrics.n_false_negatives, 3)
        # self.assertEqual(metrics.n_false_positives, 165)
        # self.assertEqual(metrics.n_pred_labels, 12)
        # self.assertEqual(metrics.n_true_labels, 12)
        # self.assertEqual(metrics.n_true_positives, 0)


if __name__ == '__main__':
    unittest.main()
