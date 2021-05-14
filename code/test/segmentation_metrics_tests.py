import os
import unittest

import numpy as np
import segmentation_models as sm

from constants import TRAIN_DIR
from utils.dataloader import SimpleDataLoader
from utils.metrics import Metrics


class SegmentationMetricsTests(unittest.TestCase):

    def setUp(self):
        self.simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(TRAIN_DIR, "images"),
            masks_folder_path=os.path.join(TRAIN_DIR, "masks"),
            resize_to=(256, 256),
            size=3
        )
        self.model = sm.Unet(activation='sigmoid')
        self.metrics = Metrics()

    @property
    def mask(self):
        mask = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]]
        )
        mask = np.expand_dims(mask, axis=2)
        return mask

    @property
    def mask2(self):
        mask2 = np.array(
            [[0., 0., 0., 0., 0.],
             [0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.],
             [0., 1., 1., 1., 0.],
             [0., 0., 0., 0., 0.]]
        )
        mask2 = np.expand_dims(mask2, axis=2)
        return mask2

    @property
    def predicted_mask(self):
        predicted_mask = np.array(
            [[1., 1., 1., 1., 1.],  # 5 true positives
             [1., 0., 0., 0., 0.],  # 1 true positive, 4 false negatives
             [0., 0., 0., 0., 0.],  # 5 true negatives
             [0., 0., 0., 0., 0.],  # 5 true negatives
             [1., 1., 1., 0., 0.]]  # 3 false positives, 2 true negatives
        )
        predicted_mask = np.expand_dims(predicted_mask, axis=2)
        return predicted_mask

    @property
    def predicted_mask2(self):
        predicted_mask2 = np.array(
            [[0., 0., 0., 0., 0.],  # 5 true negatives
             [0., 1., 1., 1., 0.],  # 2 true negatives, 3 true positives
             [0., 1., 1., 1., 0.],  # 2 true negatives, 3 true positives
             [0., 1., 1., 1., 0.],  # 2 true negatives, 3 true positives
             [0., 0., 0., 0., 0.]]  # 5 true negatives
        )
        predicted_mask2 = np.expand_dims(predicted_mask2, axis=2)
        return predicted_mask2

    def tearDown(self):
        self.simple_data_loader = None
        self.model = None
        self.metrics = None

    def test_segmentation_metrics(self):
        metrics = self.metrics.calculate(self.mask, self.predicted_mask)
        self.assertEqual(metrics["n_images"], 1)
        self.assertEqual(metrics["n_true_positives"], 6)
        self.assertEqual(metrics["n_true_negatives"], 12)
        self.assertEqual(metrics["n_false_positives"], 3)
        self.assertEqual(metrics["n_false_negatives"], 4)
        self.assertEqual(metrics["iou_score"], 0.46153846153846156)
        self.assertEqual(metrics["threshold_jaccard_index"], 0.0)
        self.assertEqual(metrics["jaccard_similarity_index"], 0.46153846153846156)
        self.assertEqual(metrics["dice"], 0.631578947368421)
        self.assertEqual(metrics["f1_score"], 0.3157894736842105)
        self.assertEqual(metrics["sensitivity"], 0.6)
        self.assertEqual(metrics["specificity"], 0.8)
        self.assertEqual(metrics["accuracy"], 0.72)

    def test_segmentation_metrics_2(self):
        metrics = self.metrics.calculate(self.mask2, self.predicted_mask2)
        self.assertEqual(metrics["n_images"], 1)
        self.assertEqual(metrics["n_true_positives"], 9)
        self.assertEqual(metrics["n_true_negatives"], 16)
        self.assertEqual(metrics["n_false_positives"], 0)
        self.assertEqual(metrics["n_false_negatives"], 0)
        self.assertEqual(metrics["iou_score"], 1.0)
        self.assertEqual(metrics["threshold_jaccard_index"], 1.0)
        self.assertEqual(metrics["jaccard_similarity_index"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_segmentation_metrics_batch(self):
        masks = np.stack([self.mask, self.mask2])
        predicted_masks = np.stack([self.predicted_mask, self.predicted_mask2])
        metrics = self.metrics.calculate_batch(masks, predicted_masks)
        self.assertEqual(metrics["n_images"], 2)
        self.assertEqual(metrics["n_true_positives"], 15)
        self.assertEqual(metrics["n_true_negatives"], 28)
        self.assertEqual(metrics["n_false_positives"], 3)
        self.assertEqual(metrics["n_false_negatives"], 4)
        self.assertEqual(metrics["iou_score"], 0.7307692307692308)
        self.assertEqual(metrics["threshold_jaccard_index"], 0.5)
        self.assertEqual(metrics["jaccard_similarity_index"], 0.7307692307692308)
        self.assertEqual(metrics["dice"], 0.8157894736842105)
        self.assertEqual(metrics["f1_score"], 0.40789473684210525)
        self.assertEqual(metrics["sensitivity"], 0.8)
        self.assertEqual(metrics["specificity"], 0.9)
        self.assertEqual(metrics["accuracy"], 0.86)

    def test_calculate_segmentation_metrics(self):
        data = self.simple_data_loader.get_images_masks()
        image = data["images"][0]
        mask = data["masks"][0]

        test_image = np.expand_dims(image, axis=0)
        predicted_mask = self.model.predict(test_image).round().squeeze(axis=0)

        metrics = self.metrics.calculate(
            mask=mask,
            predicted_mask=predicted_mask
        )
        self.assertEqual(metrics["n_images"], 1)
        self.assertIsNotNone(metrics["n_true_positives"])
        self.assertIsNotNone(metrics["n_true_negatives"])
        self.assertIsNotNone(metrics["n_false_positives"])
        self.assertIsNotNone(metrics["n_false_negatives"])
        self.assertIsNotNone(metrics["iou_score"])
        self.assertIsNotNone(metrics["threshold_jaccard_index"])
        self.assertIsNotNone(metrics["jaccard_similarity_index"])
        self.assertIsNotNone(metrics["dice"])
        self.assertIsNotNone(metrics["f1_score"])
        self.assertIsNotNone(metrics["sensitivity"])
        self.assertIsNotNone(metrics["specificity"])
        self.assertIsNotNone(metrics["accuracy"])

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

        metrics = self.metrics.calculate_batch(
            masks=masks,
            predicted_masks=predicted_masks
        )
        self.assertEqual(metrics["n_images"], 3)
        self.assertIsNotNone(metrics["n_true_positives"])
        self.assertIsNotNone(metrics["n_true_negatives"])
        self.assertIsNotNone(metrics["n_false_positives"])
        self.assertIsNotNone(metrics["n_false_negatives"])
        self.assertIsNotNone(metrics["iou_score"])
        self.assertIsNotNone(metrics["threshold_jaccard_index"])
        self.assertIsNotNone(metrics["jaccard_similarity_index"])
        self.assertIsNotNone(metrics["dice"])
        self.assertIsNotNone(metrics["f1_score"])
        self.assertIsNotNone(metrics["sensitivity"])
        self.assertIsNotNone(metrics["specificity"])
        self.assertIsNotNone(metrics["accuracy"])


if __name__ == '__main__':
    unittest.main()
