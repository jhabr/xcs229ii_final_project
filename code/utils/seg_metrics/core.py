import numpy as np


class BinarySegmentationMetric:

    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def calculate(self, mask, predicted_mask) -> dict:
        self.__calc_simple_metrics(mask, predicted_mask)
        results = {
            "n_images": 1,
            "n_true_positives": self.true_positives,
            "n_true_negatives": self.true_negatives,
            "n_false_positives": self.false_positives,
            "n_false_negatives": self.false_negatives,
            "iou": self.iou,
            "jaccard_index": self.jaccard,
            "dice": self.dice,
            "f1_score": self.f1_score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "accuracy": self.accuracy
        }

        return results

    def __calc_simple_metrics(self, mask, predicted_mask):
        assert mask.shape == predicted_mask.shape
        assert len(mask.shape) == len(predicted_mask.shape) == 3
        # assert binary mask
        assert mask.shape[-1] == 1 and predicted_mask.shape[-1] == 1
        # reshape to only 2 dimensions
        mask = mask.squeeze()
        predicted_mask = predicted_mask.squeeze()

        height, width = mask.shape

        for i in range(height):
            for j in range(width):
                mask_pixel_value = mask[i][j]
                predicted_mask_pixel_value = predicted_mask[i][j]
                if mask_pixel_value == predicted_mask_pixel_value:
                    if mask_pixel_value == 1:
                        self.true_positives += 1
                    else:
                        self.true_negatives += 1
                else:
                    if predicted_mask_pixel_value == 0:
                        self.false_negatives += 1
                    else:
                        self.false_positives += 1

        assert self.true_positives + \
               self.true_negatives + \
               self.false_positives + \
               self.false_negatives == height * width

    @property
    def tp(self):
        return self.true_positives

    @property
    def tn(self):
        return self.true_negatives

    @property
    def fp(self):
        return self.false_positives

    @property
    def fn(self):
        return self.false_negatives

    def iou(self, mask, predicted_mask):
        intersection = np.logical_and(mask, predicted_mask)
        union = np.logical_or(mask, predicted_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    @property
    def jaccard(self):
        return self.tp / (self.tp + self.fn + self.fp)

    @property
    def dice(self):
        return (2 * self.tp) / (2 * self.tp + self.fn + self.fp)

    @property
    def f1_score(self):
        return self.tp / (2 * self.tp + self.fn + self.fp)

    @property
    def sensitivity(self):
        return self.tp / (self.tp + self.fn)

    @property
    def specificity(self):
        return self.tn / (self.tn + self.fp)

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)
