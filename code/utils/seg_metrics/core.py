import numpy as np


class BinarySegmentationMetric:

    def __init__(self):
        self.true_positives = []
        self.true_negatives = []
        self.false_positives = []
        self.false_negatives = []

    def calculate(self, mask, predicted_mask) -> dict:
        self.__calc_simple_metrics(mask, predicted_mask)
        results = {
            "n_images": 1,
            "n_true_positives": self.tp,
            "n_true_negatives": self.tn,
            "n_false_positives": self.fp,
            "n_false_negatives": self.fn,
            "iou": self.per_object_iou(mask, predicted_mask),
            "jaccard": self.jaccard,
            "dice": self.dice,
            "f1_score": self.f1_score,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "accuracy": self.accuracy
        }

        return results

    def calculate_batch(self, masks, predicted_masks) -> dict:
        assert masks.shape == predicted_masks.shape
        assert len(masks.shape) == 4

        iou = []
        for i in range(len(masks)):
            mask = masks[i]
            predicted_mask = predicted_masks[i]
            results = self.calculate(mask, predicted_mask)
            iou.append(results["iou"])

        results = {
            "n_images": len(masks),
            "n_true_positives": self.tp,
            "n_true_negatives": self.tn,
            "n_false_positives": self.fp,
            "n_false_negatives": self.fn,
            "iou": np.mean(iou),
            "jaccard": self.jaccard,
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

        true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

        for i in range(height):
            for j in range(width):
                mask_pixel_value = mask[i][j]
                predicted_mask_pixel_value = predicted_mask[i][j]
                if mask_pixel_value == predicted_mask_pixel_value:
                    if mask_pixel_value == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if predicted_mask_pixel_value == 0:
                        false_negatives += 1
                    else:
                        false_positives += 1

        assert true_positives + \
               true_negatives + \
               false_positives + \
               false_negatives == height * width

        self.true_positives.append(true_positives)
        self.true_negatives.append(true_negatives)
        self.false_positives.append(false_positives)
        self.false_negatives.append(false_negatives)

    @property
    def tp(self):
        return sum(self.true_positives)

    @property
    def tn(self):
        return sum(self.true_negatives)

    @property
    def fp(self):
        return sum(self.false_positives)

    @property
    def fn(self):
        return sum(self.false_negatives)

    def per_object_iou(self, mask, predicted_mask):
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
