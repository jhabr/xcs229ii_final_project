import utils.segmentation_metrics as sm
import numpy as np

import utils.segmentation_metrics as sm


class Metrics:

    def calculate(self, model, image, mask, strict=True, iou_threshold=0.65):
        assert model is not None and image is not None and mask is not None

        image = np.expand_dims(image, axis=0)
        predicted_mask = model.predict(image).round().squeeze(axis=0)
        results = sm.calculate(mask, predicted_mask, strict=strict, iou_threshold=iou_threshold).results
        return results

    def calculate_batch(self, model, images, masks, strict=True, iou_threshold=0.65):
        assert model is not None and images is not None and masks is not None

        metrics = []

        for index, image in enumerate(images):
            mask = masks[index]
            results = self.calculate(model, image, mask, strict=strict, iou_threshold=iou_threshold)
            metrics.append(results)

        return sm.MetricResults.merge(metrics)
