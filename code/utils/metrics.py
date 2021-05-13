import utils.segmentation_metrics as sm
import numpy as np


class Metrics:

    def calculate(self, model, images, masks, strict=True, iou_threshold=0.65):
        assert model is not None and images is not None and masks is not None

        metrics = []

        for index, image in enumerate(images):
            image = np.expand_dims(image, axis=0)
            predicted_mask = model.predict(image).round().squeeze(axis=0)
            mask = masks[index]
            results = sm.calculate(mask, predicted_mask, strict=strict, iou_threshold=iou_threshold).results
            metrics.append(results)

        return sm.MetricResults.merge(metrics)
