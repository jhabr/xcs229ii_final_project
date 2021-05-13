import utils.segmentation_metrics as sm


class Metrics:

    def calculate(self, mask, predicted_mask, strict=True, iou_threshold=0.65):
        assert mask is not None and predicted_mask is not None

        results = sm.calculate(mask, predicted_mask, strict=strict, iou_threshold=iou_threshold).results
        return results

    def calculate_batch(self, masks, predicted_masks, strict=True, iou_threshold=0.65):
        assert masks is not None and predicted_masks is not None

        metrics = []

        for index, image in enumerate(masks):
            mask = masks[index]
            predicted_mask = predicted_masks[index]
            results = self.calculate(mask, predicted_mask, strict=strict, iou_threshold=iou_threshold)
            metrics.append(results)

        return sm.MetricResults.merge(metrics)
