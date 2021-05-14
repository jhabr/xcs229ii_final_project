from utils.segmentation_metrics.core import BinarySegmentationMetric


class Metrics:

    def calculate(self, mask, predicted_mask, jaccard_similarity_index_threshold=0.65):
        assert mask is not None and predicted_mask is not None

        binary_segmentation_metric = BinarySegmentationMetric(
            jaccard_similarity_index_threshold=jaccard_similarity_index_threshold
        )
        results = binary_segmentation_metric.calculate(mask=mask, predicted_mask=predicted_mask)
        return results

    def calculate_batch(self, masks, predicted_masks, jaccard_similarity_index_threshold=0.65):
        assert masks is not None and predicted_masks is not None

        binary_segmentation_metric = BinarySegmentationMetric(
            jaccard_similarity_index_threshold=jaccard_similarity_index_threshold
        )

        results = binary_segmentation_metric.calculate_batch(masks=masks, predicted_masks=predicted_masks)
        return results
