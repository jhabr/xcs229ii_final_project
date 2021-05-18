import os

from constants import TEST_DIR
from utils.helper import NotebookHelper
import numpy as np
from utils.metrics import Metrics
import segmentation_models as sm


def calculate_full_resolution_metrics(azure_model: sm.Unet):
    test_images, test_masks = NotebookHelper().load_images(TEST_DIR, load_masks=True)
    predicted_masks_list = []

    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0)
        predicted_mask = azure_model.predict(test_image).round().squeeze(axis=0)
        predicted_masks_list.append(predicted_mask)

    predicted_masks = np.stack(predicted_masks_list)

    assert len(predicted_masks.shape) == len(test_masks)
    assert predicted_masks.shape[-1] == 1

    metrics = Metrics().calculate_batch(
        masks=test_masks,
        predicted_masks=predicted_masks
    )

    print(metrics)


if __name__ == '__main__':
    print("Calculating metrics on full resolution test images...")
    # AZURE_PATH = "/home/scpdxcs/projects/xcs229ii_final_project/archive"
    AZURE_PATH = "/Users/jh/ai/courses/stanford/3-ML_RL/final_project/xcs229ii_final_project/code/experiments/export/azure"
    # adapt architecture to experiment
    azure_model = sm.Unet(encoder_weights=None, activation='sigmoid')
    # load the correct weights from training
    print("-- Loading model weights...")
    azure_model.load_weights(os.path.join(AZURE_PATH, "2021_05_14_2245", "baseline", "baseline_01_weights_only.h5"))
    # calculate metrics on full resolution test images
    print("-- Calculating metrics...")
    calculate_full_resolution_metrics(azure_model=azure_model)
    print("Done.")
