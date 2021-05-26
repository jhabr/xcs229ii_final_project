import os

from constants import TEST_DIR
from utils.helper import NotebookHelper
import numpy as np
from utils.metrics import Metrics
import segmentation_models as sm
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def calculate_full_resolution_metrics(azure_model: sm.Unet, resize_to_predict=(256,192)):

    print(f"-- Resizing images with size {resize_to_predict} only for prediction...")
    
    test_images, test_masks = NotebookHelper().load_images(TEST_DIR, load_masks=True, normalize=False)
    print("Loaded images: ", len(test_images), len(test_masks))
       
    predicted_masks_list = []
    resized_mask_list = []
  
    for index in range(len(test_images)):
        test_image = test_images[index]
        mask_image = test_masks[index]
        o_height, o_width = mask_image.shape[:-1]
        test_image = cv2.resize(test_image, resize_to_predict, interpolation=cv2.INTER_NEAREST)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)

        predicted_mask = azure_model.predict(test_image).round().squeeze(axis=0)
        predicted_mask_resized = cv2.resize(predicted_mask, (o_width, o_height), interpolation=cv2.INTER_NEAREST)
        predicted_mask_resized = predicted_mask_resized*255
        predicted_mask_resized = np.uint8(predicted_mask_resized)  
        predicted_masks_list.append(predicted_mask_resized)

    assert len(predicted_masks_list) == len(test_masks)

    metrics = Metrics().calculate_batch_different_sizes(
        masks=test_masks,
        predicted_masks=predicted_masks_list
    )

    print(metrics)

if __name__ == '__main__':
    print("Calculating metrics on full resolution test images...")
    WEIGHTS_FILE_PATH = "/local/mnt/workspace/paragk/classes/xcs229ii/project/ISIC/xcs229ii_final_project/code/experiments/export/17_LR_0P1_B8_weights_only.h5"
    # adapt architecture to experiment
    azure_model = sm.Unet("inceptionv3", encoder_weights="imagenet", activation='sigmoid')
    # load the correct weights from training
    print("-- Loading model weights...")
    azure_model.load_weights(WEIGHTS_FILE_PATH)
    # calculate metrics on full resolution test images
    print("-- Calculating metrics...")
    calculate_full_resolution_metrics(azure_model=azure_model)
    print("Done.")
  

