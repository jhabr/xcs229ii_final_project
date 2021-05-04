import cv2
import numpy as np


class ImagePreprocessor:
    IMAGE_SIZE = (512, 512)

    def apply_image_default(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalization step to get numbers between (0, 1)
        image /= 255.0
        image = cv2.resize(image, ImagePreprocessor.IMAGE_SIZE)

        return image

    def apply_mask_default(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # normalization step to get numbers between (0, 1)
        mask /= 255.0
        mask = cv2.resize(mask, ImagePreprocessor.IMAGE_SIZE)
        mask = np.expand_dims(mask, axis=2)
        return mask
