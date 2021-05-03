import cv2
import albumentations as A
import segmentation_models as sm
import numpy as np


class ImagePreprocessor:
    IMAGE_SIZE = (512, 512)

    def __init__(self, backbone):
        self.preprocessing = self.__default_preprocessing(sm.get_preprocessing(backbone))

    def perform_image_default(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        image = cv2.resize(image, ImagePreprocessor.IMAGE_SIZE)
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']

        return image

    def perform_mask_default(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
        mask = cv2.resize(mask, ImagePreprocessor.IMAGE_SIZE)
        mask = np.expand_dims(mask, axis=2)
        return mask

    def __default_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """
        _transform = [A.Lambda(image=preprocessing_fn)]
        return A.Compose(_transform)
