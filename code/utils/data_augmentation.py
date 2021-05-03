import os

import cv2
from albumentations import ImageOnlyTransform
import numpy as np

from constants import DATA_DIR


class AdvancedHairAugmentation(ImageOnlyTransform):
    """
    Hair augmentation:
    https://arxiv.org/abs/1904.09169

    https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176

    https://www2.cs.sfu.ca/~hamarneh/software/hairsim/Software.html

    https://www.kaggle.com/nroman/melanoma-hairs
    https://www.kaggle.com/shogoaraki/whitehairs

    """

    def __init__(self, hairs: int = 5, always_apply=False, p=0.5):
        self.hairs = hairs
        self.hairs_folder = os.path.join(DATA_DIR, "augmentation", "hairs")
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        no_hairs = np.random.randint(0, self.hairs)

        if not no_hairs:
            return image

        height, width, _ = image.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(no_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, np.random.choice(hair_images)))
            hair = cv2.flip(hair, np.random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, np.random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hairs image width and height
            roi_ho = np.random.randint(0, image.shape[0] - hair.shape[0])
            roi_wo = np.random.randint(0, image.shape[1] - hair.shape[1])
            roi = image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            image2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(image2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            image_background = cv2.bitwise_and(roi, roi, mask=mask_inv)
            hair_foreground = cv2.bitwise_and(hair, hair, mask=mask)

            dst = cv2.add(image_background, hair_foreground, dtype=cv2.CV_64F)
            image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return image

    def get_params_dependent_on_targets(self, params):
        super().get_params_dependent_on_targets(params)

    def get_transform_init_args_names(self):
        super().get_transform_init_args_names()
