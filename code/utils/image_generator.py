import os
import random

import cv2

from constants import DATA_DIR
from utils.augmentation import DataAugmentation, AdvancedHairAugmentation
from utils.dataloader import SimpleDataLoader


class ImageGenerator:
    ORIGINAL_IMAGE_DIR = os.path.join(DATA_DIR, "original")
    GENERATED_IMAGE_DIR = os.path.join(DATA_DIR, "generated")

    def __init__(self):
        self.simple_data_loader = SimpleDataLoader(
            images_path=os.path.join(ImageGenerator.ORIGINAL_IMAGE_DIR, "images"),
            mask_path=os.path.join(ImageGenerator.ORIGINAL_IMAGE_DIR, "masks"),
            normalize=False,
            resize=False,
            size=3
        )
        self.data_augmentation = DataAugmentation()
        self.hair_augmentation = AdvancedHairAugmentation()

    def generate_images(self, size=300):
        print("Reading original mages...")

        images_masks = self.simple_data_loader.get_images_masks()
        dataset_size = len(images_masks)

        print(f"Generating {size} augmented images...")

        for i in range(size):
            index = random.randint(0, dataset_size)
            image = images_masks["images"][index]
            mask = images_masks["masks"][index]

            advanced_augmentations = self.data_augmentation.apply_advanced(image, mask)
            augmented_image = advanced_augmentations["image"]
            augmented_mask = advanced_augmentations["mask"]

            if index % 2 == 1:
                augmented_image = self.hair_augmentation.apply(augmented_image)

            augmented_image_file_path = os.path.join(
                ImageGenerator.GENERATED_IMAGE_DIR,
                "images",
                f"augmented_image_{i}.jpg"
            )
            augmented_mask_file_name = os.path.join(
                ImageGenerator.GENERATED_IMAGE_DIR,
                "masks",
                f"augmented_mask_{i}.png"
            )
            cv2.imwrite(augmented_image_file_path, augmented_image)
            cv2.imwrite(augmented_mask_file_name, augmented_mask)

        print("Done.")


if __name__ == '__main__':
    ImageGenerator().generate_images(size=3)
