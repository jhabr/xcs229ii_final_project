import os

import cv2

from constants import DATA_DIR
from utils.augmentation import DataAugmentation, AdvancedHairAugmentation
from utils.dataloader import SimpleDataLoader


class ImageGenerator:
    ORIGINAL_IMAGE_DIR = os.path.join(DATA_DIR, "original")
    GENERATED_IMAGE_DIR = os.path.join(DATA_DIR, "generated")

    def __init__(self):
        self.data_augmentation = DataAugmentation()
        self.hair_augmentation = AdvancedHairAugmentation()

    def generate_images(self, size=300):
        print("Reading original mages...")

        simple_data_loader = SimpleDataLoader(
            images_folder_path=os.path.join(ImageGenerator.ORIGINAL_IMAGE_DIR, "images"),
            masks_folder_path=os.path.join(ImageGenerator.ORIGINAL_IMAGE_DIR, "masks"),
            normalize=False,
            resize=False,
            random_selection=True,
            size=size
        )

        images_masks = simple_data_loader.get_images_masks()

        print(f"Generating {size} augmented images...")

        for index in range(size):
            image = images_masks["images"][index]
            mask = images_masks["masks"][index]
            print(f"-- Augmenting image {index} with shape: {image.shape}, mask with shape: {mask.shape}")

            advanced_augmentations = self.data_augmentation.apply_advanced(image, mask)
            augmented_image = advanced_augmentations["image"]
            augmented_mask = advanced_augmentations["mask"]

            if index % 2 == 1:
                augmented_image = self.hair_augmentation.apply(augmented_image)

            augmented_image_file_path = os.path.join(
                ImageGenerator.GENERATED_IMAGE_DIR,
                "images",
                f"augmented_image_{index}.jpg"
            )
            augmented_mask_file_name = os.path.join(
                ImageGenerator.GENERATED_IMAGE_DIR,
                "masks",
                f"augmented_mask_{index}.png"
            )
            cv2.imwrite(augmented_image_file_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(augmented_mask_file_name, cv2.cvtColor(augmented_mask, cv2.COLOR_RGB2BGR))

        print("Done.")


if __name__ == '__main__':
    ImageGenerator().generate_images(size=606)
