import glob
import os
import random
import shutil

from constants import DATA_DIR, TEST_DIR, VALIDATION_DIR


class DatasetGenerator:
    ORIGINAL_IMAGE_DIR = os.path.join(DATA_DIR, "original")

    def generate_dataset(self, images_folder_path, masks_folder_path, max_size, destination_folder):
        images_paths = sorted(glob.glob(os.path.join(images_folder_path, "*.jpg")))
        masks_paths = sorted(glob.glob(os.path.join(masks_folder_path, "*.png")))

        print("Generating random indexes...")
        # pick random 300 images from the original image distribution
        random_indices = self.__pick_random_images(dataset_size=len(images_paths), max_size=max_size)
        print(f"Moving {max_size} images to {destination_folder}...")
        for random_index in random_indices:
            source_image_path = images_paths[random_index]
            print(f"-- Moving image {source_image_path} and mask...")
            source_mask_path = masks_paths[random_index]
            destination_image_path = os.path.join(destination_folder, "images")
            destination_mask_path = os.path.join(destination_folder, "masks")
            shutil.move(source_image_path, destination_image_path)
            shutil.move(source_mask_path, destination_mask_path)

        print("Done.")

    def __pick_random_images(self, dataset_size, max_size):
        random_indexes = set()

        while len(random_indexes) < max_size:
            random_indexes.add(random.randint(0, dataset_size))

        return random_indexes


if __name__ == '__main__':
    images_path = os.path.join(DatasetGenerator.ORIGINAL_IMAGE_DIR, "images")
    masks_path = os.path.join(DatasetGenerator.ORIGINAL_IMAGE_DIR, "masks")
    DatasetGenerator().generate_dataset(
        images_folder_path=images_path,
        masks_folder_path=masks_path,
        max_size=300,
        destination_folder=VALIDATION_DIR
    )
