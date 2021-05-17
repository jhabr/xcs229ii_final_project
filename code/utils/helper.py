import os
import numpy as np
import matplotlib.pyplot as plt

from utils.dataloader import SimpleDataLoader


class Visualisation:

    def plot_images(self, **images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()

    def plot_curves(self, history):
        plt.figure(figsize=(30, 15))
        plt.subplot(221)
        plt.plot(history['iou_score'])
        plt.plot(history['val_iou_score'])
        plt.title('IoU Score')
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(222)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # precision & recall
        plt.subplot(223)
        plt.plot(history['precision'])
        plt.plot(history['val_precision'])
        plt.title('Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(224)
        plt.plot(history['recall'])
        plt.plot(history['val_recall'])
        plt.title('Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


class NotebookHelper:
    def load_images(self, image_dir, backbone=None, load_masks=False, resize_to=None, size=None):
        simple_data_loader = SimpleDataLoader(
            backbone=backbone,
            images_folder_path=os.path.join(image_dir, "images"),
            masks_folder_path=os.path.join(image_dir, "masks"),
            resize_to=resize_to,
            size=size
        )

        images = simple_data_loader.get_images()
        masks = None

        if load_masks:
            masks = simple_data_loader.get_masks()

        return images, masks

    def plot_images_masks(self, model, images, masks=None):
        for index, image in enumerate(images):
            image = np.expand_dims(image, axis=0)
            print(f"Image shape: {image.shape}")

            predicted_mask = model.predict(image).round()
            print(f"Predicted mask shape: {predicted_mask.shape}")

            mask = None

            if masks is not None:
                mask = masks[index]
                print(f"Mask shape: {mask.shape}")

            if mask is None:
                Visualisation().plot_images(
                    image=image.squeeze(),
                    predicted_mask=predicted_mask.squeeze(axis=0)
                )
            else:
                Visualisation().plot_images(
                    image=image.squeeze(),
                    predicted_mask=predicted_mask.squeeze(axis=0),
                    mask=mask.squeeze()
                )
