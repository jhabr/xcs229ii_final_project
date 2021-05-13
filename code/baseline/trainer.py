import os

import segmentation_models as sm
import tensorflow as tf

from constants import TRAIN_DIR, VALIDATION_DIR, EXPORT_DIR
from utils.dataloader import SimpleDataLoader


class Trainer:
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 3e-5

    def __init__(self, model: sm.Unet, backbone=None):
        self.x_train_dir = os.path.join(TRAIN_DIR, 'images')
        self.y_train_dir = os.path.join(TRAIN_DIR, 'masks')
        self.x_validation_dir = os.path.join(VALIDATION_DIR, 'images')
        self.y_validation_dir = os.path.join(VALIDATION_DIR, 'masks')
        self.model = model
        self.backbone = backbone

    def get_training_data(self, dataset_size=None, image_resolution=None) -> dict:
        return SimpleDataLoader(
            backbone=self.backbone,
            images_folder_path=self.x_train_dir,
            masks_folder_path=self.y_train_dir,
            resize_to=image_resolution,
            size=dataset_size
        ).get_images_masks()

    def get_validation_data(self, dataset_size=None, image_resolution=None) -> dict:
        return SimpleDataLoader(
            backbone=self.backbone,
            images_folder_path=self.x_validation_dir,
            masks_folder_path=self.y_validation_dir,
            resize_to=image_resolution,
            size=dataset_size
        ).get_images_masks()

    def compile_model(self, learning_rate=LEARNING_RATE) -> sm.Unet:
        if self.model is None:
            raise AttributeError("Model must not be None.")
        self.model.compile(
            tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=sm.losses.bce_jaccard_loss,
            metrics=[
                sm.metrics.iou_score,
                sm.metrics.f1_score,
                sm.metrics.f2_score,
                sm.metrics.recall,
                sm.metrics.precision
            ]
        )

        return self.model

    def __get_callbacks(self, identifier: str = "-") -> list:
        model_name = f"{identifier}_baseline.h5"
        model_path = os.path.join(EXPORT_DIR, model_name)
        return [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                save_weights_only=True,
                save_best_only=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=4,
                verbose=1,
                epsilon=1e-5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=1e-5,
                verbose=1,
                patience=8,
                restore_best_weights=True
            )
        ]

    def train_from_simple_dataloader(
            self, identifier: str, learning_rate=LEARNING_RATE, dataset_size=None, batch_size=None, epochs=None,
            image_resolution=(512, 512)
    ):
        training_data = self.get_training_data(dataset_size=dataset_size, image_resolution=image_resolution)
        validation_data = self.get_validation_data(dataset_size=dataset_size, image_resolution=image_resolution)
        batch_size = batch_size if batch_size else Trainer.BATCH_SIZE

        return self.compile_model(learning_rate=learning_rate).fit(
            x=training_data['images'],
            y=training_data['masks'],
            batch_size=batch_size,
            steps_per_epoch=len(training_data['images']) // batch_size,
            epochs=epochs if epochs else Trainer.EPOCHS,
            validation_data=(validation_data["images"], validation_data["masks"]),
            validation_steps=len(validation_data['images']) // batch_size,
            callbacks=self.__get_callbacks(identifier=identifier),
            verbose=2
        )
