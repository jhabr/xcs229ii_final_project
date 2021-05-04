import os

import segmentation_models as sm
import tensorflow as tf

from baseline.dataloader import DataLoader, SimpleDataLoader
from baseline.dataset import Dataset
from constants import PROJECT_DIR, TRAIN_DIR, VALIDATION_DIR


class Trainer:
    BACKBONE = 'resnet34'
    BATCH_SIZE = 1
    EPOCHS = 1

    def __init__(self):
        self.x_train_dir = os.path.join(TRAIN_DIR, 'images')
        self.y_train_dir = os.path.join(TRAIN_DIR, 'masks')
        self.x_validation_dir = os.path.join(VALIDATION_DIR, 'images')
        self.y_validation_dir = os.path.join(VALIDATION_DIR, 'masks')


    def __get_training_dataset(self) -> Dataset:
        return Dataset(
            self.x_train_dir,
            self.y_train_dir,
            #preprocessing=self.preprocessing
        )

    def __get_training_data(self, dataset_size=None) -> dict:
        return SimpleDataLoader(
            backbone=Trainer.BACKBONE,
            images_path=self.x_train_dir,
            mask_path=self.y_train_dir,
            size=dataset_size
        ).get_images_masks()

    def __get_validation_dataset(self) -> Dataset:
        return Dataset(
            self.x_validation_dir,
            self.y_validation_dir,
            # preprocessing=self.preprocessing
        )

    def __get_validation_data(self, dataset_size=None) -> dict:
        return SimpleDataLoader(
            backbone=Trainer.BACKBONE,
            images_path=self.x_validation_dir,
            mask_path=self.y_validation_dir,
            size=dataset_size
        ).get_images_masks()

    def __get_train_data_loader(self) -> DataLoader:
        return DataLoader(self.__get_training_dataset(), batch_size=Trainer.BATCH_SIZE, shuffle=True)

    def __get_valid_data_loader(self) -> DataLoader:
        return DataLoader(self.__get_validation_dataset(), batch_size=1, shuffle=False)

    def get_model(self) -> sm.Unet:
        model = sm.Unet(Trainer.BACKBONE, encoder_weights='imagenet', activation='sigmoid')
        model.compile(
            tf.keras.optimizers.Adam(3e-5),
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score]
        )

        return model

    def __get_callbacks(self) -> list:
        model_path = os.path.join(PROJECT_DIR, "baseline", "export", "baseline.h5")
        return [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                save_weights_only=True,
                save_best_only=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau()
        ]

    def train_from_dataloader(self):
        train_data_loader = self.__get_train_data_loader()
        valid_data_loader = self.__get_valid_data_loader()

        return self.get_model().fit(
            train_data_loader,
            steps_per_epoch=len(train_data_loader),
            epochs=Trainer.EPOCHS,
            callbacks=self.__get_callbacks(),
            validation_data=valid_data_loader,
            validation_steps=len(valid_data_loader),
            verbose=2
        )

    def train_from_simple_dataloader(self, dataset_size=None, batch_size=None, epochs=None):
        training_data = self.__get_training_data(dataset_size=dataset_size)
        validation_data = self.__get_validation_data(dataset_size=dataset_size)

        return self.get_model().fit(
            x=training_data['images'],
            y=training_data['masks'],
            batch_size=batch_size if batch_size else Trainer.BATCH_SIZE,
            steps_per_epoch=len(training_data['images']) // Trainer.BATCH_SIZE,
            epochs=epochs if epochs else Trainer.EPOCHS,
            validation_data=(validation_data["images"], validation_data["masks"]),
            validation_steps=len(validation_data['images']) // Trainer.BATCH_SIZE,
            callbacks=self.__get_callbacks(),
            verbose=2
        )
