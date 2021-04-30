import os

import segmentation_models as sm
import tensorflow as tf

from baseline.data_augmentation import DataAugmentation
from baseline.dataloader import DataLoader, SimpleDataLoader
from baseline.dataset import Dataset
from constants import PROJECT_DIR, TRAIN_DIR, VALIDATION_DIR


class Trainer:
    BACKBONE = 'resnet34'
    BATCH_SIZE = 16
    EPOCHS = 1

    def __get_training_dataset(self) -> Dataset:
        x_train_dir = os.path.join(TRAIN_DIR, 'images')
        y_train_dir = os.path.join(TRAIN_DIR, 'masks')
        return Dataset(
            x_train_dir,
            y_train_dir,
            preprocessing=DataAugmentation().get_preprocessing(sm.get_preprocessing(Trainer.BACKBONE))
        )

    def __get_training_data(self) -> tuple:
        x_train_dir = os.path.join(TRAIN_DIR, 'images')
        y_train_dir = os.path.join(TRAIN_DIR, 'masks')
        return SimpleDataLoader(
            images_path=x_train_dir,
            mask_path=y_train_dir,
            preprocessing=DataAugmentation().get_preprocessing(sm.get_preprocessing(Trainer.BACKBONE))
        ).get_images_masks()

    def __get_validation_dataset(self) -> Dataset:
        x_validation_dir = os.path.join(VALIDATION_DIR, 'images')
        y_validation_dir = os.path.join(VALIDATION_DIR, 'masks')
        return Dataset(
            x_validation_dir,
            y_validation_dir,
            # preprocessing=DataAugmentation().get_preprocessing(sm.get_preprocessing(Trainer.BACKBONE))
        )

    def __get_validation_data(self) -> tuple:
        x_validation_dir = os.path.join(VALIDATION_DIR, 'images')
        y_validation_dir = os.path.join(VALIDATION_DIR, 'masks')
        return SimpleDataLoader(
            images_path=x_validation_dir,
            mask_path=y_validation_dir,
            preprocessing=DataAugmentation().get_preprocessing(sm.get_preprocessing(Trainer.BACKBONE))
        ).get_images_masks()


    def __get_train_data_loader(self) -> DataLoader:
        return DataLoader(self.__get_training_dataset(), batch_size=Trainer.BATCH_SIZE, shuffle=True)

    def __get_valid_data_loader(self) -> DataLoader:
        return  DataLoader(self.__get_validation_dataset(), batch_size=1, shuffle=False)

    def __get_model(self) -> sm.Unet:
        model = sm.Unet(Trainer.BACKBONE, encoder_weights='imagenet', activation='sigmoid')
        model.compile(
            tf.keras.optimizers.Adam(3e-5),
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score, sm.metrics.FScore]
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

        return self.__get_model().fit(
            train_data_loader,
            steps_per_epoch=len(train_data_loader),
            epochs=Trainer.EPOCHS,
            callbacks=self.__get_callbacks(),
            validation_data=valid_data_loader,
            validation_steps=len(valid_data_loader)
        )

    def train_from_simple_dataloader(self):
        training_data = self.__get_training_data()
        validation_data = self.__get_validation_data()

        return self.__get_model().fit(
            x=training_data[0],
            y=training_data[1],
            batch_size=Trainer.BATCH_SIZE,
            steps_per_epoch=len(training_data[0] // Trainer.BATCH_SIZE),
            epochs=Trainer.EPOCHS,
            validation_data=validation_data,
            validation_steps=len(validation_data[0] // Trainer.BATCH_SIZE),
            callbacks=self.__get_callbacks()
        )
