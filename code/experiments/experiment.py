import os
import pickle
from datetime import datetime

import segmentation_models as sm
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from baseline.baselinetrainer import BaselineTrainer
from constants import EXPORT_DIR, TRAIN_DIR, VALIDATION_DIR
from transformers.trans_u_net.datasets.isic_dataset import ISICDataset
from transformers.trans_u_net.my_trainer import TransUNetLightning


class Experiment:

    def __init__(self, identifier):
        self.identifier = identifier

    def run(self, model: sm.Unet, backbone: str = None, dataset_size=None, batch_size=None, epochs=None,
            image_resolution=None):
        print(f"{datetime.now()}: Starting << experiment {self.identifier} >> training...")
        history = self._configure_training(
            model=model, backbone=backbone, dataset_size=dataset_size, batch_size=batch_size, epochs=epochs,
            image_resolution=image_resolution
        )

        history_path = os.path.join(EXPORT_DIR, f"{self.identifier}_train_history.pkl")
        print(f"Exporting history to {history_path}...")
        pickle.dump(history.history, open(history_path, "wb"))
        print("Done.")

        print(f"Verifying history content...")
        loaded_history = pickle.load(open(history_path, "rb"))
        assert history.history["loss"] == loaded_history["loss"]
        print("Done.")
        print(f"{datetime.now()}: Done training.")

    def _configure_training(
            self, model: sm.Unet, backbone: str = None, dataset_size=None, batch_size=None, epochs=None,
            image_resolution=None
    ):
        raise NotImplementedError


class BaselineExperiment(Experiment):
    def _configure_training(
            self, model: sm.Unet, backbone: str = None, dataset_size=None, batch_size=None, epochs=None,
            image_resolution=None
    ):
        return BaselineTrainer(model=model, backbone=backbone).train_from_simple_dataloader(
            identifier=self.identifier,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            image_resolution=image_resolution
        )


class TransUNetExperiment:
    def __init__(self, identifier, model, dataset_size=None, batch_size=24, resize_to=(224, 224), epochs=100,
                 callbacks=None):
        self.identifier = identifier
        self.model = model
        self.train_loader = DataLoader(
            dataset=ISICDataset(resize_to=resize_to, size=dataset_size, image_dir=TRAIN_DIR),
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        self.validation_loader = DataLoader(
            dataset=ISICDataset(resize_to=resize_to, size=dataset_size, image_dir=VALIDATION_DIR),
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        self.trainer = Trainer(
            default_root_dir=os.path.join(EXPORT_DIR, "trans_u_net", "lightning"),
            max_epochs=epochs,
            callbacks=callbacks
        )
        self.lightning_model = TransUNetLightning(model=self.model)

    def run(self):
        print(f"{datetime.now()}: Starting << experiment {self.identifier} >> training...")
        self.trainer.fit(self.lightning_model, self.train_loader, self.validation_loader)
        print(f"{datetime.now()}: Done training.")
