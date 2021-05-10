from datetime import datetime
import os
import pickle

from baseline.trainer import Trainer
from constants import EXPORT_DIR
import segmentation_models as sm


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
            self, model: sm.Unet, backbone: str = None, dataset_size=None, batch_size=None, epochs=None, image_resolution=None
    ):
        raise NotImplementedError


class BaselineExperiment(Experiment):
    def _configure_training(
            self, model: sm.Unet, backbone: str = None, dataset_size=None, batch_size=None, epochs=None, image_resolution=None
    ):
        return Trainer(model=model, backbone=backbone).train_from_simple_dataloader(
            identifier=self.identifier,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            image_resolution=image_resolution
        )
