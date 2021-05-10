from datetime import datetime
import os
import pickle

from baseline.trainer import Trainer
from constants import BASELINE_DIR
import segmentation_models as sm


class Experiment:

    def __init__(self, identifier):
        self.identifier = identifier

    def run(self, model: sm.Unet, dataset_size=None, batch_size=None, epochs=None, image_resolution=None):
        print(f"{datetime.now()}: Starting << experiment {self.identifier} >> training...")
        history = self._configure_training(
            model=model, dataset_size=dataset_size, batch_size=batch_size, epochs=epochs,
            image_resolution=image_resolution
        )

        history_path = os.path.join(BASELINE_DIR, "export", f"experiment_{self.identifier}_train_history.pkl")
        print(f"Exporting history to {history_path}...")
        pickle.dump(history.history, open(history_path, "wb"))
        print("Done.")

        print(f"Verifying history content...")
        loaded_history = pickle.load(open(history_path, "rb"))
        assert history.history["loss"] == loaded_history["loss"]
        print("Done.")
        print(f"{datetime.now()}: Done training.")

    def _configure_training(self, model: sm.Unet, dataset_size=None, batch_size=None, epochs=None, image_resolution=None):
        raise NotImplementedError


class BaselineExperiment(Experiment):
    def _configure_training(
            self, model: sm.Unet, dataset_size=None, batch_size=None, epochs=None, image_resolution=None
    ):
        return Trainer().train_from_simple_dataloader(
            identifier=self.identifier,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
            model=model,
            image_resolution=image_resolution
        )
