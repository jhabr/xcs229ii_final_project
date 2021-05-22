import os

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from datetime import datetime

from constants import EXPORT_DIR, TRAIN_DIR, VALIDATION_DIR
from transformers.trans_u_net.datasets.isic_dataset import ISICDataset
from transformers.trans_u_net.my_trainer import TransUNetLightning


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