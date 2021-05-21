import os

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from constants import TRAIN_DIR, VALIDATION_DIR, TRANSFORMER_DIR
from transformers.trans_u_net.datasets.isic_dataset import ISICDataset
from transformers.trans_u_net.losses import JaccardLoss


class TransUNetLightnig(LightningModule):
    def __init__(self, transformer, learning_rate=0.01):
        super().__init__()
        self.transformer = transformer
        self.learning_rate = learning_rate
        # self.bce_loss = SoftBCEWithLogitsLoss()
        self.bce_loss = BCEWithLogitsLoss()
        self.jaccard_loss = JaccardLoss(mode='binary')

    def forward(self, x):
        prediction = self.transformer(x)
        return prediction

    def configure_optimizers(self):
        optimizer = optim.SGD(self.transformer.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer=optimizer,
                factor=0.1,
                patience=1,
                verbose=True,
                eps=1e-5
            ),
            'monitor': 'val_loss'
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        image_batch, label_batch = batch
        outputs = self.transformer(image_batch)
        loss = 0.5 * self.bce_loss(outputs, label_batch) + 0.5 * self.jaccard_loss(outputs, label_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image_batch, label_batch = batch
        outputs = self.transformer(image_batch)
        loss = 0.5 * self.bce_loss(outputs, label_batch) + 0.5 * self.jaccard_loss(outputs, label_batch)
        self.log('val_loss', loss)


def my_trainer_isic(args, transformer, dataset_size=None):
    batch_size = args.batch_size

    train_dataset = ISICDataset(
        resize_to=(args.img_size, args.img_size), size=dataset_size, image_dir=TRAIN_DIR
    )
    valid_dataset = ISICDataset(
        resize_to=(args.img_size, args.img_size), size=dataset_size, image_dir=VALIDATION_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    export_path = os.path.join(TRANSFORMER_DIR, "trans_u_net", "export", "lightning")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            min_delta=1e-5,
            verbose=True,
            patience=8
        ),
        ModelCheckpoint(
            dirpath=export_path,
            filename="{epoch}-{val_loss:.2f}.pth",
            monitor='val_loss',
            verbose=True,
            save_weights_only=False,
            mode='min'
        )
    ]
    lightning_model = TransUNetLightnig(transformer=transformer)
    trainer = Trainer(
        default_root_dir=os.path.join(TRANSFORMER_DIR, "trans_u_net", "export", "lightning"),
        max_epochs=10,
        callbacks=callbacks
    )
    trainer.fit(lightning_model, train_loader, valid_loader)
