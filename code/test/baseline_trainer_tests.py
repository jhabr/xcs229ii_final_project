import unittest

from baseline.trainer import Trainer
import segmentation_models as sm


class TrainerTests(unittest.TestCase):

    def test_trainer(self):
        trainer = Trainer(model=sm.Unet())
        history = trainer.train_from_simple_dataloader(identifier="00", dataset_size=1, batch_size=1, epochs=1)
        self.assertEqual(len(history.history["loss"]), 1)
        self.assertEqual(len(history.history["val_loss"]), 1)
