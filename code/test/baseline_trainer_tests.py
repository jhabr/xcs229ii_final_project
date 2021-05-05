import unittest

from baseline.trainer import Trainer


class TrainerTests(unittest.TestCase):

    def test_trainer(self):
        history = Trainer().train_from_simple_dataloader(dataset_size=1, batch_size=1, epochs=1)
        self.assertEqual(len(history.history["loss"]), 1)
        self.assertEqual(len(history.history["val_loss"]), 1)
