import os
import pickle
from datetime import datetime

from baseline.trainer import Trainer
from constants import BASELINE_DIR

if __name__ == '__main__':
    print(f"{datetime.now()}: Starting training...")
    history = Trainer().train_from_simple_dataloader()
    # history = Trainer().train_from_simple_dataloader(dataset_size=1, batch_size=1, epochs=1)

    history_path = os.path.join(BASELINE_DIR, "export", "train_history.pkl")
    print(f"Exporting history to {history_path}...")
    pickle.dump(history.history, open(history_path, "wb"))
    print("Done.")

    print(f"Verifying history content...")
    loaded_history = pickle.load(open(history_path, "rb"))
    assert history.history["loss"] == loaded_history["loss"]
    print("Done.")
    print(f"{datetime.now()}: Done training.")
