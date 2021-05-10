import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")
BASELINE_DIR = os.path.join(PROJECT_DIR, "baseline")
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiments")
EXPORT_DIR = os.path.join(EXPERIMENT_DIR, "export")
