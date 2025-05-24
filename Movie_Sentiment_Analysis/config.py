import os
from pathlib import Path
import random
import numpy as np
import torch
import mlflow
import time

# === RANDOM SEEDS ===
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === TRACKING CONFIG ===
TIMESTAMP = Path(os.getenv("RUN_TIMESTAMP", "default"))
EXPERIMENT_NAME = f"movie-sentiment-{int(time.time())}"

# === MLflow Directory ===
MODEL_REGISTRY = Path("/tmp/Movie_Sentiment_Analysis")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file:///" + str(MODEL_REGISTRY.resolve())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# === CLASS MAPPING ===
index_to_class = {0: "negative", 1: "neutral", 2: "positive"}
class_to_index = {v: k for k, v in index_to_class.items()}

# === TRAINING PARAMS ===
DEFAULT_TRAIN_CONFIG = {
    "dropout_p": 0.3,
    "lr": 2e-5,
    "lr_factor": 0.8,
    "lr_patience": 2,
    "num_epochs": 6,
    "batch_size": 32,
    "num_classes": 3
}