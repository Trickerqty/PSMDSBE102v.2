import time
import ray
from datetime import datetime
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune import Tuner
from ray.train.torch import TorchTrainer
from ray.air.config import RunConfig, CheckpointConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from train import train_loop_per_worker
from config import DEFAULT_TRAIN_CONFIG, MLFLOW_TRACKING_URI
from data import load_clean_data, create_ray_datasets, preprocess

# === MLflow Callback ===
experiment_name = f"movie-sentiment-tune-{int(time.time())}"
mlflow_callback = MLflowLoggerCallback(
    tracking_uri=MLFLOW_TRACKING_URI,
    experiment_name=experiment_name,
    save_artifact=True
)

# === Checkpoint and Run Config ===
checkpoint_config = CheckpointConfig(
    num_to_keep=1,
    checkpoint_score_attribute="val_loss",
    checkpoint_score_order="min"
)

run_config = RunConfig(
    callbacks=[mlflow_callback],
    checkpoint_config=checkpoint_config,
    name=f"Movie_Sentiment_Tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# === Load and preprocess data ===
df = load_clean_data(sample_size=250)
train_ds, val_ds = create_ray_datasets(df)
train_ds = train_ds.map_batches(preprocess, batch_format="pandas")
val_ds = val_ds.map_batches(preprocess, batch_format="pandas")

# === Hyperparameter Search Space ===
initial_params = [{"train_loop_config": DEFAULT_TRAIN_CONFIG}]
search_alg = HyperOptSearch(points_to_evaluate=initial_params)
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

param_space = {
    "train_loop_config": {
        "dropout_p": tune.uniform(0.3, 0.9),
        "lr": tune.loguniform(1e-5, 5e-4),
        "lr_factor": tune.uniform(0.1, 0.9),
        "lr_patience": tune.randint(1, 6),
        "num_epochs": 3,
        "batch_size": 32,
        "num_classes": 3
    }
}

scheduler = AsyncHyperBandScheduler(max_t=6, grace_period=2)

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={},
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    datasets={"train": train_ds, "val": val_ds},
)

tuner = Tuner(
    trainable=trainer,
    run_config=run_config,
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=2
    ),
    param_space=param_space
)

results = tuner.fit()
print("\nTuning complete. Best result:")
print(results.get_best_result(metric="val_loss", mode="min").metrics)