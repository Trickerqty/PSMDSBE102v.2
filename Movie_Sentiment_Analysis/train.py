import os
import random
import numpy as np
import torch
import torch.nn as nn

from ray import train
from ray.train import report
from ray.train import get_dataset_shard, Checkpoint, get_context
from ray.train.torch import TorchCheckpoint, TorchTrainer, prepare_model
from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
from models import BertClassifier
from utils import custom_collate_fn
from config import DEFAULT_TRAIN_CONFIG
from data import load_clean_data, create_ray_datasets, preprocess
from datetime import datetime
import ray


# === Training Logic ===
def train_loop_per_worker(config):
    def set_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seeds()

    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    train_ds = get_dataset_shard("train")
    val_ds = get_dataset_shard("val")

    model = prepare_model(BertClassifier())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)
    criterion = nn.CrossEntropyLoss()

    batch_size_per_worker = batch_size // get_context().get_world_size()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_ds.iter_torch_batches(batch_size=batch_size_per_worker, collate_fn=custom_collate_fn):
            optimizer.zero_grad()
            output = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(output, batch["labels"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_ds.iter_torch_batches(batch_size=batch_size_per_worker, collate_fn=custom_collate_fn):
                output = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(output, batch["labels"])
                val_loss += loss.item()

        scheduler.step(val_loss)

        metrics = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())
        report(metrics=metrics, checkpoint=checkpoint)

# === Entry point to run training directly ===
def train():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    df = load_clean_data()
    train_ds, val_ds = create_ray_datasets(df)
    train_ds = train_ds.map_batches(preprocess, batch_format="pandas")
    val_ds = val_ds.map_batches(preprocess, batch_format="pandas")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = RunConfig(
        name=f"Movie_Sentiment_Train_Run_{timestamp}",
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min"
        )
    )

    # Override training config
    config = DEFAULT_TRAIN_CONFIG.copy()
    config["num_epochs"] = 2
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds}
    )

    results = trainer.fit()
    print("\nTraining complete. Final metrics:")
    print(results.metrics)
    print("Best checkpoint path:", results.checkpoint.path)

if __name__ == "__main__":
    train()
