import torch
import numpy as np
import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from config import index_to_class
from utils import custom_collate_fn
from collections import OrderedDict
from cleanlab.filter import find_label_issues
from snorkel.slicing import PandasSFApplier, slicing_function

# === Evaluation Function ===
def evaluate(ds, predictor):
    outputs = ds.iter_torch_batches(batch_size=16, collate_fn=custom_collate_fn)
    predictions = []
    y_true = []

    for batch in outputs:
        result = predictor(batch)
        predictions.extend(result["output"])
        y_true.extend(batch["labels"].cpu().numpy())

    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, predictions)

    return {
        "accuracy": accuracy,
        "report": report,
        "y_true": y_true,
        "y_pred": predictions
    }

# === Detailed Analysis ===
def analyze_predictions(y_true, y_pred, y_prob=None):
    metrics = {"overall": {}, "class": {}, "slices": {}}

    overall = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics["overall"] = {
        "precision": overall[0],
        "recall": overall[1],
        "f1": overall[2],
        "num_samples": float(len(y_true))
    }

    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    for i, name in index_to_class.items():
        metrics["class"][name] = {
            "precision": per_class[0][i],
            "recall": per_class[1][i],
            "f1": per_class[2][i],
            "num_samples": float(per_class[3][i])
        }

    sorted_tags_by_f1 = OrderedDict(
        sorted(metrics["class"].items(), key=lambda item: item[1]["f1"], reverse=True)
    )

    if y_prob is not None:
        label_issues = find_label_issues(labels=y_true, pred_probs=y_prob, return_indices_ranked_by="self_confidence")
        metrics["label_issues"] = label_issues

    return metrics, sorted_tags_by_f1

# === Slice Functions ===
@slicing_function()
def positive_text(x):
    return "good" in x.Text.lower() or "excellent" in x.Text.lower()

@slicing_function()
def short_text(x):
    return len(x.Text.split()) < 8

def evaluate_slices(df, y_true, y_pred):
    slicing_functions = [positive_text, short_text]
    applier = PandasSFApplier(slicing_functions)
    slices = applier.apply(df)
    val_slices = slices[df.index]
    metrics = {}

    for slice_name in val_slices.dtype.names:
        mask = val_slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                np.array(y_true)[mask], np.array(y_pred)[mask], average="micro", zero_division=0
            )
            metrics[slice_name] = {
                "precision": slice_metrics[0],
                "recall": slice_metrics[1],
                "f1": slice_metrics[2],
                "num_samples": int(sum(mask))
            }
    return metrics