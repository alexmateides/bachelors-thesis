"""Prediction, aggregation, and metric helpers for segment- and TIC-level evaluation."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import EvalPrediction

from config import ExperimentConfig
from utils import save_json


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute the standard segment-level classification metrics."""

    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax over the class dimension."""

    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)


@torch.no_grad()
def predict_dataset(cfg: ExperimentConfig, model, dataset, batch_size: int, device: str, collator) -> Dict[
    str, np.ndarray]:
    """Run model inference over a dataset and return raw prediction arrays."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.DATALOADER_NUM_WORKERS,
        pin_memory=cfg.DATALOADER_PIN_MEMORY,
    )
    model.eval()
    all_logits, all_labels, all_tic_ids, all_segment_ids, all_n_segments = [], [], [], [], []
    for batch in loader:
        labels = batch["labels"]
        tic_ids = batch["tic_id"]
        segment_ids = batch["segment_id"]
        n_segments = batch["n_segments"]
        ignore_keys = {"labels", "tic_id", "segment_id", "n_segments", "sample_weight"}
        model_inputs = {k: v.to(device) for k, v in batch.items() if k not in ignore_keys}
        logits = model(**model_inputs)["logits"]
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_tic_ids.append(tic_ids.detach().cpu().numpy())
        all_segment_ids.append(segment_ids.detach().cpu().numpy())
        all_n_segments.append(n_segments.detach().cpu().numpy())
    return {
        "logits": np.concatenate(all_logits, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "tic_ids": np.concatenate(all_tic_ids, axis=0),
        "segment_ids": np.concatenate(all_segment_ids, axis=0),
        "n_segments": np.concatenate(all_n_segments, axis=0),
    }


def compute_segment_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute segment-level metrics from raw logits and integer labels."""

    preds = np.argmax(logits, axis=1)
    return {
        "segment_accuracy": accuracy_score(labels, preds),
        "segment_f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "segment_f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "segment_precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "segment_recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def aggregate_tic_predictions_mean_prob(
        logits: np.ndarray, labels: np.ndarray, tic_ids: np.ndarray
) -> Dict[str, np.ndarray]:
    """Aggregate segment predictions to TIC-level predictions by mean probability."""

    probs = softmax_numpy(logits)
    tic_true, tic_pred, tic_ids_out, tic_probs_out = [], [], [], []
    for tic_id in sorted(np.unique(tic_ids)):
        m = tic_ids == tic_id
        tic_labels = labels[m]
        unique_labels = np.unique(tic_labels)
        if len(unique_labels) != 1:
            raise ValueError(f"TIC {tic_id} has multiple true labels: {unique_labels.tolist()}")
        mean_probs = probs[m].mean(axis=0)
        tic_ids_out.append(int(tic_id))
        tic_true.append(int(unique_labels[0]))
        tic_pred.append(int(np.argmax(mean_probs)))
        tic_probs_out.append(mean_probs)
    return {
        "tic_ids": np.asarray(tic_ids_out, dtype=np.int64),
        "tic_true": np.asarray(tic_true, dtype=np.int64),
        "tic_pred": np.asarray(tic_pred, dtype=np.int64),
        "tic_probs": np.asarray(tic_probs_out, dtype=np.float64),
    }


def compute_tic_mean_prob_metrics(
        logits: np.ndarray, labels: np.ndarray, tic_ids: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compute TIC-level metrics and return the aggregated intermediate arrays."""

    agg = aggregate_tic_predictions_mean_prob(logits=logits, labels=labels, tic_ids=tic_ids)
    y_true = agg["tic_true"]
    y_pred = agg["tic_pred"]
    metrics = {
        "tic_meanprob_accuracy": accuracy_score(y_true, y_pred),
        "tic_meanprob_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "tic_meanprob_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "tic_meanprob_precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "tic_meanprob_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "tic_count": int(len(y_true)),
    }
    return metrics, agg


def evaluate_dataset_segment_and_tic(
        cfg: ExperimentConfig,
        model,
        dataset,
        split_name: str,
        output_dir: str,
        class_names: List[str],
        device: str,
        collator,
) -> Dict[str, float]:
    """Evaluate a split, persist artifacts, and return the summary metrics."""

    # run the model on all samples and collect predictions
    pred = predict_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset,
        batch_size=cfg.BATCH_SIZE,
        device=device,
        collator=collator,
    )

    logits = pred["logits"]
    labels = pred["labels"]
    tic_ids = pred["tic_ids"]

    # compute segment-level metrics (accuracy, F1, precision, recall)
    segment_metrics = compute_segment_metrics(logits, labels)

    # aggregate to TIC level and compute TIC-level metrics
    tic_metrics, tic_agg = compute_tic_mean_prob_metrics(logits, labels, tic_ids)

    # merge both metric sets
    metrics = {**segment_metrics, **tic_metrics}

    # print summary to console
    print("\n" + "=" * 60)
    print(f"{split_name.upper()} METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # print TIC-level classification report
    print(f"\n{split_name.upper()} TIC-level classification report, mean-prob aggregation:")
    print(
        classification_report(tic_agg["tic_true"], tic_agg["tic_pred"], target_names=class_names, zero_division=0)
    )

    # persist evaluation results to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save metrics as JSON
    save_json(output_dir / f"{split_name}_segment_and_tic_metrics.json", metrics)

    # save detailed classification report
    report_dict = classification_report(
        tic_agg["tic_true"], tic_agg["tic_pred"], target_names=class_names, zero_division=0, output_dict=True
    )
    save_json(output_dir / f"{split_name}_tic_meanprob_classification_report.json", report_dict)

    # compute and save confusion matrix as row-normalized percentages
    cm_counts = confusion_matrix(tic_agg["tic_true"], tic_agg["tic_pred"], labels=np.arange(len(class_names)))
    cm = np.divide(
        cm_counts.astype(np.float64),
        cm_counts.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm_counts, dtype=np.float64),
        where=cm_counts.sum(axis=1, keepdims=True) != 0,
    ) * 100.0
    np.save(output_dir / f"{split_name}_tic_meanprob_confusion_matrix.npy", cm)

    # save detailed predictions for further analysis
    np.savez_compressed(
        output_dir / f"{split_name}_tic_meanprob_predictions.npz",
        tic_ids=tic_agg["tic_ids"],
        tic_true=tic_agg["tic_true"],
        tic_pred=tic_agg["tic_pred"],
        tic_probs=tic_agg["tic_probs"],
        class_names=np.asarray(class_names),
    )

    # render confusion matrix as an image
    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, values_format=".1f", xticks_rotation=45)
    ax.set_title(f"{split_name} TIC confusion matrix, mean-prob aggregation")
    ax.set_ylabel("True label (%)")
    ax.set_xlabel("Predicted label (%)")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_tic_meanprob_confusion_matrix.png", dpi=200)
    plt.close(fig)

    return metrics


def get_tic_prediction_pairs(logits: np.ndarray, labels: np.ndarray, tic_ids: np.ndarray, id2label: Dict[int, str]) -> Dict[int, Tuple[str, str]]:
    """Return a dictionary mapping TIC IDs to (predicted_label, true_label) pairs using mean probability aggregation.

    Args:
        logits: Model output logits
        labels: True label IDs
        tic_ids: TIC identifiers for aggregation
        id2label: Mapping from label ID to human-readable label string

    Returns:
        Dictionary mapping TIC ID to tuple of (predicted_label_str, true_label_str)
    """
    agg = aggregate_tic_predictions_mean_prob(logits=logits, labels=labels, tic_ids=tic_ids)
    return {int(tic_id): (id2label[int(predicted_label)], id2label[int(true_label)]) for tic_id, predicted_label, true_label in
            zip(agg["tic_ids"], agg["tic_pred"], agg["tic_true"])}
