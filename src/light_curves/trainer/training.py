"""Training loop extensions, callbacks, and plotting utilities."""

import inspect
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer, TrainerCallback, TrainingArguments

from config import ExperimentConfig
from evaluation import compute_tic_mean_prob_metrics, predict_dataset
from utils import enabled_flux_branches, get_patch_size, save_json


class TICBalancedLossTrainer(Trainer):
    """
    HuggingFace Transformers Trainer wrapper that combines class weighting with inverse-segment weights
    """
    def __init__(
        self,
        *args,
        cfg: ExperimentConfig,
        class_weights=None,
        train_sample_weights=None,
        use_class_sampling: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize the trainer with configuration and class weights
        """

        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.class_weights = class_weights
        self.train_sample_weights = train_sample_weights
        self.use_class_sampling = cfg.USE_CLASS_SAMPLING if use_class_sampling is None else bool(use_class_sampling)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted cross-entropy using class and segment weights
        """
        labels = inputs["labels"]
        sample_weight = inputs["sample_weight"]

        # extract model inputs by removing metadata/weight keys
        ignore_keys = {"labels", "tic_id", "segment_id", "n_segments", "sample_weight"}
        model_inputs = {k: v for k, v in inputs.items() if k not in ignore_keys}

        # forward pass
        outputs = model(**model_inputs, labels=labels)
        logits = outputs["logits"]

        label_smoothing = float(getattr(self.args, "label_smoothing_factor", 0.0) or 0.0)
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing_factor must be in [0.0, 1.0), got {label_smoothing!r}.")

        class_weight = self.class_weights.to(logits.device) if self.class_weights is not None else None

        # compute per-segment cross-entropy weighted by class (to handle class imbalance)
        loss_per_segment = F.cross_entropy(
            logits,
            labels,
            weight=class_weight,
            reduction="none",
            label_smoothing=label_smoothing,
        )

        # apply inverse-segment-count weighting so multi-segment targets don't dominate the loss (each segment from the same TIC contributes equally)
        sample_weight = sample_weight.to(logits.device, dtype=loss_per_segment.dtype)
        loss = (loss_per_segment * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1e-12)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Use AdamW and cosine annealing for the optimization schedule
        """

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=int(num_training_steps), eta_min=1e-6)

    def _get_train_sampler(self, train_dataset=None):
        """
        Return a weighted sampler for class-balanced training when enabled
        """
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None
        if not self.use_class_sampling or self.train_sample_weights is None:
            return super()._get_train_sampler(train_dataset)
        if self.args.world_size > 1:
            return super()._get_train_sampler(train_dataset)

        weights = torch.as_tensor(self.train_sample_weights, dtype=torch.double)
        if len(weights) != len(train_dataset):
            raise ValueError(
                f"train_sample_weights length ({len(weights)}) does not match train_dataset length ({len(train_dataset)})."
            )
        if torch.sum(weights) <= 0:
            raise ValueError("train_sample_weights must contain at least one positive value.")
        return WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)


class MetricsCallback(TrainerCallback):
    """
    Collect training and validation metrics into history
    """
    def __init__(self):
        """
        Prepare the metric history buffers used for plotting
        """
        self.history = {
            "train_epoch": [],
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
            "val_precision_macro": [],
            "val_recall_macro": [],
            "val_tic_meanprob_accuracy": [],
            "val_tic_meanprob_f1_macro": [],
            "val_tic_meanprob_f1_weighted": [],
            "val_tic_meanprob_precision_macro": [],
            "val_tic_meanprob_recall_macro": [],
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Record training and evaluation logs emitted by the trainer
        """
        if not logs:
            return
        if "loss" in logs and "epoch" in logs and "eval_loss" not in logs:
            self.history["train_epoch"].append(logs["epoch"])
            self.history["train_loss"].append(logs["loss"])
        if "eval_loss" in logs and "epoch" in logs:
            self.history["epoch"].append(logs["epoch"])
            self.history["val_loss"].append(logs.get("eval_loss"))
            self.history["val_accuracy"].append(logs.get("eval_accuracy"))
            self.history["val_f1_macro"].append(logs.get("eval_f1_macro"))
            self.history["val_f1_weighted"].append(logs.get("eval_f1_weighted"))
            self.history["val_precision_macro"].append(logs.get("eval_precision_macro"))
            self.history["val_recall_macro"].append(logs.get("eval_recall_macro"))
            self.history["val_tic_meanprob_accuracy"].append(logs.get("eval_tic_meanprob_accuracy"))
            self.history["val_tic_meanprob_f1_macro"].append(logs.get("eval_tic_meanprob_f1_macro"))
            self.history["val_tic_meanprob_f1_weighted"].append(logs.get("eval_tic_meanprob_f1_weighted"))
            self.history["val_tic_meanprob_precision_macro"].append(logs.get("eval_tic_meanprob_precision_macro"))
            self.history["val_tic_meanprob_recall_macro"].append(logs.get("eval_tic_meanprob_recall_macro"))


class TICCheckpointCallback(TrainerCallback):
    """
    Save the best checkpoint according to a TIC-level validation metric
    """
    def __init__(self, cfg: ExperimentConfig, val_dataset, output_dir: str, class_names, label2id, id2label,
                 device: str, collator, metric_name: str = "tic_meanprob_f1_macro", best_model_name: Optional[str] = None,
                 metrics_callback: Optional["MetricsCallback"] = None):
        """
        Store the validation dataset and metadata required for checkpointing
        """
        self.cfg = cfg
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.class_names = class_names
        self.label2id = label2id
        self.id2label = id2label
        self.device = device
        self.collator = collator
        self.metric_name = metric_name
        self.best_model_name = best_model_name or cfg.BEST_TIC_MODEL_NAME
        self.best_metric = -float("inf")
        self.best_epoch = None
        self.best_model_path = os.path.join(output_dir, self.best_model_name)
        self.metrics_callback = metrics_callback

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        """
        Run TIC-level evaluation at the end of each epoch and checkpoint the best model
        """
        if model is None:
            return

        # compute TIC-level predictions by aggregating all segments for each TIC
        pred = predict_dataset(
            cfg=self.cfg,
            model=model,
            dataset=self.val_dataset,
            batch_size=args.per_device_eval_batch_size,
            device=self.device,
            collator=self.collator,
        )

        # compute TIC-level aggregation and metrics
        tic_metrics, _ = compute_tic_mean_prob_metrics(
            logits=pred["logits"], labels=pred["labels"], tic_ids=pred["tic_ids"]
        )
        current = float(tic_metrics[self.metric_name])

        # display metrics for this epoch
        print("\nTIC-level validation metrics for checkpoint selection:")
        for k, v in tic_metrics.items():
            print(f"  eval_{k}: {v:.6f}" if isinstance(v, float) else f"  eval_{k}: {v}")

        # merge TIC metrics into the trainer's metrics dict if provided
        if metrics is not None:
            for k, v in tic_metrics.items():
                metrics[f"eval_{k}"] = v

        # update the metrics callback history with TIC metrics for plotting
        if self.metrics_callback is not None:
            for k, v in tic_metrics.items():
                hist_key = f"val_{k}"
                if hist_key in self.metrics_callback.history:
                    self.metrics_callback.history[hist_key][-1] = v

        # persist checkpoint metadata to disk
        os.makedirs(self.output_dir, exist_ok=True)
        save_json(
            os.path.join(self.output_dir, "latest_val_tic_checkpoint_metrics.json"),
            {
                "epoch": state.epoch,
                "global_step": state.global_step,
                **tic_metrics,
                "best_metric_so_far": self.best_metric,
                "best_epoch_so_far": self.best_epoch,
            },
        )

        # save model if it achieved the best metric so far
        if current > self.best_metric:
            self.best_metric = current
            self.best_epoch = state.epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": self.label2id,
                    "id2label": self.id2label,
                    "class_names": self.class_names,
                    "best_epoch": self.best_epoch,
                    "best_global_step": state.global_step,
                    "best_metric_name": self.metric_name,
                    "best_metric": self.best_metric,
                    "tic_metrics": tic_metrics,
                },
                self.best_model_path,
            )
            print(
                f"Saved new best TIC-level model to {self.best_model_path} with {self.metric_name}={self.best_metric:.6f}"
            )


def build_training_args(cfg: ExperimentConfig, output_dir: Union[str, Path], device: str = "cuda") -> TrainingArguments:
    """
    Create the Hugging Face TrainingArguments for trainer configuration

    Args:
        cfg: ExperimentConfig
        output_dir: Output directory for training artifacts
        device: Device to use ("cpu" or "cuda"). When "cpu", disables GPU-specific features.
    """
    # Disable GPU-specific features when using CPU
    use_cuda = device != "cpu"
    fp16 = False
    bf16 = False

    common_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        label_smoothing_factor=cfg.LABEL_SMOOTHING_FACTOR,
        save_strategy="no",
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        logging_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        bf16=bf16,
        report_to="wandb",
        max_grad_norm=cfg.MAX_GRAD_NORM,
        remove_unused_columns=False,
        seed=cfg.RANDOM_STATE,
        data_seed=cfg.RANDOM_STATE,
        dataloader_num_workers=cfg.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=cfg.DATALOADER_PIN_MEMORY,
        full_determinism=False,
        disable_tqdm=True,
    )
    params = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in params:
        common_kwargs["eval_strategy"] = "epoch"
    else:
        common_kwargs["evaluation_strategy"] = "epoch"
    if "save_safetensors" in params:
        common_kwargs["save_safetensors"] = cfg.SAVE_SAFETENSORS
    return TrainingArguments(**common_kwargs)


def save_training_plots(hist: Dict[str, List[Any]], output_dir: Union[str, Path]) -> None:
    """
    Render the recorded training history into a combined training curves figure
    """
    output_dir = Path(output_dir)

    # only proceed if validation data exist
    if len(hist["epoch"]) == 0:
        return

    # create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # extract epoch numbers for x-axis alignment
    epochs = hist["epoch"]

    # left subplot: validation accuracy and F1 macro
    axes[0].plot(epochs, hist["val_accuracy"], marker="o", label="Val Accuracy", color='lightblue')
    axes[0].plot(epochs, hist["val_f1_macro"], marker="o", label="Val Macro F1", color='darkblue')

    # add TIC-level validation metrics if available (filter out None values)
    tic_acc = hist["val_tic_meanprob_accuracy"]
    tic_f1 = hist["val_tic_meanprob_f1_macro"]
    if any(v is not None for v in tic_acc):
        axes[0].plot(epochs, tic_acc, marker="o", label="TIC Val Accuracy", color='lightgreen')
    if any(v is not None for v in tic_f1):
        axes[0].plot(epochs, tic_f1, marker="o", label="TIC Val Macro F1", color='darkgreen')

    axes[0].set_title("Validation Metrics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # right subplot: train and validation loss
    train_x = list(range(1, len(hist["train_loss"]) + 1))
    axes[1].plot(train_x, hist["train_loss"], marker="o", label="Train Loss")
    axes[1].plot(epochs, hist["val_loss"], marker="o", label="Val Loss")
    axes[1].set_title("Train & Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # save the combined figure
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def dump_hyperparams(cfg: ExperimentConfig, class_names: List[str], best_tic_model_path: Optional[Union[str, Path]],
                     tic_checkpoint_callback: TICCheckpointCallback) -> Dict[str, Any]:
    """
    Serialize the resolved experiment settings
    """
    out = asdict(cfg)
    if cfg.BRANCH_ENCODER_TYPE == "transformer":
        positional_encoding = "learned absolute token-index embeddings after patch embedding"
        patch_embedding_features = [
            "masked_mean_per_channel",
            "masked_std_per_channel",
            "masked_min_per_channel",
            "masked_max_per_channel",
            "masked_linear_slope_per_channel",
            "valid_fraction",
        ]
    elif cfg.BRANCH_ENCODER_TYPE == "qwen":
        positional_encoding = "Qwen pretrained backbone with learned absolute patch positions and final-token pooling"
        patch_embedding_features = [
            "sliding_window_raw_flux_patch_projection",
            "learned_absolute_patch_positions",
        ]
    elif cfg.BRANCH_ENCODER_TYPE == "chronos2":
        positional_encoding = "Chronos-2 pretrained backbone with internal patching and masked mean pooling over context tokens"
        patch_embedding_features = []
    else:
        raise ValueError(f"Unknown encoder_type: {cfg.BRANCH_ENCODER_TYPE}")
    out.update(
        {
            "CLASS_NAMES": class_names,
            "POSITIONAL_ENCODING": positional_encoding,
            "ACTIVE_BRANCH_ENCODER": cfg.BRANCH_ENCODER_TYPE,
            "BACKBONE_MODEL_NAME": cfg.BACKBONE_MODEL_NAME,
            "USE_LORA": cfg.USE_LORA,
            "LORA_CONFIG": {
                "r": cfg.LORA_R,
                "alpha": cfg.LORA_ALPHA,
                "dropout": cfg.LORA_DROPOUT,
                "bias": cfg.LORA_BIAS,
            },
            "PATCH_EMBEDDING_FEATURES": patch_embedding_features,
            "RESOLVED_PATCH_SIZE": {branch: get_patch_size(cfg, branch) for branch in enabled_flux_branches(cfg) + (["phase"] if cfg.USE_PHASE_BRANCH else [])},
            "CLASS_WEIGHT_SOURCE": "TIC-level training label distribution, one label per TIC",
            "CLASS_SAMPLING": cfg.USE_CLASS_SAMPLING,
            "CLASS_SAMPLING_SOURCE": "segment-level inverse class frequency over the training split",
            "SEGMENT_SAMPLE_WEIGHT": "1 / n_segments from each segment file",
            "TRAINING_OBJECTIVE": "segment cross entropy weighted by TIC-level class weights and inverse-n_segments sample weights",
            "CHECKPOINT_SELECTION_METRIC": "validation TIC-level mean-probability macro F1",
            "TIC_LEVEL_AGGREGATION": "mean softmax probability over segments",
            "MASK_SOURCE": "valid_mask_* arrays from NPZ files",
            "DETERMINISTIC_ALGORITHMS": True,
            "CUBLAS_WORKSPACE_CONFIG_EFFECTIVE": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDNN_DETERMINISTIC": torch.backends.cudnn.deterministic,
            "CUDNN_BENCHMARK": torch.backends.cudnn.benchmark,
            "ALLOW_TF32_MATMUL": torch.backends.cuda.matmul.allow_tf32,
            "ALLOW_TF32_CUDNN": torch.backends.cudnn.allow_tf32,
            "BEST_TIC_MODEL_CHECKPOINT": str(best_tic_model_path) if best_tic_model_path is not None else None,
            "BEST_TIC_MODEL_METRIC": getattr(tic_checkpoint_callback, "best_metric", None),
            "BEST_TIC_MODEL_EPOCH": getattr(tic_checkpoint_callback, "best_epoch", None),
        }
    )
    return out

