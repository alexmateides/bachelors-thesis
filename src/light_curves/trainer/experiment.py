"""End-to-end experiment orchestration for a single training run."""

import argparse
import time
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from config import ExperimentConfig, load_config
from dataset import (
    LightCurveCollator,
    LightCurveNPZDataset,
    collect_files_labels_and_groups,
    compute_tic_level_class_weights,
    group_stratified_split,
    summarize_valid_fractions,
    warn_if_split_missing_classes,
)
from evaluation import evaluate_dataset_segment_and_tic, compute_metrics
from utils import enabled_flux_branches, get_patch_size, get_tic_id_from_file, save_json, safe_string, \
    set_reproducibility
from model import MultiBranchClassifier
from training import (
    MetricsCallback,
    TICBalancedLossTrainer,
    TICCheckpointCallback,
    build_training_args,
    dump_hyperparams,
    save_training_plots,
)


def compute_segment_class_sampling_weights(train_labels_str, label2id, n_classes):
    """
    Compute per-segment weights for inverse-frequency class sampling
    """
    y_train = np.array([label2id[label] for label in train_labels_str], dtype=np.int64)
    class_counts = np.bincount(y_train, minlength=n_classes)

    # sanity check
    missing_classes = np.where(class_counts == 0)[0]
    if len(missing_classes) > 0:
        missing_names = [label for label, idx in label2id.items() if idx in missing_classes]
        raise ValueError(
            f"Cannot enable class sampling because these classes are missing from the training split: {missing_names}."
        )

    per_class_weights = len(y_train) / (n_classes * class_counts.astype(np.float64))
    sample_weights = per_class_weights[y_train]
    return torch.tensor(sample_weights, dtype=torch.double), class_counts, per_class_weights


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a full train/val/test cycle and persist all experiment outputs
    """
    # duration monitoring
    start_time = time.perf_counter()
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
    set_reproducibility(config)

    # --- GPU diagnostics ---
    print("INFO: torch.cuda.is_available():", torch.cuda.is_available())
    print("INFO: torch.cuda.device_count():", torch.cuda.device_count())
    try:
        print("INFO: torch.version.cuda:", torch.version.cuda)
    except Exception:
        pass
    try:
        print("INFO: cudnn_version:", torch.backends.cudnn.version())
    except Exception:
        pass
    print("INFO: CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        try:
            cur = torch.cuda.current_device()
            print("INFO: current_device:", cur)
            print("INFO: device_name:", torch.cuda.get_device_name(cur))
        except Exception as e:
            print("INFO: cuda device info error:", e)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- output directory setup ---
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "resolved_config.json", asdict(config))
    save_json(output_dir / "experiment_timing.json", {"status": "running", "start_time_local": wall_start})


    # --- dataset metadata ---
    metadata_path = Path(config.DATA_DIR) / "metadata.npz"
    if metadata_path.exists():
        with np.load(metadata_path, allow_pickle=True) as meta:
            if "target_length" in meta:
                print("Metadata target_length:", int(meta["target_length"]))
            if "total_saved_segments" in meta:
                print("Metadata total_saved_segments:", int(meta["total_saved_segments"]))
    else:
        print("Warning: metadata.npz not found in DATA_DIR.")

    all_files, all_label_strings, y_all, groups_all, segments_all, n_segments_all, label2id, id2label = collect_files_labels_and_groups(
        config, config.DATA_DIR, config.ALLOWED_CLASSES
    )
    if len(all_files) == 0:
        raise RuntimeError("No files left after filtering. Check DATA_DIR / ALLOWED_CLASSES.")

    # .npz data sanity check
    with np.load(all_files[0], allow_pickle=True) as npz:
        print("Example file:", all_files[0].name)
        for branch in enabled_flux_branches(config):
            print(
                branch,
                "flux shape:",
                np.asarray(npz[f"flux_{branch}"]).shape,
                "mask shape:",
                np.asarray(npz[f"valid_mask_{branch}"]).shape,
                "patch_size:",
                get_patch_size(config, branch) if config.USE_PATCH_EMBEDDING else None,
            )
        if config.USE_PHASE_BRANCH:
            print(
                "phase",
                "folded flux shape:",
                np.asarray(npz["flux_folded_2min"]).shape,
                "folded mask shape:",
                np.asarray(npz["valid_mask_folded_2min"]).shape,
                "patch_size:",
                get_patch_size(config, "phase") if config.USE_PATCH_EMBEDDING else None,
            )

    # split into train/val/test by individual TICs so segments from one star don't end up in multiple subsets
    train_files, val_files, test_files, train_labels_str, val_labels_str, test_labels_str = group_stratified_split(
        file_paths=all_files,
        labels_str=all_label_strings,
        groups=groups_all,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # load datasets
    train_ds = LightCurveNPZDataset(config, train_files, label2id)
    val_ds = LightCurveNPZDataset(config, val_files, label2id)
    test_ds = LightCurveNPZDataset(config, test_files, label2id) if config.RUN_TEST_EVALUATION else None

    class_names = [id2label[i] for i in range(len(id2label))]
    n_classes = len(class_names)
    warn_if_split_missing_classes("val", val_labels_str, label2id, n_classes)
    if config.RUN_TEST_EVALUATION:
        warn_if_split_missing_classes("test", test_labels_str, label2id, n_classes)

    cls_weights, train_tic_df = compute_tic_level_class_weights(train_files, train_labels_str, label2id, n_classes)
    train_sample_weights = None
    if config.USE_CLASS_SAMPLING:
        train_sample_weights, sample_class_counts, sample_class_weights = compute_segment_class_sampling_weights(
            train_labels_str, label2id, n_classes
        )
        print("Segment-level train class counts:", {id2label[i]: int(sample_class_counts[i]) for i in range(n_classes)})
        print("Segment-level class sampling weights:", sample_class_weights)
    train_groups = sorted(set(get_tic_id_from_file(fp) for fp in train_files))
    val_groups = sorted(set(get_tic_id_from_file(fp) for fp in val_files))
    test_groups = sorted(set(get_tic_id_from_file(fp) for fp in test_files))

    print(f"Train files: {len(train_ds)}, Val files: {len(val_ds)}, Test files: {len(test_files)}")
    print(f"Train TICs: {len(train_groups)}, Val TICs: {len(val_groups)}, Test TICs: {len(test_groups)}")
    print("Class names:", class_names)
    print("TIC-level class weights:", cls_weights)
    print("USE_PATCH_EMBEDDING:", config.USE_PATCH_EMBEDDING)
    print("PATCH_SIZE:", config.PATCH_SIZE)
    print("MIN_VALID_POINTS_PER_PATCH:", config.MIN_VALID_POINTS_PER_PATCH)
    print("MAX_POSITION_EMBEDDINGS:", config.MAX_POSITION_EMBEDDINGS)
    print("BRANCH_ENCODER_TYPE:", config.BRANCH_ENCODER_TYPE)
    print("BACKBONE_MODEL_NAME:", config.BACKBONE_MODEL_NAME)
    print("USE_LORA:", config.USE_LORA)
    print("LORA_R:", config.LORA_R)
    print("LORA_ALPHA:", config.LORA_ALPHA)
    print("LORA_DROPOUT:", config.LORA_DROPOUT)
    print("USE_CLASS_SAMPLING:", config.USE_CLASS_SAMPLING)
    print("RUN_TEST_EVALUATION:", config.RUN_TEST_EVALUATION)
    print("LOAD_DATASET_IN_MEMORY:", config.LOAD_DATASET_IN_MEMORY)

    assert len(set(train_groups) & set(val_groups)) == 0
    assert len(set(train_groups) & set(test_groups)) == 0
    assert len(set(val_groups) & set(test_groups)) == 0
    summarize_valid_fractions(train_files, enabled_flux_branches(config))

    branch_input_dim = 1 + int(config.USE_FLUX_ERR)
    extra_input_dim = 1 + 10 + 10 if config.USE_EXTRA_FEATURES_BRANCH else 0
    model = MultiBranchClassifier(
        cfg=config,
        n_classes=n_classes,
        branch_input_dim=branch_input_dim,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        ff_dim=config.FF_DIM,
        dropout=config.DROPOUT,
        extra_input_dim=extra_input_dim,
        extra_hidden_dim=config.EXTRA_MLP_HIDDEN,
        extra_out_dim=config.EXTRA_MLP_OUT,
    ).to(device)

    collator = LightCurveCollator(config)
    training_args = build_training_args(config, output_dir, device=device)
    metrics_callback = MetricsCallback()
    tic_checkpoint_callback = TICCheckpointCallback(
        cfg=config,
        val_dataset=val_ds,
        output_dir=str(output_dir),
        class_names=class_names,
        label2id=label2id,
        id2label=id2label,
        device=device,
        collator=collator,
        metric_name="tic_meanprob_f1_macro",
        best_model_name=config.BEST_TIC_MODEL_NAME,
        metrics_callback=metrics_callback,
    )

    trainer = TICBalancedLossTrainer(
        cfg=config,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=cls_weights,
        train_sample_weights=train_sample_weights,
        use_class_sampling=config.USE_CLASS_SAMPLING,
        callbacks=[metrics_callback, tic_checkpoint_callback],
    )

    trainer.train()

    best_tic_model_path = output_dir / config.BEST_TIC_MODEL_NAME
    if best_tic_model_path.exists():
        # the original model is still in memory, loading into CPU prevents crashes in tight VRAM situations
        ckpt = torch.load(best_tic_model_path, map_location='cpu')
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.model.to(device)
        print(
            f"Loaded best TIC-level model from {best_tic_model_path} at epoch={ckpt.get('best_epoch')} with {ckpt.get('best_metric_name')}={ckpt.get('best_metric')}"
        )
    else:
        print(f"Warning: best TIC-level checkpoint not found at {best_tic_model_path}. Using final model state.")

    hist = metrics_callback.history
    print(hist)
    save_json(output_dir / "training_history.json", hist)
    save_json(
        output_dir / "hyperparams.json",
        dump_hyperparams(
            config,
            class_names,
            best_tic_model_path if best_tic_model_path.exists() else None,
            tic_checkpoint_callback,
        ),
    )
    save_training_plots(hist, output_dir)

    val_final_metrics = evaluate_dataset_segment_and_tic(config, trainer.model, val_ds, "val", str(output_dir),
                                                         class_names, device, collator)
    final_metrics = {"val": val_final_metrics}
    if config.RUN_TEST_EVALUATION:
        if test_ds is None:
            raise RuntimeError("RUN_TEST_EVALUATION=True but test_ds was not constructed.")
        test_final_metrics = evaluate_dataset_segment_and_tic(config, trainer.model, test_ds, "test", str(output_dir),
                                                              class_names, device, collator)
        final_metrics["test"] = test_final_metrics
    else:
        print("Skipping test evaluation because RUN_TEST_EVALUATION=False")

    duration_seconds = float(time.perf_counter() - start_time)
    timing = {
        "status": "completed",
        "start_time_local": wall_start,
        "end_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_seconds / 60.0,
        "duration_hours": duration_seconds / 3600.0,
    }
    save_json(output_dir / "experiment_timing.json", timing)
    final_metrics["timing"] = timing
    save_json(output_dir / "final_segment_and_tic_metrics.json", final_metrics)

    best_model_path = output_dir / config.BEST_MODEL_NAME
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "class_names": class_names,
            "hyperparams": dump_hyperparams(
                config,
                class_names,
                best_tic_model_path if best_tic_model_path.exists() else None,
                tic_checkpoint_callback,
            ),
            "final_metrics": final_metrics,
        },
        best_model_path,
    )
    print(f"Saved final best TIC-selected model package to: {best_model_path}")
    return final_metrics


def main() -> None:
    """CLI entrypoint for the modular trainer."""

    parser = argparse.ArgumentParser(description="Train light-curve classifier from a JSON config.")
    parser.add_argument("--config", type=str, required=False, default=None, help="Path to resolved JSON config.")
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        run_experiment(config)

    # catch downstream errors
    except Exception as exc:
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            output_dir / "experiment_timing.json",
            {
                "status": "failed",
                "end_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
