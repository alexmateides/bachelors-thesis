### Data and preprocessing

The processed data are stored on git LFS as an archive, the current setup expects them in `/data/lightcurves`.

The preprocessing pipeline can be configured in the code and can be run as a script:

```shell
python preprocess_lightcurves.py
```

### Light Curve Experiments

The framework enables running of multiple experiments via JSON configs:

```json
{
  "EXPERIMENT_NAME": "default_original_trainer",
  "DATA_DIR": "../../data/processed_4_channel_multi_256",
  "OUTPUT_DIR": "../experiment_results/B_all_default",
  "ARCHITECTURE": "v1.4_final",
  "BRANCH_ENCODER_TYPE": "transformer",
  "BACKBONE_MODEL_NAME": null,
  "USE_LORA": true,
  "LORA_R": 8,
  "LORA_ALPHA": 16,
  "LORA_DROPOUT": 0.1,
  "LORA_BIAS": "none",
  "ALLOWED_CLASSES": [
    "eclipsing",
    "pulsating",
    "rotating"
  ],
  "USE_BRANCH_2MIN": true,
  "USE_BRANCH_8MIN": true,
  "USE_BRANCH_32MIN": true,
  "USE_BRANCH_128MIN": true,
  "USE_PHASE_BRANCH": true,
  "USE_EXTRA_FEATURES_BRANCH": true,
  "RUN_TEST_EVALUATION": false,
  "LOAD_DATASET_IN_MEMORY": true,
  "USE_FLUX_ERR": true,
  "NORMALIZE_FLUX_ERR": true,
  "LOG1P_FLUX_ERR": true,
  "MAX_POSITION_EMBEDDINGS": 256,
  "POOLING_MODE": "cls",
  "D_MODEL": 256,
  "N_HEADS": 8,
  "N_LAYERS": 4,
  "FF_DIM": 1024,
  "DROPOUT": 0.15,
  "USE_PATCH_EMBEDDING": true,
  "PATCH_SIZE": {
    "2min": 16,
    "8min": 16,
    "32min": 16,
    "128min": 16,
    "phase": 16
  },
  "MIN_VALID_POINTS_PER_PATCH": 1,
  "FUSION_HIDDEN": 256,
  "EXTRA_MLP_HIDDEN": 128,
  "EXTRA_MLP_OUT": 128,
  "BATCH_SIZE": 64,
  "EPOCHS": 50,
  "LEARNING_RATE": 5e-05,
  "WEIGHT_DECAY": 0.01,
  "MAX_GRAD_NORM": 1.0,
  "VAL_SIZE": 0.15,
  "TEST_SIZE": 0.15,
  "BEST_MODEL_NAME": "best_model.pt",
  "BEST_TIC_MODEL_NAME": "best_tic_model.pt",
  "FP16": null,
  "BF16": false,
  "SAVE_SAFETENSORS": true,
  "DATALOADER_NUM_WORKERS": 0,
  "DATALOADER_PIN_MEMORY": false,
  "SAVE_TOTAL_LIMIT": null,
  "RANDOM_STATE": 2026,
  "CUDA_VISIBLE_DEVICES": "0",
  "CUBLAS_WORKSPACE_CONFIG": null,
  "LABEL_SMOOTHING_FACTOR": 0.0,
  "USE_CLASS_SAMPLING": false
}
```

The user can set only partial parameters, in that case, the rest is inferred from the default config

```json
{
  "EXPERIMENT_NAME": "example_experiment",
  "DATA_DIR": "../../../data/lightcurves/processed_4_channel_multi_256",
  "OUTPUT_DIR": "../experiment_results/example_experiment",
  "ARCHITECTURE": "v1.4_final",
  "BRANCH_ENCODER_TYPE": "transformer",
  "ALLOWED_CLASSES": [
    "pulsating",
    "eclipsing",
    "rotating"
  ],
  "USE_BRANCH_2MIN": false,
  "USE_BRANCH_8MIN": true,
  "USE_BRANCH_32MIN": false,
  "USE_BRANCH_128MIN": false,
  "USE_PHASE_BRANCH": false,
  "USE_EXTRA_FEATURES_BRANCH": false
}
```

### Supported Models

The following `BRANCH_ENCODER_TYPE` values are supported:

1. `transformer` - basic transformer trained from scratch
2. `chronos` - Chronos-2 fine-tune
3. `qwen` - Qwen 2.5 0.5B fine-tune
4. `cnn` - Baseline 2-layer CNN architecture

### Running Experiments

**0. Install required dependencies**

> dependencies are in `src/requirements.txt`

```shell
pip install -Ur src/requirements.txt
```

**1. Go into the `trainer` directory**

```shell
cd src/lightcurves/trainer
```

**2. Run the driver script with a parameter specifying a directory that contains the experiment configs**

```shell
python driver.py --config-dir ../example_experiment/
```

### PyTorch numerical instability

When testing the function of the code in a brand-new environment with updated PyTorch, we encountered an issue where the gradient exploded (grad norm approached +inf) and training collapsed, probably due to numerical instability.
It may also be an AMD ROCm distribution specific issue.

PyTorch version used for all the experiments:

```
Mame: torch
Version: 2.9.1+rocm6.3
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
```

PyTorch version in a new environment that produced the gradient collapse:

```
Name: torch
Version: 2.13.0.dev20260514+rocm7.2
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
```