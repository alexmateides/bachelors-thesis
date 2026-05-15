"""Experiment configuration and JSON config loading helpers for hyper_sweep."""

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ExperimentConfig:
    """
    Resolved configuration for one training run
    The fields mirror the switches used across the data pipeline, model, and training loop
    The class intentionally stays declarative so config files can override only the values they care about
    """
    RANDOM_STATE: int = 2026
    CUDA_VISIBLE_DEVICES: Optional[str] = "0"
    CUBLAS_WORKSPACE_CONFIG: Optional[str] = None

    DATA_DIR: str = "../data/processed_4_channel_multi_256"
    OUTPUT_DIR: str = "results/B_default"
    EXPERIMENT_NAME: Optional[str] = None

    ARCHITECTURE: str = "v1.4_final"
    BRANCH_ENCODER_TYPE: str = "transformer"
    BACKBONE_MODEL_NAME: Optional[str] = None
    USE_LORA: bool = False
    LORA_R: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.1
    LORA_BIAS: str = "none"
    ALLOWED_CLASSES: List[str] = field(default_factory=lambda: ["eclipsing", "pulsating", "rotating"])

    USE_BRANCH_2MIN: bool = True
    USE_BRANCH_8MIN: bool = True
    USE_BRANCH_32MIN: bool = True
    USE_BRANCH_128MIN: bool = True
    USE_PHASE_BRANCH: bool = True
    USE_EXTRA_FEATURES_BRANCH: bool = True

    RUN_TEST_EVALUATION: bool = True
    LOAD_DATASET_IN_MEMORY: bool = True

    USE_FLUX_ERR: bool = True
    NORMALIZE_FLUX_ERR: bool = True
    LOG1P_FLUX_ERR: bool = True

    MAX_POSITION_EMBEDDINGS: int = 256

    POOLING_MODE: str = "cls"
    D_MODEL: int = 128
    N_HEADS: int = 8
    N_LAYERS: int = 4
    FF_DIM: int = 512
    DROPOUT: float = 0.2

    USE_PATCH_EMBEDDING: bool = True
    PATCH_SIZE: Union[int, Dict[str, int]] = field(
        default_factory=lambda: {"2min": 16, "8min": 16, "32min": 16, "128min": 16, "phase": 16}
    )
    MIN_VALID_POINTS_PER_PATCH: int = 1

    FUSION_HIDDEN: int = 256
    EXTRA_MLP_HIDDEN: int = 128
    EXTRA_MLP_OUT: int = 128

    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 5e-5
    WEIGHT_DECAY: float = 1e-2
    LABEL_SMOOTHING_FACTOR: float = 0.0
    USE_CLASS_SAMPLING: bool = False
    MAX_GRAD_NORM: float = 1.0

    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15

    BEST_MODEL_NAME: str = "best_model.pt"
    BEST_TIC_MODEL_NAME: str = "best_tic_model.pt"

    FP16: Optional[bool] = None
    BF16: bool = False
    SAVE_SAFETENSORS: bool = True
    DATALOADER_NUM_WORKERS: int = 0
    DATALOADER_PIN_MEMORY: bool = False
    SAVE_TOTAL_LIMIT: Optional[int] = 1

    def __post_init__(self) -> None:
        encoder_type = self.BRANCH_ENCODER_TYPE.lower().strip()
        if encoder_type not in {"transformer", "qwen", "chronos2", "cnn"}:
            raise ValueError(
                f"Invalid BRANCH_ENCODER_TYPE={self.BRANCH_ENCODER_TYPE!r}. Expected one of: transformer, qwen, chronos2, cnn."
            )
        self.BRANCH_ENCODER_TYPE = encoder_type

        if not 0.0 <= float(self.LABEL_SMOOTHING_FACTOR) < 1.0:
            raise ValueError(
                f"LABEL_SMOOTHING_FACTOR must be in [0.0, 1.0), got {self.LABEL_SMOOTHING_FACTOR!r}."
            )


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base without mutating either
    Nested dictionaries are merged key-by-key; all other values are replaced
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: Optional[Union[str, Path]] = None) -> ExperimentConfig:
    """
    Load a JSON config file into ExperimentConfig
    If config_path is None, the defaults are returned
    """
    cfg_dict = asdict(ExperimentConfig())

    if config_path is not None:
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            user_cfg = json.load(f)

        unknown = sorted(set(user_cfg) - set(cfg_dict))

        if unknown:
            raise ValueError(f"Unknown config keys in {config_path}: {unknown}")

        cfg_dict = deep_update(cfg_dict, user_cfg)

    return ExperimentConfig(**cfg_dict)
