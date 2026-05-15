import os
import random
from json import dump as json_dump
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
from transformers import set_seed

from config import ExperimentConfig


def save_json(path: Union[str, Path], obj: Any) -> None:
    """
    pretty print obj to path as a JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json_dump(obj, f, indent=2)


def set_reproducibility(cfg: ExperimentConfig) -> None:
    """
    Set environment variables and library seeds for deterministic runs
    """
    if cfg.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.CUDA_VISIBLE_DEVICES)
    os.environ["PYTHONHASHSEED"] = str(cfg.RANDOM_STATE)
    if cfg.CUBLAS_WORKSPACE_CONFIG:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg.CUBLAS_WORKSPACE_CONFIG

    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)
    torch.manual_seed(cfg.RANDOM_STATE)
    torch.cuda.manual_seed(cfg.RANDOM_STATE)
    torch.cuda.manual_seed_all(cfg.RANDOM_STATE)
    set_seed(cfg.RANDOM_STATE)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # kept failing on the ROCM architecture for some reason
    torch.use_deterministic_algorithms(False, warn_only=True)


def safe_string(x: Any) -> str:
    """
    Normalize a scalar numpy-like object to a python string
    """
    arr = np.asarray(x).reshape(-1)
    if len(arr) != 1:
        raise ValueError(f"Expected scalar string-like value, got shape {np.asarray(x).shape}")
    return str(arr[0])


def read_scalar_from_npz(npz: Any, key: str, default: Any = None) -> Any:
    """
    Read a scalar field from the .npz archive.
    """
    if key not in npz:
        if default is not None:
            return default
        raise KeyError(f"Missing required field: {key}")
    arr = np.asarray(npz[key]).reshape(-1)
    if len(arr) != 1:
        raise ValueError(f"Expected scalar for {key}, got shape {np.asarray(npz[key]).shape}")
    return arr[0]


def read_int_from_npz(npz, key: str, default=None) -> int:
    """
    Read an integer field from a .npz archive
    """
    return int(read_scalar_from_npz(npz, key, default=default))


def read_label_from_npz(npz: Any) -> str:
    """
    Read the string class label from a .npz archive
    """
    if "label" in npz:
        return safe_string(npz["label"])
    raise KeyError("NPZ file missing 'label' field")


def get_tic_id_from_file(fp: Union[str, Path]) -> int:
    """
    Read the TIC identifier from a saved .npz segment file
    """
    with np.load(fp, allow_pickle=True) as npz:
        return read_int_from_npz(npz, "tic_id")


def load_npz_files(data_dir: Union[str, Path]) -> List[Path]:
    """
    Return all training NPZ files in data_dir except metadata.npz
    """
    data_dir = Path(data_dir)
    files = sorted([p for p in data_dir.glob("*.npz") if p.name != "metadata.npz"])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return files


def enabled_flux_branches(cfg: ExperimentConfig) -> List[str]:
    """
    Return the flux branches enabled by the configuration.
    """
    out = []
    if cfg.USE_BRANCH_2MIN:
        out.append("2min")
    if cfg.USE_BRANCH_8MIN:
        out.append("8min")
    if cfg.USE_BRANCH_32MIN:
        out.append("32min")
    if cfg.USE_BRANCH_128MIN:
        out.append("128min")
    return out


def get_patch_size(cfg: ExperimentConfig, branch_name: str) -> int:
    """
    Resolve the patch size for a branch, handling both scalar and mapping configs
    """
    if isinstance(cfg.PATCH_SIZE, dict):
        patch_size = int(cfg.PATCH_SIZE.get(branch_name, 1))
    else:
        patch_size = int(cfg.PATCH_SIZE)
    if patch_size < 1:
        raise ValueError(f"Invalid patch size for branch {branch_name}: {patch_size}")
    return patch_size
