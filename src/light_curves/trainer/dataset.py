"""NPZ validation, dataset splitting, dataset loading, and batching helpers."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from config import ExperimentConfig
from utils import enabled_flux_branches, load_npz_files, read_int_from_npz, read_label_from_npz


def collect_files_labels_and_groups(
    cfg: ExperimentConfig,
    data_dir: Union[str, Path],
    allowed_classes: List[str],
) -> Tuple[List[Path], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Scan .npz files, validate schema, and build label/group arrays

    The function performs lightweight structural checks before training starts so
    bad files are skipped early with a readable reason. It also constructs the
    label/id mapping from the surviving classes.
    """

    all_files = load_npz_files(data_dir)
    filtered_files, filtered_labels, filtered_groups = [], [], []
    filtered_segments, filtered_n_segments, bad_files = [], [], []
    flux_branches = enabled_flux_branches(cfg)

    # ensure at least one input branch is enabled
    if len(flux_branches) == 0 and not cfg.USE_PHASE_BRANCH and not cfg.USE_EXTRA_FEATURES_BRANCH:
        raise ValueError("At least one branch must be enabled.")

    # validate and filter each NPZ file
    for fp in all_files:
        try:
            with np.load(fp, allow_pickle=True) as npz:
                # extract metadata
                label = read_label_from_npz(npz)
                tic_id = read_int_from_npz(npz, "tic_id")
                segment_id = read_int_from_npz(npz, "segment_id", default=0)
                n_segments = read_int_from_npz(npz, "n_segments", default=1)

                if n_segments < 1:
                    raise ValueError(f"Invalid n_segments={n_segments}")

                # collect required field names based on enabled branches
                required_keys: List[str] = []
                for branch in flux_branches:
                    required_keys.extend([f"time_{branch}", f"flux_{branch}", f"valid_mask_{branch}"])
                    if cfg.USE_FLUX_ERR:
                        required_keys.append(f"flux_err_{branch}")

                if cfg.USE_PHASE_BRANCH:
                    required_keys.extend(["phase_2min", "flux_folded_2min", "valid_mask_folded_2min"])
                    if cfg.USE_FLUX_ERR:
                        required_keys.append("flux_err_folded_2min")

                if cfg.USE_EXTRA_FEATURES_BRANCH:
                    required_keys.extend(["amplitude", "top_10_periods", "top_10_powers"])

                # verify all required fields are present
                for k in required_keys:
                    if k not in npz:
                        raise KeyError(f"Missing required field: {k}")

                # validate flux branches have consistent dimensions
                for branch in flux_branches:
                    t = np.asarray(npz[f"time_{branch}"]).reshape(-1)
                    f = np.asarray(npz[f"flux_{branch}"]).reshape(-1)
                    m = np.asarray(npz[f"valid_mask_{branch}"]).reshape(-1)

                    if not (len(t) == len(f) == len(m)):
                        raise ValueError(
                            f"Length mismatch in {branch}: time={len(t)}, flux={len(f)}, mask={len(m)}"
                        )

                    if cfg.USE_FLUX_ERR:
                        e = np.asarray(npz[f"flux_err_{branch}"]).reshape(-1)
                        if len(e) != len(t):
                            raise ValueError(f"Length mismatch in {branch}: time={len(t)}, flux_err={len(e)}")

                # validate folded phase branch if enabled
                if cfg.USE_PHASE_BRANCH:
                    p = np.asarray(npz["phase_2min"]).reshape(-1)
                    f = np.asarray(npz["flux_folded_2min"]).reshape(-1)
                    m = np.asarray(npz["valid_mask_folded_2min"]).reshape(-1)

                    if not (len(p) == len(f) == len(m)):
                        raise ValueError(
                            f"Length mismatch in folded branch: phase={len(p)}, flux={len(f)}, mask={len(m)}"
                        )

                    if cfg.USE_FLUX_ERR:
                        e = np.asarray(npz["flux_err_folded_2min"]).reshape(-1)
                        if len(e) != len(p):
                            raise ValueError(f"Length mismatch in folded branch: phase={len(p)}, flux_err={len(e)}")

            # skip if class is not in the allowed list
            if allowed_classes and label not in allowed_classes:
                continue

            # file passed all checks, add it to the final set
            filtered_files.append(fp)
            filtered_labels.append(label)
            filtered_groups.append(tic_id)
            filtered_segments.append(segment_id)
            filtered_n_segments.append(n_segments)

        except Exception as e:
            # log bad files but continue processing others
            bad_files.append((fp.name, str(e)))

    # build label-to-id and id-to-label mappings from sorted unique classes
    class_names = sorted(set(filtered_labels))
    label2id = {label: i for i, label in enumerate(class_names)}
    id2label = {i: label for label, i in label2id.items()}

    # convert to numpy arrays for downstream use
    y = np.array([label2id[label] for label in filtered_labels], dtype=np.int64)
    groups = np.array(filtered_groups, dtype=np.int64)
    segments = np.array(filtered_segments, dtype=np.int64)
    n_segments_arr = np.array(filtered_n_segments, dtype=np.int64)

    # print summary
    print(f"Found total NPZ files: {len(all_files)}")
    print(f"Kept files: {len(filtered_files)}")
    print(f"Unique TIC IDs kept: {len(np.unique(groups))}")
    print(f"Enabled flux branches: {flux_branches}")
    print(f"USE_PHASE_BRANCH: {cfg.USE_PHASE_BRANCH}")
    print(f"USE_EXTRA_FEATURES_BRANCH: {cfg.USE_EXTRA_FEATURES_BRANCH}")

    if len(n_segments_arr) > 0:
        print(
            f"n_segments summary: min={n_segments_arr.min()}, median={np.median(n_segments_arr):.1f}, max={n_segments_arr.max()}"
        )

    if bad_files:
        print(f"Skipped malformed files: {len(bad_files)}")
        for name, err in bad_files[:10]:
            print(f"  {name}: {err}")

    return filtered_files, filtered_labels, y, groups, segments, n_segments_arr, label2id, id2label


def group_stratified_split(
        file_paths: List[Path],
        labels_str: List[str],
        groups: np.ndarray,
        val_size: float,
        test_size: float,
        random_state: int,
) -> Tuple[List[Path], List[Path], List[Path], List[str], List[str], List[str]]:
    """
    Split files into train/validation/test sets while keeping TICs intact to prevent data leakage
    """
    df = pd.DataFrame({"file_path": [str(p) for p in file_paths], "label": labels_str, "group": groups})
    label_per_group = df.groupby("group")["label"].nunique()

    # sanity check
    bad_groups = label_per_group[label_per_group > 1]
    if len(bad_groups) > 0:
        raise ValueError(f"Found TIC IDs with multiple labels: {bad_groups.index.tolist()[:10]}")

    group_df = df.groupby("group", as_index=False).first()[["group", "label"]]
    label_counts = group_df["label"].value_counts()
    stratify_stage1 = group_df["label"].values if label_counts.min() >= 2 else None
    if stratify_stage1 is None:
        print(
            "Warning: not enough groups per class for stratified train/temp split. Falling back to non-stratified split."
        )

    train_groups, temp_groups = train_test_split(
        group_df["group"].values,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=stratify_stage1,
    )

    temp_group_df = group_df[group_df["group"].isin(temp_groups)].copy()
    relative_test_size = test_size / (val_size + test_size)
    temp_counts = temp_group_df["label"].value_counts()
    stratify_stage2 = temp_group_df["label"].values if len(temp_counts) > 1 and temp_counts.min() >= 2 else None
    if stratify_stage2 is None:
        print(
            "Warning: not enough groups per class for stratified val/test split. Falling back to non-stratified split."
        )

    val_groups, test_groups = train_test_split(
        temp_group_df["group"].values,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=stratify_stage2,
    )

    train_groups, val_groups, test_groups = set(train_groups.tolist()), set(val_groups.tolist()), set(
        test_groups.tolist()
    )
    train_files, val_files, test_files, train_labels, val_labels, test_labels = [], [], [], [], [], []

    for fp, label, group in zip(file_paths, labels_str, groups):
        if group in train_groups:
            train_files.append(fp)
            train_labels.append(label)
        elif group in val_groups:
            val_files.append(fp)
            val_labels.append(label)
        elif group in test_groups:
            test_files.append(fp)
            test_labels.append(label)
        else:
            raise RuntimeError(f"Group {group} was not assigned to any split")

    return train_files, val_files, test_files, train_labels, val_labels, test_labels


def summarize_valid_fractions(files: List[Path], branches: List[str]) -> None:
    """
    Print summary statistics for the valid-mask coverage in each branch
    """
    if not files:
        return
    stats = {b: [] for b in branches}
    for fp in files:
        with np.load(fp, allow_pickle=True) as npz:
            for b in branches:
                stats[b].append(float(np.mean(np.asarray(npz[f"valid_mask_{b}"], dtype=bool))))
    for b, vals in stats.items():
        vals = np.asarray(vals, dtype=np.float64)
        print(
            f"{b} valid fraction: mean={np.mean(vals):.4f}, min={np.min(vals):.4f}, p05={np.percentile(vals, 5):.4f}, median={np.median(vals):.4f}"
        )

def compute_tic_level_class_weights(
        train_files: List[Path],
        train_labels_str: List[str],
        label2id: Dict[str, int],
        n_classes: int,
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Compute balanced class weights at the TIC level

    The training objective is segment-level, but the class weight is estimated per TIC since loss is additionally weighted by the segment count
    """
    rows = []
    for fp, label in zip(train_files, train_labels_str):
        with np.load(fp, allow_pickle=True) as npz:
            tic_id = read_int_from_npz(npz, "tic_id")
        rows.append({"tic_id": tic_id, "label": label})

    df = pd.DataFrame(rows)
    label_per_tic = df.groupby("tic_id")["label"].nunique()
    bad = label_per_tic[label_per_tic > 1]

    # sanity check
    if len(bad) > 0:
        raise ValueError(f"Found TIC IDs with multiple labels in training split: {bad.index.tolist()[:10]}")

    tic_df = df.groupby("tic_id", as_index=False).first()
    y_train_tic = np.array([label2id[x] for x in tic_df["label"]], dtype=np.int64)
    missing_train_classes = sorted(set(range(n_classes)) - set(np.unique(y_train_tic)))

    # sanity check
    if missing_train_classes:
        missing_names = [label for label, idx in label2id.items() if idx in missing_train_classes]
        raise ValueError(
            f"These classes are missing from the TIC-level training split: {missing_names}. Use a different random seed or a custom group split."
        )

    class_weights_np = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y_train_tic)
    tic_class_counts = {label: int((tic_df["label"] == label).sum()) for label in sorted(tic_df["label"].unique())}
    print("TIC-level train class counts:", tic_class_counts)
    print("TIC-level class weights:", class_weights_np)
    return torch.tensor(class_weights_np, dtype=torch.float32), tic_df


def warn_if_split_missing_classes(split_name: str, labels_str: List[str], label2id: Dict[str, int],
                                  n_classes: int) -> None:
    """Print a warning if a split does not contain all classes."""

    y_split = np.array([label2id[x] for x in labels_str], dtype=np.int64)
    missing = sorted(set(range(n_classes)) - set(np.unique(y_split)))
    if missing:
        id2label_local = {i: label for label, i in label2id.items()}
        missing_names = [id2label_local[i] for i in missing]
        print(f"Warning: {split_name} split is missing classes: {missing_names}")


def load_record_from_npz(cfg: ExperimentConfig, fp: Union[str, Path], label2id: Dict[str, int]) -> Dict[str, Any]:
    """Load one NPZ file into a Python record ready for tensor conversion."""

    fp = Path(fp)
    with np.load(fp, allow_pickle=True) as npz:
        label = read_label_from_npz(npz)
        tic_id = read_int_from_npz(npz, "tic_id")
        segment_id = read_int_from_npz(npz, "segment_id", default=0)
        n_segments = read_int_from_npz(npz, "n_segments", default=1)
        if n_segments < 1:
            raise ValueError(f"{fp} has invalid n_segments={n_segments}")

        record: Dict[str, Any] = {
            "file_path": str(fp),
            "label": label,
            "label_id": int(label2id[label]),
            "tic_id": int(tic_id),
            "segment_id": int(segment_id),
            "n_segments": int(n_segments),
            "sample_weight": float(1.0 / float(n_segments)),
        }

        for branch in enabled_flux_branches(cfg):
            # Arrays are copied so downstream tensor conversion can safely assume contiguous one-dimensional inputs
            record[f"time_{branch}"] = np.asarray(npz[f"time_{branch}"], dtype=np.float32).reshape(-1).copy()
            record[f"flux_{branch}"] = np.asarray(npz[f"flux_{branch}"], dtype=np.float32).reshape(-1).copy()
            record[f"valid_mask_{branch}"] = np.asarray(npz[f"valid_mask_{branch}"], dtype=bool).reshape(-1).copy()
            if cfg.USE_FLUX_ERR:
                record[f"flux_err_{branch}"] = np.asarray(npz[f"flux_err_{branch}"], dtype=np.float32).reshape(
                    -1).copy()

        if cfg.USE_PHASE_BRANCH:
            record["phase_2min"] = np.asarray(npz["phase_2min"], dtype=np.float32).reshape(-1).copy()
            record["flux_folded_2min"] = np.asarray(npz["flux_folded_2min"], dtype=np.float32).reshape(-1).copy()
            record["valid_mask_folded_2min"] = np.asarray(npz["valid_mask_folded_2min"], dtype=bool).reshape(-1).copy()
            if cfg.USE_FLUX_ERR:
                record["flux_err_folded_2min"] = np.asarray(npz["flux_err_folded_2min"], dtype=np.float32).reshape(
                    -1).copy()

        if cfg.USE_EXTRA_FEATURES_BRANCH:
            amplitude = np.asarray(npz["amplitude"], dtype=np.float32).reshape(-1)
            top_10_periods = np.asarray(npz["top_10_periods"], dtype=np.float32).reshape(-1)
            top_10_powers = np.asarray(npz["top_10_powers"], dtype=np.float32).reshape(-1)
            record["extra_features"] = np.concatenate([amplitude, top_10_periods, top_10_powers], axis=0).astype(
                np.float32, copy=True
            )
    return record


def load_records_in_memory(
        cfg: ExperimentConfig, file_paths: List[Path], label2id: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Load all selected NPZ files into RAM for faster repeated access."""

    records = []
    for i, fp in enumerate(file_paths, start=1):
        records.append(load_record_from_npz(cfg, fp, label2id))
        if i % 1000 == 0:
            print(f"Loaded {i} records into memory...")
    print(f"Loaded {len(records)} records into memory.")
    return records


class LightCurveNPZDataset(Dataset):
    """Torch dataset that lazily or eagerly loads light-curve NPZ records."""

    def __init__(
            self,
            cfg: ExperimentConfig,
            file_paths: List[Path],
            label2id: Dict[str, int],
            load_in_memory: Optional[bool] = True,
    ):
        """Create a dataset from a list of NPZ files and a label mapping."""

        self.cfg = cfg
        self.file_paths = list(file_paths)
        self.label2id = label2id
        self.flux_branches = enabled_flux_branches(cfg)
        self.load_in_memory = cfg.LOAD_DATASET_IN_MEMORY if load_in_memory is None else bool(load_in_memory)
        self.records: Optional[List[Dict[str, Any]]] = None
        if self.load_in_memory:
            self.records = load_records_in_memory(self.cfg, self.file_paths, self.label2id)

    def __len__(self):
        """Return the number of segment files in the dataset."""

        return len(self.file_paths)

    @staticmethod
    def _tensor_1d(x, dtype=torch.float32):
        """Convert an array-like object into a one-dimensional tensor."""

        return torch.tensor(np.asarray(x).reshape(-1), dtype=dtype)

    def _item_from_record(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert a loaded record into the tensor dictionary used by training."""

        item: Dict[str, torch.Tensor] = {}
        for branch in self.flux_branches:
            item[f"time_{branch}"] = self._tensor_1d(record[f"time_{branch}"], dtype=torch.float32)
            item[f"flux_{branch}"] = self._tensor_1d(record[f"flux_{branch}"], dtype=torch.float32)
            item[f"valid_mask_{branch}"] = self._tensor_1d(record[f"valid_mask_{branch}"], dtype=torch.bool)
            if self.cfg.USE_FLUX_ERR:
                item[f"flux_err_{branch}"] = self._tensor_1d(record[f"flux_err_{branch}"], dtype=torch.float32)

        if self.cfg.USE_PHASE_BRANCH:
            item["phase_2min"] = self._tensor_1d(record["phase_2min"], dtype=torch.float32)
            item["flux_folded_2min"] = self._tensor_1d(record["flux_folded_2min"], dtype=torch.float32)
            item["valid_mask_folded_2min"] = self._tensor_1d(record["valid_mask_folded_2min"], dtype=torch.bool)
            if self.cfg.USE_FLUX_ERR:
                item["flux_err_folded_2min"] = self._tensor_1d(record["flux_err_folded_2min"], dtype=torch.float32)

        if self.cfg.USE_EXTRA_FEATURES_BRANCH:
            item["extra_features"] = self._tensor_1d(record["extra_features"], dtype=torch.float32)

        item["labels"] = torch.tensor(record["label_id"], dtype=torch.long)
        item["tic_id"] = torch.tensor(record["tic_id"], dtype=torch.long)
        item["segment_id"] = torch.tensor(record["segment_id"], dtype=torch.long)
        item["n_segments"] = torch.tensor(record["n_segments"], dtype=torch.long)
        item["sample_weight"] = torch.tensor(record["sample_weight"], dtype=torch.float32)
        return item

    def __getitem__(self, idx):
        """Load one sample, either from memory or directly from disk."""

        records = self.records
        if records is not None:
            return self._item_from_record(records[idx])
        record = load_record_from_npz(self.cfg, self.file_paths[idx], self.label2id)
        return self._item_from_record(record)


class LightCurveCollator:
    """Pad and normalize a list of dataset items into one batch."""

    def __init__(self, cfg: ExperimentConfig):
        """Store the configuration that controls batch preprocessing."""

        self.cfg = cfg

    @staticmethod
    def pad_1d_sequences(sequences, pad_value=0.0):
        """Right-pad variable-length 1D tensors and return the padding mask."""

        lengths = [len(x) for x in sequences]
        max_len = max(lengths)
        out = torch.full((len(sequences), max_len), fill_value=pad_value, dtype=sequences[0].dtype)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
        for i, seq in enumerate(sequences):
            n = len(seq)
            out[i, :n] = seq
            mask[i, :n] = True
        return out, mask

    @staticmethod
    def pad_bool_sequences(sequences, pad_value=False):
        """Right-pad boolean sequences used as saved-validity masks."""

        lengths = [len(x) for x in sequences]
        max_len = max(lengths)
        out = torch.full((len(sequences), max_len), fill_value=pad_value, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            n = len(seq)
            out[i, :n] = seq.bool()
        return out

    @staticmethod
    def masked_standardize_1d(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Standardize values using only the positions marked valid by ``mask``."""

        # handle NaN and inf values before computation
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # convert mask to float for arithmetic (True = 1, False = 0)
        mask_f = mask.to(dtype=x.dtype)

        # count valid points per sample, avoiding division by zero
        count = torch.clamp(mask_f.sum(dim=1, keepdim=True), min=1.0)

        # compute mean using only valid positions
        mean = (x * mask_f).sum(dim=1, keepdim=True) / count

        # compute variance: E[(x - mean)^2]
        var = (((x - mean) ** 2) * mask_f).sum(dim=1, keepdim=True) / count

        # compute std with numerical stability
        std = torch.sqrt(torch.clamp(var, min=eps))

        # normalize
        x_norm = (x - mean) / std

        # restore zeros at invalid positions
        return torch.where(mask, x_norm, torch.zeros_like(x_norm))

    def normalize_flux_err_tensor(self, flux_err: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply optional log scaling and masked standardization to flux errors."""

        flux_err = torch.nan_to_num(flux_err, nan=0.0, posinf=0.0, neginf=0.0)

        # zero out invalid positions before transformation
        flux_err = torch.where(mask, flux_err, torch.zeros_like(flux_err))

        # optional: compress the scale with log(1 + x) to handle outliers better
        if self.cfg.LOG1P_FLUX_ERR:
            flux_err = torch.log1p(torch.clamp(flux_err, min=0.0))

        # optional: standardize using valid positions only
        if self.cfg.NORMALIZE_FLUX_ERR:
            flux_err = self.masked_standardize_1d(flux_err, mask)

        # mask out invalid positions again after transformations
        return torch.where(mask, flux_err, torch.zeros_like(flux_err))

    @staticmethod
    def normalize_extra_features(extra_features: torch.Tensor) -> torch.Tensor:
        """Normalize the handcrafted feature vector into a more stable scale.

        The first element is the amplitude; the next 10 values are periods; the
        final 10 are power estimates. We log-scale the amplitude, normalize the
        periods relative to the strongest period, and rescale powers by the per-
        sample maximum so the model sees comparable magnitudes.
        """

        extra_features = torch.nan_to_num(extra_features, nan=0.0, posinf=0.0, neginf=0.0)

        # only normalize if we have 21+ features (1 amplitude + 10 periods + 10 powers)
        if extra_features.size(1) >= 21:
            # extract each type of feature
            amplitude = extra_features[:, 0:1]
            periods = extra_features[:, 1:11]
            powers = extra_features[:, 11:21]

            # log-scale amplitude to compress outliers (additional handling after the sigma clipping during preprocessing)
            amplitude = torch.log1p(torch.clamp(amplitude, min=0.0))

            # normalize periods relative to the strongest (first) period to keep the model from overfitting to absolute period values
            top1 = torch.clamp(periods[:, 0:1], min=1e-6)
            periods = periods / top1

            # normalize powers by the per-sample maximum so all samples see power values in a consistent range [0, 1]
            max_power = torch.clamp(powers.max(dim=1, keepdim=True).values, min=1e-6)
            powers = powers / max_power

            # concatenate transformed features back together
            extra_features = torch.cat([amplitude, periods, powers], dim=1)

        return extra_features

    def build_branch_tensor(self, flux, flux_err=None):
        """Stack branch channels into the model's input layout."""

        features = [flux.unsqueeze(-1)]
        if self.cfg.USE_FLUX_ERR and flux_err is not None:
            flux_err = torch.nan_to_num(flux_err, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(flux_err.unsqueeze(-1))
        return torch.cat(features, dim=-1)

    def __call__(self, features):
        """Collate a batch and apply masking-aware normalization."""

        # stack metadata tensors (same length per batch item)
        batch = {
            "labels": torch.stack([f["labels"] for f in features], dim=0),
            "tic_id": torch.stack([f["tic_id"] for f in features], dim=0),
            "segment_id": torch.stack([f["segment_id"] for f in features], dim=0),
            "n_segments": torch.stack([f["n_segments"] for f in features], dim=0),
            "sample_weight": torch.stack([f["sample_weight"] for f in features], dim=0),
        }

        # process each flux branch: pad sequences, apply masks, and normalize
        for branch in enabled_flux_branches(self.cfg):
            # right-pad time and flux sequences to the same length
            time_tensor, length_mask = self.pad_1d_sequences([f[f"time_{branch}"] for f in features], pad_value=0.0)
            flux, _ = self.pad_1d_sequences([f[f"flux_{branch}"] for f in features], pad_value=0.0)

            # recover the original validity mask before padding
            saved_valid_mask = self.pad_bool_sequences([f[f"valid_mask_{branch}"] for f in features], pad_value=False)

            # load flux errors if present
            if self.cfg.USE_FLUX_ERR:
                flux_err, _ = self.pad_1d_sequences([f[f"flux_err_{branch}"] for f in features], pad_value=0.0)
            else:
                flux_err = None

            # combine length mask with saved validity: a position is valid only if both it wasn't added by padding and the original data marked it as valid
            mask = length_mask & saved_valid_mask

            # sanity check - ensure no sample has all-invalid positions in this branch
            if not mask.any(dim=1).all():
                bad = (~mask.any(dim=1)).nonzero(as_tuple=False).reshape(-1).tolist()
                raise ValueError(f"Found samples with no valid points in branch {branch}: batch indices {bad}")

            # zero out padded and invalid positions so they don't affect statistics
            time_tensor = torch.where(mask, time_tensor, torch.zeros_like(time_tensor))
            flux = torch.where(mask, flux, torch.zeros_like(flux))

            # normalize flux error
            if flux_err is not None:
                flux_err = self.normalize_flux_err_tensor(flux_err, mask)

            # stack flux and error as multi-channel input
            batch[f"{branch}_inputs"] = self.build_branch_tensor(flux=flux, flux_err=flux_err)
            batch[f"{branch}_times"] = time_tensor
            batch[f"{branch}_attention_mask"] = mask

        # process the phase/folded branch
        if self.cfg.USE_PHASE_BRANCH:
            # similar processing as flux branches but using phase instead of time
            phase, length_mask = self.pad_1d_sequences([f["phase_2min"] for f in features], pad_value=0.0)
            flux_folded, _ = self.pad_1d_sequences([f["flux_folded_2min"] for f in features], pad_value=0.0)
            saved_valid_mask = self.pad_bool_sequences([f["valid_mask_folded_2min"] for f in features], pad_value=False)

            if self.cfg.USE_FLUX_ERR:
                flux_err_folded, _ = self.pad_1d_sequences([f["flux_err_folded_2min"] for f in features], pad_value=0.0)
            else:
                flux_err_folded = None

            phase_mask = length_mask & saved_valid_mask

            if not phase_mask.any(dim=1).all():
                bad = (~phase_mask.any(dim=1)).nonzero(as_tuple=False).reshape(-1).tolist()
                raise ValueError(f"Found samples with no valid points in folded phase branch: batch indices {bad}")

            phase = torch.where(phase_mask, phase, torch.zeros_like(phase))
            flux_folded = torch.where(phase_mask, flux_folded, torch.zeros_like(flux_folded))

            if flux_err_folded is not None:
                flux_err_folded = self.normalize_flux_err_tensor(flux_err_folded, phase_mask)

            batch["phase_inputs"] = self.build_branch_tensor(flux=flux_folded, flux_err=flux_err_folded)

            # clamp phase to [0, 1] since it represents a fraction of a period
            batch["phase_times"] = torch.clamp(phase, 0.0, 1.0)
            batch["phase_attention_mask"] = phase_mask

        # process extra features
        if self.cfg.USE_EXTRA_FEATURES_BRANCH:
            extra_features = torch.stack([f["extra_features"] for f in features], dim=0)
            batch["extra_features"] = self.normalize_extra_features(extra_features)

        return batch
