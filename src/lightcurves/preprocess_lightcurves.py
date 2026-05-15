import os
from multiprocessing import Pool, cpu_count

import astropy.units as u
import lightkurve as lk
import numpy as np
import pandas as pd

# %% === Configuration ===

INDEX_CSV = "../../data/provided/index_binary.csv"
LIGHTCURVE_DIR = "../../data/provided"
OUTPUT_DIR = "../../data/processed_4_channel_binary_256_test"

ID_COLUMN = "id"
LABEL_COLUMN = "label"

# ALLOWED_CLASSES = ["eclipsing", "rotating", "pulsating"]
ALLOWED_CLASSES = ["variable", "non-variable"]

USE_FLUX_ERR = True

REMOVE_NANS = True
REMOVE_NONFINITE = True
OUTLIER_SIGMA = 5
MIN_POINTS_AFTER_CLEANING = 32

BASE_CADENCE_MINUTES = 2.0

BINNING_FACTORS = {
    "2min": 1,
    "8min": 4,
    "32min": 16,
    "128min": 64,
}

STANDARDIZE_FLUX = True

# == Segmentation ==
CREATE_SEGMENTS = True

# target length of each saved physical-prefix channel
# with max bin factor 64, each full raw segment has TARGET_LENGTH * 64 raw points
TARGET_LENGTH = 256

# minimum raw points required to keep a partial segment
# set to the second largest view (32-min)
MIN_RAW_POINTS_PER_SEGMENT = TARGET_LENGTH * BINNING_FACTORS["32min"]

# None == no overlap
SEGMENT_STRIDE_RAW_POINTS = TARGET_LENGTH * BINNING_FACTORS["32min"]

# == Period identification ==
PERIODOGRAM_OVERSAMPLE_FACTOR = 10
MIN_PERIOD_FLOOR_DAYS = 0.02
MAX_PERIOD_CEILING_DAYS = 100.0
MIN_PERIOD_MULTIPLE_OF_CADENCE = 5.0
MAX_PERIOD_FRACTION_OF_BASELINE = 0.9
TOP_K_PERIODS = 10

SAVE_FLOAT32 = True
NUM_PROCESSES = max(1, cpu_count() - 1)


# %% === Helper functions ===

def minmax_standardize(
        x: np.ndarray,
        valid_mask: np.ndarray | None = None,
        eps: float = 1e-8,
) -> np.ndarray:
    """
    Min/max standardization.

    If valid_mask is provided, min/max are computed oly over valid points.
    Invalid positions are kept at 0.0 after standardization.

    Output range for valid points:
        [-1, 1]
    """
    x = np.asarray(x, dtype=np.float64)

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        stat_values = x[valid_mask & np.isfinite(x)]
    else:
        stat_values = x[np.isfinite(x)]

    if len(stat_values) == 0:
        return np.zeros_like(x, dtype=np.float64)

    x_min = np.nanmin(stat_values)
    x_max = np.nanmax(stat_values)
    scale = x_max - x_min

    if not np.isfinite(scale) or scale < eps:
        y = np.zeros_like(x, dtype=np.float64)
    else:
        y = 2.0 * ((x - x_min) / scale) - 1.0

    if valid_mask is not None:
        y = np.where(valid_mask, y, 0.0)

    return y


def sigma_clip_mask(x: np.ndarray, sigma: float | None) -> np.ndarray:
    """
    Sigma clipping for outlier handling
    """
    if len(x) == 0:
        return np.zeros(0, dtype=bool)

    if sigma is None:
        return np.ones(len(x), dtype=bool)

    med = np.nanmedian(x)
    std = np.nanstd(x)

    if not np.isfinite(std) or std <= 0:
        return np.ones(len(x), dtype=bool)

    return np.abs(x - med) <= sigma * std


def load_index(index_csv: str) -> pd.DataFrame:
    """
    Helper for loading the index file
    """
    df = pd.read_csv(index_csv)
    required = {ID_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Index file missing required columns: {missing}")

    return df


def read_lightcurve_csv(path: str) -> pd.DataFrame:
    """
    Helper for processing the raw light curve CSV files
    """
    df = pd.read_csv(path)

    required = {"time", "flux"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Light curve file {path} missing required columns: {missing}")

    if "flux_err" not in df.columns:
        df["flux_err"] = np.nan

    return df[["time", "flux", "flux_err"]].copy()


def cast_array(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) if SAVE_FLOAT32 else x.astype(np.float64)


def cast_float(x: float):
    return np.float32(x) if SAVE_FLOAT32 else np.float64(x)


def get_bin_size_days(bin_factor: int) -> float:
    return (BASE_CADENCE_MINUTES * bin_factor) / (24.0 * 60.0)


def get_max_bin_factor() -> int:
    return int(max(BINNING_FACTORS.values()))


def get_raw_segment_size(target_length: int = TARGET_LENGTH) -> int:
    """
    Full raw minimum-cadence segment size
    """
    return int(np.ceil(target_length * get_max_bin_factor()))


def get_raw_segment_stride(target_length: int = TARGET_LENGTH) -> int:
    """
    Raw-point offset between consecutive segment starts

    Defaults to the full raw segment size, preserving the original non-overlapping behavior
    """
    if SEGMENT_STRIDE_RAW_POINTS is None:
        stride = get_raw_segment_size(target_length)
    else:
        stride = int(SEGMENT_STRIDE_RAW_POINTS)

    if stride <= 0:
        raise ValueError(f"SEGMENT_STRIDE_RAW_POINTS must be positive, got {stride}")

    return stride


def clean_flux_err_for_interpolation(flux_err: np.ndarray) -> np.ndarray:
    """
    Replace missing/nonfinite flux errors with a finite fallback
    Missing errors are not converted to zero unless no finite errors exist
    """
    flux_err = np.asarray(flux_err, dtype=np.float64)
    finite = np.isfinite(flux_err) & (flux_err >= 0)

    if np.any(finite):
        fill = np.nanmedian(flux_err[finite])
    else:
        fill = 0.0

    return np.where(finite, flux_err, fill)


def compute_amplitude(flux: np.ndarray) -> float:
    """
    Amplitude computation helper

    amp = max(flux) - min(flux) over valid flux values
    """
    if len(flux) == 0:
        return 0.0

    finite = np.isfinite(flux)

    if not np.any(finite):
        return 0.0

    return float(np.nanmax(flux[finite]) - np.nanmin(flux[finite]))


# %% === Resampling helpers ===

def resample_sequence_to_time_grid(
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        target_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample onto a provided physical time grid.

    Returns:
        target_time
        flux_new
        flux_err_new
        valid_mask
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    flux = np.asarray(flux, dtype=np.float64).reshape(-1)
    flux_err = np.asarray(flux_err, dtype=np.float64).reshape(-1)
    target_time = np.asarray(target_time, dtype=np.float64).reshape(-1)

    target_len = len(target_time)

    if target_len == 0:
        return (
            target_time,
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=bool),
        )

    if len(time) == 0:
        return (
            target_time,
            np.zeros(target_len, dtype=np.float64),
            np.zeros(target_len, dtype=np.float64),
            np.zeros(target_len, dtype=bool),
        )

    finite = np.isfinite(time) & np.isfinite(flux)

    if not np.any(finite):
        return (
            target_time,
            np.zeros(target_len, dtype=np.float64),
            np.zeros(target_len, dtype=np.float64),
            np.zeros(target_len, dtype=bool),
        )

    time = time[finite]
    flux = flux[finite]
    flux_err = flux_err[finite]

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]

    unique_time, unique_idx = np.unique(time, return_index=True)
    time = unique_time
    flux = flux[unique_idx]
    flux_err = flux_err[unique_idx]

    if len(time) == 1:
        flux_new = np.zeros(target_len, dtype=np.float64)
        flux_err_new = np.zeros(target_len, dtype=np.float64)
        valid_mask = np.zeros(target_len, dtype=bool)

        nearest_idx = int(np.argmin(np.abs(target_time - time[0])))
        flux_new[nearest_idx] = flux[0]
        flux_err_new[nearest_idx] = clean_flux_err_for_interpolation(flux_err)[0]
        valid_mask[nearest_idx] = True

        return target_time, flux_new, flux_err_new, valid_mask

    flux_err_clean = clean_flux_err_for_interpolation(flux_err)

    flux_new = np.interp(
        target_time,
        time,
        flux,
        left=0.0,
        right=0.0,
    )

    flux_err_new = np.interp(
        target_time,
        time,
        flux_err_clean,
        left=0.0,
        right=0.0,
    )

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) > 0:
        median_dt = float(np.median(dt))
    else:
        median_dt = get_bin_size_days(1)

    left_idx = np.searchsorted(time, target_time, side="right") - 1
    left_idx = np.clip(left_idx, 0, len(time) - 1)

    right_idx = np.clip(left_idx + 1, 0, len(time) - 1)

    nearest_dist = np.minimum(
        np.abs(target_time - time[left_idx]),
        np.abs(target_time - time[right_idx]),
    )

    within_observed_range = (target_time >= time[0]) & (target_time <= time[-1])
    close_to_observation = nearest_dist <= 2.5 * median_dt

    valid_mask = within_observed_range & close_to_observation

    flux_new = np.where(valid_mask, flux_new, 0.0)
    flux_err_new = np.where(valid_mask, flux_err_new, 0.0)

    return target_time, flux_new, flux_err_new, valid_mask


def resample_folded_curve_by_phase(
        phase: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        target_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample a folded light curve onto a regular phase grid
    Uses periodic extension so interpolation behaves correctly near phase 0/1
    """
    phase = np.asarray(phase, dtype=np.float64).reshape(-1)
    flux = np.asarray(flux, dtype=np.float64).reshape(-1)
    flux_err = np.asarray(flux_err, dtype=np.float64).reshape(-1)

    phase_grid = np.linspace(0, 1, target_len, endpoint=False, dtype=np.float64)

    finite = np.isfinite(phase) & np.isfinite(flux)

    if not np.any(finite):
        return (
            phase_grid,
            np.zeros(target_len, dtype=np.float64),
            np.zeros(target_len, dtype=np.float64),
            np.zeros(target_len, dtype=bool),
        )

    phase = phase[finite]
    flux = flux[finite]
    flux_err = flux_err[finite]

    phase = np.mod(phase, 1.0)

    order = np.argsort(phase)
    phase = phase[order]
    flux = flux[order]
    flux_err = flux_err[order]

    unique_phase, unique_idx = np.unique(phase, return_index=True)
    phase = unique_phase
    flux = flux[unique_idx]
    flux_err = flux_err[unique_idx]

    if len(phase) == 1:
        ferr = clean_flux_err_for_interpolation(flux_err)[0]

        flux_grid = np.full(target_len, flux[0], dtype=np.float64)
        flux_err_grid = np.full(target_len, ferr, dtype=np.float64)
        valid_mask = np.ones(target_len, dtype=bool)

        return phase_grid, flux_grid, flux_err_grid, valid_mask

    flux_err_clean = clean_flux_err_for_interpolation(flux_err)

    phase_ext = np.concatenate([phase - 1.0, phase, phase + 1.0])
    flux_ext = np.concatenate([flux, flux, flux])
    flux_err_ext = np.concatenate([flux_err_clean, flux_err_clean, flux_err_clean])

    flux_grid = np.interp(phase_grid, phase_ext, flux_ext)
    flux_err_grid = np.interp(phase_grid, phase_ext, flux_err_ext)

    nearest_idx = np.searchsorted(phase, phase_grid, side="left")
    left_idx = np.clip(nearest_idx - 1, 0, len(phase) - 1)
    right_idx = np.clip(nearest_idx, 0, len(phase) - 1)

    left_dist = np.abs(phase_grid - phase[left_idx])
    right_dist = np.abs(phase_grid - phase[right_idx])

    left_dist = np.minimum(left_dist, 1.0 - left_dist)
    right_dist = np.minimum(right_dist, 1.0 - right_dist)

    nearest_dist = np.minimum(left_dist, right_dist)

    if len(phase) > 1:
        phase_sorted = np.sort(phase)
        phase_diffs = np.diff(phase_sorted)
        phase_diffs = phase_diffs[np.isfinite(phase_diffs) & (phase_diffs > 0)]

        if len(phase_diffs) > 0:
            median_phase_dt = np.median(phase_diffs)
        else:
            median_phase_dt = 1.0 / len(phase)
    else:
        median_phase_dt = 1.0

    if not np.isfinite(median_phase_dt) or median_phase_dt <= 0:
        median_phase_dt = 1.0 / max(len(phase), 1)

    valid_mask = nearest_dist <= 2.5 * median_phase_dt

    return phase_grid, flux_grid, flux_err_grid, valid_mask


# %% === Physical-prefix branch preprocessing ===
def bin_prefix_to_target_length(
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        bin_factor: int,
        target_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a fixed-length physical-prefix branch from a possibly-short raw segment

    Full-segment behavior:
        2min   bin_factor=1  -> first 1  * target_length raw values
        8min   bin_factor=4  -> first 4  * target_length raw values, binned
        32min  bin_factor=16 -> first 16 * target_length raw values, binned
        128min bin_factor=64 -> first 64 * target_length raw values, binned

    Short-segment behavior:
        unavailable tail values are set to 0 and marked invalid.
    """
    if bin_factor < 1:
        raise ValueError(f"bin_factor must be >= 1, got {bin_factor}")

    required_raw_points = int(bin_factor * target_length)

    if len(time) == 0:
        branch_time = np.arange(target_length, dtype=np.float64) * get_bin_size_days(bin_factor)

        return (
            branch_time,
            np.zeros(target_length, dtype=np.float64),
            np.zeros(target_length, dtype=np.float64),
            np.zeros(target_length, dtype=bool),
        )

    available_raw_points = min(len(time), required_raw_points)

    prefix_time = time[:available_raw_points]
    prefix_flux = flux[:available_raw_points]
    prefix_flux_err = flux_err[:available_raw_points]

    base_dt = get_bin_size_days(1)
    highres_time_grid = np.arange(required_raw_points, dtype=np.float64) * base_dt

    (
        highres_time,
        highres_flux,
        highres_flux_err,
        highres_valid,
    ) = resample_sequence_to_time_grid(
        time=prefix_time,
        flux=prefix_flux,
        flux_err=prefix_flux_err,
        target_time=highres_time_grid,
    )

    if bin_factor == 1:
        return highres_time, highres_flux, highres_flux_err, highres_valid

    t = highres_time.reshape(target_length, bin_factor)
    f = highres_flux.reshape(target_length, bin_factor)
    e = highres_flux_err.reshape(target_length, bin_factor)
    m = highres_valid.reshape(target_length, bin_factor)

    valid_counts = np.sum(m, axis=1)
    has_valid = valid_counts > 0

    binned_time = np.mean(t, axis=1)

    binned_flux = np.zeros(target_length, dtype=np.float64)
    binned_flux_err = np.zeros(target_length, dtype=np.float64)

    binned_flux[has_valid] = (
            np.sum(f[has_valid] * m[has_valid], axis=1) / valid_counts[has_valid]
    )

    binned_flux_err[has_valid] = (
            np.sqrt(np.sum((e[has_valid] * m[has_valid]) ** 2, axis=1))
            / valid_counts[has_valid]
    )

    binned_valid = has_valid

    return binned_time, binned_flux, binned_flux_err, binned_valid


def preprocess_branch_fixed_prefix(
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        bin_factor: int,
        target_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build one fixed-length physical-prefix branch.
    """
    branch_time, branch_flux, branch_flux_err, branch_valid = bin_prefix_to_target_length(
        time=time,
        flux=flux,
        flux_err=flux_err,
        bin_factor=bin_factor,
        target_length=target_length,
    )

    branch_n_points_pre_standardization = len(branch_flux)

    if STANDARDIZE_FLUX:
        branch_flux = minmax_standardize(branch_flux, valid_mask=branch_valid)

    return (
        branch_time,
        branch_flux,
        branch_flux_err,
        branch_valid,
        branch_n_points_pre_standardization,
    )


# %% === Segmentation ===

def split_into_fixed_raw_segments(
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        target_length: int,
):
    """
    Split cleaned, sorted, minimum-cadence light curve into fixed raw windows.

    Raw segment length is:
        TARGET_LENGTH * max(BINNING_FACTORS)

    Segment start offset is:
        get_raw_segment_stride(target_length)

    This allows overlap when:
        SEGMENT_STRIDE_RAW_POINTS < raw_segment_size

    Incomplete final segments are kept if they contain at least MIN_RAW_POINTS_PER_SEGMENT raw points.

    Returns tuples:
        (
            segment_idx,
            segment_start_raw_index,
            t_seg,
            f_seg,
            e_seg,
        )
    """
    n = len(time)
    raw_segment_size = get_raw_segment_size(target_length)
    raw_segment_stride = get_raw_segment_stride(target_length)

    if raw_segment_size <= 0:
        raise ValueError(f"Invalid raw_segment_size: {raw_segment_size}")

    if raw_segment_stride <= 0:
        raise ValueError(f"Invalid raw_segment_stride: {raw_segment_stride}")

    if n < MIN_RAW_POINTS_PER_SEGMENT:
        return []

    segments = []

    segment_idx = 0
    start = 0

    # iteratively create segments until reaching end of the curve
    while start < n:
        end = min(start + raw_segment_size, n)

        t_seg = time[start:end]
        f_seg = flux[start:end]
        e_seg = flux_err[start:end]

        if len(t_seg) >= MIN_RAW_POINTS_PER_SEGMENT:
            t_seg = t_seg - t_seg[0]
            segments.append((segment_idx, start, t_seg, f_seg, e_seg))
            segment_idx += 1

        start += raw_segment_stride

    return segments


# %% === Period and folding ===
def fold_lightcurve(
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Folds the light curve alongside the top period
    """
    if len(time) == 0:
        return time.copy(), flux.copy(), flux_err.copy()

    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"Invalid folding period: {period}")

    phase = np.mod(time, period) / period
    order = np.argsort(phase)

    return phase[order], flux[order], flux_err[order]


def compute_periodogram_features_lightkurve(
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        top_k: int = TOP_K_PERIODS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Use the Lomb-Scargle method implemented in lightkurve to compute the periodogram features.
    """
    # handling edge cases and sanity checks
    if len(time) < MIN_POINTS_AFTER_CLEANING:
        raise ValueError(
            f"Too few points to estimate period: {len(time)} < {MIN_POINTS_AFTER_CLEANING}"
        )

    if not np.all(np.diff(time) >= 0):
        raise ValueError("Time array must be sorted before period estimation.")

    baseline = float(time[-1] - time[0])

    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(f"Invalid time baseline for period search: {baseline}")

    # initial setup
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        raise ValueError("Unable to infer cadence for period search.")

    median_dt = float(np.median(dt))

    minimum_period = max(
        MIN_PERIOD_FLOOR_DAYS,
        MIN_PERIOD_MULTIPLE_OF_CADENCE * median_dt,
    )

    maximum_period = min(
        MAX_PERIOD_CEILING_DAYS,
        MAX_PERIOD_FRACTION_OF_BASELINE * baseline,
    )

    if maximum_period <= minimum_period:
        maximum_period = max(minimum_period * 1.1, minimum_period + 1e-4)

    # light curve preparation
    if USE_FLUX_ERR and np.any(np.isfinite(flux_err) & (flux_err > 0)):
        flux_err_clean = clean_flux_err_for_interpolation(flux_err)

        lc = lk.LightCurve(
            time=time * u.day,
            flux=flux,
            flux_err=flux_err_clean,
        )
    else:
        lc = lk.LightCurve(
            time=time * u.day,
            flux=flux,
        )

    # compute periodogram
    pg = lc.to_periodogram(
        method="lombscargle",
        minimum_period=minimum_period * u.day,
        maximum_period=maximum_period * u.day,
        oversample_factor=PERIODOGRAM_OVERSAMPLE_FACTOR,
    )

    # extract values
    top_1_period = float(pg.period_at_max_power.to_value(u.day))

    periods = np.asarray(pg.period.to_value(u.day), dtype=np.float64)
    powers = np.asarray(pg.power.value, dtype=np.float64)

    valid = np.isfinite(periods) & np.isfinite(powers) & (periods > 0)
    periods = periods[valid]
    powers = powers[valid]

    if len(periods) == 0:
        raise ValueError("Periodogram returned no valid period/power samples.")

    order = np.argsort(powers)[::-1]
    top_periods = periods[order][:top_k]
    top_powers = powers[order][:top_k]

    if len(top_periods) < top_k:
        pad_n = top_k - len(top_periods)
        top_periods = np.pad(top_periods, (0, pad_n), constant_values=np.nan)
        top_powers = np.pad(top_powers, (0, pad_n), constant_values=np.nan)

    if not np.isfinite(top_1_period) or top_1_period <= 0:
        raise ValueError(f"Invalid top_1 period returned by Lightkurve: {top_1_period}")

    return top_1_period, top_periods, top_powers


# %% === Main preprocessing ===
def preprocess_single_lightcurve(
        lc_id: int,
        lc_label: str,
        lightcurve_dir: str,
) -> list[dict]:
    # initial setup
    lc_path = os.path.join(lightcurve_dir, f"{lc_id}.lc")

    if not os.path.exists(lc_path):
        raise FileNotFoundError(f"Missing lightcurve file: {lc_path}")

    df = read_lightcurve_csv(lc_path)

    time = np.asarray(df["time"].values, dtype=np.float64)
    flux = np.asarray(df["flux"].values, dtype=np.float64)
    flux_err = np.asarray(df["flux_err"].values, dtype=np.float64)

    original_n_points = len(time)

    # invalid point detection
    mask = np.ones(original_n_points, dtype=bool)

    # cleaning
    if REMOVE_NONFINITE:
        finite_mask = np.isfinite(time) & np.isfinite(flux)

        if USE_FLUX_ERR:
            finite_mask &= np.isfinite(flux_err) | np.isnan(flux_err)

        mask &= finite_mask

    if REMOVE_NANS:
        nan_mask = (~np.isnan(time)) & (~np.isnan(flux))
        mask &= nan_mask

    # masking invalid positions
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    if len(time) < MIN_POINTS_AFTER_CLEANING:
        raise ValueError(
            f"Too few points after finite/NaN filtering: "
            f"{len(time)} < {MIN_POINTS_AFTER_CLEANING}"
        )

    # cleaning outliers
    outlier_mask = sigma_clip_mask(flux, sigma=OUTLIER_SIGMA)
    time = time[outlier_mask]
    flux = flux[outlier_mask]
    flux_err = flux_err[outlier_mask]

    if len(time) < MIN_POINTS_AFTER_CLEANING:
        raise ValueError(
            f"Too few points after outlier removal: "
            f"{len(time)} < {MIN_POINTS_AFTER_CLEANING}"
        )

    # additionally failsafe sort by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]
    flux_err = flux_err[sort_idx]

    time = time - time[0]

    cleaned_raw_n_points = len(time)

    # standardize flux
    period_flux = minmax_standardize(flux)

    # compute periodogram
    top_1_period, top_10_periods, top_10_powers = compute_periodogram_features_lightkurve(
        time=time,
        flux=period_flux,
        flux_err=flux_err,
        top_k=TOP_K_PERIODS,
    )

    # split into segments
    if CREATE_SEGMENTS:
        segments_raw = split_into_fixed_raw_segments(
            time=time,
            flux=flux,
            flux_err=flux_err,
            target_length=TARGET_LENGTH,
        )

    # 1 segment if splitting disable
    else:
        raw_segment_size = get_raw_segment_size(TARGET_LENGTH)
        segment_length = min(len(time), raw_segment_size)

        if segment_length < MIN_RAW_POINTS_PER_SEGMENT:
            raise ValueError(
                f"Not enough points for one segment: "
                f"need at least {MIN_RAW_POINTS_PER_SEGMENT}, got {len(time)}"
            )

        segments_raw = [
            (
                0,
                0,
                time[:segment_length] - time[0],
                flux[:segment_length],
                flux_err[:segment_length],
            )
        ]

    if len(segments_raw) == 0:
        raise ValueError(
            f"No valid fixed raw segments produced. "
            f"cleaned_raw_n_points={cleaned_raw_n_points}, "
            f"required_full_raw_segment_size={get_raw_segment_size(TARGET_LENGTH)}, "
            f"raw_segment_stride={get_raw_segment_stride(TARGET_LENGTH)}, "
            f"MIN_RAW_POINTS_PER_SEGMENT={MIN_RAW_POINTS_PER_SEGMENT}, "
            f"TARGET_LENGTH={TARGET_LENGTH}, "
            f"max_bin_factor={get_max_bin_factor()}"
        )

    processed_segments = []

    # preprocess each segment
    for (segment_idx, segment_start_raw_index, seg_time_raw, seg_flux_raw, seg_flux_err_raw,) in segments_raw:
        processed_branches = {}

        # amplitude computation
        raw_amplitude_full_segment = compute_amplitude(seg_flux_raw)
        raw_amplitude_by_branch = {}

        for branch_name, bin_factor in BINNING_FACTORS.items():
            raw_points_requested = int(bin_factor * TARGET_LENGTH)
            raw_points_available = min(len(seg_flux_raw), raw_points_requested)

            raw_amplitude_by_branch[branch_name] = compute_amplitude(
                seg_flux_raw[:raw_points_available]
            )

        # preprocessing of each flux channel
        for branch_name, bin_factor in BINNING_FACTORS.items():
            (
                branch_time,
                branch_flux,
                branch_flux_err,
                branch_valid,
                branch_n_points,
            ) = preprocess_branch_fixed_prefix(
                time=seg_time_raw,
                flux=seg_flux_raw,
                flux_err=seg_flux_err_raw,
                bin_factor=bin_factor,
                target_length=TARGET_LENGTH,
            )

            raw_points_requested = int(bin_factor * TARGET_LENGTH)
            raw_points_available = min(len(seg_time_raw), raw_points_requested)

            processed_branches[branch_name] = {
                "time": cast_array(branch_time),
                "flux": cast_array(branch_flux),
                "flux_err": cast_array(branch_flux_err),
                "valid_mask": branch_valid.astype(bool),
                "valid_fraction": cast_float(np.mean(branch_valid)),
                "valid_points": np.int32(np.sum(branch_valid)),
                "segment_n_points": np.int32(branch_n_points),
                "saved_length": np.int32(len(branch_time)),
                "bin_factor": np.int32(bin_factor),
                "bin_size_days": cast_float(get_bin_size_days(bin_factor)),
                "raw_points_requested": np.int32(raw_points_requested),
                "raw_points_available": np.int32(raw_points_available),
                "raw_points_consumed": np.int32(raw_points_available),
            }

        # phase-folded
        seg_phase_2min, seg_flux_folded_2min, seg_flux_err_folded_2min = fold_lightcurve(
            time=np.asarray(seg_time_raw, dtype=np.float64),
            flux=np.asarray(seg_flux_raw, dtype=np.float64),
            flux_err=np.asarray(seg_flux_err_raw, dtype=np.float64),
            period=top_1_period,
        )

        saved_folded_length_2min_before_resample = len(seg_phase_2min)

        (
            seg_phase_2min,
            seg_flux_folded_2min,
            seg_flux_err_folded_2min,
            seg_folded_valid_mask_2min,
        ) = resample_folded_curve_by_phase(
            phase=seg_phase_2min,
            flux=seg_flux_folded_2min,
            flux_err=seg_flux_err_folded_2min,
            target_len=TARGET_LENGTH,
        )

        # flux standardization
        if STANDARDIZE_FLUX:
            seg_flux_folded_2min = minmax_standardize(
                seg_flux_folded_2min,
                valid_mask=seg_folded_valid_mask_2min,
            )

        sample = {
            "phase_2min": cast_array(seg_phase_2min),
            "flux_folded_2min": cast_array(seg_flux_folded_2min),
            "flux_err_folded_2min": cast_array(seg_flux_err_folded_2min),
            "valid_mask_folded_2min": seg_folded_valid_mask_2min.astype(bool),
            "valid_fraction_folded_2min": cast_float(np.mean(seg_folded_valid_mask_2min)),
            "valid_points_folded_2min": np.int32(np.sum(seg_folded_valid_mask_2min)),

            "label": str(lc_label),
            "label_original": str(lc_label),
            "tic_id": np.int64(lc_id),
            "segment_id": np.int32(segment_idx),
            "segment_start_raw_index": np.int32(segment_start_raw_index),
            "n_segments_total": np.int32(len(segments_raw)),

            "amplitude": cast_float(raw_amplitude_full_segment),
            "amplitude_raw_full_segment": cast_float(raw_amplitude_full_segment),

            "top_1_period": cast_float(top_1_period),
            "top_10_periods": cast_array(top_10_periods),
            "top_10_powers": cast_array(top_10_powers),

            "original_n_points": np.int32(original_n_points),
            "cleaned_n_points_raw": np.int32(cleaned_raw_n_points),
            "raw_segment_size": np.int32(get_raw_segment_size(TARGET_LENGTH)),
            "raw_segment_stride": np.int32(get_raw_segment_stride(TARGET_LENGTH)),
            "actual_raw_segment_length": np.int32(len(seg_time_raw)),
            "max_bin_factor": np.int32(get_max_bin_factor()),
            "target_length": np.int32(TARGET_LENGTH),
            "min_raw_points_per_segment": np.int32(MIN_RAW_POINTS_PER_SEGMENT),

            "saved_folded_length_2min": np.int32(len(seg_phase_2min)),
            "folded_n_points_2min_pre_resample": np.int32(
                saved_folded_length_2min_before_resample
            ),
        }

        for branch_name, amplitude_value in raw_amplitude_by_branch.items():
            sample[f"amplitude_raw_{branch_name}_prefix"] = cast_float(amplitude_value)

        for branch_name, branch_data in processed_branches.items():
            sample[f"time_{branch_name}"] = branch_data["time"]
            sample[f"flux_{branch_name}"] = branch_data["flux"]
            sample[f"flux_err_{branch_name}"] = branch_data["flux_err"]
            sample[f"valid_mask_{branch_name}"] = branch_data["valid_mask"]

            sample[f"valid_fraction_{branch_name}"] = branch_data["valid_fraction"]
            sample[f"valid_points_{branch_name}"] = branch_data["valid_points"]

            sample[f"segment_n_points_{branch_name}"] = branch_data["segment_n_points"]
            sample[f"saved_length_{branch_name}"] = branch_data["saved_length"]
            sample[f"bin_factor_{branch_name}"] = branch_data["bin_factor"]
            sample[f"bin_size_days_{branch_name}"] = branch_data["bin_size_days"]

            sample[f"raw_points_requested_{branch_name}"] = branch_data[
                "raw_points_requested"
            ]
            sample[f"raw_points_available_{branch_name}"] = branch_data[
                "raw_points_available"
            ]
            sample[f"raw_points_consumed_{branch_name}"] = branch_data[
                "raw_points_consumed"
            ]

        processed_segments.append(sample)

    return processed_segments


# %% === Saving ===
def save_processed_lightcurve(output_path: str, sample: dict) -> None:
    """
    .npz saving helper.
    """
    save_kwargs = {
        # 2 min
        "time_2min": sample["time_2min"],
        "flux_2min": sample["flux_2min"],
        "flux_err_2min": sample["flux_err_2min"],
        "valid_mask_2min": sample["valid_mask_2min"],
        "valid_fraction_2min": sample["valid_fraction_2min"],
        "valid_points_2min": sample["valid_points_2min"],

        # 8 min
        "time_8min": sample["time_8min"],
        "flux_8min": sample["flux_8min"],
        "flux_err_8min": sample["flux_err_8min"],
        "valid_mask_8min": sample["valid_mask_8min"],
        "valid_fraction_8min": sample["valid_fraction_8min"],
        "valid_points_8min": sample["valid_points_8min"],

        # 32 min
        "time_32min": sample["time_32min"],
        "flux_32min": sample["flux_32min"],
        "flux_err_32min": sample["flux_err_32min"],
        "valid_mask_32min": sample["valid_mask_32min"],
        "valid_fraction_32min": sample["valid_fraction_32min"],
        "valid_points_32min": sample["valid_points_32min"],

        # 128 min
        "time_128min": sample["time_128min"],
        "flux_128min": sample["flux_128min"],
        "flux_err_128min": sample["flux_err_128min"],
        "valid_mask_128min": sample["valid_mask_128min"],
        "valid_fraction_128min": sample["valid_fraction_128min"],
        "valid_points_128min": sample["valid_points_128min"],

        # folded 2 min
        "phase_2min": sample["phase_2min"],
        "flux_folded_2min": sample["flux_folded_2min"],
        "flux_err_folded_2min": sample["flux_err_folded_2min"],
        "valid_mask_folded_2min": sample["valid_mask_folded_2min"],
        "valid_fraction_folded_2min": sample["valid_fraction_folded_2min"],
        "valid_points_folded_2min": sample["valid_points_folded_2min"],

        # labels and identifiers
        "label": sample["label"],
        "label_original": sample["label_original"],
        "tic_id": sample["tic_id"],
        "segment_id": sample["segment_id"],
        "segment_start_raw_index": sample["segment_start_raw_index"],

        # Number of segments produced from this TIC/light curve.
        "n_segments": sample["n_segments_total"],
        "n_segments_total": sample["n_segments_total"],

        # amplitudes calculated before standardization
        "amplitude": sample["amplitude"],
        "amplitude_raw_full_segment": sample["amplitude_raw_full_segment"],
        "amplitude_raw_2min_prefix": sample["amplitude_raw_2min_prefix"],
        "amplitude_raw_8min_prefix": sample["amplitude_raw_8min_prefix"],
        "amplitude_raw_32min_prefix": sample["amplitude_raw_32min_prefix"],
        "amplitude_raw_128min_prefix": sample["amplitude_raw_128min_prefix"],

        # period features
        "top_1_period": sample["top_1_period"],
        "top_10_periods": sample["top_10_periods"],
        "top_10_powers": sample["top_10_powers"],

        # metadata
        "original_n_points": sample["original_n_points"],
        "cleaned_n_points_raw": sample["cleaned_n_points_raw"],
        "raw_segment_size": sample["raw_segment_size"],
        "raw_segment_stride": sample["raw_segment_stride"],
        "actual_raw_segment_length": sample["actual_raw_segment_length"],
        "max_bin_factor": sample["max_bin_factor"],
        "target_length": sample["target_length"],
        "min_raw_points_per_segment": sample["min_raw_points_per_segment"],

        "segment_n_points_2min": sample["segment_n_points_2min"],
        "segment_n_points_8min": sample["segment_n_points_8min"],
        "segment_n_points_32min": sample["segment_n_points_32min"],
        "segment_n_points_128min": sample["segment_n_points_128min"],

        "saved_length_2min": sample["saved_length_2min"],
        "saved_length_8min": sample["saved_length_8min"],
        "saved_length_32min": sample["saved_length_32min"],
        "saved_length_128min": sample["saved_length_128min"],

        "saved_folded_length_2min": sample["saved_folded_length_2min"],
        "folded_n_points_2min_pre_resample": sample[
            "folded_n_points_2min_pre_resample"
        ],

        "bin_factor_2min": sample["bin_factor_2min"],
        "bin_factor_8min": sample["bin_factor_8min"],
        "bin_factor_32min": sample["bin_factor_32min"],
        "bin_factor_128min": sample["bin_factor_128min"],

        "bin_size_days_2min": sample["bin_size_days_2min"],
        "bin_size_days_8min": sample["bin_size_days_8min"],
        "bin_size_days_32min": sample["bin_size_days_32min"],
        "bin_size_days_128min": sample["bin_size_days_128min"],

        "raw_points_requested_2min": sample["raw_points_requested_2min"],
        "raw_points_requested_8min": sample["raw_points_requested_8min"],
        "raw_points_requested_32min": sample["raw_points_requested_32min"],
        "raw_points_requested_128min": sample["raw_points_requested_128min"],

        "raw_points_available_2min": sample["raw_points_available_2min"],
        "raw_points_available_8min": sample["raw_points_available_8min"],
        "raw_points_available_32min": sample["raw_points_available_32min"],
        "raw_points_available_128min": sample["raw_points_available_128min"],

        "raw_points_consumed_2min": sample["raw_points_consumed_2min"],
        "raw_points_consumed_8min": sample["raw_points_consumed_8min"],
        "raw_points_consumed_32min": sample["raw_points_consumed_32min"],
        "raw_points_consumed_128min": sample["raw_points_consumed_128min"],
    }

    np.savez_compressed(output_path, **save_kwargs)


# %% === Worker ===
def process_worker(args):
    """
    Helper function for processing individual TICs.
    """
    row_idx, lc_id, lc_label, lightcurve_dir, output_dir = args

    try:
        samples = preprocess_single_lightcurve(
            lc_id=lc_id,
            lc_label=lc_label,
            lightcurve_dir=lightcurve_dir,
        )

        final_label = samples[0]["label"]

        if ALLOWED_CLASSES is not None and final_label not in ALLOWED_CLASSES:
            return {
                "success": False,
                "skipped": True,
                "id": lc_id,
                "label": final_label,
                "error": f"Filtered out by ALLOWED_CLASSES: {final_label}",
            }

        output_paths = []

        saved_lengths_2min = []
        saved_lengths_8min = []
        saved_lengths_32min = []
        saved_lengths_128min = []
        saved_folded_lengths_2min = []

        valid_fractions_2min = []
        valid_fractions_8min = []
        valid_fractions_32min = []
        valid_fractions_128min = []

        actual_raw_segment_lengths = []
        segment_start_raw_indices = []

        top_1_period = float(samples[0]["top_1_period"])
        cleaned_n_points_raw = int(samples[0]["cleaned_n_points_raw"])

        for sample in samples:
            segment_id = int(sample["segment_id"])
            output_path = os.path.join(output_dir, f"{lc_id}_seg{segment_id:04d}.npz")

            save_processed_lightcurve(output_path, sample)
            output_paths.append(output_path)

            saved_lengths_2min.append(int(sample["saved_length_2min"]))
            saved_lengths_8min.append(int(sample["saved_length_8min"]))
            saved_lengths_32min.append(int(sample["saved_length_32min"]))
            saved_lengths_128min.append(int(sample["saved_length_128min"]))
            saved_folded_lengths_2min.append(int(sample["saved_folded_length_2min"]))

            valid_fractions_2min.append(float(sample["valid_fraction_2min"]))
            valid_fractions_8min.append(float(sample["valid_fraction_8min"]))
            valid_fractions_32min.append(float(sample["valid_fraction_32min"]))
            valid_fractions_128min.append(float(sample["valid_fraction_128min"]))

            actual_raw_segment_lengths.append(int(sample["actual_raw_segment_length"]))
            segment_start_raw_indices.append(int(sample["segment_start_raw_index"]))

        return {
            "success": True,
            "skipped": False,
            "id": lc_id,
            "label": final_label,
            "top_1_period": top_1_period,
            "output_paths": output_paths,
            "n_segments": len(samples),
            "saved_lengths_2min": saved_lengths_2min,
            "saved_lengths_8min": saved_lengths_8min,
            "saved_lengths_32min": saved_lengths_32min,
            "saved_lengths_128min": saved_lengths_128min,
            "saved_folded_lengths_2min": saved_folded_lengths_2min,
            "valid_fractions_2min": valid_fractions_2min,
            "valid_fractions_8min": valid_fractions_8min,
            "valid_fractions_32min": valid_fractions_32min,
            "valid_fractions_128min": valid_fractions_128min,
            "actual_raw_segment_lengths": actual_raw_segment_lengths,
            "segment_start_raw_indices": segment_start_raw_indices,
            "cleaned_n_points_raw": cleaned_n_points_raw,
        }

    except Exception as e:
        return {
            "success": False,
            "skipped": False,
            "id": lc_id,
            "label": str(lc_label),
            "error": str(e),
        }


# %% === Driver Code ===

def preprocess_all():
    """
    Driver function that handles the processing of the entire dataset
    """
    # initial setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index_df = load_index(INDEX_CSV)

    tasks = []

    for row_idx, row in index_df.iterrows():
        lc_id = int(row[ID_COLUMN])
        lc_label = str(row[LABEL_COLUMN])

        if lc_label in ALLOWED_CLASSES:
            tasks.append((row_idx, lc_id, lc_label, LIGHTCURVE_DIR, OUTPUT_DIR))

    raw_segment_size = get_raw_segment_size(TARGET_LENGTH)
    raw_segment_stride = get_raw_segment_stride(TARGET_LENGTH)
    segment_overlap_raw_points = max(0, raw_segment_size - raw_segment_stride)
    segment_overlap_fraction = max(0.0, 1.0 - raw_segment_stride / raw_segment_size)

    # --- start diagnostics ---
    print(f"Found {len(tasks)} light curves in index.")
    print(f"Using {NUM_PROCESSES} processes.")
    print(f"Saving outputs to: {OUTPUT_DIR}")
    print(f"TARGET_LENGTH: {TARGET_LENGTH}")
    print(f"BASE_CADENCE_MINUTES: {BASE_CADENCE_MINUTES}")
    print(f"BINNING_FACTORS: {BINNING_FACTORS}")
    print(f"MAX_BIN_FACTOR: {get_max_bin_factor()}")
    print(f"RAW_SEGMENT_SIZE: {raw_segment_size}")
    print(f"RAW_SEGMENT_STRIDE: {raw_segment_stride}")
    print(f"SEGMENT_OVERLAP_RAW_POINTS: {segment_overlap_raw_points}")
    print(f"SEGMENT_OVERLAP_FRACTION: {segment_overlap_fraction:.4f}")
    print(f"MIN_RAW_POINTS_PER_SEGMENT: {MIN_RAW_POINTS_PER_SEGMENT}")
    print(f"PERIODOGRAM_OVERSAMPLE_FACTOR: {PERIODOGRAM_OVERSAMPLE_FACTOR}")
    print(f"TOP_K_PERIODS: {TOP_K_PERIODS}")

    success_ids = []
    skipped_ids = []
    failed_ids = []

    label_counts = {}
    cleaned_lengths_raw = []
    top_1_periods = []

    saved_lengths_2min_all = []
    saved_lengths_8min_all = []
    saved_lengths_32min_all = []
    saved_lengths_128min_all = []
    saved_folded_lengths_2min_all = []

    valid_fractions_2min_all = []
    valid_fractions_8min_all = []
    valid_fractions_32min_all = []
    valid_fractions_128min_all = []

    actual_raw_segment_lengths_all = []
    segment_start_raw_indices_all = []

    total_saved_segments = 0

    # uses multiprocessing pool to speed up the processing
    # maxtasksperchild kills and launches again the worker process after some amount of light curves was processed
    # this prevents any possible memory leaks and memory overflow
    with Pool(processes=NUM_PROCESSES, maxtasksperchild=10) as pool:
        for i, result in enumerate(
                pool.imap_unordered(process_worker, tasks, chunksize=1),
                # imap for sharing logging information between processes
                start=1,
        ):
            if result["success"]:
                success_ids.append(result["id"])

                label = result["label"]
                label_counts[label] = label_counts.get(label, 0) + result["n_segments"]

                cleaned_lengths_raw.append(result["cleaned_n_points_raw"])
                top_1_periods.append(result["top_1_period"])

                saved_lengths_2min_all.extend(result["saved_lengths_2min"])
                saved_lengths_8min_all.extend(result["saved_lengths_8min"])
                saved_lengths_32min_all.extend(result["saved_lengths_32min"])
                saved_lengths_128min_all.extend(result["saved_lengths_128min"])
                saved_folded_lengths_2min_all.extend(result["saved_folded_lengths_2min"])

                valid_fractions_2min_all.extend(result["valid_fractions_2min"])
                valid_fractions_8min_all.extend(result["valid_fractions_8min"])
                valid_fractions_32min_all.extend(result["valid_fractions_32min"])
                valid_fractions_128min_all.extend(result["valid_fractions_128min"])

                actual_raw_segment_lengths_all.extend(result["actual_raw_segment_lengths"])
                segment_start_raw_indices_all.extend(result["segment_start_raw_indices"])

                total_saved_segments += result["n_segments"]

            else:
                if result.get("skipped", False):
                    skipped_ids.append(result["id"])
                else:
                    failed_ids.append(result["id"])
                    print(f"Failed ID {result['id']}: {result['error']}")

            # debug log each 25 processed light curves
            if i % 25 == 0 or i == len(tasks):
                print(
                    f"Progress {i}/{len(tasks)} | "
                    f"ok_tics={len(success_ids)} | "
                    f"saved_segments={total_saved_segments} | "
                    f"skipped={len(skipped_ids)} | "
                    f"failed={len(failed_ids)}"
                )

    label_names = np.asarray(list(label_counts.keys()), dtype=str)
    label_count_values = np.asarray(
        [label_counts[label] for label in label_names],
        dtype=np.int32,
    )

    # --- save metadata ---
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.npz")

    metadata_kwargs = {
        "success_ids": np.asarray(success_ids, dtype=np.int64),
        "skipped_ids": np.asarray(skipped_ids, dtype=np.int64),
        "failed_ids": np.asarray(failed_ids, dtype=np.int64),

        "label_names": label_names,
        "label_counts": label_count_values,

        "cleaned_lengths_raw": np.asarray(cleaned_lengths_raw, dtype=np.int32),
        "top_1_periods": np.asarray(
            top_1_periods,
            dtype=np.float32 if SAVE_FLOAT32 else np.float64,
        ),

        "saved_lengths_2min": np.asarray(saved_lengths_2min_all, dtype=np.int32),
        "saved_lengths_8min": np.asarray(saved_lengths_8min_all, dtype=np.int32),
        "saved_lengths_32min": np.asarray(saved_lengths_32min_all, dtype=np.int32),
        "saved_lengths_128min": np.asarray(saved_lengths_128min_all, dtype=np.int32),
        "saved_folded_lengths_2min": np.asarray(
            saved_folded_lengths_2min_all,
            dtype=np.int32,
        ),

        "valid_fractions_2min": np.asarray(
            valid_fractions_2min_all,
            dtype=np.float32 if SAVE_FLOAT32 else np.float64,
        ),
        "valid_fractions_8min": np.asarray(
            valid_fractions_8min_all,
            dtype=np.float32 if SAVE_FLOAT32 else np.float64,
        ),
        "valid_fractions_32min": np.asarray(
            valid_fractions_32min_all,
            dtype=np.float32 if SAVE_FLOAT32 else np.float64,
        ),
        "valid_fractions_128min": np.asarray(
            valid_fractions_128min_all,
            dtype=np.float32 if SAVE_FLOAT32 else np.float64,
        ),

        "actual_raw_segment_lengths": np.asarray(
            actual_raw_segment_lengths_all,
            dtype=np.int32,
        ),
        "segment_start_raw_indices": np.asarray(
            segment_start_raw_indices_all,
            dtype=np.int32,
        ),

        "total_saved_segments": np.int32(total_saved_segments),
        "target_length": np.int32(TARGET_LENGTH),
        "max_bin_factor": np.int32(get_max_bin_factor()),
        "raw_segment_size": np.int32(raw_segment_size),
        "raw_segment_stride": np.int32(raw_segment_stride),
        "segment_overlap_raw_points": np.int32(segment_overlap_raw_points),
        "segment_overlap_fraction": np.float32(segment_overlap_fraction),
        "min_raw_points_per_segment": np.int32(MIN_RAW_POINTS_PER_SEGMENT),
        "base_cadence_minutes": np.float32(BASE_CADENCE_MINUTES),
    }

    np.savez(metadata_path, **metadata_kwargs)

    # --- diagnostics ---
    print("\n" + "=" * 60)
    print("Preprocessing finished")
    print(f"Successful TICs: {len(success_ids)}")
    print(f"Saved segments:   {total_saved_segments}")
    print(f"Skipped TICs:     {len(skipped_ids)}")
    print(f"Failed TICs:      {len(failed_ids)}")
    print(f"Raw segment size: {raw_segment_size}")
    print(f"Raw segment stride: {raw_segment_stride}")
    print(f"Segment overlap raw points: {segment_overlap_raw_points}")
    print(f"Segment overlap fraction: {segment_overlap_fraction:.4f}")

    if cleaned_lengths_raw:
        print(f"Min cleaned raw length: {int(np.min(cleaned_lengths_raw))}")
        print(f"Max cleaned raw length: {int(np.max(cleaned_lengths_raw))}")
        print(f"Mean cleaned raw length: {float(np.mean(cleaned_lengths_raw)):.2f}")

    if actual_raw_segment_lengths_all:
        print(f"Min actual raw segment length: {int(np.min(actual_raw_segment_lengths_all))}")
        print(f"Max actual raw segment length: {int(np.max(actual_raw_segment_lengths_all))}")
        print(
            f"Mean actual raw segment length: "
            f"{float(np.mean(actual_raw_segment_lengths_all)):.2f}"
        )

    if top_1_periods:
        print(f"Min top_1 period: {float(np.min(top_1_periods)):.6f} d")
        print(f"Max top_1 period: {float(np.max(top_1_periods)):.6f} d")
        print(f"Mean top_1 period: {float(np.mean(top_1_periods)):.6f} d")

    if saved_lengths_2min_all:
        print(f"Saved 2min length per segment: {int(saved_lengths_2min_all[0])}")

    if saved_lengths_8min_all:
        print(f"Saved 8min length per segment: {int(saved_lengths_8min_all[0])}")

    if saved_lengths_32min_all:
        print(f"Saved 32min length per segment: {int(saved_lengths_32min_all[0])}")

    if saved_lengths_128min_all:
        print(f"Saved 128min length per segment: {int(saved_lengths_128min_all[0])}")

    if saved_folded_lengths_2min_all:
        print(
            f"Saved folded 2min length per segment: "
            f"{int(saved_folded_lengths_2min_all[0])}"
        )

    if valid_fractions_2min_all:
        print(f"Mean valid fraction 2min:   {float(np.mean(valid_fractions_2min_all)):.4f}")
        print(f"Mean valid fraction 8min:   {float(np.mean(valid_fractions_8min_all)):.4f}")
        print(f"Mean valid fraction 32min:  {float(np.mean(valid_fractions_32min_all)):.4f}")
        print(f"Mean valid fraction 128min: {float(np.mean(valid_fractions_128min_all)):.4f}")

    print("Label counts, segment-level:")

    for label, count in sorted(label_counts.items(), key=lambda x: x[0]):
        print(f"  {label}: {count}")

    print("=" * 60)


if __name__ == "__main__":
    preprocess_all()
