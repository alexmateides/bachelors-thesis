# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import wfdb


# reproducibility
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# dataset constants
MITDB_DIR = "./mitdb"
FS = 360
BEAT_WINDOW = 250
BEAT_BEFORE = 125

# cuda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# map annotations to classes
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}
CLASS_NAMES = ['N (Normal)', 'S (Supra)', 'V (Ventri)', 'F (Fusion)', 'Q (Unknown)']
N_CLASSES = len(CLASS_NAMES)


def load_record_beats(record_path: str):
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    signal = rec.p_signal[:, 0].astype(np.float32)
    signal_len = len(signal)

    segments, labels = [], []
    for r_idx, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI_MAP:
            continue
        start = r_idx - BEAT_BEFORE
        end = start + BEAT_WINDOW
        if start < 0 or end > signal_len:
            continue

        seg = signal[start:end]
        mu, sigma = seg.mean(), seg.std() + 1e-8 # prevent zero division
        seg = (seg - mu) / sigma

        segments.append(seg)
        labels.append(AAMI_MAP[sym])

    return np.array(segments, dtype=np.float32), np.array(labels, dtype=np.int64)


# load data
record_ids = sorted(set(
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(MITDB_DIR, "*.hea"))
))
print(f"Found {len(record_ids)} records")

all_segs, all_labels = [], []
for rid in record_ids:
    path = os.path.join(MITDB_DIR, rid)
    segs, labs = load_record_beats(path)
    all_segs.append(segs)
    all_labels.append(labs)

X = np.concatenate(all_segs, axis=0)
y = np.concatenate(all_labels, axis=0)

print(f"Total beats: {len(X)}")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt}")


class BeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # (L,)
        if self.augment:
            x = x + torch.randn_like(x) * 0.05
            x = x * (0.9 + 0.2 * torch.rand(1).item())
        return {
            "input_values": x,  # (L,)
            "labels": self.y[idx]
        }


X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.3,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val,
    test_size=0.3,
    random_state=SEED,
    stratify=y_val
)

# train / val splits
train_ds = BeatDataset(X_tr, y_tr, augment=True)
val_ds = BeatDataset(X_val, y_val, augment=False)

cls_counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
cls_weights = 1.0 / (cls_counts + 1e-6)
cls_weights = cls_weights / cls_weights.sum() * N_CLASSES
cls_weights = torch.tensor(cls_weights, dtype=torch.float32)

# test split
test_ds = BeatDataset(X_test, y_test, augment=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print("Class weights:", cls_weights.tolist())
