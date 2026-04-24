import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_data import (
    scan_raw_data,
    CSVRecord,
    DataManifest,
    RAW_DATA_DIR,
    PROJECT_ROOT,
)
from data.dataset import KeystrokeDynamicsDataset

# local paths / constants
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOW_SIZE = 50     # N = 50 keystrokes per window
WINDOW_STRIDE = 25   # 50 % overlap for more training data
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


# --- feature extraction ---

def extract_features_from_csv(csv_path: Path) -> Optional[np.ndarray]:
    """
    Read a single CSV and compute the 4 temporal features.

    Columns expected: VK, HT, FT
    Derived features (per keystroke i):
        HT_i  = Hold Time       (directly from CSV)
        FT_i  = Flight Time     (directly from CSV)
        PP_i  = Press-to-Press  = HT_{i-1} + FT_i
        RR_i  = Release-to-Release = FT_i + HT_i

    The first keystroke has FT = -1 (sentinel for "no previous key").
    We drop the first row because PP requires HT_{i-1}.

    Returns
    -------
    np.ndarray of shape (num_keystrokes, 4) — [HT, FT, PP, RR]
    or None if the file is malformed / too short.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"WARNING: Failed to read {csv_path.name}: {e}")
        return None

    if not {"VK", "HT", "FT"}.issubset(df.columns):
        print(f"WARNING: Missing expected columns in {csv_path.name}")
        return None

    ht = df["HT"].values.astype(np.float32)
    ft = df["FT"].values.astype(np.float32)

    if len(ht) < 2:
        return None

    # Compute PP and RR starting from index 1
    # PP_i = HT_{i-1} + FT_i
    pp = ht[:-1] + ft[1:]
    # RR_i = FT_i + HT_i
    rr = ft[1:] + ht[1:]

    # Trim HT, FT to match PP/RR length (drop first keystroke)
    ht_trimmed = ht[1:]
    ft_trimmed = ft[1:]

    features = np.stack([ht_trimmed, ft_trimmed, pp, rr], axis=1)  # (T-1, 4)

    # Replace negative / NaN values with 0 (defensive)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, 0.0, None)

    return features


# --- windowing ---

def segment_into_windows(
    features: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> np.ndarray:
    """
    Segment a keystroke feature array into overlapping windows.

    Parameters
    ----------
    features : (T, 4)
    window_size : int
    stride : int

    Returns
    -------
    np.ndarray of shape (num_windows, 4, window_size)
        Channel-first layout for Conv1d compatibility.
    """
    T = features.shape[0]
    if T < window_size:
        return np.empty((0, 4, window_size), dtype=np.float32)

    windows = []
    for start in range(0, T - window_size + 1, stride):
        win = features[start : start + window_size]   # (W, 4)
        windows.append(win.T)                          # (4, W)

    return np.array(windows, dtype=np.float32)  # (N, 4, W)


# --- train/val/test splits ---

def _process_records(
    records: List[CSVRecord], desc: str = ""
) -> Dict[str, np.ndarray]:
    """
    Process a list of CSVRecords into per-subject window arrays.

    Returns {subject_uid: (N_windows, 4, 50)}.
    """
    subject_windows: Dict[str, List[np.ndarray]] = {}

    for rec in tqdm(records, desc=desc, unit="file"):
        feats = extract_features_from_csv(rec.filepath)
        if feats is None:
            continue
        wins = segment_into_windows(feats)
        if wins.shape[0] == 0:
            continue
        uid = rec.subject_uid
        subject_windows.setdefault(uid, []).append(wins)

    # Concatenate per subject
    result = {}
    for uid, win_list in subject_windows.items():
        result[uid] = np.concatenate(win_list, axis=0)

    return result


def build_splits(manifest: DataManifest) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Build train / val / test tensors with strict subject-disjoint splitting.

    Train: genuine human ONLY (70 % of subjects)
    Val:   genuine human  (15 %) + ALL synthesized bot samples for those subjects
    Test:  genuine human  (15 %) + ALL synthesized bot samples for those subjects

    Returns
    -------
    train_data, train_labels,
    val_data,   val_labels,
    test_data,  test_labels
        data shapes  = (N, 4, 50)   float32
        label shapes = (N,)         int64   (1 = human, 0 = bot)
    """
    print("INFO: Processing human records …")
    human_by_subject = _process_records(manifest.human_records, desc="Human CSVs")

    all_subjects = sorted(human_by_subject.keys())
    print(f"INFO: Total unique subjects with ≥1 window: {len(all_subjects)}")

    # deterministic subject split
    # Hash subjects to get reproducible ordering independent of filesystem
    def _hash_key(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()

    all_subjects_sorted = sorted(all_subjects, key=_hash_key)

    train_subjects, temp_subjects = train_test_split(
        all_subjects_sorted,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
    )

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    print(
        f"INFO: Subject split → Train: {len(train_set)}  |  "
        f"Val: {len(val_set)}  |  Test: {len(test_set)}"
    )

    # combine human windows
    train_human = np.concatenate(
        [human_by_subject[s] for s in sorted(train_set)], axis=0
    )
    val_human = np.concatenate(
        [human_by_subject[s] for s in sorted(val_set)], axis=0
    )
    test_human = np.concatenate(
        [human_by_subject[s] for s in sorted(test_set)], axis=0
    )

    print(
        f"INFO: Human windows → Train: {train_human.shape[0]:,}  |  "
        f"Val: {val_human.shape[0]:,}  |  Test: {test_human.shape[0]:,}"
    )

    # bot data only goes in val and test
    print("INFO: Processing synthesized (bot) records …")
    synth_records_val = [
        r for r in manifest.synth_records if r.subject_uid in val_set
    ]
    synth_records_test = [
        r for r in manifest.synth_records if r.subject_uid in test_set
    ]

    synth_by_subject_val = _process_records(synth_records_val, desc="Synth Val")
    synth_by_subject_test = _process_records(synth_records_test, desc="Synth Test")

    val_synth_windows = (
        np.concatenate(list(synth_by_subject_val.values()), axis=0)
        if synth_by_subject_val
        else np.empty((0, 4, WINDOW_SIZE), dtype=np.float32)
    )
    test_synth_windows = (
        np.concatenate(list(synth_by_subject_test.values()), axis=0)
        if synth_by_subject_test
        else np.empty((0, 4, WINDOW_SIZE), dtype=np.float32)
    )

    print(
        f"INFO: Bot windows   → Val: {val_synth_windows.shape[0]:,}  |  "
        f"Test: {test_synth_windows.shape[0]:,}"
    )

    # merge it all together
    # Train: human only  (label=1)
    train_data = train_human
    train_labels = np.ones(train_human.shape[0], dtype=np.int64)

    # Val:  human (1) + bots (0)
    val_data = np.concatenate([val_human, val_synth_windows], axis=0)
    val_labels = np.concatenate([
        np.ones(val_human.shape[0], dtype=np.int64),
        np.zeros(val_synth_windows.shape[0], dtype=np.int64),
    ])

    # Test: human (1) + bots (0)
    test_data = np.concatenate([test_human, test_synth_windows], axis=0)
    test_labels = np.concatenate([
        np.ones(test_human.shape[0], dtype=np.int64),
        np.zeros(test_synth_windows.shape[0], dtype=np.int64),
    ])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


# --- normalization & saving ---

def normalise_and_save(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Apply Z-score normalisation (fit on train) and save .pt tensors.

    Normalisation is per-channel: each of the 4 feature channels is
    independently standardised using μ and σ computed on the training set.

    Saves:
        train.pt  → {"data": Tensor, "labels": Tensor, "mean": Tensor, "std": Tensor}
        val.pt    → {"data": Tensor, "labels": Tensor}
        test.pt   → {"data": Tensor, "labels": Tensor}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-channel mean / std from training data
    # train_data shape: (N, 4, 50)
    mean = train_data.mean(axis=(0, 2), keepdims=True)  # (1, 4, 1)
    std = train_data.std(axis=(0, 2), keepdims=True) + 1e-8

    print(f"INFO: Channel means: {mean.squeeze()}")
    print(f"INFO: Channel stds:  {std.squeeze()}")

    def _normalise(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((arr - mean) / std).astype(np.float32))

    train_tensor = _normalise(train_data)
    val_tensor = _normalise(val_data)
    test_tensor = _normalise(test_data)

    train_labels_t = torch.from_numpy(train_labels)
    val_labels_t = torch.from_numpy(val_labels)
    test_labels_t = torch.from_numpy(test_labels)

    # Save
    torch.save(
        {
            "data": train_tensor,
            "labels": train_labels_t,
            "mean": torch.from_numpy(mean.squeeze().astype(np.float32)),
            "std": torch.from_numpy(std.squeeze().astype(np.float32)),
        },
        output_dir / "train.pt",
    )
    torch.save(
        {"data": val_tensor, "labels": val_labels_t},
        output_dir / "val.pt",
    )
    torch.save(
        {"data": test_tensor, "labels": test_labels_t},
        output_dir / "test.pt",
    )

    print(f"INFO: Saved train.pt  → data {train_tensor.shape}, labels {train_labels_t.shape}")
    print(f"INFO: Saved val.pt    → data {val_tensor.shape},  labels {val_labels_t.shape}")
    print(f"INFO: Saved test.pt   → data {test_tensor.shape},  labels {test_labels_t.shape}")
    print(f"INFO: Binary tensors written to {output_dir}")


# --- standalone run ---

def main():
    print("=" * 60)
    print("INFO: Phase 2: Preprocessing & Binary Compilation")
    print("=" * 60)

    # Phase 1 — scan & validate
    manifest = scan_raw_data()

    # Phase 2 — extract, window, split, save
    (
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
    ) = build_splits(manifest)

    normalise_and_save(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
    )

    # sanity check the dataset class
    print("-" * 60)
    print("INFO: Smoke-testing KeystrokeDynamicsDataset …")
    train_ds = KeystrokeDynamicsDataset(PROCESSED_DIR / "train.pt")
    x_sample, y_sample = train_ds[0]
    print(f"INFO:   train_ds[0]  →  x.shape={x_sample.shape}, label={y_sample}")

    val_ds = KeystrokeDynamicsDataset(PROCESSED_DIR / "val.pt")
    x_sample, y_sample = val_ds[0]
    print(f"INFO:   val_ds[0]    →  x.shape={x_sample.shape}, label={y_sample}")

    print("=" * 60)
    print("INFO: Phase 2 COMPLETE — binary tensors compiled successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
