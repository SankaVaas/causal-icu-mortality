"""
data/dataset.py
===============
PyTorch Dataset and DataLoader for the Causal Temporal GNN.

Each sample is a dict produced by preprocess.py:
    X       float32 [T, F]   — normalised feature values (LOCF-imputed)
    mask    float32 [T, F]   — 1 = originally observed, 0 = imputed
    delta   float32 [T, F]   — hours since last real observation
    times   float32 [T]      — hours since ICU admission
    y       int               — in-hospital mortality label
    meta    dict              — stay_id, subject_id, age, gender, los_hours

The collate function pads all samples in a batch to the same sequence length
(the longest in the batch) and returns:
    X       float32 [B, T_max, F]
    mask    float32 [B, T_max, F]
    delta   float32 [B, T_max, F]
    times   float32 [B, T_max]
    lengths int64   [B]           — actual T per sample (for packing)
    y       int64   [B]
    meta    list[dict]

Usage
-----
    from data.dataset import ICUDataset, build_dataloaders

    train_loader, val_loader, test_loader, stats = build_dataloaders(
        processed_dir="data/processed/",
        batch_size=32,
    )
    for batch in train_loader:
        X, mask, delta, times, lengths, y, meta = batch
        ...
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ICUDataset(Dataset):
    """
    Wraps a list of preprocessed patient dicts into a PyTorch Dataset.

    Parameters
    ----------
    samples     : list of dicts (output of preprocess.py)
    max_seq_len : if set, truncate / pad each sample to this length
    augment     : if True, randomly drop 10% of observations during training
                  (acts as regularisation, forces model to handle missingness)
    """

    def __init__(
        self,
        samples: list,
        max_seq_len: Optional[int] = None,
        augment: bool = False,
    ):
        self.samples     = samples
        self.max_seq_len = max_seq_len
        self.augment     = augment

        # Infer feature count from first sample
        self.n_features  = samples[0]["X"].shape[1] if samples else 17
        self.n_timesteps = samples[0]["X"].shape[0] if samples else 48

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        X     = s["X"].copy()     # [T, F]
        mask  = s["mask"].copy()  # [T, F]
        delta = s["delta"].copy() # [T, F]
        times = s["times"].copy() # [T]
        y     = int(s["y"])

        T = X.shape[0]

        # ── Optional truncation ───────────────────────────────────────────
        if self.max_seq_len is not None and T > self.max_seq_len:
            X     = X[:self.max_seq_len]
            mask  = mask[:self.max_seq_len]
            delta = delta[:self.max_seq_len]
            times = times[:self.max_seq_len]
            T     = self.max_seq_len

        # ── Optional augmentation (random observation dropout) ────────────
        if self.augment:
            drop = np.random.rand(*mask.shape) < 0.10   # 10% drop rate
            mask  = mask * (1 - drop.astype(np.float32))
            # Where we dropped, revert X to 0 (like true missing)
            X     = X * mask

        return {
            "X":      torch.tensor(X,     dtype=torch.float32),
            "mask":   torch.tensor(mask,  dtype=torch.float32),
            "delta":  torch.tensor(delta, dtype=torch.float32),
            "times":  torch.tensor(times, dtype=torch.float32),
            "length": torch.tensor(T,     dtype=torch.long),
            "y":      torch.tensor(y,     dtype=torch.long),
            "meta":   s["meta"],
        }


def collate_fn(batch: list) -> tuple:
    """
    Pad a list of samples to the max sequence length in the batch.

    Shorter sequences are right-padded with zeros (mask = 0 marks as missing,
    so the model ignores padded positions naturally).
    """
    T_max = max(b["X"].shape[0] for b in batch)
    B     = len(batch)
    F     = batch[0]["X"].shape[1]

    X_pad     = torch.zeros(B, T_max, F,    dtype=torch.float32)
    mask_pad  = torch.zeros(B, T_max, F,    dtype=torch.float32)
    delta_pad = torch.zeros(B, T_max, F,    dtype=torch.float32)
    times_pad = torch.zeros(B, T_max,       dtype=torch.float32)
    lengths   = torch.zeros(B,              dtype=torch.long)
    ys        = torch.zeros(B,              dtype=torch.long)
    metas     = []

    for i, b in enumerate(batch):
        T = b["X"].shape[0]
        X_pad[i, :T]     = b["X"]
        mask_pad[i, :T]  = b["mask"]
        delta_pad[i, :T] = b["delta"]
        times_pad[i, :T] = b["times"]
        lengths[i]        = b["length"]
        ys[i]             = b["y"]
        metas.append(b["meta"])

    return X_pad, mask_pad, delta_pad, times_pad, lengths, ys, metas


def build_dataloaders(
    processed_dir: str = "data/processed/",
    batch_size:    int  = 32,
    num_workers:   int  = 0,       # 0 = main process (safe on Windows/CPU)
    max_seq_len:   Optional[int] = None,
    augment_train: bool = True,
    pin_memory:    bool = False,   # set True if GPU available
) -> tuple:
    """
    Load preprocessed splits and build DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, stats_dict
    """
    pdir = Path(processed_dir)

    def load_pkl(name: str) -> list:
        path = pdir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run data/preprocess.py first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    train_samples = load_pkl("train.pkl")
    val_samples   = load_pkl("val.pkl")
    test_samples  = load_pkl("test.pkl")

    stats_path = pdir / "stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    train_ds = ICUDataset(train_samples, max_seq_len=max_seq_len, augment=augment_train)
    val_ds   = ICUDataset(val_samples,   max_seq_len=max_seq_len, augment=False)
    test_ds  = ICUDataset(test_samples,  max_seq_len=max_seq_len, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"DataLoaders ready:")
    print(f"  train: {len(train_ds):>4} samples  "
          f"({len(train_loader)} batches @ bs={batch_size})")
    print(f"  val:   {len(val_ds):>4} samples  "
          f"({len(val_loader)} batches)")
    print(f"  test:  {len(test_ds):>4} samples  "
          f"({len(test_loader)} batches)")

    mort_train = np.mean([s["y"] for s in train_samples])
    mort_val   = np.mean([s["y"] for s in val_samples])
    mort_test  = np.mean([s["y"] for s in test_samples])
    print(f"  Mortality — train: {mort_train*100:.1f}%  "
          f"val: {mort_val*100:.1f}%  test: {mort_test*100:.1f}%")
    print(f"  Tensor shape per sample: X=[{train_ds.n_timesteps}, {train_ds.n_features}]")

    return train_loader, val_loader, test_loader, stats


# ── Convenience helpers ────────────────────────────────────────────────────

def get_feature_names(processed_dir: str = "data/processed/") -> list:
    """Return the ordered list of feature names."""
    stats_path = Path(processed_dir) / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)["features"]
    # Fallback to hardcoded list
    return [
        "heart_rate", "map", "resp_rate", "spo2", "temp_c",
        "gcs_eye", "gcs_verbal", "gcs_motor",
        "creatinine", "bun", "wbc", "hemoglobin",
        "lactate", "ph", "pao2", "platelets", "glucose",
    ]


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for focal loss.
    Returns tensor([w_negative, w_positive]).
    """
    labels = []
    for *_, ys, _meta in train_loader:
        labels.extend(ys.tolist())
    labels = np.array(labels)
    n_pos  = labels.sum()
    n_neg  = len(labels) - n_pos
    w_pos  = len(labels) / (2.0 * n_pos)  if n_pos > 0 else 1.0
    w_neg  = len(labels) / (2.0 * n_neg)  if n_neg > 0 else 1.0
    print(f"  Class weights → negative: {w_neg:.2f}  positive: {w_pos:.2f}")
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


# ── Quick smoke-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    processed_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/"

    print(f"\nSmoke-testing DataLoader from: {processed_dir}\n")
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        processed_dir=processed_dir,
        batch_size=8,
        augment_train=False,
    )

    print("\nInspecting one training batch:")
    X, mask, delta, times, lengths, y, meta = next(iter(train_loader))
    print(f"  X      : {tuple(X.shape)}   dtype={X.dtype}")
    print(f"  mask   : {tuple(mask.shape)}")
    print(f"  delta  : {tuple(delta.shape)}   max_delta={delta.max():.1f}h")
    print(f"  times  : {tuple(times.shape)}   range [{times.min():.0f}, {times.max():.0f}]h")
    print(f"  lengths: {lengths.tolist()}")
    print(f"  y      : {y.tolist()}")
    print(f"  meta[0]: {meta[0]}")

    features = get_feature_names(processed_dir)
    print(f"\nFeatures ({len(features)}): {features}")

    obs_rate = mask.mean().item()
    print(f"\nMean observation rate in batch: {obs_rate*100:.1f}%")
    print(f"  (remaining {(1-obs_rate)*100:.1f}% are LOCF-imputed — "
          f"mask tells the model which are real)")

    cw = get_class_weights(train_loader)
    print(f"\n✓ All checks passed. Ready to train.")