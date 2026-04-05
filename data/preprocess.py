"""
data/preprocess.py
==================
Turn raw MIMIC-IV .csv.gz files into per-patient time-series tensors
ready for the Causal TGAT model.

For each ICU stay we produce:
    X       float32 [T, F]   — F=17 features, T hourly timesteps (≤48h)
    mask    float32 [T, F]   — 1 = observed, 0 = imputed  (GRU-D style)
    delta   float32 [T, F]   — hours since last real observation per feature
    times   float32 [T]      — hours since ICU admission (Time2Vec input)
    y       int               — in-hospital mortality (0/1)
    meta    dict              — stay_id, subject_id, los_hours, gender, age

Output
------
data/processed/
    train.pkl   list[dict]  (~70% of cohort)
    val.pkl     list[dict]  (~15%)
    test.pkl    list[dict]  (~15%)
    stats.json  feature mean/std computed on train set (for normalisation)

Usage
-----
    python data/preprocess.py [--data-dir ...] [--cohort data/cohort.csv]
                              [--window-hours 48] [--bin-size 1]
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Feature catalogue (must match extract_mimic.py) ──────────────────────
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220181: "map",
    220052: "map_arterial",
    220210: "resp_rate",
    224690: "resp_rate_total",
    220277: "spo2",
    223762: "temp_c",
    223761: "temp_f",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

LAB_ITEMIDS = {
    50912: "creatinine",
    51006: "bun",
    51301: "wbc",
    51222: "hemoglobin",
    50813: "lactate",
    50820: "ph",
    50821: "pao2",
    50816: "fio2_lab",
    51265: "platelets",
    50931: "glucose",
}

# Merge MAP (invasive + non-invasive → single "map" column)
MAP_ALIASES = {"map_arterial": "map"}

# Final ordered feature list — 17 features
FEATURES = [
    "heart_rate", "map", "resp_rate", "spo2", "temp_c",
    "gcs_eye", "gcs_verbal", "gcs_motor",
    "creatinine", "bun", "wbc", "hemoglobin",
    "lactate", "ph", "pao2", "platelets", "glucose",
]
N_FEATURES = len(FEATURES)

# Physiological plausibility ranges — values outside are set to NaN
PLAUSIBILITY = {
    "heart_rate":   (0,   300),
    "map":          (0,   300),
    "resp_rate":    (0,    70),
    "spo2":         (50,  100),
    "temp_c":       (25,   45),
    "gcs_eye":      (1,     4),
    "gcs_verbal":   (1,     5),
    "gcs_motor":    (1,     6),
    "creatinine":   (0,    50),
    "bun":          (0,   200),
    "wbc":          (0,   500),
    "hemoglobin":   (0,    25),
    "lactate":      (0,    40),
    "ph":           (6.5,   8),
    "pao2":         (0,   700),
    "platelets":    (0, 2_000),
    "glucose":      (0,  2000),
}


def read_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)


def fahrenheit_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """Convert temp_f → temp_c and drop temp_f column."""
    if "temp_f" in df.columns and "temp_c" in df.columns:
        mask = df["temp_c"].isna() & df["temp_f"].notna()
        df.loc[mask, "temp_c"] = (df.loc[mask, "temp_f"] - 32) * 5 / 9
    if "temp_f" in df.columns:
        df.drop(columns=["temp_f"], inplace=True)
    return df


def merge_map(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer invasive MAP, fallback to non-invasive."""
    if "map_arterial" in df.columns and "map" in df.columns:
        df["map"] = df["map_arterial"].combine_first(df["map"])
        df.drop(columns=["map_arterial"], inplace=True)
    elif "map_arterial" in df.columns:
        df.rename(columns={"map_arterial": "map"}, inplace=True)
    return df


def merge_resp_rate(df: pd.DataFrame) -> pd.DataFrame:
    if "resp_rate_total" in df.columns and "resp_rate" in df.columns:
        df["resp_rate"] = df["resp_rate"].combine_first(df["resp_rate_total"])
        df.drop(columns=["resp_rate_total"], inplace=True)
    elif "resp_rate_total" in df.columns:
        df.rename(columns={"resp_rate_total": "resp_rate"}, inplace=True)
    return df


def apply_plausibility(df: pd.DataFrame) -> pd.DataFrame:
    for feat, (lo, hi) in PLAUSIBILITY.items():
        if feat in df.columns:
            df.loc[~df[feat].between(lo, hi, inclusive="both"), feat] = np.nan
    return df


def load_vitals(data_dir: Path, stay_ids: set) -> pd.DataFrame:
    """
    Load chartevents for target stays, pivot to wide format.
    Returns DataFrame with columns: stay_id, charttime, <vital_names>
    """
    ce = read_gz(data_dir / "icu" / "chartevents.csv.gz")
    ce = ce[ce["stay_id"].isin(stay_ids) & ce["itemid"].isin(VITAL_ITEMIDS)]
    ce = ce[["stay_id", "charttime", "itemid", "valuenum"]].dropna(subset=["valuenum"])
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["feature"] = ce["itemid"].map(VITAL_ITEMIDS)
    ce.drop(columns=["itemid"], inplace=True)

    # Aggregate: if multiple readings at same charttime, take median
    ce = (
        ce.groupby(["stay_id", "charttime", "feature"])["valuenum"]
        .median()
        .reset_index()
    )
    vitals = ce.pivot_table(
        index=["stay_id", "charttime"], columns="feature",
        values="valuenum", aggfunc="first"
    ).reset_index()
    vitals.columns.name = None
    return vitals


def load_labs(data_dir: Path, hadm_map: dict) -> pd.DataFrame:
    """
    Load labevents for target admissions, pivot to wide format.
    Returns DataFrame with columns: stay_id, charttime, <lab_names>
    """
    le = read_gz(data_dir / "hosp" / "labevents.csv.gz")
    hadm_ids = set(hadm_map.keys())
    le = le[le["hadm_id"].isin(hadm_ids) & le["itemid"].isin(LAB_ITEMIDS)]
    le = le[["hadm_id", "charttime", "itemid", "valuenum"]].dropna(subset=["valuenum"])
    le["charttime"] = pd.to_datetime(le["charttime"])
    le["stay_id"] = le["hadm_id"].map(hadm_map)
    le["feature"] = le["itemid"].map(LAB_ITEMIDS)
    le.drop(columns=["itemid", "hadm_id"], inplace=True)

    le = (
        le.groupby(["stay_id", "charttime", "feature"])["valuenum"]
        .median()
        .reset_index()
    )
    labs = le.pivot_table(
        index=["stay_id", "charttime"], columns="feature",
        values="valuenum", aggfunc="first"
    ).reset_index()
    labs.columns.name = None
    return labs


def build_patient_tensor(
    stay_row: pd.Series,
    events: pd.DataFrame,
    window_hours: int,
    bin_size: float,
) -> dict:
    """
    Build a single patient's tensor from the merged event DataFrame.

    Parameters
    ----------
    stay_row   : row from cohort DataFrame
    events     : all events for this stay_id (stay_id, charttime, <features>)
    window_hours : max hours to include (48)
    bin_size     : hourly bin size in hours (1.0)

    Returns
    -------
    dict with keys: X, mask, delta, times, y, meta
    """
    stay_id    = stay_row["stay_id"]
    intime     = pd.to_datetime(stay_row["intime"])
    los_hours  = float(stay_row["los_hours"])
    y          = int(stay_row["mortality_hosp"])

    stay_events = events[events["stay_id"] == stay_id].copy()
    stay_events["hours"] = (
        (stay_events["charttime"] - intime).dt.total_seconds() / 3600
    )
    # Keep only events within the window and after admission
    stay_events = stay_events[
        (stay_events["hours"] >= 0) &
        (stay_events["hours"] <= window_hours)
    ]

    # Create hourly bins
    n_bins = int(window_hours / bin_size)
    X     = np.full((n_bins, N_FEATURES), np.nan, dtype=np.float32)
    times = np.arange(n_bins, dtype=np.float32) * bin_size  # hours since admit

    if len(stay_events) > 0:
        stay_events["bin"] = np.floor(
            stay_events["hours"] / bin_size
        ).astype(int).clip(0, n_bins - 1)

        for feat_idx, feat in enumerate(FEATURES):
            if feat not in stay_events.columns:
                continue
            feat_obs = stay_events[["bin", feat]].dropna(subset=[feat])
            for _, row in feat_obs.iterrows():
                b = int(row["bin"])
                X[b, feat_idx] = row[feat]

    # ── Observation mask ─────────────────────────────────────────────────
    # mask[t, f] = 1 if observed at bin t, 0 otherwise
    mask = (~np.isnan(X)).astype(np.float32)

    # ── Time-since-last-observation (delta) ──────────────────────────────
    # GRU-D: δ[t, f] = gap in hours since last real observation
    # If never observed before t, use t itself (time since start)
    delta = np.zeros_like(X, dtype=np.float32)
    for f in range(N_FEATURES):
        last_obs_time = 0.0
        for t in range(n_bins):
            if mask[t, f] == 1:
                delta[t, f] = times[t] - last_obs_time
                last_obs_time = times[t]
            else:
                delta[t, f] = times[t] - last_obs_time

    # ── LOCF imputation ──────────────────────────────────────────────────
    # Forward-fill observed values; remaining NaNs → 0 (handled by mask)
    for f in range(N_FEATURES):
        last_val = np.nan
        for t in range(n_bins):
            if mask[t, f] == 1:
                last_val = X[t, f]
            elif not np.isnan(last_val):
                X[t, f] = last_val

    # Replace remaining NaN (never observed) with 0 — mask encodes missingness
    np.nan_to_num(X, copy=False, nan=0.0)

    return {
        "stay_id":   stay_id,
        "X":         X,       # [T, F] float32
        "mask":      mask,    # [T, F] float32
        "delta":     delta,   # [T, F] float32  hours since last obs
        "times":     times,   # [T]    float32  hours since admission
        "y":         y,
        "meta": {
            "stay_id":    stay_id,
            "subject_id": stay_row["subject_id"],
            "los_hours":  los_hours,
            "gender":     stay_row["gender"],
            "age":        stay_row["anchor_age"],
        },
    }


def compute_train_stats(samples: list) -> dict:
    """
    Compute per-feature mean and std on observed values in the training set.
    Used for z-score normalisation at model input time.
    """
    all_vals = [[] for _ in range(N_FEATURES)]
    for s in samples:
        X, mask = s["X"], s["mask"]
        for f in range(N_FEATURES):
            obs = X[:, f][mask[:, f] == 1]
            all_vals[f].extend(obs.tolist())

    stats = {}
    for f, feat in enumerate(FEATURES):
        vals = np.array(all_vals[f], dtype=np.float64)
        if len(vals) > 1:
            stats[feat] = {"mean": float(vals.mean()), "std": float(vals.std())}
        else:
            stats[feat] = {"mean": 0.0, "std": 1.0}
    return stats


def normalise_samples(samples: list, stats: dict) -> list:
    """Z-score normalise X using train-set statistics. Mask unchanged."""
    normed = []
    for s in samples:
        s = dict(s)  # shallow copy
        X = s["X"].copy()
        for f, feat in enumerate(FEATURES):
            mean = stats[feat]["mean"]
            std  = stats[feat]["std"] if stats[feat]["std"] > 1e-8 else 1.0
            X[:, f] = (X[:, f] - mean) / std
        s["X"] = X
        normed.append(s)
    return normed


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC-IV cohort")
    parser.add_argument("--data-dir",  type=str, default=None)
    parser.add_argument("--cohort",    type=str, default=None,
                        help="Path to cohort.csv (output of extract_mimic.py)")
    parser.add_argument("--out-dir",   type=str, default=None,
                        help="Output directory (default: data/processed/)")
    parser.add_argument("--window-hours", type=int, default=48,
                        help="Hours of ICU stay to use (default 48)")
    parser.add_argument("--bin-size",  type=float, default=1.0,
                        help="Hourly bin size in hours (default 1.0)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # ── Resolve paths ─────────────────────────────────────────────────────
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        candidates = list(script_dir.glob("mimic-iv*"))
        if not candidates:
            raise FileNotFoundError(
                "Cannot find mimic-iv demo folder. Use --data-dir."
            )
        data_dir = candidates[0]

    cohort_path = Path(args.cohort) if args.cohort else script_dir / "cohort.csv"
    out_dir     = Path(args.out_dir) if args.out_dir else script_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMIMIC-IV Preprocessor")
    print(f"  Data dir    : {data_dir}")
    print(f"  Cohort      : {cohort_path}")
    print(f"  Output      : {out_dir}")
    print(f"  Window      : {args.window_hours}h  |  Bin size: {args.bin_size}h")
    print(f"  Features    : {N_FEATURES} → {FEATURES}\n")

    # ── Load cohort ───────────────────────────────────────────────────────
    cohort = pd.read_csv(cohort_path)
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    print(f"  Cohort loaded: {len(cohort):,} stays  |  "
          f"mortality {cohort['mortality_hosp'].mean()*100:.1f}%")

    stay_ids = set(cohort["stay_id"].tolist())
    hadm_map = dict(zip(cohort["hadm_id"], cohort["stay_id"]))

    # ── Load raw events ───────────────────────────────────────────────────
    print("\n  Loading vitals (chartevents)...")
    vitals = load_vitals(data_dir, stay_ids)
    vitals = fahrenheit_to_celsius(vitals)
    vitals = merge_map(vitals)
    vitals = merge_resp_rate(vitals)
    vitals = apply_plausibility(vitals)

    print("  Loading labs (labevents)...")
    labs = load_labs(data_dir, hadm_map)
    labs = apply_plausibility(labs)

    # ── Merge vitals + labs ──────────────────────────────────────────────
    print("  Merging vitals and labs...")
    events = pd.merge(
        vitals, labs,
        on=["stay_id", "charttime"], how="outer"
    ).sort_values(["stay_id", "charttime"])

    # Ensure all feature columns exist (may be absent in demo)
    for feat in FEATURES:
        if feat not in events.columns:
            events[feat] = np.nan

    # ── Build per-patient tensors ─────────────────────────────────────────
    print(f"\n  Building tensors for {len(cohort):,} patients...")
    samples = []
    skipped = 0
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), ncols=80):
        try:
            s = build_patient_tensor(row, events, args.window_hours, args.bin_size)
            samples.append(s)
        except Exception as e:
            skipped += 1
            print(f"    Warning: stay {row['stay_id']} skipped — {e}")

    print(f"  → {len(samples):,} tensors built  |  {skipped} skipped")

    # ── Train / val / test split ──────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(samples))
    n   = len(samples)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train = [samples[i] for i in train_idx]
    val   = [samples[i] for i in val_idx]
    test  = [samples[i] for i in test_idx]

    print(f"\n  Split → train: {len(train)} | val: {len(val)} | test: {len(test)}")

    # ── Compute normalisation stats on train only ─────────────────────────
    print("  Computing normalisation stats on train set...")
    stats = compute_train_stats(train)

    # ── Normalise all splits ──────────────────────────────────────────────
    train = normalise_samples(train, stats)
    val   = normalise_samples(val,   stats)
    test  = normalise_samples(test,  stats)

    # ── Save ──────────────────────────────────────────────────────────────
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = out_dir / f"{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(split_data, f)
        mort = np.mean([s["y"] for s in split_data])
        print(f"  ✓ {split_name}.pkl  — {len(split_data)} stays  "
              f"  mortality {mort*100:.1f}%")

    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({"features": FEATURES, "feature_stats": stats}, f, indent=2)
    print(f"  ✓ stats.json saved\n")

    print("Feature statistics (train set):")
    print(f"  {'Feature':<22}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*40}")
    for feat in FEATURES:
        m = stats[feat]["mean"]
        s = stats[feat]["std"]
        print(f"  {feat:<22}  {m:>8.2f}  {s:>8.2f}")

    print("\nNext step:  python data/dataset.py  (to verify DataLoader)")


if __name__ == "__main__":
    main()