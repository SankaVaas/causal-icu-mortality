"""
data/extract_mimic.py
=====================
Extract a clean ICU cohort from MIMIC-IV Clinical Database Demo v2.2.

Reads directly from the .csv.gz files in your data folder — no SQL, no
PostgreSQL, no unzipping needed.

Output
------
data/cohort.csv   — one row per ICU stay with:
    stay_id, subject_id, hadm_id, gender, anchor_age,
    intime, outtime, los_hours, mortality_hosp (0/1)

Usage
-----
    python data/extract_mimic.py [--data-dir DATA_DIR]

    DATA_DIR defaults to the folder containing this script
    (i.e. data/mimic-iv-clinical-database-demo-2.2/)
"""

import argparse
import os
import gzip
import pandas as pd
from pathlib import Path


# ── itemid reference (MetaVision / MIMIC-IV) ──────────────────────────────
# chartevents vitals
VITAL_ITEMIDS = {
    220045: "heart_rate",           # Heart Rate
    220179: "sbp",                  # Non Invasive Blood Pressure systolic
    220180: "dbp",                  # Non Invasive Blood Pressure diastolic
    220181: "map",                  # Non Invasive Blood Pressure mean
    220052: "map_arterial",         # Arterial Blood Pressure mean (invasive)
    220050: "sbp_arterial",         # Arterial Blood Pressure systolic
    220051: "dbp_arterial",         # Arterial Blood Pressure diastolic
    220210: "resp_rate",            # Respiratory Rate
    224690: "resp_rate_total",      # Respiratory Rate (Total)
    220277: "spo2",                 # O2 saturation pulseoxymetry
    223762: "temp_c",               # Temperature Celsius
    223761: "temp_f",               # Temperature Fahrenheit
    220739: "gcs_eye",              # GCS - Eye Opening
    223900: "gcs_verbal",           # GCS - Verbal Response
    223901: "gcs_motor",            # GCS - Motor Response
    226512: "weight_kg",            # Admission Weight (Kg)
}

# labevents labs
LAB_ITEMIDS = {
    50912: "creatinine",            # Creatinine
    51006: "bun",                   # Urea Nitrogen (BUN)
    51301: "wbc",                   # White Blood Cells
    51222: "hemoglobin",            # Hemoglobin
    50813: "lactate",               # Lactate
    50820: "ph",                    # pH
    50821: "pao2",                  # pO2
    50816: "fio2_lab",              # FiO2 (from ABG)
    51265: "platelets",             # Platelet Count
    50931: "glucose",               # Glucose
    50902: "chloride",              # Chloride
    50882: "bicarbonate",           # Bicarbonate
    50971: "potassium",             # Potassium
    50983: "sodium",                # Sodium
}

ALL_FEATURE_NAMES = sorted(set(VITAL_ITEMIDS.values()) | set(LAB_ITEMIDS.values()))


def read_gz(path: Path) -> pd.DataFrame:
    """Read a .csv.gz file into a DataFrame."""
    print(f"  Reading {path.name} ...", end=" ", flush=True)
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    print(f"{len(df):,} rows")
    return df


def build_cohort(data_dir: Path) -> pd.DataFrame:
    """
    Build the ICU stay cohort with demographics and mortality labels.

    Join chain:
        icustays  →  patients (age, gender)
                  →  admissions (hospital discharge + death)
    """
    hosp = data_dir / "hosp"
    icu  = data_dir / "icu"

    # ── 1. ICU stays ─────────────────────────────────────────────────────
    icustays = read_gz(icu / "icustays.csv.gz")
    # columns: subject_id, hadm_id, stay_id, first_careunit, last_careunit,
    #          intime, outtime, los

    icustays["intime"]  = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (
        (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
    )

    # Keep only adult first ICU stays >= 24 hours
    icustays = (
        icustays
        .sort_values(["subject_id", "intime"])
        .groupby("subject_id", group_keys=False)
        .first()          # first ICU stay per patient
        .reset_index()
    )
    icustays = icustays[icustays["los_hours"] >= 24].copy()
    print(f"  → {len(icustays):,} stays after filtering (first stay, ≥24h)")

    # ── 2. Patient demographics ──────────────────────────────────────────
    patients = read_gz(hosp / "patients.csv.gz")
    # columns: subject_id, gender, anchor_age, anchor_year,
    #          anchor_year_group, dod

    # anchor_age is age at anchor_year — close enough for our purposes
    cohort = icustays.merge(
        patients[["subject_id", "gender", "anchor_age", "dod"]],
        on="subject_id", how="left"
    )

    # ── 3. Hospital admissions — for in-hospital mortality label ─────────
    admissions = read_gz(hosp / "admissions.csv.gz")
    # columns: subject_id, hadm_id, admittime, dischtime, deathtime,
    #          admission_type, insurance, ...
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    cohort = cohort.merge(
        admissions[["hadm_id", "deathtime", "dischtime", "hospital_expire_flag"]],
        on="hadm_id", how="left"
    )

    # ── 4. Mortality label ───────────────────────────────────────────────
    # hospital_expire_flag == 1  →  died in hospital (in admissions table)
    # Fallback: deathtime is not null and within stay
    cohort["mortality_hosp"] = cohort["hospital_expire_flag"].fillna(0).astype(int)

    # ── 5. Age filter ────────────────────────────────────────────────────
    cohort = cohort[cohort["anchor_age"] >= 18].copy()

    # ── 6. Select and rename final columns ──────────────────────────────
    out_cols = [
        "stay_id", "subject_id", "hadm_id",
        "gender", "anchor_age",
        "intime", "outtime", "los_hours",
        "first_careunit", "last_careunit",
        "mortality_hosp",
    ]
    cohort = cohort[out_cols].copy()

    print(f"\n  ✓ Final cohort: {len(cohort):,} ICU stays")
    print(f"    Mortality rate: {cohort['mortality_hosp'].mean()*100:.1f}%")
    print(f"    Median LOS: {cohort['los_hours'].median():.1f}h")
    print(f"    Gender split: {dict(cohort['gender'].value_counts())}")
    return cohort


def report_feature_availability(data_dir: Path, cohort: pd.DataFrame) -> None:
    """
    Print how many stays have at least one measurement for each feature.
    Helps you understand data density before preprocessing.
    """
    print("\n── Feature availability in demo cohort ──")
    stay_ids = set(cohort["stay_id"].tolist())

    icu  = data_dir / "icu"
    hosp = data_dir / "hosp"

    # chartevents
    ce = read_gz(icu / "chartevents.csv.gz")
    ce = ce[ce["stay_id"].isin(stay_ids) & ce["itemid"].isin(VITAL_ITEMIDS)]
    for itemid, name in VITAL_ITEMIDS.items():
        n = ce[ce["itemid"] == itemid]["stay_id"].nunique()
        pct = 100 * n / len(cohort) if len(cohort) > 0 else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {name:<22} [{bar}] {pct:5.1f}% ({n}/{len(cohort)} stays)")

    # labevents
    le = read_gz(hosp / "labevents.csv.gz")
    # labevents uses hadm_id not stay_id; map via cohort
    hadm_ids = set(cohort["hadm_id"].dropna().tolist())
    le = le[le["hadm_id"].isin(hadm_ids) & le["itemid"].isin(LAB_ITEMIDS)]
    for itemid, name in LAB_ITEMIDS.items():
        n = le[le["itemid"] == itemid]["hadm_id"].nunique()
        pct = 100 * n / len(cohort) if len(cohort) > 0 else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {name:<22} [{bar}] {pct:5.1f}% ({n}/{len(cohort)} stays)")


def main():
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV ICU cohort")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to mimic-iv-clinical-database-demo-2.2/ folder"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path (default: <data_dir>/../cohort.csv)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print feature availability report after extraction"
    )
    args = parser.parse_args()

    # Resolve data directory
    script_dir = Path(__file__).parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Auto-discover the demo folder inside the data/ directory
        candidates = list(script_dir.glob("mimic-iv*"))
        if not candidates:
            raise FileNotFoundError(
                "Could not find mimic-iv-clinical-database-demo-2.2/ inside "
                f"{script_dir}. Pass --data-dir explicitly."
            )
        data_dir = candidates[0]

    out_path = Path(args.out) if args.out else script_dir / "cohort.csv"

    print(f"\nMIMIC-IV Demo Cohort Extractor")
    print(f"  Source : {data_dir}")
    print(f"  Output : {out_path}\n")

    cohort = build_cohort(data_dir)
    cohort.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved cohort → {out_path}")

    if args.report:
        report_feature_availability(data_dir, cohort)

    print("\nNext step:  python data/preprocess.py")


if __name__ == "__main__":
    main()