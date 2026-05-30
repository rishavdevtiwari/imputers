"""Clean + process district_monthly_climate.csv into a model-ready table.

Self-contained (no internal package deps). Produces a supervised dataset for
MONSOON DROUGHT-RISK prediction — the climate risk behind the persona's lost
potato season.

Pipeline
--------
1. Load raw monthly climate panel (NASA POWER, 62 districts x 1981-2019 x 12mo).
2. CLEAN: validate ranges + physical consistency; drop redundant collinear
   columns; compute a Data Quality Index (DQI) before/after.
3. PROCESS: aggregate monthly -> seasonal per (district, year):
     - monsoon (JJAS, months 6-9) precipitation  -> source of the LABEL
     - pre-monsoon MAM (Mar-May) + winter DJF     -> antecedent FEATURES
     - previous-year monsoon (persistence)        -> antecedent FEATURE
4. LABEL: SPI-like z-score of monsoon precip per district, standardized using
   TRAIN years only (leakage-free); drought = z < THRESHOLD.
5. Write data/processed/drought_dataset.csv + reports/{dqi_report,eda_summary}.json

Run:  python3 src/build_dataset.py     (from repo root)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "district_monthly_climate.csv"
OUT_CSV = ROOT / "data" / "processed" / "drought_dataset.csv"
REPORTS = ROOT / "reports"

TRAIN_MAX_YEAR = 2009          # train = years <= this; test = later (time split)
DROUGHT_Z = -0.8               # SPI-like threshold: driest ~1 in 5 monsoons
MONSOON = [6, 7, 8, 9]         # JJAS
PRE_MONSOON = [3, 4, 5]        # MAM

# Redundant / highly collinear columns to drop (EDA found |r| > 0.95).
# Kept: T2M, T2M_MAX, T2M_MIN, T2M_RANGE, RH2M, QV2M, PS, PRECTOT, WS10M.
DROP_COLS = ["TS", "T2MWET", "WS50M", "WS50M_MAX", "WS50M_MIN", "WS50M_RANGE",
             "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE"]

VALIDITY = {  # plausible physical ranges for validity scoring
    "PRECTOT": (0.0, 2000.0), "RH2M": (0.0, 100.0), "T2M": (-40.0, 45.0),
    "T2M_MAX": (-40.0, 50.0), "T2M_MIN": (-50.0, 40.0), "PS": (40.0, 110.0),
    "QV2M": (0.0, 40.0), "WS10M": (0.0, 30.0),
}


# --------------------------------------------------------------------------- #
# Data Quality Index (DQI) — inlined so this script has no internal deps
# --------------------------------------------------------------------------- #
def compute_dqi(df: pd.DataFrame, validity: dict, unique_subset: list) -> dict:
    """DQI = 100 * mean(completeness, validity, consistency, uniqueness)."""
    total = df.size
    completeness = 1.0 - df.isna().sum().sum() / total if total else 1.0

    checked = valid = 0
    for col, (lo, hi) in validity.items():
        if col in df:
            s = df[col].dropna()
            checked += len(s)
            valid += int(s.between(lo, hi).sum())
    validity_score = valid / checked if checked else 1.0

    # consistency: physical rules (True = violation)
    viol = total_chk = 0
    if {"T2M_MAX", "T2M_MIN"}.issubset(df.columns):
        for mask in [
            df["T2M_MAX"] < df["T2M_MIN"],
            (df["T2M"] < df["T2M_MIN"]) | (df["T2M"] > df["T2M_MAX"]),
            (df["RH2M"] < 0) | (df["RH2M"] > 100),
            df["PRECTOT"] < 0,
        ]:
            viol += int(mask.sum())
            total_chk += len(mask)
    consistency = 1.0 - viol / total_chk if total_chk else 1.0

    n = len(df)
    uniqueness = 1.0 - df.duplicated(subset=unique_subset).sum() / n if n else 1.0

    score = 100.0 * (0.25 * completeness + 0.25 * validity_score
                     + 0.25 * consistency + 0.25 * uniqueness)
    return {"completeness": round(completeness, 4), "validity": round(validity_score, 4),
            "consistency": round(consistency, 4), "uniqueness": round(uniqueness, 4),
            "dqi": round(score, 2)}


# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df


def clean(df: pd.DataFrame) -> tuple:
    """Validate + drop redundant columns; return (clean_df, dqi_report)."""
    key = ["DISTRICT", "YEAR", "MONTH"]
    before = compute_dqi(df, VALIDITY, key)
    out = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    after = compute_dqi(out, {k: v for k, v in VALIDITY.items() if k in out}, key)
    report = {
        "dataset": "district_monthly_climate",
        "before": before, "after": after, "rows": int(len(out)),
        "dropped_redundant_columns": [c for c in DROP_COLS if c in df.columns],
        "note": "Source is already pristine (no NaN/dups/sentinels). Cleaning "
                "removes collinear columns to reduce multicollinearity.",
    }
    return out, report


def _winter_year(month: int, year: int) -> int:
    """Dec belongs to the NEXT year's winter so DJF spans the year boundary."""
    return year + 1 if month == 12 else year


def _season_agg(df: pd.DataFrame, months: list, prefix: str) -> pd.DataFrame:
    sub = df[df["MONTH"].isin(months)]
    return sub.groupby(["DISTRICT", "YEAR"]).agg(**{
        f"{prefix}_T2M": ("T2M", "mean"),
        f"{prefix}_T2M_MAX": ("T2M_MAX", "mean"),
        f"{prefix}_T2M_MIN": ("T2M_MIN", "mean"),
        f"{prefix}_T2M_RANGE": ("T2M_RANGE", "mean"),
        f"{prefix}_RH2M": ("RH2M", "mean"),
        f"{prefix}_QV2M": ("QV2M", "mean"),
        f"{prefix}_PS": ("PS", "mean"),
        f"{prefix}_WS10M": ("WS10M", "mean"),
        f"{prefix}_PRECTOT": ("PRECTOT", "sum"),
    }).reset_index()


def process(df: pd.DataFrame) -> tuple:
    """Aggregate to seasonal; build leakage-free label + antecedent features."""
    static = df.groupby("DISTRICT").agg(LAT=("LAT", "first"),
                                        LON=("LON", "first")).reset_index()

    # monsoon precip (label source)
    monsoon = (df[df["MONTH"].isin(MONSOON)]
               .groupby(["DISTRICT", "YEAR"])["PRECTOT"].sum()
               .reset_index().rename(columns={"PRECTOT": "monsoon_precip"}))

    # pre-monsoon (MAM) antecedent features
    mam = _season_agg(df, PRE_MONSOON, "mam")

    # winter (DJF) antecedent features, spanning the year boundary
    wdf = df.copy()
    wdf["WYEAR"] = [_winter_year(m, y) for m, y in zip(wdf["MONTH"], wdf["YEAR"])]
    winter = (wdf[wdf["MONTH"].isin([12, 1, 2])]
              .groupby(["DISTRICT", "WYEAR"]).agg(
                  djf_T2M=("T2M", "mean"), djf_RH2M=("RH2M", "mean"),
                  djf_PRECTOT=("PRECTOT", "sum"), djf_count=("MONTH", "count"))
              .reset_index().rename(columns={"WYEAR": "YEAR"}))
    winter = winter[winter["djf_count"] == 3].drop(columns="djf_count")

    # leakage-free SPI-like label (per-district mean/std from TRAIN years only)
    train = monsoon[monsoon["YEAR"] <= TRAIN_MAX_YEAR]
    stats = (train.groupby("DISTRICT")["monsoon_precip"].agg(["mean", "std"])
             .rename(columns={"mean": "m_mean", "std": "m_std"}))
    monsoon = monsoon.merge(stats, on="DISTRICT", how="left")
    monsoon["monsoon_z"] = (monsoon["monsoon_precip"] - monsoon["m_mean"]) / monsoon["m_std"]
    monsoon["drought"] = (monsoon["monsoon_z"] < DROUGHT_Z).astype(int)

    # persistence feature: previous-year monsoon
    prev = monsoon[["DISTRICT", "YEAR", "monsoon_precip", "monsoon_z"]].copy()
    prev["YEAR"] += 1
    prev = prev.rename(columns={"monsoon_precip": "prev_monsoon_precip",
                                "monsoon_z": "prev_monsoon_z"})

    # assemble: each label-year joined to its antecedent features
    data = monsoon[["DISTRICT", "YEAR", "monsoon_precip", "monsoon_z", "drought"]]
    data = (data.merge(mam, on=["DISTRICT", "YEAR"], how="left")
                .merge(winter, on=["DISTRICT", "YEAR"], how="left")
                .merge(prev, on=["DISTRICT", "YEAR"], how="left")
                .merge(static, on="DISTRICT", how="left"))

    n0 = len(data)
    data = data.dropna(subset=["mam_T2M", "djf_T2M", "prev_monsoon_precip"]).reset_index(drop=True)

    summary = {
        "rows_total": int(n0), "rows_modelable": int(len(data)),
        "dropped_no_antecedent": int(n0 - len(data)),
        "year_range": [int(data.YEAR.min()), int(data.YEAR.max())],
        "n_districts": int(data.DISTRICT.nunique()),
        "drought_rate_overall": round(float(data.drought.mean()), 4),
        "drought_rate_train": round(float(data[data.YEAR <= TRAIN_MAX_YEAR].drought.mean()), 4),
        "drought_rate_test": round(float(data[data.YEAR > TRAIN_MAX_YEAR].drought.mean()), 4),
        "threshold_z": DROUGHT_Z, "train_max_year": TRAIN_MAX_YEAR,
    }
    return data, summary


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("1) Loading raw ...")
    raw = load_raw()

    print("2) Cleaning (validate + drop redundant) ...")
    cleaned, dqi = clean(raw)
    (REPORTS / "dqi_report.json").write_text(json.dumps(dqi, indent=2))
    print(f"   DQI before={dqi['before']['dqi']}  after={dqi['after']['dqi']}  "
          f"dropped={len(dqi['dropped_redundant_columns'])} collinear cols")

    print("3) Processing (seasonal aggregation + leakage-free label) ...")
    data, summary = process(cleaned)
    data.to_csv(OUT_CSV, index=False)
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"   wrote {OUT_CSV.relative_to(ROOT)}  shape={data.shape}")
    print(f"   drought rate: overall={summary['drought_rate_overall']:.0%} "
          f"train={summary['drought_rate_train']:.0%} test={summary['drought_rate_test']:.0%}")
    print("   feature columns:",
          [c for c in data.columns if c not in
           ("DISTRICT", "YEAR", "monsoon_precip", "monsoon_z", "drought")])


if __name__ == "__main__":
    main()
