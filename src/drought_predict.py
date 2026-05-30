"""Inference for the monsoon drought-risk model.

Loads the trained model + feature list + tuned operating threshold and scores
rows from the processed dataset. Importable by the Streamlit dashboard, and
runnable from the CLI:

    python3 src/drought_predict.py --district Dhading --year 2018
    python3 src/drought_predict.py --district Dhading            # all years
    python3 src/drought_predict.py --list                        # list districts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "drought_model.joblib"
FEATURES_PATH = ROOT / "models" / "drought_features.joblib"
THRESHOLD_PATH = ROOT / "models" / "drought_threshold.joblib"
DATA_PATH = ROOT / "data" / "processed" / "drought_dataset.csv"

_SETUP_HINT = ("Artifacts not found. Build them first:\n"
               "  python3 src/drought_preprocess.py\n"
               "  python3 src/drought_model_selection.py")


def artifacts_exist() -> bool:
    return MODEL_PATH.exists() and FEATURES_PATH.exists() and DATA_PATH.exists()


def load_artifacts():
    """Return (model, feature_list). Raises FileNotFoundError with a hint."""
    if not artifacts_exist():
        raise FileNotFoundError(_SETUP_HINT)
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


def load_threshold(default: float = 0.5) -> float:
    """Tuned operating threshold persisted by model selection (recall-targeted).

    Falls back to `default` if the artifact is missing (older models)."""
    if THRESHOLD_PATH.exists():
        meta = joblib.load(THRESHOLD_PATH)
        return float(meta.get("threshold", default)) if isinstance(meta, dict) else float(meta)
    return default


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(_SETUP_HINT)
    return pd.read_csv(DATA_PATH)


def predict_frame(df: pd.DataFrame, threshold: float | None = None,
                  model=None, features=None) -> pd.DataFrame:
    """Add drought_prob + drought_pred columns to a frame of feature rows.

    threshold=None uses the persisted tuned operating threshold."""
    if model is None:
        model, features = load_artifacts()
    if threshold is None:
        threshold = load_threshold()
    out = df.copy()
    out["drought_prob"] = model.predict_proba(out[features])[:, 1].round(4)
    out["drought_pred"] = (out["drought_prob"] >= threshold).astype(int)
    return out


def predict_district(district: str, threshold: float | None = None,
                     model=None, features=None, data=None) -> pd.DataFrame:
    """Score every available year for one district."""
    data = load_dataset() if data is None else data
    sub = data[data["DISTRICT"] == district]
    if sub.empty:
        raise ValueError(f"District not found: {district}")
    return predict_frame(sub, threshold, model, features)


def predict_district_year(district: str, year: int,
                          threshold: float | None = None) -> dict:
    """Score a single district-year; returns a tidy dict."""
    row = predict_district(district, threshold)
    row = row[row["YEAR"] == year]
    if row.empty:
        raise ValueError(f"No record for {district} {year}")
    r = row.iloc[0]
    return {
        "district": district, "year": int(year),
        "monsoon_precip_mm": round(float(r["monsoon_precip"]), 1),
        "drought_probability": float(r["drought_prob"]),
        "predicted_drought": bool(r["drought_pred"]),
        "actual_drought": bool(r["drought"]),
    }


def _cli() -> None:
    p = argparse.ArgumentParser(description="Drought-risk inference")
    p.add_argument("--district")
    p.add_argument("--year", type=int)
    p.add_argument("--threshold", type=float, default=None,
                   help="decision threshold (default: persisted tuned threshold)")
    p.add_argument("--list", action="store_true", help="list available districts")
    args = p.parse_args()

    if args.list:
        print(", ".join(sorted(load_dataset()["DISTRICT"].unique())))
        return
    if not args.district:
        p.error("provide --district (or --list)")

    if args.year:
        import json
        print(json.dumps(predict_district_year(args.district, args.year, args.threshold), indent=2))
    else:
        res = predict_district(args.district, args.threshold)
        cols = ["DISTRICT", "YEAR", "monsoon_precip", "drought", "drought_pred", "drought_prob"]
        print(res[cols].to_string(index=False))


if __name__ == "__main__":
    _cli()
