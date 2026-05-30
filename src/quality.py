"""Layer 2 - Data Quality (the `imputers` core).

Clean -> validate -> impute -> score. Computes a Data Quality Index (DQI) on
each source before and after cleaning, and writes reports/dqi.json.

DQI = 100 * (w_comp*Completeness + w_valid*Validity
             + w_consist*Consistency + w_uniq*Uniqueness)
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, load_config


# --------------------------------------------------------------------------- #
# DQI metrics (each returns a value in [0, 1])
# --------------------------------------------------------------------------- #
def completeness(df: pd.DataFrame) -> float:
    """1 - (missing cells / total cells)."""
    total = df.size
    return 1.0 if total == 0 else 1.0 - df.isna().sum().sum() / total


def uniqueness(df: pd.DataFrame, subset: list[str] | None = None) -> float:
    """1 - (duplicate rows / total rows)."""
    n = len(df)
    return 1.0 if n == 0 else 1.0 - df.duplicated(subset=subset).sum() / n


def validity(df: pd.DataFrame, rules: dict[str, tuple[float, float]]) -> float:
    """Fraction of values within configured [min, max] ranges.

    `rules` maps column -> (min, max). Columns not present are skipped.
    """
    checked = valid = 0
    for col, (lo, hi) in rules.items():
        if col not in df:
            continue
        series = df[col].dropna()
        checked += len(series)
        valid += series.between(lo, hi).sum()
    return 1.0 if checked == 0 else valid / checked


def consistency(df: pd.DataFrame, checks: list[Callable[[pd.DataFrame], pd.Series]]) -> float:
    """1 - (cross-field rule violations / total checks).

    Each check returns a boolean Series where True = violation.
    """
    if not checks:
        return 1.0
    violations = total = 0
    for check in checks:
        mask = check(df)
        violations += int(mask.sum())
        total += len(mask)
    return 1.0 if total == 0 else 1.0 - violations / total


def compute_dqi(
    df: pd.DataFrame,
    validity_rules: dict[str, tuple[float, float]] | None = None,
    consistency_checks: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
    unique_subset: list[str] | None = None,
) -> dict[str, float]:
    """Compute the four sub-metrics and the weighted DQI (0-100)."""
    w = load_config()["dqi"]["weights"]
    comp = completeness(df)
    valid = validity(df, validity_rules or {})
    consist = consistency(df, consistency_checks or [])
    uniq = uniqueness(df, unique_subset)
    score = 100.0 * (
        w["completeness"] * comp
        + w["validity"] * valid
        + w["consistency"] * consist
        + w["uniqueness"] * uniq
    )
    return {
        "completeness": round(comp, 4),
        "validity": round(valid, 4),
        "consistency": round(consist, 4),
        "uniqueness": round(uniq, 4),
        "dqi": round(score, 2),
    }


# --------------------------------------------------------------------------- #
# Cleaning + imputation
# --------------------------------------------------------------------------- #
def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NASA POWER -999 sentinels with NaN and interpolate gaps."""
    fill = load_config()["sources"]["nasa_power"]["fill_value"]
    out = df.replace(fill, np.nan)
    return out.interpolate(method="linear", limit_direction="both")


def impute_yields(df: pd.DataFrame, group_col: str = "agro_zone") -> pd.DataFrame:
    """Impute missing district x crop yields using the agro-zone group mean.

    A simple, defensible baseline; swap for KNNImputer if time allows.
    TODO: optionally use sklearn.impute.KNNImputer grouped by agro_zone.
    """
    out = df.copy()
    out["yield_kg_ha"] = out.groupby([group_col, "crop"])["yield_kg_ha"].transform(
        lambda s: s.fillna(s.mean())
    )
    return out


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return boolean mask of IQR outliers (for skewed distributions)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


def save_dqi_report(report: dict, name: str = "dqi.json") -> None:
    """Persist a before/after DQI report to reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
