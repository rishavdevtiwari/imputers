"""Layer 4a - Crop suitability.

Train/serve a classifier mapping (N, P, K, temperature, humidity, pH, rainfall)
-> crop, using the Crop Recommendation dataset relabelled to Nepal crops.
Outputs a suitability probability per crop, used downstream to scale yield.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import SiteFeatures

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train_model(dataset_csv: str | Path, model_out: str | Path) -> float:
    """Train a RandomForest suitability classifier; return validation accuracy.

    TODO: load dataset, train/test split, fit RandomForestClassifier,
    persist with joblib, report accuracy vs a majority-class baseline.
    """
    raise NotImplementedError


def load_model(model_path: str | Path):
    """Load the persisted suitability model."""
    raise NotImplementedError


def score_crops(site: SiteFeatures, model=None) -> dict[str, float]:
    """Return {crop: suitability_probability in [0, 1]} for a site.

    TODO: build the feature row from `site` and call model.predict_proba.
    """
    raise NotImplementedError
