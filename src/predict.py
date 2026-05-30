"""
Inference module for the Crop Recommendation Ensemble ML system.

Loads persisted model artifacts and generates crop recommendations
from soil and climate feature inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_pipeline import FEATURE_COLUMNS, prepare_inference_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


@dataclass
class PredictionResult:
    """Structured output for a single crop recommendation."""

    recommended_crop: str
    confidence: float
    probabilities: dict[str, float]


class CropRecommendationPredictor:
    """
    Wrapper around persisted ensemble model artifacts for inference.

    Parameters
    ----------
    model_dir:
        Directory containing `ensemble_model.joblib`, `feature_scaler.joblib`,
        and optional `training_metadata.json`.
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "ensemble_model.joblib")
        self.scaler: StandardScaler = joblib.load(self.model_dir / "feature_scaler.joblib")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.model_dir / "training_metadata.json"
        if not metadata_path.exists():
            return {"feature_columns": FEATURE_COLUMNS}
        with open(metadata_path, encoding="utf-8") as fh:
            return json.load(fh)

    def predict_from_dict(self, features: dict[str, float]) -> PredictionResult:
        """
        Generate a crop recommendation from a feature dictionary.

        Parameters
        ----------
        features:
            Mapping of feature names to numeric values.

        Returns
        -------
        PredictionResult
            Recommended crop label, confidence score, and class probabilities.
        """
        input_df = pd.DataFrame([features])
        return self.predict_batch(input_df)[0]

    def predict_batch(self, df: pd.DataFrame) -> list[PredictionResult]:
        """
        Generate crop recommendations for multiple input records.

        Parameters
        ----------
        df:
            Dataframe containing raw feature columns.

        Returns
        -------
        list[PredictionResult]
            One prediction result per input row.
        """
        feature_columns = self.metadata.get("feature_columns", FEATURE_COLUMNS)
        X = prepare_inference_data(df, scaler=self.scaler, feature_columns=feature_columns)

        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        classes = self.model.classes_

        results: list[PredictionResult] = []
        for index, label in enumerate(labels):
            prob_vector = probabilities[index]
            best_index = int(np.argmax(prob_vector))
            results.append(
                PredictionResult(
                    recommended_crop=str(label),
                    confidence=float(prob_vector[best_index]),
                    probabilities={
                        str(classes[class_index]): float(prob_vector[class_index])
                        for class_index in range(len(classes))
                    },
                )
            )
        return results


def predict_single(
    features: dict[str, float],
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> PredictionResult:
    """
    Convenience function for one-off predictions.

    Parameters
    ----------
    features:
        Mapping of feature names to numeric values.
    model_dir:
        Directory containing trained model artifacts.

    Returns
    -------
    PredictionResult
        Recommended crop and associated probabilities.
    """
    predictor = CropRecommendationPredictor(model_dir=model_dir)
    return predictor.predict_from_dict(features)
