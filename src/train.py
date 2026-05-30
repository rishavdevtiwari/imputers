"""
Training script for the Ensemble Crop Recommendation model.

Orchestrates data loading, preprocessing, ensemble model fitting,
and persistence of artifacts to the models/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, prepare_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "crop_recommendation.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def build_ensemble_model(random_state: int = 42) -> VotingClassifier:
    """
    Construct a soft-voting ensemble of Random Forest and XGBoost classifiers.

    Parameters
    ----------
    random_state:
        Seed for reproducible training.

    Returns
    -------
    VotingClassifier
        Configured ensemble estimator.
    """
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_classifier = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    return VotingClassifier(
        estimators=[
            ("random_forest", random_forest),
            ("xgboost", xgb_classifier),
        ],
        voting="soft",
    )


def train(
    data_path: Path,
    model_dir: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train the ensemble model and persist artifacts.

    Parameters
    ----------
    data_path:
        Path to the raw training CSV.
    model_dir:
        Directory where model artifacts will be saved.
    test_size:
        Fraction of data reserved for hold-out evaluation.
    random_state:
        Seed for reproducible splits and estimators.

    Returns
    -------
    dict
        Training summary including accuracy and classification report.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y, scaler = prepare_training_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_ensemble_model(random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(y_test, predictions, output_dict=True)

    joblib.dump(model, model_dir / "ensemble_model.joblib")
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "test_size": test_size,
        "random_state": random_state,
        "accuracy": accuracy,
        "classification_report": report,
        "classes": sorted(np.unique(y).astype(str).tolist()),
    }
    with open(model_dir / "training_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train the Crop Recommendation ensemble model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw training CSV file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for saved model artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out test fraction for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = train(
        data_path=args.data_path,
        model_dir=args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Training complete. Hold-out accuracy: {summary['accuracy']:.4f}")
    print(f"Artifacts saved to: {args.model_dir.resolve()}")


if __name__ == "__main__":
    main()
