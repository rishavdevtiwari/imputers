"""
Data pipeline module for the Crop Recommendation Ensemble ML system.

Provides reusable functions for loading tabular datasets, cleaning records,
and scaling soil and climate feature columns prior to model training or inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Canonical feature columns used across training and inference.
FEATURE_COLUMNS: list[str] = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

# Default target column for crop recommendation datasets.
TARGET_COLUMN: str = "label"


def load_dataset(
    path: str | Path,
    *,
    parse_dates: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load a tabular dataset from a CSV file.

    Parameters
    ----------
    path:
        Filesystem path to the CSV file.
    parse_dates:
        Optional column names to parse as datetime values.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with a reset integer index.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the file extension is not supported (currently CSV only).
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix != ".csv":
        raise ValueError(
            f"Unsupported file format '{suffix}'. Only '.csv' files are supported."
        )

    df = pd.read_csv(file_path, parse_dates=list(parse_dates) if parse_dates else None)
    return df.reset_index(drop=True)


def clean_data(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    target_column: Optional[str] = TARGET_COLUMN,
    knn_neighbors: int = 5,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Clean a crop recommendation dataset.

    Steps performed:
      1. Drop exact duplicate rows (optional).
      2. Coerce feature columns to numeric, replacing invalid values with NaN.
      3. Impute missing feature values using KNNImputer.
      4. Drop rows with missing target labels when a target column is specified.

    Parameters
    ----------
    df:
        Raw input dataframe.
    feature_columns:
        Columns to clean and impute.
    target_column:
        Label column name. Rows with missing labels are removed when set.
    knn_neighbors:
        Number of neighbors for KNN imputation.
    drop_duplicates:
        Whether to remove duplicate rows before imputation.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with imputed feature values.
    """
    cleaned = df.copy()
    features = list(feature_columns)

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in features:
        if column not in cleaned.columns:
            raise KeyError(f"Expected feature column '{column}' not found in dataset.")
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if target_column and target_column in cleaned.columns:
        cleaned = cleaned.dropna(subset=[target_column])

    imputer = KNNImputer(n_neighbors=knn_neighbors)
    cleaned.loc[:, features] = imputer.fit_transform(cleaned[features])

    return cleaned.reset_index(drop=True)


def scale_features(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize feature columns using sklearn StandardScaler.

    Parameters
    ----------
    df:
        Input dataframe containing feature columns.
    feature_columns:
        Columns to scale.
    scaler:
        Pre-fitted scaler for inference. When None, a new StandardScaler is created.
    fit:
        If True, fit the scaler on the provided data. Set to False during inference
        when passing a pre-fitted scaler.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled dataframe copy and the fitted (or reused) scaler instance.
    """
    scaled_df = df.copy()
    features = list(feature_columns)

    missing = [column for column in features if column not in scaled_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns for scaling: {missing}")

    active_scaler = scaler or StandardScaler()

    if fit:
        scaled_values = active_scaler.fit_transform(scaled_df[features])
    else:
        if scaler is None:
            raise ValueError("A pre-fitted scaler must be provided when fit=False.")
        scaled_values = active_scaler.transform(scaled_df[features])

    scaled_df[features] = pd.DataFrame(
        scaled_values,
        columns=features,
        index=scaled_df.index,
    )
    return scaled_df, active_scaler


def prepare_training_data(
    path: str | Path,
    *,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    knn_neighbors: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    End-to-end helper: load, clean, and scale a labeled training dataset.

    Parameters
    ----------
    path:
        Path to the raw CSV training file.
    feature_columns:
        Feature columns to clean and scale.
    target_column:
        Name of the crop label column.
    knn_neighbors:
        Number of neighbors for KNN imputation.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, StandardScaler]
        Scaled feature matrix, target series, and fitted StandardScaler.
    """
    raw_df = load_dataset(path)
    cleaned_df = clean_data(
        raw_df,
        feature_columns=feature_columns,
        target_column=target_column,
        knn_neighbors=knn_neighbors,
    )

    scaled_df, scaler = scale_features(
        cleaned_df,
        feature_columns=feature_columns,
        fit=True,
    )

    X = scaled_df[list(feature_columns)]
    y = cleaned_df[target_column]
    return X, y, scaler


def prepare_inference_data(
    df: pd.DataFrame,
    *,
    scaler: StandardScaler,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Clean and scale a dataframe for model inference.

    Parameters
    ----------
    df:
        Raw or partially prepared input records.
    scaler:
        Fitted StandardScaler from training.
    feature_columns:
        Feature columns to clean and scale.
    knn_neighbors:
        Number of neighbors for KNN imputation.

    Returns
    -------
    pd.DataFrame
        Scaled feature matrix ready for prediction.
    """
    cleaned_df = clean_data(
        df,
        feature_columns=feature_columns,
        target_column=None,
        knn_neighbors=knn_neighbors,
    )
    scaled_df, _ = scale_features(
        cleaned_df,
        feature_columns=feature_columns,
        scaler=scaler,
        fit=False,
    )
    return scaled_df[list(feature_columns)]
