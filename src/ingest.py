"""Layer 1 - Ingestion & cache.

Fetches each external source and caches it offline (so the live demo never
depends on a flaky API call). Pre-cache all 77 district centroids before the
event; SoilGrids can be fetched per location on demand.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DIR, load_config


def fetch_weather(lat: float, lon: float, cache: bool = True) -> pd.DataFrame:
    """Fetch daily weather history from NASA POWER for one point.

    Returns a DataFrame indexed by date with columns:
    T2M_MAX, T2M_MIN, T2M, PRECTOTCORR, RH2M.
    NASA POWER uses -999 as a fill value -> handled in `quality.clean_weather`.

    TODO: build request URL from config['sources']['nasa_power'] and parse JSON.
    """
    raise NotImplementedError


def fetch_soil(lat: float, lon: float) -> dict[str, float]:
    """Fetch soil properties (pH, nitrogen, texture) from SoilGrids for a point.

    TODO: query SoilGrids REST API; return {'phh2o':..., 'nitrogen':..., ...}.
    """
    raise NotImplementedError


def load_yield_table() -> pd.DataFrame:
    """Load district-level yield table (MoALD + FAOSTAT).

    Expected columns: district, agro_zone, crop, yield_kg_ha, year.
    TODO: read cached CSV from data/raw/.
    """
    raise NotImplementedError


def load_price_history() -> pd.DataFrame:
    """Load Kalimati daily wholesale price history.

    Expected columns: date, commodity, min, max, avg (NPR/kg).
    TODO: read cached CSV from data/raw/.
    """
    raise NotImplementedError


def load_disease_kb() -> pd.DataFrame:
    """Load the crop x disease knowledge base shipped in data/disease_kb.csv."""
    return pd.read_csv(Path(RAW_DIR).parent / "disease_kb.csv")


def cache_district_centroids() -> pd.DataFrame:
    """Return / build the 77-district centroid table used to pre-cache weather.

    Expected columns: district, agro_zone, lat, lon.
    TODO: ship a static CSV of district centroids in data/raw/.
    """
    raise NotImplementedError
