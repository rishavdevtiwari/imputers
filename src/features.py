"""Layer 3 - Feature assembly (fusion).

Join the cleaned sources on location (-> district + agro-zone) + season + crop,
and reduce time-series to the summaries the models need.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SiteFeatures:
    """Everything the analysis pipeline needs for one location + season."""

    district: str
    agro_zone: str
    lat: float
    lon: float
    season: str
    # weather summary
    temp_mean_c: float = 0.0
    temp_min_c: float = 0.0
    temp_max_c: float = 0.0
    rainfall_mm: float = 0.0
    humidity_pct: float = 0.0
    # soil
    soil_ph: float = 0.0
    soil_nitrogen: float = 0.0
    soil_texture: str = "loam"
    # raw context kept for explainability
    extras: dict = field(default_factory=dict)


def summarize_weather(weather: pd.DataFrame, season: str) -> dict[str, float]:
    """Reduce a daily weather frame to seasonal mean temp/rainfall/humidity.

    TODO: filter to the season's months, then aggregate.
    """
    raise NotImplementedError


def build_site_features(
    district: str, season: str, weather: pd.DataFrame, soil: dict
) -> SiteFeatures:
    """Assemble a SiteFeatures object from cleaned weather + soil inputs."""
    raise NotImplementedError


def seasonal_price(prices: pd.DataFrame, commodity: str, season: str) -> tuple[float, float]:
    """Return (median_price, relative_volatility) for a crop in a season.

    Volatility (std/mean) feeds the Monte Carlo risk model.
    """
    raise NotImplementedError
