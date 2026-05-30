"""
NARC (National Soil Science Research Center) API client utilities.

Handles live soil-data requests, JSON parsing, 100-hectare grid sampling,
and localized fallback payloads when the API is unreachable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from statistics import mean
from typing import Any, Callable, Optional

import requests

# Documented endpoint (legacy) and current live path observed in production.
NARC_SOILDATA_URLS: tuple[str, ...] = (
    "https://soil.narc.gov.np/api/soildata",
    "https://soil.narc.gov.np/soil/api/soildata",
)
NARC_TIMEOUT_SECONDS = 4
GRID_OFFSET_DEGREES = 0.0045
NARC_OUTSIDE_CROP_LAND_MESSAGE = "Please select the crop land"

# NARC doc example — Kailali, Sudurpaschim (returns live soil survey data).
NARC_DEMO_LAT = 28.574
NARC_DEMO_LON = 80.807

# Preset coordinates verified against the live NARC `/soil/api/soildata` endpoint.
NARC_VERIFIED_COORDS: dict[str, tuple[float, float]] = {
    "Chitwan (Terai Hub)": (27.4500, 84.3500),
    "Jhapa (Eastern Plains)": (26.6400, 87.9800),
    "Kailali (Far-West Basin)": (28.7000, 80.6000),
}

@dataclass
class GridNarcCoverage:
    """NARC crop-land availability for each macro-plot sample point."""

    live_count: int
    total: int
    points: list[tuple[str, float, float, bool]]  # label, lat, lon, is_live

    @property
    def is_full_coverage(self) -> bool:
        return self.live_count == self.total

    @property
    def is_partial_coverage(self) -> bool:
        return 0 < self.live_count < self.total


# When a perimeter point sits outside NARC crop land, retry closer to center (still inside 100 ha).
INSET_SAMPLE_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.2)


@dataclass
class NarcFetchResult:
    """Outcome of a single NARC soil-data request."""

    payload: Optional[dict[str, Any]] = None
    failure_reason: Optional[str] = None


@dataclass
class NarcParsedSample:
    """Normalized soil sample extracted from a NARC payload."""

    label: str
    lat: float
    lon: float
    source: str
    ph: Optional[float]
    total_nitrogen: Optional[float]
    p2o5: Optional[float]
    potassium: Optional[float]
    province: str
    district: str
    palika: str
    sample_detail: str = "live"
    raw_payload: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def ml_features(self) -> dict[str, float]:
        """Map NARC schema keys to ensemble model feature names."""
        return map_narc_to_ml_features(
            {
                "ph": self.ph,
                "total_nitrogen": self.total_nitrogen,
                "p2o5": self.p2o5,
                "potassium": self.potassium,
            }
        )


@dataclass
class MacroPlotEvaluation:
    """Aggregated results for a 100-hectare macro plot evaluation."""

    center_lat: float
    center_lon: float
    samples: list[NarcParsedSample]
    live_count: int
    fallback_count: int

    @property
    def regional_context(self) -> dict[str, str]:
        """Use the center sample for primary regional context."""
        center = next((sample for sample in self.samples if sample.label == "Center"), self.samples[0])
        return {
            "province": center.province,
            "district": center.district,
            "palika": center.palika,
        }

    @property
    def mean_narc_metrics(self) -> dict[str, Optional[float]]:
        """Statistical mean of parsed NARC soil metrics across all sub-points."""
        return {
            "ph": _safe_mean([sample.ph for sample in self.samples]),
            "total_nitrogen": _safe_mean([sample.total_nitrogen for sample in self.samples]),
            "p2o5": _safe_mean([sample.p2o5 for sample in self.samples]),
            "potassium": _safe_mean([sample.potassium for sample in self.samples]),
        }

    @property
    def mean_ml_features(self) -> dict[str, float]:
        """Mean ML-ready N, P, K, ph values across the macro plot."""
        mapped = map_narc_to_ml_features(self.mean_narc_metrics)
        return {key: value for key, value in mapped.items() if value is not None}


def fetch_live_narc_data(lat: float, lon: float) -> Optional[dict[str, Any]]:
    """
    Request live soil data from the NARC API for a coordinate pair.

    Tries documented and current endpoint paths. Returns None when the request
    times out, returns a non-200 status, or the coordinate is outside mapped
    crop land.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees.

    Returns
    -------
    dict or None
        Parsed JSON soil record on success, otherwise None.
    """
    return fetch_live_narc_data_detailed(lat, lon).payload


def preview_narc_coverage(lat: float, lon: float) -> NarcFetchResult:
    """
    Check whether the plot center lies on NARC mapped crop land.

    Used for UI preflight warnings before the 5-point macro plot evaluation.
    """
    return fetch_live_narc_data_detailed(lat, lon)


def assess_grid_narc_coverage(lat: float, lon: float) -> GridNarcCoverage:
    """
    Check all five macro-plot sample points against NARC mapped crop land.

    Nepal geofence and NARC crop-land coverage are different — a plot can be
    fully inside Nepal while straddling the edge of the soil survey polygon.
    """
    points: list[tuple[str, float, float, bool]] = []
    live_count = 0

    for label, point_lat, point_lon in build_macro_plot_grid(lat, lon):
        is_live = fetch_live_narc_data(point_lat, point_lon) is not None
        if is_live:
            live_count += 1
        points.append((label, point_lat, point_lon, is_live))

    return GridNarcCoverage(live_count=live_count, total=len(points), points=points)


def _fetch_live_with_inset(
    center_lat: float,
    center_lon: float,
    target_lat: float,
    target_lon: float,
) -> tuple[Optional[dict[str, Any]], float, float, str]:
    """
    Try the nominal grid coordinate, then progressively inset toward center.

    Keeps samples inside the 100-hectare circle while maximizing live NARC hits
    when the plot straddles a crop-land survey boundary.
    """
    last_reason = "Coordinate outside NARC mapped crop land."

    for fraction in INSET_SAMPLE_FRACTIONS:
        sample_lat = center_lat + (target_lat - center_lat) * fraction
        sample_lon = center_lon + (target_lon - center_lon) * fraction
        result = fetch_live_narc_data_detailed(sample_lat, sample_lon)
        if result.payload is not None:
            detail = "live" if fraction >= 0.999 else f"live-inset-{int(fraction * 100)}pct"
            return result.payload, sample_lat, sample_lon, detail
        if result.failure_reason:
            last_reason = result.failure_reason

    return None, target_lat, target_lon, last_reason


def fetch_live_narc_data_detailed(lat: float, lon: float) -> NarcFetchResult:
    """Like fetch_live_narc_data, but includes a human-readable failure reason."""
    headers = {
        "User-Agent": "CropRecommendationHackathon/1.0",
        "Accept": "application/json",
    }

    last_reason = "NARC API unreachable (timeout or HTTP error)."

    for base_url in NARC_SOILDATA_URLS:
        url = f"{base_url}?lat={lat}&lon={lon}"
        try:
            response = requests.get(url, timeout=NARC_TIMEOUT_SECONDS, headers=headers)
            if response.status_code != 200:
                last_reason = f"HTTP {response.status_code} from NARC API."
                continue

            payload = _unwrap_narc_payload(response.json())
            if not _is_valid_narc_soil_payload(payload):
                message = str(payload.get("result", "")).strip()
                if NARC_OUTSIDE_CROP_LAND_MESSAGE.lower() in message.lower():
                    last_reason = (
                        "Coordinate outside NARC mapped crop land — "
                        "pick a rural/agricultural point (see NARC soil map)."
                    )
                else:
                    last_reason = message or "NARC returned no soil record for this point."
                continue

            return NarcFetchResult(payload=payload)
        except (requests.RequestException, ValueError):
            continue

    return NarcFetchResult(failure_reason=last_reason)


def build_fallback_narc_payload(lat: float, lon: float) -> dict[str, Any]:
    """
    Build a realistic localized fallback payload matching the NARC JSON schema.

    Values are lightly perturbed by coordinate hash so perimeter quadrants differ
    slightly, simulating spatial soil variability within the macro plot.
    """
    seed = abs(hash(f"{lat:.4f}:{lon:.4f}")) % 1000
    jitter = (seed % 17) - 8

    return {
        "coord": {"lat": lat, "lon": lon, "elevation": round(1.2 + (seed % 50) / 100, 2)},
        "ph": f"{6.35 + jitter * 0.03:.2f} ",
        "organic_matter": f"{1.45 + jitter * 0.02:.2f} %",
        "total_nitrogen": f"{0.09 + jitter * 0.002:.3f} %",
        "potassium": f"{180.0 + jitter * 2.5:.2f} kg/ha",
        "p2o5": f"{36.5 + jitter * 0.8:.1f} kg/ha",
        "boron": f"{0.12 + (seed % 5) * 0.01:.2f} ppm",
        "zinc": f"{0.38 + (seed % 7) * 0.02:.2f} ppm",
        "sand": f"{32.0 + jitter:.2f} %",
        "clay": f"{18.0 + jitter * 0.5:.2f} %",
        "slit": f"{50.0 - jitter:.2f} %",
        "parentsoil": "Residual non calcareous",
        "province": _infer_province(lat, lon),
        "district": _infer_district(lat, lon),
        "palika": _infer_palika(lat, lon),
        "source": "fallback",
    }


def parse_narc_payload(
    payload: dict[str, Any],
    *,
    label: str,
    lat: float,
    lon: float,
    source: str,
    sample_detail: str = "live",
) -> NarcParsedSample:
    """Extract documented NARC keys and cast soil metrics to float."""
    return NarcParsedSample(
        label=label,
        lat=lat,
        lon=lon,
        source=source,
        ph=_parse_narc_numeric(payload.get("ph")),
        total_nitrogen=_parse_narc_numeric(payload.get("total_nitrogen")),
        p2o5=_parse_narc_numeric(payload.get("p2o5")),
        potassium=_parse_narc_numeric(payload.get("potassium")),
        province=str(payload.get("province", "Unknown")).strip(),
        district=str(payload.get("district", "Unknown")).strip(),
        palika=str(payload.get("palika", "Unknown")).strip(),
        sample_detail=sample_detail,
        raw_payload=payload,
    )


def map_narc_to_ml_features(parsed_metrics: dict[str, Optional[float]]) -> dict[str, float]:
    """
    Map NARC soil keys to ML feature names used by the ensemble model.

    NARC reports total nitrogen as a percentage; values below 10 are scaled to
    approximate the kg/ha range used in the crop recommendation training set.
    """
    mapped: dict[str, float] = {}

    ph_value = parsed_metrics.get("ph")
    if ph_value is not None:
        mapped["ph"] = float(ph_value)

    nitrogen = parsed_metrics.get("total_nitrogen")
    if nitrogen is not None:
        mapped["N"] = float(nitrogen * 1000 if nitrogen < 10 else nitrogen)

    phosphorus = parsed_metrics.get("p2o5")
    if phosphorus is not None:
        mapped["P"] = float(phosphorus)

    potassium = parsed_metrics.get("potassium")
    if potassium is not None:
        mapped["K"] = float(potassium)

    return mapped


def build_macro_plot_grid(lat: float, lon: float) -> list[tuple[str, float, float]]:
    """
    Build the 5-point spatial grid for a 100-hectare macro plot.

    Points: Center, North-West, North-East, South-West, South-East.
    Perimeter quadrants are offset by +/- 0.0045 degrees from center.
    """
    offset = GRID_OFFSET_DEGREES
    return [
        ("Center", lat, lon),
        ("North-West", lat + offset, lon - offset),
        ("North-East", lat + offset, lon + offset),
        ("South-West", lat - offset, lon - offset),
        ("South-East", lat - offset, lon + offset),
    ]


def evaluate_macro_plot(
    lat: float,
    lon: float,
    *,
    log_fn: Optional[LogFn] = None,
) -> MacroPlotEvaluation:
    """
    Sample soil data across the 100-hectare macro plot grid.

    Attempts live NARC API calls for each sub-point and falls back to localized
    synthetic payloads when the connection fails or times out.
    """
    samples: list[NarcParsedSample] = []
    live_count = 0
    fallback_count = 0

    for label, point_lat, point_lon in build_macro_plot_grid(lat, lon):
        live_payload, sample_lat, sample_lon, fetch_detail = _fetch_live_with_inset(
            lat, lon, point_lat, point_lon
        )

        if live_payload is not None:
            source = "live"
            live_count += 1
            if log_fn:
                if fetch_detail == "live":
                    log_fn(
                        "success",
                        f"{label}: live NARC data retrieved ({sample_lat:.4f}, {sample_lon:.4f}).",
                    )
                else:
                    log_fn(
                        "success",
                        f"{label}: perimeter outside crop land — inset sample at "
                        f"{fetch_detail.replace('live-inset-', '')} toward center "
                        f"({sample_lat:.4f}, {sample_lon:.4f}) — live NARC data retrieved.",
                    )
        else:
            live_payload = build_fallback_narc_payload(point_lat, point_lon)
            source = "fallback"
            sample_lat, sample_lon = point_lat, point_lon
            fallback_count += 1
            reason = str(fetch_detail)
            if log_fn:
                log_fn(
                    "warning",
                    f"{label}: {reason} — using localized fallback "
                    f"({sample_lat:.4f}, {sample_lon:.4f}).",
                )

        samples.append(
            parse_narc_payload(
                live_payload,
                label=label,
                lat=sample_lat,
                lon=sample_lon,
                source=source,
                sample_detail=fetch_detail if source == "live" else "fallback",
            )
        )

    return MacroPlotEvaluation(
        center_lat=lat,
        center_lon=lon,
        samples=samples,
        live_count=live_count,
        fallback_count=fallback_count,
    )


def _unwrap_narc_payload(payload: Any) -> dict[str, Any]:
    """Normalize list or nested API responses to a single record dictionary."""
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Empty NARC API response list.")
        first = payload[0]
        if isinstance(first, dict):
            return first
        raise ValueError("Unexpected NARC list item type.")

    if isinstance(payload, dict):
        for key in ("results", "data", "soildata", "soil"):
            nested = payload.get(key)
            if isinstance(nested, list) and nested:
                return nested[0]
            if isinstance(nested, dict):
                return nested
        return payload

    raise ValueError("Unsupported NARC payload structure.")


def _is_valid_narc_soil_payload(payload: dict[str, Any]) -> bool:
    """True when the payload contains soil survey fields rather than an error message."""
    if not payload:
        return False

    result_only = set(payload.keys()) == {"result"}
    if result_only:
        return False

    return any(key in payload for key in ("ph", "total_nitrogen", "p2o5", "potassium"))


def _parse_narc_numeric(value: Any) -> Optional[float]:
    """Extract the first numeric token from a NARC string (supports HTML-wrapped values)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = unescape(str(value))
    text = re.sub(r"<[^>]+>", "", text)
    match = re.search(r"[-+]?\d*\.?\d+", text.strip())
    if not match:
        return None
    return float(match.group())


def _safe_mean(values: list[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return float(mean(numeric))


def _infer_province(lat: float, lon: float) -> str:
    if lat > 28.5:
        return "Sudurpaschim"
    if lat > 27.8:
        return "Lumbini"
    if lon > 85.2 and lat > 27.4:
        return "Bagmati"
    if lon > 87.0:
        return "Koshi"
    return "Madhesh"


def _infer_district(lat: float, lon: float) -> str:
    if 27.6 <= lat <= 27.8 and 85.2 <= lon <= 85.4:
        return "Kathmandu"
    if lat > 28.0:
        return "Kailali"
    if lon > 87.0:
        return "Morang"
    return "Parsa"


def _infer_palika(lat: float, lon: float) -> str:
    if 27.6 <= lat <= 27.8 and 85.2 <= lon <= 85.4:
        return "Kathmandu Metropolitan City"
    if lat > 28.0:
        return "Kailari Gaunpalika"
    return "Birgunj Metropolitan City"
