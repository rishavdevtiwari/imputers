"""
Geospatial helpers for Nepal boundary validation and Folium map rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import folium

from narc_client import GRID_OFFSET_DEGREES, build_macro_plot_grid

NEPAL_LAT_MIN = 26.3
NEPAL_LAT_MAX = 30.5
NEPAL_LON_MIN = 80.0
NEPAL_LON_MAX = 88.3

# 100 hectares ≈ 1 km² circular plot → r = sqrt(1e6 / π) ≈ 564 m
MACRO_PLOT_RADIUS_METERS = 564


@dataclass
class GeofenceResult:
    """Outcome of a Nepal boundary check for a macro plot."""

    is_valid: bool
    message: str
    violating_points: list[tuple[str, float, float]]


def is_within_nepal(lat: float, lon: float) -> bool:
    """Return True when a coordinate lies inside Nepal's bounding box."""
    return (
        NEPAL_LAT_MIN <= lat <= NEPAL_LAT_MAX
        and NEPAL_LON_MIN <= lon <= NEPAL_LON_MAX
    )


def validate_macro_plot_geofence(lat: float, lon: float) -> GeofenceResult:
    """
    Validate center and all 100-hectare quadrant sample points against Nepal bounds.

    Perimeter points use the same +/- 0.0045° offset as the NARC sampling grid.
    """
    violating: list[tuple[str, float, float]] = []
    for label, point_lat, point_lon in build_macro_plot_grid(lat, lon):
        if not is_within_nepal(point_lat, point_lon):
            violating.append((label, point_lat, point_lon))

    if violating:
        details = "; ".join(
            f"{label} ({point_lat:.4f}°, {point_lon:.4f}°)" for label, point_lat, point_lon in violating
        )
        return GeofenceResult(
            is_valid=False,
            message=(
                "Boundary alert: one or more macro-plot sample points fall outside Nepal's "
                f"valid range (Lat {NEPAL_LAT_MIN}°–{NEPAL_LAT_MAX}° N, "
                f"Lon {NEPAL_LON_MIN}°–{NEPAL_LON_MAX}° E). "
                f"Offending point(s): {details}."
            ),
            violating_points=violating,
        )

    return GeofenceResult(
        is_valid=True,
        message="All macro-plot sample points are within Nepal's operational boundary.",
        violating_points=[],
    )


def build_location_map(
    lat: float,
    lon: float,
    *,
    zoom: int = 12,
    grid_coverage: list[tuple[str, float, float, bool]] | None = None,
) -> folium.Map:
    """Build a Folium map centered on the plot with a green pin and 100-ha boundary circle."""
    nepali_map = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=True)

    folium.Marker(
        location=[lat, lon],
        popup=f"Plot center<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}",
        tooltip="100-ha macro plot center",
        icon=folium.Icon(color="green", icon="leaf"),
    ).add_to(nepali_map)

    folium.Circle(
        location=[lat, lon],
        radius=MACRO_PLOT_RADIUS_METERS,
        color="#2ecc71",
        fill=True,
        fill_color="#2ecc71",
        fill_opacity=0.15,
        weight=2,
        popup="100-hectare boundary (~564 m radius)",
    ).add_to(nepali_map)

    coverage_lookup = (
        {label: is_live for label, _, _, is_live in grid_coverage}
        if grid_coverage
        else {}
    )

    for label, point_lat, point_lon in build_macro_plot_grid(lat, lon):
        is_live = coverage_lookup.get(label, True)
        marker_color = "#27ae60" if is_live else "#e67e22"
        status = "NARC crop land" if is_live else "Outside NARC crop land"
        folium.CircleMarker(
            location=[point_lat, point_lon],
            radius=8,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            popup=f"{label}<br>{point_lat:.4f}, {point_lon:.4f}<br>{status}",
        ).add_to(nepali_map)

    folium.Rectangle(
        bounds=[
            [NEPAL_LAT_MIN, NEPAL_LON_MIN],
            [NEPAL_LAT_MAX, NEPAL_LON_MAX],
        ],
        color="#e74c3c",
        weight=1,
        fill=False,
        dash_array="6",
        popup="Nepal operational bounding box",
    ).add_to(nepali_map)

    return nepali_map
