"""Layer 4d - Economics.

profit (NPR/ha) = yield (kg/ha) * price (NPR/kg) - cost (NPR/ha)
roi (%)         = profit / cost * 100
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import load_config


@dataclass
class Economics:
    crop: str
    yield_kg_ha: float
    price_npr_kg: float
    cost_npr_ha: float
    revenue_npr_ha: float
    profit_npr_ha: float
    roi_pct: float


def compute_economics(crop: str, yield_kg_ha: float, price_npr_kg: float) -> Economics:
    """Compute revenue, profit, and ROI for a crop."""
    cost = load_config()["crops"][crop]["cost_npr_ha"]
    revenue = yield_kg_ha * price_npr_kg
    profit = revenue - cost
    roi = (profit / cost * 100.0) if cost else 0.0
    return Economics(
        crop=crop,
        yield_kg_ha=yield_kg_ha,
        price_npr_kg=price_npr_kg,
        cost_npr_ha=cost,
        revenue_npr_ha=revenue,
        profit_npr_ha=profit,
        roi_pct=roi,
    )
