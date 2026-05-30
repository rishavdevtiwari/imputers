"""Layer 4c - Coherent yield estimate.

yield = potential_yield (crop ceiling, from MoALD/config)
        * suitability (0-1, from classifier)
        * (1 - expected_loss) (from disease_risk)

Making suitability + loss scale the yield ensures the soil/weather analysis
actually moves the final number (no disconnect between inputs and output).
"""
from __future__ import annotations

from .config import load_config


def estimate_yield(crop: str, suitability: float, expected_loss: float) -> float:
    """Return estimated yield (kg/ha) for a crop given suitability and loss."""
    crops = load_config()["crops"]
    if crop not in crops:
        raise KeyError(f"Unknown crop: {crop}")
    potential = crops[crop]["potential_yield_kg_ha"]
    suitability = max(0.0, min(1.0, suitability))
    expected_loss = max(0.0, min(1.0, expected_loss))
    return potential * suitability * (1.0 - expected_loss)
