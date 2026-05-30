"""Layer 4b - Disease & abiotic-stress risk.

Given site conditions, compute for each crop:
  - Disease Favorability Index (DFI) in [0, 1] per disease
  - Expected yield loss (%) = DFI * published max loss
  - Abiotic stress risk (drought / frost / heat)
These reduce the coherent yield estimate and feed the crop-failure probability.

NOTE: This is a favorability proxy, NOT field surveillance. It ignores
resistant varieties and fungicide/irrigation use. State this in outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import SiteFeatures


@dataclass
class DiseaseThreat:
    disease: str
    favorability: float      # DFI in [0, 1]
    expected_loss: float     # fraction of yield, [0, 1]


def disease_favorability(site: SiteFeatures, disease_row: pd.Series) -> float:
    """Score how favorable conditions are for one disease (0-1).

    Combine closeness of temp/humidity/rainfall/soil-pH to the disease's ideal
    ranges (e.g., product or weighted mean of per-factor membership scores).
    For late blight, optionally apply the Hutton Criteria as a hard trigger.

    TODO: implement membership scoring from disease_kb columns.
    """
    raise NotImplementedError


def crop_disease_threats(site: SiteFeatures, crop: str, kb: pd.DataFrame) -> list[DiseaseThreat]:
    """Return ranked DiseaseThreats for a crop at a site."""
    raise NotImplementedError


def abiotic_stress(site: SiteFeatures, crop: str) -> dict[str, float]:
    """Return {'drought':p, 'frost':p, 'heat':p} risk fractions for a crop.

    Drought: season rainfall vs crop need (config.stress.drought_rainfall_ratio).
    Frost/heat: min/max temp vs configured thresholds.
    TODO: implement using config['stress'].
    """
    raise NotImplementedError


def expected_total_loss(threats: list[DiseaseThreat], stress: dict[str, float]) -> float:
    """Combine disease + abiotic losses into a single expected-loss fraction.

    Use complementary multiplication so combined loss stays in [0, 1]:
        1 - prod(1 - loss_i)
    """
    survive = 1.0
    for t in threats:
        survive *= (1.0 - t.expected_loss)
    for p in stress.values():
        survive *= (1.0 - p)
    return 1.0 - survive
