"""Orchestrator - runs ingest -> quality -> features -> analysis -> decision.

Usage:
    from src.pipeline import recommend
    verdicts = recommend(district="Dhading", season="summer")
"""
from __future__ import annotations

from .config import load_config
from .decision import CropVerdict, monte_carlo_profit, rank_crops
from .economics import compute_economics
from .yield_est import estimate_yield


def recommend(district: str, season: str | None = None) -> list[CropVerdict]:
    """End-to-end recommendation for a district + season.

    Steps:
      1. ingest: load cached weather/soil/yield/price (+ live SoilGrids by point)
      2. quality: clean + impute + DQI
      3. features: build SiteFeatures + seasonal price/volatility
      4. suitability: score each crop
      5. disease_risk: expected loss + failure probability per crop
      6. yield_est: coherent yield
      7. economics: profit/ROI
      8. decision: Monte Carlo -> rank -> baseline switch-value -> explain

    TODO: wire the modules together once their implementations land.
    """
    cfg = load_config()
    season = season or cfg["project"]["default_season"]
    raise NotImplementedError(
        "Wire ingest/quality/features/suitability/disease_risk here."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop Profit & Risk Advisor")
    parser.add_argument("--district", required=True)
    parser.add_argument("--season", default=None)
    args = parser.parse_args()

    for v in recommend(args.district, args.season):
        print(f"{v.crop:12s} profit={v.profit_npr_ha:>12,.0f}  "
              f"P(profit)={v.prob_profit:.0%}  failure={v.failure_risk:.0%}")
