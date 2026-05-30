"""Layer 5 - Decision engine.

Monte Carlo over price + yield + disease occurrence -> P(profit) and
crop-failure risk. Rank crops by risk-adjusted profit, compare against the
persona's default crop (baseline), and produce a plain-language explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import load_config


@dataclass
class CropVerdict:
    crop: str
    suitability: float
    est_yield_kg_ha: float
    profit_npr_ha: float          # expected (mean) profit
    profit_p10: float             # 10th percentile (downside)
    profit_p90: float             # 90th percentile (upside)
    prob_profit: float            # P(profit > 0)
    failure_risk: float           # P(yield loss > threshold)
    top_diseases: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def monte_carlo_profit(
    expected_yield: float,
    price_mean: float,
    price_vol: float,
    cost: float,
    failure_prob: float,
) -> dict[str, float]:
    """Simulate profit distribution; return mean, p10, p90, P(profit), failure.

    Each iteration: sample price (normal around mean, sd = vol*mean), sample a
    failure event (Bernoulli(failure_prob) -> yield collapses), compute profit.
    """
    mc = load_config()["monte_carlo"]
    rng = np.random.default_rng(mc["seed"])
    n = mc["iterations"]
    vol = max(price_vol, mc["price_volatility_floor"])

    prices = rng.normal(price_mean, vol * price_mean, n).clip(min=0)
    failed = rng.random(n) < failure_prob
    yields = np.where(failed, expected_yield * 0.1, expected_yield)
    profits = yields * prices - cost

    return {
        "mean": float(profits.mean()),
        "p10": float(np.percentile(profits, 10)),
        "p90": float(np.percentile(profits, 90)),
        "prob_profit": float((profits > 0).mean()),
    }


def rank_crops(verdicts: list[CropVerdict]) -> list[CropVerdict]:
    """Sort by risk-adjusted profit = profit / (1 + failure_risk)."""
    return sorted(
        verdicts,
        key=lambda v: v.profit_npr_ha / (1.0 + v.failure_risk),
        reverse=True,
    )


def switch_value(recommended: CropVerdict, baseline: CropVerdict) -> float:
    """NPR/ha gained by switching from the baseline crop to the recommendation."""
    return recommended.profit_npr_ha - baseline.profit_npr_ha


def explain(verdict: CropVerdict) -> list[str]:
    """Produce human-readable reasons (pH ok, temp ok, low disease, good price).

    TODO: derive from the feature vector + risk components.
    """
    return verdict.reasons
