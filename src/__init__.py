"""Crop Profit & Risk Advisor - source package.

Pipeline layers:
    ingest      -> fetch + cache raw data sources
    quality     -> clean, validate, impute, score (DQI)  [the `imputers` core]
    features    -> fuse sources into per-crop feature vectors
    suitability -> classify which crops suit soil+climate
    disease_risk-> disease favorability + abiotic stress -> expected yield loss
    yield_est   -> coherent yield = potential * suitability * (1 - loss)
    economics   -> profit/loss, ROI, price volatility
    decision    -> Monte Carlo, ranking, baseline comparison, explainer
    pipeline    -> orchestrates ingest -> decision
"""

__version__ = "0.1.0"
