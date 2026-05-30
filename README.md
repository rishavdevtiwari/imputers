# 🌱 Crop Profit & Risk Advisor

> *"We tell Nepali farmers not just what grows — but what survives, what pays, how likely it is to pay, and why."*

A data-driven advisor for Nepali smallholder farmers. Given a **district + season**,
it fuses weather, soil, crop science, disease risk, and market prices to recommend
**which crop to plant** — with an estimated yield, a profit/loss range, the probability
of profit, and the risk of crop failure.

Built for **DataForGood Nepal 2026** — Cluster 3, Track 8 (Agriculture, Food Security
& Rural Livelihoods).

---

## The problem (persona)

**Suchit Ratna Bajracharya, 45**, a smallholder in **Dhading (Hill zone)** with ~0.5 ha
of rain-fed land. Last year he planted potatoes out of habit. The monsoon arrived three
weeks late; drought stunted the crop, a humid spell triggered late blight, and he lost
over half his harvest. With his plot locked under the failing crop, he missed the window
to replant — losing the whole season and falling deeper into cooperative debt.

> *"I wasn't stubborn. I just didn't know."*

**The recurring decision:** which single crop to plant — one that suits his soil and the
expected weather, can withstand likely drought and disease, and will still sell at a profit.

---

## Architecture

```
DATA SOURCES  ->  1. INGEST/CACHE  ->  2. DATA QUALITY (DQI/impute)
   ->  3. FEATURE FUSION  ->  4. ANALYSIS  ->  5. DECISION  ->  6. STREAMLIT APP
```

| Layer | Module | Responsibility |
|-------|--------|----------------|
| 1 Ingest | `src/ingest.py` | Fetch + cache NASA POWER, SoilGrids, MoALD/FAOSTAT, Kalimati |
| 2 Quality | `src/quality.py` | Clean, validate, impute, **score DQI** (the project core) |
| 3 Features | `src/features.py` | Fuse sources by location + season + crop |
| 4a Suitability | `src/suitability.py` | RF classifier: conditions → suitable crops |
| 4b Disease/stress | `src/disease_risk.py` | Favorability → expected yield loss + failure risk |
| 4c Yield | `src/yield_est.py` | `potential × suitability × (1 − loss)` |
| 4d Economics | `src/economics.py` | profit = yield × price − cost; ROI |
| 5 Decision | `src/decision.py` | Monte Carlo → P(profit), ranking, baseline, explainer |
| 6 App | `app.py` | Streamlit crop cards + DQI panel |

---

## Data sources

| Source | Use | Access |
|--------|-----|--------|
| NASA POWER | weather/temperature | Free API (`community=AG`) |
| ISRIC SoilGrids | soil pH, N, texture | Free REST API by lat/lon |
| MoALD + FAOSTAT | district yields | Public reports / CSV |
| Kalimati Market | wholesale prices | Public daily feed |
| CABI / FAO / NARC | disease knowledge base | `data/disease_kb.csv` |

---

## Data Quality Index (DQI)

```
DQI = 100 × (0.25·Completeness + 0.25·Validity + 0.25·Consistency + 0.25·Uniqueness)
```

- **Completeness** = 1 − missing/total
- **Validity** = values within configured ranges
- **Consistency** = 1 − cross-field rule violations
- **Uniqueness** = 1 − duplicates/total

Computed **before and after** cleaning per source → `reports/dqi.json`.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

The pipeline modules are **scaffolded with signatures + docstrings**; fill in the
`NotImplementedError` stubs (start with `ingest.py`, `quality.py`, `features.py`,
`suitability.py`, `disease_risk.py`) to enable live recommendations. `quality.py`,
`yield_est.py`, `economics.py`, and `decision.py` already contain working logic.

CLI:
```bash
python -m src.pipeline --district Dhading --season summer
```

---

## Honesty & limitations

- Advisory only — **not a yield guarantee**; all estimates carry uncertainty bands.
- Kalimati prices reflect the **Kathmandu market**, not local prices.
- Disease risk is a **favorability proxy**, not field surveillance; it ignores
  resistant varieties and fungicide/irrigation use.
- Soil/weather data are **gridded** → not field-exact.
- Cost-of-cultivation figures are partly **indicative assumptions** (see `config.yaml`).
- No personal or farmer data is collected.

---

## Configuration

All thresholds, weights, crop parameters, and disease rules live in **`config.yaml`**
and **`data/disease_kb.csv`** — change behaviour without touching code.

## AI-tool disclosure

Project scaffolding assisted by an AI coding assistant (per hackathon rules). All
datasets must be cited with their licences in the final submission.
