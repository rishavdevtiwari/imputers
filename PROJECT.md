# 🌱 Crop Profit & Risk Advisor — Project Detail

> *"We tell Nepali farmers not just what grows — but what survives, what pays, how likely it is to pay, and why."*

---

## 1. Event context

| | |
|---|---|
| **Hackathon** | DataForGood Nepal 2026 — Analytics for Society |
| **Cluster** | 3 — Economy, Livelihoods & Opportunity |
| **Track** | T8 — Agriculture, Food Security & Rural Livelihoods |
| **Format** | 10-hour sprint → working artifact + 4-minute pitch |
| **Repo** | `rishavdevtiwari/imputers` |

---

## 2. The problem & persona

### Persona — a real human
**Suchit Ratna Bajracharya, 45**, a smallholder farmer in **Dhading district (Hill zone)** with
~0.5 hectare of terraced, rain-fed land, two hours off the Prithvi Highway. He owns a basic
Android phone, has limited literacy, takes a yearly seed-and-fertilizer loan from his village
cooperative, and has no access to agronomic advice. He represents the ~2 million smallholder
households in Nepal who plant by **tradition, not data**.

### The lost season (origin story)
Last year, Suchit planted potatoes out of habit. The monsoon arrived three weeks late, and the
resulting drought stunted his crop. A sudden humid spell then triggered an outbreak of late
blight, and he lost more than half of his harvest. Because his entire plot was tied up with the
failing crop, he missed the window to replant anything else — losing the whole season and
falling deeper into debt with his cooperative. As he put it: *"I wasn't stubborn. I just didn't
know."*

### The recurring decision
**Every planting season:** which single crop to plant on his plot — one that suits his soil and
the expected weather, can withstand likely drought and disease, and will still sell at a profit.

### Problem statement (one sentence)
*Smallholder farmers like Suchit choose crops by tradition without knowing which crop their soil
and climate suit, how likely it is to fail to drought or disease, or whether it will sell at a
profit — so they risk total crop loss and market losses every season.*

---

## 3. Solution overview

A data-driven advisor that, given a **district + season**, fuses **weather + soil + crop science +
disease risk + market prices** to recommend **which crop to plant**, returning for each candidate:

- **Estimated yield** (kg/ha)
- **Profit/loss range** (NPR/ha)
- **Probability of profit** — P(profit)
- **Crop-failure risk** — combined disease + drought/frost/heat
- **Top disease threats** with expected loss
- A plain-language **"why this crop"** explanation

The headline differentiator: most crop recommenders stop at *"what grows."* This one adds
**survival risk** and **profitability** — the two things that actually decide whether Suchit
keeps his land or sinks into debt.

---

## 4. How it works (decision flow)

```
location + season
   │
   ▼
1. WEATHER   → seasonal temp, rainfall, humidity            (NASA POWER)
   │
   ▼
2. SOIL      → pH, nitrogen, texture                        (SoilGrids/NSSRC)
   │
   ▼
3. SUITABILITY → which crops fit these conditions           (RF classifier)
   │
   ▼
3.5 DISEASE & STRESS RISK → expected yield loss + failure   (disease KB + thresholds)
   │
   ▼
4. YIELD     → potential × suitability × (1 − loss)         (coherent estimate)
   │
   ▼
5. ECONOMICS → yield × price − cost → profit, ROI, volatility (Kalimati prices)
   │
   ▼
6. DECISION  → Monte Carlo → P(profit) + failure risk,
               rank by risk-adjusted profit, baseline switch-value, explain
   │
   ▼
Ranked crop cards (Streamlit)
```

---

## 5. Architecture (layered, data-only)

| Layer | Module | Responsibility |
|-------|--------|----------------|
| 1 Ingest | `src/ingest.py` | Fetch + cache NASA POWER, SoilGrids, MoALD/FAOSTAT, Kalimati |
| 2 Quality | `src/quality.py` | Clean, validate, impute, **score DQI** (project core) |
| 3 Features | `src/features.py` | Fuse sources by location + season + crop |
| 4a Suitability | `src/suitability.py` | RF classifier: conditions → suitable crops |
| 4b Disease/stress | `src/disease_risk.py` | Favorability → expected yield loss + failure risk |
| 4c Yield | `src/yield_est.py` | `potential × suitability × (1 − loss)` |
| 4d Economics | `src/economics.py` | profit = yield × price − cost; ROI |
| 5 Decision | `src/decision.py` | Monte Carlo → P(profit), ranking, baseline, explainer |
| 6 App | `app.py` | Streamlit crop cards + DQI panel |

A **data-quality layer wraps every source**, making the `imputers` identity (cleaning +
imputation + scoring) the project's visible strength.

---

## 6. Data sources

| Source | Use | Access | License note | Synthetic fallback |
|--------|-----|--------|--------------|--------------------|
| **NASA POWER** | weather / temperature | Free API (`community=AG`) | Open | seasonal sinusoid + monsoon gamma rainfall |
| **ISRIC SoilGrids** | soil pH, N, texture | Free REST API by lat/lon | CC-BY 4.0 | sample pH ~ N(6.0, 0.7), N/texture by zone |
| **Crop Recommendation dataset** | train suitability classifier | Kaggle CSV | Open | already complete; relabel to Nepal crops |
| **MoALD + FAOSTAT** | district yields | Public reports / CSV | Open (verify) | yield ~ suitability × zone mean |
| **Kalimati Market** | wholesale prices | Public daily feed | Open | recent price ± seasonal noise |
| **CABI / FAO / NARC** | disease knowledge base | `data/disease_kb.csv` | Cited literature | documented agronomic ranges |

> All datasets must be **cited with their licence** in the final pitch (hackathon rule).
> Any synthetic data must be **clearly labelled** and its generation method documented.

---

## 7. Data Quality Index (DQI)

```
DQI = 100 × (0.25·Completeness + 0.25·Validity + 0.25·Consistency + 0.25·Uniqueness)
```

| Metric | Definition |
|--------|------------|
| **Completeness** | `1 − missing cells / total cells` |
| **Validity** | fraction of values within configured ranges (pH 3–9, etc.) |
| **Consistency** | `1 − cross-field rule violations / checks` |
| **Uniqueness** | `1 − duplicate rows / total rows` |

Computed **before and after** cleaning per source → written to `reports/dqi.json`. Weights are
configurable in `config.yaml`; raise *Validity* for sensor-like data, *Uniqueness* for scraped data.

---

## 8. Disease & crop-failure risk module

Given site conditions, for each crop:

- **Disease Favorability Index (DFI)** ∈ [0, 1] — how closely temp / humidity / rainfall / soil-pH
  match a disease's ideal range (late blight may use the citable **Hutton Criteria** trigger).
- **Expected yield loss (%)** = `DFI × published max loss`.
- **Abiotic stress** — drought (season rainfall vs crop need), frost, heat thresholds.
- **Combined expected loss** = `1 − Π(1 − loss_i)` across disease + stress factors.
- Feeds the yield estimate and the Monte Carlo **crop-failure risk**.

Knowledge base: `data/disease_kb.csv` — 15 diseases across 9 crops (e.g., potato/tomato late
blight, rice blast, wheat stripe rust, cauliflower clubroot) with favorable conditions, soil
factors, and max yield-loss figures.

> **Honesty:** this is a *favorability proxy*, not field surveillance. It ignores resistant
> varieties and fungicide/irrigation use — stated in every output.

---

## 9. Decision engine

- **Monte Carlo** (configurable iterations) samples price (around the seasonal mean with
  historical volatility) and a Bernoulli failure event → distribution of profit.
- Returns **mean profit, P10/P90 band, P(profit), and crop-failure risk**.
- **Ranking:** risk-adjusted profit = `profit / (1 + failure_risk)`.
- **Baseline:** compares the recommendation against the persona's default crop (potato) →
  *"switching could earn ~NPR N more/ha."*
- **Explainer:** plain-language reasons (pH ✓, temp ✓, low disease pressure, good price).

---

## 10. Example output (what Suchit sees)

> 🥔 **Potato** — Suitability 0.78 | Est. profit +NPR 90k *(if it survives)*
> ⚠ **Late blight: HIGH** (cool + wet + poor drainage) → up to 80% loss
> ⚠ **Drought risk: HIGH** this season
> **Crop-failure risk: 61%** → *not recommended this season*
>
> ✅ **Recommended: Cauliflower** — profit +NPR 120k | failure risk 18% | low disease pressure
> *"Suits your soil (pH 6.2), matches expected temperature, low disease risk, strong market price."*

---

## 11. Tech stack

- **Data/ML:** Python 3.11, pandas, NumPy, scikit-learn, (optional) DuckDB
- **Geo:** SoilGrids REST, (optional) GeoPandas for district joins
- **App:** Streamlit + Plotly
- **Config:** YAML (`config.yaml`) + CSV knowledge base (`data/disease_kb.csv`)

---

## 12. Repository structure

```
imputers/
├── config.yaml              # thresholds, DQI weights, crop params, stress/MC settings
├── data/
│   ├── raw/                 # cached sources (git-ignored)
│   ├── clean/               # post-DQI outputs (git-ignored)
│   └── disease_kb.csv       # crop × disease knowledge base
├── reports/dqi.json         # before/after quality scores
├── src/
│   ├── ingest.py            # Layer 1 — fetch + cache
│   ├── quality.py           # Layer 2 — clean / impute / DQI  ✅ implemented
│   ├── features.py          # Layer 3 — fusion
│   ├── suitability.py       # Layer 4a — classifier
│   ├── disease_risk.py      # Layer 4b — favorability + stress
│   ├── yield_est.py         # Layer 4c — coherent yield        ✅ implemented
│   ├── economics.py         # Layer 4d — profit / ROI          ✅ implemented
│   ├── decision.py          # Layer 5  — Monte Carlo / ranking ✅ implemented
│   └── pipeline.py          # orchestrator
├── app.py                   # Layer 6 — Streamlit
├── requirements.txt
└── README.md
```

---

## 13. 10-hour build plan

| Block | Focus |
|-------|-------|
| H1–2 | Fetch + cache all sources; cleaning + **DQI before/after** (the `imputers` moment) |
| H3–4 | Train suitability classifier; validate vs majority-class baseline |
| H5–6 | Disease/stress risk + coherent yield + economics |
| H7 | Fusion pipeline end-to-end for 2–3 contrasting districts |
| H8–9 | Streamlit app (pin/district → ranked profit cards + charts) |
| H10 | README + AI-tool disclosure + ethics/limits + 4-min pitch |

---

## 14. Evaluation (baseline comparison)

- Compare the recommended crop against the **default crop (potato)** across past price/weather
  years: *"recommended earns ~NPR N more/ha and cuts failure risk from X% to Y%."*
- Suitability classifier reported against a **majority-class baseline**.
- Honest framing: an estimate with uncertainty bands, not a guarantee.

---

## 15. Judging-criteria mapping

| Criterion (weight) | How this project scores |
|---|---|
| **Data Engine (30%)** | 5 messy sources fused + imputed + **DQI** scored + a real baseline backtest |
| **Working Artifact (20%)** | Farmer enters location → actionable money + risk answer, live on a laptop |
| **Local Relevance / Ethics / Honesty (20%)** | Nepal soil/weather/Kalimati data; explicit limitations section |
| **Problem & Persona (15%)** | Suchit — a specific farmer with a dated, costly decision |
| **Pitch & Demo (15%)** | *"What survives, what pays, how likely, and why"* + an honest profit range |

---

## 16. Ethics & limitations

- **Advisory only — not a yield guarantee**; all estimates carry uncertainty bands.
- Kalimati prices reflect the **Kathmandu market**, not local prices.
- Disease risk is a **favorability proxy**, not field surveillance; ignores resistant
  varieties and fungicide/irrigation use.
- Soil/weather data are **gridded** → not field-exact.
- Cost-of-cultivation figures are partly **indicative assumptions** (see `config.yaml`).
- **No personal or farmer data** is collected.

---

## 17. Future work (roadmap, not in scope for the sprint)

- Optional **soil-moisture + temperature sensors** for in-season drought/disease alerts
  (the planting decision itself is well served by data; sensors help *after* planting).
- Crop-rotation and multi-season planning.
- Nepali-language and voice/SMS interface for low-literacy farmers.
- Market-glut warning that connects one farmer's choice to district-wide planting trends.

---

## 18. AI-tool disclosure

Project scaffolding and documentation were assisted by an AI coding assistant, as permitted by
the hackathon rules. All datasets used must be cited with their licences in the final submission.
