# Drought-Risk Modeling — Analysis & Justification

This document explains, step by step, **what was done and why**, turning the
`data/raw/district_monthly_climate.csv` panel into a tuned, leakage-free model
that predicts **monsoon drought risk** *before planting* — the exact climate
failure behind the persona's lost potato season.

Reproduce everything from the repo root:
```bash
python3 src/drought_preprocess.py        # clean + engineer -> data/processed/drought_dataset.csv + reports/
python3 src/drought_model_selection.py   # tune + compare + select -> models/ + reports/model_comparison.json
python3 src/drought_predict.py --district Dhading --year 2018   # inference (uses the tuned threshold)
```

---

## Step 1 — Understand the dataset

`district_monthly_climate.csv` is a NASA POWER **monthly climate panel**:
29,016 rows × 24 columns = **62 districts × 39 years (1981-2019) × 12 months**, a
perfectly balanced panel of precipitation, humidity, temperatures, pressure and
wind (10 m & 50 m).

It is already pristine: 0 missing, 0 `-999` sentinels, 0 duplicates; every date
parses and matches `YEAR/MONTH`; all physical checks pass (`T_max ≥ T_min`,
`RH ∈ [0,100]`, `precip ≥ 0`). `PRECTOT` is strongly right-skewed; temperatures
are heavily collinear (`T2M ≈ TS ≈ T2M_MIN ≈ T2M_MAX`, r > 0.95).

So the value is **not** in scrubbing dirt — it is in **defining a useful target**
and **engineering leakage-free, predictive features**.

---

## Step 2 — Define the target (leakage-free)

The file has no crop/yield label — it is climate only. We define a binary
**monsoon (Jun-Sep) drought** label per district-year, because the monsoon
delivers ~80% of Nepal's rain and *is* the persona's disaster.

- Aggregate monthly → **monsoon precipitation total** per (district, year).
- Standardize **per district** into an SPI-like z-score; **drought = z < −0.8**
  (driest ≈ 1 year in 5).
- The per-district mean/std use **training years only (≤ 2009)**, so future
  statistics never leak into labels. Resulting drought rate ≈ 25%.

---

## Step 3 — Clean (DQI before/after)

Validated physical ranges/consistency (all pass) and dropped **9 redundant
collinear columns** (`TS`, `T2MWET`, all `WS50M*`, `WS10M_MAX/MIN/RANGE`),
keeping one representative per concept. A **Data Quality Index** is scored
before/after: **100 → 100** (the data was already clean; the measurable gain is
reduced dimensionality and multicollinearity, not a higher score).

---

## Step 4 — Feature engineering (the main upgrade)

Monthly rows can't be modeled for a once-a-year decision, so we aggregate to
**one row per (district, year)** using **antecedent-only** features — everything
knowable *before* the monsoon starts (no look-ahead leakage). On top of the raw
seasonal aggregates we add **15 engineered features**, all leakage-free:

| Group | Features | Why |
|---|---|---|
| **Per-district anomaly z-scores** (train-only mean/std) | `mam_PRECTOT_z`, `mam_RH2M_z`, `mam_QV2M_z`, `mam_T2M_z`, `mam_T2M_MAX_z`, `mam_T2M_RANGE_z`, `mam_PS_z`, `mam_WS10M_z`, `djf_PRECTOT_z`, `djf_RH2M_z`, `djf_T2M_z` | Removing each district's baseline exposes the *deviation* that precedes drought. Raw temperatures are ~0-correlated with drought; their **anomalies are not** (e.g. `mam_T2M_MAX` r≈0.02 → `mam_T2M_MAX_z` r≈0.15). |
| **Agronomic dryness** | `mam_vpd` (vapour-pressure deficit, Tetens), `mam_djf_precip_ratio` | VPD = atmospheric "thirst"; a standard drought-stress indicator built only from antecedent vars. |
| **Multi-year persistence** | `prev2_monsoon_z` (lag-2), `roll3_monsoon_z` (trailing 3-yr mean) | Droughts **cluster across years** — the trailing 3-year monsoon mean is among the strongest single signals (r≈0.36). |

Plus the originals: pre-monsoon (MAM) and winter (DJF) aggregates, previous-year
monsoon (`prev_monsoon_precip/z`), and location (`LAT/LON`). **Total: 31
features.** Monsoon-period variables (circular with the label), `YEAR`, and the
`DISTRICT` name are excluded on purpose. → **2,356 modelable district-years.**

> Anomaly stats and lag features use **training-period / past-only** values, so
> no future information leaks into any feature.

---

## Step 5 — Evaluation design: forward-chaining cross-validation

A drought tool is used to **predict the next season from the past**, so the
honest estimate of skill is **forward-chaining (expanding-window) CV**: for each
cutoff year *Y*, **train on all district-years ≤ Y and predict the next 2 years**.
We pool every out-of-fold (future) prediction and score once. This sweeps recent
years (validation 2000-2019, 10 folds) and matches deployment.

**Why not a single 2009 split?** It is unstable here: the 1982-2009 decades carry
a *weak* pre-monsoon signal (so within-train CV rewards a near-constant,
over-regularized model), while the 2010s — drifted against the 1982-2009
climatology — look *unusually separable*. We still report a **2010+ holdout**
(train < 2010) for transparency, but treat it as optimistic.

- **Tuning metric:** pooled forward-chaining **PR-AUC** (average precision) — the
  right metric for an imbalanced (~25%) positive class.
- **Hyperparameter tuning:** randomized search per model whose objective *is* the
  pooled forward-chaining PR-AUC (`ParameterSampler`, 20-25 samples/model).
- **Selection:** highest forward-chaining PR-AUC.

---

## Step 6 — Baselines + tuned models, compared

`fwdPR/fwdROC` = pooled forward-chaining; `h*` = 2010+ holdout (0.5 threshold).

| model | fwdPR | fwdROC | holdout PR | holdout ROC | h-F1 | h-recall | h-prec |
|---|---|---|---|---|---|---|---|
| Baseline · MostFrequent | – | – | 0.200 | 0.500 | 0.000 | 0.000 | 0.000 |
| Baseline · Stratified | – | – | 0.205 | 0.515 | 0.252 | 0.298 | 0.218 |
| Baseline · Persistence | – | – | 0.660 | 0.755 | 0.663 | 0.500 | 0.984 |
| **LogisticRegression** | **0.506** | **0.791** | **0.864** | **0.917** | **0.728** | **0.573** | **1.000** |
| KNN | 0.241 | 0.614 | 0.540 | 0.748 | 0.047 | 0.024 | 1.000 |
| RandomForest | 0.239 | 0.677 | 0.698 | 0.856 | 0.000 | 0.000 | 0.000 |
| HistGradientBoosting | 0.256 | 0.687 | 0.717 | 0.868 | 0.078 | 0.040 | 1.000 |
| XGBoost | 0.250 | 0.690 | 0.622 | 0.798 | 0.078 | 0.040 | 1.000 |

**Selected: Logistic Regression** — `StandardScaler` + elastic-net LogReg
(`C ≈ 7.03`, `l1_ratio = 0.75`, `class_weight="balanced"`, `solver="saga"`).

**Why:**
1. **Highest forward-chaining PR-AUC (0.506) and ROC-AUC (0.791)** — by far
   (trees/KNN sit at 0.24-0.26 PR-AUC). Selected on forward CV, not a lucky split.
2. It **beats the strong persistence baseline** on the holdout (PR-AUC 0.86 vs
   0.66) — the engineered pre-monsoon climate adds real value over "last year was dry."
3. Tree models rank far worse on forward PR-AUC and (untuned-threshold) fire
   almost no warnings; LogReg with balanced weights is both the strongest and the
   simplest, interpretable choice for a farmer-facing tool.

---

## Step 7 — Decision-threshold tuning (what makes it usable)

The default 0.5 cutoff is wrong for imbalanced data — it gives low recall. For an
**early-warning** system a *missed* drought costs a farmer far more than a false
alarm, so the operating point is set by a **recall target**: the most precise
threshold that still catches **≥ 60% of droughts** on the pooled forward-OOF.

- **Operating threshold = 0.190** (persisted in `models/drought_threshold.joblib`).
- Forward-OOF @0.5: recall 0.374, precision 0.445. Forward-OOF @0.190: **recall
  0.604**, precision 0.348.
- (We use a recall target rather than F-beta because the forward precision
  *plateaus* ~0.35 across a wide recall band, where F-beta would collapse to a
  trivially low threshold. Youden's J = 0.042 and the F1-optimal = 0.051 are also
  recorded.)

The Streamlit dashboard defaults to this threshold and exposes a slider so a
risk-averse user can trade recall vs. precision.

---

## Step 8 — What drives the prediction (explainability)

Model-agnostic **permutation importance** (PR-AUC drop when a feature is shuffled,
on the 2010+ holdout) — top drivers:

`mam_RH2M_z` (0.59), `mam_T2M_z` (0.58), `mam_PRECTOT` (0.51), `djf_T2M` (0.49),
`mam_PS` (0.40), `mam_T2M_MAX_z` (0.31), `mam_vpd` (0.27).

A **dry/warm pre-monsoon anomaly** (low humidity, high temperature, high VPD) is
the dominant precursor of a drought monsoon — agronomically sensible, and the
**engineered anomaly features are the most important inputs**, validating Step 4.

**Saved artifacts:** `models/drought_model.joblib`, `models/drought_features.joblib`,
`models/drought_threshold.joblib`, `reports/model_comparison.json`.

---

## Honest limitations

- **Forward-chaining PR-AUC ≈ 0.51 / ROC ≈ 0.79 is the realistic skill.** The
  2010+ holdout (PR-AUC 0.86) is optimistic — recent droughts are very separable
  against the 1982-2009 climatology, so do not over-read it.
- At a high-recall operating point, forward precision is ~0.35 (≈ 1 in 3 warnings
  is a true drought, vs. a 25% base rate) — seasonal drought is genuinely hard to
  forecast a season ahead. This is decision support, **not** a guarantee.
- Coefficients/importances are **directional**; multicollinearity among climate
  vars means they explain intuition, not strict causation.
- Drought is defined statistically (SPI-like), not from on-the-ground impact records.

---

## How it fits the project

This is the **drought-risk engine** of the Crop Profit & Risk Advisor: given a
district, it flags the chance of a dry monsoon **before planting**, feeding the
crop-failure-risk score that would have warned the persona away from a doomed
potato season.
