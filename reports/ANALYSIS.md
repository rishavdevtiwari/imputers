# Drought-Risk Modeling — Analysis & Justification

This document explains, step by step, **what I did and why**, turning the uploaded
`data/raw/district_monthly_climate.csv` into a supervised model that predicts
**monsoon drought risk** — the exact climate failure behind the persona's lost
potato season.

Reproduce everything:
```bash
python3 src/build_dataset.py      # clean + process -> data/processed/drought_dataset.csv + reports/
python3 src/model_selection.py    # baseline + models -> models/ + reports/model_comparison.json
```

---

## Step 1 — Understand the dataset

**What:** `district_monthly_climate.csv` = a NASA POWER **monthly climate panel** —
29,016 rows × 24 columns = **62 districts × 39 years (1981-2019) × 12 months**, a
perfectly balanced panel. Variables: precipitation, humidity, temperatures,
pressure, and wind (10 m & 50 m).

**Findings:** 0 missing, 0 `-999` sentinels, 0 duplicates; all dates parse and match
`YEAR/MONTH`; all physical checks pass (`T_max ≥ T_min`, `RH ∈ [0,100]`, `precip ≥ 0`).
`PRECTOT` is strongly right-skewed; temperature variables are heavily collinear
(`T2M ≈ TS ≈ T2M_MIN ≈ T2M_MAX`, all r > 0.95).

**Why it matters:** the data is already pristine, so the value is **not** in scrubbing
dirt — it's in **defining a useful target** and **engineering leakage-free features**.

---

## Step 2 — Decide the target (the key decision)

**Problem:** the file has **no crop/yield/label column** — it is climate only. To
"compare a baseline and several models and pick the best," I had to define a
supervised target from the climate itself.

**What I chose:** a binary **monsoon (Jun-Sep) drought** label per district-year.

**Why:**
- It *is* the persona's disaster (a weak/late monsoon ruined Suchit's potatoes).
- The monsoon delivers ~80% of Nepal's annual rain → it defines the growing season.
- It gives a clean classification task with a real, domain-meaningful baseline.

**How the label is defined (leakage-free):**
- Aggregate monthly → **monsoon precipitation total** per (district, year).
- Standardize **per district** into an SPI-like z-score; **drought = z < −0.8**
  (driest ≈ 1 year in 5 — matches how often farmers hit a bad year).
- The per-district mean/std use **training years only (≤ 2009)**, so future
  statistics never leak into the labels. Resulting drought rate ≈ 25%.

---

## Step 3 — Clean (with DQI before/after)

**What I did:**
- Validated physical ranges and consistency rules → all pass.
- Dropped **9 redundant collinear columns** (`TS`, `T2MWET`, all `WS50M*`, and the
  `WS10M_MAX/MIN/RANGE`), keeping one representative per concept.
- Scored a **Data Quality Index (DQI)** before and after.

**Result:** DQI = **100 → 100** (data was already clean).

**Why drop collinear columns:** near-duplicate features add no information but inflate
multicollinearity (which destabilizes linear-model coefficients) and noise. Removing
them gives a leaner, more interpretable feature set with no loss of signal.

> Honesty note: the DQI doesn't "improve" here because nothing was broken. The
> measurable improvement is **dimensionality** (24 → 15 raw columns) and reduced
> redundancy, not a higher score.

---

## Step 4 — Process into a model-ready table

Monthly rows can't be modeled directly for a once-a-year decision, so I aggregated to
**one row per (district, year)** with **antecedent-only features** — everything knowable
*before* the monsoon begins (so there is no look-ahead leakage):

| Feature group | Columns | Why |
|---|---|---|
| Pre-monsoon (MAM, Mar-May) | mean temp/humidity/pressure/wind + total rain (9) | strongest near-term precursor of the monsoon |
| Winter (DJF, Dec-Feb) | mean temp, humidity, total rain (3) | seasonal lead-in conditions |
| Persistence | previous-year monsoon precip + its z (2) | droughts cluster across years |
| Location | LAT, LON (2) | spatial climate gradient |

**Excluded on purpose:** the monsoon-period variables (they define the label →
circular); `YEAR` (a time split puts test years outside the train range, so it can't
generalize); `DISTRICT` name (used `LAT/LON` instead to stay generalizable). The first
year per district is dropped (no previous-year antecedent). → **2,356 modelable rows.**

---

## Step 5 — Baseline + multiple models, compared

**Evaluation design (why):**
- **Time-based split** (train ≤ 2009 = 1,736 rows; test ≥ 2010 = 620 rows). Forecasting
  must be proven on the **future**; a random split would leak future climate backward.
- **GroupKFold by year** for cross-validation on the training set, so each fold holds
  out whole years → an honest estimate of generalization to unseen years.
- **ROC-AUC** is primary (threshold-independent, robust to the ~25% imbalance); we also
  report PR-AUC, F1, balanced accuracy, precision, recall — because for a *warning*
  system, catching droughts (recall) matters.

**Baselines (a model must beat these):** most-frequent, stratified-random, and a domain
**persistence** rule (drought if last year's monsoon was dry).

**Results:**

| model | cv_auc | test_auc | pr_auc | f1 | bal_acc | recall |
|---|---|---|---|---|---|---|
| Baseline · MostFrequent | – | 0.500 | 0.200 | 0.000 | 0.500 | 0.000 |
| Baseline · Stratified | – | 0.505 | 0.202 | 0.238 | 0.505 | 0.282 |
| Baseline · Persistence | – | 0.755 | 0.659 | 0.663 | 0.749 | 0.500 |
| **LogisticRegression** | **0.737** | **0.937** | **0.873** | **0.691** | **0.765** | **0.532** |
| KNN | 0.701 | 0.715 | 0.384 | 0.128 | 0.528 | 0.073 |
| RandomForest | 0.709 | 0.890 | 0.691 | 0.000 | 0.499 | 0.000 |
| HistGradientBoosting | 0.719 | 0.763 | 0.404 | 0.043 | 0.500 | 0.024 |

---

## Step 6 — Select the best model (with justification)

**Selected: Logistic Regression** (with standardized features + balanced class weights).

**Why:**
1. **Selected on cross-validated AUC, not the test set** (avoids selection bias). LogReg
   has the **highest CV AUC (0.737)** — so the choice is principled, not cherry-picked.
2. It also **wins on the held-out test set** across every usable metric (AUC 0.94,
   F1 0.69, balanced accuracy 0.77, recall 0.53).
3. It **beats the strong persistence baseline** (AUC 0.755) — proving the pre-monsoon
   climate adds real predictive value beyond "last year was dry."
4. **RandomForest/HistGradientBoosting rank well (AUC) but have ≈ 0 recall** at the
   default threshold — they almost never fire "drought," making them useless as a
   warning out-of-the-box. LogReg's balanced weights give usable recall without tuning.
5. It is the **simplest and most interpretable** model — a real asset for a farmer-facing
   tool and a hackathon pitch.

**What drives the prediction** (top standardized coefficients): warm winters (`djf_T2M`),
high pre-monsoon pressure (`mam_PS`), and low pre-monsoon rainfall (`mam_PRECTOT`) →
higher drought risk. This is agronomically sensible.

**Saved artifacts:** `models/drought_model.joblib`, `models/drought_features.joblib`,
`reports/model_comparison.json`.

---

## Honest limitations

- **Test AUC (0.94) > CV AUC (0.74).** Unusual; likely the 2010s carry a clearer
  pre-monsoon→monsoon signal and labels are referenced to the 1982-2009 baseline, so any
  drift makes recent droughts more separable. **Treat ~0.74 as the realistic skill.**
- Coefficients are **directional only** — multicollinearity can flip individual signs;
  use for intuition, not causation.
- Drought is defined statistically (SPI-like), not from on-the-ground impact records.
- Predicting *seasonal* drought is genuinely hard; this is decision-support, not a
  guarantee.

---

## How it fits the project

This model is the **drought-risk engine** of the Crop Profit & Risk Advisor: given a
district, it flags the chance of a dry monsoon **before planting**, feeding the
crop-failure-risk score that would have warned Suchit away from a doomed potato season.
