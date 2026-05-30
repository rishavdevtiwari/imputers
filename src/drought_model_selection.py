"""Train, tune, compare, and select the best monsoon DROUGHT-RISK model.

This is a farmer-facing EARLY-WARNING task: before planting, flag the chance
that the coming monsoon (Jun-Sep) will be dry. Every choice below targets the
skill of a *deployed forecast*: predict the NEXT season from the past.

Why FORWARD-CHAINING (expanding-window) cross-validation, POOLED?
----------------------------------------------------------------
For each cutoff year Y we TRAIN on all district-years <= Y and PREDICT the next
`horizon` years. We then POOL every out-of-fold (future) prediction and score
once. Folds always train on the past and test on the future, the validation set
sweeps recent years, and pooling avoids the high variance of scoring tiny
single-year folds. This is the honest 'predict next season' skill and it is
what we TUNE on, SELECT on, and tune the THRESHOLD on.

(A single 2009 split is misleading here: 1982-2009 carries a weak pre-monsoon
signal so within-train CV rewards a near-constant, over-regularised model, while
the drifted 2010s look unusually separable. GroupKFold PR-AUC barely separates
configs. Pooled forward-chaining is stable and deployment-faithful.)

Other principles
----------------
* HYPERPARAMETER TUNING: randomized search whose objective IS the pooled
  forward-chaining PR-AUC (average precision) — the right metric for an
  imbalanced (~25%) positive class.
* DECISION-THRESHOLD TUNING on the pooled forward-OOF: 0.5 gives ~0 recall on
  imbalanced data. The operating point requires a minimum sensitivity (catch
  >= TARGET_RECALL of droughts) and is then as precise as possible — because a
  MISSED drought costs a farmer more than a false alarm, and because F-beta
  collapses on this flat precision plateau. The threshold is persisted.
* BASELINES (most-frequent, stratified, domain persistence) set the bar.
* A 2010+ HOLDOUT (trained on <2010) is reported per model for transparency and
  used for model-agnostic permutation importance.

Run:  python3 src/drought_model_selection.py     (from repo root)
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")  # silence cosmetic core-count warning

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "drought_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"

HOLDOUT_FROM_YEAR = 2010      # secondary single-split holdout (train < this)
DROUGHT_Z = -0.8
SEED = 42
TARGET_RECALL = 0.60          # operating point: a warning must catch >= 60% of droughts
CV_HORIZON = 2                # forward-chaining: validate the next N years
CV_MIN_TRAIN_YEARS = 18       # first validation year ~ 2000 (enough history)

NON_FEATURES = {"DISTRICT", "YEAR", "monsoon_precip", "monsoon_z", "drought"}
TARGET = "drought"


def get_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


# --------------------------------------------------------------------------- #
# Forward-chaining (expanding-window) splitter by YEAR
# --------------------------------------------------------------------------- #
class ExpandingYearSplit:
    """Fold k trains on YEAR <= cutoff_k and validates the next `horizon` years.
    Cutoffs step by `horizon`, starting once `min_train_years` of history exist."""

    def __init__(self, horizon=CV_HORIZON, min_train_years=CV_MIN_TRAIN_YEARS):
        self.horizon = horizon
        self.min_train_years = min_train_years

    def folds(self, years: np.ndarray):
        years = np.asarray(years)
        uniq = np.array(sorted(pd.unique(years)))
        last = uniq.max()
        cutoff = uniq[self.min_train_years - 1]
        out = []
        while cutoff < last:
            val_years = [cutoff + k for k in range(1, self.horizon + 1)]
            val_mask = np.isin(years, val_years)
            tr_mask = years <= cutoff
            if val_mask.any() and tr_mask.any():
                out.append((np.where(tr_mask)[0], np.where(val_mask)[0]))
            cutoff += self.horizon
        return out


def forward_oof(estimator, X, y, years, splitter) -> tuple[np.ndarray, np.ndarray]:
    """Pool the next-year predictions across expanding-window folds (one model
    per fold, trained only on that fold's past)."""
    yt, ys = [], []
    for tr_idx, va_idx in splitter.folds(years):
        m = clone(estimator)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        ys.append(m.predict_proba(X.iloc[va_idx])[:, 1])
        yt.append(y.iloc[va_idx].to_numpy())
    return np.concatenate(yt), np.concatenate(ys)


# --------------------------------------------------------------------------- #
# Candidate models + hyperparameter search spaces
# --------------------------------------------------------------------------- #
def model_specs(pos_weight: float) -> dict:
    """name -> (estimator, param_distributions, n_iter)."""
    return {
        "LogisticRegression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, solver="saga",
                                                 class_weight="balanced",
                                                 random_state=SEED))]),
            {"clf__C": loguniform(1e-2, 1e1),
             "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]},
            25,
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 15, 21, 31, 41],
             "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
            24,
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 300, 400],
             "max_depth": [None, 6, 10, 16],
             "min_samples_leaf": [1, 2, 4, 8],
             "max_features": ["sqrt", "log2", 0.5],
             "class_weight": ["balanced", "balanced_subsample"]},
            20,
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
            {"learning_rate": loguniform(0.01, 0.3),
             "max_depth": [None, 3, 5, 7],
             "max_leaf_nodes": [15, 31, 63],
             "min_samples_leaf": [10, 20, 40],
             "l2_regularization": [0.0, 0.1, 1.0],
             "max_iter": [200, 400, 600]},
            20,
        ),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", scale_pos_weight=pos_weight,
                          random_state=SEED, n_jobs=-1),
            {"n_estimators": [200, 400, 600],
             "max_depth": [2, 3, 4, 5],
             "learning_rate": loguniform(0.01, 0.3),
             "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.7, 0.8, 1.0],
             "min_child_weight": [1, 3, 5],
             "gamma": [0.0, 0.5, 1.0],
             "reg_lambda": [1.0, 5.0]},
            20,
        ),
    }


def search_forward(estimator, dist, n_iter, X, y, years, splitter):
    """Randomized search whose objective is the POOLED forward-chaining PR-AUC.

    Returns (best_params, best_pr_auc, best_roc_auc, (oof_true, oof_score))."""
    best = None
    for params in ParameterSampler(dist, n_iter=n_iter, random_state=SEED):
        est = clone(estimator).set_params(**params)
        yt, ys = forward_oof(est, X, y, years, splitter)
        pr = average_precision_score(yt, ys)
        if best is None or pr > best[1]:
            best = (params, pr, float(roc_auc_score(yt, ys)), (yt, ys))
    return best


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def clf_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def fbeta_optimal_threshold(y_true, y_score, beta: float) -> float:
    """Threshold maximising F-beta (kept only for reporting / reference)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]
    denom = (beta**2 * prec) + rec
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denom > 0, (1 + beta**2) * prec * rec / denom, 0.0)
    return float(thr[int(np.argmax(fbeta))]) if len(thr) else 0.5


def threshold_for_recall(y_true, y_score, target: float) -> float:
    """Most precise (highest) threshold whose recall is still >= `target`.

    Standard way to set an early-warning operating point: require a minimum
    sensitivity (catch >= target fraction of droughts), then be as precise as
    possible. Robust to a flat precision plateau, where F-beta would collapse to
    the lowest threshold."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    rec_t = rec[:-1]                       # align recall with thresholds
    ok = np.where(rec_t >= target)[0]
    if len(ok) == 0 or len(thr) == 0:
        return float(thr.min()) if len(thr) else 0.5
    return float(thr[ok[np.argmax(thr[ok])]])


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))])


def metrics_at(y_true, y_score, thr) -> dict:
    return clf_metrics(y_true, (y_score >= thr).astype(int), y_score)


# --------------------------------------------------------------------------- #
def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA).sort_values(["YEAR", "DISTRICT"]).reset_index(drop=True)
    features = get_features(df)
    X, y, years = df[features], df[TARGET], df["YEAR"].to_numpy()
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    splitter = ExpandingYearSplit()
    folds = splitter.folds(years)
    val_years = sorted({int(years[va].min()) for _, va in folds})
    print(f"features={len(features)} | rows={len(df)} | drought rate={y.mean():.0%} | "
          f"scale_pos_weight={pos_weight:.2f}")
    print(f"forward-chaining CV: {len(folds)} folds, validating years "
          f"{min(val_years)}-{int(years.max())} (horizon={CV_HORIZON})\n")

    tr_mask = years < HOLDOUT_FROM_YEAR
    Xh_tr, yh_tr = X[tr_mask], y[tr_mask]
    Xh_te, yh_te = X[~tr_mask], y[~tr_mask]

    results: dict[str, dict] = {}
    best_estimators: dict[str, object] = {}
    best_params: dict[str, dict] = {}
    oof_cache: dict[str, tuple] = {}

    # ---------- Baselines (reported on the 2010+ holdout) ----------
    for name, strat in [("Baseline_MostFrequent", "most_frequent"),
                        ("Baseline_Stratified", "stratified")]:
        d = DummyClassifier(strategy=strat, random_state=SEED).fit(Xh_tr, yh_tr)
        results[name] = {"fwd_pr_auc": None, "fwd_roc_auc": None,
                         "holdout": clf_metrics(yh_te, d.predict(Xh_te),
                                                d.predict_proba(Xh_te)[:, 1])}
    pers_pred = (Xh_te["prev_monsoon_z"] < DROUGHT_Z).astype(int).to_numpy()
    pers_score = (-Xh_te["prev_monsoon_z"]).to_numpy()
    results["Baseline_Persistence"] = {"fwd_pr_auc": None, "fwd_roc_auc": None,
                                       "holdout": clf_metrics(yh_te, pers_pred, pers_score)}

    # ---------- Tune each candidate against pooled forward-chaining PR-AUC ----
    for name, (estimator, dist, n_iter) in model_specs(pos_weight).items():
        print(f"tuning {name} ({n_iter} samples x {len(folds)} forward folds) ...",
              flush=True)
        params, fwd_pr, fwd_roc, oof = search_forward(estimator, dist, n_iter,
                                                      X, y, years, splitter)
        tuned = clone(estimator).set_params(**params)
        best_params[name] = {k: (round(v, 5) if isinstance(v, float) else v)
                             for k, v in params.items()}
        oof_cache[name] = oof
        hmodel = clone(tuned).fit(Xh_tr, yh_tr)
        hscore = hmodel.predict_proba(Xh_te)[:, 1]
        results[name] = {
            "fwd_pr_auc": round(fwd_pr, 4), "fwd_roc_auc": round(fwd_roc, 4),
            "holdout": clf_metrics(yh_te, hmodel.predict(Xh_te), hscore),
        }
        best_estimators[name] = tuned
        print(f"   fwd_pr_auc={fwd_pr:.4f} fwd_roc_auc={fwd_roc:.4f}  "
              f"best={best_params[name]}")

    # ---------- Select best by POOLED FORWARD-CHAINING PR-AUC ----------
    best = max(best_estimators, key=lambda n: results[n]["fwd_pr_auc"])
    best_est = best_estimators[best]
    print(f"\nSELECTED: {best} (highest forward PR-AUC = {results[best]['fwd_pr_auc']:.4f})")

    # ---------- Tune decision threshold on the selected model's forward-OOF ----
    # Operating point for an early-warning system: catch >= TARGET_RECALL of
    # droughts, then be as precise as possible. (F-beta would collapse to the
    # lowest threshold on this flat precision plateau, so we use a recall target.)
    oof_true, oof_score = oof_cache[best]
    thr = threshold_for_recall(oof_true, oof_score, TARGET_RECALL)
    thr_youden = youden_threshold(oof_true, oof_score)
    thr_f1 = fbeta_optimal_threshold(oof_true, oof_score, beta=1.0)
    fwd_default = metrics_at(oof_true, oof_score, 0.5)
    fwd_tuned = metrics_at(oof_true, oof_score, thr)
    print(f"tuned threshold (recall>={TARGET_RECALL:.0%} on pooled forward-OOF) = "
          f"{thr:.3f}  [youden={thr_youden:.3f}, F1-opt={thr_f1:.3f}]")
    print(f"forward-OOF @0.5  : recall={fwd_default['recall']:.3f} "
          f"precision={fwd_default['precision']:.3f} f1={fwd_default['f1']:.3f}")
    print(f"forward-OOF @{thr:.2f}: recall={fwd_tuned['recall']:.3f} "
          f"precision={fwd_tuned['precision']:.3f} f1={fwd_tuned['f1']:.3f}")

    # ---------- Refit best on ALL data and persist (model + features + threshold)
    final_model = clone(best_est).fit(X, y)
    joblib.dump(final_model, MODEL_DIR / "drought_model.joblib")
    joblib.dump(features, MODEL_DIR / "drought_features.joblib")
    joblib.dump({"threshold": round(thr, 4), "target_recall": TARGET_RECALL,
                 "youden_threshold": round(thr_youden, 4),
                 "f1_threshold": round(thr_f1, 4),
                 "tuned_on": "pooled forward out-of-fold predictions"},
                MODEL_DIR / "drought_threshold.joblib")

    # ---------- Explainability: permutation importance on the 2010+ holdout ----
    perm_model = clone(best_est).fit(Xh_tr, yh_tr)
    perm = permutation_importance(perm_model, Xh_te, yh_te, scoring="average_precision",
                                  n_repeats=15, random_state=SEED, n_jobs=-1)
    importance = dict(sorted(zip(features, perm.importances_mean.round(4)),
                             key=lambda x: -x[1]))

    report = {
        "task": "monsoon_drought_early_warning",
        "n_features": len(features), "features": features,
        "evaluation": {
            "primary": "pooled forward-chaining (expanding-window) CV by YEAR",
            "cv_folds": len(folds), "cv_horizon_years": CV_HORIZON,
            "cv_validation_years": [min(val_years), int(years.max())],
            "tuning_objective": "pooled forward-chaining PR-AUC (average precision)",
            "secondary_holdout": {"train_before_year": HOLDOUT_FROM_YEAR,
                                  "holdout_rows": int((~tr_mask).sum())},
        },
        "results": results,
        "best_params": best_params,
        "selected_model": best,
        "selection_rule": "highest pooled forward-chaining PR-AUC; estimates real "
                          "'predict the next season' skill across many cutoffs and "
                          "is stable, unlike a single train/test split.",
        "operating_threshold": {
            "value": round(thr, 4), "policy": f"recall >= {TARGET_RECALL:.0%}",
            "tuned_on": "pooled forward out-of-fold predictions",
            "youden_threshold": round(thr_youden, 4),
            "f1_optimal_threshold": round(thr_f1, 4),
            "forward_oof_at_default_0.5": fwd_default,
            "forward_oof_at_tuned_threshold": fwd_tuned,
            "rationale": "A missed drought costs a farmer more than a false alarm, so "
                         f"the operating point requires recall >= {TARGET_RECALL:.0%} "
                         "(then maximises precision). 0.5 gives near-zero recall on "
                         "imbalanced data, and F-beta collapses on the flat precision "
                         "plateau, so a recall target is used instead.",
        },
        "permutation_importance_holdout": importance,
        "notes": [
            "PRIMARY metric is pooled forward-chaining PR-AUC: it answers 'how well "
            "does this predict next season?' and sweeps recent years, matching "
            "deployment. Treat it as the realistic skill.",
            "A single 2009 split is unstable here (weak 1982-2009 signal vs drifted, "
            "very separable 2010s); the 2010+ holdout numbers are reported but are "
            "optimistic. Forward-chaining averages over many cutoffs.",
            "The 0.5 threshold is a poor operating point on imbalanced data; the tuned "
            "threshold restores usable recall. Probabilities are unchanged - the "
            "dashboard slider lets users trade recall vs precision.",
            "Permutation importance (PR-AUC drop when a feature is shuffled) is "
            "model-agnostic and less biased than tree impurity importance.",
        ],
    }
    (REPORTS / "model_comparison.json").write_text(json.dumps(report, indent=2))

    # ---------- Console comparison table ----------
    print(f"\n{'model':22s} {'fwdPR':>6s} {'fwdROC':>6s} {'hPR':>6s} {'hROC':>6s} "
          f"{'hF1':>5s} {'hRec':>5s} {'hPrec':>5s}")
    print("-" * 70)
    for name, r in results.items():
        h = r["holdout"]
        fpr = f"{r['fwd_pr_auc']:.3f}" if r["fwd_pr_auc"] is not None else "  -  "
        frc = f"{r['fwd_roc_auc']:.3f}" if r["fwd_roc_auc"] is not None else "  -  "
        print(f"{name:22s} {fpr:>6s} {frc:>6s} {h['pr_auc']:>6.3f} "
              f"{h['roc_auc']:>6.3f} {h['f1']:>5.3f} {h['recall']:>5.3f} "
              f"{h['precision']:>5.3f}")
    print("-" * 70)
    print(f"SELECTED: {best}  ->  models/drought_model.joblib "
          f"(operating threshold={thr:.3f})")
    print("top features (perm. importance):",
          ", ".join(f"{k}={v}" for k, v in list(importance.items())[:6]))


if __name__ == "__main__":
    main()
