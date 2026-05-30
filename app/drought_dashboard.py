"""Drought-Risk Dashboard — test the monsoon drought model interactively.

Run from the repo root:
    streamlit run app/drought_dashboard.py

Lets you pick a district and see, year by year, the model's predicted drought
probability vs. what actually happened — plus overall held-out test metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# make `src` importable when run via `streamlit run app/...`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.drought_predict import (artifacts_exist, load_artifacts,  # noqa: E402
                                 load_dataset, predict_district)

REPORTS = ROOT / "reports"
TEST_START_YEAR = 2010

st.set_page_config(page_title="Drought-Risk Dashboard", page_icon="🌧️", layout="wide")
st.title("🌧️ Monsoon Drought-Risk Dashboard")
st.caption("Predicting a weak monsoon (Jun-Sep) per district from pre-monsoon climate — "
           "the failure behind the persona's lost potato season.")

# ----------------------------------------------------------------------------- #
# Guard: artifacts must exist
# ----------------------------------------------------------------------------- #
if not artifacts_exist():
    st.error(
        "Model/data not found. Build them first from the repo root:\n\n"
        "```\npython3 src/drought_build_dataset.py\npython3 src/drought_model_selection.py\n```"
    )
    st.stop()


@st.cache_resource
def _load():
    model, features = load_artifacts()
    data = load_dataset()
    comp = {}
    fp = REPORTS / "model_comparison.json"
    if fp.exists():
        comp = json.loads(fp.read_text())
    return model, features, data, comp


model, features, data, comp = _load()

# ----------------------------------------------------------------------------- #
# Top metrics (from model_comparison.json)
# ----------------------------------------------------------------------------- #
sel = comp.get("selected_model", "—")
res = comp.get("results", {}).get(sel, {})
test = res.get("test", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", sel)
c2.metric("CV ROC-AUC", res.get("cv_roc_auc", "—"))
c3.metric("Test ROC-AUC", test.get("roc_auc", "—"))
c4.metric("Test recall", test.get("recall", "—"))

st.divider()

# ----------------------------------------------------------------------------- #
# Sidebar controls
# ----------------------------------------------------------------------------- #
districts = sorted(data["DISTRICT"].unique())
default_ix = districts.index("Dhading") if "Dhading" in districts else 0
with st.sidebar:
    st.header("Controls")
    district = st.selectbox("District", districts, index=default_ix)
    threshold = st.slider("Drought decision threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Probability above this is flagged as drought.")
    st.caption("Lower the threshold to catch more droughts (higher recall, "
               "more false alarms).")

# ----------------------------------------------------------------------------- #
# Per-district predictions
# ----------------------------------------------------------------------------- #
pred = predict_district(district, threshold=threshold, model=model,
                        features=features, data=data).sort_values("YEAR")

st.subheader(f"📍 {district}: predicted vs. actual, by year")

fig = go.Figure()
fig.add_bar(x=pred["YEAR"], y=pred["monsoon_precip"], name="Monsoon rain (mm)",
            marker_color="#9ecae1", yaxis="y")
fig.add_scatter(x=pred["YEAR"], y=pred["drought_prob"], name="Drought probability",
                mode="lines+markers", line=dict(color="#d62728"), yaxis="y2")
# mark actual drought years
dy = pred[pred["drought"] == 1]
fig.add_scatter(x=dy["YEAR"], y=dy["monsoon_precip"], name="Actual drought",
                mode="markers", marker=dict(color="black", size=11, symbol="x"),
                yaxis="y")
# threshold reference line drawn explicitly on the probability axis (y2)
if len(pred):
    fig.add_scatter(x=[pred["YEAR"].min(), pred["YEAR"].max()],
                    y=[threshold, threshold], mode="lines", name="threshold",
                    line=dict(color="#d62728", dash="dot"), yaxis="y2")
fig.update_layout(
    height=430, hovermode="x unified", legend=dict(orientation="h", y=1.12),
    yaxis=dict(title="Monsoon rain (mm)"),
    yaxis2=dict(title="Drought probability", overlaying="y", side="right",
                range=[0, 1]),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# per-district hit summary
correct = int((pred["drought"] == pred["drought_pred"]).sum())
st.write(f"**{correct}/{len(pred)}** years predicted correctly for {district} "
         f"(threshold = {threshold:.2f}).")

with st.expander("See the yearly table"):
    show = pred[["YEAR", "monsoon_precip", "monsoon_z", "drought",
                 "drought_pred", "drought_prob"]].rename(columns={
        "monsoon_precip": "rain_mm", "drought": "actual",
        "drought_pred": "predicted", "drought_prob": "probability"})
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# ----------------------------------------------------------------------------- #
# Overall held-out test performance (all districts, years >= 2010)
# ----------------------------------------------------------------------------- #
st.subheader(f"📊 Held-out test performance (all districts, {TEST_START_YEAR}+)")
left, right = st.columns(2)

with left:
    cm = test.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, index=["actual: no", "actual: drought"],
                             columns=["pred: no", "pred: drought"])
        st.write("**Confusion matrix** (default 0.5 threshold)")
        st.dataframe(cm_df, use_container_width=True)
    st.caption(f"F1={test.get('f1','—')} · balanced acc={test.get('balanced_acc','—')} "
               f"· PR-AUC={test.get('pr_auc','—')}")

with right:
    feats_imp = comp.get("top_features") or {}
    if feats_imp:
        imp = pd.Series(feats_imp).head(8).iloc[::-1]
        bar = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color="#2ca02c"))
        bar.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                          title="Top model features (coef./importance)")
        st.plotly_chart(bar, use_container_width=True)

with st.expander("ℹ️ How to read this / honest limits"):
    st.markdown(
        "- **Probability** = model's estimated chance of a drought monsoon, from "
        "pre-monsoon (winter + spring) climate only — knowable *before* planting.\n"
        "- Selected on **cross-validated AUC**; treat **~0.74** as realistic skill "
        "(test AUC is optimistic — see `reports/ANALYSIS.md`).\n"
        "- Decision support, **not** a guarantee. Drought is defined statistically "
        "(SPI-like), not from on-the-ground impact records."
    )

"""Drought-Risk Dashboard — test the monsoon drought model interactively.

Run from the repo root:
    streamlit run app/drought_dashboard.py

Lets you pick a district and see, year by year, the model's predicted drought
probability vs. what actually happened — plus overall held-out test metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# make `src` importable when run via `streamlit run app/...`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict_drought import (artifacts_exist, load_artifacts,  # noqa: E402
                                 load_dataset, predict_district)

REPORTS = ROOT / "reports"
TEST_START_YEAR = 2010

st.set_page_config(page_title="Drought-Risk Dashboard", page_icon="🌧️", layout="wide")
st.title("🌧️ Monsoon Drought-Risk Dashboard")
st.caption("Predicting a weak monsoon (Jun-Sep) per district from pre-monsoon climate — "
           "the failure behind the persona's lost potato season.")

# ----------------------------------------------------------------------------- #
# Guard: artifacts must exist
# ----------------------------------------------------------------------------- #
if not artifacts_exist():
    st.error(
        "Model/data not found. Build them first from the repo root:\n\n"
        "```\npython3 src/build_dataset.py\npython3 src/model_selection.py\n```"
    )
    st.stop()


@st.cache_resource
def _load():
    model, features = load_artifacts()
    data = load_dataset()
    comp = {}
    fp = REPORTS / "model_comparison.json"
    if fp.exists():
        comp = json.loads(fp.read_text())
    return model, features, data, comp


model, features, data, comp = _load()

# ----------------------------------------------------------------------------- #
# Top metrics (from model_comparison.json)
# ----------------------------------------------------------------------------- #
sel = comp.get("selected_model", "—")
res = comp.get("results", {}).get(sel, {})
test = res.get("test", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", sel)
c2.metric("CV ROC-AUC", res.get("cv_roc_auc", "—"))
c3.metric("Test ROC-AUC", test.get("roc_auc", "—"))
c4.metric("Test recall", test.get("recall", "—"))

st.divider()

# ----------------------------------------------------------------------------- #
# Sidebar controls
# ----------------------------------------------------------------------------- #
districts = sorted(data["DISTRICT"].unique())
default_ix = districts.index("Dhading") if "Dhading" in districts else 0
with st.sidebar:
    st.header("Controls")
    district = st.selectbox("District", districts, index=default_ix)
    threshold = st.slider("Drought decision threshold", 0.0, 1.0, 0.5, 0.05,
                          help="Probability above this is flagged as drought.")
    st.caption("Lower the threshold to catch more droughts (higher recall, "
               "more false alarms).")

# ----------------------------------------------------------------------------- #
# Per-district predictions
# ----------------------------------------------------------------------------- #
pred = predict_district(district, threshold=threshold, model=model,
                        features=features, data=data).sort_values("YEAR")

st.subheader(f"📍 {district}: predicted vs. actual, by year")

fig = go.Figure()
fig.add_bar(x=pred["YEAR"], y=pred["monsoon_precip"], name="Monsoon rain (mm)",
            marker_color="#9ecae1", yaxis="y")
fig.add_scatter(x=pred["YEAR"], y=pred["drought_prob"], name="Drought probability",
                mode="lines+markers", line=dict(color="#d62728"), yaxis="y2")
# mark actual drought years
dy = pred[pred["drought"] == 1]
fig.add_scatter(x=dy["YEAR"], y=dy["monsoon_precip"], name="Actual drought",
                mode="markers", marker=dict(color="black", size=11, symbol="x"),
                yaxis="y")
# threshold reference line drawn explicitly on the probability axis (y2)
if len(pred):
    fig.add_scatter(x=[pred["YEAR"].min(), pred["YEAR"].max()],
                    y=[threshold, threshold], mode="lines", name="threshold",
                    line=dict(color="#d62728", dash="dot"), yaxis="y2")
fig.update_layout(
    height=430, hovermode="x unified", legend=dict(orientation="h", y=1.12),
    yaxis=dict(title="Monsoon rain (mm)"),
    yaxis2=dict(title="Drought probability", overlaying="y", side="right",
                range=[0, 1]),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# per-district hit summary
correct = int((pred["drought"] == pred["drought_pred"]).sum())
st.write(f"**{correct}/{len(pred)}** years predicted correctly for {district} "
         f"(threshold = {threshold:.2f}).")

with st.expander("See the yearly table"):
    show = pred[["YEAR", "monsoon_precip", "monsoon_z", "drought",
                 "drought_pred", "drought_prob"]].rename(columns={
        "monsoon_precip": "rain_mm", "drought": "actual",
        "drought_pred": "predicted", "drought_prob": "probability"})
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# ----------------------------------------------------------------------------- #
# Overall held-out test performance (all districts, years >= 2010)
# ----------------------------------------------------------------------------- #
st.subheader(f"📊 Held-out test performance (all districts, {TEST_START_YEAR}+)")
left, right = st.columns(2)

with left:
    cm = test.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, index=["actual: no", "actual: drought"],
                             columns=["pred: no", "pred: drought"])
        st.write("**Confusion matrix** (default 0.5 threshold)")
        st.dataframe(cm_df, use_container_width=True)
    st.caption(f"F1={test.get('f1','—')} · balanced acc={test.get('balanced_acc','—')} "
               f"· PR-AUC={test.get('pr_auc','—')}")

with right:
    feats_imp = comp.get("top_features") or {}
    if feats_imp:
        imp = pd.Series(feats_imp).head(8).iloc[::-1]
        bar = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker_color="#2ca02c"))
        bar.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                          title="Top model features (coef./importance)")
        st.plotly_chart(bar, use_container_width=True)

with st.expander("ℹ️ How to read this / honest limits"):
    st.markdown(
        "- **Probability** = model's estimated chance of a drought monsoon, from "
        "pre-monsoon (winter + spring) climate only — knowable *before* planting.\n"
        "- Selected on **cross-validated AUC**; treat **~0.74** as realistic skill "
        "(test AUC is optimistic — see `reports/ANALYSIS.md`).\n"
        "- Decision support, **not** a guarantee. Drought is defined statistically "
        "(SPI-like), not from on-the-ground impact records."
    )
