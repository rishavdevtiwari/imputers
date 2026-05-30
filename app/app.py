"""
Unified Streamlit dashboard for the Krishi-AI system.

Combines two tools behind a single sidebar navigation:

  1. 🌾 Crop Recommendation System
       Offline-resilient NARC soil lookup + climate inputs -> ensemble crop pick.
  2. 🌧️ Monsoon Drought-Risk Dashboard
       Predicts a weak monsoon (Jun-Sep) per district from pre-monsoon climate.

Run from the repo root:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px  # noqa: F401  (kept for parity / future use)
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & import setup (cover both `predict` in src/ and `src.drought_predict`)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports"

for path in (str(PROJECT_ROOT), str(SRC_DIR), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from predict import CropRecommendationPredictor  # noqa: E402

# Drought inference module is optional at import time; guard so a missing/broken
# drought module never takes down the crop tool.
try:
    from src.drought_predict import (  # noqa: E402
        artifacts_exist as drought_artifacts_exist,
        load_artifacts as load_drought_artifacts,
        load_dataset as load_drought_dataset,
        predict_district as predict_drought_district,
    )

    DROUGHT_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - defensive import guard
    DROUGHT_IMPORT_ERROR = str(exc)

# --------------------------------------------------------------------------- #
# Page config (must be the first/only Streamlit call of its kind)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Krishi-AI Dashboard",
    page_icon="🌱",
    layout="wide",
)

TEST_START_YEAR = 2010


# ==============================================================================
# CROP RECOMMENDATION: data, helpers & page
# ==============================================================================

PROVINCE_MAP = {
    "Taplejung": "Koshi", "Sankhuwasabha": "Koshi", "Solukhumbu": "Koshi", "Okhaldhunga": "Koshi", "Khotang": "Koshi", "Bhojpur": "Koshi", "Dhankuta": "Koshi", "Terhathum": "Koshi", "Panchthar": "Koshi", "Ilam": "Koshi", "Jhapa": "Koshi", "Morang": "Koshi", "Sunsari": "Koshi", "Udayapur": "Koshi",
    "Saptari": "Madhesh", "Siraha": "Madhesh", "Dhanusha": "Madhesh", "Mahottari": "Madhesh", "Sarlahi": "Madhesh", "Rautahat": "Madhesh", "Bara": "Madhesh", "Parsa": "Madhesh",
    "Sindhuli": "Bagmati", "Ramechhap": "Bagmati", "Dolakha": "Bagmati", "Sindhupalchok": "Bagmati", "Kavrepalanchok": "Bagmati", "Lalitpur": "Bagmati", "Bhaktapur": "Bagmati", "Kathmandu": "Bagmati", "Nuwakot": "Bagmati", "Rasuwa": "Bagmati", "Dhading": "Bagmati", "Makwanpur": "Bagmati", "Chitwan": "Bagmati",
    "Gorkha": "Gandaki", "Lamjung": "Gandaki", "Tanahun": "Gandaki", "Syangja": "Gandaki", "Kaski": "Gandaki", "Manang": "Gandaki", "Mustang": "Gandaki", "Myagdi": "Gandaki", "Parbat": "Gandaki", "Baglung": "Gandaki", "Nawalparasi East": "Gandaki",
    "Gulmi": "Lumbini", "Palpa": "Lumbini", "Nawalparasi West": "Lumbini", "Rupandehi": "Lumbini", "Kapilvastu": "Lumbini", "Arghakhanchi": "Lumbini", "Pyuthan": "Lumbini", "Rolpa": "Lumbini", "Rukum East": "Lumbini", "Dang": "Lumbini", "Banke": "Lumbini", "Bardiya": "Lumbini",
    "Rukum West": "Karnali", "Salyan": "Karnali", "Surkhet": "Karnali", "Dailekh": "Karnali", "Jajarkot": "Karnali", "Dolpa": "Karnali", "Jumla": "Karnali", "Kalikot": "Karnali", "Mugu": "Karnali", "Humla": "Karnali",
    "Bajura": "Sudurpaschim", "Bajhang": "Sudurpaschim", "Achham": "Sudurpaschim", "Doti": "Sudurpaschim", "Kailali": "Sudurpaschim", "Kanchanpur": "Sudurpaschim", "Dadeldhura": "Sudurpaschim", "Baitadi": "Sudurpaschim", "Darchula": "Sudurpaschim",
}

CLIMATE_FEATURES = ["temperature", "humidity", "rainfall"]
DEFAULT_CLIMATE = {
    "temperature": 25.0,
    "humidity": 80.0,
    "rainfall": 202.0,
}


@st.cache_data
def load_offline_db() -> dict:
    db_path = PROJECT_ROOT / "data" / "local_soil_db.json"
    if not db_path.exists():
        return {}
    with open(db_path, "r", encoding="utf-8") as f:
        raw_db = json.load(f)

    grouped_db: dict = {}
    for district, payload in raw_db.items():
        prov = PROVINCE_MAP.get(district, "Unknown Province")
        grouped_db.setdefault(prov, {})[district] = payload
    return grouped_db


def _clean_numeric(val) -> float | None:
    """Clean string percentages like '1.67 %' or HTML tags into floats."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val)
    # Strip HTML tags like <span style='color:red'>
    text = re.sub(r"<[^>]+>", "", text)
    match = re.search(r"[-+]?\d*\.?\d+", text.strip())
    if match:
        return float(match.group())
    return None


@st.cache_resource
def load_predictor() -> CropRecommendationPredictor | None:
    """Load trained model artifacts if available."""
    model_dir = PROJECT_ROOT / "models"
    model_path = model_dir / "ensemble_model.joblib"
    if not model_path.exists():
        return None
    return CropRecommendationPredictor(model_dir=model_dir)


def render_crop_page() -> None:
    """Crop Recommendation System: NARC soil lookup + climate -> crop pick."""
    st.title("🌾 Ensemble Crop Recommendation System")
    st.caption("Offline-Resilient Architecture: Live Edge API with Instant Fallback")

    offline_db = load_offline_db()
    if not offline_db:
        st.error("Offline soil database not found at `data/local_soil_db.json`.")
        return

    # Sidebar - 2-step UI
    with st.sidebar:
        st.header("1. Geographic Selector")

        selected_province = st.selectbox("Select Province", list(offline_db.keys()))
        districts = list(offline_db[selected_province].keys())
        selected_district = st.selectbox("Select District", districts)

        district_data = offline_db[selected_province][selected_district]
        target_lat = district_data["lat"]
        target_lon = district_data["lon"]

        st.text_input("Designated Latitude", value=f"{target_lat:.4f}", disabled=True)
        st.text_input("Designated Longitude", value=f"{target_lon:.4f}", disabled=True)

        st.divider()
        st.header("2. Climate Inputs")
        st.caption("Enter local weather estimates.")
        climate_inputs = {}
        for feature in CLIMATE_FEATURES:
            label = feature.replace("_", " ").title()
            climate_inputs[feature] = st.number_input(
                label,
                value=float(DEFAULT_CLIMATE[feature]),
                format="%.2f",
            )

        st.divider()
        evaluate_clicked = st.button(
            "Evaluate Regional Profile", type="primary", use_container_width=True
        )

    if not evaluate_clicked:
        st.info(
            "Select a Province and District from the sidebar, then click "
            "**Evaluate Regional Profile**."
        )
        return

    payload = None
    is_offline = False

    # Live API Attempt
    with st.spinner("Fetching live NARC data..."):
        try:
            url = f"https://soil.narc.gov.np/soil/api/soildata?lat={target_lat}&lon={target_lon}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=7.0)
            response.raise_for_status()

            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                payload = data[0]
            elif isinstance(data, dict):
                # Check for nested structures
                for key in ("results", "data", "soildata", "soil"):
                    nested = data.get(key)
                    if isinstance(nested, list) and nested:
                        payload = nested[0]
                        break
                    if isinstance(nested, dict):
                        payload = nested
                        break
                if payload is None:
                    payload = data

            # Check for non-arable land or missing soil keys
            if (
                not payload
                or payload.get("result") == "Please select the crop land"
                or "ph" not in payload
            ):
                is_offline = True
                st.info(
                    "Coordinates landed on non-arable land (forest, mountain, or water body)."
                )
                st.warning(
                    "📡 **Local Edge Engine Active**: Seamlessly reading data from "
                    "Krishi-AI's offline geospatial cache."
                )
                payload = district_data

        except requests.exceptions.RequestException:
            is_offline = True
            st.warning(
                "📡 **Local Edge Engine Active**: Live connection timed out or failed. "
                "Seamlessly reading data from Krishi-AI's offline geospatial cache."
            )
            payload = district_data

    if not payload:
        st.error("No soil data available for this location.")
        return

    ph_val = _clean_numeric(payload.get("ph"))
    n_val = _clean_numeric(payload.get("total_nitrogen"))
    p_val = _clean_numeric(payload.get("p2o5"))
    k_val = _clean_numeric(payload.get("potassium"))
    om_val = _clean_numeric(payload.get("organic_matter"))

    # Scale Nitrogen for ML if < 10 (per previous codebase rules)
    ml_n = (n_val * 1000) if (n_val is not None and n_val < 10) else n_val

    # Display Metrics
    st.subheader("Regional Soil Metrics")
    if is_offline:
        st.caption(f"Source: Offline Mock DB ({selected_province} / {selected_district})")
    else:
        prov = payload.get("province", selected_province)
        dist = payload.get("district", selected_district)
        st.caption(f"Source: Live NARC API ({prov} / {dist})")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("pH", f"{ph_val:.2f}" if ph_val else "—")
    m2.metric("Nitrogen (N)", f"{n_val:.3f} %" if n_val else "—")
    m3.metric("P2O5 (P)", f"{p_val:.2f} kg/ha" if p_val else "—")
    m4.metric("Potassium (K)", f"{k_val:.2f} kg/ha" if k_val else "—")
    m5.metric("Org. Matter", f"{om_val:.2f} %" if om_val else "—")

    st.divider()
    st.subheader("Crop Recommendation")

    # Build Vector
    model_inputs = {
        "N": float(ml_n or 0.0),
        "P": float(p_val or 0.0),
        "K": float(k_val or 0.0),
        "ph": float(ph_val or 0.0),
        "temperature": float(climate_inputs["temperature"]),
        "humidity": float(climate_inputs["humidity"]),
        "rainfall": float(climate_inputs["rainfall"]),
    }

    st.dataframe(pd.DataFrame([model_inputs]), use_container_width=True)

    predictor = load_predictor()
    if predictor is None:
        st.error("No trained model found. Please train the model first.")
        return

    try:
        result = predictor.predict_from_dict(model_inputs)

        # Get Top 3 choices
        prob_df = (
            pd.DataFrame(
                {
                    "crop": list(result.probabilities.keys()),
                    "probability": list(result.probabilities.values()),
                }
            )
            .sort_values("probability", ascending=False)
            .head(3)
        )

        cols = st.columns(3)
        for i, row in enumerate(prob_df.itertuples()):
            crop_name = row.crop.title()
            prob = row.probability
            cols[i].metric(f"#{i + 1} Recommendation", crop_name, f"{prob:.1%} Confidence")

    except Exception as exc:
        st.error(f"Prediction error: {exc}")


# ==============================================================================
# DROUGHT RISK: data & page
# ==============================================================================

@st.cache_resource
def load_drought_bundle():
    """Load drought model, features, dataset, and the comparison report."""
    model, features = load_drought_artifacts()
    data = load_drought_dataset()
    comp = {}
    fp = REPORTS / "model_comparison.json"
    if fp.exists():
        comp = json.loads(fp.read_text())
    return model, features, data, comp


def render_drought_page() -> None:
    """Monsoon Drought-Risk Dashboard: per-district predictions + test metrics."""
    st.title("🌧️ Monsoon Drought-Risk Dashboard")
    st.caption(
        "Predicting a weak monsoon (Jun-Sep) per district from pre-monsoon climate — "
        "the failure behind the persona's lost potato season."
    )

    if DROUGHT_IMPORT_ERROR is not None:
        st.error(f"Drought module failed to import:\n\n```\n{DROUGHT_IMPORT_ERROR}\n```")
        return

    # Guard: artifacts must exist
    if not drought_artifacts_exist():
        st.error(
            "Model/data not found. Build them first from the repo root:\n\n"
            "```\npython3 src/drought_build_dataset.py\n"
            "python3 src/drought_model_selection.py\n```"
        )
        return

    model, features, data, comp = load_drought_bundle()

    # Top metrics (from model_comparison.json)
    sel = comp.get("selected_model", "—")
    res = comp.get("results", {}).get(sel, {})
    test = res.get("test", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", sel)
    c2.metric("CV ROC-AUC", res.get("cv_roc_auc", "—"))
    c3.metric("Test ROC-AUC", test.get("roc_auc", "—"))
    c4.metric("Test recall", test.get("recall", "—"))

    st.divider()

    # Sidebar controls
    districts = sorted(data["DISTRICT"].unique())
    default_ix = districts.index("Dhading") if "Dhading" in districts else 0
    with st.sidebar:
        st.header("Controls")
        district = st.selectbox("District", districts, index=default_ix)
        threshold = st.slider(
            "Drought decision threshold", 0.0, 1.0, 0.5, 0.05,
            help="Probability above this is flagged as drought.",
        )
        st.caption(
            "Lower the threshold to catch more droughts (higher recall, more false alarms)."
        )

    # Per-district predictions
    pred = predict_drought_district(
        district, threshold=threshold, model=model, features=features, data=data
    ).sort_values("YEAR")

    st.subheader(f"📍 {district}: predicted vs. actual, by year")

    fig = go.Figure()
    fig.add_bar(
        x=pred["YEAR"], y=pred["monsoon_precip"], name="Monsoon rain (mm)",
        marker_color="#9ecae1", yaxis="y",
    )
    fig.add_scatter(
        x=pred["YEAR"], y=pred["drought_prob"], name="Drought probability",
        mode="lines+markers", line=dict(color="#d62728"), yaxis="y2",
    )
    # mark actual drought years
    dy = pred[pred["drought"] == 1]
    fig.add_scatter(
        x=dy["YEAR"], y=dy["monsoon_precip"], name="Actual drought",
        mode="markers", marker=dict(color="black", size=11, symbol="x"), yaxis="y",
    )
    # threshold reference line drawn explicitly on the probability axis (y2)
    if len(pred):
        fig.add_scatter(
            x=[pred["YEAR"].min(), pred["YEAR"].max()], y=[threshold, threshold],
            mode="lines", name="threshold",
            line=dict(color="#d62728", dash="dot"), yaxis="y2",
        )
    fig.update_layout(
        height=430, hovermode="x unified", legend=dict(orientation="h", y=1.12),
        yaxis=dict(title="Monsoon rain (mm)"),
        yaxis2=dict(title="Drought probability", overlaying="y", side="right", range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # per-district hit summary
    correct = int((pred["drought"] == pred["drought_pred"]).sum())
    st.write(
        f"**{correct}/{len(pred)}** years predicted correctly for {district} "
        f"(threshold = {threshold:.2f})."
    )

    with st.expander("See the yearly table"):
        show = pred[
            ["YEAR", "monsoon_precip", "monsoon_z", "drought", "drought_pred", "drought_prob"]
        ].rename(
            columns={
                "monsoon_precip": "rain_mm", "drought": "actual",
                "drought_pred": "predicted", "drought_prob": "probability",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()

    # Overall held-out test performance (all districts, years >= 2010)
    st.subheader(f"📊 Held-out test performance (all districts, {TEST_START_YEAR}+)")
    left, right = st.columns(2)

    with left:
        cm = test.get("confusion_matrix")
        if cm:
            cm_df = pd.DataFrame(
                cm, index=["actual: no", "actual: drought"],
                columns=["pred: no", "pred: drought"],
            )
            st.write("**Confusion matrix** (default 0.5 threshold)")
            st.dataframe(cm_df, use_container_width=True)
        st.caption(
            f"F1={test.get('f1', '—')} · balanced acc={test.get('balanced_acc', '—')} "
            f"· PR-AUC={test.get('pr_auc', '—')}"
        )

    with right:
        feats_imp = comp.get("top_features") or {}
        if feats_imp:
            imp = pd.Series(feats_imp).head(8).iloc[::-1]
            bar = go.Figure(
                go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2ca02c")
            )
            bar.update_layout(
                height=300, margin=dict(l=10, r=10, t=30, b=10),
                title="Top model features (coef./importance)",
            )
            st.plotly_chart(bar, use_container_width=True)

    with st.expander("ℹ️ How to read this / honest limits"):
        st.markdown(
            "- **Probability** = model's estimated chance of a drought monsoon, from "
            "pre-monsoon (winter + spring) climate only — knowable *before* planting.\n"
            "- Selected on **cross-validated AUC**; treat **~0.74** as realistic skill "
            "(test AUC is optimistic — see `reports/drought_report.md`).\n"
            "- Decision support, **not** a guarantee. Drought is defined statistically "
            "(SPI-like), not from on-the-ground impact records."
        )


# ==============================================================================
# NAVIGATION
# ==============================================================================

PAGES = {
    "🌾 Crop Recommendation": render_crop_page,
    "🌧️ Drought Risk": render_drought_page,
}


def main() -> None:
    with st.sidebar:
        st.title("🌱 Krishi-AI")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()

    PAGES[choice]()


if __name__ == "__main__":
    main()
