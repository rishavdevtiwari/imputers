"""
Streamlit frontend for the Crop Recommendation Ensemble ML system.

Offline-Resilient District Selector Edition.
"""

from __future__ import annotations

import sys
import re
import json
from pathlib import Path
import requests

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = Path(__file__).resolve().parent
for path in (str(SRC_DIR), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from predict import CropRecommendationPredictor  # noqa: E402

# ==============================================================================
# 1. UNIFIED GEOGRAPHIC & SOIL OFFLINE DATABASE
# ==============================================================================

PROVINCE_MAP = {
    "Taplejung": "Koshi", "Sankhuwasabha": "Koshi", "Solukhumbu": "Koshi", "Okhaldhunga": "Koshi", "Khotang": "Koshi", "Bhojpur": "Koshi", "Dhankuta": "Koshi", "Terhathum": "Koshi", "Panchthar": "Koshi", "Ilam": "Koshi", "Jhapa": "Koshi", "Morang": "Koshi", "Sunsari": "Koshi", "Udayapur": "Koshi",
    "Saptari": "Madhesh", "Siraha": "Madhesh", "Dhanusha": "Madhesh", "Mahottari": "Madhesh", "Sarlahi": "Madhesh", "Rautahat": "Madhesh", "Bara": "Madhesh", "Parsa": "Madhesh",
    "Sindhuli": "Bagmati", "Ramechhap": "Bagmati", "Dolakha": "Bagmati", "Sindhupalchok": "Bagmati", "Kavrepalanchok": "Bagmati", "Lalitpur": "Bagmati", "Bhaktapur": "Bagmati", "Kathmandu": "Bagmati", "Nuwakot": "Bagmati", "Rasuwa": "Bagmati", "Dhading": "Bagmati", "Makwanpur": "Bagmati", "Chitwan": "Bagmati",
    "Gorkha": "Gandaki", "Lamjung": "Gandaki", "Tanahun": "Gandaki", "Syangja": "Gandaki", "Kaski": "Gandaki", "Manang": "Gandaki", "Mustang": "Gandaki", "Myagdi": "Gandaki", "Parbat": "Gandaki", "Baglung": "Gandaki", "Nawalparasi East": "Gandaki",
    "Gulmi": "Lumbini", "Palpa": "Lumbini", "Nawalparasi West": "Lumbini", "Rupandehi": "Lumbini", "Kapilvastu": "Lumbini", "Arghakhanchi": "Lumbini", "Pyuthan": "Lumbini", "Rolpa": "Lumbini", "Rukum East": "Lumbini", "Dang": "Lumbini", "Banke": "Lumbini", "Bardiya": "Lumbini",
    "Rukum West": "Karnali", "Salyan": "Karnali", "Surkhet": "Karnali", "Dailekh": "Karnali", "Jajarkot": "Karnali", "Dolpa": "Karnali", "Jumla": "Karnali", "Kalikot": "Karnali", "Mugu": "Karnali", "Humla": "Karnali",
    "Bajura": "Sudurpaschim", "Bajhang": "Sudurpaschim", "Achham": "Sudurpaschim", "Doti": "Sudurpaschim", "Kailali": "Sudurpaschim", "Kanchanpur": "Sudurpaschim", "Dadeldhura": "Sudurpaschim", "Baitadi": "Sudurpaschim", "Darchula": "Sudurpaschim"
}

@st.cache_data
def load_offline_db() -> dict:
    db_path = PROJECT_ROOT / "data" / "local_soil_db.json"
    if not db_path.exists():
        return {}
    with open(db_path, "r", encoding="utf-8") as f:
        raw_db = json.load(f)
        
    grouped_db = {}
    for district, payload in raw_db.items():
        prov = PROVINCE_MAP.get(district, "Unknown Province")
        if prov not in grouped_db:
            grouped_db[prov] = {}
        grouped_db[prov][district] = payload
    return grouped_db

OFFLINE_DB = load_offline_db()

CLIMATE_FEATURES = ["temperature", "humidity", "rainfall"]
DEFAULT_CLIMATE = {
    "temperature": 25.0,
    "humidity": 80.0,
    "rainfall": 202.0,
}

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
)

def _clean_numeric(val) -> float | None:
    """Clean string percentages like '1.67 %' or HTML tags into floats."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val)
    # Strip HTML tags like <span style='color:red'>
    text = re.sub(r'<[^>]+>', '', text)
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

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

st.title("🌾 Ensemble Crop Recommendation System")
st.caption("Offline-Resilient Architecture: Live Edge API with Instant Fallback")

# Sidebar - 2-step UI
with st.sidebar:
    st.header("1. Geographic Selector")
    
    selected_province = st.selectbox("Select Province", list(OFFLINE_DB.keys()))
    districts = list(OFFLINE_DB[selected_province].keys())
    selected_district = st.selectbox("Select District", districts)
    
    district_data = OFFLINE_DB[selected_province][selected_district]
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
    evaluate_clicked = st.button("Evaluate Regional Profile", type="primary", use_container_width=True)

if evaluate_clicked:
    payload = None
    is_offline = False
    
    # Live API Attempt
    with st.spinner("Fetching live NARC data..."):
        try:
            url = f"https://soil.narc.gov.np/soil/api/soildata?lat={target_lat}&lon={target_lon}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
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
            if not payload or payload.get("result") == "Please select the crop land" or "ph" not in payload:
                is_offline = True
                st.info("Coordinates landed on non-arable land (forest, mountain, or water body).")
                st.warning("📡 **Local Edge Engine Active**: Seamlessly reading data from Krishi-AI's offline geospatial cache.")
                payload = district_data
                
        except requests.exceptions.RequestException as e:
            is_offline = True
            st.warning("📡 **Local Edge Engine Active**: Live connection timed out or failed. Seamlessly reading data from Krishi-AI's offline geospatial cache.")
            payload = district_data
    
    if payload:
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
        else:
            try:
                result = predictor.predict_from_dict(model_inputs)
                
                # Get Top 3 choices
                prob_df = pd.DataFrame(
                    {
                        "crop": list(result.probabilities.keys()),
                        "probability": list(result.probabilities.values()),
                    }
                ).sort_values("probability", ascending=False).head(3)
                
                col_c1, col_c2, col_c3 = st.columns(3)
                cols = [col_c1, col_c2, col_c3]
                
                for i, row in enumerate(prob_df.itertuples()):
                    crop_name = row.crop.title()
                    prob = row.probability
                    cols[i].metric(f"#{i+1} Recommendation", crop_name, f"{prob:.1%} Confidence")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

else:
    st.info("Select a Province and District from the sidebar, then click **Evaluate Regional Profile**.")
