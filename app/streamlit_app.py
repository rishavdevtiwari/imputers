"""
Streamlit frontend for the Crop Recommendation Ensemble ML system.

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import FEATURE_COLUMNS  # noqa: E402
from predict import CropRecommendationPredictor  # noqa: E402

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Ensemble Crop Recommendation System")
st.caption(
    "Data-for-Good hackathon prototype — soil & climate inputs → ML ensemble recommendation"
)

DEFAULTS = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 25.0,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 202.0,
}


@st.cache_resource
def load_predictor() -> CropRecommendationPredictor | None:
    """Load trained model artifacts if available."""
    model_dir = PROJECT_ROOT / "models"
    model_path = model_dir / "ensemble_model.joblib"
    if not model_path.exists():
        return None
    return CropRecommendationPredictor(model_dir=model_dir)


with st.sidebar:
    st.header("Soil & Climate Inputs")
    inputs = {}
    for feature in FEATURE_COLUMNS:
        label = feature.replace("_", " ").title()
        inputs[feature] = st.number_input(
            label,
            value=float(DEFAULTS.get(feature, 0.0)),
            format="%.2f",
        )

predictor = load_predictor()

if predictor is None:
    st.warning(
        "No trained model found. Place your dataset in `data/raw/` and run:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.subheader("Preview Input Vector")
    st.dataframe(pd.DataFrame([inputs]), use_container_width=True)
else:
    if st.button("Recommend Crop", type="primary", use_container_width=True):
        result = predictor.predict_from_dict(inputs)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Recommended Crop", result.recommended_crop)
            st.metric("Confidence", f"{result.confidence:.1%}")

        with col2:
            prob_df = pd.DataFrame(
                {
                    "crop": list(result.probabilities.keys()),
                    "probability": list(result.probabilities.values()),
                }
            ).sort_values("probability", ascending=True)

            fig = px.bar(
                prob_df,
                x="probability",
                y="crop",
                orientation="h",
                title="Class Probabilities",
                labels={"probability": "Probability", "crop": "Crop"},
            )
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
