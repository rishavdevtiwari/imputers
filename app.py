"""Layer 6 - Streamlit presentation.

Pick a district + season -> ranked crop cards showing estimated yield,
profit range, P(profit), crop-failure risk, top disease threats, and a
plain-language "why this crop". Includes a DQI before/after panel.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.config import load_config

st.set_page_config(page_title="Crop Profit & Risk Advisor", layout="wide")

cfg = load_config()

st.title("🌱 Crop Profit & Risk Advisor")
st.caption(
    "We tell Nepali farmers not just what grows - but what survives, "
    "what pays, how likely it is to pay, and why."
)

with st.sidebar:
    st.header("Your field")
    district = st.text_input("District", value="Dhading")
    season = st.selectbox("Season", ["summer", "winter", "monsoon"])
    run = st.button("Recommend crops", type="primary")

if run:
    try:
        from src.pipeline import recommend

        verdicts = recommend(district=district, season=season)
        for v in verdicts:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Crop", v.crop.title())
                c2.metric("Est. profit (NPR/ha)", f"{v.profit_npr_ha:,.0f}")
                c3.metric("P(profit)", f"{v.prob_profit:.0%}")
                c4.metric("Crop-failure risk", f"{v.failure_risk:.0%}")
                if v.top_diseases:
                    st.warning("Top disease threats: " + ", ".join(v.top_diseases))
                if v.reasons:
                    st.write("**Why this crop:** " + "; ".join(v.reasons))
    except NotImplementedError:
        st.info(
            "Pipeline modules are scaffolded but not yet implemented. "
            "Fill in src/ingest.py, quality.py, features.py, suitability.py, "
            "and disease_risk.py to enable live recommendations."
        )
else:
    st.info("Enter a district and season, then click **Recommend crops**.")
