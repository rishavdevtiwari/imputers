# 🌾 Krishi-AI: Precision Agriculture Suite

Krishi-AI is an intelligent, multi-model AI dashboard designed specifically for the Nepalese agricultural sector. By combining state-of-the-art machine learning models with live APIs and hardware sensors, Krishi-AI provides farmers, researchers, and agronomists with actionable insights for optimal crop selection, yield prediction, and economic forecasting.

---

## 🌟 Core Features

- **🧠 3-Tier Machine Learning Architecture**:
  - **Crop Recommendation**: Classifies the optimal crop based on soil and climate vectors.
  - **Yield Prediction**: Regressive modeling predicting the expected metric tonnage of crop yield based on farm area and local inputs.
  - **Drought Risk Assessment**: Probabilistic assessment forecasting rainfall deficits over a 3-month rolling window.
  
- **🌍 Instant Multi-Language Support**:
  Fully integrated English (🇬🇧) and Nepali (🇳🇵) translations across the entire dashboard, including dynamically translated machine learning outputs.

- **📡 Live Data Integrations**:
  - **NARC Soil API**: Automatically fetches subsurface soil metrics (pH, N, P, K) based on regional coordinates.
  - **Kalimati Market Scraper**: Live web scraping engine equipped with a Nepali-to-English translation mapping to fetch real-time wholesale market prices for ROI calculations.
  
- **💸 Economic Viability Projections**:
  Projects Net Profit Margins, Gross Revenue, and Asset/Labor costs scaling dynamically with the selected Farm Cultivation Area.

---

## 🛠️ Technology Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/) (100% Native Architecture + Custom Glassmorphism CSS)
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn
- **Data Engineering**: Pandas, NumPy
- **Scraping & APIs**: Requests, BeautifulSoup4, RapidFuzz

---

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/softwarica-college-class/softwarica-hackathon-2026-imputers.git
   cd softwarica-hackathon-2026-imputers
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run app/app.py
   ```
   The dashboard will automatically open at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app/
│   └── app.py                  # Main Streamlit Dashboard application
├── data/
│   ├── crop_prices.txt         # Offline fallback cache for Kalimati market prices
│   └── local_soil_db.json      # Offline geospatial fallback for NARC soil API
├── models/                     # Pre-trained ML models (.joblib files)
├── src/
│   ├── profit.py               # Financial calculation & Kalimati web scraping logic
│   ├── drought_predict.py      # Drought risk inference logic
│   ├── yield.py                # Yield regression training script
│   ├── train_drought_model.py  # Model training script for drought risk
│   ├── train.py                # Model training script for crop recommendation
│   └── rece.py                 # Serial port listener for hardware sensors
└── requirements.txt            # Project dependencies for deployment
```

---
*Built for the Softwarica Hackathon 2026.* 🚀
