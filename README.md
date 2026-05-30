[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5U5e_941)
# 🌾 Krishi-AI: Precision Agriculture Suite

Krishi-AI is an AI-powered precision agriculture platform developed to support farmers, researchers, and agricultural stakeholders in Nepal. The platform combines machine learning, environmental datasets, market intelligence, and IoT sensor integration to provide data-driven insights for crop selection, precipitation forecasting, yield prediction, and profitability analysis.

By leveraging Nepal-specific agricultural data and modern AI techniques, Krishi-AI helps farmers make informed decisions that improve productivity, optimize resource utilization, and increase economic returns.

---

# 🚀 Problem Statement

Agriculture contributes significantly to Nepal's economy, yet farmers often face challenges such as:

* Uncertainty in selecting suitable crops
* Unpredictable weather conditions
* Limited access to soil testing facilities
* Lack of yield forecasting tools
* Difficulty estimating cultivation profitability
* Limited access to real-time market information

Krishi-AI addresses these challenges through an integrated intelligent decision-support system.

---

# 🌟 Key Features

## 🧠 Multi-Model AI Architecture

Krishi-AI utilizes three interconnected machine learning models:

### 1️⃣ Crop Recommendation Model

Determines the most suitable crop based on:

* Soil pH
* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Rainfall
* Soil Moisture

**Output:** Recommended crop for cultivation.

---

### 2️⃣ Yield Prediction Model

Predicts expected crop production based on:

* Selected crop
* Farm cultivation area
* Soil characteristics
* Environmental conditions

**Output:** Estimated crop yield in metric tons.

---

### 3️⃣ Precipitation Prediction Model

Forecasts future rainfall patterns using climate and historical weather data.

Input Parameters:

* Temperature
* Humidity
* Historical rainfall records
* Seasonal weather indicators

**Output:**

* Expected precipitation level
* Rainfall probability
* Rainfall classification (Low, Medium, High)

This helps farmers optimize irrigation planning, planting schedules, and crop selection strategies.

---

## 🌍 Multi-Language Support

The dashboard supports:

* 🇬🇧 English
* 🇳🇵 Nepali

Predictions and recommendations are dynamically translated to improve accessibility for local farmers.

---

## 📡 Real-Time Data Integration

### Soil Intelligence

The platform can retrieve soil information from:

* NARC Soil Database
* Local Offline Soil Database
* IoT-Based Soil Sensors

Supported soil parameters include:

* Soil pH
* Soil Moisture
* Nitrogen
* Phosphorus
* Potassium

---

### Market Intelligence

Krishi-AI integrates with Kalimati market data sources to provide:

* Live wholesale crop prices
* Crop profitability estimation
* ROI calculations

RapidFuzz-based matching is used to align crop names from multiple sources.

---

## 💰 Economic Viability Forecasting

The profitability engine estimates:

* Cultivation Cost
* Labor Cost
* Operational Expenses
* Gross Revenue
* Net Profit
* Return on Investment (ROI)

Calculations dynamically adjust according to farm size and predicted yield.

---

# 🔄 System Workflow

```text
                  Soil Sensor Data
                         │
                         ▼
              Soil Feature Extraction
                         │
                         ▼
              Crop Recommendation Model
                         │
                         ▼
                Recommended Crop
                         │
                         ▼
                 Yield Prediction
                         │
                         ▼
              Profitability Forecast
                         │
                         ▼
                Revenue & ROI Report


      Historical Climate Data
                    │
                    ▼
       Precipitation Prediction Model
                    │
                    ▼
          Rainfall Forecast Report
```

---

# 🛠 Technology Stack

## Frontend

* Streamlit
* HTML/CSS
* Custom Glassmorphism UI

## Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn

## Data Processing

* Pandas
* NumPy

## Web Scraping & APIs

* Requests
* BeautifulSoup4
* RapidFuzz

## Hardware Integration

* ESP32
* Arduino
* Serial Communication

---

# 📊 Machine Learning Models

| Model                    | Type                        | Purpose                      |
| ------------------------ | --------------------------- | ---------------------------- |
| Crop Recommendation      | Classification              | Recommend suitable crops     |
| Yield Prediction         | Regression                  | Predict crop yield           |
| Precipitation Prediction | Classification / Regression | Forecast rainfall conditions |

---

# 📚 Data Sources

The project uses publicly available datasets from trusted agricultural and climate repositories.

## 1. Open Data Nepal

Source:

https://opendatanepal.com/datasets?q=soil%20pH%20and%20moisture%20for%20Nepal

Dataset Contents:

* Soil pH measurements
* Soil moisture records
* Regional agricultural information
* Nepal-specific soil characteristics

Usage:

* Crop recommendation model training
* Soil suitability analysis
* Soil feature engineering

---

## 2. NASA Climate & Agricultural Dataset

Source:

https://data.mendeley.com/datasets/8v757rr4st/1/files/98242fd3-1912-4a59-ab26-23d97b454218

Dataset Contents:

* Rainfall observations
* Temperature measurements
* Humidity records
* Climate indicators

Usage:

* Precipitation prediction model
* Yield forecasting
* Climate feature extraction

---

## 3. Kaggle Fertilizer Recommendation Dataset

Source:

https://www.kaggle.com/code/analyticaobscura/optimal-fertilizers-eda-playground-0-38265/notebook

Dataset Contents:

* Soil nutrient information
* Fertilizer recommendations
* Crop-soil relationships

Usage:

* Crop recommendation model
* Soil nutrient analysis
* Agricultural decision support

---

# 📡 Sensor Calibration & Limitations

Krishi-AI supports integration with low-cost soil sensors through ESP32 and Arduino devices.

## Calibration Process

### Soil Moisture Sensor Calibration

1. Record sensor value in completely dry soil.
2. Record sensor value in fully saturated soil.
3. Normalize readings between dry and wet values.
4. Convert normalized values into moisture percentages.

### Soil pH Sensor Calibration

1. Use standard pH buffer solutions.
2. Calibrate using pH 4, pH 7, and pH 10 references.
3. Adjust sensor offsets before deployment.

---

## Sensor Limitations

### Soil Moisture Sensor

Limitations:

* Accuracy varies with soil composition.
* Random readings may occur due to sensor faults.
* Sensor corrosion can occur over long-term use.
* Requires periodic recalibration.
* Extreme temperatures may affect measurements.

### Soil pH Sensor

Limitations:

* Sensitive to temperature fluctuations.
* Requires regular cleaning.
* Accuracy is lower than laboratory-grade testing equipment.
* Readings may vary due to soil heterogeneity.
* Requires routine recalibration.

### General Hardware Limitations

* Environmental noise may affect readings.
* Sensor drift may occur over time.
* Extreme weather conditions may impact accuracy.
* Internet connectivity is required for live data retrieval.
* Low-cost sensors provide estimations and should not replace laboratory analysis.

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/softwarica-college-class/softwarica-hackathon-2026-imputers.git

cd softwarica-hackathon-2026-imputers
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Launch Application

```bash
streamlit run app/app.py
```

Application URL:

```text
http://localhost:8501
```

---

# 📂 Project Structure

```text
softwarica-hackathon-2026-imputers
│
├── app/
│   └── app.py
│
├── data/
│   ├── crop_prices.txt
│   └── local_soil_db.json
│
├── models/
│   ├── crop_model.joblib
│   ├── yield_model.joblib
│   └── precipitation_model.joblib
│
├── src/
│   ├── train.py
│   ├── yield.py
│   ├── precipitation_predict.py
│   ├── train_precipitation_model.py
│   ├── profit.py
│   └── rece.py
│
├── requirements.txt
│
└── README.md
```

---

# 🔮 Future Enhancements

* Satellite imagery integration
* Disease detection using computer vision
* Mobile application development
* GPS-enabled farm mapping
* Fertilizer recommendation engine
* Weather forecast API integration
* Government subsidy recommendation support

---

# 👨‍💻 Team Imputers

Developed during **Softwarica Hackathon 2026** to empower Nepalese agriculture through Artificial Intelligence, Machine Learning, Data Analytics, and IoT technologies.

### Technologies Used

* Streamlit
* Scikit-Learn
* XGBoost
* Pandas
* NumPy
* ESP32
* Arduino
* RapidFuzz

### AI-Assisted Development Tools

* ChatGPT
* Cursor
* DeepSeek
* Kiro
* Anti Gravity

### Mission

To provide Nepalese farmers with affordable, data-driven agricultural intelligence that improves productivity, sustainability, and profitability.

---

# 📜 Disclaimer

Krishi-AI provides AI-generated recommendations based on available datasets and sensor readings. Predictions should be used as decision-support tools and not as a replacement for professional agricultural consultation or laboratory soil testing.

---

# 🌱 Empowering Nepalese Farmers Through Data-Driven Agriculture

