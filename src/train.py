import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from preprocess import build_preprocessor
from pathlib import Path


# =========================
# PROJECT PATH (ROBUST FIX)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(
    PROJECT_ROOT / "data/processed/cleaned_soil_crop.csv"
)

X = df.drop("label", axis=1)
y = df["label"]


# ------------------------
# Encode labels
# ------------------------
le = LabelEncoder()
y = le.fit_transform(y)


# ------------------------
# Split dataset
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------
# Feature separation
# ------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# ------------------------
# Preprocessing pipeline
# ------------------------
preprocessor = build_preprocessor(numeric_features, categorical_features)


# =========================================================
# MODEL 1: Logistic Regression
# =========================================================
log_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))


# =========================================================
# MODEL 2: Random Forest
# =========================================================
rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))


# =========================================================
# MODEL 3: XGBoost
# =========================================================
xgb_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="mlogloss",
        tree_method="hist"
    ))
])

xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))


# ------------------------
# Results
# ------------------------
print("\nMODEL COMPARISON")
print("-----------------")
print(f"Logistic Regression: {log_acc:.4f}")
print(f"Random Forest      : {rf_acc:.4f}")
print(f"XGBoost            : {xgb_acc:.4f}")


# ------------------------
# Select BEST MODEL
# ------------------------
models = {
    "logistic_regression": (log_acc, log_model),
    "random_forest": (rf_acc, rf_model),
    "xgboost": (xgb_acc, xgb_model)
}

best_name = max(models, key=lambda k: models[k][0])
best_model = models[best_name][1]

print(f"\n🏆 BEST MODEL: {best_name}")


# ------------------------
# SAVE ARTIFACTS (ROBUST)
# ------------------------
joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

print(f"\nModel saved successfully in: {MODEL_DIR}")