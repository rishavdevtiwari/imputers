import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "drought_dataset.csv"
MODEL_PATH = ROOT / "models" / "drought_model.joblib"
FEATURES_PATH = ROOT / "models" / "drought_features.joblib"

FEATURES = [
    'mam_T2M', 'mam_T2M_MAX', 'mam_T2M_MIN', 'mam_T2M_RANGE', 
    'mam_RH2M', 'mam_QV2M', 'mam_PS', 'mam_WS10M', 'mam_PRECTOT', 
    'djf_T2M', 'djf_RH2M', 'djf_PRECTOT', 'prev_monsoon_precip', 
    'prev_monsoon_z', 'LAT', 'LON'
]

TARGET = 'drought'

def evaluate_model(pipeline, X, y):
    """Run Stratified 5-Fold CV and return average train/val F1 scores."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        
        train_scores.append(f1_score(y_train, y_train_pred, average='macro'))
        val_scores.append(f1_score(y_val, y_val_pred, average='macro'))
        
    return np.mean(train_scores), np.mean(val_scores)

def main():
    print("=" * 60)
    print("DROUGHT MODEL ANTI-OVERFITTING PIPELINE")
    print("=" * 60)
    
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        return
        
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    
    X = df[FEATURES]
    y = df[TARGET]
    
    print("\nClass Distribution:")
    print(y.value_counts())
    
    # Starting regularization parameters
    current_depth = 5
    min_samples_split = 15
    min_samples_leaf = 10
    
    best_pipeline = None
    converged = False
    
    while current_depth >= 1 and not converged:
        print(f"\n--- Testing max_depth={current_depth} ---")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=current_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        train_f1, val_f1 = evaluate_model(pipeline, X, y)
        diff = abs(train_f1 - val_f1)
        
        print(f"Training F1 (Macro):   {train_f1:.4f}")
        print(f"Validation F1 (Macro): {val_f1:.4f}")
        print(f"Difference:            {diff:.4f} (Target: <= 0.05)")
        
        if diff <= 0.05:
            print(f"-> CONVERGENCE ACHIEVED at max_depth={current_depth}!")
            best_pipeline = pipeline
            converged = True
        else:
            print(f"-> Model is still overfitting. Reducing max_depth...")
            current_depth -= 1
            
    if not converged:
        print("\n[WARNING] Could not converge within 5%. Using lowest depth configuration.")
        best_pipeline = pipeline
        
    # Re-fit the stable generalized model on ALL data
    print("\nRe-fitting final model on complete dataset...")
    best_pipeline.fit(X, y)
    
    print("\n" + "=" * 60)
    print("FINAL MODEL CLASSIFICATION REPORT (ON FULL SET)")
    print("=" * 60)
    y_pred = best_pipeline.predict(X)
    print(classification_report(y, y_pred, target_names=["No Drought", "Drought"]))
    
    cm = confusion_matrix(y, y_pred)
    print("CONFUSION MATRIX:")
    print(pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"]))
    
    # Save robust artifacts
    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)
    print(f"\n[OK] Saved highly-regularized model to {MODEL_PATH}")
    print(f"[OK] Saved features tracking to {FEATURES_PATH}")
    
if __name__ == '__main__':
    main()
