import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv(r"data\raw\CropYieldNepal.csv")

# Display basic info
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Data Preprocessing
print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Remove rows with zero yield (invalid data)
df = df[df['yield_kg/ha'] > 0].copy()
print(f"Rows after removing zero yield: {len(df)}")

# Handle any missing values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Districts', 'crop_type']

for col in categorical_columns:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Define features (X) and target (y)
feature_columns = [
    'crop_type_encoded',
    'Districts_encoded',
    'avg_pH_value',
    'fertilizer_in_MT',
    'avg_temp_C',
    'max_temp_C', 
    'min_temp_C',
    'avg_relative_humidity',
    'avg_rainfall_mm_per_year',
    'total_solar_radiation_kWh/m2',
    'avg_wind_speed_m/s',
    'Area'
]

# Verify all features exist
available_features = [col for col in feature_columns if col in df.columns]
missing_features = [col for col in feature_columns if col not in df.columns]

if missing_features:
    print(f"\nWarning: Missing features: {missing_features}")
    feature_columns = available_features

X = df[feature_columns]
y = df['yield_kg/ha']

print(f"\nFeatures used ({len(feature_columns)} features):")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i}. {feature}")

print(f"\nTarget variable: yield_kg/ha (crop yield in kilograms per hectare)")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to evaluate
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

# Evaluate each model
print("\n" + "=" * 80)
print("MODEL EVALUATION RESULTS")
print("=" * 80)

results = []
best_model = None
best_r2 = -np.inf
best_model_name = ""

for name, model in models.items():
    try:
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            'Model': name,
            'CV Mean R2': cv_scores.mean(),
            'CV Std R2': cv_scores.std(),
            'Test R2': r2,
            'MAE': mae,
            'RMSE': rmse
        })
        
        print(f"\n{name}:")
        print(f"  Cross-validation R2: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
        print(f"  Test R2: {r2:.4f}")
        print(f"  MAE: {mae:.2f} kg/ha")
        print(f"  RMSE: {rmse:.2f} kg/ha")
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            
    except Exception as e:
        print(f"\n{name}: Failed - {str(e)}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R2', ascending=False)

print("\n" + "=" * 80)
print("MODEL RANKING (Best to Worst)")
print("=" * 80)
print(results_df.to_string(index=False))

print(f"\n[BEST MODEL] {best_model_name}")
print(f"   Test R2 Score: {best_r2:.4f}")

# Hyperparameter tuning for the best model
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING FOR BEST MODEL")
print("=" * 80)

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tuned_model = RandomForestRegressor(random_state=42)
    
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    tuned_model = XGBRegressor(random_state=42, verbosity=0)
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    tuned_model = GradientBoostingRegressor(random_state=42)
else:
    tuned_model = best_model
    param_grid = {}

if param_grid:
    grid_search = GridSearchCV(tuned_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_

# Train final model on all training data
print("\n" + "=" * 80)
print("TRAINING FINAL MODEL")
print("=" * 80)

best_model.fit(X_train_scaled, y_train)

# Final evaluation
y_pred_final = best_model.predict(X_test_scaled)
final_r2 = r2_score(y_test, y_pred_final)
final_mae = mean_absolute_error(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

print(f"Final Model Performance:")
print(f"  R2 Score: {final_r2:.4f}")
print(f"  MAE: {final_mae:.2f} kg/ha")
print(f"  RMSE: {final_rmse:.2f} kg/ha")

# Feature importance (for tree-based models)
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features for predicting yield:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Calculate cumulative importance
    feature_importance['Cumulative Importance'] = feature_importance['Importance'].cumsum()
    print(f"\nCumulative importance of top 3 features: {feature_importance.head(3)['Importance'].sum()*100:.1f}%")
    print(f"Cumulative importance of top 5 features: {feature_importance.head(5)['Importance'].sum()*100:.1f}%")

# Save the model, scaler, and encoders
print("\n" + "=" * 80)
print("SAVING MODEL AND RELATED FILES")
print("=" * 80)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/best_yield_prediction_model.joblib')
joblib.dump(scaler, 'models/feature_scaler.joblib')
joblib.dump(label_encoders, 'models/label_encoders.joblib')
joblib.dump(feature_columns, 'models/feature_columns.joblib')

print("[OK] Model saved as 'models/best_yield_prediction_model.joblib'")
print("[OK] Scaler saved as 'models/feature_scaler.joblib'")
print("[OK] Label encoders saved as 'models/label_encoders.joblib'")
print("[OK] Feature columns saved as 'models/feature_columns.joblib'")

print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)
print(f"""
Model Information:
-----------------
* Model Type: {best_model_name}
* Number of Features: {len(feature_columns)}
* Training Samples: {len(X_train)}
* Test Samples: {len(X_test)}
* Performance: R2 = {final_r2:.3f}, MAE = {final_mae:.1f} kg/ha

Input Features Required:
{chr(10).join([f"  * {col}" for col in feature_columns])}

Expected Output:
* yield_kg/ha: Predicted crop yield in kilograms per hectare
* Range: Based on your data (likely 0-3000+ kg/ha depending on crop)
* Interpretation: Higher values indicate better predicted yield
""")

# Example prediction function
print("\n" + "=" * 80)
print("EXAMPLE USAGE")
print("=" * 80)

def predict_yield(district, crop_type, area, avg_temp, max_temp, min_temp, 
                  humidity, rainfall, solar_radiation, wind_speed, ph_value, fertilizer):
    """
    Predict crop yield based on input parameters
    
    Parameters:
    - district: Name of district
    - crop_type: Type of crop (Paddy, Maize, Millet, etc.)
    - area: Area in hectares
    - avg_temp: Average temperature in Celsius
    - max_temp: Maximum temperature in Celsius
    - min_temp: Minimum temperature in Celsius
    - humidity: Average relative humidity
    - rainfall: Average rainfall in mm/year
    - solar_radiation: Total solar radiation in kWh/m2
    - wind_speed: Average wind speed in m/s
    - ph_value: Soil pH value
    - fertilizer: Fertilizer amount in MT
    
    Returns:
    - Predicted yield in kg/ha
    """
    # Encode categorical variables
    district_encoded = label_encoders['Districts'].transform([district])[0]
    crop_encoded = label_encoders['crop_type'].transform([crop_type])[0]
    
    # Create feature array
    features = np.array([[
        crop_encoded,
        district_encoded,
        ph_value,
        fertilizer,
        avg_temp,
        max_temp,
        min_temp,
        humidity,
        rainfall,
        solar_radiation,
        wind_speed,
        area
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = best_model.predict(features_scaled)[0]
    
    return prediction

# Example prediction
print("Example prediction for Achham district, Paddy crop:")
example_yield = predict_yield(
    district='Achham',
    crop_type='Paddy', 
    area=9450,
    avg_temp=13.06,
    max_temp=18.97,
    min_temp=8.11,
    humidity=61.28,
    rainfall=1245.93,
    solar_radiation=6522.01,
    wind_speed=1.56,
    ph_value=6.1,
    fertilizer=76
)

print(f"Predicted yield: {example_yield:.0f} kg/ha")
print(f"Actual yield from data: 2400 kg/ha")
print(f"Prediction accuracy: {(1 - abs(example_yield - 2400)/2400)*100:.1f}%")

print("\n[SUCCESS] Model creation complete! You can now load and use 'models/best_yield_prediction_model.joblib'")