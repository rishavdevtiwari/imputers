"""
CROP RECOMMENDATION SYSTEM
Recommends the best crop based on weather and soil conditions using machine learning

This system uses your existing crop yield data to learn which crops grow best
under different environmental conditions and provides recommendations with
confidence scores.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================

def load_and_prepare_data():
    """
    Load crop data and prepare it for the recommendation model.
    
    Returns:
        X: Feature matrix (weather and soil conditions)
        y: Target vector (crop types)
        feature_columns: List of feature names used
    """
    # Load your existing dataset
    df = pd.read_csv(r"data\raw\CropYieldNepal.csv")
    
    print("="*80)
    print("CROP RECOMMENDATION SYSTEM - DATA PREPARATION")
    print("="*80)
    print(f"Original dataset shape: {df.shape}")
    
    # Remove rows with zero yield as they represent invalid or missing data
    df = df[df['yield_kg/ha'] > 0].copy()
    print(f"After removing zero yield: {df.shape}")
    
    # Define features for crop recommendation (weather + soil conditions)
    feature_columns = [
        'avg_temp_C',                    # Average temperature in Celsius
        'max_temp_C',                    # Maximum temperature in Celsius
        'min_temp_C',                    # Minimum temperature in Celsius
        'avg_relative_humidity',         # Average relative humidity percentage
        'avg_rainfall_mm_per_year',      # Annual rainfall in millimeters
        'total_solar_radiation_kWh/m2',  # Solar radiation received
        'avg_wind_speed_m/s',            # Average wind speed
        'avg_pH_value',                  # Soil pH level (acidity/alkalinity)
        'fertilizer_in_MT',              # Fertilizer amount in metric tons
        'Area'                           # Land area in hectares
    ]
    
    # Target variable - which crop was planted
    target_column = 'crop_type'
    
    # Check which features are available in the dataset
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        print(f"\nWarning: Missing features: {missing_features}")
    
    X = df[available_features]  # Features: weather and soil conditions
    y = df[target_column]        # Target: crop type
    
    print(f"\nFeatures for recommendation ({len(available_features)} features):")
    for i, feature in enumerate(available_features, 1):
        print(f"   {i}. {feature}")
    
    print(f"\nTarget: {target_column}")
    print(f"   Unique crops: {y.nunique()}")
    print(f"   Crops: {y.unique()}")
    
    return X, y, available_features


# ============================================================================
# PART 2: CROP RECOMMENDATION MODEL
# ============================================================================

class CropRecommendationModel:
    """
    Machine Learning model for crop recommendation.
    Trains multiple classifiers and selects the best performing one.
    """
    
    def __init__(self):
        """Initialize the recommendation model with required components."""
        self.scaler = StandardScaler()           # For feature normalization
        self.label_encoder = LabelEncoder()      # For encoding crop names
        self.model = None                         # Trained model
        self.best_model_name = None               # Name of best model
        self.feature_columns = None               # List of feature names
        self.crop_classes = None                  # List of crop types
        
    def train_models(self, X, y, feature_columns):
        """
        Train multiple models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_columns: Names of features used
            
        Returns:
            float: Best accuracy achieved
        """
        self.feature_columns = feature_columns
        
        # Encode crop names to numbers (e.g., 'Paddy' -> 0, 'Maize' -> 1)
        y_encoded = self.label_encoder.fit_transform(y)
        self.crop_classes = self.label_encoder.classes_
        
        print(f"\nEncoded crops:")
        for i, crop in enumerate(self.crop_classes):
            print(f"   {i}: {crop}")
        
        # Split data into training (80%) and testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Normalize features for better model performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nData split:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Define multiple classification models to evaluate
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        # Evaluate each model
        results = []
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        print("\n" + "="*80)
        print("MODEL EVALUATION FOR CROP RECOMMENDATION")
        print("="*80)
        
        for name, model in models.items():
            try:
                # Perform 5-fold cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                # Train the model
                model.fit(X_train_scaled, y_train)
                
                # Test the model
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'CV Mean Accuracy': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Test Accuracy': accuracy
                })
                
                print(f"\n{name}:")
                print(f"   Cross-validation Accuracy: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
                print(f"   Test Accuracy: {accuracy:.4f}")
                
                # Track the best performing model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"\n{name}: Failed - {str(e)}")
        
        # Display model ranking
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL RANKING (Best to Worst)")
        print("="*80)
        print(results_df.to_string(index=False))
        
        print(f"\n[BEST MODEL] {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Hyperparameter tuning for the best model
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING")
        print("="*80)
        
        # Define tuning parameters based on the best model
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            tuned_model = RandomForestClassifier(random_state=42)
            
        elif best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
            tuned_model = XGBClassifier(random_state=42, verbosity=0)
            
        elif best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
            tuned_model = GradientBoostingClassifier(random_state=42)
        else:
            tuned_model = best_model
            param_grid = {}
        
        # Perform grid search for optimal hyperparameters
        if param_grid:
            grid_search = GridSearchCV(tuned_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            best_model = grid_search.best_estimator_
            
            # Evaluate tuned model
            y_pred_tuned = best_model.predict(X_test_scaled)
            tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
            print(f"Tuned model test accuracy: {tuned_accuracy:.4f}")
            
            if tuned_accuracy > best_accuracy:
                best_accuracy = tuned_accuracy
                print("[IMPROVED] Tuned model performs better!")
        
        # Store the final model
        self.model = best_model
        self.best_model_name = best_model_name
        
        # Final evaluation on test set
        y_pred_final = self.model.predict(X_test_scaled)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        print("\n" + "="*80)
        print("FINAL MODEL PERFORMANCE")
        print("="*80)
        print(f"Model: {self.best_model_name}")
        print(f"Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        
        # Detailed classification report
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred_final, target_names=self.crop_classes))
        
        # Feature importance analysis (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "="*80)
            print("FEATURE IMPORTANCE FOR CROP RECOMMENDATION")
            print("="*80)
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nMost important factors for crop recommendation:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"   {row['Feature']}: {row['Importance']*100:.2f}%")
        
        return final_accuracy
    
    def save_model(self, filepath='models/crop_recommendation_model.joblib'):
        """
        Save the trained model and all associated components to disk.
        
        Args:
            filepath: Path where to save the model
        """
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, f'{filepath}')
        joblib.dump(self.scaler, 'models/recommendation_scaler.joblib')
        joblib.dump(self.label_encoder, 'models/crop_label_encoder.joblib')
        joblib.dump(self.feature_columns, 'models/recommendation_features.joblib')
        
        print(f"\n[SAVED] Model saved to {filepath}")
        print(f"[SAVED] Scaler saved to models/recommendation_scaler.joblib")
        print(f"[SAVED] Label encoder saved to models/crop_label_encoder.joblib")
    
    def load_model(self, filepath='models/crop_recommendation_model.joblib'):
        """
        Load a previously trained model from disk.
        
        Args:
            filepath: Path where the model is stored
        """
        self.model = joblib.load(filepath)
        self.scaler = joblib.load('models/recommendation_scaler.joblib')
        self.label_encoder = joblib.load('models/crop_label_encoder.joblib')
        self.feature_columns = joblib.load('models/recommendation_features.joblib')
        self.crop_classes = self.label_encoder.classes_
        print(f"[LOADED] Model loaded from {filepath}")
    
    def recommend_crop(self, weather_data, soil_data, area_ha=1, top_n=3):
        """
        Recommend the best crop based on weather and soil conditions.
        
        Args:
            weather_data: Dictionary with temperature, humidity, rainfall, etc.
            soil_data: Dictionary with pH, fertilizer, etc.
            area_ha: Area in hectares
            top_n: Number of top recommendations to return
            
        Returns:
            List of dictionaries with crop recommendations and confidence scores
        """
        # Prepare input feature array in the correct order
        features = np.array([[
            weather_data.get('avg_temp', 25.0),
            weather_data.get('max_temp', 32.0),
            weather_data.get('min_temp', 18.0),
            weather_data.get('humidity', 65.0),
            weather_data.get('rainfall', 1200.0),
            weather_data.get('solar_radiation', 6500.0),
            weather_data.get('wind_speed', 2.5),
            soil_data.get('ph_value', 6.5),
            soil_data.get('fertilizer', 100.0),
            area_ha
        ]])
        
        # Normalize features using the same scaler used in training
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities for all crops
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get indices of top N recommendations
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'crop': self.crop_classes[idx],
                'confidence': probabilities[idx] * 100,
                'suitability_score': probabilities[idx] * 100
            })
        
        return recommendations
    
    def get_crop_suitability(self, crop_name, weather_data, soil_data, area_ha=1):
        """
        Get suitability score for a specific crop.
        
        Args:
            crop_name: Name of the crop to check
            weather_data: Dictionary with weather conditions
            soil_data: Dictionary with soil conditions
            area_ha: Area in hectares
            
        Returns:
            Dictionary with suitability information for the requested crop
        """
        # Get all recommendations
        all_recommendations = self.recommend_crop(weather_data, soil_data, area_ha, top_n=len(self.crop_classes))
        
        # Find the specific crop
        for rec in all_recommendations:
            if rec['crop'].lower() == crop_name.lower():
                return rec
        
        # Return zero suitability if crop not found
        return {'crop': crop_name, 'confidence': 0, 'suitability_score': 0}
    
    def compare_crops(self, weather_data, soil_data, area_ha=1):
        """
        Compare all crops and return sorted list by suitability.
        
        Args:
            weather_data: Dictionary with weather conditions
            soil_data: Dictionary with soil conditions
            area_ha: Area in hectares
            
        Returns:
            DataFrame with all crops sorted by suitability score
        """
        features = np.array([[
            weather_data.get('avg_temp', 25.0),
            weather_data.get('max_temp', 32.0),
            weather_data.get('min_temp', 18.0),
            weather_data.get('humidity', 65.0),
            weather_data.get('rainfall', 1200.0),
            weather_data.get('solar_radiation', 6500.0),
            weather_data.get('wind_speed', 2.5),
            soil_data.get('ph_value', 6.5),
            soil_data.get('fertilizer', 100.0),
            area_ha
        ]])
        
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create sorted comparison dataframe
        comparison = pd.DataFrame({
            'Crop': self.crop_classes,
            'Suitability_Score': probabilities * 100
        }).sort_values('Suitability_Score', ascending=False)
        
        return comparison


# ============================================================================
# PART 3: INTEGRATION WITH YIELD PREDICTION
# ============================================================================

class IntegratedCropAdvisor:
    """
    Integrated system that combines three key functionalities:
    1. Crop recommendation - Which crop to plant based on conditions
    2. Yield prediction - How much the crop will produce
    3. Profit calculation - Expected financial returns
    
    This provides a complete farming decision support system.
    """
    
    def __init__(self):
        """Initialize the integrated advisor with both recommendation and yield models."""
        self.recommendation_model = CropRecommendationModel()
        self.yield_model = None
        self.yield_scaler = None
        self.yield_encoders = None
        
        # Try to load existing yield prediction model
        self.load_yield_model()
        
    def load_yield_model(self):
        """Load the pre-trained yield prediction model if available."""
        try:
            self.yield_model = joblib.load('models/best_yield_prediction_model.joblib')
            self.yield_scaler = joblib.load('models/feature_scaler.joblib')
            self.yield_encoders = joblib.load('models/label_encoders.joblib')
            print("[LOADED] Yield prediction model loaded successfully")
        except:
            print("[WARNING] Yield prediction model not found. Will use fallback estimates.")
            self.yield_model = None
    
    def train_recommendation_model(self):
        """
        Train the crop recommendation model on your existing data.
        
        Returns:
            float: Accuracy of the trained model
        """
        X, y, features = load_and_prepare_data()
        accuracy = self.recommendation_model.train_models(X, y, features)
        self.recommendation_model.save_model()
        return accuracy
    
    def predict_yield(self, district, crop_type, area_ha, weather_data, soil_data):
        """
        Predict crop yield based on conditions and location.
        
        Args:
            district: Name of the district
            crop_type: Type of crop
            area_ha: Area in hectares
            weather_data: Dictionary with weather conditions
            soil_data: Dictionary with soil conditions
            
        Returns:
            float: Predicted yield in kg per hectare
        """
        # Use trained model if available
        if self.yield_model and self.yield_scaler and self.yield_encoders:
            try:
                # Encode categorical variables
                district_encoded = self.yield_encoders['Districts'].transform([district])[0]
                crop_encoded = self.yield_encoders['crop_type'].transform([crop_type])[0]
                
                # Prepare features in the order expected by the model
                features = np.array([[
                    crop_encoded,
                    district_encoded,
                    soil_data.get('ph_value', 6.5),
                    soil_data.get('fertilizer', 100.0),
                    weather_data.get('avg_temp', 25.0),
                    weather_data.get('max_temp', 32.0),
                    weather_data.get('min_temp', 18.0),
                    weather_data.get('humidity', 65.0),
                    weather_data.get('rainfall', 1200.0),
                    weather_data.get('solar_radiation', 6500.0),
                    weather_data.get('wind_speed', 2.5),
                    area_ha
                ]])
                
                # Normalize and predict
                features_scaled = self.yield_scaler.transform(features)
                yield_kg = self.yield_model.predict(features_scaled)[0]
                return yield_kg
                
            except Exception as e:
                print(f"Yield prediction error: {e}")
                return self._estimate_yield_fallback(crop_type)
        else:
            return self._estimate_yield_fallback(crop_type)
    
    def _estimate_yield_fallback(self, crop_type):
        """
        Provide fallback yield estimates when the trained model is unavailable.
        
        Args:
            crop_type: Type of crop
            
        Returns:
            float: Estimated yield in kg per hectare
        """
        estimates = {
            'paddy': 3500, 'rice': 4000, 'maize': 2800, 'millet': 1800,
            'buckwheat': 1200, 'potato': 18000, 'tomato': 25000,
            'onion': 15000, 'garlic': 8000, 'wheat': 2800
        }
        return estimates.get(crop_type.lower(), 2000)
    
    def calculate_profit(self, crop_name, yield_kg_per_ha, area_ha=1):
        """
        Calculate expected profit based on yield and market prices.
        
        Args:
            crop_name: Name of the crop
            yield_kg_per_ha: Expected yield in kg per hectare
            area_ha: Area in hectares
            
        Returns:
            Dictionary with profit analysis
        """
        # Market prices in NPR per kg (should be updated from live sources)
        market_prices = {
            'paddy': 35, 'rice': 50, 'maize': 31, 'millet': 27.5,
            'potato': 50, 'tomato': 65, 'onion': 55, 'garlic': 150,
            'wheat': 33, 'buckwheat': 33
        }
        
        # Production costs in NPR per kg
        production_costs = {
            'paddy': 12, 'rice': 18, 'maize': 14, 'millet': 16,
            'potato': 20, 'tomato': 25, 'onion': 22, 'garlic': 35,
            'wheat': 15, 'buckwheat': 17
        }
        
        price = market_prices.get(crop_name.lower(), 40)
        cost_per_kg = production_costs.get(crop_name.lower(), 15)
        
        total_yield = yield_kg_per_ha * area_ha
        revenue = total_yield * price
        cost = total_yield * cost_per_kg
        profit = revenue - cost
        
        return {
            'price_per_kg': price,
            'production_cost_per_kg': cost_per_kg,
            'total_yield_kg': total_yield,
            'revenue_npr': revenue,
            'cost_npr': cost,
            'profit_npr': profit,
            'roi_percent': (profit / cost) * 100 if cost > 0 else 0
        }
    
    def get_complete_advice(self, district, area_ha, weather_data, soil_data, top_n=3):
        """
        Get complete farming advice including recommendations, yield, and profit.
        
        This is the main function to use - it provides a comprehensive analysis
        of what to plant and what returns to expect.
        
        Args:
            district: Name of the district
            area_ha: Area in hectares
            weather_data: Dictionary with weather conditions
            soil_data: Dictionary with soil conditions
            top_n: Number of top recommendations to return
            
        Returns:
            List of dictionaries with complete analysis for each recommended crop
        """
        print("\n" + "="*80)
        print("INTEGRATED CROP ADVISOR")
        print("="*80)
        
        print("\nINPUT CONDITIONS:")
        print(f"   District: {district}")
        print(f"   Area: {area_ha} hectares")
        print(f"\n   Weather Conditions:")
        for key, value in weather_data.items():
            print(f"   - {key}: {value}")
        print(f"\n   Soil Conditions:")
        for key, value in soil_data.items():
            print(f"   - {key}: {value}")
        
        # Get crop recommendations from the ML model
        recommendations = self.recommendation_model.recommend_crop(
            weather_data, soil_data, area_ha, top_n
        )
        
        print("\n" + "="*80)
        print("TOP CROP RECOMMENDATIONS")
        print("="*80)
        
        full_results = []
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['crop'].upper()}")
            print(f"   Suitability: {rec['suitability_score']:.1f}%")
            
            # Predict expected yield
            predicted_yield = self.predict_yield(
                district, rec['crop'], area_ha, weather_data, soil_data
            )
            print(f"   Expected Yield: {predicted_yield:.0f} kg/ha")
            
            # Calculate profit
            profit = self.calculate_profit(rec['crop'], predicted_yield, area_ha)
            print(f"   Expected Revenue: NPR {profit['revenue_npr']:,.2f}")
            print(f"   Production Cost: NPR {profit['cost_npr']:,.2f}")
            print(f"   Net Profit: NPR {profit['profit_npr']:,.2f}")
            print(f"   ROI: {profit['roi_percent']:.1f}%")
            
            full_results.append({
                'rank': i,
                'crop': rec['crop'],
                'suitability': rec['suitability_score'],
                'yield_kg_ha': predicted_yield,
                'revenue_npr': profit['revenue_npr'],
                'cost_npr': profit['cost_npr'],
                'profit_npr': profit['profit_npr'],
                'roi_percent': profit['roi_percent']
            })
        
        # Highlight the best crop recommendation
        best = full_results[0]
        print("\n" + "="*80)
        print(f"BEST CROP FOR YOUR FARM: {best['crop'].upper()}")
        print("="*80)
        print(f"   Reason: {best['suitability']:.1f}% suitability for your conditions")
        print(f"   Expected profit: NPR {best['profit_npr']:,.2f} from {area_ha} hectares")
        print(f"   Expected ROI: {best['roi_percent']:.1f}%")
        
        return full_results


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to demonstrate the crop recommendation system.
    Trains the model and shows example recommendations.
    """
    
    print("\n" + "="*80)
    print("CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING")
    print("="*80)
    
    # Initialize the integrated advisor
    advisor = IntegratedCropAdvisor()
    
    # Train the recommendation model on your existing data
    print("\n[STEP 1] Training Crop Recommendation Model...")
    accuracy = advisor.train_recommendation_model()
    
    # Example 1: Get recommendations for specific conditions
    print("\n" + "="*80)
    print("[STEP 2] Getting Crop Recommendations")
    print("="*80)
    
    # Define weather conditions (example values)
    weather = {
        'avg_temp': 26.5,
        'max_temp': 33.0,
        'min_temp': 20.0,
        'humidity': 70.0,
        'rainfall': 1400.0,
        'solar_radiation': 6500.0,
        'wind_speed': 2.0
    }
    
    # Define soil conditions (example values)
    soil = {
        'ph_value': 6.3,
        'fertilizer': 85.0
    }
    
    # Get complete farming advice
    results = advisor.get_complete_advice(
        district="Achham",
        area_ha=2,
        weather_data=weather,
        soil_data=soil,
        top_n=5
    )
    
    # Example 2: Test different environmental scenarios
    print("\n" + "="*80)
    print("[STEP 3] Testing Different Environmental Scenarios")
    print("="*80)
    
    scenarios = [
        {
            'name': "High Rainfall Area",
            'weather': {'avg_temp': 24, 'max_temp': 30, 'min_temp': 18, 
                       'humidity': 80, 'rainfall': 2000, 
                       'solar_radiation': 6000, 'wind_speed': 2.5},
            'soil': {'ph_value': 5.8, 'fertilizer': 100}
        },
        {
            'name': "Dry Area",
            'weather': {'avg_temp': 30, 'max_temp': 38, 'min_temp': 22, 
                       'humidity': 45, 'rainfall': 600, 
                       'solar_radiation': 7000, 'wind_speed': 3.5},
            'soil': {'ph_value': 7.2, 'fertilizer': 120}
        },
        {
            'name': "Temperate Area",
            'weather': {'avg_temp': 18, 'max_temp': 24, 'min_temp': 12, 
                       'humidity': 65, 'rainfall': 1200, 
                       'solar_radiation': 5500, 'wind_speed': 1.8},
            'soil': {'ph_value': 6.8, 'fertilizer': 90}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        recommendations = advisor.recommendation_model.recommend_crop(
            scenario['weather'], scenario['soil'], top_n=3
        )
        print("Top recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['crop']} ({rec['suitability_score']:.1f}% suitability)")
    
    # Example 3: Check suitability for specific crops
    print("\n" + "="*80)
    print("[STEP 4] Checking Specific Crop Suitability")
    print("="*80)
    
    test_crops = ['paddy', 'maize', 'potato', 'tomato']
    for crop in test_crops:
        suitability = advisor.recommendation_model.get_crop_suitability(
            crop, weather, soil
        )
        print(f"{crop.capitalize()}: {suitability['suitability_score']:.1f}% suitable")
    
    print("\n" + "="*80)
    print("CROP RECOMMENDATION SYSTEM READY!")
    print("="*80)
    print("\nHow to use this system:")
    print("1. Model saved in 'models/crop_recommendation_model.joblib'")
    print("2. Use advisor.get_complete_advice() for full farm analysis")
    print("3. Use advisor.recommendation_model.recommend_crop() for quick recommendations")
    print("4. Use advisor.recommendation_model.compare_crops() to see all options")


if __name__ == "__main__":
    main()