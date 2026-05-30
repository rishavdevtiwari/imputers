import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data\processed\cleaned_soil_crop.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Encode target
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Identify columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_cols
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_cols
    )
])

# Transform data
X_processed = preprocessor.fit_transform(X)

# Feature names
encoded_features = preprocessor.named_transformers_[
    "cat"
].named_steps["encoder"].get_feature_names_out(
    categorical_cols
)

all_features = numeric_cols + list(encoded_features)

# Model
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_processed, y)

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 20 Important Features\n")
print(importance_df.head(20))

importance_df.to_csv(
    "processed/feature_importance.csv",
    index=False
)

print(
    "\nSaved: processed/feature_importance.csv"
)