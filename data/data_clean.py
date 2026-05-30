import pandas as pd
import os

os.makedirs("processed", exist_ok=True)

df = pd.read_csv("ra/soilPHandcrop.csv")

print("Original Shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Standardize column names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace("-", "_")
)

# Fill missing values

numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

if "soilcolor" in df.columns:
    df["soilcolor"] = df["soilcolor"].fillna(
        df["soilcolor"].mode()[0]
    )

output_file = "processed/cleaned_soil_crop.csv"

df.to_csv(output_file, index=False)

print("Saved:", output_file)
print(df.head())