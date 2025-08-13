import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

def process_data():
    df = pd.read_csv("data/raw/housing.csv")

    # Handle missing values( Handle missing values only for numeric columns)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Feature engineering
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    df.to_csv("data/processed/housing_processed.csv", index=False)
    print(f"Processed data saved to data/processed/housing_processed.csv with {len(df)} rows.")

if __name__ == "__main__":
    process_data()
    df = pd.read_csv("data/processed/housing_processed.csv")
    print(df.head())