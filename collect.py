import os
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

# Dataset URL
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"

def collect_data():
    print("Downloading dataset...")
    df = pd.read_csv(url)
    print(df.head())
    df.to_csv("data/raw/housing.csv", index=False)
    print(f"Data saved to data/raw/housing.csv with {len(df)} rows.")

if __name__ == "__main__":
    collect_data()