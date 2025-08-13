import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import time
import matplotlib.pyplot as plt

# Make sure xgboost and lightgbm are installed: pip install xgboost lightgbm
import xgboost as xgb
import lightgbm as lgb

os.makedirs("models", exist_ok=True)

def train_models():
    df = pd.read_csv("data/processed/housing_processed.csv")

    # One-hot encode and clean column names for XGBoost compatibility
    df = pd.get_dummies(df, columns=["ocean_proximity"])
    df.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace(' ', '_') for col in df.columns]

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # Train Random Forest model
    start = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    print(f"Random Forest Model MAE: {rf_mae:.2f}")
    joblib.dump(rf_model, "models/house_price_model.joblib")
    print("Random Forest model saved to models/house_price_model.joblib")
    results.append({"Model": "Random Forest", "MAE": rf_mae, "Train Time (s)": rf_time})

    # Train XGBoost model
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    print(f"XGBoost Model MAE: {xgb_mae:.2f}")
    joblib.dump(xgb_model, "models/house_price_model_xgb.joblib")
    print("XGBoost model saved to models/house_price_model_xgb.joblib")
    results.append({"Model": "XGBoost", "MAE": xgb_mae, "Train Time (s)": xgb_time})

    # Train LightGBM model
    start = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    lgb_time = time.time() - start
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"LightGBM Model MAE: {mae:.2f}")
    joblib.dump(model, "models/house_price_model_lgb.joblib")
    print("LightGBM model saved to models/house_price_model_lgb.joblib")
    results.append({"Model": "LightGBM", "MAE": mae, "Train Time (s)": lgb_time})

    # Show results table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(results_df)

    # Plot MAE comparison
    plt.figure(figsize=(8,5))
    plt.bar(results_df["Model"], results_df["MAE"], color=["skyblue", "orange", "green"])
    plt.ylabel("Mean Absolute Error (lower is better)")
    plt.title("Model Performance Comparison")
    plt.savefig("data/processed/model_comparison_mae.png")
    plt.show()

    # Plot training time comparison
    plt.figure(figsize=(8,5))
    plt.bar(results_df["Model"], results_df["Train Time (s)"], color=["skyblue", "orange", "green"])
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison")
    plt.savefig("data/processed/model_comparison_time.png")
    plt.show()
if __name__ == "__main__":
    train_models()