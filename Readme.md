# house_price_pipeline

End-to-end mini pipeline: data collection → ETL → model training (RandomForest / XGBoost / LightGBM) → visualization.

## Run locally
1. Create venv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Pipeline:
   - `python scripts/collect.py`
   - `python scripts/etl.py`
   - `python scripts/train.py`  # or xgb/lgb versions
   - `python scripts/compare_models.py`
4. Visualize: `python scripts/visualize.py` or `streamlit run scripts/compare_dashboard.py`
