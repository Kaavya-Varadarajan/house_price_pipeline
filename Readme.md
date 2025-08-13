
<p align="left">
   <img src="icon_house.png" alt="House Icon" width="60" />
</p>

# House Price Prediction Pipeline

This project provides a comprehensive, end-to-end machine learning pipeline for predicting house prices. The pipeline covers all stages from data collection and preprocessing to model training, evaluation, and visualization.

## Features
- **Data Collection:** Automated retrieval of housing data.
- **ETL (Extract, Transform, Load):** Data cleaning and preprocessing for model readiness.
- **Model Training:** Supports Random Forest, XGBoost, and LightGBM algorithms.
- **Model Comparison:** Evaluate and compare model performance using MAE and training time.
- **Visualization:** Generate insightful plots and dashboards for data and model results.

## Getting Started

### Prerequisites
- Python 3.7+
- Recommended: Create and activate a virtual environment

### Installation
1. Create a virtual environment:
   - Windows: `python -m venv venv && venv\Scripts\activate`
   - macOS/Linux: `python -m venv venv && source venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`

### Running the Pipeline
1. **Data Collection:**
   - `python collect.py`
2. **ETL (Preprocessing):**
   - `python etl.py`
3. **Model Training:**
   - `python train.py`  
     (You may modify the script to use RandomForest, XGBoost, or LightGBM as needed.)
4. **Visualization:**
   - `python visualize.py`

### Output
- Processed data and results are saved in the `data/processed/` directory.
- Visualizations and model comparison plots are also saved in this directory.

## Project Structure
```
house_price_pipeline/
├── collect.py
├── etl.py
├── train.py
├── visualize.py
├── datasets_ames_housing.py
├── data/
│   ├── raw/
│   │   └── housing.csv
│   └── processed/
│       ├── housing_processed.csv
│       ├── model_comparison_mae.png
│       ├── model_comparison_time.png
│       └── price_distribution.png
├── requirement.txt.txt
├── LICENSE
└── Readme.md
```

## Author
**Kaavya Varadarajan**
