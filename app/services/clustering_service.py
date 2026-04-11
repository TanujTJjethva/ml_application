from pathlib import Path
import numpy as np
from flask import jsonify
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
from app.data.feature_engineering import final_mapped_data, risk_final_mapped_data

def predictRiskCluster():
    df = final_mapped_data()

    # Keep Company separately
    df = df.groupby('Company').tail(1)
    companies = df['Company'].values

    # Features only
    X = df.drop(columns=['target_volatility', 'Date', 'next_high', 'next_low', 'stock'])

    # Load model
    pipeline = joblib.load('models/clusteringmodel.pkl')

    clusters = pipeline.predict(X)

    # Attach results
    result = pd.DataFrame({
        'Company': companies,
        'cluster': clusters
    })

    return jsonify(result.to_dict(orient='records'))