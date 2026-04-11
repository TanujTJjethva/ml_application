from pathlib import Path
import numpy as np
from flask import jsonify
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
from app.data.feature_engineering import final_mapped_data, risk_final_mapped_data

def predictVolatility():
    df = final_mapped_data()
    df = df.groupby('Company').tail(1)
    X = df.drop(columns=['target_volatility', 'Date', 'next_high', 'next_low', 'stock'])

    # Load model
    model = joblib.load('models/volatile_model.pkl')

    # Predict
    predictions = model.predict(X)

    # Attach predictions to output
    df['predicted_volatility'] = predictions

    return jsonify(df.to_dict(orient='records'))

def predictRisk():
    df = risk_final_mapped_data()
    df = df.groupby('Company').tail(1)
    X = df.drop(columns=['Date', 'next_high', 'next_low', 'volatility_pct', 'target_volatility_risk', 'stock'])

    # Load model
    model = joblib.load('models/risk_volatile_model.pkl')

    # Predict
    predictions = model.predict(X)

    # Load label encoder
    decode_class = joblib.load('models/risk_volatile_label_encoder.pkl')

    # Attach predictions to output
    df['predicted_volatility'] = decode_class.inverse_transform(predictions)

    return jsonify(df.to_dict(orient='records'))