from flask import Blueprint
from pathlib import Path

main_bp = Blueprint('main_api', __name__)

BASE_DIR = Path(__file__).resolve().parents[2]
from app.services.prediction_service import predictVolatility, predictRisk
from app.services.clustering_service import predictRiskCluster
from app.services.sentiment_analysis_service import sentiment_prediction

@main_bp.route('/volatility')
def volatility():
    return predictVolatility()

@main_bp.route('/risk_volatility')
def risk_volatility():
    return predictRisk()

@main_bp.route('/risk_volatility_cluster')
def risk_volatility_cluster():
    return predictRiskCluster()

@main_bp.route('/sentiment_analysis_route/<stock>', methods=['GET'])
def sentiment_prediction_api(stock):
    return sentiment_prediction(stock=stock)