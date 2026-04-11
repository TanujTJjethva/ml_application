from flask import Blueprint
from pathlib import Path
from flask import render_template

frontend_dashboard = Blueprint('ui', __name__)

BASE_DIR = Path(__file__).resolve().parents[2]

from app.config import env_type

@frontend_dashboard.route('/')
def index():

    import sys
    import pip
    import flask
    import joblib
    import pandas
    import yfinance
    import sklearn
    import matplotlib
    import seaborn
    import xgboost
    import tensorflow

    version_data = {
        'Environment': env_type(),
        'pip': pip.__version__,
        'Python': sys.version.split()[0],
        'Flask': flask.__version__,
        'joblib': joblib.__version__,
        'pandas': pandas.__version__,
        'yfinance': yfinance.__version__,
        'scikit-learn': sklearn.__version__,
        'matplotlib': matplotlib.__version__,
        'seaborn': seaborn.__version__,
        'xgboost': xgboost.__version__,
        'tensorflow': tensorflow.__version__
    }

    return render_template('dashboard.html', version_data=version_data)