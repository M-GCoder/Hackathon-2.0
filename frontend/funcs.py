import sys
import os
import pickle
import json
import subprocess
import time
import socket
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Ensure api logic is reachable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = os.environ.get("API_URL", "http://localhost:8000")

CARRIERS = {
    'AA': 'American Airlines', 'DL': 'Delta Air Lines',
    'UA': 'United Airlines', 'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways', 'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines', 'F9': 'Frontier Airlines',
    'G4': 'Allegiant Air', 'HA': 'Hawaiian Airlines'
}

AIRPORTS = [
    'ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS',
    'PHX', 'MIA', 'SEA', 'IAH', 'JFK', 'EWR', 'SFO', 'MSP',
    'BOS', 'DTW', 'FLL', 'PHL', 'LGA', 'BWI', 'SLC', 'SAN',
    'IAD', 'DCA', 'MDW', 'TPA', 'PDX', 'HNL'
]

def start_backend():
    """Start FastAPI in the background if not running."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', 8000)) != 0:
                print("🚀 Starting FastAPI backend...")
                subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])
                time.sleep(5) # Wait for startup
    except Exception as e:
        print(f"Error starting backend: {e}")

def generate_shap_explanation(input_data: dict):
    """Generate SHAP explanation for a prediction."""
    try:
        import shap
        sys.path.insert(0, PROJECT_ROOT) # Make sure api logic is reachable

        model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
        scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
        feature_cols_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "flights_processed.csv")

        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_cols_path, data_path]):
            return None, None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(feature_cols_path, 'rb') as f:
            feature_cols = pickle.load(f)

        # Prepare input features
        from api.main import CARRIER_MAP, prepare_features
        from pydantic import BaseModel

        class FI(BaseModel):
            month: int = input_data.get('month', 1)
            day_of_month: int = input_data.get('day_of_month', 1)
            day_of_week: int = input_data.get('day_of_week', 1)
            dep_hour: int = input_data.get('dep_hour', 12)
            dep_minute: int = input_data.get('dep_minute', 0)
            carrier: str = input_data.get('carrier', 'AA')
            origin: str = input_data.get('origin', 'ATL')
            dest: str = input_data.get('dest', 'LAX')
            distance: int = input_data.get('distance', 1000)
            dep_delay: float = input_data.get('dep_delay', 0)
            taxi_out: int = input_data.get('taxi_out', 15)
            crs_elapsed_time: int = input_data.get('crs_elapsed_time', None)

        fi = FI()
        X = prepare_features(fi)
        X_scaled = scaler.transform(X)

        # Get SHAP values using TreeExplainer or KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        except Exception:
            # Fallback to KernelExplainer with background sample
            bg_data = pd.read_csv(data_path)
            exclude = ['Year', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest',
                       'ArrDelay', 'Flight_Number_Reporting_Airline', 'is_delayed']
            bg_features = bg_data[[c for c in bg_data.columns if c not in exclude]].head(100)
            bg_scaled = scaler.transform(bg_features)
            explainer = shap.KernelExplainer(model.predict_proba, bg_scaled[:50])
            shap_values = explainer.shap_values(X_scaled)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1]).flatten()  # Class 1 (delayed)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim > 1:
            sv = shap_values[0].flatten()
        else:
            sv = np.array(shap_values).flatten()

        # Ensure sv length matches feature_cols
        if len(sv) != len(feature_cols):
            sv = sv[:len(feature_cols)]

        return sv, feature_cols
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
        return None, None
