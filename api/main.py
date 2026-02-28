"""
FastAPI Inference Endpoint — Serve flight delay predictions.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve project root robustly
_this_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_this_dir)
if not os.path.isdir(os.path.join(PROJECT_ROOT, "models")):
    PROJECT_ROOT = os.getcwd()
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
PREDICTION_LOG = os.path.join(PROJECT_ROOT, "monitoring", "prediction_log.csv")

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predict whether a flight will be delayed (>15 min arrival delay)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model ──────────────────────────────────────────────
model = None
scaler = None
feature_cols = None


def load_model():
    global model, scaler, feature_cols
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FEATURE_COLS_PATH, 'rb') as f:
            feature_cols = pickle.load(f)
        logger.info(f"✅ Model loaded: {type(model).__name__}")
        logger.info(f"   Features: {len(feature_cols)}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")


@app.on_event("startup")
def startup():
    load_model()


# ─── Request/Response Models ─────────────────────────────────
class FlightInput(BaseModel):
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    day_of_week: int = Field(..., ge=1, le=7, description="Day of week (1=Mon, 7=Sun)")
    dep_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0-23)")
    dep_minute: int = Field(0, ge=0, le=59, description="Scheduled departure minute (0-59)")
    carrier: str = Field(..., description="Airline carrier code (e.g., AA, DL, UA)")
    origin: str = Field(..., description="Origin airport code (e.g., ATL, JFK)")
    dest: str = Field(..., description="Destination airport code (e.g., LAX, ORD)")
    distance: int = Field(..., ge=50, le=6000, description="Flight distance in miles")
    dep_delay: float = Field(0, description="Departure delay in minutes")
    taxi_out: int = Field(15, ge=0, description="Taxi-out time in minutes")
    crs_elapsed_time: int = Field(None, description="Scheduled elapsed time in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "month": 7,
                "day_of_month": 15,
                "day_of_week": 3,
                "dep_hour": 14,
                "dep_minute": 30,
                "carrier": "AA",
                "origin": "JFK",
                "dest": "LAX",
                "distance": 2475,
                "dep_delay": 5,
                "taxi_out": 18,
                "crs_elapsed_time": 330
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    delay_probability: float
    is_delayed: int
    confidence: float
    model_type: str
    features_used: int
    timestamp: str


# ─── Carrier encoding map ────────────────────────────────────
CARRIER_MAP = {
    'AA': 0, 'AS': 1, 'B6': 2, 'DL': 3, 'F9': 4,
    'G4': 5, 'HA': 6, 'NK': 7, 'UA': 8, 'WN': 9
}

AIRPORT_LIST = [
    'ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS',
    'PHX', 'MIA', 'SEA', 'IAH', 'JFK', 'EWR', 'SFO', 'MSP',
    'BOS', 'DTW', 'FLL', 'PHL', 'LGA', 'BWI', 'SLC', 'SAN',
    'IAD', 'DCA', 'MDW', 'TPA', 'PDX', 'HNL'
]


def prepare_features(input_data: FlightInput) -> np.ndarray:
    """Convert API input into model feature vector."""
    # Base features
    features = {
        'Month': input_data.month,
        'DayofMonth': input_data.day_of_month,
        'DayOfWeek': input_data.day_of_week,
        'dep_hour': input_data.dep_hour,
        'dep_minute': input_data.dep_minute,
        'carrier_encoded': CARRIER_MAP.get(input_data.carrier.upper(), 0),
        'origin_freq': 1.0 / len(AIRPORT_LIST),  # Uniform approximation
        'dest_freq': 1.0 / len(AIRPORT_LIST),
        'Distance': input_data.distance,
        'log_distance': np.log1p(input_data.distance),
        'is_weekend': 1 if input_data.day_of_week >= 6 else 0,
        'dep_delay_flag': 1 if input_data.dep_delay > 0 else 0,
        'DepDelay': input_data.dep_delay,
        'TaxiOut': input_data.taxi_out,
        'CRSElapsedTime': input_data.crs_elapsed_time or int(input_data.distance / 8 + 30),
    }

    # Time of day one-hot
    hour = input_data.dep_hour
    features['time_of_day_morning'] = 1 if 6 < hour <= 12 else 0
    features['time_of_day_afternoon'] = 1 if 12 < hour <= 18 else 0
    features['time_of_day_evening'] = 1 if 18 < hour <= 24 else 0

    # Season one-hot
    month = input_data.month
    if month in [3, 4, 5]:
        features['season_spring'] = 1
    else:
        features['season_spring'] = 0
    if month in [6, 7, 8]:
        features['season_summer'] = 1
    else:
        features['season_summer'] = 0
    if month in [9, 10, 11]:
        features['season_fall'] = 1
    else:
        features['season_fall'] = 0

    # Distance group one-hot
    dist = input_data.distance
    features['distance_group_medium'] = 1 if 500 < dist <= 1000 else 0
    features['distance_group_long'] = 1 if 1000 < dist <= 2000 else 0
    features['distance_group_ultra_long'] = 1 if dist > 2000 else 0

    # Build feature vector in correct order
    if feature_cols is not None:
        feature_vector = []
        for col in feature_cols:
            feature_vector.append(features.get(col, 0))
        return np.array([feature_vector])
    else:
        return np.array([list(features.values())])


def log_prediction(input_data: FlightInput, prediction: int, probability: float):
    """Log prediction to CSV for monitoring."""
    os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'carrier': input_data.carrier,
        'origin': input_data.origin,
        'dest': input_data.dest,
        'month': input_data.month,
        'dep_hour': input_data.dep_hour,
        'distance': input_data.distance,
        'dep_delay': input_data.dep_delay,
        'prediction': prediction,
        'probability': round(probability, 4)
    }

    # Append to CSV
    df = pd.DataFrame([log_entry])
    header = not os.path.exists(PREDICTION_LOG)
    df.to_csv(PREDICTION_LOG, mode='a', header=header, index=False)


# ─── API Endpoints ───────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Flight Delay Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": type(model).__name__ if model else None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightInput):
    """Predict flight delay probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        X = prepare_features(flight)

        # Scale features
        X_scaled = scaler.transform(X) if scaler else X

        # Predict
        prediction = int(model.predict(X_scaled)[0])
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X_scaled)[0][1])
        else:
            probability = float(prediction)

        confidence = max(probability, 1 - probability)

        # Log prediction
        try:
            log_prediction(flight, prediction, probability)
        except Exception as e:
            logger.warning(f"Prediction logging failed: {e}")

        return PredictionResponse(
            prediction="DELAYED" if prediction == 1 else "ON TIME",
            delay_probability=round(probability, 4),
            is_delayed=prediction,
            confidence=round(confidence, 4),
            model_type=type(model).__name__,
            features_used=len(feature_cols) if feature_cols else 0,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = {
        "model_type": type(model).__name__,
        "features": feature_cols if feature_cols else [],
        "num_features": len(feature_cols) if feature_cols else 0,
        "carriers_supported": list(CARRIER_MAP.keys()),
        "airports_supported": AIRPORT_LIST,
    }

    # Add model-specific info
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        info['max_depth'] = model.max_depth
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if feature_cols:
            info['top_features'] = dict(sorted(
                zip(feature_cols, importances.tolist()),
                key=lambda x: x[1], reverse=True
            )[:10])

    return info


@app.get("/predictions/history")
def prediction_history(limit: int = 50):
    """Get recent prediction history."""
    if not os.path.exists(PREDICTION_LOG):
        return {"predictions": [], "total": 0}

    df = pd.read_csv(PREDICTION_LOG)
    total = len(df)
    recent = df.tail(limit).to_dict(orient='records')

    return {"predictions": recent, "total": total}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
