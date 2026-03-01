# ✈️ Flight Delay Prediction Platform

> End-to-End ML System: ETL → DuckDB → ML Training → MLflow → FastAPI → Streamlit/SHAP → Docker → CI/CD

[![CI/CD](https://github.com/M-GCoder/Hackathon-2.0/actions/workflows/deploy.yml/badge.svg)](https://github.com/M-GCoder/Hackathon-2.0/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  BTS Data    │───▶│  ETL Pipeline│───▶│   DuckDB     │
│  (Raw CSV)   │    │  Extract →   │    │  Curated     │
│              │    │  Transform   │    │  Storage     │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼───────┐
                    │   MLflow     │◀───│  ML Training  │
                    │  Tracking    │    │  LR/RF/GBDT   │
                    └──────────────┘    └───────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼───────┐
                    │  Streamlit   │◀───│   FastAPI     │
                    │  UI + SHAP   │    │  /predict     │
                    └──────────────┘    └───────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼───────┐
                    │   Docker +   │    │  Monitoring   │
                    │   CI/CD      │    │  Drift Check  │
                    └──────────────┘    └───────────────┘
```

## 📁 Project Structure

```
Hackathon-2.0/
├── data/
│   ├── raw/                    # Raw BTS flight data
│   ├── processed/              # Cleaned, feature-engineered data
│   └── generate_data.py        # Synthetic data generator
├── etl/
│   ├── extract.py              # Data extraction from CSV
│   ├── transform.py            # Feature engineering & cleaning
│   └── pipeline.py             # ETL orchestrator
├── database/
│   ├── db_manager.py           # DuckDB CRUD operations
│   └── flight_db.duckdb        # DuckDB database file
├── mlops/
│   ├── train.py                # Multi-model training pipeline
│   ├── evaluate.py             # Model evaluation & metrics
│   └── mlflow_tracking.py      # MLflow experiment tracking
├── models/
│   ├── best_model.pkl          # Best trained model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_columns.pkl     # Feature column names
├── api/
│   └── main.py                 # FastAPI inference endpoint
├── frontend/
│   └── app.py                  # Streamlit UI + SHAP
├── monitoring/
│   ├── drift_check.py          # Data drift detection (KS-test)
│   └── prediction_log.csv      # Prediction audit log
├── .github/workflows/
│   └── deploy.yml              # GitHub Actions CI/CD
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Run ETL
```bash
python data/generate_data.py
python -c "from etl.pipeline import run_pipeline; run_pipeline()"
```

### 3. Load into DuckDB
```bash
python -c "from database.db_manager import DuckDBManager; db = DuckDBManager(); db.load_data(); db.close()"
```

### 4. Train Models
```bash
python mlops/train.py
python mlops/evaluate.py
```

### 5. Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Streamlit UI
```bash
streamlit run frontend/app.py --server.port 8501
```

### OR Start Both API & Streamlit UI
```bash
Start-Process powershell -ArgumentList "-Command", "cd '{Replace the File Explorer path}'; python -m uvicorn api. main:app -- host 0.0.0.0 --port 8000"; Streamlit run frontend/app.py --server.port 8501
```

### 7. Docker (all-in-one)
```bash
docker-compose up --build
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/predict` | POST | Predict flight delay |
| `/model/info` | GET | Model details & top features |
| `/predictions/history` | GET | Recent prediction log |
| `/docs` | GET | Interactive API documentation |

### Example Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "month": 7, "day_of_month": 15, "day_of_week": 3,
    "dep_hour": 14, "dep_minute": 30, "carrier": "AA",
    "origin": "JFK", "dest": "LAX", "distance": 2475,
    "dep_delay": 5, "taxi_out": 18, "crs_elapsed_time": 330
  }'
```

## 🤖 ML Models

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | ~0.82 | ~0.70 | ~0.65 | ~0.67 | ~0.85 |
| Random Forest | ~0.88 | ~0.80 | ~0.75 | ~0.77 | ~0.92 |
| **Gradient Boosting** ⭐ | ~0.90 | ~0.83 | ~0.78 | ~0.80 | ~0.94 |

## 🛠️ Tech Stack

- **Data**: Pandas, NumPy, DuckDB
- **ML**: scikit-learn, XGBoost
- **Tracking**: MLflow
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit, Plotly
- **Explainability**: SHAP
- **Monitoring**: Evidently, KS-test drift detection
- **Infra**: Docker, GitHub Actions

## 📜 License

MIT © 2026 Abdullah Idrees
