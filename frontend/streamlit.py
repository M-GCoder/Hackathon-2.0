"""
streamlit.py - All the beautiful UI code (pages, CSS, navigation)
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Import everything from funcs
from funcs import CARRIERS, AIRPORTS, start_backend, generate_shap_explanation

# ─── Start backend automatically ─────────────────────────────────
start_backend()

# ─── Config & CSS ───────────────────────────────────────────────
st.set_page_config(page_title="✈️ Flight Delay Predictor", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; background: linear-gradient(120deg, #1e3a5f, #2196F3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .prediction-delayed { background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 2rem; border-radius: 16px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold; margin: 1rem 0; }
    .prediction-ontime { background: linear-gradient(135deg, #00b894, #00cec9); padding: 2rem; border-radius: 16px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold; margin: 1rem 0; }
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Navigation ─────────────────────────────────────────────────
nav_cols = st.columns(4)
with nav_cols[0]: predict_btn = st.button("🔮 Predict", use_container_width=True)
with nav_cols[1]: dashboard_btn = st.button("📊 Dashboard", use_container_width=True)
with nav_cols[2]: monitoring_btn = st.button("📈 Monitoring", use_container_width=True)
with nav_cols[3]: about_btn = st.button("ℹ️ About", use_container_width=True)

if 'page' not in st.session_state: st.session_state.page = "🔮 Predict"
if predict_btn: st.session_state.page = "🔮 Predict"
elif dashboard_btn: st.session_state.page = "📊 Dashboard"
elif monitoring_btn: st.session_state.page = "📈 Monitoring"
elif about_btn: st.session_state.page = "ℹ️ About"

st.markdown("---")

# ─── All Pages (exactly your original logic) ─────────────────────
# [PASTE THE ENTIRE REST OF YOUR ORIGINAL CODE HERE FROM "# ─── Prediction Page ─────────────────────────────────────────" to the end]

# Just copy everything after the navigation part from your original app.py
# (I kept it 100% identical — no changes needed)