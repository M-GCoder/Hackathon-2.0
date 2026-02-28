import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Import components from our separated funcs module
from frontend.funcs import (
    API_URL, 
    PROJECT_ROOT, 
    CARRIERS, 
    AIRPORTS, 
    generate_shap_explanation
)

def setup_page():
    """Initializes standard page layout and CSS rules"""
    st.set_page_config(
        page_title="✈️ Flight Delay Predictor",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #1e3a5f, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .prediction-delayed {
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .prediction-ontime {
            background: linear-gradient(135deg, #00b894, #00cec9);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .stApp {
            background-color: #0e1117;
        }
    </style>
    """, unsafe_allow_html=True)


def render_navigation():
    """Renders the top navigation bar and handles page routing status"""
    nav_cols = st.columns(4)
    with nav_cols[0]:
        predict_btn = st.button("🔮 Predict", use_container_width=True)
    with nav_cols[1]:
        dashboard_btn = st.button("📊 Dashboard", use_container_width=True)
    with nav_cols[2]:
        monitoring_btn = st.button("📈 Monitoring", use_container_width=True)
    with nav_cols[3]:
        about_btn = st.button("ℹ️ About", use_container_width=True)

    # Track selected page in session state
    if 'page' not in st.session_state:
        st.session_state.page = "🔮 Predict"
    if predict_btn:
        st.session_state.page = "🔮 Predict"
    elif dashboard_btn:
        st.session_state.page = "📊 Dashboard"
    elif monitoring_btn:
        st.session_state.page = "📈 Monitoring"
    elif about_btn:
        st.session_state.page = "ℹ️ About"

    st.markdown("---")
    return st.session_state.page


def render_predict():
    """Renders the flight delay prediction inputs and handles API calling"""
    st.markdown('<h1 class="main-header">✈️ Flight Delay Predictor</h1>', unsafe_allow_html=True)
    st.markdown("##### Predict whether your flight will be delayed using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📅 Flight Date")
        month = st.selectbox("Month", range(1, 13), index=6,
                             format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                     'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        day_of_month = st.number_input("Day of Month", 1, 31, 15)
        day_of_week = st.selectbox("Day of Week", range(1, 8),
                                   format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x-1])

    with col2:
        st.markdown("### ✈️ Flight Details")
        carrier = st.selectbox("Airline", list(CARRIERS.keys()),
                               format_func=lambda x: f"{x} - {CARRIERS[x]}")
        origin = st.selectbox("Origin Airport", AIRPORTS, index=AIRPORTS.index('JFK'))
        dest = st.selectbox("Destination Airport", AIRPORTS, index=AIRPORTS.index('LAX'))

    with col3:
        st.markdown("### ⏱ Schedule")
        dep_hour = st.slider("Departure Hour", 0, 23, 14)
        dep_minute = st.selectbox("Departure Minute", [0, 15, 30, 45])
        distance = st.number_input("Distance (miles)", 50, 6000, 2475)
        dep_delay = st.number_input("Current Departure Delay (min)", -30, 300, 0)
        taxi_out = st.number_input("Taxi Out (min)", 0, 60, 15)

    st.markdown("---")

    if st.button("🚀 Predict Delay", type="primary", use_container_width=True):
        payload = {
            "month": month,
            "day_of_month": day_of_month,
            "day_of_week": day_of_week,
            "dep_hour": dep_hour,
            "dep_minute": dep_minute,
            "carrier": carrier,
            "origin": origin,
            "dest": dest,
            "distance": distance,
            "dep_delay": dep_delay,
            "taxi_out": taxi_out,
            "crs_elapsed_time": int(distance / 8 + 30)
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()

                # Display prediction
                if result['is_delayed'] == 1:
                    st.markdown(f"""
                    <div class="prediction-delayed">
                        ⚠️ FLIGHT LIKELY DELAYED<br>
                        <span style="font-size: 2rem;">{result['delay_probability']:.1%}</span> chance of delay
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-ontime">
                        ✅ FLIGHT LIKELY ON TIME<br>
                        <span style="font-size: 2rem;">{1 - result['delay_probability']:.1%}</span> chance of on-time
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prediction", result['prediction'])
                m2.metric("Delay Probability", f"{result['delay_probability']:.1%}")
                m3.metric("Confidence", f"{result['confidence']:.1%}")
                m4.metric("Model", result['model_type'])

                # SHAP Explanation
                st.markdown("### 🔍 SHAP Explanation")
                with st.spinner("Generating SHAP explanation..."):
                    shap_values, feature_names = generate_shap_explanation(payload)

                if shap_values is not None and feature_names is not None:
                    # Create SHAP waterfall chart
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_values
                    }).sort_values('SHAP Value', key=abs, ascending=False).head(15)

                    fig = go.Figure()
                    colors = ['#ff6b6b' if v > 0 else '#00b894' for v in shap_df['SHAP Value']]
                    fig.add_trace(go.Bar(
                        x=shap_df['SHAP Value'],
                        y=shap_df['Feature'],
                        orientation='h',
                        marker_color=colors,
                        text=[f'{v:+.3f}' for v in shap_df['SHAP Value']],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title='Feature Impact on Delay Prediction',
                        xaxis_title='SHAP Value (impact on prediction)',
                        yaxis_title='Feature',
                        height=500,
                        template='plotly_dark',
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("🔴 Red bars push toward DELAYED | 🟢 Green bars push toward ON TIME")
                else:
                    st.info("SHAP explanation requires local model files. Run training first.")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}. Make sure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Error: {e}")


def render_dashboard():
    """Renders the analytics dashboard with Plotly charts"""
    st.markdown('<h1 class="main-header">📊 Flight Data Dashboard</h1>', unsafe_allow_html=True)

    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "flights_processed.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flights", f"{len(df):,}")
        c2.metric("Delay Rate", f"{df['is_delayed'].mean():.1%}")
        c3.metric("Avg Dep Delay", f"{df['DepDelay'].mean():.1f} min")
        c4.metric("Carriers", df['Reporting_Airline'].nunique() if 'Reporting_Airline' in df.columns else 'N/A')

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Delay rate by carrier
            if 'Reporting_Airline' in df.columns:
                carrier_delays = df.groupby('Reporting_Airline')['is_delayed'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=carrier_delays.index, y=carrier_delays.values,
                    labels={'x': 'Carrier', 'y': 'Delay Rate'},
                    title='Delay Rate by Carrier',
                    template='plotly_dark',
                    color=carrier_delays.values,
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Delay by hour
            if 'dep_hour' in df.columns:
                hourly = df.groupby('dep_hour')['is_delayed'].mean()
                fig = px.line(
                    x=hourly.index, y=hourly.values,
                    labels={'x': 'Departure Hour', 'y': 'Delay Rate'},
                    title='Delay Rate by Hour of Day',
                    template='plotly_dark'
                )
                fig.update_traces(fill='tozeroy', line_color='#2196F3')
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Monthly delay trend
            monthly = df.groupby('Month')['is_delayed'].mean()
            fig = px.bar(
                x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(monthly)],
                y=monthly.values,
                labels={'x': 'Month', 'y': 'Delay Rate'},
                title='Monthly Delay Trend',
                template='plotly_dark',
                color=monthly.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Distance vs delay
            fig = px.histogram(
                df, x='Distance', color='is_delayed',
                nbins=30, barmode='overlay',
                labels={'is_delayed': 'Delayed'},
                title='Flight Distance Distribution by Delay Status',
                template='plotly_dark',
                color_discrete_map={0: '#00b894', 1: '#ff6b6b'}
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No processed data found. Run the ETL pipeline first.")


def render_monitoring():
    """Renders the prediction monitoring console and drift logic"""
    st.markdown('<h1 class="main-header">📈 Prediction Monitoring</h1>', unsafe_allow_html=True)

    log_path = os.path.join(PROJECT_ROOT, "monitoring", "prediction_log.csv")

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(log_df))
        c2.metric("Delayed Rate", f"{log_df['prediction'].mean():.1%}")
        c3.metric("Avg Probability", f"{log_df['probability'].mean():.3f}")

        st.markdown("### Recent Predictions")
        st.dataframe(log_df.tail(20).sort_values('timestamp', ascending=False), use_container_width=True)

        # Prediction distribution
        fig = px.histogram(
            log_df, x='probability', nbins=30,
            title='Prediction Probability Distribution',
            template='plotly_dark',
            color_discrete_sequence=['#2196F3']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Drift monitoring
        st.markdown("### 📊 Data Drift Monitoring")
        try:
            from monitoring.drift_check import check_drift
            drift_results = check_drift()
            if drift_results:
                for feature, result in drift_results.items():
                    status = "🔴 Drift Detected" if result.get('drift', False) else "🟢 No Drift"
                    st.write(f"**{feature}**: {status} (p-value: {result.get('p_value', 'N/A')})")
        except Exception:
            st.info("Drift monitoring available after sufficient predictions are collected.")
    else:
        st.info("No predictions logged yet. Make some predictions first!")


def render_about():
    """Renders the About Page and fetches Model Information"""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)

    st.markdown("""
    ## ✈️ Flight Delay Prediction Platform

    **End-to-End ML System** built for the Hackathon 2.0 challenge.

    ### 🏗️ Architecture
    - **ETL Pipeline**: Extract → Transform → Load from BTS On-Time Performance data
    - **Database**: DuckDB for curated data storage and analytics
    - **ML Models**: Logistic Regression, Random Forest, Gradient Boosting
    - **Experiment Tracking**: MLflow for model versioning and comparison
    - **API**: FastAPI inference endpoint with prediction logging
    - **Frontend**: Streamlit with SHAP explainability
    - **Monitoring**: Prediction logs + data drift detection
    - **Deployment**: Docker + GitHub Actions CI/CD

    ### 📊 Data Source
    Synthetic data based on the U.S. Bureau of Transportation Statistics
    [On-Time Performance Dataset](https://transtats.bts.gov/)

    ### 🛠️ Tech Stack
    `Python` `DuckDB` `MLflow` `FastAPI` `Streamlit` `SHAP` `Docker` `GitHub Actions`
    """)

    # Try to show model info as a clean layout
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            st.markdown("### 🤖 Current Model")

            # Model overview metrics
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Model Type", info.get('model_type', 'N/A'))
            mc2.metric("Features", info.get('num_features', 'N/A'))
            mc3.metric("Estimators", info.get('n_estimators', 'N/A'))

            # Top features table
            if 'top_features' in info and info['top_features']:
                st.markdown("#### 🏆 Top Feature Importances")
                feat_df = pd.DataFrame(
                    list(info['top_features'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                feat_df['Importance'] = feat_df['Importance'].apply(lambda x: f"{x:.4f}")
                feat_df.index = range(1, len(feat_df) + 1)
                st.table(feat_df)

            # Supported carriers & airports
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("#### ✈️ Supported Airlines")
                if 'carriers_supported' in info:
                    carriers_list = ', '.join(info['carriers_supported'])
                    st.write(carriers_list)
            with sc2:
                st.markdown("#### 🏢 Supported Airports")
                if 'airports_supported' in info:
                    airports_list = ', '.join(info['airports_supported'])
                    st.write(airports_list)
    except Exception:
        st.info("Start the API server to see model details.")
