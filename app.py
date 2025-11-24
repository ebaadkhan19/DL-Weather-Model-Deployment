import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from PIL import Image
import base64
from scipy import stats # For KS Test
import data_updater  # Import the new module


# -----------------------------
# Config / paths
# -----------------------------
st.set_page_config(page_title="Pakistan Weather Forecast — LSTM Demo",
                   layout="wide", page_icon="🌤️")

ROOT = os.path.abspath(".")
DATA_CSV = os.path.join(ROOT, "Weather_2025.csv")
PREPROCESSED = os.path.join(ROOT, "data_master_fixed_preprocessed.csv")
MODELS_DIR = os.path.join(ROOT, "models")
SCALERS_DIR = os.path.join(ROOT, "scalers")
IMAGES_DIR = os.path.join(ROOT, "images")
CONTENT_DIR = os.path.join(ROOT, "content")
CHALLENGES_MD = os.path.join(CONTENT_DIR, "challenges.md")

# Fixed paths: removed trailing dot to match actual filenames
CITY_MODELS = {
    "Karachi": {
        "model": os.path.join(MODELS_DIR, "lstm_Karachi.keras"),
        "scaler_x": os.path.join(SCALERS_DIR, "scaler_x_Karachi"),
        "scaler_y": os.path.join(SCALERS_DIR, "scaler_y_Karachi"),
    },
    "Lahore": {
        "model": os.path.join(MODELS_DIR, "lstm_Lahore.keras"),
        "scaler_x": os.path.join(SCALERS_DIR, "scaler_x_Lahore"),
        "scaler_y": os.path.join(SCALERS_DIR, "scaler_y_Lahore"),
    },
    "Islamabad": {
        "model": os.path.join(MODELS_DIR, "lstm_Islamabad.keras"),
        "scaler_x": os.path.join(SCALERS_DIR, "scaler_x_Islamabad"),
        "scaler_y": os.path.join(SCALERS_DIR, "scaler_y_Islamabad"),
    },
    "Quetta": {
        "model": os.path.join(MODELS_DIR, "lstm_Quetta.keras"),
        "scaler_x": os.path.join(SCALERS_DIR, "scaler_x_Quetta"),
        "scaler_y": os.path.join(SCALERS_DIR, "scaler_y_Quetta"),
    },
}

BASE_FEATURES = [
    'tavg','pressure','dew_point','cloud_cover','latitude','longitude','elevation',
    'year','month','dayofyear','day_sin','day_cos',
    'tmin_lag1','tmax_lag1','tavg_lag1','humidity_lag1','wspd_lag1','prcp_lag1',
    'pressure_lag1','dew_point_lag1','cloud_cover_lag1'
]

TARGETS = ['tmin','tmax','humidity','wspd','prcp']
LAG_COLS = ['tmin','tmax','tavg','humidity','wspd','prcp','pressure','dew_point','cloud_cover']

# -----------------------------
# Custom CSS & Styling
# -----------------------------
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and cover the entire screen
    '''
    # set bg name
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-attachment: fixed;
             background-position: center;
         }}
         /* Glassmorphism for containers */
         .stMarkdown, .stDataFrame, .stPlotlyChart {{
             background-color: rgba(255, 255, 255, 0.85);
             padding: 20px;
             border-radius: 15px;
             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
             margin-bottom: 20px;
         }}
         .stMetric {{
             background-color: rgba(255, 255, 255, 0.9);
             padding: 15px;
             border-radius: 10px;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
             text-align: center;
         }}
         h1, h2, h3 {{
             color: #003f5c;
             text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
         }}
         .sidebar .sidebar-content {{
             background-color: rgba(255, 255, 255, 0.95);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

bg_path = os.path.join(IMAGES_DIR, "background.jpg")
if os.path.exists(bg_path):
    set_bg_hack(bg_path)

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def load_raw_data():
    if not os.path.exists(DATA_CSV):
        return pd.DataFrame()
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    df.columns = [c.strip() for c in df.columns]
    df['city'] = df['city'].astype(str).str.strip().str.title()
    return df

@st.cache_data
def load_preprocessed():
    if os.path.exists(PREPROCESSED):
        df = pd.read_csv(PREPROCESSED, parse_dates=["date"])
        df.columns = [c.strip() for c in df.columns]
        return df
    df = load_raw_data()
    if df.empty:
        return df
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['day_sin'] = np.sin(2*np.pi*df['dayofyear']/365.0)
    df['day_cos'] = np.cos(2*np.pi*df['dayofyear']/365.0)
    for col in LAG_COLS:
        df[f"{col}_lag1"] = df.groupby('city')[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_resource
def load_city_model_and_scalers(city_name, allow_master_fallback=True):
    entry = CITY_MODELS.get(city_name, {})
    model = None; scaler_x = None; scaler_y = None
    
    try:
        if entry.get("model") and os.path.exists(entry["model"]):
            model = tf.keras.models.load_model(entry["model"])
    except Exception as e:
        st.sidebar.error(f"Error loading model for {city_name}: {e}")
        
    try:
        if entry.get("scaler_x") and os.path.exists(entry["scaler_x"]):
            scaler_x = joblib.load(entry["scaler_x"])
    except Exception as e:
        st.sidebar.error(f"Error loading scaler_x for {city_name}: {e}")
        
    try:
        if entry.get("scaler_y") and os.path.exists(entry["scaler_y"]):
            scaler_y = joblib.load(entry["scaler_y"])
    except Exception as e:
        st.sidebar.error(f"Error loading scaler_y for {city_name}: {e}")

    if allow_master_fallback and (model is None or scaler_x is None or scaler_y is None):
        master_model = os.path.join(ROOT, "lstm_master_fixed.keras")
        sx = os.path.join(ROOT, "scaler_x_fixed.gz") # Assuming .gz based on typical joblib usage, or check file
        sy = os.path.join(ROOT, "scaler_y_fixed.gz")
        
        # Check for non-gz versions if gz not found
        if not os.path.exists(sx): sx = os.path.join(ROOT, "scaler_x_fixed")
        if not os.path.exists(sy): sy = os.path.join(ROOT, "scaler_y_fixed")

        try:
            if model is None and os.path.exists(master_model):
                model = tf.keras.models.load_model(master_model)
        except Exception: pass
        try:
            if scaler_x is None and os.path.exists(sx):
                scaler_x = joblib.load(sx)
        except Exception: pass
        try:
            if scaler_y is None and os.path.exists(sy):
                scaler_y = joblib.load(sy)
        except Exception: pass
            
    return model, scaler_x, scaler_y

def create_rolling_plot(df_city, value_col='tavg'):
    dfc = df_city.sort_values('date').copy()
    dfc['rolling30'] = dfc[value_col].rolling(window=30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc[value_col], name='Daily', line=dict(color='rgba(0, 63, 92, 0.3)')))
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['rolling30'], name='30-day MA', line=dict(color='#003f5c', width=3)))
    fig.update_layout(
        title=f"{value_col.upper()} Trend",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40,l=0,r=0,b=0), 
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    logo_path = os.path.join(IMAGES_DIR, "logo.jpg")
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)
    
    st.title("Navigation")
    menu = st.radio("Main Menu", ["Home", "EDA", "Challenges & Solutions", "Forecast / Predict", "Model Insights (LIME)", "Data Drift Monitor", "Downloads", "About"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Data Update Section
    st.subheader("Data Management")
    if st.button("🔄 Sync with Open-Meteo"):
        with st.spinner("Fetching latest weather data..."):
            status_msg = data_updater.update_weather_dataset(DATA_CSV)
            
            # If update happened, clear cache to reload new data
            if "Successfully updated" in status_msg:
                st.cache_data.clear()
                # Also remove preprocessed file to force regeneration
                if os.path.exists(PREPROCESSED):
                    os.remove(PREPROCESSED)
                st.success(status_msg)
                st.rerun()
            else:
                st.info(status_msg)

    
    st.markdown("---")
    st.caption("© 2025 Weather AI Team")

# -----------------------------
# Main Content
# -----------------------------

if menu == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🌤️ Pakistan Weather Forecast")
        st.markdown("### Powered by Multi-Output LSTM Networks")
        st.markdown("""
        Welcome to the next generation of weather forecasting. This application leverages advanced Deep Learning models to predict weather patterns across major cities in Pakistan.
        
        **Key Features:**
        *   **City-Specific Models**: Tailored LSTMs for Karachi, Lahore, Islamabad, and Quetta.
        *   **Multi-Target Prediction**: Forecasts Temperature, Humidity, Wind Speed, and Precipitation simultaneously.
        *   **Interactive EDA**: Explore historical weather trends with dynamic visualizations.
        """)
    with col2:
        map_path = os.path.join(IMAGES_DIR, "pakistan_map.gif")
        if os.path.exists(map_path):
            st.image(map_path, use_container_width=True)

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df = load_preprocessed()
    
    if df.empty:
        st.error("Data not found. Please ensure 'Weather_2025.csv' is in the root directory.")
    else:
        col_sel, col_empty = st.columns([1, 2])
        with col_sel:
            city_select = st.selectbox("Select City", sorted(df['city'].unique()))
        
        df_city = df[df['city'] == city_select].sort_values('date').reset_index(drop=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_rolling_plot(df_city, 'tavg'), use_container_width=True)
        with col2:
            st.plotly_chart(create_rolling_plot(df_city, 'humidity'), use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            fig_pr = px.bar(df_city, x='date', y='prcp', title="Daily Precipitation (mm)", color_discrete_sequence=['#58508d'])
            fig_pr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pr, use_container_width=True)
        with col4:
            fig_ws = px.histogram(df_city, x='wspd', nbins=30, title="Wind Speed Distribution", color_discrete_sequence=['#bc5090'])
            fig_ws.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_ws, use_container_width=True)

elif menu == "Challenges & Solutions":
    st.title("🧩 Challenges & Solutions")
    if os.path.exists(CHALLENGES_MD):
        with open(CHALLENGES_MD, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Challenges document not found.")

elif menu == "Forecast / Predict":
    st.title("🔮 Weather Prediction")
    st.markdown("Generate forecasts using our trained LSTM models.")
    
    df_all = load_preprocessed()
    if df_all.empty:
        st.error("Data not available.")
    else:
        col_c, col_d, col_b = st.columns([1, 1, 1])
        with col_c:
            city = st.selectbox("Select City", sorted(df_all['city'].unique()))
        with col_d:
            predict_date = st.date_input("Target Date", value=(datetime.now() + timedelta(days=1)).date())
        with col_b:
            st.write("") # Spacer
            st.write("")
            st.checkbox("Apply Smart Drift Correction", key="apply_bias", value=True, help="Automatically adjusts prediction based on the last 14 days of data (captures recent cold/heat waves).")
            run_btn = st.button("Run Prediction", type="primary", use_container_width=True)
            
        if run_btn:
            with st.spinner(f"Loading model for {city} and generating prediction..."):
                # Load Data for Drift Calculation (Bias Correction)
                df_all_city = df_all[df_all['city'] == city]
                # Determine if we need bias correction
                # Simple approach: Calculate mean shift for tmax/tmin between 2025 and history
                # This is a simplified "Drift Adaptation"
                
                bias_correction = {}
                if st.session_state.get("apply_bias", False):
                     # Smart Drift Correction: Use last 14 days to capture recent trends (e.g. cold snaps)
                     df_curr = df_all_city[df_all_city['year'] >= 2025].tail(14)
                     
                     if not df_curr.empty:
                         # Match historical data to the same days of year
                         curr_doy = df_curr['dayofyear'].unique()
                         df_ref = df_all_city[(df_all_city['year'] < 2025) & (df_all_city['dayofyear'].isin(curr_doy))]
                         
                         if not df_ref.empty:
                             # Calculate deltas
                             # Targets: 'tmin','tmax','humidity','wspd','prcp'
                             # Indices: 0:tmin, 1:tmax, 2:humidity, 3:wspd, 4:prcp
                             bias_correction[0] = df_curr['tmin'].mean() - df_ref['tmin'].mean()
                             bias_correction[1] = df_curr['tmax'].mean() - df_ref['tmax'].mean()
                             bias_correction[2] = df_curr['humidity'].mean() - df_ref['humidity'].mean()
                             bias_correction[3] = df_curr['wspd'].mean() - df_ref['wspd'].mean()
                             bias_correction[4] = df_curr['prcp'].mean() - df_ref['prcp'].mean()
                         
                model, scaler_x, scaler_y = load_city_model_and_scalers(city)
                
                if model is None:
                    st.error(f"Could not load model for {city}. Please check model files.")
                else:
                    sel_date = pd.to_datetime(predict_date)
                    window = df_all[(df_all['city']==city) & (df_all['date'] < sel_date)].sort_values("date").tail(30)
                    
                    if window.shape[0] < 30:
                        st.warning(f"Insufficient historical data. Need 30 days prior to {sel_date.date()}, found {window.shape[0]}.")
                    else:
                        missing = [f for f in BASE_FEATURES if f not in window.columns]
                        if missing:
                            st.error(f"Missing features: {missing}")
                        else:
                            # Add city one-hot encoding (required by scaler)
                            target_cities = ["Islamabad", "Karachi", "Lahore", "Quetta"]
                            for c in target_cities:
                                window[f"city_{c}"] = 1.0 if city == c else 0.0
                            
                            # Extend features list
                            full_features = BASE_FEATURES + [f"city_{c}" for c in target_cities]
                            
                            Xw = window[full_features].values
                            try:
                                Xs = scaler_x.transform(Xw)
                            except:
                                Xs = scaler_x.transform(np.array(Xw))
                                
                            Xs = Xs.reshape(1, Xs.shape[0], Xs.shape[1])
                            yp = model.predict(Xs)
                            
                            try:
                                y_pred = scaler_y.inverse_transform(yp)[0]
                            except:
                                y_pred = yp[0]
                            
                            st.success(f"Prediction for {sel_date.date()} successful!")
                            
                            # Post-processing: Clamp negative values for physical constraints
                            # Indices: 0:tmin, 1:tmax, 2:humidity, 3:wspd, 4:prcp
                            y_pred[2] = max(0.0, y_pred[2]) # Humidity
                            y_pred[3] = max(0.0, y_pred[3]) # Wind Speed
                            y_pred[4] = max(0.0, y_pred[4]) # Precipitation
                            
                            # Apply Bias Correction if enabled
                            if bias_correction:
                                for idx, shift in bias_correction.items():
                                    y_pred[idx] += shift
                                    
                                # Re-clamp after bias
                                y_pred[2] = max(0.0, y_pred[2])
                                y_pred[3] = max(0.0, y_pred[3])
                                y_pred[4] = max(0.0, y_pred[4])
                                
                                st.info(f"ℹ️ **Drift Correction Applied**: Adjusted Tmax by {bias_correction[1]:+.2f}°C, Tmin by {bias_correction[0]:+.2f}°C based on 2025 trends.")
                            
                            m1, m2, m3, m4, m5 = st.columns(5)
                            m1.metric("Min Temp", f"{y_pred[0]:.1f} °C")
                            m2.metric("Max Temp", f"{y_pred[1]:.1f} °C")
                            m3.metric("Humidity", f"{y_pred[2]:.1f} %")
                            m4.metric("Wind Speed", f"{y_pred[3]:.1f} km/h")
                            m5.metric("Precipitation", f"{y_pred[4]:.1f} mm")

elif menu == "Model Insights (LIME)":
    st.title("🧠 Model Explainability (LIME)")
    st.markdown("""
    **Local Interpretable Model-agnostic Explanations (LIME)** helps us understand *why* the model made a specific prediction.
    Below are the analysis reports for each city's model.
    """)
    
    # Path to analysis text
    lime_md_path = os.path.join(CONTENT_DIR, "lime_analysis.md")
    
    # Display Text Analysis
    if os.path.exists(lime_md_path):
        with open(lime_md_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("No analysis text found.")
        
    st.markdown("---")
    st.subheader("Visual Explanations")
    
    # Display Images in a grid
    lime_images_dir = os.path.join(IMAGES_DIR, "lime")
    
    # Map city names to expected filenames (flexible matching)
    # User said: islamabad_lime, karachi_lime, etc. (likely .png)
    
    cities = ["Karachi", "Lahore", "Islamabad", "Quetta"]
    
    cols = st.columns(2)
    
    for idx, city in enumerate(cities):
        # Try to find the file
        expected_filename = f"{city.lower()}_lime.png"
        img_path = os.path.join(lime_images_dir, expected_filename)
        
        # Check if exists (or try without extension if needed, but assuming png based on user prompt)
        if not os.path.exists(img_path):
             # Try jpg just in case
             img_path = os.path.join(lime_images_dir, f"{city.lower()}_lime.jpg")
        
        with cols[idx % 2]:
            st.markdown(f"### {city}")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"LIME Analysis for {city}", use_container_width=True)
            else:
                st.warning(f"Image not found: {expected_filename}")


elif menu == "Data Drift Monitor":
    st.title("📉 Data Drift Monitor")
    st.markdown("""
    **Goal**: Detect if recent weather patterns (2025+) deviate significantly from historical training data (2000-2024).
    Significant drift can indicate that the model's training assumptions are no longer valid.
    """)
    
    df = load_preprocessed()
    if df.empty:
        st.error("Data not available.")
    else:
        # Sidebar Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            drift_city = st.selectbox("Select City", sorted(df['city'].unique()), key="drift_city")
        with col_ctrl2:
            # Numerical columns only
            num_cols = ['tmax', 'tmin', 'tavg', 'humidity', 'wspd', 'prcp', 'pressure', 'dew_point', 'cloud_cover']
            drift_feat = st.selectbox("Select Feature", num_cols, key="drift_feat")
            
        # Sensitivity Control
        p_threshold = st.slider("Drift Sensitivity (P-Value Threshold)", 0.001, 0.1, 0.01, format="%.3f", help="Lower value = Stricter test (harder to trigger drift). Higher value = More sensitive.")
            
        # Filter Data
        df_city = df[df['city'] == drift_city]
        
        # Split: Reference (Historical) vs Current (2025+)
        # Assuming training was up to 2024 end.
        df_ref = df_city[df_city['year'] < 2025]
        df_curr = df_city[df_city['year'] >= 2025]
        
        if df_ref.empty or df_curr.empty:
            st.warning("Insufficient data for comparison. Need both historical (<2025) and recent (2025+) data.")
        else:
            # 0. Seasonality Correction (Apples-to-Apples)
            # If 2025 is partial (e.g., only up to Nov), we should only compare against Jan-Nov of history.
            curr_months = df_curr['month'].unique()
            if len(curr_months) < 12:
                st.info(f"ℹ️ **Seasonality Correction**: 2025 data is partial (Months: {min(curr_months)}-{max(curr_months)}). Filtering historical data to match these months for a fair comparison.")
                df_ref = df_ref[df_ref['month'].isin(curr_months)]
            
            # 1. Statistical Metrics
            st.subheader("Statistical Analysis")
            
            ref_mean = df_ref[drift_feat].mean()
            curr_mean = df_curr[drift_feat].mean()
            delta = curr_mean - ref_mean
            
            # KS Test
            ks_stat, p_value = stats.ks_2samp(df_ref[drift_feat], df_curr[drift_feat])
            
            # Display Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Historical Mean", f"{ref_mean:.2f}")
            m2.metric("Recent Mean (2025+)", f"{curr_mean:.2f}", delta=f"{delta:.2f}")
            m3.metric("KS Statistic", f"{ks_stat:.3f}")
            
            # P-Value Interpretation
            is_drift = p_value < p_threshold
            drift_color = "inverse" if is_drift else "normal" # Red if drift (low p), Green if stable
            drift_label = "DRIFT DETECTED" if is_drift else "Stable"
            
            m4.metric("P-Value (Drift)", f"{p_value:.4f}", delta=drift_label, delta_color=drift_color)
            
            if is_drift:
                st.error(f"⚠️ **Drift Detected**: The distribution of {drift_feat} in 2025 is significantly different from history (p < {p_threshold}).")
                
                # Contextual Advice
                if drift_feat == 'tmin':
                    st.warning("🌡️ **Tmin Drift**: Minimum temperatures (nights) often rise faster due to Urban Heat Island effects or climate change. This is a common form of genuine drift.")
                elif drift_feat in ['prcp', 'wspd']:
                    st.info("🌧️ **High Variance**: Precipitation and Wind Speed are naturally chaotic. Statistical tests often flag them as 'drifted' even if the general pattern is similar. Rely more on the visual histogram.")
                
                st.markdown("""
                **Recommendation**: 
                1. **Adjust Sensitivity**: Try lowering the threshold slider if you think this is a false alarm.
                2. **Retrain**: If `tmin` drift persists, the model might underestimate night temperatures. Consider retraining with recent data.
                """)
            else:
                st.success(f"✅ **Stable**: No significant drift detected for {drift_feat}.")
                
            # 2. Visualization
            st.subheader("Distribution Comparison")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_ref[drift_feat], 
                name='Historical (2000-2024)',
                opacity=0.6,
                marker_color='#003f5c',
                histnorm='probability density'
            ))
            fig.add_trace(go.Histogram(
                x=df_curr[drift_feat], 
                name='Recent (2025+)',
                opacity=0.6,
                marker_color='#ff6361',
                histnorm='probability density'
            ))
            
            fig.update_layout(
                barmode='overlay',
                title=f"Distribution of {drift_feat} in {drift_city}",
                xaxis_title=drift_feat,
                yaxis_title="Density",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)


elif menu == "Downloads":
    st.title("📥 Downloads")
    st.markdown("Access project artifacts and datasets.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Datasets")
        if os.path.exists(DATA_CSV):
            with open(DATA_CSV, "rb") as f:
                st.download_button("Download Raw CSV", f, file_name="Weather_2025.csv", mime="text/csv")
        if os.path.exists(PREPROCESSED):
            with open(PREPROCESSED, "rb") as f:
                st.download_button("Download Preprocessed Data", f, file_name="preprocessed_weather.csv", mime="text/csv")
                
    with c2:
        st.subheader("Models")
        for city_name, info in CITY_MODELS.items():
            if os.path.exists(info["model"]):
                with open(info["model"], "rb") as f:
                    st.download_button(f"Download {city_name} Model", f, file_name=f"lstm_{city_name}.keras")

elif menu == "About":
    st.title("ℹ️ About")
    st.info("This project was developed for the Deep Learning Final Project.")
    st.markdown("""
    **Methodology:**
    1.  **Data Collection**: Aggregated weather data from 2000-2025.
    2.  **Preprocessing**: Lag features, cyclical encoding, and scaling.
    3.  **Modeling**: LSTM networks trained on 30-day sequences.
    4.  **Evaluation**: MAE/RMSE metrics on a holdout set (2025).
    
    **Tech Stack:**
    *   Python, TensorFlow/Keras
    *   Streamlit, Plotly
    *   Pandas, NumPy, Scikit-learn
    """)
