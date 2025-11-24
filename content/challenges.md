# Project Challenges & Solutions

## 1. Data Collection

### **Challenge**
Weather data from **2000–2025** needed cleaning, merging, fixing missing values, and standardizing units. Different cities had different levels of missingness.

### **How we solved it**
*   Consolidated all sources into a unified CSV.
*   Removed impossible entries (negative humidity, pressure anomalies, etc.).
*   Standardized datetime formats.
*   Ensured consistent city naming (Islamabad, Karachi, Lahore, Quetta).
*   Used **forward-fill + backward-fill + interpolation**.

---

## 2. Preprocessing

### **Challenge**
Deep models require high-quality structured inputs. We needed:
*   Correct lags
*   Proper time features
*   Normalized numerical ranges
*   No leakage between train/val/test

### **How we solved it**
*   Added **lag-1 features** for all meteorological variables.
*   Added cyclical features using **sin/cos transformation**.
*   Ensured strict city-wise sorting to avoid leakage.
*   Dropped rows only created by lagging.

---

## 3. Data Pipeline Design

### **Challenge**
Pipeline had to support:
*   Multiple models (one per city)
*   Multiple scalers
*   Consistent feature engineering
*   Prediction mode for any future date

### **How we solved it**
*   Created reusable functions for:
    *   Lag generation
    *   Scaling
    *   Window creation (30-step input)
*   Stored scalers and models separately per city.

---

## 4. Data Drift & Model Updates

### **Challenge**
Weather is non-stationary:
*   Climate changes
*   New patterns appear every year
*   Model performance degrades with fresh data

### **How we solved it**
*   Implemented a drift trigger monitoring pipeline:
    *   Compare rolling 30-day distributions.
    *   Flag deviations in mean/variance.
*   Automated retraining every X months.
*   Kept **2025 data** as final test holdout.

---

## 5. Model Training

### **Challenge**
Finding the best architecture for forecasting 5 targets: **tmin, tmax, humidity, windspeed, precipitation**.

### **How we solved it**
*   Tried ARIMAX.
*   Tried XGBoost.
*   **Finalized on LSTM**.
*   Used:
    *   30-day lookback
    *   2× LSTM layers
    *   Dropout for regularization
    *   EarlyStopping + ModelCheckpoint

---

## 6. Finetuning Pretrained Models

### **Challenge**
Could a Pakistan-specific model benefit from pretrained global weather models?

### **How we solved it**
*   Tested finetuning on NOAA-based pretrained weights.
*   **Result**: Performance didn’t improve (Pakistan patterns differ greatly).
*   We switched to city-specific, from-scratch models.
*   Achieved far better accuracy.

---

## 7. Application Development (Streamlit)

### **Challenge**
Designing an interactive and visually appealing app that includes:
*   City selection
*   Forecasting
*   EDA
*   Performance comparison
*   Visual explanations

### **How we solved it**
*   Used Streamlit multipage-style layout.
*   Added clean sidebar navigation.
*   Integrated animations (map GIF).
*   Added downloadable resources.

---

## 8. Model Evaluation

### **Challenge**
Multiple models & targets → difficult to compare fairly.

### **How we solved it**
*   Created unified comparison tables.
*   Used **MAE** and **RMSE**.
*   City-level breakdown.
*   Baseline vs LSTM comparison.
*   Final models (2025 holdout performance) show:
    *   **tmin/tmax**: Highly accurate.
    *   **humidity/wind**: Improved but still noisy.
    *   **precipitation**: Remains the hardest.

---

## 🎯 Overall Achievement
This project demonstrates a full real-world ML pipeline with:
*   City-level models
*   Robust feature engineering
*   Multiple model experiments
*   A fully functional app
*   Clean architecture
*   Real-world forecasting capability

