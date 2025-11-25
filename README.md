
````markdown
# 🌤️ PakWeather AI: Hyper-Local Deep Learning Forecast

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Deployed-success)]()

**PakWeather AI** is an end-to-end Deep Learning solution designed to predict daily weather conditions for Pakistan's major cities. Unlike generic national forecasts, this project utilizes **city-specific LSTM (Long Short-Term Memory)** networks to capture unique micro-climate patterns, distinguishing between the humid coast of **Karachi** and the arid highlands of **Quetta**.

---

## 📌 Project Overview

Accurate weather forecasting in Pakistan is critical for agriculture, urban planning, and disaster management. Traditional statistical models often fail to capture non-linear complexities, and general models "average out" the drastic climatic differences across the country.

**Our Solution:**
We built a multi-output regression model capable of predicting **5 key weather variables** simultaneously for the next day:
1.  Minimum Temperature (`tmin`)
2.  Maximum Temperature (`tmax`)
3.  Humidity (`humidity`)
4.  Wind Speed (`wspd`)
5.  Precipitation (`prcp`)

---

## 🚀 Key Features

* **🧠 Custom LSTM Architecture:** Stacked LSTM layers designed to learn long-term temporal dependencies from 25 years of historical data.
* **🏙️ Specialized "Expert" Models:** Instead of one global model, we trained **4 separate models** (Karachi, Lahore, Islamabad, Quetta) to eliminate regional bias.
* **🔄 Automated Data Pipeline:** Includes a Python script (`update_data.py`) that auto-fetches the latest weather data from the **Open-Meteo API** to keep predictions current.
* **📉 Drift Monitoring:** A dedicated module to detect statistical drift between historical training data and recent climate shifts.
* **🔍 Explainable AI (XAI):** Integrated **LIME** analysis to visualize *why* the model made a specific prediction (e.g., "High temp predicted because yesterday was hot").
* **📊 Interactive Dashboard:** A full-stack Streamlit application for visualization and live forecasting.

---

## 📊 Model Performance

We benchmarked our LSTM against a Statistical Baseline (ARIMAX) and a Machine Learning Baseline (XGBoost). The LSTM significantly outperformed both on the daily prediction task.

**Final Test Set Results (MAE - Mean Absolute Error):**

| City | Min Temp (°C) | Max Temp (°C) | Humidity (%) | Wind Speed (km/h) | Rain (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Karachi** | 0.77 | 1.15 | 5.46 | 3.92 | 1.66 |
| **Lahore** | 0.97 | 1.24 | 3.83 | 3.67 | 2.92 |
| **Islamabad** | 1.17 | 1.27 | 5.05 | 3.45 | 4.20 |
| **Quetta** | 1.15 | 1.72 | 6.32 | 2.69 | 1.31 |

> **Result:** We achieved a temperature prediction error of **~1.0°C**, proving the model is highly effective for daily planning.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras (LSTM, Dense, Dropout)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (MinMaxScaling)
* **Visualization:** Plotly, Matplotlib
* **Deployment:** Streamlit
* **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
* **Data Source:** Open-Meteo API

---

## 📂 Project Structure

```text
├── models/                  # Saved .keras models for each city
├── scalers/                 # Saved .gz scalers for normalization
├── notebooks/               # Jupyter notebooks for EDA & Training
├── app.py                   # Main Streamlit Dashboard application
├── update_data.py           # Script to fetch latest weather data
├── Weather_2025.csv         # Master dataset (2000-Present)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
````

-----

## 💻 Installation & Usage

### 1\. Clone the Repository

```bash
git clone [https://github.com/YourUsername/DL-Weather-Model-Deployment.git](https://github.com/YourUsername/DL-Weather-Model-Deployment.git)
cd DL-Weather-Model-Deployment
```

### 2\. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the Dashboard

```bash
streamlit run app.py
```

-----

## 🤖 How It Works

1.  **Input:** The model takes a sequence of the past **30 days** of weather data.
2.  **Processing:** \* Data is scaled (0-1).
      * Lag features (Yesterday's weather) and Cyclical features (Day of Year Sine/Cosine) are engineered.
      * The data is passed through 2 LSTM layers to extract temporal patterns.
3.  **Output:** The model outputs 5 simultaneous values representing the forecast for the next day.
4.  **Update:** If the dataset is outdated, `update_data.py` fetches the gap data from the API so the model always sees "yesterday's" weather.

-----

## 👥 Contributors

  * **Ebaad Khan** (DT-22045)
  * **Ezaan Khan** (DT-22046)
  * **Syed Ahmed Ali** (DT-22301)
  * **Muhammad Khuzaima Hassan** (DT-22302)

**Supervised by:** Dr. Murk Marvi
[cite\_start]**Course:** CT-468 Deep Learning, NED University of Engineering & Technology [cite: 4, 7-12]

-----

*Note: This project was developed for educational purposes as part of a Final Year Deep Learning curriculum.*

```
```
