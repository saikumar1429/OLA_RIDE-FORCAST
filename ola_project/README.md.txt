# 🛵 Ola Bike Ride Demand Forecasting System

An end-to-end Machine Learning project that predicts hourly Ola bike ride demand using advanced time-series feature engineering and XGBoost regression.

This project demonstrates real-world demand forecasting, feature engineering, model deployment, and interactive visualization using Streamlit.

---

## 🚀 Project Overview

Urban mobility platforms experience fluctuating ride demand due to time patterns, user behavior, and external conditions.  
This system predicts hourly ride demand to help optimize:

- 🚗 Driver allocation
- 📈 Dynamic pricing strategies
- ⏱ Reduced wait times
- 📊 Operational efficiency

---

## 🧠 Key Features

- Time-series feature engineering (hour, weekday, month)
- Lag features (lag_1, lag_24)
- Rolling statistics (24-hour rolling mean & std)
- XGBoost regression model
- Time-aware train/test split (no data leakage)
- Model serialization using Pickle
- Interactive Streamlit dashboard
- Multiple advanced visualizations

---


---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- XGBoost
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 📊 Feature Engineering

The model uses advanced time-series features:

- Hour of day
- Day of week
- Weekend flag
- Lag_1 (previous hour demand)
- Lag_24 (same hour previous day)
- Rolling Mean (24-hour)
- Rolling Standard Deviation (24-hour)

These features capture short-term and daily demand patterns.

---

## 📈 Model Performance Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Time-based data splitting ensures realistic evaluation.

---

## 🌐 Running the Streamlit App

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

streamlit run app.py

