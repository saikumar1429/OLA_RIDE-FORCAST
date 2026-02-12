

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Ola Ride Forecast", layout="wide")

st.title("🛵 Ola Bike Ride Demand Forecasting Dashboard")


@st.cache_resource
def load_model():
    with open("ola_xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()



@st.cache_data
def load_data():
    df = pd.read_csv("ola.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values("datetime")
    return df

df = load_data()

st.sidebar.header("🔮 Predict Future Ride Demand")

temperature = st.sidebar.slider("Temperature", 0, 50, 30)
rain = st.sidebar.selectbox("Rain", [0, 1])
holiday = st.sidebar.selectbox("Holiday", [0, 1])
hour = st.sidebar.slider("Hour", 0, 23, 18)
day = st.sidebar.slider("Day", 1, 31, 12)
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)

is_weekend = 1 if day_of_week >= 5 else 0

lag_1 = st.sidebar.number_input("Last Hour Demand (lag_1)", value=100)
lag_24 = st.sidebar.number_input("Yesterday Same Hour (lag_24)", value=90)
rolling_mean_24 = st.sidebar.number_input("Rolling Mean 24h", value=95)
rolling_std_24 = st.sidebar.number_input("Rolling Std 24h", value=10)



if st.sidebar.button("Predict Ride Demand"):

    input_data = np.array([[
        temperature,
        rain,
        holiday,
        hour,
        day,
        month,
        day_of_week,
        is_weekend,
        lag_1,
        lag_24,
        rolling_mean_24,
        rolling_std_24
    ]])

    prediction = model.predict(input_data)[0]

    st.sidebar.success(f"Predicted Ride Demand: {round(prediction,2)}")



st.subheader("📊 Ride Demand Over Time")

fig1, ax1 = plt.subplots(figsize=(12,5))
ax1.plot(df['datetime'], df['count'])
ax1.set_title("Ride Demand Trend")
st.pyplot(fig1)


st.subheader("📈 Hourly Demand Pattern")

hourly_avg = df.groupby(df['datetime'].dt.hour)['count'].mean()

fig2, ax2 = plt.subplots()
hourly_avg.plot(kind='bar', ax=ax2)
ax2.set_title("Average Ride Demand by Hour")
st.pyplot(fig2)


st.subheader("📅 Weekly Demand Pattern")

weekly_avg = df.groupby(df['datetime'].dt.dayofweek)['count'].mean()

fig3, ax3 = plt.subplots()
weekly_avg.plot(kind='bar', ax=ax3)
ax3.set_title("Average Ride Demand by Day")
st.pyplot(fig3)


st.subheader("🔥 Demand Heatmap (Day vs Hour)")

df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

pivot_table = df.pivot_table(
    values='count',
    index='day_of_week',
    columns='hour',
    aggfunc='mean'
)

fig4, ax4 = plt.subplots(figsize=(12,5))
sns.heatmap(pivot_table, cmap='YlGnBu', ax=ax4)
ax4.set_title("Demand Heatmap")
st.pyplot(fig4)



st.subheader("📊 Feature Importance")

features = [
    'temperature',
    'rain',
    'holiday',
    'hour',
    'day',
    'month',
    'day_of_week',
    'is_weekend',
    'lag_1',
    'lag_24',
    'rolling_mean_24',
    'rolling_std_24'
]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig5, ax5 = plt.subplots(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax5)
ax5.set_title("Feature Importance")
st.pyplot(fig5)

st.success("✅ Ola Forecasting Dashboard Ready")
