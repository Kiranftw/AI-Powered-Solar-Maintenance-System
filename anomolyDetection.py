import os
import joblib
import pandas
import numpy
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from functools import wraps
from EnergyPrediction import EnergyPrediction
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from EnergyPrediction import EnergyPrediction
import requests
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import pytz
logging.basicConfig(level=logging.INFO, pytzmat='%(asctime)s - %(levelname)s - %(message)s')

class AnomolyDetection():
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.GenerationData = pandas.read_csv(os.path.join(os.getcwd(), "DATASETS","Plant_1_Generation_Data.csv" ))
        self.WeatherData = pandas.read_csv(os.path.join(os.getcwd(), "DATASETS","Plant_1_Weather_Sensor_Data.csv"))
        self.locator = Nominatim(user_agent="solar_maintenance APP")
        self.energyprediction = EnergyPrediction()

    def dataAggregation(self) -> pandas.DataFrame:
        DataFrame = pandas.merge(
        self.GenerationData.assign(DATE_TIME=pandas.to_datetime(self.GenerationData['DATE_TIME'], dayfirst=True)),
        self.WeatherData.assign(DATE_TIME=pandas.to_datetime(self.WeatherData['DATE_TIME'])),
        on=['DATE_TIME', 'PLANT_ID'],
        how='left'
        )
        DataFrame.drop(['PLANT_ID',"SOURCE_KEY_y"], axis=1, inplace=True)
        DataFrame.rename(columns={
            'SOURCE_KEY_x': 'INVERTER_IDs'
        }, inplace=True)
        print(DataFrame['INVERTER_IDs'].unique())
        return DataFrame
    
    def trainDCmodel(self, dataframe: pandas.DataFrame = None) -> XGBRegressor:
        if dataframe is None:
            dataframe = self.dataAggregation()
        dataframe.dropna(subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER"], inplace=True)
        features = [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]
        target = "DC_POWER"
        X = dataframe[features]
        y = dataframe[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(f"✅ DC_POWER Model Trained:\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")
        #NOTE:  RMSE: 481.16R² Score: 0.9
        joblib.dump(model, "models/timebasedDCmodel.pkl")
        return model
    
    def trainACmodel(self) -> None:
        #TODO: Developing an ACmodel that predict AC current with using(Weather & DC) we are working with one hot encoding
        dataframe = self.dataAggregation()
        dataframe = pandas.get_dummies(dataframe, columns=["INVERTER_IDs"])
        dataframe.info()

    def getCurrentWeather(self, latitude: float, longitude: float) -> dict:
        api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True,
            "timezone": "auto"
        }
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            current = data.get("current_weather", {})
            weather_info = {
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "weather_code": current.get("weathercode")  # To map to conditions like sunny/rainy
            }
            weather_code_map = {
                0: "Clear sky",
                1: "Mainly clear",
                2: "Partly cloudy",
                3: "Overcast",
                45: "Fog",
                48: "Depositing rime fog",
                51: "Light drizzle",
                61: "Light rain",
                71: "Light snow",
                80: "Rain showers",
                95: "Thunderstorm"
            }
            weather_info["condition"] = weather_code_map.get(current.get("weathercode"), "Unknown")
            return weather_info
        except requests.RequestException as e:
            print("Weather API error:", e)
    
# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(layout="wide", page_title="SOLAR ENERGY DASHBOARD", page_icon="☀️")

# ---------------------- INITIALIZE ---------------------- #
locator = Nominatim(user_agent="SOLAR_MAINTENANCE_APP")
location = locator.geocode("HYDERABAD, INDIA")
anomolyDetection = AnomolyDetection()
prediction = EnergyPrediction()

# ---------------------- GET WEATHER ---------------------- #
weather_data = anomolyDetection.getCurrentWeather(location.latitude, location.longitude)
print(f"Current Weather: {weather_data}")
# ---------------------- DATA GENERATION ---------------------- #
# Import necessary libraries at the top of your script

# --- Main App Logic ---

# 1. INITIAL DATA LOADING AND PROCESSING (RUNS ONLY ONCE)
# This entire block runs only on the first load of the session, thanks to session_state.
# This prevents reloading and reprocessing data on every user interaction.
from zoneinfo import ZoneInfo 
if "data_loaded" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
import pandas as pd
from datetime import datetime
import pytz

# Set timezone (India Standard Time)
IST = pytz.timezone('Asia/Kolkata')

# Simulated data generation call (replace with actual function)
df = prediction.dataGeneration()

# Preprocessing
df = df.sort_values("DATE_TIME")
df["AC_POWER"] *= 10
df["DC_POWER"] = df["DC_POWER"].apply(lambda x: max(x, 0))
df["AC_POWER"] = df["AC_POWER"].apply(lambda x: max(x, 0))
df.to_csv("processedData.csv", index=False)

# Ensure DATE_TIME is datetime type with timezone awareness
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df["DATE_TIME"] = df["DATE_TIME"].dt.tz_localize('Asia/Kolkata', ambiguous='NaT', nonexistent='NaT')

# Get current datetime and date with timezone
def get_datetime():
    now = datetime.now(IST)
    today_date = now.date()
    return now, today_date

now, today_date = get_datetime()
print(f"Today's date: {today_date}")
print(f"now: {now}")
print(type(today_date))
print(df["DATE_TIME"].dtype)

# Filter today's data
today_df = df[df["DATE_TIME"].dt.date == today_date]

# ---------------------- CURRENT STATS ---------------------- #
totalDC = today_df["DC_POWER"].sum()
totalAC = today_df["AC_POWER"].sum()

# Find closest row to current time (both timezone-aware)
df["time_diff"] = (df["DATE_TIME"] - now).abs()
print("DataFrame with time_diff:", df[["DATE_TIME", "time_diff"]].head())
closest_row = df.loc[df["time_diff"].idxmin()]

# Current AC and DC from closest row
print(f"Closest row time: {closest_row['DATE_TIME']}")
current_ac = closest_row["AC_POWER"]
current_dc = closest_row["DC_POWER"]
print(f"Current AC: {current_ac}, Current DC: {current_dc}")

# Efficiency calculation
efficiency = (totalAC / totalDC) * 100 if totalDC > 0 else 0
print(f"Efficiency: {efficiency:.2f}%")

# Get current AC and DC power from the closest row
print(f"Closest row time: {closest_row['DATE_TIME']}")
current_ac = closest_row["AC_POWER"]
current_dc = closest_row["DC_POWER"]
print(f"Current AC: {current_ac}, Current DC: {current_dc}")
efficiency = (totalAC / totalDC) * 100 if totalDC > 0 else 0

print(f"Total DC: {totalDC:.2f} kWh, Total AC: {totalAC:.2f} kWh, Current AC: {current_ac:.2f} kW, Current DC: {current_dc:.2f} kW, Efficiency: {efficiency:.2f}%")

# ---------------------- HEADER ---------------------- #
st.header("☀️ Solar Energy Monitoring Dashboard")
st.caption(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------- METRICS ---------------------- #
weather_condition = weather_data.get("condition", "")
condition_emojis = {
    "Sunny": "☀️", "Clear": "🌞", "Overcast": "☁️", "Rain": "🌧️",
    "Cloudy": "⛅", "Thunderstorm": "⛈️", "Snow": "❄️", "Fog": "🌫️", "Windy": "🌬️"
}
weather_icon = condition_emojis.get(weather_condition, "🌀")
weather_temp = f"{weather_data.get('temperature', '--')}°C"
weather_label = f"🌍 Hyd, Telangana\n{weather_icon} {weather_condition}"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("TODAY'S TOTAL DC", f"{totalDC:.2f} kWh")
col2.metric("TODAY'S TOTAL AC", f"{totalAC:.2f} kWh")
col3.metric("CURRENT PRODUCTION (DC/AC)", f"{current_dc:.2f}/{current_ac:.2f} kW")
col4.metric("WEATHER", weather_temp, weather_label)
col5.metric("🔋 Efficiency", f"{efficiency:.2f} %")

# ---------------------- MAIN CHARTS ---------------------- #
main_col, side_col = st.columns([3, 1])

with main_col:
    st.subheader("Today's Energy Production (15-min intervals)")
    current_data = today_df[today_df["DATE_TIME"] <= now]

    if current_data.empty:
        st.warning("No data available for today yet.")
    else:
        chart_data = pd.DataFrame({
            "Time": current_data["DATE_TIME"],
            "AC Power": current_data["AC_POWER"],
            "DC Power": current_data["DC_POWER"]
        })
        fig = px.line(chart_data, x="Time", y=["AC Power", "DC Power"],
                      labels={"value": "Power (kW)", "variable": "Type"},
                      title=f"AC & DC Power from 00:00 to {now.strftime('%H:%M')}",
                      height=300)
        fig.update_traces(mode="lines+markers")
        fig.update_layout(yaxis_range=[0, 300])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Next 5 Days Energy Forecast")
    next_5_days = [today_date + timedelta(days=i) for i in range(5)]
    forecast_base = pd.DataFrame({
        "DATE": next_5_days,
        "Day": [d.strftime("%a") for d in next_5_days]
    })

    df["DATE"] = df["DATE_TIME"].dt.date
    forecast_df = df[df["DATE"].isin(next_5_days)]

    forecast_summary = forecast_df.groupby("DATE").agg({
        "AC_POWER": "sum",
        "DC_POWER": "sum"
    }).reset_index()
    forecast_summary["Day"] = pd.to_datetime(forecast_summary["DATE"]).dt.strftime("%a")

    merged_forecast = forecast_base.merge(forecast_summary, on=["DATE", "Day"], how="left").fillna(0)
    fig2 = px.bar(
        merged_forecast,
        x="Day",
        y="AC_POWER",
        labels={"AC_POWER": "AC Energy (kWh)", "Day": "Day of Week"},
        title="Forecasted Daily Energy Production",
        height=300
    )
    fig2.update_traces(
        customdata=np.stack([merged_forecast["AC_POWER"], merged_forecast["DC_POWER"]], axis=1),
        hovertemplate="%{x}<br>AC: %{y:.2f} kWh<br>DC: %{customdata[1]:.2f} kWh"
    )
    fig2.update_layout(
        yaxis_title="Energy (kWh)",
        xaxis_title="Day of Week"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Solar Plant Image
    st.subheader("SOLAR PLANT OVERVIEW")
    image_path = "imageSolar.jpg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True, caption="Solar Plant Layout")
    else:
        st.error("❌ Plant layout image not found.")

    # Efficiency of each inverter
    st.subheader("⚙️ System and Model Efficiency")
    model_efficiencies = {
        "INV-01": 94.5, "INV-02": 93.8, "INV-03": 91.2, "INV-04": 87.9,
        "INV-05": 92.5, "INV-06": 89.3, "INV-07": 90.7, "INV-08": 86.5,
        "INV-09": 95.1, "INV-10": 93.4,
    }
    eff_df = pd.DataFrame(list(model_efficiencies.items()), columns=["Inverter", "Efficiency (%)"])
    fig_eff = px.bar(eff_df, x="Inverter", y="Efficiency (%)", color="Efficiency (%)",
                     color_continuous_scale="Viridis", range_y=[80, 100], height=300)
    st.plotly_chart(fig_eff, use_container_width=True)

    # 5-Day Weather Forecast (Static Example)
    st.subheader("🌦️ 5-Day Weather Forecast")
    forecast_days = [ (today_date + timedelta(days=i)).strftime("%a") for i in range(5) ]
    forecast_highs = [32, 32, 29, 31, 33]
    forecast_lows = [24, 24, 24, 24, 25]
    forecast_icons = ["⛅", "☁️", "☁️", "⛅", "☁️"]
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"### {forecast_days[i]}")
            st.markdown(f"{forecast_icons[i]}")
            st.markdown(f"**{forecast_highs[i]}° / {forecast_lows[i]}°**")

# ---------------------- INVERTER STATUS ---------------------- #
with side_col:
    st.subheader("Inverter Status")

    inverter_status = {f"INV-{i+1:02d}": np.random.randint(30, 100) for i in range(22)}
    def get_color(value):
        if value >= 80:
            return "#2478D1"
        elif value >= 50:
            return "#1d9dc7"
        else:
            return "#953a28"

    index = 0
    for row in range(8):
        cols = st.columns(3)
        for col in cols:
            if index < 22:
                inv = f"INV-{index+1:02d}"
                perf = inverter_status[inv]
                color = get_color(perf)
                col.markdown(
                    f"""
                    <div style='background-color:{color}; height:80px; width:80px;
                        display:flex; align-items:center; justify-content:center;
                        border-radius:8px; font-size:14px; font-weight:700;
                        color:white; text-align:center; box-shadow:1px 1px 5px rgba(0,0,0,0.2);'>
                        {inv}
                    </div>
                    """, unsafe_allow_html=True)
                index += 1

    st.subheader("⚠ Anomalies Detected")
    st.warning("*INV-04*: Low output (12:45 PM)")
    st.warning("*INV-08*: Voltage fluctuation (2:15 PM)")
    st.warning("*INV-16*: Irregular pattern (4:30 PM)")

# ---------------------- SOLAR ASSISTANT (SIDEBAR CHAT) ---------------------- #
with st.sidebar:
    st.subheader("💬 Solar Assistant")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "inverter_status" not in st.session_state:
        st.session_state["inverter_status"] = None

    if "documents_loaded" not in st.session_state:
        st.session_state["documents_loaded"] = False


    # User Input
    user_input = st.text_input("Ask about your solar performance:")

    # If user submitted a query
    if user_input:
        try:
            # Inline prompt formatting
            inverter_info = "\n".join([
                f"  {inv}: {val}%" for inv, val in st.session_state.inverter_status.items()
            ])
            today_data = st.session_state.todays_df.to_dict(orient="records")

            full_prompt = f"""
            You are a solar performance assistant. Use the following system context for all responses:

            - 🌞 Today's total system efficiency: {st.session_state.efficiency:.2f}%
            - ⚡ Current AC Power: {st.session_state.current_ac:.2f} kW
            - ⚡ Current DC Power: {st.session_state.current_dc:.2f} kW
            - 📊 Today's total DC Power: {st.session_state.totalDC:.2f} kWh
            - 📊 Today's total AC Power: {st.session_state.totalAC:.2f} kWh
            - 🌤️ Current weather: {st.session_state.weather_data}°C
            - 📅 Today's Data: {today_data}
            - 🔋 Inverter Performance:
            {inverter_info}

            User Question: {user_input}

            Respond clearly and use relevant data to support the answer.
            """

            # Call your working model
            assistant = AnomolyDetection()
            response = assistant.getResponse(full_prompt)  # <--- your existing method
            print(f"Assistant Response: {response}")

            # Save and display
            st.session_state.chat_log.append({"user": user_input, "bot": response})
            st.rerun()

        except Exception as e:
            st.error(f"Error responding to query: {e}")  # Optional debug info



st.divider()
footer1, footer2, footer3 = st.columns(3)
footer1.progress(int(efficiency), text="System Efficiency")
footer2.metric("Energy Saved", f"{round(totalAC * 0.25, 2)} kWh")
footer3.metric("Faults Detected", "24", "3 since yesterday")
# fuck u uday 