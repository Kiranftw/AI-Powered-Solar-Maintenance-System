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
import datetime
from geopy.geocoders import Nominatim
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from datetime import datetime, timedelta
import time
from EnergyPrediction import EnergyPrediction
import requests
from typing import List, Dict, Any
import markdown
import google.generativeai as genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationSummaryBufferMemory
import pytesseract
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, CSVLoader, Docx2txtLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pytesseract.pytesseract.tesseract_cmd = r'/home/kiranftw/OFFICE/AI-Powered-Solar-Maintenance-System/venv/bin/pytesseract' 


class AnomolyDetection():
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.GenerationData = pandas.read_csv(os.path.join(os.getcwd(), "DATASETS","Plant_1_Generation_Data.csv" ))
        self.WeatherData = pandas.read_csv(os.path.join(os.getcwd(), "DATASETS","Plant_1_Weather_Sensor_Data.csv"))
        self.locator = Nominatim(user_agent="solar_maintenance APP")
        self.energyprediction = EnergyPrediction()
        _API = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=_API)
        self.chatmodel = genai.GenerativeModel(
                model_name='models/gemini-2.5-flash',
                generation_config={},
                safety_settings={},
                tools=None,
                system_instruction=None,
            )
        self.chatmemory = None
        self.corpus = []
    
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
    
    def trainDCmodel(self, dataframe: pandas.DataFrame = None) -> None:
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
    
    def documentLoader(self, file_path: str) -> list:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_extension == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif file_extension in [".jpg", ".jpeg", ".png"]:
            # For images, we can use OCR to extract text
            from langchain_community.document_loaders import ImageLoader
            loader = ImageLoader(file_path, pytesseract=pytesseract.image_to_string)

        return loader.load()
    
    def chatbot(self, document: str) -> markdown.markdown:
        content: str = ""
        if not document:
            content = "No document provided."
        else:
            try:
                documents = self.documentLoader(document)
                if not documents:
                    content = "No content found in the document."
                else:
                    self.corpus = documents
                    self.chatmemory = ConversationSummaryBufferMemory(llm=self.chatmodel, max_token_limit=10000, return_messages=True)
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are a helpful assistant."),
                        HumanMessage(content="What is the content of the document?"),
                        AIMessage(content="Please summarize the document.")
                    ])
                    runnable = prompt | RunnablePassthrough()
                    response = runnable.invoke(self.corpus)
                    content = response.content
            except Exception as e:
                logging.error(f"Error processing document: {e}")
                content = "An error occurred while processing the document."
        return markdown.markdown(content)

st.set_page_config(layout="wide", page_title="SOLAR ENERGY DASHBOARD", page_icon="☀️")
from EnergyPrediction import EnergyPrediction
locator = Nominatim(user_agent="SOLAR_MAINTENANCE_APP")
city = "HYDERABAD, INDIA"
location = locator.geocode(city)

anomolyDetection = AnomolyDetection()
prediction = EnergyPrediction()

weather_data = anomolyDetection.getCurrentWeather(location.latitude, location.longitude)

if 'last_refresh' not in st.session_state:
   st.session_state.last_refresh = datetime.now()

dataframe = prediction.dataGeneration()
dataframe = dataframe.sort_values("DATE_TIME")
dataframe['AC_POWER'] = dataframe['AC_POWER'] * 10
dataframe['DC_POWER'] = dataframe['DC_POWER'].apply(lambda x: max(x, 0))
dataframe['AC_POWER'] = dataframe['AC_POWER'].apply(lambda x: max(x, 0))
dataframe.to_csv("processedData.csv", index=False)
logging.info("Data saved to processedData.csv")

dataframe["DATE_TIME"] = pd.to_datetime(dataframe["DATE_TIME"])
target_day = dataframe["DATE_TIME"].dt.date.min()

start_time = pd.Timestamp(target_day)
end_time = start_time + timedelta(days=1)

todaysDataFrame = dataframe[
    (dataframe["DATE_TIME"] >= start_time) & (dataframe["DATE_TIME"] <= end_time)
]
print(len(todaysDataFrame))
for index, row in todaysDataFrame.iterrows():
    print(f"Timestamp: {row['DATE_TIME']}, AC Power: {row['AC_POWER']}, DC Power: {row['DC_POWER']}")


totalDC = todaysDataFrame["DC_POWER"].sum()
totalAC = todaysDataFrame["AC_POWER"].sum()

now = datetime.now()
print(now)
dataframe["time_diff"] = (dataframe["DATE_TIME"] - now).abs()
closest_row = dataframe.loc[dataframe["time_diff"].idxmin()]


current_ac = closest_row["AC_POWER"]
current_dc = closest_row["DC_POWER"]

efficiency = (totalAC / totalDC) * 100 if totalDC > 0 else 0

print("Today's total DC:", totalDC)
print("Today's total AC:", totalAC)
print("Current Production (DC/AC):", f"{current_dc:.2f}/{current_ac:.2f} kW")
print("Efficiency:", f"{efficiency:.2f}%")
print("Filtered Data (first 10 rows):")
print(todaysDataFrame.head(10))
dataframe["DATE_TIME"] = pd.to_datetime(dataframe["DATE_TIME"])

unique_dates = dataframe["DATE_TIME"].dt.date.unique()
unique_dates.sort()

for day in unique_dates:
    start_time = pd.Timestamp(day)
    end_time = start_time + timedelta(days=1)

    day_df = dataframe[(dataframe["DATE_TIME"] >= start_time) & (dataframe["DATE_TIME"] <= end_time)]

    total_dc = day_df["DC_POWER"].sum()
    total_ac = day_df["AC_POWER"].sum()

    efficiency = (total_ac / total_dc) * 100 if total_dc > 0 else 0

    print(f"\nDate: {day}")
    print(f"  Total DC Power: {total_dc:.2f} kW")
    print(f"  Total AC Power: {total_ac:.2f} kW")
    print(f"  Efficiency: {efficiency:.2f}%")

st.header("☀️ Solar Energy Monitoring Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

condition_emojis = {
    "Sunny": "☀️", "Clear": "🌞", "Overcast": "☁️", "Rain": "🌧️",
    "Cloudy": "⛅", "Thunderstorm": "⛈️", "Snow": "❄️", "Fog": "🌫️", "Windy": "🌬️"
}
weather_condition = weather_data.get("condition", "")
weather_icon = condition_emojis.get(weather_condition, "🌀")
weather_temp = f"{weather_data.get('temperature', '--')}°C"
weather_label = f"🌍 Hyd, Telangana\n{weather_icon} {weather_condition}"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("TODAY'S TOTAL DC", f"{totalDC:.2f} kWh")
with col2:
    st.metric("TODAY'S TOTAL AC", f"{totalAC:.2f} kWh")
with col3:
    st.metric("CURRENT PRODUCTION (DC/AC)", f"{current_dc:.2f}/{current_ac:.2f} kW")
with col4:
    st.metric("WEATHER", weather_temp, weather_label)
with col5:
    st.metric("🔋 Efficiency", f"{efficiency:.2f} %")

main_col, side_col = st.columns([3, 1])
with main_col:
    st.subheader("Today's Energy Production (15-min intervals)")
    today = datetime.now().date()
    now = datetime.now()
    today_df = dataframe[(dataframe["DATE_TIME"].dt.date == today) & (dataframe["DATE_TIME"] <= now)]

    if today_df.empty:
        st.warning("No data available for today yet.")
    else:
        chart_data = pd.DataFrame({
            "Time": today_df["DATE_TIME"],
            "AC Power": today_df["AC_POWER"],
            "DC Power": today_df["DC_POWER"]
        })
        fig = px.line(
            chart_data, x="Time", y=["AC Power", "DC Power"],
            labels={"value": "Power (kW)", "variable": "Type"},
            height=300,
            title=f"AC & DC Power from 00:00 to {now.strftime('%H:%M')}"
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(
            yaxis_range=[0, 300],
            yaxis_title="Power (kW)",
            legend_title_text='Power Type'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Next 5 Days Energy Forecast")
    next_5_days = [today + timedelta(days=i) for i in range(5)]
    complete_forecast = pd.DataFrame({
        "DATE": next_5_days,
        "Day": [d.strftime("%a") for d in next_5_days]
    })

    dataframe["DATE"] = dataframe["DATE_TIME"].dt.date
    forecast_df = dataframe[dataframe["DATE"].isin(next_5_days)]

    forecast_summary = forecast_df.groupby("DATE").agg({
        "AC_POWER": "sum",
        "DC_POWER": "sum"
    }).reset_index()
    forecast_summary["Day"] = pd.to_datetime(forecast_summary["DATE"]).dt.strftime("%a")

    forecast_merged = complete_forecast.merge(
        forecast_summary, on=["DATE", "Day"], how="left"
    ).fillna({"AC_POWER": 0, "DC_POWER": 0})
    custom = np.stack([forecast_merged["AC_POWER"], forecast_merged["DC_POWER"]], axis=1)

    fig2 = px.bar(
        forecast_merged,
        x="Day",
        y="AC_POWER",
        labels={"AC_POWER": "AC Energy (kWh)", "Day": "Day of Week"},
        height=300,
        title="Forecasted Daily Energy Production"
    )
    fig2.update_traces(
        customdata=custom,
        hovertemplate="%{x}<br>AC: %{y:.2f} kWh<br>DC: %{customdata[1]:.2f} kWh"
    )

    max_val = max(forecast_merged["AC_POWER"].max(), 1)
    y_step = 10000
    y_max = ((int(max_val / y_step) + 1) * y_step)
    y_ticks = list(range(0, y_max + 1, y_step))
    y_labels = [f"{int(v / 1000)}k" for v in y_ticks]

    fig2.update_layout(
        yaxis=dict(
            tickvals=y_ticks,
            ticktext=y_labels,
            range=[0, y_max]
        ),
        xaxis_title="Day of Week",
        yaxis_title="Energy (kWh)",
        margin=dict(t=50, b=50)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("SOLAR PLANT OVERVIEW")
    image_path = "/home/kiranftw/OFFICE/AI-Powered-Solar-Maintenance-System/imageSolar.jpg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True, caption="Solar Plant Layout")
    else:
        st.error("❌ Plant layout image not found.")

 
    #TODO Now we have to work on the inverter part of the code
    st.subheader("⚙️ System and Model Efficiency")
    model_efficiencies = {
        "INV-01": 94.5, "INV-02": 93.8, "INV-03": 91.2, "INV-04": 87.9,
        "INV-05": 92.5, "INV-06": 89.3, "INV-07": 90.7, "INV-08": 86.5,
        "INV-09": 95.1, "INV-10": 93.4,
    }
    eff_df = pd.DataFrame(list(model_efficiencies.items()), columns=["Inverter", "Efficiency (%)"])
    fig_eff = px.bar(
        eff_df, x="Inverter", y="Efficiency (%)", color="Efficiency (%)",
        color_continuous_scale="Viridis", range_y=[80, 100], height=300
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    # New 5-day weather forecast block (static or replace with real data)
    st.subheader("🌦️ 5-Day Weather Forecast")
    # Example static; replace with actual data if available
    forecast_days = [ (today + timedelta(days=i)).strftime("%a") for i in range(5) ]
    forecast_highs = [32, 32, 29, 31, 33]
    forecast_lows = [24, 24, 24, 24, 25]
    forecast_icons = ["⛅", "☁️", "☁️", "⛅", "☁️"]
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"### {forecast_days[i]}")
            st.markdown(f"{forecast_icons[i]}")
            st.markdown(f"**{forecast_highs[i]}° / {forecast_lows[i]}°**")

with side_col:
    st.subheader("Inverter Status")

    # Simulated performance data (replace with real values if needed)
    inverter_status = {}
    for i in range(22):
        performance = np.random.randint(30, 100)  # Simulated values
        inverter_status[f"INV-{i+1:02d}"] = performance

    def get_color(value):
        if value >= 80:
            return "#2478D1"  # Blue
        elif value >= 50:
            return "#1d9dc7"  # Orange
        else:
            return "#953a28"  # Red

    index = 0
    for row in range(8):
        cols = st.columns(3)
        for col in cols:
            if index < 22:
                inv_name = f"INV-{index+1:02d}"
                perf = inverter_status[inv_name]
                color = get_color(perf)
                col.markdown(
                    f"""
                    <div style='
                        background-color: {color};
                        height: 80px;
                        width: 80px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: 700;
                        color: white;
                        text-align: center;
                        box-shadow: 1px 1px 5px rgba(0,0,0,0.2);
                        margin-bottom: 10px;
                    ' title='{inv_name}: {perf}%'>
                        {inv_name}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                col.empty()
            index += 1

    st.subheader("⚠ Anomalies Detected")
    st.warning("*INV-04*: Low output (12:45 PM)")
    st.warning("*INV-08*: Voltage fluctuation (2:15 PM)")
    st.warning("*INV-16*: Irregular pattern (4:30 PM)")

with st.sidebar:
    st.subheader("💬 Solar Assistant")
    chat_history = st.expander("Chat History", expanded=True)
    user_input = st.text_input("Ask about your solar performance:")
    if user_input:
        bot_response = "Based on current data, your system is performing at 92% efficiency. No critical issues detected."
        with chat_history:
            st.text(f"You: {user_input}")
            st.text(f"Assistant: {bot_response}")

st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.progress(int(efficiency), text="System Efficiency")
with footer_col2:
    energy_saved_kwh = round(totalAC * 0.25, 2)
    st.metric("Energy Saved", f"{energy_saved_kwh} kWh")
with footer_col3:
    st.metric("Faults Detected", "24", "3 since yesterday")

dataframe["DATE_TIME"] = pd.to_datetime(dataframe["DATE_TIME"])

unique_dates = dataframe["DATE_TIME"].dt.date.unique()
unique_dates.sort()

for day in unique_dates:
    start_time = pd.Timestamp(day)
    end_time = start_time + timedelta(days=1)

    day_df = dataframe[(dataframe["DATE_TIME"] >= start_time) & (dataframe["DATE_TIME"] <= end_time)]

    total_dc = day_df["DC_POWER"].sum()
    total_ac = day_df["AC_POWER"].sum()

    efficiency = (total_ac / total_dc) * 100 if total_dc > 0 else 0

    print(f"\nDate: {day}")
    print(f"  Total DC Power: {total_dc:.2f} kW")
    print(f"  Total AC Power: {total_ac:.2f} kW")
    print(f"  Efficiency: {efficiency:.2f}%")

