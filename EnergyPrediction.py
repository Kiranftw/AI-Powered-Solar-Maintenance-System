import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv, find_dotenv
from functools import wraps
import logging
import joblib
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row
import requests
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnergyPrediction(object):
    def __init__(self) -> None:
        load_dotenv(find_dotenv())
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(self.DIR):
            if filename.endswith('.csv') and filename == "Processed_data.csv":
                self.dataframe: pd.DataFrame = pd.read_csv(os.path.join(self.DIR, filename))
                logging.info(f"Loaded data from {filename}")
        self.RegressionModel: LinearRegression = LinearRegression()
        self.ForestModel: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.modelpath = os.path.join(self.DIR, "energy_prediction_model.pkl")
        self.sparksession = SparkSession.builder.appName("SolarMaintenence Model").getOrCreate()
        self.WEATHER_API = os.getenv("WEATHER_API")
        print(self.WEATHER_API)

    @staticmethod
    def ExceptionHandelling(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"An error occurred in {func.__name__}: {e}")
                raise
        return wrapper
   
    @ExceptionHandelling
    #NOTE: 
    def modeltraining(self) -> None:
        X = self.dataframe[['AMBIENT_TEMPERATURE', 'IRRADIATION']]
        y = self.dataframe['MODULE_TEMPERATURE']

        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=0.2, 
                                                            random_state=42)
        self.RegressionModel.fit(X_train, y_train)

        y_pred = self.RegressionModel.predict(X_test)

        print("Test Results:")
        print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        print("R2 Score:", r2_score(y_test, y_pred))
 
        joblib.dump(self.RegressionModel, self.modelpath)
        logging.info(f"Model trained and saved to {self.modelpath}")
    
    @ExceptionHandelling
    def predictModuleTemperature(self, ambient_temperature, irradiation) -> float:
        if not os.path.exists(self.modelpath):
            logging.error("Model file does not exist. Please train the model first.")
            return None

        model = joblib.load(self.modelpath)

        input_df = pd.DataFrame([[ambient_temperature, irradiation]], 
                                columns=['AMBIENT_TEMPERATURE', 'IRRADIATION'])

        prediction = model.predict(input_df)
        logging.info(f"Predicted Module Temperature: {prediction[0]}")
        return float(prediction[0])
    
    @ExceptionHandelling
    def getWeatherData(self,latitude, longitude, days: int) -> pd.DataFrame:
        api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "auto",
            "forecast_days": days
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df.rename(columns={
                'time': 'Timestamp',
                'temperature_2m': 'Ambient Temp (°C)',
                'shortwave_radiation': 'Irradiance (W/m²)'
            }, inplace=True)

            # Interpolate to 15-minute intervals
            df.set_index('Timestamp', inplace=True)
            interpolated_df: pd.DataFrame = df.resample('15min').interpolate(method='linear')
            interpolated_df.reset_index(inplace=True)

            return interpolated_df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
        except KeyError:
            print("Error: Unexpected data format received from API.")
            return None
        
    @ExceptionHandelling
    def ModelPrediction(self):
        dc_model_path = os.path.join(self.DIR, "models", "DC_POWER_MODEL")
        ac_model_path = os.path.join(self.DIR, "models", "AC_POWER_MODEL")
        
        dc_model = PipelineModel.load(dc_model_path)
        ac_model = PipelineModel.load(ac_model_path)

        month = int(input("Enter month: "))
        day = int(input("Enter day: "))
        ambient_temperature = float(input("Enter ambient temperature: "))
        irradiation = float(input("Enter irradiation: "))

        module_temperature = self.predictModuleTemperature(ambient_temperature, irradiation)
        print(f"Predicted Module Temperature: {module_temperature}")
        if module_temperature is None:
            return
        
        InputRow = Row("month", "day_of_month", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION")
        input_row = InputRow(month, day, ambient_temperature, module_temperature, irradiation)

        input_df = self.sparksession.createDataFrame([input_row])

        dc_pred = dc_model.transform(input_df).select("prediction").first()[0]
        ac_pred = ac_model.transform(input_df).select("prediction").first()[0]

        print(f"DC Prediction: {dc_pred}")
        print(f"AC Prediction: {ac_pred}")

    @ExceptionHandelling
    def dataGeneration(self, latitude: float = 17.3850, longitude: float =  78.4867) -> pd.DataFrame:
        dataframe: pd.DataFrame = self.getWeatherData(latitude, longitude, days=5)
        if dataframe is None:
            logging.error("Failed to fetch weather data.")
            return None
        logging.info("REAL TIME WEATHER DATA")
        #NOTE.: The weather API is giving irradiance in W/m², we need to convert it to kW/m² -> kWh_per_m2 = (irradiance_W_per_m2) * (15 / 60) / 1000
        dataframe['IRRADIATION (kWh/m²)'] = (dataframe['Irradiance (W/m²)'] * (15 / 60)) / 1000
        moduleTemperature = list()
        for index, row in dataframe.iterrows():
            ambient_temperature = row['Ambient Temp (°C)']
            irradiation = row['IRRADIATION (kWh/m²)']
            module_temp = self.predictModuleTemperature(ambient_temperature, irradiation)
            if module_temp is not None:
                moduleTemperature.append(module_temp)
            else:
                moduleTemperature.append(None)
        dataframe['MODULE_TEMPERATURE'] = moduleTemperature

if __name__ == "__main__":
    energy_prediction = EnergyPrediction()
    energy_prediction.dataGeneration()