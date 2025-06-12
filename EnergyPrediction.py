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
import requests
import datetime
import time
from markdown import markdown

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnergyPrediction(object):
    def __init__(self, modelname: str = "models/gemini-2.0-flash") -> None:
        load_dotenv(find_dotenv())
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(self.DIR):
            if filename.endswith('.csv') and filename == "Processed_data.csv":
                self.dataframe: pd.DataFrame = pd.read_csv(os.path.join(self.DIR, filename))
                logging.info(f"Loaded data from {filename}")
        self.RegressionModel: LinearRegression = LinearRegression()
        self.ForestModel: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.modelpath = os.path.join(self.DIR, "energy_prediction_model.pkl")
        self.WEATHER_API = os.getenv("WEATHER_API")
        genai.configure(api_key=os.getenv("GENAI_API_KEY"))
        for model in genai.list_models():
            pass
        self.GenerativeMODEL: genai.GenerativeModel = genai.GenerativeModel(
            model_name='models/gemini-2.0-flash',
            safety_settings={},
            tools=None,
            system_instruction=None,
        )

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
    def getResponse(self, prompt: str) -> str:
        try:
            response = self.GenerativeMODEL.generate_content(prompt)
            response.resolve()
            if response.text is None:
                logging.error("No response text received from the model.")
                return None
            return response.text
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return None
   
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
    def ModelPrediction(self, ambient_temperature: float, irradiation: float, module_temperature: float) -> tuple:
        SCALARmodel = joblib.load(os.path.join(self.DIR, "models", "INPUT_SCALER.pkl"))
        ACmodel     = joblib.load(os.path.join(self.DIR, "models", "RF_AC_POWER_MODEL.pkl"))
        DCmodel     = joblib.load(os.path.join(self.DIR, "models", "RF_DC_POWER_MODEL.pkl"))
        if not all([SCALARmodel, ACmodel, DCmodel]):
            logging.error("One or more model files are missing. Please ensure all models are trained and saved.")
            return None
        input_data = pd.DataFrame([[ambient_temperature, irradiation, module_temperature]])

        scaled_input = SCALARmodel.transform(input_data)

        dc_prediction = DCmodel.predict(scaled_input)
        ac_prediction = ACmodel.predict(scaled_input)

        return dc_prediction[0], ac_prediction[0]

    @ExceptionHandelling
    def dataGeneration(self, latitude: float = 17.3850, longitude: float =  78.4867) -> pd.DataFrame:
        dataframe: pd.DataFrame = self.getWeatherData(latitude, longitude, days=2)
        if dataframe is None:
            logging.error("Failed to fetch weather data.")
            return None
        logging.info("REAL TIME WEATHER DATA")
        #NOTE.: The weather API is giving irradiance in W/m², we need to convert it to kW/m² -> kWh_per_m2 = (irradiance_W_per_m2) * (15 / 60) / 1000
        starttime = datetime.datetime.now()
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
        print(dataframe.head(10))
        print(dataframe.info())
        #TODO: add new column DC current and AC current usinf the self.ModelPrediction method
        for index, row in dataframe.iterrows():
            ambient_temperature = row['Ambient Temp (°C)']
            irradiation = row['IRRADIATION (kWh/m²)']
            module_temperature = row['MODULE_TEMPERATURE']
            if pd.isna(module_temperature):
                continue
            dc_current, ac_current = self.ModelPrediction(ambient_temperature, irradiation, module_temperature)
            dataframe.at[index, 'DC_CURRENT'] = dc_current
            dataframe.at[index, 'AC_CURRENT'] = ac_current
        print("TIME TAKEN:",starttime - datetime.datetime.now())
        dataframe.to_csv(os.path.join(self.DIR, "ProcessedModel_data.csv"), index=False)
        logging.info("Data generation completed and saved to Processed_data.csv")
        print(dataframe.to_string(index=False))
        print(dataframe.info())
        # Save the dataframe to a CSV file
        return dataframe
    
    @ExceptionHandelling
    def detect_anomalies(self, dataframe: pd.DataFrame, model, target_column, features):
        X = dataframe[features]
        predicted_power = model.predict(X)
        error = abs(dataframe[target_column] - predicted_power)
        threshold_error = error.quantile(0.95)
        anomalies = dataframe[error > threshold_error]
        return anomalies
    
    @ExceptionHandelling
    def anomalyDetection(self, dataframe: pd.DataFrame) -> dict:
        if dataframe is None or dataframe.empty:
            logging.error("Dataframe is empty or None.")
            return None

        ACmodel = joblib.load(os.path.join(self.DIR, "models", "ac_model.pkl"))
        DCmodel = joblib.load(os.path.join(self.DIR, "models", "dc_model.pkl"))
        if not all([ACmodel, DCmodel]):
            logging.error("One or more model files are missing. Please ensure all models are trained and saved.")
            return None

        # Rename columns to match the feature names used during training
        dataframe = dataframe.rename(columns={
            "IRRADIATION (kWh/m²)": "IRRADIATION",
            "Ambient Temp (°C)": "AMBIENT_TEMPERATURE"
        })

        features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]

        ac_anomalies = self.detect_anomalies(dataframe, ACmodel, "AC_CURRENT", features)
        dc_anomalies = self.detect_anomalies(dataframe, DCmodel, "DC_CURRENT", features)

        return {
            "ac_anomalies": ac_anomalies,
            "dc_anomalies": dc_anomalies
    }
if __name__ == "__main__":
    energy_prediction = EnergyPrediction()
    dataframe: pd.DataFrame = energy_prediction.dataGeneration()
    anomalies = energy_prediction.anomalyDetection(dataframe)
    if anomalies is not None:
        ac_anomalies = anomalies["ac_anomalies"]
        dc_anomalies = anomalies["dc_anomalies"]

        prompt = f"""
        We've detected anomalies in our solar power plant data. 
        The anomalies are as follows:

        AC Anomalies:
        {ac_anomalies}

        DC Anomalies:
        {dc_anomalies}

        Provide a detailed analysis of the possible causes of these anomalies and suggest potential actions to mitigate them.
        Consider factors such as weather conditions, equipment performance, and maintenance schedules.And make the point small and concise.
        """

        print("AC Anomalies:")
        print(ac_anomalies)
        print("DC Anomalies:")
        print(dc_anomalies)

        analysis = energy_prediction.getResponse(prompt)
        print(markdown.markdown(analysis))
    else:
        print("No anomalies detected or error occurred.")