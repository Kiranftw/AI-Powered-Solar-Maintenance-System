import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
import logging
import numpy as np
import pandas
from dotenv import load_dotenv, find_dotenv
from functools import wraps
import logging
import joblib
import requests
import datetime
import xgboost as xgb
import datetime
from tabulate import tabulate
from scipy.stats import zscore
from markdown import markdown
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnergyPrediction(object):
    """
    NOTE: Avoid showing AC & DC current from 12:00 to 5:45am & 
    """
    def __init__(self, modelname: str = "models/gemini-2.0-flash") -> None:
        load_dotenv(find_dotenv())
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(self.DIR):
            if filename.endswith('.csv') and filename == "Processed_data.csv":
                self.dataframe: pd.DataFrame = pd.read_csv(os.path.join(self.DIR, filename))
                logging.info(f"Loaded data from {filename}")
        self.RegressionModel = LinearRegression()
        self.ForestModel: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.modelpath = os.path.join(self.DIR, "energy_prediction_model.pkl")
        self.WEATHER_API = os.getenv("WEATHER_API")
        self.geolocator = Nominatim(user_agent="solar_maintenance APP")
        self.XGboostModel: xgb.XGBRegressor = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.GenerationData = pandas.read_csv(os.path.join(os.getcwd(), "DATASETS","Plant_1_Generation_Data.csv" ))
        self.WeatherData = pandas.read_csv(os.path.join(os.getcwd(), "DATASETS","Plant_1_Weather_Sensor_Data.csv"))
    
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
                return ""
            return response.text
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return ""
   
    @ExceptionHandelling
    def modeltraining(self) -> None:
        X = self.dataframe[['AMBIENT_TEMPERATURE', 'IRRADIATION']]
        y = self.dataframe['MODULE_TEMPERATURE']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.RegressionModel = xgb.XGBRegressor(
        objective='reg:squarederror',  # For regression tasks
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
        )

        self.RegressionModel.fit(X_train, y_train)

        y_pred = self.RegressionModel.predict(X_test)

        print("✅ XGBoost Model Test Results:")
        print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
        print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
        print("R² Score:", r2_score(y_test, y_pred))

        # Save model
        joblib.dump(self.RegressionModel, self.modelpath)
        logging.info(f"🚀 XGBoost model trained and saved to {self.modelpath}")
    
    @ExceptionHandelling
    def predictModuleTemperature(self, ambient_temperature, irradiation) -> float:
        if not os.path.exists(self.modelpath):
            logging.error("Model file does not exist. Please train the model first.")
            return ambient_temperature  # or some other default value

        model = joblib.load(self.modelpath)
        input_df = pd.DataFrame([[ambient_temperature, irradiation]], 
                                columns=['AMBIENT_TEMPERATURE', 'IRRADIATION'])

        prediction = model.predict(input_df)
        if np.isnan(prediction[0]):
            logging.warning("Predicted Module Temperature is NaN. Using ambient temperature instead.")
            return ambient_temperature 
        # logging.info(f"Predicted Module Temperature: {prediction[0]}")
        return float(prediction[0])
    
    def convertIrradiance(self, irradiance: float) -> float:
        """Converts irradiance from W/m² to kWh/m² assuming 15-minute intervals."""
        return (irradiance * (15 / 60)) / 1000
    
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
            return pd.DataFrame()
        except KeyError:
            print("Error: Unexpected data format received from API.")
            return pd.DataFrame()
    
    from typing import Optional

    def trainDCmodel(self, dataframe: Optional[pandas.DataFrame] = None) -> xgb.XGBRegressor:
        if dataframe is None:
            dataframe = self.dataAggregation()
        dataframe['DATE_TIME'] = pd.to_datetime(dataframe['DATE_TIME'])

        dataframe['HOUR'] = dataframe['DATE_TIME'].dt.hour
        dataframe['MINUTE'] = dataframe['DATE_TIME'].dt.minute
        dataframe['DAY_OF_WEEK'] = dataframe['DATE_TIME'].dt.dayofweek
        dataframe['IS_DAYLIGHT'] = dataframe['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

        dataframe.dropna(subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER"], inplace=True)

        features = [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
            "HOUR",
            "MINUTE",
            "DAY_OF_WEEK",
            "IS_DAYLIGHT"
        ]
        target = "DC_POWER"
        X = dataframe[features]
        y = dataframe[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(f"DC_POWER Model Trained:\nRMSE: {rmse:.2f}\nR² Score: {r2:.2f}")
        joblib.dump(model, "models/timebasedDCmodel.pkl")
        return model
    
    @ExceptionHandelling
    def testACmodel(self):
        # Load trained AC model
        self.ACmodel = joblib.load(os.path.join(self.DIR, "models", "simpleACmodel.pkl"))
        file_path = os.path.join(os.getcwd(), "Combined_Generation_Weather_Data.csv")
        test_dataframe = pd.read_csv(file_path)
        print("Columns in test data:", test_dataframe.columns)
        test_dataframe['DATE_TIME'] = pd.to_datetime(test_dataframe['DATE_TIME'])

        test_dataframe.dropna(subset=[
            'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
            'DC_POWER', 'AC_POWER'
        ], inplace=True)

        test_dataframe['HOUR'] = test_dataframe['DATE_TIME'].dt.hour
        test_dataframe['MINUTE'] = test_dataframe['DATE_TIME'].dt.minute
        test_dataframe['DAY_OF_WEEK'] = test_dataframe['DATE_TIME'].dt.dayofweek
        test_dataframe['IS_DAYLIGHT'] = test_dataframe['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)
        feature_cols = [
            'AMBIENT_TEMPERATURE',
            'MODULE_TEMPERATURE',
            'IRRADIATION',
            'HOUR',
            'MINUTE',
            'DAY_OF_WEEK',
            'IS_DAYLIGHT',
            'DC_POWER'
        ]
        X_test = test_dataframe[feature_cols]
        y_true = test_dataframe['AC_POWER']
        y_pred = self.ACmodel.predict(X_test)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print("\nSample Predictions:")
        test_dataframe['PREDICTED_AC_POWER'] = y_pred
        for index, row in test_dataframe.head(100).iterrows():
            print(f"{row['DATE_TIME']} | Actual AC: {row['AC_POWER']:.2f} | Predicted AC: {row['PREDICTED_AC_POWER']:.2f}")

        print(f"\nR² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        return test_dataframe
    
    @ExceptionHandelling
    def simpleACmodel(self):
        dataframe = self.dataAggregation()
        dataframe.dropna(subset=[
            'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
            'DC_POWER', 'AC_POWER'
        ], inplace=True)

        dataframe['HOUR'] = dataframe['DATE_TIME'].dt.hour
        dataframe['MINUTE'] = dataframe['DATE_TIME'].dt.minute
        dataframe['DAY_OF_WEEK'] = dataframe['DATE_TIME'].dt.dayofweek
        dataframe['IS_DAYLIGHT'] = dataframe['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

        feature_cols = [
            'AMBIENT_TEMPERATURE',
            'MODULE_TEMPERATURE',
            'IRRADIATION',
            'HOUR',
            'MINUTE',
            'DAY_OF_WEEK',
            'IS_DAYLIGHT',
            'DC_POWER'
        ]
        X = dataframe[feature_cols]
        y = dataframe['AC_POWER']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        regressionModel: xgb.XGBRegressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        regressionModel.fit(X_train, y_train)

        y_pred_train = regressionModel.predict(X_train)
        y_pred_test = regressionModel.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"✅ Model Training Complete")
        print(f"🔎 Train R² Score: {r2_train:.4f}, RMSE: {rmse_train:.2f} watts")
        print(f"🔎 Test  R² Score: {r2_test:.4f}, RMSE: {rmse_test:.2f} watts")

        model_output_path = os.path.join(self.DIR, "models", "simpleACmodel.pkl")
        joblib.dump(regressionModel, model_output_path)
        print(f"💾 Model saved to: {model_output_path}")

        return regressionModel
    
    @ExceptionHandelling
    def simpleDCmodel(self):
        dataframe = self.dataAggregation()
        dataframe.dropna(subset=[
            'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
            'DC_POWER'
        ], inplace=True)
        dataframe['HOUR'] = dataframe['DATE_TIME'].dt.hour
        dataframe['MINUTE'] = dataframe['DATE_TIME'].dt.minute
        dataframe['DAY_OF_WEEK'] = dataframe['DATE_TIME'].dt.dayofweek
        dataframe['IS_DAYLIGHT'] = dataframe['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

        feature_cols = [
            'AMBIENT_TEMPERATURE',
            'MODULE_TEMPERATURE',
            'IRRADIATION',
            'HOUR',
            'MINUTE',
            'DAY_OF_WEEK',
            'IS_DAYLIGHT'
        ]
        X = dataframe[feature_cols]
        y = dataframe['DC_POWER']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        regressionModel: xgb.XGBRegressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        regressionModel.fit(X_train, y_train)

        y_pred_train = regressionModel.predict(X_train)
        y_pred_test = regressionModel.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"✅ DC Model Training Complete")
        print(f"🔎 Train R² Score: {r2_train:.4f}, RMSE: {rmse_train:.2f} watts")
        print(f"🔎 Test  R² Score: {r2_test:.4f}, RMSE: {rmse_test:.2f} watts")

        model_output_path = os.path.join(self.DIR, "models", "simpleDCmodel.pkl")
        joblib.dump(regressionModel, model_output_path)
        print(f"💾 DC Model saved to: {model_output_path}")

        return regressionModel
    
    def trainACmodel(self) -> RandomForestRegressor:
        dataframe = self.dataAggregation()
        dataframe['DATE_TIME'] = pd.to_datetime(dataframe['DATE_TIME'])
        dataframe['HOUR'] = dataframe['DATE_TIME'].dt.hour
        dataframe['MINUTE'] = dataframe['DATE_TIME'].dt.minute  # ✅ Include MINUTE
        dataframe['DAY_OF_WEEK'] = dataframe['DATE_TIME'].dt.dayofweek
        dataframe = pd.get_dummies(dataframe, columns=["INVERTER_IDs"])
        dataframe.dropna(inplace=True)
        dataframe.sort_values("DATE_TIME", inplace=True)

        features = ['DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                    'HOUR', 'MINUTE', 'DAY_OF_WEEK'] + \
                [col for col in dataframe.columns if col.startswith('INVERTER_IDs_')]

        X = dataframe[features]
        y = dataframe['AC_POWER']

        split_idx = int(0.8 * len(dataframe))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(self.DIR, "models", "ac_model.pkl"))

        y_pred = model.predict(X_test)
        print("AC Model Trained")
        print("R² Score:", r2_score(y_test, y_pred))
        print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
        X_test = X_test.copy()
        X_test["PREDICTED_AC"] = y_pred
        X_test["ACTUAL_AC"] = y_test.values
        X_test["TIMESTAMP"] = dataframe.iloc[split_idx:]["DATE_TIME"].values
        X_test.to_csv(os.path.join(self.DIR, "DATASETS", "timebasedACmodel.csv"), index=False)
        joblib.dump(model, os.path.join(self.DIR, "models", "timebasedACmodel.pkl"))
        return model
        
    @ExceptionHandelling
    def dataGeneration(self, days: int = 5) -> pd.DataFrame:
        self.DCmodel = joblib.load(os.path.join(self.DIR, "models", "simpleDCmodel.pkl"))
        self.ACmodel = joblib.load(os.path.join(self.DIR, "models", "simpleACmodel.pkl"))
        city = "HYDERABAD, INDIA"
        location = self.geolocator.geocode(city)
        import asyncio
        if hasattr(location, '__await__'):
            location = asyncio.get_event_loop().run_until_complete(location)
        if location is None:
            raise ValueError(f"Could not geocode city: {city}")
        weatherData = self.getWeatherData(location.latitude, location.longitude, days)

        weatherData.rename(columns={
            'Ambient Temp (°C)': 'AMBIENT_TEMPERATURE',
            'Irradiance (W/m²)': 'IRRADIATION',
            'Timestamp': 'DATE_TIME'
        }, inplace=True)

        weatherData['IRRADIATION'] = (weatherData['IRRADIATION'] * (15 / 60)) / 1000  # Convert to kWh/m²
        weatherData['DATE_TIME'] = pd.to_datetime(weatherData['DATE_TIME'])
        weatherData['HOUR'] = weatherData['DATE_TIME'].dt.hour
        weatherData['MINUTE'] = weatherData['DATE_TIME'].dt.minute
        weatherData['DAY_OF_WEEK'] = weatherData['DATE_TIME'].dt.dayofweek
        weatherData['IS_DAYLIGHT'] = weatherData['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

        weatherData['MODULE_TEMPERATURE'] = [
            self.predictModuleTemperature(row['AMBIENT_TEMPERATURE'], row['IRRADIATION'])
            for _, row in weatherData.iterrows()
        ]
        weatherData['DC_POWER'] = [
            self.DCmodel.predict(pd.DataFrame([{
                'AMBIENT_TEMPERATURE': row['AMBIENT_TEMPERATURE'],
                'MODULE_TEMPERATURE': row['MODULE_TEMPERATURE'],
                'IRRADIATION': row['IRRADIATION'],
                'HOUR': row['HOUR'],
                'MINUTE': row['MINUTE'],
                'DAY_OF_WEEK': row['DAY_OF_WEEK'],
                'IS_DAYLIGHT': row['IS_DAYLIGHT']
            }]))[0]
            for _, row in weatherData.iterrows()
        ]
        weatherData['AC_POWER'] = [
            self.ACmodel.predict(pd.DataFrame([{
                'AMBIENT_TEMPERATURE': row['AMBIENT_TEMPERATURE'],
                'MODULE_TEMPERATURE': row['MODULE_TEMPERATURE'],
                'IRRADIATION': row['IRRADIATION'],
                'HOUR': row['HOUR'],
                'MINUTE': row['MINUTE'],
                'DAY_OF_WEEK': row['DAY_OF_WEEK'],
                'IS_DAYLIGHT': row['IS_DAYLIGHT'],
                'DC_POWER': row['DC_POWER']
            }]))[0]
            for _, row in weatherData.iterrows()
        ]
        weatherData.to_csv(os.path.join(self.DIR, "DATASETS", "Generated_Weather_Data.csv"), index=False)
        logging.info("Generated weather data saved to: Generated_Weather_Data.csv")
        return weatherData
    
    @ExceptionHandelling
    def anomalyDetection(self, days: int = 5) -> pd.DataFrame:
        self.DCmodel = joblib.load(os.path.join(self.DIR, "models", "simpleDCmodel.pkl"))
        self.ACmodel = joblib.load(os.path.join(self.DIR, "models", "timebasedACmodel.pkl"))
        city = "HYDERABAD, INDIA"
        location = self.geolocator.geocode(city)
        weatherDataFrame = self.getWeatherData(location.latitude, location.longitude, days)
        weatherDataFrame.rename(columns={
            'Ambient Temp (°C)': 'AMBIENT_TEMPERATURE',
            'Irradiance (W/m²)': 'IRRADIATION',
            'Timestamp': 'DATE_TIME'
        }, inplace=True)

        # Preprocess weather features
        weatherDataFrame['IRRADIATION'] = (weatherDataFrame['IRRADIATION'] * (15 / 60)) / 1000
        weatherDataFrame['DATE_TIME'] = pd.to_datetime(weatherDataFrame['DATE_TIME'])
        weatherDataFrame['HOUR'] = weatherDataFrame['DATE_TIME'].dt.hour
        weatherDataFrame['MINUTE'] = weatherDataFrame['DATE_TIME'].dt.minute
        weatherDataFrame['DAY_OF_WEEK'] = weatherDataFrame['DATE_TIME'].dt.dayofweek
        weatherDataFrame['IS_DAYLIGHT'] = weatherDataFrame['HOUR'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

        weatherDataFrame['MODULE_TEMPERATURE'] = [
            self.predictModuleTemperature(row['AMBIENT_TEMPERATURE'], row['IRRADIATION'])
            for _, row in weatherDataFrame.iterrows()
        ]
        weatherDataFrame['DC_POWER'] = [
            self.DCmodel.predict(pd.DataFrame([{
                'AMBIENT_TEMPERATURE': row['AMBIENT_TEMPERATURE'],
                'MODULE_TEMPERATURE': row['MODULE_TEMPERATURE'],
                'IRRADIATION': row['IRRADIATION'],
                'HOUR': row['HOUR'],
                'MINUTE': row['MINUTE'],
                'DAY_OF_WEEK': row['DAY_OF_WEEK'],
                'IS_DAYLIGHT': row['IS_DAYLIGHT']
            }]))[0]
            for _, row in weatherDataFrame.iterrows()
        ]
        mergedDataFrame = self.dataAggregation()
        inverterIDs = mergedDataFrame['INVERTER_IDs'].unique()
        inverterFeatureColumns = [f"INVERTER_IDs_{inv}" for inv in inverterIDs]

        expected_features = list(self.ACmodel.feature_names_in_)

        acPredictionsList = []
        for inverterID in inverterIDs:
            # Start from weather + DC predictions
            inverterDataFrame = weatherDataFrame.copy()

            # One-hot encode this inverter
            for column in inverterFeatureColumns:
                inverterDataFrame[column] = 1 if column == f"INVERTER_IDs_{inverterID}" else 0

            # Ensure all expected_features are present
            missing = [f for f in expected_features if f not in inverterDataFrame.columns]
            if missing:
                raise ValueError(f"Missing expected features for AC model: {missing}")

            # Prepare data for prediction in correct order
            predInput = inverterDataFrame[expected_features]

            # Predict AC power
            predicted_ac = self.ACmodel.predict(predInput)

            # Keep fullDataFrame for reporting, assign predictions
            fullDataFrame = inverterDataFrame.copy()
            fullDataFrame['PREDICTED_AC'] = predicted_ac
            fullDataFrame['INVERTER_ID'] = inverterID

            acPredictionsList.append(fullDataFrame)

        # Combine all inverter predictions
        ACDataFrame = pd.concat(acPredictionsList, ignore_index=True)

        # Save predicted AC data
        predictedPath = os.path.join(self.DIR, "DATASETS", "AC_Predicted_Data.csv")
        ACDataFrame.to_csv(predictedPath, index=False)
        logging.info(f"AC predicted data saved to: {predictedPath}")

        # Anomaly detection via Z-score per inverter
        ACDataFrame['Z_SCORE'] = ACDataFrame.groupby("INVERTER_ID")["PREDICTED_AC"].transform(lambda x: zscore(x, ddof=0))
        ACDataFrame['ANOMALY'] = ACDataFrame['Z_SCORE'].abs() > 2.5  # threshold can be tuned

        # Save anomaly-marked data
        anomalyPath = os.path.join(self.DIR, "DATASETS", "AC_Predicted_Data_With_Anomalies.csv")
        ACDataFrame.to_csv(anomalyPath, index=False)
        logging.info(f"Anomaly-marked data saved to: {anomalyPath}")

        sampleInverter = inverterIDs[0]
        sampleSubset = ACDataFrame[ACDataFrame['INVERTER_ID'] == sampleInverter]
        print(f"Sample Predictions for Inverter: {sampleInverter}")
        print(sampleSubset[['DATE_TIME', 'IRRADIATION', 'AMBIENT_TEMPERATURE',
                            'MODULE_TEMPERATURE', 'DC_POWER', 'PREDICTED_AC']].head(10))

        print("Anomaly Summary:")
        for inv in inverterIDs:
            subset = ACDataFrame[(ACDataFrame['INVERTER_ID'] == inv) & (ACDataFrame['ANOMALY'])]
            if not subset.empty:
                print(f"Inverter {inv} anomalies:")
                print(subset[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'IRRADIATION', 'HOUR', 'MINUTE',
       'DAY_OF_WEEK', 'IS_DAYLIGHT', 'MODULE_TEMPERATURE', 'DC_POWER', 'PREDICTED_AC', 'INVERTER_ID']].head())
            else:
                print(f"Inverter {inv}: No anomalies detected.")

        return ACDataFrame

if __name__ == "__main__":
    energyprediction = EnergyPrediction()
    data = energyprediction.dataGeneration()
    print(data.head(20))
