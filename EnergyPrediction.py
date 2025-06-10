import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import google.generativeai as genai
import logging
from functools import wraps
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnergyPrediction(object):
    def __init__(self) -> None:
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(self.DIR):
            if filename.endswith('.csv') and filename == "Processed_data.csv":
                self.dataframe: pd.DataFrame = pd.read_csv(os.path.join(self.DIR, filename))
                logging.info(f"Loaded data from {filename}")
        self.RegressionModel: LinearRegression = LinearRegression()
        self.modelpath = os.path.join(self.DIR, "energy_prediction_model.pkl")
    
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
        return prediction[0]

def main():
    predictor = EnergyPrediction()

    ambient_temperatures = predictor.dataframe['AMBIENT_TEMPERATURE'].iloc[:100].values
    irradiations = predictor.dataframe['IRRADIATION'].iloc[:100].values

    if not os.path.exists(predictor.modelpath):
        print("Model not trained. Training now...")
        predictor.modeltraining()

    predictor.RegressionModel = joblib.load(predictor.modelpath)

    predictions = []
    for at, ir in zip(ambient_temperatures, irradiations):
        prediction = predictor.predictModuleTemperature(at, ir)
        predictions.append(float(prediction))

    print("Predictions for first 100 records:")
    print(predictions)

if __name__ == "__main__":
    main()