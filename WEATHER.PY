import requests
import pandas as pd

def get_solar_weather_data(latitude, longitude):
    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,shortwave_radiation",
        "timezone": "auto",
        "forecast_days": 5
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
        interpolated_df = df.resample('15T').interpolate(method='linear')
        interpolated_df.reset_index(inplace=True)

        return interpolated_df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except KeyError:
        print("Error: Unexpected data format received from API.")
        return None

if __name__ == "__main__":
    TARGET_LATITUDE = 17.3850
    TARGET_LONGITUDE = 78.4867

    weather_data = get_solar_weather_data(TARGET_LATITUDE, TARGET_LONGITUDE)
    if weather_data is not None:
        print(weather_data.to_string(index=False))