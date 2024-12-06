import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

# Step 1: Read the Parquet file into a DataFrame
# parquet_file = 'your_file.parquet'  # Replace with your Parquet file path
parquet_file = 'smoothed_time_series.parquet'
df = pd.read_parquet(parquet_file)

# Ensure that the Parquet file contains 'ds' (date) and 'y' (values) columns
# If necessary, convert 'ds' column to datetime format
df['ds'] = pd.to_datetime(df['ds'])

# Step 2: Initialize the ARIMA model (p, d, q)
# Define the ARIMA model parameters
model = ARIMA(p=1, d=1, q=1)

# Step 3: Create the StatsForecast object and forecast future values
sf = StatsForecast(df=df, models=[model], freq='ms')  # Adjust 'freq' if needed (e.g., 'D' for daily)
forecast = sf.forecast(horizon=100)  # Forecast the next 10 periods

# Step 4: Display the forecasted results
print(forecast)
