import pandas as pd
from statsforecast import StatsForecast
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from statsforecast.models import (
    HoltWinters,
    CrostonClassic as Croston,
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)


# Y_df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
# Step 1: Read the CSV file into a DataFrame
csv_file = 'smoothed_time_series.csv'  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Parquet file
parquet_file = 'smoothed_time_series.parquet'  # Replace with your desired Parquet file name
df.to_parquet(parquet_file, engine='pyarrow')  # Use 'fastparquet' or 'pyarrow' as the engine


# Y_df = pd.read_parquet('m4-hourly.parquet')
Y_df = pd.read_parquet('smoothed_time_series.parquet')
# csv_filename = 'm4-hourly.csv'  # Define the filename for the CSV
# Y_df.to_csv(csv_filename, index=False)  # Save the DataFrame as a CSV file without the index

Y_df.head()

uids = Y_df['unique_id'].unique()[:2] # Select 10 ids to make the example faster
Y_df = Y_df.query('unique_id in @uids')
# Y_df = Y_df.groupby('unique_id').tail(7 * 24) #Select last 7 days of data to make example faster

# fig = StatsForecast.plot(Y_df)
# fig.show()

# Create a list of models and instantiation parameters
models = [
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=1000),
    HistoricAverage(),
    DOT(season_length=1000)
]

# Instantiate StatsForecast class as sf
sf = StatsForecast(
    models=models,
    freq=1,
    fallback_model = SeasonalNaive(season_length=1000),
    n_jobs=-1,
)

forecasts_df = sf.forecast(df=Y_df, h=4000, level=[90])
forecasts_df.head()


forecast_plot = sf.plot(Y_df,forecasts_df)
forecast_plot.show()




