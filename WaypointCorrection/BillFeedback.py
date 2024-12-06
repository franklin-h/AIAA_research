import pandas as pd
from statsforecast import StatsForecast
from ray import tune

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.tsdataset import TimeSeriesDataset

from neuralforecast.auto import AutoGRU

csv_file = 'smoothed_time_series_20sGru.csv'  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Parquet file
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S.%f')  # Adjust format if necessary
parquet_file = 'smoothed_time_series_long.parquet'  # Replace with your desired Parquet file name
df.to_parquet(parquet_file, engine='pyarrow')

Y_df = pd.read_parquet(parquet_file)
dataset, *_ = TimeSeriesDataset.from_df(Y_df)

# Y_df.head()
#
# uids = Y_df['unique_id'].unique()[:1] # Select 10 ids to make the example faster
# Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)
# Y_df['milliseconds'] = Y_df['ds'].dt.microsecond // 1000

initial_data = StatsForecast.plot(Y_df, engine='matplotlib')
initial_data.show()

# Use your own config or AutoGRU.default_config
# config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8,temporal_cols=["milliseconds"]
#               )
# model = AutoGRU(h=12, config=config, num_samples=1, cpus=1)
#
# # Fit and predict
# model.fit(dataset=Y_df)
# fcst_df = model.predict(dataset=Y_df)
#
# # Optuna
# model = AutoGRU(h=12, config=config, backend='optuna')
# dataset, *_ = TimeSeriesDataset.from_df(Y_df)

# config = dict(
#     max_steps=2,
#     val_check_steps=1,
#     input_size=-1,
#     encoder_hidden_size=8,
#     # temporal_cols=['milliseconds']  # Include milliseconds as a temporal feature
# )

model = AutoGRU(h=12, config=None, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
fcst_df = model.predict(dataset=dataset)
#
# nf.fit(df=Y_df)
#
# fcst_df = nf.predict()
fcst_df.columns = fcst_df.columns.str.replace('-median', '')
fcst_df.head()

csv_file = "predicted_timeseries_short.csv"
fcst_df.to_csv(csv_file, index=True)
fcst_csv = pd.read_csv(csv_file)

parquet_file = 'predicted_timeseries_short.parquet'  # Replace with your desired Parquet file name
fcst_csv.to_parquet(parquet_file, engine='pyarrow')
fcst_df_pq = pd.read_parquet(parquet_file)
forecast_plt = StatsForecast.plot(Y_df,fcst_df_pq, engine='matplotlib',level=[80, 90])
forecast_plt.show()

# StatsForecast.plot(Y_df, fcst_df, engine='matplotlib', max_insample_length=48 * 3, level=[80, 90])
