import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1" # ARM chips on mac don't yet support certain features. make sure this is the VERY FIRST command to run.

from statsforecast import StatsForecast

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengersDF as Y_df
from neuralforecast.auto import AutoGRU

import torch

# Split train/test and declare time series dataset
Y_train_df = Y_df[Y_df.ds<='1959-12-31'] # 132 train
Y_test_df = Y_df[Y_df.ds>'1959-12-31']   # 12 test
dataset, *_ = TimeSeriesDataset.from_df(Y_train_df)


initial_data = StatsForecast.plot(Y_df, engine='matplotlib')
initial_data.show()

airplaneData = "exampleAirplaneData.csv"
Y_df.to_csv(airplaneData, index=True)

# Use your own config or AutoGRU.default_config
config = dict(max_steps=2, val_check_steps=1, input_size=-1, encoder_hidden_size=8)
model = AutoGRU(h=120, config=config, num_samples=1, cpus=1)

# Fit and predict
model.fit(dataset=dataset)
fcst_df = model.predict(dataset=dataset)

# Optuna
# model = AutoGRU(h=120, config=None, backend='optuna')


# fcst_df.columns = fcst_df.columns.str.replace('-median', '')
# fcst_df.head()
#
# csv_file = "predicted_timeseries_short.csv"
# fcst_df.to_csv(csv_file, index=True)
# fcst_csv = pd.read_csv(csv_file)
#
# parquet_file = 'predicted_timeseries_short.parquet'  # Replace with your desired Parquet file name
# fcst_csv.to_parquet(parquet_file, engine='pyarrow')
# fcst_df_pq = pd.read_parquet(parquet_file)
# forecast_plt = StatsForecast.plot(Y_df,fcst_df_pq, engine='matplotlib',level=[80, 90])
# forecast_plt.show()

flattened_array = fcst_df.flatten()
plt.plot(flattened_array)
plt.show()  # Display the plot