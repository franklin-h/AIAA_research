import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1" # ARM chips on mac don't yet support certain features. make sure this is the VERY FIRST command to run.

import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

csv_file = 'smoothed_time_series_3s_1_rnn.csv'  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Step 2: Convert 'ds' column to datetime
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S.%f')  # Adjust format if necessary

# Step 3: Add a lagged column
lag_value = 100  # Define the lag
df['y_[lag100]'] = df['y']  # Initialize the lagged column with the same values

# Apply lag starting from the 101st datapoint
df.loc[lag_value:, 'y_[lag100]'] = df['y'].shift(lag_value)[lag_value:]

# Step 4: Save the DataFrame as a Parquet file
parquet_file = 'smoothed_time_series_short.parquet'  # Replace with your desired Parquet file name
df.to_parquet(parquet_file, engine='pyarrow')

# Step 5: Reload the Parquet file to verify
Y_df = pd.read_parquet(parquet_file)

Y_df.to_csv("y_df.csv")


Y_train_df = Y_df[Y_df.ds<Y_df['ds'].values[-500]] # 2500 train
Y_test_df = Y_df[Y_df.ds>=Y_df['ds'].values[-500]].reset_index(drop=True) # 500 test

timeseriesStatic = pd.read_csv("smoothed_timeseries_static.csv")

fcst = NeuralForecast(
    models=[RNN(h=400,
                input_size=-1,
                inference_input_size=40,
                loss=MQLoss(level=[80, 90]),
                scaler_type='robust',
                encoder_n_layers=2,
                encoder_hidden_size=128,
                context_size=10,
                decoder_hidden_size=128,
                decoder_layers=2,
                max_steps=50,
                futr_exog_list=['y_[lag100]'],
                hist_exog_list=['y_[lag100]'],
                # stat_exog_list=['h1'],
                )
    ],
    freq='D' # Don't use M!!
)
fcst.fit(df=Y_train_df,val_size=400)
# fcst.get_missing_future(futr_df=Y_test_df)
forecasts = fcst.predict(futr_df=Y_test_df)

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='H1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['RNN-median'], c='blue', label='median')
plt.fill_between(x=plot_df['ds'][-12:],
                 y1=plot_df['RNN-lo-90'][-12:].values,
                 y2=plot_df['RNN-hi-90'][-12:].values,
                 alpha=0.4, label='level 90')
plt.legend()
plt.grid()
plt.plot()
plt.show()

