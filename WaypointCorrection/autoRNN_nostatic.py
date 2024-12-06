import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1" # ARM chips on mac don't yet support certain features. make sure this is the VERY FIRST command to run.

import os
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.utils import AirPassengersPanel

# Set environment variable for PyTorch on Mac with ARM chips
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # ARM chips on Mac don't yet support certain features

# Prepare the training and test data
AirPassengersPanelFiltered = AirPassengersPanel[AirPassengersPanel['unique_id'] == 'Airline1']
# AirPassengersPanelFiltered.to_parquet(filtered_parquet_file, engine='pyarrow', index=False)


Y_train_df = AirPassengersPanelFiltered[AirPassengersPanelFiltered.ds < AirPassengersPanelFiltered['ds'].values[-12]]  # 132 train
Y_test_df = AirPassengersPanelFiltered[AirPassengersPanelFiltered.ds >= AirPassengersPanelFiltered['ds'].values[-12]].reset_index(drop=True)  # 12 test

# Initialize and fit the model (no static_df, no stat_exog_list)
fcst = NeuralForecast(
    models=[RNN(
        h=12,
        input_size=-1,
        inference_input_size=24,
        loss=MQLoss(level=[80, 90]),
        scaler_type='robust',
        encoder_n_layers=2,
        encoder_hidden_size=128,
        context_size=10,
        decoder_hidden_size=128,
        decoder_layers=2,
        max_steps=10,
        futr_exog_list=['y_[lag12]'],  # Using only future exogenous (lag12)
        hist_exog_list=['y_[lag12]'],  # Using only historical exogenous (lag12)
    )],
    freq='M'
)

# Fit the model without static_df
fcst.fit(df=Y_train_df, val_size=12)

# Predict using the trained model
forecasts = fcst.predict(futr_df=Y_test_df)

# Process the forecasts and the true data for plotting
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id', 'ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

# Filter the data to only include 'Airline1' (no need for other unique_id values)
plot_df = plot_df[plot_df.unique_id == 'Airline1'].drop('unique_id', axis=1)

# Plot the results
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')  # Actual values
plt.plot(plot_df['ds'], plot_df['RNN-median'], c='blue', label='Median')  # Forecasted median
plt.fill_between(
    x=plot_df['ds'][-12:],
    y1=plot_df['RNN-lo-90'][-12:].values,
    y2=plot_df['RNN-hi-90'][-12:].values,
    alpha=0.4, label='Level 90'
)
plt.legend()
plt.grid()
plt.show()
