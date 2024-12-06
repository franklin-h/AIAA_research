import pandas as pd
from statsforecast import StatsForecast
from ray import tune

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss

import matplotlib.pyplot as plt
csv_file = 'smoothed_time_series.csv'  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Parquet file
parquet_file = 'smoothed_time_series.parquet'  # Replace with your desired Parquet file name
df.to_parquet(parquet_file, engine='pyarrow')

Y_df = pd.read_parquet('smoothed_time_series.parquet')

Y_df.head()

uids = Y_df['unique_id'].unique()[:10]
Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)

# plotTurb = StatsForecast.plot(Y_df, engine='matplotlib')
# plotTurb.show()

config_nhits = {
    "input_size": tune.choice([100, 100*2, 100*3]),              # Length of input window. 375 ms characteristic turbulence
    "start_padding_enabled": True,
    "n_blocks": 5*[1],                                              # Length of input window
    "mlp_units": 5 * [[64, 64]],      # changed from 128 to 64                             # Length of input window
    "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],
                                      [8, 4, 2, 1, 1]]),            # MaxPooling Kernel size
    "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
                                      [1, 1, 1, 1, 1]]),            # Interpolation expressivity ratios
    "learning_rate": tune.loguniform(1e-4, 5e-3),                   # Initial Learning rate
    "scaler_type": tune.choice([None]),                             # Scaler type
    "max_steps": tune.choice([700]),                               # Max number of training iterations
    "batch_size": tune.choice([1, 4, 10]),                          # Number of series in batch
    "windows_batch_size": tune.choice([128, 256]),             # Number of windows in batch
    "random_seed": tune.randint(1, 20),                             # Random seed
}

config_lstm = {
    "input_size": tune.choice([48, 48*2, 48*3]),              # Length of input window
    "encoder_hidden_size": tune.choice([64, 128]),            # Hidden size of LSTM cells
    "encoder_n_layers": tune.choice([2,4]),                   # Number of layers in LSTM
    "learning_rate": tune.loguniform(1e-4, 1e-2),             # Initial Learning rate
    "scaler_type": tune.choice(['robust']),                   # Scaler type
    "max_steps": tune.choice([500, 1000]),                    # Max number of training iterations
    "batch_size": tune.choice([1, 4]),                        # Number of series in batch
    "random_seed": tune.randint(1, 20),                       # Random seed
}

nf = NeuralForecast(
    models=[
        AutoNHITS(h=3000, config=config_nhits, loss=MQLoss(), num_samples=5),
        # AutoLSTM(h=48, config=config_lstm, loss=MQLoss(), num_samples=2),
    ],
    freq=1
)

nf.fit(df=Y_df)

fcst_df = nf.predict()
fcst_df.columns = fcst_df.columns.str.replace('-median', '')
fcst_df.head()

# for some reason there are only H1 to H9, so we need to drop the 10th from Y_df
uids = Y_df['unique_id'].unique()[:9]
Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)

# Path for the output CSV file
csv_file = "predicted_timeseries_smallerBlocks.csv"

# Load the Parquet file into a Pandas DataFrame
# df = pd.read_parquet(fcst_df)

# Export the DataFrame to a CSV file
fcst_df.to_csv(csv_file, index=True)

# predictedCsv = 'smoothed_time_series.csv'  # Replace with your CSV file name
pf = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Parquet file
parquet_file = 'predicted_timeseries.parquet'  # Replace with your desired Parquet file name
pf.to_parquet(parquet_file, engine='pyarrow')
# StatsForecast.plot(Y_df, fcst_df, engine='matplotlib', max_insample_length=3000 * 3, level=[80, 90])
#
# # print(uids)
# plotTurbPrediction = StatsForecast.plot(fcst_df, engine='matplotlib')
# plotTurbPrediction.show()

# forecast_parquet_file = parquet_fi  # Replace with your desired Parquet file name
# df.to_parquet(parquet_file, engine='pyarrow')
fcst_df_pq = pd.read_parquet(parquet_file)
forecast_plt = StatsForecast.plot(Y_df,fcst_df_pq, engine='matplotlib')
forecast_plt.show()

# Select rows 1 to 2000 (Pandas uses 0-based indexing, so we slice as 0:2000)
df = fcst_df_pq
subset_df = df.iloc[0:100]

# Extract the 2nd and 3rd columns (again, 0-based indexing)
x = subset_df.iloc[:, 1]  # 2nd column
y = subset_df.iloc[:, 2]  # 3rd column

# Plot the data
plt.figure(figsize=(10, 6))  # Optional: Set the figure size
plt.plot(x, y, label="Data", color="blue")  # Create the plot
plt.xlabel("2nd Column (X-axis)")  # Label for X-axis
plt.ylabel("3rd Column (Y-axis)")  # Label for Y-axis
plt.title("Plot of 3rd Column vs. 2nd Column (Rows 1-2000)")  # Title
plt.legend()  # Add a legend
plt.grid(True)  # Optional: Add grid lines
plt.show()  # Display the plot