import pandas as pd
from statsforecast import StatsForecast
from ray import tune

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.utils import AirPassengersDF as Y_df
csv_file = 'smoothed_time_series_2s.csv'  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Step 2: Save the DataFrame as a Parquet file
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S.%f')  # Adjust format if necessary
parquet_file = 'smoothed_time_series_short.parquet'  # Replace with your desired Parquet file name
df.to_parquet(parquet_file, engine='pyarrow')

Y_df = pd.read_parquet(parquet_file)


Y_df.head()

uids = Y_df['unique_id'].unique()[:10] # Select 10 ids to make the example faster
Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)


initial_data = StatsForecast.plot(Y_df, engine='matplotlib')
initial_data.show()

# config_nhits = {
#     "input_size": tune.choice([100, 150, 200]),              # Length of input window
#     "start_padding_enabled": True,
#     # "n_blocks": 5*[1],
#     "n_blocks": tune.choice([[1], [1, 1, 1], [2, 2, 2]]),
#     # Length of input window
#     # "mlp_units": 5 * [[64, 64]],                                  # Length of input window
#     "mlp_units": tune.choice([[64, 32], [32, 16], [16, 8]]),
#     # "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],
#     #                                   [8, 4, 2, 1, 1]]),            # MaxPooling Kernel size
#
#     "n_pool_kernel_size": tune.choice([[2, 2, 2, 1, 1], [4, 4, 2, 1, 1]]),
#     # "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
#     #                                   [1, 1, 1, 1, 1]]),            # Interpolation expressivity ratios
#
#     "n_freq_downsample": tune.choice([[4, 2, 1, 1, 1], [2, 2, 2, 1, 1]]),
#     "learning_rate": tune.loguniform(1e-4, 1e-3),                   # Initial Learning rate
#     "scaler_type": tune.choice([None]),                             # Scaler type
#     # "max_steps": tune.choice([1000]),                               # Max number of training iterations
#     "max_steps": tune.choice([500, 800]),
#     # "batch_size": tune.choice([1, 4, 10]),                          # Number of series in batch
#     "batch_size": tune.choice([16, 32, 64]),
#     "windows_batch_size": tune.choice([128, 256]),             # Number of windows in batch
#     # "random_seed": tune.randint(1, 20),                             # Random seed
#     "random_seed": tune.choice([42])
# }

config_nhits = {
    "input_size": tune.choice([100,150,200]),              # Length of input window
    "start_padding_enabled": True,
    "n_blocks": tune.choice([[1], [1, 1, 1], [2, 2, 2]]),                                      # Length of input window
    "mlp_units": 5 * [[64, 64]],                                   # Length of input window
    # "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],
    #                                   [8, 4, 2, 1, 1]]),            # MaxPooling Kernel size
    "n_pool_kernel_size": tune.choice([[2, 2, 2, 1, 1], [4, 4, 2, 1, 1]]),
    # "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
    #                                   [1, 1, 1, 1, 1]]),            # Interpolation expressivity ratios
    "n_freq_downsample": tune.choice([[4, 2, 1, 1, 1], [2, 2, 2, 1, 1]]),
    # "learning_rate": tune.loguniform(1e-4, 1e-2),                   # Initial Learning rate
    # "learning_rate": tune.loguniform(1e-4, 1e-3), # works worse than 1e-2 learning rate.
    # "learning_rate": tune.loguniform(1e-4, 5e-2),
    "learning_rate": tune.loguniform(1e-3, 5e-2),
    "scaler_type": tune.choice([None]),                             # Scaler type
    # "max_steps": tune.choice([1000]),                               # Max number of training iterations
    "max_steps": tune.choice([500, 800]),
    "batch_size": tune.choice([1, 4, 10]),                          # Number of series in batch
    "windows_batch_size": tune.choice([128, 256, 512]),             # Number of windows in batch
    "random_seed": tune.randint(1, 20),                             # Random seed
}

config_lstm = {
    "input_size": tune.choice([48, 48*2, 48*3]),              # Length of input window
    "encoder_hidden_size": tune.choice([64, 128]),            # Hidden size of LSTM cells
    "encoder_n_layers": tune.choice([2,4]),                   # Number of layers in LSTM
    "learning_rate": tune.loguniform(1e-4, 1e-3),             # Initial Learning rate
    "scaler_type": tune.choice(['robust']),                   # Scaler type
    "max_steps": tune.choice([250, 500]),                    # Max number of training iterations
    "batch_size": tune.choice([1, 4]),                        # Number of series in batch
    "random_seed": tune.randint(1, 20),                       # Random seed
}

nf = NeuralForecast(
    models=[
        AutoNHITS(h=1000, config=config_nhits, loss=MQLoss(), num_samples=5),
        # AutoLSTM(h=300, config=config_lstm, loss=MQLoss(), num_samples=5),
    ],
    freq='ms'
)
nf.fit(df=Y_df)

fcst_df = nf.predict()
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
