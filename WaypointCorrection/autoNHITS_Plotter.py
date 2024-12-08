
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
import pandas as pd

# csv_file = 'smoothed_time_series_2s.csv'  # Replace with your CSV file name
# df = pd.read_csv(csv_file)
#
# # Step 2: Save the DataFrame as a Parquet file
# df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S.%f')  # Adjust format if necessary
# parquet_file = 'smoothed_time_series_short.parquet'  # Replace with your desired Parquet file name
# df.to_parquet(parquet_file, engine='pyarrow')
#
# Y_df = pd.read_parquet(parquet_file)


parquet_file = 'NHITS_predicted_timeseries_short.parquet'  # Replace with your desired Parquet file name
fcst_df_pq = pd.read_parquet(parquet_file)
# forecast_plt = StatsForecast.plot(Y_df,fcst_df_pq, engine='matplotlib',level=[80, 90])
# forecast_plt.show()


test_csv_file = 'smoothed_time_series_3s.csv'  # Replace with your CSV file name
test_df = pd.read_csv(test_csv_file)
# Step 2: Save the DataFrame as a Parquet file
test_df['ds'] = pd.to_datetime(test_df['ds'], format='%Y-%m-%d %H:%M:%S.%f')  # Adjust format if necessary
parquet_file = 'smoothed_time_series_short.parquet'  # Replace with your desired Parquet file name
test_df.to_parquet(parquet_file, engine='pyarrow')

test_Y_df = pd.read_parquet(parquet_file)
forecast_plt = StatsForecast.plot(test_Y_df, fcst_df_pq, engine='matplotlib',level=[80,90])
forecast_plt.show()