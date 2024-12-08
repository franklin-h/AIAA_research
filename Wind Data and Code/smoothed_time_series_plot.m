% Load the full CSV file
close all 
data = readtable('../WaypointCorrection/smoothed_time_series_30s.csv');
data_prediction = readtable('predicted_timeseries_short 3.csv'); 

% Convert 'ds' column to datetime format
data.ds = datetime(data.ds, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
data_prediction.ds = datetime(data_prediction.ds,'InputFormat','yyyy-MM-dd HH:mm:ss.SSS'); 


% Filter the data for unique_id = H3
selected_series = "H1"; 
h3_data = data(strcmp(data.unique_id, selected_series), :);
data_pred_filtered = data_prediction(strcmp(data_prediction.unique_id, selected_series), :);

% Further filter for timestamps within 0 to 2 seconds
start_time = datetime('2024-01-01 00:00:00.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
end_time = datetime('2024-01-01 00:00:30.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
h3_filtered = h3_data(h3_data.ds >= start_time & h3_data.ds < end_time, :);


% Plot the filtered data
figure;
hold on 
plot(h3_filtered.ds, h3_filtered.y, 'LineWidth', 1.5);
plot(data_pred_filtered.ds,smoothdata(data_pred_filtered.AutoNHITS,"movmean",10),'LineWidth',1.5); 

xlabel('Time (ds)');
ylabel('Value (y)');
% title('Time Series for H3 (0 to 2 seconds)');
grid on;
