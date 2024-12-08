% smoothed_time_series_10s_1_rnn_ms matches with deep_copy_forecasts

% Load the data
% historic_data = readtable('smoothed_time_series_10s_1_rnn_ms.csv');
% forecast_data = readtable('deep_copy_forecasts.csv');

historic_data1 = readtable('FinalPredictions/smoothed_time_series_12s_part1_uDirec.csv'); 
historic_data2 = readtable('FinalPredictions/smoothed_time_series_12s_part2_uDirec.csv');
forecast_data1 = readtable('FinalPredictions/deep_copy_forecasts_uDirec.csv');
forecast_data2 = readtable('FinalPredictions/deep_copy_forecasts2_uDirec.csv');

forecast_data = vertcat(forecast_data1,forecast_data2); 


% Extract relevant columns
time_smoothed1 = historic_data1.ds;
y_smoothed1 = historic_data1.y;

time_smoothed2 = historic_data2.ds;
y_smoothed2 = historic_data2.y;


time_forecast_time = forecast_data.ds;
rnn_values = forecast_data.RNN_median;
rnn_lo_90 = forecast_data.RNN_lo_90;
rnn_hi_90 = forecast_data.RNN_hi_90;



% Create the plots
fig = figure;
hold on;

% % Plot smoothed time series
plot(time_smoothed1, y_smoothed1, 'k', 'DisplayName', 'Smoothed Time Series',LineWidth=2);
plot(time_smoothed2, y_smoothed2, 'k', 'DisplayName', 'Smoothed Time Series',LineWidth=2);


% Plot 90% confidence interval as shaded region
smoothing_window = 20; 
rnn_lo_90_smoothed = smoothdata(rnn_lo_90,"movmedian",smoothing_window); 
rnn_hi_90_smoothed = smoothdata(rnn_hi_90,"movmedian",smoothing_window); 

plot(time_forecast_time, rnn_lo_90_smoothed, 'r--', 'DisplayName', '90% CI Lower Bound');
plot(time_forecast_time, rnn_hi_90_smoothed, 'r--', 'DisplayName', '90% CI Upper Bound');
fill([time_forecast_time',fliplr(time_forecast_time')], [rnn_lo_90_smoothed',fliplr(rnn_hi_90_smoothed')], 'b', 'EdgeColor','none', 'FaceAlpha',0.25)
% fliplr can only accept row vector, not column!! 

% % Plot the RNN predicted time series
plot(time_forecast_time, smoothdata(rnn_values,"movmedian",smoothing_window), 'b', 'DisplayName', 'RNN Prediction',LineWidth=2);
xlim([6000, inf])

ax = gca; 
ax.FontSize = 15; 
ax.TickLabelInterpreter = 'latex';

xlabel("Time (ms)",Interpreter='latex') 
ylabel("Wind Turbulence Deviation from Mean (m/s)",Interpreter='latex')
last_4000_elements = historic_data2(end-3999:end,:);
rms_error = sqrt(mean((last_4000_elements(:,3) - rnn_values).^2)); 

legend("True Speed","Median Forecasted Speed","90% CI Lower and Upper Bound")
[~, z] = zoomPlot((1:800).', T.LoadFactor, [615 740], [0.2 0.5 0.4 0.4]);

%%%%%%%%% 

