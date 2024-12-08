% % Parameters
% close all 
% fs = 1000;    % Sampling frequency (Hz)
% T_full = 10;   % Total duration for full data (seconds)
% T_trunc = 9;  % Duration for truncated data (seconds)
% N_full = T_full * fs;   % Number of samples for full data
% N_trunc = T_trunc * fs; % Number of samples for truncated data
% dt = 1 / fs;  % Time step
% L = 10;       % Integral scale (m)
% sigma = 3;    % Turbulence intensity (standard deviation)
% U = 10;       % Mean wind speed (m/s)
% alpha = 1;    % PSD scaling factor
% 
% % Arbitrary starting date and time
% start_time = datetime(2024, 1, 1, 0, 0, 0, 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
% 
% % Initialize storage for all time series
% all_data_full = table();
% all_data_trunc = table();
% 
% % Generate num_series time series
% num_series = 1; 
% for i = 1:num_series
%     % Frequency vector (positive frequencies)
%     f = (0:N_full/2) * (fs / N_full);
% 
%     % Von Karman PSD (normalized for 1D flow)
%     S_vk = alpha * (sigma^2 * L / U) ./ ((1 + (1.339 * f * L / U).^2).^(5/6));
% 
%     % Generate random phase and amplitude for positive frequencies
%     random_phase = 2 * pi * rand(size(S_vk));
%     random_amplitude = sqrt(S_vk) .* (randn(size(S_vk)) + 1i * randn(size(S_vk)));
% 
%     % Construct the full spectrum (enforce Hermitian symmetry)
%     X = [random_amplitude, conj(flip(random_amplitude(2:end-1)))];
% 
%     % Generate time series using inverse FFT
%     time_series = real(ifft(X, 'symmetric')) * sqrt(fs);
% 
%     % Smooth the time series
%     smoothed_time_series = smoothdata(time_series);
% 
%     % Create "unique_id" column
%     unique_id = repmat("H" + string(i), N_full, 1);
% 
%     % Create "ds" column (timestamps with millisecond precision)
%     ds_full = start_time + days(0:N_full-1)';
%     ds_trunc = start_time + days(0:N_trunc-1)'; % Truncated timestamps
% 
%     % Create "y" column (time series values)
%     y_full = smoothed_time_series(:);
%     y_trunc = smoothed_time_series(1:N_trunc);
% 
%     trend_full = 0:N_full-1;  % rnn 
%     trend_trunc = 0:N_trunc-1; % rnn 
%     % Combine into tables
%     series_table_full = table(unique_id, ds_full, y_full, trend_full', 'VariableNames', {'unique_id', 'ds', 'y', 'trend'});
%     series_table_trunc = table(unique_id(1:N_trunc), ds_trunc, y_trunc', trend_trunc', 'VariableNames', {'unique_id', 'ds', 'y', 'trend'});
% 
%     % Append to storage tables
%     all_data_full = [all_data_full; series_table_full]; %#ok<AGROW>
%     all_data_trunc = [all_data_trunc; series_table_trunc]; %#ok<AGROW>
%     plot(y_full)
% end
% % plot(series_table_full)
% % Write to CSV files
% csv_full_filename = '../WaypointCorrection/smoothed_time_series_10s_1_rnn.csv'; % Full data filename
% csv_trunc_filename = '../WaypointCorrection/smoothed_time_series_9s_1_rnn.csv'; % Truncated data filename
% writetable(all_data_full, csv_full_filename);
% writetable(all_data_trunc, csv_trunc_filename);
% 
% disp(['10 smoothed time series (30 seconds) have been written to the CSV file: ', csv_full_filename]);
% disp(['10 smoothed time series (20 seconds) have been written to the CSV file: ', csv_trunc_filename]);

% Parameters
close all 
fs = 1000;    % Sampling frequency (Hz)
T_full = 12;   % Total duration for full data (seconds)
T_trunc = 10;  % Duration for truncated data (seconds)
N_full = T_full * fs;   % Number of samples for full data
N_trunc = T_trunc * fs; % Number of samples for truncated data
dt = 1 / fs;  % Time step
L = 10;       % Integral scale (m)
sigma = 3;    % Turbulence intensity (standard deviation)
U = 10;       % Mean wind speed (m/s)
alpha = 1;    % PSD scaling factor

% Arbitrary starting date and time
% start_time = datetime(2024, 1, 1, 0, 0, 0, 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
start_time = 0; 

% Initialize storage for all time series
all_data_full = table();
all_data_trunc = table();

% Generate num_series time series
num_series = 1; 
for i = 1:num_series
    % Frequency vector (positive frequencies)
    f = (0:N_full/2) * (fs / N_full);

    % Von Karman PSD (normalized for 1D flow)
    S_vk = alpha * (sigma^2 * L / U) ./ ((1 + (1.339 * f * L / U).^2).^(5/6));

    % Generate random phase and amplitude for positive frequencies
    random_phase = 2 * pi * rand(size(S_vk));
    random_amplitude = sqrt(S_vk) .* (randn(size(S_vk)) + 1i * randn(size(S_vk)));

    % Construct the full spectrum (enforce Hermitian symmetry)
    X = [random_amplitude, conj(flip(random_amplitude(2:end-1)))];

    % Generate time series using inverse FFT
    time_series = real(ifft(X, 'symmetric')) * sqrt(fs);

    % Smooth the time series
    smoothed_time_series = smoothdata(time_series);

    % Create "unique_id" column
    unique_id = repmat("H" + string(i), N_full, 1);

    % Create "ds" column (timestamps with millisecond precision)
    ds_full = start_time + 0:N_full-1;
    ds_trunc = start_time + 0:N_trunc-1; % Truncated timestamps

    % Create "y" column (time series values)
    y_full = smoothed_time_series(:);
    y_trunc = smoothed_time_series(1:N_trunc);

    trend_full = 0:N_full-1;  % rnn 
    trend_trunc = 0:N_trunc-1; % rnn 
    % Combine into tables
    series_table_full = table(unique_id, ds_full', y_full, trend_full', 'VariableNames', {'unique_id', 'ds', 'y', 'trend'});
    series_table_trunc = table(unique_id(1:N_trunc), ds_trunc', y_trunc', trend_trunc', 'VariableNames', {'unique_id', 'ds', 'y', 'trend'});

    % Append to storage tables
    all_data_full = [all_data_full; series_table_full]; %#ok<AGROW>
    all_data_trunc = [all_data_trunc; series_table_trunc]; %#ok<AGROW>
    plot(y_full)
end
% plot(series_table_full)
% Write to CSV files
csv_full_filename = '../WaypointCorrection/smoothed_time_series_12s_part1.csv'; % Full data filename
csv_full2_filename = '../WaypointCorrection/smoothed_time_series_12s_part2.csv'; % Full data filename
csv_trunc_filename = '../WaypointCorrection/smoothed_time_series_9s_1_rnn.csv'; % Truncated data filename
writetable(all_data_full(1:10000,:), csv_full_filename);
writetable(all_data_full(2001:12000,:),csv_full2_filename); 
writetable(all_data_trunc, csv_trunc_filename);

disp(['10 smoothed time series (30 seconds) have been written to the CSV file: ', csv_full_filename]);
disp(['10 smoothed time series (20 seconds) have been written to the CSV file: ', csv_trunc_filename]);
