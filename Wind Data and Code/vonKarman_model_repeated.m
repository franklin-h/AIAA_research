% Parameters
fs = 1000;    % Sampling frequency (Hz)
T = 1;       % Total duration (seconds)
N = T * fs;   % Number of samples
dt = 1 / fs;  % Time step
L = 10;       % Integral scale (m)
sigma = 3;    % Turbulence intensity (standard deviation)
U = 10;       % Mean wind speed (m/s)
alpha = 1;    % PSD scaling factor

% Initialize storage for all time series
all_data = table();
t = (0:N-1) * dt;
% Generate 10 time series
for i = 1:10
    % Frequency vector (positive frequencies)
    f = (0:N/2) * (fs / N);

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
    unique_id = repmat("H" + string(i), N, 1);

    % Create "ds" column (integers starting from 1)
    ds = (1:N)';

    % Create "y" column (time series values)
    y = smoothed_time_series(:);

    % Combine into a table
    series_table = table(unique_id, ds, y, 'VariableNames', {'unique_id', 'ds', 'y'});

    plot(t,y)
    % Append to all_data table
    all_data = [all_data; series_table];
end

% Write to a CSV file
csv_filename = '../WaypointCorrection/smoothed_time_series_short.csv'; % Desired filename
writetable(all_data, csv_filename);

disp(['10 smoothed time series have been written to the CSV file: ', csv_filename]);
