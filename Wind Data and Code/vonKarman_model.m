% MATLAB script to generate a time series from a noisy Von Karman model

% Parameters
fs = 1000;                  % Sampling frequency (Hz)
T = 30;                    % Total duration (seconds)
N = T * fs;                % Number of samples
dt = 1 / fs;               % Time step
L = 10;                    % Integral scale (m)
sigma = 3;                 % Turbulence intensity (standard deviation)
U = 10;                    % Mean wind speed (m/s)
kappa = 0.4;               % von Karman constant
alpha = 1;                 % PSD scaling factor

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

% Add Gaussian noise to the time series
noise_std = 0.1;           % Standard deviation of noise
noisy_time_series = time_series + noise_std * randn(size(time_series));


smoothed_time_series = smoothdata(time_series); 
% Time vector
t = (0:N-1) * dt;

% Plot the results
figure;
subplot(2, 1, 1);
plot(t, time_series, 'b');
title('Clean Time Series from Von Karman Model');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(t, smoothed_time_series, 'r');
title('Noisy Time Series');
xlabel('Time (s)');
ylabel('Amplitude');

% Step 1: Create the "unique_id" column
unique_id = repmat("H1", length(smoothed_time_series), 1); % Repeat "H1" for each row

% Step 2: Create the "ds" column (integers starting from 1)
ds = (1:length(smoothed_time_series))'; % Column of integers

% Step 3: Create the "y" column (time series values)
y = smoothed_time_series(:); % Ensure column format

% Step 4: Combine into a table
T = table(unique_id, ds, y, 'VariableNames', {'unique_id', 'ds', 'y'});

% Step 5: Write the table to a CSV file
csv_filename = '../WaypointCorrection/smoothed_time_series.csv'; % Desired filename
writetable(T, csv_filename);
