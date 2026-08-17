% 05_reconstruction_comparison.m
%
% This is the core comparison of the thesis: a source emits one sample
% per second into a queue (Bernoulli arrivals at rate = arrival_rate,
% Bernoulli service at rate = service_rate). At the receiver we compare
% three ways of reconstructing the original 480-second signal from
% whatever arrived:
%   - Online:          hold the last successfully received sample (zero-
%                       order hold in real time -- what a live monitor sees)
%   - Offline:          same hold-last-value rule but interpolated after
%                       the fact over the *whole* record (a genie that
%                       fills any still-empty samples using every packet
%                       received during the run)
%   - Partial offline:  like offline, but interpolation only uses packets
%                       received within the same ~10 s block, capturing
%                       what's feasible with a small look-back buffer
%                       instead of the full record
% MSE against the true signal is computed for each method across a sweep
% of arrival rates, at a fixed service rate.
%
% Cleaned up from: matlab codes/ppg_signal_analysis.m
% Fixes: relative data path; original script mixed a signals-file
% (125 Hz) time base with a hardcoded "480" end time meant for the
% numerics file (1 Hz) -- this version explicitly uses the 1 Hz
% numerics-style series, which is what the 480 assumption actually
% describes. Variable names clarified; otherwise the logic (queue,
% last-value-hold, block-wise offline fill) matches the original.

data = csvread('../data/sample_Numerics.csv', 1, 0);
time_column = data(:, 1);       % 1..480 seconds
amplitude_column = data(:, 2);  % HR-like trend, one sample/sec

simulation_time = numel(time_column); % 480
service_rate = 0.9;
block_size = 10; % seconds, for "partial offline" look-back window

arrival_rates = 0.1:0.1:0.9;
mse_online = zeros(size(arrival_rates));
mse_offline = zeros(size(arrival_rates));
mse_partial_offline = zeros(size(arrival_rates));

for r = 1:numel(arrival_rates)
    arrival_rate = arrival_rates(r);

    queue = [];
    online_signal = nan(1, simulation_time);
    received_pairs = []; % [value, time] for every packet the receiver ever got
    last_value = median(amplitude_column);

    for i = 1:simulation_time
        if rand() < arrival_rate
            queue(end+1, :) = [amplitude_column(i), i];
        end
        if rand() < service_rate && ~isempty(queue)
            last_value = queue(1,1);
            received_pairs = [received_pairs; queue(1,:)];
            queue(1,:) = [];
        end
        online_signal(i) = last_value; % zero-order hold, in real time
    end
    mse_online(r) = sum((amplitude_column(:) - online_signal(:)).^2) / simulation_time;

    % ---- offline: fill every sample using the full set of received packets ----
    offline_signal = fill_with_last_received(received_pairs, amplitude_column, simulation_time);
    mse_offline(r) = sum((amplitude_column(:) - offline_signal(:)).^2) / simulation_time;

    % ---- partial offline: only look back within block_size-second blocks ----
    partial_pairs = [];
    for b = 1:block_size:simulation_time
        block_end = min(b+block_size-1, simulation_time);
        in_block = received_pairs(received_pairs(:,2) >= b & received_pairs(:,2) <= block_end, :);
        partial_pairs = [partial_pairs; in_block];
    end
    partial_signal = fill_with_last_received(partial_pairs, amplitude_column, simulation_time);
    mse_partial_offline(r) = sum((amplitude_column(:) - partial_signal(:)).^2) / simulation_time;

    printf('arrival_rate = %.1f  online MSE = %.4f  offline MSE = %.4f  partial-offline MSE = %.4f\n', ...
        arrival_rate, mse_online(r), mse_offline(r), mse_partial_offline(r));
end

figure('visible','off');
plot(arrival_rates, mse_online, '-o', arrival_rates, mse_offline, '-s', arrival_rates, mse_partial_offline, '-^');
xlabel('Arrival rate \lambda'); ylabel('Mean Squared Error');
legend('Online reconstruction', 'Offline reconstruction', 'Partial-offline reconstruction', 'Location', 'northeast');
title(sprintf('Reconstruction MSE vs arrival rate (service rate = %.1f)', service_rate));
grid on;
print('../results/05_reconstruction_comparison.png', '-dpng', '-r120');
disp('Saved results/05_reconstruction_comparison.png');
