% 01_channel_aoi_simulation.m
%
% Age-of-Information (AoI) queueing simulation for the sensor -> receiver
% link: samples arrive as a Bernoulli process (rate = arrival_rate per
% slot) into a single-server queue that serves at rate = service_rate per
% slot. We sweep the arrival rate and record the mean squared error (MSE)
% between what was sent and what's most recently available at the
% receiver, plus the mean Age of Information, once for a constant service
% rate and once for a linearly increasing (time-varying) service rate.
%
% Cleaned up from: matlab codes/channel1.m and channel2.m
% Fixes: hardcoded 40000-slot / 40-point sweep kept the same, but both
% cases are now run in one script for direct comparison, and results are
% saved instead of only printed.

arrival_rate_interval = round(linspace(0.1, 0.78, 20) * 1000) / 1000; % coarser than original (40 pts) for speed
simulation_time = 10000;                                    % coarser than original (40000) for speed

MSE_const   = zeros(1, numel(arrival_rate_interval));
AoI_const   = zeros(1, numel(arrival_rate_interval));
MSE_varying = zeros(1, numel(arrival_rate_interval));
AoI_varying = zeros(1, numel(arrival_rate_interval));

for index = 1:numel(arrival_rate_interval)
    arrival_rate = arrival_rate_interval(index);

    % ---- constant service rate (channel1.m) ----
    service_rate = 0.8;
    queue = [];
    squared_error = zeros(1, simulation_time);
    aoi = zeros(1, simulation_time);
    for i = 1:simulation_time
        if rand() < arrival_rate
            queue(end+1, :) = [randn()*0.1, i];
        end
        if rand() < service_rate && ~isempty(queue)
            arrival_info = queue(1, :);
            queue(1, :) = [];
            delay = i - arrival_info(2);
            squared_error(i) = delay^2;
            aoi(i) = delay;
        end
    end
    MSE_const(index) = mean(squared_error);
    AoI_const(index) = mean(aoi);

    % ---- time-varying service rate (channel2.m) ----
    service_rate_t = linspace(0.8, 1.2, simulation_time);
    queue = [];
    squared_error = zeros(1, simulation_time);
    aoi = zeros(1, simulation_time);
    for i = 1:simulation_time
        if rand() < arrival_rate
            queue(end+1, :) = [randn()*0.1, i];
        end
        if rand() < service_rate_t(i) && ~isempty(queue)
            arrival_info = queue(1, :);
            queue(1, :) = [];
            delay = i - arrival_info(2);
            squared_error(i) = delay^2;
            aoi(i) = delay;
        end
    end
    MSE_varying(index) = mean(squared_error);
    AoI_varying(index) = mean(aoi);

    printf('arrival_rate = %.3f done\n', arrival_rate);
end

figure('visible','off');
subplot(2,1,1);
plot(arrival_rate_interval, MSE_const, '-o', arrival_rate_interval, MSE_varying, '-s');
xlabel('Arrival rate \lambda'); ylabel('Mean Squared Error');
legend('Constant service rate', 'Time-varying service rate', 'Location', 'northwest');
title('AoI-driven reconstruction MSE vs arrival rate'); grid on;

subplot(2,1,2);
plot(arrival_rate_interval, AoI_const, '-o', arrival_rate_interval, AoI_varying, '-s');
xlabel('Arrival rate \lambda'); ylabel('Mean AoI (slots)');
legend('Constant service rate', 'Time-varying service rate', 'Location', 'northeast');
title('Mean Age of Information vs arrival rate'); grid on;

print('../results/01_channel_aoi_simulation.png', '-dpng', '-r120');
disp('Saved results/01_channel_aoi_simulation.png');
