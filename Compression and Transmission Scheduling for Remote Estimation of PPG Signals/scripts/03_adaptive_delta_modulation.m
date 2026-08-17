% 03_adaptive_delta_modulation.m
%
% Adaptive delta modulation (ADM): like linear DM, but the step size grows
% when consecutive bits agree (tracking fast slopes) and shrinks otherwise
% -- this avoids the granular noise / slope-overload trade-off that a
% fixed delta forces you into.
%
% Cleaned up from: matlab codes/adaptive_delta_compression.m
% Fixes: relative data path, PLETH column, comments explaining the
% step-doubling rule; logic otherwise kept the same as the original.

data = csvread('../data/sample_Signals.csv', 1, 0);
amplitude_column = data(:, 3); % PLETH

x = amplitude_column(1:400);
delta0 = 0.01;   % minimum step size
delta = delta0;
sig = 0;
bits = zeros(1, numel(x));

for i = 1:numel(x)
    if x(i) >= sig
        bits(i) = 1;
    else
        bits(i) = -1;
    end
    if i > 1
        delta = abs(delta) * bits(i) + delta0 * bits(i-1); % grow/shrink rule
    end
    sig = sig + delta;
end

% reconstruct at the receiver from the bitstream alone
delta = delta0;
sig = delta * bits(1);
reconstructed = zeros(1, numel(bits));
reconstructed(1) = sig;
for i = 2:numel(bits)
    delta = abs(delta) * bits(i) + delta0 * bits(i-1);
    sig = sig + delta;
    reconstructed(i) = sig;
end

mse = sum((x(:) - reconstructed(:)).^2) / numel(x);
printf('Adaptive delta modulation MSE: %.6f\n', mse);

figure('visible','off');
subplot(3,1,1); plot(x); title('Original PPG segment'); xlabel('Sample'); ylabel('Amplitude'); grid on;
subplot(3,1,2); stairs(bits); title('ADM bitstream (+1 / -1)'); xlabel('Sample'); ylabel('bit'); grid on;
subplot(3,1,3); plot(x); hold on; plot(reconstructed, '--');
legend('original','reconstructed');
title(sprintf('ADM reconstruction (MSE = %.4f)', mse)); xlabel('Sample'); ylabel('Amplitude'); grid on;
print('../results/03_adaptive_delta_modulation.png', '-dpng', '-r120');
disp('Saved results/03_adaptive_delta_modulation.png');
