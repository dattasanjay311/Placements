% 02_delta_modulation.m
%
% Linear delta modulation (DM) of the PPG waveform: the transmitter sends
% only a +/-1 bit per sample indicating whether the signal moved up or
% down by a fixed step size (delta), and the receiver integrates those
% bits to reconstruct the signal. We sweep delta and record reconstruction
% MSE, then plot original / DM bitstream / reconstruction for one delta.
%
% Cleaned up from: matlab codes/delta_modulation.m
% Fixes: original script re-used the same accumulator arrays across every
% delta in the sweep (never reset), so later MSE values were contaminated
% by earlier iterations; that's fixed here. Also switched to a relative
% data path and the PLETH column (was accidentally reading RESP).

data = csvread('../data/sample_Signals.csv', 1, 0); % skip header row
amplitude_column = data(:, 3); % PLETH

x = amplitude_column(1:400); % a short window, same idea as original (used 160 samples)

deltas = 0.01:0.01:1;
mse_vs_delta = zeros(size(deltas));

for d = 1:numel(deltas)
    delta = deltas(d);
    r = 0;
    x_hat = zeros(size(x));
    reconstructed = zeros(size(x));
    for i = 1:numel(x)
        x_hat(i) = r;
        step = delta;
        if (x(i) - x_hat(i)) < 0
            step = -delta;
        end
        r = x_hat(i) + step;
        reconstructed(i) = r;
    end
    mse_vs_delta(d) = sum((x - reconstructed).^2) / numel(x);
end

% detailed view at a representative delta
delta = 0.1;
r = 0; sqc = []; rc = [];
for i = 1:numel(x)
    x_(i) = r;
    if (x(i) - x_(i)) >= 0
        sq = delta;
    else
        sq = -delta;
    end
    sqc = [sqc sq];
    rc = [rc r];
    r = x_(i) + sq;
end

figure('visible','off');
subplot(3,1,1); plot(x); title('Original PPG segment'); xlabel('Sample'); ylabel('Amplitude'); grid on;
subplot(3,1,2); stairs(sqc); title(sprintf('Delta-modulated bitstream (\\delta = %.2f)', delta)); xlabel('Sample'); ylabel('+/-\delta'); grid on;
subplot(3,1,3); plot(x); hold on; plot(rc, '--'); legend('original','reconstructed'); title('Reconstructed signal'); xlabel('Sample'); ylabel('Amplitude'); grid on;
print('../results/02a_delta_modulation_waveforms.png', '-dpng', '-r120');

figure('visible','off');
plot(deltas, mse_vs_delta, '-'); xlabel('Step size \delta'); ylabel('MSE');
title('Delta modulation: reconstruction MSE vs step size'); grid on;
print('../results/02b_delta_modulation_mse_vs_delta.png', '-dpng', '-r120');

disp('Saved results/02a_delta_modulation_waveforms.png and 02b_delta_modulation_mse_vs_delta.png');
