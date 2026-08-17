% 04_dct_compression.m
%
% Windowed DCT compression of the PPG signal: normalize each window,
% take its DCT, keep only enough low-frequency coefficients to capture a
% target fraction of the signal energy (the "energy threshold"), discard
% the rest, and reconstruct with the inverse DCT. We sweep the energy
% threshold and report reconstruction MSE and compression ratio.
%
% Cleaned up from: matlab codes/dct_my.m and
% matlab codes/partial_dct_paper_varying_energy_threshold.m
% Fixes: relative data path; original scripts required the DCT/signal
% toolbox function names available in Octave's 'signal' package.

pkg load signal; % provides dct/idct in Octave

data = csvread('../data/sample_Signals.csv', 1, 0);
amplitude_column = data(:, 3); % PLETH

window_size = 500;
n_windows = floor(numel(amplitude_column) / window_size);
energy_thresholds = 0.5:0.05:0.99;

mse_vs_threshold = zeros(size(energy_thresholds));
compression_ratio_vs_threshold = zeros(size(energy_thresholds));

for k = 1:numel(energy_thresholds)
    energy_threshold = energy_thresholds(k);
    reconstructed = [];
    original_norm = [];
    n_coeffs_used = 0;

    for w = 1:n_windows
        raw = amplitude_column((w-1)*window_size+1 : w*window_size);
        normalized = (raw - mean(raw)) / std(raw);
        coeffs = dct(normalized);
        cumulative_energy = cumsum(coeffs.^2) / sum(coeffs.^2);
        threshold_index = find(cumulative_energy >= energy_threshold, 1);
        compressed = coeffs;
        compressed(threshold_index+1:end) = 0;
        reconstructed = [reconstructed; idct(compressed)];
        original_norm = [original_norm; normalized];
        n_coeffs_used = n_coeffs_used + threshold_index;
    end

    mse_vs_threshold(k) = sum((original_norm - reconstructed).^2) / numel(original_norm);
    compression_ratio_vs_threshold(k) = (n_windows * window_size) / n_coeffs_used;
end

% detailed view at one representative threshold
energy_threshold = 0.9;
reconstructed = []; original_norm = []; last_coeffs = [];
for w = 1:n_windows
    raw = amplitude_column((w-1)*window_size+1 : w*window_size);
    normalized = (raw - mean(raw)) / std(raw);
    coeffs = dct(normalized);
    cumulative_energy = cumsum(coeffs.^2) / sum(coeffs.^2);
    threshold_index = find(cumulative_energy >= energy_threshold, 1);
    compressed = coeffs;
    compressed(threshold_index+1:end) = 0;
    reconstructed = [reconstructed; idct(compressed)];
    original_norm = [original_norm; normalized];
    last_coeffs = coeffs;
end

figure('visible','off');
subplot(3,1,1); plot(original_norm(1:2000)); title('Normalized original PPG (first 4 windows)'); grid on;
subplot(3,1,2); plot(reconstructed(1:2000)); title(sprintf('DCT reconstruction (energy threshold = %.2f)', energy_threshold)); grid on;
subplot(3,1,3); plot(last_coeffs); title('DCT coefficients of last window'); grid on;
print('../results/04a_dct_compression_waveforms.png', '-dpng', '-r120');

figure('visible','off');
subplot(2,1,1); plot(energy_thresholds, mse_vs_threshold, '-o'); xlabel('Energy threshold'); ylabel('MSE'); title('DCT compression: MSE vs energy threshold'); grid on;
subplot(2,1,2); plot(energy_thresholds, compression_ratio_vs_threshold, '-o'); xlabel('Energy threshold'); ylabel('Compression ratio'); title('DCT compression: compression ratio vs energy threshold'); grid on;
print('../results/04b_dct_compression_vs_threshold.png', '-dpng', '-r120');

disp('Saved results/04a_dct_compression_waveforms.png and 04b_dct_compression_vs_threshold.png');
