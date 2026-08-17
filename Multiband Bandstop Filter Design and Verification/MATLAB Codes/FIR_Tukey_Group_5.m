% Name : Datla Ravi Teja
% Roll No : 25M1064

clc; clear; close all;

f_samp = 650e3;

% ---------------- Band Edge specifications ----------------
tolerance = 0.15;

% Filter 1
f1_fp1 = 80e3;  f1_fs1 = 85e3;  f1_fs2 = 115e3;  f1_fp2 = 120e3;
f1_fp1_n = 2*f1_fp1/f_samp;
f1_fs1_n = 2*f1_fs1/f_samp;
f1_fs2_n = 2*f1_fs2/f_samp;
f1_fp2_n = 2*f1_fp2/f_samp;

% Filter 2
f2_fp1 = 185e3; f2_fs1 = 190e3; f2_fs2 = 220e3; f2_fp2 = 225e3;
f2_fp1_n = 2*f2_fp1/f_samp;
f2_fs1_n = 2*f2_fs1/f_samp;
f2_fs2_n = 2*f2_fs2/f_samp;
f2_fp2_n = 2*f2_fp2/f_samp;

% ---------------- Filter Parameters ----------------
n = 129;
if mod(n,2) == 0
    n = n + 1; % ensure odd order
end

% ---------------- Custom Tukey Window (based on given formula) ----------------
alpha = 0.5; % 0 < alpha < 1
N = (n-1)/2; % symmetric about zero
k = -N:N;

w = zeros(1, n);

for i = 1:n
    ki = abs(k(i));
    if ki <= alpha*N
        w(i) = 1;
    elseif ki <= N
        w(i) = 0.5 * (1 + cos(((ki - alpha*N) * pi) / ((1 - alpha)*N)));
    else
        w(i) = 0;
    end
end

tuk = w; % final window

% ---------------- Plot the Tukey Window ----------------
figure;
subplot(2,1,1);
plot(k, tuk, 'LineWidth', 1.5);
xlabel('Sample Index (k)'); ylabel('Amplitude');
title(sprintf('Custom Tukey Window (α = %.2f)', alpha));
grid on;

% Frequency response of window
[Hw, fw] = freqz(tuk, 1, 1024, f_samp);
subplot(2,1,2);
plot(fw, 20*log10(abs(Hw)), 'LineWidth', 1.5);
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
title('Frequency Response of Tukey Window');
grid on;

% ---------------- Ideal LPF Function ----------------
function h = ideal_lp(cutoff, n)
    t = -floor(n/2):floor(n/2);
    h = (sin(cutoff .* t)) ./ (pi .* t);
    h(floor(n/2) + 1) = cutoff / pi;
    h = h .* (abs(t) <= (n/2));
end

% ---------------- Filter 1 (Bandstop) ----------------
bs_ideal1 =  ideal_lp(pi,n) -ideal_lp(((f1_fp2_n+f1_fs2_n)/2)*pi,n) + ideal_lp(((f1_fp1_n+f1_fs1_n)/2)*pi,n);
FIR_BandStop1 = bs_ideal1 .* tuk;
fvtool(FIR_BandStop1);
[H,f] = freqz(FIR_BandStop1, 1, 2048, f_samp);
figure; plot(f, abs(H)); grid on;
title(sprintf('Bandstop 1 (Custom Tukey, α=%.2f)', alpha));
xlabel('Frequency (Hz)'); ylabel('|H(f)|');

% ---------------- Filter 2 (Bandstop) ----------------
bs_ideal2 =  ideal_lp(pi,n) -ideal_lp(((f2_fp2_n+f2_fs2_n)/2)*pi,n) + ideal_lp(((f2_fp1_n+f2_fs1_n)/2)*pi,n);
FIR_BandStop2 = bs_ideal2 .* tuk;
fvtool(FIR_BandStop2);
[H,f] = freqz(FIR_BandStop2,1,2048,f_samp);
figure; plot(f, abs(H)); grid on;
title(sprintf('Bandstop 2 (Custom Tukey, α=%.2f)', alpha));
xlabel('Frequency (Hz)'); ylabel('|H(f)|');

% ---------------- Convolution (Combined) ----------------
FIR_BandStop = conv(FIR_BandStop1, FIR_BandStop2);
fvtool(FIR_BandStop);
[H,f] = freqz(FIR_BandStop, 1, 2048, f_samp);
figure; plot(f, abs(H)); grid on;
title('Combined Bandstop Filter (Convolution)');
xlabel('Frequency (Hz)'); ylabel('|H(f)|');

% ---------------- Direct Combined Design ----------------
bs_ideal =  ideal_lp(pi,n) -ideal_lp(((f2_fp2_n+f2_fs2_n)/2)*pi,n) + ideal_lp(((f2_fp1_n+f2_fs1_n)/2)*pi,n)-ideal_lp(((f1_fp2_n+f1_fs2_n)/2)*pi,n) + ideal_lp(((f1_fp1_n+f1_fs1_n)/2)*pi,n);;
FIR_BandStop3 = bs_ideal .* tuk;
fvtool(FIR_BandStop3);
[H,f] = freqz(FIR_BandStop3,1,2048,f_samp);
figure; plot(f, abs(H)); grid on;
title(sprintf('Direct Combined Bandstop (Custom Tukey, α=%.2f)', alpha));
xlabel('Frequency (Hz)'); ylabel('|H(f)|');

figure;
plot(f, unwrap(angle(H)));
title('Magnitude Response - Direct Combined Bandstop Filter');
xlabel('Frequency (Hz)');
ylabel('|H(f)|');
grid on;

disp('h[n] array =')
% Print 5 numbers per line, with 3 decimal places
nCols = 5;
for i = 1:length(FIR_BandStop3)
    fprintf('%8.3f', FIR_BandStop3(i));  % Print each number with width=8, 3 decimals
    if mod(i, nCols) == 0 || i == length(FIR_BandStop3)
        fprintf('\n');  % Newline every 5 numbers or at the end
    end
end