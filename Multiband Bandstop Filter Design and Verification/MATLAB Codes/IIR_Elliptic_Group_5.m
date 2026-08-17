% EE 603:  EXTRAORDINARY EFFORTS - GROUP 5
% TYPE 6 (ELLIPTIC MULTI-BANDSTOP) FILTER DESIGN - CASCADE
% Filter Number: 103
% Group 1 of Frequency bands: 85 kHz to 115 kHz
% Group 2 of Frequency bands: 190 kHz to 220 kHz

clc; clear; close all;
% This script requires the Symbolic Math Toolbox
syms s sl z Omega_L;

%% ========================================================================
% Section 1: Global Specifications and LPF Prototype
% =========================================================================

% --- System Specifications ---
fs = 650e3;         % Sampling frequency
delta_p = 0.15;     % Passband ripple (Magnitude 0.85 to 1.0)
delta_s = 0.15;     % Stopband attenuation (Magnitude 0 to 0.15)
As_dB = -20*log10(delta_s);   % 16.48 dB (Stopband spec is the same for each filter)

% --- Tighter Specs for Cascade Design ---
% The final passband mag_min is (1-delta_p) = 0.85.
% In cascade, H_final = H1 * H2.
% The worst-case passband gain is (mag_min_1) * (mag_min_2).
% To ensure (mag_min_each)^2 >= 0.85, we must have:
mag_min_each = sqrt(1 - delta_p); % sqrt(0.85) = approx 0.922
Ap_dB_each = -20*log10(mag_min_each); % approx 0.702 dB
% -----------------------------------------

N_proto = 3; % From running ellipap2 for parameters
fprintf('--- LPF Prototype Implementation ---\n');
fprintf('Using N=%d LPF Prototype.\n', N_proto);
fprintf('Original Ap_dB for final filter: %.2f dB\n', -20*log10(1-delta_p));
fprintf('Tighter Ap_dB for *each* filter in cascade: %.3f dB\n', Ap_dB_each);

% --- build H_LPF(sl) ---
[z_proto, p_proto, k_proto] = ellipap2(N_proto, Ap_dB_each, As_dB);

% Convert Z/P/K vectors to symbolic transfer function
num_poly = k_proto * real(poly(z_proto));
den_poly = real(poly(p_proto));

%% --- LPF Prototype Details Printed---
fprintf('\n--- LPF PROTOTYPE (N=%d) DETAILS ---\n', N_proto);
fprintf('Zeros (z_proto):\n');
disp(z_proto);
fprintf('Poles (p_proto):\n');
disp(p_proto);
fprintf('Gain (k_proto):\n');
disp(k_proto);
fprintf('Numerator Coefficients H_LPF(sl) (num_poly):\n');
disp(num_poly);
fprintf('Denominator Coefficients H_LPF(sl) (den_poly):\n');
disp(den_poly);

% Create symbolic LPF prototype function
H_LPF(sl) = poly2sym(num_poly, sl) / poly2sym(den_poly, sl);
fprintf('Successfully built N=%d symbolic LPF prototype.\n', N_proto);

% --- Assignment Specs ---
M = 103;
Q = floor(M/11); % Q = 9
R = mod(M, 11);   % R = 4
transition = 5e3;

% --- BSF 1 (85-115 kHz Stopband) ---
fprintf('\n--- BSF 1 Specifications ---\n');
f_p1_1 = (40 + 5*Q)*1e3 - transition; % 80k
f_p2_1 = (70 + 5*Q)*1e3 + transition; % 120k
f_s1_1 = (40 + 5*Q)*1e3;              % 85k
f_s2_1 = (70 + 5*Q)*1e3;              % 115k
fprintf('Passbands: 0-%.0f kHz and %.0f-%.0f kHz\n', f_p1_1/1e3, f_p2_1/1e3, fs/2e3);
fprintf('Stopband:  %.0f-%.0f kHz\n', f_s1_1/1e3, f_s2_1/1e3);

%% --- BSF1 Normalized Specs Printed---
fprintf('\n--- BSF1: Normalized Digital Specs (w) ---\n');
fprintf('w_p1_1 (Passband): %.4f*pi rad\n', (f_p1_1 / (fs/2)));
fprintf('w_p2_1 (Passband): %.4f*pi rad\n', (f_p2_1 / (fs/2)));
fprintf('w_s1_1 (Stopband): %.4f*pi rad\n', (f_s1_1 / (fs/2)));
fprintf('w_s2_1 (Stopband): %.4f*pi rad\n', (f_s2_1 / (fs/2)));

% Pre-warp all BSF1 frequencies
Omega_p1_1 = 2*fs*tan(pi*f_p1_1/fs);
Omega_p2_1 = 2*fs*tan(pi*f_p2_1/fs);
Omega_s1_1 = 2*fs*tan(pi*f_s1_1/fs);
Omega_s2_1 = 2*fs*tan(pi*f_s2_1/fs);
Omega_0_1 = sqrt(Omega_p1_1 * Omega_p2_1);
B_1 = Omega_p2_1 - Omega_p1_1;

%% --- BSF1 Analog Specs Printed---
fprintf('\n--- BSF1: Analog Specs (Omega) ---\n');
fprintf('Omega_p1_1: %.4e rad/s\n', Omega_p1_1);
fprintf('Omega_p2_1: %.4e rad/s\n', Omega_p2_1);
fprintf('Omega_s1_1: %.4e rad/s\n', Omega_s1_1);
fprintf('Omega_s2_1: %.4e rad/s\n', Omega_s2_1);
fprintf('\n--- BSF1: LPF Transformation Params ---\n');
fprintf('Omega_0_1: %.4e rad/s\n', Omega_0_1);
fprintf('B_1:       %.4e rad/s\n', B_1);

% --- BSF 2 (190-220 kHz Stopband) ---
fprintf('\n--- BSF 2 Specifications ---\n');
f_p1_2 = (170 + 5*R)*1e3 - transition; % 185k
f_p2_2 = (200 + 5*R)*1e3 + transition; % 225k
f_s1_2 = (170 + 5*R)*1e3;              % 190k
f_s2_2 = (200 + 5*R)*1e3;              % 220k
fprintf('Passbands: 0-%.0f kHz and %.0f-%.0f kHz\n', f_p1_2/1e3, f_p2_2/1e3, fs/2e3);
fprintf('Stopband:  %.0f-%.0f kHz\n', f_s1_2/1e3, f_s2_2/1e3);

%% --- BSF2 Normalized Specs Printed---
fprintf('\n--- BSF2: Normalized Digital Specs (w) ---\n');
fprintf('w_p1_2 (Passband): %.4f*pi rad\n', (f_p1_2 / (fs/2)));
fprintf('w_p2_2 (Passband): %.4f*pi rad\n', (f_p2_2 / (fs/2)));
fprintf('w_s1_2 (Stopband): %.4f*pi rad\n', (f_s1_2 / (fs/2)));
fprintf('w_s2_2 (Stopband): %.4f*pi rad\n', (f_s2_2 / (fs/2)));

% Pre-warp all BSF2 frequencies
Omega_p1_2 = 2*fs*tan(pi*f_p1_2/fs);
Omega_p2_2 = 2*fs*tan(pi*f_p2_2/fs);
Omega_s1_2 = 2*fs*tan(pi*f_s1_2/fs);
Omega_s2_2 = 2*fs*tan(pi*f_s2_2/fs);
Omega_0_2 = sqrt(Omega_p1_2 * Omega_p2_2);
B_2 = Omega_p2_2 - Omega_p1_2;

%% --- BSF2 Analog Specs Printed---
fprintf('\n--- BSF2: Analog Specs (Omega) ---\n');
fprintf('Omega_p1_2: %.4e rad/s\n', Omega_p1_2);
fprintf('Omega_p2_2: %.4e rad/s\n', Omega_p2_2);
fprintf('Omega_s1_2: %.4e rad/s\n', Omega_s1_2);
fprintf('Omega_s2_2: %.4e rad/s\n', Omega_s2_2);
fprintf('\n--- BSF2: LPF Transformation Params ---\n');
fprintf('Omega_0_2: %.4e rad/s\n', Omega_0_2);
fprintf('B_2:       %.4e rad/s\n', B_2);

%% ========================================================================
% Section 2: Symbolic Transformation 
% =========================================================================

% --- Design BSF 1 ---
fprintf('\nStarting symbolic design for BSF1... (This may take a minute)\n');
% 1. Analog LP -> Analog BSF
H_BSF1(s) = subs(H_LPF, sl, (B_1 * s) / (s^2 + Omega_0_1^2));

%% --- BSF1 Analog H(s) Coeffs Printed---
fprintf('\n--- BSF1: Analog BSF H(s) Coefficients ---');
[Ns1s, Ds1s] = numden(H_BSF1);
Ns1_analog = sym2poly(Ns1s);
Ds1_analog = sym2poly(Ds1s);
k_a1 = Ds1_analog(1);
Ns1_analog = Ns1_analog / k_a1;
Ds1_analog = Ds1_analog / k_a1;
disp('Numerator N(s):'); disp(Ns1_analog);
disp('Denominator D(s):'); disp(Ds1_analog);

% 2. Analog BSF -> Digital BSF (using un-normalized Bilinear Transform)
H_z1(z) = subs(H_BSF1, s, 2*fs*(z-1)/(z+1));
% 3. Extract coefficients
[Nz1s, Dz1s] = numden(H_z1);
Nz1 = sym2poly(Nz1s);
Dz1 = sym2poly(Dz1s);
k1 = Dz1(1);
Nz1 = Nz1 / k1;
Dz1 = Dz1 / k1;
fprintf('Symbolic BSF1 complete.\n');

%% --- BSF1 Digital H(z) Coeffs Printed---
fprintf('\n--- BSF1: Digital BSF H(z) Coefficients ---');
disp('Numerator N(z):'); disp(Nz1);
disp('Denominator D(z):'); disp(Dz1);

% --- Design BSF 2 ---
fprintf('\nStarting symbolic design for BSF2... (This may take another minute)\n');
% 1. Analog LP -> Analog BSF
H_BSF2(s) = subs(H_LPF, sl, (B_2 * s) / (s^2 + Omega_0_2^2));

%% --- BSF2 Analog H(s) Coeffs Printed---
fprintf('\n--- BSF2: Analog BSF H(s) Coefficients ---');
[Ns2s, Ds2s] = numden(H_BSF2);
Ns2_analog = sym2poly(Ns2s);
Ds2_analog = sym2poly(Ds2s);
k_a2 = Ds2_analog(1);
Ns2_analog = Ns2_analog / k_a2;
Ds2_analog = Ds2_analog / k_a2;
disp('Numerator N(s):'); disp(Ns2_analog);
disp('Denominator D(s):'); disp(Ds2_analog);

% 2. Analog BSF -> Digital BSF
H_z2(z) = subs(H_BSF2, s, 2*fs*(z-1)/(z+1));
% 3. Extract coefficients
[Nz2s, Dz2s] = numden(H_z2);
Nz2 = sym2poly(Nz2s);
Dz2 = sym2poly(Dz2s);
k2 = Dz2(1);
Nz2 = Nz2 / k2;
Dz2 = Dz2 / k2;
fprintf('Symbolic BSF2 complete.\n');

%% --- BSF2 Digital H(z) Coeffs Printed---
fprintf('\n--- BSF2: Digital BSF H(z) Coefficients ---');
disp('Numerator N(z):'); disp(Nz2);
disp('Denominator D(z):'); disp(Dz2);

%% ========================================================================
% Section 3: Cascade, Plotting, and Analysis
% =========================================================================
fprintf('\nCascading filters and plotting...\n');

% Cascade two filters by convolving their coefficients
Nz_final = conv(Nz1, Nz2);
Dz_final = conv(Dz1, Dz2);

% Normalize final filter
k_final = Dz_final(1);
Nz_final = Nz_final / k_final;
Dz_final = Dz_final / k_final;

% Display the first few coefficients
N_total = length(Dz_final) - 1;
fprintf('\n--- Final Cascaded Filter Coefficients (N_total = %d) ---\n', N_total);
disp('Numerator (first 10):'); disp(Nz_final(1:min(end,10)));
disp('Denominator (first 10):'); disp(Dz_final(1:min(end,10)));

% --- Generate frequency response data for static plots ---
n_fft = 8192; 
[Hc, fc] = freqz(Nz_final, Dz_final, n_fft, fs);
[gd, f_gd] = grpdelay(Nz_final, Dz_final, n_fft, fs);

% --- Static Plot 1: Final Magnitude Response ---
figure('Name', 'Final Cascade - Magnitude Response');
plot(fc/1e3, abs(Hc), 'b', 'LineWidth', 1.5);
grid on;
title(sprintf('Final Cascaded Magnitude Response (N=%d)', N_total));
xlabel('Frequency (kHz)');
ylabel('Magnitude |H(e^{j\omega})|');
hold on;
yline(1.0, 'k--', 'LineWidth', 1);
% plotting spec lines (0.85 and 0.15)
yline(1-delta_p, 'r--', sprintf('Final Spec (1 - \\delta_p) = %.2f', 1-delta_p), 'LineWidth', 1);
yline(delta_s, 'r--', sprintf('Final Spec (\\delta_s) = %.2f', delta_s), 'LineWidth', 1);
xline(f_p1_1/1e3, 'r:', 'fp1_1', 'LineWidth', 1);
xline(f_p2_1/1e3, 'r:', 'fp2_1', 'LineWidth', 1);
xline(f_s1_1/1e3, 'r:', 'fs1_1', 'LineWidth', 1);
xline(f_s2_1/1e3, 'r:', 'fs2_1', 'LineWidth', 1);
xline(f_p1_2/1e3, 'r:', 'fp1_2', 'LineWidth', 1);
xline(f_p2_2/1e3, 'r:', 'fp2_2', 'LineWidth', 1);
xline(f_s1_2/1e3, 'r:', 'fs1_2', 'LineWidth', 1);
xline(f_s2_2/1e3, 'r:', 'fs2_2', 'LineWidth', 1);
hold off;
axis([0 fs/2e3 0 1.2]); 

% --- Static Plot 2: Final Phase Response ---
figure('Name', 'Final Cascade - Phase Response');
plot(fc/1e3, unwrap(angle(Hc))*180/pi, 'b', 'LineWidth', 1.5);
grid on;
title('Final Cascaded Phase Response (Unwrapped)');
xlabel('Frequency (kHz)');
ylabel('Phase (degrees)');
axis([0 fs/2e3 -inf inf]);

% --- Static Plot 3: Final Group Delay ---
figure('Name', 'Final Cascade - Group Delay');
plot(f_gd/1e3, gd, 'b', 'LineWidth', 1.5);
grid on;
title('Final Cascaded Group Delay');
xlabel('Frequency (kHz)');
ylabel('Group Delay (samples)');
axis([0 fs/2e3 0 inf]); 


% --- FVTool Analysis  ---

% FVTool for BSF1 (Group 1)
fprintf('\nLaunching FVTool for BSF1...\n');
h1 = fvtool(Nz1, Dz1, 'Fs', fs, 'Color', 'white', ...
    'Name', sprintf('BSF1 (N=%d)', length(Dz1)-1));

% FVTool for BSF2 (Group 2)
fprintf('\nLaunching FVTool for BSF2...\n');
h2 = fvtool(Nz2, Dz2, 'Fs', fs, 'Color', 'white', ...
    'Name', sprintf('BSF2 (N=%d)', length(Dz2)-1));

% FVTool for the Final Cascaded Filter
fprintf('\nLaunching FVTool for Final Filter (Cascade)...\n');
h_final = fvtool(Nz_final, Dz_final, 'Fs', fs, 'Color', 'white', ...
    'Name', sprintf('Final Cascaded Elliptic BSF (N=%d)', N_total));

fprintf('\nScript complete. All plots generated.\n');

%------------------HELPER FUNCTION--------------

function [z, p, H0, B, A] = ellipap2(N, Ap, As)
% ellipap2 - custom function to compute Elliptic (Cauer) analog lowpass filter prototype.

if nargin == 0
    help ellipap2;
    return;
end
Gp = 10^(-Ap / 20); % passband gain
ep = sqrt(10^(Ap / 10) - 1); % ripple factors
es = sqrt(10^(As / 10) - 1);
k1 = ep / es;
k = ellipdeg(N, k1); % solve degree equation
L = floor(N / 2);
r = mod(N, 2); % L is the number of second-order sections
i = (1:L)';
ui = (2 * i - 1) / N;
zeta_i = cde(ui, k); % zeros of elliptic rational function
z_custom = 1j ./ (k * zeta_i); % filter zeros = poles of elliptic rational function
% solution of sn(j*v0*N*K1, k1) = j/ep
v0 = -1j * asne(1j / ep, k1) / N; 
% filter poles
p_custom = 1j * cde(ui - 1j * v0, k); 
% first-order pole, needed when N is odd
p0 = 1j * sne(1j * v0, k); 
[z, p, H0] = ellipap(N, Ap, As);
% second-order numerator sections
B = [ones(L, 1), -2 * real(1 ./ z_custom), abs(1 ./ z_custom).^2]; 
% second-order denominator sections
A = [ones(L, 1), -2 * real(1 ./ p_custom), abs(1 ./ p_custom).^2]; 
if r == 0 % prepend first-order sections
    B = [Gp, 0, 0; B];
    A = [1, 0, 0; A];
else
    B = [1, 0, 0; B];
    A = [1, -real(1 / p0), 0; A];
end
z_custom = cplxpair([z_custom; conj(z_custom)]); % append conjugate zeros
p_custom = cplxpair([p_custom; conj(p_custom)]); % append conjugate poles
if r == 1
    p_custom = [p_custom; p0]; % append first-order pole when N is odd
end
H0_custom = Gp^(1 - r); % dc gain
end