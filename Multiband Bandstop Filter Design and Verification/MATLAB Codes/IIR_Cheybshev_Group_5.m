
clear; clc; close all;

% --- Shared Filter Specifications ---
overall_min_gain = 0.85;
f_samp = 650e3;
omega_ls_1 =1.288;
D2 = 43.44;
individual_min_gain = sqrt(overall_min_gain);
D1_tight = 1/(individual_min_gain^2) - 1;
fprintf('D1 tight value:');
disp(D1_tight);
order_formula_num = acosh(sqrt(D2 / D1_tight));
epsilon = sqrt(D1_tight);

%% --- Group 1 Filter Design (N=5) ---
N1_cal = ceil(acosh(sqrt(D2/D1_tight))/(acosh(omega_ls_1)));
fprintf('N1_calculated_value: ');
disp(N1_cal);
N1_choice = 1; % Define a choice for the filter order
N1=max(N1_cal,N1_choice);

fp1_1 = 80e3;
fp2_1 = 120e3;

% Design the first filter using the custom function
[nz1, dz1, ns1, ds1] = designBandstopFilter(N1, fp1_1, fp2_1, f_samp, epsilon);

% --- Print Coefficients for Filter 1 ---
fprintf('--- ANALOG H(s) COEFFICIENTS ---\n');
fprintf('\n-- Band Stop Filter (Group-1) [analog] --\n');
fprintf('Numerator_Group_1(s):'); disp(ns1);
fprintf('Denominator_Group_1(s):'); disp(ds1);

fprintf('\n\n--- DIGITAL H(z) COEFFICIENTS ---\n');
fprintf('\n-- Band Stop Filter (Group-1) [Digital] --\n');
fprintf('Numerator_Group_1(z):'); disp(nz1);
fprintf('Denominator_Group_1(z):'); disp(dz1);

% --- Plot Responses for Filter 1 ---
figure('Name', 'Group 1 Filter Response');
subplot(2,1,1);
[H1, f1] = freqz(nz1, dz1, 2^16, f_samp);
plot(f1, abs(H1));
title('Magnitude Response of BSF 1 (85-115 kHz)');
xlabel('Frequency (Hz)'); ylabel('Magnitude'); grid on;


subplot(2,1,2);
phase1 = unwrap(angle(H1)) * 180/pi;
plot(f1, phase1);
title('Phase Response of BSF 1 (85-115 kHz)');
xlabel('Frequency (Hz)'); ylabel('Phase (degrees)'); grid on; axis tight;


%% --- Group 2 Filter Design (N=4) ---
omega_ls_2 =1.35278;
N2_cal = ceil(acosh(sqrt(D2/D1_tight))/(acosh(omega_ls_2)));
fprintf('N2_calculated_value: ');
disp(N2_cal);
N2_choice = 1; % Define a choice for the filter order
N2=max(N2_cal,N2_choice);

fp1_2 = 185e3;
fp2_2 = 225e3;

% Design the second filter using the custom function
[nz2, dz2, ns2, ds2] = designBandstopFilter(N2, fp1_2, fp2_2, f_samp, epsilon);

% --- Print Coefficients for Filter 2 ---
fprintf('\n\n--- ANALOG H(s) COEFFICIENTS ---\n');
fprintf('\n-- Band Stop Filter (Group-2) [analog] --\n');
fprintf('Numerator_Group_2(s):'); disp(ns2);
fprintf('Denominator_Group_2(s):'); disp(ds2);

fprintf('\n\n--- DIGITAL H(z) COEFFICIENTS ---\n');
fprintf('\n-- Band Stop Filter (Group-2) [Digital] --\n');
fprintf('Numerator_Group_2(z):'); disp(nz2);
fprintf('Denominator_Group_2(z):'); disp(dz2);

% --- Plot Responses for Filter 2 ---
figure('Name', 'Group 2 Filter Response');
subplot(2,1,1);
[H2, f2] = freqz(nz2, dz2, 2^16, f_samp);
plot(f2, abs(H2));
title('Magnitude Response of BSF 2 (190-220 kHz)');
xlabel('Frequency (Hz)'); ylabel('Magnitude'); grid on;

subplot(2,1,2);
phase2 = unwrap(angle(H2)) * 180/pi;
plot(f2, phase2);
title('Phase Response of BSF 2 (190-220 kHz)');
xlabel('Frequency (Hz)'); ylabel('Phase (degrees)'); grid on; axis tight;

%% --- Cascade the two filters ---
% Cascade the filters using the custom function
[nz_cascade, dz_cascade] = cascadeFilters(nz1, dz1, nz2, dz2);

% --- Print Coefficients for Cascaded Filter ---
fprintf('\n\n--- CASCADED DIGITAL H(z) COEFFICIENTS ---\n');
fprintf('Numerator_cascade(z):'); disp(nz_cascade);
fprintf('Denominator_cascade(z):'); disp(dz_cascade);

% --- Plot Responses for Cascaded Filter ---
figure('Name', 'Cascaded Filter Response');
[H_cascade, f_cascade] = freqz(nz_cascade, dz_cascade, 2^16, f_samp);
plot(f_cascade, abs(H_cascade));
title('Magnitude Response of Cascaded Filter');
xlabel('Frequency (Hz)'); ylabel('Magnitude'); grid on; axis tight;


figure('Name', 'Cascaded Filter Phase Response');
phase_cascade = unwrap(angle(H_cascade)) * 180/pi;
plot(f_cascade, phase_cascade);
title('Phase Response of Cascaded Filter');
xlabel('Frequency (Hz)'); ylabel('Phase (degrees)'); grid on; axis tight;


% Also plot with fvtool as in the original code
fvtool(nz_cascade, dz_cascade, 'Fs', f_samp);


% =========================================================================
% FUNCTION DEFINITIONS
% =========================================================================

function [nz, dz, ns_analog, ds_analog] = designBandstopFilter(N, fp1, fp2, f_samp, epsilon)
    % This function designs a single Chebyshev Type 1 band-stop filter.
    % INPUTS:
    %   N       = Filter order
    %   fp1     = Lower passband edge frequency (Hz)
    %   fp2     = Upper passband edge frequency (Hz)
    %   f_samp  = Sampling frequency (Hz)
    %   epsilon = Ripple parameter
    % OUTPUTS:
    %   nz, dz        = Numerator and denominator of the final DIGITAL filter
    %   ns_analog, ds_analog = Numerator and denominator of the intermediate ANALOG filter

    % 1. Calculate Poles of the LPF Prototype
    poles = zeros(N, 1);
    phi = (1/N) * asinh(1/epsilon);
    for k = 1:N
        theta = (pi/2) * (2*k - 1) / N;
        real_part = -sin(theta) * sinh(phi);
        imag_part = cos(theta) * cosh(phi);
        poles(k) = real_part + 1i * imag_part;
    end
    
    % 2. Create Analog LPF Prototype Transfer Function
    den_proto = real(poly(poles));
    % Correctly handle numerator for even vs. odd order
    if mod(N, 2) == 1 % N is odd
        num_proto = den_proto(end);
    else % N is even
        num_proto = den_proto(end) / sqrt(1 + epsilon^2);
    end

    % 3. Transform to Analog Band-Stop Filter
    % Pre-warp frequencies
    wp1 = tan(pi*fp1/f_samp);
    wp2 = tan(pi*fp2/f_samp);
    % Parameters for lowpass-to-bandstop transformation
    W0 = sqrt(wp1*wp2);
    B = wp2-wp1;
    
    % Perform transformation using symbolic math
    syms s z;
    analog_lpf(s) = poly2sym(num_proto, s) / poly2sym(den_proto, s);
    % NOTE: The transformation for BSF from LPF is s -> B*s / (s^2 + W0^2)
    analog_bsf(s) = analog_lpf( (B*s) / (s*s + W0^2) );
    
    % Get analog coefficients for printing
    [ns, ds] = numden(analog_bsf(s));
    ns_analog = sym2poly(expand(ns));
    ds_analog = sym2poly(expand(ds));
    k_analog = ds_analog(1);
    ds_analog = ds_analog / k_analog;
    ns_analog = ns_analog / k_analog;

    % 4. Perform Bilinear Transform to get Digital Filter
    discrete_bsf(z) = analog_bsf((z-1)/(z+1));
    
    % Extract digital coefficients
    [nz_sym, dz_sym] = numden(discrete_bsf(z));
    nz = sym2poly(expand(nz_sym));
    dz = sym2poly(expand(dz_sym));
    
    % Normalize
    k_digital = dz(1);
    dz = dz / k_digital;
    nz = nz / k_digital;
end


function [nz_out, dz_out] = cascadeFilters(nz1, dz1, nz2, dz2)
    % This function cascades two digital filters by convolving their
    % numerator and denominator coefficients.
    
    nz_out = conv(nz1, nz2);
    dz_out = conv(dz1, dz2);
    
    % Normalize the final coefficients
    k = dz_out(1);
    dz_out = dz_out / k;
    nz_out = nz_out / k;
end