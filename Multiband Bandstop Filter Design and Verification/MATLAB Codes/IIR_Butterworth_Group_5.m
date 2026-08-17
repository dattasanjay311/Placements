% EE 603: EXTRAORDINARY EFFORTS - GROUP 5
% TYPE 4 (BUTTERWORTH MULTI-BANDSTOP) FILTER DESIGN - CASCADE
% Filter Number: 103
% Group 1 of Frequency bands: 85 kHz to 115 kHz
% Group 2 of Frequency bands: 190 kHz to 220 kHz

clc;
clear;
close all;

% Calling the bandstop() function with relevant arguments
[ds1, ns1, nz1, dz1, poles1, Wo1, B1] = bandstop(85, 80, 120, 115, 650, 1);
[ds2, ns2, nz2, dz2, poles2, Wo2, B2] = bandstop(190, 185, 225, 220, 650, 2);

% bandstop() function
function [ds, ns, nz, dz, poles, Wo, B] = bandstop(fs1, fp1, fp2, fs2, f_samp, a)

% Normalize frequency using Bilinear Transformation
Ws1 = tan((fs1*pi)/f_samp);
Wp1 = tan((fp1*pi)/f_samp);
Wp2 = tan((fp2*pi)/f_samp);
Ws2 = tan((fs2*pi)/f_samp);

% Parameters for BP-LP Frequency Transformation
Wo = sqrt(Wp1*Wp2);
B = Wp2 - Wp1;

% BS-LP transformation for Bandstop filter
Wls1 = (B*Ws1)/(Wo^2 - Ws1^2);
Wls2 = (B*Ws2)/(Wo^2 - Ws2^2);
Wlp = 1;
Wls = min(abs(Wls1), abs(Wls2));

% Parameters for Butterworth Approximation
D1 = (1/(0.85^2)) - 1;
D2 = (1/(0.15^2)) - 1;
N = ceil((log(D2/D1))/(2*log(Wls/Wlp))); % Order

% Range of Wc
n = (1/(2*N));
Wc1 = 1/(D1^n);
Wc2 = Wls/(D2^n);
Wc = (Wc1 + Wc2)/2;

% Angles for poles
k = 0:2*N-1;
theta = (pi/2) + (2*k + 1)*pi/(2*N);

% Poles in rectangular form
real_part = Wc * cos(theta);
imag_part = Wc * sin(theta);
poles = real_part + 1i*imag_part;

% To plot all the Poles of the Butterworth Analog LPF T/F
figure;
plot(real(poles), imag(poles), 'x', 'MarkerSize', 10, 'LineWidth', 2);
title(sprintf('Poles of Analog LPF %d (Butterworth)', a));
xlabel('Real Part'); 
ylabel('Imaginary Part'); 
grid on; 
axis equal;

% Keep only left-half plane poles (Re(p) < 0)
idx = real_part < 0;
real_part = real_part(idx);
imag_part = imag_part(idx);
poles = poles(idx);

% To plot only Left Half Poles of the Butterworth Analog LPF T/F
figure;
plot(real(poles), imag(poles), 'x', 'MarkerSize', 10, 'LineWidth', 2);
title(sprintf('Left Half Poles of Analog LPF %d (Butterworth)', a));
xlabel('Real Part'); 
ylabel('Imaginary Part'); 
grid on; 
axis equal;

% To Display the Poles in Rectangular Form
disp(table(real_part.', imag_part.', poles.'));

% Transfer Function for Analog LPF
[num, den] = zp2tf([], poles, Wc^N);
syms s z;
analog_lpf(s) = poly2sym(num, s) / poly2sym(den, s);

% Freq Tranformation from Analog LPF to Analog BSF
analog_bsf(s) = analog_lpf((B*s)/(s^2 + Wo^2));

% Bilinear Transformation (s = (z-1)/(z+1))
discrete_bsf(z) = analog_bsf((z-1)/(z+1));

% To compute and Display co-effs of Analog LPF T/F
[ns1, ds1] = numden(analog_lpf(s));
ns1 = sym2poly(ns1); 
ds1 = sym2poly(ds1);
k = ds1(1); 
ds1 = ds1/k; 
ns1 = ns1/k;
fprintf('Analog LPF %d Numerator Coefficients\n',a);
disp(ns1);
fprintf('Analog LPF %d Denominator Coefficients\n',a);
disp(ds1);

% To compute and Display co-effs of Analog BPF T/F
[ns, ds] = numden(analog_bsf(s));
ns = sym2poly(ns); 
ds = sym2poly(ds);
k = ds(1); 
ds = ds/k; 
ns = ns/k;
fprintf('Analog BSF %d Numerator Coefficients\n',a);
disp(ns);
fprintf('Analog BSF %d Denominator Coefficients\n',a);
disp(ds);

% To compute and Display co-effs of Discrete BPF T/F
[nz, dz] = numden(discrete_bsf(z));
nz = sym2poly(nz); 
dz = sym2poly(dz);
k = dz(1); 
dz = dz/k; 
nz = nz/k;
fprintf('Discrete BSF %d Numerator Coefficients\n',a);
disp(nz);
fprintf('Discrete BSF %d Denominator Coefficients\n',a);
disp(dz);
end

% To display the Analog Lowpass System Function
den1 = real(poly(poles1)); 
den2 = real(poly(poles2));
num1 = real(prod(-poles1)); 
num2 = real(prod(-poles2));
H1_s = tf(num1, den1)
H2_s = tf(num2, den2)

% To visualize the magnitude and phase response of the discrete BPFs
% In terms of Normalised Frequency (rad/sample)
fvtool(nz1, dz1);
fvtool(nz2, dz2);

% To plot the frequency response of the discrete BPFs 
% In terms of Un-normalized Frequency (Hz)
[H1,f] = freqz(nz1,dz1,1024*1024,650e3);
figure; 
plot(f,abs(H1),'LineWidth', 2); 
grid;
title('BSF1: Stopband: 85kHz-115kHz'); 
xlabel('Frequency (Hz)'); 
ylabel('|H(f)|');

[H2,f] = freqz(nz2,dz2,1024*1024,650e3);
figure; 
plot(f,abs(H2),'LineWidth', 2); 
grid;
title('BSF2: Stopband: 190kHz-220kHz'); 
xlabel('Frequency (Hz)'); 
ylabel('|H(f)|');

% Filter coefficients
b1 = nz1; a1 = dz1; % Filter 1
b2 = nz2; a2 = dz2; % Filter 2
num_cascade = conv(b1, b2);
den_cascade = conv(a1, a2);

% Normalize
k = den_cascade(1);
num_cascade = num_cascade / k;
den_cascade = den_cascade / k;

fprintf('Cascaded Multi-BSF Numerator Coefficients\n');
disp(num_cascade);
fprintf('Cascaded Multi-BSF Denominator Coefficients\n');
disp(den_cascade);

% To visualize the magnitude and phase response of the discrete Multi-BSF 
% In terms of Normalised Frequency (rad/sample)
fvtool(num_cascade, den_cascade);

% To plot the frequency response of the discrete Multi-BSF 
% In terms of Un-normalized Frequency (Hz)
[H_multi,f] = freqz(num_cascade,den_cascade,1024*1024,650e3);
figure; 
plot(f,abs(H_multi),'LineWidth', 2); 
grid;
title('Cascade Multi-BSF: Stopbands: 85kHz-115kHz and 190kHz-220kHz');
xlabel('Frequency (Hz)'); 
ylabel('|H(f)|');
