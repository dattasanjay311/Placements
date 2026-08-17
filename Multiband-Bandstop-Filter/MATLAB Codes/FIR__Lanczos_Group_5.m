% Name : Datla Ravi Teja
% Roll No : 25M1064

f_samp = 650e3;

%Band Edge speifications for filter 1
tolarance=0.15;
f1_fp1 = 80e3;
f1_fs1 = 85e3;
f1_fs2 = 115e3;
f1_fp2 = 120e3;


f1_fp1_n = 2*f1_fp1/f_samp;
f1_fs1_n = 2*f1_fs1/f_samp;
f1_fs2_n = 2*f1_fs2/f_samp;
f1_fp2_n = 2*f1_fp2/f_samp;

%Band Edge speifications for filter 2
f2_fp1 = 185e3;
f2_fs1 = 190e3;
f2_fs2 = 220e3;
f2_fp2 = 225e3;


f2_fp1_n = 2*f2_fp1/f_samp;
f2_fs1_n = 2*f2_fs1/f_samp;
f2_fs2_n = 2*f2_fs2/f_samp;
f2_fp2_n = 2*f2_fp2/f_samp;


function h = ideal_lp(cutoff, n)

    t = -floor(n/2):floor(n/2);
    % Calculate the ideal low-pass filter response
    h = (sin(cutoff * t)) ./ (pi * t);
    h(floor(n/2) + 1) = cutoff / pi; % Handle the zero division case
    h = h .* (abs(t) <= (n/2)); % Apply the cutoff
end

function l = Lanczos_window(n)

    t = -floor(n/2):floor(n/2);
    l = (sin(2*t*pi/(2*n))) ./ (2*t*pi/(2*n));
    l(floor(n/2) + 1) = 1; % Handle the zero division case
    l = l .* (abs(t) <= (n/2)); % Apply the cutoff
end

n=129;

if mod(n, 2) == 0
    n = n + 1; % Make n odd
end


%Ideal bandstop impulse response of length "n"

bs_ideal1 =  ideal_lp(pi,n) -ideal_lp(((f1_fp2_n+f1_fs2_n)/2)*pi,n) + ideal_lp(((f1_fp1_n+f1_fs1_n)/2)*pi,n);

lanc= Lanczos_window(n);
disp(lanc)

FIR_BandStop1 = bs_ideal1 .* lanc;
fvtool(FIR_BandStop1);         %frequency response

%magnitude response
[H,f] = freqz(FIR_BandStop1,1,1024, f_samp);
plot(f,abs(H))
grid


%Ideal bandstop impulse response of length "n"

bs_ideal2 =  ideal_lp(pi,n) -ideal_lp(((f2_fp2_n+f2_fs2_n)/2)*pi,n) + ideal_lp(((f2_fp1_n+f2_fs1_n)/2)*pi,n);

FIR_BandStop2 = bs_ideal2 .* lanc;
fvtool(FIR_BandStop2);         %frequency response

%magnitude response
[H,f] = freqz(FIR_BandStop2,1,1024, f_samp);
figure;
plot(f,abs(H))
grid

FIR_BandStop=conv(FIR_BandStop1,FIR_BandStop2);

fvtool(FIR_BandStop);         %frequency response

%magnitude response
[H,f] = freqz(FIR_BandStop,1,1024, f_samp);
figure;
plot(f,abs(H))
grid

%Ideal bandstop impulse response of length "n"
bs_ideal =  ideal_lp(pi,n) -ideal_lp(((f2_fp2_n+f2_fs2_n)/2)*pi,n) + ideal_lp(((f2_fp1_n+f2_fs1_n)/2)*pi,n)-ideal_lp(((f1_fp2_n+f1_fs2_n)/2)*pi,n) + ideal_lp(((f1_fp1_n+f1_fs1_n)/2)*pi,n);;


FIR_BandStop3 = bs_ideal .* lanc;
fvtool(FIR_BandStop3);         %frequency response

%magnitude response
[H,f] = freqz(FIR_BandStop3,1,1024, f_samp);
figure;
plot(f,abs(H))
grid

figure;
plot(f, unwrap(angle(H)));
grid


disp('h[n] array =')
% Print 5 numbers per line, with 3 decimal places
nCols = 5;
for i = 1:length(FIR_BandStop3)
    fprintf('%8.3f', FIR_BandStop3(i));  % Print each number with width=8, 3 decimals
    if mod(i, nCols) == 0 || i == length(FIR_BandStop3)
        fprintf('\n');  % Newline every 5 numbers or at the end
    end
end
