% EE 603: EXTRAORDINARY EFFORTS - GROUP 5
% TYPE 8 (FIR Filter using Kaiser Window) 
% Filter Number: 103
% Group 1 of Frequency bands: 85 kHz to 115 kHz
% Group 2 of Frequency bands: 190 kHz to 220 kHz

% For varying beta value manually , go to code line 56
% For varying the margin for filter coefficients (N_min + mar) go to code line 114


%Band Edge specifications - User Input
clc;
clear;
close all;

%====== Calculation of freq Bands ===========


D = 9;
R = 4;
del=0.15;
% Calculating stop band edges

% StopBand 1            % StopBand 2
fs1=40e3+5e3*D;             fs3=170e3+5e3*R;    
fs2=70e3+5e3*D;             fs4=200e3+5e3*R;

fp1=fs1-5e3;              fp3=fs3-5e3;
fp2=fs2+5e3;              fp4=fs4+5e3;

f_samp = 650e3;             % Sampling rate

%=============== Fn to cal discrete freq from analog freq (in Hz)

function w = A2Dfreq (f)        % f to be in Hz
    
    f_samp = 650e3;             % Sampling rate
    w = 2*pi*(f/f_samp);
end


%============== Function to calculate N for LPF====================

function [L,M,N,beta] = Ncal (fs,fp,del,mar)

    wantOddLength = true;       % Type-I linear phase

    %Kaiser parameters
        A = -20*log10(del);
        if(A < 21)
            beta = 0;
        elseif(A <=50)
            beta = 0.5842*(A-21)^0.4 + 0.07886*(A-21);
        else
            beta = 0.1102*(A-8.7);
        end
        beta = 5;
        % Wt calculation
             
        %Wt = 2*pi*((fs-fp)/f_samp);
        Wt = A2Dfreq((fs-fp));

        M=ceil ((A-8)/(2.285*Wt))+mar;
        L     = M + 1;                      % filter LENGTH
        
        % Force odd length (Type-I)
        if wantOddLength && mod(L,2)==0
            L = L + 1;                      % make length odd
            M = L - 1;
        
        end
        N=M/2;
end



%===============  Function for calculation h[n] for LPF - Kaiser equation
% Indexing :: Theory : -N to N but MATLAB : 0 to 2N (M) . Both length = L


function [hn] = lpf_kaiser(M,N,beta,fp,fs)

       
        % cut off freq
        fc=(fp+fs)/2;
        wc= A2Dfreq(fc); 
    
        m  = 0:M;            % M = 2N
        n  = m - N;          % shift to centered index

     % Ideal LPF   
        hd = zeros(size(m));
        nc = (n ~= 0);
        hd(nc)  = sin(wc * n(nc)) ./ (pi * n(nc));   % for n ≠ 0
        hd(~nc) = wc / pi;                          % for n = 0 (center)
    
     % Kaiser window

        temp1 = ((2*m)/M)-1;
        temp1 = temp1.^2;
        x = beta *sqrt(max(0,1-temp1));
        
        win = zeros(size(m));

        win= (besseli (0,x))/ besseli(0,beta); % index m = 0 to M

        %  ---- Final LP coefficients ----
        hn = hd .* win;                       % windowed ideal response by element wise multiplication
end


% Margins to increase length of the window
mar1=118;
mar2=mar1;
mar3=mar1;
mar4=mar1;

%===================== BSF 1 ==============================
%  BPF_1 = LPF2-LPF1 
%  BSF_1 = 1-BPF_1


% =========== LPF_1=================


[L_lp1,M_lp1,N_lp1, beta_lp1] = Ncal(fs1,fp1,del,mar1);

hn_lpf1 = lpf_kaiser(M_lp1,N_lp1,beta_lp1,fp1,fs1);


% =========== LPF_2=================


% Stop band and passband  to be inverted
[L_lp2,M_lp2,N_lp2, beta_lp2] = Ncal(fp2,fs2,del,mar2);

hn_lpf2 = lpf_kaiser(M_lp2,N_lp2,beta_lp2,fs2,fp2);



% ======= BUILD BPF1 =======

% Ensure hn_lpf1 and hn_lpf2 have same length
L_bsf = max(length(hn_lpf1), length(hn_lpf2));
hn_lpf1 = [hn_lpf1 zeros(1, L_bsf-length(hn_lpf1))];
hn_lpf2 = [hn_lpf2 zeros(1, L_bsf-length(hn_lpf2))];

% BPF1 = LP_high - LP_low  (this is the bandpass occupying the notch)
hn_bpf1 = hn_lpf2 - hn_lpf1;

% Single BSF1 (delta - BPF1) 
center1 = (L_bsf-1)/2;                 % center index (0-based math)
delta1 = zeros(1, L_bsf);
delta1(center1 + 1) = 1;
hn_bsf1 = delta1 - hn_bpf1;            % this is the correct single-notch BSF







fvtool(hn_bsf1);




figure;
freqz(hn_bpf1,1,1024,f_samp);
title('LPF 2 Frequency Response');

figure;
freqz(hn_bsf1,1,1024,f_samp);
title('Bandstop Filter Frequency Response');


%===================== BSF 2 ==============================
%  BPF_2=  LPF4 - LPF3 
%  BSF_2 = 1 - BPF_2


% =========== LPF_3=================


[L_lp3,M_lp3,N_lp3, beta_lp3] = Ncal(fs3,fp3,del,mar3);

hn_lpf3 = lpf_kaiser(M_lp3,N_lp3,beta_lp3,fp3,fs3);


% =========== LPF_4=================


% Stop band and passband  to be inverted
[L_lp4,M_lp4,N_lp4, beta_lp4] = Ncal(fp4,fs4,del,mar4);

hn_lpf4 = lpf_kaiser(M_lp4,N_lp4,beta_lp4,fs4,fp4);

% ======= BUILD BPF2  =======

L_bsf2 = max(length(hn_lpf3), length(hn_lpf4));
hn_lpf3 = [hn_lpf3 zeros(1, L_bsf2-length(hn_lpf3))];
hn_lpf4 = [hn_lpf4 zeros(1, L_bsf2-length(hn_lpf4))];

% BPF2 = LP_high - LP_low
hn_bpf2 = hn_lpf4 - hn_lpf3;

% Optional single BSF2
center2 = (L_bsf2-1)/2;
delta2 = zeros(1, L_bsf2);
delta2(center2 + 1) = 1;
hn_bsf2 = delta2 - hn_bpf2;






fvtool(hn_bsf2);




figure;
freqz(hn_bpf2,1,1024,f_samp);
title('LPF 2 Frequency Response');

figure;
freqz(hn_bsf2,1,1024,f_samp);
title('Bandstop Filter Frequency Response');

% ======= MULTI-BAND COMBINATION =======

% Bring both BPF vectors to common length L_multi
L_multi = max(length(hn_bpf1), length(hn_bpf2));
hn_bpf1 = [hn_bpf1 zeros(1, L_multi - length(hn_bpf1))];
hn_bpf2 = [hn_bpf2 zeros(1, L_multi - length(hn_bpf2))];

% Sum BPFs (this is additive in the frequency domain for passbands)
hn_bpf_multi = hn_bpf1 + hn_bpf2;    % sum of bandpass components

% Single delta at the common center
center_multi = (L_multi - 1)/2;
delta_imp = zeros(1, L_multi);
delta_imp(center_multi + 1) = 1;

% Final multi-notch BSF (single subtraction)
hn_bsf_multi = delta_imp - hn_bpf_multi;

% % Optional: enforce exact symmetry (numerical safety)
% hn_bsf_multi = 0.5*(hn_bsf_multi + fliplr(hn_bsf_multi));



fvtool(hn_bsf_multi);

figure;
[H,f_Hz] = freqz(hn_bsf_multi,1,1024, f_samp);


plot(f_Hz, (abs(H)));
grid on;
title(' (Magnitude Response)');
xlabel('Frequency (Hz)');
ylabel('Magnitude ');


figure;
p = plot(f_Hz, abs(H)); hold on;
grid on;
marks = [fs1 fs2 fp1 fp2 fs3 fs4 fp3 fp4];  % points to mark

for k = 1:length(marks)
    f = marks(k);
    [~, idx] = min(abs(f_Hz - f));           % nearest freq bin
    datatip(p, f_Hz(idx), abs(H(idx)));
end


% Top right: Phase response vs freq (Hz)
figure;
plot(f_Hz, unwrap(angle(H)));
grid on;
title('(Phase Response)');
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');



save('kaiser_multiband_taps.txt', 'hn_bsf_multi', '-ascii');







