% generate_sample_ppg_data.m
%
% The original scripts in this project were written against the BIDMC PPG
% and Respiration Database (record bidmc_05), hosted on PhysioNet:
%   https://physionet.org/content/bidmc/1.0.0/
% That file is not redistributed here (PhysioNet asks that people download
% it themselves), so this script synthesizes a stand-in signal with the
% same shape/format the rest of the code expects:
%   - data/sample_Signals.csv   : Time, RESP, PLETH   (125 Hz, 480 s)
%   - data/sample_Numerics.csv  : Time, HR             (1 Hz,   480 s)
%
% Swap these files for the real bidmc_05_Signals.csv / bidmc_05_Numerics.csv
% to reproduce the thesis's actual reported numbers -- just point the
% csvFilePath variable at the top of each script to the real files.

rand('seed', 42); randn('seed', 42);

fs = 125;                 % Hz, matches BIDMC sampling rate
duration_s = 480;         % 8 minutes, matches BIDMC recording length
n = fs * duration_s;
t = (1:n)';                % sample index, mirrors how 'Time' is used as an index in the original code

% --- synthetic PPG (PLETH) -------------------------------------------------
hr_bpm = 72 + 4*sin(2*pi*t/(fs*60));          % slow heart-rate drift
hr_hz = hr_bpm/60;
phase = cumsum(2*pi*hr_hz/fs);
resp_hz = 0.25;                                % ~15 breaths/min
pleth = sin(phase) + 0.25*sin(2*phase) ...     % dicrotic-notch-like harmonic
        + 0.15*sin(2*pi*resp_hz*t/fs) ...       % respiratory modulation
        + 0.03*randn(n,1);
resp = sin(2*pi*resp_hz*t/fs) + 0.05*randn(n,1);

fid = fopen('../data/sample_Signals.csv', 'w');
fprintf(fid, 'Time,RESP,PLETH\n');
fclose(fid);
dlmwrite('../data/sample_Signals.csv', [t resp pleth], '-append', 'precision', '%.6f');

% --- synthetic numerics (e.g. heart rate trend, 1 Hz) ----------------------
tn = (1:duration_s)';
hr = 72 + 4*sin(2*pi*tn/60) + 0.5*randn(duration_s,1);

fid = fopen('../data/sample_Numerics.csv', 'w');
fprintf(fid, 'Time,HR\n');
fclose(fid);
dlmwrite('../data/sample_Numerics.csv', [tn hr], '-append', 'precision', '%.6f');

printf('Wrote data/sample_Signals.csv (%d rows) and data/sample_Numerics.csv (%d rows)\n', n, duration_s);
