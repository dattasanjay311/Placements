% run_all.m -- regenerates the sample data and all five result figures.
% Usage (from the scripts/ folder):  octave run_all.m
run('generate_sample_ppg_data.m');
run('01_channel_aoi_simulation.m');
run('02_delta_modulation.m');
run('03_adaptive_delta_modulation.m');
run('04_dct_compression.m');
run('05_reconstruction_comparison.m');
disp('All results regenerated in ../results/');
