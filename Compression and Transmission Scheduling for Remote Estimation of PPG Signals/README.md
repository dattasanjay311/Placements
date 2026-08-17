# AoI-based Adaptive Compression & Transmission Scheduling for PPG Signals

This folder is a cleaned-up, runnable subset of the MATLAB code from the B.Tech
thesis **"Age of Information based Adaptive compression and transmission
scheduling for remote estimation of PPG signals"** (Powrohitham Sanjay Datta,
IIST Thiruvananthapuram, 2024, supervised by Dr. Vineeth B.S.).

The original `matlab codes/` folder had ~24 scripts, including scratch files
(`untitled*.m`), duplicates, and hardcoded Windows paths (`D:\project\...`)
pointing at the BIDMC PPG dataset. This redo keeps the five techniques that
the thesis actually evaluates, fixes the bugs listed below, makes every path
relative, and re-runs everything to produce fresh result figures.

## What the project is about

Photoplethysmography (PPG) sensors stream data faster than it's often
practical to transmit. The thesis studies two questions:
1. **How much can you compress a PPG waveform** before reconstruction quality
   (MSE) suffers too much? (delta modulation, adaptive delta modulation, DCT)
2. **Given a lossy/rate-limited channel** (modeled as a Bernoulli
   arrival/service queue), **how should you reconstruct the signal** at the
   receiver — in real time, or by using knowledge of later packets? (Age of
   Information framing; online vs. offline vs. partial-offline reconstruction)

## Folder structure

```
redo/
  data/                       synthetic PPG data used by all scripts (see note below)
  scripts/
    generate_sample_ppg_data.m    creates data/sample_Signals.csv and sample_Numerics.csv
    01_channel_aoi_simulation.m   AoI/MSE vs arrival rate, constant vs time-varying service
    02_delta_modulation.m         linear delta modulation, MSE vs step size
    03_adaptive_delta_modulation.m adaptive delta modulation
    04_dct_compression.m          windowed DCT compression, MSE & compression ratio vs energy threshold
    05_reconstruction_comparison.m online vs offline vs partial-offline reconstruction (core thesis result)
    fill_with_last_received.m     helper used by script 05
    run_all.m                     regenerates data + all figures
  results/                     PNG figures produced by the scripts above (already generated)
```

## About the data — please read

The original scripts load `bidmc_05_Signals.csv` / `bidmc_05_Numerics.csv`
from the **BIDMC PPG and Respiration Database** on PhysioNet
(https://physionet.org/content/bidmc/1.0.0/). That file isn't included in
this repo (PhysioNet's terms ask users to download it themselves, and it
wasn't in the uploaded project archive either), so `generate_sample_ppg_data.m`
synthesizes a stand-in signal with the same shape and sampling format:
- 125 Hz, 8-minute PPG-like waveform (heart-beat-shaped pulses + respiratory
  modulation + noise) → `data/sample_Signals.csv`
- 1 Hz, 480-sample slowly-varying trend (stands in for the Numerics file) →
  `data/sample_Numerics.csv`

**To reproduce the thesis's actual reported numbers**, download
`bidmc_05_Signals.csv` and `bidmc_05_Numerics.csv` from PhysioNet, drop them
into `data/`, and point the `csvFilePath`/`csvread(...)` lines at the top of
each script to those files instead.

## How to run

Requires [GNU Octave](https://octave.org/) (tested on 8.4) with the
`signal` package (`pkg install -forge signal`, or `apt install octave-signal`
on Ubuntu/Debian) for `dct`/`idct` in script 04.

```
cd scripts
octave run_all.m
```

This regenerates `data/sample_*.csv` and all PNGs in `results/`. Each script
can also be run individually, e.g. `octave 05_reconstruction_comparison.m`.

## What each result shows

| File | Shows |
|---|---|
| `01_channel_aoi_simulation.png` | MSE and mean Age of Information vs. arrival rate, for a fixed vs. a time-varying service rate. MSE/AoI blow up as arrival rate approaches the service rate — classic queueing instability. |
| `02a_delta_modulation_waveforms.png` | A PPG segment, its 1-bit delta-modulated bitstream, and the reconstruction. |
| `02b_delta_modulation_mse_vs_delta.png` | Reconstruction MSE vs. step size — shows the granular-noise (small delta) vs. slope-overload (large delta) trade-off. |
| `03_adaptive_delta_modulation.png` | Same idea as (02) but with an adaptive step size — reaches a lower MSE than fixed-delta DM at a comparable step size. |
| `04a/04b_dct_compression*.png` | Windowed DCT compression: waveform reconstruction at one energy threshold, and MSE/compression-ratio swept across thresholds. |
| `05_reconstruction_comparison.png` | **Core result**: MSE vs. arrival rate for online (real-time hold-last-value), offline (full-record best fill), and partial-offline (small look-back buffer) reconstruction. Offline/partial-offline clearly beat online, especially as arrival rate rises and the online estimator becomes prone to holding stale values for long, unlucky stretches. |

## Bugs fixed from the original scripts

- **Hardcoded absolute Windows paths** (`D:\project\bidmc_csv\...`) replaced
  with relative paths into `data/`.
- **`delta_modulation.m`** read column 2 (RESP) instead of column 3 (PLETH)
  of the signals file — likely an oversight; fixed to use the PPG channel
  consistently across all scripts.
- **`delta_modulation.m`**'s MSE-vs-delta sweep never reset its accumulator
  arrays (`sqc`, `rc`) between iterations, so every delta's MSE was
  contaminated by all previous iterations. Fixed by resetting state per
  iteration.
- **`ppg_signal_analysis.m`** mixed a 125 Hz signals-file time base with a
  hardcoded `480`-second end time that actually describes the 1 Hz numerics
  file. Script 05 here uses the 1 Hz series explicitly, matching what the
  480-second assumption was written for.
- Removed dead/scratch code (`untitled*.m`, `quantization.m`'s reference to
  an undefined `amplitude_column`, and duplicate near-identical scripts like
  `dct_paper.m` / `partial_dct_paper.m` / `mse_for_three_types_of_reconstructions.m`,
  which are all variations on `dct_my.m` / `partial_dct_paper_varying_energy_threshold.m`
  / `ppg_signal_analysis.m` already covered above).
- Simulation lengths in script 01 were reduced from the original 40,000
  time-steps × 40 arrival-rate points to 10,000 × 20 for faster iteration;
  bump these back up if you want a smoother curve.

## Not reproduced here

The uploaded archive also contained `final_presentation.pdf`, the thesis
report (`sc20b107.pdf`), an IIST thesis LaTeX template, and a large
`results/` folder of pre-generated `.fig`/`.png` plots from the original
(real-data) runs — those are left as-is; this redo only concerns the
MATLAB/Octave code and produces fresh figures from synthetic data.
