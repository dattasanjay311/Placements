# Multiband Bandstop Filter Design

## EE 603 --- Digital Signal Processing and its Applications

A MATLAB- and GNU Radio-based study of **multiband bandstop filter
design**, comparing IIR and FIR approaches and validating the resulting
filter coefficients in a practical signal-processing flow graph.

This project was completed as part of **EE 603: Digital Signal
Processing and its Applications** at **IIT Bombay**.

------------------------------------------------------------------------

## Project Overview

The objective of this project was to design a multiband bandstop filter
satisfying specified frequency and tolerance requirements, investigate
different IIR and FIR design techniques, compare their performance and
resource requirements, and verify the designed coefficients using GNU
Radio.

The target sampling frequency was **650 kHz**, with two stopbands:

-   **85--115 kHz**
-   **190--220 kHz**

The corresponding passbands are:

-   **0--80 kHz**
-   **120--185 kHz**
-   **225--325 kHz**

The final design was evaluated for passband and stopband tolerance
requirements.

------------------------------------------------------------------------

## IIR Filter Design

Three classical IIR approximations were studied:

-   **Butterworth**
-   **Chebyshev**
-   **Elliptic**

The filters were designed by starting from analog low-pass prototypes
and applying the required frequency transformations to obtain multiband
bandstop responses.

### Butterworth

Butterworth filters were investigated for their monotonic magnitude
response and relatively smooth transition characteristics.

### Chebyshev

Chebyshev filters were studied for their sharper transition compared
with Butterworth designs, at the cost of passband ripple.

### Elliptic

Elliptic filters provided the sharpest transition among the three IIR
approaches, with ripple in both passband and stopband.

For the final multiband realization, two elliptic bandstop filters were
cascaded. The resulting design had a **total order of 12** and satisfied
the required multiband stopband specifications.

------------------------------------------------------------------------

## FIR Filter Design

FIR filters were designed using window-based techniques.

The following windows were investigated:

-   Kaiser
-   Hamming
-   Triangular (Bartlett)
-   Tukey
-   Lanczos
-   Blackman

The **Kaiser window** was studied in greater detail by varying its shape
parameter β to examine the trade-off between ripple, transition
sharpness, and number of coefficients.

### FIR Comparison

  ------------------------------------------------------------------------
  Window                      Number of Coefficients Observation
  --------------------- ---------------------------- ---------------------
  Kaiser (β = 0)                                 109 Ripples in passband
                                                     and stopband

  Kaiser (β = 2)                                 129 Reduced ripple
                                                     compared with β = 0

  Kaiser (β = 5)                                 197 Smoother response
                                                     with a less sharp
                                                     transition

  Hamming                                        199 Smooth and
                                                     well-attenuated
                                                     response

  Triangular                                     199 Smooth response but
                                                     gradual transition

  Tukey                                          129 Trade-off between
                                                     rectangular and
                                                     tapered behavior

  Lanczos                                        129 Better tolerance than
                                                     rectangular-like
                                                     response

  Blackman                                       229 Low ripple, but did
                                                     not meet the required
                                                     stopband tolerance
                                                     for the given
                                                     transition bandwidth
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## IIR vs FIR

The project also compares the two major filter classes.

### IIR

-   Recursive implementation
-   Requires fewer coefficients
-   Sharper transitions for lower filter orders
-   Computationally efficient
-   Generally has nonlinear phase

### FIR

-   Non-recursive implementation
-   Can provide linear phase
-   Numerically stable
-   Usually requires more coefficients
-   Suitable when phase accuracy is important

The study highlights the trade-off between **computational efficiency
and phase characteristics** when choosing between IIR and FIR filters.

------------------------------------------------------------------------

## GNU Radio Verification

The designed FIR coefficients were integrated into **GNU Radio** to
verify their practical behavior.

Two experiments were performed:

1.  A noise signal was generated and passed through the FIR bandstop
    filter.
2.  An audio signal was intentionally contaminated with noise,
    upconverted so that the interference fell within the filter's
    stopband, and then passed through the designed FIR filter.

The filtered signal was subsequently downconverted and analyzed in the
frequency domain.

The GNU Radio results showed that the unwanted spectral components were
significantly suppressed, providing practical verification of the
MATLAB-designed filter coefficients.

------------------------------------------------------------------------

## Repository Structure

``` text
.
├── MATLAB Codes/
│   ├── FIR_Blackman_Group_5.m
│   ├── FIR_Hamming_Group_5.m
│   ├── FIR_Kaiser_Group_5.m
│   ├── FIR_Tukey_Group_5.m
│   ├── FIR__Lanczos_Group_5.m
│   ├── FIR__Triangle_Group_5.m
│   ├── IIR_Butterworth_Group_5.m
│   ├── IIR_Cheybshev_Group_5.m
│   └── IIR_Elliptic_Group_5.m
│
├── GNU Radio files/
│   ├── music_test.grc
│   └── noise.grc
│
├── Group5_EE603_ExtraordinaryEfforts.pptx
└── EE603_Extraordinary_Efforts_Filter_Design_Group5_Finalreport.pdf
```

------------------------------------------------------------------------

## Tools Used

-   **MATLAB** --- filter design, coefficient generation,
    frequency-response analysis, and comparison
-   **GNU Radio** --- practical verification of FIR filter coefficients
    and signal denoising
-   **Python/C++** --- not used in the submitted implementation

------------------------------------------------------------------------

## Key Takeaways

-   Designed multiband bandstop filters using both **IIR and FIR**
    techniques.
-   Compared Butterworth, Chebyshev, and Elliptic IIR filters.
-   Investigated multiple FIR window functions and their
    coefficient/ripple trade-offs.
-   Studied the effect of the Kaiser window parameter on filter
    performance.
-   Compared filter order, number of coefficients, magnitude response,
    phase response, and resource requirements.
-   Verified MATLAB-generated FIR coefficients in **GNU Radio**.
-   Demonstrated practical suppression of unwanted spectral components
    from a noisy audio signal.

------------------------------------------------------------------------

-------------------------- -------------
  D Ravi Teja                25M1064
  **Powrohitham Sanjay D**   **25M1071**
  Nikitha P R                25M1076
  Aashita Shyam              25M1088
  Durga Prasad               25M1160
  G Siddharth Reddy          25M1276

------------------------------------------------------------------------

## Course

**EE 603 --- Digital Signal Processing and its Applications**\
**Indian Institute of Technology Bombay**\
**Group 5**
