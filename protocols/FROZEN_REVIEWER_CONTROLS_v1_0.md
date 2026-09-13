# SGIT manuscript reviewer controls — frozen plan v1.0

## Purpose
This suite targets the main methodological objections raised against the Imaging Neuroscience manuscript while leaving the already confirmed Level-5A / Level-6A representation unchanged.

The empirical discovery under test is **not** the algebraic identity itself. The testable claim is that the amplitude-weighted angular term (AP) preserves cross-run visual-category geometry better than amplitude differences (AMP), and that this asymmetry is reproducible.

## Status and evidential class
The Level-5A and Level-6A primary outcomes are already known. Therefore the analyses below are **post-confirmatory robustness and falsification controls frozen before their own outcomes are inspected**. They must not be described as a new prospective confirmation cohort.

## Frozen common pipeline
- 102-magnetometer COGITATE-derived FineTF inputs represented by the frozen first K=16 graph modes.
- Five runs, four categories: face, object, letter, false-font.
- Temporal theta 4–8 Hz and alpha 8–13 Hz, fourth-order Butterworth SOS, zero-phase filtering, Hilbert analytic signal.
- Primary poststimulus window 50–250 ms.
- Rank-2 category-independent common-subspace removal in the joint theta+alpha feature space.
- Per-coordinate alpha power normalization.
- Exact squared-distance decomposition
  D_FULL = D_AMP + D_AP,
  where D_AMP is radial/amplitude difference and D_AP = 2 A A' [1-cos(delta phi)] summed over coordinates.
- Fifteen unique disjoint 2-vs-2 run partitions and participant-level mean RDM reliability.

## C1 — amplitude-preserving phase null
For each partition and each half-field separately, amplitudes are held **exactly** fixed. At every (mode,time) coordinate, the four category phases are randomly permuted, independently between the two halves. This preserves the category amplitudes and the coordinate-wise phase multiset while destroying category-specific phase assignment and cross-half phase organization.

Frozen B=199, seed 2026091201.

Primary diagnostics:
1. observed AP reliability versus the phase-null population distribution;
2. observed FULL reliability versus the phase-null population distribution;
3. AP−AMP before versus after phase randomization;
4. numerical audit that amplitudes are preserved to floating-point tolerance.

## C2 — basis generality rather than basis privilege
Three controls test whether AP>AMP is an artifact of a particular graph coordinate system.

### C2a fixed random rotations
Forty-nine deterministic random orthogonal KxK rotations of the same 16-dimensional subspace are applied before the frozen residualization/power-normalization steps. The same rotation is used for all subjects and runs within a replicate. We report the distribution of population AP−AMP and FULL reliability across rotations. This asks whether the phase-relational asymmetry survives alternative coordinates; it does **not** require the GFT basis to be uniquely optimal.

### C2b out-of-bag PCA
For each 2-vs-2 partition, the one unused run is used only to fit a KxK PCA rotation. That rotation is then applied unchanged to both evaluated halves. Category labels are not used to fit PCA.

### C2c rank-matched sensor-coordinate reconstruction
The frozen K=16 graph coefficients are mapped back through the frozen graph eigenvectors to the 102 sensor coordinates. The same residualization and power normalization are then applied in sensor coordinates. Because the FineTF archive contains only the first 16 graph modes, this is explicitly a **rank-matched reconstructed sensor-coordinate baseline**, not a full raw-sensor or source-space analysis.

## C3 — far-prestimulus negative control
The identity-basis pipeline is repeated in the matched 200-ms window -450 to -250 ms. The key statistic is the paired participant-level difference between poststimulus and prestimulus FULL/AP reliability. This tests whether the reported geometry is simply a stable baseline property or a zero-phase-filter leakage artifact near stimulus onset.

## C4 — metric robustness
The same six-entry RDMs are compared under Spearman rho (historical primary), Pearson r, and Kendall tau-b. This addresses the discreteness/volatility of a six-distance RDM without altering the neural representation.

## Cohorts
Exactly the archived frozen participant lists are used: 30 Level-5A CA participants and 30 Level-6A strict-template CB participants. No participant is added or replaced based on the new control outcomes.

## Interpretation
A strong result would be a convergent pattern in both cohorts:
- observed AP and FULL exceed the amplitude-preserving phase null;
- AP>AMP persists in OOB-PCA and rank-matched sensor coordinates and across most random rotations;
- poststimulus reliability exceeds far-prestimulus reliability;
- AP>AMP is not dependent on Spearman alone.

A failure of any control is scientifically informative and must be reported. No failed control may be rescued by changing the frozen bands, windows, ranks, seeds, participant lists, or distance definitions.
