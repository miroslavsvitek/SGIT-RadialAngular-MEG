# SGIT PAPER03 Phase-Organization Null — Frozen Protocol v1.0

**Frozen status:** frozen after inspection of PAPER02 and before inspection of any PAPER03 phase-null outcome.

## Scientific purpose

PAPER02 established that both the amplitude-product scaffold and the phase-dependent components of the exact complex distance are reproducible in the same two non-overlapping 30-participant cohorts. PAPER03 asks the remaining falsification question: **is the observed phase-dependent reproducibility greater than expected when category-specific phase assignment is destroyed while amplitudes are held exactly fixed?**

Only two pre-specified components are tested:

\[
D_{\mathrm{INT}}(i,j)=-2\sum_m A_{i,m}A_{j,m}\cos(\phi_{i,m}-\phi_{j,m}),
\]

\[
D_{\mathrm{ANGLE}}(i,j)=2\sum_m\left[1-\cos(\phi_{i,m}-\phi_{j,m})\right].
\]

The first is amplitude-weighted phase interference; the second removes amplitude weights and tests angular organization alone.

## Frozen neural representation

No neural parameter is changed relative to the confirmed Level-5A / strict-template Level-6A pipeline and PAPER02:

- same CA Level-5A cohort, N=30;
- same CB strict-template Level-6A cohort, N=30;
- five runs;
- 102-magnetometer graph template compressed to frozen K=16 graph modes;
- theta 4–8 Hz and alpha 8–13 Hz analytic signals;
- post-stimulus 50–250 ms window;
- rank-2 category-independent common-subspace removal;
- identical modal power normalization;
- all 15 unique disjoint 2-vs-2 run partitions;
- primary RDM association: Spearman correlation.

PAPER03 must numerically reproduce the PAPER02 participant-level INTERFERENCE and ANGLE scores before the null results are accepted. The archived PAPER02 participant values are included as a read-only reference table.

## Phase-randomization null

For each participant, run partition, and compared half separately:

1. construct the frozen complex category field \(Y_c(m)=A_c(m)e^{i\phi_c(m)}\);
2. at each coordinate \(m\), keep the four amplitudes \(A_c(m)\) exactly attached to their original categories;
3. randomly permute the four category phase labels \(\{\phi_c(m)\}\) independently at that coordinate;
4. use independent random permutations for the two compared halves;
5. recompute INTERFERENCE and ANGLE RDMs and their cross-half Spearman correlation.

This null is deliberately the same category-phase-label logic used in PAPER01 C1, but PAPER03 applies it directly to the two phase-specific components rather than to AP/FULL.

**Number of null realizations:** 999 per participant.  
**Frozen seed:** 2026091211 plus a stable participant-specific hash.  
**No alternate shuffle is searched if the primary null fails.**

## Population inference

For each cohort and component separately:

1. observed cohort mean reliability is the arithmetic mean of the 30 observed participant scores;
2. the population null distribution contains 999 cohort means, one for each aligned null replicate across participants;
3. empirical p uses the plus-one correction:

\[
p_{\rm emp}=\frac{1+\#\{R^{(b)}_{\rm null}\ge R_{\rm obs}\}}{999+1};
\]

4. independently, each participant's observed score is compared with that participant's mean null score; the population mean of these paired differences is tested by a one-sided subject-level sign-flip test with 100,000 realizations;
5. both criteria must satisfy p < 0.025 and the paired mean effect must be positive.

## Frozen decision

- **PHASE_ORGANIZATION_CONFIRMED:** both INTERFERENCE and ANGLE pass in both cohorts.
- **WEIGHTED_PHASE_ORGANIZATION_ONLY:** INTERFERENCE passes in both cohorts, ANGLE does not.
- **ANGULAR_PHASE_ORGANIZATION_ONLY:** ANGLE passes in both cohorts, INTERFERENCE does not.
- **PHASE_ORGANIZATION_NOT_CONFIRMED:** neither endpoint passes in both cohorts.

No endpoint substitution or threshold adaptation is permitted after outcomes are inspected.

## Interpretation limits

A PASS would support the claim that the observed category geometry contains reproducible category-specific phase organization beyond an amplitude-preserving category-phase permutation null. It would **not** prove physiological PAC, a universal phase code, causality, single-trial invariance, or consciousness specificity. Because PAPER03 reuses cohorts already analyzed in earlier SGIT levels, it is a targeted frozen falsification analysis rather than a third independent population replication.
