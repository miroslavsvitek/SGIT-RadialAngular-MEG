# SGIT PAPER02 — Frozen interference decomposition

**Frozen before PAPER02 component outcomes are inspected.**

## Question
The previous reviewer-control run showed that AP > AMP is robust to basis changes and metrics, but an amplitude-preserving phase shuffle did not reduce AP reliability. PAPER02 therefore asks what part of the exact complex distance is reproducible: a multiplicative amplitude scaffold, a phase-specific interference term, or both.

For two complex coordinates

`z_i = A_i exp(i phi_i)`, `z_j = A_j exp(i phi_j)`,

define, after summing over all frozen alpha-field coordinates `m`,

- `ENERGY = sum(A_i^2 + A_j^2)`
- `PRODUCT = sum(2 A_i A_j)`
- `COSINE = sum(2 A_i A_j cos(delta phi))`
- `INTERFERENCE = -COSINE`
- `ANGLE = sum(2 [1-cos(delta phi)])` (diagnostic phase-only geometry)

The exact audit identities are

- `AMP = ENERGY - PRODUCT`
- `AP = PRODUCT + INTERFERENCE`
- `FULL = ENERGY + INTERFERENCE`

The scientific novelty is **not** these identities. The empirical question is which component geometry is reproducible across independent run ensembles and across the two frozen participant cohorts.

## Frozen neural pipeline
Identical to Level-5A / Level-6A and PAPER01: 102 magnetometers projected to the frozen first 16 graph modes; theta 4–8 Hz and alpha 8–13 Hz analytic signals; 50–250 ms; rank-2 category-independent common-subspace removal; alpha power normalization; four categories; all 15 disjoint 2-vs-2 run partitions.

## Endpoints
For every participant and component, the primary reliability is mean Spearman correlation between the six category-pair values from the two independent run halves across the 15 partitions.

Primary phase-specific evidence requires **both cohorts** to satisfy:
1. mean participant INTERFERENCE reliability > 0, one-sided subject sign-flip `p < 0.025`; and
2. symmetric cross-half INTERFERENCE-to-FULL congruence > 0, one-sided subject sign-flip `p < 0.025`.

PRODUCT reliability > 0 with the same criterion in both cohorts establishes a reproducible multiplicative amplitude scaffold.

No effect-size floor is introduced for the new components because no independent prior magnitude estimate exists. Bootstrap 95% CIs are descriptive. Pearson and Kendall are robustness metrics only.

## Decision table
- **MIXED_PHASE_AND_PRODUCT**: phase-specific criterion passes in both cohorts and PRODUCT passes in both cohorts.
- **PHASE_SPECIFIC_WITHOUT_PRODUCT_DOMINANCE**: phase-specific criterion passes in both cohorts but PRODUCT fails in at least one.
- **PRODUCT_SCAFFOLD_DOMINANT**: PRODUCT passes in both cohorts but the phase-specific criterion fails.
- **INDETERMINATE**: otherwise.

## Guardrail
These are new frozen endpoints on previously used datasets, not a new independent acquisition. Any positive phase-specific result should still be independently replicated later if used as a central mechanistic claim.
