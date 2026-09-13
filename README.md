# Reproducible radial-angular geometry of visual category representations in human MEG

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22734855.svg)](https://doi.org/10.5281/zenodo.22734855)

This repository accompanies the manuscript **“Reproducible Radial-Angular Geometry of Visual Category Representations in Human MEG: Phase Organization Beyond Amplitude Structure”** (Miroslav Svítek; prepared for *Imaging Neuroscience*).

## Scientific purpose

The repository supports the finding that a complex early-alpha MEG representation contains two complementary sources of repeatable visual-category structure:

- a **radial amplitude scaffold**, including highly reproducible amplitude-product structure; and
- **category-specific angular phase organization**, demonstrated by amplitude-free ANGLE reliability and by amplitude-preserving category-phase null tests.

The central exact identities are

```text
FULL = AMP + AP = ENERGY + INTERFERENCE
AMP  = ENERGY - PRODUCT
AP   = PRODUCT + INTERFERENCE
```

with `ANGLE = sum 2[1-cos(delta_phi)]` used as an amplitude-free phase diagnostic. These algebraic identities are not themselves the discovery; the empirical result concerns cross-run representational reliability and phase-specific null tests.

## Public source data

The source COGITATE MEG/EEG dataset is publicly archived by the Max Planck Society:

- original-format data: DOI `10.17617/1.WQA3-WK71`
- BIDS-formatted data: DOI `10.17617/1.nzwp-7j89`
- data descriptor: Liu et al. (2026), *Scientific Data*, DOI `10.1038/s41597-026-07350-9`

The large neuroimaging dataset is **not redistributed** in this repository.

## Quick audit without neural data

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
python verify_manifest.py
python verify_archive.py
python reproduce_article.py
```

The audit verifies cross-stage numerical consistency and regenerates article figures from archived compact outputs.

## Analysis sequence

Public-facing scripts are in `analysis/`:

1. `01_primary_confirmatory_reliability.py`
2. `02_robustness_and_negative_controls.py`
3. `03_radial_angular_component_decomposition.py`
4. `04_amplitude_preserving_phase_null.py`

Historical filenames and frozen protocol snapshots are retained for provenance. See `analysis/README.md` and `docs/PROVENANCE.md`.

## Repository layout

- `analysis/` - public-facing and archived analysis scripts
- `results/` - compact numerical outputs used in the manuscript
- `tests/` - numerical self-tests
- `protocols/` and `reanalysis_packages/` - frozen analysis snapshots and templates
- `pipeline/` - upstream COGITATE-to-FineTF conversion components
- `article_source/` - manuscript v6 source, figures, PDFs, and supplement
- `docs/` - data access, provenance, and interpretation guardrails
- `figures/` - regenerated manuscript figures

## Interpretation boundary

The repository supports a repeatable **ensemble-level radial-angular representational geometry** in a defined early-alpha MEG signal subspace. It does not establish a clinical biomarker, causal communication mechanism, physiological phase-amplitude coupling, source-space anatomy, a single-trial decoder, or consciousness specificity.

## Clinical relevance

The immediate implication for clinical neurophysiology is methodological: candidate oscillatory biomarkers should be benchmarked against power-only baselines because amplitude reduction can discard repeatable angular information. Translation requires patient cohorts, source-space validation, test-retest studies, and independent replication.

## Citation and archival release

The archived reproducibility package is available on Zenodo:

**Version 1.1.0 DOI:** [10.5281/zenodo.22734855](https://doi.org/10.5281/zenodo.22734855)

Recommended software citation:

> Svítek, M. (2026). *Radial-Angular MEG Representational Geometry: Reproducibility Code and Archived Results* (Version 1.1.0). Zenodo. https://doi.org/10.5281/zenodo.22734855

A machine-readable citation is provided in `CITATION.cff`.

## License

Software code is released under the MIT License. Textual documentation, figures, and other non-code materials are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), unless otherwise stated. Source COGITATE data remain subject to the terms of their original repositories.
