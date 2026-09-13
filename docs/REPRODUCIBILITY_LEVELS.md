# Reproducibility levels

## Level A - manuscript audit without neural data

Run:

```bash
python verify_archive.py
python reproduce_article.py
```

This verifies cross-stage numerical consistency and regenerates the manuscript figures/tables from the archived participant/cohort outputs. It does **not** recompute neural signals.

## Level B - exact PAPER01-03 reanalysis from FineTF derivatives

Provide the frozen CA and CB FineTF roots and run the three scripts in `src/` against `reanalysis_packages/PAPER01`, `PAPER02`, and `PAPER03`. These packages contain the exact graph template/configuration/reference snapshots expected by the runners.

PAPER01-03 intentionally reuse the same two participant cohorts. They are frozen falsification/mechanistic analyses, not additional population acquisitions.

## Level C - upstream conversion from public COGITATE data

The public raw/BIDS data are not redistributed. Historical converter components are in `pipeline/`. Upstream reproduction requires the public COGITATE resource, its metadata, MNE/MNE-BIDS, and the same frozen graph/preprocessing definitions. Dataset DOIs are given in `README.md` and `docs/DATA_ACCESS.md`.

Because the dataset is large and remote repository structures can change, this archive does not hard-code a browser download URL. The manuscript's numerical claims can be audited at Level A independently of data download, and the manuscript analyses can be rerun at Level B when the FineTF derivatives are available.
