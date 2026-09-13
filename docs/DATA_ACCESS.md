# Data access and non-redistribution

This software repository intentionally does not bundle the raw COGITATE M-EEG dataset because the full public release is hundreds of gigabytes and has its own persistent repository and terms of use.

Persistent source records:
- `https://doi.org/10.17617/1.WQA3-WK71` (original-format M-EEG)
- `https://doi.org/10.17617/1.nzwp-7j89` (BIDS M-EEG)

The BIDS data descriptor reports 100 participants, simultaneous MEG/EEG, eye tracking and structural MRI acquired across two centers with four visual categories (faces, objects, letters, false fonts). The SGIT analysis uses derived 102-magnetometer FineTF files. Historical conversion code is included under `pipeline/` for transparent inspection and re-execution.

The manuscript's two 30-participant cohorts are non-overlapping participant sets. The archived Level-6 result explicitly cautions that identifier prefixes should not be treated as acquisition-site labels without joining official metadata.
