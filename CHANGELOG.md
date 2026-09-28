# Changelog

## Preprint v28 - Focused presentation, 2026-09-28

- Move the item-selection derivation and constrained-update proposition to
  appendices, with main-text summaries and cross-references.
- Condense the discussion of amortized neural IRT and remove the generic roadmap.
- Move environment inventories and source identifiers to reproducibility notes;
  keep build and deposit instructions in the source-bundle README.
- Omit fixed-parameter rows from recovery tables, retaining them in the CSVs.
- Preserve the Further Research and Extensions sections, numerical results,
  fitting software, and planned 2,000 replications per condition.

## Preprint v27 - Study presentation, 2026-09-28

- Name the numerical studies by scientific purpose, without manuscript-development
  labels in the article or supplement.
- Add a compact comparison of the fitting configurations, streamline the WILD
  narrative, and present the 2,000-replication plan under Further Research.
- Preserve study-specific denominators, exploratory status, and inference limits.
- Verify unchanged mathematical blocks, numerical tables, and result-file hashes;
  rebuild the PDFs independently from the Overleaf ZIP.

## Preprint v26 - Focused supplement, 2026-09-28

- Reduce the supplement from 35 to 11 pages, retaining the current article's
  parameter-recovery tables, predictive diagnostics, and WILD estimates.
- Remove the development history and earlier-analysis appendices from the
  supplement; preserve them in previous releases.
- Correct the main paper's pointers to supplementary and machine-readable
  results. Keep reported values and the planned 2,000-replication target unchanged.
- Rebuild both PDFs and verify the Overleaf ZIP in a clean directory.

## Preprint v25 - Replication plan, 2026-09-28

- Set the planned study to 2,000 assigned replications per condition across
  24 conditions: 48,000 datasets and 184,000 applicable fitting attempts.
- Update the corresponding maximum binomial Monte Carlo standard error to .011.
- Preserve the completed results, numerical software, and frozen execution
  protocol. This is a documentation update, not a new simulation run.

## 1.4.0rc1 - Preprint release, 2026-09-28

- Publish a focused `LaplaceIRT` API for the 1PL and diagonal/full 2PL.
- Retain the normalized moment-map calculations from frozen version 1.3.1.
- Reject infinite responses, malformed initial values, and invalid indices.
- Raise on a nonfinite posterior step rather than combining estimates from
  different iterations. Clear a previous fit before a refit begins.
- Use 64-bit arithmetic during fitting and preserve DataFrame identifiers.
- Check predictive scoring failures and keep existing output directories intact.
- Add method documentation, a synthetic example, command-line exports,
  numerical regression tests, and saved preprint result tables.

These changes do not rerun or change the preprint's reported analyses.
The former `TransposedIRT` and `ValidatedLaplaceIRT` names are not part of
this narrowed API. Archived diagnostics remain available in the test
reference source; they are not claims of validated parameter inference.
