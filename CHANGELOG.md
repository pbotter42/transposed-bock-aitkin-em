# Changelog

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
