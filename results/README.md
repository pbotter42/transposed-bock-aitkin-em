# Preprint Reporting Snapshot

These 32 CSV/JSON files accompany the September 2026 preprint. They are
unchanged copies of the manuscript's reporting snapshot; `manifest.json`
records their SHA-256 hashes. No new fits were used to prepare this release.

- Top-level simulation files describe the historical 16-condition study,
  with 100 assigned response matrices per condition. Assignment and
  completion do not mean every fit converged. See `outcomes.csv`,
  `outcomes_by_cell.csv`, and `metric_availability.csv` before interpreting
  the conditional performance and recovery summaries.
- `revised_snapshot/` describes the separate 69-matrix numerical diagnostic
  study. Its records must not be pooled with the historical simulation.
- Top-level `wild_*.csv` files contain the historical WILD application
  summaries, capability estimates, and predictive bootstrap contrasts.
- `validation_wild/` contains the separately reported revised WILD fits
  and their predictive comparisons.

`theta_recovery.csv` concerns capability estimates; `population_recovery.csv`
and `population_errors.csv` concern item-population parameters. Some original
item-level recovery vectors were not retained. Missing metrics are not
zeros and cannot be reconstructed from a successful convergence label.
Bootstrap intervals for predictive contrasts are not parameter intervals.

The CSV headers preserve the names used in the reporting pipeline, including
legacy method labels. Interpret those labels using the preprint and its
supplement rather than treating them as names of the current package API.
