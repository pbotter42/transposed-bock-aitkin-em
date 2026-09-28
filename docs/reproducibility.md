# What This Release Reproduces

This is preprint software, version `1.4.0rc1`, not a final journal release.
Its public interface is limited to the normalized Laplace moment map for
the 1PL and diagonal/full 2PL. It does not expose the archived AGHQ,
constrained-map, or parameter-inference interfaces.

Three records must be distinguished:

| Record | Status and relationship to this repository |
| --- | --- |
| Simulation comparison | 100 assigned replications in each of 16 conditions; saved summaries in `results/`. Its fitting implementation and stopping rule differ from this public package. |
| WILD application | Seven candidate fits to 65 models and 98,032 retained items, with 2,000 item-cluster predictive bootstrap resamples per contrast. The TBAEM fits use frozen package 1.3.1. |
| Planned confirmation study | 2,000 assigned replications per condition in 24 conditions: 48,000 matrices and 184,000 applicable fits; future work, not results supplied by this release. |

The planned target does not change an existing frozen run or its replication
ledger. The numerical software and retained results are unchanged.

`tests/reference_v1.3.1.zip` is the unchanged source of the diagnostic package,
included for regression tests and provenance, not installed as the public
API. `tests/reference_manifest.json` records each archived source file's
SHA-256. The archive contains older interfaces as well as the normalized
map; the regression tests invoke only the latter. For deterministic small
matrices, the tests compare capability estimates, item modes, posterior
nodes, covariance, objective values, and iteration traces for all three
model specifications. Agreement is checked to `1e-10` in these tests; it is
not a claim of identical output on every platform or every dataset.

Version 1.3.2 changed diagnostic labels. This release additionally narrows
the API, validates inputs, fixes numerical-failure handling, preserves IDs,
and scopes 64-bit arithmetic to fitting. It does not replace saved outcomes
with newly computed estimates or retroactively reclassify failed fits.
The reported population-recovery and capability-recovery summaries are
unchanged. See [CHANGELOG.md](../CHANGELOG.md).

The result CSVs are copies of the preprint reporting snapshot. Their hashes
are recorded in `results/manifest.json`; `results/README.md` describes the
groups. They are aggregate and selected estimate exports, not the complete
run archive. WILD response data, historical missing parameter vectors,
full bootstrap draws, and machine-specific simulation supervisors are not
redistributed here. The study-specific environments are recorded below.
This repository is therefore **not a one-command
reproduction of every table from raw data**. It supplies the estimator,
testable source lineage, saved reporting tables, and the paper's buildable
source. No production simulations were rerun to prepare this release.

## Computational Environments

These are the environments used for the reported analyses, not a claim
that installing the current package recreates every fitted model.

| Component | 100-replication comparison | WILD application |
| --- | --- | --- |
| Common environment | Python 3.11; NumPy 1.26.4; SciPy 1.13.1; pandas 2.3.3; JAX 0.4.30; PyTorch 2.2.2; py-irt 0.7.1 | Common PyTorch 2.2.2 environment |
| Transposed estimator | Archived local package source; not the current public release | Frozen package 1.3.1 |
| `torch_measure` | Isolated adapter, development label 0.1.dev1, PyTorch 2.13.0 | Development label 0.1.1.dev23, PyTorch 2.2.2 |

Both `torch_measure` adapters use source commit
`a461a8f24eb2960f692afdac0a2202c7d1d507ad`. The differing installed development
labels do not denote different source commits. The paper's simulation and
application sections specify their stopping rules; an environment version
alone does not establish estimator equivalence.

## Manuscript Source

The Overleaf ZIP in the GitHub release compiles the main paper and
supplement independently without running a model. The `preprint-v29` bundle
contains only the 100-replication simulation and WILD application result
files. The separate 69-dataset assessment is no longer part of the article;
its unmodified records remain in `results/revised_snapshot/` for provenance
and must not be pooled with the reported simulation. Previous releases
remain available. No saved estimates or fitting code changed.
The GitHub tag fixes this software snapshot;
it is not a DOI or a claim that the preprint has been accepted by a journal.
