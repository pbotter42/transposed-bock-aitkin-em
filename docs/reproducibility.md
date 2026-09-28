# What This Release Reproduces

This is preprint software, version `1.4.0rc1`, not a final journal release.
Its public interface is limited to the normalized Laplace moment map for
the 1PL and diagonal/full 2PL. It does not expose the archived AGHQ,
constrained-map, or parameter-inference interfaces.

Three records must be distinguished:

| Record | Status and relationship to this repository |
| --- | --- |
| Historical simulation | 100 assigned replications in each of 16 conditions; saved summaries in `results/`. It used an earlier implementation and stopping rule, not this public package. |
| Revised numerical diagnostic | 69 matrices, 256 applicable fits, frozen package 1.3.1; reported separately in the preprint. The new public estimator retains its normalized fitting map. |
| Planned confirmation study | 200 assigned replications in 24 conditions; future work, not results supplied by this release. |

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
redistributed here. The older study's exact environment and adapters are
described in the paper. This repository is therefore **not a one-command
reproduction of every table from raw data**. It supplies the estimator,
testable source lineage, saved reporting tables, and the paper's buildable
source. No production simulations were rerun to prepare this release.

The Overleaf ZIP in the GitHub release compiles the main paper and
supplement independently. The GitHub tag fixes this software snapshot;
it is not a DOI or a claim that the preprint has been accepted by a journal.
