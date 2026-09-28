# API

`LaplaceIRT(model="2pl", covariance=None, max_iterations=1000,
min_iterations=20, patience=5, damping=0.5, parameter_tolerance=2e-4,
prediction_tolerance=1e-5, item_newton_steps=20, progress_callback=None)`

`covariance=None` selects diagonal for the 1PL and full for the 2PL.
The callback receives a diagnostic dictionary at iteration 1 and every
20 iterations. See [method.md](method.md) for the controls' meaning.

| Operation | Result |
| --- | --- |
| `fit(responses, theta_init=None, covariance_init=None)` | Fits in place and returns the estimator |
| `subject_frame()` | Subject IDs and capability estimates |
| `item_frame()` | Item IDs, `b`, `a`, `latent_b`, and (2PL) `latent_alpha` |
| `summary()` | Model settings, population covariance, and diagnostics |
| `history_frame()` | One row per completed numerical iteration |
| `predict_pairs([0,1], [2,3])` | Posterior-averaged probabilities for two subject/item pairs |
| `score_subjects(new_responses)` | Capabilities for new subjects on the same item columns |

The matrix is subject by item, with `0/1` observations and `NaN` or `-1`
for missing responses. Every fitted subject and item must have an observation.
At least two subjects and two items are required by the initialization;
this is not a statistical sufficiency claim. Infinite and nonbinary values
are errors. No response-based missingness correction is performed.

Pandas DataFrame labels are retained, or supply `ResponseMatrix(data,
subject_ids=..., item_ids=...)`. Prediction indices are zero-based positions,
not those labels. Scoring a new subject permits unobserved item columns
but requires at least one observed response in each row.

`theta_init` must contain one finite value per subject. `covariance_init`
must be positive definite and symmetric, of shape `(1,1)` or `(2,2)`; a
diagonal model requires a diagonal initialization. Initial capability
location is normalized together with the item population.

## CSV Interface

```sh
transposed-em responses.csv --model 2pl --covariance full --output fit-output
```

The CSV has a header, subject IDs in the first column, and one item per
remaining column. Empty cells are missing responses. The output directory
must be new or empty. It receives `subjects.csv`, `items.csv`, `history.csv`,
and `summary.json`. Check the stopping reason even when the command exits
successfully: an iteration limit is not a successful convergence claim.
Undefined diagnostic values are JSON `null`, not nonstandard `Infinity`.

## Development

```sh
git clone https://github.com/pbotter42/transposed-bock-aitkin-em.git
cd transposed-bock-aitkin-em
python -m pip install -e '.[dev]'
pytest -q
ruff check .
ruff format --check .
python -m build
```

The tests use small deterministic matrices; they do not run the production
simulation or WILD application. JAX's first fit includes compilation time.
The 64-bit fitting context restores the previous JAX precision setting.
