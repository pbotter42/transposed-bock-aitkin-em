import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.experimental import enable_x64
from scipy.special import expit

import transposed_em.validated as implementation
from transposed_em import LaplaceIRT, ResponseMatrix
from transposed_em.item_step import logistic_derivatives
from transposed_em.posterior import laplace_cubature
from transposed_em.theta_step import theta_objective, update_theta


@pytest.fixture
def responses():
    rng = np.random.default_rng(720)
    theta = np.linspace(-1.5, 1.5, 10)
    b, alpha = rng.normal(size=(2, 40))
    probability = expit(np.exp(0.3 * alpha) * (theta[:, None] - b))
    matrix = rng.binomial(1, probability).astype(float)
    matrix[0, 0] = np.nan
    return matrix


@pytest.fixture(scope="module")
def frozen_class(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference")
    with ZipFile(Path(__file__).parent / "reference_v1.3.1.zip") as archive:
        archive.extractall(root)
    init = root / "src/transposed_em/__init__.py"
    spec = importlib.util.spec_from_file_location(
        "frozen_transposed_em", init, submodule_search_locations=[str(init.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ValidatedLaplaceIRT


@pytest.mark.parametrize(
    "model,covariance", [("1pl", "diagonal"), ("2pl", "diagonal"), ("2pl", "full")]
)
def test_frozen_numerical_equivalence(responses, frozen_class, model, covariance):
    options = {"model": model, "covariance": covariance, "max_iterations": 4}
    with enable_x64():
        expected = frozen_class(**options).fit(responses)
    previous_precision = jax.config.x64_enabled
    actual = LaplaceIRT(**options).fit(responses)
    assert jax.config.x64_enabled == previous_precision
    for name in (
        "theta_",
        "latent_modes_",
        "posterior_nodes_",
        "node_weights_",
        "item_posterior_covariance_",
        "item_laplace_nll_",
    ):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-10, atol=1e-10
        )
    np.testing.assert_allclose(
        actual.family_moments_.raw_cov,
        expected.family_moments_.raw_cov,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual.history_frame(), expected.history_frame(), rtol=1e-10, atol=1e-10
    )
    assert actual.stopping_reason_ == "iteration_limit"
    assert not actual.summary()["fixed_point_validated"]
    assert not actual.summary()["parameter_standard_errors_validated"]
    assert actual.item_frame().shape[0] == responses.shape[1]
    assert actual.subject_frame().shape[0] == responses.shape[0]
    probability = actual.predict_pairs([0, 1], [1, 2])
    assert np.all((probability > 0) & (probability < 1))
    np.testing.assert_allclose(
        actual.score_subjects(responses[:2]), expected.score_subjects(responses[:2])
    )


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 0.5, 2.0])
def test_invalid_response(responses, bad):
    responses[0, 0] = bad
    with pytest.raises(ValueError):
        ResponseMatrix(responses)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "3pl"},
        {"model": "4pl"},
        {"max_iterations": 0},
        {"max_iterations": 1.5},
        {"patience": True},
        {"damping": float("nan")},
        {"parameter_tolerance": float("inf")},
        {"prediction_tolerance": -1},
        {"model": "1pl", "covariance": "full"},
        {"progress_callback": 1},
    ],
)
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        LaplaceIRT(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta_init": [0]},
        {"theta_init": [np.nan] * 10},
        {"covariance_init": [[1]]},
        {"covariance_init": [[1, 2], [0, 1]]},
        {"covariance_init": [[1, 2], [2, 1]]},
        {"covariance_init": [[np.inf, 0], [0, 1]]},
    ],
)
def test_invalid_initial_values(responses, kwargs):
    with pytest.raises(ValueError):
        LaplaceIRT(max_iterations=2).fit(responses, **kwargs)


def test_diagonal_initial_covariance(responses):
    with pytest.raises(ValueError, match="diagonal"):
        LaplaceIRT(covariance="diagonal").fit(
            responses, covariance_init=[[1, 0.1], [0.1, 1]]
        )


def test_missing_and_labels(responses):
    frame = pd.DataFrame(
        responses,
        index=[f"system-{i}" for i in range(10)],
        columns=[f"item-{i}" for i in range(40)],
    )
    frame.iloc[0, 0] = -1
    fit = LaplaceIRT(model="1pl", max_iterations=2).fit(frame)
    assert fit.subject_frame().subject_id.tolist() == list(frame.index)
    assert fit.item_frame().item_id.tolist() == list(frame.columns)
    assert np.all(fit.item_frame().a == 1)
    np.testing.assert_allclose(
        fit.score_subjects(frame.iloc[:2]), fit.score_subjects(responses[:2])
    )
    for row in (np.full((1, 40), np.nan), np.full((1, 40), 0.5), np.full((1, 40), np.inf)):
        with pytest.raises(ValueError):
            fit.score_subjects(row)
    for subjects, items in (
        ([-1], [0]),
        ([10], [0]),
        ([0], [40]),
        ([0.5], [0]),
        ([0, 1], [0]),
    ):
        with pytest.raises(ValueError):
            fit.predict_pairs(subjects, items)
    for attr in (
        "score_subjects",
        "predict_pairs",
        "summary",
        "item_frame",
        "subject_frame",
    ):
        assert callable(getattr(fit, attr))


def test_nonfinite_step_does_not_return_mixed_state(responses, monkeypatch):
    factory = implementation.build_item_stepper

    def failing_factory(*args):
        original = factory(*args)
        calls = 0

        def step(*values):
            nonlocal calls
            calls += 1
            result = original(*values)
            if calls == 2:
                return SimpleNamespace(
                    latent_modes=result.latent_modes,
                    hessian=result.hessian,
                    laplace_nll=np.array([np.nan]),
                )
            return result

        return step

    fit = LaplaceIRT(max_iterations=3).fit(responses)
    monkeypatch.setattr(implementation, "build_item_stepper", failing_factory)
    with pytest.raises(FloatingPointError, match="iteration 2"):
        fit.fit(responses)
    assert fit.theta_ is None and fit.posterior_nodes_ is None
    assert fit.stopping_reason_ == "numerical_failure"
    assert len(fit.history_) == 1
    with pytest.raises(RuntimeError):
        fit.summary()


def test_cubature_moments_before_clipping():
    modes = np.array([[0.3, -0.2], [0.2, 0.1]])
    hessian = np.array([[[2.0, 0.2], [0.2, 3.0]]] * 2)
    nodes, covariance, weights = laplace_cubature(modes, hessian)
    np.testing.assert_allclose(np.einsum("k,jkd->jd", weights, nodes), modes)
    centered = nodes - modes[:, None, :]
    np.testing.assert_allclose(
        np.einsum("k,jkd,jke->jde", weights, centered, centered), covariance
    )
    np.testing.assert_allclose(covariance, np.linalg.inv(hessian))


def test_score_update_matches_objective():
    nodes = np.array([[[-1.0], [-0.5]], [[0.5], [1.0]]])
    weights = np.full(2, 0.5)
    y = np.array([[1.0, 0.0]])
    theta = update_theta([0.8], y, np.ones_like(y), nodes, weights, 1, 1.0, (-6, 6))[0]
    parameters = {"a": np.ones((2, 2)), "b": nodes[:, :, 0]}
    value = theta_objective(theta, y[0], [1, 1], parameters, weights)
    assert value < theta_objective(theta - 0.1, y[0], [1, 1], parameters, weights)
    assert value < theta_objective(theta + 0.1, y[0], [1, 1], parameters, weights)
    for response, expected in ((0, -6), (1, 6)):
        result = update_theta(
            [0], np.full_like(y, response), np.ones_like(y), nodes, weights, 1, 1.0, (-6, 6)
        )
        assert result[0] == expected


def test_cli(responses, tmp_path):
    path, output = tmp_path / "input.csv", tmp_path / "result"
    pd.DataFrame(responses).to_csv(path)
    command = [
        sys.executable,
        "-m",
        "transposed_em.cli",
        str(path),
        "--output",
        str(output),
        "--model",
        "1pl",
        "--max-iterations",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["stopping_reason"] == "iteration_limit"
    assert len(pd.read_csv(output / "subjects.csv")) == 10
    assert len(pd.read_csv(output / "items.csv")) == 40
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0


@pytest.mark.parametrize("dimension", [1, 2])
def test_item_derivatives(dimension):
    with enable_x64():
        latent = jnp.array([0.3, -0.4][:dimension])
        theta = jnp.array([-1.5, -0.2, 0.7, 1.3])
        y, mask = jnp.array([0.0, 1.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 0.0, 1.0])

        def objective(z):
            a = jnp.exp(z[1]) if dimension == 2 else 1.0
            linear = a * (theta - z[0])
            return jnp.sum(mask * (jax.nn.softplus(linear) - y * linear))

        gradient, hessian = logistic_derivatives(latent, y, mask, theta, dimension)
        np.testing.assert_allclose(
            gradient, jax.grad(objective)(latent), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            hessian, jax.hessian(objective)(latent), rtol=1e-12, atol=1e-12
        )


def test_reference_integrity():
    directory = Path(__file__).parent
    manifest = json.loads((directory / "reference_manifest.json").read_text())
    archive = directory / "reference_v1.3.1.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with ZipFile(archive) as source:
        assert set(source.namelist()) == set(manifest["source_files"])
        for name, digest in manifest["source_files"].items():
            assert hashlib.sha256(source.read(name)).hexdigest() == digest


def test_stopping_rule_retains_checked_state(responses):
    fit = LaplaceIRT(
        model="1pl",
        max_iterations=20,
        min_iterations=2,
        patience=2,
        parameter_tolerance=1e6,
        prediction_tolerance=1e6,
    ).fit(responses)
    assert fit.stopping_reason_ == "fixed_point_criteria"
    assert fit.history_[-1]["stable_iterations"] == 2
    assert fit.summary()["fixed_point_validated"]
    assert not fit.summary()["approximate_fisher_root_checked"]
    assert not fit.summary()["parameter_standard_errors_validated"]
