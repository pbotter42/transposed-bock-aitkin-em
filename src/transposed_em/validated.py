"""Laplace posterior-moment iteration with explicit fixed-point diagnostics.

This algorithm is not an optimizer of the Laplace log integral. The default
expanded-and-normalized moment map is checked for numerical stability only.
No parameter standard errors are inferred from these diagnostics.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from jax.experimental import enable_x64
from scipy.special import expit

from .config import FitConfig
from .data import coerce_responses, item_major, theta_start
from .estimator import AlternatingRandomItemIRT
from .families import NormalFamily
from .families.common import pack_cholesky
from .families.normal import NormalState
from .item_step import build_item_stepper
from .models import natural_item_parameters_numpy
from .posterior import laplace_cubature
from .theta_step import update_theta


def set_population(family, covariance):
    covariance = np.asarray(covariance, dtype=float)
    diagonal, offdiagonal = pack_cholesky(np.linalg.cholesky(covariance))
    family.state = NormalState(np.zeros(len(covariance)), diagonal, offdiagonal)


def fisher_residual(theta, nodes, weights, responses, mask, covariance, diagonal):
    """Normalized posterior-averaged scores in zero-mean free coordinates."""
    dimension = nodes.shape[-1]
    score = np.zeros(len(theta))
    for start in range(0, len(nodes), 256):
        z, w = nodes[start : start + 256], weights[start : start + 256]
        a = np.exp(z[:, :, 1]) if dimension == 2 else np.ones(z.shape[:2])
        p = expit(a[:, None, :] * (theta[None, :, None] - z[:, None, :, 0]))
        score += np.einsum(
            "jik,jk,jk->i",
            mask[start : start + 256, :, None]
            * (responses[start : start + 256, :, None] - p),
            a,
            w,
        )
    score /= np.maximum(mask.sum(axis=0), 1)
    second = np.einsum("jk,jkd,jke->de", weights, nodes, nodes) / len(nodes)
    inverse = np.linalg.inv(covariance)
    covariance_score = 0.5 * inverse @ (second - covariance) @ inverse
    if diagonal:
        free = np.diag(covariance_score)
    else:
        free = covariance_score[np.triu_indices(dimension)].copy()
        if dimension == 2:
            free[1] *= 2
    return {
        "theta_score_max": float(np.abs(score).max()),
        "covariance_score_max": float(np.abs(free).max()),
        "theta_score": score,
        "covariance_score": free,
    }


class LaplaceIRT(AlternatingRandomItemIRT):
    """Fit the normalized Laplace posterior-moment map.

    Rows are subjects (or systems); columns are binary items. The 2PL uses
    normal random effects in (b, log(a)) with zero population means. Choose
    diagonal covariance to fix their correlation to zero, or full to estimate
    it. The 1PL fixes a=1. See docs/method.md for updates and numerical limits.

    Stopping tolerances concern the iteration, not exact likelihood scores.
    A fit at the iteration limit is returned with that status, never relabeled
    as converged. Nonfinite numerical steps raise FloatingPointError.
    """

    def __init__(
        self,
        model="2pl",
        covariance=None,
        max_iterations=1000,
        min_iterations=20,
        patience=5,
        damping=0.5,
        parameter_tolerance=2e-4,
        prediction_tolerance=1e-5,
        item_newton_steps=20,
        progress_callback=None,
    ):
        if covariance is None:
            covariance = "diagonal" if model == "1pl" else "full"
        super().__init__(
            FitConfig(
                model=model,
                covariance=covariance,
                max_iterations=max_iterations,
                min_iterations=min(min_iterations, max_iterations),
                convergence_patience=patience,
                item_newton_steps=item_newton_steps,
            )
        )
        if not 0 < damping <= 1:
            raise ValueError("damping must be in (0, 1]")
        for name, value in (
            ("parameter_tolerance", parameter_tolerance),
            ("prediction_tolerance", prediction_tolerance),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable")
        self.damping = damping
        self.parameter_tolerance = parameter_tolerance
        self.prediction_tolerance = prediction_tolerance
        self.diagnostics_ = {}
        self.progress_callback = progress_callback

    @enable_x64()
    def fit(self, matrix, theta_init=None, covariance_init=None):
        """Fit in 64-bit arithmetic without changing JAX's global precision setting."""
        self._clear_fit()
        data = coerce_responses(matrix)
        self.subject_ids_ = data.subject_ids
        self.item_ids_ = data.item_ids
        matrix = data.to_numpy()
        if min(matrix.shape) < 2:
            raise ValueError("fitting requires at least two subjects and two items")
        if theta_init is not None:
            theta_init = np.asarray(theta_init, dtype=float)
            if theta_init.shape != (matrix.shape[0],) or not np.isfinite(theta_init).all():
                raise ValueError(
                    "theta_init must be a finite vector with one entry per subject"
                )
        if covariance_init is not None:
            covariance_init = np.asarray(covariance_init, dtype=float)
            dimension = self.model_dimension
            if (
                covariance_init.shape != (dimension, dimension)
                or not np.isfinite(covariance_init).all()
            ):
                raise ValueError(
                    f"covariance_init must be a finite {dimension} by {dimension} matrix"
                )
            if not np.allclose(covariance_init, covariance_init.T, rtol=0, atol=1e-12):
                raise ValueError("covariance_init must be symmetric")
            if np.linalg.eigvalsh(covariance_init).min() <= 0:
                raise ValueError("covariance_init must be positive definite")
            if self.config.covariance == "diagonal" and np.any(
                covariance_init != np.diag(np.diag(covariance_init))
            ):
                raise ValueError("covariance_init must be diagonal for this model")
        started = time.perf_counter()
        y, mask = item_major(matrix)
        self.item_responses_, self.item_mask_ = y, mask
        theta = theta_start(matrix) if theta_init is None else np.asarray(theta_init).copy()
        latent = self._initial_latent(matrix, theta)
        self.family_ = NormalFamily(self.model_dimension, self.config)
        self.family_.fit(latent)
        latent, theta, moments = self._normalize_identification(latent, theta)
        covariance = (
            moments.raw_cov
            if covariance_init is None
            else np.asarray(covariance_init).copy()
        )
        set_population(self.family_, covariance)
        stepper = build_item_stepper(
            self.model_dimension,
            self.family_,
            self.config.item_newton_steps,
            self.config.ridge,
        )
        flat = np.flatnonzero(np.isfinite(matrix))
        flat = flat[np.linspace(0, len(flat) - 1, min(4096, len(flat)), dtype=int)]
        row, col = flat // matrix.shape[1], flat % matrix.shape[1]
        previous_predictions = None
        stable = 0
        self.history_ = []
        self.stopping_reason_ = "iteration_limit"
        for iteration in range(1, self.config.max_iterations + 1):
            step = stepper(latent, y, mask, theta, self.family_.state_dict(), 1.0)
            if not all(
                np.isfinite(x).all()
                for x in (step.latent_modes, step.hessian, step.laplace_nll)
            ):
                self.stopping_reason_ = "numerical_failure"
                raise FloatingPointError(
                    f"Nonfinite Laplace posterior at iteration {iteration}"
                )
            latent = step.latent_modes
            nodes, posterior_cov, weights = laplace_cubature(latent, step.hessian)
            weights = np.broadcast_to(weights, nodes.shape[:2])
            predictions = self._predict_with_nodes(theta, nodes, weights, row, col)
            expanded = copy.deepcopy(self.family_)
            expanded.fit_weighted(nodes, weights)
            raw = expanded.moments()
            alpha_shift = raw.raw_mean[1] if self.model_dimension == 2 else 0.0
            scale, shift = np.exp(alpha_shift), raw.raw_mean[0]
            mapped_nodes = nodes.copy()
            mapped_nodes[:, :, 0] = scale * (nodes[:, :, 0] - shift)
            if self.model_dimension == 2:
                mapped_nodes[:, :, 1] -= alpha_shift
            transform = np.eye(self.model_dimension)
            transform[0, 0] = scale
            candidate_cov = transform @ raw.raw_cov @ transform.T
            theta_for_update = scale * (theta - shift)
            candidate_theta = update_theta(
                theta_for_update,
                y.T,
                mask.T,
                mapped_nodes,
                weights,
                self.model_dimension,
                1.0,
                self.config.theta_bounds,
            )
            theta_residual = float(np.max(np.abs(candidate_theta - theta)))
            population_residual = float(np.max(np.abs(candidate_cov - covariance)))
            prediction_change = (
                float("inf")
                if previous_predictions is None
                else float(np.sqrt(np.mean((predictions - previous_predictions) ** 2)))
            )
            criteria = (
                iteration >= self.config.min_iterations
                and theta_residual < self.parameter_tolerance
                and population_residual < self.parameter_tolerance
                and float(np.max(step.mode_gradient_max)) < 1e-4
                and prediction_change < self.prediction_tolerance
            )
            stable = stable + 1 if criteria else 0
            self.history_.append(
                {
                    "iteration": iteration,
                    "theta_map_residual": theta_residual,
                    "covariance_map_residual": population_residual,
                    "prediction_rms_change": prediction_change,
                    "laplace_nll": float(step.laplace_nll.sum()),
                    "stable_iterations": stable,
                }
            )
            self.history_[-1]["item_mode_gradient_max"] = float(
                np.max(step.mode_gradient_max)
            )
            if self.progress_callback is not None and (
                iteration == 1 or iteration % 20 == 0
            ):
                self.progress_callback(self.history_[-1])
            # On success retain the state just tested, not an unchecked final M-step.
            if stable >= self.config.convergence_patience:
                self.stopping_reason_ = "fixed_point_criteria"
                break
            if iteration == self.config.max_iterations:
                break
            theta = (1 - self.damping) * theta + self.damping * candidate_theta
            covariance = (1 - self.damping) * covariance + self.damping * candidate_cov
            set_population(self.family_, covariance)
            previous_predictions = predictions
        self.theta_, self.latent_modes_ = theta, latent
        self.family_moments_ = self.family_.moments()
        self.posterior_nodes_, self.node_weights_ = nodes, weights
        self.item_hessian_, self.item_posterior_covariance_ = step.hessian, posterior_cov
        self.item_params_ = natural_item_parameters_numpy(latent, self.model_dimension)
        self.item_laplace_nll_ = step.laplace_nll
        self.objective_increase_count_ = sum(
            a["laplace_nll"] > b["laplace_nll"]
            for a, b in zip(self.history_[1:], self.history_[:-1])
        )
        residuals = fisher_residual(
            theta, nodes, weights, y, mask, covariance, self.config.covariance == "diagonal"
        )
        self.diagnostics_ = {
            key: value
            for key, value in residuals.items()
            if not isinstance(value, np.ndarray)
        }
        self.diagnostics_.update(
            self.history_[-1],
            stopping_reason=self.stopping_reason_,
            elapsed_seconds=time.perf_counter() - started,
            theta_boundary_count=int(np.sum(np.abs(theta) > 5.999)),
            posterior_node_boundary_count=int(np.sum(np.abs(nodes) >= 10.0)),
            covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
            update_rule="normalized",
            approximate_fisher_root_checked=False,
            fixed_point_validated=self.stopping_reason_ == "fixed_point_criteria",
        )
        return self

    def summary(self):
        """Population estimates and diagnostics; no inferential standard errors."""
        if self.theta_ is None:
            raise RuntimeError("Model has not been fit yet")
        result = {
            "model": self.config.model,
            "covariance": self.config.covariance,
            "integration": "laplace_gaussian_cubature",
            "n_subjects": len(self.theta_),
            "n_items": len(self.latent_modes_),
            "outer_iterations": len(self.history_),
            "population_mean": self.family_moments_.raw_mean.tolist(),
            "population_covariance": self.family_moments_.raw_cov.tolist(),
            "population_correlation": self.family_moments_.raw_corr.tolist(),
            "objective_increase_count": self.objective_increase_count_,
        }
        result.update(self.diagnostics_)
        result.update(
            theta_estimator="laplace_posterior_moment_map",
            parameter_tolerance=self.parameter_tolerance,
            prediction_tolerance=self.prediction_tolerance,
            parameter_standard_errors_validated=False,
        )
        return result
