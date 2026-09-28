"""Shared initialization and prediction for the Laplace estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit

from .data import theta_start
from .models import (
    clamp_probability_numpy,
    irt_probability_numpy,
    natural_item_parameters_numpy,
)
from .theta_step import update_theta_predictive


class AlternatingRandomItemIRT:
    def __init__(self, config):
        self.config = config
        self._clear_fit()

    def _clear_fit(self):
        for name in (
            "theta_",
            "latent_modes_",
            "item_params_",
            "family_",
            "family_moments_",
            "posterior_nodes_",
            "node_weights_",
            "item_hessian_",
            "item_posterior_covariance_",
            "item_laplace_nll_",
            "item_responses_",
            "item_mask_",
        ):
            setattr(self, name, None)
        self.history_ = []
        self.diagnostics_ = {}
        self.stopping_reason_ = "not_started"

    @property
    def model_dimension(self):
        return self.config.dimension

    @property
    def population_dimension(self):
        return self.config.dimension

    def _initial_latent(self, matrix: np.ndarray, theta: np.ndarray) -> np.ndarray:
        dim = self.model_dimension
        observed = np.isfinite(matrix)
        responses = np.nan_to_num(matrix, nan=0.0)
        counts = np.maximum(np.sum(observed, axis=0), 1)
        p = np.sum(responses, axis=0) / counts
        fallback = float(np.nanmean(matrix))
        p = np.where(np.isfinite(p), p, fallback)
        p = clamp_probability_numpy(p, eps=1e-4)
        latent = np.zeros((matrix.shape[1], dim), dtype=float)
        if dim >= 2:
            # A zero alpha vector places the initial discrimination variance on
            # the covariance boundary and can trap the 2PL at a 1PL solution.
            # The first-order logistic slope below gives the population update
            # a finite, data-informed interior start without fitting p separate
            # logistic regressions.
            theta_column = np.asarray(theta, dtype=float)[:, None]
            item_theta_mean = np.sum(observed * theta_column, axis=0) / counts
            centered_theta = theta_column - item_theta_mean
            item_variance = np.sum(observed * centered_theta**2, axis=0) / counts
            covariance = (
                np.sum(observed * centered_theta * (responses - p), axis=0) / counts
            )
            information = item_variance * p * (1.0 - p)
            slope = covariance / np.maximum(information, 1e-3)
            slope = np.clip(slope, 0.30, 3.50)
            alpha = np.log(slope)
            alpha -= float(np.mean(alpha))
            latent[:, 1] = alpha
            latent[:, 0] = item_theta_mean - logit(p) / np.exp(alpha)
        else:
            latent[:, 0] = np.mean(theta) - logit(p)
        return np.clip(latent, -10.0, 10.0)

    def _normalize_identification(self, latent, theta):
        if self.family_ is None:
            raise RuntimeError("Family is not initialized.")
        moments = self.family_.moments()
        shift = float(moments.b_center)
        latent = np.asarray(latent, dtype=float).copy()
        theta = np.asarray(theta, dtype=float).copy()
        latent[:, 0] = latent[:, 0] - shift
        theta = theta - shift
        self.family_.affine_normalize_b(shift=shift, scale=1.0)
        return latent, theta, self.family_.moments()

    def _predict_with_nodes(self, theta, nodes, weights, person_index, item_index):
        selected = nodes[item_index]
        n_pairs, n_nodes, dimension = selected.shape
        params = natural_item_parameters_numpy(
            selected.reshape(n_pairs * n_nodes, dimension),
            model_dim=self.model_dimension,
            a_scale=1.0,
        )
        probability = irt_probability_numpy(
            theta=np.repeat(theta[person_index], n_nodes),
            a=params["a"],
            b=params["b"],
        ).reshape(n_pairs, n_nodes)
        weights = np.asarray(weights)
        selected_weights = (
            weights[item_index]
            if weights.ndim == 2
            else np.broadcast_to(weights, probability.shape)
        )
        return np.sum(probability * selected_weights, axis=1)

    def predict_pairs(
        self, person_index, item_index, theta: np.ndarray | None = None
    ) -> np.ndarray:
        if self.item_params_ is None or self.posterior_nodes_ is None:
            raise RuntimeError("Model has not been fit yet.")
        theta_use = self.theta_ if theta is None else np.asarray(theta, dtype=float)
        if theta_use.ndim != 1 or not np.isfinite(theta_use).all():
            raise ValueError("theta must be a finite one-dimensional vector")
        person_index = np.asarray(person_index)
        item_index = np.asarray(item_index)
        if person_index.ndim != 1 or item_index.shape != person_index.shape:
            raise ValueError("person_index and item_index must be equal-length vectors")
        for name, values, size in (
            ("person_index", person_index, len(theta_use)),
            ("item_index", item_index, len(self.posterior_nodes_)),
        ):
            if (
                values.dtype.kind not in "iu"
                or np.any(values < 0)
                or np.any(values >= size)
            ):
                raise ValueError(f"{name} must contain valid nonnegative integer indices")
        return self._predict_with_nodes(
            theta_use,
            self.posterior_nodes_,
            self.node_weights_,
            person_index,
            item_index,
        )

    def score_subjects(
        self, matrix: np.ndarray, theta_init: np.ndarray | None = None
    ) -> np.ndarray:
        """Score new subjects against the fitted item bank, in the original item order.

        This maximizes a sum of log posterior-predictive probabilities, not
        the expected log likelihood used inside fit(). No global optimum is
        guaranteed by the bounded scalar search.
        """

        if self.posterior_nodes_ is None or self.family_moments_ is None:
            raise RuntimeError("Model has not been fit yet.")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.posterior_nodes_.shape[0]:
            raise ValueError(
                f"responses must have {self.posterior_nodes_.shape[0]} item columns"
            )
        matrix = np.where(matrix == -1, np.nan, matrix)
        if np.isinf(matrix).any() or not np.isin(matrix[np.isfinite(matrix)], [0, 1]).all():
            raise ValueError("responses must be 0, 1, NaN, or -1")
        observed = np.isfinite(matrix)
        if np.any(np.sum(observed, axis=1) == 0):
            raise ValueError("every scored subject must have an observed response")
        responses = np.nan_to_num(matrix, nan=0.0)
        start = (
            theta_start(matrix)
            if theta_init is None
            else np.asarray(theta_init, dtype=float)
        )
        if start.shape != (matrix.shape[0],) or not np.isfinite(start).all():
            raise ValueError(f"theta_init must have shape ({matrix.shape[0]},)")
        return update_theta_predictive(
            theta_start=start,
            responses=responses,
            mask=observed.astype(float),
            posterior_nodes=self.posterior_nodes_,
            node_weights=self.node_weights_,
            model_dim=self.model_dimension,
            bounds=self.config.theta_bounds,
        )

    def item_frame(self) -> pd.DataFrame:
        """Conditional item modes; a=exp(alpha_mode), not the posterior mean of a."""
        if self.item_params_ is None or self.latent_modes_ is None:
            raise RuntimeError("Model has not been fit yet.")
        data = {
            "item_id": self.item_ids_ or np.arange(self.latent_modes_.shape[0]),
            "b": self.item_params_["b"],
            "a": self.item_params_["a"],
            "latent_b": self.latent_modes_[:, 0],
        }
        if self.model_dimension >= 2:
            data["latent_alpha"] = self.latent_modes_[:, 1]
        return pd.DataFrame(data)

    def subject_frame(self) -> pd.DataFrame:
        """Fitted capability estimates in the response matrix's row order."""
        if self.theta_ is None:
            raise RuntimeError("Model has not been fit yet")
        return pd.DataFrame(
            {
                "subject_id": self.subject_ids_ or np.arange(len(self.theta_)),
                "theta": self.theta_,
            }
        )

    def latent_correlation_frame(self) -> pd.DataFrame:
        if self.family_moments_ is None:
            raise RuntimeError("Model has not been fit yet.")
        names = ["b", "alpha"][: self.population_dimension]
        rows = []
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                rows.append(
                    {
                        "parameter_1": name_i,
                        "parameter_2": name_j,
                        "correlation": float(self.family_moments_.raw_corr[i, j]),
                    }
                )
        return pd.DataFrame(rows)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)
