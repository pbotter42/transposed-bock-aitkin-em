"""Stable expected-log-likelihood scoring and frozen-bank predictive scoring."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from .models import irt_probability_numpy, natural_item_parameters_numpy


def theta_objective(
    theta,
    responses,
    mask,
    node_params,
    node_weights,
):
    responses = np.asarray(responses, dtype=float)[:, None]
    mask = np.asarray(mask, dtype=float)[:, None]
    linear = node_params["a"] * (theta - node_params["b"])
    ll = mask * (responses * linear - np.logaddexp(0.0, linear))
    return -float(np.sum(ll * node_weights))


def update_theta(
    theta_start,
    responses,
    mask,
    posterior_nodes,
    node_weights,
    model_dim,
    a_scale,
    bounds,
):
    theta = np.asarray(theta_start, dtype=float).copy()
    responses = np.asarray(responses, dtype=float)
    mask = np.asarray(mask, dtype=float)
    n_items, n_nodes, dimension = posterior_nodes.shape
    flat_params = natural_item_parameters_numpy(
        posterior_nodes.reshape(n_items * n_nodes, dimension),
        model_dim=model_dim,
        a_scale=a_scale,
    )
    node_params = {
        name: values.reshape(n_items, n_nodes) for name, values in flat_params.items()
    }
    node_weights = np.asarray(node_weights, dtype=float)
    if node_weights.ndim == 1:
        node_weights = np.broadcast_to(node_weights, (n_items, n_nodes))
    if node_weights.shape != (n_items, n_nodes):
        raise ValueError("node_weights have an incompatible shape")
    for i in range(theta.shape[0]):
        if not np.any(mask[i]):
            raise ValueError("Each subject needs an observed response for scoring")

        def score(value, i=i):
            probability = expit(node_params["a"] * (value - node_params["b"]))
            return float(
                np.sum(
                    mask[i, :, None]
                    * node_weights
                    * node_params["a"]
                    * (responses[i, :, None] - probability)
                )
            )

        lower, upper = bounds
        if score(lower) <= 0:
            theta[i] = lower
        elif score(upper) >= 0:
            theta[i] = upper
        else:
            theta[i] = brentq(score, lower, upper, xtol=1e-9, rtol=1e-12)
    return theta


def predictive_theta_objective(
    theta,
    responses,
    mask,
    node_params,
    node_weights,
):
    probability = irt_probability_numpy(theta=theta, a=node_params["a"], b=node_params["b"])
    integrated = np.sum(probability * node_weights, axis=1)
    integrated = np.clip(integrated, 1e-10, 1.0 - 1e-10)
    log_likelihood = mask * (
        responses * np.log(integrated) + (1.0 - responses) * np.log1p(-integrated)
    )
    return -float(np.sum(log_likelihood))


def update_theta_predictive(
    theta_start,
    responses,
    mask,
    posterior_nodes,
    node_weights,
    model_dim,
    bounds,
):
    """Score new subjects by the frozen-bank predictive marginal likelihood."""

    theta = np.asarray(theta_start, dtype=float).copy()
    responses = np.asarray(responses, dtype=float)
    mask = np.asarray(mask, dtype=float)
    n_items, n_nodes, dimension = posterior_nodes.shape
    flat_params = natural_item_parameters_numpy(
        posterior_nodes.reshape(n_items * n_nodes, dimension), model_dim=model_dim
    )
    node_params = {
        name: values.reshape(n_items, n_nodes) for name, values in flat_params.items()
    }
    weights = np.asarray(node_weights, dtype=float)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n_items, n_nodes))
    if weights.shape != (n_items, n_nodes):
        raise ValueError("node_weights have an incompatible shape")
    for i in range(theta.shape[0]):
        result = minimize_scalar(
            predictive_theta_objective,
            bounds=bounds,
            method="bounded",
            args=(responses[i], mask[i], node_params, weights),
            options={"xatol": 1e-4, "maxiter": 200},
        )
        if not result.success or not np.isfinite(result.fun):
            raise FloatingPointError(f"Predictive scoring failed for subject {i}")
        theta[i] = float(result.x)
    return theta
