from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import expit


def clamp_probability_numpy(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.clip(values, eps, 1.0 - eps)


def natural_item_parameters_numpy(
    latent: np.ndarray, model_dim: int, a_scale: float = 1.0
) -> dict[str, np.ndarray]:
    latent = np.atleast_2d(np.asarray(latent, dtype=float))
    difficulty = latent[:, 0]
    discrimination = (
        np.exp(np.clip(latent[:, 1], -20.0, 20.0))
        if model_dim == 2
        else np.ones(latent.shape[0])
    )
    return {"a": discrimination, "b": difficulty}


def natural_item_parameters_jax(latent, model_dim: int, a_scale: float = 1.0):
    difficulty = latent[0]
    discrimination = jnp.exp(jnp.clip(latent[1], -20.0, 20.0)) if model_dim == 2 else 1.0
    return discrimination, difficulty


def irt_probability_numpy(theta, a, b) -> np.ndarray:
    linear_predictor = np.asarray(a) * (np.asarray(theta) - np.asarray(b))
    return clamp_probability_numpy(expit(linear_predictor))


def irt_probability_jax(theta, a, b):
    return jnp.clip(jax_sigmoid(a * (theta - b)), 1e-6, 1.0 - 1e-6)


def jax_sigmoid(value):
    return 1.0 / (1.0 + jnp.exp(-value))
