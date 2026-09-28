from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .models import natural_item_parameters_jax


@dataclass
class ItemStepResult:
    latent_modes: np.ndarray
    laplace_nll: np.ndarray
    joint_nll: np.ndarray
    hessian: np.ndarray
    mode_gradient_max: np.ndarray


_CACHE: dict[tuple, Callable] = {}


def logistic_derivatives(latent, responses, mask, theta, dimension):
    """Analytic likelihood derivatives, checked against automatic differentiation."""
    a = jnp.exp(latent[1]) if dimension == 2 else jnp.array(1.0)
    linear = a * (theta - latent[0])
    probability = jax.nn.sigmoid(linear)
    residual = mask * (probability - responses)
    weight = mask * probability * (1.0 - probability)
    gb = -a * jnp.sum(residual)
    hbb = a * a * jnp.sum(weight)
    if dimension == 1:
        return jnp.array([gb]), jnp.array([[hbb]])
    ga = jnp.sum(residual * linear)
    hba = -a * jnp.sum(weight * linear + residual)
    haa = jnp.sum(weight * linear**2 + residual * linear)
    return jnp.array([gb, ga]), jnp.array([[hbb, hba], [hba, haa]])


def build_item_stepper(model_dim, family, newton_steps=6, ridge=1e-4):
    """Create a compiled item-wise Newton update for a normal item population."""

    key = (model_dim, family.dim, int(newton_steps), float(ridge))
    if key in _CACHE:
        return _CACHE[key]
    eye = jnp.eye(model_dim)

    def item_nll(latent, responses, mask, theta, family_state, a_scale):
        log_prior = family.logpdf_jax(latent, family_state)
        discrimination, difficulty = natural_item_parameters_jax(latent, model_dim, a_scale)
        linear = discrimination * (theta - difficulty)
        likelihood = mask * (responses * linear - jax.nn.softplus(linear))
        return -(log_prior + jnp.sum(likelihood))

    prior_gradient = jax.grad(lambda z, state: -family.logpdf_jax(z, state))
    prior_hessian = jax.hessian(lambda z, state: -family.logpdf_jax(z, state))

    def gradient(latent, responses, mask, theta, family_state, a_scale):
        likelihood, _ = logistic_derivatives(latent, responses, mask, theta, model_dim)
        return likelihood + prior_gradient(latent, family_state)

    def hessian(latent, responses, mask, theta, family_state, a_scale):
        _, likelihood = logistic_derivatives(latent, responses, mask, theta, model_dim)
        return likelihood + prior_hessian(latent, family_state)

    def stabilize(matrix):
        matrix = jnp.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = jnp.linalg.eigh(matrix)
        values = jnp.maximum(values, ridge)
        return (vectors * values) @ vectors.T

    def solve(latent0, responses, mask, theta, family_state, a_scale):
        def update(latent, _):
            score = gradient(latent, responses, mask, theta, family_state, a_scale)
            information = stabilize(
                hessian(latent, responses, mask, theta, family_state, a_scale) + ridge * eye
            )
            step = jnp.linalg.solve(information, score)
            step = jnp.nan_to_num(step, nan=0.0, posinf=2.0, neginf=-2.0)
            # Limit the vector length without changing its descent direction.
            step = step * jnp.minimum(1.0, 2.0 / jnp.maximum(jnp.max(jnp.abs(step)), 1e-12))
            value = item_nll(latent, responses, mask, theta, family_state, a_scale)

            def backtrack(state):
                index, accepted, candidate = state
                trial = jnp.clip(latent - (0.75 * 0.5**index) * step, -10.0, 10.0)
                trial_value = item_nll(trial, responses, mask, theta, family_state, a_scale)
                # Accept only a finite descent step for the conditional mode.
                improves = jnp.isfinite(trial_value) & (trial_value <= value)
                take = (~accepted) & improves
                return index + 1, accepted | improves, jnp.where(take, trial, candidate)

            _, _, candidate = jax.lax.while_loop(
                lambda state: (state[0] < 12) & (~state[1]),
                backtrack,
                (jnp.array(0), jnp.array(False), latent),
            )
            return candidate, None

        latent, _ = jax.lax.scan(update, latent0, xs=None, length=newton_steps)
        information = stabilize(
            hessian(latent, responses, mask, theta, family_state, a_scale) + ridge * eye
        )
        log_determinant = jnp.sum(jnp.log(jnp.linalg.eigvalsh(information)))
        joint_nll = item_nll(latent, responses, mask, theta, family_state, a_scale)
        laplace_nll = joint_nll + 0.5 * (
            log_determinant - model_dim * jnp.log(2.0 * jnp.pi)
        )
        mode_gradient = jnp.max(
            jnp.abs(gradient(latent, responses, mask, theta, family_state, a_scale))
        )
        return latent, laplace_nll, joint_nll, information, mode_gradient

    batched = jax.jit(jax.vmap(solve, in_axes=(0, 0, 0, None, None, None)))

    def run(latent0, item_responses, item_mask, theta, family_state, a_scale):
        latent, laplace, joint, information, mode_gradient = batched(
            jnp.asarray(latent0),
            jnp.asarray(item_responses),
            jnp.asarray(item_mask),
            jnp.asarray(theta),
            family_state,
            jnp.asarray(float(a_scale)),
        )
        return ItemStepResult(
            latent_modes=np.asarray(latent),
            laplace_nll=np.asarray(laplace),
            joint_nll=np.asarray(joint),
            hessian=np.asarray(information),
            mode_gradient_max=np.asarray(mode_gradient),
        )

    _CACHE[key] = run
    return run
