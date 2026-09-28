from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ..config import FitConfig
from .base import RandomItemFamily
from .common import (
    FamilyMoments,
    build_cholesky,
    build_cholesky_jax,
    covariance_to_correlation,
    gaussian_start,
    pack_cholesky,
    stabilize_covariance,
)


@dataclass
class NormalState:
    loc: np.ndarray
    raw_diag: np.ndarray
    raw_offdiag: np.ndarray


class NormalFamily(RandomItemFamily):
    name = "normal"

    def __init__(self, dim: int, config: FitConfig):
        super().__init__(dim=dim, config=config)
        self.state: NormalState | None = None

    @staticmethod
    def logpdf_jax(z, state):
        loc = state["loc"]
        raw_diag = state["raw_diag"]
        raw_offdiag = state["raw_offdiag"]
        dim = loc.shape[0]
        chol = build_cholesky_jax(raw_diag, raw_offdiag, dim)
        centered = z - loc
        solve = jsp.linalg.solve_triangular(chol, centered, lower=True)
        quad = jnp.dot(solve, solve)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + logdet + quad)

    def fit(self, latent: np.ndarray) -> NormalFamily:
        mean, cov = gaussian_start(latent)
        if self.config.covariance == "diagonal":
            cov = np.diag(np.diag(cov))
        chol = np.linalg.cholesky(stabilize_covariance(cov))
        raw_diag, raw_offdiag = pack_cholesky(chol)
        self.state = NormalState(
            loc=np.asarray(mean, dtype=float),
            raw_diag=np.asarray(raw_diag, dtype=float),
            raw_offdiag=np.asarray(raw_offdiag, dtype=float),
        )
        return self

    def fit_weighted(self, nodes: np.ndarray, weights: np.ndarray) -> NormalFamily:
        """Update Gaussian moments from item-specific posterior quadrature."""

        nodes = np.asarray(nodes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if nodes.ndim != 3 or weights.shape != nodes.shape[:2]:
            raise ValueError("nodes and weights have incompatible shapes")
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        item_means = np.einsum("jk,jkd->jd", weights, nodes)
        mean = np.mean(item_means, axis=0)
        second = np.einsum("jk,jkd,jke->de", weights, nodes, nodes) / nodes.shape[0]
        cov = stabilize_covariance(second - np.outer(mean, mean))
        if self.config.covariance == "diagonal":
            cov = np.diag(np.diag(cov))
        chol = np.linalg.cholesky(stabilize_covariance(cov))
        raw_diag, raw_offdiag = pack_cholesky(chol)
        self.state = NormalState(mean, raw_diag, raw_offdiag)
        return self

    def covariance(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Family has not been fit yet.")
        chol = build_cholesky(self.state.raw_diag, self.state.raw_offdiag, self.dim)
        cov = stabilize_covariance(chol @ chol.T)
        if self.config.covariance == "diagonal":
            cov = np.diag(np.diag(cov))
        return cov

    def moments(self) -> FamilyMoments:
        if self.state is None:
            raise RuntimeError("Family has not been fit yet.")
        cov = self.covariance()
        corr = covariance_to_correlation(cov)
        return FamilyMoments(
            raw_mean=self.state.loc.copy(),
            raw_cov=cov,
            raw_corr=corr,
            b_center=float(self.state.loc[0]),
            a_scale=1.0,
        )

    def state_dict(self):
        if self.state is None:
            raise RuntimeError("Family has not been fit yet.")
        return {
            "loc": jnp.asarray(self.state.loc),
            "raw_diag": jnp.asarray(self.state.raw_diag),
            "raw_offdiag": jnp.asarray(self.state.raw_offdiag),
        }

    def affine_normalize_b(self, shift: float, scale: float) -> None:
        if self.state is None:
            raise RuntimeError("Family has not been fit yet.")
        cov = self.covariance()
        transform = np.eye(self.dim)
        transform[0, 0] = scale
        loc = self.state.loc.copy()
        loc[0] = scale * (loc[0] - shift)
        cov_new = transform @ cov @ transform.T
        if self.config.covariance == "diagonal":
            cov_new = np.diag(np.diag(cov_new))
        chol = np.linalg.cholesky(stabilize_covariance(cov_new))
        raw_diag, raw_offdiag = pack_cholesky(chol)
        self.state = NormalState(
            loc=loc,
            raw_diag=raw_diag,
            raw_offdiag=raw_offdiag,
        )
