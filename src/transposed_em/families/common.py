from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class FamilyMoments:
    raw_mean: np.ndarray
    raw_cov: np.ndarray
    raw_corr: np.ndarray
    b_center: float
    a_scale: float


def stabilize_covariance(cov: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=-1e6)
    cov = 0.5 * (cov + cov.T)
    try:
        evals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        diag = np.diag(np.maximum(np.diag(cov), ridge))
        cov = diag
        evals = np.linalg.eigvalsh(cov)
    min_eval = float(np.min(evals))
    if min_eval < ridge:
        cov = cov + np.eye(cov.shape[0]) * (ridge - min_eval + ridge)
    return cov


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(sd, sd)
    corr[np.diag_indices_from(corr)] = 1.0
    return corr


def build_cholesky(raw_diag: np.ndarray, raw_offdiag: np.ndarray, dim: int) -> np.ndarray:
    chol = np.zeros((dim, dim), dtype=float)
    diag = np.exp(np.asarray(raw_diag, dtype=float))
    idx = 0
    for i in range(dim):
        chol[i, i] = diag[i]
        for j in range(i):
            chol[i, j] = raw_offdiag[idx]
            idx += 1
    return chol


def build_cholesky_jax(
    raw_diag: jnp.ndarray, raw_offdiag: jnp.ndarray, dim: int
) -> jnp.ndarray:
    chol = jnp.zeros((dim, dim))
    idx = 0
    for i in range(dim):
        chol = chol.at[i, i].set(jnp.exp(raw_diag[i]))
        for j in range(i):
            chol = chol.at[i, j].set(raw_offdiag[idx])
            idx += 1
    return chol


def pack_cholesky(chol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dim = chol.shape[0]
    diag = np.log(np.maximum(np.diag(chol), 1e-8))
    offdiag = []
    for i in range(dim):
        for j in range(i):
            offdiag.append(chol[i, j])
    return diag, np.asarray(offdiag, dtype=float)


def gaussian_start(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latent = np.asarray(latent, dtype=float)
    latent = np.nan_to_num(latent, nan=0.0, posinf=10.0, neginf=-10.0)
    latent = np.clip(latent, -10.0, 10.0)
    mean = np.mean(latent, axis=0)
    cov_raw = np.cov(latent, rowvar=False)
    if np.ndim(cov_raw) == 0:
        cov_raw = np.array([[float(cov_raw)]], dtype=float)
    cov = stabilize_covariance(cov_raw)
    return mean, cov
