from __future__ import annotations

import numpy as np


def laplace_cubature(
    modes: np.ndarray,
    hessians: np.ndarray,
    eigenvalue_floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a moment-matching cubature rule for Gaussian Laplace posteriors."""

    modes = np.asarray(modes, dtype=float)
    hessians = np.asarray(hessians, dtype=float)
    n_items, dimension = modes.shape
    nodes = np.empty((n_items, 2 * dimension, dimension), dtype=float)
    radius = np.sqrt(float(dimension))
    symmetric = 0.5 * (hessians + hessians.swapaxes(1, 2))
    symmetric = np.nan_to_num(symmetric, nan=0.0, posinf=1e8, neginf=-1e8)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, eigenvalue_floor, 1e8)
    covariances = (eigenvectors / eigenvalues[:, None, :]) @ eigenvectors.swapaxes(1, 2)
    covariances = 0.5 * (covariances + covariances.swapaxes(1, 2))
    offsets = radius * (eigenvectors / np.sqrt(eigenvalues[:, None, :])).swapaxes(1, 2)
    nodes[:, :dimension] = modes[:, None, :] + offsets
    nodes[:, dimension:] = modes[:, None, :] - offsets

    weights = np.full(2 * dimension, 1.0 / (2.0 * dimension), dtype=float)
    nodes = np.nan_to_num(nodes, nan=0.0, posinf=10.0, neginf=-10.0)
    covariances = np.nan_to_num(covariances, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(nodes, -10.0, 10.0), covariances, weights
