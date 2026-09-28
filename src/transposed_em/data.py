from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import logit


class ResponseMatrix:
    """Validated subject-by-item binary response matrix.

    Missing responses may be represented by ``numpy.nan`` or ``-1``.
    """

    def __init__(
        self,
        data,
        subject_ids: Sequence[str] | None = None,
        item_ids: Sequence[str] | None = None,
    ) -> None:
        array = np.asarray(data, dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"responses must be two-dimensional; got {array.ndim} dimensions"
            )
        array = np.where(array == -1, np.nan, array)
        if np.isinf(array).any():
            raise ValueError(
                "infinite responses are invalid; use NaN or -1 for missing data"
            )
        observed = array[np.isfinite(array)]
        if observed.size == 0:
            raise ValueError("responses contain no observed values")
        if not np.all(np.isin(observed, (0.0, 1.0))):
            raise ValueError("observed responses must be 0 or 1")
        if np.any(np.sum(np.isfinite(array), axis=1) == 0):
            raise ValueError("every subject must have at least one observed response")
        if np.any(np.sum(np.isfinite(array), axis=0) == 0):
            raise ValueError("every item must have at least one observed response")
        self._data = array
        self.subject_ids = _validated_ids(subject_ids, array.shape[0], "subject_ids")
        self.item_ids = _validated_ids(item_ids, array.shape[1], "item_ids")

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> ResponseMatrix:
        return cls(frame, frame.index.astype(str), frame.columns.astype(str))

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def n_subjects(self) -> int:
        return self.shape[0]

    @property
    def n_items(self) -> int:
        return self.shape[1]

    @property
    def density(self) -> float:
        return float(np.mean(np.isfinite(self._data)))

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self._data.copy() if copy else self._data

    def __array__(self, dtype=None, copy=None):
        array = self.to_numpy(copy=copy is not False)
        return array.astype(dtype) if dtype is not None else array

    def __repr__(self) -> str:
        return (
            f"ResponseMatrix(n_subjects={self.n_subjects}, n_items={self.n_items}, "
            f"density={self.density:.3f})"
        )


def coerce_responses(data) -> ResponseMatrix:
    if isinstance(data, ResponseMatrix):
        return data
    if isinstance(data, pd.DataFrame):
        return ResponseMatrix.from_dataframe(data)
    return ResponseMatrix(data)


def item_major(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.nan_to_num(matrix.T, nan=0.0).astype(np.float64),
        np.isfinite(matrix.T).astype(np.float64),
    )


def theta_start(matrix: np.ndarray) -> np.ndarray:
    means = np.clip(np.nanmean(matrix, axis=1), 1e-4, 1.0 - 1e-4)
    theta = logit(means)
    theta -= np.mean(theta)
    scale = np.std(theta)
    return theta / scale if np.isfinite(scale) and scale > 0 else theta


def _validated_ids(values: Sequence[str] | None, size: int, name: str) -> list[str] | None:
    if values is None:
        return None
    result = [str(value) for value in values]
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result
