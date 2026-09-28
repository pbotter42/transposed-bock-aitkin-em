from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import FitConfig
from .common import FamilyMoments


class RandomItemFamily(ABC):
    name = "base"

    def __init__(self, dim: int, config: FitConfig):
        self.dim = int(dim)
        self.config = config

    @abstractmethod
    def fit(self, latent: np.ndarray) -> RandomItemFamily:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def moments(self) -> FamilyMoments:
        raise NotImplementedError

    @abstractmethod
    def affine_normalize_b(self, shift: float, scale: float) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def logpdf_jax(z, state):
        raise NotImplementedError
