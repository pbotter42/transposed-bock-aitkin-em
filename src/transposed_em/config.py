"""Controls used by the preprint's Laplace moment map."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class FitConfig:
    model: str = "2pl"
    covariance: str = "full"
    max_iterations: int = 1000
    min_iterations: int = 20
    convergence_patience: int = 5
    item_newton_steps: int = 20
    theta_bounds: tuple[float, float] = (-6.0, 6.0)
    ridge: float = 1e-4

    def __post_init__(self):
        if self.model not in {"1pl", "2pl"}:
            raise ValueError("model must be '1pl' or '2pl'")
        if self.covariance not in {"diagonal", "full"}:
            raise ValueError("covariance must be 'diagonal' or 'full'")
        if self.model == "1pl" and self.covariance != "diagonal":
            raise ValueError("The 1PL has one item coordinate; use diagonal covariance")
        for name in (
            "max_iterations",
            "min_iterations",
            "convergence_patience",
            "item_newton_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations cannot exceed max_iterations")

    @property
    def dimension(self):
        return 1 if self.model == "1pl" else 2
