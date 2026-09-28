"""Small synthetic example, not a replication of the paper's simulation."""

import numpy as np
from scipy.special import expit

from transposed_em import LaplaceIRT

rng = np.random.default_rng(42)
theta = np.linspace(-1.5, 1.5, 12)
b = rng.normal(size=80)
a = np.exp(rng.normal(0, 0.3, size=80))
responses = rng.binomial(1, expit(a * (theta[:, None] - b)))

fit = LaplaceIRT(model="2pl", covariance="full", max_iterations=40).fit(responses)
print(fit.subject_frame().to_string(index=False))
print("Stopping reason:", fit.stopping_reason_)
print("Population covariance (b, log(a)):")
print(fit.family_moments_.raw_cov)
print("Final diagnostics:", fit.diagnostics_)
