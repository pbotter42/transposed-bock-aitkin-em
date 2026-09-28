# Method

For subject `i` and item `j`, the response model is

$$P(Y_{ij}=1\mid\theta_i,b_j,\alpha_j)
  =\operatorname{logit}^{-1}\{\exp(\alpha_j)(\theta_i-b_j)\}.$$

The 1PL fixes discrimination to one. The 2PL sets `a=exp(alpha)`.
Capabilities `theta` are fixed parameters. Items are independent draws from
a normal population: `b ~ N(0, sigma_b^2)` in the 1PL, or
`(b, alpha) ~ N(0, Sigma)` in the 2PL. A diagonal `Sigma` fixes
`Corr(b, log(a))=0`; a full `Sigma` estimates it. This is not the correlation
between `b` and `a` on the original discrimination scale.

## Why Transposed?

The item-wise marginal likelihood is

$$\ell(\theta,\Sigma)=\sum_j\log\int
  \left\{\prod_{i\in O_j}p_{ij}(z)^{y_{ij}}
  [1-p_{ij}(z)]^{1-y_{ij}}\right\}\phi(z;0,\Sigma)\,dz.$$

`O_j` contains the subjects observed on item `j`. The integration dimension
is one or two, regardless of the number of subjects. Conventional
[Bock-Aitkin EM](https://doi.org/10.1007/BF02293801) integrates over subject
effects; here the item effects are integrated out. Conditional independence
and the normal item-population model are substantive assumptions.

Fixing the item means at zero identifies location and, in the 2PL, scale.
The model does not standardize the fitted capabilities to mean zero and
variance one. The number of items is the replication dimension in the
paper's asymptotic argument; adding more items does not establish that a
small-subject Laplace approximation is accurate.

## Implemented Iteration

1. Find each item's conditional mode by damped Newton steps. Approximate
   its posterior by a Gaussian whose covariance is the inverse regularized
   negative-log-posterior Hessian.
2. Represent that Gaussian by `2d` symmetric cubature nodes with equal
   weights, where `d` is the item-effect dimension. Before clipping, these
   nodes match its mean and covariance. They are not AGHQ nodes.
3. Pool the posterior moments, obtaining a mean `m` and covariance `S`.
   Set `s=exp(m_alpha)` for the 2PL and `s=1` for the 1PL. Transform
   `b*=s(b-m_b)`, `theta*=s(theta-m_b)`, and `alpha*=alpha-m_alpha`.
   This preserves the response probabilities before bounding. The candidate
   population covariance is `diag(s,1) S diag(s,1)` in the 2PL, or `S` in
   the 1PL; diagonal fits retain only the diagonal.
4. With the transformed posterior nodes fixed, solve each subject's
   expected-log-likelihood score equation within `[-6,6]`:

   $$\sum_{j\in O_i}\sum_k w_{jk}a_{jk}
     \{y_{ij}-\operatorname{logit}^{-1}[a_{jk}(\theta_i-b_{jk})]\}=0.$$

   Its derivative is nonpositive. If no interior root exists, choose the
   appropriate bound. Brent's method finds interior roots.
5. Dampen both capability and covariance updates by `damping` (default
   `0.5`), and repeat. Retain the state actually checked at termination,
   not an unchecked extra update.

This expanded-and-normalized moment map is approximate. We do not claim
that it monotonically increases either the exact marginal likelihood or
its Laplace approximation, or that a fixed point solves either score
equation. The reported posterior-averaged Fisher-score residuals are
diagnostics, not derivatives of the complete Laplace log integral.

## Stopping and Safeguards

The default requires five consecutive qualifying iterations, after at least
20 iterations. The undamped maximum changes in `theta` and population
covariance must each be below `2e-4`; RMS prediction change must be below
`1e-5`; the maximum item-mode gradient must be below `1e-4`.
Prediction change is measured on up to 4,096 deterministic observed pairs.
The maximum is 1,000 iterations, with 20 item Newton steps per iteration.
For shorter requested fits, the minimum iteration count is capped at the
requested maximum.

Item modes and cubature nodes are bounded to `[-10,10]`; capabilities are
bounded to `[-6,6]`. Hessian regularization is `1e-4`; the cubature precision
eigenvalues are clipped to `[1e-4,1e8]`. Population covariance stabilization
uses a `1e-6` floor. Clipping changes the nominal Gaussian moments and is
reported through boundary diagnostics. These are numerical safeguards,
not extra model parameters or evidence of successful estimation.

An iteration limit returns estimates with `stopping_reason_="iteration_limit"`.
A nonfinite posterior step raises an exception and leaves no fitted result.
No failed fit is silently retried. A small score residual, a stable moment
map, and an accurate integral are different numerical properties.

## New Subjects and Uncertainty

`score_subjects` holds the fitted item posterior bank fixed and maximizes
the sum of logs of the posterior-averaged response probabilities. That
objective differs from the expected log likelihood inside `fit`.
It uses a bounded scalar search, without a global-optimum guarantee, and
requires responses in the original item order. It is for new subjects on
the existing items, not calibration of new items.

`theta_` is a point estimate. `item_frame()` exports conditional modes in
latent coordinates and their transforms; `a=exp(alpha_mode)` is not
`E[a | Y]`. The population covariance is `family_moments_.raw_cov`, in
the order `(b, log(a))`. The package does not report parameter standard
errors, confidence intervals, or coverage. The paper's item-cluster
predictive bootstrap is a separate analysis and does not provide those
parameter intervals.
