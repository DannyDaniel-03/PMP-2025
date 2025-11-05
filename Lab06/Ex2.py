import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class PosteriorSummary:
    alpha_post: float
    beta_post: float
    mean: float
    std: float
    map: float
    hdi_low: float
    hdi_high: float


def gamma_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Gamma PDF with 'rate' parameterization: f(x) = beta^alpha / Gamma(alpha) * x^(alpha-1) * e^(-beta x)
    Defined for x >= 0.
    """
    # Use log-domain for numerical stability, then exp back
    log_pdf = alpha * math.log(beta) - math.lgamma(alpha) + (alpha - 1.0) * np.log(x, where=(x > 0), out=np.full_like(x, -np.inf)) - beta * x
    # For x == 0 and alpha < 1, log_pdf is -inf as expected; np.exp handles it.
    return np.exp(log_pdf)


def gamma_mode(alpha: float, beta: float) -> float:
    """
    Mode (MAP for a Gamma prior/posterior) in 'rate' form: (alpha - 1)/beta, valid for alpha > 1; else 0.
    """
    return (alpha - 1.0) / beta if alpha > 1.0 else 0.0


def gamma_hdi(alpha: float, beta: float, cred_mass: float = 0.94, grid_size: int = 200_000) -> Tuple[float, float]:
    """
    Numerically approximate the Highest Density Interval (HDI) for a Gamma(alpha, beta) using a dense grid.
    No SciPy dependency. Assumes the Gamma is unimodal (alpha > 1 gives strict unimodality; for alpha <= 1,
    this procedure still returns a contiguous interval).
    """
    # Moments to set a generous finite grid upper bound
    mean = alpha / beta
    std = math.sqrt(alpha) / beta

    # Upper bound: large enough to capture ~all mass (conservative)
    upper = max(5.0 * mean, mean + 40.0 * std, 20.0)  # ensures a safe cap even for tiny means

    xs = np.linspace(0.0, upper, grid_size, dtype=np.float64)
    dx = xs[1] - xs[0]

    pdf = gamma_pdf(xs, alpha, beta)

    # Normalize discrete mass (so sum(pdf) * dx ≈ 1.0)
    total_mass = np.sum(pdf) * dx
    if total_mass == 0.0 or not np.isfinite(total_mass):
        raise RuntimeError("Failed to normalize posterior density; try increasing grid_size or upper bound.")

    # Sort by density (descending) to find the smallest density threshold covering cred_mass
    order = np.argsort(-pdf)
    pdf_sorted = pdf[order]
    xs_sorted = xs[order]

    cummass = np.cumsum(pdf_sorted) * dx
    k = np.searchsorted(cummass, cred_mass)  # first index where cumulative mass >= cred_mass
    chosen = xs_sorted[: max(1, k)]

    # HDI is the narrowest interval with that mass for unimodal densities -> min/max of top-density set
    hdi_low = float(np.min(chosen))
    hdi_high = float(np.max(chosen))
    return hdi_low, hdi_high


def solve_ex2(
    sum_counts: int = 180,
    n_hours: int = 10,
    alpha0: Optional[float] = None,
    beta0: Optional[float] = None,
    *,
    # Alternative prior specification via prior mean m0 and "equivalent hours" h0 (alpha0 = m0*h0, beta0 = h0)
    m0: Optional[float] = 18.0,
    h0: Optional[float] = 1.0,
    cred_mass: float = 0.94,
) -> PosteriorSummary:
    """
    Compute the Gamma–Poisson posterior for λ given:
      - Data: sum of counts over n hours
      - Prior: either (alpha0, beta0) directly, OR via (m0, h0) where alpha0=m0*h0, beta0=h0 (equivalent-hours trick)

    Returns posterior parameters and the 94% HDI + MAP.
    """
    if alpha0 is None or beta0 is None:
        if m0 is None or h0 is None:
            raise ValueError("Provide either (alpha0, beta0) or (m0, h0).")
        alpha0 = float(m0) * float(h0)
        beta0 = float(h0)

    alpha_post = alpha0 + float(sum_counts)
    beta_post = beta0 + float(n_hours)

    mean = alpha_post / beta_post
    std = math.sqrt(alpha_post) / beta_post
    mode = gamma_mode(alpha_post, beta_post)
    hdi_low, hdi_high = gamma_hdi(alpha_post, beta_post, cred_mass=cred_mass)

    return PosteriorSummary(
        alpha_post=alpha_post,
        beta_post=beta_post,
        mean=mean,
        std=std,
        map=mode,
        hdi_low=hdi_low,
        hdi_high=hdi_high,
    )


if __name__ == "__main__":
    # Example usage for the exercise:
    # Data: 180 calls over 10 hours
    # Prior: set via (m0, h0). Here we choose a weakly informative prior of 1 "equivalent hour" centered at 18 calls/hour.
    result = solve_ex2(sum_counts=180, n_hours=10, m0=18.0, h0=1.0, cred_mass=0.94)

    print("Posterior λ | data ~ Gamma(alpha_post, beta_post) with 'rate' β")
    print(f"  alpha_post = {result.alpha_post:.6g}")
    print(f"  beta_post  = {result.beta_post:.6g}")
    print()
    print(f"Posterior mean  = {result.mean:.6f} calls/hour")
    print(f"Posterior std   = {result.std:.6f} calls/hour")
    print(f"MAP (mode)      = {result.map:.6f} calls/hour")
    print(f"94% HDI         = [{result.hdi_low:.6f}, {result.hdi_high:.6f}] calls/hour")
