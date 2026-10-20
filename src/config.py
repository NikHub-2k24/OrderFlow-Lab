"""Configuration schemas and input validation for OrderFlow Lab."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class HawkesParams:
    """Parameters for bivariate Hawkes process with exponential kernel.

    lambda_buy(t)  = mu + alpha_self * x_buy(t)  + alpha_cross * x_sell(t)
    lambda_sell(t) = mu + alpha_self * x_sell(t) + alpha_cross * x_buy(t)

    Time domain is normalized to t in [0, 1]. All rate/decay parameters are
    expressed per simulation session.
    """

    mu: float = 30.0  # Base intensity per side per session
    alpha_self: float = 8.0  # Self-excitation coefficient
    alpha_cross: float = 4.0  # Cross-excitation coefficient
    beta: float = 20.0  # Exponential decay rate per session

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"Baseline intensity mu must be strictly positive (> 0), got {self.mu}.")
        if self.beta <= 0:
            raise ValueError(f"Decay parameter beta must be strictly positive (> 0), got {self.beta}.")
        if self.alpha_self < 0:
            raise ValueError(f"Self-excitation alpha_self must be non-negative (>= 0), got {self.alpha_self}.")
        if self.alpha_cross < 0:
            raise ValueError(f"Cross-excitation alpha_cross must be non-negative (>= 0), got {self.alpha_cross}.")

        ratio = (self.alpha_self + self.alpha_cross) / self.beta
        if ratio >= 1.0:
            raise ValueError(
                f"Hawkes process is unstable: branching ratio (alpha_self + alpha_cross) / beta = "
                f"({self.alpha_self} + {self.alpha_cross}) / {self.beta} = {ratio:.4f} >= 1.0. "
                f"System must be subcritical (branching ratio < 1.0) to guarantee stationary finite intensity."
            )

    @property
    def branching_ratio(self) -> float:
        """Total branching ratio: (alpha_self + alpha_cross) / beta."""
        return (self.alpha_self + self.alpha_cross) / self.beta

    @property
    def theoretical_mean_rate_per_side(self) -> float:
        """Theoretical stationary mean event intensity per side: mu / (1 - branching_ratio)."""
        return self.mu / (1.0 - self.branching_ratio)

    @property
    def theoretical_total_rate(self) -> float:
        """Theoretical total event intensity across both sides."""
        return 2.0 * self.theoretical_mean_rate_per_side


@dataclass(frozen=True)
class MarketParams:
    """Parameters for synthetic reference-price evolution."""

    s0: float = 100.0  # Initial reference mid-price
    sigma: float = 0.05  # Log-price Brownian volatility per sqrt(session)
    impact_bps: float = 2.0  # Signed price impact per order event in basis points

    def __post_init__(self) -> None:
        if self.s0 <= 0:
            raise ValueError(f"Initial price s0 must be strictly positive (> 0), got {self.s0}.")
        if self.sigma < 0:
            raise ValueError(f"Volatility sigma must be non-negative (>= 0), got {self.sigma}.")
        if self.impact_bps < 0:
            raise ValueError(f"Impact impact_bps must be non-negative (>= 0), got {self.impact_bps}.")

    @property
    def impact_log_increment(self) -> float:
        """Log-price increment corresponding to one basis point impact: impact_bps / 10000."""
        return self.impact_bps / 10_000.0


@dataclass(frozen=True)
class PolicyParams:
    """Parameters governing dealer quoting policies, fees, and execution."""

    half_spread_bps: float = 5.0  # Baseline half-spread from mid in basis points
    max_inventory: int = 10  # Hard absolute inventory limit |q| <= max_inventory
    p0: float = 0.8  # Reference fill probability at zero distance
    kappa: float = 0.15  # Fill probability decay rate with respect to quote distance in bps
    fee_bps: float = 0.5  # Transaction fee in basis points of notional
    inventory_skew_bps: float = 1.0  # Quote skew per unit of signed inventory (Policy B)

    def __post_init__(self) -> None:
        if self.half_spread_bps <= 0:
            raise ValueError(f"half_spread_bps must be strictly positive (> 0), got {self.half_spread_bps}.")
        if self.max_inventory < 1:
            raise ValueError(f"max_inventory must be at least 1, got {self.max_inventory}.")
        if not (0.0 < self.p0 <= 1.0):
            raise ValueError(f"Reference fill probability p0 must be in (0, 1], got {self.p0}.")
        if self.kappa < 0:
            raise ValueError(f"kappa must be non-negative (>= 0), got {self.kappa}.")
        if self.fee_bps < 0:
            raise ValueError(f"fee_bps must be non-negative (>= 0), got {self.fee_bps}.")
        if self.inventory_skew_bps < 0:
            raise ValueError(f"inventory_skew_bps must be non-negative (>= 0), got {self.inventory_skew_bps}.")

    @property
    def max_skew_bps(self) -> float:
        """Maximum allowable inventory skew: 0.8 * half_spread_bps."""
        return 0.8 * self.half_spread_bps


@dataclass(frozen=True)
class ExperimentConfig:
    """Full simulation setup configuration."""

    hawkes: HawkesParams = field(default_factory=HawkesParams)
    market: MarketParams = field(default_factory=MarketParams)
    policy: PolicyParams = field(default_factory=PolicyParams)
    n_sessions: int = 200  # Number of Monte Carlo sessions
    seed: int = 42
    t_max: float = 1.0

    def __post_init__(self) -> None:
        if self.n_sessions < 1:
            raise ValueError(f"n_sessions must be at least 1, got {self.n_sessions}.")
        if self.t_max <= 0:
            raise ValueError(f"Session horizon t_max must be positive, got {self.t_max}.")
