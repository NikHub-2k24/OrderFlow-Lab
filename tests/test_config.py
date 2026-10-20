"""Tests for configuration schemas and parameter validation."""

import pytest
from src.config import HawkesParams, MarketParams, PolicyParams, ExperimentConfig


def test_hawkes_params_valid():
    p = HawkesParams(mu=20.0, alpha_self=6.0, alpha_cross=4.0, beta=15.0)
    assert p.branching_ratio == pytest.approx(10.0 / 15.0)
    assert p.theoretical_mean_rate_per_side == pytest.approx(20.0 / (1.0 - (10.0 / 15.0)))
    assert p.theoretical_total_rate == pytest.approx(2.0 * p.theoretical_mean_rate_per_side)


def test_hawkes_params_rejects_unstable():
    # Exactly critical (ratio == 1.0)
    with pytest.raises(ValueError, match="Hawkes process is unstable"):
        HawkesParams(mu=10.0, alpha_self=5.0, alpha_cross=5.0, beta=10.0)

    # Supercritical (ratio > 1.0)
    with pytest.raises(ValueError, match="Hawkes process is unstable"):
        HawkesParams(mu=10.0, alpha_self=8.0, alpha_cross=6.0, beta=10.0)


def test_hawkes_params_rejects_non_positive_baseline():
    with pytest.raises(ValueError, match="mu must be strictly positive"):
        HawkesParams(mu=0.0)

    with pytest.raises(ValueError, match="mu must be strictly positive"):
        HawkesParams(mu=-5.0)


def test_hawkes_params_rejects_non_positive_beta():
    with pytest.raises(ValueError, match="beta must be strictly positive"):
        HawkesParams(beta=0.0)

    with pytest.raises(ValueError, match="beta must be strictly positive"):
        HawkesParams(beta=-10.0)


def test_market_params_validation():
    p = MarketParams(s0=100.0, sigma=0.05, impact_bps=2.0)
    assert p.s0 == 100.0
    assert p.impact_log_increment == pytest.approx(2.0 / 10_000.0)

    with pytest.raises(ValueError, match="s0 must be strictly positive"):
        MarketParams(s0=0.0)

    with pytest.raises(ValueError, match="sigma must be non-negative"):
        MarketParams(sigma=-0.01)


def test_policy_params_validation():
    p = PolicyParams(half_spread_bps=5.0, max_inventory=10, p0=0.8, kappa=0.15, fee_bps=0.5)
    assert p.max_skew_bps == pytest.approx(0.8 * 5.0)

    with pytest.raises(ValueError, match="half_spread_bps must be strictly positive"):
        PolicyParams(half_spread_bps=0.0)

    with pytest.raises(ValueError, match="max_inventory must be at least 1"):
        PolicyParams(max_inventory=0)

    with pytest.raises(ValueError, match="p0 must be in"):
        PolicyParams(p0=0.0)

    with pytest.raises(ValueError, match="p0 must be in"):
        PolicyParams(p0=1.2)
