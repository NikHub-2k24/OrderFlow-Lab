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


def test_hawkes_params_finite_horizon_expected_count():
    # Symmetric parameters: mu=20, a=10, beta=20 -> eta=0.5, gamma=10
    # Over T=1.0: E[N(1)] = (20/0.5)*1 - (20*0.5/0.5)*(1 - exp(-10))/10 = 40 - 20*(1 - exp(-10))/10 = 40 - 2 = 38
    p = HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0)
    cnt = p.expected_count_per_side(t_max=1.0)
    assert cnt == pytest.approx(38.0, rel=1e-4)

    # Matched rate
    rate = p.finite_horizon_matched_rate_per_side(t_max=1.0)
    assert rate == pytest.approx(38.0, rel=1e-4)
    # Stationary rate is 40.0
    assert p.stationary_mean_rate_per_side == pytest.approx(40.0)
    assert rate < p.stationary_mean_rate_per_side

    # Zero excitation: eta = 0
    p_zero = HawkesParams(mu=15.0, alpha_self=0.0, alpha_cross=0.0, beta=10.0)
    assert p_zero.expected_count_per_side(t_max=2.0) == pytest.approx(30.0)
    assert p_zero.finite_horizon_matched_rate_per_side(t_max=2.0) == pytest.approx(15.0)

    # Non-positive horizon edge cases
    assert p.expected_count_per_side(t_max=0.0) == 0.0
    assert p.finite_horizon_matched_rate_per_side(t_max=0.0) == p.mu

