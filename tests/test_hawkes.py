"""Unit tests for bivariate Hawkes process and matched Poisson simulation."""

import math
import numpy as np
import pytest
from src.config import HawkesParams
from src.hawkes import (
    simulate_hawkes_bivariate,
    simulate_poisson_matched,
    compute_continuous_intensities,
    EVENT_BUY,
    EVENT_SELL,
)


def test_hawkes_intensities_nonnegative_and_ordered():
    params = HawkesParams(mu=30.0, alpha_self=10.0, alpha_cross=5.0, beta=25.0)
    events = simulate_hawkes_bivariate(params, t_max=1.0, rng=42)

    assert len(events) > 0
    # Check bounds
    assert np.all(events.timestamps >= 0.0)
    assert np.all(events.timestamps <= 1.0)

    # Check strict ascending order
    diffs = np.diff(events.timestamps)
    assert np.all(diffs >= 0.0)

    # Check intensities are strictly positive (>= mu)
    assert np.all(events.lambda_buy >= params.mu)
    assert np.all(events.lambda_sell >= params.mu)

    # Check valid event types
    valid_types = {EVENT_BUY, EVENT_SELL}
    assert set(np.unique(events.event_types)).issubset(valid_types)


def test_hawkes_intensity_decays_between_events():
    params = HawkesParams(mu=20.0, alpha_self=8.0, alpha_cross=4.0, beta=15.0)
    events = simulate_hawkes_bivariate(params, t_max=1.0, rng=123)

    if len(events) >= 2:
        t1 = events.timestamps[0]
        t2 = events.timestamps[1]
        grid = np.linspace(t1 + 1e-4, t2 - 1e-4, 10)
        lb, ls = compute_continuous_intensities(events, params, grid)

        # Between events, state must decay monotonically towards mu
        assert np.all(np.diff(lb) <= 1e-9)
        assert np.all(np.diff(ls) <= 1e-9)


def test_finite_horizon_expected_count_formula():
    # mu=20, alpha_self=5, alpha_cross=5, beta=20
    # a = 10, eta = 0.5, gamma = 10
    # E[N(1.0)] = (20 / 0.5) * 1 - (20 * 0.5 / 0.5) * (1 - exp(-10)) / 10
    #           = 40 - 20 * (1 - exp(-10)) / 10 = 40 - 2 * (1 - exp(-10))
    params = HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0)
    expected_side = params.expected_count_per_side(t_max=1.0)
    analytical = 40.0 - 2.0 * (1.0 - math.exp(-10.0))
    assert expected_side == pytest.approx(analytical, rel=1e-9)

    matched_rate = params.finite_horizon_matched_rate_per_side(t_max=1.0)
    assert matched_rate == pytest.approx(expected_side / 1.0, rel=1e-9)


def test_zero_excitation_case():
    # When alpha = 0 (eta = 0), process reduces to homogeneous Poisson with rate mu
    params = HawkesParams(mu=25.0, alpha_self=0.0, alpha_cross=0.0, beta=10.0)
    assert params.branching_ratio == 0.0

    for t_h in [0.25, 0.5, 1.0, 2.5]:
        exp_count = params.expected_count_per_side(t_max=t_h)
        assert exp_count == pytest.approx(25.0 * t_h, rel=1e-9)
        matched_rate = params.finite_horizon_matched_rate_per_side(t_max=t_h)
        assert matched_rate == pytest.approx(25.0, rel=1e-9)


def test_multiple_session_horizons():
    params = HawkesParams(mu=20.0, alpha_self=8.0, alpha_cross=4.0, beta=20.0)
    # a = 12, eta = 12/20 = 0.6, gamma = 8.0, stationary rate = 20 / 0.4 = 50.0
    for t_h in [0.1, 0.5, 1.0, 2.0, 5.0]:
        exp_count = params.expected_count_per_side(t_max=t_h)
        rate = params.finite_horizon_matched_rate_per_side(t_max=t_h)
        # Expected count must be strictly positive and strictly less than stationary * t_h
        assert 0.0 < exp_count < params.stationary_mean_rate_per_side * t_h
        # Rate must lie between mu and stationary rate
        assert params.mu <= rate < params.stationary_mean_rate_per_side


def test_monte_carlo_convergence_to_finite_horizon():
    # Verify that simulated average counts for BOTH Hawkes and Poisson match E[N(T)]
    params = HawkesParams(mu=20.0, alpha_self=6.0, alpha_cross=4.0, beta=20.0)
    t_max = 1.0
    expected_per_side = params.expected_count_per_side(t_max=t_max)
    expected_total = 2.0 * expected_per_side

    n_runs = 400
    hawkes_counts = []
    poisson_counts = []
    rng = np.random.default_rng(2026)

    for _ in range(n_runs):
        ev_h = simulate_hawkes_bivariate(params, t_max=t_max, rng=rng)
        ev_p = simulate_poisson_matched(params, t_max=t_max, rng=rng)
        hawkes_counts.append(len(ev_h))
        poisson_counts.append(len(ev_p))

    mean_h = float(np.mean(hawkes_counts))
    mean_p = float(np.mean(poisson_counts))

    # Standard error of mean is roughly std / sqrt(N) ~ 10 / 20 = 0.5
    # Both should match the theoretical finite-horizon expectation within 3 standard errors (~2.0)
    assert abs(mean_h - expected_total) < 2.5, f"Hawkes mean {mean_h} diverged from theory {expected_total}"
    assert abs(mean_p - expected_total) < 2.5, f"Poisson mean {mean_p} diverged from theory {expected_total}"
    assert abs(mean_h - mean_p) < 2.5, f"Hawkes {mean_h} and Poisson {mean_p} diverged from each other"


def test_reproducibility_with_seed():
    params = HawkesParams(mu=25.0, alpha_self=7.0, alpha_cross=3.0, beta=20.0)

    ev1 = simulate_hawkes_bivariate(params, t_max=1.0, rng=42)
    ev2 = simulate_hawkes_bivariate(params, t_max=1.0, rng=42)

    np.testing.assert_array_equal(ev1.timestamps, ev2.timestamps)
    np.testing.assert_array_equal(ev1.event_types, ev2.event_types)
    np.testing.assert_allclose(ev1.lambda_buy, ev2.lambda_buy)
