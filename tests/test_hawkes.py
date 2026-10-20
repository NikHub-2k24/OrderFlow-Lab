"""Unit tests for bivariate Hawkes process and matched Poisson simulation."""

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
        # Check intensity decay on a fine grid between first two events
        t1 = events.timestamps[0]
        t2 = events.timestamps[1]
        grid = np.linspace(t1 + 1e-4, t2 - 1e-4, 10)
        lb, ls = compute_continuous_intensities(events, params, grid)

        # Between events, state must decay monotonically towards mu
        assert np.all(np.diff(lb) <= 1e-9)
        assert np.all(np.diff(ls) <= 1e-9)


def test_hawkes_mean_intensity_matches_theory():
    # mu = 25, branching ratio = (6 + 4) / 20 = 0.5
    # Theoretical rate per side = 25 / (1 - 0.5) = 50.0
    # Expected total events per session (t_max=1.0) = 100.0
    params = HawkesParams(mu=25.0, alpha_self=6.0, alpha_cross=4.0, beta=20.0)
    assert params.branching_ratio == 0.5
    assert params.theoretical_mean_rate_per_side == 50.0
    assert params.theoretical_total_rate == 100.0

    n_runs = 350
    counts = []
    rng = np.random.default_rng(999)

    for _ in range(n_runs):
        ev = simulate_hawkes_bivariate(params, t_max=1.0, rng=rng)
        counts.append(len(ev))

    empirical_mean = np.mean(counts)
    # Hawkes with stationary zero-state start exhibits slight ramp-up from mu to asymptotic rate,
    # so finite-horizon [0, 1] mean is slightly lower than asymptotic steady-state (typically ~90-100).
    # Check that empirical mean is within statistical bounds of theoretical expectation.
    assert 80.0 <= empirical_mean <= 110.0


def test_poisson_matched_mean():
    params = HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0)
    # Branching ratio = 10 / 20 = 0.5, mean rate per side = 40.0, total = 80.0
    assert params.theoretical_total_rate == 80.0

    n_runs = 200
    counts = []
    rng = np.random.default_rng(777)

    for _ in range(n_runs):
        ev = simulate_poisson_matched(params, t_max=1.0, rng=rng)
        counts.append(len(ev))

    empirical_mean = np.mean(counts)
    assert 74.0 <= empirical_mean <= 86.0


def test_reproducibility_with_seed():
    params = HawkesParams(mu=25.0, alpha_self=7.0, alpha_cross=3.0, beta=20.0)

    ev1 = simulate_hawkes_bivariate(params, t_max=1.0, rng=42)
    ev2 = simulate_hawkes_bivariate(params, t_max=1.0, rng=42)

    np.testing.assert_array_equal(ev1.timestamps, ev2.timestamps)
    np.testing.assert_array_equal(ev1.event_types, ev2.event_types)
    np.testing.assert_allclose(ev1.lambda_buy, ev2.lambda_buy)
