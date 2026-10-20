"""Unit tests for synthetic reference-price process."""

import numpy as np
import pytest
from src.config import HawkesParams, MarketParams
from src.hawkes import simulate_hawkes_bivariate, PointProcessEvents, EVENT_BUY, EVENT_SELL
from src.price_process import simulate_reference_price_path


def test_price_process_positive_and_impact_direction():
    h_params = HawkesParams(mu=20.0, alpha_self=6.0, alpha_cross=4.0, beta=15.0)
    m_params = MarketParams(s0=100.0, sigma=0.04, impact_bps=3.0)

    events = simulate_hawkes_bivariate(h_params, t_max=1.0, rng=42)
    path = simulate_reference_price_path(events, m_params, t_max=1.0, rng=42)

    assert len(path.timestamps) >= len(events)
    # Price must remain strictly positive
    assert np.all(path.mid_prices > 0.0)
    assert np.all(path.pre_event_mid_prices > 0.0)
    assert path.initial_mid_price == pytest.approx(100.0)

    # Test pre vs post event impact relationship
    impact_factor = np.exp(m_params.impact_log_increment)

    for i, p_idx in enumerate(path.event_indices):
        side = events.event_types[i]
        pre_m = path.pre_event_mid_prices[p_idx]
        post_m = path.mid_prices[p_idx]

        if side == EVENT_BUY:
            # Buy order shifts price up
            expected = pre_m * impact_factor
            assert post_m == pytest.approx(expected, rel=1e-7)
        elif side == EVENT_SELL:
            # Sell order shifts price down
            expected = pre_m / impact_factor
            assert post_m == pytest.approx(expected, rel=1e-7)


def test_price_process_zero_volatility_pure_impact():
    # With sigma = 0, price changes ONLY at events by exact impact
    m_params = MarketParams(s0=100.0, sigma=0.0, impact_bps=5.0)
    events = PointProcessEvents(
        timestamps=np.array([0.2, 0.5, 0.8]),
        event_types=np.array([EVENT_BUY, EVENT_BUY, EVENT_SELL]),
        lambda_buy=np.array([20.0, 20.0, 20.0]),
        lambda_sell=np.array([20.0, 20.0, 20.0]),
        process_type="custom",
    )

    path = simulate_reference_price_path(events, m_params, t_max=1.0, rng=1)
    step = 5.0 / 10_000.0

    # Net impact is 2 buys - 1 sell = +1 step
    expected_terminal = 100.0 * np.exp(1 * step)
    assert path.terminal_mid_price == pytest.approx(expected_terminal, rel=1e-7)


def test_price_process_reproducibility():
    h_params = HawkesParams(mu=20.0, alpha_self=6.0, alpha_cross=4.0, beta=15.0)
    m_params = MarketParams(s0=100.0, sigma=0.05, impact_bps=2.0)
    events = simulate_hawkes_bivariate(h_params, t_max=1.0, rng=10)

    p1 = simulate_reference_price_path(events, m_params, t_max=1.0, rng=99)
    p2 = simulate_reference_price_path(events, m_params, t_max=1.0, rng=99)

    np.testing.assert_allclose(p1.mid_prices, p2.mid_prices)
    np.testing.assert_allclose(p1.noise_increments, p2.noise_increments)
