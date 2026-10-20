"""Unit tests for single session simulation and paired market making paths."""

import numpy as np
import pytest
from src.config import HawkesParams, MarketParams, PolicyParams
from src.simulation import simulate_paired_session


def test_paired_simulation_invariants():
    h_params = HawkesParams(mu=30.0, alpha_self=10.0, alpha_cross=5.0, beta=20.0)
    m_params = MarketParams(s0=100.0, sigma=0.06, impact_bps=3.0)
    p_params = PolicyParams(
        half_spread_bps=6.0,
        max_inventory=8,
        p0=0.85,
        kappa=0.15,
        fee_bps=0.5,
        inventory_skew_bps=1.0,
    )

    for seed in [1, 42, 100, 2024]:
        res = simulate_paired_session(
            hawkes_params=h_params,
            market_params=m_params,
            policy_params=p_params,
            t_max=1.0,
            rng=seed,
        )

        # 1. Check strict hard inventory bounds for Policy A and Policy B
        inv_a = res.trace_a.record.inventory
        inv_b = res.trace_b.record.inventory

        assert np.all(inv_a <= p_params.max_inventory), f"Policy A exceeded +Q_max at seed {seed}"
        assert np.all(inv_a >= -p_params.max_inventory), f"Policy A fell below -Q_max at seed {seed}"
        assert np.all(inv_b <= p_params.max_inventory), f"Policy B exceeded +Q_max at seed {seed}"
        assert np.all(inv_b >= -p_params.max_inventory), f"Policy B fell below -Q_max at seed {seed}"

        # 2. Check terminal P&L is less than or equal to pre-liquidation MTM
        assert res.metrics_a.net_terminal_pnl <= res.metrics_a.mtm_terminal_pnl + 1e-8
        assert res.metrics_b.net_terminal_pnl <= res.metrics_b.mtm_terminal_pnl + 1e-8

        # 3. Check drawdowns are non-negative
        assert np.all(res.trace_a.record.drawdowns >= -1e-9)
        assert np.all(res.trace_b.record.drawdowns >= -1e-9)


def test_paired_simulation_identical_market_path():
    h_params = HawkesParams(mu=20.0, alpha_self=6.0, alpha_cross=4.0, beta=15.0)
    m_params = MarketParams(s0=100.0, sigma=0.04, impact_bps=2.0)
    p_params = PolicyParams(half_spread_bps=5.0, max_inventory=10)

    res = simulate_paired_session(h_params, m_params, p_params, rng=42)

    # Both policies must be evaluated on the exact same price path timestamps and mid prices
    np.testing.assert_array_equal(res.trace_a.record.timestamps, res.trace_b.record.timestamps)
    np.testing.assert_allclose(res.trace_a.record.mid_prices, res.trace_b.record.mid_prices)


def test_paired_simulation_reproducibility():
    h_params = HawkesParams(mu=25.0, alpha_self=7.0, alpha_cross=3.0, beta=20.0)
    m_params = MarketParams(s0=100.0, sigma=0.05, impact_bps=2.5)
    p_params = PolicyParams(half_spread_bps=5.0, max_inventory=10)

    res1 = simulate_paired_session(h_params, m_params, p_params, rng=123)
    res2 = simulate_paired_session(h_params, m_params, p_params, rng=123)

    assert res1.metrics_a.net_terminal_pnl == pytest.approx(res2.metrics_a.net_terminal_pnl)
    assert res1.metrics_b.net_terminal_pnl == pytest.approx(res2.metrics_b.net_terminal_pnl)
    np.testing.assert_array_equal(res1.trace_a.record.inventory, res2.trace_a.record.inventory)
    np.testing.assert_array_equal(res1.trace_b.record.inventory, res2.trace_b.record.inventory)
