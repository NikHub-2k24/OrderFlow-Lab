"""Unit tests for dealer accounting, fee deduction, and terminal liquidation."""

import numpy as np
import pytest
from src.config import PolicyParams
from src.accounting import calculate_terminal_liquidation, SessionMetrics


def test_terminal_liquidation_flat_inventory():
    params = PolicyParams(half_spread_bps=5.0, fee_bps=0.5)
    terminal_cash = 150.0
    terminal_mid = 100.0

    net_pnl, penalty, spread_cost, fee = calculate_terminal_liquidation(
        terminal_cash=terminal_cash,
        terminal_inventory=0,
        terminal_mid=terminal_mid,
        policy_params=params,
    )

    assert net_pnl == 150.0
    assert penalty == 0.0
    assert spread_cost == 0.0
    assert fee == 0.0


def test_terminal_liquidation_long_inventory():
    # Long 10 units at mid = 100.0. Pre-liquidation MTM equity = 1000.0 + cash
    params = PolicyParams(half_spread_bps=10.0, fee_bps=1.0)
    terminal_cash = 50.0
    terminal_mid = 100.0
    q = 10

    net_pnl, penalty, spread_cost, fee = calculate_terminal_liquidation(
        terminal_cash=terminal_cash,
        terminal_inventory=q,
        terminal_mid=terminal_mid,
        policy_params=params,
    )

    p_liq = 100.0 * np.exp(-10.0 / 10_000.0)
    expected_proceeds = 10 * p_liq
    expected_fee = expected_proceeds * (1.0 / 10_000.0)
    expected_net = expected_proceeds - expected_fee

    assert net_pnl == pytest.approx(terminal_cash + expected_net)
    assert penalty > 0.0
    assert spread_cost > 0.0
    assert fee > 0.0

    # Key requirement: Terminal net P&L must be STRICTLY LESS than pre-liquidation MTM equity
    pre_mtm_equity = terminal_cash + q * terminal_mid
    assert net_pnl < pre_mtm_equity


def test_terminal_liquidation_short_inventory():
    # Short 10 units at mid = 100.0. Pre-liquidation MTM equity = -1000.0 + cash
    params = PolicyParams(half_spread_bps=10.0, fee_bps=1.0)
    terminal_cash = 2000.0
    terminal_mid = 100.0
    q = -10

    net_pnl, penalty, spread_cost, fee = calculate_terminal_liquidation(
        terminal_cash=terminal_cash,
        terminal_inventory=q,
        terminal_mid=terminal_mid,
        policy_params=params,
    )

    p_liq = 100.0 * np.exp(+10.0 / 10_000.0)
    expected_cost = 10 * p_liq
    expected_fee = expected_cost * (1.0 / 10_000.0)
    expected_net_outflow = expected_cost + expected_fee

    assert net_pnl == pytest.approx(terminal_cash - expected_net_outflow)
    assert penalty > 0.0
    assert spread_cost > 0.0
    assert fee > 0.0

    # Key requirement: Terminal net P&L must be STRICTLY LESS than pre-liquidation MTM equity
    pre_mtm_equity = terminal_cash + q * terminal_mid
    assert net_pnl < pre_mtm_equity


def test_fees_and_liquidation_never_increase_pnl():
    # Fuzz check across different inventory and fee combinations
    for q in [-20, -5, 0, 5, 20]:
        for fee_bps in [0.0, 0.5, 2.0]:
            for spread_bps in [1.0, 5.0, 20.0]:
                params = PolicyParams(half_spread_bps=spread_bps, fee_bps=fee_bps)
                cash = 100.0
                mid = 105.0

                net_pnl, penalty, _, _ = calculate_terminal_liquidation(
                    terminal_cash=cash,
                    terminal_inventory=q,
                    terminal_mid=mid,
                    policy_params=params,
                )

                mtm = cash + q * mid
                assert net_pnl <= mtm + 1e-10, "Terminal P&L cannot exceed MTM equity!"
                assert penalty >= -1e-10, "Liquidation penalty must be non-negative!"


def test_session_metrics_constructor():
    # Test instantiation with explicit event_sampled_max_drawdown and omitting max_drawdown
    metrics = SessionMetrics(
        net_terminal_pnl=-12.5,
        mtm_terminal_pnl=-10.0,
        liquidation_cost=2.5,
        spread_crossing_cost=2.0,
        liquidation_fee=0.5,
        total_fees=1.5,
        intra_session_fees=1.0,
        n_fills=14,
        n_buy_fills=8,
        n_sell_fills=6,
        traded_notional=1400.0,
        terminal_inventory=-2,
        max_abs_inventory=6,
        time_average_abs_inventory=2.75,
        inventory_utilization=0.6,
        event_sampled_max_drawdown=15.0,
    )

    assert metrics.net_terminal_pnl == -12.5
    assert metrics.time_average_abs_inventory == 2.75
    assert metrics.max_abs_inventory == 6
    assert metrics.event_sampled_max_drawdown == 15.0
    # Auto-populated alias and is_loss
    assert metrics.max_drawdown == 15.0
    assert metrics.is_loss is True


def test_inventory_integration_calculation_exact_cases():
    # Case 1: Constant inventory q=4 over [0, 1]
    t1 = np.array([0.0, 0.3, 0.7, 1.0])
    q1 = np.array([4, 4, 4, 4])
    dt1 = np.diff(t1)
    integral1 = np.sum(np.abs(q1[:-1]) * dt1) / (t1[-1] - t1[0])
    assert integral1 == pytest.approx(4.0)

    # Case 2: Half time at +5, half time at -5
    t2 = np.array([0.0, 0.5, 1.0])
    q2 = np.array([5, -5, -5])
    dt2 = np.diff(t2)
    integral2 = np.sum(np.abs(q2[:-1]) * dt2) / (t2[-1] - t2[0])
    # |5| * 0.5 + |-5| * 0.5 = 2.5 + 2.5 = 5.0
    assert integral2 == pytest.approx(5.0)

    # Case 3: Step-wise ramp: 0 from [0, 0.2], 2 from [0.2, 0.6], -4 from [0.6, 1.0]
    t3 = np.array([0.0, 0.2, 0.6, 1.0])
    q3 = np.array([0, 2, -4, 0])
    dt3 = np.diff(t3)
    # 0 * 0.2 + 2 * 0.4 + 4 * 0.4 = 0 + 0.8 + 1.6 = 2.4
    integral3 = np.sum(np.abs(q3[:-1]) * dt3) / (t3[-1] - t3[0])
    assert integral3 == pytest.approx(2.4)

