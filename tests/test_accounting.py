"""Unit tests for dealer accounting, fee deduction, and terminal liquidation."""

import numpy as np
import pytest
from src.config import PolicyParams
from src.accounting import calculate_terminal_liquidation


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
