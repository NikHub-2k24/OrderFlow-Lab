"""Unit tests for quoting policies and execution mechanics."""

import numpy as np
import pytest
from src.config import PolicyParams
from src.policy import (
    SymmetricQuotingPolicy,
    InventoryAwareQuotingPolicy,
    compute_fill_probability,
)


def test_symmetric_policy_constant_spread():
    params = PolicyParams(half_spread_bps=5.0, max_inventory=10)
    policy = SymmetricQuotingPolicy(params)

    mid = 100.0
    quotes_flat = policy.calculate_quotes(mid, inventory=0)
    quotes_long = policy.calculate_quotes(mid, inventory=5)
    quotes_short = policy.calculate_quotes(mid, inventory=-5)

    assert quotes_flat.bid_dist_bps == 5.0
    assert quotes_flat.ask_dist_bps == 5.0
    assert quotes_long.bid_dist_bps == 5.0
    assert quotes_long.ask_dist_bps == 5.0
    assert quotes_short.bid_dist_bps == 5.0
    assert quotes_short.ask_dist_bps == 5.0


def test_inventory_aware_quote_skew_direction():
    params = PolicyParams(half_spread_bps=10.0, max_inventory=10, inventory_skew_bps=1.0)
    policy = InventoryAwareQuotingPolicy(params)
    mid = 100.0

    # Flat inventory (q = 0): no skew
    q0 = policy.calculate_quotes(mid, inventory=0)
    assert q0.skew_bps == 0.0
    assert q0.bid_dist_bps == 10.0
    assert q0.ask_dist_bps == 10.0

    # Positive inventory (q = +4):
    # Skew = +4.0 bps
    # Bid distance increases (bid moves farther from mid, lower price)
    # Ask distance decreases (ask moves closer to mid, lower price)
    q_pos = policy.calculate_quotes(mid, inventory=4)
    assert q_pos.skew_bps == 4.0
    assert q_pos.bid_dist_bps == 14.0  # 10 + 4
    assert q_pos.ask_dist_bps == 6.0   # 10 - 4
    assert q_pos.bid_price < q0.bid_price
    assert q_pos.ask_price < q0.ask_price

    # Negative inventory (q = -4):
    # Skew = -4.0 bps
    # Bid distance decreases (bid moves closer to mid, higher price)
    # Ask distance increases (ask moves farther from mid, higher price)
    q_neg = policy.calculate_quotes(mid, inventory=-4)
    assert q_neg.skew_bps == -4.0
    assert q_neg.bid_dist_bps == 6.0   # 10 - 4
    assert q_neg.ask_dist_bps == 14.0  # 10 + 4
    assert q_neg.bid_price > q0.bid_price
    assert q_neg.ask_price > q0.ask_price


def test_inventory_aware_skew_clamping():
    params = PolicyParams(half_spread_bps=10.0, max_inventory=10, inventory_skew_bps=2.0)
    policy = InventoryAwareQuotingPolicy(params)
    mid = 100.0

    # For q = 10, raw skew = 20.0, but max allowed skew is 0.8 * 10 = 8.0 bps
    q_max = policy.calculate_quotes(mid, inventory=10)
    assert q_max.skew_bps == 8.0
    assert q_max.bid_dist_bps == 18.0  # 10 + 8
    assert q_max.ask_dist_bps == 2.0   # 10 - 8
    # Distances must remain strictly positive
    assert q_max.bid_dist_bps > 0
    assert q_max.ask_dist_bps > 0

    # For q = -10, raw skew = -20.0, clamped to -8.0 bps
    q_min = policy.calculate_quotes(mid, inventory=-10)
    assert q_min.skew_bps == -8.0
    assert q_min.bid_dist_bps == 2.0
    assert q_min.ask_dist_bps == 18.0


def test_hard_inventory_limit_quote_disabling():
    params = PolicyParams(half_spread_bps=5.0, max_inventory=5)
    policy_sym = SymmetricQuotingPolicy(params)
    policy_inv = InventoryAwareQuotingPolicy(params)

    for pol in [policy_sym, policy_inv]:
        # At positive limit (+5): cannot buy more -> bid is disabled
        q_pos = pol.calculate_quotes(100.0, inventory=5)
        assert q_pos.bid_active is False
        assert q_pos.ask_active is True

        # At negative limit (-5): cannot sell more -> ask is disabled
        q_neg = pol.calculate_quotes(100.0, inventory=-5)
        assert q_neg.bid_active is True
        assert q_neg.ask_active is False


def test_fill_probability_function():
    # p_fill = min(1, p0 * exp(-kappa * dist))
    p0 = 0.8
    kappa = 0.1

    # At distance 0
    assert compute_fill_probability(0.0, p0, kappa, is_active=True) == pytest.approx(0.8)

    # Monotonic decrease
    p1 = compute_fill_probability(5.0, p0, kappa, is_active=True)
    p2 = compute_fill_probability(10.0, p0, kappa, is_active=True)
    assert 0.0 < p2 < p1 < 0.8

    # When disabled
    assert compute_fill_probability(5.0, p0, kappa, is_active=False) == 0.0
