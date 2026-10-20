"""Simulation orchestration for single-session and paired market-making paths."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from src.config import HawkesParams, MarketParams, PolicyParams
from src.hawkes import (
    PointProcessEvents,
    simulate_hawkes_bivariate,
    simulate_poisson_matched,
    EVENT_BUY,
    EVENT_SELL,
)
from src.price_process import MarketPricePath, simulate_reference_price_path
from src.policy import (
    BaseMarketMakerPolicy,
    SymmetricQuotingPolicy,
    InventoryAwareQuotingPolicy,
    QuotePair,
)
from src.accounting import (
    SessionAccountingRecord,
    SessionMetrics,
    calculate_terminal_liquidation,
)


@dataclass
class PolicyExecutionTrace:
    """Records full policy actions and quotes along a price path."""

    metrics: SessionMetrics
    record: SessionAccountingRecord
    quotes: List[QuotePair]


@dataclass
class PairedSessionResult:
    """Result of running both Policy A and Policy B on the exact same market path."""

    events: PointProcessEvents
    price_path: MarketPricePath
    trace_a: PolicyExecutionTrace
    trace_b: PolicyExecutionTrace

    @property
    def metrics_a(self) -> SessionMetrics:
        return self.trace_a.metrics

    @property
    def metrics_b(self) -> SessionMetrics:
        return self.trace_b.metrics

    @property
    def pnl_diff_b_minus_a(self) -> float:
        """Net P&L of Policy B minus Policy A on this exact market path."""
        return self.metrics_b.net_terminal_pnl - self.metrics_a.net_terminal_pnl


def run_policy_on_path(
    policy: BaseMarketMakerPolicy,
    events: PointProcessEvents,
    price_path: MarketPricePath,
    policy_params: PolicyParams,
    fill_uniforms: np.ndarray,
) -> PolicyExecutionTrace:
    """Execute a market-making policy on a pre-generated price path with predetermined fill draws.

    Parameters
    ----------
    policy : BaseMarketMakerPolicy
        Symmetric or Inventory-Aware policy instance.
    events : PointProcessEvents
        Order flow events.
    price_path : MarketPricePath
        Synthetic reference-price path.
    policy_params : PolicyParams
        Parameters governing spread, fees, and inventory limits.
    fill_uniforms : np.ndarray
        Array of uniform random numbers in [0, 1] for fill evaluation.

    Returns
    -------
    PolicyExecutionTrace
        Contains metrics, time-series trajectory, and posted quotes.
    """
    n_points = len(price_path.timestamps)
    n_events = len(events)
    max_inv = policy_params.max_inventory
    fee_rate = policy_params.fee_bps / 10_000.0

    # Arrays to record time-series evolution
    cash_arr = np.zeros(n_points, dtype=np.float64)
    inv_arr = np.zeros(n_points, dtype=np.int32)
    mtm_equity = np.zeros(n_points, dtype=np.float64)
    drawdowns = np.zeros(n_points, dtype=np.float64)
    fill_events = np.zeros(n_points, dtype=bool)
    fill_sides = np.zeros(n_points, dtype=np.int32)
    fill_prices = np.zeros(n_points, dtype=np.float64)

    quotes_recorded: List[QuotePair] = []

    current_cash = 0.0
    current_inv = 0
    peak_equity = 0.0
    intra_fees = 0.0
    traded_notional = 0.0
    n_buy_fills = 0
    n_sell_fills = 0
    max_abs_inv = 0

    # Initial state at t=0
    mtm_equity[0] = current_cash + current_inv * price_path.mid_prices[0]
    peak_equity = mtm_equity[0]

    # Map event index to price path index
    event_indices = price_path.event_indices

    for i in range(n_events):
        path_idx = event_indices[i]

        # Prior points between events (if any) maintain current cash and inventory
        prev_idx = 0 if i == 0 else event_indices[i - 1]
        for p in range(prev_idx + 1, path_idx):
            cash_arr[p] = current_cash
            inv_arr[p] = current_inv
            eq = current_cash + current_inv * price_path.mid_prices[p]
            mtm_equity[p] = eq
            peak_equity = max(peak_equity, eq)
            drawdowns[p] = peak_equity - eq

        # Reference mid-price right before order impact
        pre_mid = price_path.pre_event_mid_prices[path_idx]
        event_type = events.event_types[i]
        u_fill = fill_uniforms[i]

        # Evaluate policy fill against incoming market order
        is_filled, exec_price, quote_pair = policy.evaluate_order(
            event_type=event_type,
            mid_price=pre_mid,
            inventory=current_inv,
            fill_uniform_draw=u_fill,
        )
        quotes_recorded.append(quote_pair)

        if is_filled:
            fill_events[path_idx] = True
            fill_prices[path_idx] = exec_price

            if event_type == EVENT_BUY:
                # Buyer-initiated order lifted dealer ask: dealer sells 1 unit
                fee = exec_price * fee_rate
                current_cash += (exec_price - fee)
                current_inv -= 1
                intra_fees += fee
                traded_notional += exec_price
                n_sell_fills += 1
                fill_sides[path_idx] = -1  # Dealer sold

            elif event_type == EVENT_SELL:
                # Seller-initiated order hit dealer bid: dealer buys 1 unit
                fee = exec_price * fee_rate
                current_cash -= (exec_price + fee)
                current_inv += 1
                intra_fees += fee
                traded_notional += exec_price
                n_buy_fills += 1
                fill_sides[path_idx] = 1   # Dealer bought

        # Assert strict inventory bounds
        if not (-max_inv <= current_inv <= max_inv):
            raise RuntimeError(
                f"Invariant violation in {policy.policy_name}: inventory {current_inv} exceeded limit +/-{max_inv}"
            )

        max_abs_inv = max(max_abs_inv, abs(current_inv))

        # Update state at event point (post-impact mid-price)
        post_mid = price_path.mid_prices[path_idx]
        eq = current_cash + current_inv * post_mid
        cash_arr[path_idx] = current_cash
        inv_arr[path_idx] = current_inv
        mtm_equity[path_idx] = eq
        peak_equity = max(peak_equity, eq)
        drawdowns[path_idx] = peak_equity - eq

    # Carry forward through terminal point
    last_event_idx = event_indices[-1] if n_events > 0 else 0
    for p in range(last_event_idx + 1, n_points):
        cash_arr[p] = current_cash
        inv_arr[p] = current_inv
        eq = current_cash + current_inv * price_path.mid_prices[p]
        mtm_equity[p] = eq
        peak_equity = max(peak_equity, eq)
        drawdowns[p] = peak_equity - eq

    # Calculate terminal liquidation
    terminal_mid = price_path.terminal_mid_price
    net_pnl, liq_cost, spread_cost, liq_fee = calculate_terminal_liquidation(
        terminal_cash=current_cash,
        terminal_inventory=current_inv,
        terminal_mid=terminal_mid,
        policy_params=policy_params,
    )

    mtm_terminal_pnl = current_cash + current_inv * terminal_mid
    total_fees = intra_fees + liq_fee
    max_mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    inv_util = float(max_abs_inv) / float(max_inv) if max_inv > 0 else 0.0

    metrics = SessionMetrics(
        net_terminal_pnl=net_pnl,
        mtm_terminal_pnl=mtm_terminal_pnl,
        liquidation_cost=liq_cost,
        spread_crossing_cost=spread_cost,
        liquidation_fee=liq_fee,
        total_fees=total_fees,
        intra_session_fees=intra_fees,
        n_fills=n_buy_fills + n_sell_fills,
        n_buy_fills=n_buy_fills,
        n_sell_fills=n_sell_fills,
        traded_notional=traded_notional,
        terminal_inventory=current_inv,
        max_abs_inventory=max_abs_inv,
        inventory_utilization=inv_util,
        max_drawdown=max_mdd,
        is_loss=(net_pnl < 0.0),
    )

    record = SessionAccountingRecord(
        timestamps=price_path.timestamps,
        cash=cash_arr,
        inventory=inv_arr,
        mid_prices=price_path.mid_prices,
        mtm_equity=mtm_equity,
        drawdowns=drawdowns,
        fill_events=fill_events,
        fill_sides=fill_sides,
        fill_prices=fill_prices,
    )

    return PolicyExecutionTrace(
        metrics=metrics,
        record=record,
        quotes=quotes_recorded,
    )


def simulate_paired_session(
    hawkes_params: HawkesParams,
    market_params: MarketParams,
    policy_params: PolicyParams,
    t_max: float = 1.0,
    use_poisson: bool = False,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> PairedSessionResult:
    """Generate one synthetic market path and execute both Policy A and Policy B under Common Random Numbers.

    Parameters
    ----------
    hawkes_params : HawkesParams
        Hawkes parameters.
    market_params : MarketParams
        Market parameters.
    policy_params : PolicyParams
        Quoting policy parameters.
    t_max : float
        Session horizon (default 1.0).
    use_poisson : bool
        If True, use matched Poisson arrival process instead of Hawkes.
    rng : np.random.Generator or int, optional
        RNG for reproducibility.

    Returns
    -------
    PairedSessionResult
        Market path and traces for both policies.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # 1. Generate Order Flow Events
    if use_poisson:
        events = simulate_poisson_matched(hawkes_params, t_max=t_max, rng=rng)
    else:
        events = simulate_hawkes_bivariate(hawkes_params, t_max=t_max, rng=rng)

    # 2. Generate Synthetic Reference-Price Path
    price_path = simulate_reference_price_path(events, market_params, t_max=t_max, rng=rng)

    # 3. Generate Common Random Numbers for fill decisions
    n_events = len(events)
    fill_uniforms = rng.uniform(0.0, 1.0, size=n_events)

    # 4. Instantiate both policies
    policy_a = SymmetricQuotingPolicy(policy_params)
    policy_b = InventoryAwareQuotingPolicy(policy_params)

    # 5. Execute both policies on identical market path with common random fill draws
    trace_a = run_policy_on_path(policy_a, events, price_path, policy_params, fill_uniforms)
    trace_b = run_policy_on_path(policy_b, events, price_path, policy_params, fill_uniforms)

    return PairedSessionResult(
        events=events,
        price_path=price_path,
        trace_a=trace_a,
        trace_b=trace_b,
    )
