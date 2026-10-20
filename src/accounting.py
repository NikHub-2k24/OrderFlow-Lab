"""Accounting engine for dealer cash, inventory, mark-to-market equity, fees, and liquidation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.config import PolicyParams


@dataclass
class SessionAccountingRecord:
    """Detailed time-series trajectory of dealer accounting state during a single session."""

    timestamps: np.ndarray        # Time points (events + initial + terminal)
    cash: np.ndarray              # Cash balance trajectory
    inventory: np.ndarray         # Signed inventory trajectory
    mid_prices: np.ndarray        # Reference mid-price trajectory
    mtm_equity: np.ndarray        # Mark-to-market equity: cash + inventory * mid
    drawdowns: np.ndarray         # Peak-to-trough drawdowns
    fill_events: np.ndarray       # Boolean array indicating where fills occurred
    fill_sides: np.ndarray        # Side of fill (+1 for dealer bought, -1 for dealer sold, 0 for none)
    fill_prices: np.ndarray       # Execution prices for fills


@dataclass
class SessionMetrics:
    """Comprehensive performance and risk metrics for a single simulation session."""

    net_terminal_pnl: float       # Final P&L net of intra-session fees, spread crossing, and liq fees
    mtm_terminal_pnl: float       # Final equity marked to mid before terminal liquidation
    liquidation_cost: float       # Total cost of liquidating residual inventory (spread + fee)
    spread_crossing_cost: float   # Cost of crossing base half-spread during residual liquidation
    liquidation_fee: float        # Fee paid on terminal liquidation trade
    total_fees: float             # Total transaction fees paid (intra-session + liquidation)
    intra_session_fees: float     # Transaction fees paid during the session
    n_fills: int                  # Total number of fills
    n_buy_fills: int              # Fills where dealer bought (seller hit dealer bid)
    n_sell_fills: int             # Fills where dealer sold (buyer lifted dealer ask)
    traded_notional: float        # Total traded dollar notional
    terminal_inventory: int       # Signed inventory at session end
    max_abs_inventory: int        # Peak absolute inventory during session
    time_average_abs_inventory: float  # Time-weighted integral(abs(q(t)) dt) / T
    inventory_utilization: float  # max_abs_inventory / max_inventory
    event_sampled_max_drawdown: float  # Event-time-sampled peak-to-trough equity drawdown
    max_drawdown: Optional[float] = None  # Backward-compatible alias for event_sampled_max_drawdown
    is_loss: Optional[bool] = None  # True if net_terminal_pnl < 0

    def __post_init__(self) -> None:
        if self.max_drawdown is None:
            self.max_drawdown = self.event_sampled_max_drawdown
        if self.is_loss is None:
            self.is_loss = bool(self.net_terminal_pnl < 0.0)


def calculate_terminal_liquidation(
    terminal_cash: float,
    terminal_inventory: int,
    terminal_mid: float,
    policy_params: PolicyParams,
) -> tuple[float, float, float, float]:
    """Calculate terminal liquidation proceeds, crossing cost, and fee.

    At session end (t=1.0), residual inventory is liquidated:
    - If q > 0: Sell q at mid * exp(-half_spread_bps / 10000)
    - If q < 0: Buy back |q| at mid * exp(+half_spread_bps / 10000)
    - Liquidating trade incurs fee_bps of liquidation notional.

    Returns
    -------
    tuple[float, float, float, float]
        (net_terminal_pnl, total_liquidation_penalty, spread_crossing_cost, liquidation_fee)
    """
    half_spread_step = policy_params.half_spread_bps / 10_000.0
    fee_rate = policy_params.fee_bps / 10_000.0

    q = terminal_inventory
    s_t = terminal_mid

    if q == 0:
        return float(terminal_cash), 0.0, 0.0, 0.0

    if q > 0:
        # Dealer is net long: must sell q units at dealer bid
        p_liq = s_t * np.exp(-half_spread_step)
        gross_notional = float(q) * p_liq
        fee = gross_notional * fee_rate
        net_proceeds = gross_notional - fee

        final_cash = terminal_cash + net_proceeds
        # Pre-liquidation inventory MTM value is q * s_t
        pre_liq_mtm_value = float(q) * s_t
        spread_cost = pre_liq_mtm_value - gross_notional
        total_penalty = pre_liq_mtm_value - net_proceeds

        return float(final_cash), float(total_penalty), float(spread_cost), float(fee)

    else:  # q < 0
        # Dealer is net short: must buy back |q| units at dealer ask
        abs_q = abs(q)
        p_liq = s_t * np.exp(+half_spread_step)
        gross_cost = float(abs_q) * p_liq
        fee = gross_cost * fee_rate
        net_cost = gross_cost + fee

        final_cash = terminal_cash - net_cost
        # Pre-liquidation inventory MTM value is q * s_t (negative liability)
        # Liquidating eliminates liability at net_cost
        pre_liq_liability = float(abs_q) * s_t
        spread_cost = gross_cost - pre_liq_liability
        total_penalty = net_cost - pre_liq_liability

        return float(final_cash), float(total_penalty), float(spread_cost), float(fee)
