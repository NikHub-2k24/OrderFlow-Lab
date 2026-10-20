"""Market-making quoting policies: Symmetric (Policy A) and Inventory-Aware (Policy B)."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from src.config import PolicyParams


@dataclass(frozen=True)
class QuotePair:
    """Represents posted bid and ask quotes, their distances from mid, and activation flags."""

    bid_price: float
    ask_price: float
    bid_dist_bps: float
    ask_dist_bps: float
    bid_active: bool
    ask_active: bool
    skew_bps: float


def compute_fill_probability(
    distance_bps: float,
    p0: float,
    kappa: float,
    is_active: bool,
) -> float:
    """Calculate conditional fill probability given quote distance in basis points.

    p_fill = min(1.0, p0 * exp(-kappa * distance_bps))  if is_active else 0.0

    Note: This is a stylized conditional fill model. It does not model explicit
    order queue position, order cancellations, or partial fills.
    """
    if not is_active:
        return 0.0
    val = p0 * np.exp(-kappa * max(distance_bps, 0.0))
    return float(np.clip(val, 0.0, 1.0))


class BaseMarketMakerPolicy(ABC):
    """Abstract base class for market maker quoting policies."""

    def __init__(self, params: PolicyParams) -> None:
        self.params = params

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Name of the policy."""
        pass

    @abstractmethod
    def calculate_quotes(self, mid_price: float, inventory: int) -> QuotePair:
        """Calculate posted bid and ask quotes given current mid-price and signed inventory."""
        pass

    def evaluate_order(
        self,
        event_type: int,  # +1 for BUY (hits ask), -1 for SELL (hits bid)
        mid_price: float,
        inventory: int,
        fill_uniform_draw: float,
    ) -> Tuple[bool, float, QuotePair]:
        """Evaluate whether an incoming market order fills against the dealer's quote.

        Parameters
        ----------
        event_type : int
            +1 for buyer-initiated order (lifts ask), -1 for seller-initiated order (hits bid).
        mid_price : float
            Mid-price before order impact.
        inventory : int
            Current signed dealer inventory.
        fill_uniform_draw : float
            Random draw in [0, 1] for Common Random Numbers comparison.

        Returns
        -------
        Tuple[bool, float, QuotePair]
            (is_filled, execution_price, quotes)
        """
        quotes = self.calculate_quotes(mid_price, inventory)

        if event_type == 1:
            # Buyer-initiated order can lift dealer's ask
            p_fill = compute_fill_probability(
                quotes.ask_dist_bps, self.params.p0, self.params.kappa, quotes.ask_active
            )
            is_filled = (fill_uniform_draw <= p_fill) and quotes.ask_active
            exec_price = quotes.ask_price if is_filled else 0.0
            return is_filled, exec_price, quotes

        elif event_type == -1:
            # Seller-initiated order can hit dealer's bid
            p_fill = compute_fill_probability(
                quotes.bid_dist_bps, self.params.p0, self.params.kappa, quotes.bid_active
            )
            is_filled = (fill_uniform_draw <= p_fill) and quotes.bid_active
            exec_price = quotes.bid_price if is_filled else 0.0
            return is_filled, exec_price, quotes

        return False, 0.0, quotes


class SymmetricQuotingPolicy(BaseMarketMakerPolicy):
    """Policy A: Symmetric Quoting.

    - Constant half-spread around mid-price.
    - No inventory skew.
    - Disables bids at +max_inventory and asks at -max_inventory.
    """

    @property
    def policy_name(self) -> str:
        return "Policy A: Symmetric"

    def calculate_quotes(self, mid_price: float, inventory: int) -> QuotePair:
        half_spread_bps = self.params.half_spread_bps
        dist_factor = half_spread_bps / 10_000.0

        bid_price = mid_price * np.exp(-dist_factor)
        ask_price = mid_price * np.exp(+dist_factor)

        # Enforce hard inventory limits
        bid_active = inventory < self.params.max_inventory
        ask_active = inventory > -self.params.max_inventory

        return QuotePair(
            bid_price=float(bid_price),
            ask_price=float(ask_price),
            bid_dist_bps=float(half_spread_bps),
            ask_dist_bps=float(half_spread_bps),
            bid_active=bid_active,
            ask_active=ask_active,
            skew_bps=0.0,
        )


class InventoryAwareQuotingPolicy(BaseMarketMakerPolicy):
    """Policy B: Inventory-Aware Quoting.

    - Skew = clip(inventory_skew_bps * q, -0.8 * half_spread_bps, +0.8 * half_spread_bps)
    - Bid distance = half_spread_bps + skew_bps
    - Ask distance = half_spread_bps - skew_bps
    - Positive inventory (long): bid moves further from mid, ask moves closer to mid.
    - Negative inventory (short): bid moves closer to mid, ask moves further from mid.
    - Enforces hard inventory limits: disables bids at +max_inventory, asks at -max_inventory.
    """

    @property
    def policy_name(self) -> str:
        return "Policy B: Inventory-Aware"

    def calculate_quotes(self, mid_price: float, inventory: int) -> QuotePair:
        half_spread_bps = self.params.half_spread_bps
        max_skew = 0.8 * half_spread_bps

        # Calculate bounded inventory skew
        raw_skew = self.params.inventory_skew_bps * inventory
        skew_bps = float(np.clip(raw_skew, -max_skew, max_skew))

        bid_dist_bps = half_spread_bps + skew_bps
        ask_dist_bps = half_spread_bps - skew_bps

        # Guarantee positivity (at least 0.2 * half_spread_bps)
        bid_dist_bps = max(bid_dist_bps, 0.1 * half_spread_bps)
        ask_dist_bps = max(ask_dist_bps, 0.1 * half_spread_bps)

        bid_price = mid_price * np.exp(-bid_dist_bps / 10_000.0)
        ask_price = mid_price * np.exp(+ask_dist_bps / 10_000.0)

        # Enforce hard inventory limits
        bid_active = inventory < self.params.max_inventory
        ask_active = inventory > -self.params.max_inventory

        return QuotePair(
            bid_price=float(bid_price),
            ask_price=float(ask_price),
            bid_dist_bps=float(bid_dist_bps),
            ask_dist_bps=float(ask_dist_bps),
            bid_active=bid_active,
            ask_active=ask_active,
            skew_bps=skew_bps,
        )
