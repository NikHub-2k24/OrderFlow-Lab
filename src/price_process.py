"""Synthetic reference-price process simulation with Brownian noise and signed order-flow impact."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from src.config import MarketParams
from src.hawkes import PointProcessEvents, EVENT_BUY, EVENT_SELL


@dataclass
class MarketPricePath:
    """Stores the synthetic reference-price trajectory across all events in a session."""

    timestamps: np.ndarray             # Event times including t=0 and terminal t_max
    mid_prices: np.ndarray             # Reference mid-price at each timestamp (post-impact)
    pre_event_mid_prices: np.ndarray   # Mid-price immediately before each event (pre-impact)
    event_indices: np.ndarray          # Indices corresponding to order-flow events
    noise_increments: np.ndarray       # Brownian diffusion increments in log space
    impact_increments: np.ndarray      # Signed price impact increments in log space

    @property
    def terminal_mid_price(self) -> float:
        return float(self.mid_prices[-1])

    @property
    def initial_mid_price(self) -> float:
        return float(self.mid_prices[0])

    @property
    def total_return_bps(self) -> float:
        return float(10_000.0 * np.log(self.terminal_mid_price / self.initial_mid_price))


def simulate_reference_price_path(
    events: PointProcessEvents,
    market_params: MarketParams,
    t_max: float = 1.0,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> MarketPricePath:
    """Generate the synthetic reference-price path driven by Brownian diffusion and order impacts.

    Between consecutive events t_{i-1} and t_i:
      delta_log_price_diffusion = sigma * sqrt(delta_t) * Z_i,  where Z_i ~ N(0, 1)

    At event t_i:
      Signed log-price impact = +impact_bps / 10000 for BUY (+1)
                              = -impact_bps / 10000 for SELL (-1)

    Parameters
    ----------
    events : PointProcessEvents
        Simulated order-flow event times and types.
    market_params : MarketParams
        Market parameters including s0, sigma, and impact_bps.
    t_max : float
        Session horizon (default 1.0).
    rng : np.random.Generator or int, optional
        RNG for Brownian noise draws.

    Returns
    -------
    MarketPricePath
        Complete reference-price path.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    s0 = market_params.s0
    sigma = market_params.sigma
    impact_step = market_params.impact_log_increment

    n_events = len(events)

    # Time points: t=0, all event times, and terminal t_max (if not already at t_max)
    t_list = [0.0]
    is_event_mask = [False]
    event_signs = [0]

    for i in range(n_events):
        t_list.append(events.timestamps[i])
        is_event_mask.append(True)
        event_signs.append(events.event_types[i])

    # Append terminal time if last event is before t_max
    if len(t_list) == 1 or t_list[-1] < t_max:
        t_list.append(t_max)
        is_event_mask.append(False)
        event_signs.append(0)

    timestamps = np.array(t_list, dtype=np.float64)
    n_points = len(timestamps)

    log_mid = np.zeros(n_points, dtype=np.float64)
    pre_event_mid = np.zeros(n_points, dtype=np.float64)
    noise_inc = np.zeros(n_points, dtype=np.float64)
    impact_inc = np.zeros(n_points, dtype=np.float64)

    log_mid[0] = np.log(s0)
    pre_event_mid[0] = s0

    for i in range(1, n_points):
        dt = timestamps[i] - timestamps[i - 1]
        dt = max(dt, 0.0)

        # Brownian diffusion between steps
        z = rng.standard_normal()
        diffusion = sigma * np.sqrt(dt) * z
        noise_inc[i] = diffusion

        # Mid-price right before the event occurs
        log_pre = log_mid[i - 1] + diffusion
        pre_event_mid[i] = np.exp(log_pre)

        # Apply event price impact if this point is an order event
        if is_event_mask[i]:
            impact = float(event_signs[i]) * impact_step
            impact_inc[i] = impact
            log_mid[i] = log_pre + impact
        else:
            impact_inc[i] = 0.0
            log_mid[i] = log_pre

    mid_prices = np.exp(log_mid)
    mid_prices[0] = s0
    event_indices = np.where(is_event_mask)[0]

    return MarketPricePath(
        timestamps=timestamps,
        mid_prices=mid_prices,
        pre_event_mid_prices=pre_event_mid,
        event_indices=event_indices,
        noise_increments=noise_inc,
        impact_increments=impact_inc,
    )
