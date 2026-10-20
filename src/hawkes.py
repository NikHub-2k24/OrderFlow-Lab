"""Bivariate Hawkes process and matched Poisson process simulation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from src.config import HawkesParams


# Event type constants
EVENT_BUY: int = 1   # Buyer-initiated market order (hits market-maker ask)
EVENT_SELL: int = -1  # Seller-initiated market order (hits market-maker bid)


@dataclass
class PointProcessEvents:
    """Container for simulated point-process events."""

    timestamps: np.ndarray        # 1D float array of event times in [0, t_max]
    event_types: np.ndarray       # 1D int array (+1 for BUY, -1 for SELL)
    lambda_buy: np.ndarray        # 1D float array of intensity right before event
    lambda_sell: np.ndarray       # 1D float array of intensity right before event
    process_type: str             # "hawkes" or "poisson"

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def n_buys(self) -> int:
        return int(np.sum(self.event_types == EVENT_BUY))

    @property
    def n_sells(self) -> int:
        return int(np.sum(self.event_types == EVENT_SELL))


def simulate_hawkes_bivariate(
    params: HawkesParams,
    t_max: float = 1.0,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> PointProcessEvents:
    """Simulate a continuous-time bivariate Hawkes process using Ogata's thinning algorithm.

    Intensity equations:
      lambda_buy(t)  = mu + alpha_self * x_buy(t)  + alpha_cross * x_sell(t)
      lambda_sell(t) = mu + alpha_self * x_sell(t) + alpha_cross * x_buy(t)

    Decay between events:
      x_buy(t + dt)  = x_buy(t) * exp(-beta * dt)
      x_sell(t + dt) = x_sell(t) * exp(-beta * dt)

    Upon accepted event at t:
      buy:  x_buy(t+)  = x_buy(t) + 1,  x_sell(t+) = x_sell(t)
      sell: x_sell(t+) = x_sell(t) + 1, x_buy(t+)  = x_buy(t)

    Parameters
    ----------
    params : HawkesParams
        Validated parameters for the bivariate Hawkes process.
    t_max : float
        Session length (default 1.0).
    rng : np.random.Generator or int, optional
        Random number generator or seed.

    Returns
    -------
    PointProcessEvents
        Recorded event timestamps, event types, and pre-event intensities.
    """
    # Enforce parameter stability
    if params.branching_ratio >= 1.0:
        raise ValueError(
            f"Cannot simulate unstable Hawkes process: branching ratio {params.branching_ratio:.4f} >= 1.0"
        )

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    mu = params.mu
    alpha_self = params.alpha_self
    alpha_cross = params.alpha_cross
    beta = params.beta

    # Simulation state variables
    t = 0.0
    t_last = 0.0
    x_buy_state = 0.0
    x_sell_state = 0.0

    # Right after t=0, state is 0, total intensity is 2 * mu
    lambda_tot_upper = 2.0 * mu

    times_list: List[float] = []
    types_list: List[int] = []
    l_buy_list: List[float] = []
    l_sell_list: List[float] = []

    # Ogata thinning loop
    while t < t_max:
        if lambda_tot_upper <= 0:
            break

        # Candidate inter-arrival time from exponential with rate lambda_tot_upper
        w = rng.exponential(1.0 / lambda_tot_upper)
        t_cand = t + w

        if t_cand > t_max:
            break

        # Decay states from t_last to t_cand
        dt = t_cand - t_last
        decay = np.exp(-beta * dt)
        x_buy_cand = x_buy_state * decay
        x_sell_cand = x_sell_state * decay

        # Evaluate intensities at candidate time t_cand
        lam_buy_cand = mu + alpha_self * x_buy_cand + alpha_cross * x_sell_cand
        lam_sell_cand = mu + alpha_self * x_sell_cand + alpha_cross * x_buy_cand
        lam_tot_cand = lam_buy_cand + lam_sell_cand

        # Thinning acceptance test: accept with probability lam_tot_cand / lambda_tot_upper
        u1 = rng.uniform(0.0, 1.0)
        if u1 <= (lam_tot_cand / lambda_tot_upper):
            # Candidate accepted! Attribute event type
            u2 = rng.uniform(0.0, 1.0)
            if u2 <= (lam_buy_cand / lam_tot_cand):
                event_type = EVENT_BUY
                x_buy_state = x_buy_cand + 1.0
                x_sell_state = x_sell_cand
            else:
                event_type = EVENT_SELL
                x_buy_state = x_buy_cand
                x_sell_state = x_sell_cand + 1.0

            times_list.append(t_cand)
            types_list.append(event_type)
            l_buy_list.append(lam_buy_cand)
            l_sell_list.append(lam_sell_cand)

            # Advance state time
            t_last = t_cand
            t = t_cand

            # Immediately after the jump, total intensity jumps by (alpha_self + alpha_cross)
            lambda_tot_upper = lam_tot_cand + (alpha_self + alpha_cross)
        else:
            # Candidate rejected.
            # Advance current time to candidate time
            t = t_cand
            # Since intensity decreases monotonically between events, lam_tot_cand
            # is a valid, tighter upper bound for subsequent times
            lambda_tot_upper = max(lam_tot_cand, 2.0 * mu)

    return PointProcessEvents(
        timestamps=np.array(times_list, dtype=np.float64),
        event_types=np.array(types_list, dtype=np.int32),
        lambda_buy=np.array(l_buy_list, dtype=np.float64),
        lambda_sell=np.array(l_sell_list, dtype=np.float64),
        process_type="hawkes",
    )


def simulate_poisson_matched(
    params: HawkesParams,
    t_max: float = 1.0,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> PointProcessEvents:
    """Simulate a matched Poisson process with the same theoretical mean intensity per side.

    Theoretical mean rate per side:
      mean_rate = mu / (1 - branching_ratio)
    Total arrival rate:
      total_rate = 2 * mean_rate

    Each arrival is independently assigned BUY or SELL with probability 0.5.

    Parameters
    ----------
    params : HawkesParams
        Hawkes parameters from which the matched mean rate is computed.
    t_max : float
        Session length (default 1.0).
    rng : np.random.Generator or int, optional
        Random number generator or seed.

    Returns
    -------
    PointProcessEvents
        Recorded Poisson event timestamps, event types, and constant intensities.
    """
    if params.branching_ratio >= 1.0:
        raise ValueError(
            f"Cannot compute matched Poisson rate for unstable Hawkes process (ratio {params.branching_ratio:.4f} >= 1.0)"
        )

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    mean_rate_per_side = params.theoretical_mean_rate_per_side
    total_rate = 2.0 * mean_rate_per_side

    times_list: List[float] = []
    types_list: List[int] = []

    t = 0.0
    while t < t_max:
        w = rng.exponential(1.0 / total_rate)
        t += w
        if t > t_max:
            break
        # Assign side with equal probability
        side = EVENT_BUY if rng.uniform(0.0, 1.0) < 0.5 else EVENT_SELL
        times_list.append(t)
        types_list.append(side)

    n_events = len(times_list)
    return PointProcessEvents(
        timestamps=np.array(times_list, dtype=np.float64),
        event_types=np.array(types_list, dtype=np.int32),
        lambda_buy=np.full(n_events, mean_rate_per_side, dtype=np.float64),
        lambda_sell=np.full(n_events, mean_rate_per_side, dtype=np.float64),
        process_type="poisson",
    )


def compute_continuous_intensities(
    events: PointProcessEvents,
    params: HawkesParams,
    grid_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact Hawkes intensities along an arbitrary time grid.

    Parameters
    ----------
    events : PointProcessEvents
        The historical events.
    params : HawkesParams
        Hawkes parameters.
    grid_times : np.ndarray
        Sorted 1D array of evaluation timestamps.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (lambda_buy_grid, lambda_sell_grid)
    """
    if events.process_type == "poisson":
        rate = params.theoretical_mean_rate_per_side
        return np.full_like(grid_times, rate), np.full_like(grid_times, rate)

    mu = params.mu
    alpha_self = params.alpha_self
    alpha_cross = params.alpha_cross
    beta = params.beta

    t_events = events.timestamps
    types = events.event_types
    n_events = len(t_events)

    l_buy = np.full(len(grid_times), mu, dtype=np.float64)
    l_sell = np.full(len(grid_times), mu, dtype=np.float64)

    if n_events == 0:
        return l_buy, l_sell

    # Compute intensity trajectory sequentially
    event_idx = 0
    x_buy = 0.0
    x_sell = 0.0
    t_prev = 0.0

    for i, t in enumerate(grid_times):
        # Catch up events strictly before t
        while event_idx < n_events and t_events[event_idx] <= t:
            te = t_events[event_idx]
            dt = te - t_prev
            decay = np.exp(-beta * dt)
            x_buy = x_buy * decay
            x_sell = x_sell * decay

            if types[event_idx] == EVENT_BUY:
                x_buy += 1.0
            else:
                x_sell += 1.0

            t_prev = te
            event_idx += 1

        dt = t - t_prev
        decay = np.exp(-beta * dt) if dt > 0 else 1.0
        xb = x_buy * decay
        xs = x_sell * decay

        l_buy[i] = mu + alpha_self * xb + alpha_cross * xs
        l_sell[i] = mu + alpha_self * xs + alpha_cross * xb

    return l_buy, l_sell
