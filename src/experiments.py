"""Monte Carlo experiment runners, paired statistical analysis, and sensitivity sweeps."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    HawkesParams,
    MarketParams,
    PolicyParams,
    ExperimentConfig,
)
from src.simulation import simulate_paired_session, PairedSessionResult


def compute_bootstrap_ci(
    data: np.ndarray,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval for the sample mean.

    Parameters
    ----------
    data : np.ndarray
        1D array of sample observations.
    n_bootstraps : int
        Number of resamples (default 2000).
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    rng : np.random.Generator or int, optional
        RNG for bootstrap draws.

    Returns
    -------
    Tuple[float, float, float]
        (point_mean, ci_lower, ci_upper)
    """
    clean_data = np.asarray(data, dtype=np.float64)
    n = len(clean_data)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        val = float(clean_data[0])
        return val, val, val

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_means = np.mean(clean_data[boot_indices], axis=1)

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    point_mean = float(np.mean(clean_data))
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return point_mean, ci_lower, ci_upper


def compute_cvar(data: np.ndarray, quantile: float = 0.05) -> float:
    """Calculate Conditional Value-at-Risk (Expected Shortfall) below the given quantile.

    Parameters
    ----------
    data : np.ndarray
        Array of returns or P&Ls.
    quantile : float
        Tail probability cutoff (default 0.05 for worst 5%).

    Returns
    -------
    float
        Average outcome in the worst quantile fraction of sessions.
    """
    clean_data = np.asarray(data, dtype=np.float64)
    if len(clean_data) == 0:
        return 0.0
    cutoff = np.percentile(clean_data, quantile * 100.0)
    tail_vals = clean_data[clean_data <= cutoff]
    if len(tail_vals) == 0:
        return float(cutoff)
    return float(np.mean(tail_vals))


@dataclass
class PolicySummaryStats:
    """Aggregate statistical summary for a single policy across Monte Carlo sessions."""

    policy_name: str
    n_sessions: int
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    ci_lower: float
    ci_upper: float
    loss_probability: float
    pnl_5th_pct: float
    cvar_5th: float
    mean_event_sampled_drawdown: float
    event_sampled_drawdown_95th: float
    mean_inventory_util: float
    mean_session_max_abs_inventory: float
    mean_time_avg_abs_inventory: float
    mean_fills: float
    mean_notional: float
    mean_fees: float
    mean_liquidation_cost: float

    @property
    def mean_abs_inventory(self) -> float:
        """Backward-compatible alias for mean_session_max_abs_inventory."""
        return self.mean_session_max_abs_inventory

    @property
    def mean_drawdown(self) -> float:
        """Backward-compatible alias for mean_event_sampled_drawdown."""
        return self.mean_event_sampled_drawdown

    @property
    def drawdown_95th(self) -> float:
        """Backward-compatible alias for event_sampled_drawdown_95th."""
        return self.event_sampled_drawdown_95th


@dataclass
class IndependentMetricComparison:
    """Statistical comparison of a metric between independent Hawkes and Poisson simulation suites."""

    metric_name: str
    hawkes_val: float
    poisson_val: float
    diff_val: float            # Hawkes minus Poisson
    ci_lower: float            # 95% bootstrap CI lower bound
    ci_upper: float            # 95% bootstrap CI upper bound
    is_significant: bool       # True if 0 is outside [ci_lower, ci_upper]


@dataclass
class ProcessComparisonResult:
    """Full independent-sample comparison between Hawkes and Poisson simulation suites for a policy."""

    policy_name: str
    mean_pnl: IndependentMetricComparison
    var_5th: IndependentMetricComparison
    cvar_5th: IndependentMetricComparison
    mean_drawdown: IndependentMetricComparison
    mean_max_inventory: IndependentMetricComparison
    time_avg_inventory: IndependentMetricComparison


def compute_independent_metric_bootstrap_ci(
    data_hawkes: np.ndarray,
    data_poisson: np.ndarray,
    metric_func: Callable[[np.ndarray], float],
    metric_name: str,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> IndependentMetricComparison:
    """Compute non-parametric bootstrap confidence interval for difference between independent samples.

    Evaluates: theta_Hawkes - theta_Poisson
    Recomputes the metric function inside each resample to properly handle nonlinear statistics.
    """
    clean_h = np.asarray(data_hawkes, dtype=np.float64)
    clean_p = np.asarray(data_poisson, dtype=np.float64)

    n_h = len(clean_h)
    n_p = len(clean_p)

    if n_h == 0 or n_p == 0:
        return IndependentMetricComparison(
            metric_name=metric_name,
            hawkes_val=0.0,
            poisson_val=0.0,
            diff_val=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            is_significant=False,
        )

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    point_h = float(metric_func(clean_h))
    point_p = float(metric_func(clean_p))
    point_diff = point_h - point_p

    # Resample independently
    idx_h = rng.integers(0, n_h, size=(n_bootstraps, n_h))
    idx_p = rng.integers(0, n_p, size=(n_bootstraps, n_p))

    diffs = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        sample_h = clean_h[idx_h[b]]
        sample_p = clean_p[idx_p[b]]
        diffs[b] = metric_func(sample_h) - metric_func(sample_p)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))

    # A difference is statistically distinguishable from zero if zero is outside the CI
    is_sig = (ci_lower > 0.0) or (ci_upper < 0.0)

    return IndependentMetricComparison(
        metric_name=metric_name,
        hawkes_val=point_h,
        poisson_val=point_p,
        diff_val=point_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_sig,
    )


def compute_independent_process_comparison(
    hawkes_results: MonteCarloResults,
    poisson_results: MonteCarloResults,
    policy_key: str = "a",  # "a" or "b"
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> ProcessComparisonResult:
    """Evaluate independent-sample differences between Hawkes and Poisson suites.

    Computes bootstrap confidence intervals for:
      - Mean net terminal P&L
      - 5th percentile terminal P&L (VaR_0.05)
      - Worst 5% average P&L (CVaR_0.05 / lower-tail expected shortfall)
      - Mean event-time-sampled maximum drawdown
      - Mean session maximum absolute inventory
      - Time-average absolute inventory
    """
    df_h = hawkes_results.df_sessions
    df_p = poisson_results.df_sessions
    pol_name = "Policy A: Symmetric" if policy_key == "a" else "Policy B: Inventory-Aware"

    pnl_col = f"pnl_{policy_key}"
    dd_col = f"event_sampled_mdd_{policy_key}" if f"event_sampled_mdd_{policy_key}" in df_h else f"drawdown_{policy_key}"
    max_inv_col = f"max_abs_inv_{policy_key}"
    time_inv_col = f"time_avg_inv_{policy_key}" if f"time_avg_inv_{policy_key}" in df_h else f"max_abs_inv_{policy_key}"

    pnl_h = df_h[pnl_col].to_numpy()
    pnl_p = df_p[pnl_col].to_numpy()
    dd_h = df_h[dd_col].to_numpy()
    dd_p = df_p[dd_col].to_numpy()
    max_inv_h = df_h[max_inv_col].to_numpy()
    max_inv_p = df_p[max_inv_col].to_numpy()
    time_inv_h = df_h[time_inv_col].to_numpy()
    time_inv_p = df_p[time_inv_col].to_numpy()

    rng = np.random.default_rng(seed)

    comp_mean = compute_independent_metric_bootstrap_ci(
        pnl_h, pnl_p, metric_func=np.mean, metric_name="Mean Net P&L",
        n_bootstraps=n_bootstraps, rng=rng,
    )
    comp_var = compute_independent_metric_bootstrap_ci(
        pnl_h, pnl_p, metric_func=lambda x: np.percentile(x, 5.0), metric_name="5th Percentile VaR",
        n_bootstraps=n_bootstraps, rng=rng,
    )
    comp_cvar = compute_independent_metric_bootstrap_ci(
        pnl_h, pnl_p, metric_func=lambda x: compute_cvar(x, 0.05), metric_name="Worst 5% CVaR",
        n_bootstraps=n_bootstraps, rng=rng,
    )
    comp_dd = compute_independent_metric_bootstrap_ci(
        dd_h, dd_p, metric_func=np.mean, metric_name="Mean Event-Sampled Max Drawdown",
        n_bootstraps=n_bootstraps, rng=rng,
    )
    comp_max_inv = compute_independent_metric_bootstrap_ci(
        max_inv_h, max_inv_p, metric_func=np.mean, metric_name="Mean Peak Absolute Inventory",
        n_bootstraps=n_bootstraps, rng=rng,
    )
    comp_time_inv = compute_independent_metric_bootstrap_ci(
        time_inv_h, time_inv_p, metric_func=np.mean, metric_name="Time-Average Absolute Inventory",
        n_bootstraps=n_bootstraps, rng=rng,
    )

    return ProcessComparisonResult(
        policy_name=pol_name,
        mean_pnl=comp_mean,
        var_5th=comp_var,
        cvar_5th=comp_cvar,
        mean_drawdown=comp_dd,
        mean_max_inventory=comp_max_inv,
        time_avg_inventory=comp_time_inv,
    )


@dataclass
class PairedComparisonStats:
    """Statistical evaluation of paired differences (Policy B minus Policy A)."""

    mean_diff: float
    median_diff: float
    std_diff: float
    ci_lower: float
    ci_upper: float
    win_rate_b: float          # Fraction of sessions where Policy B outperformed Policy A
    t_stat: float
    p_value_t: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p_val: Optional[float]


@dataclass
class MonteCarloResults:
    """Results from a paired Monte Carlo simulation suite."""

    df_sessions: pd.DataFrame
    stats_a: PolicySummaryStats
    stats_b: PolicySummaryStats
    paired_stats: PairedComparisonStats
    config: ExperimentConfig
    process_type: str


def summarize_policy_metrics(
    pnl_arr: np.ndarray,
    dd_arr: np.ndarray,
    util_arr: np.ndarray,
    max_inv_arr: np.ndarray,
    fills_arr: np.ndarray,
    notional_arr: np.ndarray,
    fees_arr: np.ndarray,
    liq_arr: np.ndarray,
    policy_name: str,
    time_avg_inv_arr: Optional[np.ndarray] = None,
    seed: int = 42,
) -> PolicySummaryStats:
    """Compute aggregate risk and performance statistics for a policy."""
    n = len(pnl_arr)
    mean_pnl, ci_low, ci_high = compute_bootstrap_ci(pnl_arr, rng=seed)
    pnl_5 = float(np.percentile(pnl_arr, 5.0)) if n > 0 else 0.0
    cvar_5 = compute_cvar(pnl_arr, quantile=0.05)
    loss_prob = float(np.mean(pnl_arr < 0.0)) if n > 0 else 0.0
    dd_95 = float(np.percentile(dd_arr, 95.0)) if n > 0 else 0.0

    mean_time_inv = float(np.mean(time_avg_inv_arr)) if time_avg_inv_arr is not None and len(time_avg_inv_arr) > 0 else float(np.mean(max_inv_arr))

    return PolicySummaryStats(
        policy_name=policy_name,
        n_sessions=n,
        mean_pnl=mean_pnl,
        median_pnl=float(np.median(pnl_arr)),
        std_pnl=float(np.std(pnl_arr, ddof=1)) if n > 1 else 0.0,
        ci_lower=ci_low,
        ci_upper=ci_high,
        loss_probability=loss_prob,
        pnl_5th_pct=pnl_5,
        cvar_5th=cvar_5,
        mean_event_sampled_drawdown=float(np.mean(dd_arr)),
        event_sampled_drawdown_95th=dd_95,
        mean_inventory_util=float(np.mean(util_arr)),
        mean_session_max_abs_inventory=float(np.mean(max_inv_arr)),
        mean_time_avg_abs_inventory=mean_time_inv,
        mean_fills=float(np.mean(fills_arr)),
        mean_notional=float(np.mean(notional_arr)),
        mean_fees=float(np.mean(fees_arr)),
        mean_liquidation_cost=float(np.mean(liq_arr)),
    )


def compute_paired_statistics(
    diff_arr: np.ndarray,
    seed: int = 42,
) -> PairedComparisonStats:
    """Compute hypothesis tests and bootstrap confidence interval for paired differences."""
    n = len(diff_arr)
    mean_d, ci_l, ci_u = compute_bootstrap_ci(diff_arr, rng=seed)
    win_b = float(np.mean(diff_arr > 0.0)) if n > 0 else 0.0

    # Paired Student's t-test against zero mean difference
    if n > 1 and np.std(diff_arr) > 1e-12:
        t_res = stats.ttest_1samp(diff_arr, popmean=0.0)
        t_stat = float(t_res.statistic)
        p_val_t = float(t_res.pvalue)
    else:
        t_stat = 0.0
        p_val_t = 1.0

    # Non-parametric Wilcoxon signed-rank test
    wilcoxon_stat = None
    wilcoxon_p = None
    if n >= 10 and np.any(diff_arr != 0.0):
        try:
            w_res = stats.wilcoxon(diff_arr, alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except Exception:
            wilcoxon_stat = None
            wilcoxon_p = None

    return PairedComparisonStats(
        mean_diff=mean_d,
        median_diff=float(np.median(diff_arr)),
        std_diff=float(np.std(diff_arr, ddof=1)) if n > 1 else 0.0,
        ci_lower=ci_l,
        ci_upper=ci_u,
        win_rate_b=win_b,
        t_stat=t_stat,
        p_value_t=p_val_t,
        wilcoxon_stat=wilcoxon_stat,
        wilcoxon_p_val=wilcoxon_p,
    )


def run_monte_carlo_paired(
    config: ExperimentConfig,
    use_poisson: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MonteCarloResults:
    """Execute paired Monte Carlo simulation comparing Policy A and Policy B across paths.

    Parameters
    ----------
    config : ExperimentConfig
        Simulation parameters and number of sessions.
    use_poisson : bool
        If True, use matched Poisson order arrival process.
    progress_callback : callable, optional
        Callback invoked with (completed_sessions, total_sessions).

    Returns
    -------
    MonteCarloResults
        Aggregated session results, summary statistics, and paired differences.
    """
    n_sessions = config.n_sessions
    seed = config.seed
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_sessions)

    rows = []

    for idx, c_seed in enumerate(child_seeds):
        rng = np.random.default_rng(c_seed)
        res = simulate_paired_session(
            hawkes_params=config.hawkes,
            market_params=config.market,
            policy_params=config.policy,
            t_max=config.t_max,
            use_poisson=use_poisson,
            rng=rng,
        )

        ma = res.metrics_a
        mb = res.metrics_b

        rows.append({
            "session_id": idx,
            "n_events": len(res.events),
            "n_buys": res.events.n_buys,
            "n_sells": res.events.n_sells,
            "log_price_return_bps": res.price_path.log_price_return_bps,
            "price_return_bps": res.price_path.log_price_return_bps,
            # Policy A
            "pnl_a": ma.net_terminal_pnl,
            "mtm_pnl_a": ma.mtm_terminal_pnl,
            "liq_cost_a": ma.liquidation_cost,
            "fees_a": ma.total_fees,
            "fills_a": ma.n_fills,
            "notional_a": ma.traded_notional,
            "term_inv_a": ma.terminal_inventory,
            "max_abs_inv_a": ma.max_abs_inventory,
            "time_avg_inv_a": ma.time_average_abs_inventory,
            "inv_util_a": ma.inventory_utilization,
            "event_sampled_mdd_a": ma.event_sampled_max_drawdown,
            "drawdown_a": ma.event_sampled_max_drawdown,
            # Policy B
            "pnl_b": mb.net_terminal_pnl,
            "mtm_pnl_b": mb.mtm_terminal_pnl,
            "liq_cost_b": mb.liquidation_cost,
            "fees_b": mb.total_fees,
            "fills_b": mb.n_fills,
            "notional_b": mb.traded_notional,
            "term_inv_b": mb.terminal_inventory,
            "max_abs_inv_b": mb.max_abs_inventory,
            "time_avg_inv_b": mb.time_average_abs_inventory,
            "inv_util_b": mb.inventory_utilization,
            "event_sampled_mdd_b": mb.event_sampled_max_drawdown,
            "drawdown_b": mb.event_sampled_max_drawdown,
            # Paired difference
            "pnl_diff_b_minus_a": mb.net_terminal_pnl - ma.net_terminal_pnl,
        })

        if progress_callback is not None and (idx % 10 == 0 or idx == n_sessions - 1):
            progress_callback(idx + 1, n_sessions)

    df = pd.DataFrame(rows)

    stats_a = summarize_policy_metrics(
        pnl_arr=df["pnl_a"].to_numpy(),
        dd_arr=df["event_sampled_mdd_a"].to_numpy(),
        util_arr=df["inv_util_a"].to_numpy(),
        max_inv_arr=df["max_abs_inv_a"].to_numpy(),
        time_avg_inv_arr=df["time_avg_inv_a"].to_numpy(),
        fills_arr=df["fills_a"].to_numpy(),
        notional_arr=df["notional_a"].to_numpy(),
        fees_arr=df["fees_a"].to_numpy(),
        liq_arr=df["liq_cost_a"].to_numpy(),
        policy_name="Policy A: Symmetric",
        seed=seed,
    )

    stats_b = summarize_policy_metrics(
        pnl_arr=df["pnl_b"].to_numpy(),
        dd_arr=df["event_sampled_mdd_b"].to_numpy(),
        util_arr=df["inv_util_b"].to_numpy(),
        max_inv_arr=df["max_abs_inv_b"].to_numpy(),
        time_avg_inv_arr=df["time_avg_inv_b"].to_numpy(),
        fills_arr=df["fills_b"].to_numpy(),
        notional_arr=df["notional_b"].to_numpy(),
        fees_arr=df["fees_b"].to_numpy(),
        liq_arr=df["liq_cost_b"].to_numpy(),
        policy_name="Policy B: Inventory-Aware",
        seed=seed + 1,
    )

    paired_stats = compute_paired_statistics(
        diff_arr=df["pnl_diff_b_minus_a"].to_numpy(),
        seed=seed + 2,
    )

    return MonteCarloResults(
        df_sessions=df,
        stats_a=stats_a,
        stats_b=stats_b,
        paired_stats=paired_stats,
        config=config,
        process_type="poisson" if use_poisson else "hawkes",
    )


def run_poisson_vs_hawkes_experiment(
    config: ExperimentConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[MonteCarloResults, MonteCarloResults]:
    """Execute matched-mean Poisson vs Hawkes experiment.

    Returns
    -------
    Tuple[MonteCarloResults, MonteCarloResults]
        (hawkes_results, poisson_results)
    """
    total = config.n_sessions * 2

    def cb_hawkes(done: int, tot: int) -> None:
        if progress_callback:
            progress_callback(done, total)

    def cb_poisson(done: int, tot: int) -> None:
        if progress_callback:
            progress_callback(config.n_sessions + done, total)

    hawkes_res = run_monte_carlo_paired(config, use_poisson=False, progress_callback=cb_hawkes)
    poisson_res = run_monte_carlo_paired(config, use_poisson=True, progress_callback=cb_poisson)

    return hawkes_res, poisson_res


def run_sensitivity_sweep(
    base_config: ExperimentConfig,
    param_name: str,
    param_values: List[float],
    n_sessions: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Run sensitivity analysis across a chosen parameter grid.

    Supported parameters:
      - 'branching_ratio' (varies alpha_self + alpha_cross with fixed beta)
      - 'impact_bps'
      - 'sigma'
      - 'half_spread_bps'
      - 'inventory_skew_bps'
      - 'fee_bps'

    Returns
    -------
    pd.DataFrame
        Table of aggregated metrics for each parameter value.
    """
    total_runs = len(param_values) * n_sessions
    completed_runs = 0

    results_rows = []

    for p_val in param_values:
        # Create modified config
        h_kwargs = {
            "mu": base_config.hawkes.mu,
            "alpha_self": base_config.hawkes.alpha_self,
            "alpha_cross": base_config.hawkes.alpha_cross,
            "beta": base_config.hawkes.beta,
        }
        m_kwargs = {
            "s0": base_config.market.s0,
            "sigma": base_config.market.sigma,
            "impact_bps": base_config.market.impact_bps,
        }
        p_kwargs = {
            "half_spread_bps": base_config.policy.half_spread_bps,
            "max_inventory": base_config.policy.max_inventory,
            "p0": base_config.policy.p0,
            "kappa": base_config.policy.kappa,
            "fee_bps": base_config.policy.fee_bps,
            "inventory_skew_bps": base_config.policy.inventory_skew_bps,
        }

        if param_name == "branching_ratio":
            # Adjust alpha_self and alpha_cross proportionately
            # ratio = (alpha_self + alpha_cross) / beta
            beta = h_kwargs["beta"]
            total_alpha = float(p_val) * beta
            h_kwargs["alpha_self"] = 0.65 * total_alpha
            h_kwargs["alpha_cross"] = 0.35 * total_alpha
        elif param_name == "impact_bps":
            m_kwargs["impact_bps"] = float(p_val)
        elif param_name == "sigma":
            m_kwargs["sigma"] = float(p_val)
        elif param_name == "half_spread_bps":
            p_kwargs["half_spread_bps"] = float(p_val)
        elif param_name == "inventory_skew_bps":
            p_kwargs["inventory_skew_bps"] = float(p_val)
        elif param_name == "fee_bps":
            p_kwargs["fee_bps"] = float(p_val)
        else:
            raise ValueError(f"Unknown sweep parameter '{param_name}'")

        cfg = ExperimentConfig(
            hawkes=HawkesParams(**h_kwargs),
            market=MarketParams(**m_kwargs),
            policy=PolicyParams(**p_kwargs),
            n_sessions=n_sessions,
            seed=base_config.seed,
            t_max=base_config.t_max,
        )

        def step_cb(done: int, tot: int) -> None:
            nonlocal completed_runs
            if progress_callback:
                progress_callback(completed_runs + done, total_runs)

        mc_res = run_monte_carlo_paired(cfg, use_poisson=False, progress_callback=step_cb)
        completed_runs += n_sessions

        sa = mc_res.stats_a
        sb = mc_res.stats_b
        ps = mc_res.paired_stats

        results_rows.append({
            "param_name": param_name,
            "param_value": p_val,
            "mean_pnl_a": sa.mean_pnl,
            "mean_pnl_b": sb.mean_pnl,
            "cvar_5_a": sa.cvar_5th,
            "cvar_5_b": sb.cvar_5th,
            "mdd_a": sa.mean_drawdown,
            "mdd_b": sb.mean_drawdown,
            "inv_util_a": sa.mean_inventory_util,
            "inv_util_b": sb.mean_inventory_util,
            "loss_prob_a": sa.loss_probability,
            "loss_prob_b": sb.loss_probability,
            "mean_diff_b_minus_a": ps.mean_diff,
            "diff_ci_lower": ps.ci_lower,
            "diff_ci_upper": ps.ci_upper,
            "win_rate_b": ps.win_rate_b,
            "p_value_t": ps.p_value_t,
        })

    return pd.DataFrame(results_rows)
