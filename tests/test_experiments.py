"""Unit tests for Monte Carlo experiments, statistical tests, and parameter sweeps."""

import numpy as np
import pytest
from src.config import HawkesParams, MarketParams, PolicyParams, ExperimentConfig
from src.experiments import (
    compute_bootstrap_ci,
    compute_cvar,
    run_monte_carlo_paired,
    run_poisson_vs_hawkes_experiment,
    run_sensitivity_sweep,
)


def test_bootstrap_ci_properties():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mean_val, ci_low, ci_high = compute_bootstrap_ci(data, n_bootstraps=1000, ci_level=0.95, rng=42)

    assert mean_val == pytest.approx(5.5)
    assert ci_low <= mean_val <= ci_high
    assert 2.0 < ci_low < 5.5
    assert 5.5 < ci_high < 9.0


def test_cvar_properties():
    # Uniform 1..100
    data = np.arange(1, 101, dtype=float)
    # Worst 5% are 1, 2, 3, 4, 5
    cvar = compute_cvar(data, quantile=0.05)
    assert cvar == pytest.approx(3.0)


def test_monte_carlo_paired_runs():
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=6.0, alpha_cross=4.0, beta=15.0),
        market=MarketParams(s0=100.0, sigma=0.04, impact_bps=2.0),
        policy=PolicyParams(half_spread_bps=5.0, max_inventory=5, inventory_skew_bps=1.0),
        n_sessions=30,
        seed=101,
    )

    mc_res = run_monte_carlo_paired(cfg, use_poisson=False)

    assert len(mc_res.df_sessions) == 30
    assert mc_res.stats_a.n_sessions == 30
    assert mc_res.stats_b.n_sessions == 30

    # Paired stats
    ps = mc_res.paired_stats
    assert ps.ci_lower <= ps.mean_diff <= ps.ci_upper
    assert 0.0 <= ps.win_rate_b <= 1.0


def test_monte_carlo_standard_error_convergence():
    # As N increases from 25 to 100, standard error of the mean shrinks approximately as 1/sqrt(N)
    cfg_small = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0),
        n_sessions=25,
        seed=42,
    )
    cfg_large = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0),
        n_sessions=100,
        seed=42,
    )

    res_small = run_monte_carlo_paired(cfg_small)
    res_large = run_monte_carlo_paired(cfg_large)

    ci_width_small = res_small.stats_a.ci_upper - res_small.stats_a.ci_lower
    ci_width_large = res_large.stats_a.ci_upper - res_large.stats_a.ci_lower

    # 100 is 4x the sample size, so CI width should be approximately halved (~2x smaller)
    assert ci_width_large < ci_width_small


def test_sensitivity_sweep_execution():
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0),
        n_sessions=15,
        seed=55,
    )

    df_sweep = run_sensitivity_sweep(
        base_config=cfg,
        param_name="half_spread_bps",
        param_values=[3.0, 6.0],
        n_sessions=15,
    )

    assert len(df_sweep) == 2
    assert "mean_pnl_a" in df_sweep.columns
    assert "mean_pnl_b" in df_sweep.columns
    assert "win_rate_b" in df_sweep.columns
