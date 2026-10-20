"""Unit tests for Monte Carlo experiments, statistical tests, and parameter sweeps."""

import numpy as np
import pytest
from src.config import HawkesParams, MarketParams, PolicyParams, ExperimentConfig
from src.experiments import (
    compute_bootstrap_ci,
    compute_cvar,
    compute_independent_metric_bootstrap_ci,
    compute_independent_process_comparison,
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


def test_independent_metric_bootstrap_ci():
    # Test with shifted normal samples
    rng = np.random.default_rng(42)
    sample_a = rng.normal(loc=10.0, scale=1.0, size=200)
    sample_b = rng.normal(loc=5.0, scale=1.0, size=200)

    res = compute_independent_metric_bootstrap_ci(
        sample_a, sample_b, metric_func=np.mean, metric_name="Mean", n_bootstraps=1000, rng=rng
    )

    assert res.metric_name == "Mean"
    assert res.hawkes_val == pytest.approx(10.0, abs=0.2)
    assert res.poisson_val == pytest.approx(5.0, abs=0.2)
    assert res.diff_val == pytest.approx(5.0, abs=0.3)
    assert res.ci_lower > 0.0
    assert res.is_significant is True

    # Test with identical samples -> 0 should be covered in CI
    sample_c = rng.normal(loc=5.0, scale=1.0, size=200)
    res_null = compute_independent_metric_bootstrap_ci(
        sample_b, sample_c, metric_func=np.mean, metric_name="Mean", n_bootstraps=1000, rng=rng
    )
    assert res_null.ci_lower <= 0.0 <= res_null.ci_upper
    assert res_null.is_significant is False


def test_independent_metric_bootstrap_ci_nonlinear_cvar():
    # Verify nonlinear function (CVaR) properly bootstraps inside resample
    rng = np.random.default_rng(123)
    sample_h = rng.uniform(0.0, 100.0, size=150)
    sample_p = rng.uniform(0.0, 100.0, size=150)

    res_cvar = compute_independent_metric_bootstrap_ci(
        sample_h,
        sample_p,
        metric_func=lambda x: compute_cvar(x, quantile=0.05),
        metric_name="CVaR_0.05",
        n_bootstraps=500,
        rng=rng,
    )
    assert np.isfinite(res_cvar.ci_lower)
    assert np.isfinite(res_cvar.ci_upper)
    assert res_cvar.ci_lower <= res_cvar.diff_val <= res_cvar.ci_upper


def test_independent_process_comparison_structure():
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0),
        market=MarketParams(s0=100.0, sigma=0.04, impact_bps=2.0),
        policy=PolicyParams(half_spread_bps=5.0, max_inventory=5, inventory_skew_bps=1.0),
        n_sessions=15,
        seed=77,
    )

    res_h, res_p = run_poisson_vs_hawkes_experiment(cfg)
    comp = compute_independent_process_comparison(res_h, res_p, policy_key="a", n_bootstraps=300, seed=77)

    assert comp.policy_name == "Policy A: Symmetric"
    assert comp.mean_pnl.metric_name == "Mean Net P&L"
    assert comp.var_5th.metric_name == "5th Percentile VaR"
    assert comp.cvar_5th.metric_name == "Worst 5% CVaR"
    assert comp.mean_drawdown.metric_name == "Mean Event-Sampled Max Drawdown"
    assert comp.mean_max_inventory.metric_name == "Mean Peak Absolute Inventory"
    assert comp.time_avg_inventory.metric_name == "Time-Average Absolute Inventory"

    # Metrics should match the individual summary stats
    assert comp.mean_pnl.hawkes_val == pytest.approx(res_h.stats_a.mean_pnl)
    assert comp.mean_pnl.poisson_val == pytest.approx(res_p.stats_a.mean_pnl)
    assert comp.time_avg_inventory.hawkes_val == pytest.approx(res_h.stats_a.mean_time_avg_abs_inventory)
    assert comp.time_avg_inventory.poisson_val == pytest.approx(res_p.stats_a.mean_time_avg_abs_inventory)


def test_monte_carlo_paired_records_new_metrics_and_aliases():
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0),
        market=MarketParams(s0=100.0, sigma=0.04, impact_bps=2.0),
        policy=PolicyParams(half_spread_bps=5.0, max_inventory=5, inventory_skew_bps=1.0),
        n_sessions=20,
        seed=88,
    )

    mc_res = run_monte_carlo_paired(cfg, use_poisson=False)
    df = mc_res.df_sessions

    # Check DataFrame column presence
    assert "log_price_return_bps" in df.columns
    assert "time_avg_inv_a" in df.columns
    assert "time_avg_inv_b" in df.columns
    assert "event_sampled_mdd_a" in df.columns
    assert "event_sampled_mdd_b" in df.columns

    # Verify inventory inequality: session peak inventory must be >= time-average inventory
    assert mc_res.stats_a.mean_session_max_abs_inventory >= mc_res.stats_a.mean_time_avg_abs_inventory
    assert mc_res.stats_b.mean_session_max_abs_inventory >= mc_res.stats_b.mean_time_avg_abs_inventory

    # Check backward compatibility properties on PolicySummaryStats
    assert mc_res.stats_a.mean_abs_inventory == mc_res.stats_a.mean_session_max_abs_inventory
    assert mc_res.stats_a.mean_drawdown == mc_res.stats_a.mean_event_sampled_drawdown
    assert mc_res.stats_a.drawdown_95th == mc_res.stats_a.event_sampled_drawdown_95th


def test_monte_carlo_stream_separation_independent_suites_and_paired_policies():
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=20.0, alpha_self=5.0, alpha_cross=5.0, beta=20.0),
        market=MarketParams(s0=100.0, sigma=0.04, impact_bps=2.0),
        policy=PolicyParams(half_spread_bps=5.0, max_inventory=5, inventory_skew_bps=1.0),
        n_sessions=10,
        seed=123,
    )

    res_h1, res_p1 = run_poisson_vs_hawkes_experiment(cfg)
    res_h2, res_p2 = run_poisson_vs_hawkes_experiment(cfg)

    # 1. Reproducibility: same seed produces identical results
    np.testing.assert_array_equal(
        res_h1.df_sessions["price_return_bps"].to_numpy(),
        res_h2.df_sessions["price_return_bps"].to_numpy(),
    )
    np.testing.assert_array_equal(
        res_p1.df_sessions["price_return_bps"].to_numpy(),
        res_p2.df_sessions["price_return_bps"].to_numpy(),
    )

    # 2. Independent streams: Hawkes suite and Poisson suite have distinct price return paths
    # Because they receive independent spawned seed sequences, their Brownian and arrival draws are independent
    assert not np.allclose(
        res_h1.df_sessions["price_return_bps"].to_numpy(),
        res_p1.df_sessions["price_return_bps"].to_numpy(),
    )

    # 3. Paired testing within suite: Policy A and Policy B share the same path in each session
    assert len(res_h1.df_sessions) == 10
    assert len(res_p1.df_sessions) == 10
    # Win rate and paired differences are computed
    assert res_h1.stats_a.n_sessions == 10
    assert res_p1.stats_a.n_sessions == 10
    assert 0.0 <= res_h1.paired_stats.win_rate_b <= 1.0
    assert 0.0 <= res_p1.paired_stats.win_rate_b <= 1.0


