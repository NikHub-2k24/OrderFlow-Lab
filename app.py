"""OrderFlow Lab: Hawkes Dynamics and Inventory-Aware Market Making.

A quantitative simulation and experimentation platform comparing symmetric and
inventory-aware quoting under endogenous order-flow clustering and price impact.
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from src.config import HawkesParams, MarketParams, PolicyParams, ExperimentConfig
from src.simulation import simulate_paired_session, PairedSessionResult
from src.experiments import (
    run_monte_carlo_paired,
    run_poisson_vs_hawkes_experiment,
    run_sensitivity_sweep,
    MonteCarloResults,
)
from src.visualization import (
    plot_single_session_deep_dive,
    plot_pnl_distribution_comparison,
    plot_paired_differences,
    plot_tail_risk_metrics_bar,
    plot_poisson_vs_hawkes_comparison,
    plot_sensitivity_curve,
)

# Page configuration
st.set_page_config(
    page_title="OrderFlow Lab | Hawkes & Inventory-Aware Quoting",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle institutional CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.65rem;
        font-weight: 650;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
    }
    .stat-badge-pos {
        color: #15803d;
        font-weight: 600;
    }
    .stat-badge-neg {
        color: #b91c1c;
        font-weight: 600;
    }
    .callout-box {
        background-color: #f1f5f9;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
        color: #334155;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 3px solid #d97706;
        padding: 10px 14px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
        color: #92400e;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached simulation runners to guarantee rapid interaction and offline determinism
@st.cache_data(show_spinner=False)
def get_cached_single_session(
    mu: float,
    alpha_self: float,
    alpha_cross: float,
    beta: float,
    s0: float,
    sigma: float,
    impact_bps: float,
    half_spread_bps: float,
    max_inventory: int,
    p0: float,
    kappa: float,
    fee_bps: float,
    inventory_skew_bps: float,
    use_poisson: bool,
    seed: int,
) -> PairedSessionResult:
    h_p = HawkesParams(mu=mu, alpha_self=alpha_self, alpha_cross=alpha_cross, beta=beta)
    m_p = MarketParams(s0=s0, sigma=sigma, impact_bps=impact_bps)
    p_p = PolicyParams(
        half_spread_bps=half_spread_bps,
        max_inventory=max_inventory,
        p0=p0,
        kappa=kappa,
        fee_bps=fee_bps,
        inventory_skew_bps=inventory_skew_bps,
    )
    return simulate_paired_session(
        hawkes_params=h_p,
        market_params=m_p,
        policy_params=p_p,
        t_max=1.0,
        use_poisson=use_poisson,
        rng=seed,
    )


@st.cache_data(show_spinner="Running paired Monte Carlo simulations...")
def get_cached_monte_carlo(
    mu: float,
    alpha_self: float,
    alpha_cross: float,
    beta: float,
    s0: float,
    sigma: float,
    impact_bps: float,
    half_spread_bps: float,
    max_inventory: int,
    p0: float,
    kappa: float,
    fee_bps: float,
    inventory_skew_bps: float,
    n_sessions: int,
    use_poisson: bool,
    seed: int,
) -> MonteCarloResults:
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=mu, alpha_self=alpha_self, alpha_cross=alpha_cross, beta=beta),
        market=MarketParams(s0=s0, sigma=sigma, impact_bps=impact_bps),
        policy=PolicyParams(
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
        ),
        n_sessions=n_sessions,
        seed=seed,
    )
    return run_monte_carlo_paired(cfg, use_poisson=use_poisson)


@st.cache_data(show_spinner="Running Poisson vs Hawkes paired experiment...")
def get_cached_poisson_vs_hawkes(
    mu: float,
    alpha_self: float,
    alpha_cross: float,
    beta: float,
    s0: float,
    sigma: float,
    impact_bps: float,
    half_spread_bps: float,
    max_inventory: int,
    p0: float,
    kappa: float,
    fee_bps: float,
    inventory_skew_bps: float,
    n_sessions: int,
    seed: int,
):
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=mu, alpha_self=alpha_self, alpha_cross=alpha_cross, beta=beta),
        market=MarketParams(s0=s0, sigma=sigma, impact_bps=impact_bps),
        policy=PolicyParams(
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
        ),
        n_sessions=n_sessions,
        seed=seed,
    )
    return run_poisson_vs_hawkes_experiment(cfg)


@st.cache_data(show_spinner="Evaluating sensitivity grid...")
def get_cached_sensitivity_sweep(
    mu: float,
    alpha_self: float,
    alpha_cross: float,
    beta: float,
    s0: float,
    sigma: float,
    impact_bps: float,
    half_spread_bps: float,
    max_inventory: int,
    p0: float,
    kappa: float,
    fee_bps: float,
    inventory_skew_bps: float,
    param_name: str,
    param_values: list[float],
    n_sessions: int,
    seed: int,
) -> pd.DataFrame:
    cfg = ExperimentConfig(
        hawkes=HawkesParams(mu=mu, alpha_self=alpha_self, alpha_cross=alpha_cross, beta=beta),
        market=MarketParams(s0=s0, sigma=sigma, impact_bps=impact_bps),
        policy=PolicyParams(
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
        ),
        n_sessions=n_sessions,
        seed=seed,
    )
    return run_sensitivity_sweep(cfg, param_name, param_values, n_sessions=n_sessions)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & MODEL VALIDATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### OrderFlow Lab")
    st.markdown(
        "<span style='font-size:0.8rem; color:#64748b;'>Quantitative Microstructure Simulator</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.expander("Hawkes Order-Flow Dynamics", expanded=True):
        mu = st.number_input(
            "Base rate per side (mu)",
            min_value=1.0,
            max_value=200.0,
            value=30.0,
            step=5.0,
            help="Autonomous baseline arrival rate per side per normalized session",
        )
        alpha_self = st.number_input(
            "Self-excitation (alpha_self)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=1.0,
            help="Intensity jump induced by same-side market order",
        )
        alpha_cross = st.number_input(
            "Cross-excitation (alpha_cross)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=1.0,
            help="Intensity jump induced by opposite-side market order",
        )
        beta = st.number_input(
            "Decay rate (beta)",
            min_value=1.0,
            max_value=200.0,
            value=20.0,
            step=2.0,
            help="Exponential memory decay per normalized session",
        )

        branching_ratio = (alpha_self + alpha_cross) / beta
        is_stable = branching_ratio < 1.0

        if is_stable:
            mean_rate_side = mu / (1.0 - branching_ratio)
            tot_rate = 2.0 * mean_rate_side
            st.markdown(
                f"""
                <div style='background:#f8fafc; padding:8px; border-radius:4px; font-size:0.8rem; border:1px solid #cbd5e1;'>
                <b>Branching Ratio:</b> <code>{branching_ratio:.3f}</code> (&lt; 1.0 Subcritical)<br>
                <b>Mean Rate / Side:</b> <code>{mean_rate_side:.1f}</code> / sess<br>
                <b>Total Expected Rate:</b> <code>{tot_rate:.1f}</code> / sess
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(
                f"Unstable Hawkes Process! Branching ratio ({branching_ratio:.3f}) >= 1.0. "
                "Adjust parameters so that (alpha_self + alpha_cross) < beta."
            )

    with st.expander("Reference-Price Dynamics", expanded=False):
        s0 = st.number_input("Initial Mid Price ($)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
        sigma = st.number_input("Brownian Volatility (sigma)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        impact_bps = st.number_input("Price Impact per Order (bps)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)

    with st.expander("Market-Maker Policies & Fees", expanded=False):
        half_spread_bps = st.number_input("Quoted Half-Spread (bps)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
        max_inventory = st.number_input("Hard Inventory Limit (Q_max)", min_value=1, max_value=50, value=10, step=1)
        inventory_skew_bps = st.number_input(
            "Policy B Inventory Skew (bps/unit)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.25,
            help="Quote shift per unit of signed inventory (clamped to +/-0.8 * half_spread)",
        )
        p0 = st.number_input("Base Fill Probability (p0)", min_value=0.05, max_value=1.0, value=0.8, step=0.05)
        kappa = st.number_input("Fill Decay Sensitivity (kappa)", min_value=0.0, max_value=2.0, value=0.15, step=0.02)
        fee_bps = st.number_input("Transaction Fee (bps notional)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

    with st.expander("Monte Carlo Configuration", expanded=True):
        n_sessions = st.slider("Number of Sessions (N)", min_value=20, max_value=500, value=150, step=10)
        sim_seed = st.number_input("Simulation Seed", min_value=0, max_value=999999, value=42, step=1)

    run_btn = st.button("Recompute Simulation Suite", type="primary", use_container_width=True)

    st.markdown(
        """
        <div style='font-size:0.75rem; color:#94a3b8; margin-top:15px; line-height:1.3;'>
        <b>Offline Operational Mode:</b><br>
        All processes evaluated numerically with deterministic random streams. No runtime external APIs.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Stop execution gracefully if parameters are physically unstable
if not is_stable:
    st.error(
        f"Parameter Validation Error: Branching ratio {branching_ratio:.4f} >= 1.0 violates subcriticality. "
        "Hawkes processes with branching ratio >= 1 produce infinite intensity cascades. "
        "Please reduce alpha_self or alpha_cross, or increase decay rate beta in the sidebar."
    )
    st.stop()


# -----------------------------------------------------------------------------
# MAIN APP TABS & RESEARCH WORKFLOW
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">OrderFlow Lab: Hawkes Dynamics & Inventory-Aware Quoting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Continuous-Time Microstructure Simulation & Risk Analysis Platform | '
    'Normalized Session Time t &isin; [0, 1]</div>',
    unsafe_allow_html=True,
)

tab_overview, tab_single, tab_mc, tab_experiments, tab_methodology, tab_limitations = st.tabs([
    "Research Framework",
    "Single-Session Deep Dive",
    "Paired Monte Carlo Analysis",
    "Research Experiments",
    "Methodology & Equations",
    "Limitations & Falsification",
])


# =============================================================================
# TAB 1: OVERVIEW & RESEARCH FRAMEWORK
# =============================================================================
with tab_overview:
    st.markdown("### Research Framework & Core Hypotheses")

    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Research Question 1</div>
                <div style="font-size:0.95rem; font-weight:600; color:#1e293b; margin-top:4px;">
                Endogenous Clustering vs Poisson
                </div>
                <p style="font-size:0.82rem; color:#475569; margin-top:6px; line-height:1.4;">
                How does temporal and cross-excitation clustering in order flow distort market-maker inventory risk
                relative to memoryless Poisson arrivals with identical theoretical mean intensity?
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Research Question 2</div>
                <div style="font-size:0.95rem; font-weight:600; color:#1e293b; margin-top:4px;">
                Inventory-Aware Skew vs Symmetric Quoting
                </div>
                <p style="font-size:0.82rem; color:#475569; margin-top:6px; line-height:1.4;">
                Can asymmetric reservation-price quoting reduce inventory tail exposure and residual liquidation penalties
                without excessively sacrificing spread capture or fill frequency?
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_q3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Research Question 3</div>
                <div style="font-size:0.95rem; font-weight:600; color:#1e293b; margin-top:4px;">
                Frictions, Volatility & Adverse Selection
                </div>
                <p style="font-size:0.82rem; color:#475569; margin-top:6px; line-height:1.4;">
                How do permanent order impact (adverse selection), diffusion volatility, and transaction fees interact
                to shift terminal P&L distributions and Conditional Value-at-Risk (worst 5% CVaR)?
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Simulation Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The laboratory runs a paired, path-matched experiment where two quoting policies compete under the exact same
        order sequence and price path:
        """
    )

    st.markdown(
        """
        ```
        +----------------------------------------------------------------------------------------+
        |                                    STOCHASTIC GENERATOR                                |
        |                                                                                        |
        |   Bivariate Hawkes Process (Ogata Thinning)  OR  Matched-Mean Poisson Baseline        |
        |   Buy Arrival Rate:  lambda_buy(t)  = mu + alpha_self * x_b(t) + alpha_cross * x_s(t)  |
        |   Sell Arrival Rate: lambda_sell(t) = mu + alpha_self * x_s(t) + alpha_cross * x_b(t)  |
        +-------------------------------------------+--------------------------------------------+
                                                    |  (Order Flow Timestamps & Signs)
                                                    v
        +----------------------------------------------------------------------------------------+
        |                           SYNTHETIC REFERENCE-PRICE PATH                               |
        |                                                                                        |
        |   delta_log_price = sigma * sqrt(delta_t) * Z + sign * (impact_bps / 10000)            |
        |   (Exogenous price path: Brownian noise + broader-market signed order-flow impact)     |
        +-------------------------------------------+--------------------------------------------+
                                                    |  Identical Path & Common Random Draws
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
        +----------------------------------+                  +----------------------------------+
        |       POLICY A: SYMMETRIC        |                  |    POLICY B: INVENTORY-AWARE     |
        |                                  |                  |                                  |
        |  Bid Dist:  half_spread_bps      |                  |  Skew = clip(c*q, -0.8s, +0.8s)  |
        |  Ask Dist:  half_spread_bps      |                  |  Bid Dist: half_spread + Skew    |
        |  Hard Limit: +/- Q_max           |                  |  Ask Dist: half_spread - Skew    |
        +----------------+-----------------+                  +----------------+-----------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                                    v
        +----------------------------------------------------------------------------------------+
        |                          ACCOUNTING & LIQUIDATION ENGINE                               |
        |                                                                                        |
        |   - Execution price, fee deduction (fee_bps), cash & signed inventory tracking        |
        |   - Continuous Mark-to-Market Equity & Session Drawdown                                |
        |   - Terminal Liquidation: Residual inventory liquidated at (mid -/+ half_spread)       |
        |     plus liquidation transaction fee. Ensures no inflated MTM reporting.              |
        +----------------------------------------------------------------------------------------+
        ```
        """
    )

    col_assump, col_param_summary = st.columns([1.2, 0.8])
    with col_assump:
        st.markdown(
            """
            <div class="callout-box">
            <b>Core Assumptions & Disclosures:</b>
            <ul>
                <li><b>Stylized Order Book:</b> Fills are governed by a conditional exponential probability model
                <code>p_fill = min(1, p0 * exp(-kappa * distance))</code> rather than explicit limit-order queue priority,
                cancellations, or latency.</li>
                <li><b>Exogenous Impact:</b> Each trade permanently shifts the log mid-price by <code>+/- impact_bps</code>
                representing stylized broader market reaction, independent of whether our dealer fills.</li>
                <li><b>Terminal Liquidation Penalty:</b> Residual inventory at session end is forced to cross the base
                half-spread and pay liquidation fees. Net terminal P&L is strictly bounded above by mark-to-mid equity.</li>
                <li><b>Non-Commercial Research:</b> These are synthetic simulations for academic study of risk mechanics,
                not live trading backtests or claims of profitability.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_param_summary:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Active Parameter State</div>
                <table style="width:100%; font-size:0.8rem; color:#334155; margin-top:4px;">
                    <tr><td><b>Hawkes mu:</b></td><td>{mu:.1f} / sess</td></tr>
                    <tr><td><b>Alpha Self / Cross:</b></td><td>{alpha_self:.1f} / {alpha_cross:.1f}</td></tr>
                    <tr><td><b>Decay Beta:</b></td><td>{beta:.1f} / sess</td></tr>
                    <tr><td><b>Branching Ratio:</b></td><td>{branching_ratio:.3f}</td></tr>
                    <tr><td><b>Matched Mean Rate:</b></td><td>{mu / (1.0 - branching_ratio):.1f} / side</td></tr>
                    <tr><td><b>Brownian Sigma:</b></td><td>{sigma:.3f} / &radic;sess</td></tr>
                    <tr><td><b>Event Impact:</b></td><td>{impact_bps:.1f} bps</td></tr>
                    <tr><td><b>Half-Spread:</b></td><td>{half_spread_bps:.1f} bps</td></tr>
                    <tr><td><b>Inventory Limit:</b></td><td>&plusmn;{max_inventory} units</td></tr>
                    <tr><td><b>Inventory Skew:</b></td><td>{inventory_skew_bps:.1f} bps/unit</td></tr>
                    <tr><td><b>Fee Rate:</b></td><td>{fee_bps:.1f} bps</td></tr>
                    <tr><td><b>Simulation Seed:</b></td><td><code>{sim_seed}</code></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 2: SINGLE-SESSION DEEP DIVE
# =============================================================================
with tab_single:
    st.markdown("### Single-Session Inspection & Microstructure Micro-Dynamics")
    st.markdown(
        "Inspect event arrivals, intensity trajectories, posted bid/ask quotes, actual dealer fills, "
        "inventory evolution, and cumulative mark-to-market equity along a single market path."
    )

    col_single_ctrl1, col_single_ctrl2, col_single_ctrl3 = st.columns([1, 1, 2])
    with col_single_ctrl1:
        single_seed = st.number_input("Session Inspection Seed", value=int(sim_seed), step=1, key="single_seed")
    with col_single_ctrl2:
        compare_process = st.selectbox("Arrival Process", ["Hawkes (Clustered)", "Matched Poisson (Memoryless)"])
        use_poisson_single = (compare_process == "Matched Poisson (Memoryless)")

    # Compute single session result
    session_res = get_cached_single_session(
        mu=mu,
        alpha_self=alpha_self,
        alpha_cross=alpha_cross,
        beta=beta,
        s0=s0,
        sigma=sigma,
        impact_bps=impact_bps,
        half_spread_bps=half_spread_bps,
        max_inventory=max_inventory,
        p0=p0,
        kappa=kappa,
        fee_bps=fee_bps,
        inventory_skew_bps=inventory_skew_bps,
        use_poisson=use_poisson_single,
        seed=int(single_seed),
    )

    ma = session_res.metrics_a
    mb = session_res.metrics_b

    # Metric comparison row
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Market Orders</div>
                <div class="metric-value">{len(session_res.events)}</div>
                <div class="metric-sub">{session_res.events.n_buys} Buys | {session_res.events.n_sells} Sells</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mid-Price Return</div>
                <div class="metric-value">{session_res.price_path.total_return_bps:+.1f} <span style="font-size:0.8rem;">bps</span></div>
                <div class="metric-sub">${session_res.price_path.initial_mid_price:.2f} &rarr; ${session_res.price_path.terminal_mid_price:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m3:
        pnl_diff = mb.net_terminal_pnl - ma.net_terminal_pnl
        diff_class = "stat-badge-pos" if pnl_diff >= 0 else "stat-badge-neg"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Net P&L (Policy A vs B)</div>
                <div class="metric-value">
                    <span style="color:#475569;">${ma.net_terminal_pnl:+.2f}</span> /
                    <span style="color:#1d4ed8;">${mb.net_terminal_pnl:+.2f}</span>
                </div>
                <div class="metric-sub">Diff (B - A): <span class="{diff_class}">${pnl_diff:+.2f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Terminal Inventory</div>
                <div class="metric-value">
                    <span style="color:#475569;">{ma.terminal_inventory:+d}</span> /
                    <span style="color:#1d4ed8;">{mb.terminal_inventory:+d}</span>
                </div>
                <div class="metric-sub">Max Abs: {ma.max_abs_inventory} (A) vs {mb.max_abs_inventory} (B)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Liquidation Cost</div>
                <div class="metric-value">${ma.liquidation_cost:.2f} / ${mb.liquidation_cost:.2f}</div>
                <div class="metric-sub">Fees: ${ma.total_fees:.2f} (A) / ${mb.total_fees:.2f} (B)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4-panel interactive chart
    fig_single = plot_single_session_deep_dive(
        session_res,
        hawkes_params=HawkesParams(mu=mu, alpha_self=alpha_self, alpha_cross=alpha_cross, beta=beta),
        policy_params=PolicyParams(
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
        ),
    )
    st.plotly_chart(fig_single, use_container_width=True)


# =============================================================================
# TAB 3: PAIRED MONTE CARLO ANALYSIS
# =============================================================================
with tab_mc:
    st.markdown("### Paired Monte Carlo Simulation & Risk Distributions")
    st.markdown(
        f"Evaluating **{n_sessions} paired paths** generated under identical seeds and Common Random Numbers (CRN). "
        "Every market path is evaluated simultaneously across both Policy A (Symmetric) and Policy B (Inventory-Aware)."
    )

    mc_results = get_cached_monte_carlo(
        mu=mu,
        alpha_self=alpha_self,
        alpha_cross=alpha_cross,
        beta=beta,
        s0=s0,
        sigma=sigma,
        impact_bps=impact_bps,
        half_spread_bps=half_spread_bps,
        max_inventory=max_inventory,
        p0=p0,
        kappa=kappa,
        fee_bps=fee_bps,
        inventory_skew_bps=inventory_skew_bps,
        n_sessions=n_sessions,
        use_poisson=False,
        seed=int(sim_seed),
    )

    sa = mc_results.stats_a
    sb = mc_results.stats_b
    ps = mc_results.paired_stats

    # High-level statistical comparison table
    col_t1, col_t2 = st.columns([1.2, 0.8])
    with col_t1:
        st.markdown('<div class="section-header">Performance & Risk Metrics Across Replicates</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame(
            [
                {
                    "Metric": "Mean Net Terminal P&L",
                    "Policy A: Symmetric": f"${sa.mean_pnl:.2f} [${sa.ci_lower:.2f}, ${sa.ci_upper:.2f}]",
                    "Policy B: Inventory-Aware": f"${sb.mean_pnl:.2f} [${sb.ci_lower:.2f}, ${sb.ci_upper:.2f}]",
                    "Paired Difference (B - A)": f"${ps.mean_diff:.2f} [${ps.ci_lower:.2f}, ${ps.ci_upper:.2f}]",
                },
                {
                    "Metric": "Median Net Terminal P&L",
                    "Policy A: Symmetric": f"${sa.median_pnl:.2f}",
                    "Policy B: Inventory-Aware": f"${sb.median_pnl:.2f}",
                    "Paired Difference (B - A)": f"${ps.median_diff:.2f}",
                },
                {
                    "Metric": "Sample Standard Deviation",
                    "Policy A: Symmetric": f"${sa.std_pnl:.2f}",
                    "Policy B: Inventory-Aware": f"${sb.std_pnl:.2f}",
                    "Paired Difference (B - A)": f"${ps.std_diff:.2f}",
                },
                {
                    "Metric": "5th Percentile P&L (95% VaR)",
                    "Policy A: Symmetric": f"${sa.pnl_5th_pct:.2f}",
                    "Policy B: Inventory-Aware": f"${sb.pnl_5th_pct:.2f}",
                    "Paired Difference (B - A)": f"${sb.pnl_5th_pct - sa.pnl_5th_pct:+.2f}",
                },
                {
                    "Metric": "Worst 5% Average (Expected Shortfall / CVaR)",
                    "Policy A: Symmetric": f"${sa.cvar_5th:.2f}",
                    "Policy B: Inventory-Aware": f"${sb.cvar_5th:.2f}",
                    "Paired Difference (B - A)": f"${sb.cvar_5th - sa.cvar_5th:+.2f}",
                },
                {
                    "Metric": "Probability of Losing Session (P < 0)",
                    "Policy A: Symmetric": f"{sa.loss_probability:.1%}",
                    "Policy B: Inventory-Aware": f"{sb.loss_probability:.1%}",
                    "Paired Difference (B - A)": f"{sb.loss_probability - sa.loss_probability:+.1%}",
                },
                {
                    "Metric": "Mean Max Drawdown",
                    "Policy A: Symmetric": f"${sa.mean_drawdown:.2f}",
                    "Policy B: Inventory-Aware": f"${sb.mean_drawdown:.2f}",
                    "Paired Difference (B - A)": f"${sb.mean_drawdown - sa.mean_drawdown:+.2f}",
                },
                {
                    "Metric": "Mean Inventory Utilization (|q|_max / Q_max)",
                    "Policy A: Symmetric": f"{sa.mean_inventory_util:.1%}",
                    "Policy B: Inventory-Aware": f"{sb.mean_inventory_util:.1%}",
                    "Paired Difference (B - A)": f"{sb.mean_inventory_util - sa.mean_inventory_util:+.1%}",
                },
                {
                    "Metric": "Mean Traded Fills / Session",
                    "Policy A: Symmetric": f"{sa.mean_fills:.1f}",
                    "Policy B: Inventory-Aware": f"{sb.mean_fills:.1f}",
                    "Paired Difference (B - A)": f"{sb.mean_fills - sa.mean_fills:+.1f}",
                },
                {
                    "Metric": "Mean Liquidation Cost",
                    "Policy A: Symmetric": f"${sa.mean_liquidation_cost:.2f}",
                    "Policy B: Inventory-Aware": f"${sb.mean_liquidation_cost:.2f}",
                    "Paired Difference (B - A)": f"${sb.mean_liquidation_cost - sa.mean_liquidation_cost:+.2f}",
                },
            ]
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    with col_t2:
        st.markdown('<div class="section-header">Hypothesis Test: Policy Difference</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Paired Difference Mean (B - A)</div>
                <div class="metric-value">${ps.mean_diff:.2f}</div>
                <div class="metric-sub">95% Bootstrap CI: <b>[${ps.ci_lower:.2f}, ${ps.ci_upper:.2f}]</b></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Policy B Win Rate (Path Outperformance)</div>
                <div class="metric-value">{ps.win_rate_b:.1%}</div>
                <div class="metric-sub">Fraction of paths where Net P&L(B) &gt; Net P&L(A)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Paired Student's t-Test</div>
                <div style="font-size:0.95rem; font-weight:600; color:#1e293b;">
                    t = {ps.t_stat:.2f} | p-value = {ps.p_value_t:.4e}
                </div>
                <div class="metric-sub">
                    {'Statistically significant at 5% level' if ps.p_value_t < 0.05 else 'Not statistically significant at 5% level'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Charts: Distribution & Paired Diffs
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_dist = plot_pnl_distribution_comparison(mc_results)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_c2:
        fig_diff = plot_paired_differences(mc_results)
        st.plotly_chart(fig_diff, use_container_width=True)

    # Risk metrics bar chart
    fig_tail = plot_tail_risk_metrics_bar(mc_results)
    st.plotly_chart(fig_tail, use_container_width=True)

    st.markdown(
        """
        <div class="callout-box">
        <b>Quantitative Interpretation:</b><br>
        Inventory-aware quoting (Policy B) acts as a dynamic risk dampener: by shading bids downward and asks upward
        when long (and vice versa when short), Policy B actively attracts order flow on the inventory-reducing side.
        This compresses inventory variance, reduces the probability of hitting the hard boundary Q_max, and dampens
        terminal liquidation penalties. However, in return regimes where adverse selection is mild, Policy B may capture
        slightly fewer total fills or narrower margins on liquidation trades.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 4: RESEARCH EXPERIMENTS
# =============================================================================
with tab_experiments:
    st.markdown("### Predefined Quantitative Experiments")

    exp_choice = st.radio(
        "Select Experiment",
        [
            "Experiment A: Clustered Hawkes vs Matched-Mean Poisson Baseline",
            "Experiment B: Inventory Control Ablation (Skew Strength Sweep)",
            "Experiment C: Adverse Selection & Volatility Stress Analysis",
        ],
        horizontal=True,
    )

    if exp_choice.startswith("Experiment A"):
        st.markdown(
            """
            **Research Objective:** Isolate the pure effect of order clustering.
            We compare the Hawkes process against an independent Poisson arrival process that has the **exact same
            theoretical mean intensity per side** `mean_rate = mu / (1 - branching_ratio)`.
            Any difference in market-maker tail risk and inventory dispersion is strictly attributable to autocorrelation
            and self/cross-excitation rather than raw order volume.
            """
        )

        h_res, p_res = get_cached_poisson_vs_hawkes(
            mu=mu,
            alpha_self=alpha_self,
            alpha_cross=alpha_cross,
            beta=beta,
            s0=s0,
            sigma=sigma,
            impact_bps=impact_bps,
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
            n_sessions=n_sessions,
            seed=int(sim_seed),
        )

        fig_p_vs_h = plot_poisson_vs_hawkes_comparison(h_res, p_res)
        st.plotly_chart(fig_p_vs_h, use_container_width=True)

        col_ha1, col_ha2 = st.columns(2)
        with col_ha1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Hawkes Process Outcomes (Clustered)</div>
                    <ul style="font-size:0.82rem; color:#334155; margin:6px 0 0 16px; padding:0;">
                        <li>Policy A Mean P&L: <b>${h_res.stats_a.mean_pnl:.2f}</b> (Std: ${h_res.stats_a.std_pnl:.2f})</li>
                        <li>Policy B Mean P&L: <b>${h_res.stats_b.mean_pnl:.2f}</b> (Std: ${h_res.stats_b.std_pnl:.2f})</li>
                        <li>Policy A Worst 5% CVaR: <b>${h_res.stats_a.cvar_5th:.2f}</b></li>
                        <li>Policy B Worst 5% CVaR: <b>${h_res.stats_b.cvar_5th:.2f}</b></li>
                        <li>Policy A Mean Inv Util: <b>{h_res.stats_a.mean_inventory_util:.1%}</b></li>
                        <li>Policy B Mean Inv Util: <b>{h_res.stats_b.mean_inventory_util:.1%}</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_ha2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Poisson Baseline Outcomes (Memoryless)</div>
                    <ul style="font-size:0.82rem; color:#334155; margin:6px 0 0 16px; padding:0;">
                        <li>Policy A Mean P&L: <b>${p_res.stats_a.mean_pnl:.2f}</b> (Std: ${p_res.stats_a.std_pnl:.2f})</li>
                        <li>Policy B Mean P&L: <b>${p_res.stats_b.mean_pnl:.2f}</b> (Std: ${p_res.stats_b.std_pnl:.2f})</li>
                        <li>Policy A Worst 5% CVaR: <b>${p_res.stats_a.cvar_5th:.2f}</b></li>
                        <li>Policy B Worst 5% CVaR: <b>${p_res.stats_b.cvar_5th:.2f}</b></li>
                        <li>Policy A Mean Inv Util: <b>{p_res.stats_a.mean_inventory_util:.1%}</b></li>
                        <li>Policy B Mean Inv Util: <b>{p_res.stats_b.mean_inventory_util:.1%}</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="callout-box">
            <b>Clustering Insight:</b> Under Hawkes dynamics, consecutive orders arrive in micro-bursts. When an order
            arrives on the buy side, self-excitation increases the immediate probability of subsequent buy orders.
            Because market orders exert price impact, the dealer accumulates adverse one-sided inventory exactly during
            directional runs, causing significantly heavier left-tail drawdowns under Hawkes than under memoryless Poisson arrivals.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif exp_choice.startswith("Experiment B"):
        st.markdown(
            """
            **Research Objective:** Quantify the trade-off in inventory skew sensitivity.
            We sweep `inventory_skew_bps` from `0.0` (which collapses Policy B into Policy A) up to `4.0 bps/unit`.
            Notice how increasing skew strength progressively trims inventory utilization and tail risk, but beyond an
            optimal threshold, excessive skew reduces fill rates on quote edges.
            """
        )

        skew_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        df_skew = get_cached_sensitivity_sweep(
            mu=mu,
            alpha_self=alpha_self,
            alpha_cross=alpha_cross,
            beta=beta,
            s0=s0,
            sigma=sigma,
            impact_bps=impact_bps,
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
            param_name="inventory_skew_bps",
            param_values=skew_vals,
            n_sessions=min(n_sessions, 100),
            seed=int(sim_seed),
        )

        fig_skew = plot_sensitivity_curve(df_skew, "inventory_skew_bps (bps/unit)")
        st.plotly_chart(fig_skew, use_container_width=True)
        st.dataframe(df_skew, hide_index=True, use_container_width=True)

    else:
        st.markdown(
            """
            **Research Objective:** Stress-test policies under varying market conditions.
            Select a risk driver to sweep:
            """
        )
        sweep_target = st.selectbox(
            "Sweep Parameter",
            ["impact_bps", "sigma", "half_spread_bps", "fee_bps", "branching_ratio"],
        )

        if sweep_target == "impact_bps":
            test_vals = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
        elif sweep_target == "sigma":
            test_vals = [0.01, 0.03, 0.05, 0.08, 0.12, 0.18]
        elif sweep_target == "half_spread_bps":
            test_vals = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
        elif sweep_target == "fee_bps":
            test_vals = [0.0, 0.5, 1.0, 2.0, 4.0]
        elif sweep_target == "branching_ratio":
            test_vals = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
        else:
            test_vals = [1.0, 2.0]

        df_stress = get_cached_sensitivity_sweep(
            mu=mu,
            alpha_self=alpha_self,
            alpha_cross=alpha_cross,
            beta=beta,
            s0=s0,
            sigma=sigma,
            impact_bps=impact_bps,
            half_spread_bps=half_spread_bps,
            max_inventory=max_inventory,
            p0=p0,
            kappa=kappa,
            fee_bps=fee_bps,
            inventory_skew_bps=inventory_skew_bps,
            param_name=sweep_target,
            param_values=test_vals,
            n_sessions=min(n_sessions, 80),
            seed=int(sim_seed),
        )

        fig_stress = plot_sensitivity_curve(df_stress, sweep_target)
        st.plotly_chart(fig_stress, use_container_width=True)
        st.dataframe(df_stress, hide_index=True, use_container_width=True)


# =============================================================================
# TAB 5: METHODOLOGY & EQUATIONS
# =============================================================================
with tab_methodology:
    st.markdown("### Mathematical Methodology & Formal Specification")

    st.markdown("#### 1. Bivariate Hawkes Process with Exponential Kernel")
    st.latex(
        r"""
        \begin{aligned}
        \lambda_{\text{buy}}(t) &= \mu + \alpha_{\text{self}} \, x_{\text{buy}}(t) + \alpha_{\text{cross}} \, x_{\text{sell}}(t) \\
        \lambda_{\text{sell}}(t) &= \mu + \alpha_{\text{self}} \, x_{\text{sell}}(t) + \alpha_{\text{cross}} \, x_{\text{buy}}(t)
        \end{aligned}
        """
    )
    st.markdown(
        r"""
        Where $x_{\text{buy}}(t)$ and $x_{\text{sell}}(t)$ are decaying event-history states governed between arrivals by:
        """
    )
    st.latex(r"x_k(t + \Delta t) = x_k(t) e^{-\beta \Delta t}, \quad k \in \{\text{buy}, \text{sell}\}")
    st.markdown(
        r"""
        Immediately following an accepted buy arrival, $x_{\text{buy}}(t^+) = x_{\text{buy}}(t) + 1$;
        after a sell arrival, $x_{\text{sell}}(t^+) = x_{\text{sell}}(t) + 1$.
        The branching ratio $\eta$ is defined as:
        """
    )
    st.latex(r"\eta = \frac{\alpha_{\text{self}} + \alpha_{\text{cross}}}{\beta}")
    st.markdown(
        r"""
        **Subcritical Stability Criterion:** Stationarity requires $\eta < 1.0$.
        Under this condition, the stationary theoretical mean event intensity per side is:
        """
    )
    st.latex(r"\mathbb{E}[\lambda_{\text{buy}}] = \mathbb{E}[\lambda_{\text{sell}}] = \frac{\mu}{1 - \eta}")

    st.markdown("#### 2. Synthetic Reference-Price Dynamics")
    st.markdown(
        r"""
        Between events, the reference mid-price evolves under Brownian diffusion in log-space:
        """
    )
    st.latex(r"\Delta \ln S = \sigma \sqrt{\Delta t} \, Z, \quad Z \sim \mathcal{N}(0, 1)")
    st.markdown(
        r"""
        At each order arrival $t_i$, a permanent signed price impact is added:
        """
    )
    st.latex(
        r"\ln S(t_i) = \ln S(t_i^-) + \text{sign}_i \times \frac{\text{impact\_bps}}{10\,000}, \quad \text{sign}_i \in \{+1, -1\}"
    )

    st.markdown("#### 3. Quoting Policies & Fill Probabilities")
    st.markdown(
        r"""
        - **Policy A (Symmetric):** Posts bid and ask quotes at constant half-spread distance:
        $d_{\text{bid}} = d_{\text{ask}} = \delta_{\text{base}}$.
        - **Policy B (Inventory-Aware):** Applies signed skew bounded by $0.8 \times \delta_{\text{base}}$:
        """
    )
    st.latex(
        r"""
        \begin{aligned}
        \text{skew\_bps} &= \text{clip}\left(c \cdot q, \, -0.8 \delta_{\text{base}}, \, +0.8 \delta_{\text{base}}\right) \\
        d_{\text{bid}} &= \delta_{\text{base}} + \text{skew\_bps} \\
        d_{\text{ask}} &= \delta_{\text{base}} - \text{skew\_bps}
        \end{aligned}
        """
    )
    st.markdown(
        r"""
        Conditional fill probability given quote distance $d$ (in bps):
        """
    )
    st.latex(r"p_{\text{fill}} = \min\left(1.0, \, p_0 \, e^{-\kappa \, d}\right)")
    st.markdown(
        r"""
        When signed inventory reaches $+Q_{\max}$, bids are disabled ($p_{\text{fill, bid}} = 0$).
        When inventory reaches $-Q_{\max}$, asks are disabled ($p_{\text{fill, ask}} = 0$).
        """
    )

    st.markdown("#### 4. Terminal Liquidation & Conservative Accounting")
    st.markdown(
        r"""
        At session end ($t = 1.0$), any residual inventory $q_T \ne 0$ is liquidated by crossing the base half-spread:
        - If $q_T > 0$ (long): sell at $P_{\text{liq}} = S_T \exp(-\delta_{\text{base}} / 10\,000)$.
        - If $q_T < 0$ (short): buy back at $P_{\text{liq}} = S_T \exp(+\delta_{\text{base}} / 10\,000)$.
        - Both intra-session fills and terminal liquidation pay proportional transaction fees $\text{fee\_bps}$.
        - Consequently: $\text{Terminal Net P\&L} \le \text{Mark-to-Market Equity}$.
        """
    )


# =============================================================================
# TAB 6: LIMITATIONS & FALSIFICATION
# =============================================================================
with tab_limitations:
    st.markdown("### Empirical Limitations & Falsification Conditions")

    st.markdown(
        """
        <div class="warning-box">
        <b>Mandatory Research Disclosures:</b>
        This application is an educational and mathematical experimentation tool. All price and arrival paths
        are purely synthetic. It does not ingest live market feeds, model order book queues, or demonstrate
        commercial profitability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Stylized Modeling Limitations")
    st.markdown(
        """
        1. **Absence of Limit Order Book Queue Dynamics:**
           In real electronic limit order books (e.g., NASDAQ, CME), market makers occupy discrete queue priorities
           (FIFO, pro-rata). In this simulator, execution is governed by a smooth conditional probability function
           $p_0 \\exp(-\\kappa d)$ without queue position, partial fills, or cancellations.
        2. **Exogenous Impact Independence:**
           Price impact is treated as a fixed linear jump in log-price space, invariant to the dealer's quoted depth or fill.
           In reality, price impact is concave (square-root law) and depends on prevailing order book liquidity.
        3. **Finite Normalized Horizon:**
           The session horizon is normalized to $t \\in [0, 1]$. In practical intraday trading, sessions span hours with
           diurnal U-shaped volatility and intensity patterns (open/close surges) not captured by stationary parameters.
        4. **Constant Exponential Kernel:**
           Real market order arrivals often exhibit long memory and power-law kernels ($t^{-\\gamma}$) rather than
           single-exponential decay.
        """
    )

    st.markdown("#### What Would Falsify the Central Hypotheses?")
    st.markdown(
        """
        The central hypothesis of this study is that **inventory-aware quoting provides superior tail-risk protection
        relative to symmetric quoting in the presence of order clustering and adverse selection.**

        This hypothesis would be **falsified** if:
        1. **Vanishing Inventory Advantage:** In an empirical order flow dataset where price impact is negligible
           (impact_bps -> 0), the fee and spread losses from shading quotes in Policy B outweigh any liquidation savings,
           yielding a statistically significant negative mean difference: $\\mathbb{E}[\\Delta \\text{P\\&L}] < 0$.
        2. **Boundary Trapping Under Asymmetric Drift:** Under strong persistent exogenous drift, inventory skewing
           fails to prevent inventory saturation if the opposite arrival intensity drops towards zero, leading both policies
           to identical boundary utilization.
        3. **Adverse Queue Degradation:** If queue priority penalties for requoting exceed the benefit of distance shading,
           symmetric resting orders may outperform dynamic skewing in real matching engines.
        """
    )

# Footer
st.divider()
st.markdown(
    "<div style='font-size:0.75rem; color:#94a3b8; text-align:center;'>"
    "OrderFlow Lab &copy; 2026 | Quantitative Finance Research Application | Runs fully offline with deterministic random seeds."
    "</div>",
    unsafe_allow_html=True,
)
