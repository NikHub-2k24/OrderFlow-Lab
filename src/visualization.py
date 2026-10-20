"""Plotly visualization suite with institutional research aesthetic."""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import HawkesParams, PolicyParams
from src.hawkes import PointProcessEvents, compute_continuous_intensities, EVENT_BUY, EVENT_SELL
from src.price_process import MarketPricePath
from src.simulation import PairedSessionResult
from src.experiments import MonteCarloResults


# Restrained institutional dark palette
COLOR_BG = "#111418"          # Deep charcoal main background
COLOR_PLOT_BG = "#191F26"     # Slate surface
COLOR_PANEL_BG = "#202830"    # Elevated panel surface
COLOR_GRID = "#303943"        # Subtle border / gridline
COLOR_TEXT = "#E5E9ED"        # Primary text (soft off-white)
COLOR_MUTED = "#A6B0BB"       # Secondary / muted text (light slate)

COLOR_MID = "#E5E9ED"         # Reference Mid-Price
COLOR_POLICY_A = "#8899A6"    # Policy A: Symmetric (neutral slate gray)
COLOR_POLICY_B = "#70AFA7"    # Policy B: Inventory-Aware (primary accent muted teal)
COLOR_BUY = "#69B68A"         # Dealer Bought (soft emerald green)
COLOR_SELL = "#D77979"        # Dealer Sold (soft coral red)


def apply_institutional_layout(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent institutional research typography and dark theme to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=title or "",
            font=dict(size=13, color=COLOR_TEXT, family="Inter, system-ui, sans-serif"),
            x=0.01,
            xanchor="left",
            y=0.98,
        ),
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=11, color=COLOR_TEXT),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_PLOT_BG,
        margin=dict(l=45, r=25, t=40, b=40),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color=COLOR_TEXT),
            bgcolor="rgba(25, 31, 38, 0.85)",
            bordercolor=COLOR_GRID,
            borderwidth=1,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            linecolor=COLOR_GRID,
            tickfont=dict(size=10, color=COLOR_MUTED),
            title_font=dict(size=10, color=COLOR_MUTED),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            zeroline=False,
            linecolor=COLOR_GRID,
            tickfont=dict(size=10, color=COLOR_MUTED),
            title_font=dict(size=10, color=COLOR_MUTED),
        ),
    )
    return fig


def plot_single_session_deep_dive(
    result: PairedSessionResult,
    hawkes_params: HawkesParams,
    policy_params: PolicyParams,
) -> go.Figure:
    """Create a 4-panel multi-plot visualizing prices, quotes, intensities, inventory, and equity."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "1. Reference Mid-Price, Quoted Spreads & Market-Maker Fills",
            "2. Order-Flow Arrival Intensities (lambda_buy and lambda_sell)",
            f"3. Dealer Inventory Trajectory (Hard Limit +/-{policy_params.max_inventory})",
            "4. Mark-to-Market Equity & Drawdown Evolution",
        ],
        row_heights=[0.32, 0.22, 0.22, 0.24],
    )

    t_grid = np.linspace(0.0, 1.0, 500)
    l_buy_grid, l_sell_grid = compute_continuous_intensities(result.events, hawkes_params, t_grid)

    pp = result.price_path
    ta = result.trace_a
    tb = result.trace_b

    # --- Panel 1: Prices, Quotes, Fills ---
    fig.add_trace(
        go.Scatter(
            x=pp.timestamps,
            y=pp.mid_prices,
            mode="lines",
            line=dict(color=COLOR_MID, width=1.5),
            name="Reference Mid-Price",
        ),
        row=1, col=1,
    )

    # Policy B quotes
    if len(tb.quotes) > 0 and len(pp.event_indices) > 0:
        event_times = pp.timestamps[pp.event_indices]
        bids_b = [q.bid_price for q in tb.quotes]
        asks_b = [q.ask_price for q in tb.quotes]
        fig.add_trace(
            go.Scatter(
                x=event_times,
                y=asks_b,
                mode="lines",
                line=dict(color=COLOR_POLICY_B, width=1.0, dash="dot"),
                name="Policy B Ask Quote",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=event_times,
                y=bids_b,
                mode="lines",
                line=dict(color=COLOR_POLICY_B, width=1.0, dash="dot"),
                name="Policy B Bid Quote",
            ),
            row=1, col=1,
        )

    # Fills for Policy B
    fill_mask_b = tb.record.fill_events
    if np.any(fill_mask_b):
        fill_t = tb.record.timestamps[fill_mask_b]
        fill_p = tb.record.fill_prices[fill_mask_b]
        fill_s = tb.record.fill_sides[fill_mask_b]

        buy_mask = fill_s == 1
        sell_mask = fill_s == -1

        if np.any(buy_mask):
            fig.add_trace(
                go.Scatter(
                    x=fill_t[buy_mask],
                    y=fill_p[buy_mask],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color=COLOR_BUY),
                    name="Policy B Buy Fill (Dealer Bought)",
                ),
                row=1, col=1,
            )
        if np.any(sell_mask):
            fig.add_trace(
                go.Scatter(
                    x=fill_t[sell_mask],
                    y=fill_p[sell_mask],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color=COLOR_SELL),
                    name="Policy B Sell Fill (Dealer Sold)",
                ),
                row=1, col=1,
            )

    # --- Panel 2: Intensities ---
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=l_buy_grid,
            mode="lines",
            line=dict(color=COLOR_BUY, width=1.2),
            name="lambda_buy(t)",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=l_sell_grid,
            mode="lines",
            line=dict(color=COLOR_SELL, width=1.2),
            name="lambda_sell(t)",
        ),
        row=2, col=1,
    )
    # Baseline intensity reference line
    fig.add_hline(
        y=hawkes_params.mu,
        line=dict(color=COLOR_MUTED, dash="dash", width=1.0),
        annotation_text="Base rate mu",
        annotation_position="top left",
        row=2, col=1,
    )

    # --- Panel 3: Inventory Trajectory ---
    fig.add_trace(
        go.Scatter(
            x=ta.record.timestamps,
            y=ta.record.inventory,
            mode="lines",
            line=dict(color=COLOR_POLICY_A, width=1.5),
            name="Policy A (Symmetric) Inv",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tb.record.timestamps,
            y=tb.record.inventory,
            mode="lines",
            line=dict(color=COLOR_POLICY_B, width=1.8),
            name="Policy B (Inventory-Aware) Inv",
        ),
        row=3, col=1,
    )
    # Inventory bounds
    fig.add_hline(
        y=policy_params.max_inventory,
        line=dict(color=COLOR_SELL, dash="dash", width=1.0),
        annotation_text=f"+Q_max ({policy_params.max_inventory})",
        row=3, col=1,
    )
    fig.add_hline(
        y=-policy_params.max_inventory,
        line=dict(color=COLOR_SELL, dash="dash", width=1.0),
        annotation_text=f"-Q_max (-{policy_params.max_inventory})",
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color=COLOR_GRID, width=1.0), row=3, col=1)

    # --- Panel 4: Equity & Drawdown ---
    fig.add_trace(
        go.Scatter(
            x=ta.record.timestamps,
            y=ta.record.mtm_equity,
            mode="lines",
            line=dict(color=COLOR_POLICY_A, width=1.4),
            name="Policy A Equity",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tb.record.timestamps,
            y=tb.record.mtm_equity,
            mode="lines",
            line=dict(color=COLOR_POLICY_B, width=1.8),
            name="Policy B Equity",
        ),
        row=4, col=1,
    )
    fig.add_hline(y=0, line=dict(color=COLOR_GRID, width=1.0), row=4, col=1)

    fig.update_xaxes(title_text="Normalized Session Time t in [0, 1]", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Events / Session", row=2, col=1)
    fig.update_yaxes(title_text="Inventory (units)", row=3, col=1)
    fig.update_yaxes(title_text="MTM Equity ($)", row=4, col=1)

    apply_institutional_layout(fig, height=720, show_legend=True)
    return fig


def plot_pnl_distribution_comparison(mc_results: MonteCarloResults) -> go.Figure:
    """Histogram comparison of net terminal P&L distributions for Policy A vs Policy B."""
    df = mc_results.df_sessions
    sa = mc_results.stats_a
    sb = mc_results.stats_b

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["pnl_a"],
            name="Policy A: Symmetric",
            opacity=0.55,
            marker_color=COLOR_POLICY_A,
            nbinsx=40,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=df["pnl_b"],
            name="Policy B: Inventory-Aware",
            opacity=0.60,
            marker_color=COLOR_POLICY_B,
            nbinsx=40,
        )
    )

    # Add vertical mean lines
    fig.add_vline(
        x=sa.mean_pnl,
        line=dict(color=COLOR_POLICY_A, dash="dash", width=2.0),
        annotation_text=f"A Mean: ${sa.mean_pnl:.2f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=sb.mean_pnl,
        line=dict(color=COLOR_POLICY_B, dash="dash", width=2.0),
        annotation_text=f"B Mean: ${sb.mean_pnl:.2f}",
        annotation_position="top right",
    )
    # Zero line
    fig.add_vline(x=0.0, line=dict(color=COLOR_MID, width=1.2))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Net Terminal P&L ($)",
        yaxis_title="Number of Sessions",
    )
    apply_institutional_layout(fig, title="Net Terminal P&L Distribution Across Matched Sessions", height=380)
    return fig


def plot_paired_differences(mc_results: MonteCarloResults) -> go.Figure:
    """Plot distribution of paired P&L differences (Policy B minus Policy A) with 95% bootstrap CI."""
    df = mc_results.df_sessions
    ps = mc_results.paired_stats

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["pnl_diff_b_minus_a"],
            name="P&L Difference (B - A)",
            marker_color=COLOR_POLICY_B,
            opacity=0.75,
            nbinsx=35,
        )
    )

    # Add vertical reference lines
    fig.add_vline(x=0.0, line=dict(color=COLOR_SELL, width=1.5, dash="dash"), annotation_text="Parity (0.0)", annotation_font_color=COLOR_MUTED)
    fig.add_vline(
        x=ps.mean_diff,
        line=dict(color=COLOR_TEXT, width=2.0),
        annotation_text=f"Mean Diff: ${ps.mean_diff:.2f}",
        annotation_position="top left",
        annotation_font_color=COLOR_TEXT,
    )

    # Confidence interval shaded band
    fig.add_vrect(
        x0=ps.ci_lower,
        x1=ps.ci_upper,
        fillcolor="rgba(112, 175, 167, 0.18)",
        line_width=0,
        annotation_text=f"95% Bootstrap CI: [${ps.ci_lower:.2f}, ${ps.ci_upper:.2f}]",
        annotation_position="bottom right",
        annotation_font_color=COLOR_MUTED,
    )

    fig.update_layout(
        xaxis_title="Paired Difference (Policy B Net P&L - Policy A Net P&L, $)",
        yaxis_title="Session Count",
    )
    apply_institutional_layout(
        fig,
        title=f"Paired Policy Difference per Matched Path (Win Rate B: {ps.win_rate_b:.1%}, p-val: {ps.p_value_t:.4f})",
        height=360,
    )
    return fig


def plot_tail_risk_metrics_bar(mc_results: MonteCarloResults) -> go.Figure:
    """Grouped bar chart comparing key risk and return metrics."""
    sa = mc_results.stats_a
    sb = mc_results.stats_b

    categories = [
        "Mean Net P&L ($)",
        "Median Net P&L ($)",
        "5th Pct VaR ($)",
        "Worst 5% CVaR ($)",
        "Mean Event-Sampled MDD ($)",
        "95th Pct Event-Sampled MDD ($)",
    ]

    vals_a = [
        sa.mean_pnl, sa.median_pnl, sa.pnl_5th_pct, sa.cvar_5th,
        sa.mean_event_sampled_drawdown, sa.event_sampled_drawdown_95th,
    ]
    vals_b = [
        sb.mean_pnl, sb.median_pnl, sb.pnl_5th_pct, sb.cvar_5th,
        sb.mean_event_sampled_drawdown, sb.event_sampled_drawdown_95th,
    ]

    fig = go.Figure(
        data=[
            go.Bar(name="Policy A: Symmetric", x=categories, y=vals_a, marker_color=COLOR_POLICY_A),
            go.Bar(name="Policy B: Inventory-Aware", x=categories, y=vals_b, marker_color=COLOR_POLICY_B),
        ]
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Metric Value ($)",
    )
    apply_institutional_layout(fig, title="Risk & Return Decomposition Across All Sessions", height=360)
    return fig


def plot_poisson_vs_hawkes_comparison(
    hawkes_res: MonteCarloResults,
    poisson_res: MonteCarloResults,
) -> go.Figure:
    """Compare Policy A and B performance under Hawkes clustering vs matched Poisson arrivals."""
    h_df = hawkes_res.df_sessions
    p_df = poisson_res.df_sessions

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Terminal Net P&L: Hawkes vs Poisson",
            "Maximum Absolute Inventory Utilization",
        ],
    )

    # Boxplot for P&L
    fig.add_trace(
        go.Box(y=h_df["pnl_a"], name="Hawkes (Sym)", marker_color=COLOR_POLICY_A, boxpoints=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=h_df["pnl_b"], name="Hawkes (Inv)", marker_color=COLOR_POLICY_B, boxpoints=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=p_df["pnl_a"], name="Poisson (Sym)", marker_color="#5A6876", boxpoints=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(y=p_df["pnl_b"], name="Poisson (Inv)", marker_color="#45857C", boxpoints=False),
        row=1, col=1,
    )

    # Boxplot for Max Absolute Inventory
    fig.add_trace(
        go.Box(y=h_df["max_abs_inv_a"], name="Hawkes (Sym)", marker_color=COLOR_POLICY_A, boxpoints=False, showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Box(y=h_df["max_abs_inv_b"], name="Hawkes (Inv)", marker_color=COLOR_POLICY_B, boxpoints=False, showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Box(y=p_df["max_abs_inv_a"], name="Poisson (Sym)", marker_color="#5A6876", boxpoints=False, showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Box(y=p_df["max_abs_inv_b"], name="Poisson (Inv)", marker_color="#45857C", boxpoints=False, showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Box(y=p_df["max_abs_inv_b"], name="Poisson (Inv)", marker_color="#38bdf8", boxpoints=False, showlegend=False),
        row=1, col=2,
    )

    fig.update_yaxes(title_text="Net Terminal P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Max Absolute Inventory (units)", row=1, col=2)

    apply_institutional_layout(
        fig,
        title="Arrival Process Impact: Hawkes Clustering vs Matched Poisson Baseline",
        height=400,
    )
    return fig


def plot_sensitivity_curve(df_sweep: pd.DataFrame, param_name: str) -> go.Figure:
    """Plot sensitivity of Policy A vs Policy B metrics across the parameter sweep."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Mean Net P&L and 5% CVaR vs {param_name}",
            f"Inventory Utilization & Loss Prob vs {param_name}",
        ],
    )

    x = df_sweep["param_value"]

    # Mean P&L
    fig.add_trace(
        go.Scatter(
            x=x, y=df_sweep["mean_pnl_a"],
            mode="lines+markers",
            line=dict(color=COLOR_POLICY_A, width=2.0),
            name="Policy A Mean P&L",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df_sweep["mean_pnl_b"],
            mode="lines+markers",
            line=dict(color=COLOR_POLICY_B, width=2.0),
            name="Policy B Mean P&L",
        ),
        row=1, col=1,
    )

    # 5% CVaR (Dotted)
    fig.add_trace(
        go.Scatter(
            x=x, y=df_sweep["cvar_5_a"],
            mode="lines",
            line=dict(color=COLOR_POLICY_A, dash="dot", width=1.5),
            name="Policy A 5% CVaR",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df_sweep["cvar_5_b"],
            mode="lines",
            line=dict(color=COLOR_POLICY_B, dash="dot", width=1.5),
            name="Policy B 5% CVaR",
        ),
        row=1, col=1,
    )

    # Inventory Utilization
    fig.add_trace(
        go.Scatter(
            x=x, y=df_sweep["inv_util_a"],
            mode="lines+markers",
            line=dict(color=COLOR_POLICY_A, width=2.0),
            name="Policy A Inv Util",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df_sweep["inv_util_b"],
            mode="lines+markers",
            line=dict(color=COLOR_POLICY_B, width=2.0),
            name="Policy B Inv Util",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text=param_name, row=1, col=1)
    fig.update_xaxes(title_text=param_name, row=1, col=2)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Utilization Ratio", row=1, col=2)

    apply_institutional_layout(fig, title=f"Sensitivity Analysis: Sweep over {param_name}", height=400)
    return fig
