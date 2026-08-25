"""Plotly figure builders — themed to match the Signal Studio palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Shared plotly config for st.plotly_chart — hides the camera/zoom/pan toolbar.
CHART_CONFIG = {"displayModeBar": False}
GRID_COLOR = "#1E2027"
AXIS_COLOR = "#6B7280"
BG_COLOR = "#000000"
FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

# Hardcoded semantic colors — gain/loss NEVER depend on user picks so
# green-means-up / red-means-down stays consistent everywhere.
GAIN_COLOR = "#00E676"
LOSS_COLOR = "#FF3B30"

# Fallback tints for the 3rd+ series in multi-line charts; the first two
# slots come from the user's Chart-color pickers (comparison colors).
_EXTRA_PALETTE = ["#FFB800", "#22D3EE", "#8B5CF6", "#EC4899"]


def _comparison_colors() -> tuple[str, str]:
    """User-picked Color 1 / Color 2 for multi-series comparison charts."""
    import streamlit as st
    return (
        st.session_state.get("gain_color", "#3B82F6"),
        st.session_state.get("loss_color", "#8B5CF6"),
    )


def palette() -> list[str]:
    """Chart colorway for multi-series comparisons: Color 1, Color 2, then extras."""
    c1, c2 = _comparison_colors()
    return [c1, c2, *_EXTRA_PALETTE]


# For backwards compatibility with anything importing PALETTE.
PALETTE = ["#3B82F6", "#8B5CF6", *_EXTRA_PALETTE]


def _layout() -> dict:
    """Fresh layout dict per render so the colorway reflects the latest picks."""
    return dict(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=48, r=24, t=52, b=48),
        font=dict(color="#E8ECEF", size=12, family=FONT_FAMILY),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=AXIS_COLOR,
                   showspikes=True, spikecolor=AXIS_COLOR, spikethickness=1, spikedash="dot"),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=AXIS_COLOR),
        colorway=palette(),
    )
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)


def _line_figure(
    data: pd.DataFrame,
    title: str,
    yaxis_title: str,
    y_scale: float = 1.0,
    y_format: str = ",.2f",
    height: int = 480,
    direction_colored: bool = True,
) -> go.Figure:
    colors = palette()
    is_single = len(data.columns) == 1

    fig = go.Figure()
    for i, col in enumerate(data.columns):
        series = data[col].dropna()
        # For a single-line stock chart, color by up/down direction so the line
        # itself signals "went up = green, went down = red" — these ALWAYS use
        # the fixed gain/loss colors regardless of the user's picker choices.
        if is_single and direction_colored and len(series) >= 2:
            color = GAIN_COLOR if series.iloc[-1] >= series.iloc[0] else LOSS_COLOR
        else:
            color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col] * y_scale,
                mode="lines",
                name=col,
                line=dict(color=color, width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:{y_format}}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=height,
        legend=_LEGEND,
        **_layout(),
    )
    return fig


def fig_price_history(closes: pd.DataFrame, title: str) -> go.Figure:
    return _line_figure(closes, title, "Price (adj. close)")


def fig_pct_change(cumulative_pct: pd.DataFrame, title: str) -> go.Figure:
    return _line_figure(
        cumulative_pct,
        title,
        "Cumulative % change",
        y_scale=1.0,
        y_format="+.2f",
    )


def fig_rolling_vol(rolling: pd.DataFrame, title: str) -> go.Figure:
    from finance_dashboard.config import ROLLING_WINDOW

    return _line_figure(
        rolling,
        title,
        f"Annualized vol. (%, {ROLLING_WINDOW}-day rolling)",
        y_scale=100.0,
        y_format=".2f",
        height=420,
    )


def fig_volatility_bar(tickers: list[str], vols_pct: list[float], title: str) -> go.Figure:
    colors = palette()
    fig = go.Figure(
        data=go.Bar(
            x=tickers,
            y=vols_pct,
            marker_color=[colors[i % len(colors)] for i in range(len(tickers))],
            text=[f"{v:.2f}%" if pd.notna(v) else "—" for v in vols_pct],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="Volatility (%)",
        height=360,
        showlegend=False,
        **_layout(),
    )
    return fig


def fig_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            # Diverging scale — always red at −1, green at +1 (semantic).
            colorscale=[
                [0.0, LOSS_COLOR],
                [0.5, "#000000"],   # near-zero blends into terminal bg
                [1.0, GAIN_COLOR],
            ],
            zmid=0,
            colorbar=dict(title="ρ", tickcolor=AXIS_COLOR, tickfont=dict(color=AXIS_COLOR)),
            hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Return correlation matrix",
        height=max(360, 48 * len(corr)),
        margin=dict(l=80, r=24, t=56, b=80),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#E8ECEF", size=12, family=FONT_FAMILY),
    )
    return fig
