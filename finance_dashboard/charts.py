"""Plotly figure builders — themed to match the Signal Studio palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Signal Studio palette: teal-first, with warm accents for negatives/losses.
PALETTE = ["#14B8A6", "#8B5CF6", "#F59E0B", "#EC4899", "#22D3EE", "#F87171"]
GRID_COLOR = "#1E2A36"
AXIS_COLOR = "#8A9BA8"

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#131A22",
    plot_bgcolor="#131A22",
    margin=dict(l=48, r=24, t=52, b=48),
    font=dict(color="#E5EDF6", size=13),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=AXIS_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=AXIS_COLOR),
    colorway=PALETTE,
)
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)


def _line_figure(
    data: pd.DataFrame,
    title: str,
    yaxis_title: str,
    y_scale: float = 1.0,
    y_format: str = ",.2f",
    height: int = 480,
) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(data.columns):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col] * y_scale,
                mode="lines",
                name=col,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
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
        **_LAYOUT,
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
    fig = go.Figure(
        data=go.Bar(
            x=tickers,
            y=vols_pct,
            marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(tickers))],
            text=[f"{v:.2f}%" if pd.notna(v) else "—" for v in vols_pct],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="Volatility (%)",
        height=360,
        showlegend=False,
        **_LAYOUT,
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
            # Custom teal-to-red diverging scale matching the app palette.
            colorscale=[
                [0.0, "#F87171"],   # strong negative
                [0.5, "#131A22"],   # near-zero blends with card bg
                [1.0, "#14B8A6"],   # strong positive
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
        paper_bgcolor="#131A22",
        plot_bgcolor="#131A22",
        font=dict(color="#E5EDF6", size=13),
    )
    return fig
