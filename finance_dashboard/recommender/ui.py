"""Streamlit dashboard for the stock recommender."""

from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from finance_dashboard.charts import PALETTE
from finance_dashboard.dashboards import BaseDashboard
from finance_dashboard.data import company_name
from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import RecommendationRequest, RiskLabel
from finance_dashboard.recommender.recommender import StockRecommender

def _label_colors() -> dict[str, str]:
    """Chart color per risk label — Low/HighNegative are semantic and always fixed."""
    return {
        RiskLabel.LOW.value:            "#00E676",   # green (safe/gain-like)
        RiskLabel.MEDIUM.value:         "#FFB800",
        RiskLabel.HIGH_POSITIVE.value:  "#3B82F6",
        RiskLabel.HIGH_NEGATIVE.value:  "#FF3B30",   # red (loss-like)
    }

# Per-bucket theme for the KPI chips at the top.
_BUCKET_TONE = {
    RiskLabel.LOW: "low",
    RiskLabel.MEDIUM: "med",
    RiskLabel.HIGH_POSITIVE: "pos",
    RiskLabel.HIGH_NEGATIVE: "neg",
}


def _bucket_kpi_row(profile) -> None:
    """Render the four risk-bucket KPIs as themed chips."""
    tiles = []
    for label in RiskLabel:
        tone = _BUCKET_TONE[label]
        count = profile.counts[label]
        tiles.append(
            f'<div class="kpi {tone}">'
            f'<div class="kpi-label">{html.escape(label.value)}</div>'
            f'<div class="kpi-value">{count}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="kpi-row">{"".join(tiles)}</div>', unsafe_allow_html=True)


class RecommenderDashboard(BaseDashboard):
    """Risk-bucketed stock grouping driven by a RecommendationRequest."""

    name = "Stock recommender"

    def validate(self, tickers: list[str], period_label: str, **kwargs) -> str:
        request = kwargs.get("request")
        if not isinstance(request, RecommendationRequest):
            raise UserInputError("No recommendation request provided.")
        return request.period

    def display(self, tickers: list[str], period_label: str, **kwargs) -> None:
        self.validate(tickers, period_label, **kwargs)
        request: RecommendationRequest = kwargs["request"]

        with st.spinner("Fetching data and grouping stocks…"):
            result = StockRecommender().recommend(request)

        _bucket_kpi_row(result.profile)

        rows = []
        for r in result.recommendations:
            rows.append(
                {
                    "Ticker": r.ticker,
                    "Company": company_name(r.ticker) or "",
                    "Risk": r.risk_label.value,
                    "Risk score": round(r.risk_score, 1),
                    "Return %":     round(r.features.get("returns", 0) * 100, 2),
                    "Volatility %": round(r.features.get("volatility", 0) * 100, 2),
                }
            )
        table = pd.DataFrame(rows)

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker":       st.column_config.TextColumn(width="small"),
                "Company":      st.column_config.TextColumn(width="medium"),
                "Risk":         st.column_config.TextColumn(width="small"),
                "Risk score":   st.column_config.ProgressColumn(
                    "Risk score", min_value=0, max_value=100, format="%.0f", width="medium"
                ),
                "Return %":     st.column_config.NumberColumn(format="%+.2f %%"),
                "Volatility %": st.column_config.NumberColumn(format="%.2f %%"),
            },
        )

        chart_df = table.rename(columns={"Risk score": "score"})
        fig = px.bar(
            chart_df.sort_values("score"),
            x="score", y="Ticker",
            color="Risk",
            color_discrete_map=_label_colors(),
            orientation="h",
            title="Risk score — bucket band + within-bucket volatility rank",
        )
        fig.update_layout(
            height=max(320, 28 * len(chart_df) + 120),
            template="plotly_dark",
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(color="#E8ECEF", size=12, family="Inter, sans-serif"),
            colorway=PALETTE,
            xaxis=dict(gridcolor="#1E2027", color="#6B7280"),
            yaxis=dict(gridcolor="#1E2027", color="#6B7280"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if result.skipped:
            with st.expander(f"Skipped tickers ({len(result.skipped)})"):
                for ticker, reason in sorted(result.skipped.items()):
                    st.write(f"**{ticker}** — {reason}")

        st.caption(
            "Risk buckets are derived from historical volatility and returns. "
            "This is an educational screener, not investment advice."
        )
