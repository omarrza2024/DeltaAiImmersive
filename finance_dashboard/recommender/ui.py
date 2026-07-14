"""Streamlit dashboard for the stock recommender."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from finance_dashboard.dashboards import BaseDashboard
from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.models import RecommendationRequest, RiskLabel
from finance_dashboard.recommender.recommender import StockRecommender

_LABEL_COLORS = {
    RiskLabel.LOW.value: "#2E8B57",
    RiskLabel.MEDIUM.value: "#DAA520",
    RiskLabel.HIGH_POSITIVE.value: "#FF7F0E",
    RiskLabel.HIGH_NEGATIVE.value: "#D62728",
}


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

        profile = result.profile
        cols = st.columns(len(RiskLabel))
        for col, label in zip(cols, RiskLabel):
            col.metric(label.value, profile.counts[label])

        table = pd.DataFrame(
            [
                {
                    "Ticker": r.ticker,
                    "Risk": r.risk_label.value,
                    "Risk score": round(r.risk_score, 1),
                    **{k: round(v, 4) for k, v in r.features.items()},
                }
                for r in result.recommendations
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

        chart_df = table.rename(columns={"Risk score": "score"})
        fig = px.bar(
            chart_df.sort_values("score"),
            x="score",
            y="Ticker",
            color="Risk",
            color_discrete_map=_LABEL_COLORS,
            orientation="h",
            title="Risk score by stock — bucket first, volatility rank within it",
        )
        fig.update_layout(height=max(320, 28 * len(chart_df) + 120))
        st.plotly_chart(fig, use_container_width=True)

        if result.skipped:
            with st.expander(f"Skipped tickers ({len(result.skipped)})"):
                for ticker, reason in sorted(result.skipped.items()):
                    st.write(f"**{ticker}** — {reason}")

        st.caption(
            "Risk buckets are derived from historical volatility and returns. "
            "This is an educational screener, not investment advice."
        )
