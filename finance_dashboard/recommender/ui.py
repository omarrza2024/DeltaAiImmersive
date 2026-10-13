"""Streamlit dashboard for the stock recommender."""

from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import streamlit as st

from finance_dashboard.charts import PALETTE
from finance_dashboard.dashboards import BaseDashboard
from finance_dashboard.data import company_name
from finance_dashboard.models import UserInputError
from finance_dashboard.recommender.features import FeatureProvider
from finance_dashboard.recommender.models import FeatureName, RecommendationRequest, RiskLabel
from finance_dashboard.recommender.recommender import StockRecommender
from finance_dashboard.sentiment import (
    NEWS_LIMIT,
    SentimentResult,
    SentimentUnavailableError,
    fetch_sentiment,
    make_sentiment_lookup,
)


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


# How each selectable feature maps onto one or more table/chart columns.
# Keys are matrix column names (produced by FeatureProvider); each entry is
# a display label and an optional unit formatter.
@dataclass(frozen=True)
class _FeatureColumn:
    column: str           # key in StockRecommendation.features
    display: str          # human label
    unit: str             # "%", "", "ratio"
    chart_title: str      # heading for the per-feature bar chart
    multiply: float = 1.0 # scale factor applied before display (e.g. 100 for pct)

_FEATURE_DISPLAY: dict[FeatureName, tuple[_FeatureColumn, ...]] = {
    FeatureName.VOLATILITY: (
        _FeatureColumn("volatility", "Volatility %", "%",
                       "Annualized volatility", multiply=100.0),
    ),
    FeatureName.RETURNS: (
        _FeatureColumn("returns", "Return %", "%",
                       "Period return", multiply=100.0),
    ),
    FeatureName.SMA: (
        _FeatureColumn("sma_ratio", "SMA ratio", "ratio",
                       "SMA ratio (price / 20-day SMA − 1)"),
    ),
    FeatureName.MACD: (
        _FeatureColumn("macd_hist", "MACD hist", "",
                       "MACD histogram (normalized by price)"),
    ),
    FeatureName.FUNDAMENTALS: (
        _FeatureColumn("revenue_growth", "Revenue growth %", "%",
                       "Revenue growth YoY (SEC)", multiply=100.0),
        _FeatureColumn("profit_margin", "Profit margin %", "%",
                       "Net profit margin (SEC)", multiply=100.0),
    ),
    FeatureName.SENTIMENT: (
        _FeatureColumn("sentiment", "News sentiment", "",
                       "Average news sentiment (−1 to +1)"),
    ),
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

        # When sentiment is a selected feature, wire marketaux into the feature
        # provider so clustering uses REAL news-driven scores instead of the
        # neutral stub. The lookup gracefully falls back to 0.0 if marketaux
        # is unavailable, keeping clustering alive either way.
        sentiment_requested = FeatureName.SENTIMENT in request.features
        if sentiment_requested:
            provider = FeatureProvider(sentiment_lookup=make_sentiment_lookup())
            recommender = StockRecommender(feature_provider=provider)
        else:
            recommender = StockRecommender()

        with st.spinner("Fetching data and grouping stocks…"):
            result = recommender.recommend(request)

        _bucket_kpi_row(result.profile)

        selected_cols: list[_FeatureColumn] = [
            fc for feature in request.features
            for fc in _FEATURE_DISPLAY.get(feature, ())
        ]
        self._render_feature_table(result.recommendations, selected_cols)
        self._render_risk_score_chart(result.recommendations)
        self._render_per_feature_charts(result.recommendations, selected_cols)

        if result.skipped:
            with st.expander(f"Skipped tickers ({len(result.skipped)})"):
                for ticker, reason in sorted(result.skipped.items()):
                    st.write(f"**{ticker}** — {reason}")

        if sentiment_requested:
            _render_sentiment_panel([r.ticker for r in result.recommendations])

        st.caption(
            "Risk buckets are derived from historical volatility and returns. "
            "Additional columns and charts reflect the features YOU selected. "
            "This is an educational screener, not investment advice."
        )

    # ---- table + charts ---------------------------------------------------

    def _render_feature_table(self, recs, selected_cols: list[_FeatureColumn]) -> None:
        """Table with ticker/risk + one column per user-selected feature only."""
        rows = []
        for r in recs:
            row = {
                "Ticker": r.ticker,
                "Company": company_name(r.ticker) or "",
                "Risk": r.risk_label.value,
                "Risk score": round(r.risk_score, 1),
            }
            for fc in selected_cols:
                raw = r.features.get(fc.column)
                row[fc.display] = round(raw * fc.multiply, 2) if raw is not None else None
            rows.append(row)

        table = pd.DataFrame(rows)
        column_config: dict = {
            "Ticker":     st.column_config.TextColumn(width="small"),
            "Company":    st.column_config.TextColumn(width="medium"),
            "Risk":       st.column_config.TextColumn(width="small"),
            "Risk score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.0f", width="medium"
            ),
        }
        for fc in selected_cols:
            if fc.unit == "%":
                column_config[fc.display] = st.column_config.NumberColumn(format="%+.2f %%")
            else:
                column_config[fc.display] = st.column_config.NumberColumn(format="%.4f")

        st.dataframe(
            table, use_container_width=True, hide_index=True, column_config=column_config,
        )

    def _render_risk_score_chart(self, recs) -> None:
        """The headline bucket chart — always rendered, risk is the recommender's point."""
        df = pd.DataFrame(
            [{"Ticker": r.ticker, "score": round(r.risk_score, 1),
              "Risk": r.risk_label.value} for r in recs]
        )
        fig = px.bar(
            df.sort_values("score"),
            x="score", y="Ticker",
            color="Risk",
            color_discrete_map=_label_colors(),
            orientation="h",
            title="Risk score — bucket band + within-bucket volatility rank",
        )
        _style_bar(fig, len(df))
        st.plotly_chart(
            fig, use_container_width=True, config={"displayModeBar": False},
            key="recommender-chart-risk",
        )

    def _render_per_feature_charts(self, recs, selected_cols: list[_FeatureColumn]) -> None:
        """One bar chart per SELECTED feature — e.g. MACD alone when only MACD was picked."""
        for fc in selected_cols:
            rows = []
            for r in recs:
                raw = r.features.get(fc.column)
                if raw is None:
                    continue
                rows.append({
                    "Ticker": r.ticker,
                    fc.display: round(raw * fc.multiply, 4),
                    "Risk": r.risk_label.value,
                })
            if not rows:
                continue
            df = pd.DataFrame(rows)
            fig = px.bar(
                df.sort_values(fc.display),
                x=fc.display, y="Ticker",
                color="Risk",
                color_discrete_map=_label_colors(),
                orientation="h",
                title=fc.chart_title,
            )
            _style_bar(fig, len(df))
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False},
                key=f"recommender-chart-{fc.column}",
            )


def _style_bar(fig, n_rows: int) -> None:
    fig.update_layout(
        height=max(320, 28 * n_rows + 120),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E8ECEF", size=12, family="Inter, sans-serif"),
        colorway=PALETTE,
        xaxis=dict(gridcolor="#1E2027", color="#6B7280"),
        # "total ascending" sorts the categorical y-axis by the summed x value
        # per category — overrides plotly's default grouping-by-color so the
        # longest bar is always at the top regardless of risk-label color.
        yaxis=dict(gridcolor="#1E2027", color="#6B7280", categoryorder="total ascending"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


# ---- sentiment panel ----------------------------------------------------

# Chip classes for the sentiment label — reuses the existing .chip.pos/.neg
# green/red tokens, plus the neutral default .chip for anything near zero.
_SENTIMENT_CHIP_CLASS = {
    "Positive": "chip pos",
    "Negative": "chip neg",
    "Neutral":  "chip",
}


def _sentiment_card_html(result: SentimentResult) -> str:
    name = company_name(result.ticker) or ""
    chip = _SENTIMENT_CHIP_CLASS.get(result.label, "chip")
    article_label = (
        f"{result.article_count} article{'s' if result.article_count != 1 else ''}"
    )
    return (
        '<div class="rec-card">'
        '<div class="rec-header">'
        f'<span class="rec-ticker">{html.escape(result.ticker)}</span>'
        f'<span class="rec-name">{html.escape(name)}</span>'
        '</div>'
        '<div class="kpi-row">'
        '<div class="kpi">'
        '<div class="kpi-label">Avg sentiment</div>'
        f'<div class="kpi-value">{result.avg_score:+.3f}</div>'
        '</div>'
        '<div class="kpi">'
        '<div class="kpi-label">News articles</div>'
        f'<div class="kpi-value">{result.article_count}</div>'
        '</div>'
        '</div>'
        '<div class="chip-row">'
        f'<span class="{chip}">{html.escape(result.label)}</span>'
        f'<span class="chip">{html.escape(article_label)}</span>'
        '</div>'
        '</div>'
    )


def _render_article_list(result: SentimentResult) -> None:
    """Expandable list of the exact articles that contributed to the average."""
    # getattr guard: a cached SentimentResult pickled before `articles` was
    # added to the dataclass won't have the attribute until the cache cycles.
    articles = getattr(result, "articles", ()) or ()
    if not articles:
        st.caption("No articles were returned by marketaux for this ticker.")
        return
    for article in articles:
        meta_bits: list[str] = []
        if article.source:
            meta_bits.append(article.source)
        if article.published_at:
            # Marketaux returns ISO timestamps like 2024-01-02T15:30:00.000000Z.
            # Trim to the date for a cleaner line.
            meta_bits.append(article.published_at[:10])
        meta_bits.append(f"score {article.score:+.3f}")
        meta = " · ".join(meta_bits)

        title_md = (
            f"**[{article.title}]({article.url})**"
            if article.url else f"**{article.title}**"
        )
        st.markdown(f"{title_md}  \n_{meta}_")
        if article.description:
            st.caption(article.description)
        st.divider()


def _render_sentiment_panel(tickers: list[str]) -> None:
    """Show average marketaux sentiment per ticker — one call each, cached 30 min."""
    if not tickers:
        return
    st.markdown(f"## News sentiment (latest {NEWS_LIMIT} articles each)")
    st.caption(
        "Pulled live from marketaux. Score is the mean of each article's entity "
        "sentiment score for this ticker — range [-1, 1]. "
        "The free marketaux tier caps results at **3 articles per request**, so "
        "the shown count may be lower than the requested 10."
    )

    with st.spinner("Fetching latest news from marketaux…"):
        results: list[SentimentResult] = []
        errors: list[tuple[str, str]] = []
        for ticker in tickers:
            try:
                results.append(fetch_sentiment(ticker))
            except SentimentUnavailableError as exc:
                errors.append((ticker, str(exc)))

    if errors and not results:
        # Hard failure — API key missing, network down, etc. Show once, not per-ticker.
        st.error(f"News sentiment unavailable: {errors[0][1]}")
        return

    # Rank cards by avg sentiment DESC so the most positive news surfaces at
    # the top. Tickers with no articles land at the bottom (score 0.0, but
    # distinguishable by article_count == 0 — we keep them visible).
    results.sort(
        key=lambda r: (r.article_count > 0, r.avg_score),
        reverse=True,
    )

    # Two columns on wide screens; stacks on mobile automatically.
    cols = st.columns(2)
    for i, result in enumerate(results):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(_sentiment_card_html(result), unsafe_allow_html=True)
                with st.expander(f"View {result.article_count} article"
                                 f"{'s' if result.article_count != 1 else ''}"):
                    _render_article_list(result)

    if errors:
        with st.expander(f"Tickers with no sentiment data ({len(errors)})"):
            for ticker, reason in errors:
                st.write(f"**{ticker}** — {reason}")
