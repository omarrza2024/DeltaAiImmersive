"""Routes user selections to the appropriate dashboard."""

from __future__ import annotations

import streamlit as st

from finance_dashboard.dashboards import (
    BaseDashboard,
    CompareTwoDashboard,
    CorrelationDashboard,
    PriceMetricsDashboard,
)
from finance_dashboard.models import DashboardName, UserInputError
from finance_dashboard.recommender.ui import RecommenderDashboard


class VisualizationController:
    """
    Entry point for the UI layer.

    Maps a dashboard name + user inputs to the correct dashboard class and
    handles validation errors with user-visible messages.
    """

    _DASHBOARDS: dict[DashboardName, BaseDashboard] = {
        DashboardName.PRICE_METRICS: PriceMetricsDashboard(),
        DashboardName.COMPARE_TWO: CompareTwoDashboard(),
        DashboardName.CORRELATION: CorrelationDashboard(),
        DashboardName.RECOMMENDER: RecommenderDashboard(),
    }

    def show_appropriate_dashboard(
        self,
        dashboard: DashboardName | str,
        *,
        tickers: list[str],
        period_label: str,
        **kwargs,
    ) -> None:
        """
        Validate inputs and render the selected dashboard.

        Raises UserInputError for invalid input (also shown via st.error when caught by caller).
        """
        if isinstance(dashboard, str):
            try:
                dashboard = DashboardName(dashboard)
            except ValueError as exc:
                raise UserInputError(f"Unknown dashboard: {dashboard}") from exc

        impl = self._DASHBOARDS.get(dashboard)
        if impl is None:
            raise UserInputError(f"Unknown dashboard: {dashboard}")

        try:
            impl.display(tickers, period_label, **kwargs)
        except UserInputError:
            raise
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            raise
