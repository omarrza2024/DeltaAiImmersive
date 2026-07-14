"""Tests for dashboard routing and input validation."""

import pytest

from finance_dashboard.controller import VisualizationController
from finance_dashboard.models import DashboardName, UserInputError


def test_controller_rejects_unknown_dashboard():
    controller = VisualizationController()
    with pytest.raises(UserInputError, match="Unknown dashboard"):
        controller.show_appropriate_dashboard("not-a-dashboard", tickers=["AAPL"], period_label="1 month")


def test_controller_price_dashboard_requires_ticker():
    controller = VisualizationController()
    with pytest.raises(UserInputError, match="Enter a ticker"):
        controller.show_appropriate_dashboard(
            DashboardName.PRICE_METRICS,
            tickers=[""],
            period_label="1 month",
        )


def test_controller_compare_requires_two_distinct_tickers():
    controller = VisualizationController()
    with pytest.raises(UserInputError, match="both tickers"):
        controller.show_appropriate_dashboard(
            DashboardName.COMPARE_TWO,
            tickers=["AAPL"],
            period_label="1 year",
        )
    with pytest.raises(UserInputError, match="different tickers"):
        controller.show_appropriate_dashboard(
            DashboardName.COMPARE_TWO,
            tickers=["AAPL", "AAPL"],
            period_label="1 year",
        )


def test_controller_recommender_requires_request():
    controller = VisualizationController()
    with pytest.raises(UserInputError, match="request"):
        controller.show_appropriate_dashboard(
            DashboardName.RECOMMENDER,
            tickers=[],
            period_label="1 year",
        )


def test_controller_correlation_requires_at_least_two_tickers():
    controller = VisualizationController()
    with pytest.raises(UserInputError, match="at least two"):
        controller.show_appropriate_dashboard(
            DashboardName.CORRELATION,
            tickers=["AAPL"],
            period_label="1 year",
        )
