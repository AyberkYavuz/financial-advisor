"""Unit tests for the optimizer node (portfolio valuation + constraint checks).

Tests are offline: no network or LLM calls are made; the optimizer is driven
directly with in-memory market data and the standard test payload.
"""

from __future__ import annotations

import pytest

from graph import (
    AgentState,
    PortfolioConstraints,
    PortfolioItem,
    TEST_CONSTRAINTS,
    TEST_PORTFOLIO,
)
from graph.nodes.optimizer import optimize_portfolio, optimizer_node


def _info(ticker, price, dividend_yield, sector="technology"):
    return {
        "ticker": ticker,
        "current_price": price,
        "dividend_yield": dividend_yield,
        "sector": sector,
    }


def _market_data(infos):
    return {
        "source": "yfinance",
        "tickers": list(infos),
        "data": {
            t: {"ticker": t, "status": "ok", "is_mock": False, "info": info}
            for t, info in infos.items()
        },
    }


#: Prices: AAPL 220, MSFT 400, GOOGL 160  => total value 46,000.
SAMPLE = _market_data(
    {
        "AAPL": _info("AAPL", 220.0, 0.005),  # 0.5% yield
        "MSFT": _info("MSFT", 400.0, 0.007),  # 0.7% yield
        "GOOGL": _info("GOOGL", 160.0, 0.0),  # 0.0% yield
    }
)

CONSTRAINTS = PortfolioConstraints(
    max_risk="medium", min_dividend_yield=2.0, sector="technology"
)


def test_total_valuation_and_weights():
    result = optimize_portfolio(TEST_PORTFOLIO, CONSTRAINTS, SAMPLE)
    assert result["total_value"] == pytest.approx(46000.0)
    weights = {h["ticker"]: h["weight_pct"] for h in result["holdings"]}
    assert weights["AAPL"] == pytest.approx(22000 / 46000 * 100, rel=1e-3)
    assert weights["MSFT"] == pytest.approx(20000 / 46000 * 100, rel=1e-3)
    assert weights["GOOGL"] == pytest.approx(4000 / 46000 * 100, rel=1e-3)


def test_weighted_dividend_yield_and_violation():
    result = optimize_portfolio(TEST_PORTFOLIO, CONSTRAINTS, SAMPLE)
    expected = (22000 * 0.5 + 20000 * 0.7 + 4000 * 0.0) / 46000
    assert result["metrics"]["weighted_dividend_yield_pct"] == pytest.approx(
        expected, rel=1e-3
    )
    assert result["compliant"] is False
    assert any("dividend yield" in v for v in result["violations"])
    assert any("reallocate" in r for r in result["recommendations"])


def test_sector_misalignment_triggers_violation():
    md = _market_data(
        {
            "AAPL": _info("AAPL", 220.0, 0.005),
            "MSFT": _info("MSFT", 400.0, 0.007, sector="Healthcare"),
            "GOOGL": _info("GOOGL", 160.0, 0.0),
        }
    )
    result = optimize_portfolio(TEST_PORTFOLIO, CONSTRAINTS, md)
    assert result["metrics"]["sector_mismatch_tickers"] == ["MSFT"]
    assert result["metrics"]["sector_match_count"] == 2
    assert any("sector" in v for v in result["violations"])
    assert any("MSFT" in r for r in result["recommendations"])


def test_sector_matching_is_case_insensitive():
    md = _market_data(
        {
            "AAPL": _info("AAPL", 220.0, 0.005, sector="Technology"),
            "MSFT": _info("MSFT", 400.0, 0.007, sector="Technology"),
            "GOOGL": _info("GOOGL", 160.0, 0.0, sector="Technology"),
        }
    )
    result = optimize_portfolio(TEST_PORTFOLIO, CONSTRAINTS, md)
    assert result["metrics"]["sector_mismatch_tickers"] == []


def test_compliant_portfolio_has_no_violations():
    md = _market_data(
        {
            "AAPL": _info("AAPL", 220.0, 0.03),  # 3%
            "MSFT": _info("MSFT", 400.0, 0.02),  # 2%
            "GOOGL": _info("GOOGL", 160.0, 0.02),  # 2%
        }
    )
    result = optimize_portfolio(TEST_PORTFOLIO, CONSTRAINTS, md)
    assert result["compliant"] is True
    assert result["violations"] == []
    assert any("no rebalancing" in r for r in result["recommendations"])


def test_optimizer_node_stores_result_in_state():
    state = AgentState(
        query="Optimize this portfolio",
        portfolio=TEST_PORTFOLIO,
        constraints=TEST_CONSTRAINTS,
        market_data=SAMPLE,
    )
    out = optimizer_node(state)
    assert "optimization_result" in out
    result = out["optimization_result"]
    assert result["total_value"] == pytest.approx(46000.0)
    assert result["compliant"] is False  # 0.54% < 2.0%


def test_fallback_to_test_payload_when_empty():
    state = AgentState(query="Optimize this portfolio")  # no holdings/constraints/market data
    out = optimizer_node(state)
    result = out["optimization_result"]
    assert [h["ticker"] for h in result["holdings"]] == ["AAPL", "MSFT", "GOOGL"]
    assert result["constraints"]["sector"] == "technology"
    assert result["constraints"]["min_dividend_yield"] == 2.0
    assert result["total_value"] == pytest.approx(46000.0)
    assert all(h["is_mock"] for h in result["holdings"])
    assert result["compliant"] is False


def test_dividend_yield_fraction_converted_to_percent():
    md = _market_data({"AAPL": _info("AAPL", 100.0, 0.005)})
    result = optimize_portfolio(
        [PortfolioItem(ticker="AAPL", shares=10)], CONSTRAINTS, md
    )
    assert result["holdings"][0]["dividend_yield_pct"] == pytest.approx(0.5)
