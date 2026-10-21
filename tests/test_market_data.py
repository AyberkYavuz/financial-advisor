"""Unit tests for the market data node (Yahoo Finance / yfinance).

Tests are offline: the ``yfinance`` module is faked via monkeypatching so no
network calls are made.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import graph.nodes.market_data as md
from graph import TEST_PORTFOLIO, AgentState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeTicker:
    """Fake ``yfinance.Ticker``."""

    def __init__(self, info: dict[str, Any] | None = None, history: Any = None):
        self._info = info if info is not None else {}
        self._history = history

    @property
    def info(self) -> dict[str, Any]:
        return self._info

    def history(self, period: str | None = None, interval: str | None = None) -> Any:
        if self._history is None:
            return pd.DataFrame()
        return self._history


class FakeYF:
    """Fake ``yfinance`` module exposing ``.Ticker()``."""

    def __init__(self, tickers: dict[str, FakeTicker]):
        self._tickers = tickers

    def Ticker(self, symbol: str) -> FakeTicker:
        return self._tickers[symbol]


def make_history_df() -> pd.DataFrame:
    """Return a small realistic OHLCV DataFrame."""
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.5, 102.5],
            "Volume": [1_000_000, 1_200_000],
        },
        index=index,
    )


def make_state(tickers: list[str] | None = None) -> AgentState:
    from graph import PortfolioItem

    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    return AgentState(portfolio=[PortfolioItem(ticker=t, shares=1) for t in tickers])


# ---------------------------------------------------------------------------
# get_stock_info
# ---------------------------------------------------------------------------
def test_get_stock_info_normalizes_fields(monkeypatch):
    fake = FakeYF(
        {
            "AAPL": FakeTicker(
                info={
                    "currentPrice": 220.5,
                    "currency": "USD",
                    "marketCap": 3_000_000_000_000,
                    "dividendYield": 0.5,  # yfinance reports percentage points (0.5 = 0.5%)
                    "sector": "Technology",
                    "beta": 1.2,
                    "longName": "Apple Inc.",
                    "ignoredKey": "should not appear",
                }
            )
        }
    )
    monkeypatch.setattr(md, "_get_yf", lambda: fake)

    info = md.get_stock_info("AAPL")
    assert info["ticker"] == "AAPL"
    assert info["current_price"] == 220.5
    assert info["currency"] == "USD"
    assert info["market_cap"] == 3_000_000_000_000
    assert info["dividend_yield"] == 0.005  # normalized to fraction (0.5% -> 0.005)
    assert info["sector"] == "Technology"
    assert info["beta"] == 1.2
    assert info["long_name"] == "Apple Inc."
    assert "ignoredKey" not in info


def test_get_stock_info_empty_info_raises(monkeypatch):
    monkeypatch.setattr(md, "_get_yf", lambda: FakeYF({"BAD": FakeTicker(info={})}))
    with pytest.raises(ValueError):
        md.get_stock_info("BAD")


def test_get_stock_info_network_error_propagates(monkeypatch):
    class ExplodingYF:
        def Ticker(self, symbol: str) -> None:
            raise ConnectionError("network down")

    monkeypatch.setattr(md, "_get_yf", lambda: ExplodingYF())
    with pytest.raises(ConnectionError):
        md.get_stock_info("AAPL")


# ---------------------------------------------------------------------------
# get_historical_stock_prices
# ---------------------------------------------------------------------------
def test_get_historical_stock_prices_normalizes(monkeypatch):
    monkeypatch.setattr(md, "_get_yf", lambda: FakeYF({"AAPL": FakeTicker(history=make_history_df())}))

    rows = md.get_historical_stock_prices("AAPL", period="5d", interval="1d")
    assert len(rows) == 2
    assert rows[0]["date"] == "2024-01-02"
    assert rows[0]["open"] == 100.0
    assert rows[0]["high"] == 102.0
    assert rows[0]["low"] == 99.0
    assert rows[0]["close"] == 101.5
    assert rows[0]["volume"] == 1_000_000
    assert rows[1]["date"] == "2024-01-03"


def test_get_historical_stock_prices_empty_raises(monkeypatch):
    monkeypatch.setattr(md, "_get_yf", lambda: FakeYF({"BAD": FakeTicker(history=pd.DataFrame())}))
    with pytest.raises(ValueError):
        md.get_historical_stock_prices("BAD")


# ---------------------------------------------------------------------------
# market_data_node
# ---------------------------------------------------------------------------
def test_market_data_node_aggregates_and_updates_state(monkeypatch):
    monkeypatch.setattr(
        md,
        "get_stock_info",
        lambda t: {"ticker": t, "current_price": 100.0, "sector": "technology"},
    )
    monkeypatch.setattr(
        md,
        "get_historical_stock_prices",
        lambda t, period="1mo", interval="1d": [{"date": "2024-01-02", "close": 99.0}],
    )

    out = md.market_data_node(make_state())
    result = out["market_data"]

    assert result["source"] == "yfinance"
    assert "fetched_at" in result
    assert set(result["tickers"]) == {"AAPL", "MSFT", "GOOGL"}
    assert set(result["data"].keys()) == {"AAPL", "MSFT", "GOOGL"}

    aapl = result["data"]["AAPL"]
    assert aapl["status"] == "ok"
    assert aapl["is_mock"] is False
    assert aapl["info"]["current_price"] == 100.0
    assert aapl["historical"][0]["close"] == 99.0

    assert "AAPL" in out["final_response"]


def test_market_data_node_full_fallback_on_error(monkeypatch):
    def boom_info(ticker: str) -> dict[str, Any]:
        raise RuntimeError("network down")

    def boom_history(ticker: str, period: str = "1mo", interval: str = "1d") -> list:
        raise RuntimeError("network down")

    monkeypatch.setattr(md, "get_stock_info", boom_info)
    monkeypatch.setattr(md, "get_historical_stock_prices", boom_history)

    out = md.market_data_node(make_state())
    aapl = out["market_data"]["data"]["AAPL"]

    assert aapl["status"] == "error"
    assert aapl["is_mock"] is True
    assert aapl["error"]  # non-empty error message
    assert aapl["info"]["is_mock"] is True
    assert aapl["historical"][0]["is_mock"] is True


def test_market_data_node_partial_fallback(monkeypatch):
    monkeypatch.setattr(
        md,
        "get_stock_info",
        lambda t: {"ticker": t, "current_price": 123.0},
    )

    def boom_history(ticker: str, period: str = "1mo", interval: str = "1d") -> list:
        raise RuntimeError("history down")

    monkeypatch.setattr(md, "get_historical_stock_prices", boom_history)

    out = md.market_data_node(make_state(["AAPL"]))
    aapl = out["market_data"]["data"]["AAPL"]

    assert aapl["status"] == "partial"
    assert aapl["is_mock"] is False  # info was real
    assert aapl["info"]["current_price"] == 123.0
    assert aapl["historical"][0]["is_mock"] is True


def test_market_data_node_invalid_ticker_falls_back_but_others_ok(monkeypatch):
    def info_for(ticker: str) -> dict[str, Any]:
        if ticker.upper() == "BOGUS":
            raise ValueError("invalid ticker")
        return {"ticker": ticker, "current_price": 10.0}

    monkeypatch.setattr(md, "get_stock_info", info_for)
    monkeypatch.setattr(md, "get_historical_stock_prices", lambda t, period="1mo", interval="1d": [])

    out = md.market_data_node(make_state(["AAPL", "BOGUS"]))
    data = out["market_data"]["data"]

    assert data["AAPL"]["status"] == "ok"
    assert data["BOGUS"]["status"] == "error"
    assert data["BOGUS"]["is_mock"] is True


def test_market_data_node_empty_portfolio():
    out = md.market_data_node(make_state([]))
    result = out["market_data"]
    assert result["tickers"] == []
    assert result["data"] == {}
    assert "0 ticker" in out["final_response"]
