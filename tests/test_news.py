"""Unit tests for the news node (Yahoo Finance news + sentiment tagging).

Tests are offline: the ``yfinance`` module is faked via monkeypatching so no
network calls are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

import graph.nodes.news as nd
from graph import PortfolioItem, AgentState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeTicker:
    """Fake ``yfinance.Ticker`` exposing a ``.news`` property."""

    def __init__(self, news: list[dict[str, Any]] | None = None):
        self._news = news if news is not None else []

    @property
    def news(self) -> list[dict[str, Any]]:
        return self._news


class FakeYF:
    """Fake ``yfinance`` module exposing ``.Ticker()``."""

    def __init__(self, tickers: dict[str, FakeTicker]):
        self._tickers = tickers

    def Ticker(self, symbol: str) -> FakeTicker:
        return self._tickers[symbol]


def make_article(title: str, **overrides: Any) -> dict[str, Any]:
    article = {
        "title": title,
        "publisher": "Yahoo Finance",
        "link": "https://finance.yahoo.com/news/example",
        "providerPublishTime": 1_700_000_000,
    }
    article.update(overrides)
    return article


def make_state(tickers: list[str] | None = None) -> AgentState:
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    return AgentState(portfolio=[PortfolioItem(ticker=t, shares=1) for t in tickers])


# ---------------------------------------------------------------------------
# Sentiment & keyword tagging
# ---------------------------------------------------------------------------
def test_classify_sentiment_positive():
    assert nd._classify_sentiment("Apple beats earnings estimates") == "positive"


def test_classify_sentiment_negative():
    assert nd._classify_sentiment("Apple misses earnings estimates") == "negative"


def test_classify_sentiment_neutral():
    assert nd._classify_sentiment("Apple announces a new product") == "neutral"


def test_classify_sentiment_positive_outweighs_negative():
    assert nd._classify_sentiment("Apple beats estimates and surges but faces a lawsuit") == "positive"


def test_extract_keywords_word_boundary():
    # "ai" should match a standalone word but not a substring of "said".
    keywords = nd._extract_keywords("Apple's AI chip drives growth")
    assert "ai" in keywords
    assert "chip" in keywords
    assert "ai" not in nd._extract_keywords("The analyst said the deal is done")


def test_extract_keywords_multiword_phrase():
    assert "artificial intelligence" in nd._extract_keywords("Apple pushes artificial intelligence")


# ---------------------------------------------------------------------------
# get_yahoo_finance_news
# ---------------------------------------------------------------------------
def test_get_yahoo_finance_news_normalizes(monkeypatch):
    fake = FakeYF(
        {
            "AAPL": FakeTicker(
                news=[make_article("Apple beats earnings expectations")]
            )
        }
    )
    monkeypatch.setattr(nd, "_get_yf", lambda: fake)

    articles = nd.get_yahoo_finance_news("AAPL")
    assert len(articles) == 1
    article = articles[0]
    assert article["ticker"] == "AAPL"
    assert article["title"] == "Apple beats earnings expectations"
    assert article["publisher"] == "Yahoo Finance"
    assert article["link"] == "https://finance.yahoo.com/news/example"
    assert article["published_at"] == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc).isoformat()
    assert article["sentiment"] == "positive"
    assert "earnings" in article["keywords"]


def test_get_yahoo_finance_news_respects_limit(monkeypatch):
    fake = FakeYF(
        {"AAPL": FakeTicker(news=[make_article(f"Title {i}") for i in range(5)])}
    )
    monkeypatch.setattr(nd, "_get_yf", lambda: fake)
    assert len(nd.get_yahoo_finance_news("AAPL", limit=2)) == 2


def test_get_yahoo_finance_news_empty_raises(monkeypatch):
    monkeypatch.setattr(nd, "_get_yf", lambda: FakeYF({"BAD": FakeTicker(news=[])}))
    with pytest.raises(ValueError):
        nd.get_yahoo_finance_news("BAD")


def test_get_yahoo_finance_news_network_error_propagates(monkeypatch):
    class ExplodingYF:
        def Ticker(self, symbol: str) -> None:
            raise ConnectionError("network down")

    monkeypatch.setattr(nd, "_get_yf", lambda: ExplodingYF())
    with pytest.raises(ConnectionError):
        nd.get_yahoo_finance_news("AAPL")


# ---------------------------------------------------------------------------
# news_node
# ---------------------------------------------------------------------------
def test_news_node_aggregates_and_updates_state(monkeypatch):
    monkeypatch.setattr(
        nd,
        "get_yahoo_finance_news",
        lambda t, limit=10: [
            {
                "ticker": t.upper(),
                "title": "Great news",
                "publisher": "YF",
                "link": "x",
                "published_at": None,
                "sentiment": "positive",
                "keywords": [],
            }
        ],
    )

    out = nd.news_node(make_state())
    results = out["news_results"]

    assert len(results) == 3
    assert {r["ticker"] for r in results} == {"AAPL", "MSFT", "GOOGL"}

    aapl = next(r for r in results if r["ticker"] == "AAPL")
    assert aapl["status"] == "ok"
    assert aapl["is_mock"] is False
    assert aapl["source"] == "yahoo_finance_mcp"
    assert aapl["articles"][0]["sentiment"] == "positive"

    assert "AAPL" in out["final_response"]


def test_news_node_full_fallback_on_error(monkeypatch):
    def boom(ticker: str, limit: int = 10) -> list:
        raise RuntimeError("rate limit exceeded")

    monkeypatch.setattr(nd, "get_yahoo_finance_news", boom)

    out = nd.news_node(make_state(["AAPL"]))
    entry = out["news_results"][0]

    assert entry["status"] == "error"
    assert entry["is_mock"] is True
    assert entry["error"]  # non-empty error message
    assert entry["articles"][0]["is_mock"] is True


def test_news_node_invalid_ticker_falls_back_but_others_ok(monkeypatch):
    def news_for(ticker: str, limit: int = 10) -> list:
        if ticker.upper() == "BOGUS":
            raise ValueError("invalid ticker")
        return [
            {
                "ticker": ticker.upper(),
                "title": "ok",
                "publisher": "YF",
                "link": "x",
                "published_at": None,
                "sentiment": "neutral",
                "keywords": [],
            }
        ]

    monkeypatch.setattr(nd, "get_yahoo_finance_news", news_for)

    out = nd.news_node(make_state(["AAPL", "BOGUS"]))
    results = {r["ticker"]: r for r in out["news_results"]}

    assert results["AAPL"]["status"] == "ok"
    assert results["BOGUS"]["status"] == "error"
    assert results["BOGUS"]["is_mock"] is True


def test_news_node_empty_portfolio():
    out = nd.news_node(make_state([]))
    assert out["news_results"] == []
    assert "0 ticker" in out["final_response"]
