"""News node — latest news + sentiment tagging via Yahoo Finance.

This node fetches the latest news articles for every portfolio ticker using
``get_yahoo_finance_news`` (the Yahoo Finance MCP tool documented in
``yahoo_mcp_reference.md``, backed by ``yfinance``'s ``Ticker.news``). For each
article it extracts the title, publisher, link, and publication timestamp, and
applies lightweight keyword-based sentiment tagging. Results are stored under
the ``news_results`` key of :class:`~graph.state.AgentState`.

Every fetch is wrapped in per-ticker exception handling so that network errors,
rate limits, or missing news degrade to deterministic mock payloads rather than
crashing the graph.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)

#: Maximum number of news articles fetched per ticker.
DEFAULT_NEWS_LIMIT = 10

#: Positive-sentiment lexicon (lowercase).
POSITIVE_KEYWORDS: tuple[str, ...] = (
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "outperform",
    "growth",
    "grows",
    "surge",
    "surges",
    "rally",
    "rallies",
    "record",
    "profit",
    "profits",
    "gain",
    "gains",
    "strong",
    "bullish",
    "jump",
    "jumps",
    "rise",
    "rises",
    "soar",
    "soars",
    "win",
    "wins",
    "boost",
    "boosted",
    "success",
)

#: Negative-sentiment lexicon (lowercase).
NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "underperform",
    "decline",
    "declines",
    "drop",
    "drops",
    "fall",
    "falls",
    "loss",
    "losses",
    "lawsuit",
    "fine",
    "probe",
    "investigation",
    "bearish",
    "weak",
    "slump",
    "slumps",
    "plunge",
    "plunges",
    "cut",
    "cuts",
    "warning",
    "risk",
    "recall",
    "layoff",
    "layoffs",
    "scandal",
    "fraud",
)

#: Finance/topic lexicon used for lightweight keyword tagging (lowercase).
TOPIC_KEYWORDS: tuple[str, ...] = (
    "earnings",
    "revenue",
    "guidance",
    "forecast",
    "dividend",
    "buyback",
    "acquisition",
    "merger",
    "ipo",
    "ai",
    "artificial intelligence",
    "chip",
    "semiconductor",
    "iphone",
    "cloud",
    "azure",
    "lawsuit",
    "regulation",
    "antitrust",
    "layoffs",
    "hiring",
    "product",
    "launch",
    "partnership",
    "investment",
    "stock split",
    "upgrade",
    "downgrade",
)


def _get_yf() -> Any:
    """Return the ``yfinance`` module, imported lazily."""
    import yfinance as yf

    return yf


def _first(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value among ``keys`` in ``item``."""
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return default


def _format_timestamp(value: Any) -> str | None:
    """Normalize a publication timestamp to an ISO-8601 UTC string."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            return str(int(value))
    return str(value)


def _contains(text: str, keyword: str) -> bool:
    """Word-boundary match for single words; substring for multi-word phrases."""
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def _classify_sentiment(text: str) -> str:
    """Return ``positive`` / ``negative`` / ``neutral`` from keyword counts."""
    lowered = (text or "").lower()
    positive = sum(1 for keyword in POSITIVE_KEYWORDS if _contains(lowered, keyword))
    negative = sum(1 for keyword in NEGATIVE_KEYWORDS if _contains(lowered, keyword))
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def _extract_keywords(text: str) -> list[str]:
    """Return sorted matched finance/topic keywords found in the title."""
    lowered = (text or "").lower()
    return sorted({keyword for keyword in TOPIC_KEYWORDS if _contains(lowered, keyword)})


def _normalize_news_article(ticker: str, item: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields of interest from a raw Yahoo news article."""
    title = str(_first(item, "title", "headline") or "").strip()
    return {
        "ticker": ticker.upper(),
        "title": title,
        "publisher": str(_first(item, "publisher", "provider", "source", "site") or "").strip(),
        "link": str(_first(item, "link", "url", "canonicalUrl") or "").strip(),
        "published_at": _format_timestamp(
            _first(item, "providerPublishTime", "publishTime", "publishedAt", "pubDate")
        ),
        "sentiment": _classify_sentiment(title),
        "keywords": _extract_keywords(title),
    }


# ---------------------------------------------------------------------------
# Yahoo Finance fetcher (mirrors the MCP tool pattern)
# ---------------------------------------------------------------------------
def get_yahoo_finance_news(
    ticker: str,
    limit: int = DEFAULT_NEWS_LIMIT,
) -> list[dict[str, Any]]:
    """Fetch the latest news for a ticker (MCP ``get_yahoo_finance_news``).

    Raises:
        ValueError: If Yahoo Finance returns no news for the ticker.
    """
    yf = _get_yf()
    raw = yf.Ticker(ticker).news or []
    if not raw:
        raise ValueError(f"No news returned for ticker {ticker!r}")
    return [_normalize_news_article(ticker, item) for item in raw[:limit]]


# ---------------------------------------------------------------------------
# Mock fallbacks
# ---------------------------------------------------------------------------
def _mock_news(ticker: str) -> list[dict[str, Any]]:
    """Return a deterministic mock news payload for a ticker (used on failure)."""
    upper = ticker.upper()
    return [
        {
            "ticker": upper,
            "title": f"{upper} market update (mock)",
            "publisher": "Yahoo Finance",
            "link": f"https://finance.yahoo.com/quote/{upper}",
            "published_at": None,
            "sentiment": "neutral",
            "keywords": [],
            "is_mock": True,
        }
    ]


def _fetch_ticker_news(ticker: str) -> dict[str, Any]:
    """Fetch news for a single ticker with graceful degradation."""
    entry: dict[str, Any] = {
        "ticker": ticker.upper(),
        "source": "yahoo_finance_mcp",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "error": None,
        "is_mock": False,
        "articles": [],
    }

    try:
        entry["articles"] = get_yahoo_finance_news(ticker)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any failure
        logger.warning("get_yahoo_finance_news failed for %s: %s", ticker, exc)
        entry["status"] = "error"
        entry["error"] = str(exc)
        entry["is_mock"] = True
        entry["articles"] = _mock_news(ticker)

    return entry


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------
def news_node(state: AgentState) -> dict[str, Any]:
    """Fetch latest news for every portfolio ticker.

    The structured result is stored under the ``news_results`` state key (a
    list of per-ticker entries) for downstream nodes and the supervisor.
    """
    tickers = [item.ticker for item in state.portfolio]

    news_results: list[dict[str, Any]] = [_fetch_ticker_news(ticker) for ticker in tickers]

    return {
        "news_results": news_results,
        "final_response": (
            f"[news] Fetched latest news for {len(tickers)} ticker(s): "
            f"{', '.join(tickers) or 'none'}."
        ),
    }


__all__ = [
    "DEFAULT_NEWS_LIMIT",
    "POSITIVE_KEYWORDS",
    "NEGATIVE_KEYWORDS",
    "TOPIC_KEYWORDS",
    "get_yahoo_finance_news",
    "news_node",
]
