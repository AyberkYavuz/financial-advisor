"""Market data node — real-time and historical stock data via Yahoo Finance.

This node fetches current price, key metrics, and historical OHLCV data for
every ticker in the portfolio, and stores the result under the ``market_data``
key of :class:`~graph.state.AgentState`.

The implementation uses the ``yfinance`` library directly, mirroring the two
MCP tools documented in ``yahoo_mcp_reference.md``:

* ``get_stock_info``              -> :func:`get_stock_info`
* ``get_historical_stock_prices`` -> :func:`get_historical_stock_prices`

Every fetch is wrapped in per-ticker exception handling so that network errors,
rate limits, or invalid tickers degrade to deterministic mock payloads rather
than crashing the graph.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)

#: Default period / interval for historical OHLCV requests.
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1d"

#: Mapping of ``yfinance`` ``info`` keys -> normalized snake_case keys kept in
#: the structured output.
_INFO_FIELDS: dict[str, str] = {
    "currentPrice": "current_price",
    "regularMarketPrice": "regular_market_price",
    "currency": "currency",
    "marketCap": "market_cap",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "dividendYield": "dividend_yield",
    "dividendRate": "dividend_rate",
    "sector": "sector",
    "industry": "industry",
    "beta": "beta",
    "fiftyTwoWeekHigh": "fifty_two_week_high",
    "fiftyTwoWeekLow": "fifty_two_week_low",
    "longName": "long_name",
    "shortName": "short_name",
}

#: Deterministic baseline prices used by the mock fallback for known tickers.
_MOCK_BASELINES: dict[str, float] = {
    "AAPL": 220.0,
    "MSFT": 400.0,
    "GOOGL": 160.0,
}


def _get_yf() -> Any:
    """Return the ``yfinance`` module, imported lazily."""
    import yfinance as yf

    return yf


def _jsonable(value: Any) -> Any:
    """Convert a value to a JSON-safe Python native (``NaN``/``inf`` -> ``None``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    try:
        as_float = float(value)
        return None if (math.isnan(as_float) or math.isinf(as_float)) else as_float
    except (TypeError, ValueError):
        return str(value)


def _normalize_info(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    """Extract a curated, snake_case subset of a ``yfinance`` info dict."""
    normalized: dict[str, Any] = {"ticker": ticker}
    for source_key, target_key in _INFO_FIELDS.items():
        if source_key in info:
            normalized[target_key] = _jsonable(info[source_key])
    return normalized


def _normalize_history(df: Any) -> list[dict[str, Any]]:
    """Convert a ``yfinance`` OHLCV ``DataFrame`` into a list of records."""
    records: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        records.append(
            {
                "date": index.strftime("%Y-%m-%d") if hasattr(index, "strftime") else str(index),
                "open": _jsonable(row.get("Open")),
                "high": _jsonable(row.get("High")),
                "low": _jsonable(row.get("Low")),
                "close": _jsonable(row.get("Close")),
                "volume": _jsonable(row.get("Volume")),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Yahoo Finance fetchers (mirror the MCP tool patterns)
# ---------------------------------------------------------------------------
def get_stock_info(ticker: str) -> dict[str, Any]:
    """Fetch current price and key metrics for a ticker (MCP ``get_stock_info``).

    Raises:
        ValueError: If Yahoo Finance returns no data for the ticker.
    """
    yf = _get_yf()
    info = yf.Ticker(ticker).info or {}
    if not info:
        raise ValueError(f"No stock info returned for ticker {ticker!r}")
    return _normalize_info(ticker, info)


def get_historical_stock_prices(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> list[dict[str, Any]]:
    """Fetch historical OHLCV data (MCP ``get_historical_stock_prices``).

    Raises:
        ValueError: If Yahoo Finance returns no history for the ticker.
    """
    yf = _get_yf()
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df is None or getattr(df, "empty", True):
        raise ValueError(f"No historical data returned for ticker {ticker!r}")
    return _normalize_history(df)


# ---------------------------------------------------------------------------
# Mock fallbacks
# ---------------------------------------------------------------------------
def _mock_stock_info(ticker: str) -> dict[str, Any]:
    """Return a deterministic mock payload for a ticker (used on failure)."""
    upper = ticker.upper()
    return {
        "ticker": upper,
        "current_price": _MOCK_BASELINES.get(upper, 0.0),
        "currency": "USD",
        "market_cap": None,
        "trailing_pe": None,
        "dividend_yield": None,
        "sector": "technology" if upper in _MOCK_BASELINES else None,
        "is_mock": True,
    }


def _mock_historical(ticker: str) -> list[dict[str, Any]]:
    """Return a deterministic mock OHLCV payload for a ticker (used on failure)."""
    upper = ticker.upper()
    price = _MOCK_BASELINES.get(upper, 0.0)
    return [
        {
            "date": "2024-01-02",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0,
            "is_mock": True,
        }
    ]


def _fetch_ticker(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Fetch info + history for a single ticker with graceful degradation."""
    logger.info("Market data fetching ticker: %s", ticker)
    result: dict[str, Any] = {
        "ticker": ticker.upper(),
        "status": "ok",
        "error": None,
        "is_mock": False,
        "info": None,
        "historical": None,
    }

    try:
        result["info"] = get_stock_info(ticker)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any failure
        logger.warning("get_stock_info failed for %s: %s", ticker, exc)
        result["status"] = "error"
        result["error"] = str(exc)
        result["is_mock"] = True
        result["info"] = _mock_stock_info(ticker)

    try:
        result["historical"] = get_historical_stock_prices(ticker, period, interval)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any failure
        logger.warning("get_historical_stock_prices failed for %s: %s", ticker, exc)
        if result["status"] == "ok":
            result["status"] = "partial"
        result["error"] = f"{result['error'] or ''}; {exc}".strip("; ")
        result["historical"] = _mock_historical(ticker)

    logger.info(
        "Market data ticker %s processed: status=%s is_mock=%s",
        result["ticker"],
        result["status"],
        result["is_mock"],
    )
    return result


# ---------------------------------------------------------------------------
# Graph node
# ---------------------------------------------------------------------------
def market_data_node(state: AgentState) -> dict[str, Any]:
    """Fetch real-time + historical data for every portfolio ticker.

    The structured result is stored under the ``market_data`` state key for
    downstream nodes (e.g. the optimizer) and the supervisor to consume.
    """
    logger.info("Market data node called")
    tickers = [item.ticker for item in state.portfolio]
    logger.info("Market data processing tickers: %s", tickers)

    data = {ticker: _fetch_ticker(ticker) for ticker in tickers}

    market_data: dict[str, Any] = {
        "source": "yfinance",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "data": data,
    }

    return {
        "market_data": market_data,
        "final_response": (
            f"[market_data] Retrieved data for {len(tickers)} ticker(s): "
            f"{', '.join(tickers) or 'none'}."
        ),
    }


__all__ = [
    "DEFAULT_PERIOD",
    "DEFAULT_INTERVAL",
    "get_stock_info",
    "get_historical_stock_prices",
    "market_data_node",
]
