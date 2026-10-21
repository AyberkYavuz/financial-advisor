"""Node implementations for the multi-agent financial advisor graph."""

from .market_data import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    get_historical_stock_prices,
    get_stock_info,
    market_data_node,
)
from .news import (
    DEFAULT_NEWS_LIMIT,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    TOPIC_KEYWORDS,
    get_yahoo_finance_news,
    news_node,
)
from .optimizer import optimizer_node
from .rag import rag_node
from .supervisor import (
    DEFAULT_NODE,
    PLAN_FILE,
    PROJECT_ROOT,
    ROUTING_TABLE,
    SupervisorPlan,
    PlanStepDraft,
    build_supervisor_prompt,
    decompose_query,
    get_llm,
    render_plan_md,
    route_supervisor,
    supervisor_node,
    write_plan_md,
)

__all__ = [
    # Nodes
    "supervisor_node",
    "route_supervisor",
    "market_data_node",
    "rag_node",
    "news_node",
    "optimizer_node",
    # Market data helpers
    "get_stock_info",
    "get_historical_stock_prices",
    "DEFAULT_PERIOD",
    "DEFAULT_INTERVAL",
    # News helpers
    "get_yahoo_finance_news",
    "DEFAULT_NEWS_LIMIT",
    "POSITIVE_KEYWORDS",
    "NEGATIVE_KEYWORDS",
    "TOPIC_KEYWORDS",
    # Supervisor helpers
    "PROJECT_ROOT",
    "PLAN_FILE",
    "ROUTING_TABLE",
    "DEFAULT_NODE",
    "PlanStepDraft",
    "SupervisorPlan",
    "get_llm",
    "build_supervisor_prompt",
    "decompose_query",
    "render_plan_md",
    "write_plan_md",
]
