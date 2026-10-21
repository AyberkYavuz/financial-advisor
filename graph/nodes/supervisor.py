"""Intelligent supervisor node for the multi-agent financial advisor.

The supervisor is responsible for two things:

1. **Task decomposition** — it analyzes the user query (plus portfolio and
   constraint context) and, using the DeepSeek LLM, breaks the request into an
   ordered list of sub-tasks. Each sub-task is assigned to a specialist worker
   (``market_data``, ``rag``, ``news`` or ``optimizer``).

2. **Progress tracking** — it materializes the plan into a root-level
   ``plan.md`` (with Markdown checkboxes) and keeps that file in sync as each
   sub-task is executed, finally emitting a ``FINISH`` route when the plan is
   complete.

When ``DEEPSEEK_API_KEY`` is not available (or the LLM call fails), the
supervisor degrades gracefully to a deterministic keyword router so the graph
remains fully exercisable without any external dependency.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..state import (
    AgentState,
    FINISH,
    MARKET_DATA,
    NEWS,
    OPTIMIZER,
    RAG,
    PlanStep,
    WorkerRoute,
)

logger = logging.getLogger(__name__)

#: Absolute path to the project root (``graph/nodes/supervisor.py`` -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Root-level execution plan file managed by the supervisor.
PLAN_FILE = PROJECT_ROOT / "plan.md"

# Load the root ``.env`` so ``DEEPSEEK_API_KEY`` is available even when the
# module is imported outside of ``langgraph dev`` (e.g. unit tests or scripts).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv is a project dependency
    pass


# ---------------------------------------------------------------------------
# LLM structured-output schemas
# ---------------------------------------------------------------------------
class PlanStepDraft(BaseModel):
    """A single sub-task as returned by the supervisor LLM."""

    description: str = Field(..., description="Concrete, actionable sub-task description.")
    route: WorkerRoute = Field(..., description="Specialist worker to execute this sub-task.")


class SupervisorPlan(BaseModel):
    """Structured plan returned by the supervisor LLM."""

    steps: list[PlanStepDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic fallback router (no LLM required)
# ---------------------------------------------------------------------------
#: Ordered routing table. Order matters: the first table whose keyword matches
#: the lower-cased query wins, so more specific keywords come first.
ROUTING_TABLE: dict[str, tuple[str, ...]] = {
    OPTIMIZER: ("optimize", "rebalance", "reallocate", "constraint", "allocation"),
    NEWS: ("news", "trend", "trends", "sector", "headline", "latest"),
    RAG: (
        "risk",
        "performance",
        "financial",
        "report",
        "10-k",
        "10k",
        "annual",
        "revenue",
        "earnings",
        "risk factor",
    ),
    MARKET_DATA: ("price", "quote", "market data", "current", "stock price", "ticker"),
}

#: Fallback worker when no routing keyword matches.
DEFAULT_NODE = RAG


def _keyword_route(query: str) -> str:
    """Return the worker implied by the query, using the keyword table."""
    lowered = (query or "").lower()
    for node, keywords in ROUTING_TABLE.items():
        if any(keyword in lowered for keyword in keywords):
            return node
    return DEFAULT_NODE


def _fallback_plan(query: str) -> list[PlanStep]:
    """Build a single-step plan using the deterministic keyword router."""
    return [PlanStep(id=1, description=f"Process request: {query}", route=_keyword_route(query))]


# ---------------------------------------------------------------------------
# DeepSeek LLM integration
# ---------------------------------------------------------------------------
_llm: Any | None = None


def get_llm() -> Any | None:
    """Return a lazily-initialized ``ChatOpenAI`` bound to the DeepSeek API.

    Returns ``None`` when ``DEEPSEEK_API_KEY`` is not set so callers can fall
    back to the deterministic router.
    """
    global _llm
    if _llm is not None:
        return _llm

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0,
    )
    return _llm


def _holdings_text(state: AgentState) -> str:
    if not state.portfolio:
        return "none"
    return ", ".join(f"{item.ticker} ({item.shares} shares)" for item in state.portfolio)


def build_supervisor_prompt(state: AgentState) -> list[dict[str, str]]:
    """Construct the system + user messages sent to the supervisor LLM."""
    constraints = state.constraints
    system = (
        "You are the supervisor of a multi-agent financial advisory system. "
        "Decompose the user request into an ordered list of concrete sub-tasks, "
        "assigning each sub-task to exactly one specialist worker.\n\n"
        "Available workers:\n"
        f"- {MARKET_DATA}: fetch real-time market data (Yahoo Finance MCP)\n"
        f"- {RAG}: retrieve and reason over financial documents (10-K / annual reports)\n"
        f"- {NEWS}: fetch the latest news and market analysis (Web Search MCP)\n"
        f"- {OPTIMIZER}: evaluate and optimize the portfolio against its constraints\n\n"
        "Return the sub-tasks in execution order. Be specific and actionable. "
        "A simple request may map to a single sub-task; a complex request may map "
        "to several."
    )
    user = (
        f"User request: {state.query}\n"
        f"Portfolio: {_holdings_text(state)}\n"
        f"Constraints: max_risk={constraints.max_risk}, "
        f"min_dividend_yield={constraints.min_dividend_yield}, "
        f"sector={constraints.sector or 'any'}\n\n"
        "Decompose this request into sub-tasks and assign each to a worker."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _drafts_to_steps(drafts: list[PlanStepDraft]) -> list[PlanStep]:
    """Convert LLM drafts into numbered :class:`PlanStep` objects."""
    return [
        PlanStep(id=index, description=draft.description, route=draft.route)
        for index, draft in enumerate(drafts, start=1)
    ]


def _llm_decompose(llm: Any, state: AgentState) -> list[PlanStep]:
    """Call the LLM (structured output) to decompose the query into a plan."""
    structured = llm.with_structured_output(SupervisorPlan)
    result: SupervisorPlan = structured.invoke(build_supervisor_prompt(state))
    return _drafts_to_steps(result.steps)


def decompose_query(state: AgentState) -> list[PlanStep]:
    """Decompose ``state.query`` into an ordered list of sub-tasks.

    Uses the DeepSeek LLM when available and degrades to the keyword router on
    any failure (missing key or LLM error).
    """
    llm = get_llm()
    if llm is None:
        return _fallback_plan(state.query)
    try:
        return _llm_decompose(llm, state)
    except Exception as exc:  # noqa: BLE001 - graceful degradation is intentional
        logger.warning("Supervisor LLM call failed (%s); falling back to keyword router.", exc)
        return _fallback_plan(state.query)


# ---------------------------------------------------------------------------
# plan.md management
# ---------------------------------------------------------------------------
def render_plan_md(query: str, plan: list[PlanStep]) -> str:
    """Render the execution plan as Markdown with checkbox lists."""
    lines = [
        "# Financial Advisor — Execution Plan",
        "",
        f"**Query:** {query}",
        "",
        "## Sub-tasks",
        "",
    ]
    for step in plan:
        box = "[x]" if step.status == "completed" else "[ ]"
        marker = " *(in progress)*" if step.status == "in_progress" else ""
        lines.append(f"- {box} {step.id}. {step.description} — `{step.route}`{marker}")

    total = len(plan)
    done = sum(1 for step in plan if step.status == "completed")
    lines += [
        "",
        f"_Progress: {done}/{total} completed_",
    ]
    return "\n".join(lines) + "\n"


def write_plan_md(query: str, plan: list[PlanStep], path: Path | None = None) -> Path:
    """Write (or overwrite) the execution plan to ``plan.md``.

    Args:
        query: The user query the plan belongs to.
        plan: The current plan steps (with their statuses).
        path: Optional override target path (used by tests). Defaults to the
            root-level :data:`PLAN_FILE`.
    """
    target = Path(path) if path is not None else PLAN_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_plan_md(query, plan), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------
def _summarize(plan: list[PlanStep]) -> str:
    """Build a concise final-response summary from a completed plan."""
    if not plan:
        return "No action required for this request."
    descriptions = "; ".join(step.description for step in plan)
    return f"Completed {len(plan)} sub-task(s): {descriptions}."


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Plan the request (first entry) or advance progress (subsequent entries).

    * **Planning phase** — ``state.plan`` is empty: decompose the query, write
      ``plan.md``, mark the first step ``in_progress``, and route to it.
    * **Progress phase** — ``state.plan`` is populated: mark the step at
      ``state.current_step`` as ``completed``, update ``plan.md``, and route to
      the next step — or to ``FINISH`` once every step is complete.
    """
    plan = [step.model_copy() for step in state.plan]

    if not plan:
        steps = decompose_query(state)
        if not steps:
            write_plan_md(state.query, [])
            return {
                "plan": [],
                "current_step": 0,
                "next_node": FINISH,
                "final_response": "No action required for this request.",
            }
        steps[0].status = "in_progress"
        write_plan_md(state.query, steps)
        return {
            "plan": steps,
            "current_step": 0,
            "next_node": steps[0].route,
        }

    # Progress phase: mark the current step completed.
    current = state.current_step
    if 0 <= current < len(plan):
        plan[current].status = "completed"

    next_index = current + 1
    if next_index < len(plan):
        plan[next_index].status = "in_progress"
        write_plan_md(state.query, plan)
        return {
            "plan": plan,
            "current_step": next_index,
            "next_node": plan[next_index].route,
        }

    # All steps complete.
    write_plan_md(state.query, plan)
    return {
        "plan": plan,
        "current_step": next_index,
        "next_node": FINISH,
        "final_response": _summarize(plan),
    }


def route_supervisor(state: AgentState) -> str:
    """Conditional-edge function returning the node selected by the supervisor."""
    return state.next_node


__all__ = [
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
    "supervisor_node",
    "route_supervisor",
]
