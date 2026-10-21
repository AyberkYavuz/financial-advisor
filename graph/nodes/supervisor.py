"""Intelligent supervisor node for the multi-agent financial advisor.

The supervisor is responsible for two things:

1. **Task decomposition** — it analyzes the user query (plus portfolio and
   constraint context) and, using the configured LLM provider (see
   :mod:`graph.llm`), breaks the request into an ordered list of sub-tasks.
   Each sub-task is assigned to a specialist worker (``market_data``, ``rag``,
   ``news`` or ``optimizer``).

2. **Progress tracking** — it materializes the plan into a root-level
   ``plan.md`` (with Markdown checkboxes) and keeps that file in sync as each
   sub-task is executed, finally emitting a ``FINISH`` route when the plan is
   complete.

When the configured provider's API key is not available (or the LLM call
fails), the supervisor degrades gracefully to a deterministic keyword router so
the graph remains fully exercisable without any external dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..llm import ask_structured, get_llm
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
# LLM integration
# ---------------------------------------------------------------------------
# ``get_llm`` and ``ask_structured`` are imported from :mod:`graph.llm` (see the
# imports at the top of this module) so LLM access is provider-agnostic and
# configured via the ``LLM_PROVIDER`` environment variable.


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
    """Call the LLM (provider-agnostic JSON structured output) to plan."""
    messages = build_supervisor_prompt(state)
    system = messages[0]["content"]
    user = messages[1]["content"]
    result: SupervisorPlan = ask_structured(llm, SupervisorPlan, system=system, user=user)
    return _drafts_to_steps(result.steps)


def decompose_query(state: AgentState) -> list[PlanStep]:
    """Decompose ``state.query`` into an ordered list of sub-tasks.

    Uses the configured LLM provider when available and degrades to the keyword
    router on any failure (missing key or LLM error).
    """
    llm = get_llm()
    if llm is None:
        logger.info("Supervisor: no LLM configured; using keyword router")
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
    logger.info("Supervisor node called")
    logger.info("Supervisor query: %s", state.query)

    plan = [step.model_copy() for step in state.plan]

    if not plan:
        steps = decompose_query(state)
        if not steps:
            logger.info("Supervisor: empty plan -> FINISH")
            write_plan_md(state.query, [])
            return {
                "plan": [],
                "current_step": 0,
                "next_node": FINISH,
                "final_response": "No action required for this request.",
            }
        steps[0].status = "in_progress"
        write_plan_md(state.query, steps)
        logger.info("Supervisor routing to: %s", steps[0].route)
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
        logger.info("Supervisor routing to: %s", plan[next_index].route)
        return {
            "plan": plan,
            "current_step": next_index,
            "next_node": plan[next_index].route,
        }

    # All steps complete.
    write_plan_md(state.query, plan)
    logger.info("Supervisor: plan complete -> FINISH")
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
    "ask_structured",
    "build_supervisor_prompt",
    "decompose_query",
    "render_plan_md",
    "write_plan_md",
    "supervisor_node",
    "route_supervisor",
]
