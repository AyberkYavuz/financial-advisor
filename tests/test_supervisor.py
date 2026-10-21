"""Unit tests for the intelligent supervisor node.

These tests are deterministic and offline: the DeepSeek LLM is replaced with
fakes via monkeypatching, and ``plan.md`` is redirected to a temporary path so
the real project-root file is never touched.
"""

from __future__ import annotations

from typing import Any

import pytest

import graph.nodes.supervisor as sup
from graph import (
    FINISH,
    MARKET_DATA,
    NEWS,
    OPTIMIZER,
    RAG,
    TEST_CONSTRAINTS,
    TEST_PORTFOLIO,
    AgentState,
    PlanStep,
)
from graph.nodes.supervisor import (
    PlanStepDraft,
    SupervisorPlan,
    decompose_query,
    render_plan_md,
    route_supervisor,
    supervisor_node,
    write_plan_md,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeResponse:
    """Minimal response object exposing a ``.content`` string."""

    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Fake LLM whose ``invoke`` returns a fixed plan serialized as JSON."""

    def __init__(self, plan: SupervisorPlan):
        self._plan = plan

    def invoke(self, messages: Any) -> _FakeResponse:
        return _FakeResponse(self._plan.model_dump_json())


class ExplodingLLM:
    """Fake LLM that raises, to exercise the graceful-degradation path."""

    def invoke(self, messages: Any) -> None:
        raise RuntimeError("boom")


def make_state(query: str = "Optimize this portfolio") -> AgentState:
    return AgentState(query=query, portfolio=TEST_PORTFOLIO, constraints=TEST_CONSTRAINTS)


def two_step_plan() -> SupervisorPlan:
    return SupervisorPlan(
        steps=[
            PlanStepDraft(description="Fetch market data for AAPL, MSFT, GOOGL", route=MARKET_DATA),
            PlanStepDraft(description="Optimize the portfolio", route=OPTIMIZER),
        ]
    )


# ---------------------------------------------------------------------------
# plan.md rendering / writing
# ---------------------------------------------------------------------------
def test_render_plan_md_uses_checkboxes():
    plan = [
        PlanStep(id=1, description="A", route=RAG, status="completed"),
        PlanStep(id=2, description="B", route=NEWS, status="in_progress"),
        PlanStep(id=3, description="C", route=OPTIMIZER, status="pending"),
    ]
    md = render_plan_md("hello", plan)
    assert "- [x] 1. A — `rag`" in md
    assert "- [ ] 2. B — `news` *(in progress)*" in md
    assert "- [ ] 3. C — `optimizer`" in md
    assert "_Progress: 1/3 completed_" in md
    assert "**Query:** hello" in md


def test_write_plan_md_creates_and_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")

    target = write_plan_md("q1", [PlanStep(id=1, description="A", route=RAG)])
    assert target.exists()
    assert "**Query:** q1" in target.read_text(encoding="utf-8")

    # Overwrite with new content.
    write_plan_md("q2", [PlanStep(id=1, description="B", route=NEWS, status="completed")])
    text = target.read_text(encoding="utf-8")
    assert "**Query:** q2" in text
    assert "- [x] 1. B — `news`" in text
    assert "q1" not in text


# ---------------------------------------------------------------------------
# Decomposition (fallback + LLM)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How has AAPL's financial performance been over the last 5 years?", RAG),
        ("What are MSFT's risk factors?", RAG),
        ("What trends are being seen in the Tech sector?", NEWS),
        ("Optimize this portfolio", OPTIMIZER),
        ("What do the latest news say about the portfolio?", NEWS),
        ("What is the current price of AAPL?", MARKET_DATA),
    ],
)
def test_fallback_decompose_routes_correctly(monkeypatch, query, expected_route):
    monkeypatch.setattr(sup, "get_llm", lambda: None)
    steps = decompose_query(make_state(query))
    assert len(steps) == 1
    assert steps[0].route == expected_route
    assert steps[0].status == "pending"


def test_llm_decompose_multi_step(monkeypatch):
    monkeypatch.setattr(sup, "get_llm", lambda: FakeLLM(two_step_plan()))
    steps = decompose_query(make_state("Analyze and optimize my portfolio"))
    assert [s.route for s in steps] == [MARKET_DATA, OPTIMIZER]
    assert [s.id for s in steps] == [1, 2]
    assert all(s.status == "pending" for s in steps)


def test_decompose_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr(sup, "get_llm", lambda: ExplodingLLM())
    steps = decompose_query(make_state("What are MSFT's risk factors?"))
    assert len(steps) == 1
    assert steps[0].route == RAG  # keyword fallback


def test_build_supervisor_prompt_includes_context():
    state = make_state("Optimize this portfolio")
    messages = sup.build_supervisor_prompt(state)
    joined = " ".join(m["content"] for m in messages)
    assert "Optimize this portfolio" in joined
    assert "AAPL (100 shares)" in joined
    assert "MSFT (50 shares)" in joined
    assert "GOOGL (25 shares)" in joined
    assert "max_risk=medium" in joined
    assert "min_dividend_yield=2.0" in joined
    assert "technology" in joined


# ---------------------------------------------------------------------------
# Supervisor node: planning / progress / finish
# ---------------------------------------------------------------------------
def test_supervisor_planning_writes_plan_md(monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: FakeLLM(two_step_plan()))

    out = supervisor_node(make_state())
    assert out["next_node"] == MARKET_DATA
    assert out["current_step"] == 0
    assert out["plan"][0].status == "in_progress"
    assert out["plan"][1].status == "pending"

    text = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "- [ ] 1. Fetch market data for AAPL, MSFT, GOOGL — `market_data` *(in progress)*" in text
    assert "- [ ] 2. Optimize the portfolio — `optimizer`" in text


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("Optimize this portfolio", OPTIMIZER),
        ("What is the current price of AAPL?", MARKET_DATA),
        ("What are AAPL's risk factors?", RAG),
        ("Get the latest news about AAPL", NEWS),
    ],
)
def test_supervisor_node_routes_by_intent(monkeypatch, tmp_path, query, expected_route):
    """Non-RAG and RAG intents each route to the correct worker (single-turn)."""
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: None)  # keyword router

    out = supervisor_node(make_state(query))
    assert out["next_node"] == expected_route
    assert out["planned_query"] == query


def test_supervisor_replans_when_query_changes(monkeypatch, tmp_path):
    """A new query in a multi-turn conversation replaces the stale plan."""
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: FakeLLM(two_step_plan()))

    # Turn 1: a plan is built for the first query.
    state = make_state("What are AAPL's risk factors?")
    out1 = supervisor_node(state)
    assert out1["next_node"] == MARKET_DATA
    assert out1["planned_query"] == "What are AAPL's risk factors?"

    # Turn 2: a new query arrives while the old plan is still in state.
    state2 = state.model_copy(update=out1).model_copy(
        update={"query": "Optimize this portfolio"}
    )
    out2 = supervisor_node(state2)

    # The supervisor must re-plan, not continue the stale plan.
    assert out2["planned_query"] == "Optimize this portfolio"
    assert out2["current_step"] == 0
    assert out2["plan"][0].status == "in_progress"


def test_supervisor_progress_then_finish(monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: FakeLLM(two_step_plan()))
    monkeypatch.setattr(sup, "synthesize_final_response", lambda state: "ADVISORY REPORT")

    # Planning phase.
    state = make_state()
    out1 = supervisor_node(state)
    assert out1["next_node"] == MARKET_DATA

    # Worker returns -> progress phase: step 0 completed, step 1 in progress.
    state = state.model_copy(update=out1)
    out2 = supervisor_node(state)
    assert out2["next_node"] == OPTIMIZER
    assert out2["current_step"] == 1
    assert out2["plan"][0].status == "completed"
    assert out2["plan"][1].status == "in_progress"

    # Worker returns -> all complete -> FINISH.
    state = state.model_copy(update=out2)
    out3 = supervisor_node(state)
    assert out3["next_node"] == FINISH
    assert out3["plan"][0].status == "completed"
    assert out3["plan"][1].status == "completed"
    assert out3["final_response"] == "ADVISORY REPORT"

    text = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "- [x] 1. Fetch market data for AAPL, MSFT, GOOGL — `market_data`" in text
    assert "- [x] 2. Optimize the portfolio — `optimizer`" in text
    assert "_Progress: 2/2 completed_" in text


def test_supervisor_empty_plan_finishes(monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: FakeLLM(SupervisorPlan(steps=[])))

    out = supervisor_node(make_state("Hello"))
    assert out["next_node"] == FINISH
    assert out["plan"] == []
    assert out["final_response"] == "No action required for this request."


# ---------------------------------------------------------------------------
# Final response synthesis
# ---------------------------------------------------------------------------
def test_synthesize_final_response_with_llm(monkeypatch):
    class ReportLLM:
        def invoke(self, messages):
            return _FakeResponse(
                "## Executive Summary\n\nA synthesized report with insights."
            )

    monkeypatch.setattr(sup, "get_llm", lambda: ReportLLM())
    result = sup.synthesize_final_response(make_state("What are AAPL's risk factors?"))
    assert "Executive Summary" in result
    assert "synthesized report" in result


def test_synthesize_final_response_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(sup, "get_llm", lambda: None)
    state = AgentState(
        query="What are AAPL's risk factors?",
        portfolio=TEST_PORTFOLIO,
        constraints=TEST_CONSTRAINTS,
        retrieved_docs=[
            {
                "source": "AAPL 10-K (2023).pdf",
                "page": 8,
                "text": "Item 1A. Risk Factors The Company's business...",
            }
        ],
        optimization_result={
            "total_value": 46000.0,
            "compliant": False,
            "violations": ["weighted dividend yield 0.54% is below the minimum 2.00%"],
            "recommendations": ["Rebalance toward higher-yield assets."],
        },
    )
    result = sup.synthesize_final_response(state)
    assert "Executive Summary" in result
    assert "Key Insights from Filings" in result
    assert "Portfolio Performance & Valuation" in result
    assert "Constraint & Sector Analysis" in result
    assert "Actionable Recommendations" in result
    assert "$46,000.00" in result


def test_synthesize_final_response_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr(sup, "get_llm", lambda: ExplodingLLM())
    result = sup.synthesize_final_response(make_state("What are AAPL's risk factors?"))
    assert "Executive Summary" in result


def test_route_supervisor_returns_next_node():
    state = make_state().model_copy(update={"next_node": NEWS})
    assert route_supervisor(state) == NEWS


# ---------------------------------------------------------------------------
# Graph state-transition integrity
# ---------------------------------------------------------------------------
def test_graph_state_transition_integrity(compiled_graph, monkeypatch, tmp_path):
    monkeypatch.setattr(sup, "PLAN_FILE", tmp_path / "plan.md")
    monkeypatch.setattr(sup, "get_llm", lambda: FakeLLM(two_step_plan()))
    monkeypatch.setattr(sup, "synthesize_final_response", lambda state: "ADVISORY REPORT")

    result = compiled_graph.invoke(
        {
            "query": "Analyze and optimize my portfolio",
            "portfolio": TEST_PORTFOLIO,
            "constraints": TEST_CONSTRAINTS,
        }
    )

    # The run should have looped supervisor -> worker -> supervisor -> worker
    # -> supervisor and terminated on FINISH.
    assert result["next_node"] == FINISH
    assert [s.status for s in result["plan"]] == ["completed", "completed"]
    assert result["current_step"] == 2
    assert result["final_response"] == "ADVISORY REPORT"

    text = (tmp_path / "plan.md").read_text(encoding="utf-8")
    assert "_Progress: 2/2 completed_" in text
    assert text.count("[x]") == 2
