"""LangGraph state graph wiring for the Hifazat orchestrator.

Flow:
    START → classify_intent → [conditional] → agent_node → risk_gate → END

The conditional edge routes on Intent enum to one of four agent nodes.
risk_gate is the single exit point — all agents converge here.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    classify_intent,
    risk_gate,
    run_committee_ledger,
    run_hisaab_sathi,
    run_scam_shield,
    run_unknown_handler,
)
from app.graph.state import GraphState, Intent


def _route_to_agent(state: GraphState) -> str:
    """Conditional routing function: intent → agent node name."""
    match state.get("intent"):
        case Intent.SCAM_CHECK:
            return "run_scam_shield"
        case Intent.BUDGET_QUERY:
            return "run_hisaab_sathi"
        case Intent.LEDGER_ACTION:
            return "run_committee_ledger"
        case _:
            return "run_unknown_handler"


def build_graph() -> StateGraph:
    """Construct and compile the orchestrator graph."""
    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("run_scam_shield", run_scam_shield)
    builder.add_node("run_hisaab_sathi", run_hisaab_sathi)
    builder.add_node("run_committee_ledger", run_committee_ledger)
    builder.add_node("run_unknown_handler", run_unknown_handler)
    builder.add_node("risk_gate", risk_gate)

    # Entry
    builder.set_entry_point("classify_intent")

    # Conditional edge: classify_intent → agent
    builder.add_conditional_edges(
        "classify_intent",
        _route_to_agent,
        {
            "run_scam_shield": "run_scam_shield",
            "run_hisaab_sathi": "run_hisaab_sathi",
            "run_committee_ledger": "run_committee_ledger",
            "run_unknown_handler": "run_unknown_handler",
        },
    )

    # All agents converge to risk_gate
    builder.add_edge("run_scam_shield", "risk_gate")
    builder.add_edge("run_hisaab_sathi", "risk_gate")
    builder.add_edge("run_committee_ledger", "risk_gate")
    builder.add_edge("run_unknown_handler", "risk_gate")

    # risk_gate → END
    builder.add_edge("risk_gate", END)

    return builder.compile()


# Module-level compiled graph — import and use directly
graph = build_graph()
