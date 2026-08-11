"""Tests for the LangGraph orchestrator — intent routing and risk gating.

Covers Definition of Done items:
- All three intents route to the correct agent skeleton
- write-tier responses never return a direct reply
- read-tier responses return a direct reply
- Unknown intent returns a fallback
"""

from __future__ import annotations

import pytest

from app.graph.graph import graph
from app.graph.state import Intent


@pytest.mark.asyncio
async def test_scam_keywords_route_to_scam_shield(db_path, sample_scam_message):
    """'OTP' keyword → SCAM_CHECK intent → scam_shield agent."""
    state = {"message": sample_scam_message}
    result = await graph.ainvoke(state)

    assert result["intent"] == Intent.SCAM_CHECK
    assert result["agent_response"].agent == "scam_shield"
    assert result["final_reply"]["status"] == "ok"


@pytest.mark.asyncio
async def test_budget_keywords_route_to_hisaab_sathi(db_path, sample_budget_message):
    """'kitna bachaon' keyword → BUDGET_QUERY intent → hisaab_sathi agent."""
    state = {"message": sample_budget_message}
    result = await graph.ainvoke(state)

    assert result["intent"] == Intent.BUDGET_QUERY
    assert result["agent_response"].agent == "hisaab_sathi"
    assert result["final_reply"]["status"] == "ok"


@pytest.mark.asyncio
async def test_ledger_keywords_route_to_committee_ledger(db_path, sample_ledger_message):
    """'committee' keyword → LEDGER_ACTION intent → committee_ledger agent."""
    state = {"message": sample_ledger_message}
    result = await graph.ainvoke(state)

    assert result["intent"] == Intent.LEDGER_ACTION
    assert result["agent_response"].agent == "committee_ledger"


@pytest.mark.asyncio
async def test_unknown_intent_returns_fallback(db_path, sample_unknown_message):
    """No keyword match → UNKNOWN intent → canned fallback reply."""
    state = {"message": sample_unknown_message}
    result = await graph.ainvoke(state)

    assert result["intent"] == Intent.UNKNOWN
    assert result["agent_response"].agent == "unknown_handler"
    assert result["final_reply"]["status"] == "ok"
    assert "samajh nahi aaya" in result["final_reply"]["reply"]


@pytest.mark.asyncio
async def test_write_risk_never_returns_direct_reply(db_path, sample_ledger_message):
    """Write-tier agent (committee_ledger) must ALWAYS land in approval queue.

    This is the critical architectural invariant: write-tier actions
    never return a direct reply — they return pending_approval.
    """
    state = {"message": sample_ledger_message}
    result = await graph.ainvoke(state)

    assert result["final_reply"]["status"] == "pending_approval"
    assert "approval_id" in result["final_reply"]
    assert result["final_reply"]["status"] != "ok"


@pytest.mark.asyncio
async def test_read_risk_returns_direct_reply(db_path, sample_budget_message):
    """Read-tier agent (hisaab_sathi) returns reply directly, no queue."""
    state = {"message": sample_budget_message}
    result = await graph.ainvoke(state)

    assert result["final_reply"]["status"] == "ok"
    assert "reply" in result["final_reply"]
    assert result["final_reply"].get("approval_id") is None
