"""Tests for agent fallback behavior and scam confidence escalation.

6 tests total:
- 3 fallback tests: each agent returns exact Day 1-2 canned response
  when provider fails (FailingProvider raises TimeoutError)
- 2 scam confidence tests: high confidence → write-tier, low → read-tier
- 1 structural test: proposed_action is never a copy of reply_text
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agents import committee_ledger, hisaab_sathi, scam_shield
from app.graph.state import IncomingMessage, RiskLevel
from app.llm.provider import FakeProvider, FailingProvider


def _make_message(text: str = "test message") -> IncomingMessage:
    return IncomingMessage(
        user_id="user_test",
        channel="whatsapp",
        text=text,
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fallback tests — the single most important tests in this phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scam_shield_fallback_on_provider_failure():
    """FailingProvider raises TimeoutError → exact Day 1-2 canned response."""
    scam_shield._provider = FailingProvider()
    msg = _make_message("some suspicious message about OTP")

    response = await scam_shield.handle(msg)

    assert response.agent == "scam_shield"
    assert response.risk_level == RiskLevel.READ
    assert "OTP" in response.reply_text
    assert response.metadata.get("phase") == "stub"


@pytest.mark.asyncio
async def test_hisaab_sathi_fallback_on_provider_failure():
    """FailingProvider raises TimeoutError → exact Day 1-2 canned response."""
    hisaab_sathi._provider = FailingProvider()
    msg = _make_message("kitna bachaon?")

    response = await hisaab_sathi.handle(msg)

    assert response.agent == "hisaab_sathi"
    assert response.risk_level == RiskLevel.READ
    assert "50-30-20" in response.reply_text
    assert response.metadata.get("phase") == "stub"


@pytest.mark.asyncio
async def test_committee_ledger_fallback_on_provider_failure():
    """FailingProvider raises TimeoutError → exact Day 1-2 canned response."""
    committee_ledger._provider = FailingProvider()
    msg = _make_message("committee ka hissa jama karo")

    response = await committee_ledger.handle(msg)

    assert response.agent == "committee_ledger"
    assert response.risk_level == RiskLevel.WRITE
    assert "approval" in response.reply_text.lower() or "darkar" in response.reply_text.lower()
    assert response.metadata.get("phase") == "stub"


# ---------------------------------------------------------------------------
# Scam Shield confidence-based escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scam_shield_high_confidence_escalates_to_write():
    """Confidence ≥ 0.8 → WRITE tier, proposed_action has flag_sender."""
    scam_shield._provider = FakeProvider(scam_confidence=0.9, scam_is_scam=True)
    msg = _make_message("aapka account band ho raha hai, OTP bhejein")

    response = await scam_shield.handle(msg)

    assert response.risk_level == RiskLevel.WRITE
    assert response.requires_approval is True
    assert response.metadata["confidence"] == 0.9
    assert response.metadata["proposed_action"]["action"] == "flag_sender"


@pytest.mark.asyncio
async def test_scam_shield_low_confidence_stays_read():
    """Confidence < 0.8 → READ tier, no approval needed."""
    scam_shield._provider = FakeProvider(scam_confidence=0.5, scam_is_scam=True)
    msg = _make_message("yeh kya hai?")

    response = await scam_shield.handle(msg)

    assert response.risk_level == RiskLevel.READ
    assert response.requires_approval is False
    assert response.metadata["confidence"] == 0.5


# ---------------------------------------------------------------------------
# Structural invariant: proposed_action ≠ reply_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposed_action_never_equals_reply_text():
    """Write-tier response: proposed_action must be a structured dict,
    never a copy of reply_text."""
    committee_ledger._provider = FakeProvider()
    msg = _make_message("committee mein 1000 jama karo")

    response = await committee_ledger.handle(msg)

    assert response.risk_level == RiskLevel.WRITE
    proposed = response.metadata.get("proposed_action")
    assert proposed is not None
    assert isinstance(proposed, dict)
    assert proposed != response.reply_text
    assert "action" in proposed
