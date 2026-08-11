"""Tests for hybrid intent classification.

2 tests:
- Keyword match short-circuits before any LLM call (provider never invoked)
- No-keyword message falls through to provider.classify()
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.graph import nodes
from app.graph.state import IncomingMessage, Intent
from app.llm.provider import FakeProvider


def _make_message(text: str) -> IncomingMessage:
    return IncomingMessage(
        user_id="user_test",
        channel="whatsapp",
        text=text,
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_keyword_match_skips_provider():
    """Send 'OTP bhejo' with a FakeProvider → provider.classify never called."""
    provider = FakeProvider()
    nodes._provider = provider

    state = {"message": _make_message("Mera OTP code bhejo")}
    result = await nodes.classify_intent(state)

    assert result["intent"] == Intent.SCAM_CHECK
    assert provider.classify_call_count == 0


@pytest.mark.asyncio
async def test_no_keyword_falls_through_to_provider():
    """Send 'help me please' with FakeProvider → provider.classify() is called."""
    provider = FakeProvider()
    nodes._provider = provider

    state = {"message": _make_message("help me please")}
    result = await nodes.classify_intent(state)

    assert provider.classify_call_count == 1
    assert result["intent"] == Intent.UNKNOWN
