"""Integration test — real Gemini API call.

Excluded from default pytest run via @pytest.mark.integration.
Run manually: pytest -m integration -v

Requires GEMINI_API_KEY environment variable to be set.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_classify_real_call():
    """Smoke test: one real Gemini call to verify API key + structured output.

    Sends a clear Urdu scam message to provider.classify() and verifies
    the structured ClassificationResult parses correctly.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set — skipping integration test")

    from app.llm.provider import GeminiProvider

    provider = GeminiProvider(api_key=api_key)
    result = await provider.classify("Aapka account band ho raha hai, apna OTP bhejein")

    assert result.intent is not None
    assert 0.0 <= result.confidence <= 1.0
