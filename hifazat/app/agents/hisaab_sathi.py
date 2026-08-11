"""Hisaab Sathi agent — LLM-backed with Day 1-2 canned fallback.

Day 3-4: real budgeting advice via provider. Always read-tier.
Falls back to exact Day 1-2 canned 50-30-20 response on any
provider failure or timeout.
"""

from __future__ import annotations

from app.graph.state import AgentResponse, IncomingMessage, RiskLevel

_provider = None


def configure(provider) -> None:
    """Set the LLM provider. Called at app startup from main.py."""
    global _provider
    _provider = provider


def _canned_response() -> AgentResponse:
    """Exact Day 1-2 canned budgeting advice."""
    return AgentResponse(
        agent="hisaab_sathi",
        reply_text=(
            "Apni income ka 50% zarooriyat par, 30% khwahishat par, "
            "aur 20% bachat mein rakhein. Yeh 50-30-20 ka asool hai."
        ),
        risk_level=RiskLevel.READ,
        requires_approval=False,
        metadata={"phase": "stub"},
    )


async def handle(message: IncomingMessage) -> AgentResponse:
    """Provide budgeting advice. Always read-tier.

    On any provider failure: exact Day 1-2 canned response.
    """
    if _provider is not None:
        try:
            advice = await _provider.budget_advice(message.text)
            return AgentResponse(
                agent="hisaab_sathi",
                reply_text=advice.urdu_advice,
                risk_level=RiskLevel.READ,
                requires_approval=False,
                metadata={"suggested_saving_pct": advice.suggested_saving_pct},
            )
        except Exception:
            pass

    return _canned_response()
