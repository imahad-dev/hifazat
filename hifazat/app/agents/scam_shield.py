"""Scam Shield agent — LLM-backed with Day 1-2 canned fallback.

Day 3-4: real scam analysis via provider, with confidence-based
write-tier escalation. Falls back to exact Day 1-2 canned response
on any provider failure or timeout.

Confidence threshold: 0.8 (deliberately conservative). A threshold
set too low floods the approval queue with low-confidence flags.
"""

from __future__ import annotations

from app.graph.state import AgentResponse, IncomingMessage, RiskLevel

SCAM_CONFIDENCE_THRESHOLD = 0.8

_provider = None


def configure(provider) -> None:
    """Set the LLM provider. Called at app startup from main.py."""
    global _provider
    _provider = provider


def _canned_response() -> AgentResponse:
    """Exact Day 1-2 canned scam awareness advice."""
    return AgentResponse(
        agent="scam_shield",
        reply_text=(
            "Yeh message scam ho sakta hai. Apna OTP ya code kisi ko "
            "mat dijiye. Bank kabhi phone par code nahi maangta."
        ),
        risk_level=RiskLevel.READ,
        requires_approval=False,
        metadata={"phase": "stub", "confidence": "n/a"},
    )


async def handle(message: IncomingMessage) -> AgentResponse:
    """Analyze message for scam patterns.

    When provider is available:
    - High confidence (≥0.8): WRITE tier, flag sender for human review
    - Low confidence (<0.8): READ tier, advice only

    On any provider failure: exact Day 1-2 canned response.
    """
    if _provider is not None:
        try:
            analysis = await _provider.analyze_scam(message.text)

            if analysis.confidence >= SCAM_CONFIDENCE_THRESHOLD:
                return AgentResponse(
                    agent="scam_shield",
                    reply_text=analysis.urdu_explanation,
                    risk_level=RiskLevel.WRITE,
                    requires_approval=True,
                    metadata={
                        "confidence": analysis.confidence,
                        "proposed_action": {
                            "action": "flag_sender",
                            "user_id": message.user_id,
                            "confidence": analysis.confidence,
                            "description": (
                                f"Flag sender as potential scammer "
                                f"(confidence: {analysis.confidence})"
                            ),
                            "source_text": message.text,
                        },
                    },
                )
            else:
                return AgentResponse(
                    agent="scam_shield",
                    reply_text=analysis.urdu_explanation,
                    risk_level=RiskLevel.READ,
                    requires_approval=False,
                    metadata={"confidence": analysis.confidence},
                )
        except Exception:
            pass

    return _canned_response()
