"""Scam Shield agent — LLM-backed with Day 1-2 canned fallback.

Day 3-4: real scam analysis via provider, with confidence-based
write-tier escalation. Falls back to exact Day 1-2 canned response
on any provider failure or timeout.

Day 7 tuning:
- Threshold updated from 0.8 (Day 3-4 guess) to 0.75 (measured).
- All 8 scam eval examples scored 0.95-1.00 confidence, well above 0.75.
- 0.75 provides headroom for borderline cases not yet in the eval set.
- Escalation now requires BOTH is_scam=True AND confidence >= threshold.
  Previously only checked confidence, causing false positives: a benign
  message like 'mera OTP nahi aa raha' got confidence=1.0 (model is sure
  it's NOT a scam) and would incorrectly escalate to WRITE tier.
"""

from __future__ import annotations

from app.graph.state import AgentResponse, IncomingMessage, RiskLevel

# Day 7: measured against 14-example eval set with real Gemini calls.
# All scam examples scored >= 0.95. 0.75 catches all known patterns
# while leaving room for borderline cases not yet in the eval set.
# Recall dropped sharply above 0.90 in partial results.
SCAM_CONFIDENCE_THRESHOLD = 0.75

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
    - is_scam=True AND high confidence (>=0.75): WRITE tier, flag for review
    - Otherwise: READ tier, advice only (includes safe messages
      with high confidence — the model is confident it's NOT a scam)

    On any provider failure: exact Day 1-2 canned response.
    """
    if _provider is not None:
        try:
            analysis = await _provider.analyze_scam(message.text)

            if analysis.is_scam and analysis.confidence >= SCAM_CONFIDENCE_THRESHOLD:
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
