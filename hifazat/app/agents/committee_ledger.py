"""Committee Ledger agent — LLM-backed with Day 1-2 canned fallback.

Day 3-4: real ledger draft parsing via provider. Always write-tier.
Ledger mutations always need human approval — no exceptions.
Falls back to exact Day 1-2 canned response on any provider failure.
"""

from __future__ import annotations

from app.graph.state import AgentResponse, IncomingMessage, RiskLevel

_provider = None


def configure(provider) -> None:
    """Set the LLM provider. Called at app startup from main.py."""
    global _provider
    _provider = provider


def _canned_response(message: IncomingMessage) -> AgentResponse:
    """Exact Day 1-2 canned ledger response."""
    return AgentResponse(
        agent="committee_ledger",
        reply_text=(
            "Committee mein aapka hissa darj kiya jaayega. "
            "Is action ke liye approval darkar hai."
        ),
        risk_level=RiskLevel.WRITE,
        requires_approval=True,
        metadata={
            "phase": "stub",
            "proposed_action": {
                "action": "record_contribution",
                "user_id": message.user_id,
                "description": f"Record committee contribution for user {message.user_id}",
                "source_text": message.text,
            },
        },
    )


async def handle(message: IncomingMessage) -> AgentResponse:
    """Parse message into a ledger draft. Always write-tier.

    On any provider failure: exact Day 1-2 canned response.
    proposed_action is built from LedgerDraft fields — distinct
    from reply_text by construction.
    """
    if _provider is not None:
        try:
            draft = await _provider.draft_ledger_action(message.text)
            return AgentResponse(
                agent="committee_ledger",
                reply_text=draft.urdu_confirmation_draft,
                risk_level=RiskLevel.WRITE,
                requires_approval=True,
                metadata={
                    "proposed_action": {
                        "action": draft.action_type,
                        "amount": draft.amount,
                        "member_reference": draft.member_reference,
                        "user_id": message.user_id,
                        "description": (
                            f"{draft.action_type}: "
                            f"amount={draft.amount}, "
                            f"member={draft.member_reference}"
                        ),
                        "source_text": message.text,
                    },
                },
            )
        except Exception:
            pass

    return _canned_response(message)
