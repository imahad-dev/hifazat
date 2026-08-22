"""LangGraph node functions for the Hifazat orchestrator.

Day 3-4: six nodes.
1. classify_intent — hybrid: keyword fast-path first, LLM fallback for ambiguous
2. run_scam_shield — async, delegates to scam_shield.handle()
3. run_hisaab_sathi — async, delegates to hisaab_sathi.handle()
4. run_committee_ledger — async, delegates to committee_ledger.handle()
5. run_unknown_handler — canned fallback for unrecognised intents
6. risk_gate — routes by risk level, enqueues write-tier to approval queue

Provider injection: module-level _provider variable, set via set_provider()
at app startup. Tests reset it to None via autouse fixture.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from app.agents import committee_ledger, hisaab_sathi, scam_shield
from app.approval import queue as approval_queue
from app.approval.models import ApprovalItem
from app.graph.state import GraphState, Intent, RiskLevel

# ---------------------------------------------------------------------------
# Provider injection
# ---------------------------------------------------------------------------

_provider = None


def set_provider(provider) -> None:
    """Set the LLM provider for hybrid classification."""
    global _provider
    _provider = provider


# ---------------------------------------------------------------------------
# Keyword lists — embedded here, easy to extract to config later
# ---------------------------------------------------------------------------

SCAM_KEYWORDS: list[str] = [
    # OTP / code solicitation
    "otp",
    "code bhejo",
    "code do",
    "code batayein",
    "code share",
    "verify karo",
    "verify karna",
    # Account-lock / freeze urgency
    "locked",
    "account band",
    "account block",
    "account freeze",
    "permanently band",
    # Prize / lottery
    "prize",
    "inam",
    "lucky draw",
    "lucky winner",
    "jeete hain",
    "claim karne",
    # PIN / CNIC solicitation
    "atm pin",
    "cnic number",
    "bank details bhejein",
    # Investment / Ponzi
    "double paise",
    "double paisa",
    "investment scheme",
    # Authority impersonation
    "bank verify",
    "sbp se",
    "nadra record",
    # Advance-fee
    "delivery charges",
    "sirf charges",
    # Mobile-money impersonation
    "jazzcash se call",
    "easypaisa se call",
    # Urgency framing
    "turant",
    "abhi karo",
    "warna",
    "offer khatam",
]

BUDGET_KEYWORDS: list[str] = [
    "bachat",
    "budget",
    "kitna bachaon",
    "paisa",
    "kharcha",
    "saving",
    "salary",
    "income",
    "expense",
]

LEDGER_KEYWORDS: list[str] = [
    "committee",
    "bisi",
    "ledger",
    "hissa",
    "jama",
    "installment",
    "member",
]


# ---------------------------------------------------------------------------
# Node: classify_intent (hybrid)
# ---------------------------------------------------------------------------


async def classify_intent(state: GraphState) -> dict:
    """Hybrid intent classification.

    1. Keyword fast-path: zero cost, zero latency, zero rate-limit impact
    2. LLM fallback: only when no keyword matches (ambiguous case)
    3. Final fallback: UNKNOWN if LLM also fails

    This is the correct production pattern: cheap deterministic heuristics
    before expensive model calls.
    """
    text = state["message"].text.lower()

    for keyword in SCAM_KEYWORDS:
        if keyword in text:
            return {"intent": Intent.SCAM_CHECK}

    for keyword in BUDGET_KEYWORDS:
        if keyword in text:
            return {"intent": Intent.BUDGET_QUERY}

    for keyword in LEDGER_KEYWORDS:
        if keyword in text:
            return {"intent": Intent.LEDGER_ACTION}

    if _provider is not None:
        try:
            result = await _provider.classify(text)
            return {"intent": result.intent}
        except Exception:
            pass

    return {"intent": Intent.UNKNOWN}


# ---------------------------------------------------------------------------
# Node: agent runners (now async)
# ---------------------------------------------------------------------------


async def run_scam_shield(state: GraphState) -> dict:
    """Delegate to scam_shield agent. May escalate to write-tier."""
    response = await scam_shield.handle(state["message"])
    return {"agent_response": response}


async def run_hisaab_sathi(state: GraphState) -> dict:
    """Delegate to hisaab_sathi agent. Always read-tier."""
    response = await hisaab_sathi.handle(state["message"])
    return {"agent_response": response}


async def run_committee_ledger(state: GraphState) -> dict:
    """Delegate to committee_ledger agent. Always write-tier."""
    response = await committee_ledger.handle(state["message"])
    return {"agent_response": response}


async def run_unknown_handler(state: GraphState) -> dict:
    """Fallback for unrecognised intents. Read-level canned reply."""
    from app.graph.state import AgentResponse

    return {
        "agent_response": AgentResponse(
            agent="unknown_handler",
            reply_text="Mujhe samajh nahi aaya. Kya aap dobara bata sakte hain?",
            risk_level=RiskLevel.READ,
            requires_approval=False,
            metadata={"phase": "stub"},
        )
    }


# ---------------------------------------------------------------------------
# Node: risk_gate
# ---------------------------------------------------------------------------


async def risk_gate(state: GraphState) -> dict:
    """Route by risk level.

    - read/draft: return reply directly
    - write/destructive: generate UUID, build ApprovalItem, enqueue,
      return pending_approval status with the generated approval_id

    proposed_action is sourced from agent_response.metadata["proposed_action"]
    (structured, reviewer-facing), NOT from reply_text (end-user friendly).
    This node is async because enqueue() hits aiosqlite.
    """
    response = state["agent_response"]

    if response.risk_level in (RiskLevel.WRITE, RiskLevel.DESTRUCTIVE):
        approval_id = str(uuid.uuid4())

        raw_action = response.metadata.get("proposed_action")
        if raw_action is not None:
            proposed_action = json.dumps(raw_action, ensure_ascii=False)
        else:
            proposed_action = json.dumps(
                {"_unstructured": True, "text": response.reply_text},
                ensure_ascii=False,
            )

        item = ApprovalItem(
            id=approval_id,
            user_id=state["message"].user_id,
            agent=response.agent,
            proposed_action=proposed_action,
            risk_level=response.risk_level,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        await approval_queue.enqueue(item)

        return {
            "final_reply": {
                "status": "pending_approval",
                "approval_id": approval_id,
            }
        }

    return {
        "final_reply": {
            "status": "ok",
            "reply": response.reply_text,
        }
    }
