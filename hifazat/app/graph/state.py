"""Core schemas for the Hifazat orchestration graph.

Two layers by design:
- Pydantic BaseModel for API boundary validation (IncomingMessage, AgentResponse)
- TypedDict for LangGraph state (GraphState) — avoids silent dict conversion
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class RiskLevel(str, Enum):
    READ = "read"
    DRAFT = "draft"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class Intent(str, Enum):
    SCAM_CHECK = "scam_check"
    BUDGET_QUERY = "budget_query"
    LEDGER_ACTION = "ledger_action"
    UNKNOWN = "unknown"


class IncomingMessage(BaseModel):
    user_id: str
    channel: Literal["whatsapp", "voice"]
    text: Optional[str] = None
    audio_path: Optional[str] = None
    language: Literal["ur", "pa", "en"] = "ur"
    received_at: datetime


class AgentResponse(BaseModel):
    agent: str
    reply_text: str
    risk_level: RiskLevel
    requires_approval: bool
    metadata: dict = {}


class GraphState(TypedDict, total=False):
    """LangGraph state container.

    Uses TypedDict (not BaseModel) because LangGraph merges state via
    dict update. Pydantic models get silently cast to dicts during
    merging, losing validators and causing attribute-access bugs.
    """

    message: IncomingMessage
    intent: Optional[Intent]
    agent_response: Optional[AgentResponse]
    final_reply: Optional[dict]
