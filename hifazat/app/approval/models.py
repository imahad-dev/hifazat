"""Approval queue data models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.graph.state import RiskLevel


class ApprovalItem(BaseModel):
    id: str
    user_id: str
    agent: str
    proposed_action: str
    risk_level: RiskLevel
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime


class DecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
