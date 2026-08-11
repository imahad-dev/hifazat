"""Tests for the approval queue — CRUD operations and full API roundtrip.

Covers Definition of Done items:
- GET /approvals and decision endpoint work against real SQLite rows
- Full roundtrip (POST /messages → queued → approved → resolved) passes
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.approval import queue as approval_queue
from app.approval.models import ApprovalItem
from app.graph.state import RiskLevel


@pytest.mark.asyncio
async def test_enqueue_and_list_pending(db_path):
    """Enqueue 2 items → list_pending() returns exactly 2."""
    for i in range(2):
        item = ApprovalItem(
            id=str(uuid.uuid4()),
            user_id=f"user_{i}",
            agent="committee_ledger",
            proposed_action=f"Test action {i}",
            risk_level=RiskLevel.WRITE,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        await approval_queue.enqueue(item)

    pending = await approval_queue.list_pending()
    assert len(pending) == 2
    assert all(item.status == "pending" for item in pending)


@pytest.mark.asyncio
async def test_resolve_approved(db_path):
    """Resolve an item as 'approved' → status changes, item returned."""
    item_id = str(uuid.uuid4())
    item = ApprovalItem(
        id=item_id,
        user_id="user_resolve",
        agent="committee_ledger",
        proposed_action="Approve test",
        risk_level=RiskLevel.WRITE,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    await approval_queue.enqueue(item)

    resolved = await approval_queue.resolve(item_id, "approved")
    assert resolved is not None
    assert resolved.status == "approved"
    assert resolved.id == item_id

    # Verify it no longer appears in pending
    pending = await approval_queue.list_pending()
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_resolve_rejected(db_path):
    """Resolve an item as 'rejected' → status changes."""
    item_id = str(uuid.uuid4())
    item = ApprovalItem(
        id=item_id,
        user_id="user_reject",
        agent="committee_ledger",
        proposed_action="Reject test",
        risk_level=RiskLevel.WRITE,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    await approval_queue.enqueue(item)

    resolved = await approval_queue.resolve(item_id, "rejected")
    assert resolved is not None
    assert resolved.status == "rejected"


@pytest.mark.asyncio
async def test_resolve_nonexistent_returns_none(db_path):
    """Resolving a random UUID returns None, not an error."""
    result = await approval_queue.resolve(str(uuid.uuid4()), "approved")
    assert result is None


@pytest.mark.asyncio
async def test_full_roundtrip_via_api(async_client):
    """Full roundtrip: POST /messages → GET /approvals → POST decision → resolved.

    This is the critical end-to-end test proving the async approval
    pattern works: message in → queued → human decides → resolved.
    """
    # 1. Send a committee ledger message (write-tier → queued)
    msg_payload = {
        "user_id": "roundtrip_user",
        "channel": "whatsapp",
        "text": "Committee ka hissa jama karo",
        "language": "ur",
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = await async_client.post("/messages", json=msg_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending_approval"
    approval_id = data["approval_id"]

    # 2. List pending approvals — should contain our item
    resp = await async_client.get("/approvals")
    assert resp.status_code == 200
    pending = resp.json()
    assert len(pending) >= 1
    assert any(item["id"] == approval_id for item in pending)

    # 3. Approve the item
    resp = await async_client.post(
        f"/approvals/{approval_id}/decision",
        json={"decision": "approved"},
    )
    assert resp.status_code == 200
    resolved = resp.json()
    assert resolved["status"] == "approved"
    assert resolved["id"] == approval_id

    # 4. Verify it's no longer pending
    resp = await async_client.get("/approvals")
    pending_after = resp.json()
    assert not any(item["id"] == approval_id for item in pending_after)
