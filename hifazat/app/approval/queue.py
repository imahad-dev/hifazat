"""Async SQLite-backed approval queue.

Fire-and-forget pattern: risk_gate enqueues and returns immediately.
Approval decision arrives later via a separate HTTP endpoint.

Connection strategy: each function opens/closes its own aiosqlite
connection. SQLite doesn't benefit from pooling, and short-lived
connections keep the module stateless enough for testing.
"""

from __future__ import annotations

import os
from datetime import datetime

import aiosqlite

from app.approval.models import ApprovalItem

_db_path: str = ""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);
"""


async def init_db(db_path: str | None = None) -> None:
    """Initialise the approval database.

    Resolution order for db_path:
    1. Explicit argument (used by tests)
    2. APPROVAL_DB_PATH environment variable
    3. ./hifazat_approvals.db fallback
    """
    global _db_path
    if db_path is not None:
        _db_path = db_path
    else:
        _db_path = os.environ.get("APPROVAL_DB_PATH", "./hifazat_approvals.db")

    async with aiosqlite.connect(_db_path) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()


async def enqueue(item: ApprovalItem) -> str:
    """Insert a pending approval item. Returns the item ID."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            INSERT INTO approvals (id, user_id, agent, proposed_action,
                                   risk_level, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.user_id,
                item.agent,
                item.proposed_action,
                item.risk_level.value,
                item.status,
                item.created_at.isoformat(),
            ),
        )
        await db.commit()
    return item.id


async def list_pending() -> list[ApprovalItem]:
    """Return all items with status='pending'."""
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at"
        )
        rows = await cursor.fetchall()
    return [_row_to_item(row) for row in rows]


async def resolve(item_id: str, decision: str) -> ApprovalItem | None:
    """Update status to 'approved' or 'rejected'. Returns updated item or None."""
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(
            "UPDATE approvals SET status = ? WHERE id = ?",
            (decision, item_id),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM approvals WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()

    if row is None:
        return None
    return _row_to_item(row)


def _row_to_item(row: aiosqlite.Row) -> ApprovalItem:
    """Convert a database row to an ApprovalItem."""
    return ApprovalItem(
        id=row["id"],
        user_id=row["user_id"],
        agent=row["agent"],
        proposed_action=row["proposed_action"],
        risk_level=row["risk_level"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
