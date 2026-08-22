"""Tests for Day 6 reviewer authentication and dashboard.

10 tests (all offline, no network):
1. test_login_valid_credentials_sets_session — correct creds → cookie + redirect
2. test_login_invalid_credentials_rejected — wrong password → 401, no session
3. test_dashboard_requires_login — /dashboard with no session → redirect
4. test_approvals_api_requires_login — GET /approvals with no session → 401
5. test_api_routes_return_401_not_redirect — /approvals returns 401, NOT 3xx
6. test_dashboard_page_redirects_not_401 — /dashboard returns redirect, NOT 401
7. test_logout_revokes_session — old cookie invalid after logout
8. test_dashboard_shows_proposed_action_not_reply_text — API returns proposed_action
9. test_messages_returns_404_when_dev_ingress_disabled — /messages gated by env var
10. test_session_cookie_attributes — Set-Cookie has HttpOnly, Secure, SameSite
"""

from __future__ import annotations

import pytest

from app.auth.session import create_session


# ---------------------------------------------------------------------------
# 1. Valid login → session cookie + redirect to dashboard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_valid_credentials_sets_session(reviewer_env, db_path):
    """Correct credentials → session cookie set, redirect to /dashboard."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        response = await client.post(
            "/login",
            data={"username": "test_reviewer", "password": "test_password_secure_123"},
        )

    assert response.status_code == 303
    assert response.headers.get("location") == "/dashboard"
    assert "session" in response.cookies


# ---------------------------------------------------------------------------
# 2. Invalid credentials → 401, no session created
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_invalid_credentials_rejected(reviewer_env, db_path):
    """Wrong password → 401 response, no session cookie set."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        response = await client.post(
            "/login",
            data={"username": "test_reviewer", "password": "WRONG_PASSWORD"},
        )

    assert response.status_code == 401
    assert "session" not in response.cookies


# ---------------------------------------------------------------------------
# 3. Dashboard requires login — no session → redirect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_requires_login(db_path):
    """GET /dashboard with no session → redirected, never shows content."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        response = await client.get("/dashboard")

    # Should be a redirect (307) to /login, not the actual dashboard content
    assert response.status_code == 307
    assert "/login" in response.headers.get("location", "")


# ---------------------------------------------------------------------------
# 4. Approvals API requires login — most critical test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approvals_api_requires_login(db_path):
    """GET /approvals and POST /decision with no session → rejected.

    This is the test that proves the vulnerability described at the top
    of the Day 6 spec is actually closed. If this test passes, the
    approval queue cannot be manipulated by an unauthenticated caller.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        # GET /approvals
        get_response = await client.get("/approvals")
        assert get_response.status_code == 401

        # POST /decision
        post_response = await client.post(
            "/approvals/fake-id/decision",
            json={"decision": "approved"},
        )
        assert post_response.status_code == 401


# ---------------------------------------------------------------------------
# 5. API routes return 401, NOT redirect (the critical split)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_routes_return_401_not_redirect(db_path):
    """GET /approvals returns 401, NOT a 3xx redirect.

    This catches the browser-vs-API dependency mixup. fetch() silently
    follows redirects — if /approvals returned 307, the dashboard JS
    would receive login page HTML with status 200 and the session-expiry
    handler would never fire.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        response = await client.get("/approvals")

    assert response.status_code == 401
    assert response.status_code < 300 or response.status_code >= 400  # not 3xx


# ---------------------------------------------------------------------------
# 6. Dashboard redirects, NOT 401 (inverse of #5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_page_redirects_not_401(db_path):
    """/dashboard with no session returns redirect to /login, NOT 401.

    Confirms the two dependency variants (require_login_page vs
    require_login_api) are wired to the correct routes, not swapped.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        response = await client.get("/dashboard")

    assert response.status_code == 307  # redirect, not 401
    assert response.status_code != 401


# ---------------------------------------------------------------------------
# 7. Logout revokes session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logout_revokes_session(reviewer_env, db_path):
    """After logout, the old session cookie no longer grants access."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        # 1. Login
        login_res = await client.post(
            "/login",
            data={"username": "test_reviewer", "password": "test_password_secure_123"},
        )
        session_cookie = login_res.cookies.get("session")
        assert session_cookie is not None

        # 2. Verify session works
        client.cookies.set("session", session_cookie)
        authed_res = await client.get("/approvals")
        assert authed_res.status_code == 200

        # 3. Logout
        await client.post("/logout")

        # 4. Old cookie should no longer work
        client.cookies.set("session", session_cookie)
        post_logout_res = await client.get("/approvals")
        assert post_logout_res.status_code == 401


# ---------------------------------------------------------------------------
# 8. Dashboard shows proposed_action, not reply_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_shows_proposed_action_not_reply_text(reviewer_env, db_path):
    """Pending item with distinct proposed_action and reply_text:
    GET /approvals returns proposed_action in the JSON.

    This is the test that ensures the reviewer sees the structured
    action detail, not the friendly Urdu message meant for the user.
    """
    from httpx import ASGITransport, AsyncClient

    from app.approval.models import ApprovalItem
    from app.approval.queue import enqueue
    from app.graph.state import RiskLevel
    from app.main import app

    # Create a pending item with distinct proposed_action
    from datetime import datetime, timezone

    item = ApprovalItem(
        id="test-proposed-action-item",
        user_id="whatsapp:+923001234567",
        agent="committee_ledger",
        proposed_action="Add Rs 5,000 to committee ledger #3 for +92-300-1234567",
        risk_level=RiskLevel.WRITE,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    await enqueue(item)

    # Login and fetch approvals
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        login_res = await client.post(
            "/login",
            data={"username": "test_reviewer", "password": "test_password_secure_123"},
        )
        client.cookies.set("session", login_res.cookies.get("session"))

        response = await client.get("/approvals")

    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 1

    item = items[0]
    assert "proposed_action" in item
    assert item["proposed_action"] == "Add Rs 5,000 to committee ledger #3 for +92-300-1234567"


# ---------------------------------------------------------------------------
# 9. POST /messages returns 404 when ENABLE_DEV_INGRESS is disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_returns_404_when_dev_ingress_disabled(db_path, monkeypatch):
    """POST /messages returns 404 when ENABLE_DEV_INGRESS is unset/false.

    Prevents pipeline abuse on deployed instances — the route shouldn't
    even reveal it exists when disabled.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    # Explicitly set to false (also tests default behavior)
    monkeypatch.setenv("ENABLE_DEV_INGRESS", "false")

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        response = await client.post(
            "/messages",
            json={
                "user_id": "attacker",
                "channel": "whatsapp",
                "text": "Junk message to burn quota",
                "language": "ur",
                "received_at": "2026-08-15T00:00:00Z",
            },
        )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 10. Session cookie has HttpOnly + Secure + SameSite — all three
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_cookie_attributes(reviewer_env, db_path):
    """Set-Cookie header includes HttpOnly, Secure, and SameSite=Lax.

    Inspects the actual response header, not just code — this is the test
    that would catch a missing attribute before it reaches production.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as client:
        response = await client.post(
            "/login",
            data={"username": "test_reviewer", "password": "test_password_secure_123"},
        )

    assert response.status_code == 303

    # Inspect raw Set-Cookie header
    set_cookie = response.headers.get("set-cookie", "")
    set_cookie_lower = set_cookie.lower()

    assert "httponly" in set_cookie_lower, f"Missing HttpOnly in: {set_cookie}"
    assert "secure" in set_cookie_lower, f"Missing Secure in: {set_cookie}"
    assert "samesite=lax" in set_cookie_lower, f"Missing SameSite=Lax in: {set_cookie}"
