"""Session-based reviewer authentication.

In-memory opaque token sessions — no new dependencies, no signed payloads.
Tokens generated via secrets.token_urlsafe(32), stored server-side in a set.
Credential check is timing-safe via secrets.compare_digest.

Accepted trade-off: sessions lost on server restart / Render sleep.
This is a documented limitation, not a bug — reviewer just re-logs in.

Cookie attributes: HttpOnly + Secure + SameSite=Lax — all three, always.

Two dependency variants exist (require_login_page vs require_login_api)
because fetch() silently follows redirects. See docstrings for details.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

_valid_sessions: set[str] = set()


def create_session() -> str:
    """Generate an opaque session token and register it server-side."""
    token = secrets.token_urlsafe(32)
    _valid_sessions.add(token)
    return token


def is_valid(token: str | None) -> bool:
    """Check if a session token is currently valid (set membership)."""
    return token is not None and token in _valid_sessions


def revoke(token: str) -> None:
    """Invalidate a session token. Idempotent — safe to call twice."""
    _valid_sessions.discard(token)


def check_credentials(username: str, password: str) -> bool:
    """Timing-safe credential check against env vars.

    Uses secrets.compare_digest to avoid leaking password length
    or content through response-time differences.
    """
    expected_user = os.environ.get("REVIEWER_USERNAME", "")
    expected_pass = os.environ.get("REVIEWER_PASSWORD", "")

    if not expected_user or not expected_pass:
        return False

    user_ok = secrets.compare_digest(username.encode(), expected_user.encode())
    pass_ok = secrets.compare_digest(password.encode(), expected_pass.encode())
    return user_ok and pass_ok


async def require_login_page(request: Request) -> None:
    """Dependency for browser-navigated pages (e.g. /dashboard).

    Returns a 307 redirect to /login on missing/invalid session.
    Browsers follow this naturally and show the login page.
    """
    token = request.cookies.get("session")
    if not is_valid(token):
        raise HTTPException(
            status_code=307,
            headers={"Location": "/login"},
        )


async def require_login_api(request: Request) -> None:
    """Dependency for fetch()-called API routes (e.g. /approvals).

    Returns 401 on missing/invalid session — NOT a redirect.

    Why this is separate from require_login_page:
    fetch() silently follows HTTP redirects. If an API route returned
    307 → /login, the JS would receive the login page's HTML with
    status 200, and the session-expiry handler (checking for 401)
    would never fire. The dashboard would just break confusingly
    instead of redirecting cleanly.
    """
    token = request.cookies.get("session")
    if not is_valid(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
