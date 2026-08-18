"""FastAPI endpoints for the Hifazat orchestrator.

Day 6: Eight endpoints.
- POST /messages — dev/testing ingress, gated by ENABLE_DEV_INGRESS (default off)
- POST /webhooks/twilio/whatsapp — Twilio webhook (fast-ack + background)
- GET /login — serves the login page (unauthenticated)
- POST /login — validates credentials, sets session cookie
- POST /logout — revokes session, clears cookie
- GET /dashboard — reviewer dashboard (behind require_login_page)
- GET /approvals — list pending items (behind require_login_api)
- POST /approvals/{item_id}/decision — approve/reject (behind require_login_api)

Auth split: /dashboard uses require_login_page (307 redirect).
/approvals and /decision use require_login_api (401).
This is deliberate — see session.py docstrings for why.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from app.approval import queue as approval_queue
from app.approval.models import ApprovalItem, DecisionRequest
from app.auth.session import (
    check_credentials,
    create_session,
    require_login_api,
    require_login_page,
    revoke,
)
from app.channels.twilio_whatsapp import (
    build_incoming_message,
    download_media,
    has_audio_attachment,
    validate_signature,
)
from app.graph.graph import graph
from app.graph.state import IncomingMessage
from app.pipeline import process_and_reply

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter()


# ---------------------------------------------------------------------------
# Direct API ingress (dev/testing — gated by ENABLE_DEV_INGRESS)
# ---------------------------------------------------------------------------


@router.post("/messages")
async def handle_message(message: IncomingMessage) -> dict:
    """Ingest a message, run the orchestrator graph, return result.

    Gated by ENABLE_DEV_INGRESS env var (default: false).
    Returns 404 when disabled — not 401, to avoid revealing the route exists.

    This was scoped as a local dev-testing convenience back on Day 1-2.
    Left open on a deployed instance, anyone who finds the URL can trigger
    the full pipeline directly — burning Gemini quota, writing junk into
    the approval queue. 404 on production is the right default.
    """
    if os.environ.get("ENABLE_DEV_INGRESS", "false").lower() != "true":
        raise HTTPException(status_code=404)

    if message.audio_path and not message.text:
        from app.stt.whisper_engine import transcribe

        message = message.model_copy(
            update={"text": await transcribe(message.audio_path, message.language)}
        )

    if not message.text:
        raise HTTPException(
            status_code=422,
            detail="No text or audio_path provided",
        )

    initial_state = {"message": message}
    result = await graph.ainvoke(initial_state)
    return result["final_reply"]


# ---------------------------------------------------------------------------
# Auth: login, logout
# ---------------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
async def login_page() -> HTMLResponse:
    """Serve the login page. No auth required."""
    html_path = STATIC_DIR / "login.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.post("/login")
async def login(request: Request) -> Response:
    """Validate credentials, set session cookie, redirect to dashboard.

    Cookie attributes: HttpOnly + Secure + SameSite=Lax — all three.
    Returns 401 on invalid credentials (not a redirect).
    """
    form = await request.form()
    username = str(form.get("username", ""))
    password = str(form.get("password", ""))

    if not check_credentials(username, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_session()
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return response


@router.post("/logout")
async def logout(request: Request) -> Response:
    """Revoke session, clear cookie, redirect to login."""
    token = request.cookies.get("session", "")
    revoke(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="session", path="/")
    return response


# ---------------------------------------------------------------------------
# Dashboard (behind require_login_page — 307 redirect on no session)
# ---------------------------------------------------------------------------


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(_=Depends(require_login_page)) -> HTMLResponse:
    """Serve the reviewer dashboard. Requires valid session."""
    html_path = STATIC_DIR / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Twilio WhatsApp webhook (uses its own signature validation, no session auth)
# ---------------------------------------------------------------------------


@router.post("/webhooks/twilio/whatsapp")
async def twilio_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    """Twilio WhatsApp webhook — fast ack + background processing.

    Does ZERO synchronous work beyond:
    1. Signature validation (CPU-only, sub-ms)
    2. Form parsing (CPU-only, sub-ms)
    3. Media download (if audio attachment — small file, well within 15s)

    The full pipeline (STT + graph + TTS + reply) runs entirely in
    BackgroundTasks, decoupled from Twilio's 15-second timeout clock.
    """
    form = await request.form()
    params = dict(form)

    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    if not validate_signature(url, params, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    message = build_incoming_message(params)

    if has_audio_attachment(params):
        audio_path = await download_media(params["MediaUrl0"])
        message = message.model_copy(update={"audio_path": audio_path})

    reply_to = params["From"]
    background_tasks.add_task(process_and_reply, message, reply_to)

    from twilio.twiml.messaging_response import MessagingResponse

    return Response(
        content=str(MessagingResponse()),
        media_type="application/xml",
    )


# ---------------------------------------------------------------------------
# Approval queue (behind require_login_api — 401 on no session)
# ---------------------------------------------------------------------------


@router.get("/approvals", response_model=list[ApprovalItem])
async def list_approvals(_=Depends(require_login_api)) -> list[ApprovalItem]:
    """List all pending approval items. Requires valid session."""
    return await approval_queue.list_pending()


@router.post("/approvals/{item_id}/decision", response_model=ApprovalItem)
async def decide_approval(
    item_id: str,
    body: DecisionRequest,
    background_tasks: BackgroundTasks,
    _=Depends(require_login_api),
) -> ApprovalItem:
    """Approve or reject a queued action. Requires valid session.

    If the user came via WhatsApp (user_id starts with 'whatsapp:'),
    sends an async notification with TTS voice via BackgroundTasks —
    consistent with the voice-first design (every reply gets voice).
    """
    item = await approval_queue.resolve(item_id, body.decision)
    if item is None:
        raise HTTPException(
            status_code=404, detail=f"Approval item '{item_id}' not found"
        )

    if item.user_id.startswith("whatsapp:"):
        from app.pipeline import notify_approval_resolution

        background_tasks.add_task(
            notify_approval_resolution, item.user_id, body.decision
        )

    return item
