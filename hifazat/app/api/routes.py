"""FastAPI endpoints for the Hifazat orchestrator.

Day 5: Five endpoints.
- POST /messages — direct API ingress (dev/testing)
- POST /webhooks/twilio/whatsapp — Twilio webhook (fast-ack + background)
- GET /approvals — list pending approval items
- POST /approvals/{item_id}/decision — approve/reject + async notification

The webhook handler does ZERO synchronous work beyond parse + ack.
The full pipeline runs in BackgroundTasks, decoupled from Twilio's
15-second timeout clock.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from app.approval import queue as approval_queue
from app.approval.models import ApprovalItem, DecisionRequest
from app.channels.twilio_whatsapp import (
    build_incoming_message,
    download_media,
    has_audio_attachment,
    validate_signature,
)
from app.graph.graph import graph
from app.graph.state import IncomingMessage
from app.pipeline import process_and_reply

router = APIRouter()


# ---------------------------------------------------------------------------
# Direct API ingress (dev/testing — curl, httpie, etc.)
# ---------------------------------------------------------------------------


@router.post("/messages")
async def handle_message(message: IncomingMessage) -> dict:
    """Ingest a message, run the orchestrator graph, return result.

    STT pre-processing: any message with audio_path and no text gets
    transcribed before entering the graph. Channel-agnostic.
    """
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
# Twilio WhatsApp webhook
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
# Approval queue
# ---------------------------------------------------------------------------


@router.get("/approvals", response_model=list[ApprovalItem])
async def list_approvals() -> list[ApprovalItem]:
    """List all pending approval items."""
    return await approval_queue.list_pending()


@router.post("/approvals/{item_id}/decision", response_model=ApprovalItem)
async def decide_approval(
    item_id: str,
    body: DecisionRequest,
    background_tasks: BackgroundTasks,
) -> ApprovalItem:
    """Approve or reject a queued action.

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

