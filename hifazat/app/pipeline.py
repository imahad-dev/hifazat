"""Background pipeline: STT → graph → TTS → send reply.

Extracted from routes.py to keep the webhook handler thin.
Called via BackgroundTasks — completely decoupled from Twilio's 15s clock.

TTS failure degrades to text-only, never blocks the reply.
Audio cleanup runs lazily before each TTS call.

send_voice_reply() is the shared voice-first reply primitive:
used by both process_and_reply (graph pipeline) and
notify_approval_resolution (approval decision notifications).
"""

from __future__ import annotations

import os

from app.channels.twilio_whatsapp import send_whatsapp_reply
from app.graph.graph import graph
from app.graph.state import IncomingMessage
from app.stt.whisper_engine import transcribe
from app.tts.cleanup import cleanup_old_audio
from app.tts.mms_tts_engine import GENERATED_AUDIO_DIR, synthesize


async def send_voice_reply(
    to: str,
    text: str,
) -> None:
    """Send a WhatsApp reply with optional TTS voice attachment.

    This is the voice-first reply primitive used everywhere:
    - process_and_reply() (graph pipeline responses)
    - notify_approval_resolution() (approval decision notifications)

    TTS failure degrades to text-only. Never blocks. Never raises.
    Audio cleanup runs lazily before each synthesis call.
    """
    media_url = None
    try:
        cleanup_old_audio(GENERATED_AUDIO_DIR)
        audio_path = await synthesize(text)
        base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
        if base_url and audio_path:
            media_url = f"{base_url}/audio/{os.path.basename(audio_path)}"
    except Exception:
        media_url = None

    await send_whatsapp_reply(to, text, media_url=media_url)


async def notify_approval_resolution(user_id: str, decision: str) -> None:
    """Send approval-resolution notification with voice.

    Called from decide_approval via BackgroundTasks.
    Uses the same voice-first send_voice_reply() as the main pipeline —
    approval notifications get TTS too, consistent with voice-first design.
    """
    if decision == "approved":
        text = "Aapki darkhwast manzoor ho gayi hai."
    else:
        text = "Aapki darkhwast radd ho gayi hai."

    await send_voice_reply(user_id, text)


async def process_and_reply(message: IncomingMessage, reply_to: str) -> None:
    """Full pipeline: STT → graph → TTS → send reply.

    Called from BackgroundTasks — completely decoupled from Twilio's clock.
    TTS failure degrades to text-only, never blocks the reply.

    Steps:
    1. Transcribe audio if present (STT)
    2. Run the orchestrator graph (classify → agent → risk_gate)
    3. Send text + optional voice via send_voice_reply()
    """
    # 1. STT if audio
    if message.audio_path and not message.text:
        try:
            transcribed = await transcribe(message.audio_path, message.language)
            message = message.model_copy(update={"text": transcribed})
        except Exception:
            pass

    if not message.text:
        await send_whatsapp_reply(reply_to, "Mujhe samajh nahi aaya.")
        return

    # 2. Run graph
    result = await graph.ainvoke({"message": message})
    final = result["final_reply"]

    if final["status"] == "pending_approval":
        reply_text = (
            "Aapki darkhwast approval ke liye bhej di gayi hai. "
            "Jab faisla hoga, hum aapko bataein ge."
        )
    else:
        reply_text = final["reply"]

    # 3. Send with voice
    await send_voice_reply(reply_to, reply_text)
