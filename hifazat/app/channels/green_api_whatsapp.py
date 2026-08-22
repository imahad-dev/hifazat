"""Green API WhatsApp channel — parse, download, send.

Replaces twilio_whatsapp.py. Four responsibilities:
1. Parse Green API JSON webhook payload → IncomingMessage
2. Detect audio attachments via messageData.typeMessage
3. Download media from plain public URL (no auth needed — confirmed via test)
4. Send outbound replies via Green API SDK (sync, wrapped in asyncio.to_thread)

Idempotency: in-memory bounded set of recently-seen idMessage values.
If Green API retries a webhook (slow ack, network blip), duplicate
deliveries are silently dropped instead of reprocessed. This prevents
duplicate approval queue items or double-sent replies.

Format bridge:
- Internal user_id format: "whatsapp:+923001234567" (provider-agnostic)
- Green API chatId format:  "923001234567@c.us"
- Conversion happens at the channel boundary only — nothing downstream
  of this module knows or cares which provider is active.

The SDK's Python client is confirmed synchronous (v0.0.54).
All sending calls use asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.graph.state import IncomingMessage

MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 10.0
_MAX_SEEN_MESSAGES = 1000

# Bounded dedup set — OrderedDict used as an ordered set (values are None).
# FIFO eviction when size exceeds _MAX_SEEN_MESSAGES.
_seen_messages: OrderedDict[str, None] = OrderedDict()


def _to_chat_id(whatsapp_id: str) -> str:
    """Convert internal 'whatsapp:+923...' format to Green API '923...@c.us'."""
    phone = whatsapp_id.removeprefix("whatsapp:").lstrip("+")
    return f"{phone}@c.us"


def _from_chat_id(chat_id: str) -> str:
    """Convert Green API '923...@c.us' format to internal 'whatsapp:+923...'."""
    phone = chat_id.removesuffix("@c.us")
    return f"whatsapp:+{phone}"


def is_duplicate(id_message: str) -> bool:
    """Check if this message ID was already seen. Thread-safe at single-worker scale.

    Returns True if duplicate (caller should silently drop).
    Returns False on first sight (registers the ID for future checks).
    """
    if id_message in _seen_messages:
        return True
    _seen_messages[id_message] = None
    if len(_seen_messages) > _MAX_SEEN_MESSAGES:
        _seen_messages.popitem(last=False)
    return False


def build_incoming_message(body: dict) -> IncomingMessage:
    """Parse Green API JSON webhook body into IncomingMessage.

    Extracts sender from senderData.sender (e.g. '923248817336@c.us')
    and converts to internal 'whatsapp:+923248817336' format.

    Text extracted from messageData.textMessageData.textMessage
    for textMessage types. Audio handled separately via download_media.
    """
    sender = body.get("senderData", {}).get("sender", "")
    user_id = _from_chat_id(sender)

    message_data = body.get("messageData", {})
    type_message = message_data.get("typeMessage", "")

    text = None
    if type_message == "textMessage":
        text_data = message_data.get("textMessageData", {})
        text = text_data.get("textMessage", "").strip() or None

    return IncomingMessage(
        user_id=user_id,
        channel="whatsapp",
        text=text,
        audio_path=None,
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


def has_audio_attachment(body: dict) -> bool:
    """Check if the webhook body contains an audio message."""
    message_data = body.get("messageData", {})
    return message_data.get("typeMessage") == "audioMessage"


def get_audio_download_url(body: dict) -> Optional[str]:
    """Extract the downloadUrl from an audio message payload.

    Returns None if the field is missing or empty.
    """
    file_data = body.get("messageData", {}).get("fileMessageData", {})
    url = file_data.get("downloadUrl", "")
    return url if url else None


async def download_media(download_url: str) -> Optional[str]:
    """Download audio from Green API's media URL. Returns local file path.

    Green API's downloadUrl is a plain public DigitalOcean Spaces URL —
    confirmed via live test. No auth header needed (unlike Twilio's
    Basic Auth requirement).

    Wrapped in 10s timeout + try/except. On ANY failure (network,
    timeout, disk), returns None. The caller dispatches with
    audio_path=None, and the pipeline treats it as a text-only
    message rather than crashing.
    """
    try:
        async with httpx.AsyncClient(
            timeout=MEDIA_DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            response = await asyncio.wait_for(
                client.get(download_url),
                timeout=MEDIA_DOWNLOAD_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            suffix = ".ogg"
            content_type = response.headers.get("content-type", "")
            if "wav" in content_type:
                suffix = ".wav"
            elif "mpeg" in content_type or "mp3" in content_type:
                suffix = ".mp3"

            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix="hifazat_media_"
            )
            tmp.write(response.content)
            tmp.close()
            return tmp.name

    except Exception:
        return None


def _get_green_api_client():
    """Lazy-init the Green API client from env vars.

    Not cached at module level — env vars may change between tests.
    The SDK client is lightweight (no persistent connection),
    so re-creating per call is acceptable.
    """
    from whatsapp_api_client_python import API

    id_instance = os.environ.get("GREEN_API_ID_INSTANCE", "")
    api_token = os.environ.get("GREEN_API_TOKEN_INSTANCE", "")
    return API.GreenAPI(id_instance, api_token)


async def send_whatsapp_reply(
    to: str,
    body: str,
    media_url: Optional[str] = None,
) -> None:
    """Send reply via Green API SDK.

    Used for both immediate graph replies (from BackgroundTasks)
    and async approval-resolution notifications.

    The SDK's Python client is synchronous (confirmed v0.0.54) —
    every call runs in asyncio.to_thread() to avoid blocking
    the event loop. Same pattern as Whisper and MMS-TTS.

    Converts internal 'whatsapp:+923...' format to Green API
    '923...@c.us' chatId at the channel boundary.
    """
    green_api = _get_green_api_client()
    chat_id = _to_chat_id(to)

    if media_url:
        await asyncio.to_thread(
            green_api.sending.sendFileByUrl,
            chat_id,
            media_url,
            "voice_reply.wav",
            body,
        )
    else:
        await asyncio.to_thread(
            green_api.sending.sendMessage,
            chat_id,
            body,
        )
