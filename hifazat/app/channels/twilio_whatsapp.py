"""Twilio WhatsApp channel — parse, validate, download, send.

Four responsibilities:
1. Parse Twilio form-encoded webhook payload → IncomingMessage
2. Validate X-Twilio-Signature (SDK-based, not hand-rolled)
3. Download media with HTTP Basic Auth (Account SID + Auth Token)
4. Send outbound replies via Twilio REST API

Fix #2: download_media wrapped in 10s timeout + try/except.
On failure returns None (caller dispatches with audio_path=None)
instead of risking a 500 to Twilio.

Fix #6: no host-specific logic anywhere. All external config
read from env vars (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, etc).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.graph.state import IncomingMessage

MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 10.0


def build_incoming_message(form: dict) -> IncomingMessage:
    """Parse Twilio form-encoded webhook payload into IncomingMessage.

    Twilio fields:
    - From: "whatsapp:+923001234567"
    - Body: text content (may be empty for voice-only messages)
    - NumMedia: count of attached media
    - MediaUrl0: URL of first media attachment
    - MediaContentType0: MIME type of first attachment
    - MessageSid: Twilio's message identifier
    """
    body = form.get("Body", "").strip()
    return IncomingMessage(
        user_id=form["From"],
        channel="whatsapp",
        text=body if body else None,
        audio_path=None,
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


def has_audio_attachment(form: dict) -> bool:
    """Check if the webhook payload contains an audio attachment."""
    return (
        int(form.get("NumMedia", "0")) > 0
        and form.get("MediaContentType0", "").startswith("audio/")
    )


def validate_signature(request_url: str, params: dict, signature: str) -> bool:
    """Validate X-Twilio-Signature using the SDK RequestValidator.

    Uses Twilio's own HMAC implementation — never hand-rolled.
    Returns True if signature is valid.
    """
    from twilio.request_validator import RequestValidator

    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    validator = RequestValidator(auth_token)
    return validator.validate(request_url, params, signature)


async def download_media(media_url: str) -> Optional[str]:
    """Download Twilio media with HTTP Basic Auth. Returns local file path.

    Fix #2: wrapped in 10s timeout + try/except. On ANY failure
    (network, timeout, disk), returns None instead of raising.
    The caller dispatches with audio_path=None, and the pipeline
    treats it as a text-only message rather than crashing.

    Twilio media URLs require HTTP Basic Auth:
    - Username: TWILIO_ACCOUNT_SID
    - Password: TWILIO_AUTH_TOKEN
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")

    try:
        async with httpx.AsyncClient(
            auth=(account_sid, auth_token),
            timeout=MEDIA_DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
        ) as client:
            response = await asyncio.wait_for(
                client.get(media_url),
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


async def send_whatsapp_reply(
    to: str,
    body: str,
    media_url: Optional[str] = None,
) -> None:
    """Send reply via Twilio REST API.

    Used for both immediate graph replies (from BackgroundTasks)
    and async approval-resolution notifications.

    Twilio's client.messages.create() is synchronous — runs in a thread.
    """
    from twilio.rest import Client

    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_WHATSAPP_NUMBER"]

    client = Client(account_sid, auth_token)

    kwargs = {
        "from_": from_number,
        "to": to,
        "body": body,
    }
    if media_url:
        kwargs["media_url"] = [media_url]

    await asyncio.to_thread(client.messages.create, **kwargs)
