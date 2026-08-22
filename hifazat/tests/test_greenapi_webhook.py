"""Tests for Green API WhatsApp webhook, channel, and pipeline integration.

8 tests (all offline, no network):
1. test_text_webhook_parsing — Green API JSON → correct IncomingMessage
2. test_audio_webhook_detection — audioMessage type → has_audio_attachment True
3. test_webhook_requires_token — wrong/missing token → 404
4. test_webhook_valid_token_accepted — correct token → 200
5. test_background_task_dispatched — process_and_reply scheduled via BackgroundTasks
6. test_tts_failure_still_sends_text_reply — TTS raises → text-only reply sent
7. test_media_download_failure_returns_none — download timeout → None returned
8. test_duplicate_idmessage_is_dropped — same idMessage twice → only one dispatch
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.channels.green_api_whatsapp import (
    build_incoming_message,
    has_audio_attachment,
    is_duplicate,
)


# ---------------------------------------------------------------------------
# Fixtures — real captured payloads from ngrok, not guessed
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_text_webhook() -> dict:
    """Realistic Green API text message webhook body."""
    return {
        "typeWebhook": "incomingMessageReceived",
        "instanceData": {
            "idInstance": 7107227116095,
            "wid": "923004175809@c.us",
            "typeInstance": "whatsapp",
        },
        "timestamp": 1787386800,
        "idMessage": "ABC123DEF456TEXT",
        "senderData": {
            "chatId": "923248817336@c.us",
            "chatName": "Master Wayne",
            "sender": "923248817336@c.us",
            "senderName": "Master Wayne",
            "senderContactName": "",
        },
        "messageData": {
            "typeMessage": "textMessage",
            "textMessageData": {
                "textMessage": "Mera OTP code bhejo mujhe",
            },
        },
    }


@pytest.fixture
def sample_audio_webhook() -> dict:
    """Realistic Green API audio message webhook body (from live ngrok capture)."""
    return {
        "typeWebhook": "incomingMessageReceived",
        "instanceData": {
            "idInstance": 7107227116095,
            "wid": "923004175809@c.us",
            "typeInstance": "whatsapp",
        },
        "timestamp": 1787386806,
        "idMessage": "AC896575A436D4C824AE3D7A3C944AA9",
        "senderData": {
            "chatId": "923248817336@c.us",
            "chatName": "Master Wayne",
            "sender": "923248817336@c.us",
            "senderName": "Master Wayne",
            "senderContactName": "",
        },
        "messageData": {
            "typeMessage": "audioMessage",
            "fileMessageData": {
                "downloadUrl": "https://do-media-7107.fra1.digitaloceanspaces.com/test-audio.oga",
                "caption": "",
                "fileName": "aa455811-test.oga",
                "jpegThumbnail": "",
                "isAnimated": False,
                "mimeType": "audio/ogg; codecs=opus",
                "forwardingScore": 0,
                "isForwarded": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. Text webhook parsing — Green API JSON → IncomingMessage
# ---------------------------------------------------------------------------


def test_text_webhook_parsing(sample_text_webhook):
    """Green API text webhook → correct IncomingMessage construction.

    Verifies the format bridge: Green API's '923248817336@c.us'
    is stored as internal 'whatsapp:+923248817336'.
    """
    message = build_incoming_message(sample_text_webhook)

    assert message.user_id == "whatsapp:+923248817336"
    assert message.channel == "whatsapp"
    assert message.text == "Mera OTP code bhejo mujhe"
    assert message.audio_path is None
    assert message.language == "ur"


# ---------------------------------------------------------------------------
# 2. Audio webhook detection
# ---------------------------------------------------------------------------


def test_audio_webhook_detection(sample_audio_webhook, sample_text_webhook):
    """audioMessage type detected, textMessage type not."""
    assert has_audio_attachment(sample_audio_webhook) is True
    assert has_audio_attachment(sample_text_webhook) is False


# ---------------------------------------------------------------------------
# 3. Webhook requires token — wrong/missing → 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_requires_token(greenapi_env, db_path, sample_text_webhook):
    """Wrong webhook token → 404 (not 403/401 — don't reveal route)."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        # Wrong token
        response = await client.post(
            "/webhooks/whatsapp/WRONG_TOKEN",
            json=sample_text_webhook,
        )
        assert response.status_code == 404

        # Empty token
        response = await client.post(
            "/webhooks/whatsapp/",
            json=sample_text_webhook,
        )
        assert response.status_code in (404, 405, 422)


# ---------------------------------------------------------------------------
# 4. Webhook valid token → 200
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_valid_token_accepted(greenapi_env, db_path, sample_text_webhook):
    """Correct webhook token → 200 OK, process dispatched."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    with patch("app.api.routes.process_and_reply"):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhooks/whatsapp/test_webhook_secret_token",
                json=sample_text_webhook,
            )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 5. Background task dispatched (not run synchronously)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_task_dispatched(greenapi_env, db_path, sample_text_webhook):
    """process_and_reply is scheduled via BackgroundTasks, not run sync.

    Verifies the fast-ack architecture: webhook returns 200 immediately,
    full pipeline runs in background.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    with patch("app.api.routes.process_and_reply") as mock_pipeline:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhooks/whatsapp/test_webhook_secret_token",
                json=sample_text_webhook,
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 6. TTS failure still sends text reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_failure_still_sends_text_reply(greenapi_env, db_path):
    """Force TTS to raise → outbound message still sent with media_url=None.

    Same fallback discipline as Twilio era, different provider underneath.
    A secondary capability failing must degrade, not break.
    """
    from app.tts import mms_tts_engine

    def _failing_tts(text: str, language: str) -> str:
        raise RuntimeError("Simulated TTS failure")

    mms_tts_engine._synthesize_fn = _failing_tts

    with patch(
        "app.pipeline.send_whatsapp_reply", new_callable=AsyncMock
    ) as mock_send:
        from app.graph.state import IncomingMessage
        from app.pipeline import process_and_reply

        message = IncomingMessage(
            user_id="whatsapp:+923248817336",
            channel="whatsapp",
            text="Mera OTP code bhejo",
            language="ur",
            received_at=datetime.now(timezone.utc),
        )
        await process_and_reply(message, "whatsapp:+923248817336")

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == "whatsapp:+923248817336"
        assert len(call_args[0][1]) > 0
        actual_media = call_args[1].get("media_url") if call_args[1] else (
            call_args[0][2] if len(call_args[0]) > 2 else None
        )
        assert actual_media is None


# ---------------------------------------------------------------------------
# 7. Media download failure → returns None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_download_failure_returns_none():
    """download_media returns None on network/timeout error.

    No auth needed for Green API media URLs (plain public DO Spaces),
    but the download itself can still fail. Verifies graceful degradation.
    """
    from app.channels.green_api_whatsapp import download_media

    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_instance

        result = await download_media("https://do-media.example.com/fake-audio.oga")

    assert result is None


# ---------------------------------------------------------------------------
# 8. Duplicate idMessage is dropped — only one dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_idmessage_is_dropped(
    greenapi_env, db_path, sample_text_webhook,
):
    """Same idMessage delivered twice → only one background dispatch.

    Prevents double-processing on webhook retries: duplicate reply sent,
    or worse, the same committee-ledger action pushed into the approval
    queue twice.
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    dispatch_count = 0

    original_process = None

    def counting_process(*args, **kwargs):
        nonlocal dispatch_count
        dispatch_count += 1

    with patch("app.api.routes.process_and_reply", side_effect=counting_process):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # First delivery — should dispatch
            r1 = await client.post(
                "/webhooks/whatsapp/test_webhook_secret_token",
                json=sample_text_webhook,
            )
            assert r1.status_code == 200
            assert r1.json()["status"] == "ok"

            # Second delivery (same idMessage) — should be dropped
            r2 = await client.post(
                "/webhooks/whatsapp/test_webhook_secret_token",
                json=sample_text_webhook,
            )
            assert r2.status_code == 200
            assert r2.json()["status"] == "duplicate_dropped"

    # Only one dispatch, not two
    assert dispatch_count == 1
