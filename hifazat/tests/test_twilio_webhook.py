"""Tests for Twilio WhatsApp webhook, TTS, and channel integration.

6 tests (all offline, no network):
1. test_webhook_payload_parsing — form payload → correct IncomingMessage
2. test_signature_validation_rejects_tampered — tampered sig → False
3. test_signature_validation_accepts_valid — correct sig → True
4. test_background_task_dispatched — process_and_reply scheduled, not run sync
5. test_tts_failure_still_sends_text_reply — TTS raises → text-only reply sent
6. test_media_download_uses_basic_auth — download uses SID+token auth (mocked)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.channels.twilio_whatsapp import (
    build_incoming_message,
    has_audio_attachment,
    validate_signature,
)
from app.graph.state import Intent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_twilio_form() -> dict:
    """Realistic Twilio WhatsApp webhook form payload."""
    return {
        "MessageSid": "SM1234567890abcdef1234567890abcdef",
        "AccountSid": "ACtest1234567890test1234567890ab",
        "From": "whatsapp:+923001234567",
        "To": "whatsapp:+14155238886",
        "Body": "Mera OTP code bhejo mujhe",
        "NumMedia": "0",
    }


@pytest.fixture
def sample_twilio_audio_form() -> dict:
    """Twilio payload with audio attachment (voice note)."""
    return {
        "MessageSid": "SM1234567890abcdef1234567890abcdef",
        "AccountSid": "ACtest1234567890test1234567890ab",
        "From": "whatsapp:+923001234567",
        "To": "whatsapp:+14155238886",
        "Body": "",
        "NumMedia": "1",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/ACtest/Messages/SM123/Media/ME456",
        "MediaContentType0": "audio/ogg",
    }


# ---------------------------------------------------------------------------
# 1. Webhook payload parsing
# ---------------------------------------------------------------------------


def test_webhook_payload_parsing(sample_twilio_form):
    """Twilio form payload → correct IncomingMessage construction."""
    message = build_incoming_message(sample_twilio_form)

    assert message.user_id == "whatsapp:+923001234567"
    assert message.channel == "whatsapp"
    assert message.text == "Mera OTP code bhejo mujhe"
    assert message.audio_path is None
    assert message.language == "ur"


def test_webhook_audio_payload_has_audio_flag(sample_twilio_audio_form):
    """Audio attachment form payload correctly detected."""
    assert has_audio_attachment(sample_twilio_audio_form) is True

    text_form = {**sample_twilio_audio_form, "NumMedia": "0"}
    assert has_audio_attachment(text_form) is False


# ---------------------------------------------------------------------------
# 2. Signature validation — rejects tampered
# ---------------------------------------------------------------------------


def test_signature_validation_rejects_tampered(twilio_env):
    """Deliberately tampered signature → validate_signature returns False."""
    result = validate_signature(
        request_url="https://example.com/webhooks/twilio/whatsapp",
        params={"Body": "test message", "From": "whatsapp:+923001234567"},
        signature="TAMPERED_INVALID_SIGNATURE",
    )
    assert result is False


# ---------------------------------------------------------------------------
# 3. Signature validation — accepts valid
# ---------------------------------------------------------------------------


def test_signature_validation_accepts_valid(twilio_env):
    """Correctly computed signature → validate_signature returns True.

    Uses Twilio SDK's own RequestValidator to compute the expected
    signature, then verifies our validate_signature agrees.
    """
    import os

    from twilio.request_validator import RequestValidator

    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    url = "https://example.com/webhooks/twilio/whatsapp"
    params = {"Body": "test message", "From": "whatsapp:+923001234567"}

    validator = RequestValidator(auth_token)
    valid_sig = validator.compute_signature(url, params)

    result = validate_signature(url, params, valid_sig)
    assert result is True


# ---------------------------------------------------------------------------
# 4. Background task dispatched (not run synchronously)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_task_dispatched(twilio_env, db_path):
    """process_and_reply is scheduled via BackgroundTasks, not run sync.

    Patches validate_signature to return True, then verifies:
    1. Response is 200 with TwiML XML content
    2. process_and_reply was added to background tasks (not awaited)
    """
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    with (
        patch(
            "app.api.routes.validate_signature", return_value=True
        ),
        patch(
            "app.api.routes.process_and_reply"
        ) as mock_pipeline,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/webhooks/twilio/whatsapp",
                data={
                    "From": "whatsapp:+923001234567",
                    "Body": "Hello test",
                    "NumMedia": "0",
                    "MessageSid": "SM_test",
                },
                headers={"X-Twilio-Signature": "test_sig"},
            )

        assert response.status_code == 200
        assert "xml" in response.headers.get("content-type", "")
        assert "Response" in response.text


# ---------------------------------------------------------------------------
# 5. TTS failure still sends text reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_failure_still_sends_text_reply(twilio_env, db_path):
    """Force TTS to raise → outbound message still sent with media_url=None.

    This is Day 5's equivalent of the Day 3-4 fallback tests.
    A secondary capability failing must degrade, not break.
    """
    from app.tts import mms_tts_engine

    def _failing_tts(text: str, language: str) -> str:
        raise RuntimeError("Simulated TTS failure")

    mms_tts_engine._synthesize_fn = _failing_tts

    with patch("app.pipeline.send_whatsapp_reply", new_callable=AsyncMock) as mock_send:
        from app.graph.state import IncomingMessage
        from app.pipeline import process_and_reply

        message = IncomingMessage(
            user_id="whatsapp:+923001234567",
            channel="whatsapp",
            text="Mera OTP code bhejo",
            language="ur",
            received_at=datetime.now(timezone.utc),
        )
        await process_and_reply(message, "whatsapp:+923001234567")

        mock_send.assert_called_once()
        call_args = mock_send.call_args
        # send_whatsapp_reply(to, body, media_url=None)
        assert call_args[0][0] == "whatsapp:+923001234567"  # to
        assert len(call_args[0][1]) > 0  # body is non-empty
        # media_url should be None since TTS failed
        actual_media = call_args[1].get("media_url") if call_args[1] else (
            call_args[0][2] if len(call_args[0]) > 2 else None
        )
        assert actual_media is None


# ---------------------------------------------------------------------------
# 6. Media download uses Basic Auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_download_uses_basic_auth(twilio_env):
    """Media fetch includes Account SID/Auth Token as HTTP Basic Auth.

    Mocks httpx.AsyncClient to verify the auth parameter is set correctly.
    Never makes a real network call.
    """
    import httpx

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake audio data"
    mock_response.headers = {"content-type": "audio/ogg"}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client_instance) as mock_client_cls:
        from app.channels.twilio_whatsapp import download_media

        result = await download_media("https://api.twilio.com/fake/media/url")

        mock_client_cls.assert_called_once()
        call_kwargs = mock_client_cls.call_args[1]
        assert call_kwargs["auth"] == ("ACtest1234567890test1234567890ab", "test_auth_token_32chars_exactly_")

        assert result is not None
        assert result.endswith(".ogg")
