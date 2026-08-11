"""Integration tests — real Twilio Sandbox + real MMS-TTS.

Excluded from default pytest run via @pytest.mark.integration.
Run manually: pytest -m integration -v

Requires live credentials in environment variables.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_twilio_sandbox_send_message():
    """Smoke test: send one real WhatsApp message via Twilio Sandbox.

    Verifies Account SID, Auth Token, and Sandbox number are valid.
    Sends to a pre-joined test number specified by TEST_WHATSAPP_TO.

    Requires:
    - TWILIO_ACCOUNT_SID (live)
    - TWILIO_AUTH_TOKEN (live)
    - TWILIO_WHATSAPP_NUMBER (Sandbox number)
    - TEST_WHATSAPP_TO (a number already joined to the Sandbox)
    """
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_WHATSAPP_NUMBER")
    to_number = os.environ.get("TEST_WHATSAPP_TO")

    if not all([account_sid, auth_token, from_number, to_number]):
        pytest.skip(
            "Missing Twilio credentials or TEST_WHATSAPP_TO — "
            "skipping Sandbox integration test"
        )

    from app.channels.twilio_whatsapp import send_whatsapp_reply

    await send_whatsapp_reply(
        to=to_number,
        body="Hifazat integration test — if you see this, the Sandbox is working.",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mms_tts_real_synthesis():
    """Smoke test: one real MMS-TTS Urdu synthesis call.

    Verifies the model loads and produces a valid .wav file.
    Downloads ~200MB model on first run — slow, run manually only.

    Requires: torch, transformers, scipy installed.
    """
    try:
        import scipy  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("torch/transformers/scipy not installed — skipping TTS integration test")

    from app.tts.mms_tts_engine import synthesize

    result_path = await synthesize("یہ ایک ٹیسٹ ہے", language="ur")

    assert result_path is not None
    assert result_path.endswith(".wav")
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 100

    os.remove(result_path)
