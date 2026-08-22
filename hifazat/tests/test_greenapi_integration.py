"""Integration tests — real Green API + real MMS-TTS.

Excluded from default pytest run via @pytest.mark.integration.
Run manually: pytest -m integration -v

Requires live credentials in environment variables.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_greenapi_send_message():
    """Smoke test: send one real WhatsApp message via Green API.

    Verifies idInstance, apiTokenInstance are valid and QR-linked.
    Sends to a test number specified by TEST_WHATSAPP_TO.

    Requires:
    - GREEN_API_ID_INSTANCE (live)
    - GREEN_API_TOKEN_INSTANCE (live)
    - TEST_WHATSAPP_TO (e.g. 'whatsapp:+923001234567')
    """
    id_instance = os.environ.get("GREEN_API_ID_INSTANCE")
    api_token = os.environ.get("GREEN_API_TOKEN_INSTANCE")
    to_number = os.environ.get("TEST_WHATSAPP_TO")

    if not all([id_instance, api_token, to_number]):
        pytest.skip(
            "Missing Green API credentials or TEST_WHATSAPP_TO — "
            "skipping integration test"
        )

    from app.channels.green_api_whatsapp import send_whatsapp_reply

    await send_whatsapp_reply(
        to=to_number,
        body="Hifazat integration test — if you see this, Green API is working.",
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
