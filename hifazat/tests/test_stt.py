"""Tests for STT pre-processing across both channel types.

2 tests:
- channel='voice' + audio_path → transcribes and classifies correctly
- channel='whatsapp' + audio_path → same behavior (primary demo scenario)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.graph.graph import graph
from app.graph.state import IncomingMessage, Intent
from app.stt import whisper_engine


@pytest.mark.asyncio
async def test_voice_message_transcribes_before_classification(db_path):
    """Stub transcribe() → 'OTP bhejo' → scam_shield agent triggered.

    Verifies the full pipeline:
    1. Voice message arrives with audio_path, no text
    2. STT stub transcribes to 'OTP bhejo'
    3. routes.py populates text before graph.ainvoke()
    4. Graph classifies as SCAM_CHECK, routes to scam_shield
    """
    original_fn = whisper_engine._transcribe_fn

    def _stub_transcribe(audio_path: str, language: str) -> str:
        assert audio_path == "/tmp/test_audio.wav"
        return "Mera OTP code bhejo"

    whisper_engine._transcribe_fn = _stub_transcribe

    try:
        message = IncomingMessage(
            user_id="user_voice",
            channel="voice",
            text=None,
            audio_path="/tmp/test_audio.wav",
            language="ur",
            received_at=datetime.now(timezone.utc),
        )

        from app.stt.whisper_engine import transcribe

        transcribed_text = await transcribe(message.audio_path, message.language)

        populated_message = message.model_copy(update={"text": transcribed_text})

        state = {"message": populated_message}
        result = await graph.ainvoke(state)

        assert result["intent"] == Intent.SCAM_CHECK
        assert result["agent_response"].agent == "scam_shield"
    finally:
        whisper_engine._transcribe_fn = original_fn


@pytest.mark.asyncio
async def test_whatsapp_voice_note_transcribes_correctly(db_path):
    """WhatsApp voice note: channel='whatsapp' + audio_path set, no text.

    This is the primary demo scenario. The STT trigger must fire on
    audio_path presence regardless of channel — WhatsApp voice notes
    arrive with channel='whatsapp', not channel='voice'.

    Verifies the fix from review item #4: the old condition
    `channel == 'voice' and audio_path` would silently skip this case.
    """
    original_fn = whisper_engine._transcribe_fn

    def _stub_transcribe(audio_path: str, language: str) -> str:
        assert audio_path == "/tmp/whatsapp_voice_note.ogg"
        return "Committee ka hissa jama karo"

    whisper_engine._transcribe_fn = _stub_transcribe

    try:
        message = IncomingMessage(
            user_id="user_whatsapp_voice",
            channel="whatsapp",
            text=None,
            audio_path="/tmp/whatsapp_voice_note.ogg",
            language="ur",
            received_at=datetime.now(timezone.utc),
        )

        from app.stt.whisper_engine import transcribe

        transcribed_text = await transcribe(message.audio_path, message.language)

        populated_message = message.model_copy(update={"text": transcribed_text})

        state = {"message": populated_message}
        result = await graph.ainvoke(state)

        assert result["intent"] == Intent.LEDGER_ACTION
        assert result["agent_response"].agent == "committee_ledger"
        assert result["final_reply"]["status"] == "pending_approval"
    finally:
        whisper_engine._transcribe_fn = original_fn
