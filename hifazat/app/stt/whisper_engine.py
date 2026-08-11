"""Whisper-based speech-to-text engine using faster-whisper.

Lazy-loads the model on first voice request — not at app startup.
Runs transcription in a thread via asyncio.to_thread() to avoid
blocking the event loop.

Tests swap _transcribe_fn to avoid loading the real 460MB model.
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable, Optional

_model = None
_transcribe_fn: Optional[Callable] = None


def _load_model():
    """Lazy-load the WhisperModel. Called once on first voice request."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        model_size = os.environ.get("WHISPER_MODEL_SIZE", "small")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def _default_transcribe(audio_path: str, language: str) -> str:
    """Synchronous transcription using faster-whisper. Runs in a thread."""
    model = _load_model()
    segments, _info = model.transcribe(audio_path, language=language)
    return " ".join(segment.text.strip() for segment in segments)


async def transcribe(audio_path: str, language: str = "ur") -> str:
    """Async transcription entry point.

    Uses _transcribe_fn if set (for tests), otherwise loads the
    real model and runs in a thread.
    """
    fn = _transcribe_fn or _default_transcribe
    return await asyncio.to_thread(fn, audio_path, language)
