"""MMS-TTS engine for Urdu voice output.

Uses facebook/mms-tts-urd-script_arabic (VITS-based, CPU).
Same shape as whisper_engine.py: lazy-loaded, thread-offloaded,
test-injectable via _synthesize_fn.

Fix #1 verified: _load_model() caches into module-level globals
(_model, _tokenizer) with explicit `global` + `if is None` guard.
Same pattern as whisper_engine._load_model().
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Callable, Optional

_model = None
_tokenizer = None
_synthesize_fn: Optional[Callable] = None

GENERATED_AUDIO_DIR = "generated_audio"


def _load_model():
    """Lazy-load VitsModel + AutoTokenizer. Called once on first TTS request.

    Caches into module-level _model and _tokenizer globals.
    Subsequent calls return the cached instances immediately.
    """
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        model_id = os.environ.get(
            "MMS_TTS_MODEL", "facebook/mms-tts-urd-script_arabic"
        )
        _model = VitsModel.from_pretrained(model_id)
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model.eval()
    return _model, _tokenizer


def _default_synthesize(text: str, language: str) -> str:
    """Synchronous TTS synthesis. Runs in a thread.

    Returns path to the generated .wav file in GENERATED_AUDIO_DIR.
    """
    import scipy.io.wavfile
    import torch

    model, tokenizer = _load_model()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        waveform = model(**inputs).waveform

    os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
    filename = f"{uuid.uuid4()}.wav"
    out_path = os.path.join(GENERATED_AUDIO_DIR, filename)
    scipy.io.wavfile.write(
        out_path,
        rate=model.config.sampling_rate,
        data=waveform.squeeze().numpy(),
    )
    return out_path


async def synthesize(text: str, language: str = "ur") -> str:
    """Async TTS entry point.

    Uses _synthesize_fn if set (for tests), otherwise loads the
    real model and runs in a thread.

    Returns the file path to the generated .wav audio.
    """
    fn = _synthesize_fn or _default_synthesize
    return await asyncio.to_thread(fn, text, language)
