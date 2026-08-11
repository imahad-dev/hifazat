"""Audio file cleanup utility.

Deletes generated audio files older than MAX_AGE_HOURS.
Called lazily on each TTS request — no cron, no separate process.
Acceptable for demo/hackathon scale; replace with a background
scheduler or object-storage TTL policy in production.
"""

from __future__ import annotations

import os
import time

MAX_AGE_HOURS = 2
MAX_AGE_SECONDS = MAX_AGE_HOURS * 3600


def cleanup_old_audio(directory: str) -> int:
    """Delete .wav files older than MAX_AGE_HOURS. Returns count deleted.

    Safe: ignores missing directory, ignores permission errors on
    individual files, never raises.
    """
    if not os.path.isdir(directory):
        return 0

    now = time.time()
    deleted = 0

    for filename in os.listdir(directory):
        if not filename.endswith(".wav"):
            continue
        filepath = os.path.join(directory, filename)
        try:
            if now - os.path.getmtime(filepath) > MAX_AGE_SECONDS:
                os.remove(filepath)
                deleted += 1
        except OSError:
            continue

    return deleted
