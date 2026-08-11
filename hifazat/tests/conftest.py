"""Shared test fixtures for Hifazat.

Key design decisions:
- Function-scoped DB fixture: init_db() runs fresh before EVERY test
- Each test gets its own temp SQLite file via tmp_path
- Autouse provider reset: all 4 module-level provider globals reset
  to None before AND after every test — prevents state leaking
- Autouse TTS reset: _synthesize_fn reset to None between tests
- No shared state between tests, no cleanup needed
- TestClient uses the real FastAPI app with overridden DB path
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.approval import queue as approval_queue
from app.graph.state import IncomingMessage
from app.main import app


@pytest.fixture(autouse=True)
def _reset_providers():
    """Reset all module-level provider globals before AND after every test.

    Prevents provider state from leaking between tests. Covers:
    - scam_shield._provider
    - hisaab_sathi._provider
    - committee_ledger._provider
    - nodes._provider (hybrid classification)
    """
    from app.agents import committee_ledger, hisaab_sathi, scam_shield
    from app.graph import nodes

    scam_shield._provider = None
    hisaab_sathi._provider = None
    committee_ledger._provider = None
    nodes._provider = None

    yield

    scam_shield._provider = None
    hisaab_sathi._provider = None
    committee_ledger._provider = None
    nodes._provider = None


@pytest.fixture(autouse=True)
def _reset_tts():
    """Reset TTS engine _synthesize_fn before AND after every test.

    Prevents a test that stubs TTS from leaking the stub into the next test.
    """
    from app.tts import mms_tts_engine

    mms_tts_engine._synthesize_fn = None
    yield
    mms_tts_engine._synthesize_fn = None


@pytest_asyncio.fixture
async def db_path(tmp_path):
    """Function-scoped: create a fresh SQLite DB for each test."""
    path = str(tmp_path / "test_approvals.db")
    await approval_queue.init_db(path)
    return path


@pytest.fixture
def sample_scam_message() -> IncomingMessage:
    return IncomingMessage(
        user_id="user_001",
        channel="whatsapp",
        text="Mera OTP code bhejo",
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_budget_message() -> IncomingMessage:
    return IncomingMessage(
        user_id="user_002",
        channel="whatsapp",
        text="Kitna bachaon is mahine?",
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_ledger_message() -> IncomingMessage:
    return IncomingMessage(
        user_id="user_003",
        channel="whatsapp",
        text="Committee ka hissa jama karo",
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_unknown_message() -> IncomingMessage:
    return IncomingMessage(
        user_id="user_004",
        channel="whatsapp",
        text="Hello ji kya haal hai",
        language="ur",
        received_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def fake_provider():
    """FakeProvider with default deterministic responses."""
    from app.llm.provider import FakeProvider

    return FakeProvider()


@pytest.fixture
def failing_provider():
    """Provider that raises TimeoutError on every call."""
    from app.llm.provider import FailingProvider

    return FailingProvider()


@pytest.fixture
def twilio_env(monkeypatch):
    """Set Twilio env vars for webhook testing.

    Uses test-only values — no real Twilio credentials.
    """
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtest1234567890test1234567890ab")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test_auth_token_32chars_exactly_")
    monkeypatch.setenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://test.example.com")


@pytest_asyncio.fixture
async def async_client(db_path):
    """Function-scoped async test client with fresh DB.

    Uses httpx AsyncClient with ASGITransport for async endpoint testing.
    The db_path fixture ensures init_db() runs before the client is created.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
