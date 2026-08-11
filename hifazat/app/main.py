"""Hifazat FastAPI application entry point.

Day 5: configures LLM providers + mounts static audio for TTS output.
No host-specific logic — all external config read from env vars.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.approval.queue import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB and configure providers on startup."""
    await init_db()

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        from app.llm.provider import GeminiProvider

        provider = GeminiProvider(api_key=api_key)
    else:
        provider = None

    from app.agents import committee_ledger, hisaab_sathi, scam_shield
    from app.graph import nodes

    scam_shield.configure(provider)
    hisaab_sathi.configure(provider)
    committee_ledger.configure(provider)
    nodes.set_provider(provider)

    yield


app = FastAPI(
    title="Hifazat",
    description="Financial inclusion orchestrator — Day 5 WhatsApp + Voice",
    version="0.3.0",
    lifespan=lifespan,
)
app.include_router(router)

os.makedirs("generated_audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")
