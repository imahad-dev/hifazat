"""Hifazat FastAPI application entry point.

Day 6: configures LLM providers, mounts static assets (audio + dashboard).
No host-specific logic — all external config read from env vars.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.approval.queue import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB and configure providers on startup."""
    await init_db()

    provider_name = os.environ.get("LLM_PROVIDER", "groq")

    provider = None
    if provider_name == "groq" and os.environ.get("GROQ_API_KEY"):
        from app.llm.provider import GroqProvider

        provider = GroqProvider(
            api_key=os.environ["GROQ_API_KEY"],
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        )
    elif provider_name == "gemini" and os.environ.get("GEMINI_API_KEY"):
        from app.llm.provider import GeminiProvider

        provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])

    from app.agents import committee_ledger, hisaab_sathi, scam_shield
    from app.graph import nodes

    scam_shield.configure(provider)
    hisaab_sathi.configure(provider)
    committee_ledger.configure(provider)
    nodes.set_provider(provider)

    yield


app = FastAPI(
    title="Hifazat",
    description="Financial inclusion orchestrator — Day 6 Auth + Dashboard",
    version="0.4.0",
    lifespan=lifespan,
)
app.include_router(router)

# Static mounts — order matters (more specific first)
os.makedirs("generated_audio", exist_ok=True)
app.mount("/audio", StaticFiles(directory="generated_audio"), name="audio")

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
