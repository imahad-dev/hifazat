# Hifazat — حفاظت

> Financial inclusion orchestrator for Pakistan's unbanked population.
> Alibaba Cloud AI Hackathon 2026 (Bano Qabil / Alkhidmat Foundation)

Hifazat is a WhatsApp-based AI assistant that helps Pakistani daily-wage earners with:

- **🛡️ Scam Shield** — Detects OTP scams, fake bank calls, and prize fraud in Urdu/Punjabi voice messages
- **📊 Hisaab Sathi** — Simple budgeting advice tailored for irregular income (50-30-20 rule)
- **📒 Committee Ledger** — Records bisi/chit-fund contributions with mandatory human approval

## Architecture

```
WhatsApp → Green API Webhook → FastAPI (fast-ack <500ms)
                                   ↓ BackgroundTasks
                              Whisper STT (if voice)
                                   ↓
                           Hybrid Intent Classifier
                        (keyword fast-path → Gemini LLM fallback)
                                   ↓
                       Agent (Gemini structured output → canned fallback)
                                   ↓
                              Risk Gate
                           ↙          ↘
                      READ              WRITE
                 (reply now)     (approval queue → human review)
                                   ↓
                             MMS-TTS (Urdu voice reply, optional)
                                   ↓
                          Green API SDK → WhatsApp
```

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestrator | LangGraph state graph |
| API | FastAPI + uvicorn |
| LLM | Gemini 2.5 Flash (google-genai SDK) |
| STT | faster-whisper (Whisper small, CPU) |
| TTS | MMS-TTS (facebook/mms-tts-urd-script_arabic) |
| Channel | Green API (QR-linked WhatsApp) |
| Data | aiosqlite (approval queue) |
| Auth | Session-based (opaque token, in-memory) |
| Schemas | Pydantic v2 |

## Quick Start

```bash
# Clone and install
git clone https://github.com/imahad-dev/hifazat.git
cd hifazat/hifazat
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (Green API, Gemini, etc.)

# Run
uvicorn app.main:app --reload

# Test
pytest tests/ -v
```

## Project Status

| Phase | Status | Tests |
|---|---|---|
| Day 1-2: Skeleton | ✅ Complete | 11 tests |
| Day 3-4: LLM + STT | ✅ Complete | +10 tests |
| Day 5: WhatsApp + TTS | ✅ Complete | +7 tests |
| Day 6: Auth + Dashboard | ✅ Complete | +10 tests |
| Day 6b: Green API Swap | ✅ Complete | 8 tests (rewrote) |
| Day 7: Fraud fine-tuning | ⏳ Pending | — |
| Day 8: Reliability drill | ⏳ Pending | — |

**Current:** 39 passed, 3 deselected (integration), 0.89s

## Design Principles

1. **Fallback discipline** — Every external call (LLM, STT, TTS) has a timeout and falls back to a canned response. A secondary capability failing degrades the experience, never breaks it.
2. **Keyword before LLM** — Deterministic keyword matching runs first (zero cost). LLM only fires for ambiguous messages.
3. **Human-in-the-loop** — Write-tier actions (committee ledger, high-confidence scam flags) always go through human approval. No autonomous financial actions.
4. **Fast webhook ack** — Provider-agnostic fast-ack architecture. Zero synchronous work in the webhook handler. Full pipeline runs in BackgroundTasks.
5. **Idempotency** — Duplicate webhook deliveries are silently dropped via a bounded in-memory dedup set. Prevents double-processing on retries.

## License

MIT
