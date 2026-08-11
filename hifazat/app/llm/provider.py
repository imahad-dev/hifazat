"""LLM provider abstraction for Hifazat.

Protocol + three implementations:
- GeminiProvider: real Gemini API calls via google-genai SDK
- FakeProvider: deterministic canned values for tests
- FailingProvider: raises TimeoutError on every call for fallback tests

Design: providers are dumb and honest. On failure they raise — the
caller (agent node) decides the fallback. No silent error swallowing.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.graph.state import Intent


# ---------------------------------------------------------------------------
# Response schemas — structured output from LLM
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    intent: Intent = Field(description="One of: scam_check, budget_query, ledger_action, unknown")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")


class ScamAnalysis(BaseModel):
    is_scam: bool = Field(description="Whether the message appears to be a scam")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    urdu_explanation: str = Field(description="Plain Urdu explanation for the user")


class BudgetAdvice(BaseModel):
    urdu_advice: str = Field(description="Budgeting advice in plain Urdu")
    suggested_saving_pct: int = Field(ge=0, le=100, description="Suggested saving percentage")


class LedgerDraft(BaseModel):
    action_type: str = Field(description="e.g. record_contribution, record_payout")
    amount: Optional[float] = Field(default=None, description="Amount if mentioned")
    member_reference: Optional[str] = Field(default=None, description="Member name/ID if mentioned")
    urdu_confirmation_draft: str = Field(description="Draft confirmation in Urdu for the user")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    async def classify(self, text: str) -> ClassificationResult: ...
    async def analyze_scam(self, text: str) -> ScamAnalysis: ...
    async def budget_advice(self, text: str) -> BudgetAdvice: ...
    async def draft_ledger_action(self, text: str) -> LedgerDraft: ...


# ---------------------------------------------------------------------------
# GeminiProvider — real API calls
# ---------------------------------------------------------------------------

_TIMEOUT_SECONDS = 8.0


class GeminiProvider:
    """Real Gemini provider using google-genai SDK.

    Uses client.models.generate_content() with structured output.
    Every method has an 8-second timeout. On failure: raises, never
    falls back — the caller decides the fallback.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
    ) -> None:
        from google import genai

        self._model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=api_key)

    async def _generate(self, prompt: str, schema: type[BaseModel], system_prompt: str) -> BaseModel:
        """Shared generate_content call with timeout."""
        from app.llm.prompts import get_system_prompt

        async def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "system_instruction": system_prompt,
                },
            )
            return response.parsed

        return await asyncio.wait_for(_call(), timeout=_TIMEOUT_SECONDS)

    async def classify(self, text: str) -> ClassificationResult:
        from app.llm.prompts import CLASSIFY_SYSTEM_PROMPT

        return await self._generate(text, ClassificationResult, CLASSIFY_SYSTEM_PROMPT)

    async def analyze_scam(self, text: str) -> ScamAnalysis:
        from app.llm.prompts import SCAM_SHIELD_SYSTEM_PROMPT

        return await self._generate(text, ScamAnalysis, SCAM_SHIELD_SYSTEM_PROMPT)

    async def budget_advice(self, text: str) -> BudgetAdvice:
        from app.llm.prompts import HISAAB_SATHI_SYSTEM_PROMPT

        return await self._generate(text, BudgetAdvice, HISAAB_SATHI_SYSTEM_PROMPT)

    async def draft_ledger_action(self, text: str) -> LedgerDraft:
        from app.llm.prompts import COMMITTEE_LEDGER_SYSTEM_PROMPT

        return await self._generate(text, LedgerDraft, COMMITTEE_LEDGER_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# FakeProvider — deterministic test doubles
# ---------------------------------------------------------------------------


class FakeProvider:
    """Deterministic canned values for testing. No network calls.

    Configurable via constructor args for scam analysis confidence tests.
    """

    def __init__(
        self,
        scam_confidence: float = 0.9,
        scam_is_scam: bool = True,
    ) -> None:
        self.scam_confidence = scam_confidence
        self.scam_is_scam = scam_is_scam
        self.classify_call_count = 0
        self.analyze_scam_call_count = 0
        self.budget_advice_call_count = 0
        self.draft_ledger_call_count = 0

    async def classify(self, text: str) -> ClassificationResult:
        self.classify_call_count += 1
        return ClassificationResult(intent=Intent.UNKNOWN, confidence=0.95)

    async def analyze_scam(self, text: str) -> ScamAnalysis:
        self.analyze_scam_call_count += 1
        return ScamAnalysis(
            is_scam=self.scam_is_scam,
            confidence=self.scam_confidence,
            urdu_explanation="Yeh message mashkook hai. Apna OTP kisi ko mat dijiye.",
        )

    async def budget_advice(self, text: str) -> BudgetAdvice:
        self.budget_advice_call_count += 1
        return BudgetAdvice(
            urdu_advice="Apni income ka 50% zarooriyat, 30% khwahishat, 20% bachat mein rakhein.",
            suggested_saving_pct=20,
        )

    async def draft_ledger_action(self, text: str) -> LedgerDraft:
        self.draft_ledger_call_count += 1
        return LedgerDraft(
            action_type="record_contribution",
            amount=1000.0,
            member_reference="user_test",
            urdu_confirmation_draft="Committee mein 1000 rupay ka hissa darj kiya jaayega.",
        )


class FailingProvider:
    """Every method raises TimeoutError. Tests fallback paths."""

    async def classify(self, text: str) -> ClassificationResult:
        raise TimeoutError("Simulated LLM timeout")

    async def analyze_scam(self, text: str) -> ScamAnalysis:
        raise TimeoutError("Simulated LLM timeout")

    async def budget_advice(self, text: str) -> BudgetAdvice:
        raise TimeoutError("Simulated LLM timeout")

    async def draft_ledger_action(self, text: str) -> LedgerDraft:
        raise TimeoutError("Simulated LLM timeout")
