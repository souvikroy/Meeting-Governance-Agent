"""Typed domain model. The LLM reports *evidence* (per-policy matches); the
orchestrator's precedence resolver derives the *action*. The model never emits
the final action directly — see resolver.py."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    COMMIT = "COMMIT"                  # persist as-is
    DROP = "DROP"                      # no copy, no trace
    REDACT = "REDACT"                  # persist with sensitive value masked
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"  # persist + legal-review marker
    CONSENT_GATE = "CONSENT_GATE"     # non-consented speaker: never committed


class Speaker(BaseModel):
    name: str
    role: str = ""
    side: str = ""
    consent: bool


class Utterance(BaseModel):
    """One governed unit. `text` is populated only after STT; for non-consented
    speakers it stays empty because we never transcribe them (ephemerality)."""
    id: str
    beat: int
    speaker: str
    consent: bool
    audio_file: str
    text: str = ""


# --- Structured verdict (what the LLM returns) ----------------------------
class PolicyMatch(BaseModel):
    policy_id: int = Field(ge=1, le=4)   # consent (5) is enforced in code, pre-LLM
    matched: bool
    rationale: str = ""
    span: Optional[str] = None           # offending substring, for targeting


class Redaction(BaseModel):
    value: str                            # literal substring to mask out
    reason: str = ""


class Verdict(BaseModel):
    policy_matches: list[PolicyMatch] = Field(default_factory=list)
    redactions: list[Redaction] = Field(default_factory=list)
    confidence: float = 0.0
    overall_reasoning: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def matched_ids(self) -> set[int]:
        return {m.policy_id for m in self.policy_matches if m.matched}


# --- Decision record (what the orchestrator produces) ---------------------
class Decision(BaseModel):
    """The outcome for one utterance. NOTE: serializers in store.py decide what
    is safe to persist — this object may transiently hold sensitive text in
    memory, but DROP/CONSENT_GATE content is never written to any artifact."""
    utterance_id: str
    beat: int
    speaker: str
    action: Action
    policy_ids: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    unsure: bool = False
    refine_iterations: int = 0
    # `committed_text` is the (possibly redacted) text actually persisted.
    # Empty for DROP / CONSENT_GATE.
    committed_text: str = ""
    # `reason` is a generic, non-leaking explanation safe for logs.
    reason: str = ""
