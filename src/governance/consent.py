"""Consent gate — Policy 5. A hard, code-level invariant applied BEFORE any
transcription or inference. Non-consent always wins; the LLM never sees, and
Deepgram never transcribes, a non-consented speaker's audio."""
from __future__ import annotations

from .models import Utterance


def is_gated(utt: Utterance) -> bool:
    """True if this utterance must be declined at the consent gate."""
    return not utt.consent
