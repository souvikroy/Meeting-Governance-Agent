"""Rolling trailing window — the bounded, online context the agent reasons over.

Invariants:
  * Trailing only. Entries are added AFTER a decision; the agent can never see
    utterance N+1 while deciding utterance N (no look-ahead).
  * Holds committed/sanitized text only. On DROP / CONSENT_GATE the raw text is
    NEVER stored — a content-free placeholder is appended instead, so dropped
    content cannot linger in the buffer or leak into a later prompt.
  * Bounded by count (WINDOW_SIZE) and by tokens (WINDOW_TOKEN_CAP)."""
from __future__ import annotations

from dataclasses import dataclass

from . import config

_PLACEHOLDER = "[utterance withheld]"

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def _ntok(s: str) -> int:
        return len(_enc.encode(s))
except Exception:  # pragma: no cover - fallback if tiktoken unavailable
    def _ntok(s: str) -> int:
        return max(1, len(s) // 4)


@dataclass
class WindowEntry:
    utterance_id: str
    speaker: str
    text: str          # committed/sanitized text, or the content-free placeholder
    withheld: bool


class RollingWindow:
    def __init__(self, size: int | None = None, token_cap: int | None = None):
        self.size = size or config.WINDOW_SIZE
        self.token_cap = token_cap or config.WINDOW_TOKEN_CAP
        self._entries: list[WindowEntry] = []

    def add_committed(self, utterance_id: str, speaker: str, text: str) -> None:
        self._entries.append(WindowEntry(utterance_id, speaker, text, withheld=False))
        self._trim()

    def add_withheld(self, utterance_id: str, speaker: str) -> None:
        """Append a content-free placeholder for a DROP / CONSENT_GATE utterance.
        The placeholder carries NO topic, only that something was withheld — this
        preserves conversational continuity without leaking what was removed."""
        self._entries.append(WindowEntry(utterance_id, speaker, _PLACEHOLDER, withheld=True))
        self._trim()

    def _trim(self) -> None:
        if len(self._entries) > self.size:
            self._entries = self._entries[-self.size :]
        # Token cap: drop oldest until under cap.
        while self._entries and self._total_tokens() > self.token_cap:
            self._entries.pop(0)

    def _total_tokens(self) -> int:
        return sum(_ntok(f"{e.speaker}: {e.text}") for e in self._entries)

    def render(self) -> str:
        if not self._entries:
            return "(no prior context)"
        return "\n".join(f"{e.speaker}: {e.text}" for e in self._entries)

    def contains_text(self, needle: str) -> bool:
        """Test helper: is `needle` present in any retained entry's text?"""
        return any(needle in e.text for e in self._entries)
