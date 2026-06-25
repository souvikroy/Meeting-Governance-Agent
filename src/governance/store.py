"""Durable, ephemerality-respecting output stores.

Three artifacts in out/:
  * transcript.jsonl   — the governed recording: COMMIT (full), REDACT (masked),
                         FLAG_FOR_REVIEW (full + marker). DROP / CONSENT_GATE
                         write NOTHING.
  * review_queue.jsonl — FLAG and unsure escalations. Stores utterance_id +
                         (masked) preview only — never raw droppable text.
  * decisions.jsonl    — audit log: action, policy ids, confidence, generic
                         reason. NEVER the raw text or cleartext redaction value,
                         and never the model's verbatim reasoning for DROP/REDACT.

A DROP leaves no copy in any of these (or in stdout — see agent.py)."""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .models import Action, Decision

_TRANSCRIPT = config.OUT_DIR / "transcript.jsonl"
_REVIEW = config.OUT_DIR / "review_queue.jsonl"
_AUDIT = config.OUT_DIR / "decisions.jsonl"


def reset() -> None:
    for p in (_TRANSCRIPT, _REVIEW, _AUDIT):
        if p.exists():
            p.unlink()


def _append(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _preview(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def persist(decision: Decision) -> None:
    """Write the decision to the appropriate stores, honoring ephemerality."""
    # 1) Durable transcript — only for content we are allowed to keep.
    if decision.action in (Action.COMMIT, Action.REDACT, Action.FLAG_FOR_REVIEW):
        entry = {
            "utterance_id": decision.utterance_id,
            "beat": decision.beat,
            "speaker": decision.speaker,
            "text": decision.committed_text,        # already masked for REDACT
            "action": decision.action.value,
        }
        if decision.action == Action.FLAG_FOR_REVIEW:
            entry["legal_review"] = True
            entry["flag_reason"] = decision.reason
        if decision.action == Action.REDACT:
            entry["redacted"] = True
        _append(_TRANSCRIPT, entry)

    # 2) Review queue — flags + unsure escalations. Preview is of the *committed*
    #    (already-safe) text only; for contained content there is nothing to show.
    if decision.action == Action.FLAG_FOR_REVIEW or (
        decision.unsure and decision.action != Action.COMMIT
    ):
        _append(_REVIEW, {
            "utterance_id": decision.utterance_id,
            "beat": decision.beat,
            "speaker": decision.speaker,
            "policy_ids": decision.policy_ids,
            "reason": decision.reason,
            "preview": _preview(decision.committed_text) if decision.committed_text else "(content withheld)",
        })

    # 3) Audit log — provenance for EVERY decision, but never the content.
    _append(_AUDIT, {
        "utterance_id": decision.utterance_id,
        "beat": decision.beat,
        "speaker": decision.speaker,
        "action": decision.action.value,
        "policy_ids": decision.policy_ids,
        "confidence": round(decision.confidence, 3),
        "unsure": decision.unsure,
        "refine_iterations": decision.refine_iterations,
        "reason": decision.reason,
    })
