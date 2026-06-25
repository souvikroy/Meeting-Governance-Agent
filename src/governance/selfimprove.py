"""Self-improving context memory.

After each decision the system updates its memory:
  * in-session KG + vector store grow with SANITIZED facts (kept content only;
    dropped/non-consented utterances contribute a content-free provenance node
    and nothing retrievable);
  * borderline/unsure decisions trigger a GENERATED lesson — the LLM reflects on
    an ABSTRACT, content-free description of the situation (which policy, what
    action, the confidence — never the utterance text) and writes one durable
    guideline, stored for retrieval in future sessions;
  * lightweight calibration counters persist in state/priors.json.

Nothing sensitive is ever persisted across sessions: no transcript content, no
codename usage, no numbers — lessons are generated from non-sensitive signals
only and are content-free by construction."""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from pydantic import BaseModel

from . import config
from .kg import KnowledgeGraph
from .llm import LLM
from .models import Action, Decision
from .vectorstore import VectorMemory

_LESSONS_PATH = config.STATE_DIR / "lessons.json"
_PRIORS_PATH = config.STATE_DIR / "priors.json"


class Lesson(BaseModel):
    guideline: str


_LESSON_SYSTEM = (
    "You distill durable, NON-SENSITIVE operating lessons for a meeting-governance "
    "classifier. You are given an ABSTRACT description of a borderline decision — "
    "no transcript content, only which policy was implicated, the action taken, and "
    "the confidence. Write exactly ONE general guideline (<=200 characters) that "
    "would help handle similar borderline cases better next time. Never invent or "
    "reference any specific content, name, number, or codename. Output ONLY JSON: "
    '{"guideline":"..."}'
)


class SelfImprover:
    def __init__(self, vmem: VectorMemory, llm: Optional[LLM] = None):
        self.vmem = vmem
        self.llm = llm
        self.lessons: dict[str, str] = {}
        self.priors: dict = {"counts": {}, "sessions": 0}
        self._policy_names = self._load_policy_names()
        self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        if _LESSONS_PATH.exists():
            self.lessons = json.loads(_LESSONS_PATH.read_text())
        if _PRIORS_PATH.exists():
            self.priors = json.loads(_PRIORS_PATH.read_text())
        for lid, text in self.lessons.items():
            try:
                self.vmem.add(lid, text, {"kind": "lesson"}, persistent=True)
            except Exception:
                pass  # already present

    def _load_policy_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        for t in config.load_ontology().get("topics", []):
            if t.get("policy_id"):
                names[int(t["policy_id"])] = t.get("name", f"policy {t['policy_id']}")
        return names

    # --- main hook ---------------------------------------------------------
    def record(self, kg: KnowledgeGraph, decision: Decision,
               utterance_text: str) -> None:
        # 1) Grow the in-session memory (sanitized).
        if decision.action in (Action.COMMIT, Action.REDACT, Action.FLAG_FOR_REVIEW):
            kg.add_committed_utterance(
                decision.utterance_id, decision.speaker,
                decision.policy_ids, decision.committed_text,
            )
            summary = (f"{decision.speaker} [{decision.action.value}] "
                       f"beat {decision.beat}: {decision.committed_text}")
            try:
                self.vmem.add(f"sess::{decision.utterance_id}", summary,
                              {"kind": "session", "action": decision.action.value})
            except Exception:
                pass
        else:  # DROP / CONSENT_GATE -> provenance only, no content
            kg.add_withheld_utterance(decision.utterance_id, decision.policy_ids)

        # 2) Generate (not look up) a lesson from borderline/unsure decisions.
        if decision.unsure:
            self._learn(decision)

        # 3) Calibration counters.
        c = self.priors.setdefault("counts", {})
        c[decision.action.value] = c.get(decision.action.value, 0) + 1

    # --- generation --------------------------------------------------------
    def _situation(self, decision: Decision) -> str:
        pols = decision.policy_ids or []
        names = [self._policy_names.get(p, f"policy {p}") for p in pols]
        subject = ", ".join(names) if names else "a possibly-sensitive topic"
        return (
            f"A borderline utterance may have implicated {subject}. The classifier "
            f"was unsure (confidence {decision.confidence:.2f}) and the system took "
            f"action {decision.action.value} (containment/escalation). No content is "
            f"available. What general guideline should improve handling next time?"
        )

    def _learn(self, decision: Decision) -> None:
        if self.llm is None:
            return  # generation unavailable; we do not fall back to a hardcoded table
        try:
            result = self.llm.structured(_LESSON_SYSTEM, self._situation(decision), Lesson)
        except Exception:
            result = None
        if not result or not result.guideline.strip():
            return
        text = result.guideline.strip()
        lid = "lesson::" + hashlib.sha256(text.lower().encode()).hexdigest()[:12]
        if lid in self.lessons:
            return  # dedupe identical lessons
        self.lessons[lid] = text
        try:
            self.vmem.add(lid, text, {"kind": "lesson"}, persistent=True)
        except Exception:
            pass

    def finalize(self) -> None:
        self.priors["sessions"] = self.priors.get("sessions", 0) + 1
        _LESSONS_PATH.write_text(json.dumps(self.lessons, indent=2))
        _PRIORS_PATH.write_text(json.dumps(self.priors, indent=2))
