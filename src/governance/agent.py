"""The governance agent — the window → infer → act orchestration loop.

Owns every code-level invariant: the consent gate (pre-STT/pre-LLM), context
assembly, the recursive-refine loop, the precedence resolver, action execution,
window + KG/vector updates, and the commit barrier (each utterance is fully
resolved and persisted before the next begins). The LLM is consulted only as the
policy-check tool; it cannot override consent, precedence, or ephemerality."""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config, consent, policy_tool, store
from .context import ContextAssembler
from .kg import KnowledgeGraph
from .llm import LLM
from .models import Action, Decision, Utterance, Verdict
from .resolver import resolve
from .selfimprove import SelfImprover
from .stt import STT
from .vectorstore import VectorMemory
from .window import RollingWindow

_REDACTION_MARK = "[REDACTED]"

_ACTION_GLYPH = {
    Action.COMMIT: "✓ COMMIT",
    Action.REDACT: "▒ REDACT",
    Action.FLAG_FOR_REVIEW: "⚑ FLAG ",
    Action.DROP: "✗ DROP ",
    Action.CONSENT_GATE: "⛔ GATE ",
}


_DIGIT_RUN = re.compile(r"\d[\d\s\-\(\)\.]*\d")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _mask(text: str, values: list[str]) -> str:
    """Remove each sensitive value from the text, replacing it with a marker.

    Robust to formatting drift between what the model returns and what STT
    produced: in addition to literal variants, any contiguous digit run whose
    digits match a numeric redaction value is masked (so 8847-220193-04,
    884722019304, and 8847 220193 04 are all caught)."""
    masked = text
    # 1) literal variants (covers non-numeric values too)
    for v in values:
        v = v.strip()
        if not v:
            continue
        for variant in {v, re.sub(r"\s+", " ", v), v.replace("-", " "), v.replace(" ", "")}:
            if variant and variant in masked:
                masked = masked.replace(variant, _REDACTION_MARK)
    # 2) digit-normalized run matching
    targets = {_digits(v) for v in values if len(_digits(v)) >= 6}
    if targets:
        def repl(m: re.Match) -> str:
            return _REDACTION_MARK if _digits(m.group(0)) in targets else m.group(0)
        masked = _DIGIT_RUN.sub(repl, masked)
    return masked


def _still_contains(text: str, values: list[str]) -> bool:
    """True if any redaction value's cleartext (literal or digit-normalized) is
    still present in `text`."""
    for v in values:
        v = v.strip()
        if v and v in text:
            return True
    targets = {_digits(v) for v in values if len(_digits(v)) >= 6}
    if targets:
        for run in _DIGIT_RUN.findall(text):
            if _digits(run) in targets:
                return True
    return False


class Orchestrator:
    def __init__(self, *, graphrag: bool = True, use_cache: bool = False,
                 verbose: bool = False, llm: LLM | None = None, stt: STT | None = None):
        manifest = json.loads(config.MANIFEST_PATH.read_text())
        self.utterances = [Utterance(**u) for u in manifest]
        speakers = {
            u["speaker"]: {"role": u.get("role", ""), "side": u.get("side", ""),
                           "consent": u["consent"]}
            for u in manifest
        }

        self.window = RollingWindow()
        self.kg = KnowledgeGraph()
        self.kg.seed(speakers)
        self.llm = llm or LLM()
        self.vmem = VectorMemory()
        self.vmem.seed_policies(_policy_clauses())
        self.improver = SelfImprover(self.vmem, llm=self.llm)
        self.context = ContextAssembler(self.window, self.kg, self.vmem, graphrag=graphrag)

        self.stt = stt or STT(use_cache=use_cache)
        self.verbose = verbose
        self.graphrag = graphrag

    # --- main loop ---------------------------------------------------------
    def run(self) -> list[Decision]:
        store.reset()
        decisions: list[Decision] = []
        print(f"\n=== Governance run (mode={'GraphRAG' if self.graphrag else 'simple-window'}, "
              f"model={self.llm.model}) ===\n")
        for utt in self.utterances:
            decision, text = self._process(utt)
            self._commit(decision, text)        # commit barrier
            decisions.append(decision)
            self._log(decision)
        self.improver.finalize()
        print(f"\n=== Done. {len(decisions)} utterances governed. Outputs in out/ ===\n")
        return decisions

    # --- per-utterance -----------------------------------------------------
    def _process(self, utt: Utterance) -> tuple[Decision, str]:
        # 1) CONSENT GATE — hard, before any transcription or inference.
        if consent.is_gated(utt):
            return (Decision(
                utterance_id=utt.id, beat=utt.beat, speaker=utt.speaker,
                action=Action.CONSENT_GATE, policy_ids=[5], confidence=1.0,
                reason="non-consented speaker — declined at gate (never transcribed)",
            ), "")

        # 2) STT — real, live, this clip only.
        text = self.stt.transcribe(Path(config.PROJECT_ROOT / utt.audio_file))
        if not text.strip():
            return (Decision(
                utterance_id=utt.id, beat=utt.beat, speaker=utt.speaker,
                action=Action.COMMIT, confidence=1.0, reason="no speech detected",
            ), "")

        # 3-5) Context assembly -> policy tool -> recursive refine.
        verdict = policy_tool.check_against_policies(
            text, self.context.assemble(text, level=0), self.llm)
        res = resolve(verdict)
        # Monotonic caution: if the first read suspects a strict no-copy topic
        # (compensation/codename) and is unsure, added context may CONFIRM the
        # concern but must never downgrade it to a clear keep.
        drop_suspect = res.unsure and res.action == Action.DROP
        refine = 0
        while res.unsure and refine < config.MAX_REFINE:
            refine += 1
            ctx = self.context.assemble(text, level=refine)
            v2 = policy_tool.check_against_policies(text, ctx, self.llm)
            r2 = resolve(v2)
            verdict, res = v2, r2
            if not res.unsure:
                break
        if drop_suspect and res.action == Action.COMMIT:
            res = type(res)(Action.DROP, res.policy_ids or [1], res.confidence,
                            True, "no-copy-topic suspicion not resolved — contained")

        # 6) Build the committed text per action, with a redaction safety net.
        action = res.action
        committed = ""
        if action == Action.REDACT and verdict is not None:
            values = [r.value for r in verdict.redactions]
            committed = _mask(text, values)
            if _still_contains(committed, values):
                # Could not fully mask (e.g. STT formatting drift) -> contain.
                action = Action.DROP
                committed = ""
                res = type(res)(Action.DROP, res.policy_ids, res.confidence, True,
                                "redaction could not be verified — contained")
        elif action in (Action.COMMIT, Action.FLAG_FOR_REVIEW):
            committed = text

        decision = Decision(
            utterance_id=utt.id, beat=utt.beat, speaker=utt.speaker, action=action,
            policy_ids=res.policy_ids, confidence=res.confidence, unsure=res.unsure,
            refine_iterations=refine, committed_text=committed, reason=res.reason,
        )
        return decision, text

    # --- commit barrier ----------------------------------------------------
    def _commit(self, decision: Decision, text: str) -> None:
        store.persist(decision)
        if decision.action in (Action.COMMIT, Action.REDACT, Action.FLAG_FOR_REVIEW):
            self.window.add_committed(decision.utterance_id, decision.speaker,
                                      decision.committed_text)
            # KG/vector grow with the SAFE committed text only.
            self.improver.record(self.kg, decision, decision.committed_text)
        else:  # DROP / CONSENT_GATE — content-free placeholder + provenance only
            self.window.add_withheld(decision.utterance_id, decision.speaker)
            self.improver.record(self.kg, decision, "")
        # `text` goes out of scope here; for DROP/CONSENT_GATE it was never stored
        # anywhere downstream.

    def _log(self, d: Decision) -> None:
        glyph = _ACTION_GLYPH[d.action]
        pol = f"P{','.join(map(str, d.policy_ids))}" if d.policy_ids else "—"
        unsure = " [UNSURE]" if d.unsure else ""
        refine = f" refine×{d.refine_iterations}" if d.refine_iterations else ""
        # Never print content for DROP / CONSENT_GATE.
        if d.action in (Action.DROP, Action.CONSENT_GATE):
            tail = "content withheld"
        else:
            preview = " ".join(d.committed_text.split())
            tail = (preview[:70] + "…") if len(preview) > 70 else preview
        print(f"  {glyph} {d.utterance_id} b{d.beat:<2} {d.speaker:18.18s} "
              f"{pol:<6} c={d.confidence:.2f}{unsure}{refine}  | {tail}")


def _policy_clauses() -> list[tuple[str, str]]:
    """Split policies.md into clause chunks for retrieval seeding."""
    text = config.POLICIES_PATH.read_text()
    chunks = re.split(r"\n## ", text)
    out = []
    for i, c in enumerate(chunks):
        c = c.strip()
        if c and not c.startswith("# Governance"):
            out.append((f"policy::{i}", "## " + c if not c.startswith("##") else c))
    return out
