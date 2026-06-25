"""Offline end-to-end integration test of the full agent loop, using the
deterministic stand-ins in offline_sim.py (no API keys). Validates orchestration,
precedence, redaction, ephemerality, and store hygiene. The graded path
(run.py --live) exercises the same loop with real Deepgram + OpenRouter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from governance import config
from governance.models import Action
from offline_sim import FakeLLM, FakeSTT, build_manifest

EXPECTED = {
    "u01": Action.COMMIT, "u02": Action.COMMIT, "u03": Action.DROP,
    "u04": Action.COMMIT, "u05": Action.REDACT, "u06": Action.REDACT,
    "u07": Action.DROP, "u08": Action.CONSENT_GATE, "u09": Action.FLAG_FOR_REVIEW,
    "u10": Action.COMMIT, "u11": Action.COMMIT, "u13": Action.COMMIT,
}


def _run():
    build_manifest()
    from governance.agent import Orchestrator
    return {d.utterance_id: d for d in
            Orchestrator(graphrag=True, llm=FakeLLM(), stt=FakeSTT()).run()}


def test_decision_matrix_offline():
    d = _run()
    for uid, action in EXPECTED.items():
        assert d[uid].action == action, f"{uid}: got {d[uid].action}, want {action}"
    # Beat 10 is the deliberate unsure showcase.
    assert d["u12"].unsure and d["u12"].action in {Action.DROP, Action.FLAG_FOR_REVIEW}
    # Non-consented speaker never transcribed.
    assert d["u08"].committed_text == ""


def test_ephemerality_offline():
    _run()
    produced = "\n".join(
        p.read_text(errors="ignore")
        for base in (config.OUT_DIR, config.STATE_DIR)
        for p in base.rglob("*") if p.is_file()
    )
    # Sensitive cleartext must be absent from everything produced.
    for forbidden in ("4012 8888 3320 7741", "4012888833207741",
                      "8847-220193-04", "884722019304",
                      "240,000", "retention bonus", "equity grant",
                      "internal codename for this acquisition"):
        assert forbidden not in produced, f"leaked: {forbidden!r}"
    # Trap content that must be kept IS present.
    assert "Everest Ascent" in produced
    assert "555" in produced


def test_redaction_masks_in_transcript():
    import json
    _run()
    rows = [json.loads(l) for l in
            (config.OUT_DIR / "transcript.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["utterance_id"]: r for r in rows}
    assert "[REDACTED]" in by_id["u05"]["text"]
    assert "8847-220193-04" not in by_id["u05"]["text"]
    # Dropped/gated utterances are absent from the transcript entirely.
    assert "u03" not in by_id and "u07" not in by_id and "u08" not in by_id
