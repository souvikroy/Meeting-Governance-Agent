"""End-to-end matrix test against the real pipeline (Deepgram + OpenRouter).

Marked `live`: it needs OPENROUTER_API_KEY + DEEPGRAM_API_KEY and generated audio
(`scripts/generate_audio.py`). It asserts on ACTIONS/OUTCOMES, not on exact model
reasoning strings (verdicts are not bit-reproducible). Auto-skips when keys or the
manifest/audio are absent.

Run:  uv run pytest -m live -q
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from governance import config  # noqa: E402
from governance.models import Action  # noqa: E402

pytestmark = pytest.mark.live

# Expected action per utterance id. Beat 10 is the deliberate "unsure" showcase.
EXPECTED = {
    "u01": {Action.COMMIT},
    "u02": {Action.COMMIT},
    "u03": {Action.DROP},                       # codename tied to the deal
    "u04": {Action.COMMIT},
    "u05": {Action.REDACT},                     # bank account
    "u06": {Action.REDACT},                     # card number
    "u07": {Action.DROP},                       # compensation figures
    "u08": {Action.CONSENT_GATE},               # Tomas — non-consented
    "u09": {Action.FLAG_FOR_REVIEW},            # safety defect + missed filing
    "u10": {Action.COMMIT},                     # movie title — NOT the codename
    "u11": {Action.COMMIT},                     # phone number — NOT a financial id
    "u13": {Action.COMMIT},
}


def _ready() -> bool:
    return bool(
        os.getenv("OPENROUTER_API_KEY")
        and os.getenv("DEEPGRAM_API_KEY")
        and config.MANIFEST_PATH.exists()
    )


@pytest.mark.skipif(not _ready(), reason="needs API keys + generated audio/manifest")
def test_beat_decision_matrix():
    from governance.agent import Orchestrator

    decisions = {d.utterance_id: d for d in Orchestrator(graphrag=True).run()}

    for uid, allowed in EXPECTED.items():
        assert decisions[uid].action in allowed, (
            f"{uid}: got {decisions[uid].action}, expected one of {allowed}"
        )

    # Beat 10 ('comp philosophy') should be contained/escalated, not a confident COMMIT.
    u12 = decisions["u12"]
    assert u12.unsure or u12.action in {Action.DROP, Action.FLAG_FOR_REVIEW}, (
        "Beat 10 ('comp philosophy') should be contained/escalated, not a confident COMMIT"
    )
    # Non-consented speaker (u08) is never transcribed -> no content.
    assert decisions["u08"].committed_text == ""


@pytest.mark.skipif(not _ready(), reason="needs API keys + generated audio/manifest")
def test_drop_leaves_no_trace():
    """After a full run, the produced artifacts contain none of the sensitive
    cleartext (the strongest ephemerality check)."""
    from governance.agent import Orchestrator

    Orchestrator(graphrag=True).run()
    produced = "\n".join(
        p.read_text(errors="ignore")
        for d in (config.OUT_DIR, config.STATE_DIR)
        for p in d.rglob("*") if p.is_file()
    )
    for forbidden in ("4012", "8847-220193-04", "884722019304", "240,000",
                      "two hundred forty thousand"):
        assert forbidden not in produced, f"leaked sensitive content: {forbidden!r}"
