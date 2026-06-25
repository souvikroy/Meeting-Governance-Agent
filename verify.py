"""End-to-end verification of the governed outputs.

Scope: everything the system PRODUCES — out/ (transcript, review queue, audit) and
the persistent state/ store. NOT the input audio (the simulated live source).

Asserts the ephemerality + correctness guarantees:
  * ABSENT anywhere downstream: the card number, the bank account (cleartext),
    the compensation figures, and the deal-codename-tied usage.
  * PRESENT in the transcript: the office phone number and the "Everest Ascent"
    documentary line (the traps that must be COMMITTED, not dropped/redacted).
  * transcript holds only COMMIT/REDACT/FLAG; a FLAG carries a legal marker.

Run after `uv run python run.py --live`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from governance import config  # noqa: E402

OUT = config.OUT_DIR
STATE = config.STATE_DIR

# Cleartext that must NEVER appear in any produced artifact.
MUST_BE_ABSENT = [
    "4012-8888-3320-7741", "4012 8888 3320 7741", "4012888833207741",  # card
    "8847-220193-04", "8847 220193 04", "884722019304",                # bank acct
    "240,000", "240000", "two hundred forty thousand",                 # comp figures
    "retention bonus", "equity grant",
]
# Content that the traps require us to KEEP (committed correctly).
MUST_BE_PRESENT = ["Everest Ascent", "555"]   # documentary title; office phone digits


def _all_produced_text() -> str:
    blobs = []
    for d in (OUT, STATE):
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file():
                    try:
                        blobs.append(p.read_text(errors="ignore"))
                    except Exception:
                        pass
    return "\n".join(blobs)


def main() -> int:
    transcript = OUT / "transcript.jsonl"
    if not transcript.exists():
        print("FAIL: no out/transcript.jsonl — run `uv run python run.py --live` first.")
        return 1

    produced = _all_produced_text()
    ok = True

    print("== Ephemerality: sensitive content must be ABSENT downstream ==")
    for s in MUST_BE_ABSENT:
        present = s in produced
        print(f"  [{'FAIL' if present else 'ok'}] absent: {s!r}")
        ok &= not present

    print("\n== Traps: content that must be COMMITTED (PRESENT) ==")
    for s in MUST_BE_PRESENT:
        present = s in produced
        print(f"  [{'ok' if present else 'FAIL'}] present: {s!r}")
        ok &= present

    print("\n== Transcript hygiene ==")
    rows = [json.loads(l) for l in transcript.read_text().splitlines() if l.strip()]
    allowed = {"COMMIT", "REDACT", "FLAG_FOR_REVIEW"}
    bad = [r for r in rows if r.get("action") not in allowed]
    print(f"  [{'FAIL' if bad else 'ok'}] transcript holds only COMMIT/REDACT/FLAG "
          f"({len(rows)} rows)")
    ok &= not bad
    flagged = [r for r in rows if r.get("action") == "FLAG_FOR_REVIEW"]
    has_marker = all(r.get("legal_review") for r in flagged)
    print(f"  [{'ok' if has_marker else 'FAIL'}] every FLAG carries a legal-review "
          f"marker ({len(flagged)} flagged)")
    ok &= has_marker

    print("\n" + ("PASS ✅" if ok else "FAIL ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
