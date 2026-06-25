"""Deterministic stand-ins for an OFFLINE end-to-end validation of the
orchestration. This is a WIRING test harness — NOT the graded path. The graded
run (`run.py --live`) uses real Deepgram STT and real OpenRouter LLM inference.

  * FakeSTT  — returns a pre-baked transcript per clip, mimicking what Deepgram's
               `smart_format` would emit (digits formatted), so REDACT masking and
               verify.py are exercised exactly as in the live path.
  * FakeLLM  — a transparent rule-based ORACLE that produces structured Verdicts.
               It is the test's expected-answer source, deliberately simple; it is
               NOT the product's classifier.

Together they drive the full agent loop (consent gate, GraphRAG context assembly
with REAL embeddings/Chroma/KG, resolver, precedence, redaction, stores) without
any external API key."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governance import config
from governance.models import PolicyMatch, Redaction, Verdict

# Simulated Deepgram transcripts (digits formatted, as smart_format would).
SIM: dict[str, str] = {
    "u01": "Good morning everyone, and thanks for joining. Can everyone hear me okay? Let's do a quick roll call. We have me, Maya, leading from Northwind, Raj on diligence, and from Cendara we have Lena and Tomas.",
    "u02": "Great. Here is the agenda for today's session. We will walk through the financials, the customer contracts, the IP position, and any open liabilities. I would like to keep us moving so we cover all four.",
    "u03": "Before we dive in, let's align on where we are with Project Atlas, which is our internal codename for this acquisition, so everyone is on the same page about the transaction timeline.",
    "u04": "Sure. Last fiscal year we did about 42 million dollars in revenue. Our top three customers account for roughly 60 percent of that, and annual churn has been holding around 8 percent.",
    "u05": "For the diligence escrow, let me read out the bank account. The account number is 8847-220193-04.",
    "u06": "And for the expense reconciliation question, the corporate card number is 4012 8888 3320 7741.",
    "u07": "Now let's talk retention for the key engineers so we can model deal cost. For the lead robotics engineer we are looking at a base salary of 240,000 dollars, plus an equity grant worth about 300,000, and a retention bonus of 50,000.",
    "u08": "(non-consented speaker — never transcribed)",
    "u09": "I should flag something on the product side. On the last firmware release, we shipped a build that we knew had a safety defect, and we did not file the required regulatory disclosure in time.",
    "u10": "On a lighter note, totally unrelated, did anyone catch that mountain climbing documentary, Project Atlas, The Everest Ascent? It was fantastic.",
    "u11": "Oh, and Lena, for the follow up materials, Cendara's main office phone number is 415-555-0198, correct?",
    "u12": "Good. One scheduling note. The board is planning to discuss our overall comp philosophy at the offsite next quarter, but that is a separate conversation for another day.",
    "u13": "Okay, I think that is everything for today. Let's reconvene next Thursday at the same time. Thanks everyone, really appreciate it.",
}

_NUM = re.compile(r"([0-9][0-9 ,\-]{6,}[0-9])")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_manifest() -> None:
    """Write scenario/manifest.json from the script (no audio needed offline)."""
    script = json.loads(config.SCRIPT_PATH.read_text())
    speakers = script["speakers"]
    manifest = []
    for u in script["utterances"]:
        meta = speakers[u["speaker"]]
        manifest.append({
            "id": u["id"], "beat": u["beat"], "speaker": u["speaker"],
            "role": meta.get("role", ""), "side": meta.get("side", ""),
            "consent": bool(meta.get("consent", True)),
            "audio_file": f"audio/{u['id']}_{slug(u['speaker'])}.mp3",
        })
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


class FakeSTT:
    def transcribe(self, audio_path) -> str:
        uid = Path(audio_path).stem.split("_")[0]
        return SIM.get(uid, "")


class FakeLLM:
    """Transparent rule-based verdict oracle (test only)."""
    model = "offline-sim"

    def structured(self, system: str, user: str, schema):
        # Lesson generation (self-improvement) asks for a different schema.
        if schema.__name__ == "Lesson":
            return schema(guideline="For borderline meta-references to a sensitive "
                                    "topic without disclosed content, escalate for "
                                    "human review rather than committing.")

        m = re.search(r'CURRENT UTTERANCE TO CLASSIFY:\s*"(.*?)"', user, re.S)
        text = (m.group(1) if m else "").strip()
        low = text.lower()

        matches: list[PolicyMatch] = []
        redactions: list[Redaction] = []
        conf = 0.95

        if "account number" in low or "card number" in low:
            num = _NUM.search(text)
            if num:
                val = num.group(1).strip().rstrip(".").strip()
                redactions.append(Redaction(value=val, reason="financial identifier"))
                matches.append(PolicyMatch(policy_id=4, matched=True,
                                           rationale="account/card number", span=val))
        elif any(k in low for k in ("salary", "bonus", "equity", "retention")) and any(c.isdigit() for c in text):
            matches.append(PolicyMatch(policy_id=1, matched=True, rationale="compensation figures"))
        elif "codename" in low and "atlas" in low and any(
            k in low for k in ("acquisition", "transaction", "deal")
        ):
            matches.append(PolicyMatch(policy_id=2, matched=True, rationale="codename tied to the deal"))
        elif "everest ascent" in low:
            pass  # documentary title -> commit
        elif "phone number" in low:
            pass  # not a financial identifier -> commit
        elif any(k in low for k in ("safety defect", "did not file", "regulatory disclosure")):
            matches.append(PolicyMatch(policy_id=3, matched=True, rationale="legal/regulatory exposure"))
        elif "comp philosophy" in low or ("comp" in low and "philosophy" in low):
            matches.append(PolicyMatch(policy_id=1, matched=False, span="comp philosophy",
                                       rationale="meta-reference to compensation, no figures"))
            conf = 0.3

        return Verdict(policy_matches=matches, redactions=redactions,
                       confidence=conf, overall_reasoning="offline oracle")
