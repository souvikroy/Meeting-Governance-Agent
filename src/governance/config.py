"""Central configuration. Reads .env; exposes provider settings and tunable constants.

Nothing sensitive is hard-coded. Keys come from the environment / .env.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# --- Providers -------------------------------------------------------------
# LLM: all generative calls go through OpenRouter (OpenAI-compatible API).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Strong reasoning model by default; fully swappable without code changes.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

# STT: real speech-to-text.
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2")

# TTS: scenario audio generation (a means to an end).
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

# Embeddings: generated via OpenRouter (the only embedding source); the resulting
# vectors are stored LOCALLY (Chroma under state/).
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")

# --- Governance / context tunables ----------------------------------------
# Rolling trailing window: last N committed-or-placeholder utterances. See README
# for the justification of WINDOW_SIZE.
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "4"))
WINDOW_TOKEN_CAP = int(os.getenv("WINDOW_TOKEN_CAP", "1500"))

# Below this confidence -> "unsure" -> conservative containment + escalation.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
# Drop-class actions (compensation/codename) below this confidence are still
# contained, but ALSO escalated for human review — we don't treat a lukewarm
# call on a strict no-copy topic as fully settled.
HIGH_CONFIDENCE = float(os.getenv("HIGH_CONFIDENCE", "0.8"))
# Recursive RAG self-refinement: max extra retrieve+re-query iterations when unsure.
MAX_REFINE = int(os.getenv("MAX_REFINE", "2"))

# GraphRAG retrieval breadth/depth.
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "5"))
GRAPH_DEPTH = int(os.getenv("GRAPH_DEPTH", "2"))

# --- Paths -----------------------------------------------------------------
AUDIO_DIR = PROJECT_ROOT / "audio"
OUT_DIR = PROJECT_ROOT / "out"
STATE_DIR = PROJECT_ROOT / "state"          # persistent cross-session, non-sensitive only
SCENARIO_DIR = PROJECT_ROOT / "scenario"
POLICIES_PATH = PROJECT_ROOT / "policies" / "policies.md"
SCRIPT_PATH = SCENARIO_DIR / "script.json"
MANIFEST_PATH = SCENARIO_DIR / "manifest.json"
# Config-controlled governance ontology (topics + entities) for the knowledge
# graph — NOT hardcoded in kg.py. Regenerate topics from the policies with
# scripts/derive_topics.py.
ONTOLOGY_PATH = SCENARIO_DIR / "ontology.json"
# Per-policy reasoning guidance, derived from the policies (scripts: governance.guidance).
# The policy-check system prompt is assembled from policies.md + this guidance at runtime.
GUIDANCE_PATH = SCENARIO_DIR / "policy_guidance.json"

# Protected deal codename, known a priori from Policy 2 (NOT learned from dropped
# content). Seeds the knowledge graph as a protected entity.
DEAL_CODENAME = "Project Atlas"

for _d in (AUDIO_DIR, OUT_DIR, STATE_DIR, SCENARIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# Built-in fallback so the KG still seeds if the ontology file is missing.
_DEFAULT_ONTOLOGY = {
    "entities": [],
    "topics": [
        {"name": "compensation", "note": "salary/bonus/equity/severance/pay — drop entirely (Policy 1)", "policy_id": 1},
        {"name": "legal exposure", "note": "admission of breach/defect/missed filing — keep but flag (Policy 3)", "policy_id": 3},
        {"name": "financial identifier", "note": "bank account / card numbers — redact the digits (Policy 4); phone numbers are not covered", "policy_id": 4},
    ],
}


def load_ontology() -> dict:
    """Load the config-controlled governance ontology (topics + entities). Falls
    back to a built-in default if the file is missing or unreadable."""
    if ONTOLOGY_PATH.exists():
        try:
            data = json.loads(ONTOLOGY_PATH.read_text())
            return {
                "entities": data.get("entities", _DEFAULT_ONTOLOGY["entities"]),
                "topics": data.get("topics", _DEFAULT_ONTOLOGY["topics"]),
            }
        except (json.JSONDecodeError, OSError):
            return _DEFAULT_ONTOLOGY
    return _DEFAULT_ONTOLOGY


def require(*names: str) -> None:
    """Fail fast with a clear message if a required key is missing."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            f"Copy .env.example to .env and fill them in."
        )
