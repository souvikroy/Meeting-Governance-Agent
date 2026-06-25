"""Embeddings via OpenRouter (the only embedding source), with an in-process
cache to avoid re-embedding identical strings. The resulting VECTORS are stored
locally (Chroma, on disk under state/) — generation is remote, storage is local.

Ephemerality: we embed only (a) the current utterance — which is already sent to
the LLM for classification, the same documented egress — and (b) sanitized/kept
or non-sensitive text (committed summaries, policy clauses, abstracted lessons).
Non-consented speech is never embedded because it is never transcribed."""
from __future__ import annotations

from functools import lru_cache

from . import config

_cache: dict[str, list[float]] = {}


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    config.require("OPENROUTER_API_KEY")
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://localhost/meeting-governance-agent",
            "X-Title": "Meeting Governance Agent",
        },
    )


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    out: list[list[float] | None] = [None] * len(texts)
    missing: list[str] = []
    idx: list[int] = []
    for i, t in enumerate(texts):
        if t in _cache:
            out[i] = _cache[t]
        else:
            missing.append(t)
            idx.append(i)
    if missing:
        resp = _client().embeddings.create(model=config.EMBED_MODEL, input=missing)
        for j, d in enumerate(resp.data):
            vec = list(d.embedding)
            _cache[missing[j]] = vec
            out[idx[j]] = vec
    return [v for v in out]  # type: ignore[return-value]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
