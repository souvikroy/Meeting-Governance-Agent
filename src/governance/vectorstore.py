"""Two-tier vector memory backed by Chroma, with local embeddings.

  * ephemeral  — in-memory, lives for one meeting run, then gone. Holds the
                 sanitized in-session working set (entity facts, verdict
                 provenance). Purged automatically when the process exits.
  * persistent — on disk under state/. Holds NON-SENSITIVE only: policy clauses
                 and abstracted "lessons" reused across sessions.

Sanitization is enforced by the caller (selfimprove.py / kg seeding). This module
only stores what it is given."""
from __future__ import annotations

from typing import Optional

from . import config, embeddings


class VectorMemory:
    def __init__(self):
        import chromadb

        self._ephemeral_client = chromadb.EphemeralClient()
        self._persistent_client = chromadb.PersistentClient(
            path=str(config.STATE_DIR / "chroma")
        )
        self._ephemeral = self._ephemeral_client.get_or_create_collection(
            name="session", metadata={"hnsw:space": "cosine"}
        )
        self._persistent = self._persistent_client.get_or_create_collection(
            name="lessons", metadata={"hnsw:space": "cosine"}
        )

    def _coll(self, persistent: bool):
        return self._persistent if persistent else self._ephemeral

    def add(self, id_: str, text: str, metadata: Optional[dict] = None,
            persistent: bool = False) -> None:
        self._coll(persistent).add(
            ids=[id_],
            embeddings=[embeddings.embed_one(text)],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def query(self, text: str, k: int = 5, persistent: bool = False) -> list[dict]:
        coll = self._coll(persistent)
        if coll.count() == 0:
            return []
        res = coll.query(
            query_embeddings=[embeddings.embed_one(text)],
            n_results=min(k, coll.count()),
        )
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"document": doc, "metadata": meta or {}, "distance": dist})
        return out

    def seed_policies(self, clauses: list[tuple[str, str]]) -> None:
        """Index policy clauses into the persistent store once (id, text)."""
        existing = set(self._persistent.get().get("ids", []))
        for cid, text in clauses:
            if cid not in existing:
                self.add(cid, text, {"kind": "policy"}, persistent=True)
