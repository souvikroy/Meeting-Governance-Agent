"""Recursive GraphRAG context assembler.

Builds the context the policy-check tool reasons over, from three sources:
  1. the bounded trailing window (recent committed/placeholder turns);
  2. knowledge-graph expansion (entities mentioned -> neighbor facts, N hops);
  3. vector retrieval over the in-session working set + persistent lessons.

"Recursive" in two senses: (a) the KG expansion walks neighbors-of-neighbors to
GRAPH_DEPTH; (b) when the agent is unsure it calls assemble() at a higher `level`,
which widens retrieval (more hops, larger k, pull lessons) and re-queries —
bounded by MAX_REFINE.

A `simple` mode disables 2+3 and uses only the trailing window, so we can A/B the
lift the context layer provides."""
from __future__ import annotations

from . import config
from .kg import KnowledgeGraph
from .vectorstore import VectorMemory
from .window import RollingWindow


class ContextAssembler:
    def __init__(self, window: RollingWindow, kg: KnowledgeGraph,
                 vmem: VectorMemory, graphrag: bool = True):
        self.window = window
        self.kg = kg
        self.vmem = vmem
        self.graphrag = graphrag

    def assemble(self, utterance_text: str, level: int = 0) -> str:
        parts = [f"RECENT TURNS (trailing window):\n{self.window.render()}"]

        if self.graphrag and utterance_text.strip():
            depth = config.GRAPH_DEPTH + level
            kg_facts = self.kg.neighbors_text(utterance_text, depth=depth)
            if kg_facts:
                parts.append("KNOWLEDGE-GRAPH FACTS (entities & relationships):\n"
                             + "\n".join(kg_facts))

            k = config.RETRIEVE_K + level * 3
            session_hits = self.vmem.query(utterance_text, k=k, persistent=False)
            if session_hits:
                parts.append("RELATED EARLIER (this meeting):\n"
                             + "\n".join(f"- {h['document']}" for h in session_hits))

            lessons = self.vmem.query(utterance_text, k=k, persistent=True)
            # Surface only abstracted lessons here (policies already live in the
            # system prompt). Filter to lesson-kind docs.
            lesson_docs = [h for h in lessons if h["metadata"].get("kind") == "lesson"]
            if lesson_docs:
                parts.append("LEARNED GUIDANCE (from prior sessions):\n"
                             + "\n".join(f"- {h['document']}" for h in lesson_docs))

        return "\n\n".join(parts)
