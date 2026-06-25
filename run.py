"""CLI entrypoint for the meeting-governance agent.

Examples:
  uv run python run.py --live          # real Deepgram STT + OpenRouter, GraphRAG context
  uv run python run.py --simple-window # baseline: trailing window only (A/B the context layer)
  uv run python run.py --cache         # reuse cached STT during iteration (NOT for the graded run)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from governance import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time meeting-governance agent")
    ap.add_argument("--live", action="store_true",
                    help="transcribe live (default); explicit flag for the graded run")
    ap.add_argument("--cache", action="store_true",
                    help="reuse on-disk STT cache for iteration (bypassed by the graded run)")
    ap.add_argument("--simple-window", action="store_true",
                    help="disable GraphRAG/KG; use only the trailing window")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    config.require("OPENROUTER_API_KEY", "DEEPGRAM_API_KEY")
    if not config.MANIFEST_PATH.exists():
        raise SystemExit("No manifest. Run: uv run python scripts/generate_audio.py")

    from governance.agent import Orchestrator

    orch = Orchestrator(
        graphrag=not args.simple_window,
        use_cache=args.cache and not args.live,
        verbose=args.verbose,
    )
    orch.run()


if __name__ == "__main__":
    main()
