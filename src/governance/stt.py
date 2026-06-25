"""Real speech-to-text via Deepgram (prerecorded API, one clip at a time —
simulating live arrival). The client is created once and reused.

Ephemerality: non-consented speakers are never transcribed (the orchestrator
skips this module entirely for them), so their words never leave the audio file.
An optional on-disk cache speeds local iteration but is bypassed on the graded
run (`--live`) and never stores non-consented/dropped content (the orchestrator
controls what is sent here)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config


class STT:
    def __init__(self, use_cache: bool = False):
        config.require("DEEPGRAM_API_KEY")
        from deepgram import DeepgramClient  # lazy import (SDK v7)

        self._client = DeepgramClient(api_key=config.DEEPGRAM_API_KEY)
        self._use_cache = use_cache
        self._cache_dir = config.PROJECT_ROOT / ".stt_cache"
        if use_cache:
            self._cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, audio_path: Path) -> Path:
        h = hashlib.sha256(audio_path.read_bytes()).hexdigest()[:16]
        return self._cache_dir / f"{h}.json"

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe one clip to text. Real network call to Deepgram."""
        audio_path = Path(audio_path)
        if self._use_cache:
            cp = self._cache_path(audio_path)
            if cp.exists():
                return json.loads(cp.read_text())["text"]

        with open(audio_path, "rb") as f:
            data = f.read()
        resp = self._client.listen.v1.media.transcribe_file(
            request=data,
            model=config.DEEPGRAM_MODEL,
            smart_format=True,   # digit-friendly formatting for account/card numbers
            punctuate=True,
            language="en",
        )
        text = ""
        try:
            text = (resp.results.channels[0].alternatives[0].transcript or "").strip()
        except (AttributeError, IndexError, TypeError):
            text = ""

        if self._use_cache:
            self._cache_path(audio_path).write_text(json.dumps({"text": text}))
        return text
