import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governance import consent
from governance.models import Utterance


def _u(speaker, consented):
    return Utterance(id="u", beat=1, speaker=speaker, consent=consented,
                     audio_file="x.mp3")


def test_non_consented_is_gated():
    assert consent.is_gated(_u("Tomas Herrera", False))


def test_consented_is_not_gated():
    assert not consent.is_gated(_u("Maya Okafor", True))


def test_gate_operates_per_speaker():
    # One non-consented speaker is gated while others are governed normally.
    mix = [_u("Maya", True), _u("Tomas", False), _u("Lena", True)]
    gated = [u.speaker for u in mix if consent.is_gated(u)]
    assert gated == ["Tomas"]
