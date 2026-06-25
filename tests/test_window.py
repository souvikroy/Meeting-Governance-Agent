import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governance.window import RollingWindow


def test_committed_text_is_visible():
    w = RollingWindow(size=4)
    w.add_committed("u1", "Maya", "the revenue was forty two million")
    assert "forty two million" in w.render()


def test_withheld_leaves_no_trace():
    w = RollingWindow(size=4)
    secret = "base salary of two hundred forty thousand"
    w.add_withheld("u1", "Maya")          # a DROP/CONSENT_GATE utterance
    assert not w.contains_text(secret)     # raw content never stored
    assert "withheld" in w.render().lower()
    assert secret not in w.render()


def test_count_bound():
    w = RollingWindow(size=3)
    for i in range(6):
        w.add_committed(f"u{i}", "S", f"line {i}")
    rendered = w.render()
    assert "line 5" in rendered and "line 0" not in rendered


def test_placeholder_is_content_free():
    # The placeholder must not encode the dropped topic.
    w = RollingWindow(size=4)
    w.add_withheld("u1", "Maya")
    assert "compensation" not in w.render().lower()
    assert "codename" not in w.render().lower()
