"""Template synthesizer — regex-driven cases carried over from the original
in-place tests in ``mente.synthesis``."""
from __future__ import annotations

from mente.synthesis import TemplateSynthesizer as ReexportedTemplateSynthesizer
from mente.synthesizers import Synthesizer
from mente.synthesizers.template import TemplateSynthesizer


def test_reexport_is_same_class() -> None:
    # Back-compat: `from mente.synthesis import TemplateSynthesizer` must
    # resolve to the same class as the new canonical location.
    assert ReexportedTemplateSynthesizer is TemplateSynthesizer


def test_protocol_conformance() -> None:
    assert isinstance(TemplateSynthesizer(), Synthesizer)


def test_fib() -> None:
    s = TemplateSynthesizer()
    out = s.synthesize("compute the 10th fibonacci number")
    assert out is not None
    source, entry, args = out
    assert entry == "fib"
    assert args == {"n": 10}
    assert "def fib(n):" in source


def test_factorial() -> None:
    s = TemplateSynthesizer()
    out = s.synthesize("what is the factorial of 7")
    assert out is not None
    source, entry, args = out
    assert entry == "factorial"
    assert args == {"n": 7}
    assert "def factorial(n):" in source


def test_power() -> None:
    s = TemplateSynthesizer()
    out = s.synthesize("2 to the power of 10")
    assert out is not None
    source, entry, args = out
    assert entry == "power"
    assert args == {"a": 2, "b": 10}
    assert "def power(a, b):" in source


def test_unknown_returns_none() -> None:
    s = TemplateSynthesizer()
    assert s.synthesize("what is the capital of France") is None
