import sys

import pytest

from vramcalc import estimate, gpus
from vramcalc.estimate import FitResult


@pytest.mark.parametrize("query", ["RTX 3090", "rtx3090", "rtx-3090", "3090", "GeForce RTX 3090"])
def test_lookup_normalizes(query):
    name, size = gpus.lookup(query)
    assert name == "RTX 3090"
    assert size == 24 * 2**30


def test_lookup_distinguishes_ti():
    assert gpus.lookup("3090 ti")[0] == "RTX 3090 Ti"
    assert gpus.lookup("3090")[0] == "RTX 3090"


def test_lookup_unknown_suggests_candidates():
    with pytest.raises(KeyError, match="A100"):
        gpus.lookup("A1000")


def test_resolve_number():
    assert gpus.resolve(24) == ("24 GB", 24 * 10**9)
    assert gpus.resolve(11.5) == ("11.5 GB", 11_500_000_000)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_resolve_bad_number(bad):
    with pytest.raises(ValueError, match="gpu"):
        gpus.resolve(bad)


def test_detect_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with pytest.raises(RuntimeError, match="GPU name or GB"):
        gpus.resolve(None)


def test_fits(tiny_cfg):
    e = estimate(tiny_cfg, seq_len=8)              # total ~236 KB + 0.5 GB overhead
    r = e.fits(1)
    assert isinstance(r, FitResult)
    assert r.ok is True
    assert r.gpu_name == "1 GB"
    assert r.usable_bytes == int(10**9 * gpus.USABLE_RATIO)
    assert r.required_bytes == e.total + e.overhead
    assert r.margin_bytes == r.usable_bytes - r.required_bytes
    assert r.suggestions == []
    assert "fits" in r.message


def test_does_not_fit(tiny_cfg):
    r = estimate(tiny_cfg, seq_len=8).fits(0.4)   # 0.4 GB < 0.5 GB overhead alone
    assert r.ok is False
    assert r.margin_bytes < 0
    assert "short" in r.message


def test_detect_shuts_down_nvml(monkeypatch):
    calls = []

    class FakeInfo:
        total = 24 * 2**30

    class FakeNvml:
        @staticmethod
        def nvmlInit():
            calls.append("init")

        @staticmethod
        def nvmlDeviceGetHandleByIndex(i):
            return i

        @staticmethod
        def nvmlDeviceGetName(h):
            return b"Fake GPU"

        @staticmethod
        def nvmlDeviceGetMemoryInfo(h):
            return FakeInfo

        @staticmethod
        def nvmlShutdown():
            calls.append("shutdown")

    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml)
    assert gpus.detect() == ("Fake GPU", 24 * 2**30)
    assert calls == ["init", "shutdown"]
