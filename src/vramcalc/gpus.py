"""Built-in GPU capacity table and resolution of the `gpu` argument of fits()."""

from __future__ import annotations

import math
import re

GIB = 2**30

# Nominal capacity in GiB. Keys are canonical display names.
GPUS: dict[str, int] = {name: gib * GIB for name, gib in {
    "RTX 3060": 12, "RTX 3070": 8, "RTX 3080": 10, "RTX 3080 Ti": 12,
    "RTX 3090": 24, "RTX 3090 Ti": 24,
    "RTX 4060": 8, "RTX 4060 Ti": 16, "RTX 4070": 12, "RTX 4070 Ti": 12,
    "RTX 4070 Ti Super": 16, "RTX 4080": 16, "RTX 4080 Super": 16, "RTX 4090": 24,
    "RTX 5070": 12, "RTX 5070 Ti": 16, "RTX 5080": 16, "RTX 5090": 32,
    "T4": 16, "V100 16GB": 16, "V100 32GB": 32,
    "A10": 24, "A40": 48, "A100 40GB": 40, "A100 80GB": 80,
    "H100 80GB": 80, "H200": 141, "L4": 24, "L40S": 48, "B200": 192,
}.items()}

# Provisional safety reserve for an otherwise idle GPU. Confirm with bench.
USABLE_RATIO = 0.97

_STRIP = re.compile(r"[\s\-_]+")
_PREFIXES = ("nvidia", "geforce", "tesla", "rtx")
_DETECT_HINT = "cannot detect GPU; pass a GPU name or GB number"


def _key(name: str) -> str:
    k = _STRIP.sub("", name.lower())
    for p in _PREFIXES:
        k = k.removeprefix(p)
    return k


_INDEX = {_key(n): n for n in GPUS}


def lookup(name: str) -> tuple[str, int]:
    k = _key(name)
    if k in _INDEX:
        canonical = _INDEX[k]
        return canonical, GPUS[canonical]
    candidates = [n for key, n in _INDEX.items() if k and (k in key or key in k)]
    hint = f"; did you mean {', '.join(candidates)}?" if candidates else ""
    raise KeyError(f"unknown GPU {name!r}{hint}")


def detect() -> tuple[str, int]:
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(_DETECT_HINT) from exc
    if pynvml is None:  # sys.modules poisoned in tests
        raise RuntimeError(_DETECT_HINT)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    except Exception as exc:  # pynvml raises NVMLError subclasses
        raise RuntimeError(_DETECT_HINT) from exc
    return (name.decode() if isinstance(name, bytes) else name), int(total)


def resolve(gpu) -> tuple[str, int]:
    if gpu is None:
        return detect()
    if isinstance(gpu, bool) or not isinstance(gpu, (int, float, str)):
        raise ValueError(f"gpu must be None, a GB number or a name, got {gpu!r}")  # noqa: TRY004
    if isinstance(gpu, (int, float)):
        if not math.isfinite(gpu) or gpu <= 0:
            raise ValueError(f"gpu GB must be a positive finite number, got {gpu!r}")
        return f"{gpu:g} GB", int(gpu * 10**9)
    return lookup(gpu)
