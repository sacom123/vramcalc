"""Compose per-term formulas into an Estimate for a given mode."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import memory
from .arch import Arch
from .loaders import load_arch

GB = 10**9

MODES = ("inference", "train")
ATTN_IMPLS = ("eager", "sdpa", "flash")
OPTIMIZERS = ("adamw", "sgd")
PRECISIONS = ("amp", "pure")


@dataclass(frozen=True)
class EstimateParams:
    arch: Arch
    mode: str
    dtype: str
    batch: int
    seq_len: int
    lora_rank: int | None
    grad_checkpoint: bool
    optimizer: str
    precision: str
    attn_impl: str
    overhead_gb: float

    def replace(self, **changes) -> EstimateParams:
        return EstimateParams(**{**asdict_shallow(self), **changes})


def asdict_shallow(params: EstimateParams) -> dict:
    """asdict() without recursing into Arch, so Arch stays an Arch."""
    return {f: getattr(params, f) for f in EstimateParams.__dataclass_fields__}


@dataclass(frozen=True)
class Suggestion:
    change: dict
    changes_objective: bool
    estimate: Estimate
    margin_bytes: int


@dataclass(frozen=True)
class FitResult:
    ok: bool
    gpu_name: str
    gpu_bytes: int
    usable_bytes: int
    required_bytes: int
    margin_bytes: int
    suggestions: list[Suggestion]
    message: str

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("ok", "gpu_name", "gpu_bytes", "usable_bytes",
                                            "required_bytes", "margin_bytes", "message")}
        d["suggestions"] = [
            {"change": s.change, "changes_objective": s.changes_objective,
             "margin_bytes": s.margin_bytes, "total": s.estimate.total}
            for s in self.suggestions
        ]
        return d


@dataclass(frozen=True)
class Estimate:
    weights: int
    kv_cache: int
    gradients: int
    optimizer: int
    activations: int
    autocast_cache: int
    step_workspace: int
    overhead: int
    params: EstimateParams = field(repr=False)

    @property
    def persistent(self) -> int:
        return self.weights + self.kv_cache + self.gradients + self.optimizer

    @property
    def peak_extra(self) -> int:
        return max(self.activations + self.autocast_cache, self.step_workspace)

    @property
    def total(self) -> int:
        return self.persistent + self.peak_extra

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in _TERMS}
        d.update(persistent=self.persistent, peak_extra=self.peak_extra, total=self.total)
        params = asdict_shallow(self.params)
        params["arch"] = asdict(self.params.arch)
        d["params"] = params
        return d

    def fits(self, gpu=None) -> FitResult:
        from . import gpus  # local import keeps gpus optional for pure estimation
        from .suggest import suggest

        name, size = gpus.resolve(gpu)
        usable = int(size * gpus.USABLE_RATIO)
        required = self.total + self.overhead
        margin = usable - required
        ok = required <= usable
        suggestions = [] if ok else suggest(self, usable)
        if ok:
            message = f"fits {name}: {required / GB:.1f} GB of {usable / GB:.1f} GB usable"
        elif suggestions:
            n = len(suggestions)
            message = (f"short by {-margin / GB:.1f} GB on {name}; "
                       f"{n} single adjustment{'s fit' if n != 1 else ' fits'}")
        else:
            message = (f"short by {-margin / GB:.1f} GB on {name}; "
                       "no tested single adjustment fits. "
                       "Combine adjustments or consider quantization or distributed training")
        return FitResult(ok, name, size, usable, required, margin, suggestions, message)

    def breakdown(self) -> str:
        rows = [(k, getattr(self, k)) for k in _TERMS if k != "overhead" and getattr(self, k)]
        rows += [("total", self.total), ("overhead", self.overhead)]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}}  {v / GB:8.2f} GB" for k, v in rows)


_TERMS = ("weights", "kv_cache", "gradients", "optimizer",
          "activations", "autocast_cache", "step_workspace", "overhead")


def _validate(p: EstimateParams) -> None:
    if p.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {p.mode!r}")
    if p.dtype not in memory.DTYPE_BYTES:
        raise ValueError(f"dtype must be one of {tuple(memory.DTYPE_BYTES)}, got {p.dtype!r}")
    if p.attn_impl not in ATTN_IMPLS:
        raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {p.attn_impl!r}")
    if p.optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {p.optimizer!r}")
    if p.precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {p.precision!r}")
    for name in ("batch", "seq_len"):
        v = getattr(p, name)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    if p.lora_rank is not None:
        if p.mode != "train":
            raise ValueError("lora_rank requires mode='train'")
        if not isinstance(p.lora_rank, int) or isinstance(p.lora_rank, bool) or p.lora_rank <= 0:
            raise ValueError(f"lora_rank must be a positive integer, got {p.lora_rank!r}")
    if (not isinstance(p.overhead_gb, (int, float)) or not math.isfinite(p.overhead_gb)
            or p.overhead_gb < 0):
        raise ValueError(f"overhead_gb must be a finite number >= 0, got {p.overhead_gb!r}")


def estimate(
    model,
    *,
    mode: str = "inference",
    dtype: str = "bf16",
    batch: int = 1,
    seq_len: int = 2048,
    lora_rank: int | None = None,
    grad_checkpoint: bool = False,
    optimizer: str = "adamw",
    precision: str = "amp",
    attn_impl: str = "sdpa",
    overhead_gb: float = 0.75,
) -> Estimate:
    params = EstimateParams(
        arch=load_arch(model), mode=mode, dtype=dtype, batch=batch, seq_len=seq_len,
        lora_rank=lora_rank, grad_checkpoint=grad_checkpoint, optimizer=optimizer,
        precision=precision, attn_impl=attn_impl, overhead_gb=overhead_gb,
    )
    return estimate_from_params(params)


def estimate_from_params(p: EstimateParams) -> Estimate:
    _validate(p)
    B = memory.DTYPE_BYTES[p.dtype]
    overhead = int(p.overhead_gb * GB)
    if p.mode == "inference":
        return Estimate(
            weights=memory.weights(p.arch, B),
            kv_cache=memory.kv_cache(p.arch, p.batch, p.seq_len, B),
            gradients=0, optimizer=0,
            activations=memory.inference_activations(p.arch, p.batch, p.seq_len, p.attn_impl, B),
            autocast_cache=0, step_workspace=0,
            overhead=overhead, params=p,
        )
    if p.dtype == "fp32" and p.precision == "amp":
        p = p.replace(precision="pure")  # nothing to autocast
    S = 4 if p.precision == "amp" else B
    lora = p.lora_rank is not None
    trainable = memory.lora_params(p.arch, p.lora_rank) if lora else p.arch.param_count()
    adapter_bytes = 4 if lora else S
    weights = memory.weights(p.arch, S) + (trainable * 4 if lora else 0)
    return Estimate(
        weights=weights,
        kv_cache=0,
        gradients=memory.gradients(trainable, adapter_bytes),
        optimizer=memory.optimizer_states(trainable, p.optimizer, adapter_bytes),
        activations=memory.train_activations(
            p.arch, p.batch, p.seq_len, p.attn_impl, B,
            grad_checkpoint=p.grad_checkpoint, lora_rank=p.lora_rank,
        ),
        autocast_cache=memory.autocast_cache(
            p.arch, p.precision, B, lora=lora, grad_checkpoint=p.grad_checkpoint,
        ),
        step_workspace=memory.step_workspace(trainable, adapter_bytes),
        overhead=overhead, params=p,
    )
