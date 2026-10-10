"""Measure real GPU memory with torch and compare against the prediction. Spec section 7."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .estimate import GB, Estimate, EstimateParams, estimate

BUCKETS = ("weights", "states", "peak", "total", "overhead")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
_ATTN_IMPL = {"eager": "eager", "sdpa": "sdpa", "flash": "flash_attention_2"}


def buckets(est: Estimate) -> dict[str, int]:
    inference = est.params.mode == "inference"
    return {
        "weights": est.weights,
        "states": est.gradients + est.optimizer,
        "peak": est.peak_extra + (est.kv_cache if inference else 0),
        "total": est.total,
        "overhead": est.overhead,
    }


@dataclass(frozen=True)
class BenchResult:
    status: str                   # "ok" | "oom" | "failed"
    error: str | None
    params: EstimateParams
    predicted: dict[str, int]
    measured: dict[str, int]
    gpu_name: str
    versions: dict[str, str]
    timestamp: str

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["params"]["arch"] = dataclasses.asdict(self.params.arch)
        return d

    def table(self) -> str:
        lines = [f"{'bucket':<10}{'predicted':>12}{'measured':>12}{'error':>9}"]
        for k in BUCKETS:
            p = self.predicted.get(k, 0)
            m = self.measured.get(k)
            err = "N/A" if not m else f"{(p - m) / m * 100:+.1f}%"
            m_txt = f"{m / GB:.2f}" if m is not None else "-"
            lines.append(f"{k:<10}{p / GB:>12.2f}{m_txt:>12}{err:>9}")
        if self.status != "ok":
            lines.append(f"status: {self.status}: {self.error}")
        return "\n".join(lines)


def write_record(result: BenchResult, path: str | Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


_ABS_PATH = re.compile(r"(?<![\w])/[^\s'\"]+")


def _sanitize(message: str, model_path) -> str:
    """Keep local paths out of recorded errors (spec 7.4)."""
    message = message.replace(str(model_path), "<model>")
    return _ABS_PATH.sub("<path>", message)[:500]


def _require_bench_deps():
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "bench needs torch, transformers and peft; install vramcalc[bench]") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("bench needs a CUDA device")
    return torch, transformers, peft


def bench(model_path, *, mode="inference", dtype="bf16", batch=1, seq_len=2048, lora_rank=None,
          grad_checkpoint=False, optimizer="adamw", precision="amp", attn_impl="sdpa",
          overhead_gb=0.75, steps=3, record=None) -> BenchResult:
    # Pre-validation: everything here raises; nothing after model load does.
    compute_fp32 = dtype == "fp32" and (mode == "inference" or precision == "pure")
    if attn_impl == "sdpa" and compute_fp32:
        # The pinned efficient/flash backends do not serve fp32 GQA; math would store scores.
        raise ValueError("attn_impl='sdpa' cannot be pinned for fp32 compute; use attn_impl='eager'")
    torch, transformers, peft = _require_bench_deps()
    if attn_impl == "flash":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise ImportError("attn_impl='flash' needs the flash-attn package") from exc
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    est = estimate(model_path, mode=mode, dtype=dtype, batch=batch, seq_len=seq_len,
                   lora_rank=lora_rank, grad_checkpoint=grad_checkpoint, optimizer=optimizer,
                   precision=precision, attn_impl=attn_impl, overhead_gb=overhead_gb)
    p = est.params
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "peft": peft.__version__}
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    status, error, measured = "ok", None, {}
    try:
        measured = _measure(torch, transformers, peft, model_path, p, steps)
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", _sanitize((str(exc).splitlines() or [""])[0], model_path)
    except Exception as exc:  # noqa: BLE001 - runtime failure after model load, spec 7.3
        status, error = "failed", _sanitize(f"{type(exc).__name__}: {exc}", model_path)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    result = BenchResult(status, error, p, buckets(est), measured, gpu_name, versions, stamp)
    if record is not None:
        write_record(result, record)
    return result


def _reset(torch):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _measure(torch, transformers, peft, model_path, p: EstimateParams, steps: int) -> dict[str, int]:
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[p.dtype]
    load_dtype = torch.float32 if (p.mode == "train" and p.precision == "amp") else torch_dtype
    measured: dict[str, int] = {}

    _reset(torch)
    preexisting = torch.cuda.memory_allocated()  # tensors the caller already holds
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=load_dtype, attn_implementation=_ATTN_IMPL[p.attn_impl],
    ).cuda()
    if p.lora_rank is not None:
        cfg = peft.LoraConfig(r=p.lora_rank, lora_alpha=p.lora_rank, lora_dropout=0.0,
                              target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = peft.get_peft_model(model, cfg)
    torch.cuda.synchronize()
    measured["weights"] = torch.cuda.memory_allocated() - preexisting

    input_ids = torch.randint(0, p.arch.vocab_size, (p.batch, p.seq_len), device="cuda")

    if p.attn_impl == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def attn_ctx():
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION])
    else:
        attn_ctx = contextlib.nullcontext

    if p.mode == "inference":
        model.eval()
        _reset(torch)
        with torch.inference_mode(), attn_ctx():
            out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        measured["peak"] = torch.cuda.max_memory_allocated() - preexisting - measured["weights"]
        measured["states"] = 0
        del out
    else:
        model.train()
        model.config.use_cache = False
        if p.grad_checkpoint:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        trainable = [q for q in model.parameters() if q.requires_grad]
        if p.optimizer == "adamw":
            opt = torch.optim.AdamW(trainable, lr=1e-5, foreach=True)
        else:
            opt = torch.optim.SGD(trainable, lr=1e-5, momentum=0.9, foreach=True)
        amp = p.precision == "amp"
        if amp:
            def autocast():
                return torch.autocast("cuda", dtype=torch_dtype)
        else:
            autocast = contextlib.nullcontext

        def step():
            # attn_ctx stays active through backward so checkpoint recomputation
            # uses the same SDPA backend as the forward pass
            with attn_ctx():
                with autocast():
                    out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                    loss = out.loss
                loss.backward()
            del out, loss  # release logits before step() so workspace does not overlap (spec 5.6)
            opt.step()
            opt.zero_grad(set_to_none=False)

        step()  # warm-up
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        measured["states"] = baseline - preexisting - measured["weights"]
        peak = 0
        for _ in range(steps):
            _reset(torch)
            step()
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated() - baseline)
        measured["peak"] = peak
        del opt

    measured["total"] = measured["weights"] + measured["states"] + measured["peak"]
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    measured["overhead"] = (total - free) - torch.cuda.memory_reserved()
    return measured  # model and inputs are released when this frame returns
