"""Command-line interface: `vramcalc MODEL ...`, `vramcalc bench ...`, `vramcalc gpus`."""

from __future__ import annotations

import argparse
import json
import sys

from . import gpus
from .estimate import GB, estimate

_SUBCOMMANDS = ("bench", "gpus")


def _add_estimate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train", action="store_true")
    p.add_argument("--lora", type=int, metavar="R")
    p.add_argument("--dtype", default="bf16", choices=("fp32", "bf16", "fp16"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--optimizer", default="adamw", choices=("adamw", "sgd"))
    p.add_argument("--precision", default="amp", choices=("amp", "pure"))
    p.add_argument("--attn", default="sdpa", choices=("eager", "sdpa", "flash"))
    p.add_argument("--overhead-gb", type=float, default=0.75)


def _estimate_kwargs(a: argparse.Namespace) -> dict:
    return {
        "mode": "train" if a.train else "inference", "dtype": a.dtype, "batch": a.batch,
        "seq_len": a.seq, "lora_rank": a.lora, "grad_checkpoint": a.grad_checkpoint,
        "optimizer": a.optimizer, "precision": a.precision, "attn_impl": a.attn,
        "overhead_gb": a.overhead_gb,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vramcalc", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    est = sub.add_parser("estimate", help="predict memory (default when MODEL is given)")
    est.add_argument("model")
    _add_estimate_args(est)
    est.add_argument("--gpu", help="GPU name or GB number; exit 1 if the model does not fit")
    est.add_argument("--json", action="store_true")

    sub.add_parser("gpus", help="list built-in GPU table")

    bench = sub.add_parser(
        "bench", help="measure on the local GPU and compare (needs vramcalc[bench])")
    bench.add_argument("model_path")
    _add_estimate_args(bench)
    bench.add_argument("--steps", type=int, default=3)
    bench.add_argument("--record", metavar="PATH")
    bench.add_argument("--json", action="store_true")
    return parser


def _parse_gpu(value: str | None):
    """Table names win over numbers so that `--gpu 3090` means the card, not 3090 GB."""
    if value is None:
        return None
    if gpus._key(value) in gpus._INDEX:
        return value
    try:
        return float(value)
    except ValueError:
        return value


def cmd_estimate(a: argparse.Namespace) -> int:
    e = estimate(a.model, **_estimate_kwargs(a))
    fit = e.fits(_parse_gpu(a.gpu)) if a.gpu is not None else None
    if a.json:
        out = {"estimate": e.to_dict()}
        if fit is not None:
            out["fit"] = fit.to_dict()
        print(json.dumps(out, indent=2))
    else:
        print(e.breakdown())
        if fit is not None:
            print()
            print(fit.message)
            for s in fit.suggestions:
                tag = " (changes training objective)" if s.changes_objective else ""
                change = ", ".join(f"{k}={v}" for k, v in s.change.items())
                need = (s.estimate.total + s.estimate.overhead) / GB
                print(f"  -> {change}: {need:.1f} GB{tag}")
    return 0 if fit is None or fit.ok else 1


def cmd_gpus(_: argparse.Namespace) -> int:
    for name, size in gpus.GPUS.items():
        print(f"{name:<20} {size / gpus.GIB:>5.0f} GiB")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench import bench

    kwargs = _estimate_kwargs(a)
    r = bench(a.model_path, steps=a.steps, record=a.record, **kwargs)
    print(json.dumps(r.to_dict(), indent=2) if a.json else r.table())
    return 0 if r.status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in _SUBCOMMANDS and argv[0] != "estimate" and not argv[0].startswith("-"):
        argv.insert(0, "estimate")
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.command is None:
        parser.print_help()
        return 2
    handler = {"estimate": cmd_estimate, "gpus": cmd_gpus, "bench": cmd_bench}[a.command]
    try:
        return handler(a)
    except (ValueError, KeyError, FileNotFoundError, ImportError, RuntimeError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        print(f"vramcalc: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
