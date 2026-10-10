"""Resolve the `model` argument of estimate() into an Arch."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .arch import Arch

_HF_HINT = "is not a local path; install vramcalc[hf] to load from the Hub"


def load_arch(model) -> Arch:
    if isinstance(model, Arch):
        return model
    if isinstance(model, dict):
        return Arch.from_config(model)
    path = Path(model)
    if path.is_dir():
        config = path / "config.json"
        if not config.is_file():
            raise FileNotFoundError(f"no config.json in directory {path}")
        return _named(Arch.from_config(_read_json(config)), path.name)
    if path.is_file():
        return _named(Arch.from_config(_read_json(path)), path.parent.name)
    repo_id = str(model)
    return _named(Arch.from_config(_download_config(repo_id)), repo_id.split("/")[-1])


def _named(arch: Arch, fallback: str) -> Arch:
    """Configs without _name_or_path get the input's own name so records stay identifiable."""
    return arch if arch.name else dataclasses.replace(arch, name=fallback)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def _download_config(repo_id: str) -> dict:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(f"{repo_id!r} {_HF_HINT}") from exc
    if huggingface_hub is None:  # sys.modules poisoned in tests
        raise ImportError(f"{repo_id!r} {_HF_HINT}")
    try:
        local = huggingface_hub.hf_hub_download(repo_id, "config.json")
    except Exception as exc:
        raise FileNotFoundError(
            f"cannot fetch config.json for {repo_id!r}: {exc}"
            " (gated repo? run `huggingface-cli login`; offline? pass a local path)"
        ) from exc
    return _read_json(Path(local))
