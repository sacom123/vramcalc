import json
import sys

import pytest

from vramcalc.arch import Arch
from vramcalc.loaders import load_arch

from .conftest import CONFIG_DIR


def test_dict(tiny_cfg):
    assert isinstance(load_arch(tiny_cfg), Arch)


def test_file_path():
    arch = load_arch(CONFIG_DIR / "llama-3.1-8b.json")
    assert arch.num_layers == 32


def test_dir_path(tmp_path, tiny_cfg):
    (tmp_path / "config.json").write_text(json.dumps(tiny_cfg))
    assert load_arch(str(tmp_path)).hidden_size == 64


def test_dir_without_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_arch(tmp_path)


def test_hub_id_without_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"vramcalc\[hf\]"):
        load_arch("meta-llama/Llama-3.1-8B")


def test_hub_id_uses_hf_hub_download(monkeypatch, tmp_path, tiny_cfg):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tiny_cfg))
    calls = []

    class FakeHub:
        @staticmethod
        def hf_hub_download(repo_id, filename):
            calls.append((repo_id, filename))
            return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    assert load_arch("org/model").hidden_size == 64
    assert calls == [("org/model", "config.json")]


def test_hub_errors_become_clean_errors(monkeypatch):
    class FakeHub:
        @staticmethod
        def hf_hub_download(repo_id, filename):
            raise OSError("401 Client Error: gated repo")

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    with pytest.raises(FileNotFoundError, match=r"org/gated.*gated repo"):
        load_arch("org/gated")


def test_malformed_config_names_the_file(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="config.json"):
        load_arch(bad)


def test_name_falls_back_to_path_when_config_has_none(tmp_path, tiny_cfg):
    d = tmp_path / "Qwen2.5-1.5B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(tiny_cfg))
    assert load_arch(d).name == "Qwen2.5-1.5B"
    assert load_arch(tiny_cfg).name == ""
