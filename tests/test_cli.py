import json
import subprocess
import sys

from .conftest import CONFIG_DIR

CFG = str(CONFIG_DIR / "llama-3.2-1b.json")


def run(*args):
    return subprocess.run([sys.executable, "-m", "vramcalc.cli", *args],
                          capture_output=True, text=True, check=False)


def test_estimate_text():
    r = run(CFG, "--seq", "4096")
    assert r.returncode == 0
    assert "weights" in r.stdout and "total" in r.stdout


def test_estimate_json_and_fit():
    r = run(CFG, "--train", "--lora", "16", "--gpu", "RTX 3090", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["estimate"]["params"]["lora_rank"] == 16
    assert d["fit"]["gpu_name"] == "RTX 3090"
    assert d["fit"]["ok"] is True


def test_exit_1_when_not_fit():
    r = run(CFG, "--train", "--seq", "8192", "--batch", "8", "--gpu", "8")
    assert r.returncode == 1
    assert "short by" in r.stdout


def test_missing_config_dir_is_clean_error(tmp_path):
    r = run(str(tmp_path))
    assert r.returncode == 2
    assert "config.json" in r.stderr
    assert "Traceback" not in r.stderr


def test_unknown_gpu_is_clean_error():
    r = run(CFG, "--gpu", "A1000")
    assert r.returncode == 2
    assert "unknown GPU" in r.stderr


def test_gpus_subcommand():
    r = run("gpus")
    assert r.returncode == 0
    assert "RTX 3090 Ti" in r.stdout and "H100" in r.stdout


def test_bare_gpu_number_that_is_a_model_name_uses_table():
    r = run(CFG, "--train", "--gpu", "3090", "--json")
    d = json.loads(r.stdout)
    assert d["fit"]["gpu_name"] == "RTX 3090"
    assert d["fit"]["ok"] is False


def test_gpu_name_with_suffix_and_plain_number():
    assert json.loads(run(CFG, "--gpu", "3090 ti", "--json").stdout)["fit"]["gpu_name"] == "RTX 3090 Ti"
    assert json.loads(run(CFG, "--gpu", "80", "--json").stdout)["fit"]["gpu_name"] == "80 GB"
