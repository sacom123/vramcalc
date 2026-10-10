import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_make_table(tmp_path):
    rows = [
        {"status": "ok", "gpu_name": "NVIDIA GeForce RTX 3090 Ti",
         "params": {"arch": {"name": "Llama-3.2-1B"}, "mode": "inference", "dtype": "bf16",
                    "batch": 1, "seq_len": 2048, "lora_rank": None, "grad_checkpoint": False,
                    "precision": "amp", "attn_impl": "sdpa"},
         "predicted": {"total": 3_000_000_000}, "measured": {"total": 3_100_000_000}},
        {"status": "oom", "gpu_name": "NVIDIA GeForce RTX 3090 Ti",
         "params": {"arch": {"name": "Llama-3.1-8B"}, "mode": "train", "dtype": "bf16",
                    "batch": 1, "seq_len": 2048, "lora_rank": None, "grad_checkpoint": False,
                    "precision": "amp", "attn_impl": "sdpa"},
         "predicted": {"total": 90_000_000_000}, "measured": {}},
    ]
    src = tmp_path / "r.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "make_table.py"), str(src)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "| Llama-3.2-1B | inference" in r.stdout
    assert "-3.2%" in r.stdout
    assert "| Llama-3.1-8B | train" in r.stdout and "OOM" in r.stdout
