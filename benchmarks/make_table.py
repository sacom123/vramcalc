"""Render benchmarks/results.jsonl as a Markdown table for the README.

Usage: python benchmarks/make_table.py [results.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GB = 10**9


def main(path: str) -> int:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    print("| model | mode | setting | GPU | predicted | measured | error |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        p = r["params"]
        parts = [p["dtype"], f"b={p['batch']}", f"s={p['seq_len']}", p["attn_impl"]]
        if p["mode"] == "train":
            parts.append(f"lora={p['lora_rank']}" if p["lora_rank"] else "full")
            parts.append(p["precision"])
            if p["grad_checkpoint"]:
                parts.append("ckpt")
        pred = r["predicted"]["total"] / GB
        if r["status"] == "ok":
            meas = r["measured"]["total"] / GB
            err = f"{(pred - meas) / meas * 100:+.1f}%"
            meas_txt = f"{meas:.2f} GB"
        else:
            meas_txt, err = r["status"].upper(), "-"
        gpu = r["gpu_name"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        name = p["arch"]["name"].removeprefix("Meta-").removeprefix("meta-")
        print(f"| {name} | {p['mode']} | {' '.join(parts)} | {gpu} | "
              f"{pred:.2f} GB | {meas_txt} | {err} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results.jsonl"))
