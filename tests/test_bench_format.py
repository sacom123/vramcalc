import json

from vramcalc import estimate
from vramcalc.bench import BenchResult, _sanitize, buckets, write_record


def test_buckets_inference(tiny_cfg):
    e = estimate(tiny_cfg, seq_len=8)
    b = buckets(e)
    assert b["weights"] == e.weights
    assert b["states"] == 0
    assert b["peak"] == e.peak_extra + e.kv_cache
    assert b["total"] == e.total
    assert b["overhead"] == e.overhead


def test_buckets_train(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=8)
    b = buckets(e)
    assert b["states"] == e.gradients + e.optimizer
    assert b["peak"] == e.peak_extra


def _result(tiny_cfg, **kw):
    e = estimate(tiny_cfg, seq_len=8)
    fields = {
        "status": "ok", "error": None, "params": e.params, "predicted": buckets(e),
        "measured": {"weights": 200_000, "states": 0, "peak": 25_000,
                     "total": 225_000, "overhead": 400_000_000},
        "gpu_name": "Test GPU", "versions": {"torch": "2.4.0"},
        "timestamp": "2026-10-10T00:00:00+00:00",
    }
    return BenchResult(**{**fields, **kw})


def test_table_reports_error_and_na(tiny_cfg):
    text = _result(tiny_cfg).table()
    assert "weights" in text and "%" in text
    assert "N/A" in text            # states measured 0 in inference


def test_to_dict_roundtrip_json(tiny_cfg):
    d = _result(tiny_cfg).to_dict()
    s = json.dumps(d)
    back = json.loads(s)
    assert back["status"] == "ok"
    assert back["params"]["arch"]["hidden_size"] == 64
    assert back["predicted"]["weights"] == 213_632
    assert "hostname" not in s and "/Users" not in s


def test_sanitize_strips_model_path():
    assert _sanitize("failed to load /Users/me/models/llama",
                     "/Users/me/models/llama") == "failed to load <model>"


def test_write_record_appends_lines(tiny_cfg, tmp_path):
    path = tmp_path / "r.jsonl"
    write_record(_result(tiny_cfg), path)
    write_record(_result(tiny_cfg, status="oom", error="CUDA out of memory"), path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["status"] == "oom"


def test_sanitize_strips_any_absolute_path():
    msg = "No such file: /home/u/.cache/huggingface/hub/x/config.json while loading m"
    out = _sanitize(msg, "m")
    assert "/home" not in out and "<path>" in out


def test_fp32_pure_sdpa_rejected_before_loading_deps(tiny_cfg):
    import pytest

    from vramcalc.bench import bench
    with pytest.raises(ValueError, match="eager"):
        bench(tiny_cfg, mode="train", dtype="fp32", precision="pure", attn_impl="sdpa")
    with pytest.raises(ValueError, match="eager"):
        bench(tiny_cfg, mode="inference", dtype="fp32", attn_impl="sdpa")
