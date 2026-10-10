import pytest

from vramcalc import estimate


def test_inference_terms(tiny_cfg):
    e = estimate(tiny_cfg, batch=1, seq_len=8)
    assert e.weights == 213_632
    assert e.kv_cache == 2048
    assert e.activations == 6144
    assert e.gradients == e.optimizer == e.autocast_cache == e.step_workspace == 0
    assert e.persistent == 213_632 + 2048
    assert e.peak_extra == 6144
    assert e.total == e.persistent + e.peak_extra
    assert e.overhead == 750_000_000


def test_params_preserved(tiny_cfg):
    e = estimate(tiny_cfg, dtype="fp16", seq_len=16, attn_impl="eager")
    assert e.params.dtype == "fp16"
    assert e.params.seq_len == 16
    assert e.params.attn_impl == "eager"
    assert e.params.arch.hidden_size == 64


def test_to_dict_nests_params(tiny_cfg):
    d = estimate(tiny_cfg, seq_len=8).to_dict()
    assert d["weights"] == 213_632
    assert d["total"] == d["weights"] + d["kv_cache"] + d["activations"]
    assert d["params"]["optimizer"] == "adamw"
    assert d["params"]["arch"]["hidden_size"] == 64


def test_breakdown_mentions_terms(tiny_cfg):
    text = estimate(tiny_cfg, seq_len=8).breakdown()
    assert "weights" in text and "kv_cache" in text and "total" in text
    assert "GB" in text
    assert text.count("overhead") == 1


@pytest.mark.parametrize("kwargs,match", [
    ({"batch": 0}, "batch"),
    ({"seq_len": -1}, "seq_len"),
    ({"dtype": "int8"}, "dtype"),
    ({"mode": "eval"}, "mode"),
    ({"attn_impl": "math"}, "attn_impl"),
    ({"optimizer": "adam"}, "optimizer"),
    ({"precision": "fp32"}, "precision"),
    ({"overhead_gb": -1}, "overhead_gb"),
    ({"overhead_gb": float("inf")}, "overhead_gb"),
    ({"lora_rank": 8}, "lora_rank"),           # inference + lora
    ({"mode": "train", "lora_rank": 0}, "lora_rank"),
])
def test_invalid_arguments(tiny_cfg, kwargs, match):
    with pytest.raises(ValueError, match=match):
        estimate(tiny_cfg, **kwargs)


def test_train_amp_full(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=8)
    P = 106_816
    assert e.weights == P * 4
    assert e.gradients == P * 4
    assert e.optimizer == P * 8
    assert e.step_workspace == P * 4
    assert e.autocast_cache == 90_112 * 2
    assert e.activations == 73_728
    assert e.kv_cache == 0
    assert e.peak_extra == max(e.activations + e.autocast_cache, e.step_workspace)
    assert e.total == e.persistent + e.peak_extra


def test_forward_peak_dominates_with_long_sequence(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=64)
    assert e.activations + e.autocast_cache > e.step_workspace
    assert e.peak_extra == e.activations + e.autocast_cache


def test_train_pure_bf16(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=8, precision="pure")
    assert e.weights == 106_816 * 2
    assert e.gradients == 106_816 * 2
    assert e.autocast_cache == 0


def test_fp32_amp_is_treated_as_pure(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=8, dtype="fp32", precision="amp")
    assert e.autocast_cache == 0
    assert e.params.precision == "pure"


def test_train_lora(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=8, lora_rank=4, precision="pure")
    P_t = 3584
    assert e.weights == 106_816 * 2 + P_t * 4
    assert e.gradients == P_t * 4
    assert e.optimizer == P_t * 8
    assert e.step_workspace == P_t * 4
    assert e.activations == 73_728 + 1024


def test_step_workspace_can_dominate_peak(tiny_cfg):
    # seq_len=1, batch=1 makes activations tiny; pure fp32 full training has step_workspace = P*4
    e = estimate(tiny_cfg, mode="train", seq_len=1, dtype="fp32", precision="pure",
                 attn_impl="flash")
    assert e.step_workspace > e.activations + e.autocast_cache
    assert e.peak_extra == e.step_workspace
