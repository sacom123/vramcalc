from vramcalc import estimate
from vramcalc.suggest import suggest

from .conftest import load_config


def _usable_for(e, factor):
    """Usable capacity as a fraction of what this estimate needs, so tests control fit/not-fit."""
    return int((e.total + e.overhead) * factor)


def test_no_candidates_in_minimal_inference(tiny_cfg):
    e = estimate(tiny_cfg, seq_len=512, batch=1, overhead_gb=0)
    assert suggest(e, _usable_for(e, 0.5)) == []


def test_train_candidate_order_and_labels():
    # A real-sized model: on the tiny arch LoRA/SGD/pure save less than 1%, so use Llama-3.2-1B
    e = estimate(load_config("llama-3.2-1b"), mode="train", seq_len=2048, batch=2, overhead_gb=0)
    # usable = 99% of required: every candidate that reduces anything fits
    out = suggest(e, _usable_for(e, 0.99))
    changes = [s.change for s in out]
    assert changes[0] == {"grad_checkpoint": True}
    assert changes[1] == {"batch": 1}
    assert changes[2] == {"seq_len": 1024}
    assert {"lora_rank": 16} in changes
    assert {"optimizer": "sgd"} in changes
    assert {"precision": "pure"} in changes
    labels = {tuple(s.change.items()): s.changes_objective for s in out}
    assert labels[(("grad_checkpoint", True),)] is False
    assert labels[(("batch", 1),)] is True
    assert labels[(("seq_len", 1024),)] is True
    assert labels[(("lora_rank", 16),)] is True
    for s in out:
        assert s.margin_bytes >= 0
        assert s.estimate.total + s.estimate.overhead <= _usable_for(e, 0.99)


def test_inference_reductions_do_not_change_objective(tiny_cfg):
    e = estimate(tiny_cfg, seq_len=4096, batch=4, overhead_gb=0)
    out = suggest(e, _usable_for(e, 0.9))
    assert all(s.changes_objective is False for s in out)
    assert all(set(s.change) <= {"batch", "seq_len"} for s in out)


def test_halving_stops_at_first_fit(tiny_cfg):
    e = estimate(tiny_cfg, seq_len=4096, batch=8, overhead_gb=0, attn_impl="flash")
    # activations and kv scale with b*s; need about 1/4 -> batch 2, seq 1024
    out = suggest(e, _usable_for(e, 0.3))
    by_key = {next(iter(s.change)): s.change for s in out}
    assert by_key["batch"] == {"batch": 2}
    assert by_key["seq_len"] == {"seq_len": 1024}


def test_nothing_fits(tiny_cfg):
    e = estimate(tiny_cfg, mode="train", seq_len=512, batch=1, overhead_gb=0)
    assert suggest(e, 1) == []
    r = e.fits(1e-9)
    assert r.suggestions == [] and "no tested single adjustment" in r.message
