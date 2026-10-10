"""Adjustment suggestions when an estimate does not fit. Spec section 6.3."""

from __future__ import annotations

from .estimate import Estimate, EstimateParams, Suggestion, estimate_from_params

MIN_SEQ = 512
MIN_BATCH = 1


def _halving(p: EstimateParams, key: str, floor: int, usable: int):
    """Return the first halved value of `key` whose estimate fits, or None."""
    value = getattr(p, key)
    while value > floor:
        value = max(value // 2, floor)
        e = estimate_from_params(p.replace(**{key: value}))
        if e.total + e.overhead <= usable:
            return {key: value}
    return None


def _candidates(p: EstimateParams, usable: int):
    train = p.mode == "train"
    if train and not p.grad_checkpoint:
        yield {"grad_checkpoint": True}, False
    for key, floor in (("batch", MIN_BATCH), ("seq_len", MIN_SEQ)):
        hit = _halving(p, key, floor, usable)
        if hit:
            yield hit, train
    if train and p.lora_rank is None:
        yield {"lora_rank": 16}, True
    if train and p.optimizer == "adamw":
        yield {"optimizer": "sgd"}, True
    if train and p.precision == "amp":
        yield {"precision": "pure"}, True


def suggest(est: Estimate, usable_bytes: int) -> list[Suggestion]:
    out: list[Suggestion] = []
    for change, changes_objective in _candidates(est.params, usable_bytes):
        e = estimate_from_params(est.params.replace(**change))
        margin = usable_bytes - (e.total + e.overhead)
        if margin >= 0:
            out.append(Suggestion(change, changes_objective, e, margin))
    return out
