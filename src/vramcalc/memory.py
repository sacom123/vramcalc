"""Per-term memory formulas. Every function returns bytes unless its name says elements.

See spec section 5. Symbols: L layers, h hidden, a heads, q = a*head_dim,
kv = kv_heads*head_dim, i intermediate, V vocab, b batch, s seq_len.
"""

from __future__ import annotations

from .arch import Arch

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

# RMSNorm keeps fp32 intermediates for backward; count per layer. Confirm with bench.
NORM_FP32_TENSORS = 2


def weights(arch: Arch, dtype_bytes: int) -> int:
    return arch.param_count() * dtype_bytes


def kv_cache(arch: Arch, batch: int, seq_len: int, dtype_bytes: int) -> int:
    return 2 * arch.num_layers * batch * seq_len * arch.kv * dtype_bytes


def per_layer_act(arch: Arch, batch: int, seq_len: int, attn_impl: str, dtype_bytes: int) -> int:
    """Elements saved per layer for backward (also the transient per-layer footprint in inference)."""
    h, q, kv, i, a = arch.hidden_size, arch.q, arch.kv, arch.intermediate_size, arch.num_heads
    tokens = seq_len * batch
    elements = tokens * (2 * h + q + 2 * kv + q + h + 4 * i)
    # The +2h term is empirical. A6000 runs on Llama-3.2-1B/3B and Qwen2.5-1.5B leave a
    # residual of about two hidden-sized tensors per layer after the logits term is fixed.
    # Split it into named tensors when a model family shows >10% peak error.
    elements += tokens * 2 * h
    if attn_impl in ("sdpa", "eager") and kv < q:
        elements += 2 * tokens * q  # repeat_kv materializes K and V at query width
    if attn_impl == "eager":
        # score matrix in dtype plus fp32 softmax, expressed in dtype-sized elements
        elements += a * seq_len * seq_len * batch * (dtype_bytes + 4) // dtype_bytes
    return elements


def inference_activations(
    arch: Arch, batch: int, seq_len: int, attn_impl: str, dtype_bytes: int
) -> int:
    """Forward-only peak: prefill logits plus a small working set.

    Per-layer buffers are freed as the forward proceeds, so they do not survive to the peak
    (measured: kv_cache + logits explains the peak within 30 MB on 1B to 8B models).
    """
    tokens = seq_len * batch
    logits = tokens * arch.vocab_size * dtype_bytes
    working = 2 * tokens * arch.hidden_size * dtype_bytes
    if attn_impl == "eager":
        # score matrix in dtype plus fp32 softmax for one layer, and repeat_kv K/V expansion
        working += arch.num_heads * seq_len * seq_len * batch * (dtype_bytes + 4)
        working += 2 * tokens * arch.q * dtype_bytes
    return logits + working


LORA_TARGETS = 4  # q_proj, k_proj, v_proj, o_proj


def lora_params(arch: Arch, rank: int) -> int:
    h, q, kv = arch.hidden_size, arch.q, arch.kv
    # in+out per target: q (h,q), k (h,kv), v (h,kv), o (q,h)
    return arch.num_layers * rank * (4 * h + 2 * q + 2 * kv)


def cast_params(arch: Arch) -> int:
    """Linear weights autocast copies; the output linear is cast once whether tied or not."""
    return arch.num_layers * (arch.attn_params() + arch.mlp_params()) + arch.embed_params()


def gradients(trainable: int, state_bytes: int) -> int:
    return trainable * state_bytes


def optimizer_states(trainable: int, optimizer: str, state_bytes: int) -> int:
    moments = 2 if optimizer == "adamw" else 1
    return trainable * moments * state_bytes


def step_workspace(trainable: int, state_bytes: int) -> int:
    """Temporary buffer of the foreach optimizer implementation, alive only during step()."""
    return trainable * state_bytes


def autocast_cache(
    arch: Arch, precision: str, dtype_bytes: int, *, lora: bool, grad_checkpoint: bool
) -> int:
    if precision != "amp":
        return 0
    full = cast_params(arch) * dtype_bytes
    if lora and grad_checkpoint:
        # frozen casts are not cached; only the layer being recomputed holds them
        return full // arch.num_layers
    return full


def train_activations(
    arch: Arch, batch: int, seq_len: int, attn_impl: str, dtype_bytes: int,
    *, grad_checkpoint: bool, lora_rank: int | None,
) -> int:
    L, h, V = arch.num_layers, arch.hidden_size, arch.vocab_size
    tokens = seq_len * batch
    layer = per_layer_act(arch, batch, seq_len, attn_impl, dtype_bytes) * dtype_bytes
    norm_fp32 = NORM_FP32_TENSORS * 4 * tokens * h * L
    lora_branch = LORA_TARGETS * L * tokens * lora_rank * 4 if lora_rank else 0
    # logits peak sits in log_softmax backward: dtype logits + saved fp32 log_softmax
    # + fp32 grad_out + fp32 grad_in (the fp32 upcast itself is freed before backward)
    logits = tokens * V * (dtype_bytes + 4 + 4 + 4)
    if grad_checkpoint:
        return L * tokens * h * dtype_bytes + layer + norm_fp32 // L + lora_branch + logits
    return L * layer + norm_fp32 + lora_branch + logits
