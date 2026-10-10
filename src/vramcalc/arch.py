"""Normalize a Hugging Face config.json into the fields the memory formulas need."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "gemma", "gemma2")

_REQUIRED = ("hidden_size", "num_hidden_layers", "num_attention_heads",
             "intermediate_size", "vocab_size")


class UnsupportedModelError(ValueError):
    """Raised when model_type is outside the supported allowlist."""


def _positive_int(cfg: dict, key: str) -> int:
    value = cfg[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Arch:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool
    norms_per_layer: int
    name: str = ""

    @property
    def q(self) -> int:
        """Width of the query projection output."""
        return self.num_heads * self.head_dim

    @property
    def kv(self) -> int:
        """Width of each of the key and value projection outputs."""
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_config(cls, cfg: dict) -> Arch:
        model_type = cfg.get("model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise UnsupportedModelError(
                f"model_type {model_type!r} is not supported; "
                f"supported: {', '.join(SUPPORTED_MODEL_TYPES)}"
            )
        missing = [k for k in _REQUIRED if k not in cfg]
        if missing:
            raise ValueError(f"config is missing required keys: {', '.join(missing)}")

        hidden = _positive_int(cfg, "hidden_size")
        layers = _positive_int(cfg, "num_hidden_layers")
        heads = _positive_int(cfg, "num_attention_heads")
        kv_heads = _positive_int(cfg, "num_key_value_heads") if "num_key_value_heads" in cfg else heads
        if heads % kv_heads != 0:
            raise ValueError(
                f"num_key_value_heads={kv_heads} must divide num_attention_heads={heads}"
            )
        if "head_dim" in cfg:
            head_dim = _positive_int(cfg, "head_dim")
        else:
            if hidden % heads != 0:
                raise ValueError(
                    f"hidden_size={hidden} is not divisible by num_attention_heads={heads}"
                )
            head_dim = hidden // heads

        is_gemma = model_type in ("gemma", "gemma2")
        name = str(cfg.get("_name_or_path", "")).rstrip("/").split("/")[-1]
        return cls(
            hidden_size=hidden,
            num_layers=layers,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=_positive_int(cfg, "intermediate_size"),
            vocab_size=_positive_int(cfg, "vocab_size"),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", is_gemma)),
            attention_bias=(model_type == "qwen2") or bool(cfg.get("attention_bias", False)),
            norms_per_layer=4 if model_type == "gemma2" else 2,
            name=name,
        )

    # Parameter-count pieces are exposed so memory.py can reuse them for P_cast.
    def attn_params(self) -> int:
        h = self.hidden_size
        return h * self.q + 2 * h * self.kv + self.q * h

    def attn_bias_params(self) -> int:
        return self.q + 2 * self.kv if self.attention_bias else 0

    def mlp_params(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    def embed_params(self) -> int:
        return self.vocab_size * self.hidden_size

    def param_count(self) -> int:
        per_layer = (self.attn_params() + self.attn_bias_params() + self.mlp_params()
                     + self.norms_per_layer * self.hidden_size)
        lm_head = 0 if self.tie_word_embeddings else self.embed_params()
        return self.embed_params() + self.num_layers * per_layer + self.hidden_size + lm_head
