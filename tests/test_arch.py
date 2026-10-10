import pytest

from vramcalc.arch import Arch, UnsupportedModelError

from .conftest import load_config

# Exact counts from safetensors metadata of the public checkpoints.
REFERENCE_PARAMS = {
    "llama-2-7b": 6_738_415_616,
    "llama-3.1-8b": 8_030_261_248,
    "llama-3.2-1b": 1_235_814_400,
    "llama-3.2-3b": 3_212_749_824,
    "mistral-7b-v0.3": 7_248_023_552,
    "qwen2.5-7b": 7_615_616_512,
    "qwen2.5-1.5b": 1_543_714_304,
    "gemma-2-2b": 2_614_341_888,
}


@pytest.mark.parametrize("name,expected", REFERENCE_PARAMS.items())
def test_param_count_matches_reference(name, expected):
    arch = Arch.from_config(load_config(name))
    assert abs(arch.param_count() - expected) / expected < 0.001


def test_tiny_param_count(tiny_cfg):
    arch = Arch.from_config(tiny_cfg)
    # embed 16384 + 2 layers * (attn 12288 + mlp 24576 + norms 128) + final norm 64 + lm_head 16384
    assert arch.param_count() == 106_816
    assert arch.q == 64
    assert arch.kv == 32


def test_defaults_by_model_type():
    gemma = Arch.from_config(load_config("gemma-2-2b"))
    assert gemma.tie_word_embeddings is True
    assert gemma.norms_per_layer == 4
    qwen = Arch.from_config(load_config("qwen2.5-7b"))
    assert qwen.attention_bias is True
    llama = Arch.from_config(load_config("llama-3.1-8b"))
    assert llama.attention_bias is False
    assert llama.head_dim == 128
    assert llama.name == "Llama-3.1-8B"


def test_unsupported_model_type(tiny_cfg):
    tiny_cfg["model_type"] = "bert"
    with pytest.raises(UnsupportedModelError, match="llama"):
        Arch.from_config(tiny_cfg)


def test_missing_key(tiny_cfg):
    del tiny_cfg["hidden_size"]
    with pytest.raises(ValueError, match="hidden_size"):
        Arch.from_config(tiny_cfg)


@pytest.mark.parametrize("key,value", [
    ("hidden_size", 0),
    ("num_hidden_layers", -1),
    ("vocab_size", 0),
    ("num_key_value_heads", 3),   # 4 % 3 != 0
])
def test_invalid_values(tiny_cfg, key, value):
    tiny_cfg[key] = value
    with pytest.raises(ValueError, match=key):
        Arch.from_config(tiny_cfg)


def test_head_dim_must_divide():
    cfg = {"model_type": "llama", "hidden_size": 100, "num_hidden_layers": 1,
           "num_attention_heads": 3, "intermediate_size": 8, "vocab_size": 8}
    with pytest.raises(ValueError, match="num_attention_heads"):
        Arch.from_config(cfg)
