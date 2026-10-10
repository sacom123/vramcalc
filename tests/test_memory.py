import pytest

from vramcalc import memory
from vramcalc.arch import Arch

from .conftest import load_config


@pytest.fixture
def tiny(tiny_cfg) -> Arch:
    return Arch.from_config(tiny_cfg)


def test_weights(tiny):
    assert memory.weights(tiny, 2) == 106_816 * 2


def test_kv_cache(tiny):
    # 2 * L(2) * b(1) * s(8) * kv(32) * 2 bytes
    assert memory.kv_cache(tiny, 1, 8, 2) == 2048


def test_per_layer_act_sdpa(tiny):
    # s*b*(2h + q + 2kv + q + h + 4i + 2h) = 8*(128+64+64+64+64+512+128) = 8192,
    # plus gqa_expand 2*8*64 = 1024
    assert memory.per_layer_act(tiny, 1, 8, "sdpa", 2) == 9216


def test_per_layer_act_flash_has_no_gqa_expand(tiny):
    assert memory.per_layer_act(tiny, 1, 8, "flash", 2) == 8192


def test_per_layer_act_eager_adds_scores(tiny):
    # eager: + a*s*s*b*(B+4)/B = 4*64*6/2 = 768
    assert memory.per_layer_act(tiny, 1, 8, "eager", 2) == 9216 + 768


def test_inference_activations(tiny):
    # logits b*s*V*B = 4096 + working set 2*s*b*h*B = 2048; no per-layer term survives to the peak
    assert memory.inference_activations(tiny, 1, 8, "sdpa", 2) == 6144


def test_inference_activations_eager(tiny):
    # + scores a*s*s*b*(B+4) = 1536 + repeat_kv 2*s*b*q*B = 2048
    assert memory.inference_activations(tiny, 1, 8, "eager", 2) == 6144 + 1536 + 2048


def test_lora_params(tiny):
    # L*r*(4h + 2q + 2kv) = 2*4*(256+128+64)
    assert memory.lora_params(tiny, 4) == 3584


def test_cast_params_includes_output_linear_once(tiny):
    # L*(attn+mlp) + V*h = 2*(12288+24576) + 16384
    assert memory.cast_params(tiny) == 90_112


def test_cast_params_tied_model_still_counts_lm_head():
    tied = Arch.from_config(load_config("llama-3.2-1b"))
    untied = Arch.from_config({**load_config("llama-3.2-1b"), "tie_word_embeddings": False})
    assert memory.cast_params(tied) == memory.cast_params(untied)
    assert tied.param_count() < untied.param_count()


def test_states():
    assert memory.gradients(100, 4) == 400
    assert memory.optimizer_states(100, "adamw", 4) == 800
    assert memory.optimizer_states(100, "sgd", 4) == 400
    assert memory.step_workspace(100, 4) == 400


def test_autocast_cache(tiny):
    full = memory.cast_params(tiny) * 2
    assert memory.autocast_cache(tiny, "amp", 2, lora=False, grad_checkpoint=True) == full
    assert memory.autocast_cache(tiny, "amp", 2, lora=True, grad_checkpoint=False) == full
    assert memory.autocast_cache(tiny, "amp", 2, lora=True, grad_checkpoint=True) == full // 2
    assert memory.autocast_cache(tiny, "pure", 2, lora=False, grad_checkpoint=False) == 0


def test_train_activations_full(tiny):
    # L*per_layer_act*B = 2*9216*2 = 36864; norm_fp32 = 2*4*8*64*2 = 8192; logits = 8*256*14 = 28672
    assert memory.train_activations(
        tiny, 1, 8, "sdpa", 2, grad_checkpoint=False, lora_rank=None) == 73_728


def test_train_activations_checkpoint(tiny):
    # L*s*b*h*B = 2048; per_layer_act*B = 18432; norm_fp32/L = 4096; logits = 28672
    assert memory.train_activations(
        tiny, 1, 8, "sdpa", 2, grad_checkpoint=True, lora_rank=None) == 53_248


def test_train_activations_lora_branch(tiny):
    # + 4*L*b*s*r*4 = 4*2*8*4*4 = 1024
    base = memory.train_activations(tiny, 1, 8, "sdpa", 2, grad_checkpoint=False, lora_rank=None)
    assert memory.train_activations(
        tiny, 1, 8, "sdpa", 2, grad_checkpoint=False, lora_rank=4) == base + 1024
