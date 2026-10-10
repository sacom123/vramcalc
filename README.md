# vramcalc

[English](#english) | [한국어](#한국어)

---

## English

Predict how much GPU memory an LLM needs before you download the weights or rent the GPU. vramcalc reads only the Hugging Face `config.json`, tells you whether the model fits a given GPU, and when it does not, which single change would make it fit. A `bench` subcommand measures real memory on your GPU and compares it with the prediction, bucket by bucket.

```python
from vramcalc import estimate

e = estimate("meta-llama/Llama-3.1-8B", mode="train", lora_rank=16,
             precision="pure", seq_len=4096)
print(e.breakdown())
r = e.fits("RTX 3090")
print(r.message)
for s in r.suggestions:
    print(s.change, s.margin_bytes)
```

```
$ vramcalc meta-llama/Llama-3.1-8B --train --lora 16 --precision pure --seq 4096 --gpu "RTX 3090"
weights            16.12 GB
gradients           0.05 GB
optimizer           0.11 GB
activations        36.92 GB
step_workspace      0.05 GB
total              53.19 GB
overhead            0.75 GB

short by 28.9 GB on RTX 3090; 1 single adjustment fits
  -> seq_len=512: 21.6 GB (changes training objective)
```

### Install

| Command | What you get |
|---|---|
| `pip install vramcalc` | Estimator and CLI. No dependencies. Takes a local `config.json` or a model directory |
| `pip install "vramcalc[hf]"` | Hugging Face Hub repo ids as input (`huggingface_hub`) |
| `pip install "vramcalc[bench]"` | `vramcalc bench` for real measurements (`torch`, `transformers`, `peft`, `nvidia-ml-py`) |

Offline or on-premises: pass a model directory, `estimate("/path/to/model")`. With `HF_HUB_OFFLINE=1`, Hub ids resolve from the local cache only.

### Why

"Will this model run on this GPU with these settings?" is a question you want answered before downloading 16 GB of weights or paying for an H100 hour. Existing calculators mostly size the weights and treat activations, optimizer state and autocast copies loosely, which is where training estimates drift by several gigabytes. vramcalc keeps one formula per memory term and ships a `bench` command that measures the same configuration on a real GPU, so every term can be checked against the allocator. The formulas are fixed; there is no learned correction factor. Measurements are evidence for fixing a formula, not a fudge applied on top.

### Scope

- Models: dense decoder-only models with `model_type` of `llama`, `mistral`, `qwen2`, `gemma` or `gemma2`. GQA, QKV bias (Qwen2), tied embeddings and Gemma 2's four norms per layer are read from the config.
- Modes: inference (with KV cache), full fine-tuning, LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Knobs: `dtype` (fp32/bf16/fp16), `batch`, `seq_len`, `grad_checkpoint`, `optimizer` (adamw/sgd), `precision` (amp/pure), `attn_impl` (eager/sdpa/flash).

`precision="amp"` models HF Trainer with `bf16=True` (fp32 weights, bf16 autocast). `precision="pure"` models loading with `torch_dtype=bf16` and training as is. The two differ by almost 2x in memory, so pick the one you actually run.

Out of scope for 0.1: MoE, MLA, quantization (int8/int4/QLoRA), distributed training (ZeRO/FSDP), multimodal and encoder models, and sliding-window KV caches (currently counted as global attention, which is conservative).

### What is counted

The prediction targets PyTorch's caching allocator (`allocated` memory). CUDA context and other memory outside the allocator is kept separate as `overhead` (default 0.75 GB) and only added in `fits()`.

| Term | Meaning |
|---|---|
| `weights` | parameter count x storage dtype; for LoRA, frozen base plus fp32 adapters |
| `kv_cache` | inference only, `2 x L x b x s x kv x bytes` |
| `gradients`, `optimizer` | over trainable parameters; AdamW keeps two moments, SGD one |
| `activations` | per-layer tensors saved for backward (SwiGLU, RMSNorm) plus the fp32 logits chain at the `log_softmax` backward peak |
| `autocast_cache` | bf16 weight copies alive during an amp forward pass |
| `step_workspace` | temporary buffer of the foreach optimizer during `step()` |

`total = weights + kv_cache + gradients + optimizer + max(activations + autocast_cache, step_workspace)`. The full formulas are in `docs/design.md`, section 5.

### Validation

`vramcalc bench` loads the model with the same arguments, records `torch.cuda.max_memory_allocated()`, and compares five buckets: weights, states, peak, total, overhead.

```
vramcalc bench /path/to/Llama-3.2-1B --train --precision pure --record benchmarks/results.jsonl
```

Environment: NVIDIA RTX A6000 48 GB, torch 2.14.1+cu130, transformers 4.57.6, peft 0.21.2. Llama weights are gated, so the `unsloth/` mirrors of the same checkpoints were used. All 19 runs land within +/-4.3% on `total`; `weights` is exact in every run.

| model | mode | setting | GPU | predicted | measured | error |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | inference | bf16 b=1 s=2048 sdpa | RTX A6000 | 3.08 GB | 3.08 GB | -0.1% |
| Llama-3.2-1B | inference | bf16 b=1 s=4096 sdpa | RTX A6000 | 3.69 GB | 3.68 GB | +0.2% |
| Llama-3.2-1B | inference | bf16 b=1 s=2048 eager | RTX A6000 | 3.90 GB | 3.95 GB | -1.2% |
| Llama-3.2-3B | inference | bf16 b=1 s=2048 sdpa | RTX A6000 | 7.21 GB | 7.21 GB | +0.0% |
| Llama-3.1-8B | inference | bf16 b=1 s=2048 sdpa | RTX A6000 | 16.89 GB | 16.88 GB | +0.0% |
| Qwen2.5-1.5B | inference | bf16 b=1 s=2048 sdpa | RTX A6000 | 3.78 GB | 3.78 GB | -0.1% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa full pure | RTX A6000 | 17.52 GB | 17.51 GB | +0.0% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa full pure ckpt | RTX A6000 | 13.95 GB | 13.76 GB | +1.4% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa full amp | RTX A6000 | 29.88 GB | 30.55 GB | -2.2% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa full amp ckpt | RTX A6000 | 26.30 GB | 25.22 GB | +4.3% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 eager full pure | RTX A6000 | 30.41 GB | 30.26 GB | +0.5% |
| Llama-3.2-3B | train | bf16 b=1 s=2048 sdpa full pure ckpt | RTX A6000 | 32.13 GB | 32.15 GB | -0.1% |
| Qwen2.5-1.5B | train | bf16 b=1 s=2048 sdpa full pure | RTX A6000 | 23.17 GB | 23.40 GB | -1.0% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa lora=16 pure | RTX A6000 | 10.17 GB | 10.13 GB | +0.4% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa lora=16 amp | RTX A6000 | 15.11 GB | 14.52 GB | +4.1% |
| Llama-3.2-1B | train | bf16 b=1 s=2048 sdpa lora=16 amp ckpt | RTX A6000 | 9.22 GB | 9.51 GB | -3.1% |
| Llama-3.2-3B | train | bf16 b=1 s=2048 sdpa lora=16 pure | RTX A6000 | 18.84 GB | 18.98 GB | -0.8% |
| Llama-3.1-8B | train | bf16 b=1 s=2048 sdpa lora=16 pure ckpt | RTX A6000 | 20.97 GB | 20.55 GB | +2.0% |
| Qwen2.5-1.5B | train | bf16 b=1 s=2048 sdpa lora=16 pure | RTX A6000 | 13.99 GB | 13.78 GB | +1.5% |

`pure` is `torch_dtype=bf16` training, `amp` is fp32 weights with bf16 autocast, `ckpt` is gradient checkpointing. Per-bucket numbers are in `benchmarks/results.jsonl`; regenerate the table with `python benchmarks/make_table.py`.

What the measurements changed in the formulas: the inference peak is just `kv_cache + logits` (per-layer buffers are freed as the forward proceeds); the training logits peak is 14 bytes per logit, not 10, because `log_softmax` backward holds its saved output and both gradient tensors at once; and each layer saves about two more hidden-sized tensors than the textbook count.

### Limits

- `bench` uses `torch.autocast` directly, not HF Trainer. Extra conversions done by Accelerate are not modelled.
- `fits()` assumes nothing else is using the GPU and takes 97% of the nominal capacity as usable.
- `attn_impl="sdpa"` assumes the memory-efficient or flash backend. If your environment falls back to the math backend, use `eager`.
- Overhead was measured on an A6000 with CUDA 13; other cards and driver versions may differ by a few hundred megabytes.

### License

MIT

---

## 한국어

모델 가중치를 받기 전, GPU를 빌리기 전에 LLM이 얼마만큼의 GPU 메모리를 필요로 하는지 예측하는 라이브러리입니다. Hugging Face `config.json`만 읽어 지정한 GPU에 들어가는지 판정하고, 들어가지 않으면 무엇을 하나 바꾸면 들어가는지 제안합니다. `bench` 명령은 실제 GPU에서 메모리를 측정해 예측과 항목별로 비교합니다.

```python
from vramcalc import estimate

e = estimate("meta-llama/Llama-3.1-8B", mode="train", lora_rank=16,
             precision="pure", seq_len=4096)
print(e.breakdown())
r = e.fits("RTX 3090")
print(r.message)
for s in r.suggestions:
    print(s.change, s.margin_bytes)
```

```
$ vramcalc meta-llama/Llama-3.1-8B --train --lora 16 --precision pure --seq 4096 --gpu "RTX 3090"
weights            16.12 GB
gradients           0.05 GB
optimizer           0.11 GB
activations        36.92 GB
step_workspace      0.05 GB
total              53.19 GB
overhead            0.75 GB

short by 28.9 GB on RTX 3090; 1 single adjustment fits
  -> seq_len=512: 21.6 GB (changes training objective)
```

### 설치

| 명령 | 포함 내용 |
|---|---|
| `pip install vramcalc` | 계산기와 CLI. 의존성 없음. 로컬 `config.json` 또는 모델 디렉터리 입력 |
| `pip install "vramcalc[hf]"` | Hugging Face Hub repo id 입력 (`huggingface_hub`) |
| `pip install "vramcalc[bench]"` | 실측 비교 `vramcalc bench` (`torch`, `transformers`, `peft`, `nvidia-ml-py`) |

오프라인이나 온프레미스 환경에서는 `estimate("/path/to/model")`처럼 모델 디렉터리를 넘기면 됩니다. `HF_HUB_OFFLINE=1`이면 Hub id도 로컬 캐시만 조회합니다.

### 만든 이유

"이 모델을 이 설정으로 이 GPU에서 돌릴 수 있는가"는 16GB 가중치를 받거나 H100을 한 시간 빌리기 전에 알아야 하는 질문입니다. 기존 계산기는 대부분 가중치 크기 위주이고 activation, optimizer 상태, autocast 복사본은 대략적으로만 다루기 때문에 학습 모드에서 수 GB씩 어긋납니다. vramcalc는 메모리 항목마다 식을 하나씩 두고, 같은 설정을 실제 GPU에서 측정하는 `bench` 명령을 함께 제공해 항목별로 allocator 값과 대조할 수 있게 했습니다. 식은 고정이며 보정 계수를 학습하지 않습니다. 실측은 식을 고치는 근거이지, 위에 덧씌우는 보정이 아닙니다.

### 지원 범위

- 모델: `model_type`이 `llama`, `mistral`, `qwen2`, `gemma`, `gemma2`인 dense decoder-only 모델. GQA, QKV bias(Qwen2), 임베딩 공유, Gemma 2의 레이어당 norm 4개를 config에서 읽어 반영합니다.
- 모드: 추론(KV cache 포함), 전체 학습, LoRA(`q_proj`, `k_proj`, `v_proj`, `o_proj` 대상)
- 옵션: `dtype`(fp32/bf16/fp16), `batch`, `seq_len`, `grad_checkpoint`, `optimizer`(adamw/sgd), `precision`(amp/pure), `attn_impl`(eager/sdpa/flash)

`precision="amp"`는 HF Trainer `bf16=True` 방식(fp32 가중치, bf16 autocast)이고, `precision="pure"`는 `torch_dtype=bf16`으로 로드해 그대로 학습하는 방식입니다. 둘의 메모리는 2배 가까이 차이가 나므로 실제로 사용하는 방식을 지정해야 합니다.

0.1에서 제외된 항목: MoE, MLA, 양자화(int8/int4/QLoRA), 분산 학습(ZeRO/FSDP), 멀티모달과 encoder 모델, sliding window KV cache(현재는 전부 global attention으로 보수적으로 계산).

### 계산 항목

예측 대상은 PyTorch caching allocator 기준 `allocated` 메모리입니다. CUDA context 등 allocator 밖 메모리는 `overhead`(기본 0.75GB)로 분리해 `fits()` 판정에서만 더합니다.

| 항목 | 내용 |
|---|---|
| `weights` | 파라미터 수 × 저장 dtype. LoRA는 frozen base + fp32 어댑터 |
| `kv_cache` | 추론 전용. `2 × L × b × s × kv × bytes` |
| `gradients`, `optimizer` | 학습 대상 파라미터 기준. AdamW는 상태 2개, SGD는 momentum 1개 |
| `activations` | 레이어당 역전파 저장분(SwiGLU, RMSNorm) + `log_softmax` 역전파 시점의 fp32 logits 체인 |
| `autocast_cache` | amp에서 forward 동안 살아 있는 bf16 가중치 복사본 |
| `step_workspace` | foreach optimizer의 `step()` 중 임시 버퍼 |

`total = weights + kv_cache + gradients + optimizer + max(activations + autocast_cache, step_workspace)`. 상세 식은 `docs/design.md` 5절에 있습니다.

### 실측 검증

`vramcalc bench`는 같은 인자로 모델을 실제로 로드해 `torch.cuda.max_memory_allocated()`를 기록하고 weights, states, peak, total, overhead 다섯 묶음으로 예측과 비교합니다.

```
vramcalc bench /path/to/Llama-3.2-1B --train --precision pure --record benchmarks/results.jsonl
```

측정 환경: NVIDIA RTX A6000 48GB, torch 2.14.1+cu130, transformers 4.57.6, peft 0.21.2. Llama 가중치는 gated라 동일 가중치의 `unsloth/` 미러를 사용했습니다. 19개 조합 모두 `total` 오차 ±4.3% 안이며 `weights`는 전부 0.0%입니다. 표는 위 영어 절과 같고, 항목별 수치는 `benchmarks/results.jsonl`에 있습니다. `python benchmarks/make_table.py`로 표를 다시 생성할 수 있습니다.

실측으로 바뀐 식: 추론 peak는 `kv_cache + logits`만으로 설명되고(레이어 버퍼는 forward 진행 중 해제됨), 학습 logits peak는 logit당 10바이트가 아니라 14바이트이며(`log_softmax` 역전파가 저장 출력과 두 gradient를 동시에 보유), 레이어마다 교과서 계산보다 hidden 크기 텐서 약 2개가 더 저장됩니다.

### 한계

- `bench`는 HF Trainer가 아니라 `torch.autocast`를 직접 사용합니다. Accelerate가 추가로 수행하는 변환은 반영하지 않습니다.
- `fits()`는 GPU에 다른 프로세스가 없다고 가정하고 표기 용량의 97%를 사용 가능 용량으로 봅니다.
- `attn_impl="sdpa"`는 memory-efficient 또는 flash backend를 전제로 합니다. math backend로 떨어지는 환경이면 `eager`로 지정해야 합니다.
- overhead는 A6000과 CUDA 13에서 측정한 값이라 다른 카드와 드라이버에서는 수백 MB 차이가 날 수 있습니다.

### 라이선스

MIT
