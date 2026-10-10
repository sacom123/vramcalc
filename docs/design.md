# vramcalc 설계 문서

작성일: 2026-10-10
상태: 1차 실측 반영 완료 (2026-10-10)

## 1. 목적

Hugging Face 형식의 `config.json`만으로 decoder-only Transformer 모델의 GPU 메모리 사용량을 실행 전에 예측하고, 지정한 GPU에 들어가는지 판정하며, 들어가지 않을 때 무엇을 바꾸면 들어가는지 제안하는 Python 라이브러리입니다.

핵심 질문은 "이 모델을 이 설정으로 이 GPU에서 돌릴 수 있는가"입니다. 모델을 받기 전, GPU를 빌리기 전에 답을 내는 것이 목적이므로 예측 정확도가 곧 라이브러리의 가치입니다. 정확도는 실측 비교 기능(`bench`)으로 검증하고, 검증 결과를 README에 공개합니다.

### 1.1 성공 기준

- 지원 모델의 파라미터 수가 HF 공개 수치(safetensors 메타데이터의 정확한 수)와 ±0.1% 안에서 일치합니다.
- RTX 3090 Ti(24GB)에서 추론, 전체 학습, LoRA 각 모드의 `total` 예측 오차가 실측 대비 ±10% 안에 들어옵니다.
- 기본 설치는 표준 라이브러리만 사용하며 오프라인 환경에서 로컬 config.json으로 동작합니다.
- PyPI에 배포되고 GitHub Actions CI가 통과합니다.

### 1.2 1차 범위 밖 (이후 버전)

- MoE(Mixtral, Qwen-MoE, DeepSeek), MLA(DeepSeek V2/3)
- 양자화(int8, int4, QLoRA)
- 분산 학습(ZeRO, FSDP, tensor parallel)
- 멀티모달, encoder 및 encoder-decoder 모델
- Gemma 2의 local/global 교차 attention을 레이어별로 구분한 KV cache 계산
- HF Trainer 실행과의 완전한 동일성. bench는 standalone AMP(`torch.autocast`)를 기준으로 하며, Accelerate가 추가로 만드는 fp32 출력 변환 등은 한계로 명시합니다.

## 2. 설계 원칙

- 메모리 항목마다 작은 순수 함수를 두고 `estimate()`가 모드에 따라 조합합니다. 각 항목을 독립적으로 테스트하고, 실측 오차가 나면 어느 항목이 틀렸는지 좁힐 수 있어야 합니다.
- 예측식은 고정합니다. 실측으로 보정 계수를 학습하지 않습니다. 실측은 식 자체를 고치는 근거입니다.
- 모델별 차이는 `Arch` 정규화 단계에서 config 키로 흡수합니다. 계산식은 모델을 구분하지 않습니다.
- 예측 대상은 PyTorch caching allocator 기준 `allocated` 메모리입니다. CUDA context 등 allocator 밖 메모리는 `overhead`로 분리하고 `fits()` 판정에서만 더합니다.
- 메모리 항목을 **영구 항목**(측정 구간 내내 살아 있음: weights, kv_cache, gradients, optimizer)과 **peak 항목**(forward 또는 step 중에만 존재: activations, autocast_cache, step_workspace)으로 구분합니다. bench는 영구 항목을 `memory_allocated()` 차분으로, peak 항목을 `max_memory_allocated()` 차분으로 측정합니다.
- 계열이 둘 이상 생길 때까지 클래스 기반 플러그인 구조는 도입하지 않습니다.

## 3. 패키지 구조

```
vramcalc/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE                      # MIT
├── .github/workflows/ci.yml
├── src/vramcalc/
│   ├── __init__.py              # estimate, bench, Arch, Estimate, FitResult 공개
│   ├── arch.py                  # Arch dataclass, config.json 정규화, param_count
│   ├── loaders.py               # dict / 로컬 경로 / HF Hub 입력 해석
│   ├── memory.py                # 항목별 계산 함수 (순수 함수, bytes 단위)
│   ├── estimate.py              # Estimate, FitResult, estimate() 조합
│   ├── gpus.py                  # 내장 GPU 테이블, pynvml 자동 감지
│   ├── suggest.py               # 조정 제안
│   ├── bench.py                 # 실측 (torch 지연 import)
│   └── cli.py                   # argparse CLI
├── benchmarks/
│   ├── results.jsonl            # 실측 기록 (저장소 검증표 원본)
│   └── make_table.py            # results.jsonl에서 README 표 생성
└── tests/
    ├── configs/                 # 실제 config.json 8개
    ├── test_arch.py
    ├── test_memory.py
    ├── test_estimate.py
    ├── test_suggest.py
    ├── test_cli.py
    └── test_bench_format.py
```

### 3.1 의존성

| extra | 패키지 | 용도 |
|---|---|---|
| (기본) | 없음 | 계산, 로컬 config, CLI |
| `hf` | `huggingface_hub` | HF Hub repo id 입력 |
| `bench` | `torch>=2.3`, `transformers>=4.46,<5`, `peft>=0.12`, `nvidia-ml-py`(import 이름은 `pynvml`) | 실측, GPU 자동 감지. transformers 4.45는 추론 logits를 무조건 fp32로 upcast하므로 제외, peft 0.12 미만은 어댑터 dtype 기본값이 다름 |
| `dev` | `pytest`, `ruff` | 개발 |

`flash-attn`은 extra에 포함하지 않습니다. `attn_impl="flash"` bench는 사용자가 직접 설치한 경우에만 동작하며, 없으면 `ImportError`를 냅니다.

Python 최소 버전은 3.10입니다. 빌드 백엔드는 hatchling, 패키지 관리는 uv를 사용합니다.

## 4. 공개 API

### 4.1 `Arch`

config.json을 계산에 필요한 필드만 뽑아 정규화한 불변 dataclass입니다.

```python
@dataclass(frozen=True)
class Arch:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int            # 없으면 num_heads
    head_dim: int                # 없으면 hidden_size // num_heads
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool         # Qwen2 계열 True
    norms_per_layer: int         # 기본 2, gemma2는 4
    name: str = ""

    @classmethod
    def from_config(cls, cfg: dict) -> "Arch": ...
    def param_count(self) -> int: ...
```

유도 값: `q = num_heads × head_dim`, `kv = num_kv_heads × head_dim`.

지원 `model_type` allowlist: `llama`, `mistral`, `qwen2`, `gemma`, `gemma2`. 목록 밖이면 `UnsupportedModelError`를 올립니다. 경고 후 진행하는 fallback은 두지 않습니다.

`sliding_window`는 `Arch`에 두지 않습니다. 1차에서는 Mistral과 Gemma 2의 sliding window를 무시하고 모든 레이어를 global attention으로 보수적으로 계산합니다. 이 정책은 1.2에 범위 밖으로 명시되어 있습니다.

`from_config` 검증 규칙:
- `hidden_size`, `num_layers`, `num_heads`, `num_kv_heads`, `head_dim`, `intermediate_size`, `vocab_size`는 양의 정수
- `num_heads % num_kv_heads == 0`
- `head_dim` 미지정 시 `hidden_size % num_heads == 0`
- 위반 시 `ValueError`에 위반한 키 이름

### 4.2 `estimate()`

```python
def estimate(
    model,                        # dict | str | Path
    *,
    mode: str = "inference",      # "inference" | "train"
    dtype: str = "bf16",          # "fp32" | "bf16" | "fp16"
    batch: int = 1,
    seq_len: int = 2048,
    lora_rank: int | None = None,
    grad_checkpoint: bool = False,
    optimizer: str = "adamw",     # "adamw" | "sgd"
    precision: str = "amp",       # "amp" | "pure" (학습만 적용)
    attn_impl: str = "sdpa",      # "eager" | "sdpa" | "flash"
    overhead_gb: float = 0.5,
) -> Estimate
```

`attn_impl` 의미:
- `"eager"`: score 행렬과 fp32 softmax를 저장하는 구현. transformers `attn_implementation="eager"`와 SDPA의 math backend가 여기에 해당합니다.
- `"sdpa"`: memory-efficient 또는 flash backend로 고정된 SDPA. score 행렬을 저장하지 않지만 GQA K/V 확장은 발생합니다.
- `"flash"`: `flash-attn` 패키지. score 행렬도 GQA 확장도 없습니다.

인자 검증 규칙:
- `batch`, `seq_len`, `lora_rank`(지정 시)는 양의 정수
- `overhead_gb`는 0 이상의 유한한 수
- `lora_rank`와 `mode="inference"` 동시 지정은 `ValueError`
- `dtype="fp32"`와 `precision="amp"`는 `pure`로 취급합니다 (autocast 대상 없음)

`model` 입력 해석(`loaders.py`):

| 입력 | 처리 |
|---|---|
| `dict` | 그대로 `Arch.from_config` |
| `config.json` 파일 경로 | 읽어서 파싱 |
| 디렉터리 경로 | 안의 `config.json` 탐색 |
| 그 외 문자열 | HF repo id로 간주. `huggingface_hub` 없으면 `pip install vramcalc[hf]` 안내 에러 |

`HF_HUB_OFFLINE=1` 환경에서는 `huggingface_hub`가 캐시만 조회하므로 추가 처리 없이 동작합니다.

### 4.3 결과 객체

```python
@dataclass(frozen=True)
class EstimateParams:
    arch: Arch
    mode: str
    dtype: str
    batch: int
    seq_len: int
    lora_rank: int | None
    grad_checkpoint: bool
    optimizer: str
    precision: str
    attn_impl: str
    overhead_gb: float

@dataclass(frozen=True)
class Estimate:
    # 영구 항목 (bytes)
    weights: int
    kv_cache: int
    gradients: int
    optimizer: int
    # peak 항목 (bytes)
    activations: int
    autocast_cache: int
    step_workspace: int
    # allocator 밖
    overhead: int
    params: EstimateParams

    @property
    def persistent(self) -> int   # weights + kv_cache + gradients + optimizer
    @property
    def peak_extra(self) -> int   # max(forward peak, step peak), 5.6 참조
    @property
    def total(self) -> int        # persistent + peak_extra, overhead 제외
    def fits(self, gpu=None) -> FitResult
    def breakdown(self) -> str    # 항목별 GB 표, GPU 무관
    def to_dict(self) -> dict     # 메모리 필드는 최상위, params는 "params" 키 아래 중첩 (optimizer 키 충돌 방지)

@dataclass(frozen=True)
class FitResult:
    ok: bool
    gpu_name: str
    gpu_bytes: int
    usable_bytes: int
    required_bytes: int           # total + overhead
    margin_bytes: int             # usable - required, 음수면 부족
    suggestions: list[Suggestion]
    message: str                  # 판정 요약, 실패 시 안내 문구

@dataclass(frozen=True)
class Suggestion:
    change: dict                  # 바뀐 인자만
    changes_objective: bool       # LoRA, sgd, pure처럼 학습 결과가 달라지는 조정이면 True
    estimate: Estimate
    margin_bytes: int
```

내부 단위는 bytes 정수이고 출력 시에만 GB(10^9)로 변환합니다.

## 5. 메모리 계산식

기호: `L` 레이어 수, `h` hidden, `a` 헤드 수, `q = a × head_dim`, `kv = num_kv_heads × head_dim`, `i` intermediate, `V` vocab, `b` batch, `s` seq_len, `n` norms_per_layer, `P` 전체 파라미터 수, `P_t` 학습 대상 파라미터 수, `P_cast` autocast 대상 파라미터 수, `B = bytes(dtype)` (bf16/fp16은 2, fp32는 4).

### 5.1 파라미터 수

```
embed      = V × h
attn       = h×q (q_proj) + 2×h×kv (k_proj, v_proj) + q×h (o_proj)
attn_bias  = q + 2×kv                  (attention_bias일 때만)
mlp        = 3 × h × i                 (gate, up, down)
norms      = n × h
per_layer  = attn + attn_bias + mlp + norms
lm_head    = 0 if tie_word_embeddings else V × h
P          = embed + L × per_layer + h + lm_head
P_cast     = L × (attn + mlp) + V × h     (autocast 대상 linear 가중치. 출력 linear는 tie 여부와 무관하게 한 번 캐스팅되므로 항상 포함)
```

### 5.2 추론

```
weights     = P × B
kv_cache    = 2 × L × b × s × kv × B
activations = b × s × V × B                      # prefill logits
            + 2 × b × s × h × B                  # 작업 집합
            + [eager] a × s × s × b × (B + 4) + 2 × b × s × q × B   # 한 레이어분 score, fp32 softmax, repeat_kv
total       = weights + kv_cache + activations
```

실측(A6000, 1B~8B) 결과 추론 peak는 `kv_cache + logits`로 30MB 안에서 설명됩니다. 레이어 버퍼는 forward가 진행되며 해제되어 peak에 남지 않으므로 항에서 제외합니다. 추론 모드에서 `gradients`, `optimizer`, `autocast_cache`, `step_workspace`는 0이고 `precision`은 무시됩니다.

### 5.3 학습 정밀도

`precision` 인자가 가중치, gradient, optimizer 상태의 저장 dtype을 결정합니다.

- `"amp"`: 모델은 fp32로 유지되고 `torch.autocast`가 linear 가중치의 bf16 복사본을 만들어 연산합니다. gradient와 optimizer 상태는 fp32입니다. HF Trainer `bf16=True`의 기본 동작과 같은 방식이지만, Accelerate의 추가 변환까지 동일하다고 보장하지는 않습니다 (1.2 참조).
- `"pure"`: `torch_dtype=bf16`으로 로드해 그대로 학습하는 방식. 모든 상태가 `dtype`입니다.

```
S (state_bytes) = 4 if precision == "amp" else B
autocast_cache (amp, 전체 학습)            = P_cast × B        # autocast 캐시가 모든 cast 복사본을 forward 동안 유지
autocast_cache (amp, LoRA, checkpoint 없음) = P_cast × B        # frozen 가중치는 캐시되지 않지만 autograd가 backward용으로 cast 텐서를 저장
autocast_cache (amp, LoRA, checkpoint 있음) = P_cast × B / L    # 재계산 중인 한 레이어분만 살아 있음
autocast_cache (pure)                      = 0
```

### 5.4 학습 상태

```
P_t (전체 학습) = P
P_t (LoRA)      = L × r × (4h + 2q + 2kv)       # q,k,v,o 대상, in+out 합
adapter_bytes   = 4 if LoRA else S                # PEFT 어댑터 기본 fp32

weights   = P × S                       (전체 학습)
          = P × S + P_t × 4             (LoRA: frozen base + fp32 어댑터)
gradients = P_t × adapter_bytes
optimizer = adamw: P_t × 2 × adapter_bytes        # m, v (영구)
            sgd:   P_t × adapter_bytes            # momentum 0.9 (영구)
step_workspace = P_t × adapter_bytes              # foreach 구현 임시 버퍼, step 중에만 존재
```

### 5.5 Activations (학습)

레이어당 역전파용 저장분을 element 수로 정의하고 dtype bytes를 곱합니다. SwiGLU MLP, RMSNorm, dropout 없음을 가정합니다. Gemma 2는 norm이 4개지만 저장되는 norm 입력은 residual 분기당 하나로 보고 `2h`를 유지합니다.

```
per_layer_act (elements)
  = s × b × (2h            # 두 residual 분기의 norm 입력
           + q + 2kv       # q, k, v 출력
           + q             # attention 출력 (o_proj 입력)
           + h             # MLP 입력
           + 4i            # gate 사전활성, 활성 후 gate, up 출력, gate×up
           + 2h)           # 실측 잔차 (A6000, 1B/3B/Qwen2.5-1.5B). 이름 붙은 텐서로 분해는 추후
  + gqa_expand
  + attn_scores

gqa_expand  = 2 × s × b × q        if attn_impl in ("sdpa", "eager") and kv < q else 0
                                    # transformers repeat_kv로 K/V를 q 폭으로 확장
attn_scores = a × s × s × b × (B + 4) / B   if attn_impl == "eager" else 0
                                    # score 행렬 + fp32 softmax. element 단위로 맞추기 위해 B로 나눔

norm_fp32   = NORM_FP32_TENSORS × 4 × s × b × h × L    # RMSNorm fp32 중간 텐서, 기본 2개 (bench로 확정)
lora_branch = 4 × L × b × s × r × 4                     # rank-r 중간 텐서 4개 (q,k,v,o), fp32
logits      = b × s × V × (B + 4 + 4 + 4)               # log_softmax 역전파 시점: logits + 저장된 fp32 log_softmax + fp32 grad_out + fp32 grad_in
                                                        # (fp32 upcast 자체는 역전파 전에 해제됨. 실측으로 10N → 14N 수정)

activations (checkpoint 없음)
  = L × per_layer_act × B + norm_fp32 + lora_branch + logits
activations (grad_checkpoint)
  = L × s × b × h × B + per_layer_act × B + norm_fp32 / L + lora_branch + logits
    # 레이어 입력만 저장, 재계산 시 한 레이어분
```

`NORM_FP32_TENSORS`는 `memory.py` 모듈 상수로 두고 bench 결과로 확정합니다. 1차 기본값은 2입니다.

### 5.6 합계

peak 항목은 서로 다른 시점에 존재합니다. forward/backward peak에는 activations와 autocast_cache가, optimizer step peak에는 step_workspace가 있습니다. 둘 중 큰 쪽이 실제 peak입니다.

```
persistent (inference) = weights + kv_cache
persistent (train)     = weights + gradients + optimizer
peak_extra (inference) = activations
peak_extra (train)     = max(activations + autocast_cache, step_workspace)
total                  = persistent + peak_extra
required (fits)        = total + overhead
```

`overhead`는 `overhead_gb` 인자(기본 0.75GB)로, CUDA context와 cuBLAS workspace 등 allocator 밖 메모리를 뜻합니다. A6000 + torch 2.14/cu130 실측값 0.74GB 기준입니다.

### 5.7 검증 전제

위 식의 상수, 특히 `per_layer_act` 항과 `NORM_FP32_TENSORS`, `overhead_gb` 기본값은 bench 실측으로 조정될 수 있습니다. 항목별 오차가 10%를 넘으면 해당 항목만 op 단위로 세분화합니다. 처음부터 transformers 구현을 op 단위로 따라가지 않는 이유는 1차 범위가 Llama 계열 하나이고, 실측으로 틀린 항을 특정할 수 있기 때문입니다.

## 6. GPU 테이블과 판정

### 6.1 테이블 (`gpus.py`)

`dict[str, int]`로 GPU 이름과 VRAM을 **bytes**로 보관합니다. 값은 `표기 GiB × 2^30`이며, 예를 들어 RTX 3090은 `24 × 2^30`입니다. 약 35종: RTX 3060/3070/3080/3090/3090 Ti, RTX 4060~4090, RTX 5070~5090, T4, V100 16/32, A10, A40, A100 40/80, H100 80, H200, L4, L40S, B200. 조회는 대소문자, 공백, 하이픈을 무시하고 `"rtx3090"`, `"RTX 3090"`, `"3090"`을 같은 항목으로 잡습니다. 모르는 이름은 `KeyError`에 유사 이름 후보를 붙여 올립니다.

### 6.2 `fits(gpu)`

| `gpu` 인자 | 처리 |
|---|---|
| `None` | `pynvml`로 장착 GPU 0번의 **전체** VRAM 조회 (여유 용량이 아님). 실패 시 "GPU 이름이나 GB 숫자를 넘기십시오" `RuntimeError` |
| `int`, `float` | GB 숫자(10^9). 양의 유한한 수가 아니면 `ValueError` |
| `str` | 테이블 조회 |

```
usable = gpu_bytes × USABLE_RATIO     # USABLE_RATIO = 0.97, 유휴 GPU 기준 임시 안전 여유. bench로 확정
ok     = required <= usable
```

`USABLE_RATIO`는 `gpus.py` 모듈 상수입니다. 다른 프로세스가 GPU를 쓰는 상황은 고려하지 않습니다. A6000 실측에서 `mem_get_info` total은 표기 용량의 98.8%였으므로 0.97은 약 1.8%의 여유를 더 둔 값입니다.

### 6.3 제안 (`suggest.py`)

`fits()`가 실패하면 아래 후보를 **각각 독립적으로** 적용해 `estimate()`를 다시 호출하고, 들어가는 후보를 `suggestions`로 반환합니다. 후보를 조합하지는 않습니다.

| 순서 | 조정 | 조건 | `changes_objective` |
|---|---|---|---|
| 1 | `grad_checkpoint=True` | 학습, 미적용 시 | False |
| 2 | `batch` 절반씩 축소, 최소 1 | batch > 1 | 학습 True, 추론 False |
| 3 | `seq_len` 절반씩 축소, 최소 512 | seq_len > 512 | 학습 True, 추론 False |
| 4 | `lora_rank=16` | 전체 학습일 때 | True |
| 5 | `optimizer="sgd"` | adamw일 때 | True |
| 6 | `precision="pure"` | amp일 때 | True |

`changes_objective`는 "학습 결과가 달라지는 조정"을 뜻합니다. 학습에서 batch 축소는 gradient accumulation 보정이 없으면 유효 batch가 바뀌고, seq_len 축소는 학습 데이터 자체가 잘리므로 둘 다 True입니다. 추론에서는 처리량과 문맥 길이만 바뀌므로 False입니다. batch 축소를 seq_len 축소보다 앞에 두는 이유는 데이터를 자르는 쪽이 더 큰 변화이기 때문입니다. 2번과 3번은 절반씩 줄이며 처음 들어가는 값에서 멈춥니다.

어느 후보도 들어가지 않으면 `suggestions`는 비어 있고 `FitResult.message`에 "시험한 단일 조정 중 들어가는 것 없음. 조정을 조합하거나 양자화, 분산 학습을 검토하십시오"를 넣습니다.

## 7. 실측 (`bench.py`)

같은 인자로 예측과 실측을 수행하고 묶음별로 비교합니다. `torch`, `transformers`, `peft`는 함수 안에서 지연 import하며, 없으면 `pip install vramcalc[bench]` 안내 에러를 냅니다. CUDA가 없으면 `RuntimeError`를 냅니다.

```python
def bench(model_path, *, mode, dtype, batch, seq_len, lora_rank, grad_checkpoint,
          optimizer, precision, attn_impl, steps=3, record: Path | None = None) -> BenchResult
```

### 7.1 고정 워크로드

| 항목 | 설정 |
|---|---|
| 모델 로드 dtype | 추론과 pure: `dtype`. amp: fp32 |
| attention | `attn_implementation`을 `eager` / `sdpa` / `flash_attention_2`로 전달. `sdpa`는 `torch.nn.attention.sdpa_kernel([EFFICIENT_ATTENTION, FLASH_ATTENTION])`로 감싸 math backend를 배제 |
| 입력 | `torch.randint(vocab, (b, s))`, labels는 input_ids 복사 |
| 추론 | `model.eval()`, `torch.inference_mode()`, `use_cache=True` |
| 학습 | `model.train()`, `use_cache=False`, grad_checkpoint 시 `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` |
| LoRA | PEFT `LoraConfig(r, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0)`, 어댑터는 weights 스냅샷 **전에** 삽입 |
| optimizer | `AdamW(foreach=True)` 또는 `SGD(momentum=0.9, foreach=True)` |
| amp | forward와 loss만 `torch.autocast("cuda", dtype)` 안에서, backward와 step은 밖에서 |
| step 수 | warm-up 1 step 후 `steps`회 측정 |

### 7.2 측정 절차

각 측정 구간 직전에 `torch.cuda.synchronize()`, `gc.collect()`, `torch.cuda.empty_cache()`, `reset_peak_memory_stats()`를 호출합니다. 이전 구간의 출력 텐서는 `del` 합니다.

1. 모델 로드 (LoRA면 어댑터 삽입까지). `memory_allocated()` 기록: **weights 실측**
2. 추론: forward 1회. `max_memory_allocated() - weights`: **kv_cache + activations 실측**. 끝나면 출력과 cache 삭제
3. 학습:
   1. optimizer 생성, warm-up 1 step 실행
   2. 출력과 loss `del`, `zero_grad(set_to_none=False)`, `synchronize()`. 이 시점의 `memory_allocated()`를 `baseline`으로 저장. `baseline - weights`: **gradients + optimizer 실측** (영구 상태는 첫 step 뒤 모두 할당되고 gradient 버퍼는 0으로 채워진 채 유지됨)
   3. `steps`회 반복. 매 반복은 peak 리셋, forward, backward, step, 출력과 loss `del`, `zero_grad(set_to_none=False)` 순서이며 반복 끝에 `max_memory_allocated() - baseline`을 기록. 반복 중 최댓값: **peak 항목 실측**
4. 종료 직전 `synchronize()` 후 `mem_get_info()`와 `memory_reserved()`를 연속으로 읽고, `(total - free) - reserved`: **overhead 실측**. 장치에 다른 프로세스가 없음을 전제로 하며 결과에 그대로 기록합니다

측정 구간 전체의 `max_memory_allocated()`는 1단계 직전에 한 번 리셋한 별도 카운터 대신, `weights + states + peak`의 합으로 보고합니다. 구간별 리셋과 전체 peak를 동시에 추적할 수 없기 때문입니다.

### 7.3 묶음과 보고

| 묶음 | 예측 | 실측 |
|---|---|---|
| weights | `weights` | 1단계 |
| states | `gradients + optimizer` | 3단계 전반 |
| peak | `peak_extra + kv_cache` (추론) / `peak_extra` (학습) | 2단계 또는 3단계 후반 |
| total | `total` | `weights + states + peak` 실측 합 |
| overhead | `overhead` | 4단계 |

오차율은 `(예측 - 실측) / 실측`이며 실측이 0인 묶음은 `N/A`로 표시합니다. 출력은 묶음, 예측 GB, 실측 GB, 오차율 표입니다.

```python
@dataclass(frozen=True)
class BenchResult:
    status: str                   # "ok" | "oom" | "failed"
    error: str | None
    params: EstimateParams
    predicted: dict[str, int]     # 묶음별 bytes
    measured: dict[str, int]      # 묶음별 bytes, 실패 시 빈 dict
    gpu_name: str
    versions: dict[str, str]      # torch, transformers, peft
    timestamp: str                # ISO 8601
```

예외 처리 경계는 **모델 로드 시작 시점**입니다.
- 로드 전(의존성 import, CUDA 확인, 인자 검증, `flash-attn` 확인)에 실패하면 9절의 예외를 그대로 올립니다. `BenchResult`는 만들어지지 않습니다.
- 로드 이후 발생한 `torch.cuda.OutOfMemoryError`는 `status="oom"`, 그 밖의 예외는 `status="failed"`와 `error` 메시지로 반환하며 예외를 전파하지 않습니다. 반환 전에 모델, optimizer, 출력 텐서를 삭제하고 `empty_cache()`를 호출합니다. 실패한 실행도 기록 대상입니다.

### 7.4 기록

`record` 인자(CLI `--record PATH`)를 지정한 경우에만 `BenchResult.to_dict()`를 해당 jsonl 파일에 한 줄로 추가합니다. 호스트명과 파일 경로는 기록하지 않으며 모델은 `Arch.name`(config의 `_name_or_path` 마지막 경로 요소)으로 식별합니다. 저장소의 `benchmarks/results.jsonl`은 검증표 원본이고, `benchmarks/make_table.py`가 README 표를 생성합니다.

### 7.5 검증 계획

RTX 3090 Ti(24GB)에서 아래 조합을 측정할 계획이었으나, 서버 접속 불가로 1차 실측은 RTX A6000(48GB, dg 서버)에서 수행했습니다. Llama 가중치는 gated라 동일 가중치의 `unsloth/` 미러를 사용했습니다. 48GB에서는 8B 전체 학습을 제외한 모든 조합이 OOM 없이 측정됩니다.

| 모델 | 추론 | 전체 학습 | LoRA |
|---|---|---|---|
| Llama-3.2-1B | bf16, s=2048/4096 | amp, pure, grad_checkpoint 유무 | r=16 |
| Llama-3.2-3B | bf16, s=2048 | pure + grad_checkpoint | r=16 |
| Llama-3.1-8B | bf16, s=2048 | 불가 | r=16, grad_checkpoint |
| Qwen2.5-1.5B | bf16, s=2048 | pure | r=16 |

attention 구현은 sdpa를 기본으로 하고 1B 모델에서 eager를 추가 측정합니다. 70B급과 MoE 검증은 RunPod에서 2차에 수행합니다.

## 8. CLI (`cli.py`)

argparse 기반입니다.

```
vramcalc MODEL [--train] [--lora R] [--dtype bf16] [--batch 1] [--seq 2048]
         [--grad-checkpoint] [--optimizer adamw] [--precision amp]
         [--attn sdpa] [--gpu NAME_OR_GB] [--json]
vramcalc bench MODEL_PATH [위와 같은 옵션] [--steps 3] [--record PATH]
vramcalc gpus
```

- `MODEL`은 dict를 제외한 모든 입력 형식을 받습니다.
- `--gpu` 지정 시 `fits()` 결과와 제안을 함께 출력하고, 들어가지 않으면 exit code 1을 반환합니다.
- `--json`은 `Estimate.to_dict()`와 `FitResult`를 직렬화합니다.
- `vramcalc gpus`는 내장 테이블을 출력합니다.

## 9. 에러 처리

| 상황 | 동작 |
|---|---|
| allowlist 밖 `model_type` | `UnsupportedModelError`, 지원 목록 안내 |
| 필요한 config 키 누락 또는 4.1 검증 위반 | `ValueError`, 키 이름 |
| 4.2 인자 검증 위반 | `ValueError`, 인자 이름 |
| HF repo id인데 `huggingface_hub` 없음 | `ImportError`, `pip install vramcalc[hf]` 안내 |
| `bench` 호출인데 `torch` 없음 | `ImportError`, `pip install vramcalc[bench]` 안내 |
| `bench` 호출인데 CUDA 없음 | `RuntimeError` (모델 로드 전) |
| `attn_impl="flash"` bench인데 `flash-attn` 없음 | `ImportError` (모델 로드 전) |
| 모르는 GPU 이름 | `KeyError`, 유사 이름 후보 |
| `fits(None)`인데 GPU 감지 불가 | `RuntimeError`, 이름이나 숫자 입력 안내 |
| bench 모델 로드 이후 OOM | `BenchResult(status="oom")` 반환, 예외 전파 없음 |
| bench 모델 로드 이후 그 밖의 예외 | `BenchResult(status="failed", error=...)` 반환, 예외 전파 없음 |

## 10. 테스트

- `tests/configs/`: Llama-2-7B, Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B, Mistral-7B-v0.3, Qwen2.5-7B, Qwen2.5-1.5B, Gemma-2-2B의 실제 config.json. `param_count()`가 safetensors 메타데이터 기준 정확한 파라미터 수와 ±0.1% 안에서 일치하는지 검사합니다. 기준값은 테스트 파일에 숫자로 고정합니다.
- `test_arch.py`: 4.1 검증 규칙의 경계값(0, 음수, 나누어떨어지지 않는 비율, allowlist 밖 model_type)을 검사합니다.
- `test_memory.py`: 손으로 계산 가능한 작은 Arch(h=64, L=2, a=4, kv_heads=2, i=128, V=256)로 항목별 함수를 검증합니다.
- `test_estimate.py`: 모드별 조합이 올바른 항목만 합산하는지, 추론에서 학습 항목이 0인지, `peak_extra`의 max 선택이 맞는지, 4.2 인자 검증을 검사합니다.
- `test_suggest.py`: 들어가지 않는 케이스에서 후보 순서, 최소값 종료, `changes_objective` 라벨, 모두 실패 시 빈 목록과 message를 검사합니다.
- `test_cli.py`: `subprocess`로 주요 명령 실행, exit code와 `--json` 출력 스키마를 검사합니다.
- `test_bench_format.py`: `BenchResult.to_dict()` 스키마와 `results.jsonl` 각 줄의 필수 필드를 검사합니다. 실측 자체는 GPU가 필요하므로 CI에서 제외합니다.

## 11. CI와 배포

- GitHub Actions: `uv sync --extra dev`, `ruff check`, `pytest`를 Python 3.10과 3.12에서 실행합니다.
- 태그 push 시 `uv build` 후 PyPI trusted publishing으로 배포합니다.
- 저장소는 공개입니다. `.gitignore`에 모델 가중치 디렉터리와 `.env`를 포함합니다.

## 12. 버전 계획

- `0.1.0`: 추론과 학습 식, LoRA, 제안, CLI, 3090 Ti 실측표
- `0.2.0`: MoE, sliding window 레이어별 KV cache, 70B급 RunPod 실측
- 이후: 양자화, MLA, 분산 학습

## 13. 구현 순서

1. `arch.py`, `loaders.py`, config 테스트 (파라미터 수 일치 확인)
2. `memory.py`, `estimate.py` 추론 모드, 단위 테스트
3. `bench.py` 추론 실측, 식 조정
4. 학습 모드 (amp/pure, gradients, optimizer, step_workspace, activations, autocast_cache), 실측, 식 조정
5. LoRA, grad_checkpoint, 실측, 식 조정
6. `gpus.py`, `suggest.py`, `cli.py`
7. README, 검증표, CI, PyPI 배포
