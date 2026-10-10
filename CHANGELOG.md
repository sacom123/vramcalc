# Changelog

## 0.1.1 (2026-10-11)

- Reject `lora_rank=True` (bool was accepted as an int)
- `fits()` GPU auto-detection now shuts NVML down after reading
- README badges

## 0.1.0 (2026-10-11)

- Inference, full fine-tuning and LoRA memory estimation for Llama, Mistral, Qwen2 and Gemma dense models
- `fits()` judgment with single-adjustment suggestions
- `bench` subcommand measuring real memory and comparing per bucket
- CLI: `vramcalc MODEL`, `vramcalc bench`, `vramcalc gpus`
