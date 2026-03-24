# local-agent-bench

A lightweight benchmarking tool that measures real-world agent performance — TTFT, prefill throughput, decode speed, and end-to-end latency — across multi-turn conversations with prefix caching. Supports Ollama, vLLM, TensorRT-LLM, and llama.cpp. One script, one dependency.

## Setup

```bash
pip install transformers
```

## Quick start (Qwen3.5-35B-A3B on Ollama)

```bash
ollama pull qwen3.5:35b

python3 local_agent_bench.py \
    --backend ollama \
    --model qwen3.5:35b \
    --tokenizer Qwen/Qwen3.5-35B-A3B \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output ollama_qwen35.json
```

See `benchmarks/` for more model configs (vLLM, TRT-LLM, etc.). Run `python3 local_agent_bench.py --help` for all options (`--repeats`, `--turns`, `--output`, etc.).

## Default turns

Simulates a 3-turn agent session with a 16K system prompt:

| Turn | Input | Output | Scenario |
|------|-------|--------|----------|
| 1 | 16K cold | 128 | First turn — process system prompt |
| 2 | 16K cached + 128 new | 128 | Follow-up chat |
| 3 | 16K cached + 4K new | 512 | Tool output (file read) |
