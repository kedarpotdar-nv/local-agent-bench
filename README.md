# local-agent-bench

A lightweight benchmarking tool that measures real-world agent performance — TTFT, prefill throughput, decode speed, and end-to-end latency — across multi-turn conversations with prefix caching. Supports Ollama, vLLM, TensorRT-LLM, and llama.cpp. One script, one dependency.

## Setup

```bash
pip install transformers
```

## Quick start

```bash
# OpenClaw profile (default) — realistic agentic workload
python3 local_agent_bench.py \
    --backend ollama \
    --model qwen3:30b \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --profile openclaw \
    --output ollama_qwen3_openclaw.json

# Legacy profile — simple 3-turn chatbot benchmark
python3 local_agent_bench.py \
    --backend ollama \
    --model qwen3:30b \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --profile legacy \
    --output ollama_qwen3_legacy.json
```

See `benchmarks/` for more model configs. Run `python3 local_agent_bench.py --help` for all options.

## Profiles

### `openclaw` (default)

Simulates a realistic OpenClaw agentic session: 8 user turns, 17 inference calls with multi-call tool chains. Derived from 112 interactive turns across 64 real OpenClaw sessions on DGX Spark.

Includes a ~7,900-token system prompt and 18 tool schemas sent via the API `tools` parameter (adds ~1,485 ISL tokens server-side). ISL grows from 9.4K to ~13.7K tokens as tool results accumulate.

| Turn | Type | Calls | Tool result | What it simulates |
|------|------|------:|------------:|-------------------|
| 1 | Chat | 1 | — | User asks a question |
| 2 | Exec | 2 | 125t | Run a shell command, respond |
| 3 | Web | 3 | 1,100t | Search twice, summarize |
| 4 | File read | 2 | 600t | Read a file, analyze |
| 5 | Chat | 1 | — | Follow-up question |
| 6 | Exec | 2 | 150t | Run + check output |
| 7 | Code edit | 4 | 825t | Read → edit → verify → respond |
| 8 | Web | 2 | 950t | Search, summarize |

Turn mix matches real usage: 32% chat, 29% exec, 17% code_edit, 10% web, 8% file_read.

### `legacy`

Original 3-turn benchmark for backward compatibility:

| Turn | Input | Output | Scenario |
|------|-------|--------|----------|
| 1 | 16K cold | 128 | First turn — process system prompt |
| 2 | 16K cached + 128 new | 128 | Follow-up chat |
| 3 | 16K cached + 4K new | 512 | Tool output (file read) |

### Custom turns

Override with `--turns` for custom workloads:

```bash
python3 local_agent_bench.py --backend ollama --model qwen3:30b \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --turns "0,8192,512|8192,256,512|8192,1024,1024"
```

## Options

```
--profile openclaw|legacy   Workload profile (default: openclaw)
--turns SPEC                Custom turn config (overrides --profile)
--repeats N                 Runs per call for median/stddev
--no-tools                  Disable tool schemas in requests
--no-system-prompt          Disable system prompt
--output FILE               Save results to JSON
```

## How the OpenClaw profile was derived

All parameters measured from real OpenClaw sessions, verified by HTTP wire intercept:

- **System prompt**: 31,475 chars (~7,900 tokens) from `systemPromptReport`
- **Tool schemas**: +1,485 tokens measured via `prompt_eval_count` delta
- **Turn mix**: 112 interactive turns across 64 sessions (heartbeat/cron excluded)
- **Tool result sizes**: P50=447 chars, P90=3,828 chars from 384 real results
- **Calls per turn**: median 2, mean 2.2
- **Thinking tokens**: not stored in ISL (verified: OpenClaw sends in `reasoning` key, Ollama ignores)

See `GROUND_TRUTH_REPORT.md` in the parent repo for full methodology.
