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

#### What a real agentic session looks like

A user message doesn't map to a single LLM call. The agent thinks, calls tools, processes results, and may loop several times before responding:

```
User: "Fix the auth bug"
  │
  ├─ Call 1: model thinks → calls read("auth.py")          ← ISL: 9,400t
  │          [tool result: 600 tokens of file content]
  │
  ├─ Call 2: model thinks → calls edit(fix)                 ← ISL: 10,050t
  │          [tool result: 25 tokens confirmation]
  │
  ├─ Call 3: model thinks → calls exec("pytest")            ← ISL: 10,125t
  │          [tool result: 200 tokens test output]
  │
  └─ Call 4: model responds "Fixed the null check in..."    ← ISL: 10,375t

User: "What time is it?"
  │
  └─ Call 5: model responds "It's 3:42 PM Pacific"          ← ISL: 10,425t
```

The benchmark models this multi-call chain pattern. Each "turn" in the profile is one of these full sequences.

#### ISL composition

```
┌─────────────────────────────────────────────────────────────────┐
│                    ISL at each inference call                    │
│                                                                 │
│  ┌───────────────────────┐                                      │
│  │   System prompt       │  ~7,900 tokens (fixed, KV-cached)   │
│  │   + tool schemas      │  +1,485 tokens (Ollama injects)     │
│  ├───────────────────────┤                                      │
│  │   Conversation ctx    │  Grows with tool results:            │
│  │   (visible output +   │    Turn 1:      0t                   │
│  │    tool results +     │    Turn 4:  +1,275t                  │
│  │    user messages)     │    Turn 7:  +2,900t                  │
│  │                       │    Turn 8:  +4,200t                  │
│  └───────────────────────┘                                      │
│                                                                 │
│  NOT in ISL: thinking tokens (sent in `reasoning` key,          │
│  ignored by Ollama — verified by HTTP wire intercept)           │
└─────────────────────────────────────────────────────────────────┘
```

#### ISL growth across the 17 calls

```
Call  ISL        What happens                          Decode
─────────────────────────────────────────────────────────────
  1   9,400t ████████████████████████████               60 t/s   T1: chat
  2   9,450t ████████████████████████████               60 t/s   T2: exec →
  3   9,575t ████████████████████████████░              60 t/s   T2: respond
  4   9,625t ████████████████████████████░              60 t/s   T3: search →
  5  10,175t ████████████████████████████░░░            59 t/s   T3: search again →
  6  10,725t ████████████████████████████░░░░░          58 t/s   T3: summarize
  7  10,775t ████████████████████████████░░░░░          58 t/s   T4: read →
  8  11,375t ████████████████████████████░░░░░░░        57 t/s   T4: analyze
  9  11,425t ████████████████████████████░░░░░░░        57 t/s   T5: chat
 10  11,475t ████████████████████████████░░░░░░░        57 t/s   T6: exec →
 11  11,625t ████████████████████████████░░░░░░░░       57 t/s   T6: respond
 12  11,675t ████████████████████████████░░░░░░░░       57 t/s   T7: read →
 13  12,275t ████████████████████████████░░░░░░░░░░     56 t/s   T7: edit →
 14  12,300t ████████████████████████████░░░░░░░░░░     56 t/s   T7: verify →
 15  12,500t ████████████████████████████░░░░░░░░░░░    55 t/s   T7: respond
 16  12,550t ████████████████████████████░░░░░░░░░░░    55 t/s   T8: search →
 17  13,500t ████████████████████████████░░░░░░░░░░░░░  54 t/s   T8: summarize

█ = system prompt (KV-cached)   ░ = conversation growth (tool results)
```

Decode throughput drops from 60 to 54 tok/s as ISL grows — this is GPU memory bandwidth physics, not a software issue.

#### Turn details

| Turn | Type | Calls | Tool result | ISL after | Decode | What it simulates |
|------|------|------:|------------:|----------:|-------:|-------------------|
| 1 | Chat | 1 | — | 9,400t | 60 | User asks a question |
| 2 | Exec | 2 | +125t | 9,575t | 60 | Run a shell command, respond |
| 3 | Web | 3 | +1,100t | 10,725t | 58 | Search twice, summarize |
| 4 | File read | 2 | +600t | 11,375t | 57 | Read a file, analyze |
| 5 | Chat | 1 | — | 11,425t | 57 | Follow-up question |
| 6 | Exec | 2 | +150t | 11,625t | 57 | Run + check output |
| 7 | Code edit | 4 | +825t | 12,500t | 55 | Read → edit → verify → respond |
| 8 | Web | 2 | +950t | 13,500t | 54 | Search, summarize |

Turn mix matches real usage: 32% chat, 29% exec, 17% code_edit, 10% web, 8% file_read.

#### OSL per inference call

The benchmark sets `num_predict` as a ceiling. The model generates however many tokens it wants — the bench measures the decode rate from whatever it produces. A thinking model generates more tokens per call than a non-thinking model, but decode tok/s is the same.

```
                                  num_predict
                                  (ceiling)
                                      │
Tool call (model decides to use a tool):
  ├── thinking: 200-1500t ────────┐   │
  └── tool call JSON: 20-50t     │   │
                                  │   ▼
Final response (model answers):   │  1500-2000
  ├── thinking: 200-1500t ────────┘
  └── visible text: 50-300t

Per-call OSL from real sessions:
  P25:   118t (non-thinking model or simple response)
  P50:   284t
  P75:   729t
  P90: 1,968t (thinking model on complex turn)
```

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

## What the benchmark measures (and doesn't)

**Measures accurately (for hardware comparison):**
- **Decode tok/s** at realistic ISL (9.4K → 13.7K) — the key hardware metric
- **Prefill throughput** with KV cache reuse across multi-call chains
- **TTFT** (time to first token) including CPU overhead
- **Projected E2E** = TTFT + num_predict / decode_tps — deterministic, comparable across hardware
- ISL growth pattern matching real agentic tool-calling workloads

**Two E2E metrics:**
- `E2E(measured)` — actual wall time for this run. Varies because models generate different token counts for the same prompt.
- `E2E(projected)` — TTFT + (target output tokens / measured decode tok/s). Deterministic. **Use this for hardware comparison.**

**Known limitation — `ignore_eos`:** Ollama's `/api/chat` does not support disabling EOS. The model may generate fewer tokens than `num_predict` on short/simple prompts. With the OpenClaw profile (7,900-token system prompt + accumulated context), the model typically generates enough tokens for stable decode measurement. Projected E2E compensates for any shortfall.

**Does not model:**
- Actual thinking depth (synthetic prompts; but decode tok/s is the same regardless of content)
- Tool execution latency (0ms; we measure the inference server, not the agent)
- Model-specific tool selection (chain length is fixed per profile)

This is intentional — the benchmark is a proxy for comparing inference hardware under agentic workloads, not a simulation of agent behavior.

## How the OpenClaw profile was derived

All parameters measured from real OpenClaw sessions, verified by HTTP wire intercept between OpenClaw and Ollama on DGX Spark (April 2026):

- **System prompt**: 31,475 chars (~7,900 tokens) from OpenClaw's `systemPromptReport`
- **Tool schemas**: +1,485 tokens measured via `prompt_eval_count` delta (with/without `tools` param)
- **Turn mix**: 112 interactive turns across 64 sessions (heartbeat/cron excluded)
- **Tool result sizes**: P50=447 chars, P90=3,828 chars from 384 real tool results
- **Calls per turn**: median 2, mean 2.2
- **Thinking tokens**: not stored in ISL (verified: OpenClaw sends in `reasoning` key, Ollama ignores)
- **ISL growth**: 45% across an 8-turn session (9.4K → 13.7K tokens)

See `GROUND_TRUTH_REPORT.md` in the parent repo for full methodology, including the wire intercept that verified thinking token behavior.
