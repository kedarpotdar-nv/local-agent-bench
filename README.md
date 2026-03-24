# local-agent-bench

Benchmark local LLM inference servers for agentic workloads. Single script, client-side, works across Ollama, vLLM, TensorRT-LLM, and llama.cpp.

## Setup

```bash
pip install transformers
```

## Usage

```bash
python3 local_agent_bench.py --backend ollama --model qwen3.5:35b --tokenizer Qwen/Qwen3.5-35B-A3B
python3 local_agent_bench.py --backend vllm --url http://localhost:8000 --model my-model --tokenizer my-model
```

Run `python3 local_agent_bench.py --help` for all options (`--repeats`, `--turns`, `--output`, etc.).

## Default turns

Simulates a 3-turn agent session with a 16K system prompt:

| Turn | Input | Output | Scenario |
|------|-------|--------|----------|
| 1 | 16K cold | 128 | First turn — process system prompt |
| 2 | 16K cached + 128 new | 128 | Follow-up chat |
| 3 | 16K cached + 4K new | 512 | Tool output (file read) |
