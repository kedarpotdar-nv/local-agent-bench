#!/usr/bin/env python3
"""
Measure TTFT and tok/s for any model on local inference servers.

Supports: Ollama, vLLM, TensorRT-LLM (trtllm-serve), llama.cpp-server
Client-side measurement — same method for all backends.
Uses Ollama's prompt_eval_duration/eval_duration when available for validation.
Requires: pip install transformers

Metrics:
  TTFT:  Time from request sent to first streaming token (thinking or content)
  tok/s: ALL output tokens (thinking + content) / decode wall time
         Token count from: server usage field > tokenizer count > chunk count

Profiles:
  openclaw  Realistic OpenClaw agentic workload (default)
            8 user turns, 17 inference calls, multi-call chains.
            Based on 112 interactive turns from 64 real OpenClaw sessions.
            ISL: 9.4K → 13.7K tokens. Includes system prompt + tool schemas.
  legacy    Original 3-turn benchmark (backward compatible)
            ISL: 16K cold → 16K cached + 4K new. No system prompt or tools.

Usage:
  # OpenClaw profile on Ollama (recommended)
  python3 local_agent_bench.py --backend ollama --model qwen3:30b \\
    --tokenizer Qwen/Qwen3-30B-A3B --profile openclaw

  # Legacy profile (backward compatible)
  python3 local_agent_bench.py --backend ollama --model qwen3:30b \\
    --tokenizer Qwen/Qwen3-30B-A3B --profile legacy

  # Custom turns (same as before)
  python3 local_agent_bench.py --backend ollama --model qwen3:30b \\
    --tokenizer Qwen/Qwen3-30B-A3B \\
    --turns "0,16384,128|16384,128,128|16384,4096,512"

  # Repeated runs for statistical confidence
  python3 local_agent_bench.py --backend ollama --model qwen3:30b \\
    --tokenizer Qwen/Qwen3-30B-A3B --profile openclaw --repeats 3
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
import urllib.request

_tokenizer = None


def get_tokenizer(tokenizer_name):
    """Load HuggingFace tokenizer once."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True)
        print(f"  Tokenizer: {tokenizer_name}")
    except Exception as e:
        print(f"  ERROR: Could not load tokenizer '{tokenizer_name}': {e}")
        print(f"  Install: pip install transformers")
        sys.exit(1)
    return _tokenizer


def generate_text(n_tokens, tokenizer, seed=42):
    """Generate exactly n_tokens of text using tokenizer encode/decode."""
    rng = random.Random(seed)
    token_ids = [rng.randint(0, tokenizer.vocab_size - 1) for _ in range(n_tokens)]
    text = tokenizer.decode(token_ids)
    # Re-encode to stabilize tokenization, truncate to exact length
    re_encoded = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(re_encoded)


def count_tokens(text, tokenizer):
    """Count tokens using the tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# OpenClaw tool schemas (18 tools, adds ~1,485 ISL tokens when sent to Ollama)
# Measured via prompt_eval_count delta on DGX Spark, April 2026.
# ---------------------------------------------------------------------------

OPENCLAW_TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Create or overwrite a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Make precise edits to files", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "apply_patch", "description": "Apply multi-file unified diffs", "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents for patterns", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "find", "description": "Find files by glob pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "ls", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "exec", "description": "Run shell commands", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "cwd": {"type": "string"}, "background": {"type": "boolean"}, "yieldMs": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "process", "description": "Manage background exec sessions", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["poll", "kill", "list"]}, "id": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "count": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_fetch", "description": "Fetch readable content from a URL", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "selector": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "browser", "description": "Control web browser", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["navigate", "click", "type", "screenshot", "scroll"]}, "url": {"type": "string"}, "selector": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "message", "description": "Send messages to channels", "parameters": {"type": "object", "properties": {"channel": {"type": "string"}, "text": {"type": "string"}, "action": {"type": "string", "enum": ["send", "reply", "react"]}}, "required": ["channel", "text"]}}},
    {"type": "function", "function": {"name": "cron", "description": "Manage cron jobs", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["create", "list", "delete"]}, "schedule": {"type": "string"}, "systemEvent": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "memory_search", "description": "Search memory files", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "memory_get", "description": "Get memory file lines", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "session_status", "description": "Show status card", "parameters": {"type": "object", "properties": {"model": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "sessions_spawn", "description": "Spawn sub-agent", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string"}, "runtime": {"type": "string", "enum": ["subagent", "acp"]}}, "required": ["prompt"]}}},
]


# ---------------------------------------------------------------------------
# OpenClaw system prompt (~7,900 tokens measured)
# Realistic approximation of the 31,475-char system prompt from OpenClaw
# 2026.3.13 with 12 skills, 18 tools, and workspace file injections.
# ---------------------------------------------------------------------------

def build_openclaw_system_prompt(tokenizer, target_tokens=7900):
    """Build a system prompt of exactly target_tokens.

    Uses the real OpenClaw structure (instructions, skills XML, workspace
    files) padded to match the measured 31,475-char / ~7,900-token prompt.
    """
    header = (
        "You are a personal assistant running inside OpenClaw.\n\n"
        "## Tooling\nTool availability (filtered by policy):\n"
        "- read, write, edit, apply_patch, grep, find, ls, exec, process\n"
        "- web_search, web_fetch, browser, message, cron\n"
        "- memory_search, memory_get, session_status, sessions_spawn\n\n"
        "## Safety\nPrioritize safety and human oversight.\n\n"
        "## Skills\n<available_skills>\n"
        "  <skill><name>todo</name><description>Task management</description>"
        "<location>~/.openclaw/workspace/skills/todo/SKILL.md</location></skill>\n"
        "  <skill><name>github</name><description>GitHub integration</description>"
        "<location>~/.openclaw/workspace/skills/github/SKILL.md</location></skill>\n"
        "  <skill><name>coding-agent</name><description>Spawn coding sub-agent</description>"
        "<location>~/.openclaw/workspace/skills/coding-agent/SKILL.md</location></skill>\n"
        "</available_skills>\n\n"
        "## Workspace Files\n"
    )
    header_tokens = count_tokens(header, tokenizer)
    remaining = target_tokens - header_tokens
    if remaining > 0:
        padding = generate_text(remaining, tokenizer, seed=8888)
        return header + padding
    return header


# ---------------------------------------------------------------------------
# OpenClaw profile: 17-call chain across 8 user turns
# ---------------------------------------------------------------------------

# Each entry: (label, prefix_tokens, new_tokens, output_tokens)
# prefix_tokens: tokens from prior context (KV-cached)
# new_tokens:    tokens added this call (tool result or user message)
# output_tokens: num_predict cap for this call
#
# Derived from 112 interactive turns across 64 real OpenClaw sessions.
# Tool result sizes match measured P50/P90 from 384 real tool results.
# Turn mix: 32% chat, 29% exec, 17% code_edit, 10% web, 8% file_read.

OPENCLAW_PROFILE = [
    # Turn 1: Chat Q&A (1 call)
    ("T1 chat: respond",           0, 0,    2000),

    # Turn 2: Run a command (2 calls)
    ("T2 exec: call tool",         0, 50,   1500),   # user msg
    ("T2 exec: respond",           0, 125,  2000),   # +125t exec result

    # Turn 3: Web search (3 calls)
    ("T3 web: search",             0, 50,   1500),   # user msg
    ("T3 web: search again",       0, 550,  1500),   # +550t search results
    ("T3 web: summarize",          0, 550,  2000),   # +550t more results

    # Turn 4: Read a file (2 calls)
    ("T4 read: call tool",         0, 50,   1500),   # user msg
    ("T4 read: analyze",           0, 600,  2000),   # +600t file content

    # Turn 5: Follow-up chat (1 call)
    ("T5 chat: respond",           0, 50,   2000),   # user msg

    # Turn 6: Exec + check (2 calls)
    ("T6 exec: call tool",         0, 50,   1500),   # user msg
    ("T6 exec: respond",           0, 150,  2000),   # +150t exec output

    # Turn 7: Edit a file — heavy turn (4 calls)
    ("T7 edit: read first",        0, 50,   1500),   # user msg
    ("T7 edit: edit file",         0, 600,  2000),   # +600t file content
    ("T7 edit: run verify",        0, 25,   1500),   # +25t edit confirm
    ("T7 edit: respond",           0, 200,  2000),   # +200t test output

    # Turn 8: Web search + summarize (2 calls)
    ("T8 web: search",             0, 50,   1500),   # user msg
    ("T8 web: summarize",          0, 950,  2000),   # +950t search results
]


# Legacy profile (backward compatible)
LEGACY_TURNS = "0,16384,128|16384,128,128|16384,4096,512"


# ---------------------------------------------------------------------------
# Backend: Ollama (native API — supports num_predict for output control)
# ---------------------------------------------------------------------------

def send_ollama(url, model, messages, max_tokens, tools=None):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/api/chat", data=data,
        headers={"Content-Type": "application/json"})

    t_start = time.perf_counter()
    t_first = None
    generated_text = ""
    server_stats = {}

    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msg = obj.get("message", {})
                    c = msg.get("content", "")
                    th = msg.get("thinking", "")
                    # TTFT = first token of any kind (thinking or content)
                    if (c or th) and t_first is None:
                        t_first = time.perf_counter()
                    # Accumulate ALL generated text (thinking + content)
                    generated_text += c + th
                    if obj.get("done"):
                        server_stats = {
                            "prompt_eval_count": obj.get("prompt_eval_count", 0),
                            "prompt_eval_ms": obj.get("prompt_eval_duration", 0) / 1e6,
                            "eval_count": obj.get("eval_count", 0),
                            "eval_ms": obj.get("eval_duration", 0) / 1e6,
                        }
                except json.JSONDecodeError:
                    pass

    t_end = time.perf_counter()
    return t_start, t_first, t_end, generated_text, server_stats


def warmup_ollama(url, model):
    """Send small request to load model, then flush KV for clean cold start."""
    print("  [warmup] Loading model...")
    send_ollama(url, model, [{"role": "user", "content": "hi"}], 1)
    print("  [warmup] Flushing KV cache...")
    import subprocess
    for ollama_bin in ["/usr/local/bin/ollama", "ollama"]:
        try:
            subprocess.run([ollama_bin, "stop", model],
                          capture_output=True, timeout=10)
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    time.sleep(2)
    send_ollama(url, model, [{"role": "user", "content": "hi"}], 1)
    print("  [warmup] Ready.\n")


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible (vLLM, TensorRT-LLM, llama.cpp-server)
# ---------------------------------------------------------------------------

def send_openai(url, model, messages, max_tokens, tools=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
        })

    t_start = time.perf_counter()
    t_first = None
    generated_text = ""
    usage_tokens = None
    usage_prompt_tokens = None

    with urllib.request.urlopen(req, timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                s = line[6:].decode("utf-8", errors="replace")
                if s.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(s)
                    choices = obj.get("choices", [])
                    if choices:
                        d = choices[0].get("delta", {})
                        c = d.get("content", "") or ""
                        r = (d.get("reasoning_content", "")
                             or d.get("reasoning", "") or "")
                        # TTFT = first token of any kind
                        if (c or r) and t_first is None:
                            t_first = time.perf_counter()
                        # Accumulate ALL generated text
                        generated_text += c + r
                    # Capture usage from final chunk (most reliable token count)
                    usage = obj.get("usage")
                    if usage and usage.get("completion_tokens"):
                        usage_tokens = usage["completion_tokens"]
                    if usage and usage.get("prompt_tokens"):
                        usage_prompt_tokens = usage["prompt_tokens"]
                except json.JSONDecodeError:
                    pass

    t_end = time.perf_counter()
    server_stats = {}
    if usage_tokens is not None:
        server_stats["usage_completion_tokens"] = usage_tokens
    if usage_prompt_tokens is not None:
        server_stats["prompt_eval_count"] = usage_prompt_tokens
    return t_start, t_first, t_end, generated_text, server_stats


def warmup_openai(url, model):
    """Send small request to warm up CUDA graphs / memory allocations."""
    print("  [warmup] Sending small request...")
    send_openai(url, model, [{"role": "user", "content": "hi"}], 1)
    print("  [warmup] Ready.\n")


# ---------------------------------------------------------------------------
# Measurement + reporting
# ---------------------------------------------------------------------------

BACKENDS = {
    "ollama":   {"send": send_ollama, "warmup": warmup_ollama},
    "vllm":     {"send": send_openai, "warmup": warmup_openai},
    "trtllm":   {"send": send_openai, "warmup": warmup_openai},
    "llamacpp": {"send": send_openai, "warmup": warmup_openai},
}

DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000",
    "trtllm": "http://localhost:8000",
    "llamacpp": "http://localhost:8080",
}


def measure_call(label, url, model, messages, max_tokens, send_fn, tokenizer,
                 tools=None):
    """Run one inference call and return metrics."""
    t_start, t_first, t_end, generated_text, server = send_fn(
        url, model, messages, max_tokens, tools=tools)

    ttft_client = (t_first - t_start) * 1000 if t_first else 0
    decode_client = (t_end - t_first) * 1000 if t_first else 0
    total_client = (t_end - t_start) * 1000

    # Token count priority: server eval_count > server usage > tokenizer count
    if server.get("eval_count"):
        tokens = server["eval_count"]
        token_source = "server eval_count"
    elif server.get("usage_completion_tokens"):
        tokens = server["usage_completion_tokens"]
        token_source = "server usage"
    elif generated_text:
        tokens = count_tokens(generated_text, tokenizer)
        token_source = "tokenizer"
    else:
        tokens = 0
        token_source = "none"

    # Decode throughput: output tokens / decode wall time
    decode_tps = tokens / (decode_client / 1000) if decode_client > 0 else 0

    # Input tokens from server (actual ISL including tool schemas)
    server_isl = server.get("prompt_eval_count", 0)

    # Prefill throughput
    prefill_ms = server.get("prompt_eval_ms", 0)
    prefill_tps = server_isl / (prefill_ms / 1000) if prefill_ms > 0 else 0

    # Projected E2E: what this call would cost if the model generated
    # exactly max_tokens. Deterministic across hardware for comparison.
    # projected_e2e = TTFT + (max_tokens / decode_tps)
    projected_e2e = (ttft_client + max_tokens / decode_tps * 1000
                     if decode_tps > 0 else total_client)

    return {
        "label": label,
        "ttft_client_ms": round(ttft_client, 1),
        "decode_client_ms": round(decode_client, 1),
        "total_client_ms": round(total_client, 1),
        "projected_e2e_ms": round(projected_e2e, 1),
        "server_isl": server_isl,
        "output_tokens": tokens,
        "output_tokens_requested": max_tokens,
        "token_source": token_source,
        "decode_tps": round(decode_tps, 1),
        "prefill_tps": round(prefill_tps, 1),
        **{f"server_{k}": v for k, v in server.items()},
    }


def print_call_result(r):
    """Print a single call result."""
    print(f"  Client TTFT:      {r['ttft_client_ms']:>8.0f}ms")
    print(f"  Client Decode:    {r['decode_client_ms']:>8.0f}ms  "
          f"({r['output_tokens']}/{r['output_tokens_requested']} tokens "
          f"[{r['token_source']}], {r['decode_tps']:.1f} tok/s)")
    print(f"  E2E (measured):   {r['total_client_ms']:>8.0f}ms")
    print(f"  E2E (projected):  {r['projected_e2e_ms']:>8.0f}ms  "
          f"(TTFT + {r['output_tokens_requested']}t / {r['decode_tps']:.1f} tok/s)")
    if r.get("server_isl"):
        print(f"  Server ISL:       {r['server_isl']:>8d} tokens")
    if r.get("server_prompt_eval_ms"):
        pe = r["server_prompt_eval_ms"]
        ev = r.get("server_eval_ms", 0)
        ec = r.get("server_eval_count", 0)
        ev_tps = ec / (ev / 1000) if ev > 0 else 0
        print(f"  Server Prefill:   {pe:>6.0f}ms  "
              f"(ISL={r['server_isl']}, {r['prefill_tps']:.0f} tok/s)")
        print(f"  Server Decode:    {ev:>6.0f}ms  "
              f"({ec} tokens, {ev_tps:.1f} tok/s)")
    print()


TOOL_ISL_OVERHEAD = 1485  # measured tokens added by 18 tool schemas on Ollama


def run_openclaw_profile(url, model, send_fn, tokenizer, repeats, tools,
                         output_file):
    """Run the OpenClaw 17-call profile."""
    sys_prompt = build_openclaw_system_prompt(tokenizer)

    if not tools:
        # Option C: pad system prompt to compensate for missing tool ISL
        # so ISL is consistent across backends that don't support tools
        print(f"  No tools — padding system prompt with ~{TOOL_ISL_OVERHEAD} tokens "
              f"to match tool ISL overhead")
        padding = generate_text(TOOL_ISL_OVERHEAD, tokenizer, seed=7777)
        sys_prompt = sys_prompt + "\n\n## Additional Context\n" + padding

    sys_prompt_tokens = count_tokens(sys_prompt, tokenizer)
    print(f"  System prompt: {sys_prompt_tokens} tokens (client-side)")
    if tools:
        print(f"  Tool schemas: {len(tools)} tools (adds ~{TOOL_ISL_OVERHEAD} tokens server-side)")
    else:
        print(f"  Tool schemas: none (ISL padded in system prompt instead)")
    print(f"  Profile: openclaw (8 turns, 17 inference calls)")

    all_results = []
    all_summaries = []

    for rep in range(repeats):
        if repeats > 1:
            print(f"\n{'='*60}")
            print(f"  Run {rep+1}/{repeats}")
            print(f"{'='*60}")

        # Build conversation: start with system prompt
        messages = [{"role": "system", "content": sys_prompt}]
        cumulative_new = 0

        run_results = []
        for i, (label, _prefix, new_tokens, out_tokens) in enumerate(OPENCLAW_PROFILE):
            # Add new tokens (user message or tool result) to conversation
            if new_tokens > 0:
                new_text = generate_text(new_tokens, tokenizer, seed=5000 + i)
                # Alternate between user and tool-result-like messages
                if "respond" in label or "summarize" in label or "analyze" in label:
                    messages.append({"role": "user", "content": new_text})
                else:
                    # Simulate tool result appended as assistant context
                    messages.append({"role": "user",
                                     "content": f"[Tool Result]\n{new_text}"})
                cumulative_new += new_tokens

            if repeats > 1:
                print(f"{label} [run {rep+1}]:")
            else:
                print(f"{label}:")

            r = measure_call(label, url, model, messages, out_tokens, send_fn,
                             tokenizer, tools=tools)
            print_call_result(r)
            run_results.append(r)

            # Append a short assistant response to conversation
            # (simulates visible output that stays in context)
            visible_response = generate_text(50, tokenizer, seed=6000 + i)
            messages.append({"role": "assistant", "content": visible_response})

        all_results.append(run_results)

    # Summary
    print(f"{'='*60}")
    print(f"  SUMMARY — {platform.node()} — {model}")
    print(f"{'='*60}")

    # Aggregate across calls and runs
    for i, (label, _, _, out_tokens) in enumerate(OPENCLAW_PROFILE):
        call_results = [run[i] for run in all_results]
        ttfts = [r["ttft_client_ms"] for r in call_results]
        e2es = [r["total_client_ms"] for r in call_results]
        proj_e2es = [r["projected_e2e_ms"] for r in call_results]
        decode_list = [r["decode_tps"] for r in call_results if r["decode_tps"] > 0]
        isls = [r["server_isl"] for r in call_results if r["server_isl"] > 0]

        ttft = statistics.median(ttfts) if ttfts else 0
        e2e = statistics.median(e2es) if e2es else 0
        proj = statistics.median(proj_e2es) if proj_e2es else 0
        dec = statistics.median(decode_list) if decode_list else 0
        isl = statistics.median(isls) if isls else 0

        all_summaries.append({
            "label": label, "ttft_ms": round(ttft),
            "e2e_ms": round(e2e), "projected_e2e_ms": round(proj),
            "decode_tps": round(dec, 1),
            "server_isl": round(isl),
            "output_tokens_requested": out_tokens,
        })

    print(f"\n  {'Call':<28s} {'ISL':>6s} {'TTFT':>7s} {'Decode':>7s} "
          f"{'E2E(m)':>8s} {'E2E(p)':>8s}")
    print(f"  {'-'*70}")
    total_proj = 0
    for s in all_summaries:
        total_proj += s["projected_e2e_ms"]
        print(f"  {s['label']:<28s} {s['server_isl']:>5.0f}t "
              f"{s['ttft_ms']:>6.0f}ms {s['decode_tps']:>6.1f} "
              f"{s['e2e_ms']:>7.0f}ms {s['projected_e2e_ms']:>7.0f}ms")
    print(f"\n  E2E(m) = measured    E2E(p) = projected (TTFT + num_predict/decode_tps)")
    print(f"  Use E2E(p) for hardware comparison — deterministic regardless of model output length.")
    print(f"  Projected session total: {total_proj/1000:.1f}s across 17 calls")

    if output_file:
        out = {
            "host": platform.node(),
            "model": model,
            "tokenizer": args.tokenizer,
            "backend": args.backend,
            "url": url,
            "profile": "openclaw",
            "repeats": repeats,
            "system_prompt_tokens": sys_prompt_tokens,
            "tool_count": len(tools) if tools else 0,
            "summaries": all_summaries,
            "results": [[{k: v for k, v in r.items()} for r in run]
                        for run in all_results],
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {output_file}")


def run_legacy_profile(url, model, send_fn, tokenizer, repeats, turns_str,
                       output_file):
    """Run the legacy turn-based profile (backward compatible)."""
    # Parse turn configs
    turns = []
    for spec in turns_str.split("|"):
        parts = spec.strip().split(",")
        if len(parts) != 3:
            print(f"Invalid turn config: {spec}")
            sys.exit(1)
        turns.append((int(parts[0]), int(parts[1]), int(parts[2])))

    # Generate text with tokenizer — exact token counts
    max_prefix = max(t[0] for t in turns)
    cold_inputs = [t[1] for t in turns if t[0] == 0]
    cold_input = max(cold_inputs) if cold_inputs else 0
    shared_len = max(max_prefix, cold_input)
    print(f"  Generating {shared_len} token prefix...")
    prefix_text = generate_text(shared_len, tokenizer, seed=42)

    # Generate per-turn suffix texts (only for warm turns)
    suffix_texts = {}
    for i, (prefix_len, new_len, _) in enumerate(turns):
        if prefix_len > 0:
            suffix_texts[i] = generate_text(new_len, tokenizer, seed=100 + i)

    all_summaries = []
    all_results = []

    for i, (prefix_len, new_len, out_len) in enumerate(turns):
        if prefix_len == 0:
            text = generate_text(new_len, tokenizer, seed=42)
            label = f"Turn {i+1}: {new_len//1024}K cold"
        else:
            text = prefix_text + " " + suffix_texts[i]
            label = f"Turn {i+1}: {prefix_len//1024}K cached + {new_len} new"

        messages = [{"role": "user", "content": text}]
        turn_results = []
        for run in range(repeats):
            if repeats > 1:
                print(f"{label} [run {run+1}/{repeats}]:")
            else:
                print(f"{label}:")
            r = measure_call(label, url, model, messages, out_len, send_fn,
                             tokenizer)
            print_call_result(r)
            turn_results.append(r)

        all_results.extend(turn_results)

        # Summarize
        ttfts = [r["ttft_client_ms"] for r in turn_results]
        e2es = [r["total_client_ms"] for r in turn_results]
        decode_list = [r["decode_tps"] for r in turn_results if r["decode_tps"] > 0]

        def med(lst): return round(statistics.median(lst), 1) if lst else 0
        def std(lst): return round(statistics.stdev(lst), 1) if len(lst) > 1 else 0

        all_summaries.append({
            "label": label,
            "ttft_median_ms": med(ttfts), "ttft_stddev_ms": std(ttfts),
            "e2e_median_ms": med(e2es), "e2e_stddev_ms": std(e2es),
            "decode_tps_median": med(decode_list),
            "decode_tps_stddev": std(decode_list),
            "runs": len(turn_results),
        })

    # Summary table
    print(f"{'='*60}")
    print(f"  SUMMARY — {platform.node()} — {model}")
    print(f"{'='*60}")
    print(f"  {'Turn':<30s} {'TTFT':>9s} {'E2E':>9s} {'decode':>9s}")
    print(f"  {'-'*60}")
    for s in all_summaries:
        print(f"  {s['label']:<30s} {s['ttft_median_ms']:>8.0f}ms "
              f"{s['e2e_median_ms']:>8.0f}ms "
              f"{s['decode_tps_median']:>7.1f}t/s")

    if output_file:
        out = {
            "host": platform.node(),
            "model": model,
            "tokenizer": args.tokenizer,
            "backend": args.backend,
            "url": url,
            "profile": "legacy",
            "turns": turns_str,
            "repeats": repeats,
            "summaries": all_summaries,
            "results": all_results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {output_file}")


def main():
    global args
    parser = argparse.ArgumentParser(
        description="Measure TTFT and tok/s for local inference servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--url", default=None,
                        help="Server URL (default: per-backend)")
    parser.add_argument("--model", required=True,
                        help="Model name for API requests")
    parser.add_argument("--tokenizer", required=True,
                        help="HuggingFace tokenizer name")
    parser.add_argument("--backend", required=True,
                        choices=["ollama", "vllm", "trtllm", "llamacpp"],
                        help="Inference backend")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per turn for median/stddev (default: 1)")
    parser.add_argument("--profile", default="openclaw",
                        choices=["openclaw", "legacy"],
                        help="Workload profile (default: openclaw)")
    parser.add_argument("--turns", default=None,
                        help="Custom turn configs (overrides --profile): "
                             "prefix,new_tokens,output_tokens (pipe-separated)")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tool schemas in requests")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Disable system prompt (legacy behavior)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    url = args.url or DEFAULT_URLS[args.backend]
    send_fn = BACKENDS[args.backend]["send"]
    warmup_fn = BACKENDS[args.backend]["warmup"]
    tokenizer = get_tokenizer(args.tokenizer)
    tools = None if args.no_tools else OPENCLAW_TOOLS

    print(f"\n{'='*60}")
    print(f"  {platform.node()} — {args.model} via {args.backend}")
    print(f"  URL: {url}")
    print(f"{'='*60}\n")

    warmup_fn(url, args.model)

    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}\n")

    if args.turns:
        # Custom turns override — use legacy runner
        run_legacy_profile(url, args.model, send_fn, tokenizer, args.repeats,
                          args.turns, args.output)
    elif args.profile == "openclaw":
        run_openclaw_profile(url, args.model, send_fn, tokenizer, args.repeats,
                            tools, args.output)
    else:
        run_legacy_profile(url, args.model, send_fn, tokenizer, args.repeats,
                          LEGACY_TURNS, args.output)


if __name__ == "__main__":
    main()
