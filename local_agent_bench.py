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

Usage:
  # Ollama
  python3 local_agent_bench.py --backend ollama --model qwen3.5:35b \
    --tokenizer Qwen/Qwen3.5-35B-A3B

  # vLLM
  python3 local_agent_bench.py --backend vllm --url http://localhost:8000 \
    --model openai/gpt-oss-120b --tokenizer openai/gpt-oss-120b

  # TensorRT-LLM
  python3 local_agent_bench.py --backend trtllm --url http://localhost:8000 \
    --model my-model --tokenizer my-model-hf-name

  # llama.cpp-server
  python3 local_agent_bench.py --backend llamacpp --url http://localhost:8080 \
    --model default --tokenizer Qwen/Qwen3.5-35B-A3B

  # Repeated runs for statistical confidence
  python3 local_agent_bench.py --backend ollama --model qwen3.5:35b \
    --tokenizer Qwen/Qwen3.5-35B-A3B --repeats 5

  # Custom turn configs
  python3 local_agent_bench.py --backend ollama --model qwen3.5:35b \
    --tokenizer Qwen/Qwen3.5-35B-A3B \
    --turns "0,16384,128|16384,128,128|16384,4096,512"
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
# Backend: Ollama (native API — supports num_predict for output control)
# ---------------------------------------------------------------------------

def send_ollama(url, model, text, max_tokens):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/chat", data=payload,
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
    send_ollama(url, model, "hi", 1)
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
    send_ollama(url, model, "hi", 1)
    print("  [warmup] Ready.\n")


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible (vLLM, TensorRT-LLM, llama.cpp-server)
# ---------------------------------------------------------------------------

def send_openai(url, model, text, max_tokens):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
        })

    t_start = time.perf_counter()
    t_first = None
    generated_text = ""
    usage_tokens = None

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
                        r = d.get("reasoning_content", "") or ""
                        # TTFT = first token of any kind
                        if (c or r) and t_first is None:
                            t_first = time.perf_counter()
                        # Accumulate ALL generated text
                        generated_text += c + r
                    # Capture usage from final chunk (most reliable token count)
                    usage = obj.get("usage")
                    if usage and usage.get("completion_tokens"):
                        usage_tokens = usage["completion_tokens"]
                except json.JSONDecodeError:
                    pass

    t_end = time.perf_counter()
    # Store usage_tokens in server_stats for the measure function
    server_stats = {}
    if usage_tokens is not None:
        server_stats["usage_completion_tokens"] = usage_tokens
    return t_start, t_first, t_end, generated_text, server_stats


def warmup_openai(url, model):
    """Send small request to warm up CUDA graphs / memory allocations."""
    print("  [warmup] Sending small request...")
    send_openai(url, model, "hi", 1)
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


def measure(label, url, model, text, max_tokens, send_fn, tokenizer,
            prefill_tokens=None):
    """Run one request and return metrics.

    prefill_tokens: number of tokens actually being prefilled (excluding cached prefix).
                    If None, assumes all input tokens are prefilled (cold turn).
    """
    t_start, t_first, t_end, generated_text, server = send_fn(
        url, model, text, max_tokens)

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

    # Prefill throughput: only count tokens actually prefilled (not cached)
    input_tokens = count_tokens(text, tokenizer)
    if prefill_tokens is None:
        prefill_tokens = input_tokens  # cold turn: everything is prefilled
    prefill_tps = prefill_tokens / (ttft_client / 1000) if ttft_client > 0 else 0

    # Warn if output is significantly shorter than requested
    if tokens > 0 and tokens < max_tokens * 0.5:
        print(f"  WARNING: requested {max_tokens} tokens, got {tokens} "
              f"(model hit EOS early)")

    return {
        "label": label,
        "ttft_client_ms": round(ttft_client, 1),
        "decode_client_ms": round(decode_client, 1),
        "total_client_ms": round(total_client, 1),
        "input_tokens": input_tokens,
        "prefill_tokens": prefill_tokens,
        "output_tokens": tokens,
        "output_tokens_requested": max_tokens,
        "token_source": token_source,
        "decode_tps": round(decode_tps, 1),
        "prefill_tps": round(prefill_tps, 1),
        **{f"server_{k}": v for k, v in server.items()},
    }


def print_result(r):
    """Print a single measurement result."""
    prefill_label = (f"{r['prefill_tokens']} prefilled"
                     if r['prefill_tokens'] < r['input_tokens']
                     else f"{r['input_tokens']} input tokens")
    print(f"  Client TTFT:    {r['ttft_client_ms']:>8.0f}ms  "
          f"({prefill_label}, {r['prefill_tps']:.0f} prefill tok/s)")
    print(f"  Client Decode:  {r['decode_client_ms']:>8.0f}ms  "
          f"({r['output_tokens']} output tokens [{r['token_source']}], "
          f"{r['decode_tps']:.1f} decode tok/s)")
    print(f"  Client E2E:     {r['total_client_ms']:>8.0f}ms")
    if r.get("server_prompt_eval_ms"):
        pe = r["server_prompt_eval_ms"]
        pc = r["server_prompt_eval_count"]
        ev = r["server_eval_ms"]
        ec = r["server_eval_count"]
        pe_tps = pc / (pe / 1000) if pe > 0 else 0
        ev_tps = ec / (ev / 1000) if ev > 0 else 0
        print(f"  Server Prefill: {pe:>6.0f}ms  ({pc} tokens, {pe_tps:.0f} tok/s)")
        print(f"  Server Decode:  {ev:>6.0f}ms  ({ec} tokens, {ev_tps:.1f} tok/s)")
    if r["output_tokens"] < r["output_tokens_requested"] * 0.5 and r["output_tokens"] > 0:
        print(f"  WARNING: Output shorter than requested "
              f"({r['output_tokens']}/{r['output_tokens_requested']})")
    print()


def run_turn(label, url, model, text, out_len, send_fn, tokenizer, repeats,
             prefill_tokens=None):
    """Run a turn N times and return all results."""
    results = []
    for run in range(repeats):
        if repeats > 1:
            print(f"{label} [run {run+1}/{repeats}]:")
        else:
            print(f"{label}:")
        r = measure(label, url, model, text, out_len, send_fn, tokenizer,
                    prefill_tokens=prefill_tokens)
        print_result(r)
        results.append(r)
    return results


def summarize(turn_results):
    """Compute median/stddev across repeated runs for a single turn."""
    ttfts = [r["ttft_client_ms"] for r in turn_results]
    e2els = [r["total_client_ms"] for r in turn_results]
    decode_tps_list = [r["decode_tps"] for r in turn_results if r["decode_tps"] > 0]
    prefill_tps_list = [r["prefill_tps"] for r in turn_results if r["prefill_tps"] > 0]

    def med(lst): return round(statistics.median(lst), 1) if lst else 0
    def std(lst): return round(statistics.stdev(lst), 1) if len(lst) > 1 else 0

    summary = {
        "label": turn_results[0]["label"],
        "ttft_median_ms": med(ttfts),
        "ttft_stddev_ms": std(ttfts),
        "e2e_median_ms": med(e2els),
        "e2e_stddev_ms": std(e2els),
        "decode_tps_median": med(decode_tps_list),
        "decode_tps_stddev": std(decode_tps_list),
        "prefill_tps_median": med(prefill_tps_list),
        "prefill_tps_stddev": std(prefill_tps_list),
        "runs": len(turn_results),
        "all_ttfts": ttfts,
        "all_e2els": e2els,
    }
    return summary


def main():
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
    parser.add_argument("--turns",
                        default="0,16384,128|16384,128,128|16384,4096,512",
                        help="Turn configs: prefix,new_tokens,output_tokens "
                             "(pipe-separated)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    url = args.url or DEFAULT_URLS[args.backend]
    send_fn = BACKENDS[args.backend]["send"]
    warmup_fn = BACKENDS[args.backend]["warmup"]
    tokenizer = get_tokenizer(args.tokenizer)

    # Parse turn configs
    turns = []
    for spec in args.turns.split("|"):
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

    print(f"\n{'='*60}")
    print(f"  {platform.node()} — {args.model} via {args.backend}")
    print(f"  URL: {url}")
    print(f"  Turns: {args.turns}")
    if args.repeats > 1:
        print(f"  Repeats: {args.repeats} per turn")
    print(f"{'='*60}\n")

    warmup_fn(url, args.model)

    print(f"{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}\n")

    all_summaries = []
    all_results = []

    for i, (prefix_len, new_len, out_len) in enumerate(turns):
        if prefix_len == 0:
            # Cold turn: all tokens are prefilled
            text = generate_text(new_len, tokenizer, seed=42)
            label = f"Turn {i+1}: {new_len//1024}K cold"
            prefill_tokens = None  # measure() will count all input tokens
        else:
            # Warm turn: only new_len tokens are prefilled, prefix is cached
            text = prefix_text + " " + suffix_texts[i]
            label = f"Turn {i+1}: {prefix_len//1024}K cached + {new_len} new"
            prefill_tokens = new_len

        turn_results = run_turn(
            label, url, args.model, text, out_len, send_fn, tokenizer,
            args.repeats, prefill_tokens=prefill_tokens)
        all_results.extend(turn_results)
        all_summaries.append(summarize(turn_results))

    # Summary table
    print(f"{'='*60}")
    print(f"  SUMMARY — {platform.node()} — {args.model} ({args.backend})")
    print(f"{'='*60}")

    if args.repeats > 1:
        print(f"  {'Turn':<30s} {'TTFT med':>9s} {'±':>5s} "
              f"{'E2E med':>9s} {'±':>5s} "
              f"{'prefill':>9s} {'decode':>9s}")
        print(f"  {'-'*80}")
        for s in all_summaries:
            print(f"  {s['label']:<30s} {s['ttft_median_ms']:>8.0f}ms "
                  f"{s['ttft_stddev_ms']:>4.0f} "
                  f"{s['e2e_median_ms']:>8.0f}ms "
                  f"{s['e2e_stddev_ms']:>4.0f} "
                  f"{s['prefill_tps_median']:>7.0f}t/s "
                  f"{s['decode_tps_median']:>7.1f}t/s")
    else:
        print(f"  {'Turn':<30s} {'TTFT':>9s} {'E2E':>9s} "
              f"{'prefill':>9s} {'decode':>9s}")
        print(f"  {'-'*70}")
        for s in all_summaries:
            print(f"  {s['label']:<30s} {s['ttft_median_ms']:>8.0f}ms "
                  f"{s['e2e_median_ms']:>8.0f}ms "
                  f"{s['prefill_tps_median']:>7.0f}t/s "
                  f"{s['decode_tps_median']:>7.1f}t/s")

    if args.output:
        out = {
            "host": platform.node(),
            "model": args.model,
            "tokenizer": args.tokenizer,
            "backend": args.backend,
            "url": url,
            "turns": args.turns,
            "repeats": args.repeats,
            "summaries": all_summaries,
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
