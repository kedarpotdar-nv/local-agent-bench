# Qwen3.5-35B-A3B

MoE model, 3B active parameters. Thinking model (reasoning tokens before content).

## Server (vLLM)

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 --name vllm-bench \
    -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface \
    -e TORCH_CUDA_ARCH_LIST=10.0 \
    -e PYTHONNOUSERSITE=1 \
    vllm/vllm-openai:v0.18.0 \
    Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 4 \
    --max-cudagraph-capture-size 128 \
    --attention-backend TRITON_ATTN \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

## Benchmark

```bash
# Clone
git clone https://github.com/kedarpotdar-nv/local-agent-bench.git
cd local-agent-bench

# vLLM
python3 local_agent_bench.py \
    --backend vllm \
    --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --tokenizer Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --url http://localhost:8000 \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output vllm_qwen35_gptq.json

# Ollama (stop vLLM first)
docker stop vllm-bench
python3 local_agent_bench.py \
    --backend ollama \
    --model qwen3.5:35b \
    --tokenizer Qwen/Qwen3.5-35B-A3B \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output ollama_qwen35.json
```

## Notes

- GPTQ-Int4 quantization, fp8 KV cache
- Prefix caching enabled (`--enable-prefix-caching`)
