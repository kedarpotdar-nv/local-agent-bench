# Nemotron-3 Super

## Server (vLLM)

```bash
# TODO: vLLM launch command
```

## Benchmark

```bash
# Clone
git clone https://github.com/kedarpotdar-nv/local-agent-bench.git
cd local-agent-bench

# vLLM
python3 local_agent_bench.py \
    --backend vllm \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
    --url http://localhost:8000 \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output vllm_nemotron3_super.json

# Ollama (stop vLLM first)
docker stop vllm-bench

ollama pull nemotron-3-super

python3 local_agent_bench.py \
    --backend ollama \
    --model nemotron-3-super \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output ollama_nemotron3_super.json
```

## Notes

- MoE model, 120B total / 12B active parameters, FP8 quantized
