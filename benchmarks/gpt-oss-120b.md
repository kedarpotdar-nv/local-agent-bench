# GPT-OSS-120B

Dense 120B parameter model.

## DGX Spark instructions

### vLLM

#### Server

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 --name vllm-bench \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/vllm_config.yaml:/workspace/config.yaml \
    -e TORCH_CUDA_ARCH_LIST=10.0 \
    -e PYTHONNOUSERSITE=1 \
    -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
    nvcr.io/nvidia/vllm:26.02-py3 \
    vllm serve openai/gpt-oss-120b \
    --config /workspace/config.yaml \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 8 \
    --disable-log-requests
```

#### Benchmark

```bash
python3 local_agent_bench.py \
    --backend vllm \
    --model openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --url http://localhost:8000 \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output vllm_gptoss120b.json
```

### Ollama

#### Server

```bash
# TODO: Ollama launch command
```

#### Benchmark

```bash
python3 local_agent_bench.py \
    --backend ollama \
    --model gpt-oss:120b \
    --tokenizer openai/gpt-oss-120b \
    --turns "0,16384,128|16384,128,128|16384,4096,512" \
    --output ollama_gptoss120b.json
```

## Notes

- Server launch commands TBD
