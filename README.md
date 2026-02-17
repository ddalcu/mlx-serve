# mlx-serve

A native Zig binary that runs MLX-format language models on Apple Silicon. Wraps Apple's `mlx-c` library to load quantized models and serve an OpenAI-compatible chat API. No Python runtime needed.

## Supported models

| Architecture | Tested with | Chat format |
|---|---|---|
| **Gemma-3** | `gemma-3-12b-it-qat-4bit` | Gemma turns |
| **Qwen3** | `Qwen3-4B-Instruct-2507-MLX-4bit` | ChatML |
| **Llama-family** | Llama, Mistral (same architecture) | ChatML |

Any 4-bit quantized MLX model using one of the above architectures should work.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Zig 0.15+](https://ziglang.org/download/)
- mlx-c (installs MLX as a dependency):

```bash
brew install mlx-c
```

## Downloading models

Install the Hugging Face CLI and download a model:

```bash
pip install huggingface-hub
```

**Gemma-3 12B (4-bit, ~8 GB):**

```bash
huggingface-cli download mlx-community/gemma-3-12b-it-qat-4bit --local-dir /path/to/gemma-3-12b
```

**Qwen3-4B (4-bit, ~2 GB):**

```bash
huggingface-cli download lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit --local-dir /path/to/qwen3-4b
```

## Build

```bash
zig build
```

The binary is output to `./zig-out/bin/mlx-serve`.

## Usage

### Interactive mode

Run a single prompt and print the response:

```bash
./zig-out/bin/mlx-serve --model /path/to/model --prompt "What is 2+2?"
```

### HTTP server mode

Start an OpenAI-compatible API server:

```bash
./zig-out/bin/mlx-serve --model /path/to/model --serve --port 8080
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | required | Path to the MLX model directory |
| `--prompt TEXT` | `"Hello"` | Prompt text for interactive mode |
| `--max-tokens N` | `100` | Maximum tokens to generate |
| `--temp F` | `0.0` | Sampling temperature (0 = greedy) |
| `--serve` | off | Start the HTTP server instead of interactive mode |
| `--port N` | `8080` | Port for the HTTP server |

## API

### GET /v1/models

Returns the loaded model info.

```bash
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3",
      "object": "model",
      "owned_by": "mlx-serve"
    }
  ]
}
```

### POST /v1/chat/completions

Generate a chat completion. Compatible with the OpenAI API format.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a haiku about programming."}
    ],
    "max_tokens": 256
  }'
```

```json
{
  "id": "chatcmpl-1771230500371",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Code lines take their form slow,\nLogic blooms, a digital art,\nBugs hide, then disappear."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 17,
    "completion_tokens": 23,
    "total_tokens": 40
  }
}
```

#### Request fields

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | array | required | Array of `{"role": "...", "content": "..."}` objects |
| `max_tokens` | int | `256` | Maximum tokens to generate |

## Performance

Benchmarked on Apple M4 (16 GB unified memory), 256-token generation:

### Gemma-3 12B (4-bit quantized)

| Metric | mlx-serve | mlx_lm (Python) | Delta |
|---|---|---|---|
| Prefill | ~29 tok/s | 32-38 tok/s | ~91% |
| Decode | ~13.5 tok/s | ~14.4 tok/s | ~94% |
| Peak memory | 6.75 GB | 7.3 GB | 8% less |

### Qwen3-4B (4-bit quantized)

| Metric | mlx-serve | mlx_lm (Python) | Delta |
|---|---|---|---|
| Prefill | ~220 tok/s | ~121 tok/s | 1.8x faster |
| Decode | ~37 tok/s | ~40 tok/s | ~91% |
| Peak memory | 2.17 GB | 2.38 GB | 9% less |

The decode gap is primarily due to mlx_lm using `mx.compile` on the forward pass, which the mlx-c API does not support with stateful KV caches.

## Project structure

```
src/
  main.zig          Entry point, CLI parsing
  mlx.zig           FFI bindings for mlx-c
  model.zig         Config parsing and weight loading (safetensors)
  tokenizer.zig     BPE tokenizer (SentencePiece and byte-level)
  transformer.zig   Forward pass: embedding, attention, MLP, norms
  generate.zig      Autoregressive generation with temperature sampling
  chat.zig          Chat template formatting (Gemma turns, ChatML)
  server.zig        HTTP server with OpenAI-compatible JSON API
```

## Limitations

- Single-threaded server (one request at a time)
- No streaming responses
- No top-k or top-p sampling (temperature only)
- Quantized models only (bits and group size read from config)

## License

MIT
