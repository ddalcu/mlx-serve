# Three-way serving-backend comparison

Target model: **Qwen3.6-35B-A3B** (35B MoE, 3B active).

- **mlx-serve**: our Zig server, MLX 6-bit affine (group 64), ~27 GB.
- **mlx-lm**: Apple's reference Python server, same MLX 6-bit weights.
- **llama.cpp**: upstream C++ server, GGUF Q6_K (~30 GB).

Raw metrics live in `tests/compare_backends.tsv`. Per-backend logs live in
`tests/pi-results/compare_<backend>.*`. This file is appended by
`tests/compare_backends_run.sh` and then hand-edited for analysis.

## How to run

```
bash tests/compare_backends_setup.sh    # one-time: downloads llama.cpp + Q6_K GGUF
bash tests/compare_backends_run.sh      # ~45 min end-to-end on one M-series box
```

Pass backend names to the driver to run only a subset:
`bash tests/compare_backends_run.sh mlx-serve llama.cpp`.

## Analysis (hand-written — edit after each run)

### 1. mlx-serve vs mlx-lm on identical weights

_TODO after run: any gap is an implementation gap in our Zig code (prefill
scheduling, KV cache, sampling). This is the primary self-check._

### 2. Tool-call reliability with enable_thinking=false

_TODO after run: parse-success rate across the three; whether llama.cpp and
mlx-lm hit the same garbage shapes we had to repair in chat.zig (v26.4.26)._

### 3. Thinking-tag streaming

_TODO after run: mlx-serve emits reasoning_content live; note whether
llama.cpp / mlx-lm emit it in-band as `<think>` text, in a separate field, or
not at all. Which is more agent-friendly?_

### 4. Agent-loop quality on the same pi harness

_TODO after run: scores should bracket each other within ±1 if the three
backends are equally healthy. A 3-point gap means something is wrong._

## Runs

_Runs are appended below automatically by `compare_backends_run.sh`._
