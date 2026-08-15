# Performance & tuning

Apple M4 Max, identical weights per engine. [benchmarks.md](../benchmarks.md) tracks decode tok/s release by release, plus the current head-to-head against other engines. Numbers come from `tests/bench.sh`, which boots the server and lets [llmprobe](https://github.com/ddalcu/llmprobe) take them: warmup discarded, median of three, same protocol for everyone.

![mlx-serve vs LM Studio · oMLX · MTPLX — Gemma 4 + Qwen 3.6, code completion (M4 Max)](perf-vs-engines.png)

*Code completion decode tok/s, v26.8.3, against LM Studio 0.4.19+2, oMLX 0.5.2 and MTPLX 2.5.3. Every bar is that engine on its **shipping defaults** — llmprobe measures the server that is running, so there is no best-config collapse and no per-model tuning. All four engines load the identical MLX weight files. MTPLX shows 0 where it can't run (it needs its own MTP artifacts), and LM Studio is absent on two rows it has no copy of. Geomean decode: **+26% over LM Studio** across the four shared models and **+25% over oMLX** across all six, with prefill +36% and +10%. The two head-to-heads that matter are on the competitors' own checkpoints: **+23% decode over oMLX** on its oQ4e (prefill level), and **+10% decode / +17% prefill over MTPLX** on its own MTPLX-Optimized build.*

## Speculative decoding

Four flavors, all greedy-equivalent (byte-identical at temp=0 within the first 30 tokens; mathematically exact at temp > 0 via the Leviathan probability-ratio sampler):

- **Native MTP** (Qwen 3.5/3.6/3.8) — checkpoints with a trained multi-token-prediction head (sidecar or baked in, like the Qwen 3.8 27B build) draft with the model's *own* head, with a controller that self-tunes depth per request. MoE sidecars supported. Auto-loads, zero setup.
- **DFlash draft companions** — a model folder can ship its own `drafter/` block-draft companion (the Muse-Glimmer builds do); the server loads it with the model, switching models keeps the speedup, and the draft size adapts to the Mac. About 2x on Muse-Glimmer. `--no-drafter` turns it off.
- **PLD** (Prompt Lookup Decoding) — model-agnostic n-gram match in `prompt + generated_tokens`. Default-on, no per-model setup. Wins on agent loops, RAG and code editing, anywhere the answer echoes the prompt.
- **Gemma 4 assistant drafter** — Google's small 4-layer cross-attention drafters, opt-in via `--drafter <dir>`. Cross-attends into the target's KV cache, so no weights are duplicated.

All four share an **adaptive prompt-time gate**: a 3-gram repetition score auto-disables speculation on novel content, so creative writing and one-shot Q&A run at parity instead of paying verify overhead. A **runtime acceptance gate** disables speculation mid-decode if per-draft acceptance falls below break-even, sticky for the rest of the request. Both apply across all four API surfaces, streaming and non-streaming, including requests with tools. Agentic tool loops are speculative decoding's best workload.

## Long context

Since v26.8.6, sliding-window models trim every attention read to their window: prompt chunks and speculative steps too, not just single-token decode. Nothing to turn on. Measured against v26.8.5 on an M4 Max, median of three runs per point:

| Model | | 16k context | 64k context |
|---|---|---|---|
| Muse-Glimmer 30B 4-bit | decode | 24.7 → **40.2 tok/s** | 8.6 → **21.0 tok/s** |
| Laguna XS 2.1 NVFP4 | prompt | 609 → **774 tok/s** | 236 → **540 tok/s** |
| Laguna XS 2.1 NVFP4 | decode | 62.7 → **77.7 tok/s** | 34.9 → **46.8 tok/s** |

The gain grows with the conversation; below roughly 8k there is nothing to trim yet.

## Tuning

The defaults are already the fast path. **If you have plenty of RAM, the fastest configuration is the one you get out of the box.** Every speed-relevant knob is a memory-for-speed trade in one direction or the other.

| Knob (CLI flag / Settings) | Default | What it does to speed | Flip it when… |
|---|---|---|---|
| **KV cache quantization** (`--kv-quant`) | off | ≈ **−10% decode** at 2–4K context; at long context the fused packed reads (auto from 8K) cut the penalty to ≈ −18% at 42K (was −47%). Saves 2× (8-bit) / 4× (4-bit) KV RAM. | …memory is the constraint: long contexts or big models on a 16 GB Mac. |
| **PLD** (`--pld`) | on | Large wins on agent loops, code editing and RAG. Auto-gates itself off on novel prose. | Leave on. `--no-pld` only for apples-to-apples benchmarks. |
| **Sampling** (`temperature` / `top_p` / `top_k`) | model defaults | Full sampling costs **~6% decode** vs greedy. | Temp 0 for benchmarks; keep sampling for chat quality. |
| **Continuous batching** (`--max-concurrent`) | 1 | ~1.6× *total* throughput at 4-way on dense models, at some per-request latency cost. | …several clients share the server. |
| **Prefix cache** (`--prefix-cache-entries/-mem`) | on | Repeated system prompts and multi-turn chats skip re-prefill. | Leave on. Cap entries on RAM-tight Macs (the app does this automatically). |
| **Native MTP** (auto when the checkpoint ships a sidecar) | on | Large gains on code and coding-agent loops. | Leave on. `--no-mtp` for benchmarks; per-request `enable_mtp` opts MoE trunks in. |
| **Drafter** (`--drafter`) | auto in the app | Big gains on Gemma 4 dense code-edit loops (+52% on E4B code, 117 → 178 tok/s). The app downloads and pairs the drafter with dense Gemma 4 models by itself; on the CLI it stays opt-in. | Leave on. The Settings toggle turns auto-pairing off. |
| **Fast decode for bf16-attention models** (`--no-decode-attn-quant` to disable) | on | Serves dense bf16 attention weights through a decode-only quantized copy: −23% per-token time on models that ship bf16 attention (poolside Laguna). Prefill keeps the dense weights. Can vary output wording slightly on those models; quantized checkpoints are unaffected. | …you want output on such a model to match the fully dense path byte for byte. |

Building from source? **Always `zig build -Doptimize=ReleaseFast`.** A bare `zig build` produces a Debug binary that's 2–4× slower and looks like a regression.
