# Nemotron Bolt / LiLiCorr measurements

These are anonymous measurements from one session. Hardware, operating-system details and local build identifiers are omitted. These results are not suitable for cross-machine comparisons.

| Prompt | Drafter off tok/s | LiLiCorr on tok/s | On/off | Accepted drafts/round |
|---|---:|---:|---:|---:|
| explanation | 73.49 | 94.22 | 1.28× | 3.68 |
| code | 70.36 | 92.89 | 1.32× | 4.65 |
| narrative | 60.55 | 57.73 | 0.95× | 1.62 |

Each cell is the median of three requests capped at 192 output tokens. The same loaded model and drafter serve both arms; `enable_drafter` changes per request. One 64-token warmup per arm is excluded. Arm order alternates off/on, on/off, off/on for each prompt. Temperature 0, top-k 1, seed 42, thinking off, context 32768, KV quantization off, prefix caches off, PLD off and MTP off. The process uses an isolated HOME. The harness schedules no concurrent inference; this is a desktop measurement, not a thermally controlled lab run.

The target is [thelogicalgate/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX](https://huggingface.co/thelogicalgate/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX/tree/5fa5821404f5e5d76ad33d8f102d688577830385), revision `5fa5821404f5e5d76ad33d8f102d688577830385`: 4-bit affine/group-64 language weights, bf16 residual/vision, and the shipped bf16 LiLiCorr sidecar (block 8, candidate top-k 8). Its flattened converted config is supplemented with the original native RADIO metadata from `config.hf.json`; weight files are unchanged. Engine source is PR head `4da739a93bad1d5cd347b5a4a60ac5cbfc11e2ab` plus the accompanying engine review fixes, built with Zig 0.17.0 ReleaseFast. Test settings and individual measurements are in [results.json](results.json).

Throughput is the server's decode timing, excluding prefill. Acceptance is accepted draft tokens divided by speculative rounds, excluding the committed `t1` token; rounds after the runtime gate falls back to plain decoding are not speculative rounds. The story prompt triggers that existing low-yield gate in all three runs. LiLiCorr improves these explanation/code cases but loses on this story case; it is not a universal speedup. All nine greedy on/off pairs produce identical content. Individual timings, content hashes, token counts, acceptance counts and runtime-gate outcomes are retained in `results.json`.

Reproduce with a local pack retaining `vision_config.model_type: "radio"`:

```sh
python3 tests/test_nemotron_bolt.py --model /path/to/Bolt-MLX --output /path/to/results --checks bench --reps 3
```

The three prompts live in that script. Use `--checks vision` for image plus 128/256-frame video checks, or `--checks restore` for a repeated request that must hit the prefix cache and complete with LiLiCorr. The GUI is unchanged and still samples 16 frames total; the engine accepts the complete client-supplied frame array.
