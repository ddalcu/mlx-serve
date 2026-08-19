# Note to testers: DFlash2 (M5 Max) + ANE prefill (M3 Ultra / M5 Max)

Two things I want measured on hardware I don't have. Everything below was validated on an M4 Max, the open questions are exactly the ones your chips answer. Background stories live in `docs/gotchas/engine-mlx.md` (sections "DFlash 2 port" and the ANE rules in root `CLAUDE.md`).

## Setup (both tests)

- macOS 26.2 or newer (hard floor, the self-built mlx needs it), Xcode 26.2+ with the Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain` if `xcrun -sdk macosx metal --version` fails).
- Build, the easy way (full details in `docs/building.md`):

  ```bash
  git clone --recurse-submodules https://github.com/ddalcu/mlx-serve && cd mlx-serve
  brew bundle install --file=Brewfile
  ./app/build.sh
  ```

  That one script stages everything (zig nightly, llama.cpp, mlx with NAX kernels asserted), builds the app + server, and signs ad-hoc with no Apple account. The server binary you bench is `zig-out/bin/mlx-serve`. Server-only alternative: `./scripts/fetch-zig.sh`, `./scripts/fetch-llama.sh && ./scripts/build-mlx.sh`, then `zig build -Doptimize=ReleaseFast`. Never bench a bare `zig build`, Debug is 2-4x slower and every number is fiction.
- Models (HF, all ungated):
  - Qwen trunk: https://huggingface.co/ddalcu/Qwen3.8-27B-MLX-Serve-4bit (or any Qwen3.8-27B MLX-Serve pack you have; it ships the MTP head)
  - Qwen DFlash2 drafter: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 (3.85 GB)
  - Muse trunk (optional round): https://huggingface.co/ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit (~33 GB; its `drafter/` subdir is the v1 assistant and auto-loads)
  - Muse DFlash2 drafter (optional round): https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2 (5.5 GB)
- Drafters are dense bf16 and get quantized to 8-bit at load automatically, nothing to convert.
- llmprobe is on npm, `npx -y llmprobe@latest`. `tests/bench.sh` drives it for you.
- Keep everything else off the GPU while measuring. Background GPU traffic poisoned cells 2x for us.

## Test 1: DFlash2 vs MTP (M5 Max only)

On M4 Max, MTP wins (novel 65.7 vs 64.6/62.5 tok/s). The reason is hardware: the DFlash block is the trunk verify width, and without the NAX m16 tile the verify falls off a cliff past M=7, so the block caps at 5. Your M5 keeps the checkpoint's block automatically (8 for the Qwen drafter, 16 for the muse one), and at block 8 the DFlash2 selector already measured +16% acceptance over plain argmax drafts on the M4 (3.69 vs 3.17 accepted/round). Whether that beats MTP on M5 is the question.

For calibration, inco's own H200 numbers for the muse drafter (block 16, SGLang): acceptance length 4.4 to 6.6 depending on task, 2.6x to 4.6x over autoregressive. Two metric notes: their "acceptance length" counts accepted+1, so our `avg_per_round` is theirs minus 1, and at block 5 our echo cells already sit at the ceiling (4.97 on their scale), which is why the M4 can't see the difference.

Traps that will make you measure the wrong thing:

1. Boot the DFlash2 arm with `--no-mtp --drafter <drafter-dir>`. The Qwen3.8 packs ship an in-checkpoint MTP head and spec priority is MTP > dflash, so without `--no-mtp` you get green dflash boot lines and MTP decode.
2. Proof of engagement is `[spec-stats] mode=dflash` (or `mode=mtp`) in the server log, never the launch flags. A cell with no engagement line is measuring plain decode. (DFlash is exempt from the prompt-side ngram spec gate as of this build, so plain llmprobe/curl bodies engage it on novel prompts; only its own runtime yield gate can disable it, and that line says so.)
3. Drafter acceptance is a THINKING-MODE property. These sidecars are trained on reasoning-mode outputs: on muse, the same prompt measured 12.5% per-draft with thinking off vs 47.8% with thinking on, and inco's eval runs high reasoning strength. Bench with thinking ON (`"enable_thinking": true`, and for muse that's its natural mode anyway), or you're benching the drafter off-distribution.

Procedure:

- Arms, one boot each, all with `--prefix-cache-entries 0 --skip-mem-preflight`:
  - `mtp`: default boot (MTP is default on this trunk)
  - `dflash2`: `--no-mtp --drafter <dir>`
  - `dflash2-v1`: same + env `MLX_SERVE_DFLASH_SELECTOR=0` (isolates the selector's contribution)
  - `serial`: `--no-mtp --no-pld`
- Two cells per arm: a novel prompt (~300-600 tokens out, thinking on) and an echo prompt (paste a ~1000 token doc, ask it to repeat it, ~1200 tokens out). Temperature 0, 3 reps, read `decode:` tok/s off the server's own log lines, take medians.
- Counterbalance: run the arms forward then reversed (8 boots), thermal drift reads as an effect otherwise.
- Also try `--draft-block-size 6` and `7` on the dflash2 arm if block 8 disappoints, the optimum may sit between.
- Optional second round on muse: trunk `ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit` (its `drafter/` subdir is the v1 assistant and auto-loads, that's your v1 arm), DFlash2 drafter `incoai/Muse-Glimmer-30B-DFlash2` (5.5 GB, needs this build: its config declares a logit softcap + output multiplier the engine now applies). Muse has no MTP, so the fight is v1 vs DFlash2 at block 16, thinking on. On M4 they tied because block 5 capped both at the ceiling.

Report: the median table, plus the `[spec-stats]` line per cell (avg_per_round + per_draft_pct + block_size). Name the machine and the pack.

### The acceptance gap is already proven on M4, only the verify price is missing

Measured here (M4 Max, Qwen trunk, thinking on, greedy, same prompt, matched at 7 drafts per round):

| arm | drafts/round | accepted/round | per-draft hit |
| :-- | --: | --: | --: |
| DFlash2, block 8 | 7.0 (always) | 3.18 | 45.4% |
| MTP, forced depth 7 | 3.9 avg | 2.13 | 54.8% |

MTP's individual drafts are more accurate but its chained drafting cannot hold depth (each draft conditions on predicted hiddens, so the chain self-limits at ~4 even when forced to 7); DFlash2 proposes all 7 in one parallel block every round and lands ~50% more accepted tokens per verification step. That matches inco's H200 direction (4.80 vs 4.28 on their accepted+1 scale; ours reads 4.18 vs 3.13 on one prompt). On M4 the block-8 verify is overpriced so MTP still wins tok/s. On your M5 the verify should be cheap, so DFlash2 winning end-to-end is the expected outcome; your job is to confirm or kill that.

Reproduce the acceptance cell alongside the tok/s arms (one boot each, same thinking-on prompt):

```bash
# DFlash2 acceptance at block 8 (7 drafts/round)
mlx-serve --model <qwen-pack> --drafter <dflash2-dir> --no-mtp --draft-block-size 8 \
  --serve --port 8123 --prefix-cache-entries 0 --skip-mem-preflight

# MTP acceptance at matched depth (forced, acceptance metric ONLY -- the
# adaptive default is the fair tok/s arm, forced depth is deliberately
# uneconomic and exists to compare drafters at equal width)
MLX_SERVE_MTP_ADAPTIVE=0 mlx-serve --model <qwen-pack> --mtp --mtp-depth 7 \
  --serve --port 8123 --prefix-cache-entries 0 --skip-mem-preflight
```

Read `avg_per_round`, `per_draft_pct`, and (mtp) `drafted=` off the `[spec-stats]` line after a request with `"enable_thinking": true`, temperature 0, max_tokens 600. Note whether MTP's drafted/attempts ratio still self-limits below 7 on your box; that number is part of the result.

## Test 2: ANE prefill offload (M3 Ultra and M5 Max)

`--ane-prefill` (opt-in, lossy int8/fp16 by design) offloads dense MLP + GDN input projections to the Neural Engine during prefill. Measured M4 Max on the 27B: +19%/+26% prefill at 16k/32k, decode untouched. M3 Ultra (older ANE gen, two of them) and M5 Max are unmeasured.

Procedure:

- Boot: `mlx-serve --model <pack> --serve --port 8123 --ane-prefill --prefix-cache-entries 0`.
- First boot compiles ANE programs. Watch for `[ane] ... ready: N/M` in the log. If N < M the compile ran out of internal disk budget, just reboot the server, the cache converges across boots. Measure only after full coverage. Needs a few GB free on the INTERNAL disk (aned stages there regardless of TMPDIR).
- Measure with llmprobe, prefill is the number that moves:
  - `npx -y llmprobe@latest --bench-only -u http://127.0.0.1:8123 -m <model-id>` gives the decode/prefill ladder, or use `./tests/bench.sh --url 127.0.0.1:8123 -m <model-id>` for paste-ready rows.
  - A/B by BOOT: on, off, on, off (same session, alternate arms per boot, never block-of-N). The off arm is the same command without `--ane-prefill`.
  - The 8k/16k/32k prefill rungs are the signal. Ignore decode, it must not move (if it does, that's a bug, report it).
- Verify engagement before believing any number: the log carries per-seam one-shot engagement lines, and `GET /props` has an `"ane"` object with eval counts. Zero engagements with a green boot means the tile width didn't match, report the log.
- Knobs worth one sweep each on new silicon (M4's defaults were measured on M4 and may not transfer):
  - `MLX_SERVE_ANE_SPLIT=0.30 / 0.40 / 0.45 / 0.50` (share of channels on the ANE; M4 optimum was 0.45 channel mode, rollover at 0.50 where the ANE becomes the critical path)
  - `MLX_SERVE_ANE_MODE=row` (the older token-row split, M4 optimum 0.40)
  - `MLX_SERVE_ANE_GDN=0` (MLP-only, isolates the GDN seam)
- RAM: the ANE holds an int8 copy of what it serves, ~11 GB on the 27B at channel 0.45. The admission gate bills it, plenty of headroom on your boxes.

Report: prefill tok/s per rung per arm (medians of the alternating boots), the winning share, and the `/props` ane object from one on-boot.

## Dual ANE on the M3 Ultra

Short answer: our code does NOT use the second ANE today, and I'd like to know if it can.

Current state: `lib/ane/ane_bridge.m` loads `_ANEInMemoryModel` programs and evaluates with `loadWithQoS:options:` / `evaluateWithQoS:options:request:` passing empty options. No device is ever named, aned schedules wherever it wants, which in practice means one ANE instance. On top of that the engine (`src/ane.zig`) is built strictly serial: one eval in flight, and the I/O surfaces are SHARED per shape class, which is only legal because evals never overlap.

Cheap probe first (no code, do this before implementing anything):

1. Count instances: `ioreg -l | grep -ci h11ane` (or look for the ANE service class of your gen), and check `sudo powermetrics --samplers ane_power -i 1000` during a 32k `--ane-prefill` run. If total ANE power on the Ultra roughly matches a single Max die's, the second ANE is idle.

If it's idle and you want to try lighting it (this is a good task for your coding agent, it's exploratory):

1. Find the affinity handle. Class-dump `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework` and look at `_ANEDeviceController` / `_ANEClient` and the accepted keys of the `options:` dicts we currently pass as `@{}` in `msv_ane_model_create` and `msv_ane_model_eval`. You're looking for anything naming a device index, instance, or affinity. If nothing exists at the model/request level, check whether separate client connections get balanced across instances.
2. If you can pin a program to an instance, the right split is the CHANNEL split we already have: give each ANE half of the current ANE channel share and kick both evals concurrently, then add both partials at the existing seam. That halves the ANE critical path, which is exactly what caps the share at 0.45/0.50 today, so expect the optimal `MLX_SERVE_ANE_SPLIT` to move up. Engine changes needed in `src/ane.zig`: two engine instances per layer, per-instance OUTPUT planes (the shared-plane trick assumes serial evals, the shared INPUT plane is fine since both only read it), kick-both/wait-both on the eval thread, and both halves billed in `engineBillBytes`. Respect the existing constraints per half: fp16 plane rows must be multiples of 32, channel boundaries align to 128.
3. Verify with powermetrics that both instances draw power, and A/B against single-ANE with the same boot-alternation discipline as above. Also note the M3-gen ANE is slower per instance than M4's, so even a working dual split may land near M4 single-ANE numbers. That's still a result worth having.

If the framework simply refuses to address the second ANE, that's also a result, write down what you tried.

## What to send back

- The median tables (arm x cell), with the engagement line per cell.
- The acceptance cells (DFlash2 block 8 vs MTP forced depth 7): avg_per_round, per_draft_pct, and MTP's drafted/attempts ratio.
- Server logs from one boot per arm (`~/.mlx-serve/logs/` or wherever you redirected them).
- `GET /props` ane object from an ANE-on boot.
- Machine, macOS version, pack used, and which llmprobe version.
