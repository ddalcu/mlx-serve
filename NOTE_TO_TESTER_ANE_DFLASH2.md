# Note to testers: DFlash2 (M3 Ultra) + ANE prefill (M3 Ultra)

**M3 Ultra ANE verdict (extensive test received 2026-08-19 — thank you)**: ANE prefill WORKS on the M3 Ultra, but the default share was worth ~nothing there (channel 0.45: +1% at 32k). That 0.45 is not an "M4" number as such: `ane.defaultShare` returns 0.35 for an M3 Ultra and 0.45 for *everything else*, and 0.45 was only ever measured on an M4 Max. The tester's sweep found the optimum at **0.35** (+8.5%/+13.7% at 16k/32k over OFF, post-reboot A/B, reproduced 3x), so this build's default share is now per-silicon: M3 Ultra boots at 0.35 automatically. Also from that round: the second ANE instance is idle (all energy on ANE0_0 — see the dual-ANE section), `powermetrics ane_power` returns empty samples on that build (use `macpow --dump | grep ANE0_`), and `/props` ane now carries `evals`/`eval_failures` so dispatch is verifiable without log-grepping. Test 2 below is ANSWERED for the M3 Ultra; what remains there is the optional dual-ANE exploration.

Two things I want measured on hardware I don't have. Everything below was validated on an M4 Max, the open questions are exactly the ones your chips answer. Background stories live in `docs/gotchas/engine-mlx.md` (sections "DFlash 2 port" and the ANE rules in root `CLAUDE.md`).

**M5 Max verdict (thanks to both testers, 2026-08-19)**: DFlash2 ties MTP at block 8 (35.2 vs 35.3 tok/s novel) — the block-8/NAX lane engages correctly and the selector holds (+17% acceptance, 1.6x accepted/verify vs MTP), there's just no tok/s win on that chip. Not a bug; the fight is settled per machine. ANE prefill LOSES on M5 Max (channel 0.45 median -11%/-7.5% at 16k/32k) — NAX prefill already covers it, so this build gates `--ane-prefill` OFF on NAX-class GPUs (`MLX_SERVE_ANE_FORCE=1` overrides for future silicon). M5 testers: nothing left to bench here, but please confirm the two test fixes at the bottom on your next run. **The open question moves to the M3 Ultra.**

## Power/thermal (laptops especially)

Measure on AC power at high charge. One tester's decode roughly DOUBLED mid-run when the machine went from battery to AC — a battery arm and an AC arm are not the same machine. On laptops, Latin-square or A-B-A the arms so a power/thermal drift reads as drift, not as an effect.

One more machine-state trap from the M3 Ultra round: mid-session, speculative decode collapsed (5.45 → 2.0 tok/step) and STAYED collapsed across mlx-serve restarts and an ANE-OFF control — a macOS reboot fully restored it. Root cause unknown (the OFF control degrading too acquits the build; the state lived in the OS). If a spec cell collapses and a fresh server boot doesn't recover it, reboot the machine and re-baseline before reporting a regression.

## Setup (all tests)

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

## Test 1: DFlash2 vs MTP (M3 Ultra)

On M4 Max, MTP wins (novel 65.7 vs 64.6/62.5 tok/s); on M5 Max they tie (see the verdict above). The M3 Ultra is the machine DFlash2 should actually WIN on: oMLX PRs #2850/#2840 shipped DFlash2 with M3 Ultra numbers of 1.33-1.43x over serial at T=0.7 on our exact model pairing (Qwen3.8-27B + the incoai drafter, block 8). This build's default block cap is now per-silicon — the M3 Ultra defaults to block 8 (matching oMLX's evidence); on older builds pass `--draft-block-size 8` explicitly. Whether our engine reproduces their 1.33-1.43x there is the question.

For calibration, inco's own H200 numbers for the muse drafter (block 16, SGLang): acceptance length 4.4 to 6.6 depending on task, 2.6x to 4.6x over autoregressive. Two metric notes: their "acceptance length" counts accepted+1, so our `avg_per_round` is theirs minus 1, and at block 5 our echo cells already sit at the ceiling (4.97 on their scale), which is why the M4 can't see the difference.

Traps that will make you measure the wrong thing:

1. Boot the DFlash2 arm with `--no-mtp --drafter <drafter-dir>`. The Qwen3.8 packs ship an in-checkpoint MTP head and spec priority is MTP > dflash, so without `--no-mtp` you get green dflash boot lines and MTP decode.
2. Proof of engagement is `[spec-stats] mode=dflash` (or `mode=mtp`) in the server log, never the launch flags. A cell with no engagement line is measuring plain decode. (DFlash is exempt from the prompt-side ngram spec gate as of this build, so plain llmprobe/curl bodies engage it on novel prompts; only its own runtime yield gate can disable it, and that line says so.)
3. Drafter acceptance is a THINKING-MODE property. These sidecars are trained on reasoning-mode outputs: on muse, the same prompt measured 12.5% per-draft with thinking off vs 47.8% with thinking on, and inco's eval runs high reasoning strength. Bench with thinking ON (`"enable_thinking": true`, and for muse that's its natural mode anyway), or you're benching the drafter off-distribution.

Procedure:

- Run **Test 3 first**. The `mtp` arm below is a default boot, which caps the adaptive controller at depth 3, and that cap has never been checked on an M3 Ultra. If Test 3 finds a better depth on your box, pass it here with `--mtp-depth N` or this comparison undersells MTP.
- Arms, one boot each, all with `--prefix-cache-entries 0 --skip-mem-preflight`:
  - `mtp`: default boot (MTP is default on this trunk), at Test 3's winning depth
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

# MTP acceptance at matched depth. **ACCEPTANCE METRIC ONLY — never use this
# boot as the MTP tok/s arm.** The adaptive default boot is the fair tok/s
# arm; forced depth is deliberately uneconomic and exists to compare
# drafters at equal width. (One tester benched MTP tok/s at forced depth 7 —
# that number undersells MTP and is not comparable to anything.)
MLX_SERVE_MTP_ADAPTIVE=0 mlx-serve --model <qwen-pack> --mtp --mtp-depth 7 \
  --serve --port 8123 --prefix-cache-entries 0 --skip-mem-preflight
```

Read `avg_per_round`, `per_draft_pct`, and (mtp) `drafted=` off the `[spec-stats]` line after a request with `"enable_thinking": true`, temperature 0, max_tokens 600. Note whether MTP's drafted/attempts ratio still self-limits below 7 on your box; that number is part of the result.

## Test 2: ANE prefill offload — ANSWERED (M3 Ultra measured 2026-08-19)

`--ane-prefill` (opt-in, lossy int8/fp16 by design) offloads dense MLP + GDN input projections to the Neural Engine during prefill. Measured M4 Max on the 27B: +19.9%/+26.2% prefill at 16k/32k (re-confirmed 2026-08-20 after the procedure-bank change, 3 boot-alternated reps). M5 Max measured a LOSS — ANE prefill is for M4-and-below, refused by name on NAX-class GPUs. **M3 Ultra measured (27B 4-bit): +8.5%/+13.7% at 16k/32k at channel share 0.35** (the 0.45 default was ~+1% — the older ANE hits its critical-path rollover earlier), decode untouched. This build defaults the M3 Ultra to 0.35 automatically; verify with `/props` (`"share":0.35`, `evals` > 0 after a long prompt). The procedure below stays for reference on any new machine.

Procedure:

- Boot: `mlx-serve --model <pack> --serve --port 8123 --ane-prefill --prefix-cache-entries 0`.
- First boot compiles ANE programs. Watch for `[ane] ... ready: N/M` in the log. If N < M the compile ran out of internal disk budget, just reboot the server, the cache converges across boots. Measure only after full coverage. Needs a few GB free on the INTERNAL disk (aned stages there regardless of TMPDIR).
- Measure with llmprobe, prefill is the number that moves:
  - `npx -y llmprobe@latest --bench-only -u http://127.0.0.1:8123 -m <model-id>` gives the decode/prefill ladder, or use `./tests/bench.sh --url 127.0.0.1:8123 -m <model-id>` for paste-ready rows.
  - **Do NOT pass `--full`.** The default ladder stops at 16k with one run per rung, which is all this test needs and is minutes instead of a quarter hour per arm. `--full` adds a 32k and a 64k rung at 3 reps each; the 64k rung tells us nothing here and the extra time is what made the first M3 Ultra round an all-day job.
  - A/B by BOOT: on, off, on, off (same session, alternate arms per boot, never block-of-N). The off arm is the same command without `--ane-prefill`.
  - The 8k and 16k prefill rungs are the signal, and 16k is the one that decides it. Ignore decode, it must not move (if it does, that's a bug, report it).
  - Optional, only if 16k looks interesting and you want the long-context point: the win grows with context (M3 Ultra measured +8.5% at 16k and +13.7% at 32k), and one 32k reading per arm is a single request rather than a ladder. `tests/prefill_ab.sh` is the shape to crib: send one long prompt with `max_tokens: 4`, read `timings.prompt_ms` off the response, and check `usage.prompt_tokens_details.cached_tokens` is under ~64 so the prefix cache did not serve the body.
- Verify engagement before believing any number: the log carries per-seam one-shot engagement lines, and `GET /props` has an `"ane"` object with eval counts. Zero engagements with a green boot means the tile width didn't match, report the log.
- Knobs worth one sweep each on any silicon whose share has never been measured. Only two have: M4 Max (0.45) and M3 Ultra (0.35), and this build picks those automatically. Every other chip — M1, M2, M3 Max/Pro, M4 Pro — currently boots at 0.45 because that is the fallback, not because anyone measured it there, so a sweep on those is worth real numbers:
  - `MLX_SERVE_ANE_SPLIT=0.30 / 0.40 / 0.45 / 0.50` (share of channels on the ANE; the M4 Max optimum was 0.45 in channel mode, rolling over at 0.50 where the ANE becomes the critical path, and the M3 Ultra's older ANE rolls over earlier at 0.35)
  - `MLX_SERVE_ANE_MODE=row` (the older token-row split, M4 optimum 0.40)
  - `MLX_SERVE_ANE_GDN=0` (MLP-only, isolates the GDN seam)
- RAM: the ANE holds an int8 copy of what it serves, ~11 GB on the 27B at channel 0.45. The admission gate bills it, plenty of headroom on your boxes.

Report: prefill tok/s per rung per arm (medians of the alternating boots), the winning share, and the `/props` ane object from one on-boot. Two boots per arm is enough to call it; four if the two disagree.

## Dual ANE on the M3 Ultra (IMPLEMENTED — please measure)

The exploration from the last round landed. `MLX_SERVE_ANE_DUAL=1` now builds TWO ANE units pinned to the two dies of your Ultra, each computing half the total channel share concurrently while the GPU computes the rest. Default OFF, and it self-disables by name on single-ANE silicon, so nothing changes unless you ask for it.

What changed under the hood, in case a boot misbehaves:

- Programs are now procedure BANKS: every covered layer is one `procedureNNN` inside a single compiled program. The private runtime only accepts about 121 resident model handles and the 27B alone wants 112, so a dual build had to stop making one handle per layer. The ready line now says `... in N banks`.
- The affinity handle is `kANEFAneInstanceHint` plus `kANEFProcedureVariantHint`, passed at compile, load and eval. Instance 0 (single-ANE) still passes the empty dict, so a non-dual boot is byte-identical to your last round.
- Compile time grows. oMLX report 37-40 s of eager bank compilation for a 64+48 layout. Ours logs its own duration and converges across boots through the content-hash cache, same as before.

### Step 1: prove both dies dispatched

Boot with the offload and dual on:

```
MLX_SERVE_ANE_DUAL=1 mlx-serve --model <pack> --serve --ane-prefill --port 8080
```

Then, after one long prompt:

```
curl -s localhost:8080/props | python3 -m json.tool | sed -n '/"ane"/,/}/p'
```

You want `"units": 2` and a `unit_evals` array where BOTH rows have non-zero `evals`. The log also carries a one-shot `[ane] dual engaged: 2 units pinned to instances 1..2, ...` line.

Now the out-of-process half, which is the one that actually matters:

```
macpow --dump | grep ANE0_    # before
# run a 16k prefill
macpow --dump | grep ANE0_    # after
```

Both `ANE0_0` and `ANE0_1` must move. If only `ANE0_0` moves while `/props` shows two units evaluating, the framework accepted our hint and ignored it — that is a real result, write it down and stop there. (`powermetrics --samplers ane_power` still returns empty samples on your build, don't bother.) If instead the boot logs an ANE load failure by name and falls back to one unit, that is also a result: the hint was rejected outright.

### Step 2: A/B dual vs single at the same total share

`MLX_SERVE_ANE_SPLIT` still means the TOTAL fraction of channels taken off the GPU — it is halved across the two units — so 0.35 means the same thing in both arms and every number from last round carries over.

Same boot-alternation discipline as the other cells: one boot per arm, alternate, no `--full`, 8k and 16k prefill from the default llmprobe ladder, medians across at least three boots per arm.

- Arm A: `MLX_SERVE_ANE_SPLIT=0.35` (single ANE, the current default)
- Arm B: `MLX_SERVE_ANE_DUAL=1 MLX_SERVE_ANE_SPLIT=0.35`

Decode must not move in either arm. If it does, something is wrong with the seam, not the ANE.

### Step 3: re-sweep the share upward

This is the part I most want. The 0.35 optimum exists because the single M3-gen ANE saturates and becomes the critical path — that is exactly what a second die removes. So under dual the optimum should sit HIGHER, and I have no way to guess where. oMLX landed at 0.53 for MLP and 0.50 for GDN on the same silicon.

Sweep `MLX_SERVE_ANE_DUAL=1` with `MLX_SERVE_ANE_SPLIT` at 0.35 / 0.45 / 0.55 / 0.65, one boot per cell, 16k and 32k. Report the medians and the `/props` `int8_bytes` per cell (the resident copy scales with the share, and on a 27B that is real memory).

If a cell fails to build, the log will say why by name. A bank that the runtime refuses halves down a ladder automatically, so partial coverage should not happen silently — but if the ready line reports fewer mlp/gdn layers than the arms above, that is worth reporting on its own.

### If it misbehaves

`MLX_SERVE_ANE_DUAL=0` (or just omitting it) is a full restore to the single-ANE path. `MLX_SERVE_ANE_BANK_MAX_BYTES=<bytes>` forces smaller banks if you hit a load failure that looks size-related. `MLX_SERVE_ANE_DUAL_SHARE_INPUT=1` makes both units read one input surface instead of getting a memcpy'd copy — do NOT use it for the measurements above, it is an untested optimisation and its failure mode is silently wrong output, not an error.

## Test 3: MTP draft depth on the M3 Ultra (do this BEFORE Test 1)

We have no draft-depth data for the M3 Ultra at all. The previous round measured ANE share and the DFlash block cap there, never this. That matters for Test 1: its MTP tok/s arm is the default boot, and the default caps the adaptive controller at depth 3. If your machine's optimum is 4 or more, that arm is running under-configured and the DFlash2-vs-MTP verdict would be skewed against MTP. So run this first and use the winning depth for Test 1's MTP arm.

Why we think the cap may be wrong on your box: the controller picks a depth inside `[1, --mtp-depth]` using a cost table (`MTP_EV_DEFAULT_COSTS`) that was fit on an M4 Max. Every *calibrated* per-silicon profile we ship is M5/NAX only, so M1 through M4 and the M3 Ultra all run the M4 Max numbers. An M1 Pro 32 GB user reports `--mtp-depth 4` beating the default.

Measured here on an M4 Max for reference (Qwen3.8-27B oQ4e, greedy, 2 boot-alternated reps, decode tok/s):

| workload | depth 3 | depth 4 | delta | accepted/round |
| :-- | --: | --: | --: | :-- |
| echo, high acceptance | 73.4 / 74.0 | 77.3 / 77.4 | +4.9% | 2.36 to 2.83 |
| novel prose, low acceptance | 48.5 / 48.5 | 46.7 / 47.1 | -3.4% | 1.14 to 1.08 |

So on the M4 it is a real tradeoff, not a free win: the fourth position pays on repetitive content and buys nothing on novel content while still costing a wider round. That is the shape to compare against.

What to run:

```
mlx-serve --model <qwen-pack> --serve --port 8080 --mtp --prefix-cache-entries 0   # cap 3
mlx-serve --model <qwen-pack> --serve --port 8080 --mtp --mtp-depth 4 --prefix-cache-entries 0
mlx-serve --model <qwen-pack> --serve --port 8080 --mtp --mtp-depth 5 --prefix-cache-entries 0
```

Two cells per arm, temperature 0, max_tokens ~220, boot-alternated, 2-3 reps, medians:

- echo: paste ~400 tokens and ask for it back twice.
- novel: ask for an original short story on a subject of your choice.

Read decode tok/s from the response's `timings.predicted_per_second` and `avg_per_round` off the `[spec-stats]` line.

**The discriminating cell is novel prose.** If depth 4+ wins on echo only and loses on novel, that is the same tradeoff we measured on M4 and the answer is that the default is fine and the controller should be demoting. If depth 4+ wins on novel prose too, your machine's round-cost curve is genuinely flatter than the M4 numbers the controller is using, which is a real per-silicon gap and worth a cost-table row of its own.

If you get the second result and want to go all the way, the refit method is in the doc comment above `MTP_EV_DEFAULT_COSTS` in `src/generate.zig`: same-session saturated sweep, `MLX_SERVE_MTP_ADAPTIVE=0` with forced depths, `--prefix-cache-entries 0`, counting only trace windows where `m_avg == N`. Send T(1), T(2), T(3), T(4), T(6), T(8) in milliseconds and I can fit the row.

## Note for anyone running --ane-prefill on any Mac

This build changed how ANE programs are packaged for everyone, not just dual: every covered layer is now a procedure inside a shared program instead of its own. Two things worth a glance in any ANE boot log, single or dual:

- The ready line reports `... in N banks`. N should be small (5 on a 27B), not one per layer.
- Zero `[ane] ... eval failed` lines. A failed eval silently recomputes that chunk on the GPU, so the answer stays correct and only the speed drops. During development a wrong dispatch index lost 200 of 210 evals and read as "banks are 23% slower" rather than as a bug. If you see those lines, the number in that run is meaningless, send the log.

## What to send back

- The median tables (arm x cell), with the engagement line per cell.
- For the dual-ANE test: `/props` ane object per arm, and the `macpow --dump | grep ANE0_` deltas for BOTH counters.
- For the MTP depth test: decode tok/s per workload per depth, with `[spec-stats]` avg_per_round beside each, and the machine.
- The acceptance cells (DFlash2 block 8 vs MTP forced depth 7): avg_per_round, per_draft_pct, and MTP's drafted/attempts ratio.
- Server logs from one boot per arm (`~/.mlx-serve/logs/` or wherever you redirected them).
- `GET /props` ane object from an ANE-on boot.
- Machine, macOS version, pack used, and which llmprobe version.

## M5 testers: two blind test fixes to confirm

Both M5-only GPU unit-test failures you reported are fixed on principled grounds without an M5 to verify on — please run `zig build test` once on your next pull and confirm:

- `deepseek_v4.zig` dec-chain parity: the composed-vs-kernel compare now acquits a mismatch only when the pre-quant value sits at an fp4 rounding midpoint between the two returned grid points (reduction-order near-tie, machine-dependent). Any other mismatch still fails.
- `minimax_h3.zig` sparseAttend closed form: tolerance widened to 2e-3 — the closed form is exact only in exact arithmetic, and your 1.9995117-vs-2 reading is fp16-accumulation scale on a different sdpa arm, not a wrong mask.

Also fixed from your reports: the live scripts now wait for the model to actually LOAD instead of proceeding at `/health` (no more local patches), `test_dflash.sh` [10] asserts the drafter's own declared block instead of a literal 16, and a missing MTP head now logs `[mtp] no head found: ...` at debug instead of silently falling back to mode=pld.
