# CUDA

mlx-serve on NVIDIA GPUs: MLX's CUDA backend on Linux (x86_64 and aarch64) and Windows. This page holds the CUDA rules and the traps behind them. AGENTS.md points here.

## Building

- **Linux**: `./scripts/build-linux.sh` from a fresh checkout. It checks prerequisites, submodules and Zig, and detects the GPU arch (`CUDA_ARCH="75;86;120"` for RTX 20-50; aarch64 too: GH200, Thor, DGX Spark). Then it runs `build-mlx-linux.sh` (applies `patches/mlx-cuda-*.patch`), `build-ds4-linux.sh`, `fetch-llama.sh` and `zig build -Doptimize=ReleaseFast`.
- **Windows**: `powershell -File scripts\build-windows.ps1`. MLX + mlx-c are built by MSVC with CUDA 13.0 + cuDNN (`build-mlx-windows.ps1`), llama.cpp comes from its release, then `zig build -Dtarget=x86_64-windows-gnu` (also cross from Linux with `lib/mlx-win` staged). Users install CUDA 13 + cuDNN 9 (`CUDA_PATH`, `CUDNN_PATH`); no CUDA DLL ships. No ds4, LAN or WebP there.
- **MLX's kernel JIT needs the toolkit headers** through `CUDA_HOME`. `mlx.exportCudaHome` fills in `/opt/cuda` for the server; tests need it exported by hand.
- **`build-linux.sh` snaps clean submodules back to their pins** (`check-submodules.sh --fix`): a submodule checked out at another commit on purpose comes back at the pin.
- **The MLX stage is keyed on its inputs**: `lib/mlx/.version` hashes the patches and the build script, so a changed patch rebuilds MLX (about 5 minutes on 16 cores with the cmake directory kept in `.work-mlx-linux`).

## Kernels

- **Custom MSL kernels run on Metal only** (`mlx.metalKernelsAvailable`; `mlx.streamIsGpu` is false on CUDA, so every Metal helper declines). mlx-omarchy's Vulkan backend translates a subset of MSL and refuses the rest at EVAL, which fails the request. Vulkan and CUDA run MLX ops.
- **A CUDA kernel of our own goes through `mlx_fast_cuda_kernel`**, not an MLX patch (`sdk/blocks/qmm/qmv_cuda.zig`, the GDN recurrence in `engines/mlx/transformer.zig`):
  - The source is the kernel BODY. MLX writes the signature from the input and output names: typed pointers (`__nv_bfloat16*`, `uint32_t*`); helpers go in the header.
  - `<input>_shape` / `_strides` / `_ndim` are passed only when the body names them, so shapes cost nothing. Read them instead of compiling one variant per shape.
  - Template arguments become `constexpr` / `using` lines at the top of the body; each distinct set is one NVRTC compile.
  - The grid is in THREADS (`set_grid(32, SK, tiles)` with `set_thread_group(32, SK, 1)` gives `blockIdx.z = tile`).
  - Inline PTX (`mma.sync`) compiles. Gate tensor-core code on the device, not the build.
  - A new file's tests run only once `src/tests.zig` imports it.
- **Decode mat-vecs run `row_dot`** (`patches/mlx-cuda-row-dot-matvec.patch`): bf16 gemv, mxfp8/mxfp4/nvfp4 `quantized_matmul` and the fp expert gather read 16 bytes per load, split short-N rows over up to 256 threads, and share each weight load across up to 8 rows of x. On GB10 the fp modes take it through 8 rows (affine stays at the rows kernel's 6, `MLX_CUDA_QMV_MAX_ROWS` decides when set): with 8 batched slots the tiled `qmm_sm80` was 9% slower per token, and a row's bits are then the same alone or batched. A gather with fewer than 8 rows per expert takes it: rows on different experts share no weights, so `qmm_sm80`'s tile bought nothing at top-10 decode.
- **A sorted fp expert gather runs CTAs per (N tile, expert)** (`patches/mlx-cuda-sorted-gather-experts.patch`): a 64-row tile over sorted rows spanned 2-3 experts at Flash Next's ~40 rows each, one serial K-loop per expert run. A popular expert's M tiles split over up to 8 CTAs, the grid kept near 16k CTAs (more empty CTAs made the down GEMM slower).
- **qwen4 decode glue runs as `cuda_kernel`s**: the hyper-connection read (`sdk/blocks/hc/hc_cuda.zig`, `hc_decode2`'s two launches: norm + down, up + mix, with the deferred write folded in) and the fp-bank MoE (`sdk/blocks/moe/moe_cuda.zig`: gate+up+SwiGLU, then down with the routing-weighted sum). The MoE kernels include `row_dot`'s arithmetic (`device/row_dot_ops.cuh`, a JIT header the `row_dot` patch installs) and use MLX's own functors, so a decode row is bit-identical to the composed chain; the hc read sums in its own order, no further from an f32 truth than the chain. A multi-row read (MTP verify) loops its rows inside each block, so the hyper-connection weights load once per read, not once per row.
- **Prefill experts over fp banks run `sdk/blocks/moe/moe_prefill_cuda.zig`**: gate+up+SwiGLU in one launch (x rows read through the routing map), the down GEMM, and the weighted top-k sum. Outputs are bit-identical to the chain: the same `mma.m16n8k16` in `qmm_sm80`'s K order on exact bf16 weights, MLX's functors, MLX's sequential bf16 sum.
- **Below 4 rows per expert a sorted fp gather runs `row_dot_sorted_kernel`** (`patches/mlx-cuda-sorted-gather-matvec.patch`): a block takes 8 sorted rows and loads a weight row once per run on the same expert, with `row_dot`'s per-row arithmetic at the same threads per row, so the bits are the per-row mat-vec's.
- **On an integrated GPU prefill runs without CUDA graphs** (`mlx.setCudaGraphs`, `patches/mlx-cuda-graphs-switch.patch`); decode, and the decode ticks interleaved into a prefill, keep them.
- **The GDN short conv runs `depthwiseConvCuda` at prefill widths**: MLX's grouped conv1d is cuDNN's direct kernel there, ~7 ms per 2048-row chunk. Its own summation order, no further from an f64 truth than cuDNN's.
- **The GDN recurrence (`gdnRecurCuda`) gives a warp 16 Dv rows of one head at prefill lengths**: the rows share each step's k/q/g/beta loads, issued a step ahead; a row's arithmetic is the one-row kernel's, so the outputs are bit-identical to one row per warp (4.6 to 2.7 ms per 2048-token call on Flash Next). Against the op chain it sums in its own order: over 8 seeds its error to an f64 truth is 0.96x the chain's in `y` and 0.88x in the state.
- **The qwen4 hyper-connection read runs `sdk/blocks/hc/hc_prefill_cuda.zig` at prefill widths**: the deferred stream write, each stream's RMS norm and weight in one launch, the sigmoid-gated mean in another, in place of the compiled closures, `rms_norm_small` and `col_reduce_small`. Bit-identical to the compiled chain.
- **One QSA decode token, and each row of a solo verify block, attends through `plugins/qwen4_exp/qsa_decode_cuda.zig`** (the CUDA twin of `plugins/qwen4_exp/qsa_decode.zig`): the indexer's picks and the tail go straight to a split pass and a merge pass in f32, replacing the index arithmetic, gathers and the masked SDPA's matmul fallback (~11 ms a token at 5k context). Not MLX's summation order: 2.6x less RMS error than the gathered SDPA against an f64 truth. Verify rows (2..16) ride a third grid axis, each with its own picks and causal tail; before, they took the dense mask arm over the whole cache, and MTP at 32k context decoded slower than plain decode (20.5 vs 21.4 tok/s, now 34.3).
- **MTP drafts shortlist on the coarse head on CUDA too**: `draftTop32`'s Metal kernels decline there, and every draft read the full bf16 `lm_head` (1.27 GB on Flash Next); `draftTop32Partition` takes the same 32 ids through MLX's argpartition (proposal side only, so outputs never depend on its tie order).
- **QSA's mask arm hands MLX's SDPA an additive mask on CUDA** (`qsaMaskArm`): MLX broadcasts a bool mask over the heads, then fills two `[B, H, S, kv]` arrays to select 0 / -inf into a third (~7 ms per layer against a 5 ms attention at 7k tokens); the same values over one head reach cuDNN as a view.
- **The compute capability is `mlx.cudaComputeCapability(.major / .minor)`** (MLX's `device_info`): sm_120 is `12.0`, GB10 is `12.1`.

## Per-device defaults

- **A fast path is on by default only where it was MEASURED**; one whose verdict differs per device keeps an env switch to measure the next device.
  - Mac-tuned paths are re-measured on CUDA: `prefillDqGemm` ran 13% slower there and stacked ~5 GB into the prefill peak (default off on CUDA), and a prefill chunk past 2048 bought no speed for GBs of peak (`autoPrefillChunkCap`).
  - A path tuned on one CUDA device loses on another: GB10 (128 GB unified memory) and a discrete RTX 5060 Ti flip on the same kernel.
- **Quantized matmul at 2..8 rows** (speculative verify, batched decode):
  - sm_120 runs `sdk/blocks/qmm/qmv_cuda.zig` (`MLX_SERVE_CUDA_QMV_TC=0|1`).
  - GB10 runs 2..6 rows through `patches/mlx-cuda-qmv-rows.patch` (`MLX_CUDA_QMV_MAX_ROWS`).
  - Other devices stay on MLX's `qmm_sm80`.
  - `qmv_cuda` declines unless the weight is 4-bit affine, group 32/64/128, bf16/f16, and `k % 128 == 0`.

## Memory

- **Where CUDA has no concurrent managed access (Windows, WSL), a CPU-stream buffer is pinned HOST memory that the GPU reads over PCIe** (decode ran 4.6x slower). Every weight loader takes `mlx.weightLoadStream()`, and `status.get{Available,Total}MemBytes` report VRAM there.
- **On an integrated GPU (DGX Spark) CUDA managed memory reads at about two thirds of the GPU pool's bandwidth** (162 against 236 GB/s on GB10; `cudaMemAdvise` and prefetch do not help): weights load on the GPU stream there (`mlx.weightsInGpuPool`), and an array built from host bytes moves with `mlx.toGpuPool`. A host read of a weight (`mlx_array_data_*`) migrates it back to managed memory, and so does a CPU-stream op that reads a GPU load (MLX's cross-stream fence): load-time passes run on `mlx.weightLoadStream()`.
- **One process loads ONE CUDA runtime**: a second cuBLAS copy left llama.cpp silently on its CPU backend. Story: `docs/gotchas/engine-mlx.md`.

## Measuring

- **A kernel microbench must rotate its weights past L2.** The 5060 Ti's L2 held a 28 MB matrix and read it at 760 GB/s on a 448 GB/s card. Rotate copies that total more than ~5x L2.
- **The forward ladder is the in-situ meter**: `MLX_SERVE_DIAG=decode-fwd-ubench-s=<rows>,decode-fwd-ubench-kv=2048 tests/fwd_ubench.sh <model> 30`. `MLX_CUDA_QMV_MAX_ROWS=1` plus `MLX_SERVE_CUDA_QMV_TC=0` is MLX's stock path.
- **Tok/s comes from llmprobe** (`tests/bench.sh --url`, `/bench`), with each arm proven by its own log: `[spec-stats] mode=` and `[qmv-cuda] tensor-core rows kernel engaged`.

## Tests

- **The Metal-kernel tests fail on CUDA by design** (~150 names). Judge a change by its failure LIST against main's on the same box, never by the count.
- **Two rollback tests flake inside the full suite** (`verify rollback permits in-place KV append…`, `offset rollback preserves dense affine growth…`): they share a buffer-identity check that the CUDA pool breaks depending on test order. Each passes alone.

## Gotchas

### A rows kernel on CUDA cores grew with every row (PR #791)

Defect: MLX's tiled `qmm_sm80` made a 2..8-row verify cost ~1.45 decode steps on an RTX 5060 Ti. A CUDA-core rows kernel (one weight read, one FMA per weight per row) won on GB10, but on the 5060 it cost ~10 ms more per row (91 ms at 8 rows against 24).
Cause: the arithmetic, not memory traffic. Staging x in shared memory did not help; both shapes grew linearly with the rows.
Fix: `sdk/blocks/qmm/qmv_cuda.zig` puts 16 weight rows x 8 x rows into one `mma.m16n8k16`, so 8 rows cost about one decode step. A lane owns 32 consecutive k of each 128-k block, so its packed words and its x slice load as 16-byte vectors. A and B take the same k permutation, which leaves the dot product unchanged. 16 warps split K over each 16-row tile.
Guard: `qmv_cuda: no worse than stock quantized_matmul against f32 truth` (rows 2..8, bf16 + f16, groups 32/64/128).

### Flash Next's 8192-row prefill failed in MLX's expert gather on a DGX Spark

Defect: NVIDIA's Flash Next FP4 pack chose an 8192-row chunk on a GB10 and every prompt past it failed with `cuGraphAddKernelNode ... invalid argument`.
Cause: MLX's `GatherQMM` runs `qmm_sm80` with one block per (token, expert) pair on grid z, capped at 65,535; 8192 rows x top-10 is 81,920. The 2048-row CUDA cap (`autoPrefillChunkCap`) bound only the load-time pin, and Flash Next picks its width per request and widens per chunk.
Fix: `mlx-cuda-gather-qmm-rhs.patch` (upstream #4554) runs sorted fp4/fp8 gathers one group per expert, ~10x faster at a 2048-row chunk; `generate.machine_prefill_chunk_cap` bounds every auto width in `effectivePrefillChunk`. Affine banks and expert widths not a multiple of 128 keep the old gather and its limit.
Guard: `gatherExpertMm: a sorted nvfp4 gather past 65,535 rows…`, `effectivePrefillChunk: the machine cap bounds every auto width…`, `chooseRequestPrefillChunk: the machine's chunk cap bounds…`.

### `bf16 + bf16` in a JIT kernel is not MLX's add

Defect: the fused MoE down launch summed the ten weighted experts as `sum = sum + w` in `__nv_bfloat16`, like MLX's col reduce, and half the outputs came out one bf16 ulp off the composed chain.
Cause: under NVRTC that `+` did not resolve to the correctly rounded bf16 add the precompiled reduce uses.
Fix: write the rounding out, `T(float(sum) + float(w))`; MLX's `Multiply`/`Sigmoid` functors (from `device/*.cuh`) are fine.
The converse bit the hyper-connection read: an `mlx_compile`d closure is itself NVRTC code, so its bf16 `Add` is that same `+`. The fused write rounded in float, matched the op-by-op chain in its test, and moved the logits in the server; it now calls `Add{}` like the closure, and the test compares against the closure.
Guard: `moe_cuda: decode rows are bit-identical to the composed gather chain`, `hc prefill on CUDA: ... bit-identical to the compiled chain`.

### Expert weights streamed at half the DRAM bandwidth

Defect: Flash Next's prefill expert GEMMs ran at 13-16 TFLOPs on a DGX Spark, about 5 ms per projection per 2048-token chunk; the MMAs alone took a third of it.
Cause: `qmm_sm80` reads each weight row 32 bytes per K tile, so a warp load touches one sector of 16 rows; a probe streams the same bank at 91 GB/s in 32-byte pieces and 175 GB/s in 64-byte ones.
Fix: `moe_prefill_cuda` reads 64+ contiguous bytes of a row per warp load, four 128-wide K stages ahead in registers, and dequantizes each stage once through a code table. It must also decline wherever MLX leaves `qmm_sm80`: below 16 rows or 4 rows per expert the sorted gather runs a mat-vec that sums in another order, and short prefill chunks changed the logits until it did (`mlxTakesExpertPath`).
Guard: the `moe_prefill_cuda:` bit-identity tests against `gather_qmm` and the chain, and `below 4 rows per expert ... the kernels decline`.

### A 173-token prompt spent 870 of its 1055 ms in the expert mat-vec

Defect: Flash Next answered a 178-token prompt later than a 331-token one (1.1 s against 0.4 s to the first token on a GB10).
Cause: below 4 routed rows per expert (prompts under ~200 tokens at 512 experts, top-10) MLX's sorted gather runs the mat-vec, one block per routed row, so each expert's weights were read once per row: 1,720 rows x 0.8 MB per projection per layer.
Fix: `row_dot_sorted_kernel` loads a weight row once per run of sorted rows on the same expert; each output keeps `row_dot`'s per-row arithmetic at the threads per row the per-row launch picks, so it is bit-identical.
Guard: `below 4 rows per expert MLX's sorted mat-vec is bit-identical to its per-row one`.

### A fixed kernel kept running its racy build from MLX's PTX cache

Defect: on a DGX Spark the same greedy request drifted from run to run (logprobs ~0.1 apart, another token after 15-150), and one 64-token prefill forward repeated in a process moved logits by up to 5. The first output to differ was a `qmm_sm80` quantized matmul whose inputs were bit-identical.
Cause: `qmm_sm80`'s epilogue writes C into the shared memory of the A/B pipeline while its last prefetches can still land. `mlx-cuda-gather-qmm-rhs.patch` (#899) waits for them, but MLX caches compiled kernels by module name under `<tmp>/mlx/<version>/ptx`, so a box that compiled the kernel before the patch kept running the racy build.
Fix: `mlx.exportPtxCacheDir` points `MLX_PTX_CACHE_DIR` (unless set) at `<tmp>/mlx-serve-ptx/<id>`, the id hashing `lib/mlx/.version` (MLX commits, patch inputs, CUDA arch), so each patch set compiles its own kernels. The stamp reaches the binary through the build graph (module `mlx_stamp`): read at configure time, it was cached and an incremental build after an MLX update kept the old id. A hand edit to `lib/mlx-src` keeps the id: delete that directory.
Guard: `ptxCacheDir: each staged MLX build compiles into its own kernel cache`, `stampId: a rebuilt MLX gets a cache of its own`; `QWEN4_TEST_MODEL=<pack> zig build test -Dtest-filter="one lazy 64-token forward"` (4 of 4 repeats differ on the pre-#899 kernel).

### Start-up took 100 s: every custom kernel compiled again in each process

Defect: on a GB10 the Flash Next server took 100 s to report healthy against 64 s before (warmup 12 s, MTP verify warmup 31 s), and a request that brought a new verify width or batch size stalled for seconds.
Cause: MLX writes its own JIT kernels to the disk kernel cache but never a `cuda_kernel` (`use_disk_cache = false`), so NVRTC compiled each of ours again in every process, about a second apiece; the hc read also took its row count as a template, one more compile per width.
Fix: `mlx-cuda-custom-kernel-disk-cache.patch` writes them to the cache (the module name hashes the generated source; the directory is keyed by the MLX stamp), and the hc read takes its row count from the input's shape. A second start: warmup 1.3 s, verify warmup 1.6 s, healthy after 58 s.
Guard: `a custom CUDA kernel is written to MLX's kernel cache, so the next process loads it instead of compiling it`.

### Every weight on a DGX Spark read at two thirds of the bandwidth

Defect: Flash Next FP4 decoded at 14 tok/s on a GB10; its `lm_head` gemv took 7.8 ms in the model and 5.3 ms in a microbench of the same shape.
Cause: native Linux loads weights on the CPU stream, into CUDA managed memory, which on an integrated GPU stays on host page tables (`pageableMemoryAccessUsesHostPageTables`); the ModelOpt expert banks were built from host bytes the same way. The bench's weights came from GPU kernels, so from the pool.
Fix: on an integrated GPU weights load on the GPU stream and host-built banks go through `mlx.toGpuPool` (MLX's `device_info` reports `integrated`). With the `row_dot` patch and the one-token MoE gather, decode went from 14.4 to about 21 tok/s.
Guard: `weight loads go to the GPU's pool on CUDA where managed memory reads slow or never migrates`.

### Prefill kernels ran up to 2.3x slower inside CUDA graphs

Defect: on a GB10 Flash Next's prefill GEMMs took 4.2 ms in the server against 1.8 ms alone (same kernel, same shapes, same buffer kinds); with `MLX_USE_CUDA_GRAPHS=0` prefill ran 1.3x faster at 7k tokens and 1.6x at 14k, decode 11% slower.
Cause: not found. MLX on the GB10 commits a graph per 20 ops or 25 MB of inputs, so prefill launched a small graph per op; clocks, power, carveout, page size, buffer kind and graph shape all checked out, and no isolated graph reproduced it.
Fix: `mlx_cuda_set_graphs` (`patches/mlx-cuda-graphs-switch.patch`) turns graphs off at runtime, committing pending nodes first; `runPrefill` drops them on an integrated GPU (`mlx.prefillKeepsCudaGraphs`) and `interleaveDecodeTickCb` restores them for its ticks. Outputs are unchanged: the same kernels launch either way.
Guard: `setCudaGraphs: work evaluated with the graphs off, on and off again is bit-identical`.

### ModelOpt's FP8 linears read from managed memory

Defect: on a GB10, 2.8 GB of Flash Next's dense FP8 weights sat in CUDA managed memory although they loaded on the GPU stream.
Cause: `modelopt_pack.adapt` reinterpreted each e4m3 tensor as u32 words with `mlx_view` on a CPU stream; evaluating that view moved the GPU load's buffer to unified memory (`Fence::update`, MLX's cross-device fence).
Fix: `adapt` runs on `mlx.weightLoadStream()`. Decode +1.5%.
Guard: `ModelOpt: an FP8 linear loaded into the GPU's pool stays there` (reads the device address with a kernel: asking MLX for the pointer migrates the buffer).

## MLX patches and what could move into mlx-serve

| Patch | What it changes | Could be a kernel we pass in? |
|---|---|---|
| `mlx-cuda-qmv-rows.patch` | rows kernel inside `quantized_matmul` | Done on sm_120 (`sdk/blocks/qmm/qmv_cuda.zig`); GB10 keeps the patch until it is measured there |
| `mlx-cuda-qmm-read-weights-once.patch` | folds `[B, M, K]` batches into rows | Yes: `qmv_cuda` flattens leading dims; the stock path could reshape before the call |
| `mlx-cuda-sdpa-head-dim-256-512.patch` | fused attention at head dims 256 and 512 | Possible, as a custom attention kernel (as `msv_attn_p256` is on Metal) |
| `mlx-cuda-gather-qmm-rhs.patch` | upstream #4554: sorted fp4/fp8 expert gather, one group per expert | No: drop it once the pin includes #4554 |
| `mlx-cuda-row-dot-matvec.patch` | 16-byte mat-vecs for bf16/fp8/fp4 rows and few-row fp gathers; `device_info` reports `integrated` | Partly: a `cuda_kernel` could take the expert gather, but gemv and `quantized_matmul` dispatch inside MLX |
| `mlx-cuda-sorted-gather-experts.patch` | the sorted fp expert gather's grid: CTAs per (N tile, expert), heavy experts split | No: it reuses `qmm_sm80`'s mainloop |
| `mlx-cuda-graphs-switch.patch` | `mlx_cuda_set_graphs`: CUDA graphs on or off at runtime | No: the encoder's mode |
| `mlx-cuda-sorted-gather-matvec.patch` | below 4 rows per expert, the sorted fp gather's mat-vec loads each expert once per run of rows | Yes: a `cuda_kernel` could take it, if it reproduced `row_dot`'s arithmetic |
| `mlx-cuda-custom-kernel-disk-cache.patch` | `cuda_kernel` modules go to the disk kernel cache like MLX's own | No: MLX's JIT cache |
| `mlx-cuda-lru-consecutive-misses.patch` | MLX's internal LRU cache | No: not a kernel |
| `mlx-cuda-windows-toolkit-dirs.patch` | DLL search folders on Windows | No: not a kernel |
