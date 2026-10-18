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
- **A CUDA kernel of our own goes through `mlx_fast_cuda_kernel`**, not an MLX patch (`qmv_cuda.zig`, the GDN recurrence in `transformer.zig`):
  - The source is the kernel BODY. MLX writes the signature from the input and output names: typed pointers (`__nv_bfloat16*`, `uint32_t*`); helpers go in the header.
  - `<input>_shape` / `_strides` / `_ndim` are passed only when the body names them, so shapes cost nothing. Read them instead of compiling one variant per shape.
  - Template arguments become `constexpr` / `using` lines at the top of the body; each distinct set is one NVRTC compile.
  - The grid is in THREADS (`set_grid(32, SK, tiles)` with `set_thread_group(32, SK, 1)` gives `blockIdx.z = tile`).
  - Inline PTX (`mma.sync`) compiles. Gate tensor-core code on the device, not the build.
  - A new file's tests run only once `src/tests.zig` imports it.
- **The compute capability is `mlx.cudaComputeCapability(.major / .minor)`** (MLX's `device_info`): sm_120 is `12.0`, GB10 is `12.1`.

## Per-device defaults

- **A fast path is on by default only where it was MEASURED**, and every one has an env switch.
  - Mac-tuned paths are re-measured on CUDA: `prefillDqGemm` ran 13% slower there and stacked ~5 GB into the prefill peak (default off on CUDA), and a prefill chunk past 2048 bought no speed for GBs of peak (`autoPrefillChunkCap`).
  - A path tuned on one CUDA device loses on another: GB10 (128 GB unified memory) and a discrete RTX 5060 Ti flip on the same kernel.
- **Quantized matmul at 2..8 rows** (speculative verify, batched decode):
  - sm_120 runs `qmv_cuda.zig` (`MLX_SERVE_CUDA_QMV_TC=0|1`).
  - GB10 runs 2..6 rows through `patches/mlx-cuda-qmv-rows.patch` (`MLX_CUDA_QMV_MAX_ROWS`).
  - Other devices stay on MLX's `qmm_sm80`.
  - `qmv_cuda` declines unless the weight is 4-bit affine, group 32/64/128, bf16/f16, and `k % 128 == 0`.

## Memory

- **Where CUDA has no concurrent managed access (Windows, WSL), a CPU-stream buffer is pinned HOST memory that the GPU reads over PCIe** (decode ran 4.6x slower). Every weight loader takes `mlx.weightLoadStream()`, and `status.get{Available,Total}MemBytes` report VRAM there.
- **One process loads ONE CUDA runtime**: a second cuBLAS copy left llama.cpp silently on its CPU backend. Story: `docs/gotchas/engine-mlx.md`.

## Measuring

- **A kernel microbench must rotate its weights past L2.** The 5060 Ti's L2 held a 28 MB matrix and read it at 760 GB/s on a 448 GB/s card. Rotate copies that total more than ~5x L2.
- **The forward ladder is the in-situ meter**: `MLX_SERVE_DECODE_FWD_UBENCH=30 MLX_SERVE_DECODE_FWD_UBENCH_S=<rows> MLX_SERVE_DECODE_FWD_UBENCH_KV=2048 tests/fwd_ubench.sh <model>`. `MLX_CUDA_QMV_MAX_ROWS=1` plus `MLX_SERVE_CUDA_QMV_TC=0` is MLX's stock path.
- **Tok/s comes from llmprobe** (`tests/bench.sh --url`, `/bench`), with each arm proven by its own log: `[spec-stats] mode=` and `[qmv-cuda] tensor-core rows kernel engaged`.

## Tests

- **The Metal-kernel tests fail on CUDA by design** (~150 names). Judge a change by its failure LIST against main's on the same box, never by the count.
- **Two rollback tests flake inside the full suite** (`verify rollback permits in-place KV append…`, `offset rollback preserves dense affine growth…`): they share a buffer-identity check that the CUDA pool breaks depending on test order. Each passes alone.

## Gotchas

### A rows kernel on CUDA cores grew with every row (PR #791)

Defect: MLX's tiled `qmm_sm80` made a 2..8-row verify cost ~1.45 decode steps on an RTX 5060 Ti. A CUDA-core rows kernel (one weight read, one FMA per weight per row) won on GB10, but on the 5060 it cost ~10 ms more per row (91 ms at 8 rows against 24).
Cause: the arithmetic, not memory traffic. Staging x in shared memory did not help; both shapes grew linearly with the rows.
Fix: `qmv_cuda.zig` puts 16 weight rows x 8 x rows into one `mma.m16n8k16`, so 8 rows cost about one decode step. A lane owns 32 consecutive k of each 128-k block, so its packed words and its x slice load as 16-byte vectors. A and B take the same k permutation, which leaves the dot product unchanged. 16 warps split K over each 16-row tile.
Guard: `qmv_cuda: no worse than stock quantized_matmul against f32 truth` (rows 2..8, bf16 + f16, groups 32/64/128).

### Flash Next's 8192-row prefill failed in MLX's expert gather on a DGX Spark

Defect: NVIDIA's Flash Next FP4 pack chose an 8192-row chunk on a GB10 and every prompt past it failed with `cuGraphAddKernelNode ... invalid argument`.
Cause: MLX's `GatherQMM` runs `qmm_sm80` with one block per (token, expert) pair on grid z, capped at 65,535; 8192 rows x top-10 is 81,920. The 2048-row CUDA cap (`autoPrefillChunkCap`) bound only the load-time pin, and Flash Next picks its width per request and widens per chunk.
Fix: `mlx-cuda-gather-qmm-rhs.patch` (upstream #4554) runs sorted fp4/fp8 gathers one group per expert, ~10x faster at a 2048-row chunk; `generate.machine_prefill_chunk_cap` bounds every auto width in `effectivePrefillChunk`. Affine banks and expert widths not a multiple of 128 keep the old gather and its limit.
Guard: `gatherExpertMm: a sorted nvfp4 gather past 65,535 rows…`, `effectivePrefillChunk: the machine cap bounds every auto width…`, `chooseRequestPrefillChunk: the machine's chunk cap bounds…`.

## MLX patches and what could move into mlx-serve

| Patch | What it changes | Could be a kernel we pass in? |
|---|---|---|
| `mlx-cuda-qmv-rows.patch` | rows kernel inside `quantized_matmul` | Done on sm_120 (`qmv_cuda.zig`); GB10 keeps the patch until it is measured there |
| `mlx-cuda-qmm-read-weights-once.patch` | folds `[B, M, K]` batches into rows | Yes: `qmv_cuda` flattens leading dims; the stock path could reshape before the call |
| `mlx-cuda-sdpa-head-dim-256-512.patch` | fused attention at head dims 256 and 512 | Possible, as a custom attention kernel (as `msv_attn_p256` is on Metal) |
| `mlx-cuda-gather-qmm-rhs.patch` | upstream #4554: sorted fp4/fp8 expert gather, one group per expert | No: drop it once the pin includes #4554 |
| `mlx-cuda-lru-consecutive-misses.patch` | MLX's internal LRU cache | No: not a kernel |
| `mlx-cuda-windows-toolkit-dirs.patch` | DLL search folders on Windows | No: not a kernel |
