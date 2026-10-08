# Plugins

How engine work from another repository ships inside mlx-serve: compiled into our build, from its repository, at a
pinned commit, with as little glue on our side as possible. This replaces the proposal on
`claude/mlx-serve-modular-arch-l2h588` (8781a6c). The goal is the same and the mechanism is much smaller: no SDK
layer, no registry, no negotiation. Plugin and host already share one MLX and one compiler.

## What a plugin is

- A Zig module in its own repository, pinned as a git submodule under `lib/<name>`. `-D<name>-dir=/abs/path` builds
  against a checkout instead.
- It imports `std`, `builtin` and one host module, `mlx_host`. It imports nothing else from the host and never
  another plugin.
- It owns only what differs from the host: a file format, a weight encoding and its kernels, or a model's forward.
  HTTP, chat templates, tool calls, sampling, stops, the scheduler, memory admission and the app stay the host's.
- The host calls it from one glue file (`src/arch/<name>.zig`) or from one call site at a seam. The plugin never
  calls back into host internals.
- Glue budget on our side: one submodule line, one `build.zig` function, the glue, one `NOTICE` paragraph.

## `mlx_host`: the one import, and the hub between plugins

`mlx_host` is the host's root module (`src/main.zig`, `src/tests.zig`, `src/ios_lib.zig`); `build.zig` passes it to
every plugin. A plugin reads only the declarations those roots re-export for it:

| Declaration | What | Used by |
|---|---|---|
| `mlx_host.mlx` | the mlx-c FFI (`src/mlx.zig`), with our error latch | mlx-serve-gguf, sushi, mlx-stream |
| `mlx_host.log` | leveled logging and the log file | sushi, mlx-stream |
| `mlx_host.io_util` | timers and the I/O helpers | sushi |
| `mlx_host.mtp_acceptance` | the draft acceptance modes the host serves | mlx-stream |

- There is one instance of each per binary. Plugin and host share MLX handles, the error latch and the log file.
- **The host is the hub.** When two plugins need the same code, it lives in the host (or in a module the host
  builds), and the roots re-export it. A plugin never imports another plugin, so pins never chain and a fix lands
  once.
- The three roots carry the same list, and so does `src/plugin_host.zig`, the root a plugin's own test build imports
  as `mlx_host`. A missing declaration is a compile error in the graph that lacks it.
- **MLX through mlx-c, with one exception.** A plugin calls MLX through `mlx_host.mlx` (missing mlx-c declarations
  are added to `src/mlx.zig`). A plugin that needs an MLX internal (a GPU wait on an event, an allocation without a
  fill) keeps its own C++ shim, compiled by our build against the staged MLX headers (mlx-stream's `csrc/`). An MLX
  bump can break such a shim; the plugin's repository fixes it and the pin moves.

## Pick the shallowest seam

The deeper the seam, the less of our stack the model gets for free. Use the shallowest one that expresses the work.

| Seam | The plugin supplies | The host keeps | Example |
|---|---|---|---|
| Format | detection, the JSON a model directory would hold, the tensor map, kernels for its encodings | the arch, KV cache, batching, spec decode, prefix cache | mlx-serve-gguf |
| Weight encoding | config parse and load checks, the matmul over its weights | the arch and everything above it | EXL3 (from sushi) |
| Arch | a model's forward, its kernels and decode state, its weight loading and memory bill | HTTP, the template, tool calls, sampling, stops, scheduling, admission | mlx-stream (DeepSeek-V4.1's EXL3 repack) |

Not a seam: **opaque engines.** ds4 and llama.cpp keep their bridges (`src/arch/ds4.zig`, `src/arch/llama.zig`).

A seam is added with its first consumer and generalized with its second.

## Wiring a plugin

1. **Pin it.** `git submodule add <url> lib/<name>`. Anything compiled into a release is pinned from a repository we
   control (a fork, as with `ddalcu/sushi`). The author keeps developing upstream, and a bump is a PR that moves the
   pin.
2. **Build it.** Add one function to `build.zig`, shaped like `addGgufModule`, and call it once per graph (server,
   tests, Linux, iOS):

   ```zig
   fn addNameModule(b: *std.Build, host: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
       const m = b.createModule(.{
           .root_source_file = nameRoot(b), // lib/<name>/src/root.zig, or -D<name>-dir
           .target = target,
           .optimize = optimize,
           .link_libc = true,
           .imports = &.{.{ .name = "mlx_host", .module = host }},
       });
       host.addImport("<name>", m);
   }
   ```

   The plugin's C sources go into its module in the same function. Plain C (a pthread read pool) and Objective-C over
   public Metal API are fine.
3. **Glue it.** `src/arch/<name>.zig` is the only host file that imports the plugin, unless the seam is a single call
   (EXL3's MoE call sits in `transformer.zig`). The glue translates host types (`ModelConfig`, `Transformer`,
   `ForwardCtx`) into the plugin's plain arguments: slices, integers, `mlx_array`.
4. **Platforms.** Pure Zig over `mlx_host.mlx` compiles on every graph. macOS-only code (Objective-C, Metal events)
   gets a stub glue file chosen by a build option (`build_options.mlx_stream`, as `macos_engines` picks
   `src/arch/ds4_stub.zig`). The stub refuses the model by name.
5. **Tests.** The plugin's own suite runs in its own repository (see below). Its root also runs as a test artifact in
   our `zig build test` (one `b.addTest` per plugin), so a host change that breaks it fails here. The glue's tests live
   in the glue file.
6. **NOTICE.** One paragraph naming the repository, its license and what it ports. The license must allow
   redistribution in our MIT / Apache-2.0 release.

## Developing a plugin on its own

The plugin's `build.zig` builds against an mlx-serve checkout (`-Dmlx-serve=../mlx-serve`). A three-line
`standalone/mlx_host.zig` re-exports the host's `src/mlx.zig` (plus `log.zig` and `io_util.zig` if it uses them).
mlx-serve-gguf works exactly this way:

- `zig build test-core`: the format reader and reference dequant, with no MLX and no GPU.
- `zig build test`: the kernels on the GPU, through the host's MLX and its staged `lib/mlx`.

To serve, build mlx-serve with `-D<name>-dir=$PWD`. No slim host and no fork are needed. mlx-stream's `build.zig`
drives the host's own build instead (`zig build test` runs the host's `mlx-stream-test` step, `zig build conformance`
its `mlx-stream-conformance` step).

## Example: mlx-serve-gguf (format seam)

`lib/mlx-serve-gguf` serves GGUF files as they are on the regular MLX path. The file stands in for a model directory.

- **Module.** `src/root.zig` exports the reader (`gguf`), the metadata-to-HF translation (`meta`), `weights` and
  `kernels`. It reads `mlx_host.mlx` only.
- **Glue.** `src/arch/mlx_gguf.zig` holds four entry points:
  - `servablePath` takes a `.gguf` only when its arch, tokenizer, tensor types and tensor names are all known.
    Anything else stays with llama.cpp or ds4.
  - `sidecar(which)` returns the `config.json` / `tokenizer.json` / `tokenizer_config.json` /
    `generation_config.json` a model directory would hold. `parseConfig`, `loadTokenizer` and `loadChatConfig` read
    it.
  - `loadWeights` returns the tensor map and `weightBytes` the preflight's disk bytes.
- **Seam in the forward.** A quantized tensor stays raw ggml blocks plus a `.scales` sentinel under `QuantMode.gguf`.
  `kernels.infoOf` at the top of `Transformer.qmatmul`, `rawEmbedding` and `gatherExpertMm` routes it to the module's
  kernels.
- **Free from the host.** The native `qwen3_5` / `qwen3_5_moe` / `gemma4` forward, batching, prefix cache and PLD.
- Detail: `docs/reference.md` "MLX GGUF engine".

## Example: EXL3 experts from sushi (weight-encoding seam)

[sushi](https://github.com/beamivalice/sushi) is a fork of mlx-serve that added EXL3 routed experts. Only the EXL3
code came in, as a plugin: the submodule `lib/sushi` (our fork's `exl3-module` branch), module `sushi_exl3`, reaching
`mlx`, `log` and `io_util` through `mlx_host`. The rest of the fork stayed out.

- **Parse.** `parseExpertQuant(quantization_config.expert_quant)` gives `ModelConfig.exl3` (rate, codebook, window),
  then `admitTopK`.
- **Load.** `checkExl3Bank` asks `trellisAdmitted` for each bank's geometry and refuses by name.
- **Forward.** `moe(s, x, bank, inds, scores, dec, verify_rows)` replaces the routed-expert matmul in qwen4_exp's MoE
  layer.
- **Free from the host.** The whole qwen4_exp trunk, MTP and batching.

mlx-stream reads the same format with its own decoder and kernels, pinned to DeepSeek-V4.1's shapes. The two do not
share code: each serves a different arch, and their kernels are tuned to it.

## Example: mlx-stream (arch seam)

[mlx-stream](https://github.com/davidtai/mlx-stream) (pinned from its `main`) serves
DeepSeek-V4.1's EXL3 streaming repack (`experts.bin` beside the trunk). It owns the model: the arch, its Metal
kernels, its EXL3, the SSD expert streamer (lookahead reads, event gates), DSpark with typical acceptance, the prefix
resume of one conversation, its weight loading past the page cache and its memory bill. Its `sdk` module (`sdk/`)
holds the arch contract's types. MLX-format V4.1 packs stay in-tree (`src/deepseek_v41.zig`).

- **Dispatch.** `parseConfig` marks a `deepseek_v41` directory with `experts.bin` (`ModelConfig.dsv41_stream`);
  `loadModelWeights` loads nothing for it.
- **Glue** (`src/arch/mlx_stream.zig`):
  - `loadBytes`: the preflight's bill, from the GPU ceiling, the wired margin and the memory in use before the load
    (sampled first, as the plugin's admission fills rows on top of it);
  - `open` / `close`: the plugin loads its weights and builds the module;
  - `begin`: a request's start, what the module's kept state resumes (never the whole prompt) and the shape its
    prompt pass bills; a prompt past the billed context is `PrefillDoesNotFit` (a 400);
  - `forward`: the prompt pass, then the decode handover once and the steps;
  - `arm` / `round`: the DSpark lane, on requests with nothing shaping the logits. `arm` takes the request's
    `SamplingParams` (temperature, top_p, top_k, min_p, a seed even when the request sent none) and every `round`
    passes them: greedy requests at the lane's typical acceptance, sampled ones by the plugin's exact speculative
    sampling;
  - `contextLength`: the billed prompts plus the generation past them, which the host advertises.
- **Host arms.** `Transformer.dsv41_ext` (init, forward, deinit; the generator hands it the whole prompt in one
  forward, `prefillsWholePrompt`; no host warm-up), the scheduler's bill and `begin`, the generator's lane, the
  server's context and prompt bill.
- **Build.** The `lib/mlx-stream` submodule, or `-Dmlx-stream-dir=/abs/path`: two modules (`sdk`, `mlx_stream`), its
  C sources against the staged MLX, its suite in `zig build test`, its conformance suite (a CPU-lane binary of its
  own, which checks the plugin against the linked MLX) as `zig build mlx-stream-conformance`. macOS only; the
  Linux and iOS graphs build `src/arch/mlx_stream_stub.zig`, which refuses the repack by name.

## Rules for plugin code

- Our `AGENTS.md` rules apply to the glue in our tree. A plugin's internals follow its own repository's rules, and
  its tests must pass in our `zig build test`.
- **Refuse by name.** An unsupported config, layout or platform is a named error plus a `log.err` line. It never
  falls back to another arch.
- **No new CLI flags or env levers for one plugin.** Use what the host already has: the GPU ceiling
  (`MLX_SERVE_GPU_CEILING_MB` for diagnostics), the OS reserve, `--wired-margin-gib`, the per-model `ctx_size`. A diagnostic env follows the host's convention (absent or `0` = off, as `diagEnvOn`).
- **Bill what you allocate.** Memory the plugin allocates is billed by the plugin (`loadBytes`, `promptBytes`), and
  each term is an upper bound it has measured.
- **No duplicated host code.** If the plugin needs something the host has, the host exposes it through `mlx_host`.
  If the host's version falls short, fix it in the host. Code only the plugin uses stays in the plugin.
- **Weight layouts start from mlx-lm.** Read mlx-lm's model file and the MLX packs on Hugging Face first. A new layout
  needs a stated reason, and a pack other engines can read beats one only the plugin reads.
- **No host types in a plugin's API, and no plugin types on host-wide structs** beyond one pointer.

## Left out of the earlier proposal, on purpose

- **Compile-time API and MLX negotiation.** The compiler already checks every call, and the submodule pin is the
  version. There is one MLX per binary because there is one `mlx_host.mlx`.
- **`claims(peek) ?Priority`, a registry and tie-breaks in `model-settings.json`.** One plugin serves each
  `model_type` or file format, so the glue's dispatch arm is the registration.
- **`source` and `engine` kinds, a slim host, conformance lanes.** None has a consumer. Each lands with its first
  real one, the way the arch seam landed with mlx-stream.

## Status

- On main: the format seam (mlx-serve-gguf); the weight-encoding seam (EXL3 from sushi).
- With DeepSeek-V4.1: the arch seam (mlx-stream, `lib/mlx-stream` at `davidtai/mlx-stream` `main`), its
  suite in `zig build test`. mlx-serve-gguf's suite runs only in its own repository today.
