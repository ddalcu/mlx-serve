# Plugins

Every model family and media backend is a plugin: a folder under `src/plugins/<name>/` compiled into the server, or a
dylib built from the plugin kit and installed at runtime. Two vendored modules (GGUF, sushi's EXL3) predate the plugin
table and plug in at seams below it; their sections close this page. mlx-stream's expert streamer is SDK code now
(`src/sdk/weights/streamed/`, `kit.streamed`).

## In-tree plugins

- **A folder per family.** `src/plugins/<name>/` holds `plugin.zig` (the table: `sdk.Plugin` with its `kind`
  (`model`, `media` or `quant`) and the `model_type`s it claims), `config.zig` (the claim and parse, MLX-free: the
  build refuses a `config.zig` whose import closure reaches `src/mlx/`, `src/engines/`, `src/host/` or `src/kernels/`)
  and an `AGENTS.md` (what it owns, its rules, its known issues). The table's contract is `src/abi/plugin.zig`;
  `Caps` declares what the engine may assume (batching, MoE, the exactness contracts in `docs/known-issues.md`).
- **Registration is a scan.** `build/plugins.zig` scans `src/plugins/*/plugin.zig` at configure time and writes the
  git-ignored `src/plugins/registry.zig`. A folder starting with `_` is a template (`_template`, `_template_media`,
  `_template_quant`): compiled and tested, never registered. Copy one to start a family.
- **Blocks come from the SDK.** `src/sdk/` holds the bases, blocks, quant formats, weight sources, spec heads and test
  harnesses a plugin builds on; a plugin never imports another plugin (shared code moves into the SDK).
- **Proof.** A plugin carries its own bar: a proof test against recorded goldens or a reference oracle, the released
  configs it claims (`src/fixtures/model-configs/`), and a registered contract for each of its Metal kernels
  (`contract.zig`, `src/sdk/testing/kernel_contract/REGISTERING.md`).
- **Units.** `-Dunits=true` builds each plugin as its own object (`zig build units-check -Dunits=true` for host
  globals; AGENTS.md "Building").

## Outside plugins (dylibs)

Step-by-step guides: [model](plugin-model.md), [media](plugin-media.md), [quant](plugin-quant.md). Every table, struct
and field: [plugin-abi.md](plugin-abi.md) (generated from `src/abi/`).

- **The kit.** `zig build kit` stages `zig-out/kit/mlx-serve-plugin-kit-<version>/` and its tarball: `build/kit/`,
  `src/kit.zig`'s import closure, mlx-c's headers, the MLX pin and the example plugins (`src/plugins/_example*`).
  `zig build test` builds every example from the staged kit, so the kit and the server cannot drift.
- **The kit** also carries `include/mlx_serve_plugin.h` (C/C++/Rust layouts, size-asserted; `src/kit_header.zig`) and
  `scripts/fetch-zig.sh` for the pinned Zig.
- **The CLI** (`src/host/plugins/cli.zig`): `mlx-serve plugin new <name> [--media|--quant|--engine]` copies the kit's
  example; `plugin check <dylib> [--model <pack>]` runs the conformance suite (`src/sdk/testing/conformance.zig`) on its
  fixture and, with `--model`, checks it claims and parses that pack (an engine: its probe takes it; that pack alone
  when the plugin has no fixture);
  `plugin fixture <dylib> <dir>` writes its tiny pack;
  `plugin install` shows the file's hash and signer (`codesign`), asks before a quarantined file (`--yes` confirms),
  and copies it into `~/.mlx-serve/plugins/` with a receipt (name, kind, version, sha256, signer, model types, an
  engine's model folders, options).
- **Loading** (`src/host/plugins/loader.zig`): at startup, when `plugins.json` says `"enabled": true`. Each dylib must
  match its receipt's hash, is negotiated (`src/abi/dylib.zig`: a plugin built against another MLX is refused) and
  joins the claims (a same-name plugin replaces a built-in; higher rank wins; a built-in wins a tie). A refusal names
  the file and the reason, and the server serves on without it. A dylib is never unloaded; `--no-plugins` loads none.
  `/props` `plugins` lists each one loaded or refused (`rebuild: true` = built for another MLX or ABI, which a rebuild
  with this release's kit fixes; the app offers Rebuild), and a failed model load names those plugins.
  `/v1/models` marks a pack an outside plugin serves with `"plugin": "<name>"`. The app's Models ▸ Plugins installs a
  catalog plugin after one approval of the fetched commit (build, install, switch on in `plugins.json`) and offers a
  server restart to load it.
  macOS only, never in the Mac App Store build.
- **Options** (`Manifest.options`, `sdk.Plugin.options`): the settings a plugin reads from its environment (`getenv`,
  or the Host table's `lever`), each a name, a kind (`bool` = `0`/`1`, `number`, `text`), a default and one line of
  help. A name is the plugin's name upper-cased, `_`, then [A-Z0-9_] (`turboexl3_ds41` → `TURBOEXL3_DS41_FRAC`),
  never `MLX_`, `MTL_`, `METAL_` or `DYLD_`; negotiation refuses any other, and the server sets no name `env.zig`
  registers (`HF_TOKEN` for a plugin named `hf`). Two plugins whose names nest (`foo`, `foo_bar`) can
  declare one name: the first loaded sets it, the other is told by name. The app draws them on the
  plugin's page and saves values in `plugins.json` (`"options": {"<plugin>": {"<NAME>": "<value>"}}`); at startup the
  server sets each declared one the file saves, over a value the shell set, before it opens the plugin (a plugin it
  then refuses gets the shell's values back), and `/props` reports what it set, the values to a loopback client only.
  A change applies at the next start (the app offers the restart). One value per plugin, not per model.

## Vendored modules: `mlx_host`, the one import

`mlx_host` is the host's root module (`src/main.zig`, `src/tests.zig`, `src/ios_lib.zig`); `build.zig` passes it to
the vendored module (`sushi_exl3`). It reads only the declarations those roots re-export:

| Declaration | What | Used by |
|---|---|---|
| `mlx_host.mlx` | the mlx-c FFI (`src/mlx/ffi.zig`), with our error latch | the in-tree EXL3 module (`sushi_exl3`) |
| `mlx_host.log` | leveled logging and the log file | `sushi_exl3` |
| `mlx_host.io_util` | timers and the I/O helpers | `sushi_exl3` |
| `mlx_host.env` | the registered environment (`src/env.zig`) | `sushi_exl3` |

- There is one instance of each per binary. Plugin and host share MLX handles, the error latch and the log file.
- **The host is the hub.** When two plugins need the same code, it lives in the host (or in a module the host
  builds), and the roots re-export it. A plugin never imports another plugin, so pins never chain and a fix lands
  once.
- The three roots carry the same list, and so does `src/plugin_host.zig`, the root a plugin's own test build imports
  as `mlx_host`. A missing declaration is a compile error in the graph that lacks it.
- **MLX through mlx-c, with one exception.** A plugin calls MLX through `mlx_host.mlx` (missing mlx-c declarations
  are added to `src/mlx/ffi.zig`). A plugin that needs an MLX internal (a GPU wait on an event, an allocation without a
  fill) reaches it through the host's `HostGpu` table, whose C++ shims our build compiles against the staged MLX
  headers (`src/sdk/weights/streamed/csrc/`). An MLX bump can break a shim; the host fixes it and the pin moves.

## The Neural Engine: `Host.ane`

A plugin reaches the ANE through the host's one runtime (`lib/ane`), never a copy of it:
`kit.host.Host.of(ctx.host).ane()` is the `HostAne` table, or null (no ANE, Linux, or a host from before the field),
and then the plugin serves on the GPU. The host owns the mechanism; the plugin owns the policy.

| The host (`HostAne`) | The plugin |
|---|---|
| compiling any MIL program (`program_create`; `kit.ane.mil` builds the text and weight blob) with the compile cache, surfaces, requests, evaluation | which ops and layers go to the ANE, the rows each die takes, when it starts and joins an evaluation, its GPU fallback |
| typed banks: SwiGLU (with DeepSeek V4's `swiglu_limit`) and GatedDeltaNet input projections over int8 per-row weights | quantizing its weights to int8 rows, the layout of its activations |
| the process's count of resident compiled models (`resident`; about 121 allowed) and its dies, shared with the host's own offloads | staying under that limit |

An evaluation runs on the calling thread. Different programs, banks and requests evaluate concurrently, one caller at
a time each.

## Pick the shallowest seam

The deeper the seam, the less of our stack the model gets for free. Use the shallowest one that expresses the work.

| Seam | The plugin supplies | The host keeps | Example |
|---|---|---|---|
| Format | detection, the JSON a model directory would hold, the tensor map, kernels for its encodings | the arch, KV cache, batching, spec decode, prefix cache | GGUF (`src/sdk/quant/gguf/`) |
| Weight encoding | config parse and load checks, the matmul over its weights | the arch and everything above it | EXL3 (`src/sdk/quant/exl3/`, from sushi) |
| Arch | a model's forward, its kernels and decode state, its weight loading and memory bill | HTTP, the template, tool calls, sampling, stops, scheduling, admission | every model plugin's table (`abi/plugin.zig`) |
| Media towers (`abi.MediaHooks`) | `prepare` (a decoded item, RGB8 frames or 16 kHz samples, as its processor lays it out), the tower, the item's placeholder run (text through the tokenizer callback) | decoding, the encoder cache, placeholder expansion, the splice, prefix-cache media keys | LFM3-VL (`src/plugins/lfm3/media.zig`) |
| Speech output (`abi.SpeechHooks`, `Parsed.speech`) | a frame of codes from the trunk's last hidden row, a frame's input rows (`Forward.embeds`), a streaming codec | the text/speech/interleaved slot machine (`engines/mlx/generate/speech.zig`), sampling windows, `.audio` events, OpenAI `modalities`, `/v1/audio/speech` + transcriptions | LFM3-Audio, Omni (`src/plugins/lfm3/speech.zig`) |

Deepest of all, an **engine plugin** (`--engine`, the kit's `_example_engine`) is a whole engine of its own behind
`abi/engine.zig`, the table ds4 (raw Metal), llama.cpp and the Neural Engine sit behind: it owns the weights, KV,
batching, sampling and its GPU runtime, and never calls MLX (its MLX pin is not read). The host keeps HTTP and every API
surface, the template, tool calls, reasoning and stops (unless the engine declares `owns_tokenizer` / `owns_template`),
queueing and admission (its `load_bytes` is the bill). It serves only the model folders its manifest names
(`Manifest.models`, by folder name; two that name one folder are ambiguous and neither serves it), over every
built-in, and runs its own thread (`Threading.own_thread`, no `step`),
or is refused by name; a model it serves keeps a folder the host can read (a config.json and a tokenizer, or a GGUF).
`plugin check` opens it on its fixture and serves one request; the kit's example passes the whole engine contract
(`abi/contract_test.zig`) in `zig build test`. An engine that wants MLX is a model plugin instead.

A seam is added with its first consumer and generalized with its second.

## Example: GGUF (format seam)

`src/sdk/quant/gguf/` (formerly the mlx-serve-gguf repo) serves GGUF files as they are on the regular MLX path. The
file stands in for a model directory.

- **Module.** `root.zig` exports the reader (`gguf`), the metadata-to-HF translation (`meta`), `weights` and
  `kernels`, over `src/mlx/ffi.zig`; `format.zig` is its quant table.
- **Glue.** `src/sdk/quant/gguf/claim.zig` holds four entry points:
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
code came in: Sushi 1.3.0's `src/exl3` in `src/sdk/quant/exl3/sushi/`, module `sushi_exl3`, reaching `mlx`, `log`
and `io_util` through `mlx_host`; `src/sdk/quant/exl3/format.zig` is its quant table. The rest of the fork stayed out.

- **Parse.** `parseExpertQuant(quantization_config.expert_quant)` gives `ModelConfig.exl3` (rate, codebook, window),
  then `admitTopK`.
- **Load.** `checkExl3Bank` asks `trellisAdmitted` for each bank's geometry and refuses by name.
- **Forward.** `exl3.routed(s, x, bank, inds, scores, spec, opts)` replaces the routed-expert matmul in qwen4_exp's
  MoE layer.
- **Free from the host.** The whole qwen4_exp trunk, MTP and batching.

A streamed EXL3 bank is not in the tree: a quant plugin serves one (`QuantTable.open_bank`, below).

## Example: streamed experts (weight seam)

A streamed load keeps its routed experts on the SSD and reads them into per-layer slot rows; the plugin's own trunk
runs everything else. The stream (`sdk/weights/streamed/expert.zig`: slot pools, the residency policy, the read pool
in `csrc/`) serves a source's fetches behind `Moe.Weights.streamed` (a `Routed` with the source's math, `settle`,
`phase` and read-ahead), and the plugin's caps say `experts_stream`:

- `stream_experts` (a per-model setting, `true` or the slot rows a layer): a standard MLX pack's affine experts
  streamed as downloaded (`streamed/mlx_pack.zig`), for MoE packs that do not fit in memory. The load bills the
  residents plus those rows. A plugin whose MoE runs on the shared block streams with two calls:
  `Streamed.forLoad(gpa, ctx, n_experts, s)` at load (null when the setting is off) and `moe.bindStreamed(&w, &x,
  "<prefix>.layers.<i>.mlp")` per trunk layer, and declares `Caps.experts_stream` (without it the setting is refused
  by name). The conformance suite then checks the streamed logits against the in-memory ones. In-tree: qwen3_moe,
  qwen3_5/qwen3_next, lfm2_moe, laguna, kolibri1, flint, hy_v3, glm5_next, bailing_hybrid, and deepseek_v41 (its own
  trunk).
- `deepseek_v41`: V4.1's EXL3 repack, through the bank its outside quant plugin opens (`QuantTable.open_bank`, an
  `abi.Bank` wrapped as the trunk's `Routed` by `sdk/quant/outside.zig`). `turboexl3_ds41` streams its EXL3 records
  over the kit's stream (`kit.streamed.expert.stream.StreamOf` over its own bank; the kit's `build.zig`
  `streamSources` compiles the read pool into the plugin).

A pack whose config declares `expert_streaming` (MiMo's and GLM-5.3-Flash's `experts.bin` banks) is refused by name:
its bank format streams through a plugin. The stream is macOS only; Linux and iOS refuse a streamed load by name.

## Rules for plugin code

- Our `AGENTS.md` rules apply to every plugin, in-tree and vendored; an outside plugin's tests are its own, and
  `plugin check` is the bar it must pass here.
- **Refuse by name.** An unsupported config, layout or platform is a named error plus a `log.err` line. It never
  falls back to another arch.
- **No new CLI flags or env levers for one plugin.** Use what the host already has: the GPU ceiling
  (`MLX_SERVE_GPU_CEILING_MB` for diagnostics), the OS reserve, `--wired-margin-gib`, the per-model `ctx_size`. A diagnostic is a `MLX_SERVE_DIAG` token (absent or `0` = off).
- **Bill what you allocate.** Memory the plugin allocates is billed by the plugin (`loadBytes`, `promptBytes`), and
  each term is an upper bound it has measured.
- **No duplicated host code.** If the plugin needs something the host has, the host exposes it through `mlx_host`.
  If the host's version falls short, fix it in the host. Code only the plugin uses stays in the plugin.
- **Weight layouts start from mlx-lm.** Read mlx-lm's model file and the MLX packs on Hugging Face first. A new layout
  needs a stated reason, and a pack other engines can read beats one only the plugin reads.
- **No host types in a plugin's API, and no plugin types on host-wide structs** beyond one pointer.
