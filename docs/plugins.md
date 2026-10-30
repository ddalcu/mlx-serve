# Plugins

Every model family and media backend is a plugin: a folder under `src/plugins/<name>/` compiled into the server, or a
dylib built from the plugin kit and installed at runtime. Three vendored modules (GGUF, sushi's EXL3, mlx-stream)
predate the plugin table and plug in at seams below it; their sections close this page.

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
- **The CLI** (`src/host/plugins/cli.zig`): `mlx-serve plugin new <name> [--media|--quant]` copies the kit's example;
  `plugin check <dylib> [--model <pack>]` runs the conformance suite (`src/sdk/testing/conformance.zig`) on its fixture
  and, with `--model`, checks it claims and parses that pack (that pack alone when the plugin has no fixture);
  `plugin fixture <dylib> <dir>` writes its tiny pack;
  `plugin install` shows the file's hash and signer (`codesign`), asks before a quarantined file (`--yes` confirms),
  and copies it into `~/.mlx-serve/plugins/` with a receipt (name, kind, version, sha256, signer, model types).
- **Loading** (`src/host/plugins/loader.zig`): at startup, when `plugins.json` says `"enabled": true`. Each dylib must
  match its receipt's hash, is negotiated (`src/abi/dylib.zig`: a plugin built against another MLX is refused) and
  joins the claims (a same-name plugin replaces a built-in; higher rank wins; a built-in wins a tie). A refusal names
  the file and the reason, and the server serves on without it. A dylib is never unloaded; `--no-plugins` loads none.
  `/v1/models` marks a pack an outside plugin serves with `"plugin": "<name>"`; the app's Settings ▸ Server lists the
  receipts with a switch each (writes `plugins.json`). macOS only, never in the Mac App Store build.

## Vendored modules: `mlx_host`, the one import

`mlx_host` is the host's root module (`src/main.zig`, `src/tests.zig`, `src/ios_lib.zig`); `build.zig` passes it to
the two vendored modules (`sushi_exl3`, `mlx_stream`). They read only the declarations those roots re-export:

| Declaration | What | Used by |
|---|---|---|
| `mlx_host.mlx` | the mlx-c FFI (`src/mlx/ffi.zig`), with our error latch | the in-tree EXL3 module (`sushi_exl3`), mlx-stream |
| `mlx_host.log` | leveled logging and the log file | `sushi_exl3`, mlx-stream |
| `mlx_host.io_util` | timers and the I/O helpers | `sushi_exl3` |
| `mlx_host.mtp_acceptance` | the draft acceptance modes the host serves | mlx-stream |

- There is one instance of each per binary. Plugin and host share MLX handles, the error latch and the log file.
- **The host is the hub.** When two plugins need the same code, it lives in the host (or in a module the host
  builds), and the roots re-export it. A plugin never imports another plugin, so pins never chain and a fix lands
  once.
- The three roots carry the same list, and so does `src/plugin_host.zig`, the root a plugin's own test build imports
  as `mlx_host`. A missing declaration is a compile error in the graph that lacks it.
- **MLX through mlx-c, with one exception.** A plugin calls MLX through `mlx_host.mlx` (missing mlx-c declarations
  are added to `src/mlx/ffi.zig`). A plugin that needs an MLX internal (a GPU wait on an event, an allocation without a
  fill) keeps its own C++ shim, compiled by our build against the staged MLX headers (mlx-stream's `csrc/`). An MLX
  bump can break such a shim; the plugin's repository fixes it and the pin moves.

## Pick the shallowest seam

The deeper the seam, the less of our stack the model gets for free. Use the shallowest one that expresses the work.

| Seam | The plugin supplies | The host keeps | Example |
|---|---|---|---|
| Format | detection, the JSON a model directory would hold, the tensor map, kernels for its encodings | the arch, KV cache, batching, spec decode, prefix cache | GGUF (`src/sdk/quant/gguf/`) |
| Weight encoding | config parse and load checks, the matmul over its weights | the arch and everything above it | EXL3 (`src/sdk/quant/exl3/`, from sushi) |
| Arch | a model's forward, its kernels and decode state, its weight loading and memory bill | HTTP, the template, tool calls, sampling, stops, scheduling, admission | mlx-stream (DeepSeek-V4.1's EXL3 repack, MiMo-V2.6-Flash, GLM-5.3) |
| Media towers (`abi.MediaHooks`) | `prepare` (a decoded item, RGB8 frames or 16 kHz samples, as its processor lays it out), the tower, the item's placeholder run (text through the tokenizer callback) | decoding, the encoder cache, placeholder expansion, the splice, prefix-cache media keys | LFM3-VL (`src/plugins/lfm3/media.zig`) |
| Speech output (`abi.SpeechHooks`, `Parsed.speech`) | a frame of codes from the trunk's last hidden row, a frame's input rows (`Forward.embeds`), a streaming codec | the text/speech/interleaved slot machine (`engines/mlx/generate/speech.zig`), sampling windows, `.audio` events, OpenAI `modalities`, `/v1/audio/speech` + transcriptions | LFM3-Audio, Omni (`src/plugins/lfm3/speech.zig`) |

Not a seam: **opaque engines.** ds4 and llama.cpp keep their bridges (`src/engines/ds4/bridge.zig`, `src/engines/llama/bridge.zig`).

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

mlx-stream reads the same format with its own decoder and kernels, pinned to DeepSeek-V4.1's shapes
(`src/sdk/quant/exl3/stream/`). The two do not share kernels: each serves a different arch, and its kernels are tuned
to it. mlx-stream's GLM-5.3 reaches sushi's decode through the format (`mlx_host.exl3`).

## Example: streamed experts (weight seam)

A streaming pack keeps its routed experts in `experts.bin` and reads them from the SSD into per-layer slot rows; the
plugin's own trunk runs everything else. Its config declares `expert_streaming`; `parseConfig` sets
`ModelConfig.expert_bank`, the plugin opens the bank at load and binds it behind `Moe.Weights.streamed`
(`sdk/weights/streamed.zig`: a `Routed` with the bank's math, `settle`, `phase` and read-ahead), and its caps say
`streamed_weights`. Three plugins serve one each:

- `deepseek_v41`: V4.1's EXL3 repack, through the SDK streamer (`sdk/weights/streamed/expert/`).
- `mimo_v2`: MiMo-V2.6-Flash's MXFP4 pack, through MiMo's streamer
  (`sdk/weights/streamed/mimo/`).
- `glm5_next`: GLM-5.3-Flash's EXL3 K=3 mul1 pack, MiMo's streamer over the GLM-shape EXL3 texts
  (`sdk/weights/streamed/glm5n/`, `glm5n_source.zig`); its `mtp.0` head's NVFP4 experts load resident.

The slot rows a layer keeps come from free memory at load (the plugin's bill), and the load bills the residents plus
those rows. GLM-5.3 (`glm_moe_dsa`, `plugins/glm_moe_dsa/`) is the one arch still behind the engine's arch seam
(`engines/mlx/modules/stream_glue.zig`: `open`, `begin`, `forward`, `arm`/`round` for DSpark, `end`,
`contextLength`), reached once per prompt or step. All of it builds as the `mlx_stream` module (macOS only; the
Linux and iOS graphs build `stream_stub.zig`, which refuses these packs by name); tests: `zig build mlx-stream-test`,
`zig build mlx-stream-conformance`.

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
