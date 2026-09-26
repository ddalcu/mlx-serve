# Engine plugins: folding community forks back in as pinned, compiled-in parts

Status: PROPOSAL (for discussion). Nothing here is implemented yet.

## Goal

Forks like [sushi](https://github.com/beamivalice/sushi) add real engine work (EXL3 expert quants, SSD expert
streaming, MiMo-V2) on top of a copy of our tree. We want that work to ship inside mlx-serve the way ds4 does:

- compiled in OUR build, from THEIR repo, at a PINNED commit;
- only the parts that differ come in, never a second copy of the server or engine core;
- a very small amount of glue on our side (target: one pin line + one registry line per plugin);
- the author develops and tests against a slim mlx-serve on their own machine, with no fork.

## Why not "just like ds4" as it is today

ds4 is the right shape (pinned submodule, `src/arch/ds4.zig` bridge, Linux stub) but the wrong amount of glue: about
80 `ds4_engine`/`llama_engine` branches spread over `server.zig`, `scheduler.zig`, `model_registry.zig`, `chat.zig`,
`gen.zig`. A third engine done that way adds another 80. The refactor below turns those branches into ONE interface.
ds4 and llama.cpp become the first two plugins, so the interface is tested by the code we already ship.

## Plugin kinds (the extension points)

A fork's delta is rarely "a whole engine". Sushi's is three separate things, and each maps to a hook we own:

| Kind | Host hook | What the plugin supplies | Host keeps | Sushi example |
|---|---|---|---|---|
| `quant` | weight load + `QLinear`/MoE matmul | format detection from `quantization_config`, tensor loader, `matmul`, `gatherMatmul` for experts | trunk, KV, batching, spec, cache | EXL3 experts (`expert_exl3*.zig`, `expert_quant.zig`) |
| `expert_source` | the MoE layer's expert fetch | where routed experts live and how they reach the GPU; a memory bill | the MoE math, routing | SSD expert streaming (`expert_stream.zig`, `expert_io.zig`, `expert_bf16_kernels.zig`, these import only `mlx` + `log`) |
| `arch` | `transformer.zig` dispatch on `model_type` | a forward over the host's `KVCache` + `ModelConfig` | HTTP, templates, tools, sampling, scheduler, prefix cache | MiMo-V2 (`mimo_source.zig`, `mimo_mtp.zig`, `mimo_vision.zig`) |
| `engine` | registry: an opaque session | open/close, tokenize, session `sync`/`eval`/`sample`/`rewind`/snapshot | HTTP, templates, tools, routing | ds4, llama.cpp (moved behind the interface) |

The deeper the kind, the more of our stack the plugin gets for free (`arch` gets batching, MTP, prefix cache; `engine`
gets none of it). Plugins pick the shallowest kind that expresses their work.

## The SDK: what a plugin may import

`src/sdk.zig` is a curated facade and the ONLY module a plugin sees:

- `sdk.api`: `{ major, minor }`. Major = breaking.
- `sdk.mlx` (our FFI), `sdk.log`, `sdk.ConfigPeek` (parsed `config.json` / `quantization_config` fields),
  `sdk.KVCache` (the `denseView`/`update` contract), `sdk.ForwardCtx`, `sdk.Linear` interface, `sdk.MemoryBill`,
  `sdk.testing` (conformance helpers, below).
- Nothing else. Internals (`transformer.zig`, `scheduler.zig`, ...) stay free to refactor, because no plugin can
  reach them.

## Negotiation (at compile time)

Each plugin exports one declaration:

```zig
pub const plugin = sdk.Plugin{
    .name = "sushi-exl3",
    .api = .{ .major = 1, .minor = 0 },   // the SDK it was built against
    .mlx = "v0.32.2",                     // the pin it was tested on
    .macos_only = true,                   // Linux/iOS graphs get no-op registration
    .provides = .{ .quant = Exl3 },       // any subset of: quant, expert_source, arch, engine
};
```

`src/plugins.zig` is the registry: one line per plugin, `@import("sushi_exl3").plugin`. At comptime the host:

1. rejects a major `api` mismatch with `@compileError` naming the plugin and both versions; a newer minor on either
   side works (optional hooks are `?fn` and default to "not implemented");
2. rejects an `mlx` pin that differs from ours (one MLX per process), so bumping MLX is one PR that bumps every pin;
3. checks every `provides` entry against its kind's interface (missing or mistyped `fn` = compile error naming it);
4. registers each entry with its hook's table.

At runtime routing is ONE question per hook, asked in registry order:
`claims(peek: sdk.ConfigPeek) ?Priority`. Discovery asks the `engine` and `arch` tables for a model, the load path asks
`quant` per weight group and `expert_source` per MoE layer. `model-settings.json` may name a plugin to break a tie.
`/v1/models` and `/props` carry `plugins: [...]` so a user can see what served a model.

## Pinning and build

- Zig plugins are dependencies in `build.zig.zon` (url + content hash), which is a reproducible pin with no
  submodule dance. C/C++ engines (ds4, llama.cpp) keep their submodule or fetch script, behind the same interface.
- The plugin's `build.zig` exposes a module WITHOUT importing the SDK itself; our `build.zig` injects our `sdk`
  module (`addImport("mlxserve", sdk)`), so there is one SDK and one MLX in the binary.
- Bump = a PR that changes the hash. CI builds every plugin and runs its conformance tests; red = the bump waits.
- The plugin's own license and attribution go into `NOTICE`, as for ds4.

## The slim host: what an author runs

The plugin repo depends on mlx-serve by a pinned tag, and its `build.zig` has two steps:

- `zig build serve`: our server built with `-Dslim=true` (no media gen, ds4, llama.cpp, ANE, app) plus this one
  plugin. It is the real server code with the heavy parts left out, so it builds in a fraction of the full time
  and behaves exactly like the full server on their plugin's models.
- `zig build conformance`: `sdk.testing` runs the invariants we would otherwise catch in review: config claims on
  fixture configs, a greedy golden vs the author's oracle dump, KV `truncate`/rewind equivalence, memory bill ≥
  measured peak, the `arch` verify invariant when MTP is claimed, and a Linux-stub build.

The same conformance step runs in OUR CI against the pinned commit, so "works on the author's machine" and "works in
mlx-serve" are the same test.

## What folding sushi back looks like

In our tree: one `build.zig.zon` entry, one line in `src/plugins.zig`, one `NOTICE` paragraph. In theirs:
`expert_stream` + `expert_io` + kernels as an `expert_source` plugin; `expert_exl3*` + `expert_quant` as a `quant`
plugin; MiMo as an `arch` plugin once `sdk.KVCache`/`ForwardCtx` cover what it reads. Their fork of our
`transformer.zig`, `server.zig`, etc. goes away. The work they do in those files either lands upstream as a normal PR
or turns out to be a hook the SDK is missing.

## Refactor plan (our side)

1. `src/sdk.zig` + `sdk.Plugin` + comptime negotiation, with hermetic tests (a fake plugin per kind, and one per
   rejection: api major, mlx pin, missing fn).
2. `engine` kind: move ds4 and llama.cpp behind it, deleting the scattered branches. Characterization tests first
   (`tests/test_ds4_serve.sh` + the llama GGUF path must stay byte-identical).
3. `quant` and `expert_source` hooks in the load path and the MoE layer (the resident path is itself the default
   `expert_source`).
4. `-Dslim` build + `sdk.testing` conformance + a template plugin repo.
5. `arch` kind (widest surface: KV, ForwardCtx, capture for spec). Last, because it freezes the most internals.
6. First external plugins: sushi `expert_source` and `quant`, with its authors.

## Open questions

- Should `arch` plugins get MTP/batching automatically, or opt in per capability (`batches_decode`, `mtp_head`)?
  Opt-in is safer: the verify and batched invariants are where silent corruption lives.
- An external plugin needs the same TDD, class-guard and gotcha rules we follow. Proposal: its conformance suite is the
  gate, and our `CLAUDE.md` rules apply to its glue in our tree, not to its internals.
- Out-of-process guests (a separate binary, loopback proxy, `--parent-pid`, which sushi already supports) remain
  the fallback for an engine that cannot share our MLX pin.
