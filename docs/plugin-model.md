# Write a model plugin

A model plugin teaches the server a new LLM, vision or embedding arch without rebuilding it. You build a dylib from the
plugin kit and install it. The host keeps HTTP, templates, tool calls, sampling, scheduling and the KV cache; you supply
the config parse, the weight load, the memory bill and the forward. Reference: [plugins.md](plugins.md).

This walkthrough ports Nanbeige4.3-33B (`model_type: llama`), start to finish.

## 1. Stage the kit and scaffold

```
zig build kit                       # in the mlx-serve checkout: zig-out/kit/mlx-serve-plugin-kit-<version>/
mlx-serve plugin new nanbeige --kit zig-out/kit/mlx-serve-plugin-kit-<version> --dir ~/work
```

Release users get the same kit as the `mlx-serve-plugin-kit` tarball (extract it to `~/.mlx-serve/kit` and skip
`--kit`). The folder has `config.zig` (claim and parse), `plugin.zig` (load, bill, forward, test fixture), `build.zig`.
It is a small Llama decoder in mlx-lm's weight names, f32, tied head, dense weights.

## 2. Claim the pack

`claims` runs with no GPU and returns `.no`, `.by_type` or `.by_marker`. The highest rank wins and **a built-in wins a
tie**, so a plugin that wants a `model_type` the server already serves must claim by marker:

```zig
fn claim(model_type: abi.Str, config_json: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
    if (!std.mem.eql(u8, model_type.slice(), "llama")) return .no;
    return if (std.mem.indexOf(u8, config_json.slice(), "\"num_query_groups\"") != null) .by_marker else .no;
}
```

Nanbeige ships `llama`; `num_query_groups` (a Megatron-style key) tells it apart, so every other Llama stays with the
built-in. A new `model_type` needs only `kit.sdk.claimsByType`. Naming the plugin like a built-in replaces that
built-in for all its packs.

## 3. Adapt the scaffold to the real checkpoint

The example is a fixture, not a loader for real packs. Check the checkpoint's `config.json` and weight index against it:

- **Dtype.** Read `torch_dtype`, set `shell.act_dtype`, and bill 2 bytes per weight for bf16/f16 (the example bills 4).
- **Head.** Nanbeige's `lm_head.weight` is untied: load it, fall back to `embed_tokens` when absent, and bill the extra
  `vocab x hidden` matrix.
- **Config JSON is untrusted.** Bounds-check every field before use (the scaffold does); refuse what you do not
  implement by name.
- **Dense only.** The example's matmul takes unquantized weights. An MLX-quantized pack (affine, mxfp4/8, nvfp4)
  runs through mlx-c's own `mlx_quantized_matmul` and `mlx_gather_qmm`; any other encoding is
  [a quant plugin](plugin-quant.md).

Weight names and config keys should match mlx-lm so packs load across engines.

## The contract in one page

A plugin is a table (`abi.ModelTable`) of C-ABI functions. The host calls them in this order:

| Call | Runs | You do |
|---|---|---|
| `claims(model_type, config_json, dir)` | discovery, no GPU | return `.no`, `.by_type` or `.by_marker` |
| `parse(dir, config_json, settings_json, span, stub, out)` | discovery (`stub` true: `config.json` only) and load | fill `out.shell` and `out.caps`; keep your own config in `out.private`; refuse by name |
| `free_parsed` | after | free `private` |
| `bill(parsed, shape, out)` | admission | `load_bytes`, `kv_per_token` (zero = engine default arithmetic); `shape` has context, slots, KV bits |
| `load(ctx, out_verdict)` | once | return your model pointer, or null with a refusal |
| `forward(model, fwd, out)` | per step | write `out.logits` (and `out.hidden` when `fwd.want_hidden`) |
| `unload` | end | free everything `load` made |

- **`parse` must set `shell`:** `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`, `max_position_embeddings`, `context_cap`, `act_dtype`,
  `model_type`, and the `eos` ids. Call `kit.generation.apply(dir, &p.shell)` on a full parse so
  `generation_config.json` supplies the sampling defaults. Chat template, tokenizer and cache facts default sensibly
  (`Parsed.chat`, `.tokenizer`, `.cache`); set them only when your family differs.
- **`caps` is a promise to the engine.** The scaffold sets `batches_decode`. Leave `moe`, `images`, `embed`,
  `spec_rollback`, `exact_*` and the rest off unless the forward honours them; the scaffold refuses a forward mode the caps
  do not declare (`abi.declaresMode`).
- **`load`:** the engine has already read the pack's tensors into `ctx.weights`; `src.tensor("model.layers.0.self_attn.q_proj.weight")`
  returns an `mlx_array` (null `ctx` = missing). Take a new reference (`mlx_array_set`) for each one you keep; the
  source is valid during `load` only. `ctx.stream` is the GPU stream for any compute.
- **`forward`:** `fwd.caches` and `fwd.positions` hold one entry per slot, `fwd.rows` tokens per slot, token ids in
  `fwd.tokens` (host) or `fwd.ids` (device array `[slots, rows]`; use whichever is set). KV lives in the engine: per
  layer call `cache_ops.append(cache, layer, k, v, 0, stream, &view)`, attend over `view.k`/`view.v`, then
  `cache_ops.release(&view)`. `fwd.logits` says `last_row`, `all_rows` or `none`. Everything you allocate per step
  is freed per step (the scaffold's `Graph` does this).
- **MLX** is the mlx-c API (`@import("mlx_c")`): ops like `mlx_fast_rms_norm`, `mlx_fast_rope`,
  `mlx_fast_scaled_dot_product_attention`, `mlx_matmul`. The kit ships no layer blocks: the server's attention,
  MoE, GDN and quant blocks live in its `src/sdk` and are not part of the plugin contract, so a plugin writes its
  layers on mlx-c (or vendors its own). `kit.sdk` holds the claim helpers and the test fixtures (`src/sdk/root.zig`).

## Read these in the kit

The kit carries the contract and a working example, so a plain decoder needs nothing from the mlx-serve repo.

- `examples/_example/plugin.zig`: a complete working decoder (about 270 lines). Start from `plugin new`, which copies it.
- [plugin-abi.md](plugin-abi.md): every table, struct, field and enum of the ABI, generated from `src/abi/`.
- `src/abi/plugin.zig`: every struct above with field comments (`Shell`, `Caps`, `Parsed`, `Bill`, `LoadCtx`,
  `Forward`, `CacheOps`, `ForwardOut`). The authoritative reference.
- `src/sdk/root.zig` and `src/kit.zig`: what `kit.sdk` and `kit.generation` export.
- `include/mlx_serve_plugin.h`: the same ABI for C, C++ or Rust plugins.

## Out of scope for an outside plugin

Vision towers, speculative decoding heads, MoE streaming, ANE offload and cluster plans are in-tree territory. They
need `Caps` bits and engine hooks that a plain decoder does not. Start from the scaffold; if the model needs more,
read `src/plugins/<similar family>/AGENTS.md` in the repo and consider
[recipes/add-a-model.md](recipes/add-a-model.md).

## 4. Check, then run on the real pack

```
cd ~/work/nanbeige && zig build test && zig build        # zig-out/lib/libnanbeige.dylib
mlx-serve plugin check ./nanbeige                         # conformance suite on the plugin's tiny fixture
mlx-serve plugin check ./nanbeige --model <pack dir>      # also: does it claim and parse this pack?
```

`plugin check` passing proves the contract, not the numerics. Serve the real pack twice and diff greedy output:

```
mlx-serve plugin install nanbeige/zig-out/lib/libnanbeige.dylib
mlx-serve --serve --port 18082 --model <pack dir>                 # plugin on
mlx-serve --serve --port 18083 --model <pack dir> --no-plugins    # built-in
```

The log line `[plugin] nanbeige serves llama (52 layers)` shows who served the pack. For Nanbeige at
`temperature: 0` the two arms returned byte-identical text, at the same decode speed.

## 5. Gotchas

- **Plugins load only when `~/.mlx-serve/plugins.json` says `"enabled": true`.** The Mac app owns that file and writes
  it back, so test under a throwaway `HOME` (copy `plugins/nanbeige.*` and write the JSON there) and leave the app alone.
- A reinstall changes the dylib hash; install again so the receipt matches or the load is refused.
- A dylib is never unloaded; restart the server after reinstalling.
- A plugin built against another MLX pin is refused at load, with the reason in the log.
- Outside plugins are macOS only.

Other kinds: [media](plugin-media.md), [quant](plugin-quant.md). In-tree model plugins (a folder under
`src/plugins/`, full SDK blocks, kernels, proofs): [recipes/add-a-model.md](recipes/add-a-model.md).
