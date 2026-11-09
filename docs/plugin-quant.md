# Write a quant plugin

A quant plugin adds a weight encoding: how a pack's tensors are stored and how the matmul over them runs. The host keeps
the arch, KV cache, batching, spec decode and prefix cache; a model that reads your encoding gets all of that for free.
Contract: `src/abi/quant.zig`. Overview: [plugins.md](plugins.md). The kit, install and test flow is in
[plugin-model.md](plugin-model.md).

```
mlx-serve plugin new myquant --quant --kit <kit dir> --dir ~/work
```

The scaffold is a stub: claim, validation and bill are real, and loading and the matmuls refuse by name until you fill
them in.

## Claim and validate

A quant plugin claims a pack by its quantization block (`quantization`, `quantization_config` or `expert_quant`,
passed as JSON), not by `model_type`:

```zig
pub fn claims(block_json: abi.Str, config_json: abi.Str) callconv(.c) abi.Claim {
    const b = parseBlock(block_json.slice()) orelse return .no;
    return if (std.mem.eql(u8, b.mode, mode)) .by_type else .no;
}
```

`validate` then checks the block against the pack and refuses unsupported bits or group sizes by name. Both run with no
GPU. `bill` returns the resident bytes for the format given the stored size.

## The table

`abi.QuantTable`:

| Field | What it does |
|---|---|
| `open` / `close` | Open the pack's weight source once per load; free it |
| `load_tensor` | One weight, as an `abi.Weight` the forward can matmul against |
| `load_bank` | A bank of MoE experts as one weight |
| `release` | Free a weight you handed out |
| `matmul` | `x @ W^T` for a weight you loaded |
| `moe` | The routed-expert matmul over a bank (indices, scores) |
| `swiglu` | Optional: routed SwiGLU experts over three banks in one dispatch (`limit` > 0 clamps); with no `inds`, a dense SwiGLU MLP you may decline (`unsupported`: the model runs its own chain) |
| `bank_rows` / `open_bank` | Optional: a streamed bank of routed experts (`experts.bin`): the slot rows a budget buys, and the bank itself (`abi.Bank`), which runs each routed call and returns its sum; null: the format streams none |

An in-tree model that knows no other form of a weight asks the loaded quant plugin claiming the pack's
`quantization_config` (or `quantization`) block (`src/sdk/quant/outside.zig`): it loads the weight through
`load_tensor` / `load_bank` and runs it through `matmul` / `swiglu`, without knowing the encoding. A streamed pack's
routed experts come from your `open_bank` (`outside.Format.openBank` wraps the `abi.Bank` as the trunk's
`sdk/weights/routed.zig` `Routed`): the bank owns its slot rows, reads and math; the trunk hands it each layer's normed
rows, routes and weights. The bank streams over the kit's expert stream (`kit.streamed.expert`: `stream.StreamOf` over
your bank type, the residency policy, the read pool; the kit's `build.zig` `streamSources` compiles the read pool's C
into your dylib, and its MLX internals are C symbols you serve over the `HostGpu` table). DeepSeek-V4.1 reads its EXL3
repack this way (the `turboexl3_ds41` plugin: its trunk, its grouped `wo_a` as one weight your `matmul` takes as
`[rows, groups, in]`, and its streamed bank). Such a pack loads its tensors as stored (no f16 narrowing).

## Rules

- A format is exact against a float64 reference. Register any Metal kernel with the kernel contract
  (`src/sdk/testing/kernel_contract/REGISTERING.md`) when it lives in the tree.
- Refuse an unsupported layout by name at `validate`, not mid-forward.
- Bill resident bytes as an upper bound; a format that expands on load (dequantized copies) bills the expanded size.
- Weight layouts start from mlx-lm and the packs already on Hugging Face, so one pack loads across engines.

## Check

```
cd ~/work/myquant && zig build test && zig build
mlx-serve plugin check ./myquant
mlx-serve plugin check ./myquant --model <pack dir>
```

In-tree examples to copy from: affine, fp8, nvfp4 in `src/sdk/quant/`, sushi's EXL3 in `src/sdk/quant/exl3/`, GGUF in
`src/sdk/quant/gguf/` (a format seam: see [plugins.md](plugins.md) "Pick the shallowest seam").

## Read these in the kit

- `examples/_example_quant/plugin.zig` and its `config.zig`: the scaffold `plugin new` copies.
- `src/abi/quant.zig`: `QuantTable`, `Weight`, `LoadCtx`, `Bank`. Field comments are the reference.
- `src/sdk/root.zig`, `include/mlx_serve_plugin.h`: the SDK exports and the C ABI.
