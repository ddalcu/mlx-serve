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

The dispatch is per load and per forward, never per layer, so a model plugin routes its quantized tensors through your
table without knowing the encoding.

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

In-tree examples to copy from: affine, fp8, nvfp4 in `src/sdk/quant/`, EXL3 in `src/sdk/quant/exl3/`, GGUF in
`src/sdk/quant/gguf/` (a format seam: see [plugins.md](plugins.md) "Pick the shallowest seam").

## Read these in the kit

- `examples/_example_quant/plugin.zig` and its `config.zig`: the scaffold `plugin new` copies.
- `src/abi/quant.zig`: `QuantTable`, `Weight`, `LoadCtx`. Field comments are the reference.
- `src/sdk/root.zig`, `include/mlx_serve_plugin.h`: the SDK exports and the C ABI.
