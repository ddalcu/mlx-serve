# Plugin SDK: what an outside model plugin can reach, and what is left

An outside plugin that runs its own forward should import the server's blocks, not copy them.

| Part | Home | State |
|---|---|---|
| mlx bindings | `kit.mlx` | done |
| norms, conv, ssm, schedulers, vae, dit, embedding, linear, qmm, mlp, rope, attention, moe, hc, gdn | `kit.blocks` | done (Metal; versioned with the kit) |
| a weight layout of its own | `kit.foreign.register` (`sdk/quant/foreign.zig`) | done |
| the MTP round policy | `kit.spec`, pure parts only | not started |
| kernel contract and parity harness | `kit.testing` | not started |
| config helpers, kv, mrope, util | `kit.sdk` | partly (`kit.sdk`, `kit.generation`) |

## Rules

- A block imports no format. Dispatch to a layout it does not read goes through `foreign`; a new format is a `Format`,
  never an `@import` in a block. The server's formats are `foreign_builtin.zig`'s list; the kit stages an empty one
  (`stagedInstead` in `build/kit.zig`: keep that seam to one small file).
- Blocks are versioned with the kit, not with the ABI. The ABI (`abi/`) stays additive and negotiated; blocks may change
  between kits, and a plugin pins one.
- Every public function of `kit.blocks` builds from the staged kit (`build/kit/blocks_check`, run by `zig build test`), and
  every block file of an exposed area is listed in `src/kit.zig` or is a test, contract or CUDA file (`kit_test.zig`).

## MTP: native rounds stay, the policy ships as a library

`qwen4_exp`'s MTP keeps running behind the engine's own symbols, unchanged: it is perf-critical there, and moving it onto
`SpecHooks.head` risks regressions for no built-in gain. A plugin with its own round policy uses `NativeDrafter`
(`Host.latch_message`, `Forward.min_depth/max_depth`, `NativeDrafter.commit_forced`). What remains is the policy code a
plugin has to copy (`engines/mlx/spec/mtp/`: the EV planner, two-chunk plans, prompt lookup, the padded head history).
Shipping it as `kit.spec`:

- Move only code with no engine dependency (pure planning and cost math) into `sdk/spec/mtp/`, and have the engine import it
  from there: the same code at a new path, no behaviour change, the MTP tests and a bench against the previous build as
  the gate.
- Code that reads the engine's `Generator` stays in the engine; a plugin states what it needs of a generator through its
  own type.

## A plugin that forks a built-in

A built-in written against engine internals (KV cache, SSM state, loader, batching) cannot be reached through the kit, so a
plugin that forks its forward carries its own copy of those and moves with mlx-serve when they change. Pin the mlx-serve
ref it is built against, generate the copy from that ref where it can, and expect to rebase. Forward stages (prefill
wavefronts, fused decode steps) work over the model's own state; an ABI for them would expose those internals, so a fork
of the forward stays a fork. `qwen4_exp` is one such built-in. A plugin that writes its model on `kit.blocks`, `kit.foreign`
and `kit.mlx` does not carry this cost.

Kernels a plugin ships stay the plugin's.
