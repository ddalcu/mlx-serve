# Registering a kernel with the contract

Every kernel registers a spec and a test type; one harness runs five gates on it (`registry.zig`). The worked example
is `src/sdk/blocks/norms/contract.zig` (add_norm's three arms, the residual-norm seam); the smallest test type is
`Row` in `controls.zig`.

## Where

One `contract.zig` per area, beside the area's Zig code (`src/sdk/blocks/<area>/contract.zig`,
`src/plugins/<name>/contract.zig`), exporting `pub fn register(set: *kc.Set) !void`, added to `areas` in
`registry.zig`. The kernel files and their launch code do not change; a private source const the contract needs
becomes `pub` (one word). The to-do list is `.bb/reports/kernel-registry-todo.tsv`.

## What a registration holds

1. A `metal.Launch` per pipeline family: name, input and output names, `source`/`header` (the wrapper's own consts),
   `tg_bytes` (threadgroup arrays at the largest instantiation), `m1`/`m2` plus `caps` (the M1/M2 limits, below),
   `adapts` (only if the wrapper probes or reads the limit at EVERY instantiation), `offline = false` for tensor-op
   kernels, `min_gen = 17` for NAX-only ones.
2. A test type with fields `rig: metal.Rig` and `launch: *const metal.Launch`, and methods:
   - `geometry(shape) ?metal.Geometry`: grid, threadgroup, outputs and template args exactly as the wrapper builds
     them, template args in the wrapper's order; null where the wrapper declines. Outputs it never writes by design
     go in `unread`.
   - `prepare(shape, fill, seed) !bool`: host values (f64, already rounded to the input dtype) for `.random`,
     `.nonfinite` (NaN, +Inf, -Inf at known rows; one row with both) and `.stress`, bound with `rig.input` /
     `rig.scalar` in the kernel's input order; false where the kernel refuses the shape.
   - `run()`: `rig.launch(self.launch, geometry, args)`.
   - `count()`, `reference(want, mag)`: the formula in float64. Round to the dtype at the op chain's points for
     values that are outputs or feed a later stage; leave the last stage unrounded (`mids` counts those roundings).
   - `simulate(acc, order, out)` when `accum = .f32`: the outputs with the sums run in `acc` (`spec.sumIn`).
   - `replaced()` when the kernel replaces an op chain or kernel: that chain's bytes on the same `rig` inputs.
   - `check(gate)` for a stated NaN rule (top-k, argmax, sampling); `deinit()`.
3. A `spec.Spec`: accumulator, `depth`, `ops`, `mids`, output element, `replaces`, `nan`, `tiles` (per dimension: tile,
   step, least size), `served` shapes (what the engine calls it at), `sums` (the fp16/bf16 margins gate 4 must prove;
   0 with a `why` where the output's rounding hides them), `known`.
4. A test in the area's contract that the declared launch equals the shipped wrapper's output bytes at a served
   shape (the drift guard; see the norms one).

## The gates

1. Non-finite: output classes equal float64's. 2. Repeat: five runs, same bytes; inputs and their 64 guard bytes
untouched (a view into a larger buffer); outputs start as NaN (`Geometry.poison`), so an unwritten element shows.
3. Shapes: served shapes plus each tiled dimension at least / tile - step / tile / tile + step / largest; bit-equal
to `replaced`, or within the float64 bound. 4. fp32 sums: integer `.stress` inputs exact in fp32 in any order, with
every other stage exact. 5. Limits: every shape's threads and threadgroup bytes against M1..M5, no device.

## Commands

- `zig build test -Dtest-filter="kernel contract"`: gates 1-5, every control (about a second on the device).
- M1/M2 limits: `MLX_SERVE_KERNEL_CONTRACT_OFFLINE=~/claude-tmp/kc zig build test -Dtest-filter="kernel contract
  offline"` compiles each instantiation with Xcode's `metal-tt` for M1 and M2 and prints the limit per shape where it
  is under 1024 or disagrees with the record. Record the least as `m1`/`m2`, per-shape exceptions as `caps`.
  INCONCLUSIVE (every thread request changes the build) leaves the record as is.

## When a gate fails

A shipped kernel that fails a gate gets `known = .{ .gate, .finding }` with the reason; the gate keeps failing
(KNOWN) until the fix lands, then the run fails until the entry goes. Never widen a bound or drop a shape to pass.
A new `.metal`/`.cu` file (a control) needs its line in `src/kernels/digests.txt`. CUDA: gate 5's tables
(`limits.cuda`, sm_80..sm_121) are ready; gates 1-4 need a CUDA rig (not built yet).
