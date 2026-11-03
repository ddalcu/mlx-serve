# Add a kernel

1. Put the source in a file: `src/kernels/<area>/` when two or more plugins use it, `src/plugins/<name>/kernels/`
   when one does. List it with its sha256 in `src/kernels/digests.txt` (`shasum -a 256`, run from `src/`).
2. Load it through `src/mlx/kernels.zig` (`source` / `resolve`; `MLX_SERVE_KERNEL_DIR` swaps the text at runtime).
3. Register it with the kernel contract: `src/sdk/testing/kernel_contract/REGISTERING.md` (five gates against a
   float64 reference; the planted controls must fail).
4. Gate it on the kernel's own arithmetic (shape, dtype, threadgroup memory), never on one model's shape, and give
   it a kill switch in `src/env.zig` only if the old path is worth comparing.
5. A/B it in situ at per-cell medians before default-on.

## Iterate fast

A ReleaseFast test build costs minutes; a kernel body edit should cost seconds.

1. Settle the wrapper first: launch grid, threadgroup size, inputs, template arguments. Each of those is a rebuild.
2. Build the ReleaseFast test binary once with your filter (`-Dtest-filter=...`) and keep it.
3. Edit the kernel in a copy tree laid out like `src/` and rerun the same binary with
   `MLX_SERVE_KERNEL_DIR=<tree>`; the log's `[kernels] <path>` line shows the override took. The server reads it
   the same way: edit, restart, no build.
4. Only text MLX receives as one whole file is swapped: a source or header assembled with `++` in Zig, or a
   `.fmt.metal` template, still needs a rebuild. Keep what you tune in the body file.
5. When done: copy the files back into `src/`, update `digests.txt`, and run the full filters once on a fresh build.

## Iterate as a plugin (TBD: measure, then rewrite or delete this section)

Untested idea: build the plugin from the kit (`zig build kit`, `mlx-serve plugin new`, its own test build against
`src/plugin_host.zig`) so a wrapper edit rebuilds a small project instead of the whole test binary, and
`plugin install` the dylib to replace the built-in in a live server. To find out:
- Time a one-kernel wrapper change: the plugin's own kit test build vs our filtered ReleaseFast test build.
- Which plugins qualify: their imports stay in what the kit exports (`zig build unit-closure`), and the engine does
  not drive them through `engines/mlx/plugin/intree.zig` (deepseek_v41's DSpark rounds do, so its dylib could not
  serve DSpark).
- Whether the kernel contract and the plugin's fixtures run from the kit build.
Write the measured times and the limits here; if it is not several times faster, delete this section.
