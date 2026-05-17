// Embedded Metal kernel sources for the ds4 inference engine.
//
// ds4's Metal loader (lib/ds4/ds4_metal.m around line 1207) honors per-kernel
// env vars (`DS4_METAL_<NAME>_SOURCE`) that point at filesystem paths. We embed
// the kernel sources here, write them to a per-binary-hash directory under
// `~/.mlx-serve/ds4-metal/<hash>/`, and set the env vars before opening the
// engine. Single-binary shipping; no patches to upstream.
//
// Names match the ds4 loader's lookup table 1:1. Adding a new kernel: drop the
// `.metal` file in lib/ds4/metal/, add a `pub const` here, and add an entry in
// `src/arch/ds4.zig` `kernel_entries`.

pub const argsort: []const u8 = @embedFile("ds4/metal/argsort.metal");
pub const bin: []const u8 = @embedFile("ds4/metal/bin.metal");
pub const concat: []const u8 = @embedFile("ds4/metal/concat.metal");
pub const cpy: []const u8 = @embedFile("ds4/metal/cpy.metal");
pub const dense: []const u8 = @embedFile("ds4/metal/dense.metal");
pub const dsv4_hc: []const u8 = @embedFile("ds4/metal/dsv4_hc.metal");
pub const dsv4_kv: []const u8 = @embedFile("ds4/metal/dsv4_kv.metal");
pub const dsv4_misc: []const u8 = @embedFile("ds4/metal/dsv4_misc.metal");
pub const dsv4_rope: []const u8 = @embedFile("ds4/metal/dsv4_rope.metal");
pub const flash_attn: []const u8 = @embedFile("ds4/metal/flash_attn.metal");
pub const get_rows: []const u8 = @embedFile("ds4/metal/get_rows.metal");
pub const glu: []const u8 = @embedFile("ds4/metal/glu.metal");
pub const moe: []const u8 = @embedFile("ds4/metal/moe.metal");
pub const norm: []const u8 = @embedFile("ds4/metal/norm.metal");
pub const repeat: []const u8 = @embedFile("ds4/metal/repeat.metal");
pub const set_rows: []const u8 = @embedFile("ds4/metal/set_rows.metal");
pub const softmax: []const u8 = @embedFile("ds4/metal/softmax.metal");
pub const sum_rows: []const u8 = @embedFile("ds4/metal/sum_rows.metal");
pub const unary: []const u8 = @embedFile("ds4/metal/unary.metal");
