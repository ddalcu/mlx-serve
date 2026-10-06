//! The dense key-visibility mask of one QSA decode row, written by one kernel. The op chain it
//! replaces (compare, select, scatter, broadcast, pad, three more compares) is about eight dependent
//! launches per attention layer; the mask is just "this token's block is in the sorted selection, or
//! the token is in the incomplete tail" (every key of a one-row step is causal-visible).
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

// grid (kv), one thread a token. `blocks` is sorted ascending with INT_MAX sentinels, so a binary
// search finds a token's block; a sentinel never equals a block id.
const SOURCE =
    \\const uint t = thread_position_in_grid.x;
    \\const int kv = int(threads_per_grid.x);
    \\const int blk = int(t) / RATIO;
    \\bool visible = int(t) >= (kv / RATIO) * RATIO;
    \\if (!visible) {
    \\  int lo = 0;
    \\  int hi = KB - 1;
    \\  while (lo <= hi) {
    \\    const int mid = (lo + hi) >> 1;
    \\    const int v = blocks[mid];
    \\    if (v == blk) { visible = true; break; }
    \\    if (v < blk) lo = mid + 1; else hi = mid - 1;
    \\  }
    \\}
    \\mask[t] = visible;
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
var env_enabled: ?bool = null;
pub var override: ?bool = null;

/// `MLX_SERVE_QSA_MASK_KERNEL=0` keeps the op chain.
fn enabled() bool {
    if (override) |v| return v;
    if (env_enabled) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_QSA_MASK_KERNEL");
    env_enabled = raw == null or raw.?[0] != '0';
    return env_enabled.?;
}

/// `[1, 1, 1, kv]` bool mask for one decode row over `blocks` `[1, 1, kb]` int32, or null outside
/// the kernel's set (the caller keeps the op chain).
pub fn rowMask(s: mlx.mlx_stream, blocks: mlx.mlx_array, kv: c_int, ratio: c_int) !?mlx.mlx_array {
    if (!enabled() or !mlx.streamIsGpu(s)) return null;
    const bs = mlx.getShape(blocks);
    if (bs.len != 3 or bs[0] != 1 or bs[1] != 1 or bs[2] < 1 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    if (kv < 1 or ratio < 1) return null;
    if (kernel == null) {
        const ins = [_][*:0]const u8{"blocks"};
        const outs = [_][*:0]const u8{"mask"};
        const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("mlxserve_qsa_row_mask", in_vec, out_vec, SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    // The mask's width is the step's kv, so the config is built per call.
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, 1, kv }, 4, .bool_));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, kv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(kv, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KB", bs[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RATIO", ratio));
    const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{blocks}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel.?, v, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    if (!engaged) {
        engaged = true;
        log.info("[qsa] one-row key mask kernel engaged (MLX_SERVE_QSA_MASK_KERNEL=0 restores the op chain)\n", .{});
    }
    return out;
}

const testing = std.testing;

test "qsa mask: the one-row kernel equals the op chain for sorted selections with sentinels and tails" {
    const xfm = @import("transformer.zig");
    defer override = null;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x9A5C);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const cases = [_]struct { kv: c_int, kb: c_int, picks: c_int }{
        .{ .kv = 8076, .kb = 512, .picks = 512 }, // a full selection, the tail empty
        .{ .kv = 8077, .kb = 512, .picks = 512 }, // a one-token tail
        .{ .kv = 8079, .kb = 512, .picks = 300 }, // sentinels at the end, a three-token tail
        .{ .kv = 2053, .kb = 512, .picks = 512 }, // just past the budget
        .{ .kv = 65, .kb = 16, .picks = 5 }, // a tiny cache
    };
    for (cases) |c| {
        const nb: usize = @intCast(@divTrunc(c.kv, ratio));
        var pool = try testing.allocator.alloc(i32, nb);
        defer testing.allocator.free(pool);
        for (pool, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(i32, pool);
        const kb: usize = @intCast(c.kb);
        const picks: usize = @min(@as(usize, @intCast(c.picks)), nb);
        const sel = try testing.allocator.alloc(i32, kb);
        defer testing.allocator.free(sel);
        @memset(sel, std.math.maxInt(i32));
        @memcpy(sel[0..picks], pool[0..picks]);
        std.mem.sort(i32, sel, {}, std.sort.asc(i32));
        const blocks = mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ 1, 1, c.kb }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);

        override = false;
        const want = try xfm.qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        override = true;
        const got = (try rowMask(s, blocks, c.kv, ratio)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(want);
        defer _ = mlx.mlx_array_free(got);
        try testing.expectEqualSlices(c_int, &[_]c_int{ 1, 1, 1, c.kv }, mlx.getShape(got));
        var wf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wf);
        var gf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gf);
        try mlx.check(mlx.mlx_astype(&wf, want, .float32, s));
        try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
        try mlx.check(mlx.mlx_array_eval(wf));
        try mlx.check(mlx.mlx_array_eval(gf));
        const n: usize = @intCast(c.kv);
        try testing.expectEqualSlices(f32, (mlx.mlx_array_data_float32(wf) orelse return error.Unreadable)[0..n], (mlx.mlx_array_data_float32(gf) orelse return error.Unreadable)[0..n]);
    }
}
