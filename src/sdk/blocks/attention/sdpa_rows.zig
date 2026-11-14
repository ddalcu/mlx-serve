//! `sdpa.rowByRowSdpa` in one launch per kind of row: query row r of a verify of S rows attends the first
//! N_r = T - S + r + 1 keys with the words of its own one-row MLX SDPA call. MLX runs a one-row call on its vector
//! kernels: the one-pass `sdpa_vector` below `planRow`'s threshold, else the two-pass `sdpa_vector_2pass_1` + `_2` at
//! a `blocks` count chosen from the keys, the device and the simdgroups. A causal call of S rows would choose its blocks
//! at S simdgroups and mask by position, so its rows are not their one-row calls' words. Here the rows are grouped by
//! their one-row plan (consecutive rows of one plan a group), and each group runs MLX's kernel statements
//! (kernels/attention/sdpa_rows_*.metal) for each of its rows over that row's own N. Groups join on the row axis.
//! Serves one sequence, as many key heads as query heads, head dims (192, 128) or (128, 128), bf16 or f16, up to
//! `max_rows` rows; false elsewhere (the caller runs rowByRowSdpa), checked before any launch. The keys and values are
//! read in place at their strides (the cache's views), as MLX's own call reads them.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const env = @import("../../../env.zig");
const log = @import("../../util/log.zig");

/// A one-row call's kernel: the one-pass vector kernel, or the two-pass kernels at `blocks` partials a row.
pub const Kind = enum { one_pass, two_pass };
pub const Plan = struct {
    kind: Kind,
    blocks: u32 = 0,
    pub fn eql(a: Plan, b: Plan) bool {
        return a.kind == b.kind and a.blocks == b.blocks;
    }
};

/// The two-pass kernel's keys per main-loop step (sdpa_vector_2pass_1's TK at these head dims: MLX's _tk1 variant
/// serves other dims only).
pub const tk: u32 = 4;
/// The most rows one call takes: a serial verify's (each row is its own plan's work, so the cap is the plans array).
pub const max_rows: c_int = 16;

/// MLX's plan for one query row over `n_keys` keys at `gqa` query heads a key head (eval_gpu's vector mode at q_len 1,
/// where the causal flag is off, and sdpa_vector_2pass's blocks at n_simds = gqa): the two-pass kernels on a large
/// device ('d' or 's', the architecture string's last letter) from 1024 keys or with gqa from 4096, else the one-pass
/// kernel. `blocks_env`: MLX_SDPA_BLOCKS (0 unset), rounded up to 32 as MLX rounds it.
pub fn planRow(devc: u8, gqa: u32, n_keys: u32, blocks_env: u32) Plan {
    const two = ((devc == 'd' or devc == 's') and n_keys >= 1024) or (gqa > 1 and n_keys >= 4096);
    if (!two) return .{ .kind = .one_pass };
    const n_simds = gqa;
    var blocks: u32 = undefined;
    if (devc == 's') {
        blocks = 64;
        if (n_keys > 1024 and n_simds > 4) {
            blocks = if (n_keys <= 8192) 128 else if (n_keys <= 32768) 256 else if (n_keys <= 65536) 512 else 1024;
        }
    } else if (devc == 'd') {
        blocks = 128;
        if (n_simds <= 2 and n_keys > 8192) {
            blocks = 256;
        } else if (n_simds >= 6) {
            if (n_keys >= 16384 and n_keys < 65536) {
                blocks = 512;
            } else if (n_keys >= 65536) {
                blocks = 1024;
            }
        }
    } else {
        blocks = if (n_simds >= 4) 64 else 32;
    }
    if (blocks_env > 0) blocks = ((blocks_env + 31) / 32) * 32;
    return .{ .kind = .two_pass, .blocks = blocks };
}

var dev_arch: ?u8 = null;

/// The device's architecture letter as MLX reads it (`get_architecture().back()`, e.g. applegpu_g17s), read once; 0
/// where it cannot be read.
pub fn deviceClass() u8 {
    if (dev_arch) |c| return c;
    var buf: [64]u8 = undefined;
    const arch = device.gpuArchitecture(&buf) orelse "";
    const c: u8 = if (arch.len > 0) arch[arch.len - 1] else 0;
    dev_arch = c;
    return c;
}

/// The head dims served: (192, 128) and (128, 128).
pub fn admits(qk: c_int, v: c_int) bool {
    return (qk == 192 and v == 128) or (qk == 128 and v == 128);
}

pub const SOURCE_1P: [:0]const u8 = kernelSource("kernels/attention/sdpa_rows_1p.metal");
pub const SOURCE_2P1: [:0]const u8 = kernelSource("kernels/attention/sdpa_rows_2p1.metal");
pub const SOURCE_2P2: [:0]const u8 = kernelSource("kernels/attention/sdpa_rows_2p2.metal");
pub const SOURCE_R1_1P: [:0]const u8 = kernelSource("kernels/attention/sdpa_rows_r1_1p.metal");
pub const SOURCE_R1_2P1: [:0]const u8 = kernelSource("kernels/attention/sdpa_rows_r1_2p1.metal");
pub const IN_ROWS = [_][*:0]const u8{ "qin", "kin", "vin", "p", "sc" };
pub const IN_PASS2 = [_][*:0]const u8{ "partials", "sums", "maxs", "p" };
pub const OUT_ROWS = [_][*:0]const u8{"oout"};
pub const OUT_PASS1 = [_][*:0]const u8{ "partials", "sums", "maxs" };

var one_pass_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pass1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pass2_kernel: ?mlx.mlx_fast_metal_kernel = null;
var r1_one_pass_kernel: ?mlx.mlx_fast_metal_kernel = null;
var r1_pass1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var engaged = false;
var r1_engaged = false;
/// Groups launched (one or two kernels each): a served verify's engagement.
pub var launches: usize = 0;

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// The operands at their own strides (no row-contiguous copy): the cache's views, read in place.
fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src.ptr, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn apply(k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    if (mlx.mlx_vector_array_size(ov) != outs.len) return error.MetalKernelBadOutputCount;
    for (outs) |*o| o.* = .{ .ctx = null };
    errdefer for (outs) |o| free(o);
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
}

/// One group's rows [r0, r0 + rows) of `plan`, n0 the keys of its first row: [1, H, rows, V] in q's dtype.
pub fn launchGroup(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, sc: mlx.mlx_array, h: c_int, qk: c_int, vd: c_int, r0: c_int, rows: c_int, n0: c_int, plan: Plan) !mlx.mlx_array {
    return launchGroupAt(s, q, k, v, sc, h, qk, vd, r0, rows, n0, plan, 0);
}

/// `launchGroup` over a rope-once cache (`attendRope1`): keys [1, 1, T, H * nope + qk - nope], values [1, 1, T, H * vd].
pub fn launchGroupRope1(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, sc: mlx.mlx_array, h: c_int, qk: c_int, vd: c_int, r0: c_int, rows: c_int, n0: c_int, plan: Plan, nope: c_int) !mlx.mlx_array {
    return launchGroupAt(s, q, k, v, sc, h, qk, vd, r0, rows, n0, plan, nope);
}

/// nope 0: the dense kernels; else the rope-once kernels at `nope` non-positional key dims a head.
fn launchGroupAt(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, sc: mlx.mlx_array, h: c_int, qk: c_int, vd: c_int, r0: c_int, rows: c_int, n0: c_int, plan: Plan, nope: c_int) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(q);
    const out_shape = [_]c_int{ 1, h, rows, vd };
    const pv = [_]i32{ r0, n0, @intCast(plan.blocks) };
    const p = mlx.mlx_array_new_data(&pv, &[_]c_int{3}, 1, .int32);
    defer free(p);
    switch (plan.kind) {
        .one_pass => {
            const kern = if (nope == 0)
                try kernel(&one_pass_kernel, "msv_sdpa_rows_1p", &IN_ROWS, &OUT_ROWS, SOURCE_1P)
            else
                try kernel(&r1_one_pass_kernel, "msv_sdpa_rows_r1_1p", &IN_ROWS, &OUT_ROWS, SOURCE_R1_1P);
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, out_shape.len, dt));
            // MLX's grid: threadgroups (B H, q_len) of 1024 threads, a launch row a query row here.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, h * 1024, rows, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", qk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", vd));
            if (nope != 0) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", nope));
            var o: [1]mlx.mlx_array = undefined;
            try apply(kern, cfg, s, &.{ q, k, v, p, sc }, &o);
            return o[0];
        },
        .two_pass => {
            const blocks: c_int = @intCast(plan.blocks);
            const k1 = if (nope == 0)
                try kernel(&pass1_kernel, "msv_sdpa_rows_2p1", &IN_ROWS, &OUT_PASS1, SOURCE_2P1)
            else
                try kernel(&r1_pass1_kernel, "msv_sdpa_rows_r1_2p1", &IN_ROWS, &OUT_PASS1, SOURCE_R1_2P1);
            const cfg1 = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg1);
            const part_shape = [_]c_int{ h, rows, blocks, vd };
            const stat_shape = [_]c_int{ h, rows, blocks };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &part_shape, part_shape.len, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &stat_shape, stat_shape.len, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg1, &stat_shape, stat_shape.len, .float32));
            // MLX's grid: threadgroups (key heads, B, blocks) of (32, gqa, q_len), a simdgroup a query row.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg1, h * 32, 1, blocks * rows));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg1, 32, 1, rows));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg1, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "D", qk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "V", vd));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "BLOCKS", blocks));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "TK", @intCast(tk)));
            if (nope != 0) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg1, "NOPE", nope));
            var po: [3]mlx.mlx_array = undefined;
            try apply(k1, cfg1, s, &.{ q, k, v, p, sc }, &po);
            defer for (po) |a| free(a);
            const k2 = try kernel(&pass2_kernel, "msv_sdpa_rows_2p2", &IN_PASS2, &OUT_ROWS, SOURCE_2P2);
            const cfg2 = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg2);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg2, &out_shape, out_shape.len, dt));
            // MLX's grid: threadgroups (B H, q_len) of 1024 threads.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg2, h * 1024, rows, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg2, 1024, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg2, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg2, "V", vd));
            var o: [1]mlx.mlx_array = undefined;
            try apply(k2, cfg2, s, &.{ po[0], po[1], po[2], p }, &o);
            return o[0];
        },
    }
}

/// `out` (the caller's handle) = rowByRowSdpa's result for queries [1, H, S, D] over keys [1, H, T, D] and values
/// [1, H, T, V] (the query rows the last S of the T): row r over the first T - S + r + 1 keys, each row's words its own
/// one-row MLX SDPA call's. False where it does not serve (nothing launched): GQA, other head dims or dtypes, past
/// `max_rows` rows, a CPU stream.
pub fn attend(out: *mlx.mlx_array, queries: mlx.mlx_array, keys: mlx.mlx_array, values: mlx.mlx_array, scale: f32, seq_len: c_int, s: mlx.mlx_stream) !bool {
    const qs = mlx.getShape(queries);
    const ks = mlx.getShape(keys);
    const vs = mlx.getShape(values);
    if (qs.len != 4 or ks.len != 4 or vs.len != 4) return false;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return false;
    if (qs[1] != ks[1] or vs[1] != ks[1] or vs[2] != ks[2] or qs[3] != ks[3]) return false;
    if (!admits(qs[3], vs[3])) return false;
    const dt = mlx.mlx_array_dtype(queries);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(keys) != dt or mlx.mlx_array_dtype(values) != dt) return false;
    if (seq_len < 1 or seq_len > max_rows or qs[2] != seq_len or ks[2] < seq_len) return false;
    if (!mlx.streamIsGpu(s)) return false;
    try attendRows(out, queries, keys, values, scale, seq_len, s, qs[1], ks[2], qs[3], vs[3], 0);
    return true;
}

/// `attend` over a rope-once MLA cache: `keys` [1, 1, T, H * nope + D - nope] holds each token's per-head
/// non-positional keys, then the one roped key every head shares; `values` [1, 1, T, H * V]. Each row's words are
/// `attend`'s over the dense keys [1, H, T, D] (nope | rope broadcast) and values [1, H, T, V] of the same cache. False
/// where it does not serve (nothing launched), as `attend`. The rows are read at their token stride, each contiguous.
pub fn attendRope1(out: *mlx.mlx_array, queries: mlx.mlx_array, keys: mlx.mlx_array, values: mlx.mlx_array, nope: c_int, scale: f32, seq_len: c_int, s: mlx.mlx_stream) !bool {
    const qs = mlx.getShape(queries);
    const ks = mlx.getShape(keys);
    const vs = mlx.getShape(values);
    if (qs.len != 4 or ks.len != 4 or vs.len != 4) return false;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1 or ks[1] != 1 or vs[1] != 1) return false;
    const h = qs[1];
    const d = qs[3];
    if (nope <= 0 or nope >= d or @rem(vs[3], h) != 0) return false;
    const vd = @divExact(vs[3], h);
    if (ks[3] != h * nope + d - nope or vs[2] != ks[2] or !admits(d, vd)) return false;
    const dt = mlx.mlx_array_dtype(queries);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(keys) != dt or mlx.mlx_array_dtype(values) != dt) return false;
    if (seq_len < 1 or seq_len > max_rows or qs[2] != seq_len or ks[2] < seq_len) return false;
    if (!mlx.streamIsGpu(s)) return false;
    try attendRows(out, queries, keys, values, scale, seq_len, s, h, ks[2], d, vd, nope);
    if (!r1_engaged) {
        r1_engaged = true;
        log.info("[sdpa-rows] rope-once engaged: {d} heads of {d} | {d}, value {d}\n", .{ h, nope, d - nope, vd });
    }
    return true;
}

/// The dense keys [B, H, T, D] and values [B, H, T, V] of a rope-once cache's `keys` [B, 1, T, H * nope + D - nope]
/// and `values` [B, 1, T, H * V]: each head's nope keys, then the shared roped key broadcast to every head. The same
/// elements as the per-head cache holds, so any attention over them gives that cache's words. Caller frees both.
pub fn denseRope1(keys: mlx.mlx_array, values: mlx.mlx_array, h: c_int, nope: c_int, s: mlx.mlx_stream, k_out: *mlx.mlx_array, v_out: *mlx.mlx_array) !void {
    const ks = mlx.getShape(keys);
    const vs = mlx.getShape(values);
    const b = ks[0];
    const t = ks[2];
    const rope = ks[3] - h * nope;
    const vd = @divExact(vs[3], h);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const ones = [_]c_int{ 1, 1, 1, 1 };
    var np = mlx.mlx_array_new();
    defer free(np);
    try mlx.check(mlx.mlx_slice(&np, keys, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ b, 1, t, h * nope }, 4, &ones, 4, s));
    var np4 = mlx.mlx_array_new();
    defer free(np4);
    try mlx.check(mlx.mlx_reshape(&np4, np, &[_]c_int{ b, t, h, nope }, 4, s));
    var np_t = mlx.mlx_array_new();
    defer free(np_t);
    try mlx.check(mlx.mlx_transpose_axes(&np_t, np4, &perm, 4, s));
    var rp = mlx.mlx_array_new();
    defer free(rp);
    try mlx.check(mlx.mlx_slice(&rp, keys, &[_]c_int{ 0, 0, 0, h * nope }, 4, &[_]c_int{ b, 1, t, ks[3] }, 4, &ones, 4, s));
    var rp_b = mlx.mlx_array_new();
    defer free(rp_b);
    try mlx.check(mlx.mlx_broadcast_to(&rp_b, rp, &[_]c_int{ b, h, t, rope }, 4, s));
    const parts = [_]mlx.mlx_array{ np_t, rp_b };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(k_out, vec, 3, s));
    var v4 = mlx.mlx_array_new();
    defer free(v4);
    try mlx.check(mlx.mlx_reshape(&v4, values, &[_]c_int{ b, t, h, vd }, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(v_out, v4, &perm, 4, s));
}

/// The groups of `seq_len` rows over `t` keys, each its one-row plan's kernels (nope 0: the dense layout).
fn attendRows(out: *mlx.mlx_array, queries: mlx.mlx_array, keys: mlx.mlx_array, values: mlx.mlx_array, scale: f32, seq_len: c_int, s: mlx.mlx_stream, h: c_int, t: c_int, qk: c_int, vd: c_int, nope: c_int) !void {
    const devc = deviceClass();
    const benv = env.int(.MLX_SDPA_BLOCKS, u32, 0);
    const n_rows: usize = @intCast(seq_len);
    var plans: [max_rows]Plan = undefined;
    for (plans[0..n_rows], 0..) |*pl, r| pl.* = planRow(devc, 1, @intCast(t - seq_len + @as(c_int, @intCast(r)) + 1), benv);
    const sv = [_]f32{scale};
    const sc = mlx.mlx_array_new_data(&sv, &[_]c_int{1}, 1, .float32);
    defer free(sc);
    var parts: [max_rows]mlx.mlx_array = undefined;
    var n_parts: usize = 0;
    defer for (parts[0..n_parts]) |a| free(a);
    var r0: usize = 0;
    while (r0 < n_rows) {
        var r1 = r0 + 1;
        while (r1 < n_rows and plans[r1].eql(plans[r0])) r1 += 1;
        const n0: c_int = t - seq_len + @as(c_int, @intCast(r0)) + 1;
        parts[n_parts] = try launchGroupAt(s, queries, keys, values, sc, h, qk, vd, @intCast(r0), @intCast(r1 - r0), n0, plans[r0], nope);
        n_parts += 1;
        r0 = r1;
    }
    if (n_parts == 1) {
        try mlx.check(mlx.mlx_array_set(out, parts[0]));
    } else {
        const vec = mlx.mlx_vector_array_new_data(&parts, n_parts);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, 2, s));
    }
    launches += n_parts;
    if (!engaged) {
        engaged = true;
        log.info("[sdpa-rows] engaged: {d} rows in {d} group(s), device class {c}\n", .{ seq_len, n_parts, if (devc == 0) '?' else devc });
    }
}

// ── tests ──

const ops = @import("../../../mlx/ops.zig");
const rowByRowSdpa = @import("sdpa.zig").rowByRowSdpa;

test "sdpa rows: MLX's one-row plan (pass, blocks) on each device class, and its override" {
    const t = std.testing;
    // A large device ('s', 'd'): one-pass below 1024 keys, two-pass from 1024.
    try t.expect(planRow('s', 1, 1023, 0).eql(.{ .kind = .one_pass }));
    try t.expect(planRow('s', 1, 1024, 0).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('s', 1, 100000, 0).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('d', 1, 8192, 0).eql(.{ .kind = .two_pass, .blocks = 128 }));
    try t.expect(planRow('d', 1, 8193, 0).eql(.{ .kind = .two_pass, .blocks = 256 }));
    // Another device: one-pass at gqa 1 whatever the keys.
    try t.expect(planRow('g', 1, 100000, 0).eql(.{ .kind = .one_pass }));
    try t.expect(planRow('g', 8, 4096, 0).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('s', 8, 2048, 0).eql(.{ .kind = .two_pass, .blocks = 128 }));
    // MLX_SDPA_BLOCKS, rounded up to 32.
    try t.expect(planRow('s', 1, 4096, 40).eql(.{ .kind = .two_pass, .blocks = 64 }));
    try t.expect(planRow('s', 1, 100, 40).eql(.{ .kind = .one_pass }));
    try t.expect(admits(192, 128) and admits(128, 128) and !admits(192, 192) and !admits(128, 64));
}

/// normal(0, 1) in `dt` of `shape`, evaluated.
fn normal(g: *ops.Graph, seed: u64, sh: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const f = try g.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, @as(mlx.mlx_dtype, .float32), @as(f32, 0), @as(f32, 1), key });
    const out = try g.astype(f, dt);
    try ops.eval(out);
    return out;
}

/// `attend` against rowByRowSdpa, bit for bit: h heads of (qk, v), `rows` rows, the first over n0 keys; the keys and
/// values the first T rows of a cache `extra` rows longer (the cache's views).
fn expectRows(h: c_int, qk: c_int, vd: c_int, rows: c_int, n0: c_int, extra: c_int, dt: mlx.mlx_dtype, seed: u64) !void {
    var g = ops.Graph.init(std.testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const t = n0 + rows - 1;
    const q = try normal(&g, seed, &.{ 1, h, rows, qk }, dt);
    const k = try g.slice(try normal(&g, seed + 1, &.{ 1, h, t + extra, qk }, dt), &.{ 0, 0, 0, 0 }, &.{ 1, h, t, qk });
    const v = try g.slice(try normal(&g, seed + 2, &.{ 1, h, t + extra, vd }, dt), &.{ 0, 0, 0, 0 }, &.{ 1, h, t, vd });
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    var want = try g.adopt(mlx.mlx_array_new());
    try rowByRowSdpa(&want, q, k, v, scale, rows, g.s);
    var got = try g.adopt(mlx.mlx_array_new());
    try std.testing.expect(try attend(&got, q, k, v, scale, rows, g.s));
    try ops.expectBitEqual(got, want);
}

// Bar: each row's words are its one-row MLX SDPA call's, short and long contexts, rows straddling the one-pass /
// two-pass threshold (1024 keys) and the blocks threshold (8192), the two-pass tails (each N mod 4), views and whole
// buffers, Ling's (192, 128) and (128, 128), bf16 and f16.
test "sdpa rows: n rows in one launch per plan are n one-row MLX SDPA calls, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var seed: u64 = 0x5EED_0A11;
    for ([_]c_int{ 100, 1017, 1024, 4093, 8190 }) |n0| {
        try expectRows(4, 192, 128, 8, n0, 37, .bfloat16, seed);
        seed += 3;
    }
    for ([_]c_int{ 1, 2, 3, 5, 16 }) |rows| {
        for ([_]c_int{ 100, 1022 }) |n0| {
            try expectRows(4, 192, 128, rows, n0, 37, .bfloat16, seed);
            seed += 3;
        }
    }
    try expectRows(4, 192, 128, 3, 2000, 0, .bfloat16, seed);
    try expectRows(4, 192, 128, 4, 1022, 37, .float16, seed + 3);
    try expectRows(4, 128, 128, 5, 1021, 37, .bfloat16, seed + 6);
}

/// A rope-once cache of `extra` rows past T: keys [1, 1, T, h * nope + rope] and values [1, 1, T, h * vd], as views.
fn rope1Cache(g: *ops.Graph, h: c_int, nope: c_int, rope: c_int, vd: c_int, t: c_int, extra: c_int, dt: mlx.mlx_dtype, seed: u64) ![2]mlx.mlx_array {
    const kw = h * nope + rope;
    const k = try g.slice(try normal(g, seed, &.{ 1, 1, t + extra, kw }, dt), &.{ 0, 0, 0, 0 }, &.{ 1, 1, t, kw });
    const v = try g.slice(try normal(g, seed + 1, &.{ 1, 1, t + extra, h * vd }, dt), &.{ 0, 0, 0, 0 }, &.{ 1, 1, t, h * vd });
    return .{ k, v };
}

/// `attendRope1` against rowByRowSdpa over the same cache's dense keys and values, bit for bit.
fn expectRope1Rows(h: c_int, rows: c_int, n0: c_int, extra: c_int, dt: mlx.mlx_dtype, seed: u64) !void {
    var g = ops.Graph.init(std.testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const t = n0 + rows - 1;
    const q = try normal(&g, seed, &.{ 1, h, rows, 192 }, dt);
    const kv = try rope1Cache(&g, h, 128, 64, 128, t, extra, dt, seed + 1);
    var dk = try g.adopt(mlx.mlx_array_new());
    var dv = try g.adopt(mlx.mlx_array_new());
    try denseRope1(kv[0], kv[1], h, 128, g.s, &dk, &dv);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var want = try g.adopt(mlx.mlx_array_new());
    try rowByRowSdpa(&want, q, dk, dv, scale, rows, g.s);
    var got = try g.adopt(mlx.mlx_array_new());
    try std.testing.expect(try attendRope1(&got, q, kv[0], kv[1], 128, scale, rows, g.s));
    try ops.expectBitEqual(got, want);
}

// Bar: over a rope-once cache each row's words are its one-row MLX SDPA call's over the dense keys (nope | rope), at
// one row (decode) and verify widths, across the one-pass / two-pass and blocks thresholds, bf16 and f16.
test "sdpa rows: a rope-once cache's rows are n one-row MLX SDPA calls over its dense keys, bit for bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var seed: u64 = 0x5EED_0B12;
    for ([_]c_int{ 1, 100, 1023, 1024, 4093, 8190, 18380 }) |n0| {
        try expectRope1Rows(4, 1, n0, 37, .bfloat16, seed);
        try expectRope1Rows(4, 4, n0, 37, .bfloat16, seed + 2);
        seed += 4;
    }
    for ([_]c_int{ 2, 3, 5, 16 }) |rows| {
        try expectRope1Rows(4, rows, 1022, 37, .bfloat16, seed);
        seed += 2;
    }
    try expectRope1Rows(64, 3, 2000, 0, .bfloat16, seed);
    try expectRope1Rows(4, 4, 1022, 37, .float16, seed + 2);
}

// Bar: a rope-once cache the kernels do not serve (other widths, a batch, past 16 rows) is declined before any launch.
test "sdpa rows: a rope-once cache of other widths, a batch or past 16 rows is declined" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = ops.Graph.init(std.testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const q = try normal(&g, 21, &.{ 1, 4, 3, 192 }, .bfloat16);
    const kv = try rope1Cache(&g, 4, 128, 64, 128, 10, 0, .bfloat16, 22);
    const before = launches;
    var out = try g.adopt(mlx.mlx_array_new());
    try std.testing.expect(!try attendRope1(&out, q, kv[0], kv[1], 96, 0.1, 3, g.s));
    const v64 = try rope1Cache(&g, 4, 128, 64, 64, 10, 0, .bfloat16, 23);
    try std.testing.expect(!try attendRope1(&out, q, kv[0], v64[1], 128, 0.1, 3, g.s));
    try std.testing.expect(!try attendRope1(&out, try normal(&g, 24, &.{ 2, 4, 3, 192 }, .bfloat16), kv[0], kv[1], 128, 0.1, 3, g.s));
    const q17 = try normal(&g, 25, &.{ 1, 4, 17, 192 }, .bfloat16);
    const kv20 = try rope1Cache(&g, 4, 128, 64, 128, 20, 0, .bfloat16, 26);
    try std.testing.expect(!try attendRope1(&out, q17, kv20[0], kv20[1], 128, 0.1, 17, g.s));
    try std.testing.expectEqual(before, launches);
}

// Bar: what it does not serve is declined before any launch.
test "sdpa rows: GQA, other head dims and dtypes, and past 16 rows are declined" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var g = ops.Graph.init(std.testing.allocator, mlx.gpuStream());
    defer g.deinit();
    const q = try normal(&g, 7, &.{ 1, 4, 3, 192 }, .bfloat16);
    const k = try normal(&g, 8, &.{ 1, 4, 10, 192 }, .bfloat16);
    const v = try normal(&g, 9, &.{ 1, 4, 10, 128 }, .bfloat16);
    const before = launches;
    var out = try g.adopt(mlx.mlx_array_new());
    try std.testing.expect(!try attend(&out, try normal(&g, 10, &.{ 1, 4, 3, 192 }, .float32), k, v, 0.1, 3, g.s));
    try std.testing.expect(!try attend(&out, q, try normal(&g, 11, &.{ 1, 2, 10, 192 }, .bfloat16), try normal(&g, 12, &.{ 1, 2, 10, 128 }, .bfloat16), 0.1, 3, g.s));
    try std.testing.expect(!try attend(&out, q, k, try normal(&g, 13, &.{ 1, 4, 10, 192 }, .bfloat16), 0.1, 3, g.s));
    try std.testing.expect(!try attend(&out, q, k, v, 0.1, 4, g.s));
    const q17 = try normal(&g, 14, &.{ 1, 4, 17, 192 }, .bfloat16);
    const k17 = try normal(&g, 15, &.{ 1, 4, 20, 192 }, .bfloat16);
    const v17 = try normal(&g, 16, &.{ 1, 4, 20, 128 }, .bfloat16);
    try std.testing.expect(!try attend(&out, q17, k17, v17, 0.1, 17, g.s));
    try std.testing.expectEqual(before, launches);
}
