//! Decode attention on the M5 tensor units whose every row has the bits of
//! the one-row step at its position, ported from TensorFold's
//! `lane_attention.py` / `stream_attention.py` (MIT, see NOTICE). Keys go in
//! 512-key chunks of 64-key tiles counted from position 0, each tile one
//! online-softmax update, the chunks merged in order. The window's rows share
//! key reads over the committed prefix's whole tiles (`PARTIAL`, 16 query rows
//! a tensor op); each node then walks the rest, its path's rows included,
//! from the first tile the prefix did not fill (`TAIL`), and `MERGE` joins the
//! chunks. A serial step is the one-node window, so a node and the step at its
//! position see the same tiles in the same order.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const row_attn = @import("row_attn.zig");

pub const MAX_ROWS = row_attn.MAX_ROWS;
pub const Tree = row_attn.Tree;
const CK: c_int = 512; // keys per chunk (part of the arithmetic)
const TK: c_int = 64; // keys per tile (part of the arithmetic)
const TILES_PER_GROUP: c_int = 16;
/// Simdgroups a TAIL threadgroup loads its gathered key/value tiles with (the first alone computes).
const TAIL_SG: c_int = 4;

pub const HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");

// The committed prefix's whole tiles [0, L), every window row seeing all of them.
pub const PARTIAL = kernelSource("kernels/attention/lane_attn_partial.metal");

// One node's keys from the first tile the prefix did not fill, its path's rows
// in depth order; chunk c0's walk continues from the state PARTIAL left there.
pub const TAIL = kernelSource("kernels/attention/lane_attn_tail.metal");

// The prefix's finished chunks, then the node's own, in order.
pub const MERGE = kernelSource("kernels/attention/lane_attn_merge.metal");

var partial_kernel: ?mlx.mlx_fast_metal_kernel = null;
var tail_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;
var placeholder: mlx.mlx_array = .{ .ctx = null };

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, row_contig: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, HEADER, row_contig, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// bf16 q/k/v, head dim 256 with rows of 8-element alignment, at most 16 query
/// heads a key head, 1..MAX_ROWS query rows.
pub fn fits(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array) bool {
    for ([_]mlx.mlx_array{ q, k, v }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    if (qs.len != 4 or ks.len != 4 or qs[0] != 1 or ks[0] != 1) return false;
    if (qs[3] != 256 or ks[3] != 256 or @rem(qs[1], ks[1]) != 0) return false;
    const g = @divExact(qs[1], ks[1]);
    return g <= 16 and qs[2] >= 1 and qs[2] <= MAX_ROWS and ks[2] >= qs[2];
}

fn addDims(cfg: mlx.mlx_fast_metal_kernel_config, g: c_int, d: c_int, extra_name: ?[*:0]const u8, extra: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", g));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CK", CK));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TK", TK));
    if (extra_name) |n| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, extra));
}

var chain_cache: [MAX_ROWS + 1]?Tree = @splat(null);

/// A chain's window depths and paths (depth r, path 0..r), built once per width.
fn chainTree(w: c_int) Tree {
    const wu: usize = @intCast(w);
    if (chain_cache[wu]) |t| return t;
    var chain: [MAX_ROWS * MAX_ROWS]i32 = undefined;
    var depth: [MAX_ROWS]i32 = undefined;
    for (0..wu) |r| {
        depth[r] = @intCast(r);
        for (0..wu) |i| chain[r * wu + i] = @intCast(i);
    }
    chain_cache[wu] = Tree{
        .depth = mlx.mlx_array_new_data(&depth, &[_]c_int{w}, 1, .int32),
        .path = mlx.mlx_array_new_data(&chain, &[_]c_int{ w, w }, 2, .int32),
        .max_depth = w - 1,
    };
    return chain_cache[wu].?;
}

/// q [1, H, W, D] of a window whose rows sit at the last W positions of k/v
/// [1, HKV, L, D] (the cache views after this step's update) -> [1, H, W, D].
/// `tree` null: a chain. False outside the kernels.
pub fn sdpa(out: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, tree: ?Tree, s: mlx.mlx_stream) !bool {
    if (!mlx.streamIsGpu(s) or !fits(q, k, v)) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const h = qs[1];
    const w = qs[2];
    const d = qs[3];
    const hkv = ks[1];
    const g = @divExact(h, hkv);
    const p = ks[2] - w; // committed keys
    const pt = @divTrunc(p, TK) * TK; // the prefix's whole tiles
    const ca = @divTrunc(pt + CK - 1, CK);

    const t = tree orelse chainTree(w);
    const maxd = t.max_depth + 1;
    const ncb = @divTrunc(p + t.max_depth, CK) - @divTrunc(pt, CK) + 1;
    const r_rows = g * w;
    const sga = @divTrunc(r_rows + 15, 16);
    const rp = 16 * sga;
    const sg = @min(sga, TILES_PER_GROUP);

    const pk = try kernel(&partial_kernel, "msv_lane_attn_partial", &.{ "Qp", "K", "V", "scale", "dims" }, &.{ "PO", "PM", "PL" }, PARTIAL, false);
    const tk = try kernel(&tail_kernel, "msv_lane_attn_tail", &.{ "QB", "K", "V", "scale", "dims", "paths", "depths", "POA", "PMA", "PLA" }, &.{ "PO", "PM", "PL" }, TAIL, false);
    const mk = try kernel(&merge_kernel, "msv_lane_attn_merge", &.{ "POA", "PMA", "PLA", "POB", "PMB", "PLB", "dims" }, &.{"OUT"}, MERGE, true);

    const scale_arr = mlx.mlx_array_new_data(&scale, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scale_arr);
    if (placeholder.ctx == null) {
        const z: [16]f32 = @splat(0);
        placeholder = mlx.mlx_array_new_data(&z, &[_]c_int{16}, 1, .float32);
    }

    // q [1, H, W, D] read in place: key head hk's rows g * W + node sit together.
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    try mlx.check(mlx.mlx_contiguous(&qc, q, false, s));

    var poa: mlx.mlx_array = placeholder;
    var pma: mlx.mlx_array = placeholder;
    var pla: mlx.mlx_array = placeholder;
    var a_outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(a_outs);
    var a_arrays: [3]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    defer for (a_arrays) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    if (ca > 0) {
        const dims_a = [_]i32{ pt, ca, w, sga };
        const da = mlx.mlx_array_new_data(&dims_a, &[_]c_int{4}, 1, .int32);
        defer _ = mlx.mlx_array_free(da);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{hkv * ca * rp * d}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{hkv * ca * rp}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{hkv * ca * rp}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, hkv * 32 * sg, ca, @divTrunc(sga + sg - 1, sg)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * sg, 1, 1));
        try addDims(cfg, g, d, "SG", sg);
        const ins = [_]mlx.mlx_array{ qc, k, v, scale_arr, da };
        const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&a_outs, pk, in_vec, cfg, s));
        for (&a_arrays, 0..) |*a, i| {
            a.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(a, a_outs, i));
        }
        poa = a_arrays[0];
        pma = a_arrays[1];
        pla = a_arrays[2];
    }

    const dims_b = [_]i32{ p, pt, w, ncb, @max(ca, 1), rp };
    const db = mlx.mlx_array_new_data(&dims_b, &[_]c_int{6}, 1, .int32);
    defer _ = mlx.mlx_array_free(db);
    var paths = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(paths);
    try mlx.check(mlx.mlx_reshape(&paths, t.path, &[_]c_int{w * maxd}, 1, s));
    const bcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(bcfg);
    const nb = hkv * ncb * w * 16;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(bcfg, &[_]c_int{nb * d}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(bcfg, &[_]c_int{nb}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(bcfg, &[_]c_int{nb}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(bcfg, hkv * 32 * TAIL_SG, ncb, w));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(bcfg, 32 * TAIL_SG, 1, 1));
    try addDims(bcfg, g, d, "MAXD", maxd);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(bcfg, "TSG", TAIL_SG));
    const bins = [_]mlx.mlx_array{ qc, k, v, scale_arr, db, paths, t.depth, poa, pma, pla };
    const bin_vec = mlx.mlx_vector_array_new_data(&bins, bins.len);
    defer _ = mlx.mlx_vector_array_free(bin_vec);
    var b_outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(b_outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&b_outs, tk, bin_vec, bcfg, s));
    var bo: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (bo) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&bo, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, b_outs, i));

    const mcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mcfg, &[_]c_int{ 1, h, w, d }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mcfg, hkv * 32, w * g, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mcfg, 32, 1, 1));
    try addDims(mcfg, g, d, null, 0);
    const mins = [_]mlx.mlx_array{ poa, pma, pla, bo[0], bo[1], bo[2], db };
    const mvec = mlx.mlx_vector_array_new_data(&mins, mins.len);
    defer _ = mlx.mlx_vector_array_free(mvec);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&res, mk, mvec, mcfg, s));
    try mlx.check(mlx.mlx_vector_array_get(out, res, 0));
    return true;
}

// ── tests ──

const testing = std.testing;

fn randBf16(shape: []const c_int, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn slice4(a: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var start = [_]c_int{ 0, 0, 0, 0 };
    var stop = [_]c_int{ sh[0], sh[1], sh[2], sh[3] };
    start[axis] = lo;
    stop[axis] = hi;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &start, 4, &stop, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    return out;
}

fn expectSame(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

fn maxAbsDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a32, b32, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var mx_ = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx_);
    try mlx.check(mlx.mlx_max(&mx_, ad, false, s));
    var worst: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&worst, mx_));
    return worst;
}

test "lane_attn: every window row equals the one-row step at its position, and matches MLX's sdpa" {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // Qwen3.8-27B: 24 query heads over 4 kv heads, head dim 256; cache views
    // longer than the keys. A short prefix (one partial tile), one where the
    // later rows' steps take a tile the window's TAIL runs into PARTIAL, and one
    // past two chunks that ends mid-tile.
    const H: c_int = 24;
    const HKV: c_int = 4;
    const D: c_int = 256;
    const CAP: c_int = 1536;
    for ([_]c_int{ 40, 300, 330, 1100 }) |L| {
        const W: c_int = 16;
        const q = try randBf16(&.{ 1, H, W, D }, 1, s);
        defer _ = mlx.mlx_array_free(q);
        const kbuf = try randBf16(&.{ 1, HKV, CAP, D }, 2, s);
        defer _ = mlx.mlx_array_free(kbuf);
        const vbuf = try randBf16(&.{ 1, HKV, CAP, D }, 3, s);
        defer _ = mlx.mlx_array_free(vbuf);
        const k = try slice4(kbuf, 2, 0, L, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try slice4(vbuf, 2, 0, L, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / 16.0;
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        try testing.expect(try sdpa(&all, q, k, v, scale, null, s));
        var r: c_int = 0;
        while (r < W) : (r += 1) {
            const keys = L - W + r + 1;
            const qr = try slice4(q, 2, r, r + 1, s);
            defer _ = mlx.mlx_array_free(qr);
            const kr = try slice4(k, 2, 0, keys, s);
            defer _ = mlx.mlx_array_free(kr);
            const vr = try slice4(v, 2, 0, keys, s);
            defer _ = mlx.mlx_array_free(vr);
            var one = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(one);
            try testing.expect(try sdpa(&one, qr, kr, vr, scale, null, s));
            const want = try slice4(all, 2, r, r + 1, s);
            defer _ = mlx.mlx_array_free(want);
            try expectSame(one, want, s);
            const none = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(none);
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, qr, kr, vr, scale, "", none, .{ .ctx = null }, false, s));
            try testing.expect(try maxAbsDiff(one, ref, s) < 2e-2);
        }
    }
}

test "lane_attn: a tree node equals the one-row step over its own path's keys" {
    if (!@import("../../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const H: c_int = 24;
    const HKV: c_int = 4;
    const D: c_int = 256;
    const W: c_int = 6;
    // rows: 0 root, 1 and 2 children of 0, 3 child of 1, 4 child of 2, 5 child of 3
    const parents = [_]i32{ -1, 0, 0, 1, 2, 3 };
    var depth: [W]i32 = undefined;
    const MAXD: c_int = 4;
    var path: [W * MAXD]i32 = @splat(0);
    for (0..W) |r| {
        depth[r] = if (parents[r] < 0) 0 else depth[@intCast(parents[r])] + 1;
        var cur: i32 = @intCast(r);
        var d = depth[r];
        while (cur >= 0) : (d -= 1) {
            path[r * MAXD + @as(usize, @intCast(d))] = cur;
            cur = parents[@intCast(cur)];
        }
    }
    for ([_]c_int{ 150, 1021 }) |P| {
        const q = try randBf16(&.{ 1, H, W, D }, 11, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try randBf16(&.{ 1, HKV, P + W, D }, 12, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try randBf16(&.{ 1, HKV, P + W, D }, 13, s);
        defer _ = mlx.mlx_array_free(v);
        const tree = Tree{
            .depth = mlx.mlx_array_new_data(&depth, &[_]c_int{W}, 1, .int32),
            .path = mlx.mlx_array_new_data(&path, &[_]c_int{ W, MAXD }, 2, .int32),
            .max_depth = MAXD - 1,
        };
        defer _ = mlx.mlx_array_free(tree.depth);
        defer _ = mlx.mlx_array_free(tree.path);
        var all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(all);
        try testing.expect(try sdpa(&all, q, k, v, 1.0 / 16.0, tree, s));
        for (0..W) |r| {
            // The node's serial step: committed keys, then its path's rows in depth order.
            const du: usize = @intCast(depth[r]);
            var idx: [1021 + MAXD]i32 = undefined;
            const pu: usize = @intCast(P);
            for (0..pu) |i| idx[i] = @intCast(i);
            for (0..du + 1) |i| idx[pu + i] = P + path[r * MAXD + i];
            const ia = mlx.mlx_array_new_data(&idx, &[_]c_int{@intCast(pu + du + 1)}, 1, .int32);
            defer _ = mlx.mlx_array_free(ia);
            var kr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kr);
            try mlx.check(mlx.mlx_take_axis(&kr, k, ia, 2, s));
            var vr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vr);
            try mlx.check(mlx.mlx_take_axis(&vr, v, ia, 2, s));
            const ri: c_int = @intCast(r);
            const qr = try slice4(q, 2, ri, ri + 1, s);
            defer _ = mlx.mlx_array_free(qr);
            var one = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(one);
            try testing.expect(try sdpa(&one, qr, kr, vr, 1.0 / 16.0, null, s));
            const want = try slice4(all, 2, ri, ri + 1, s);
            defer _ = mlx.mlx_array_free(want);
            try expectSame(one, want, s);
        }
    }
}
