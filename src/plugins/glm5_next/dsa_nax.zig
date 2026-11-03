//! DeepSeek sparse attention on the tensor units (M5-class NAX): the indexer's head-summed
//! scores and the latent attention over each query's selected rows, as MLX custom kernels.
//! GLM-5-Next (`glm5_next.zig`) and plugins (through `mlx_host`) call them; the caller checks the
//! device has NAX (`transformer.verifyQmmNaxAvailable`), the kernels check the shapes.

const std = @import("std");
const kernelSource = @import("../../mlx/kernels.zig").source;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");

/// oMLX `patches/glm_moe_dsa/indexer_nax.py` (Apache-2.0, see NOTICE): the indexer's
/// head-summed `relu(q . k) * w` per (query, pool) on the tensor units, the causal pool mask
/// folded into the epilogue. Here the scores leave in f32 (`O`), as the reference ranks them.
const INDEXER_NAX_SOURCE = kernelSource("plugins/glm5_next/kernels/dsa_indexer_nax.metal");
const INDEXER_WM: c_int = 2;
const INDEXER_WN: c_int = 2;
var indexer_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;
var indexer_nax_engaged = false;

/// Masked scores f32 [1, S, P] of queries at positions `q_start..` (`q_idx` [1, S, H, 128],
/// `w` [1, S, H] scaled) against pooled keys `pk` [1, P, 128] (`pool_len` of them complete,
/// `kpool` tokens each); a pool a query cannot see scores -1e30. Null off the GPU or outside
/// the shape; the caller checks the device has NAX.
pub fn indexerScores(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, kpool: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const qs = mlx.getShape(q_idx);
    const ps = mlx.getShape(pk);
    if (qs.len != 4 or qs[0] != 1 or qs[3] != 128 or ps.len != 3 or ps[0] != 1 or ps[2] != 128) return null;
    if (mlx.mlx_array_dtype(q_idx) != .bfloat16 or mlx.mlx_array_dtype(pk) != .bfloat16) return null;
    const sq = qs[1];
    const h = qs[2];
    const p = ps[1];
    if (sq < 1 or p < 1) return null;
    const kern = indexer_nax_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "k", "w", "params" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("msv_glm5_indexer_nax", iv, ov, INDEXER_NAX_SOURCE, SPARSE_NAX_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        indexer_nax_kernel = k;
        break :blk k;
    };
    const pdata = [6]i32{ sq, p, q_start, p, kpool, h };
    const params = mlx.mlx_array_new_data(&pdata, &[_]c_int{6}, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const tg_x = @divTrunc(p + 32 * INDEXER_WN - 1, 32 * INDEXER_WN);
    const tg_y = @divTrunc(sq + 32 * INDEXER_WM - 1, 32 * INDEXER_WM);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, sq, p }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg_x * INDEXER_WM * INDEXER_WN * 32, tg_y, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, INDEXER_WM * INDEXER_WN * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "O", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WM", INDEXER_WM));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WN", INDEXER_WN));
    const inputs = [_]mlx.mlx_array{ q_idx, pk, w, params };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    if (!indexer_nax_engaged) {
        indexer_nax_engaged = true;
        log.info("[dsa] NAX indexer scores engaged\n", .{});
    }
    return out;
}

/// oMLX `patches/glm_moe_dsa/sparse_mla_nax.py` (Apache-2.0, see NOTICE): the sparse
/// latent attention on the tensor units. One threadgroup per (query, 32-head half), 8
/// simdgroups; per 128-slot tile, scores read key rows through the indices (never gathered),
/// online exp2 softmax in f32, P as fp16 hi + lo pieces (MSV_DSA_TWO_P; the roped form one
/// piece) against bf16 values, f32 accumulate.
const SPARSE_NAX_HEADER = kernelSource("plugins/glm5_next/kernels/dsa_sparse_nax_header.metal");
const SPARSE_NAX_SOURCE = kernelSource("plugins/glm5_next/kernels/dsa_sparse_nax.metal");

var sparse_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sparse_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sparse_engaged = false;
var sparse_rope_engaged = false;

/// `sparseLatentAttention` (glm5_next) on the tensor units, for queries at the LAST rows of
/// `latent`: `q_abs` [1, H, L, 512], `latent` [1, 1, K, 512], `idx` int32 [1, L, W]. An unused
/// slot is -1 and unused pools sort last, so whole tiles skip. Null off the GPU or outside the
/// shape; the caller checks the device has NAX.
pub fn sparseLatent(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    return sparse(s, q_abs, null, latent, null, idx, scale);
}

/// `sparseLatent` with a roped part (DeepSeek-V3.2's MLA, GLM-5.3): each score adds the
/// query's roped part against the selected row's roped key, `q_lat . kv + q_pe . k_pe`, the
/// values the latent rows. `q_pe` [1, H, L, 64], `k_pe` [1, 1, K, 64]. P meets the values as
/// one fp16 piece.
pub fn sparseLatentRope(s: mlx.mlx_stream, q_lat: mlx.mlx_array, q_pe: mlx.mlx_array, latent: mlx.mlx_array, k_pe: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    return sparse(s, q_lat, q_pe, latent, k_pe, idx, scale);
}

fn sparse(s: mlx.mlx_stream, q_abs: mlx.mlx_array, q_pe: ?mlx.mlx_array, latent: mlx.mlx_array, k_pe: ?mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const qs = mlx.getShape(q_abs);
    const ls = mlx.getShape(latent);
    const is = mlx.getShape(idx);
    if (qs.len != 4 or ls.len != 4 or is.len != 3) return null;
    if (qs[0] != 1 or ls[0] != 1 or ls[1] != 1 or is[0] != 1) return null;
    if (qs[3] != 512 or ls[3] != 512 or @rem(qs[1], 32) != 0) return null;
    const dt = mlx.mlx_array_dtype(q_abs);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(latent) != dt or mlx.mlx_array_dtype(idx) != .int32) return null;
    const h = qs[1];
    const l = qs[2];
    const k = ls[2];
    const w = is[2];
    if (l < 1 or k < l or is[1] != l or w < 1) return null;
    const rope = q_pe != null;
    if (rope) {
        const ps = mlx.getShape(q_pe.?);
        const ks = mlx.getShape(k_pe.?);
        if (ps.len != 4 or ps[0] != 1 or ps[1] != h or ps[2] != l or ps[3] != 64) return null;
        if (ks.len != 4 or ks[0] != 1 or ks[1] != 1 or ks[2] != k or ks[3] != 64) return null;
        if (mlx.mlx_array_dtype(q_pe.?) != dt or mlx.mlx_array_dtype(k_pe.?) != dt) return null;
    }
    const slot = if (rope) &sparse_rope_kernel else &sparse_kernel;
    const kern = slot.* orelse blk: {
        const ins_plain = [_][*:0]const u8{ "q", "kv", "idx", "params", "scale" };
        const ins_rope = [_][*:0]const u8{ "q", "qpe", "kv", "pe", "idx", "params", "scale" };
        const ins: []const [*:0]const u8 = if (rope) &ins_rope else &ins_plain;
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kn = if (rope)
            mlx.mlx_fast_metal_kernel_new("msv_dsa_sparse_mla_rope_nax", iv, ov, "#define MSV_DSA_ROPE\n" ++ SPARSE_NAX_SOURCE, SPARSE_NAX_HEADER, true, false)
        else
            mlx.mlx_fast_metal_kernel_new("msv_glm5_sparse_mla_nax", iv, ov, "#define MSV_DSA_TWO_P\n" ++ SPARSE_NAX_SOURCE, SPARSE_NAX_HEADER, true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = kn;
        break :blk kn;
    };
    var arrays: [7]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (arrays[0..n]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    // The kernel's inputs in its order, each as the rows it indexes.
    const shapes = [5][3]c_int{ .{ h, l, 512 }, .{ h, l, 64 }, .{ k, 512, 0 }, .{ k, 64, 0 }, .{ l, w, 0 } };
    const ranks = [5]usize{ 3, 3, 2, 2, 2 };
    for ([5]?mlx.mlx_array{ q_abs, q_pe, latent, k_pe, idx }, shapes, ranks) |x_, sh, r| if (x_) |x| {
        arrays[n] = mlx.mlx_array_new();
        n += 1;
        try mlx.check(mlx.mlx_reshape(&arrays[n - 1], x, &sh, r, s));
    };
    const pdata = [4]i32{ l, k, w, k - l };
    arrays[n] = mlx.mlx_array_new_data(&pdata, &[_]c_int{4}, 1, .int32);
    n += 1;
    const sv = [1]f32{scale};
    arrays[n] = mlx.mlx_array_new_data(&sv, &[_]c_int{1}, 1, .float32);
    n += 1;

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ h, l, 512 }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, l * 256, @divExact(h, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const iv = mlx.mlx_vector_array_new_data(&arrays, n);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var o3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o3);
    try mlx.check(mlx.mlx_vector_array_get(&o3, ov, 0));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o3, &[_]c_int{ 1, h, l, 512 }, 4, s));
    const engaged = if (rope) &sparse_rope_engaged else &sparse_engaged;
    if (!engaged.*) {
        engaged.* = true;
        log.info("[dsa] NAX sparse latent attention{s} engaged\n", .{if (rope) " (roped part)" else ""});
    }
    return out;
}

const testing = std.testing;

fn testArray(alloc: std.mem.Allocator, rand: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try alloc.alloc(f32, n);
    defer alloc.free(v);
    for (v) |*x| x.* = rand.floatNorm(f32);
    const f = mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn hostF32(a: mlx.mlx_array, s: mlx.mlx_stream, alloc: std.mem.Allocator) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.MlxNoData;
    return alloc.dupe(f32, p[0..mlx.mlx_array_size(c)]);
}

test "dsa sparse latent attention with a roped part equals the latent attention over the joined rows" {
    if (mlx.noGpuBackend() or !@import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    // The second shape's 4 index slots are bound as a constant array.
    for ([_][4]c_int{ .{ 64, 300, 40, 260 }, .{ 32, 5, 1, 4 } }) |dims| try checkSparseRope(dims[0], dims[1], dims[2], dims[3]);
}

fn checkSparseRope(h: c_int, k: c_int, l: c_int, w: c_int) !void {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(13);
    const rand = prng.random();
    const q = try testArray(alloc, rand, &.{ 1, h, l, 512 }, s);
    defer _ = mlx.mlx_array_free(q);
    const qp = try testArray(alloc, rand, &.{ 1, h, l, 64 }, s);
    defer _ = mlx.mlx_array_free(qp);
    const lat = try testArray(alloc, rand, &.{ 1, 1, k, 512 }, s);
    defer _ = mlx.mlx_array_free(lat);
    const kp = try testArray(alloc, rand, &.{ 1, 1, k, 64 }, s);
    defer _ = mlx.mlx_array_free(kp);
    // Each query reads a random subset of the keys at or before it, holes as -1 (some past it too).
    const iv = try alloc.alloc(i32, @intCast(l * w));
    defer alloc.free(iv);
    for (0..@intCast(l)) |qi| {
        const pos: i32 = k - l + @as(i32, @intCast(qi));
        for (0..@intCast(w)) |j| iv[qi * @as(usize, @intCast(w)) + j] = rand.intRangeAtMost(i32, -@divTrunc(pos, 4) - 1, pos + 3);
    }
    const idx = mlx.mlx_array_new_data(iv.ptr, &[_]c_int{ 1, l, w }, 3, .int32);
    defer _ = mlx.mlx_array_free(idx);
    const got = (try sparseLatentRope(s, q, qp, lat, kp, idx, 0.0625)) orelse return error.SkipZigTest;
    defer _ = mlx.mlx_array_free(got);
    // The reference: the roped part joined to each row (q | q_pe against latent | k_pe), its first 512 values.
    var qj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qj);
    var kj = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kj);
    {
        const qv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ q, qp }, 2);
        defer _ = mlx.mlx_vector_array_free(qv);
        try mlx.check(mlx.mlx_concatenate_axis(&qj, qv, -1, s));
        const kv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ lat, kp }, 2);
        defer _ = mlx.mlx_vector_array_free(kv);
        try mlx.check(mlx.mlx_concatenate_axis(&kj, kv, -1, s));
    }
    // An unused slot or one past its query is -1 to the reference (the kernel drops both).
    for (0..@intCast(l)) |qi| {
        const pos: i32 = k - l + @as(i32, @intCast(qi));
        for (iv[qi * @as(usize, @intCast(w)) ..][0..@intCast(w)]) |*x| if (x.* > pos) {
            x.* = -1;
        };
    }
    const idx_ref = mlx.mlx_array_new_data(iv.ptr, &[_]c_int{ 1, l, w }, 3, .int32);
    defer _ = mlx.mlx_array_free(idx_ref);
    const joined = try @import("glm5_next.zig").sparseLatentAttention(s, qj, kj, idx_ref, 0.0625, 64);
    defer _ = mlx.mlx_array_free(joined);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_slice(&ref, joined, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, h, l, 512 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    const a = try hostF32(got, s, alloc);
    defer alloc.free(a);
    const b = try hostF32(ref, s, alloc);
    defer alloc.free(b);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var maxd: f32 = 0;
    for (a, b) |x, y| {
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxd = @max(maxd, @abs(x - y));
    }
    const cos = dot / @sqrt(na * nb);
    std.debug.print("[dsa nax rope] {d} heads, {d} keys, {d} queries x {d} slots: cos {d:.6} max|d| {d:.4} (|ref| {d:.3})\n", .{ h, k, l, w, cos, maxd, @sqrt(nb) });
    try testing.expect(cos > 0.9999 and maxd < 0.05);
}
