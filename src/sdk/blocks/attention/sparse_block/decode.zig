//! QSA attention for ONE decode token over dense bf16 KV: the indexer's sorted block picks go
//! straight to a split pass and a merge pass, with no mask, gather or SDPA between them.
//!
//! Split pass: a simdgroup owns one q head and `CH` of the visible keys, eight keys at a time with
//! the head dim spread over the lanes. The eight partial dots reduce in one butterfly that leaves
//! key `lane >> 2` on lanes `4k..4k+3` (9 shuffles, not 8 x 5). More heads per simdgroup lose to
//! register pressure.
const std = @import("std");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");

const CH: c_int = 32; // visible keys per simdgroup, a multiple of 8 and of RATIO
const BD: c_int = 256;

pub const HEADER = kernelSource("kernels/attention/qsa_decode_header.metal");

pub const PARTS_SOURCE = kernelSource("kernels/attention/qsa_decode_parts.metal");

// One threadgroup a head. An empty split writes l = 0, the skip flag.
pub const MERGE_SOURCE = kernelSource("kernels/attention/qsa_decode_merge.metal");

var parts_kernel: ?mlx.mlx_fast_metal_kernel = null;
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

const Key = struct { hq: c_int, hk: c_int, kb: c_int, ratio: c_int };
const Cfgs = struct { key: Key, parts: mlx.mlx_fast_metal_kernel_config, merge: mlx.mlx_fast_metal_kernel_config };
var decode_cfgs: ?Cfgs = null;

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // K/V are cache views: the kernel reads their strides.
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn configsFor(key: Key, ns: c_int) !Cfgs {
    if (decode_cfgs) |c| {
        if (std.meta.eql(c.key, key)) return c;
        _ = mlx.mlx_fast_metal_kernel_config_free(c.parts);
        _ = mlx.mlx_fast_metal_kernel_config_free(c.merge);
        decode_cfgs = null;
    }
    const gqa = @divExact(key.hq, key.hk);
    const parts = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, BD }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(parts, &[_]c_int{ key.hq, ns, 2 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(parts, ns * gqa * 32, key.hk, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(parts, gqa * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(parts, "T", .bfloat16));
    inline for (.{ .{ "GQA", gqa }, .{ "KB", key.kb }, .{ "RATIO", key.ratio }, .{ "CH", CH }, .{ "NS", ns } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(parts, kv[0], kv[1]));
    }
    const merge = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge, &[_]c_int{ 1, key.hq, 1, BD }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge, BD * key.hq, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge, BD, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(merge, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge, "NS", ns));
    decode_cfgs = .{ .key = key, .parts = parts, .merge = merge };
    return decode_cfgs.?;
}

/// `[1, Hq, 1, 256]` bf16 attention of one query row `q` over the keys `blocks` `[1, 1, kb]` int32
/// selects (RATIO tokens each) plus the incomplete tail; `k`/`v` are the dense cache views
/// `[1, Hk, kv, 256]`. Null when the geometry is outside the kernel's set (the caller keeps its arm).
pub fn attend(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, blocks: mlx.mlx_array, ratio: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or q.ctx == null or k.ctx == null or v.ctx == null or blocks.ctx == null) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks) != .int32) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    if (!std.mem.eql(c_int, ks, mlx.getShape(v))) return null;
    if (qs[0] != 1 or qs[2] != 1 or qs[3] != BD or ks[0] != 1 or ks[3] != BD or bs[0] != 1 or bs[1] != 1) return null;
    const hq = qs[1];
    const hk = ks[1];
    // An input under 8 elements binds in `constant` space, where the kernel's `device int*` fails to compile.
    if (hk <= 0 or ratio <= 0 or bs[2] < 8 or @rem(hq, hk) != 0 or @divTrunc(hq, hk) > 32 or @rem(CH, ratio) != 0) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;

    if (parts_kernel == null) {
        parts_kernel = try newKernel("msv_qsa_dec_parts", &.{ "q", "scl", "blocks", "k", "v" }, &.{ "pacc", "pml" }, PARTS_SOURCE);
        merge_kernel = try newKernel("msv_qsa_dec_merge", &.{ "pacc", "pml" }, &.{"out"}, MERGE_SOURCE);
    }
    const ns = @divTrunc(bs[2] * ratio + ratio - 1 + CH - 1, CH);
    const c = try configsFor(.{ .hq = hq, .hk = hk, .kb = bs[2], .ratio = ratio }, ns);

    const scl_data = [_]f32{attn_scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const ins = [_]mlx.mlx_array{ q, scl, blocks, k, v };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, parts_kernel.?, in_vec, c.parts, s));
    if (mlx.mlx_vector_array_size(parts) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts, 1));

    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, merge_kernel.?, merge_vec, c.merge, s));
    if (mlx.mlx_vector_array_size(o) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, o, 0));
    return out;
}

const testing = std.testing;

fn randomBf16(s: mlx.mlx_stream, shape: []const c_int, seed: u64) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 1.0, key, s));
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&b, f, .bfloat16, s));
    return b;
}

fn hostF32(s: mlx.mlx_stream, a: mlx.mlx_array) !struct { arr: mlx.mlx_array, data: []const f32 } {
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const p = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    return .{ .arr = f, .data = p[0..mlx.mlx_array_size(f)] };
}

test "qsa decode: the one-token kernel is no worse than the masked SDPA against the f64 truth" {
    const mask_mod = @import("mask.zig");
    const s = mlx.gpuStream();
    const hq: usize = 24;
    const hk: usize = 2;
    const gqa = hq / hk;
    const kb: usize = 512;
    const ratio: usize = 4;
    const cap: usize = 12288;
    const scale: f32 = 1.0 / 16.0;
    var prng = std.Random.DefaultPrng.init(0xDEC0DE);
    const rnd = prng.random();
    const kvs = [_]usize{ 2052, 2055, 4099, 8076, 8103, 11000 };
    for (kvs, 0..) |kv, case| {
        const q = try randomBf16(s, &[_]c_int{ 1, @intCast(hq), 1, BD }, 100 + case);
        defer _ = mlx.mlx_array_free(q);
        const kfull = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 200 + case);
        defer _ = mlx.mlx_array_free(kfull);
        const vfull = try randomBf16(s, &[_]c_int{ 1, @intCast(hk), @intCast(cap), BD }, 300 + case);
        defer _ = mlx.mlx_array_free(vfull);
        // The cache views a decode step sees: a leading slice of a larger buffer.
        var kview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kview);
        var vview = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vview);
        const lo = [_]c_int{ 0, 0, 0, 0 };
        const hi = [_]c_int{ 1, @intCast(hk), @intCast(kv), BD };
        const st = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&kview, kfull, &lo, 4, &hi, 4, &st, 4, s));
        try mlx.check(mlx.mlx_slice(&vview, vfull, &lo, 4, &hi, 4, &st, 4, s));

        const nb = kv / ratio;
        const pool = try testing.allocator.alloc(i32, nb);
        defer testing.allocator.free(pool);
        for (pool, 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(i32, pool);
        const sel = try testing.allocator.dupe(i32, pool[0..kb]);
        defer testing.allocator.free(sel);
        std.mem.sort(i32, sel, {}, std.sort.asc(i32));
        const blocks = mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ 1, 1, @intCast(kb) }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);

        // f64 truth over the same bf16 values.
        const qh = try hostF32(s, q);
        defer _ = mlx.mlx_array_free(qh.arr);
        const kh = try hostF32(s, kview);
        defer _ = mlx.mlx_array_free(kh.arr);
        const vh = try hostF32(s, vview);
        defer _ = mlx.mlx_array_free(vh.arr);
        var pos_list: std.ArrayList(usize) = .empty;
        defer pos_list.deinit(testing.allocator);
        for (sel) |b| for (0..ratio) |r| try pos_list.append(testing.allocator, @as(usize, @intCast(b)) * ratio + r);
        var t = nb * ratio;
        while (t < kv) : (t += 1) try pos_list.append(testing.allocator, t);
        const truth = try testing.allocator.alloc(f64, hq * BD);
        defer testing.allocator.free(truth);
        const sc = try testing.allocator.alloc(f64, pos_list.items.len);
        defer testing.allocator.free(sc);
        for (0..hq) |h| {
            const g = h / gqa;
            var mx: f64 = -std.math.inf(f64);
            for (pos_list.items, 0..) |pos, i| {
                var dot: f64 = 0;
                for (0..BD) |d| dot += @as(f64, qh.data[h * BD + d]) * @as(f64, kh.data[(g * kv + pos) * BD + d]);
                sc[i] = dot * scale;
                mx = @max(mx, sc[i]);
            }
            var z: f64 = 0;
            for (sc) |*x| {
                x.* = @exp(x.* - mx);
                z += x.*;
            }
            for (0..BD) |d| {
                var o: f64 = 0;
                for (pos_list.items, 0..) |pos, i| o += sc[i] * @as(f64, vh.data[(g * kv + pos) * BD + d]);
                truth[h * BD + d] = o / z;
            }
        }

        const got = (try attend(s, q, kview, vview, blocks, @intCast(ratio), scale)) orelse return error.KernelDeclined;
        defer _ = mlx.mlx_array_free(got);
        const mask = try mask_mod.qsaMaskFromBlocks(s, blocks, @intCast(kv), @intCast(ratio));
        defer _ = mlx.mlx_array_free(mask);
        const stock = try mask_mod.qsaMaskArm(s, q, kview, vview, scale, mask);
        defer _ = mlx.mlx_array_free(stock);
        const gh = try hostF32(s, got);
        defer _ = mlx.mlx_array_free(gh.arr);
        const sh = try hostF32(s, stock);
        defer _ = mlx.mlx_array_free(sh.arr);
        try testing.expectEqual(hq * BD, gh.data.len);
        var e_got: f64 = 0;
        var e_stock: f64 = 0;
        var n_truth: f64 = 0;
        for (truth, 0..) |tv, i| {
            try testing.expect(std.math.isFinite(gh.data[i]));
            e_got += (@as(f64, gh.data[i]) - tv) * (@as(f64, gh.data[i]) - tv);
            e_stock += (@as(f64, sh.data[i]) - tv) * (@as(f64, sh.data[i]) - tv);
            n_truth += tv * tv;
        }
        const rms_got = @sqrt(e_got / @as(f64, @floatFromInt(truth.len)));
        const rms_stock = @sqrt(e_stock / @as(f64, @floatFromInt(truth.len)));
        const rms_truth = @sqrt(n_truth / @as(f64, @floatFromInt(truth.len)));
        std.debug.print("[qsa-dec parity] kv={d} rms_err kernel={e:.3} stock={e:.3} (truth rms {e:.3})\n", .{ kv, rms_got, rms_stock, rms_truth });
        try testing.expect(rms_got <= rms_stock * 1.15);
        try testing.expect(rms_got < rms_truth * 0.02);
    }
}
