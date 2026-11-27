//! The fp4 MoE decode launches (`moe_fp4.zig`) in every fp4 layout: MXFP4 (group 32), NVFP4 (group 16) and NVFP4 with
//! ModelOpt's per-expert scales, at DeepSeek-V4.1-Flash's and GLM-5.3 Pro's MoE widths. The bars: no further from an f32
//! model than the plugin's composed path on random weights (`decode.expectNoWorse`); bit for bit the composed path where
//! every sum is exact; and the fp4 decode rules (OCP MX v1.0 e2m1 / e8m0, OFP8 E4M3FN) code by code.

const std = @import("std");
const testing = std.testing;
const mlx = @import("../../../mlx/ffi.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const moe_up = @import("moe_up.zig");
const moe_fp4 = @import("moe_fp4.zig");
const fp4 = @import("../../../sdk/quant/fp4.zig");
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;
const none: A = .{ .ctx = null };
const own = moe_up.own;
const reshape = moe_up.reshape;
const astype = moe_up.astype;
const mul = moe_up.mul;

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dims(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

/// The layouts a routed bank arrives in.
const Fmt = enum {
    mxfp4,
    nvfp4,
    nvfp4_scaled,

    fn format(f: Fmt) fp4.Format {
        return if (f == .mxfp4) .mxfp4 else .nvfp4;
    }

    fn qp(f: Fmt) QuantParams {
        return .{ .bits = 4, .group_size = f.format().group(), .mode = f.format().mode() };
    }
};
const fmts = [_]Fmt{ .mxfp4, .nvfp4, .nvfp4_scaled };
const mxfp8: QuantParams = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 };

/// A MoE's widths: hidden K, expert width N, picks, hc streams; fewer experts than the model's (the kernels read only the
/// picks' rows).
const Dims = struct { name: []const u8, k: usize, n: usize, topk: usize, hc: usize = 4, experts: usize = 10 };
/// DeepSeek-V4.1-Flash: hidden 5120, moe_intermediate 2304, 6 of 384 experts, hc 4, swiglu_limit 10.
const deepseek: Dims = .{ .name = "DeepSeek-V4.1-Flash", .k = 5120, .n = 2304, .topk = 6 };
/// GLM-5.3 Pro: hidden 6144, inter 2048, 8 of 256 experts. It has no hc streams: 4, as the plugin's expand takes them.
const glm: Dims = .{ .name = "GLM-5.3 Pro", .k = 6144, .n = 2048, .topk = 8 };

/// DeepSeek's `swiglu_limit`.
const lim_ds: f32 = 10;

/// A model handle for the gates: they read only the activation dtype.
fn bf16Model() dsv41.Dsv41Model {
    var m: dsv41.Dsv41Model = undefined;
    m.act = .bfloat16;
    return m;
}

const Gen = struct {
    seed: u64,
    n: u64 = 0,

    fn normal(g: *Gen, sc: *Scope, shape: []const c_int, std_: f32) !A {
        var key = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&key, g.seed *% 0x9E3779B97F4A7C15 +% g.n));
        _ = try sc.keep(key);
        g.n += 1;
        var o = mlx.mlx_array_new();
        return own(sc, mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, std_, key, sc.s), o);
    }
};

/// `w` (f32) quantized to `qp` as MLX packs it: words and scales.
fn quant(sc: *Scope, w: A, qp: QuantParams) ![2]A {
    var v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_quantize(&v, try astype(sc, w, .bfloat16), mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), qp.mode.cstr(), none, sc.s));
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&q, v, 0));
    _ = try sc.keep(q);
    var s = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&s, v, 1));
    _ = try sc.keep(s);
    return .{ q, s };
}

/// `[E]` bf16 from host values.
fn bf16Vec(sc: *Scope, v: []const f32) !A {
    return astype(sc, try sc.keep(mlx.mlx_array_new_data(v.ptr, &[_]c_int{ci(v.len)}, 1, .float32)), .bfloat16);
}

/// A layer at `d` in `f`: its routed banks (gate and up fused) and the mxfp8 shared expert, random weights at
/// 1 / sqrt(fan-in). `.nvfp4_scaled` draws each expert's gate, up and down scale near 2^-3 and quantizes the weights
/// divided by it, as ModelOpt's global scale leaves them; `es` holds them.
const Layer = struct {
    ly: dsv41.Layer,
    es: moe_fp4.Scales,
    /// The scales as host values (1 where none).
    hs: [3][]f32,

    fn deinit(l: *Layer) void {
        for (l.hs) |h| testing.allocator.free(h);
    }
};

fn randomLayer(sc: *Scope, g: *Gen, d: Dims, f: Fmt) !Layer {
    const E = d.experts;
    var hs: [3][]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(g.seed);
    for (&hs) |*h| {
        h.* = try testing.allocator.alloc(f32, E);
        for (h.*) |*v| v.* = if (f == .nvfp4_scaled) @floatCast(kc.host.rt(.bfloat16, 0.125 * (0.75 + 0.5 * prng.random().float(f64)))) else 1;
    }
    const kf = @sqrt(@as(f32, @floatFromInt(d.k)));
    const nf = @sqrt(@as(f32, @floatFromInt(d.n)));
    var gate = try g.normal(sc, &.{ ci(E), ci(d.n), ci(d.k) }, 1 / kf);
    var up = try g.normal(sc, &.{ ci(E), ci(d.n), ci(d.k) }, 1 / kf);
    var down = try g.normal(sc, &.{ ci(E), ci(d.k), ci(d.n) }, 1 / nf);
    var es: moe_fp4.Scales = .{};
    if (f == .nvfp4_scaled) {
        const v = [3]A{ try bf16Vec(sc, hs[0]), try bf16Vec(sc, hs[1]), try bf16Vec(sc, hs[2]) };
        es = .{ .gate = v[0], .up = v[1], .down = v[2] };
        inline for (.{ &gate, &up, &down }, 0..) |w, i| w.* = try sc.div(w.*, try reshape(sc, try astype(sc, v[i], .float32), &.{ ci(E), 1, 1 }));
    }
    const gu = try quant(sc, try sc.concat(&.{ gate, up }, 1), f.qp());
    const dn = try quant(sc, down, f.qp());
    const sgu = try quant(sc, try g.normal(sc, &.{ ci(2 * d.n), ci(d.k) }, 1 / kf), mxfp8);
    const sdn = try quant(sc, try g.normal(sc, &.{ ci(d.k), ci(d.n) }, 1 / nf), mxfp8);
    return .{ .ly = layerOf(d, f.qp(), gu, dn, sgu, sdn), .es = es, .hs = hs };
}

fn layerOf(d: Dims, qp: QuantParams, gu: [2]A, dn: [2]A, sgu: [2]A, sdn: [2]A) dsv41.Layer {
    var ly: dsv41.Layer = undefined;
    ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = dn[0], .s = dn[1], .qp = qp }, .gu = .{ .w = gu[0], .s = gu[1], .qp = qp } };
    ly.shared_gu = .{ .w = sgu[0], .s = sgu[1], .qp = mxfp8 };
    ly.shared[2] = .{ .w = sdn[0], .s = sdn[1], .qp = mxfp8 };
    ly.top_k = d.topk;
    return ly;
}

// ── the composed path: the plugin's own ops, with the scales as the SDK MoE applies them ──

/// The non-fused gate/up half: `gather_qmm` (gate and up rounded to bf16), each expert's gate / up scale on its rounded
/// output (bf16), the clamped SwiGLU in f32 times the routing weight (f32, the down scale on it), rounded to bf16; then
/// the shared expert's quantized matmul and SwiGLU: `[R, k + 1, N]`.
fn composedUp(sc: *Scope, ly: *const dsv41.Layer, h2: A, idx: A, wts: A, es: moe_fp4.Scales, lim: f32) !A {
    const sgu = ly.shared_gu.?;
    const rows = dims(h2, 0);
    const topk = dims(idx, 1);
    const k = dims(h2, 1);
    const n = dims(ly.experts.gu.?.w, 1) / 2;
    const per: []const c_int = &.{ ci(rows), ci(topk), 1, 1 };
    var gt, var up = try dsv41.gateUp(sc, &ly.experts, try reshape(sc, h2, &.{ ci(rows), 1, 1, ci(k) }), none, idx, false);
    if (es.gate) |v| gt = try mul(sc, gt, try reshape(sc, try sc.take(v, idx, 0), per));
    if (es.up) |v| up = try mul(sc, up, try reshape(sc, try sc.take(v, idx, 0), per));
    var w = try reshape(sc, wts, per);
    if (es.down) |v| w = try mul(sc, w, try reshape(sc, try sc.take(v, idx, 0), per));
    const routed = try astype(sc, try mul(sc, try dsv41.clampedSwiglu(sc, gt, up, lim), w), .bfloat16);
    var y = mlx.mlx_array_new();
    const sy = try own(sc, mlx.mlx_quantized_matmul(&y, h2, sgu.w, sgu.s, none, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", sc.s), y);
    const sh = try astype(sc, try dsv41.clampedSwiglu(sc, try moe_up.sliceLast(sc, sy, 0, n), try moe_up.sliceLast(sc, sy, n, 2 * n), lim), .bfloat16);
    return moe_up.concat(sc, &.{ try reshape(sc, routed, &.{ ci(rows), ci(topk), ci(n) }), try reshape(sc, sh, &.{ ci(rows), 1, ci(n) }) }, 1);
}

/// The non-fused down half: the routed gather over the weighted activations (bf16 a slot), summed in f32, plus the
/// shared down, rounded to bf16 (`addShared`), then `hcExpand`.
fn composedDown(sc: *Scope, ly: *const dsv41.Layer, act: A, idx: A, x1: A, sp: *const dsv41.Split) !A {
    const rows = dims(act, 0);
    const topk = dims(idx, 1);
    const k = dims(act, 2);
    const d = dims(x1, 2);
    const routed_act = try reshape(sc, try sc.slice(act, 1, 0, topk), &.{ ci(rows), ci(topk), 1, ci(k) });
    const dn = try dsv41.gatherMm(sc, routed_act, &ly.experts.down, none, idx, false);
    const routed_sum = try sc.sum(try astype(sc, try reshape(sc, dn, &.{ ci(rows), ci(topk), ci(d) }), .float32), 1, false);
    const shared_act = try reshape(sc, try sc.slice(act, 1, topk, topk + 1), &.{ ci(rows), ci(k) });
    const sy = try ly.shared[2].apply(sc, shared_act);
    const y = try astype(sc, try sc.add(routed_sum, try astype(sc, sy, .float32)), .bfloat16);
    return dsv41.hcExpand(sc, x1, y, sp);
}

// ── an f32 model from the dequantized banks ──

fn dequantF32(sc: *Scope, w: A, s: A, qp: QuantParams) !A {
    var d = mlx.mlx_array_new();
    return own(sc, mlx.mlx_dequantize(&d, w, s, none, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), qp.mode.cstr(), none, .{ .value = .float32, .has_value = true }, sc.s), d);
}

/// The gate/up activations in float64 from the dequantized banks and the host scales: `[R, k + 1, N]`.
fn truthUp(sc: *Scope, l: *const Layer, h2: A, ids: []const u32, wts: []const f32, lim: f64) ![]f32 {
    const gu = l.ly.experts.gu.?;
    const sgu = l.ly.shared_gu.?;
    const rows = dims(h2, 0);
    const topk = ids.len / rows;
    const n = dims(gu.w, 1) / 2;
    const x32 = try sc.transpose(try astype(sc, h2, .float32), &.{ 1, 0 });
    const yb = try dsv41.hostF32(try sc.matmul(try dequantF32(sc, gu.w, gu.s, gu.qp), x32)); // [E, 2N, R]
    defer testing.allocator.free(yb);
    const yh = try dsv41.hostF32(try sc.matmul(try dequantF32(sc, sgu.w, sgu.s, sgu.qp), x32)); // [2N, R]
    defer testing.allocator.free(yh);
    const out = try testing.allocator.alloc(f32, rows * (topk + 1) * n);
    for (0..rows) |row| for (0..topk + 1) |j| for (0..n) |c| {
        const g: f64, const u: f64, const wt: f64 = if (j < topk) blk: {
            const e: usize = ids[row * topk + j];
            break :blk .{ yb[(e * 2 * n + c) * rows + row] * l.hs[0][e], yb[(e * 2 * n + n + c) * rows + row] * l.hs[1][e], wts[row * topk + j] * l.hs[2][e] };
        } else .{ yh[c * rows + row], yh[(n + c) * rows + row], 1 };
        const gc = @min(g, lim);
        const uc = std.math.clamp(u, -lim, lim);
        out[(row * (topk + 1) + j) * n + c] = @floatCast(gc / (1 + @exp(-gc)) * uc * wt);
    };
    return out;
}

/// The new stream the down half makes, in float64 from the dequantized banks: `[R, hc, D]`.
fn truthDown(sc: *Scope, ly: *const dsv41.Layer, act: A, ids: []const u32, x1: A, sp: *const dsv41.Split) ![]f32 {
    const gpa = testing.allocator;
    const rows = dims(act, 0);
    const topk = dims(act, 1) - 1;
    const k = dims(act, 2);
    const n = dims(x1, 2);
    const hc = dims(x1, 1);
    const a = try dsv41.hostF32(act);
    defer gpa.free(a);
    const xs = try dsv41.hostF32(x1);
    defer gpa.free(xs);
    const post = try dsv41.hostF32(sp.post);
    defer gpa.free(post);
    const comb = try dsv41.hostF32(sp.comb);
    defer gpa.free(comb);
    const bank = try dsv41.hostF32(try dequantF32(sc, ly.experts.down.w, ly.experts.down.s, ly.experts.down.qp)); // [E, D, K]
    defer gpa.free(bank);
    const sbank = try dsv41.hostF32(try dequantF32(sc, ly.shared[2].w, ly.shared[2].s, ly.shared[2].qp)); // [D, K]
    defer gpa.free(sbank);
    const out = try gpa.alloc(f32, rows * hc * n);
    const y = try gpa.alloc(f64, n);
    defer gpa.free(y);
    for (0..rows) |r| {
        for (y, 0..) |*v, c| {
            var s: f64 = 0;
            for (0..topk + 1) |j| {
                const w = if (j < topk) bank[(@as(usize, ids[r * topk + j]) * n + c) * k ..][0..k] else sbank[c * k ..][0..k];
                const x = a[(r * (topk + 1) + j) * k ..][0..k];
                for (w, x) |wv, xv| s += @as(f64, wv) * xv;
            }
            v.* = s;
        }
        for (0..hc) |h| for (0..n) |c| {
            var acc: f64 = 0;
            for (0..hc) |j| acc += @as(f64, comb[(r * hc + j) * hc + h]) * xs[(r * hc + j) * n + c];
            out[(r * hc + h) * n + c] = @floatCast(@as(f64, post[r * hc + h]) * y[c] + acc);
        };
    }
    return out;
}

fn uniform(sc: *Scope, g: *Gen, shape: []const c_int, lo: f32, hi: f32) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, g.seed *% 31 +% g.n));
    _ = try sc.keep(key);
    g.n += 1;
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_random_uniform(&o, try sc.f(lo), try sc.f(hi), shape.ptr, shape.len, .float32, key, sc.s), o);
}

/// Every layout at `d`, rows 1, 3 and 6: both launches through their gates (`moeUpScaled`, `moeDown`) against the composed
/// path and the f32 model.
fn matrix(d: Dims) !void {
    const gpa = testing.allocator;
    const m = bf16Model();
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    for (fmts, 0..) |f, fi| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        var g: Gen = .{ .seed = 0xF04 + fi };
        var l = try randomLayer(&sc, &g, d, f);
        defer l.deinit();
        var prng = std.Random.DefaultPrng.init(0xD5 + fi);
        for ([_]usize{ 1, 3, decode.max_rows }) |rows| {
            var ids: [64]u32 = undefined;
            var ws: [64]f32 = undefined;
            moe_up.route(prng.random(), .random, rows, d.topk, @intCast(d.experts), &ids, &ws);
            const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .uint32));
            const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .float32));
            const h2 = try astype(&sc, try g.normal(&sc, &.{ ci(rows), ci(d.k) }, 1), .bfloat16);
            var label: [96]u8 = undefined;
            {
                const got = (try moe_fp4.moeUpScaled(&sc, &m, kern, &l.ly, h2, idx, wts, l.es)) orelse return error.Declined;
                const want = try composedUp(&sc, &l.ly, h2, idx, wts, l.es, lim_ds);
                const tr = try truthUp(&sc, &l, h2, ids[0 .. rows * d.topk], ws[0 .. rows * d.topk], lim_ds);
                defer gpa.free(tr);
                const gh = try dsv41.hostF32(got);
                defer gpa.free(gh);
                const wh = try dsv41.hostF32(want);
                defer gpa.free(wh);
                try decode.expectNoWorse(try std.fmt.bufPrint(&label, "moe_fp4 up {s} {s} R={d}", .{ d.name, @tagName(f), rows }), gh, wh, tr);
            }
            {
                const act = try astype(&sc, try g.normal(&sc, &.{ ci(rows), ci(d.topk + 1), ci(d.n) }, 0.5), .bfloat16);
                const x1 = try astype(&sc, try g.normal(&sc, &.{ ci(rows), ci(d.hc), ci(d.k) }, 1), .bfloat16);
                const sp: dsv41.Split = .{ .pre = none, .post = try uniform(&sc, &g, &.{ ci(rows), ci(d.hc) }, 0.05, 1.5), .comb = try uniform(&sc, &g, &.{ ci(rows), ci(d.hc), ci(d.hc) }, -0.5, 0.5) };
                const got = (try moe_fp4.moeDown(&sc, kern, &l.ly, act, idx, x1, &sp)) orelse return error.Declined;
                const want = try composedDown(&sc, &l.ly, act, idx, x1, &sp);
                const tr = try truthDown(&sc, &l.ly, act, ids[0 .. rows * d.topk], x1, &sp);
                defer gpa.free(tr);
                const gh = try dsv41.hostF32(got);
                defer gpa.free(gh);
                const wh = try dsv41.hostF32(want);
                defer gpa.free(wh);
                try decode.expectNoWorse(try std.fmt.bufPrint(&label, "moe_fp4 down {s} {s} R={d}", .{ d.name, @tagName(f), rows }), gh, wh, tr);
            }
        }
    }
}

test "dsv41 dec moe_fp4 matrix: mxfp4, nvfp4, nvfp4 with per-expert scales at DeepSeek-V4.1-Flash's widths, no further from f32 than the composed path" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try matrix(deepseek);
}

test "dsv41 dec moe_fp4 matrix: mxfp4, nvfp4, nvfp4 with per-expert scales at GLM-5.3 Pro's widths, no further from f32 than the composed path" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try matrix(glm);
}

/// Both launches at `d` under `MLX_SERVE_DSV41_FP4_READS` = `fr`, every layout, rows 1, 3 and 6, against the composed
/// path and the f32 model (`expectNoWorse`); with every row's picks rotated one expert, the fused side fails the bar.
fn readsAgree(d: Dims, fr: [:0]const u8) !void {
    const gpa = testing.allocator;
    const sw = env.override(.MLX_SERVE_DSV41_FP4_READS, fr.ptr);
    defer sw.restore();
    const m = bf16Model();
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    for (fmts, 0..) |f, fi| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        var g: Gen = .{ .seed = 0xF0A + fi };
        var l = try randomLayer(&sc, &g, d, f);
        defer l.deinit();
        var prng = std.Random.DefaultPrng.init(0xDA + fi);
        for ([_]usize{ 1, 3, decode.max_rows }) |rows| {
            var ids: [64]u32 = undefined;
            var ws: [64]f32 = undefined;
            moe_up.route(prng.random(), .random, rows, d.topk, @intCast(d.experts), &ids, &ws);
            var off: [64]u32 = undefined;
            for (off[0 .. rows * d.topk], ids[0 .. rows * d.topk]) |*o, v| o.* = (v + 1) % @as(u32, @intCast(d.experts));
            const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .uint32));
            const bad = try sc.keep(mlx.mlx_array_new_data(&off, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .uint32));
            const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .float32));
            const h2 = try astype(&sc, try g.normal(&sc, &.{ ci(rows), ci(d.k) }, 1), .bfloat16);
            var label: [128]u8 = undefined;
            {
                const want = try composedUp(&sc, &l.ly, h2, idx, wts, l.es, lim_ds);
                const tr = try truthUp(&sc, &l, h2, ids[0 .. rows * d.topk], ws[0 .. rows * d.topk], lim_ds);
                defer gpa.free(tr);
                const wh = try dsv41.hostF32(want);
                defer gpa.free(wh);
                for ([_]A{ idx, bad }, 0..) |use, k| {
                    const got = (try moe_fp4.moeUpScaled(&sc, &m, kern, &l.ly, h2, use, wts, l.es)) orelse return error.Declined;
                    const gh = try dsv41.hostF32(got);
                    defer gpa.free(gh);
                    const lab = try std.fmt.bufPrint(&label, "moe_fp4 up FP4_READS={s} {s} R={d}{s}", .{ fr, @tagName(f), rows, if (k == 1) " (picks off by one, planted)" else "" });
                    if (k == 0) try decode.expectNoWorse(lab, gh, wh, tr) else try testing.expectError(error.FurtherFromTruth, decode.expectNoWorse(lab, gh, wh, tr));
                }
            }
            {
                const act = try astype(&sc, try g.normal(&sc, &.{ ci(rows), ci(d.topk + 1), ci(d.n) }, 0.5), .bfloat16);
                const x1 = try astype(&sc, try g.normal(&sc, &.{ ci(rows), ci(d.hc), ci(d.k) }, 1), .bfloat16);
                const sp: dsv41.Split = .{ .pre = none, .post = try uniform(&sc, &g, &.{ ci(rows), ci(d.hc) }, 0.05, 1.5), .comb = try uniform(&sc, &g, &.{ ci(rows), ci(d.hc), ci(d.hc) }, -0.5, 0.5) };
                const want = try composedDown(&sc, &l.ly, act, idx, x1, &sp);
                const tr = try truthDown(&sc, &l.ly, act, ids[0 .. rows * d.topk], x1, &sp);
                defer gpa.free(tr);
                const wh = try dsv41.hostF32(want);
                defer gpa.free(wh);
                for ([_]A{ idx, bad }, 0..) |use, k| {
                    const got = (try moe_fp4.moeDown(&sc, kern, &l.ly, act, use, x1, &sp)) orelse return error.Declined;
                    const gh = try dsv41.hostF32(got);
                    defer gpa.free(gh);
                    const lab = try std.fmt.bufPrint(&label, "moe_fp4 down FP4_READS={s} {s} R={d}{s}", .{ fr, @tagName(f), rows, if (k == 1) " (picks off by one, planted)" else "" });
                    if (k == 0) try decode.expectNoWorse(lab, gh, wh, tr) else try testing.expectError(error.FurtherFromTruth, decode.expectNoWorse(lab, gh, wh, tr));
                }
            }
        }
    }
}

test "dsv41 dec moe_fp4 FP4_READS: off (MLX's reads, an equal-scale NVFP4 span in MXFP4's order) and on (bit-placement reads, each 16-group apart) are both no further from f32 than the composed path, every layout at DeepSeek-V4.1-Flash's widths, R = 1, 3, 6; picks off by one expert fail" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try readsAgree(deepseek, "0");
    try readsAgree(deepseek, "1");
}

// ── banks from host codes ──

/// e2m1 codes by value: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (bit 3 the sign).
const e2m1 = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// An fp4 bank `[lead, K]` from host nibbles (`codes`, one a weight) and group-scale bytes.
fn fp4Bank(sc: *Scope, codes: []const u8, scales: []const u8, lead: []const usize, K: usize, gs: usize) ![2]A {
    const words = try testing.allocator.alloc(u32, codes.len / 8);
    defer testing.allocator.free(words);
    for (words, 0..) |*w, i| {
        w.* = 0;
        for (0..8) |b| w.* |= @as(u32, codes[8 * i + b]) << @intCast(4 * b);
    }
    return fp4Words(sc, words, scales, lead, K, gs);
}

/// Weight `i`'s nibble of packed `words` set to `code`.
fn setCode(words: []u32, i: usize, code: u8) void {
    const sh: u5 = @intCast(4 * (i % 8));
    words[i / 8] = (words[i / 8] & ~(@as(u32, 15) << sh)) | (@as(u32, code) << sh);
}

/// An fp4 bank `[lead, K]` from packed words (eight codes a word, the first in the low nibble) and group-scale bytes.
fn fp4Words(sc: *Scope, words: []const u32, scales: []const u8, lead: []const usize, K: usize, gs: usize) ![2]A {
    var wd: [3]c_int = undefined;
    var sd: [3]c_int = undefined;
    for (lead, 0..) |v, i| {
        wd[i] = ci(v);
        sd[i] = ci(v);
    }
    wd[lead.len] = ci(K / 8);
    sd[lead.len] = ci(K / gs);
    return .{
        try sc.keep(mlx.mlx_array_new_data(words.ptr, &wd, @intCast(lead.len + 1), .uint32)),
        try sc.keep(mlx.mlx_array_new_data(scales.ptr, &sd, @intCast(lead.len + 1), .uint8)),
    };
}

/// An mxfp8 linear `[rows, K]` from host e4m3 bytes and e8m0 scale bytes.
fn mxfp8Bank(sc: *Scope, bytes: []const u8, scales: []const u8, rows: usize, K: usize) ![2]A {
    return .{
        try sc.keep(mlx.mlx_array_new_data(bytes.ptr, &[_]c_int{ ci(rows), ci(K / 4) }, 2, .uint32)),
        try sc.keep(mlx.mlx_array_new_data(scales.ptr, &[_]c_int{ ci(rows), ci(K / 32) }, 2, .uint8)),
    };
}

fn bf16Of(sc: *Scope, v: []const f32, shape: []const c_int) !A {
    return astype(sc, try sc.keep(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32)), .bfloat16);
}

/// A power of two as the format's scale byte: e8m0 (2^e = 127 + e) or e4m3 (exponent field e + 7, no mantissa).
fn pow2Code(f: fp4.Format, e: i32) u8 {
    return @intCast(if (f == .mxfp4) 127 + e else (e + 7) << 3);
}

/// Integer-valued e4m3 codes 0..8.
const int_e4m3 = [_]u8{ 0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54, 0x58 };

/// Every sum exact in f32 in any order (a few small dyadic terms a row): the launches equal the composed path bit for
/// bit, which leaves only the epilogues (roundings, per-expert scales, clamp, SwiGLU, routing weight, combine) to differ.
fn exact(d: Dims) !void {
    const gpa = testing.allocator;
    const m = bf16Model();
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    const E = d.experts;
    for (fmts, 0..) |f, fi| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        var prng = std.Random.DefaultPrng.init(0xE7AC + fi);
        const rng = prng.random();
        const ff = f.format();
        const gs = ff.group();
        // Up: x integers in [-3, 3]; each routed row 8 nonzero codes under scales 2^-2..2^1 (e4m3 with mantissas).
        const words = try gpa.alloc(u32, E * 2 * d.n * d.k / 8);
        defer gpa.free(words);
        const sbytes = try gpa.alloc(u8, E * 2 * d.n * d.k / gs);
        defer gpa.free(sbytes);
        @memset(words, 0);
        for (0..E * 2 * d.n) |r| for (0..8) |_| setCode(words, r * d.k + rng.uintLessThan(usize, d.k), 1 + rng.uintLessThan(u8, 15));
        for (sbytes) |*b| b.* = if (ff == .mxfp4) pow2Code(ff, rng.intRangeAtMost(i32, -2, 1)) else kc.host.e4m3Code(rng, -2, 2, false);
        const gu = try fp4Words(&sc, words, sbytes, &.{ E, 2 * d.n }, d.k, gs);
        const sbytes8 = try gpa.alloc(u8, 2 * d.n * d.k);
        defer gpa.free(sbytes8);
        @memset(sbytes8, 0);
        for (0..2 * d.n) |r| for (0..8) |_| {
            sbytes8[r * d.k + rng.uintLessThan(usize, d.k)] = int_e4m3[1 + rng.uintLessThan(usize, 8)] | (if (rng.boolean()) @as(u8, 0x80) else 0);
        };
        const s8s = try gpa.alloc(u8, 2 * d.n * d.k / 32);
        defer gpa.free(s8s);
        for (s8s) |*b| b.* = pow2Code(.mxfp4, rng.intRangeAtMost(i32, -2, 1));
        const sgu = try mxfp8Bank(&sc, sbytes8, s8s, 2 * d.n, d.k);
        // Down: 4 nonzero codes a row under one power-of-two scale a row, so each slot's sum is a bf16 value.
        @memset(words, 0);
        for (0..E * d.k) |r| for (0..4) |_| setCode(words, r * d.n + rng.uintLessThan(usize, d.n), 1 + rng.uintLessThan(u8, 15));
        for (0..E * d.k) |r| @memset(sbytes[r * (d.n / gs) ..][0 .. d.n / gs], pow2Code(ff, rng.intRangeAtMost(i32, -3, 2)));
        const dn = try fp4Words(&sc, words[0 .. E * d.k * d.n / 8], sbytes[0 .. E * d.k * d.n / gs], &.{ E, d.k }, d.n, gs);
        @memset(sbytes8[0 .. d.k * d.n], 0);
        for (0..d.k) |r| for (0..4) |_| {
            sbytes8[r * d.n + rng.uintLessThan(usize, d.n)] = int_e4m3[1 + rng.uintLessThan(usize, 8)] | (if (rng.boolean()) @as(u8, 0x80) else 0);
        };
        for (0..d.k) |r| @memset(s8s[r * (d.n / 32) ..][0 .. d.n / 32], pow2Code(.mxfp4, rng.intRangeAtMost(i32, -3, 2)));
        const sdn = try mxfp8Bank(&sc, sbytes8[0 .. d.k * d.n], s8s[0 .. d.k * d.n / 32], d.k, d.n);
        const ly = layerOf(d, f.qp(), gu, dn, sgu, sdn);
        var es: moe_fp4.Scales = .{};
        if (f == .nvfp4_scaled) {
            var v: [3][16]f32 = undefined;
            for (&v) |*row| for (row[0..E]) |*x| {
                x.* = @floatCast(kc.host.rt(.bfloat16, 0.125 + 1.5 * rng.float(f64)));
            };
            es = .{ .gate = try bf16Vec(&sc, v[0][0..E]), .up = try bf16Vec(&sc, v[1][0..E]), .down = try bf16Vec(&sc, v[2][0..E]) };
        }
        for ([_]usize{ 1, 3, decode.max_rows }) |rows| {
            var ids: [64]u32 = undefined;
            var ws: [64]f32 = undefined;
            moe_up.route(rng, .random, rows, d.topk, @intCast(E), &ids, &ws);
            const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .uint32));
            const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(rows), ci(d.topk) }, 2, .float32));
            const xv = try gpa.alloc(f32, rows * d.k);
            defer gpa.free(xv);
            for (xv) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
            const h2 = try bf16Of(&sc, xv, &.{ ci(rows), ci(d.k) });
            const got = (try moe_fp4.moeUpScaled(&sc, &m, kern, &ly, h2, idx, wts, es)) orelse return error.Declined;
            const want = try composedUp(&sc, &ly, h2, idx, wts, es, lim_ds);
            try expectSameBits("up", d, f, rows, got, want);
            // Down: activations in {-1, 0, 1}; the residual small integers, post and comb dyadic.
            const av = try gpa.alloc(f32, rows * (d.topk + 1) * d.n);
            defer gpa.free(av);
            for (av) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -1, 1));
            const act = try bf16Of(&sc, av, &.{ ci(rows), ci(d.topk + 1), ci(d.n) });
            const rv = try gpa.alloc(f32, rows * d.hc * d.k);
            defer gpa.free(rv);
            for (rv) |*x| x.* = @floatFromInt(rng.intRangeAtMost(i32, -4, 4));
            const x1 = try bf16Of(&sc, rv, &.{ ci(rows), ci(d.hc), ci(d.k) });
            var pv: [decode.max_rows * 32 * 33]f32 = undefined;
            for (pv[0 .. rows * d.hc * (d.hc + 1)]) |*x| x.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -4, 4))) * 0.25;
            const sp: dsv41.Split = .{
                .pre = none,
                .post = try sc.keep(mlx.mlx_array_new_data(&pv, &[_]c_int{ ci(rows), ci(d.hc) }, 2, .float32)),
                .comb = try sc.keep(mlx.mlx_array_new_data(pv[rows * d.hc ..].ptr, &[_]c_int{ ci(rows), ci(d.hc), ci(d.hc) }, 3, .float32)),
            };
            const gd = (try moe_fp4.moeDown(&sc, kern, &ly, act, idx, x1, &sp)) orelse return error.Declined;
            const wd = try composedDown(&sc, &ly, act, idx, x1, &sp);
            try expectSameBits("down", d, f, rows, gd, wd);
        }
    }
}

fn expectSameBits(what: []const u8, d: Dims, f: Fmt, rows: usize, got: A, want: A) !void {
    const g = try moe_up.bitsOf(got);
    defer testing.allocator.free(g);
    const w = try moe_up.bitsOf(want);
    defer testing.allocator.free(w);
    var diff: usize = 0;
    var first: ?usize = null;
    for (g, w, 0..) |a, b, i| if (a != b) {
        diff += 1;
        if (first == null) first = i;
    };
    if (diff > 0) {
        std.debug.print("[moe_fp4 exact] {s} {s} {s} R={d}: {d} of {d} outputs differ, first at {d}: 0x{x:0>4} vs composed 0x{x:0>4}\n", .{ what, d.name, @tagName(f), rows, diff, g.len, first.?, g[first.?], w[first.?] });
        return error.NotBitExact;
    }
}

test "dsv41 dec moe_fp4 matrix: with every sum exact, both launches equal the composed path bit for bit (mxfp4, nvfp4, nvfp4 with per-expert scales; DeepSeek-V4.1-Flash and GLM-5.3 Pro widths)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    try exact(deepseek);
    try exact(glm);
}

// ── the decode rules, code by code ──

/// e2m1 code `c` as f32 (OCP MX v1.0: all 16 finite, -0 kept).
fn e2m1Rule(c: u8) f32 {
    const v: f32 = @floatCast(e2m1[c & 7]);
    return if (c & 8 != 0) -v else v;
}

/// e8m0 byte `b` as f32: 2^(b - 127) exactly (2^-127 a subnormal), 0xFF NaN.
fn e8m0Rule(b: u8) f32 {
    if (b == 0xFF) return std.math.nan(f32);
    return @floatCast(std.math.pow(f64, 2, @as(f64, @floatFromInt(b)) - 127));
}

/// E4M3FN byte `b` as f32: 0x7F / 0xFF NaN, no infinities, subnormals exact, the sign kept on 0.
fn e4m3Rule(b: u8) f32 {
    if (b & 0x7F == 0x7F) return std.math.nan(f32);
    return @floatCast(kc.host.e4m3(b));
}

/// `v` as the GPU's float arithmetic leaves a result: a subnormal flushed to zero of its sign.
fn ftz(v: f32) f32 {
    if (v != 0 and @abs(v) < std.math.floatMin(f32)) return if (v < 0) -0.0 else 0.0;
    return v;
}

fn sameF32(a: f32, b: f32) bool {
    if (std.math.isNan(a) or std.math.isNan(b)) return std.math.isNan(a) and std.math.isNan(b);
    return @as(u32, @bitCast(a)) == @as(u32, @bitCast(b));
}

const probe_src =
    \\const uint i = thread_position_in_grid.x;
    \\const float2 p = dec_fp4_pair(i) * 16384.0f;
    \\out[4 * i] = p.x;
    \\out[4 * i + 1] = p.y;
    \\out[4 * i + 2] = mlxserve_fp4_scale_of<32>(i);
    \\out[4 * i + 3] = mlxserve_fp4_scale_of<16>(i);
;

test "dsv41 dec moe_fp4 rules: the kernel's e2m1 pairs, e8m0 and e4m3 scales decode every code as MLX's fp4.h / fp8.h read it, and MLX's dequantize is that read times the code on the GPU" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const kern = try dsv41.buildKernel("dsv41_dec_moe_fp4_probe", &.{"x"}, &.{"out"}, probe_src, moe_fp4.HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1024}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const ins = [_]A{try sc.f(0)};
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    const got = try dsv41.hostF32(try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o));
    defer gpa.free(got);
    var bad: usize = 0;
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const want = [4]f32{ e2m1Rule(b & 15), e2m1Rule(b >> 4), e8m0Rule(b), e4m3Rule(b) };
        for (want, got[4 * i ..][0..4], 0..) |w, g, j| if (!sameF32(w, g)) {
            bad += 1;
            std.debug.print("[moe_fp4 rules] byte 0x{x:0>2} {s}: got {e} (0x{x:0>8}), the rule {e}\n", .{ b, ([_][]const u8{ "e2m1 lo", "e2m1 hi", "e8m0", "e4m3" })[j], g, @as(u32, @bitCast(g)), w });
        };
    }
    try testing.expectEqual(@as(usize, 0), bad);

    // MLX's dequantize, the composed path's decode: row c of each bank holds code c (scale 2^0 / e2m1 1.0 elsewhere).
    var report: usize = 0;
    for ([_]fp4.Format{ .mxfp4, .nvfp4 }) |ff| {
        const gs = ff.group();
        // Rows 0..255: scale byte c over codes 1.0; row 256: codes 0..15 (twice for a group of 32) under scale 1.
        const codes = try gpa.alloc(u8, 257 * 32);
        defer gpa.free(codes);
        @memset(codes, 2);
        for (0..32) |j| codes[256 * 32 + j] = @intCast(j % 16);
        const sb = try gpa.alloc(u8, 257 * 32 / gs);
        defer gpa.free(sb);
        for (0..257) |r| @memset(sb[r * (32 / gs) ..][0 .. 32 / gs], if (r < 256) @as(u8, @intCast(r)) else pow2Code(ff, 0));
        const q = try fp4Bank(&sc, codes, sb, &.{257}, 32, gs);
        const deq = try dsv41.hostF32(try dequantF32(&sc, q[0], q[1], .{ .bits = 4, .group_size = gs, .mode = ff.mode() }));
        defer gpa.free(deq);
        for (0..257) |r| {
            const want: f32 = if (r < 256) (if (ff == .mxfp4) e8m0Rule(@intCast(r)) else e4m3Rule(@intCast(r))) else 0;
            for (0..32) |j| {
                // The product is a float multiply on the GPU: a subnormal one flushes to zero (sign kept).
                const w = ftz(if (r < 256) want else e2m1Rule(@intCast(j % 16)));
                const g = deq[r * 32 + j];
                if (!sameF32(w, g)) {
                    report += 1;
                    std.debug.print("[moe_fp4 rules] MLX dequantize {s} {s} 0x{x:0>2}: {e} (0x{x:0>8}), the rule {e}\n", .{ @tagName(ff), if (r < 256) "scale" else "e2m1", if (r < 256) r else j % 16, g, @as(u32, @bitCast(g)), w });
                    break;
                }
            }
        }
    }
    std.debug.print("[moe_fp4 rules] MLX dequantize: {d} codes differ from the reads\n", .{report});
    try testing.expectEqual(@as(usize, 0), report);
}

/// The hc expand's identity: post 1, comb 0, residual 0 (out = y).
fn identitySplit(sc: *Scope, rows: usize, hc: usize, n: usize) !struct { dsv41.Split, A } {
    const post = try sc.ones(&.{ ci(rows), ci(hc) }, .float32);
    const comb = try sc.zeros(&.{ ci(rows), ci(hc), ci(hc) }, .float32);
    return .{ .{ .pre = none, .post = post, .comb = comb }, try sc.zeros(&.{ ci(rows), ci(hc), ci(n) }, .bfloat16) };
}

/// f32 `v` rounded to bf16 (nearest even; NaN stays NaN), as f32.
fn bf16Round(v: f32) f32 {
    if (std.math.isNan(v)) return v;
    return @floatCast(kc.host.rt(.bfloat16, v));
}

/// The bar is MLX's composed path, column by column (NaN equals NaN); where both leave the exact value they print it.
fn checkColumns(label: []const u8, f: fp4.Format, got: []const f32, want: []const f32, mlx_got: []const f32, n: usize) !void {
    var bad: usize = 0;
    var off_exact: usize = 0;
    for (0..n) |c| {
        if (!sameF32(mlx_got[c], got[c])) {
            bad += 1;
            std.debug.print("[moe_fp4 rules] {s} {s} column {d}: kernel {e} (0x{x:0>8}), composed MLX {e} (0x{x:0>8})\n", .{ label, @tagName(f), c, got[c], @as(u32, @bitCast(got[c])), mlx_got[c], @as(u32, @bitCast(mlx_got[c])) });
        }
        if (!sameF32(want[c], mlx_got[c])) {
            off_exact += 1;
            std.debug.print("[moe_fp4 rules] composed MLX {s} {s} column {d}: {e}, the exact value {e}\n", .{ label, @tagName(f), c, mlx_got[c], want[c] });
        }
    }
    std.debug.print("[moe_fp4 rules] {s} {s}: {d} columns, kernel off MLX {d}, MLX off the exact value {d}\n", .{ label, @tagName(f), n, bad, off_exact });
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 dec moe_fp4 rules: through both launches, every scale code and every e2m1 code lands as MLX's composed path puts it, bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const m = bf16Model();
    const kern = try moe_fp4.build(std.math.inf(f32));
    defer kern.deinit();
    // Columns 0..255: the scale byte c on the code 1.0 at input 56 (unit 1, group 1 at GS 16); 256..271: e2m1 code c - 256
    // under scale 1. The other inputs are zero.
    const N = 272;
    const at = 56;
    for ([_]fp4.Format{ .mxfp4, .nvfp4 }) |ff| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const gs = ff.group();
        const want = try gpa.alloc(f32, N);
        defer gpa.free(want);
        for (want, 0..) |*w, c| {
            const v: f32 = if (c < 256) (if (ff == .mxfp4) e8m0Rule(@intCast(c)) else e4m3Rule(@intCast(c))) else e2m1Rule(@intCast(c - 256));
            // A sum starts at +0: a -0 term leaves +0.
            w.* = if (v == 0) 0 else v;
        }
        // The down launch: one row, one pick (expert 0), act 1 at input `at`; y = the column's value.
        {
            const K = 64;
            const codes = try gpa.alloc(u8, N * K);
            defer gpa.free(codes);
            @memset(codes, 0);
            const sb = try gpa.alloc(u8, N * K / gs);
            defer gpa.free(sb);
            @memset(sb, pow2Code(ff, 0));
            for (0..N) |c| {
                codes[c * K + at] = if (c < 256) 2 else @intCast(c - 256);
                if (c < 256) sb[c * (K / gs) + at / gs] = @intCast(c);
            }
            const dn = try fp4Bank(&sc, codes, sb, &.{ 1, N }, K, gs);
            const zeros8 = try gpa.alloc(u8, N * K);
            defer gpa.free(zeros8);
            @memset(zeros8, 0);
            const s8 = try gpa.alloc(u8, N * K / 32);
            defer gpa.free(s8);
            @memset(s8, 127);
            const sdn = try mxfp8Bank(&sc, zeros8, s8, N, K);
            var ly: dsv41.Layer = undefined;
            ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = dn[0], .s = dn[1], .qp = .{ .bits = 4, .group_size = gs, .mode = ff.mode() } } };
            ly.shared[2] = .{ .w = sdn[0], .s = sdn[1], .qp = mxfp8 };
            var av: [2 * K]f32 = @splat(0);
            av[at] = 1;
            const act = try bf16Of(&sc, &av, &.{ 1, 2, K });
            const idx = try sc.keep(mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{ 1, 1 }, 2, .uint32));
            const sp, const x1 = try identitySplit(&sc, 1, 4, N);
            const got = try dsv41.hostF32((try moe_fp4.moeDown(&sc, kern, &ly, act, idx, x1, &sp)) orelse return error.Declined);
            defer gpa.free(got);
            const comp = try dsv41.hostF32(try composedDown(&sc, &ly, act, idx, x1, &sp));
            defer gpa.free(comp);
            const wr = try gpa.alloc(f32, N);
            defer gpa.free(wr);
            for (wr, want) |*o, v| o.* = bf16Round(v);
            try checkColumns("down", ff, got[0..N], wr, comp[0..N], N);
        }
        // The up launch, clamp off (limit inf): the gate sum 32 (silu(32) is 32 in f32), the up sum the column's value;
        // out = bf16(32 x value), past bf16 / f32 range inf.
        {
            const K = 1024;
            const codes = try gpa.alloc(u8, 2 * N * K);
            defer gpa.free(codes);
            @memset(codes, 0);
            const sb = try gpa.alloc(u8, 2 * N * K / gs);
            defer gpa.free(sb);
            @memset(sb, pow2Code(ff, 0));
            for (0..N) |c| {
                codes[c * K] = 6; // gate: 4 x 2^3 at input 0
                sb[c * (K / gs)] = pow2Code(ff, 3);
                codes[(N + c) * K + at] = if (c < 256) 2 else @intCast(c - 256);
                if (c < 256) sb[(N + c) * (K / gs) + at / gs] = @intCast(c);
            }
            const gu = try fp4Bank(&sc, codes, sb, &.{ 1, 2 * N }, K, gs);
            const zeros8 = try gpa.alloc(u8, 2 * N * K);
            defer gpa.free(zeros8);
            @memset(zeros8, 0);
            const s8 = try gpa.alloc(u8, 2 * N * K / 32);
            defer gpa.free(s8);
            @memset(s8, 127);
            const sgu = try mxfp8Bank(&sc, zeros8, s8, 2 * N, K);
            var ly: dsv41.Layer = undefined;
            ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = none }, .gu = .{ .w = gu[0], .s = gu[1], .qp = .{ .bits = 4, .group_size = gs, .mode = ff.mode() } } };
            ly.shared_gu = .{ .w = sgu[0], .s = sgu[1], .qp = mxfp8 };
            var xv: [K]f32 = @splat(0);
            xv[0] = 1;
            xv[at] = 1;
            const h2 = try bf16Of(&sc, &xv, &.{ 1, K });
            const idx = try sc.keep(mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{ 1, 1 }, 2, .uint32));
            const wts = try sc.keep(mlx.mlx_array_new_data(&[_]f32{1}, &[_]c_int{ 1, 1 }, 2, .float32));
            const got = try dsv41.hostF32((try moe_fp4.moeUp(&sc, &m, kern, &ly, h2, idx, wts)) orelse return error.Declined);
            defer gpa.free(got);
            const comp = try dsv41.hostF32(try composedUp(&sc, &ly, h2, idx, wts, .{}, std.math.inf(f32)));
            defer gpa.free(comp);
            const wr = try gpa.alloc(f32, N);
            defer gpa.free(wr);
            for (wr, want) |*o, v| o.* = bf16Round(32 * bf16Round(v));
            try checkColumns("up", ff, got[0..N], wr, comp[0..N], N);
        }
    }
}

test "dsv41 dec moe_fp4 rules: NaN through the dot, the per-expert scales and the clamped SwiGLU stays NaN; the clamp bounds infinities; equal to the composed path bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const m = bf16Model();
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    const K = 1024;
    const E = 6;
    // Columns: 0 a NaN gate scale, 1 a NaN up scale, 2 a large gate (clamped to 10), 3 a large up and 4 a large negative up
    // (clamped to +-10), 5 a large negative gate, 6 gate -inf (its sum overflows f32: silu(-inf) is -inf x 0, NaN in MLX
    // too), the rest plain. Row r takes expert r; row 0 has a NaN input (every sum NaN). The per-expert scales
    // (`nvfp4_scaled`) are 1 but: expert 1 gate NaN, 2 up NaN, 3 down NaN, 4 gate +inf, 5 up -inf. 16 columns: MLX's
    // quantized matmuls take whole groups of 8 output rows.
    const N = 16;
    const big_x = 0x1p127;
    for (fmts) |f| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const ff = f.format();
        const gs = ff.group();
        const nan_code: u8 = if (ff == .mxfp4) 0xFF else 0x7F;
        const codes = try gpa.alloc(u8, E * 2 * N * K);
        defer gpa.free(codes);
        @memset(codes, 0);
        const sb = try gpa.alloc(u8, E * 2 * N * K / gs);
        defer gpa.free(sb);
        @memset(sb, pow2Code(ff, 0));
        for (0..E) |e| for (0..N) |c| {
            const g = (e * 2 * N + c) * K;
            const u = (e * 2 * N + N + c) * K;
            // Every column's gate 1.5 and up 2 at input 0 unless set below.
            codes[g] = 3;
            codes[u] = 4;
            switch (c) {
                0 => sb[g / gs] = nan_code,
                1 => sb[u / gs] = nan_code,
                2 => sb[g / gs] = pow2Code(ff, if (ff == .mxfp4) 20 else 8),
                3 => sb[u / gs] = pow2Code(ff, if (ff == .mxfp4) 20 else 8),
                4 => {
                    sb[u / gs] = pow2Code(ff, if (ff == .mxfp4) 20 else 8);
                    codes[u] = 12;
                },
                5 => {
                    sb[g / gs] = pow2Code(ff, if (ff == .mxfp4) 20 else 8);
                    codes[g] = 11;
                },
                6 => codes[g + 64] = 15, // -6 x the input 2^127 at 64: past f32
                else => {},
            }
        };
        const gu = try fp4Bank(&sc, codes, sb, &.{ E, 2 * N }, K, gs);
        const zeros8 = try gpa.alloc(u8, 2 * N * K);
        defer gpa.free(zeros8);
        @memset(zeros8, 0);
        const s8 = try gpa.alloc(u8, 2 * N * K / 32);
        defer gpa.free(s8);
        @memset(s8, 127);
        const sgu = try mxfp8Bank(&sc, zeros8, s8, 2 * N, K);
        var ly: dsv41.Layer = undefined;
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = none }, .gu = .{ .w = gu[0], .s = gu[1], .qp = f.qp() } };
        ly.shared_gu = .{ .w = sgu[0], .s = sgu[1], .qp = mxfp8 };
        var es: moe_fp4.Scales = .{};
        if (f == .nvfp4_scaled) {
            const nan = std.math.nan(f32);
            const inf = std.math.inf(f32);
            es = .{ .gate = try bf16Vec(&sc, &.{ 1, nan, 1, 1, inf, 1 }), .up = try bf16Vec(&sc, &.{ 1, 1, nan, 1, 1, -inf }), .down = try bf16Vec(&sc, &.{ 1, 1, 1, nan, 1, 1 }) };
        }
        const R = E;
        var xv: [R * K]f32 = @splat(0);
        for (0..R) |r| {
            xv[r * K] = 1;
            xv[r * K + 64] = big_x;
        }
        xv[128] = std.math.nan(f32);
        const h2 = try bf16Of(&sc, &xv, &.{ R, K });
        var ids: [R]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ R, 1 }, 2, .uint32));
        const wts = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }, &[_]c_int{ R, 1 }, 2, .float32));
        const got_a = (try moe_fp4.moeUpScaled(&sc, &m, kern, &ly, h2, idx, wts, es)) orelse return error.Declined;
        const got = try dsv41.hostF32(got_a);
        defer gpa.free(got);
        const comp_a = try composedUp(&sc, &ly, h2, idx, wts, es, lim_ds);
        const comp = try dsv41.hostF32(comp_a);
        defer gpa.free(comp);
        for (0..R) |r| for (0..N) |c| {
            const v = got[(r * 2 + 0) * N + c];
            // Expert 4's +inf gate scale makes column 5's negative gate -inf: NaN after the SiLU, as column 6.
            const nan_rule = r == 0 or c <= 1 or c == 6 or (f == .nvfp4_scaled and ((r >= 1 and r <= 3) or (r == 4 and c == 5)));
            if (nan_rule != std.math.isNan(v)) {
                std.debug.print("[moe_fp4 rules] NaN {s} row {d} column {d}: got {e}\n", .{ @tagName(f), r, c, v });
                return error.NanRule;
            }
            if (!nan_rule and !std.math.isFinite(v)) return error.ClampLetInfinityThrough;
            const w = comp[(r * 2 + 0) * N + c];
            if (!sameF32(v, w)) std.debug.print("[moe_fp4 rules] {s} row {d} column {d}: kernel {e}, composed MLX {e}\n", .{ @tagName(f), r, c, v, w });
        };
        try expectSameBits("nan/clamp", .{ .name = "edge", .k = K, .n = N, .topk = 1 }, f, R, got_a, comp_a);
    }
}

// ── NVIDIA's NVFP4 experts through the plugin (`dsv41_nvfp4_pack.py --experts-nvfp4`) ──

const fixture = @import("fixture.zig");
const env = @import("../../../env.zig");

test "dsv41 nvfp4 experts: the load reads MLX nvfp4 banks with their per-expert scales, and refuses a partial or non-finite set and scales beside mxfp4 by name" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const base: fixture.Opts = .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all };
    const fx = try fixture.Fixture.init(gpa, base);
    defer fx.deinit();
    const e = &fx.layer(20).experts;
    try testing.expectEqual(fp4.Format.nvfp4, fp4.of(e.gu.?.qp.mode, e.gu.?.qp.group_size).?);
    try testing.expectEqual(fp4.Format.nvfp4, fp4.of(e.down.qp.mode, e.down.qp.group_size).?);
    for (e.scale) |v| try testing.expect(v != null);
    var o = base;
    o.expert_scales = .missing_up;
    try testing.expectError(error.Dsv41ExpertScales, fixture.Fixture.init(gpa, o));
    o.expert_scales = .nan_gate;
    try testing.expectError(error.Dsv41ExpertScales, fixture.Fixture.init(gpa, o));
    o = base;
    o.experts_nvfp4 = false;
    o.experts_fp4 = true;
    try testing.expectError(error.Dsv41ExpertScales, fixture.Fixture.init(gpa, o));
}

test "dsv41 nvfp4 experts: the route carries each pick's down scale, bit for bit the unscaled weight times it" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all });
    defer fx.deinit();
    const ly = fx.layer(20);
    var bare = ly.*;
    bare.experts.scale = .{ null, null, null };
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const x = try sc.keep(try fx.normal(&.{ decode.max_rows, ci(fx.m.cfg.hidden_size) }, .bfloat16, 0, 1));
    const r = try dsv41.route(&sc, fx.m, ly, x, none);
    const r0 = try dsv41.route(&sc, fx.m, &bare, x, none);
    const want = try sc.mul(r0.wts, try sc.take(ly.experts.scale[2].?, r0.idx, 0));
    const ia = try dsv41.hostF32(r.idx);
    defer gpa.free(ia);
    const ib = try dsv41.hostF32(r0.idx);
    defer gpa.free(ib);
    try testing.expectEqualSlices(f32, ib, ia);
    const a = try dsv41.hostF32(r.wts);
    defer gpa.free(a);
    const b = try dsv41.hostF32(want);
    defer gpa.free(b);
    for (a, b) |p, q| try testing.expectEqual(@as(u32, @bitCast(q)), @as(u32, @bitCast(p)));
}

/// The layer's MoE output `[L, D]` in f32 from the dequantized banks: the gate and up outputs times their expert's
/// scale, the clamped SwiGLU, the down rows, the picks weighted by `wts` (the route already holds the down scale), plus
/// the shared expert.
fn moeTruth(sc: *Scope, ly: *const dsv41.Layer, h: A, idx: A, wts: A, lim: f32) !A {
    const gu = ly.experts.gu.?;
    const L = dims(h, 0);
    const k = dims(idx, 1);
    const n = dims(gu.w, 1) / 2;
    const E = dims(gu.w, 0);
    const hf = try astype(sc, h, .float32);
    const hu = try sc.matmul(hf, try sc.transpose(try dequantF32(sc, gu.w, gu.s, gu.qp), &.{ 0, 2, 1 })); // [E, L, 2N]
    var g = try sc.slice(hu, -1, 0, n);
    var u = try sc.slice(hu, -1, n, 2 * n);
    if (ly.experts.scale[0]) |v| g = try sc.mul(g, try reshape(sc, try astype(sc, v, .float32), &.{ ci(E), 1, 1 }));
    if (ly.experts.scale[1]) |v| u = try sc.mul(u, try reshape(sc, try astype(sc, v, .float32), &.{ ci(E), 1, 1 }));
    const a = try dsv41.clampedSwiglu(sc, g, u, lim);
    const dn = try sc.transpose(try sc.matmul(a, try sc.transpose(try dequantF32(sc, ly.experts.down.w, ly.experts.down.s, ly.experts.down.qp), &.{ 0, 2, 1 })), &.{ 1, 0, 2 }); // [L, E, D]
    const picked = try sc.takeAlong(dn, try reshape(sc, idx, &.{ ci(L), ci(k), 1 }), 1);
    const routed = try sc.sum(try sc.mul(picked, try reshape(sc, wts, &.{ ci(L), ci(k), 1 })), 1, false);
    const sgu = ly.shared_gu.?;
    const su = try sc.matmul(hf, try sc.transpose(try dequantF32(sc, sgu.w, sgu.s, sgu.qp), &.{ 1, 0 }));
    const sa = try dsv41.clampedSwiglu(sc, try sc.slice(su, -1, 0, n), try sc.slice(su, -1, n, 2 * n), lim);
    return sc.add(routed, try sc.matmul(sa, try sc.transpose(try dequantF32(sc, ly.shared[2].w, ly.shared[2].s, ly.shared[2].qp), &.{ 1, 0 })));
}

fn relRms(a: []const f32, t: []const f32) f64 {
    var e: f64 = 0;
    var s: f64 = 0;
    for (a, t) |x, r| {
        e += (x - r) * (x - r);
        s += @as(f64, r) * r;
    }
    return @sqrt(e / s);
}

test "dsv41 nvfp4 experts: decode (composed and the fused fp4 kernels) and prefill with the per-expert scales stay within the f32 model's bar" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 8, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    const lim = m.cfg.dsv4_swiglu_limit;
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    // On a NAX GPU the fused kernels serve only under their switch.
    const sw = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer sw.restore();
    for ([_]usize{ 1, decode.max_rows, 2048 }) |L| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h = try sc.keep(try fx.normal(&.{ ci(L), ci(d) }, .bfloat16, 0, 1));
        const r = try dsv41.route(&sc, m, ly, h, none);
        const truth = try moeTruth(&sc, ly, h, r.idx, r.wts, lim);
        const comp = try sc.add(try dsv41.routed(&sc, m, ly, h, r.idx, r.wts), try astype(&sc, try dsv41.shared(&sc, m, ly, h), .float32));
        var label: [64]u8 = undefined;
        if (L > decode.max_rows) {
            // The prompt: the sorted gathers (`moePrefill`, through `moeShare`).
            const got = try sc.keep(try dsv41.moeShare(m, gpa, 20, h, r.idx, r.wts));
            const g = try dsv41.hostF32(got);
            defer gpa.free(g);
            const c = try dsv41.hostF32(comp);
            defer gpa.free(c);
            const t = try dsv41.hostF32(truth);
            defer gpa.free(t);
            const eg = relRms(g, t);
            const ec = relRms(c, t);
            std.debug.print("[nvfp4 experts] prefill L={d}: prefill-truth {e:.2}, composed-truth {e:.2}\n", .{ L, eg, ec });
            try testing.expect(ec < 1e-2);
            try testing.expect(eg <= 1.25 * ec + 1e-4);
            continue;
        }
        // Decode: the fused kernels (they must take the layer: null is a decline) against the composed path, both
        // through the hc expand.
        const x1 = try sc.keep(try fx.stream(L));
        const sp: dsv41.Split = .{ .pre = none, .post = try sc.keep(try fx.uniform(&.{ ci(L), ci(hc) }, 0.05, 1.5)), .comb = try sc.keep(try fx.uniform(&.{ ci(L), ci(hc), ci(hc) }, -0.5, 0.5)) };
        const got = (try dsv41.ffnDecode(&sc, m, ly, h, r.idx, r.wts, x1, &sp)) orelse return error.Fp4KernelsDeclined;
        const want = try dsv41.hcExpand(&sc, x1, try astype(&sc, comp, .bfloat16), &sp);
        const tx = try dsv41.hcExpand(&sc, try astype(&sc, x1, .float32), truth, &sp);
        const g = try dsv41.hostF32(got);
        defer gpa.free(g);
        const w = try dsv41.hostF32(want);
        defer gpa.free(w);
        const t = try dsv41.hostF32(tx);
        defer gpa.free(t);
        try decode.expectNoWorse(try std.fmt.bufPrint(&label, "nvfp4 experts decode R={d}", .{L}), g, w, t);
        // The routed part alone: the composed path holds the scales (a scale left out moves it far more than this).
        const c = try dsv41.hostF32(comp);
        defer gpa.free(c);
        const tr = try dsv41.hostF32(truth);
        defer gpa.free(tr);
        std.debug.print("[nvfp4 experts] decode R={d}: composed-truth {e:.2}\n", .{ L, relRms(c, tr) });
        try testing.expect(relRms(c, tr) < 1e-2);
    }
}

// ── schedule invariance: a row's bytes whatever launch carries it ──

/// Each launch below repeats this many times.
const inv_reps = 50;

/// Six decode rows' inputs: the gate/up's normed rows, picks and weights; the down's activations, stream and hc split.
const RowIns = struct { h2: A, idx: A, wts: A, act: A, x1: A, post: A, comb: A };

fn rowIns(sc: *Scope, g: *Gen, rng: std.Random, d: Dims) !RowIns {
    const R = decode.max_rows;
    var ids: [decode.max_rows * 16]u32 = undefined;
    var ws: [decode.max_rows * 16]f32 = undefined;
    moe_up.route(rng, .random, R, d.topk, @intCast(d.experts), &ids, &ws);
    return .{
        .h2 = try astype(sc, try g.normal(sc, &.{ ci(R), ci(d.k) }, 1), .bfloat16),
        .idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ ci(R), ci(d.topk) }, 2, .uint32)),
        .wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ ci(R), ci(d.topk) }, 2, .float32)),
        .act = try astype(sc, try g.normal(sc, &.{ ci(R), ci(d.topk + 1), ci(d.n) }, 0.5), .bfloat16),
        .x1 = try astype(sc, try g.normal(sc, &.{ ci(R), ci(d.hc), ci(d.k) }, 1), .bfloat16),
        .post = try uniform(sc, g, &.{ ci(R), ci(d.hc) }, 0.05, 1.5),
        .comb = try uniform(sc, g, &.{ ci(R), ci(d.hc), ci(d.hc) }, -0.5, 0.5),
    };
}

/// Rows [lo, hi) of `a` in a buffer of their own, as a step or verify hands them over.
fn rowsOf(sc: *Scope, a: A, lo: usize, hi: usize) !A {
    var o = mlx.mlx_array_new();
    return own(sc, mlx.mlx_copy(&o, try sc.slice(a, 0, lo, hi), sc.s), o);
}

/// The outputs compared: the gate/up `[R, k + 1, N]`, the down over `RowIns.act` `[R, hc, D]` and the chain (the down
/// over the gate/up's output).
const stages = [_][]const u8{ "up", "down", "chain" };

/// The fused launches over rows [lo, hi) of `in`, as bits (caller frees).
fn launchRows(sc: *Scope, kern: moe_fp4.Kernels, l: *const Layer, in: RowIns, lo: usize, hi: usize) ![3][]const u16 {
    const m = bf16Model();
    const idx = try rowsOf(sc, in.idx, lo, hi);
    const x1 = try rowsOf(sc, in.x1, lo, hi);
    const sp: dsv41.Split = .{ .pre = none, .post = try rowsOf(sc, in.post, lo, hi), .comb = try rowsOf(sc, in.comb, lo, hi) };
    const up = (try moe_fp4.moeUpScaled(sc, &m, kern, &l.ly, try rowsOf(sc, in.h2, lo, hi), idx, try rowsOf(sc, in.wts, lo, hi), l.es)) orelse return error.Declined;
    const dn = (try moe_fp4.moeDown(sc, kern, &l.ly, try rowsOf(sc, in.act, lo, hi), idx, x1, &sp)) orelse return error.Declined;
    const ch = (try moe_fp4.moeDown(sc, kern, &l.ly, up, idx, x1, &sp)) orelse return error.Declined;
    const b0 = try moe_up.bitsOf(up);
    errdefer testing.allocator.free(b0);
    const b1 = try moe_up.bitsOf(dn);
    errdefer testing.allocator.free(b1);
    return .{ b0, b1, try moe_up.bitsOf(ch) };
}

/// Launches whose bits differ: a repeat from the first run (`repeat`), a row of an R-row launch from that row's own
/// one-row launch (`row`), per stage.
const InvTally = struct { repeat: [3]usize = @splat(0), row: [3]usize = @splat(0) };

/// Rows 0..5 at R = 1..6 (rows [0, R)) and each row alone, every launch `inv_reps` times.
fn rowInvariance(sc: *Scope, kern: moe_fp4.Kernels, l: *const Layer, in: RowIns, label: []const u8) !InvTally {
    const gpa = testing.allocator;
    const MR = decode.max_rows;
    var t: InvTally = .{};
    // Launch j < MR: rows [0, j + 1); j >= MR: row j - MR alone. The first run's bits.
    var first: [2 * MR][3][]const u16 = undefined;
    var have: usize = 0;
    defer for (first[0..have]) |b| for (b) |x| gpa.free(x);
    for (0..inv_reps) |rep| {
        var isc = Scope.init(gpa, sc.s);
        defer isc.deinit();
        for (0..2 * MR) |j| {
            const lo: usize = if (j < MR) 0 else j - MR;
            const hi: usize = if (j < MR) j + 1 else j - MR + 1;
            const b = try launchRows(&isc, kern, l, in, lo, hi);
            if (rep == 0) {
                first[j] = b;
                have += 1;
                continue;
            }
            defer for (b) |x| gpa.free(x);
            for (b, first[j], 0..) |x, y, s| if (!std.mem.eql(u16, x, y)) {
                t.repeat[s] += 1;
            };
        }
    }
    for (1..MR + 1) |R| for (0..R) |r| for (0..3) |s| {
        const want = first[MR + r][s];
        const got = first[R - 1][s][r * want.len ..][0..want.len];
        if (std.mem.eql(u16, got, want)) continue;
        t.row[s] += 1;
        var diff: usize = 0;
        var at: usize = 0;
        for (got, want, 0..) |a, b, i| if (a != b) {
            if (diff == 0) at = i;
            diff += 1;
        };
        std.debug.print("[moe_fp4 schedule] {s} {s} R={d} row {d}: {d} of {d} outputs differ from its one-row launch, first at {d}: 0x{x:0>4} vs 0x{x:0>4}\n", .{ label, stages[s], R, r, diff, want.len, at, got[at], want[at] });
    };
    std.debug.print("[moe_fp4 schedule] {s}: {d} reps; repeats differing up/down/chain {any}, rows differing from their one-row launch {any}\n", .{ label, inv_reps, t.repeat, t.row });
    return t;
}

test "dsv41 fp4 schedule: a row's gate/up, down and chained bytes are its one-row launch's at R = 1..6, each launch the same 50 times (DeepSeek-V4.1-Flash widths; mxfp4, nvfp4, nvfp4 with per-expert scales)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    var bad: usize = 0;
    for (fmts, 0..) |f, fi| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        var g: Gen = .{ .seed = 0x5C4E + fi };
        var l = try randomLayer(&sc, &g, deepseek, f);
        defer l.deinit();
        var prng = std.Random.DefaultPrng.init(0x5C4E + fi);
        const in = try rowIns(&sc, &g, prng.random(), deepseek);
        const t = try rowInvariance(&sc, kern, &l, in, @tagName(f));
        for (t.repeat, t.row) |a, b| bad += a + b;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 fp4 schedule: the MoE a plain step runs (ffnDecode, one row) gives each verify row's bytes at R = 1..6 (the fixture's mxfp4 and nvfp4 with per-expert scales banks, the fused fp4 kernels)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const sw = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer sw.restore();
    var bad: usize = 0;
    for ([_]bool{ false, true }) |nv| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 16, .ctx = 0, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none, .fused = true });
        defer fx.deinit();
        const m = fx.m;
        const ly = fx.layer(20);
        const MR = decode.max_rows;
        const hc = m.cfg.dsv4_hc_mult;
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h = try sc.keep(try fx.normal(&.{ ci(MR), ci(m.cfg.hidden_size) }, .bfloat16, 0, 1));
        const r = try dsv41.route(&sc, m, ly, h, none);
        const x1 = try sc.keep(try fx.stream(MR));
        const post = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc) }, 0.05, 1.5));
        const comb = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc), ci(hc) }, -0.5, 0.5));
        var first: [2 * MR][]const u16 = undefined;
        var have: usize = 0;
        defer for (first[0..have]) |b| gpa.free(b);
        var repeat: usize = 0;
        for (0..inv_reps) |rep| {
            var isc = Scope.init(gpa, mlx.gpuStream());
            defer isc.deinit();
            for (0..2 * MR) |j| {
                const lo: usize = if (j < MR) 0 else j - MR;
                const hi: usize = if (j < MR) j + 1 else j - MR + 1;
                const sp: dsv41.Split = .{ .pre = none, .post = try rowsOf(&isc, post, lo, hi), .comb = try rowsOf(&isc, comb, lo, hi) };
                const y = (try dsv41.ffnDecode(&isc, m, ly, try rowsOf(&isc, h, lo, hi), try rowsOf(&isc, r.idx, lo, hi), try rowsOf(&isc, r.wts, lo, hi), try rowsOf(&isc, x1, lo, hi), &sp)) orelse return error.Fp4KernelsDeclined;
                const b = try moe_up.bitsOf(y);
                if (rep == 0) {
                    first[j] = b;
                    have += 1;
                    continue;
                }
                defer gpa.free(b);
                if (!std.mem.eql(u16, b, first[j])) repeat += 1;
            }
        }
        var rows: usize = 0;
        for (1..MR + 1) |R| for (0..R) |row| {
            const want = first[MR + row];
            if (!std.mem.eql(u16, first[R - 1][row * want.len ..][0..want.len], want)) {
                rows += 1;
                std.debug.print("[nvfp4 experts] ffnDecode {s} R={d}: row {d} differs from the one-row step\n", .{ if (nv) "nvfp4+scales" else "mxfp4", R, row });
            }
        };
        std.debug.print("[nvfp4 experts] ffnDecode {s}: {d} reps; repeats differing {d}, verify rows differing from the one-row step {d}\n", .{ if (nv) "nvfp4+scales" else "mxfp4", inv_reps, repeat, rows });
        bad += repeat + rows;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

// ── NVFP4 that is MXFP4: NVIDIA's DeepSeek-V4.1-Flash pack ──

/// A bank as NVIDIA's DeepSeek-V4.1-Flash NVFP4 pack stores it (`nv`, group 16) and its MXFP4 conversion (`mx`, group
/// 32), the same e2m1 codes: each 32-span's two e4m3 scales one power of two 2^e, the expert's global 2^g (`g_lo` on
/// rows below `split`, `g_hi` on the rest), the e8m0 scale 2^(e + g).
const PairBank = struct { nv: [2]A, mx: [2]A };

fn pow2Banks(sc: *Scope, rng: std.Random, E: usize, rows: usize, K: usize, split: usize, g_lo: []const i32, g_hi: []const i32) !PairBank {
    const gpa = testing.allocator;
    const words = try gpa.alloc(u32, E * rows * K / 8);
    defer gpa.free(words);
    for (words) |*w| w.* = rng.int(u32);
    const nvs = try gpa.alloc(u8, E * rows * K / 16);
    defer gpa.free(nvs);
    const mxs = try gpa.alloc(u8, E * rows * K / 32);
    defer gpa.free(mxs);
    for (mxs, 0..) |*b, i| {
        const r = i / (K / 32);
        const x = r / rows;
        const e = rng.intRangeAtMost(i32, -4, 2);
        nvs[2 * i] = pow2Code(.nvfp4, e);
        nvs[2 * i + 1] = nvs[2 * i];
        b.* = pow2Code(.mxfp4, e + (if (r % rows < split) g_lo[x] else g_hi[x]));
    }
    return .{ .nv = try fp4Words(sc, words, nvs, &.{ E, rows }, K, 16), .mx = try fp4Words(sc, words, mxs, &.{ E, rows }, K, 32) };
}

/// Powers of two 2^e as `[E]` bf16.
fn pow2Vec(sc: *Scope, es: []const i32) !A {
    var v: [64]f32 = undefined;
    for (es, 0..) |e, i| v[i] = std.math.ldexp(@as(f32, 1), e);
    return bf16Vec(sc, v[0..es.len]);
}

test "dsv41 fp4 nvfp4 as mxfp4: the fused decode with FP4_READS off: NVIDIA's power-of-two NVFP4 bank with its per-expert globals gives its MXFP4 conversion's bytes through both launches, rows 1..6 (DeepSeek-V4.1-Flash widths)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const sw = env.override(.MLX_SERVE_DSV41_FP4_READS, "0");
    defer sw.restore();
    const gpa = testing.allocator;
    const d = deepseek;
    const E = d.experts;
    const m = bf16Model();
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(0x4E5F);
    const rng = prng.random();
    var gx: [3][16]i32 = undefined; // each expert's gate, up and down global exponents
    for (&gx) |*row| for (row[0..E]) |*e| {
        e.* = rng.intRangeAtMost(i32, -9, -6);
    };
    const gu = try pow2Banks(&sc, rng, E, 2 * d.n, d.k, d.n, gx[0][0..E], gx[1][0..E]);
    const dn = try pow2Banks(&sc, rng, E, d.k, d.n, d.k, gx[2][0..E], gx[2][0..E]);
    var g: Gen = .{ .seed = 0x4E5F };
    const sgu = try quant(&sc, try g.normal(&sc, &.{ ci(2 * d.n), ci(d.k) }, 1 / @sqrt(@as(f32, @floatFromInt(d.k)))), mxfp8);
    const sdn = try quant(&sc, try g.normal(&sc, &.{ ci(d.k), ci(d.n) }, 1 / @sqrt(@as(f32, @floatFromInt(d.n)))), mxfp8);
    const nv = layerOf(d, Fmt.nvfp4.qp(), gu.nv, dn.nv, sgu, sdn);
    const mx = layerOf(d, Fmt.mxfp4.qp(), gu.mx, dn.mx, sgu, sdn);
    const gsc = try pow2Vec(&sc, gx[0][0..E]);
    const usc = try pow2Vec(&sc, gx[1][0..E]);
    const dsc = try pow2Vec(&sc, gx[2][0..E]);
    const in = try rowIns(&sc, &g, rng, d);
    // The served form's routing weights: the down global on each pick's weight (`routeScaled`, f32).
    const wts_nv = try sc.mul(in.wts, try sc.take(try astype(&sc, dsc, .float32), in.idx, 0));
    var bad: usize = 0;
    for (1..decode.max_rows + 1) |R| {
        const h2 = try rowsOf(&sc, in.h2, 0, R);
        const idx = try rowsOf(&sc, in.idx, 0, R);
        const x1 = try rowsOf(&sc, in.x1, 0, R);
        const sp: dsv41.Split = .{ .pre = none, .post = try rowsOf(&sc, in.post, 0, R), .comb = try rowsOf(&sc, in.comb, 0, R) };
        const wts = try rowsOf(&sc, in.wts, 0, R);
        const up_mx = (try moe_fp4.moeUpScaled(&sc, &m, kern, &mx, h2, idx, wts, .{})) orelse return error.Declined;
        const y_mx = (try moe_fp4.moeDown(&sc, kern, &mx, up_mx, idx, x1, &sp)) orelse return error.Declined;
        // The gate/up launch with the gate and up globals; the down global in the kernel (DSCALE) and on the route.
        const up_nv = (try moe_fp4.moeUpScaled(&sc, &m, kern, &nv, h2, idx, wts, .{ .gate = gsc, .up = usc })) orelse return error.Declined;
        const up_ds = (try moe_fp4.moeUpScaled(&sc, &m, kern, &nv, h2, idx, wts, .{ .gate = gsc, .up = usc, .down = dsc })) orelse return error.Declined;
        const up_rt = (try moe_fp4.moeUpScaled(&sc, &m, kern, &nv, h2, idx, try rowsOf(&sc, wts_nv, 0, R), .{ .gate = gsc, .up = usc })) orelse return error.Declined;
        const y_ds = (try moe_fp4.moeDown(&sc, kern, &nv, up_ds, idx, x1, &sp)) orelse return error.Declined;
        const y_rt = (try moe_fp4.moeDown(&sc, kern, &nv, up_rt, idx, x1, &sp)) orelse return error.Declined;
        const cases = [_]struct { []const u8, A, A }{ .{ "up (gate and up globals)", up_nv, up_mx }, .{ "chain, down global in the launch", y_ds, y_mx }, .{ "chain, down global on the route", y_rt, y_mx } };
        for (cases) |c| {
            const a = try moe_up.bitsOf(c[1]);
            defer gpa.free(a);
            const b = try moe_up.bitsOf(c[2]);
            defer gpa.free(b);
            var diff: usize = 0;
            for (a, b) |p, q| diff += @intFromBool(p != q);
            std.debug.print("[moe_fp4 nvfp4-as-mxfp4] R={d} {s}: {d} of {d} outputs differ from the MXFP4 conversion's\n", .{ R, c[0], diff, a.len });
            bad += diff;
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// NVIDIA's NVFP4 experts and their MXFP4 conversion (`pow2Banks`) at a fixture layer's widths, with the per-expert
/// globals as bf16 `[E]` (gate, up, down).
const PackPair = struct {
    gu: PairBank,
    dn: PairBank,
    globals: [3]A,

    fn init(sc: *Scope, rng: std.Random, E: usize, d: usize, n: usize) !PackPair {
        var gx: [3][64]i32 = undefined;
        for (&gx) |*row| for (row[0..E]) |*e| {
            e.* = rng.intRangeAtMost(i32, -9, -6);
        };
        return .{
            .gu = try pow2Banks(sc, rng, E, 2 * n, d, n, gx[0][0..E], gx[1][0..E]),
            .dn = try pow2Banks(sc, rng, E, d, n, d, gx[2][0..E], gx[2][0..E]),
            .globals = .{ try pow2Vec(sc, gx[0][0..E]), try pow2Vec(sc, gx[1][0..E]), try pow2Vec(sc, gx[2][0..E]) },
        };
    }

    /// `ly`'s routed experts as the NVFP4 arm (with its globals) or the MXFP4 one (none).
    fn install(p: *const PackPair, ly: *dsv41.Layer, nv: bool) void {
        const f: Fmt = if (nv) .nvfp4 else .mxfp4;
        const gu = if (nv) p.gu.nv else p.gu.mx;
        const dn = if (nv) p.dn.nv else p.dn.mx;
        ly.experts.gate = .{ .w = none };
        ly.experts.up = .{ .w = none };
        ly.experts.gu = .{ .w = gu[0], .s = gu[1], .qp = f.qp() };
        ly.experts.down = .{ .w = dn[0], .s = dn[1], .qp = f.qp() };
        ly.experts.views = null;
        ly.experts.scale = if (nv) .{ p.globals[0], p.globals[1], p.globals[2] } else .{ null, null, null };
    }

    /// The NVFP4 arm's routing weights: the down global on each pick's weight, as `routeFrom` puts it (f32).
    fn nvWeights(p: *const PackPair, sc: *Scope, idx: A, wts: A) !A {
        return sc.mul(wts, try sc.take(try astype(sc, p.globals[2], .float32), idx, 0));
    }
};

/// Elements whose bits differ between two arrays of one dtype (bf16 or f32), printed under `label`.
fn bitDiff(label: []const u8, a: A, b: A) !usize {
    const gpa = testing.allocator;
    const x = try dsv41.hostF32(a);
    defer gpa.free(x);
    const y = try dsv41.hostF32(b);
    defer gpa.free(y);
    var diff: usize = 0;
    for (x, y) |p, q| diff += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
    std.debug.print("[nvfp4 as mxfp4] {s}: {d} of {d} outputs differ from the MXFP4 conversion's\n", .{ label, diff, x.len });
    return diff;
}

test "dsv41 fp4 nvfp4 as mxfp4: decode's routed gathers (the fused fp4 kernels off) of NVIDIA's power-of-two NVFP4 bank with its globals give its MXFP4 conversion's bytes, R = 1..6 (DeepSeek-V4.1-Flash widths)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const E = 16;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = E, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(0x9F10);
    const p = try PackPair.init(&sc, prng.random(), E, m.cfg.hidden_size, m.cfg.moe_intermediate_size);
    var bad: usize = 0;
    for (1..decode.max_rows + 1) |R| {
        const h = try sc.keep(try fx.normal(&.{ ci(R), ci(m.cfg.hidden_size) }, .bfloat16, 0, 1));
        p.install(ly, false);
        const r = try dsv41.route(&sc, m, ly, h, none);
        const y_mx = try dsv41.routed(&sc, m, ly, h, r.idx, r.wts);
        p.install(ly, true);
        const y_nv = try dsv41.routed(&sc, m, ly, h, r.idx, try p.nvWeights(&sc, r.idx, r.wts));
        var label: [48]u8 = undefined;
        bad += try bitDiff(try std.fmt.bufPrint(&label, "decode gathers R={d}", .{R}), y_nv, y_mx);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// The prefill half's bar: the NVFP4 arm applies its globals in its paired epilogue in another order than the MXFP4
/// arm's scale codes (MOE_PAIR_FP4), so the two round apart now and then; 1e-3 relative rms holds that and fails a
/// global on the wrong projection by orders of magnitude.
const nv_as_mx_prefill_bar = 1e-3;

/// `y` against the MXFP4 conversion's `want`: relative rms, printed under `label`.
fn relRmsOf(label: []const u8, y: A, want: A) !f64 {
    const gpa = testing.allocator;
    const x = try dsv41.hostF32(y);
    defer gpa.free(x);
    const t = try dsv41.hostF32(want);
    defer gpa.free(t);
    const e = relRms(x, t);
    std.debug.print("[nvfp4 as mxfp4] {s}: rel rms {e:.3} from the MXFP4 conversion's\n", .{ label, e });
    return e;
}

test "dsv41 fp4 nvfp4 as mxfp4: prefill's sorted gathers of NVIDIA's power-of-two NVFP4 bank with its globals are its MXFP4 conversion's within 1e-3 rel rms; the gate and up globals swapped fail (DeepSeek-V4.1-Flash widths)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const E = 16;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = E, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(0x9F11);
    const p = try PackPair.init(&sc, prng.random(), E, m.cfg.hidden_size, m.cfg.moe_intermediate_size);
    for ([_]usize{ 512, 2048 }) |L| {
        const h = try sc.keep(try fx.normal(&.{ ci(L), ci(m.cfg.hidden_size) }, .bfloat16, 0, 1));
        p.install(ly, false);
        const r = try dsv41.route(&sc, m, ly, h, none);
        const y_mx = try sc.keep(try dsv41.moeShare(m, gpa, 20, h, r.idx, r.wts));
        p.install(ly, true);
        const w_nv = try p.nvWeights(&sc, r.idx, r.wts);
        const y_nv = try sc.keep(try dsv41.moeShare(m, gpa, 20, h, r.idx, w_nv));
        var label: [64]u8 = undefined;
        try testing.expect(try relRmsOf(try std.fmt.bufPrint(&label, "prefill L={d}", .{L}), y_nv, y_mx) <= nv_as_mx_prefill_bar);
        // Planted: the gate's global on up and up's on the gate.
        ly.experts.scale = .{ p.globals[1], p.globals[0], p.globals[2] };
        const y_bad = try sc.keep(try dsv41.moeShare(m, gpa, 20, h, r.idx, w_nv));
        try testing.expect(try relRmsOf(try std.fmt.bufPrint(&label, "planted: globals swapped, L={d}", .{L}), y_bad, y_mx) > nv_as_mx_prefill_bar);
    }
}

test "dsv41 dec moe_fp4 rules: with FP4_READS off the shared expert's down reads every e4m3 code and every e8m0 scale code as MLX's mxfp8 composed path does, bit for bit" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const sw = env.override(.MLX_SERVE_DSV41_FP4_READS, "0");
    defer sw.restore();
    const gpa = testing.allocator;
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    // Columns 0..255: the e8m0 scale byte c on the e4m3 code 1.0 at input 40 (unit 1); 256..511: the e4m3 code c - 256
    // under scale 1. The shared slot's activation is 1 at input 40, the routed slot's bank all zero.
    const N = 512;
    const K = 64;
    const at = 40;
    const bytes = try gpa.alloc(u8, N * K);
    defer gpa.free(bytes);
    @memset(bytes, 0);
    const s8 = try gpa.alloc(u8, N * K / 32);
    defer gpa.free(s8);
    @memset(s8, 127);
    const want = try gpa.alloc(f32, N);
    defer gpa.free(want);
    for (0..N) |c| {
        bytes[c * K + at] = if (c < 256) 0x38 else @intCast(c - 256);
        if (c < 256) s8[c * (K / 32) + at / 32] = @intCast(c);
        const v = if (c < 256) e8m0Rule(@intCast(c)) else e4m3Rule(@intCast(c - 256));
        // A sum starts at +0: a -0 term leaves +0.
        want[c] = bf16Round(if (v == 0) 0 else v);
    }
    const sdn = try mxfp8Bank(&sc, bytes, s8, N, K);
    const zc = try gpa.alloc(u8, N * K);
    defer gpa.free(zc);
    @memset(zc, 0);
    const zs = try gpa.alloc(u8, N * K / 32);
    defer gpa.free(zs);
    @memset(zs, 127);
    const dn = try fp4Bank(&sc, zc, zs, &.{ 1, N }, K, 32);
    var ly: dsv41.Layer = undefined;
    ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = .{ .w = dn[0], .s = dn[1], .qp = Fmt.mxfp4.qp() } };
    ly.shared[2] = .{ .w = sdn[0], .s = sdn[1], .qp = mxfp8 };
    var av: [2 * K]f32 = @splat(0);
    av[K + at] = 1;
    const act = try bf16Of(&sc, &av, &.{ 1, 2, K });
    const idx = try sc.keep(mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{ 1, 1 }, 2, .uint32));
    const sp, const x1 = try identitySplit(&sc, 1, 4, N);
    const got = try dsv41.hostF32((try moe_fp4.moeDown(&sc, kern, &ly, act, idx, x1, &sp)) orelse return error.Declined);
    defer gpa.free(got);
    const comp = try dsv41.hostF32(try composedDown(&sc, &ly, act, idx, x1, &sp));
    defer gpa.free(comp);
    try checkColumns("shared down", .mxfp4, got[0..N], want, comp[0..N], N);
}

// ── the decode levers: today's bytes ──

/// A set of the decode levers (`moe_fp4.Levers`) as their switches set it.
const LeverSet = struct { name: []const u8, lv: moe_fp4.Levers };
const lever_off: LeverSet = .{ .name = "off", .lv = .{} };
/// The levers that keep the bytes: each, and both.
const lever_sets = [_]LeverSet{
    .{ .name = "DEC_WORD", .lv = .{ .dec_word = true } },
    .{ .name = "UP_SPLIT", .lv = .{ .up_split = true } },
    .{ .name = "both", .lv = .{ .dec_word = true, .up_split = true } },
};

/// `launchRows` with the lever switches at `ls` (and FP4_READS at `fr`).
fn launchLevers(sc: *Scope, kern: moe_fp4.Kernels, l: *const Layer, in: RowIns, lo: usize, hi: usize, ls: LeverSet, fr: [*:0]const u8) ![3][]const u16 {
    const o0 = env.override(.MLX_SERVE_DSV41_FP4_DEC_WORD, if (ls.lv.dec_word) "1" else "0");
    defer o0.restore();
    const o1 = env.override(.MLX_SERVE_DSV41_FP4_UP_SPLIT, if (ls.lv.up_split) "1" else "0");
    defer o1.restore();
    const o3 = env.override(.MLX_SERVE_DSV41_FP4_READS, fr);
    defer o3.restore();
    return launchRows(sc, kern, l, in, lo, hi);
}

/// Host routed banks for `l` at `d` in `f`: every nibble a random code (all sixteen), scales near 2^-4; NVFP4 gives
/// half of the 32-spans two equal scales (FP4_READS off sums those in MXFP4's order). `plant`: weight 0 of output 0 of
/// every expert's gate and down bank one exponent bit off (a decode that misreads one code).
fn hostRouted(sc: *Scope, l: *Layer, d: Dims, f: Fmt, seed: u64, plant: bool) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const E = d.experts;
    const gs = f.format().group();
    const banks = [2][2]usize{ .{ 2 * d.n, d.k }, .{ d.k, d.n } };
    for (banks, 0..) |rk, bi| {
        const rows = rk[0];
        const K = rk[1];
        const words = try gpa.alloc(u32, E * rows * K / 8);
        defer gpa.free(words);
        for (words) |*w| w.* = rng.int(u32);
        const sb = try gpa.alloc(u8, E * rows * K / gs);
        defer gpa.free(sb);
        var i: usize = 0;
        while (i < sb.len) : (i += 1) {
            if (gs == 32) {
                sb[i] = pow2Code(.mxfp4, rng.intRangeAtMost(i32, -5, -3));
                continue;
            }
            sb[i] = kc.host.e4m3Code(rng, -5, -3, false);
            if (i % 2 == 1 and rng.boolean()) sb[i] = sb[i - 1];
        }
        if (plant) for (0..E) |e| {
            const w0 = e * rows * K;
            setCode(words, w0, @as(u8, @intCast((words[w0 / 8] & 15) ^ 4)));
        };
        const b = try fp4Words(sc, words, sb, &.{ E, rows }, K, gs);
        if (bi == 0) l.ly.experts.gu = .{ .w = b[0], .s = b[1], .qp = f.qp() } else l.ly.experts.down = .{ .w = b[0], .s = b[1], .qp = f.qp() };
    }
}

/// Each lever set against the levers off at R = 1..6, every stage (gate/up, down, the chain): bit for bit, and the
/// planted bank (one code off) differs at every R. Returns the bytes that differ unplanted.
fn leversAgree(d: Dims, sets: []const LeverSet) !usize {
    const gpa = testing.allocator;
    const kern = try moe_fp4.build(lim_ds);
    defer kern.deinit();
    var bad: usize = 0;
    // NVFP4 with FP4_READS off as well: the equal-scale spans take MXFP4's order there.
    const cases = [_]struct { Fmt, [*:0]const u8 }{ .{ .mxfp4, "1" }, .{ .nvfp4, "1" }, .{ .nvfp4_scaled, "1" }, .{ .nvfp4, "0" }, .{ .mxfp4, "0" } };
    for (cases, 0..) |cs, ci_| {
        const f = cs[0];
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        var g: Gen = .{ .seed = 0x1E7E + ci_ };
        var l = try randomLayer(&sc, &g, d, f);
        defer l.deinit();
        var p = l;
        try hostRouted(&sc, &l, d, f, 0x1E7E + ci_, false);
        try hostRouted(&sc, &p, d, f, 0x1E7E + ci_, true);
        var prng = std.Random.DefaultPrng.init(0x1E7E + ci_);
        const in = try rowIns(&sc, &g, prng.random(), d);
        for (1..decode.max_rows + 1) |R| {
            var isc = Scope.init(gpa, sc.s);
            defer isc.deinit();
            const want = try launchLevers(&isc, kern, &l, in, 0, R, lever_off, cs[1]);
            defer for (want) |b| gpa.free(b);
            for (sets) |ls| {
                const got = try launchLevers(&isc, kern, &l, in, 0, R, ls, cs[1]);
                defer for (got) |b| gpa.free(b);
                const pl = try launchLevers(&isc, kern, &p, in, 0, R, ls, cs[1]);
                defer for (pl) |b| gpa.free(b);
                var diff: [3]usize = @splat(0);
                var seen: [3]usize = @splat(0);
                for (0..3) |s| for (got[s], pl[s], want[s]) |a, b, w| {
                    diff[s] += @intFromBool(a != w);
                    seen[s] += @intFromBool(b != w);
                };
                if (diff[0] + diff[1] + diff[2] != 0 or seen[0] == 0 or seen[1] == 0 or seen[2] == 0)
                    std.debug.print("[moe_fp4 levers] {s} {s} FP4_READS={s} R={d} {s}: up/down/chain bytes differing {any}, planted {any}\n", .{ d.name, @tagName(f), cs[1], R, ls.name, diff, seen });
                bad += diff[0] + diff[1] + diff[2];
                // The plant must show in every stage, or the comparison could not see a misread code.
                for (seen) |v| if (v == 0) return error.PlantUnseen;
            }
        }
        std.debug.print("[moe_fp4 levers] {s} {s} FP4_READS={s}: R = 1..6, {d} lever sets against the levers off\n", .{ d.name, @tagName(f), cs[1], sets.len });
    }
    return bad;
}

test "dsv41 dec moe_fp4 levers: DEC_WORD and UP_SPLIT, each and together, give today's bytes through both launches and the chain at R = 1..6 (mxfp4, nvfp4, nvfp4 with per-expert scales; FP4_READS on and off; DeepSeek-V4.1-Flash widths, both at GLM-5.3 Pro's); one misread code planted shows in every stage" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var bad = try leversAgree(deepseek, &lever_sets);
    bad += try leversAgree(glm, lever_sets[2..]);
    try testing.expectEqual(@as(usize, 0), bad);
}

/// The MoE a plain step runs over rows of fixed routes (`decode.rowsInvariant`): `ffnDecode` on the scaled routes, or
/// with `down_scale` the fused launches that carry the down scale themselves (`fuse.ffnDownScaled`, FUSE_DOWN_SCALE).
const MoeRows = struct {
    sc: *Scope,
    m: *const dsv41.Dsv41Model,
    ly: *const dsv41.Layer,
    h: A,
    idx: A,
    wts: A,
    x1: A,
    post: A,
    comb: A,
    down_scale: bool,

    pub fn runRows(c: *MoeRows, lo: usize, hi: usize) !decode.RowOuts {
        const sc = c.sc;
        const sp: dsv41.Split = .{ .pre = none, .post = try rowsOf(sc, c.post, lo, hi), .comb = try rowsOf(sc, c.comb, lo, hi) };
        const h = try rowsOf(sc, c.h, lo, hi);
        const idx = try rowsOf(sc, c.idx, lo, hi);
        const wts = try rowsOf(sc, c.wts, lo, hi);
        const x1 = try rowsOf(sc, c.x1, lo, hi);
        const y = if (c.down_scale)
            try @import("../fuse.zig").ffnDownScaled(sc, c.m, c.ly, h, idx, wts, x1, &sp)
        else
            try dsv41.ffnDecode(sc, c.m, c.ly, h, idx, try dsv41.routeScaled(sc, c.ly, idx, wts), x1, &sp);
        return .{ y orelse return error.Fp4KernelsDeclined, none, none, none, none, none, none, none };
    }
};

test "dsv41 exact rows: moe: the MoE a plain step runs gives each verify row its one-row bytes at R = 1..6, ffnDecode and the fused down scale (FUSE_DOWN_SCALE; the fixture's mxfp4, nvfp4 with per-expert scales, the iQ pack's affine and the pruned packs' 2-bit g128 banks)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const Bank = enum { mxfp4, nvfp4, affine, affine2 };
    var bad: usize = 0;
    for (std.enums.values(Bank)) |bank| {
        const nv = bank == .nvfp4;
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 16, .ctx = 0, .experts_fp4 = bank == .mxfp4, .experts_nvfp4 = nv, .experts_affine = if (bank == .affine2) 2 else null, .expert_scales = if (nv) .all else .none, .fused = true });
        defer fx.deinit();
        const m = fx.m;
        const ly = fx.layer(20);
        const MR = decode.max_rows;
        const hc = m.cfg.dsv4_hc_mult;
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const h = try sc.keep(try fx.normal(&.{ ci(MR), ci(m.cfg.hidden_size) }, .bfloat16, 0, 1));
        const r = try dsv41.route(&sc, m, ly, h, none);
        for ([_]bool{ false, true }) |down_scale| {
            // The fused down scale serves a bank with its per-expert scales alone.
            if (down_scale and !nv) continue;
            var c: MoeRows = .{ .sc = &sc, .m = m, .ly = ly, .h = h, .idx = r.idx, .wts = r.wts, .x1 = try sc.keep(try fx.stream(MR)), .post = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc) }, 0.05, 1.5)), .comb = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc), ci(hc) }, -0.5, 0.5)), .down_scale = down_scale };
            var buf: [48]u8 = undefined;
            bad += try decode.rowsInvariant(try std.fmt.bufPrint(&buf, "moe {t}{s}", .{ bank, if (down_scale) " down scale" else "" }), &c);
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
