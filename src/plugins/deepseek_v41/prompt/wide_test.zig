//! The wide prompt layer's default-off rewrites behind their switches against the stock (unfused) path at the
//! release's widths and prompt row counts: each kernel's own, then each switch alone and all of them through a
//! release-shape pair worker's prefill (`decode/fixture.zig`). A rewrite may round in another order, so each output is
//! held to a tolerance (`Err`), not to the stock path's bytes; a planted semantic bug (a weight on the wrong expert,
//! swapped per-expert scales, a dropped norm weight or mix scale, the routed term at half its scale) must fail it.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const mpair = @import("moe_pair.zig");
const fixture = @import("../decode/fixture.zig");

const testing = std.testing;
const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const hidden: usize = 5120;
const inter: usize = 2304;
const top_k = 6;

fn skip() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable();
}

/// How far `got` is from `want`, in f32 over the elements neither holds a NaN at: `l2` the difference's norm over the
/// base's, `max` its largest magnitude over the base's largest. The base is `want`, or `want - ref` where a residual
/// `ref` would hide the change under test. `nan_same`: the two hold NaNs at the same elements.
const Err = struct {
    l2: f32,
    max: f32,
    nan_same: bool,

    /// Reordered rounding: within `l2_tol` in norm and 5% of the base's largest element anywhere (bf16 keeps 8 bits, so
    /// one step either side is 0.4%; a fused epilogue drops one rounding, a retiled sum reorders them).
    fn ok(e: Err, l2_tol: f32) bool {
        return e.nan_same and e.l2 <= l2_tol and e.max <= 0.05;
    }
};

const tol_kernel: f32 = 1e-2;

fn scalar(x: A) !f32 {
    try dsv41.evalAll(&.{x});
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, x));
    return v;
}

fn errOf(want: A, got: A, ref: ?A) !Err {
    if (mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got) or !std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.ShapeOrDtypeDiffers;
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const w = try sc.astype(want, .float32);
    const g = try sc.astype(got, .float32);
    const nw = try sc.op1(mlx.mlx_logical_not, try sc.op2(mlx.mlx_equal, w, w));
    const ng = try sc.op1(mlx.mlx_logical_not, try sc.op2(mlx.mlx_equal, g, g));
    var eq = mlx.mlx_array_new();
    const same = try sc.res(mlx.mlx_array_equal(&eq, nw, ng, false, sc.s), &eq);
    try dsv41.evalAll(&.{same});
    var nan_same: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&nan_same, same));
    const zero = try sc.f(0);
    const w0 = try sc.where(nw, zero, w);
    const g0 = try sc.where(ng, zero, g);
    const d = try sc.sub(g0, w0);
    const base = if (ref) |r| try sc.sub(w0, try sc.where(nw, zero, try sc.astype(r, .float32))) else w0;
    var o1 = mlx.mlx_array_new();
    const dd = try sc.res(mlx.mlx_sum(&o1, try sc.op1(mlx.mlx_square, d), false, sc.s), &o1);
    var o2 = mlx.mlx_array_new();
    const bb = try sc.res(mlx.mlx_sum(&o2, try sc.op1(mlx.mlx_square, base), false, sc.s), &o2);
    var o3 = mlx.mlx_array_new();
    const dm = try sc.res(mlx.mlx_max(&o3, try sc.op1(mlx.mlx_abs, d), false, sc.s), &o3);
    var o4 = mlx.mlx_array_new();
    const bm = try sc.res(mlx.mlx_max(&o4, try sc.op1(mlx.mlx_abs, base), false, sc.s), &o4);
    const vb = try scalar(bb);
    const vm = try scalar(bm);
    if (!(vb > 0) or !(vm > 0)) return error.EmptyBase;
    return .{ .l2 = @sqrt(try scalar(dd) / vb), .max = try scalar(dm) / vm, .nan_same = nan_same };
}

/// `got` within `l2_tol` of `want` (printed either way, so a run shows its margin).
fn close(what: []const u8, want: A, got: A, ref: ?A, l2_tol: f32) !bool {
    const e = try errOf(want, got, ref);
    std.debug.print("{s}: l2 {e:.3} max {e:.3} nan_same {}\n", .{ what, e.l2, e.max, e.nan_same });
    return e.ok(l2_tol);
}

fn key(sc: *dsv41.Scope, seed: u64) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_key(&o, seed), &o);
}

fn normal(sc: *dsv41.Scope, shape: []const c_int, dt: mlx.mlx_dtype, std_: f32, seed: u64) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, std_, try key(sc, seed), sc.s), &o);
}

fn uniform(sc: *dsv41.Scope, shape: []const c_int, dt: mlx.mlx_dtype, lo: f32, hi: f32, seed: u64) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_uniform(&o, try sc.f(lo), try sc.f(hi), shape.ptr, shape.len, dt, try key(sc, seed), sc.s), &o);
}

// ── the paired fp4 gate and up (moe_pair.zig) ──

/// Random fp4 codes `[experts, rows, cols]` over every e2m1 value, with scale codes drawn from [lo, hi): e8m0 2^-6 to
/// 2^-4 a 32-group (MXFP4) or e4m3 2^-6 to 0.117 a 16-group (NVFP4), so many gate and up products pass the clamp.
fn fp4Bank(sc: *dsv41.Scope, seed: u64, experts: c_int, rows_: usize, cols_: usize, nv: bool) !dsv41.Bank {
    const group: c_int = if (nv) 16 else 32;
    const rows: c_int = @intCast(rows_);
    const cols: c_int = @intCast(cols_);
    var w = mlx.mlx_array_new();
    const wa = try sc.res(mlx.mlx_random_bits(&w, &[_]c_int{ experts, rows, @divExact(cols, 8) }, 3, 4, try key(sc, seed), sc.s), &w);
    var s = mlx.mlx_array_new();
    const lo: f32 = if (nv) 0x08 else 121;
    const hi: f32 = if (nv) 0x20 else 124;
    const sa = try sc.res(mlx.mlx_random_randint(&s, try sc.f(lo), try sc.f(hi), &[_]c_int{ experts, rows, @divExact(cols, group) }, 3, .uint8, try key(sc, seed + 1), sc.s), &s);
    try dsv41.evalAll(&.{ wa, sa });
    return .{ .w = wa, .s = sa, .qp = .{ .bits = 4, .group_size = @intCast(group), .mode = if (nv) .nvfp4 else .mxfp4 } };
}

const Call = struct { x: A, idx: A, wts: A, sh: A };

/// `L` rows' normed input, routes (six distinct experts a row drawn as floor(u^2 E): expert 0 takes about sqrt(E) times
/// the mean, the high ids run short), f32 weights in [0.05, 1.05) and shared-expert rows; `nan`: row 3 all NaN, row 7
/// one NaN.
fn call(sc: *dsv41.Scope, L: usize, experts: usize, seed: u64, nan: bool) !Call {
    const a = testing.allocator;
    const shape = [_]c_int{ @intCast(L), @intCast(hidden) };
    var x = try normal(sc, &shape, .bfloat16, 1, seed);
    if (nan) {
        try dsv41.evalAll(&.{x});
        const h = try a.dupe(u16, (mlx.mlx_array_data_bfloat16(x) orelse return error.NoData)[0 .. L * hidden]);
        defer a.free(h);
        @memset(h[3 * hidden ..][0..hidden], 0x7FC0);
        h[7 * hidden + 100] = 0x7FC0;
        x = try sc.keep(mlx.mlx_array_new_data(h.ptr, &shape, 2, .bfloat16));
    }
    const ids = try a.alloc(u32, L * top_k);
    defer a.free(ids);
    const ws = try a.alloc(f32, L * top_k);
    defer a.free(ws);
    var rng = std.Random.DefaultPrng.init(seed);
    for (0..L) |l| {
        var j: usize = 0;
        while (j < top_k) {
            const u = rng.random().float(f32);
            const e: u32 = @intCast(@min(experts - 1, @as(usize, @intFromFloat(u * u * @as(f32, @floatFromInt(experts))))));
            if (std.mem.indexOfScalar(u32, ids[l * top_k ..][0..j], e) != null) continue;
            ids[l * top_k + j] = e;
            ws[l * top_k + j] = 0.05 + rng.random().float(f32);
            j += 1;
        }
    }
    const idx = try sc.keep(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ @intCast(L), top_k }, 2, .uint32));
    const wts = try sc.keep(mlx.mlx_array_new_data(ws.ptr, &[_]c_int{ @intCast(L), top_k }, 2, .float32));
    return .{ .x = x, .idx = idx, .wts = wts, .sh = try normal(sc, &shape, .bfloat16, 1, seed + 1) };
}

/// `mpair.act` on `c` as `moePrefill` calls it (its pairs sorted by expert).
fn actOf(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.Experts, x: A, idx: A, wts: A) !?A {
    const L: usize = @intCast(mlx.getShape(x)[0]);
    const n: c_int = @intCast(L * top_k);
    const flat = try sc.reshape(try sc.astype(idx, .uint32), &.{n});
    var o = mlx.mlx_array_new();
    const order = try sc.res(mlx.mlx_argsort_axis(&o, flat, 0, sc.s), &o);
    const rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.u(top_k));
    const w_sorted = try sc.take(try sc.reshape(wts, &.{n}), order, 0);
    return mpair.act(sc, m, e, try sc.reshape(x, &.{ @intCast(L), 1, @intCast(hidden) }), rows, try sc.take(flat, order, 0), w_sorted);
}

/// Each row's routing weights rotated by one (a weight on the wrong expert): the planted MoE bug.
fn rolled(sc: *dsv41.Scope, wts: A) !A {
    return sc.concat(&.{ try sc.slice(wts, 1, 1, top_k), try sc.slice(wts, 1, 0, 1) }, 1);
}

test "dsv41 wide tol: the paired fp4 gate+up with moe_act in its epilogue is the split path's MoE within 1% at the release widths, MXFP4 and NVFP4 with per-expert scales, prompt row counts to 8192; a weight on the wrong expert or swapped scales fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    // The release's 384 experts under `-Dslow-tests` (7 GB of banks); else 48, whose rows an expert reach every plan.
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    const unset = env.override(.MLX_SERVE_DSV41_MOE_PAIR, null);
    defer unset.restore();
    const unset_scaled = env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "0");
    defer unset_scaled.restore();
    // Every format and row count runs; each out of tolerance is reported.
    var ok = true;
    inline for (.{ false, true }) |nv| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
        m.moe_wt_kernel = null;
        m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
        var ly: dsv41.Layer = undefined;
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        // Per-expert scales over two octaves either side of 1.
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 17, 256, 500, 3072, 5001, 8064, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x1000 + ci, L == 500);
            // No shared expert, so the routed sum is all the output holds.
            const sh = try csc.zeros(&.{ @intCast(L), @intCast(hidden) }, .bfloat16);
            const want = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
            // The other kind's switch leaves this bank on the split path.
            {
                const other = if (nv) env.override(.MLX_SERVE_DSV41_MOE_PAIR, "1") else env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "1");
                defer other.restore();
                try testing.expect(try actOf(&csc, &m, &ly.experts, c.x, c.idx, c.wts) == null);
            }
            const sw = if (nv) env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "1") else env.override(.MLX_SERVE_DSV41_MOE_PAIR, "1");
            defer sw.restore();
            // Served: the paired kernel takes this call.
            try testing.expect(try actOf(&csc, &m, &ly.experts, c.x, c.idx, c.wts) != null);
            var lb: [64]u8 = undefined;
            const label = try std.fmt.bufPrint(&lb, "moe pair {s} rows {d}", .{ if (nv) "NVFP4" else "MXFP4", L });
            const got = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
            if (!try close(label, want, got, null, tol_kernel)) {
                std.debug.print("{s}: the paired path is outside the split path's tolerance\n", .{label});
                ok = false;
            }
            // Planted: each row's weights on the wrong experts.
            const bad_w = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, try rolled(&csc, c.wts), sh);
            try testing.expect(!try close("  planted: weights rolled", want, bad_w, null, tol_kernel));
            // Planted: the gate's per-expert scale on up and up's on the gate.
            if (nv) {
                const s = ly.experts.scale;
                ly.experts.scale = .{ s[1], s[0], s[2] };
                defer ly.experts.scale = s;
                const bad_s = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
                try testing.expect(!try close("  planted: gate and up scales swapped", want, bad_s, null, tol_kernel));
            }
        }
    }
    try testing.expect(ok);
}

// ── the hc stages ──

/// A model holding only the hc kernels, at the release's widths, the fused ones built and probed.
fn hcModel(m: *dsv41.Dsv41Model) !void {
    m.cfg = .{};
    m.cfg.hidden_size = hidden;
    m.cfg.dsv4_hc_mult = 4;
    m.cfg.dsv4_hc_sinkhorn_iters = 20;
    m.s = mlx.gpuStream();
    m.act = .bfloat16;
    m.hc_tg = 1024;
    m.hc_mix_kernel = try dsv41.buildKernel("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, dsv41.HC_MIX_SOURCE, "");
    m.hc_norm_kernel = try dsv41.buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, dsv41.HC_NORM_SOURCE, "");
    m.hc_split_kernel = try dsv41.buildKernel("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, dsv41.HC_SPLIT_SOURCE, "");
    try dsv41.hcFuseInit(testing.allocator, m);
}

fn hcModelFree(m: *dsv41.Dsv41Model) void {
    for ([_]mlx.mlx_fast_metal_kernel{ m.hc_mix_kernel, m.hc_norm_kernel, m.hc_split_kernel }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
}

/// `L` rows of a wide layer's hc inputs: the stream, the block output, its split, the previous mix, the mix weights
/// and the norm.
const HcRows = struct { x: A, y: A, sp: dsv41.Split, cpre: A, hc: dsv41.Hc, w: A };

fn hcRows(sc: *dsv41.Scope, L: c_int, seed: u64) !HcRows {
    const d: c_int = @intCast(hidden);
    return .{
        .x = try normal(sc, &.{ L, 4, d }, .bfloat16, 1, seed),
        .y = try normal(sc, &.{ L, d }, .bfloat16, 1, seed + 1),
        .sp = .{ .pre = none, .post = try uniform(sc, &.{ L, 4 }, .float32, 0.1, 2, seed + 2), .comb = try uniform(sc, &.{ L, 4, 4 }, .float32, 0, 1, seed + 3) },
        .cpre = try uniform(sc, &.{ L, 4 }, .float32, 0.05, 0.45, seed + 4),
        .hc = .{ .fn_w = try normal(sc, &.{ 24, 4 * d }, .float32, 0.01, seed + 5), .scale = try uniform(sc, &.{24}, .float32, 0.5, 1.5, seed + 6), .base = try uniform(sc, &.{24}, .float32, -0.5, 0.5, seed + 7) },
        .w = try uniform(sc, &.{d}, .bfloat16, 0.5, 1.5, seed + 8),
    };
}

test "dsv41 wide tol: the FFN-side hc expand and norm in one launch are the stock launches' stream, split and norm within 1% at prompt widths; a dropped norm weight fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    const unset = env.override(.MLX_SERVE_DSV41_HC_EXPAND_NORM, null);
    defer unset.restore();
    var ok = true;
    for ([_]c_int{ 17, 256, 1027, 4096, 8192 }, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0x4C00 + 16 * @as(u64, i));
        const want = try dsv41.hcExpandPreNorm(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, r.w);
        const on = env.override(.MLX_SERVE_DSV41_HC_EXPAND_NORM, "1");
        defer on.restore();
        const got = (try dsv41.hcExpandNormFresh(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, r.w)) orelse return error.FusedHcRefused;
        const served = try dsv41.hcExpandPreNorm(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, r.w);
        const names = [_][]const u8{ "x1", "h", "pre", "post", "comb" };
        for ([_]dsv41.Staged{ got, served }) |g| for ([_][2]A{ .{ want.x1, g.x1 }, .{ want.h, g.h }, .{ want.sp.pre, g.sp.pre }, .{ want.sp.post, g.sp.post }, .{ want.sp.comb, g.sp.comb } }, names) |pr, nm| {
            var lb: [64]u8 = undefined;
            if (!try close(try std.fmt.bufPrint(&lb, "hc expand norm rows {d} {s}", .{ L, nm }), pr[0], pr[1], null, tol_kernel)) ok = false;
        };
        // Planted: the fused launch without the norm's weight (ones in its place).
        const bad = (try dsv41.hcExpandNormFresh(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, try sc.ones(&.{@intCast(hidden)}, .bfloat16))) orelse return error.FusedHcRefused;
        try testing.expect(!try close("  planted: norm weight dropped", want.h, bad.h, null, tol_kernel));
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: the hc mixes at 8 rows a threadgroup are the 4-row launches' mixes, split and norms within 1% at prompt widths; a dropped mix scale fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    const unset = env.override(.MLX_SERVE_DSV41_HC_MIX8, null);
    defer unset.restore();
    var ok = true;
    for ([_]c_int{ 17, 1027, 5001, 8192 }, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0x8800 + 16 * @as(u64, i));
        const eps = try dsv41.hcEps(&sc, &m);
        const z4 = try dsv41.hcMixWide(&sc, &m, r.x, &r.hc, eps);
        const f4 = try dsv41.hcMixNormFused(&sc, &m, r.x, r.cpre, &r.hc, r.w, r.w, eps);
        const on = env.override(.MLX_SERVE_DSV41_HC_MIX8, "1");
        defer on.restore();
        const z8 = try dsv41.hcMixWide(&sc, &m, r.x, &r.hc, eps);
        const f8 = try dsv41.hcMixNormFused(&sc, &m, r.x, r.cpre, &r.hc, r.w, r.w, eps);
        const names = [_][]const u8{ "z", "fused z", "h", "h2", "comb" };
        for ([_][2]A{ .{ z4, z8 }, .{ f4.z, f8.z }, .{ f4.n.h, f8.n.h }, .{ f4.h2, f8.h2 }, .{ f4.n.sp.comb, f8.n.sp.comb } }, names) |pr, nm| {
            var lb: [64]u8 = undefined;
            if (!try close(try std.fmt.bufPrint(&lb, "hc mix8 rows {d} {s}", .{ L, nm }), pr[0], pr[1], null, tol_kernel)) ok = false;
        }
        // Planted: 8 rows a threadgroup with the mixes' scale dropped (ones in its place).
        var hc_bad = r.hc;
        hc_bad.scale = try sc.ones(mlx.getShape(r.hc.scale), .float32);
        const bad = try dsv41.hcMixWide(&sc, &m, r.x, &hc_bad, eps);
        try testing.expect(!try close("  planted: mix scale dropped", z4, bad, null, tol_kernel));
    }
    try testing.expect(ok);
}

// ── a release-shape pair worker's prefill ──

const switches = .{ .MLX_SERVE_DSV41_MOE_PAIR, .MLX_SERVE_DSV41_MOE_PAIR_SCALED, .MLX_SERVE_DSV41_HC_EXPAND_NORM, .MLX_SERVE_DSV41_HC_MIX8, .MLX_SERVE_DSV41_WOA_JOIN };

/// The pair worker's streams and mixes after its layers over `x` / `pre`, from a fresh state (caller frees).
fn prefill(fx: *fixture.Fixture, x: A, pre: A) ![2]A {
    return dsv41.workerStep(fx.m, testing.allocator, x, pre, 0, false, false);
}

/// A pair worker's layers move its stream through rounded bf16 sums over two layers, so its tolerance is 2% of the
/// layers' change to the stream (and of the mixes).
const tol_layers: f32 = 2e-2;

/// Whether the pair worker's prefill is within `tol_layers` of `want` (the stream measured on its change from `x`).
fn closeAs(fx: *fixture.Fixture, x: A, pre: A, want: [2]A, label: []const u8, what: []const u8) !bool {
    const got = try prefill(fx, x, pre);
    defer for (got) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var ok = true;
    for (want, got, 0..) |w, g, k| {
        var lb: [128]u8 = undefined;
        const ref: ?A = if (k == 0 and std.mem.eql(c_int, mlx.getShape(w), mlx.getShape(x))) x else null;
        if (!try close(try std.fmt.bufPrint(&lb, "wide layer {s}, {s}, output {d}", .{ label, what, k }), w, g, ref, tol_layers)) ok = false;
    }
    return ok;
}

test "dsv41 wide tol: each switch, and all of them, keep a release-shape pair worker's prefill streams and mixes within 2% of the stock path's, MXFP4 and NVFP4 experts with per-expert scales; the routed term at half its scale fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    inline for (switches) |n| env.set(n, "0");
    // A ratio-2 pair (layers 8, 9) and a ratio-1 pair (20, 21), each range its own KV and index source and no Engram
    // layer (14 reads Engram, which a pair worker's fixture does not hold).
    const Case = struct { lo: usize, rows: usize };
    // Every case and switch runs; each out of tolerance is reported.
    var ok = true;
    for ([_]Case{ .{ .lo = 8, .rows = 5000 }, .{ .lo = 20, .rows = 8192 } }) |c| inline for (.{ false, true }) |nv| {
        var lb: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&lb, "{d}-{d} {s} at {d} rows", .{ c.lo, c.lo + 1, if (nv) "NVFP4" else "MXFP4", c.rows });
        const fx = fixture.Fixture.init(gpa, .{ .lo = c.lo, .hi = c.lo + 2, .ctx = 0, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none }) catch |e| {
            std.debug.print("wide layer {s}: the fixture refused: {s}\n", .{ label, @errorName(e) });
            return e;
        };
        defer fx.deinit();
        const x = try fx.stream(c.rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try fx.cpre(c.rows);
        defer _ = mlx.mlx_array_free(pre);
        const want = prefill(fx, x, pre) catch |e| {
            std.debug.print("wide layer {s}: the stock prefill failed: {s}\n", .{ label, @errorName(e) });
            return e;
        };
        defer for (want) |o| {
            _ = mlx.mlx_array_free(o);
        };
        // The layer's loaded experts take the paired kernel under their kind's switch.
        {
            var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer sc.deinit();
            const ly = fx.layer(c.lo);
            const h = try normal(&sc, &.{ @intCast(c.rows), @intCast(hidden) }, .bfloat16, 1, 0x77);
            const r = dsv41.route(&sc, fx.m, ly, h, none) catch |e| {
                std.debug.print("wide layer {s}: the route failed: {s}\n", .{ label, @errorName(e) });
                return e;
            };
            env.set(if (nv) .MLX_SERVE_DSV41_MOE_PAIR_SCALED else .MLX_SERVE_DSV41_MOE_PAIR, "1");
            defer env.set(if (nv) .MLX_SERVE_DSV41_MOE_PAIR_SCALED else .MLX_SERVE_DSV41_MOE_PAIR, "0");
            const served = (actOf(&sc, fx.m, &ly.experts, h, r.idx, r.wts) catch |e| {
                std.debug.print("wide layer {s}: the paired kernel failed: {s}\n", .{ label, @errorName(e) });
                return e;
            }) != null;
            if (!served) std.debug.print("wide layer {s}: the paired kernel declines the loaded experts (gu {}, mode {s})\n", .{ label, ly.experts.gu != null, if (ly.experts.gu) |g| @tagName(g.qp.mode) else "-" });
            try testing.expect(served);
        }
        if (!try closeAs(fx, x, pre, want, label, "the stock path again")) ok = false;
        inline for (switches) |n| {
            env.set(n, "1");
            defer env.set(n, "0");
            if (!try closeAs(fx, x, pre, want, label, @tagName(n))) ok = false;
        }
        inline for (switches) |n| env.set(n, "1");
        defer {
            inline for (switches) |n| env.set(n, "0");
        }
        if (!try closeAs(fx, x, pre, want, label, "every switch")) ok = false;
        // Planted: every switch on with the routed experts' sum at half its scale.
        const scale = fx.m.cfg.router_scaling_factor;
        fx.m.cfg.router_scaling_factor = scale / 2;
        defer fx.m.cfg.router_scaling_factor = scale;
        try testing.expect(!try closeAs(fx, x, pre, want, label, "planted: the routed term at half its scale"));
    };
    try testing.expect(ok);
}
