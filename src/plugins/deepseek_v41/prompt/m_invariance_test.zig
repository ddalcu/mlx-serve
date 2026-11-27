//! A wide prompt layer's row-wise stages at the release's shapes over its 384 experts: whether a row's words are the
//! same whatever rows share the launch (M-invariance), the premise of a layer-major prefill that runs them once over
//! every chunk's rows. Each stage runs over 17,639 rows, then over row ranges of them as launches of their own: 1,126
//! and 13,158 rows, and the 17.6K card's chunks (4,096, 4,096, 8,192 and its last 1,255). Every stage reads the same
//! inputs in both: the whole launch's, sliced.
//!
//! Every stage holds but the router: under 1,707 rows its logits come from MLX's split-K matmul (K = 5,120 >= 3 x the
//! larger of M and 384), from 1,707 on from `gate_gemm`'s regular-tile launch, so a chunk under 1,707 rows routes apart
//! from the same rows in a wider launch. Below `DENSE_ROWS` (1,024) every linear takes another kernel too.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const dsv41 = @import("../deepseek_v41.zig");
const fixture = @import("../decode/fixture.zig");
const env = @import("../../../env.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const testing = std.testing;
const Scope = dsv41.Scope;

const full_rows: usize = 17_639;
const ranges = [_][2]usize{ .{ 0, 1126 }, .{ 0, 4096 }, .{ 4096, 8192 }, .{ 8192, 16384 }, .{ 16384, full_rows }, .{ full_rows - 13_158, full_rows } };
const stages = [_][]const u8{ "hc attn mix+norm", "q_kv+wq_b+kv row+rope", "wo_b", "hc expand+ffn norm", "router", "routed+shared experts", "hc expand" };
const router = 4;
/// The rows from which MLX runs the router's [rows, 5,120] x [5,120, 384] on its regular kernel (`gate_gemm.mlxRegular`).
const split_k_rows = 1707;

/// `x`'s words, contiguous (caller frees).
fn words(x: A) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return testing.allocator.dupe(u8, p[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)]);
}

/// Elements of `got` (a launch over rows [lo, hi)) whose words differ from those rows of `whole` (the whole launch's).
fn differing(sc: *Scope, whole: A, lo: usize, hi: usize, got: A) !usize {
    const w = try words(try sc.slice(whole, 0, lo, hi));
    defer testing.allocator.free(w);
    const g = try words(got);
    defer testing.allocator.free(g);
    if (w.len != g.len) return error.ShapeMismatch;
    const isz = mlx.mlx_array_itemsize(got);
    var n: usize = 0;
    var i: usize = 0;
    while (i < g.len) : (i += isz) {
        if (!std.mem.eql(u8, w[i..][0..isz], g[i..][0..isz])) n += 1;
    }
    return n;
}

/// Rows [lo, hi) of `x` as an array of their own.
fn rowsOf(sc: *Scope, x: A, lo: usize, hi: usize) !A {
    var c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&c, try sc.slice(x, 0, lo, hi), false, sc.s));
    _ = try sc.keep(c);
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

/// One layer's stage inputs over the whole launch, each stage's taken from the stage before it.
const Ins = struct { x: A, pres: A, h1: A, sp1: dsv41.Split, o: A, y: A, x1: A, sp4: dsv41.Split, h4: A, idx: A, wts: A, e: A };

/// Stage `k` over rows [lo, hi) of `in` (all of them: the whole launch), its outputs in `out`; returns how many.
fn stage(sc: *Scope, fx: *fixture.Fixture, li: usize, k: usize, in: *const Ins, lo: usize, hi: usize, out: *[4]A) !usize {
    const m = fx.m;
    const ly = fx.layer(li);
    const whole = lo == 0 and hi == full_rows;
    const r = struct {
        fn of(s: *Scope, x: A, a: usize, b: usize, w: bool) !A {
            return if (w) x else rowsOf(s, x, a, b);
        }
    }.of;
    switch (k) {
        0 => {
            const pa = (try dsv41.attnHcWide(sc, m, try r(sc, in.x, lo, hi, whole), try r(sc, in.pres, lo, hi, whole), ly, null)).pa;
            out.* = .{ pa.h, pa.sp.pre, pa.sp.post, pa.sp.comb };
            return 4;
        },
        1 => {
            const q_tail = m.attn_gather != null and m.cfg.dsv4_rope_head_dim == 64 and m.cfg.head_dim % 64 == 0;
            const pj = try dsv41.qkvProjWith(sc, m, ly, dsv41.ropeFor(m, li).fwd, try r(sc, in.h1, lo, hi, whole), try sc.i(@intCast(lo)), q_tail);
            out.* = .{ pj.q, pj.kv, pj.qr, if (pj.qt.ctx != null) pj.qt else pj.q };
            return 4;
        },
        2 => {
            out[0] = try ly.wo_b.apply(sc, try r(sc, in.o, lo, hi, whole));
            return 1;
        },
        3 => {
            const sp: dsv41.Split = .{ .pre = try r(sc, in.sp1.pre, lo, hi, whole), .post = try r(sc, in.sp1.post, lo, hi, whole), .comb = try r(sc, in.sp1.comb, lo, hi, whole) };
            const pf = try dsv41.hcExpandPreNorm(sc, m, try r(sc, in.x, lo, hi, whole), try r(sc, in.y, lo, hi, whole), &sp, sp.pre, &ly.hc_ffn, ly.ffn_norm);
            out.* = .{ pf.x1, pf.h, pf.sp.post, pf.sp.comb };
            return 4;
        },
        4 => {
            const rt = try dsv41.route(sc, m, ly, try r(sc, in.h4, lo, hi, whole), none);
            out[0] = rt.idx;
            out[1] = rt.wts;
            return 2;
        },
        5 => {
            out[0] = try sc.keep(try dsv41.moeShare(m, testing.allocator, li, try r(sc, in.h4, lo, hi, whole), try r(sc, in.idx, lo, hi, whole), try r(sc, in.wts, lo, hi, whole)));
            return 1;
        },
        6 => {
            const sp: dsv41.Split = .{ .pre = none, .post = try r(sc, in.sp4.post, lo, hi, whole), .comb = try r(sc, in.sp4.comb, lo, hi, whole) };
            out[0] = try dsv41.hcExpand(sc, try r(sc, in.x1, lo, hi, whole), try r(sc, in.e, lo, hi, whole), &sp);
            return 1;
        },
        else => unreachable,
    }
}

/// The whole launch's stage inputs, each stage over the last's outputs (`sc` holds them).
fn inputs(sc: *Scope, fx: *fixture.Fixture, li: usize) !Ins {
    const m = fx.m;
    const ly = fx.layer(li);
    var in: Ins = undefined;
    in.x = try sc.keep(try fx.stream(full_rows));
    in.pres = try sc.keep(try fx.cpre(full_rows));
    const pa = (try dsv41.attnHcWide(sc, m, in.x, in.pres, ly, null)).pa;
    in.h1 = pa.h;
    in.sp1 = pa.sp;
    const og_in: c_int = @intCast(m.cfg.dsv4_o_groups * m.cfg.dsv4_o_lora_rank);
    in.o = try sc.keep(try fx.normal(&.{ @intCast(full_rows), og_in }, .bfloat16, 0, 1));
    in.y = try ly.wo_b.apply(sc, in.o);
    const pf = try dsv41.hcExpandPreNorm(sc, m, in.x, in.y, &in.sp1, in.sp1.pre, &ly.hc_ffn, ly.ffn_norm);
    in.x1 = pf.x1;
    in.sp4 = pf.sp;
    in.h4 = pf.h;
    const rt = try dsv41.route(sc, m, ly, in.h4, none);
    in.idx = rt.idx;
    in.wts = rt.wts;
    in.e = try sc.keep(try dsv41.moeShare(m, testing.allocator, li, in.h4, in.idx, in.wts));
    try dsv41.evalAll(&.{ in.x, in.pres, in.h1, in.sp1.pre, in.sp1.post, in.sp1.comb, in.o, in.y, in.x1, in.sp4.post, in.sp4.comb, in.h4, in.idx, in.wts, in.e });
    return in;
}

test "dsv41 prefill rows (DSV41_PREFILL_ROWS): each row-wise stage of a wide layer gives a row the same words in a launch of 17,639 rows as in one of 1,126, 4,096, 8,192, 1,255 or 13,158, but the router under 1,707 rows (384 NVFP4 experts with their scales, 384 MXFP4)" {
    if (!env.present(.DSV41_PREFILL_ROWS)) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const li: usize = 2;
    var total: [stages.len]usize = @splat(0);
    for ([_]bool{ true, false }) |nv| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = li, .hi = li + 1, .experts = 384, .ctx = 0, .experts_nvfp4 = nv, .experts_fp4 = !nv, .expert_scales = if (nv) .all else .none });
        defer fx.deinit();
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const in = try inputs(&sc, fx, li);
        for (0..stages.len) |k| {
            var ssc = Scope.init(gpa, mlx.gpuStream());
            defer ssc.deinit();
            var whole: [4]A = undefined;
            const n = try stage(&ssc, fx, li, k, &in, 0, full_rows, &whole);
            try dsv41.evalAll(whole[0..n]);
            for (ranges) |rg| {
                var rsc = Scope.init(gpa, mlx.gpuStream());
                defer rsc.deinit();
                var got: [4]A = undefined;
                _ = try stage(&rsc, fx, li, k, &in, rg[0], rg[1], &got);
                var diff: usize = 0;
                var all: usize = 0;
                for (whole[0..n], got[0..n]) |w, g| {
                    diff += try differing(&rsc, w, rg[0], rg[1], g);
                    all += mlx.mlx_array_size(g);
                }
                std.debug.print("[minv] {s} {s}: rows [{d}, {d}) ({d}): {d} of {d} words differ\n", .{ if (nv) "nvfp4" else "mxfp4", stages[k], rg[0], rg[1], rg[1] - rg[0], diff, all });
                // The router's split K below 1,707 rows routes apart: the bar's planted difference.
                if (k == router and rg[1] - rg[0] < split_k_rows) try testing.expect(diff > 0) else total[k] += diff;
            }
        }
        // A planted control: a launch whose input row 0 is negated differs at that row.
        var psc = Scope.init(gpa, mlx.gpuStream());
        defer psc.deinit();
        const o = try rowsOf(&psc, in.o, 0, 1126);
        const flip = try psc.ints(&.{ -1, 1 }, &.{ 2, 1 });
        const sign = try psc.astype(try psc.concat(&.{ try psc.slice(flip, 0, 0, 1), try psc.broadcast(try psc.slice(flip, 0, 1, 2), &.{ 1125, 1 }) }, 0), .bfloat16);
        const planted = try fx.layer(li).wo_b.apply(&psc, try psc.mul(o, sign));
        var wsc = Scope.init(gpa, mlx.gpuStream());
        defer wsc.deinit();
        const want = try fx.layer(li).wo_b.apply(&wsc, in.o);
        try testing.expect(try differing(&psc, want, 0, 1126, planted) > 0);
    }
    for (stages, total) |name, t| std.debug.print("[minv] {s}: {d} words differ over both formats and every range it holds at\n", .{ name, t });
    for (total) |t| try testing.expectEqual(@as(usize, 0), t);
}
