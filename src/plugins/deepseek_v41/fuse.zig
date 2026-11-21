//! Launch fusions, each off by default behind its own switch (`Fuse`, read at load):
//! - `down_scale`: a decode step or verify puts an NVFP4 bank's per-expert down scale on the routing weights inside the
//!   fused fp4 gate/up launch (`DSCALE`: `wts * dsc[e]`, the f32 product `routeScaled`'s take, cast and multiply make).
//! - `engram`: a decode step's or verify's Engram write takes its wkv projection straight from the table's codes in one
//!   launch (`decode/engram_kv.zig`: no dequant, cast or f32 rows in memory), then the gate; a prompt chunk's rows
//!   dequantize straight to the activation dtype (`Table.dequantTo`), not to f32 and a cast. Not bit-equal: the
//!   projection sums in another order.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const dsv41 = @import("deepseek_v41.zig");
const dkern = @import("decode/decode.zig");
const engram = @import("engram.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Fuse = struct {
    down_scale: bool = false,
    engram: bool = false,

    pub fn wanted() Fuse {
        return .{
            .down_scale = env.on(.MLX_SERVE_DSV41_FUSE_DOWN_SCALE),
            .engram = env.on(.MLX_SERVE_DSV41_FUSE_ENGRAM),
        };
    }
};

/// The fusions that have served, each logged the first time (an A/B arm's engagement lines).
pub var engaged: struct { down_scale: bool = false, engram: bool = false, engram_rows: bool = false } = .{};

fn engage(comptime which: []const u8, comptime what: []const u8) void {
    if (@field(engaged, which)) return;
    @field(engaged, which) = true;
    log.info("deepseek_v41: fused " ++ what ++ " engaged\n", .{});
}

/// `ffnDecode` over routes whose weights `wts` do not hold the layer's down scale: the fused fp4 launches with the
/// scale applied inside the gate/up launch. Null where the layer has no down scale or a launch declines; the caller
/// then scales the route.
pub fn ffnDownScaled(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h: A, idx: A, wts: A, x1: A, sp: *const dsv41.Split) !?A {
    const ds = ly.experts.scale[2] orelse return null;
    const kf = dsv41.fp4Decode(m) orelse return null;
    const act = try dkern.moe_fp4.moeUpScaled(sc, m, kf, ly, h, idx, wts, .{ .gate = ly.experts.scale[0], .up = ly.experts.scale[1], .down = ds }) orelse return null;
    const y = try dkern.moe_fp4.moeDown(sc, kf, ly, act, idx, x1, sp) orelse return null;
    engage("down_scale", "down scale (an NVFP4 bank's per-expert down scale inside the fp4 gate/up launch)");
    return y;
}

/// An Engram layer's kv `[rows, (hc + 1) * dim]` from the table's raw rows `raw` (`rows * K / head_dim` of them, a
/// row's n-gram rows in order) on the one-launch kernel. Null where it does not serve: the table not mxfp4 or mxfp8 at
/// group 32, wkv not mxfp8 group 32 in MLX arrays, activations not bf16, more rows than a decode step.
pub fn engramKv(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, raw: engram.Table.Raw, rows: usize) !?A {
    const t = &e.table;
    const l = &e.wkv;
    if (m.act != .bfloat16 or rows < 1 or rows > dkern.max_rows) return null;
    if (!((t.mode == .mxfp4 and t.bits == 4) or (t.mode == .mxfp8 and t.bits == 8)) or t.group_size != 32) return null;
    if (l.ext != null or l.s.ctx == null or l.b.ctx != null or l.qp.mode != .mxfp8 or l.qp.bits != 8 or l.qp.group_size != 32) return null;
    if (mlx.getShape(l.w).len != 2 or t.head_dim == 0 or t.wlen != t.head_dim * t.bits / 8 or t.slen != t.head_dim / 32) return null;
    const K: usize = @intCast(mlx.getShape(l.w)[1] * 4);
    if (K % t.head_dim != 0) return null;
    const n = rows * K / t.head_dim;
    if (raw.w.len < n * t.wlen or raw.s.len < n * t.slen) return null;
    const codes = try sc.keep(mlx.mlx_array_new_data(raw.w.ptr, &[_]c_int{ ci(n), ci(t.wlen / 4) }, 2, .uint32));
    const tsc = try sc.keep(mlx.mlx_array_new_data(raw.s.ptr, &[_]c_int{ ci(n), ci(t.slen) }, 2, .uint8));
    return dkern.engram_kv.launch(sc, .{ .hd = t.head_dim, .tb = t.bits }, rows, codes, tsc, l.w, l.s);
}

/// Under `engram`: the Engram write of the stream `x` `[rows, hc, dim]` from the table's raw rows (`engramKv`, then the
/// gate). Null where the switch is off or the kernel declines; the caller then dequantizes the rows.
pub fn engramWrite(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, raw: engram.Table.Raw, x: A) !?A {
    if (!m.fuse.engram) return null;
    const kv = try engramKv(sc, m, e, raw, @intCast(mlx.getShape(x)[0])) orelse return null;
    engage("engram", "Engram write (the wkv projection from the table's codes in one launch)");
    return try dsv41.engramGate(sc, m, e.qk, x, kv);
}

/// The dtype the table `t`'s rows dequantize into: the activations' under `engram` where the codes allow it.
pub fn engramRowsDtype(m: *const dsv41.Dsv41Model, t: *const engram.Table) mlx.mlx_dtype {
    if (!m.fuse.engram or !t.dequantsTo()) return .float32;
    engage("engram_rows", "Engram rows (a prompt chunk's rows straight to the activation dtype)");
    return m.act;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("decode/fixture.zig");
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;

fn free(a: A) void {
    _ = mlx.mlx_array_free(a);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// Elements whose bits differ (f32 of bf16 or f32 values, exact either way). The reference must be finite: two equal
/// NaN rows would pass vacuously.
fn differing(got: A, want: A) !usize {
    const g = try dsv41.hostF32(got);
    defer testing.allocator.free(g);
    const w = try dsv41.hostF32(want);
    defer testing.allocator.free(w);
    if (g.len != w.len) return error.ShapeMismatch;
    var n: usize = 0;
    for (g, w) |x, y| {
        if (!std.math.isFinite(y)) return error.NonFinite;
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) n += 1;
    }
    return n;
}

test "dsv41 fuse down_scale: the fp4 launches over unscaled routes, the down scale inside, give ffnDecode's bytes over the scaled routes, R = 1..6 (NVFP4 with per-expert scales); the gate's scales in its place differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const sw = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer sw.restore();
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 16, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    var bare = ly.*;
    bare.experts.scale = .{ null, null, null };
    var wrong = ly.*;
    wrong.experts.scale[2] = ly.experts.scale[0];
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const MR = dkern.max_rows;
    const hc = m.cfg.dsv4_hc_mult;
    const h = try sc.keep(try fx.normal(&.{ ci(MR), ci(m.cfg.hidden_size) }, .bfloat16, 0, 1));
    const raw = try dsv41.route(&sc, m, &bare, h, none);
    const scaled = try dsv41.route(&sc, m, ly, h, none);
    const x1 = try sc.keep(try fx.stream(MR));
    const post = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc) }, 0.05, 1.5));
    const comb = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc), ci(hc) }, -0.5, 0.5));
    var differ: usize = 0;
    var plants_equal: usize = 0;
    for (1..MR + 1) |R| {
        const sp: dsv41.Split = .{ .pre = none, .post = try sc.slice(post, 0, 0, R), .comb = try sc.slice(comb, 0, 0, R) };
        const hr = try sc.slice(h, 0, 0, R);
        const xr = try sc.slice(x1, 0, 0, R);
        const idx = try sc.slice(scaled.idx, 0, 0, R);
        const want = (try dsv41.ffnDecode(&sc, m, ly, hr, idx, try sc.slice(scaled.wts, 0, 0, R), xr, &sp)) orelse return error.Declined;
        const got = (try ffnDownScaled(&sc, m, ly, hr, idx, try sc.slice(raw.wts, 0, 0, R), xr, &sp)) orelse return error.Declined;
        const plant = (try ffnDownScaled(&sc, m, &wrong, hr, idx, try sc.slice(raw.wts, 0, 0, R), xr, &sp)) orelse return error.Declined;
        const d = try differing(got, want);
        const p = try differing(plant, want);
        std.debug.print("[fuse] down_scale R={d}: {d} elements differ from the scaled route's; with the gate's scales as the down's, {d}\n", .{ R, d, p });
        differ += d;
        if (p == 0) plants_equal += 1;
    }
    try testing.expectEqual(@as(usize, 0), differ);
    try testing.expectEqual(@as(usize, 0), plants_equal);
}

test "dsv41 fuse down_scale: decode steps of 1..6 rows through two layers give the control's streams and mixes bit for bit with the fusion on (NVFP4 with per-expert scales; MXFP4, where it stays inert)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const sw = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer sw.restore();
    var bad: usize = 0;
    for ([_]bool{ true, false }) |nv| {
        const opts: fixture.Opts = .{ .lo = 20, .hi = 22, .experts = 16, .ctx = 300, .seed = 5, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none, .fused = true };
        const a = try fixture.Fixture.init(gpa, opts);
        defer a.deinit();
        const b = try fixture.Fixture.init(gpa, opts);
        defer b.deinit();
        b.m.fuse.down_scale = true;
        engaged.down_scale = false;
        var p0: usize = opts.ctx;
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 1 }) |R| {
            const x = try a.stream(R);
            defer free(x);
            const pre = try a.cpre(R);
            defer free(pre);
            const oa = try dsv41.workerStep(a.m, gpa, x, pre, p0, false, false);
            defer for (oa) |o| free(o);
            const ob = try dsv41.workerStep(b.m, gpa, x, pre, p0, false, false);
            defer for (ob) |o| free(o);
            const ds = try differing(ob[0], oa[0]);
            const dp = try differing(ob[1], oa[1]);
            std.debug.print("[fuse] down_scale {s} R={d} at {d}: stream {d}, mix {d} elements differ\n", .{ if (nv) "nvfp4+scales" else "mxfp4", R, p0, ds, dp });
            bad += ds + dp;
            p0 += R;
        }
        try testing.expectEqual(nv, engaged.down_scale);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// Random table rows: e2m1 codes (any nibble) or e4m3 bytes (never the NaN code), e8m0 scales 2^-7 .. 2^-1.
fn rawRows(gpa: std.mem.Allocator, t: *const engram.Table, n: usize, rng: std.Random) !engram.Table.Raw {
    const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", n * t.wlen), .s = try gpa.alignedAlloc(u8, .@"4", n * t.slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
    for (raw.w) |*b| b.* = if (t.bits == 4) rng.int(u8) else @as(u8, rng.uintLessThan(u8, 2)) << 7 | (2 + rng.uintLessThan(u8, 12)) << 3 | rng.uintLessThan(u8, 8);
    for (raw.s) |*b| b.* = 120 + rng.uintLessThan(u8, 7);
    return raw;
}

fn dupRaw(gpa: std.mem.Allocator, r: engram.Table.Raw) !engram.Table.Raw {
    const out: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", r.w.len), .s = try gpa.alignedAlloc(u8, .@"4", r.s.len), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
    @memcpy(out.w, r.w);
    @memcpy(out.s, r.s);
    return out;
}

fn hostOf(a: A) ![]f32 {
    return dsv41.hostF32(a);
}

test "dsv41 fuse engram: the one-launch wkv from the table's codes is no further from f32 truth than the dequant, cast and quantized_matmul chain, and the gated write within 0.1 rel rms of the chain's, R = 1..6, mxfp4 and mxfp8 tables at the release's shape; a scale a code up, two rows swapped differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    m.fuse.engram = true;
    defer m.fuse.engram = false;
    const s = mlx.gpuStream();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    const hd: usize = 256;
    const K: usize = 24 * hd;
    const N = (hc + 1) * d;
    const ew = try fx.normal(&.{ ci(N), ci(K) }, .bfloat16, 0, 0.02);
    defer free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) free(q);
    var wd = mlx.mlx_array_new();
    defer free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, eq[0], eq[1], none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, s));
    var wdt = mlx.mlx_array_new();
    defer free(wdt);
    try mlx.check(mlx.mlx_transpose(&wdt, wd, s));
    const qk = try fx.normal(&.{ ci(hc), ci(d) }, .float32, 0.5, 0.2);
    defer free(qk);
    var prng = std.Random.DefaultPrng.init(0xE6);
    const rng = prng.random();
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    for ([_]QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        const t: engram.Table = .{ .fd = -1, .rows = 1 << 20, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen };
        const e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = qk, .table = t };
        for (1..dkern.max_rows + 1) |R| {
            const n = R * K / hd;
            const raw = try rawRows(gpa, &t, n, rng);
            defer raw.deinit(gpa);
            const rows = try sc.keep(try t.dequant(raw, n, s));
            const r2 = try sc.reshape(rows, &.{ ci(R), ci(K) });
            const got = (try engramKv(&sc, m, &e, raw, R)) orelse return error.Declined;
            const chain = try e.wkv.apply(&sc, try sc.astype(r2, .bfloat16));
            var truth = mlx.mlx_array_new();
            defer free(truth);
            try mlx.check(mlx.mlx_matmul(&truth, r2, wdt, cpu));
            const g = try hostOf(got);
            defer gpa.free(g);
            const c = try hostOf(chain);
            defer gpa.free(c);
            const tr = try hostOf(truth);
            defer gpa.free(tr);
            var label_buf: [64]u8 = undefined;
            try dkern.expectNoWorse(try std.fmt.bufPrint(&label_buf, "fuse engram kv {t} R={d}", .{ mode, R }), g, c, tr);
            // Plants: every scale a code up (the rows doubled); rows 0 and 1 swapped.
            const up = try dupRaw(gpa, raw);
            defer up.deinit(gpa);
            for (up.s) |*b| b.* += 1;
            const pu = try hostOf((try engramKv(&sc, m, &e, up, R)).?);
            defer gpa.free(pu);
            const eu = dkern.relRms(pu, c);
            var es: f64 = 1;
            if (R >= 2) {
                const sw = try dupRaw(gpa, raw);
                defer sw.deinit(gpa);
                const per = K / hd;
                @memcpy(sw.w[0 .. per * wlen], raw.w[per * wlen ..][0 .. per * wlen]);
                @memcpy(sw.w[per * wlen ..][0 .. per * wlen], raw.w[0 .. per * wlen]);
                @memcpy(sw.s[0 .. per * slen], raw.s[per * slen ..][0 .. per * slen]);
                @memcpy(sw.s[per * slen ..][0 .. per * slen], raw.s[0 .. per * slen]);
                const ps = try hostOf((try engramKv(&sc, m, &e, sw, R)).?);
                defer gpa.free(ps);
                es = dkern.relRms(ps, c);
            }
            // The gated write against the chain's (`engramApply` over the dequantized rows), on the increment.
            const x = try sc.keep(try fx.normal(&.{ ci(R), ci(hc), ci(d) }, .bfloat16, 0, 1));
            engaged.engram = false;
            const yg = (try engramWrite(&sc, m, &e, raw, x)) orelse return error.Declined;
            try testing.expect(engaged.engram);
            const yw = try dsv41.engramApply(&sc, m, &e, x, r2);
            const xh = try hostOf(x);
            defer gpa.free(xh);
            const ygh = try hostOf(yg);
            defer gpa.free(ygh);
            const ywh = try hostOf(yw);
            defer gpa.free(ywh);
            for (ygh, ywh, xh) |*a, *b, xv| {
                a.* -= xv;
                b.* -= xv;
            }
            const ew_ = dkern.relRms(ygh, ywh);
            std.debug.print("[fuse] engram {t} R={d}: write increment {e:.2} from the chain's (bar 0.1); plants: scales up {e:.2}, rows swapped {e:.2} (bar > 0.5)\n", .{ mode, R, ew_, eu, es });
            try testing.expect(ew_ <= 0.1);
            try testing.expect(eu > 0.5 and es > 0.5);
        }
    }
}

test "dsv41 fuse engram: a prompt chunk's mxfp4 and mxfp8 rows dequantize straight to bf16 as the f32 rows cast to bf16, every code under every e8m0 scale; the scales a code up differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    const hd: usize = 256;
    const n: usize = 256;
    for ([_]QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        const t: engram.Table = .{ .fd = -1, .rows = n, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen };
        try testing.expect(t.dequantsTo());
        // Row r: e8m0 code r on each of its groups; element j code j mod 2^bits, so each group runs through every code.
        const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", n * wlen), .s = try gpa.alignedAlloc(u8, .@"4", n * slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
        defer raw.deinit(gpa);
        @memset(raw.w, 0);
        for (0..n) |r| {
            @memset(raw.s[r * slen ..][0..slen], @intCast(r));
            for (0..hd) |j| {
                const c: u8 = @intCast(j % (@as(usize, 1) << @intCast(bits)));
                if (bits == 8) raw.w[r * wlen + j] = c else raw.w[r * wlen + j / 2] |= c << @intCast(4 * (j % 2));
            }
        }
        const rows = try t.dequant(raw, n, s);
        defer free(rows);
        var want = mlx.mlx_array_new();
        defer free(want);
        try mlx.check(mlx.mlx_astype(&want, rows, .bfloat16, s));
        const got = try t.dequantTo(raw, n, .bfloat16, s);
        defer free(got);
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got));
        for (raw.s) |*c| c.* +%= 1;
        const plant = try t.dequantTo(raw, n, .bfloat16, s);
        defer free(plant);
        const wb = try dkern.moe_up.bitsOf(want);
        defer gpa.free(wb);
        const gb = try dkern.moe_up.bitsOf(got);
        defer gpa.free(gb);
        const pb = try dkern.moe_up.bitsOf(plant);
        defer gpa.free(pb);
        var differ: usize = 0;
        var planted: usize = 0;
        for (wb, gb, pb) |w, g, p| {
            const w_nan = (w & 0x7f80) == 0x7f80 and (w & 0x7f) != 0;
            const g_nan = (g & 0x7f80) == 0x7f80 and (g & 0x7f) != 0;
            if (w != g and !(w_nan and g_nan)) differ += 1;
            if (p != w) planted += 1;
        }
        std.debug.print("[fuse] engram rows {t}: {d} of {d} bf16 values differ from the f32 rows cast; with the scales a code up, {d}\n", .{ mode, differ, wb.len, planted });
        try testing.expectEqual(@as(usize, 0), differ);
        try testing.expect(planted > 0);
    }
}
