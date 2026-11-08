//! DeepSeek-V4.1's decode kernels, one file a kernel (`~/plans/dsv41-decode-kernels-spec.md`): a layer's 1..6 rows in
//! about ten dependent launches, each at its weights' byte floor. What their tests and ubenches share lives here.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");

pub const fixture = @import("fixture.zig");
pub const in_proj = @import("in_proj.zig");
pub const wq_b = @import("wq_b.zig");
pub const idx_score = @import("idx_score.zig");
pub const topk = @import("topk.zig");
pub const attn = @import("attn.zig");
pub const wo_a = @import("wo_a.zig");
pub const ffn_in = @import("ffn_in.zig");
pub const moe_up = @import("moe_up.zig");
pub const moe_fp4 = @import("moe_fp4.zig");
pub const moe_down = @import("moe_down.zig");

const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

/// The most rows a decode kernel serves in one launch: a plain step, or a DSpark verify block.
pub const max_rows = 6;

/// What this machine streams from memory in practice (a plain reduction on the M5 Ultra), bytes a microsecond.
pub const stream_bytes_per_us: f64 = 1.065e6;

pub fn register(set: *kc.Set) !void {
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, ffn_in, moe_up, moe_fp4, moe_down }) |k| try k.register(set);
}

/// Root-mean-square of `got - want` over that of `want`.
pub fn relRms(got: []const f32, want: []const f32) f64 {
    var e: f64 = 0;
    var t: f64 = 0;
    for (got, want) |g, w| {
        const d = @as(f64, g) - @as(f64, w);
        e += d * d;
        t += @as(f64, w) * @as(f64, w);
    }
    return @sqrt(e / @max(t, 1e-300));
}

/// The fusion bar: `new` is no further from `truth` than today's `chain` (1.25x, plus 1e-4 of truth's scale).
pub fn expectNoWorse(label: []const u8, new: []const f32, chain: []const f32, truth: []const f32) !void {
    const en = relRms(new, truth);
    const ec = relRms(chain, truth);
    std.debug.print("[dec] {s}: new-truth {e:.2}, chain-truth {e:.2} (rel rms)\n", .{ label, en, ec });
    for (new) |v| if (!std.math.isFinite(v)) return error.NonFinite;
    if (en > 1.25 * ec + 1e-4) return error.FurtherFromTruth;
}

/// `DSV41_ATTN_BW` (on; =0 keeps the first attention-side launches).
pub fn attnBw() bool {
    const v = env.get(.DSV41_ATTN_BW);
    return v == null or v.?[0] != '0';
}

/// The ubenches run only with `DSV41_DEC_UBENCH` set (under `~/claude-tmp/dsdec/with.sh gpu`).
pub fn ubenchOn() bool {
    return env.get(.DSV41_DEC_UBENCH) != null;
}

/// Microseconds a step of a dependent chain: `ctx.step(i)` builds step i's graph (fed by step i - 1's output) and
/// returns the array the chain ends on; `n` steps evaluated at once, median of 7 chains after 2 warm-up chains.
pub fn chainUs(ctx: anytype, n: usize) !f64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        const t0 = iou.nowUsMonotonic(io);
        var last: A = .{ .ctx = null };
        for (0..n) |i| last = try ctx.step(i);
        try mlx.check(mlx.mlx_array_eval(last));
        const us = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / @as(f64, @floatFromInt(n));
        if (rep >= 2) times[rep - 2] = us;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

/// The ubench line every kernel prints: its time, achieved bandwidth and floor, and the chain it replaces.
pub fn report(name: []const u8, rows: usize, us: f64, bytes: f64, chain_us: f64) void {
    const floor = bytes / stream_bytes_per_us;
    std.debug.print("[dec-ubench] {s} R={d}: {d:.1} us, {d:.0} GB/s ({d:.0}% of floor {d:.1} us), replaced chain {d:.1} us\n", .{ name, rows, us, bytes / us / 1e3, 100 * floor / us, floor, chain_us });
}

test "dsv41 dec layers: the fused kernels decode no further from an f32 model than today's chain over seeds, one row and a verify block" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    // Ratio 1 (KV source, owner, candidate source, an index non-owner), ratio 2 (KV source pooling pairs), ratio 0.
    for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 }, .{ 0, 1 } }) |range| try layersAgree(gpa, range[0], range[1], false);
}

test "dsv41 dec layers: the fused kernels over mxfp4 experts decode no further from an f32 model than today's chain" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    // The mxfp4 kernels build only without NAX (`Dec.moe_fp4`): with NAX this compares the NAX attention against the
    // chain at a bf16 near-tie instead. The kernel contract checks the mxfp4 kernels on every GPU.
    if (device.naxAvailable()) return error.SkipZigTest;
    try layersAgree(std.testing.allocator, 20, 23, true);
}

fn layersAgree(gpa: std.mem.Allocator, lo: usize, hi: usize, fp4: bool) !void {
    // One seed lands up to 1.6x either side of the chain (five random layers amplify any rounding; under TF32 the f32
    // model itself moves 8% at the verify step), so the bar is each step's median over seeds.
    const seeds = [_]u64{ 9, 1, 2, 3, 4, 5, 6 };
    const steps = [_]usize{ 1, 1, 1, max_rows, 1 };
    var ratio: [steps.len][seeds.len]f64 = undefined;
    for (seeds, 0..) |seed, si| {
        // Same seed, so the same weights, prefill and step inputs: the fused kernels, today's chain, and the f32 model.
        var fxs: [3]*fixture.Fixture = undefined;
        var made: usize = 0;
        defer for (fxs[0..made]) |fx| fx.deinit();
        for (&fxs, [_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .float32 }, [_]bool{ true, false, false }) |*fx, act, fused| {
            fx.* = try fixture.Fixture.init(gpa, .{ .lo = lo, .hi = hi, .ctx = 601, .seed = seed, .act = act, .fused = fused, .experts_fp4 = fp4 });
            made += 1;
        }
        var p0: usize = 601;
        for (steps, 0..) |rows, k| {
            var got: [3][]f32 = undefined;
            for (fxs, &got) |fx, *g| {
                const x = try fx.stream(rows);
                defer _ = mlx.mlx_array_free(x);
                const pre = try fx.cpre(rows);
                defer _ = mlx.mlx_array_free(pre);
                const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
                defer for (out) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                g.* = try dsv41.hostF32(out[0]);
            }
            defer for (got) |g| gpa.free(g);
            for (got[0]) |v| if (!std.math.isFinite(v)) return error.NonFinite;
            ratio[k][si] = relRms(got[0], got[2]) / @max(relRms(got[1], got[2]), 1e-4);
            p0 += rows;
        }
    }
    for (&ratio, steps) |*r, rows| {
        std.mem.sort(f64, r, {}, std.sort.asc(f64));
        std.debug.print("[dec] layers {d}..{d}, {d} rows: fused/chain distance to the f32 model, median {d:.3} (min {d:.2}, max {d:.2})\n", .{ lo, hi, rows, r[seeds.len / 2], r[0], r[seeds.len - 1] });
        if (r[seeds.len / 2] > 1.15) return error.FurtherFromTruth;
    }
}

test "dsv41 dec layers ubench: a decode step through layers 20..40, fused vs today's chain, 1 and 6 rows" {
    if (!ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    // A fixture an arm: the compiled `.ffn` region keeps the MoE kernels of its first trace.
    for ([_]bool{ true, false }) |fused| {
        // 64 experts a layer: 20 layers hold ~20 GB, so a step's weights come from DRAM, not the SLC.
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 40, .experts = 64, .ctx = 4096, .fused = fused });
        defer fx.deinit();
        var p0: usize = 4096;
        for ([_]usize{ 1, max_rows }) |rows| {
            const x = try fx.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(rows);
            defer _ = mlx.mlx_array_free(pre);
            var times: [9]f64 = undefined;
            for (0..3 + times.len) |rep| {
                const t0 = iou.nowUsMonotonic(io);
                const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
                for (out) |o| _ = mlx.mlx_array_free(o);
                if (rep >= 3) times[rep - 3] = @floatFromInt(iou.nowUsMonotonic(io) - t0);
                p0 += rows;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            std.debug.print("[dec-ubench] layers 20..40 R={d} {s}: {d:.1} us a layer (median step {d:.0} us)\n", .{ rows, if (fused) "fused" else "chain", times[times.len / 2] / 20, times[times.len / 2] });
        }
    }
}

// ── the attention side at a tensor rank's shapes ──

const Shard = @import("../../../sdk/weights/shard.zig").Shard;
/// Rank 0 of four: a quarter of the heads, wq_b and wo_a rows and wo_b columns; q_kv, the hc mixes, the compressor
/// and the indexer whole.
const tp4: Shard = .{ .rank = 0, .size = 4 };
/// Weight sets a chain cycles through, so a step's weights were last read hundreds of MB before it.
const bw_sets = 80;
/// The splits a compressed layer's attention hands wo_a (`attn.splitsFor` of the window and the 512 picks).
const bw_splits = 8;
const bench_rows = [_]usize{ 1, 2, 4, max_rows };

/// `a` in new memory, evaluated.
fn copyOf(sc: *dsv41.Scope, a: A) !A {
    const o = try sc.add(a, try sc.zeros(&.{1}, mlx.mlx_array_dtype(a)));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// `ly` with every attention-side weight in new memory.
fn layerCopy(sc: *dsv41.Scope, ly: *const dsv41.Layer) !dsv41.Layer {
    var c = ly.*;
    const q = &c.q_kv.?;
    q.w = try copyOf(sc, q.w);
    q.s = try copyOf(sc, q.s);
    c.hc_attn.fn_w = try copyOf(sc, c.hc_attn.fn_w);
    inline for (.{ "wq_b", "wo_a", "wo_b" }) |f| {
        @field(c, f).w = try copyOf(sc, @field(c, f).w);
        @field(c, f).s = try copyOf(sc, @field(c, f).s);
    }
    if (c.idx) |*iw| {
        iw.weights_proj.w = try copyOf(sc, iw.weights_proj.w);
        iw.wq_b.w = try copyOf(sc, iw.wq_b.w);
        iw.wq_b.s = try copyOf(sc, iw.wq_b.s);
    }
    if (c.comp) |*cw| cw.wkv.w = try copyOf(sc, cw.wkv.w);
    return c;
}

fn nbytes(a: A) usize {
    return if (a.ctx == null) 0 else mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
}

fn linBytes(l: dsv41.Lin) f64 {
    return @floatFromInt(nbytes(l.w) + nbytes(l.s));
}

/// One arm's pipelines (today's, or `DSV41_ATTN_BW`'s); with `dep` wo_a's takes an unread input a chain hangs on.
const Arm = struct {
    in: in_proj.Kernels,
    wq: wq_b.Kernel,
    oa: wo_a.Kernel,

    fn init(bw: bool, dep: bool) !Arm {
        const in = try in_proj.Kernels.buildArm(bw);
        errdefer in.deinit();
        const wq = try wq_b.buildArm(bw);
        errdefer wq.deinit();
        if (!dep) return .{ .in = in, .wq = wq, .oa = try wo_a.buildArm(bw) };
        const oa = try dsv41.buildKernel(if (bw) "dsv41_dec_wo_a_bw_dep" else "dsv41_dec_wo_a_dep", &(wo_a.INS ++ [_][*:0]const u8{"dep"}), &wo_a.OUTS, if (bw) wo_a.BW_SOURCE else wo_a.SOURCE, wo_a.HEADER);
        return .{ .in = in, .wq = wq, .oa = .{ .k = oa, .bw = bw } };
    }

    fn deinit(a: Arm) void {
        a.in.deinit();
        a.wq.deinit();
        a.oa.deinit();
    }

    /// Every output of the three launches on `ly`, in a fixed order.
    fn outs(p: *const Arm, sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, li: usize, in: *const Ins, counter: A, buf: *[13]A) ![]A {
        const a = (try in_proj.run(sc, &p.in, m, ly, in.x, in.cpre)) orelse return error.Declined;
        const w = (try wq_b.run(sc, m, p.wq, ly, li, in.qa, in.kv, in.z, in.pos)) orelse return error.Declined;
        const o = (try wo_a.apply(sc, m, ly, p.oa, counter, dsv41.ropeFor(m, li).inv, in.meta, in.pacc, in.pm, in.pl)) orelse return error.Declined;
        var n: usize = 0;
        for ([_]?A{ a.h, a.qa, a.kv, a.iw, a.comp, a.z, w.q, w.q_idx, w.kv_row, w.sa.pre, w.sa.post, w.sa.comb, o }) |x| if (x) |v| {
            buf[n] = v;
            n += 1;
        };
        return buf[0..n];
    }
};

/// One row count's inputs for a layer's attention-side launches, and which stretch a chain runs.
const Side = struct {
    const Stretch = enum { in_proj, in_proj_wq_b, wo_a, wo_a_wo_b };

    sc: *dsv41.Scope,
    m: *const dsv41.Dsv41Model,
    li: usize,
    sets: []const dsv41.Layer,
    arm: *const Arm,
    counter: A,
    x: A,
    cpre: A,
    pos: A,
    parts: [3]A,
    meta: A,
    stretch: Stretch = .in_proj,
    prev: A = none,

    fn inProj(c: *Side, ly: *const dsv41.Layer, cpre: A) !in_proj.Out {
        return (try in_proj.run(c.sc, &c.arm.in, c.m, ly, c.x, cpre)) orelse error.Declined;
    }

    fn wqB(c: *Side, ly: *const dsv41.Layer, a: in_proj.Out) !A {
        const o = try wq_b.run(c.sc, c.m, c.arm.wq, ly, c.li, a.qa, a.kv, a.z, c.pos);
        return (o orelse return error.Declined).sa.pre;
    }

    fn woA(c: *Side, ly: *const dsv41.Layer, dep: A) !A {
        const ins = [8]A{ c.parts[0], c.parts[1], c.parts[2], ly.sink, c.meta, dsv41.ropeFor(c.m, c.li).inv, ly.wo_a.w, ly.wo_a.s };
        const rd = c.m.cfg.dsv4_rope_head_dim;
        return (try wo_a.launch(c.sc, c.arm.oa.k, rd, c.m.act, ins, c.counter, dep)) orelse error.Declined;
    }

    pub fn step(c: *Side, i: usize) !A {
        if (i == 0) {
            for (c.sc.items.items) |a| _ = mlx.mlx_array_free(a);
            c.sc.items.clearRetainingCapacity();
            c.prev = if (c.stretch == .in_proj or c.stretch == .in_proj_wq_b) c.cpre else c.meta;
        }
        const ly = &c.sets[i % c.sets.len];
        c.prev = switch (c.stretch) {
            .in_proj => (try c.inProj(ly, c.prev)).z,
            .in_proj_wq_b => try c.wqB(ly, try c.inProj(ly, c.prev)),
            .wo_a => try c.woA(ly, c.prev),
            .wo_a_wo_b => try ly.wo_b.apply(c.sc, try c.woA(ly, c.prev)),
        };
        return c.prev;
    }

    fn us(c: *Side, stretch: Stretch) !f64 {
        c.stretch = stretch;
        return moe_up.deviceUs(c, c.sets.len);
    }

    fn all(c: *Side) ![4]f64 {
        return .{ try c.us(.in_proj), try c.us(.in_proj_wq_b), try c.us(.wo_a), try c.us(.wo_a_wo_b) };
    }
};

/// A layer's 80 weight copies and one row count's inputs, for either arm.
const Bench = struct {
    held: dsv41.Scope,
    own: dsv41.Scope,
    sets: []dsv41.Layer,
    side: Side,

    fn init(gpa: std.mem.Allocator, f: *fixture.Fixture, li: usize, rows: usize, arm: *const Arm) !*Bench {
        const m = f.m;
        const b = try gpa.create(Bench);
        errdefer gpa.destroy(b);
        b.held = dsv41.Scope.init(gpa, m.s);
        errdefer b.held.deinit();
        b.own = dsv41.Scope.init(gpa, m.s);
        b.sets = try gpa.alloc(dsv41.Layer, bw_sets);
        errdefer gpa.free(b.sets);
        for (b.sets) |*st| st.* = try layerCopy(&b.held, f.layer(li));
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(m.cfg.num_attention_heads);
        const parts = [3]A{
            try b.held.keep(try f.normal(&.{ R, H, bw_splits, 512 }, .float32, 0, 1)),
            try b.held.keep(try f.normal(&.{ R, H, bw_splits }, .float32, 0, 1)),
            try b.held.keep(try f.uniform(&.{ R, H, bw_splits }, 0.5, 2)),
        };
        b.side = .{ .sc = &b.own, .m = m, .li = li, .sets = b.sets, .arm = arm, .counter = try b.held.keep(wo_a.newCounter()), .x = try b.held.keep(try f.stream(rows)), .cpre = try b.held.keep(try f.cpre(rows)), .pos = try wq_b.rowsAt(&b.held, 4096, rows), .parts = parts, .meta = try b.held.ints(&.{ 4096, R, 512, 128, 144, 0 }, &.{6}) };
        return b;
    }

    fn deinit(b: *Bench, gpa: std.mem.Allocator) void {
        b.own.deinit();
        b.held.deinit();
        gpa.free(b.sets);
        gpa.destroy(b);
    }
};

const Kind = struct { lo: usize, hi: usize, li: usize, name: []const u8, layers: usize, tp: ?Shard = tp4 };
/// A layer of each attention-side shape and how many of the release's 40 have it, at a rank of four; then a plain
/// layer of the whole model (`layers` 0: outside the rank's sums).
const kinds = [_]Kind{
    .{ .lo = 20, .hi = 25, .li = 21, .name = "plain", .layers = 32 },
    .{ .lo = 20, .hi = 25, .li = 24, .name = "index", .layers = 4 },
    .{ .lo = 20, .hi = 25, .li = 20, .name = "kv ratio 1", .layers = 1 },
    .{ .lo = 2, .hi = 3, .li = 2, .name = "kv ratio 2", .layers = 3 },
    .{ .lo = 20, .hi = 22, .li = 21, .name = "whole model", .layers = 0, .tp = null },
};

test "dsv41 dec attention side: ubench at a tensor-parallel rank of four, rows 1 2 4 6, both arms (DSV41_DEC_UBENCH)" {
    if (!ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var arms: [2]Arm = .{ try Arm.init(false, true), try Arm.init(true, true) };
    defer for (arms) |a| a.deinit();
    var fx: ?*fixture.Fixture = null;
    defer if (fx) |f| f.deinit();
    var fx_kind: ?Kind = null;
    for (kinds) |kind| {
        if (fx_kind == null or fx_kind.?.lo != kind.lo or (fx_kind.?.tp == null) != (kind.tp == null)) {
            if (fx) |f| f.deinit();
            fx = null;
            fx = try fixture.Fixture.init(gpa, .{ .lo = kind.lo, .hi = kind.hi, .experts = 2, .ctx = 0, .fused = true, .tp = kind.tp });
            fx_kind = kind;
        }
        const f = fx.?;
        const ly = f.layer(kind.li);
        for (bench_rows) |rows| {
            const b = try Bench.init(gpa, f, kind.li, rows, &arms[0]);
            defer b.deinit(gpa);
            var raw: [2][4]f64 = @splat(@splat(std.math.inf(f64)));
            for (0..6) |_| for (&arms, &raw) |*arm, *ra| {
                b.side.arm = arm;
                for (ra, try b.side.all()) |*x, v| x.* = @min(x.*, v);
            };
            var t: [2][4]f64 = undefined;
            for (&t, raw) |*ta, ra| ta.* = .{ ra[0], ra[1] - ra[0], ra[2], ra[3] - ra[2] };
            const p = in_proj.plan(f.m, ly, b.side.x).?;
            const idx_b: f64 = if (ly.idx) |iw| linBytes(iw.wq_b) else 0;
            const bytes = [4]f64{ @floatFromInt(in_proj.layerBytes(p)), linBytes(ly.wq_b) + idx_b, linBytes(ly.wo_a), linBytes(ly.wo_b) };
            for ([_][]const u8{ "in_proj", "wq_b", "wo_a", "wo_b" }, bytes, 0..) |name, by, j| {
                if (j >= 2 and kind.layers != 32 and kind.layers != 0) continue;
                std.debug.print("[attn-bw] {s} ({s}) R={d}, {d:.2} MB: today {d:.1} us {d:.0} GB/s, bw {d:.1} us {d:.0} GB/s\n", .{ name, kind.name, rows, by / 1e6, t[0][j], by / t[0][j] / 1e3, t[1][j], by / t[1][j] / 1e3 });
            }
        }
    }
}

/// Elements of `a` and `b` whose bits differ (both widened to f32, exact from bf16).
fn bitsDiffer(a: A, b: A) !usize {
    const x = try dsv41.hostF32(a);
    defer std.testing.allocator.free(x);
    const y = try dsv41.hostF32(b);
    defer std.testing.allocator.free(y);
    try std.testing.expectEqual(x.len, y.len);
    var n: usize = 0;
    for (x, y) |u, v| n += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return n;
}

/// `a` with NaN, +Inf and -Inf at a few elements (an e4m3 NaN code in a word array), evaluated.
fn planted(sc: *dsv41.Scope, a: A) !A {
    const n = mlx.mlx_array_size(a);
    const shape = mlx.getShape(a);
    const at = [_]usize{ 3, n / 3 + 17, n / 2 + 5, n - 9 };
    const dt = mlx.mlx_array_dtype(a);
    if (dt == .uint32) {
        const c = try sc.op1(mlx.mlx_copy, a);
        try mlx.check(mlx.mlx_array_eval(c));
        const words = try std.testing.allocator.dupe(u32, (mlx.mlx_array_data_uint32(c) orelse return error.NoData)[0..n]);
        defer std.testing.allocator.free(words);
        for (at) |i| words[i] = (words[i] & 0xFFFF00FF) | 0x7F00;
        return sc.keep(mlx.mlx_array_new_data(words.ptr, shape.ptr, @intCast(shape.len), .uint32));
    }
    const v = try dsv41.hostF32(a);
    defer std.testing.allocator.free(v);
    for (at, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |i, x| v[i] = x;
    const f = try sc.keep(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32));
    const o = try sc.astype(f, dt);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// `ly` with NaN and Inf planted in every attention-side weight the bw launches read.
fn plantedLayer(sc: *dsv41.Scope, ly: *const dsv41.Layer) !dsv41.Layer {
    var c = ly.*;
    c.q_kv.?.w = try planted(sc, c.q_kv.?.w);
    c.hc_attn.fn_w = try planted(sc, c.hc_attn.fn_w);
    c.wq_b.w = try planted(sc, c.wq_b.w);
    c.wo_a.w = try planted(sc, c.wo_a.w);
    if (c.idx) |*iw| {
        iw.weights_proj.w = try planted(sc, iw.weights_proj.w);
        iw.wq_b.w = try planted(sc, iw.wq_b.w);
    }
    if (c.comp) |*cw| cw.wkv.w = try planted(sc, cw.wkv.w);
    return c;
}

/// One row count's inputs to the three launches.
const Ins = struct {
    x: A,
    cpre: A,
    qa: A,
    kv: A,
    z: A,
    pos: A,
    pacc: A,
    pm: A,
    pl: A,
    meta: A,

    /// `splits` partials a (row, head), every 13th split empty as the split kernel writes one (max -inf, sum 0).
    fn init(sc: *dsv41.Scope, f: *fixture.Fixture, rows: usize, splits: usize, plant: bool) !Ins {
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(f.m.cfg.num_attention_heads);
        const S: c_int = @intCast(splits);
        const n = rows * @as(usize, @intCast(H)) * splits;
        const host = try std.testing.allocator.alloc(f32, 3 * n);
        defer std.testing.allocator.free(host);
        var prng = std.Random.DefaultPrng.init(rows * 31 + splits);
        for (0..n) |j| {
            const empty = j % 13 == 5;
            host[j] = if (empty) -std.math.inf(f32) else prng.random().floatNorm(f32);
            host[n + j] = if (empty) 0 else 0.5 + 1.5 * prng.random().float(f32);
            host[2 * n + j] = if (empty) 0 else 1;
        }
        const live = try sc.keep(mlx.mlx_array_new_data(host[2 * n ..].ptr, &[_]c_int{ R, H, S, 1 }, 4, .float32));
        var in: Ins = .{
            .x = try sc.keep(try f.stream(rows)),
            .cpre = try sc.keep(try f.cpre(rows)),
            .qa = try sc.keep(try f.normal(&.{ R, 1280 }, .bfloat16, 0, 2)),
            .kv = try sc.keep(try f.normal(&.{ R, 512 }, .bfloat16, 0, 2)),
            .z = try sc.keep(try f.normal(&.{ R, 24 }, .float32, 0, 1.5)),
            .pos = try wq_b.rowsAt(sc, 3001, rows),
            .pacc = try sc.mul(try sc.keep(try f.normal(&.{ R, H, S, 512 }, .float32, 0, 1)), live),
            .pm = try sc.keep(mlx.mlx_array_new_data(host.ptr, &[_]c_int{ R, H, S }, 3, .float32)),
            .pl = try sc.keep(mlx.mlx_array_new_data(host[n..].ptr, &[_]c_int{ R, H, S }, 3, .float32)),
            .meta = try sc.ints(&.{ 3001, R, 512, 128, 144, 0 }, &.{6}),
        };
        if (!plant) return in;
        in.x = try planted(sc, in.x);
        in.qa = try planted(sc, in.qa);
        in.pacc = try planted(sc, in.pacc);
        return in;
    }
};

test "dsv41 dec attention side: the DSV41_ATTN_BW launches write today's bytes, a rank of four and the whole model, rows 1-6, NaN and Inf planted too" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    // in_proj's launch needs M3 or newer (`in_proj.plan`).
    if (@import("../../../mlx/device.zig").deviceGeneration().gen < 15) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const arms: [2]Arm = .{ try Arm.init(false, false), try Arm.init(true, false) };
    defer for (arms) |a| a.deinit();
    var checked: usize = 0;
    for ([_]?Shard{ tp4, null }) |tp| for ([_][2]usize{ .{ 20, 25 }, .{ 2, 3 } }) |span| {
        const f = try fixture.Fixture.init(gpa, .{ .lo = span[0], .hi = span[1], .experts = 2, .ctx = 0, .tp = tp });
        defer f.deinit();
        for (span[0]..span[1]) |li| {
            if (li == 22 or li == 23) continue;
            var sc = dsv41.Scope.init(gpa, f.m.s);
            defer sc.deinit();
            const counter = try sc.keep(wo_a.newCounter());
            const lys = [2]dsv41.Layer{ f.layer(li).*, try plantedLayer(&sc, f.layer(li)) };
            // The attention's splits: 8 on a compressed layer, 4 on a window-only one, 5 to 7 early in a context.
            for (bench_rows ++ [_]usize{ 3, 5 }) |rows| for ([_]usize{ 8, 4, 5 }) |splits| for (&lys, 0..) |*ly, plant| {
                if (splits != 8 and rows > 3) continue;
                const in = try Ins.init(&sc, f, rows, splits, plant == 1);
                var bufs: [2][13]A = undefined;
                const want = try arms[0].outs(&sc, f.m, ly, li, &in, counter, &bufs[0]);
                const got = try arms[1].outs(&sc, f.m, ly, li, &in, counter, &bufs[1]);
                try std.testing.expectEqual(want.len, got.len);
                for (want, got, 0..) |u, v, k| {
                    const d = try bitsDiffer(u, v);
                    if (d != 0) std.debug.print("[attn-bw] {s} layer {d} R={d} S={d} planted {d}: output {d} differs in {d} elements\n", .{ if (tp != null) "tp4" else "whole", li, rows, splits, plant, k, d });
                    try std.testing.expectEqual(@as(usize, 0), d);
                    checked += mlx.mlx_array_size(u);
                }
            };
        }
    };
    std.debug.print("[attn-bw] bw arm equal to today's in all {d} output elements\n", .{checked});
}

test {
    _ = fixture;
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, ffn_in, moe_up, moe_down }) |k| _ = k;
}
