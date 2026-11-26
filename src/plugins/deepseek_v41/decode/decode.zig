//! DeepSeek-V4.1's decode kernels, one file a kernel (`~/plans/dsv41-decode-kernels-spec.md`): a layer's 1..6 rows in
//! about ten dependent launches, each at its weights' byte floor. What their tests and ubenches share lives here.

const std = @import("std");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");

pub const fixture = @import("fixture.zig");
pub const in_proj = @import("in_proj.zig");
pub const wq_b = @import("wq_b.zig");
pub const idx_score = @import("idx_score.zig");
pub const topk = @import("topk.zig");
pub const attn = @import("attn.zig");
pub const wo_a = @import("wo_a.zig");
pub const woa_plain = @import("woa_plain.zig");
pub const ffn_in = @import("ffn_in.zig");
pub const moe_up = @import("moe_up.zig");
pub const moe_fp4 = @import("moe_fp4.zig");
pub const moe_down = @import("moe_down.zig");
pub const engram_kv = @import("engram_kv.zig");
pub const engram_ids = @import("engram_ids.zig");
pub const engram_kvc = @import("engram_kvc.zig");
pub const fp8_codes = @import("fp8_codes.zig");

const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// The most rows a decode kernel serves in one launch: a plain step, or a DSpark verify block.
pub const max_rows = 6;

/// What this machine streams from memory in practice (a plain reduction on the M5 Ultra), bytes a microsecond.
pub const stream_bytes_per_us: f64 = 1.065e6;

pub fn register(set: *kc.Set) !void {
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, woa_plain, ffn_in, moe_up, moe_fp4, moe_down, engram_kv, engram_kvc }) |k| try k.register(set);
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

/// An output's distance from its float64 truth, summed over calls: squared error, the truth's square, the largest
/// absolute error, and the non-finite values (each one farther than anything).
pub const Dist = struct {
    se: f64 = 0,
    ss: f64 = 0,
    max: f64 = 0,
    bad: usize = 0,

    /// `got` f32 or f64 values against `truth`.
    pub fn add(d: *Dist, got: anytype, truth: []const f64) void {
        std.debug.assert(got.len == truth.len);
        for (got, truth) |g, t| {
            const v: f64 = g;
            if (!std.math.isFinite(v)) {
                d.bad += 1;
                continue;
            }
            d.se += (v - t) * (v - t);
            d.ss += t * t;
            d.max = @max(d.max, @abs(v - t));
        }
    }

    pub fn l2(d: Dist) f64 {
        return @sqrt(d.se / @max(d.ss, 1e-300));
    }

    /// No farther from float64 than `o` in l2 and in max.
    pub fn within(d: Dist, o: Dist) bool {
        return d.bad == 0 and d.se <= o.se and d.max <= o.max;
    }

    pub fn join(d: *Dist, o: Dist) void {
        d.se += o.se;
        d.ss += o.ss;
        d.max = @max(d.max, o.max);
        d.bad += o.bad;
    }
};

/// The words where `a` and `b` differ, and of those how many `a` holds nearer to `truth` and how many farther.
pub const WordDiff = struct {
    differ: usize = 0,
    nearer: usize = 0,
    farther: usize = 0,
    words: usize = 0,

    pub fn add(w: *WordDiff, a: []const f32, b: []const f32, truth: []const f64) void {
        for (a, b, truth) |x, y, t| {
            w.words += 1;
            if (@as(u32, @bitCast(x)) == @as(u32, @bitCast(y))) continue;
            w.differ += 1;
            const ex = @abs(@as(f64, x) - t);
            const ey = @abs(@as(f64, y) - t);
            w.nearer += @intFromBool(ex < ey);
            w.farther += @intFromBool(ex > ey);
        }
    }
};

/// The fusion bar: `new` is no further from `truth` than today's `chain` (1.25x, plus 1e-4 of truth's scale).
pub fn expectNoWorse(label: []const u8, new: []const f32, chain: []const f32, truth: []const f32) !void {
    const en = relRms(new, truth);
    const ec = relRms(chain, truth);
    std.debug.print("[dec] {s}: new-truth {e:.2}, chain-truth {e:.2} (rel rms)\n", .{ label, en, ec });
    for (new) |v| if (!std.math.isFinite(v)) return error.NonFinite;
    if (en > 1.25 * ec + 1e-4) return error.FurtherFromTruth;
}

/// The mxfp8 reads (MLX's, or the 2^8 fold under `FOLD`): `fp8_read_header.metal` after the e4m3 / e8m0 reads.
pub const FP8_READ_HEADER = @import("../../../sdk/quant/fp4.zig").SCALE_HEADER ++ @import("../../../mlx/kernels.zig").source("plugins/deepseek_v41/kernels/fp8_read_header.metal");

/// The mxfp8 reads' template int `FOLD` (`MLX_SERVE_DSV41_FP8_FOLD`, read every call): 1 the 2^8 fold (the default:
/// the same product on every finite code under 247, fewer operations), 0 MLX's reads (=0, or where the load found a
/// code the fold reads apart from MLX, `fp8_codes.settle`).
pub fn fp8Fold() c_int {
    return @intFromBool(lev.onUnlessZero(.FP8_FOLD) and fp8_codes.ps().fold);
}

/// A contract kernel at `FOLD` 0 (MLX's mxfp8 reads) and 1 (the fold) on the same inputs, each served shape over three
/// seeds of its random fill (finite e4m3 codes, e8m0 under 247): byte for byte, else `error.FoldDiffers`.
pub fn expectFoldSame(comptime D: type, comptime launch: *const kc.metal.Launch, p: D.Params, shapes: []const kc.spec.Shape) !void {
    const kit = @import("../../deepseek_v4/contract_kit.zig");
    const a = std.testing.allocator;
    for (shapes) |shape| for (1..4) |seed| {
        var c = kit.Case(D).init(a, launch, p);
        defer c.deinit();
        if (!try c.prepare(shape, .random, seed)) continue;
        var g = D.geometry(p, shape).?;
        const mlx_reads = try a.dupe(u8, blk: {
            try c.rig.launch(launch, g, c.args[0..c.n_in]);
            break :blk c.rig.outputBytes();
        });
        defer a.free(mlx_reads);
        for (g.targs[0..g.n_targs]) |*t| if (std.mem.orderZ(u8, t.name, "FOLD") == .eq) {
            t.v = .{ .int = 1 };
        };
        try c.rig.launch(launch, g, c.args[0..c.n_in]);
        if (!std.mem.eql(u8, mlx_reads, c.rig.outputBytes())) {
            std.debug.print("{s} {d}x{d}x{d} seed {d}: FOLD=1 bytes differ from MLX's reads\n", .{ launch.name, shape.rows, shape.n, shape.k, seed });
            return error.FoldDiffers;
        }
    };
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
    try layersAgree(std.testing.allocator, 20, 23, true);
}

test "dsv41 dec layers: under MLX_SERVE_DSV41_FUSE_WQB_RING on and off, the fused kernels decode no further from an f32 model than today's chain" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    inline for (.{ "1", "0" }) |arm| {
        var o0 = env.override(.MLX_SERVE_DSV41_FUSE_WQB_RING, arm);
        defer o0.restore();
        std.debug.print("[dec] switches ring={s}\n", .{arm});
        for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 } }) |range| try layersAgree(gpa, range[0], range[1], false);
    }
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
    if (!ubenchOn() or !mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    // `DSV41_DEC_UBENCH` naming "nvfp4" or "mxfp4": the experts in that format, the fp4 MoE kernels' levers off against
    // the arms it names (`layersFp4Levers`).
    if (env.span(.DSV41_DEC_UBENCH)) |v| for ([_][]const u8{ "nvfp4", "mxfp4" }) |f| {
        if (std.mem.indexOf(u8, v, f) != null) return layersFp4Levers(gpa, f[0] == 'n', v);
    };
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

/// The layers ubench over fp4 experts (`nv`: NVFP4 with per-expert scales, else MXFP4), R = 1..6: the fp4 MoE
/// kernels' levers (`moe_fp4.Levers`) off (A) against each B arm, A first, twice over (a fixture an arm). B arms: the
/// words "dw", "us" (one lever each) and "all" in `DSV41_DEC_UBENCH`, else the levers as their switches are set.
fn layersFp4Levers(gpa: std.mem.Allocator, nv: bool, spec_: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    const L = moe_fp4.Levers;
    // The third field: `MLX_SERVE_DSV41_MOE_DOWN_ORDER` (the down launch's column block major grid at 2 rows and up); the
    // fourth `MLX_SERVE_DSV41_FFN_IN_SINK` (ffn_in launch 2's mix tail in its own launch).
    const Arm = struct { []const u8, L, bool, bool };
    const named = [_]Arm{
        .{ "dw", .{ .dec_word = true }, false, false },
        .{ "us", .{ .up_split = true }, false, false },
        .{ "all", .{ .dec_word = true, .up_split = true }, false, false },
        .{ "ctl", .{ .dec_word = true }, moe_fp4.orderOn(), false },
        .{ "order", .{ .dec_word = true }, true, false },
        .{ "sink", .{ .dec_word = true }, moe_fp4.orderOn(), true },
    };
    var arms: [1 + named.len]Arm = undefined;
    arms[0] = .{ "A (off)", .{}, false, false };
    var n: usize = 1;
    var it = std.mem.tokenizeAny(u8, spec_, " ,:");
    while (it.next()) |w| for (named) |c| if (std.mem.eql(u8, w, c[0])) {
        arms[n] = c;
        n += 1;
    };
    if (n == 1) {
        arms[1] = .{ "as set", L.fromEnv(), moe_fp4.orderOn(), ffn_in.sinkOn() };
        n = 2;
    }
    std.debug.print("[dec-ubench] layers 20..40 {s} experts, {d} arms, two rounds\n", .{ if (nv) "nvfp4 + per-expert scales" else "mxfp4", n });
    var us: [1 + named.len][max_rows]f64 = undefined;
    for (&us) |*row| row.* = @splat(0);
    for (0..2 * n) |k| {
        const arm = arms[k % n];
        const lv = arm[1];
        const o0 = env.override(.MLX_SERVE_DSV41_FP4_DEC_WORD, if (lv.dec_word) "1" else "0");
        defer o0.restore();
        const o1 = env.override(.MLX_SERVE_DSV41_FP4_UP_SPLIT, if (lv.up_split) "1" else "0");
        defer o1.restore();
        const o2 = env.override(.MLX_SERVE_DSV41_MOE_DOWN_ORDER, if (arm[2]) "1" else "0");
        defer o2.restore();
        const o3 = env.override(.MLX_SERVE_DSV41_FFN_IN_SINK, if (arm[3]) "1" else "0");
        defer o3.restore();
        const t_fx = iou.nowUsMonotonic(io);
        // 64 experts a layer: 20 layers hold ~30 GB, so a step's weights come from DRAM, not the SLC.
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 40, .experts = 64, .ctx = 4096, .fused = true, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none });
        defer fx.deinit();
        std.debug.print("[dec-ubench] arm {s}: fixture {d:.1} s\n", .{ arm[0], @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t_fx)) / 1e6 });
        var p0: usize = 4096;
        for (1..max_rows + 1) |rows| {
            const x = try fx.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(rows);
            defer _ = mlx.mlx_array_free(pre);
            var times: [21]f64 = undefined;
            for (0..3 + times.len) |rep| {
                const t0 = iou.nowUsMonotonic(io);
                const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
                for (out) |o| _ = mlx.mlx_array_free(o);
                if (rep >= 3) times[rep - 3] = @floatFromInt(iou.nowUsMonotonic(io) - t0);
                p0 += rows;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            const med = times[times.len / 2] / 20;
            us[k % n][rows - 1] += med / 2;
            std.debug.print("[dec-ubench] layers 20..40 R={d} {s}: {d:.1} us a layer (median step {d:.0} us, p10 {d:.0}, p90 {d:.0})\n", .{ rows, arm[0], med, times[times.len / 2], times[times.len / 10], times[times.len * 9 / 10] });
        }
    }
    // Each arm against A, and an arm after "ctl" (the control's levers) against "ctl" too.
    var ctl: ?usize = null;
    for (arms[0..n], 0..) |a, j| if (std.mem.eql(u8, a[0], "ctl")) {
        ctl = j;
    };
    for (1..max_rows + 1) |rows| for (1..n) |j| {
        const a = us[0][rows - 1];
        const b = us[j][rows - 1];
        std.debug.print("[dec-ubench] layers 20..40 R={d} {s}: {d:.1} us a layer against A {d:.1} ({d:.1} us, {d:.2}%)\n", .{ rows, arms[j][0], b, a, b - a, 100 * (b - a) / a });
        if (ctl) |c| if (j > c) {
            const cu = us[c][rows - 1];
            std.debug.print("[dec-ubench] layers 20..40 R={d} {s}: {d:.1} us a layer against ctl {d:.1} ({d:.1} us, {d:.2}%)\n", .{ rows, arms[j][0], b, cu, b - cu, 100 * (b - cu) / cu });
        };
    };
}

/// One link of `HandOff`'s chain, shaped like a tensor split's decode `tpSum` past two ranks: a GPU producer of the
/// bf16 part, its cast to f32, the collective's stand-in (an add) on the CPU stream, the cast back and a GPU consumer;
/// without `sum`, the producer and consumer alone.
const HandOff = struct {
    sc: *dsv41.Scope,
    cpu: mlx.mlx_stream,
    x0: A,
    zero: A,
    prev: A,
    sum: bool,

    fn on(c: *HandOff, x: A, dt: ?mlx.mlx_dtype, s: mlx.mlx_stream) !A {
        var o = mlx.mlx_array_new();
        if (dt) |t| return c.sc.res(mlx.mlx_astype(&o, x, t, s), &o);
        return c.sc.res(mlx.mlx_add(&o, x, c.zero, s), &o);
    }

    pub fn step(c: *HandOff, _: usize) !A {
        const gpu = c.sc.s;
        var a = try c.sc.astype(try c.sc.add(c.x0, try c.sc.mul(c.prev, c.zero)), .bfloat16);
        if (c.sum) a = try c.on(try c.on(try c.on(a, .float32, gpu), null, c.cpu), .bfloat16, gpu);
        c.prev = try c.sc.add(a, c.zero);
        return c.prev;
    }
};

test "dsv41 dec ubench: a tensor sum's GPU-CPU-GPU hand-off without the network (DSV41_DEC_UBENCH)" {
    if (!ubenchOn() or !mlx.metalKernelsAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var row: [4096]f32 = undefined;
    for (&row, 0..) |*v, i| v.* = @floatFromInt(i % 7);
    const x0 = try sc.keep(mlx.mlx_array_new_data(&row, &[_]c_int{ 1, row.len }, 2, .float32));
    const zero = try sc.keep(mlx.mlx_array_new_data(&[_]f32{0}, &[_]c_int{1}, 1, .float32));
    const n = 80; // a decode token's sums under a tensor split
    var us = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
    for (0..3) |_| for (&us, [_]bool{ false, true }) |*u, sum| {
        var h: HandOff = .{ .sc = &sc, .cpu = cpu, .x0 = x0, .zero = zero, .prev = x0, .sum = sum };
        u.* = @min(u.*, try moe_up.deviceUs(&h, n));
    };
    std.debug.print("[dec-ubench] tensor-sum hand-off (MLX_METAL_FAST_SYNCH={s}): {d:.1} us a sum before any network ({d:.1} us a link for the GPU producer and consumer alone)\n", .{ env.get(.MLX_METAL_FAST_SYNCH) orelse "unset", us[1] - us[0], us[0] });
}

test {
    _ = fixture;
    _ = @import("exact_test.zig");
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, woa_plain, ffn_in, moe_up, moe_down, engram_kv, engram_ids, engram_kvc, fp8_codes }) |k| _ = k;
}

test "dsv41 dec fp8 reads: every e4m3 code at every byte reads as MLX's from_fp8, every e8m0 code as MLX's fp8.h; FOLD differs only where it overflows or misses a NaN" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    // Word w holds codes 4w .. 4w + 3 rotated by w % 4 bytes, so each code lands at each byte position.
    var words: [256]u32 = undefined;
    for (&words, 0..) |*w, i| {
        const base: u32 = @intCast((i / 4) * 4);
        const raw = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24;
        w.* = std.math.rotl(u32, raw, 8 * (i % 4));
    }
    const src =
        \\const uint i = thread_position_in_grid.x;
        \\const float4 v = FOLD != 0 ? dsv41_e4m3x4<1>(w[i]) : dsv41_e4m3x4<0>(w[i]);
        \\for (int b = 0; b < 4; ++b) e4[4 * i + b] = v[b] * 256.0f;
        \\const float2 f = FOLD != 0 ? dsv41_e8m0<1>(i) : dsv41_e8m0<0>(i);
        \\e8[2 * i] = f.x;
        \\e8[2 * i + 1] = f.y;
        \\
    ;
    const k = try dsv41.buildKernel("dsv41_fp8_read_codes", &.{"w"}, &.{ "e4", "e8" }, src, FP8_READ_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const wa = mlx.mlx_array_new_data(&words, &[_]c_int{256}, 1, .uint32);
    defer _ = mlx.mlx_array_free(wa);
    var code_bytes: [1024]u8 = undefined;
    for (&code_bytes, 0..) |*c, j| c.* = @truncate(words[j / 4] >> @intCast(8 * (j % 4)));
    const ca = mlx.mlx_array_new_data(&code_bytes, &[_]c_int{1024}, 1, .uint8);
    defer _ = mlx.mlx_array_free(ca);
    var mlx_e4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mlx_e4);
    try mlx.check(mlx.mlx_from_fp8(&mlx_e4, ca, .float32, s));
    try mlx.check(mlx.mlx_array_eval(mlx_e4));
    const want_e4 = mlx.mlx_array_data_float32(mlx_e4) orelse return error.NoData;
    for ([_]c_int{ 0, 1 }) |fold| {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1024}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{512}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "FOLD", fold));
        const iv = mlx.mlx_vector_array_new_data(&[_]A{wa}, 1);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
        var outs: [2]A = undefined;
        for (&outs, 0..) |*o, i| {
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
            try mlx.check(mlx.mlx_array_eval(o.*));
        }
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        const e4 = mlx.mlx_array_data_float32(outs[0]) orelse return error.NoData;
        const e8 = mlx.mlx_array_data_float32(outs[1]) orelse return error.NoData;
        var differ: usize = 0;
        for (0..1024) |j| {
            const nan_code = code_bytes[j] & 0x7F == 0x7F;
            const same = if (std.math.isNan(want_e4[j])) std.math.isNan(e4[j]) else @as(u32, @bitCast(e4[j])) == @as(u32, @bitCast(want_e4[j]));
            if (fold == 0 or !nan_code) try std.testing.expect(same) else differ += @intFromBool(!same);
        }
        // MLX's e8m0 read (0 is 2^-127, 0xFF NaN) and the 2^8 on the dot; the fold's 2^8 x read is +Inf from 247 up.
        for (0..256) |e| {
            const read: f32 = if (e == 0xFF) std.math.nan(f32) else @bitCast(@as(u32, if (e == 0) 0x400000 else @as(u32, @intCast(e)) << 23));
            if (fold == 0) {
                try std.testing.expect(if (e == 0xFF) std.math.isNan(e8[2 * e]) else @as(u32, @bitCast(e8[2 * e])) == @as(u32, @bitCast(read)));
                try std.testing.expectEqual(@as(f32, 256), e8[2 * e + 1]);
            } else {
                try std.testing.expectEqual(@as(f32, 1), e8[2 * e + 1]);
                // Code 0's 2^-127 is a subnormal the GPU flushes before the x256 (MLX's products flush it too).
                const want: f32 = if (e == 0) 0 else read * 256.0;
                if (e < 247) try std.testing.expectEqual(want, e8[2 * e]) else differ += @intFromBool(std.math.isInf(e8[2 * e]));
            }
        }
        // The fold reads S.1111.111 as 480 (8 bytes) and e8m0 247..255 as +Inf (9 codes).
        if (fold == 1) try std.testing.expectEqual(@as(usize, 8 + 9), differ);
    }
    _ = a;
}
