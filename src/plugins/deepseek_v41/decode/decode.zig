//! DeepSeek-V4.1's decode kernels, one file a kernel (`~/plans/dsv41-decode-kernels-spec.md`): a layer's 1..6 rows in
//! about ten dependent launches, each at its weights' byte floor. What their tests and ubenches share lives here.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
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
pub const moe_down = @import("moe_down.zig");

const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

/// The most rows a decode kernel serves in one launch: a plain step, or a DSpark verify block.
pub const max_rows = 6;

/// What this machine streams from memory in practice (a plain reduction on the M5 Ultra), bytes a microsecond.
pub const stream_bytes_per_us: f64 = 1.065e6;

pub fn register(set: *kc.Set) !void {
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, ffn_in, moe_up, moe_down }) |k| try k.register(set);
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

test "dsv41 dec layers: the fused kernels decode no further from an f32 model than today's chain, one row and a verify block" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    // Ratio 1 (KV source, owner, candidate source, an index non-owner), ratio 2 (KV source pooling pairs), ratio 0.
    for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 }, .{ 0, 1 } }) |range| try layersAgree(gpa, range[0], range[1]);
}

fn layersAgree(gpa: std.mem.Allocator, lo: usize, hi: usize) !void {
    // Same seed, so the same weights, prefill and step inputs: the fused kernels, today's chain, and the f32 reference.
    var fxs: [3]*fixture.Fixture = undefined;
    var made: usize = 0;
    defer for (fxs[0..made]) |fx| fx.deinit();
    for (&fxs, [_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .float32 }, [_]bool{ true, false, false }) |*fx, act, fused| {
        fx.* = try fixture.Fixture.init(gpa, .{ .lo = lo, .hi = hi, .ctx = 601, .seed = 9, .act = act, .fused = fused });
        made += 1;
    }
    var p0: usize = 601;
    for ([_]usize{ 1, 1, 1, max_rows, 1 }) |rows| {
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
        std.debug.print("[dec] layers {d}..{d}, {d} rows at {d}: fused vs chain {e:.2}\n", .{ lo, hi, rows, p0, relRms(got[0], got[1]) });
        try expectNoWorse("layers", got[0], got[1], got[2]);
        p0 += rows;
    }
}

test "dsv41 dec layers ubench: a decode step through layers 20..40, fused vs today's chain, 1 and 6 rows" {
    if (!ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    // 64 experts a layer: 20 layers hold ~20 GB, so a step's weights come from DRAM, not the SLC.
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 40, .experts = 64, .ctx = 4096, .fused = true });
    defer fx.deinit();
    const dec = fx.m.dec;
    defer fx.m.dec = dec;
    var p0: usize = 4096;
    for ([_]usize{ 1, max_rows }) |rows| for ([_]bool{ true, false }) |fused| {
        fx.m.dec = if (fused) dec else .{};
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
    };
}

test {
    _ = fixture;
    inline for (.{ in_proj, wq_b, idx_score, topk, attn, wo_a, ffn_in, moe_up, moe_down }) |k| _ = k;
}
