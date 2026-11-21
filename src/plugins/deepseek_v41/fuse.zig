//! Launch fusions, each off by default behind its own switch (`Fuse`, read at load):
//! - `down_scale`: a decode step or verify puts an NVFP4 bank's per-expert down scale on the routing weights inside the
//!   fused fp4 gate/up launch (`DSCALE`: `wts * dsc[e]`, the f32 product `routeScaled`'s take, cast and multiply make).

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const dsv41 = @import("deepseek_v41.zig");
const dkern = @import("decode/decode.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Fuse = struct {
    down_scale: bool = false,

    pub fn wanted() Fuse {
        return .{
            .down_scale = env.on(.MLX_SERVE_DSV41_FUSE_DOWN_SCALE),
        };
    }
};

/// The fusions that have served, each logged the first time (an A/B arm's engagement lines).
pub var engaged: Fuse = .{};

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

// ── tests ──

const testing = std.testing;
const fixture = @import("decode/fixture.zig");

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
