//! `MLX_SERVE_DSV41_PF_INT8`'s one owner (`pf_int8.Plan`, the model's `Table`): every path that once built int8
//! weights inside a request, outside the bill, now finds no table entry and runs the exact path. On the release's
//! shapes (`decode/fixture.zig`): the whole model's six layers, CED split at 3, and a pair worker's layers 20 and 21.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const pfint8 = @import("pf_int8.zig");
const fixture = @import("../decode/fixture.zig");

const testing = std.testing;
const A = mlx.mlx_array;

fn skip() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable();
}

fn tokens(n: usize, salt: usize, vocab: usize) ![]u32 {
    const p = try testing.allocator.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((i * 7919 + salt * 104729 + 11) % vocab);
    return p;
}

fn evalFree(x: A) !void {
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_array_eval(x));
}

/// The whole-model fixture at level 4: its table holds the encoder layers' (0..2) five slots, 15 weights.
fn full4(seed: u64) !*fixture.Fixture {
    const fx = try fixture.Fixture.initFull(testing.allocator, 8, 1024, seed);
    errdefer fx.deinit();
    try testing.expectEqual(@as(?usize, 3), fx.m.ced_split);
    try testing.expectEqual(@as(usize, 3), fx.m.trunk8.plan.layers);
    try testing.expectEqual(@as(usize, 15), fx.m.trunk8.count());
    return fx;
}

test "dsv41 pf int8 eligibility T1: with CED=0 a 1,200-row prompt runs the decoder layers wide, and they run exact: no int8 weight is made after load" {
    if (skip()) return error.SkipZigTest;
    const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
    defer lv.restore();
    const ced = env.override(.MLX_SERVE_DSV41_CED, "0");
    defer ced.restore();
    dsv41.resetCedEnvForTest();
    defer dsv41.resetCedEnvForTest();
    const fx = try full4(1);
    defer fx.deinit();
    const p = try tokens(1200, 1, fx.m.vocab);
    defer testing.allocator.free(p);
    const live = pfint8.live.load(.monotonic);
    const calls = pfint8.trunk_calls.load(.monotonic);
    try dsv41.reset(fx.m, testing.allocator);
    try evalFree(try dsv41.extendResumable(fx.m, testing.allocator, p, null));
    try testing.expect(pfint8.trunk_calls.load(.monotonic) > calls); // the encoder layers ran int8
    try testing.expectEqual(live, pfint8.live.load(.monotonic));
    try testing.expectEqual(@as(usize, 15), fx.m.trunk8.count());
}

test "dsv41 pf int8 eligibility T2: with CED on, a 1,200-row prompt that wants every row's logits runs the decoder layers wide, and they run exact: no int8 weight is made after load" {
    if (skip()) return error.SkipZigTest;
    const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
    defer lv.restore();
    const fx = try full4(2);
    defer fx.deinit();
    const p = try tokens(1200, 2, fx.m.vocab);
    defer testing.allocator.free(p);
    const live = pfint8.live.load(.monotonic);
    try dsv41.reset(fx.m, testing.allocator);
    try evalFree(try dsv41.extendAllResumable(fx.m, testing.allocator, p));
    try testing.expectEqual(live, pfint8.live.load(.monotonic));
    try testing.expectEqual(@as(usize, 15), fx.m.trunk8.count());
}

test "dsv41 pf int8 eligibility T3: a two-Mac pair worker is inert (single-Mac only): nothing is built at load or by a wide call on its layers" {
    if (skip()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
    defer lv.restore();
    const live = pfint8.live.load(.monotonic);
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 22, .experts = 8, .ctx = 0, .experts_fp4 = true });
    defer fx.deinit();
    try testing.expect(fx.m.pair != null);
    try testing.expect(!fx.m.trunk8.plan.on());
    try testing.expect(std.mem.indexOf(u8, fx.m.trunk8.plan.inert.?, "single-Mac only") != null);
    var sc = dsv41.Scope.init(gpa, fx.m.s);
    defer sc.deinit();
    const x = try fx.normal(&.{ 1024, @intCast(fx.m.cfg.hidden_size) }, .bfloat16, 0, 1);
    defer _ = mlx.mlx_array_free(x);
    const calls = pfint8.trunk_calls.load(.monotonic);
    const pj = try dsv41.qkvRows(&sc, fx.m, fx.layer(20), x);
    try dsv41.evalAll(&.{ pj.q, pj.kv });
    try testing.expectEqual(calls, pfint8.trunk_calls.load(.monotonic));
    try testing.expectEqual(live, pfint8.live.load(.monotonic));
}

test "dsv41 pf int8 eligibility T4: the bill holds what the load builds (levels 3 and 4); a builder planted one layer past the plan fails the load" {
    if (skip()) return error.SkipZigTest;
    for ([_][:0]const u8{ "3", "4" }, [_]usize{ 6, 15 }) |v, n| {
        const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, v.ptr);
        defer lv.restore();
        const fx = try fixture.Fixture.initFull(testing.allocator, 8, 1024, 4);
        defer fx.deinit();
        try testing.expectEqual(n, fx.m.trunk8.count());
        try testing.expectEqual(pfint8.residentBytes(&fx.m.cfg), fx.m.trunk8.bytes());
    }
    const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
    defer lv.restore();
    const live = pfint8.live.load(.monotonic);
    pfint8.plan_plant = true;
    defer pfint8.plan_plant = false;
    try testing.expectError(error.Dsv41Int8OverBill, fixture.Fixture.initFull(testing.allocator, 8, 1024, 4));
    try testing.expectEqual(live, pfint8.live.load(.monotonic));
}

test "dsv41 pf int8 eligibility T5: load, unload, load: the unload frees every int8 weight, and the second load builds its own from its own weights" {
    if (skip()) return error.SkipZigTest;
    const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
    defer lv.restore();
    const live = pfint8.live.load(.monotonic);
    {
        const fx = try full4(5);
        try testing.expectEqual(live + 15, pfint8.live.load(.monotonic));
        fx.deinit();
    }
    try testing.expectEqual(live, pfint8.live.load(.monotonic));
    const fx = try full4(6);
    defer fx.deinit();
    try testing.expectEqual(live + 15, pfint8.live.load(.monotonic));
    // Each entry is this model's: the same codes as a fresh int8 weight of its own linear.
    var sc = dsv41.Scope.init(testing.allocator, fx.m.s);
    defer sc.deinit();
    for (0..3) |li| {
        var w = (try pfint8.weightOf(&sc, &fx.layer(li).wq_b)).?;
        defer w.deinit();
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, w.wq, fx.m.trunk8.get(li, .wq_b).?.wq, false, sc.s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same: bool = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    }
}

test "dsv41 pf int8 eligibility T6: with CED=0 a perplexity window (every row's logits, a fresh state, 2,048 rows) runs the decoder layers exact: no int8 weight is made after load" {
    if (skip()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const lv = env.override(.MLX_SERVE_DSV41_PF_INT8, "4");
    defer lv.restore();
    const ced = env.override(.MLX_SERVE_DSV41_CED, "0");
    defer ced.restore();
    dsv41.resetCedEnvForTest();
    defer dsv41.resetCedEnvForTest();
    const fx = try full4(7);
    defer fx.deinit();
    const p = try tokens(2048, 7, fx.m.vocab);
    defer gpa.free(p);
    const live = pfint8.live.load(.monotonic);
    var st = try dsv41.initState(fx.m, gpa);
    defer st.deinit();
    try evalFree(try dsv41.extend(fx.m, gpa, &st, p, .all));
    try testing.expectEqual(live, pfint8.live.load(.monotonic));
}
