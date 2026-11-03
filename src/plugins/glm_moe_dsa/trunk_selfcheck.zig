//! The self-check of GLM-5.3's trunk kernel registry (`trunk_kernels`), as `sdk_ext.kernels.KernelSet` runs it and the
//! module at its construction (`runRoutes`): `compile`, the kernel builds, binds and completes one launch over zeros at
//! an instantiation (the construction's: the model's; `runSubset`: every one the manifest registers), as the routes
//! launch it (`trunk_kernels.topkLaunch`, `combineLaunch`). No numeric probe: the numerics are the device tests'
//! (topk.zig against argpartition's ids and order, experts.zig against the op chain's bits).

const std = @import("std");
const env = @import("mlx_host").env;
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const mlx = @import("sdk").mlx;
const log = @import("sdk").log;
const xk = @import("trunk_kernels.zig");
const kset = @import("../../sdk/weights/streamed/sdk_ext/kernel_set.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Check = xk.Check;

const scaffold = kset.Checks(Kernel, Check);
pub const Result = scaffold.Result;
pub const Report = scaffold.Report;
pub const Depth = kset.CheckDepth;
const Scope = kset.Scope;

/// The checks of `want`'s entries at `depth`, per kernel.
pub fn plan(reg: *const xk.Registry, want: std.EnumSet(Kernel), depth: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
    return scaffold.plan(&reg.entries, want, depth);
}

/// A route's kernel and the instantiation it launches.
pub const Route = struct { kernel: Kernel, at: xk.Instance };

/// The plan over `subset` at `depth` on the bound kernels: each kernel at every instantiation the manifest registers,
/// one result per launch.
pub fn runSubset(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, subset: []const Kernel, depth: Depth, report: *Report) !void {
    var want: std.EnumSet(Kernel) = .empty;
    for (subset) |k| want.insert(k);
    const checks = plan(reg, want, depth);
    for (&reg.entries) |*e| {
        if (!want.contains(e.kernel) or !checks.get(e.kernel).contains(.compile)) continue;
        for (e.instances) |inst| try checkOne(a, bound, e.kernel, inst, report);
    }
}

/// The construction's acceptance: each route's kernel at its instantiation (registered, else refused by name), one
/// launch over zeros, judged.
pub fn runRoutes(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, routes: []const Route, report: *Report, diag: ?*xk.Diag) !void {
    for (routes) |r| {
        _ = try reg.instance(r.kernel, r.at, diag);
        try checkOne(a, bound, r.kernel, r.at, report);
    }
    try judge(report, diag);
}

/// The verdict over a finished plan: refused (error.SelfCheckFailed, `diag` naming the first failure) unless every
/// result passed and the plan ran.
pub fn judge(report: *const Report, diag: ?*xk.Diag) error{SelfCheckFailed}!void {
    return scaffold.judge(report, "glm trunk kernels", diag);
}

/// One `compile` result of `k` at `inst`: a raised launch is recorded with its error and the latched MLX message.
fn checkOne(a: Allocator, bound: *const xk.Bound, k: Kernel, inst: xk.Instance, report: *Report) !void {
    launchZeros(a, bound, k, inst) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        var buf: [512]u8 = undefined;
        const m = mlx.takeError(&buf) orelse "";
        log.err("[glm trunk selfcheck] {t} at K {d}, H {d}: {s} {s}\n", .{ k, inst.k, inst.hidden, @errorName(e), m });
        return report.appendRaised(a, k, .compile, e, m);
    };
    try report.results.append(a, .{ .kernel = k, .check = .compile, .words = 1, .ok = true });
}

/// One launch of `k` at `inst` over zeros, evaluated: the top-k over one row (or `rows` rows) of k + 1 zeros, the combine
/// over one token's k routed rows of zeros (every slot of the parts the first part, as the route binds them).
fn launchZeros(a: Allocator, bound: *const xk.Bound, k: Kernel, inst: xk.Instance) !void {
    var sc: Scope = .{ .a = a };
    defer sc.deinit();
    var inputs: [xk.max_inputs]mlx.mlx_array = undefined;
    var n_in: usize = 0;
    var cfg: xk.LaunchConfig = undefined;
    switch (xk.familyOf(k)) {
        .topk => {
            const n = inst.k + 1;
            const zeros = try a.alloc(f32, n);
            defer a.free(zeros);
            @memset(zeros, 0);
            var params: [8]i32 = @splat(0);
            if (k == .glm_moe_dsa_topk) {
                inputs[0] = try sc.host(std.mem.sliceAsBytes(zeros), &.{ 1, 1, @intCast(n) }, .float32);
                params[0] = @intCast(n);
                n_in = 1;
                cfg = xk.topkLaunch(1, inst.k, false);
            } else {
                const rows: usize = @backingInt(k);
                const row = try sc.host(std.mem.sliceAsBytes(zeros), &.{ 1, 1, 1, @intCast(n) }, .float32);
                for (inputs[0..rows], params[0..rows]) |*in, *p| {
                    in.* = row;
                    p.* = @intCast(n);
                }
                n_in = rows;
                cfg = xk.topkLaunch(@intCast(rows), inst.k, true);
            }
            inputs[n_in] = try sc.host(std.mem.sliceAsBytes(&params), &.{8}, .int32);
            n_in += 1;
        },
        .combine => {
            const rows = try a.alloc(u16, @as(usize, inst.k) * inst.hidden);
            defer a.free(rows);
            @memset(rows, 0);
            const part = try sc.host(std.mem.sliceAsBytes(rows), &.{ @intCast(inst.k), @intCast(inst.hidden) }, .bfloat16);
            for (inputs[0..xk.max_combine_parts]) |*p| p.* = part;
            const loc = try a.alloc(u32, inst.k);
            defer a.free(loc);
            for (loc, 0..) |*l, j| l.* = @intCast(j);
            const w = try a.alloc(f32, inst.k);
            defer a.free(w);
            @memset(w, 1);
            inputs[xk.max_combine_parts] = try sc.host(std.mem.sliceAsBytes(loc), &.{@intCast(inst.k)}, .uint32);
            inputs[xk.max_combine_parts + 1] = try sc.host(std.mem.sliceAsBytes(w), &.{ 1, @intCast(inst.k) }, .float32);
            n_in = xk.max_inputs;
            cfg = xk.combineLaunch(1, inst.k, inst.hidden, .bfloat16);
        },
    }
    var outs: [xk.max_outputs]mlx.mlx_array = undefined;
    try bound.apply(k, inputs[0..n_in], &cfg, &outs);
    _ = try sc.keep(outs[0]);
    try mlx.check(mlx.mlx_array_eval(outs[0]));
}

const testing = std.testing;

test "glm trunk selfcheck: the stub device's plan gives one compile result per kernel; a scripted failure refuses by name" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const KS = kset.KernelSet(xk);
    var diag: xk.Diag = .{};
    const set = try KS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    var report: Report = .{};
    defer report.deinit(testing.allocator);
    try set.selfCheck(testing.allocator, &xk.topk_kernels, .compile, &report, &diag);
    try testing.expectEqual(xk.topk_kernels.len, report.results.items.len);
    for (report.results.items) |r| try testing.expect(r.ok and r.check == .compile);
    const bad = try KS.Set.init(testing.allocator, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .glm_moe_dsa_combine, .check = .compile } } } }, &diag);
    defer bad.deinit();
    var r2: Report = .{};
    defer r2.deinit(testing.allocator);
    try testing.expectError(error.SelfCheckFailed, bad.selfCheck(testing.allocator, &.{.glm_moe_dsa_combine}, .compile, &r2, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "glm_moe_dsa_combine") != null);
}

test "glm trunk selfcheck: every kernel builds and completes a launch at every registered instantiation on the GPU" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const set = try xk.openForTest(a, s);
    defer set.deinit();
    var report: Report = .{};
    defer report.deinit(a);
    try runSubset(a, &set.reg, &set.bound, std.enums.values(Kernel), .full, &report);
    try judge(&report, null);
    try testing.expectEqual(@as(usize, (xk.topk_kernels.len) * xk.topk_instances.len + xk.combine_instances.len), report.results.items.len);
    var diag: xk.Diag = .{};
    var r2: Report = .{};
    defer r2.deinit(a);
    try testing.expectError(error.InstanceNotRegistered, runRoutes(a, &set.reg, &set.bound, &.{.{ .kernel = .glm_moe_dsa_topk, .at = .{ .k = 1024 } }}, &r2, &diag));
}
