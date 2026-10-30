//! The kernel registry: every area's registrations and the planted controls. Gate 5 runs over all of them in
//! `zig build test` with no device; gates 1-4 run on the Metal device (`-Dtest-filter="kernel contract"`).
//! An area registers in a `contract.zig` beside its block code (see REGISTERING.md) and is listed in `areas`.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const spec = @import("spec.zig");
const gates = @import("gates.zig");
const limits = @import("limits.zig");
const metal = @import("metal.zig");

const areas = .{
    @import("controls.zig"),
    @import("../../blocks/norms/contract.zig"),
    @import("../../blocks/rope/contract.zig"),
    @import("../../blocks/ssm/contract.zig"),
    @import("../../blocks/hc/contract.zig"),
    @import("../../blocks/gdn/contract.zig"),
    @import("../../blocks/kda/contract.zig"),
    @import("../../blocks/attention/contract.zig"),
    @import("../../blocks/qmm/contract.zig"),
    @import("../../../engines/mlx/sampling/contract.zig"),
    @import("../../../engines/mlx/spec/contract.zig"),
    @import("../../../plugins/gemma4/contract.zig"),
    @import("../../../plugins/qwen4_exp/contract.zig"),
    @import("../../blocks/attention/sparse_block/contract.zig"),
    @import("../../../plugins/minimax_h3/contract.zig"),
    @import("../../../plugins/ltx/contract.zig"),
    @import("../../quant/gguf/contract.zig"),
    @import("../../quant/exl3/contract.zig"),
    @import("../../blocks/mlp/contract.zig"),
    @import("../../quant/contract.zig"),
    @import("../../blocks/dit/contract.zig"),
    @import("../../blocks/linear/contract.zig"),
    @import("../../blocks/moe/contract.zig"),
    @import("../../../plugins/deepseek_v4/contract.zig"),
    @import("../../../plugins/deepseek_v41/contract.zig"),
};

pub const Set = @import("set.zig").Set;

pub fn build(a: std.mem.Allocator) !Set {
    var set = Set.init(a);
    errdefer set.deinit();
    inline for (areas) |area| try area.register(&set);
    return set;
}

/// Gate 5 on `k`: its Metal launch at every shape of its spec (served and edges) on every Apple family, problems
/// written to `out`; returns how many (0 for a kernel with no Metal launch).
pub fn gate5(a: std.mem.Allocator, k: gates.Kernel, out: *std.ArrayList(u8)) !usize {
    const launch = k.vt.launch orelse return 0;
    const geometry = k.vt.geometry orelse return 0;
    const l = launch(k.ctx);
    const list = try spec.shapes(a, k.spec);
    defer a.free(list);
    var n: usize = 0;
    var shapes: usize = 0;
    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(a);
    for (list) |shape| if (geometry(k.ctx, shape)) |g| {
        scratch.clearRetainingCapacity();
        const here = try limits.metalAll(&.{l.record(k.spec.name, shape, g.threads())}, &scratch, a);
        if (here == 0) continue;
        if (shapes == 0) try out.print(a, "{s}  (first at {d}x{d}x{d})\n", .{ scratch.items, shape.rows, shape.n, shape.k });
        shapes += 1;
        n += here;
    };
    if (shapes > 1) try out.print(a, "  {s}: {d} shapes in all\n", .{ k.spec.name, shapes });
    return n;
}

// ── tests ──

test "kernel contract gate 5: every registered launch fits every chip, every limits control is caught" {
    const a = std.testing.allocator;
    var set = try build(a);
    defer set.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var bad: usize = 0;
    var controls: usize = 0;
    for (set.kernels.items) |k| {
        const mark = out.items.len;
        const n = try gate5(a, k, &out);
        const known = if (k.spec.known) |kn| kn.gate == .limits else false;
        if (k.control == .limits) {
            controls += 1;
            if (n == 0) {
                std.debug.print("CONTROL MISSED {s}: gate 5 found no problem\n", .{k.spec.name});
                bad += 1;
            }
            out.shrinkRetainingCapacity(mark);
        } else if (known) {
            if (n == 0) {
                std.debug.print("FAIL limits {s}: passes now; drop its known finding\n", .{k.spec.name});
                bad += 1;
            } else std.debug.print("KNOWN limits {s}: {s}\n", .{ k.spec.name, k.spec.known.?.finding });
            out.shrinkRetainingCapacity(mark);
        } else bad += n;
    }
    if (out.items.len > 0) std.debug.print("{s}", .{out.items});
    try std.testing.expect(controls >= 2);
    try std.testing.expectEqual(@as(usize, 0), bad);
}

fn shard() struct { k: usize, n: usize } {
    const raw = env.span(.MLX_SERVE_TEST_SHARD) orelse return .{ .k = 0, .n = 1 };
    const slash = std.mem.indexOfScalar(u8, raw, '/') orelse return .{ .k = 0, .n = 1 };
    const k = std.fmt.parseUnsigned(usize, raw[0..slash], 10) catch return .{ .k = 0, .n = 1 };
    const n = std.fmt.parseUnsigned(usize, raw[slash + 1 ..], 10) catch return .{ .k = 0, .n = 1 };
    return if (n == 0 or k >= n) .{ .k = 0, .n = 1 } else .{ .k = k, .n = n };
}

test "sharded: kernel contract gates 1-4: every registered kernel passes, every planted control goes red" {
    if (!builtin.os.tag.isDarwin() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = try build(a);
    defer set.deinit();
    // A sharded run (`MLX_SERVE_TEST_SHARD=k/n`) checks every n-th kernel from k.
    const sh = shard();
    var mine: std.ArrayList(gates.Kernel) = .empty;
    defer mine.deinit(a);
    for (set.kernels.items, 0..) |k, i| if (i % sh.n == sh.k) try mine.append(a, k);
    var t: gates.Tally = .{};
    try gates.runAll(a, mine.items, 0xc0de, &t);
    std.debug.print("SUMMARY kernel contract: {d}/{d} kernels pass gates 1-4, {d} known findings, {d}/{d} controls red, {d} need NAX\n", .{ t.passed, t.kernels, t.known, t.caught, t.controls, t.skipped });
    try std.testing.expectEqual(@as(usize, 0), t.problems());
}

test {
    // Each area's own tests (its drift guards) run only when a test references its file.
    inline for (areas) |area| _ = area;
    _ = @import("spec.zig");
    _ = @import("limits.zig");
    _ = @import("offline.zig");
    _ = @import("host.zig");
}
