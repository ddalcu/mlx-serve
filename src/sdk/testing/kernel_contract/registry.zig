//! The kernel registry: every area's registrations and the planted controls. Gate 5 runs over all of them in
//! `zig build test` with no device; gates 1-4 run on the Metal device, one test per area (`-Dtest-filter="kernel contract"`).
//! An area registers in a `contract.zig` beside its block code (see REGISTERING.md) and is listed in `areas`.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
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
    @import("../../../plugins/ming_tts/contract.zig"),
    @import("../../quant/gguf/contract.zig"),
    @import("../../quant/exl3/contract.zig"),
    @import("../../blocks/mlp/contract.zig"),
    @import("../../quant/contract.zig"),
    @import("../../blocks/dit/contract.zig"),
    @import("../../blocks/linear/contract.zig"),
    @import("../../blocks/moe/contract.zig"),
    @import("../../../plugins/deepseek_v4/contract.zig"),
    @import("../../../plugins/deepseek_v41/contract.zig"),
    @import("../../../plugins/idol/contract.zig"),
    @import("../../../plugins/inkling2/contract.zig"),
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

/// The areas' names, in `areas` order: the per-area gates tests below name theirs.
const area_names = [_][]const u8{
    "controls",   "norms",  "rope",     "ssm",         "hc",           "gdn",       "kda",
    "attention",  "qmm",    "sampling", "spec",        "gemma4",       "qwen4_exp", "sparse_block",
    "minimax_h3", "ltx",    "ming_tts", "gguf",        "exl3",         "mlp",       "quant",
    "dit",        "linear", "moe",      "deepseek_v4", "deepseek_v41", "idol",      "inkling2",
};

comptime {
    // One gates test per area (and per part of a split one) below: an area added to `areas` needs its own.
    if (area_names.len != areas.len) @compileError("kernel contract: a new area needs its name here and its gates 1-4 test below");
}

/// Gates 1-4 over one area's registrations, every `parts`-th from `part` (an area too slow for one test is split in
/// parts the test runner places on different shards). Each kernel's inputs are seeded by its name, so its verdict does
/// not depend on which test runs it.
fn gatesArea(comptime name: []const u8, part: usize, parts: usize) !void {
    if (!builtin.os.tag.isDarwin() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const i = comptime for (area_names, 0..) |n, j| {
        if (std.mem.eql(u8, n, name)) break j;
    } else @compileError("kernel contract: no area " ++ name);
    const a = std.testing.allocator;
    var set = Set.init(a);
    defer set.deinit();
    try areas[i].register(&set);
    var mine: std.ArrayList(gates.Kernel) = .empty;
    defer mine.deinit(a);
    for (set.kernels.items, 0..) |k, j| if (j % parts == part) try mine.append(a, k);
    var t: gates.Tally = .{};
    try gates.runAll(a, mine.items, 0xc0de, &t);
    std.debug.print("SUMMARY kernel contract: {d}/{d} kernels pass gates 1-4, {d} known findings, {d}/{d} controls red, {d} need NAX\n", .{ t.passed, t.kernels, t.known, t.caught, t.controls, t.skipped });
    try std.testing.expectEqual(@as(usize, 0), t.problems());
}

test "kernel contract: the areas' registrations are the registry's, each kernel in one area" {
    const a = std.testing.allocator;
    var all = try build(a);
    defer all.deinit();
    var n: usize = 0;
    inline for (areas) |area| {
        var one = Set.init(a);
        defer one.deinit();
        try area.register(&one);
        for (one.kernels.items, all.kernels.items[n..][0..one.kernels.items.len]) |k, want| try std.testing.expectEqualStrings(want.spec.name, k.spec.name);
        n += one.kernels.items.len;
    }
    try std.testing.expectEqual(all.kernels.items.len, n);
}

// Gates 1-4, every registered kernel passes and every planted control goes red: one test per area, a slow area in parts.
test "kernel contract gates 1-4: controls" {
    try gatesArea("controls", 0, 1);
}

test "kernel contract gates 1-4: norms" {
    try gatesArea("norms", 0, 1);
}

test "kernel contract gates 1-4: rope" {
    try gatesArea("rope", 0, 1);
}

test "kernel contract gates 1-4: ssm" {
    try gatesArea("ssm", 0, 1);
}

test "kernel contract gates 1-4: hc" {
    try gatesArea("hc", 0, 1);
}

test "kernel contract gates 1-4: gdn" {
    try gatesArea("gdn", 0, 1);
}

test "kernel contract gates 1-4: kda" {
    try gatesArea("kda", 0, 1);
}

test "kernel contract gates 1-4: attention (part 1 of 4)" {
    try gatesArea("attention", 0, 4);
}

test "kernel contract gates 1-4: attention (part 2 of 4)" {
    try gatesArea("attention", 1, 4);
}

test "kernel contract gates 1-4: attention (part 3 of 4)" {
    try gatesArea("attention", 2, 4);
}

test "kernel contract gates 1-4: attention (part 4 of 4)" {
    try gatesArea("attention", 3, 4);
}

test "kernel contract gates 1-4: qmm (part 1 of 6)" {
    try gatesArea("qmm", 0, 6);
}

test "kernel contract gates 1-4: qmm (part 2 of 6)" {
    try gatesArea("qmm", 1, 6);
}

test "kernel contract gates 1-4: qmm (part 3 of 6)" {
    try gatesArea("qmm", 2, 6);
}

test "kernel contract gates 1-4: qmm (part 4 of 6)" {
    try gatesArea("qmm", 3, 6);
}

test "kernel contract gates 1-4: qmm (part 5 of 6)" {
    try gatesArea("qmm", 4, 6);
}

test "kernel contract gates 1-4: qmm (part 6 of 6)" {
    try gatesArea("qmm", 5, 6);
}

test "kernel contract gates 1-4: sampling" {
    try gatesArea("sampling", 0, 1);
}

test "kernel contract gates 1-4: spec" {
    try gatesArea("spec", 0, 1);
}

test "kernel contract gates 1-4: gemma4" {
    try gatesArea("gemma4", 0, 1);
}

test "kernel contract gates 1-4: qwen4_exp (part 1 of 2)" {
    try gatesArea("qwen4_exp", 0, 2);
}

test "kernel contract gates 1-4: qwen4_exp (part 2 of 2)" {
    try gatesArea("qwen4_exp", 1, 2);
}

test "kernel contract gates 1-4: sparse_block" {
    try gatesArea("sparse_block", 0, 1);
}

test "kernel contract gates 1-4: minimax_h3" {
    try gatesArea("minimax_h3", 0, 1);
}

test "kernel contract gates 1-4: ltx" {
    try gatesArea("ltx", 0, 1);
}

test "kernel contract gates 1-4: ming_tts" {
    try gatesArea("ming_tts", 0, 1);
}

test "kernel contract gates 1-4: gguf (part 1 of 4)" {
    try gatesArea("gguf", 0, 4);
}

test "kernel contract gates 1-4: gguf (part 2 of 4)" {
    try gatesArea("gguf", 1, 4);
}

test "kernel contract gates 1-4: gguf (part 3 of 4)" {
    try gatesArea("gguf", 2, 4);
}

test "kernel contract gates 1-4: gguf (part 4 of 4)" {
    try gatesArea("gguf", 3, 4);
}

test "kernel contract gates 1-4: exl3" {
    try gatesArea("exl3", 0, 1);
}

test "kernel contract gates 1-4: mlp" {
    try gatesArea("mlp", 0, 1);
}

test "kernel contract gates 1-4: quant" {
    try gatesArea("quant", 0, 1);
}

test "kernel contract gates 1-4: dit" {
    try gatesArea("dit", 0, 1);
}

test "kernel contract gates 1-4: linear" {
    try gatesArea("linear", 0, 1);
}

test "kernel contract gates 1-4: moe (part 1 of 5)" {
    try gatesArea("moe", 0, 5);
}

test "kernel contract gates 1-4: moe (part 2 of 5)" {
    try gatesArea("moe", 1, 5);
}

test "kernel contract gates 1-4: moe (part 3 of 5)" {
    try gatesArea("moe", 2, 5);
}

test "kernel contract gates 1-4: moe (part 4 of 5)" {
    try gatesArea("moe", 3, 5);
}

test "kernel contract gates 1-4: moe (part 5 of 5)" {
    try gatesArea("moe", 4, 5);
}

test "kernel contract gates 1-4: deepseek_v4" {
    try gatesArea("deepseek_v4", 0, 1);
}

test "kernel contract gates 1-4: deepseek_v41 (part 1 of 13)" {
    try gatesArea("deepseek_v41", 0, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 2 of 13)" {
    try gatesArea("deepseek_v41", 1, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 3 of 13)" {
    try gatesArea("deepseek_v41", 2, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 4 of 13)" {
    try gatesArea("deepseek_v41", 3, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 5 of 13)" {
    try gatesArea("deepseek_v41", 4, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 6 of 13)" {
    try gatesArea("deepseek_v41", 5, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 7 of 13)" {
    try gatesArea("deepseek_v41", 6, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 8 of 13)" {
    try gatesArea("deepseek_v41", 7, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 9 of 13)" {
    try gatesArea("deepseek_v41", 8, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 10 of 13)" {
    try gatesArea("deepseek_v41", 9, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 11 of 13)" {
    try gatesArea("deepseek_v41", 10, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 12 of 13)" {
    try gatesArea("deepseek_v41", 11, 13);
}

test "kernel contract gates 1-4: deepseek_v41 (part 13 of 13)" {
    try gatesArea("deepseek_v41", 12, 13);
}

test "kernel contract gates 1-4: idol" {
    try gatesArea("idol", 0, 1);
}

test "kernel contract gates 1-4: inkling2 (part 1 of 2)" {
    try gatesArea("inkling2", 0, 2);
}

test "kernel contract gates 1-4: inkling2 (part 2 of 2)" {
    try gatesArea("inkling2", 1, 2);
}

test {
    // Each area's own tests (its drift guards) run only when a test references its file.
    inline for (areas) |area| _ = area;
    _ = @import("spec.zig");
    _ = @import("limits.zig");
    _ = @import("offline.zig");
    _ = @import("host.zig");
}
