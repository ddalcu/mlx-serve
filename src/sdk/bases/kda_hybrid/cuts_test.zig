//! The KDA-hybrid trunk's cuts: what the base reads in place of a plugin or a format, and the server's own formats
//! behind its hooks (`trunk.Hooks`, `builtin_formats.zig`).

const std = @import("std");
const testing = std.testing;
const trunk = @import("trunk.zig");
const builtin_formats = @import("builtin_formats.zig");
const layout = @import("../../quant/layout.zig");
const native = @import("../../quant/native.zig");
const fp8_block128 = @import("../../quant/fp8_block128.zig");
const fp8_claim = @import("../../quant/fp8_block128_claim.zig");
const fp8_grid = @import("../../quant/fp8_block128_gemv.zig");
const exl3_format = @import("../../quant/exl3/format.zig");

test "kda_hybrid cuts: the base's Layout is native's own type, every value in its place" {
    try testing.expect(layout.Layout == native.Layout);
    const want = [_][]const u8{ "dense", "affine", "nvfp4", "mxfp4", "mxfp8", "fp8_rows", "fp8_block128" };
    inline for (want, 0..) |name, i| try testing.expectEqual(@as(u8, i), @intFromEnum(@field(native.Layout, name)));
}

test "kda_hybrid cuts: the fp8 claim the trunk reads is fp8_block128's, on the GEMV plan's 64 x 128 grid" {
    try testing.expect(&fp8_block128.isBlockScaled == &fp8_claim.isBlockScaled);
    try testing.expectEqual(fp8_grid.scale_rows, fp8_claim.block_rows);
    try testing.expectEqual(fp8_grid.scale_cols, fp8_claim.block_cols);
    try testing.expectEqual(fp8_block128.block_rows, fp8_claim.block_rows);
    try testing.expectEqual(fp8_block128.block_cols, fp8_claim.block_cols);
}

test "kda_hybrid cuts: a pack that declares EXL3 experts without the format's hook is refused by name; none, no spec" {
    try testing.expectError(error.Exl3ExpertsNeedTheBuiltinFormat, trunk.exl3Spec(.{}, "{\"format\":\"exl3\",\"k\":2.5,\"codebook\":\"mcg\",\"window\":14}", 8));
    try testing.expect((try trunk.exl3Spec(.{}, "", 8)) == null);
    try testing.expect((try trunk.exl3Spec(.{ .exl3 = &builtin_formats.exl3 }, "", 8)) == null);
}

/// The `expert_quant` object of a config's text, as written.
fn expertQuantText(cfg: []const u8) []const u8 {
    const key = "\"expert_quant\"";
    const at = std.mem.indexOf(u8, cfg, key).?;
    const open = std.mem.indexOfScalarPos(u8, cfg, at + key.len, '{').?;
    var depth: usize = 0;
    for (cfg[open..], open..) |ch, i| {
        if (ch == '{') depth += 1;
        if (ch == '}') {
            depth -= 1;
            if (depth == 0) return cfg[open .. i + 1];
        }
    }
    unreachable;
}

test "kda_hybrid cuts: the built-in EXL3 hook holds the spec the format parses, inline in the trunk's Spec" {
    const cfg = @embedFile("../../../fixtures/model-configs/glm5_next-sushi-2.4bpw.json");
    const text = expertQuantText(cfg);
    var sp: trunk.Hooks.Exl3.Spec = .{};
    try builtin_formats.exl3.parse(testing.allocator, text, 8, &sp);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const wrapped = try std.fmt.allocPrint(arena.allocator(), "{{\"expert_quant\":{s}}}", .{text});
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), wrapped, .{});
    const want = try exl3_format.parse(root.object, 8);
    const got = builtin_formats.formatSpec(&sp);
    try testing.expectEqualDeep(exl3_format.decode(want), exl3_format.decode(got));
    try testing.expectEqual(want.rate, got.rate);
    try testing.expect(builtin_formats.exl3.table == &exl3_format.table);
    // the spec through the trunk's own entry
    const via = (try trunk.exl3Spec(.{ .exl3 = &builtin_formats.exl3 }, text, 8)).?;
    try testing.expectEqualDeep(exl3_format.decode(want), exl3_format.decode(builtin_formats.formatSpec(&via)));
}
