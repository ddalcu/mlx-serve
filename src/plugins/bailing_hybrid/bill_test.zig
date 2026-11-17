//! bailing_hybrid's prompt bill against the engine's. The plugin prices a streaming pack's prompt from
//! `sdk/bill/prefill.zig` (its closure stays in sdk/); this file, outside that closure, pins the figure to the engine's
//! `prefillNeededAtChunk` of the same config.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const plugin = @import("plugin.zig");
const config = plugin.config;
const toModelConfig = @import("../glm5_next/model_config.zig").toModelConfig;
const prefillNeededAtChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").prefillNeededAtChunk;

fn expectEngineBill(json: []const u8, files: []const sdk.fixture.File) !void {
    const mem: sdk.fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try std.testing.expect(config.parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    const cfg = try toModelConfig(config.of(&p));
    // The stream room's shape (16K + 1K at the wide chunk), the standard prompt, a short one, and 8-bit KV.
    const shapes = [_][3]u64{ .{ 17_408, 8192, 16 }, .{ 18_380, 4096, 16 }, .{ 900, 4096, 16 }, .{ 18_380, 4096, 8 } };
    for (shapes) |s| {
        const want = prefillNeededAtChunk(&cfg, s[0], 1, s[2], s[1], .{});
        try std.testing.expect(want > 0);
        try std.testing.expectEqual(want, plugin.prefillBytes(&p, s[0], @intCast(s[1]), @intCast(s[2]), 0));
    }
}

test "bailing_hybrid bill: a streaming pack bills its prompt as the engine bills the same config unstreamed; another pack leaves it to the engine" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try std.testing.expect(config.parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    var b: abi.Bill = .{};
    plugin.table.bill(&p, &.{ .context = 32768 }, &b);
    try std.testing.expect(b.prefill_bytes.? == &plugin.prefillBytes);
    try std.testing.expect(p.caps.prompt_last_row and p.caps.streamed_weights);
    try expectEngineBill(json, mem.files);

    const tiny = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json");
    const tmem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = tiny }} };
    const tdir = tmem.packDir();
    var q: abi.Parsed = .{};
    try std.testing.expect(config.parse(&tdir, abi.Str.of(tiny), .{}, null, true, &q).isOk());
    defer config.freeParsed(&q);
    var bq: abi.Bill = .{};
    plugin.table.bill(&q, &.{ .context = 32768 }, &bq);
    try std.testing.expect(bq.prefill_bytes == null and !q.caps.prompt_last_row);
    // The same arithmetic on an affine 4-bit pack (the dequant-weights term) still equals the engine's.
    try expectEngineBill(tiny, tmem.files);
}
