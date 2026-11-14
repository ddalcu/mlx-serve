//! bailing_hybrid's prompt bill against the engine's. The plugin prices a streaming pack's prompt from
//! `sdk/bill/prefill.zig` (its closure stays in sdk/); this file, outside that closure, pins the figure to the engine's
//! `prefillNeededAtChunk` of the same config plus the whole-prompt pass's live set past a chunk's.

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
        const pass = if (p.caps.whole_prompt_prefill) plugin.passBytes(&p, s[0], @intCast(s[1])) * 5 / 4 else 0;
        try std.testing.expectEqual(want + pass, plugin.prefillBytes(&p, s[0], @intCast(s[1]), @intCast(s[2]), 0));
    }
}

test "bailing_hybrid bill: a streaming pack bills its prompt as the engine bills the same config unstreamed plus its whole-prompt pass; another pack leaves it to the engine" {
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

// Bar: the pass's live set past a chunk's at Ling-3.1-flash's geometry (top-8, hidden 4,096, 63 KDA layers' bf16
// checkpoints of 141,410,304 B), by term.
test "bailing_hybrid bill: the whole-prompt pass bills its routed call, its spans' rows and its checkpoints past one chunk's" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try std.testing.expect(config.parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    const cp: u64 = 141_410_304;
    try std.testing.expectEqual(cp, config.of(&p).ssmCheckpointBytes(2));
    // The standard prompt at the engine's 1,536-row chunk: 16,844 routed rows past a chunk (each pair's output and its
    // weighted copy, f32), their residual rows (4 x bf16), and 12 checkpoints (one a span of 18,380 / 1,536, plus one).
    try std.testing.expectEqual(16_844 * 8 * 4096 * 8 + 4 * 16_844 * 4096 * 2 + 12 * cp, plugin.passBytes(&p, 18_380, 1536));
    // A one-span prompt holds no rows past its chunk: the backoff's checkpoint and one more.
    try std.testing.expectEqual(2 * cp, plugin.passBytes(&p, 900, 4096));
    // The routed call stops at 16,384 + 2,048 rows: at 32,767 tokens it bills 18,432 - 1,024 of them.
    try std.testing.expectEqual((18_432 - 1024) * 8 * 4096 * 8 + 4 * (32_767 - 1024) * 4096 * 2 + 32 * cp, plugin.passBytes(&p, 32_767, 1024));
}
