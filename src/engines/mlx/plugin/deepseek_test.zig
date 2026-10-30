//! The DeepSeek plugins' bills and configs pinned against the engine's own (`ModelConfig`, `prefillNeededAtChunk`).

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const config = @import("../../../plugins/deepseek_v4/config.zig");
const bill = @import("../../../plugins/deepseek_v4/bill.zig").bill;
const model = @import("../../../sdk/config/model.zig");
const v41_config = @import("../../../plugins/deepseek_v41/config.zig");
const v41_bill = @import("../../../plugins/deepseek_v41/bill.zig");
const prefillNeededAtChunk = @import("../admission/prefill_chunk.zig").prefillNeededAtChunk;
const dsv41PrefillMemoryNeeded = @import("../admission/prefill_chunk.zig").dsv41PrefillMemoryNeeded;
const resolvePrefillChunk = @import("../admission/prefill_chunk.zig").resolvePrefillChunk;
const effectivePrefillChunk = @import("../generate/prefill_chunk.zig").effectivePrefillChunk;

test "deepseek_v4 bill: the plugin's terms equal the engine's on the release config" {
    const t = std.testing;
    const sdk = @import("../../../sdk/root.zig");
    const legacy = try model.parseConfigFromJson(t.allocator, config.release_config);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = config.release_config }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try t.expect(config.parse(&dir, .of(config.release_config), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{ .context = 1 << 20 }, &b);
    try t.expectEqual(legacy.kvBytesPerToken(), b.kv_per_token);
    for ([_]u64{ 1, 100, 128, 129, 4096, 65536, 1 << 20 }) |seq| {
        try t.expectEqual(legacy.prefillAttnKeys(seq), b.prefill_attn_keys.?(&p, seq));
        try t.expectEqual(prefillNeededAtChunk(&legacy, seq, 256, 16, 4096, .{}), b.prefill_bytes.?(&p, seq, 4096, 16, 0));
    }
}

test "dsv41PrefillMemoryNeeded: state per compressed token, the chunk's stream, one span's transients" {
    const t = std.testing;
    const cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    const gb: u64 = 1 << 30;
    // A short prompt is the runtime floor and one span, nowhere near a
    // dense 40-layer 512-wide KV bill.
    try t.expect(dsv41PrefillMemoryNeeded(&cfg, 64, 4096) < gb);
    // The chunk's stream rides every layer: a wider chunk costs its rows.
    const d4 = dsv41PrefillMemoryNeeded(&cfg, 32768, 4096);
    const d8 = dsv41PrefillMemoryNeeded(&cfg, 32768, 8192);
    try t.expect(d8 > d4 + 4096 * 4 * 5120 * 2 * 2);
    // 1M tokens of context: ~3.2 KB per token of compressed caches plus the
    // [span, entries] index scores, bounded, never [span, heads, entries].
    const far = dsv41PrefillMemoryNeeded(&cfg, 1 << 20, 4096);
    try t.expect(far > (1 << 20) * 3200);
    try t.expect(far < 16 * gb);
    // The one estimator the guard reads.
    try t.expectEqual(far, prefillNeededAtChunk(&cfg, 1 << 20, 256, 16, 4096, .{}));
}

test "deepseek_v41 bill: the plugin's terms equal the engine's on the release config" {
    const t = std.testing;
    const sdk = @import("../../../sdk/root.zig");
    const json = model.dsv41_release_config;
    const legacy = try model.parseConfigFromJson(t.allocator, json);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try t.expect(v41_config.parse(&dir, .of(json), .{}, null, true, &p).isOk());
    defer v41_config.freeParsed(&p);
    var b: abi.Bill = .{};
    v41_bill.bill(&p, &.{ .context = 1 << 20 }, &b);
    try t.expectEqual(legacy.kvBytesPerToken(), b.kv_per_token);
    for ([_]u64{ 1, 100, 640, 641, 4096, 1 << 20 }) |seq| {
        try t.expectEqual(legacy.prefillAttnKeys(seq), b.prefill_attn_keys.?(&p, seq));
        try t.expectEqual(prefillNeededAtChunk(&legacy, seq, 256, 16, 4096, .{}), b.prefill_bytes.?(&p, seq, 4096, 16, 0));
    }
}

test "dsv41 prefill chunk: pinned at 8192 on a big machine, and kept there however long the prompt" {
    const t = std.testing;
    var cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    const gb: u64 = 1 << 30;
    cfg.pinned_prefill_chunk = resolvePrefillChunk(&cfg, 16, 240 * gb, 140 * gb, 0, 0);
    try t.expectEqual(@as(u32, 8192), cfg.pinned_prefill_chunk);
    for ([_]usize{ 8192, 65536, 131_072, 1 << 20 }) |ctx|
        try t.expectEqual(@as(usize, 8192), effectivePrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, ctx, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), cfg.pinned_prefill_chunk));
}
