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
const prefillAdmissionBill = @import("../admission/admit.zig").prefillAdmissionBill;
const generate_mod = @import("../generate.zig");

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

test "dsv41 prefill chunk: an explicit --prefill-chunk past 8192 is billed and run at 8192" {
    const t = std.testing;
    var cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    generate_mod.prefillCli().override = 16384;
    generate_mod.prefillCli().explicit = true;
    defer {
        generate_mod.prefillCli().override = 8192;
        generate_mod.prefillCli().explicit = false;
    }
    try t.expectEqual(@as(u64, 8192), prefillAdmissionBill(&cfg, 65536, 256, null, null, .{}).chunk);
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

test "dsv41 prefill chunk: the pass plan pins the widest whole-span pass the ladder's share bills, up to the context, the explicit width over it, and the ladder's pin where not even 8192 rows fit" {
    const t = std.testing;
    const env = @import("../../../env.zig");
    const pc = @import("../admission/prefill_chunk.zig");
    const span = @import("../../../plugins/deepseek_v41/deepseek_v41.zig").SPAN;
    var cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    const was = env.override(.MLX_SERVE_DSV41_PF_PASS_SIZE, "1");
    defer was.restore();
    const gb: u64 = 1 << 30;
    // The share bounds it on a long context: the widest whole span that fits, the next one past the cap.
    cfg.ctx_override = 1 << 20;
    const cap: u64 = 25 * gb;
    const wide = pc.dsv41PassRows(&cfg, 16, cap, 0, 8192);
    try t.expect(wide > 8192 and wide % span == 0);
    try t.expect(dsv41PrefillMemoryNeeded(&cfg, wide, wide) <= cap);
    try t.expect(dsv41PrefillMemoryNeeded(&cfg, wide + span, wide + span) > cap);
    // Under the pass plan one MoE call takes every row, and the bill says so.
    env.set(.MLX_SERVE_DSV41_PF_PASS_SIZE, "0");
    const off = dsv41PrefillMemoryNeeded(&cfg, wide, wide);
    env.set(.MLX_SERVE_DSV41_PF_PASS_SIZE, "1");
    try t.expect(dsv41PrefillMemoryNeeded(&cfg, wide, wide) > off);
    // A wider share never pins a narrower pass; the context bounds it.
    try t.expect(pc.dsv41PassRows(&cfg, 16, 2 * cap, 0, 8192) >= wide);
    cfg.ctx_override = 32768;
    try t.expectEqual(@as(u32, 32768), pc.dsv41PassRows(&cfg, 16, 400 * gb, 0, 8192));
    // The explicit width, and the ladder's pin on a box whose share takes no 8192-row pass.
    try t.expectEqual(@as(u32, 12000), pc.dsv41PassRows(&cfg, 16, cap, 12000, 8192));
    try t.expectEqual(@as(u32, 2048), pc.dsv41PassRows(&cfg, 16, gb / 4, 0, 2048));
    // However wide the share or the flag, a pass's query heads stay indexable in 32 bits.
    cfg.ctx_override = 1 << 20;
    const limit = @import("../../../plugins/deepseek_v41/deepseek_v41.zig").passRowsLimit(&cfg);
    try t.expectEqual(@as(u64, 65_280), limit);
    try t.expectEqual(@as(u32, 65_280), pc.dsv41PassRows(&cfg, 16, 4096 * gb, 0, 8192));
    try t.expectEqual(@as(u32, 65_280), pc.dsv41PassRows(&cfg, 16, cap, 1 << 20, 8192));
}

test "dsv41 prefill chunk: under the pass plan the chunk loop cuts a prompt into near-equal chunks no wider than the pinned pass; off, 8192-row chunks and the rest" {
    const t = std.testing;
    const env = @import("../../../env.zig");
    var cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    const Loop = struct {
        // As generate/init.zig picks it: the request's admitted width (the pin), the pass plan's cut of it, else the cap.
        fn cuts(c: *const model.ModelConfig, rows: usize, out: []usize) []usize {
            const pin: usize = c.pinned_prefill_chunk;
            const chunk = c.passPrefillChunk(pin, rows) orelse c.cappedPrefillChunk(16384);
            var pos: usize = 0;
            var n: usize = 0;
            while (pos < rows) : (n += 1) {
                const end = generate_mod.nextChunkEnd(pos, rows, chunk, false, 0, 0, false, false);
                out[n] = end - pos;
                pos = end;
            }
            return out[0..n];
        }
    };
    var buf: [64]usize = undefined;
    // Off: 8192-row chunks and the rest.
    {
        const was = env.override(.MLX_SERVE_DSV41_PF_PASS_SIZE, "0");
        defer was.restore();
        cfg.pinned_prefill_chunk = 8192;
        try t.expectEqualSlices(usize, &.{ 8192, 8192, 1254 }, Loop.cuts(&cfg, 17_638, &buf));
    }
    // On: the pin is the pass (`pinPrefillChunk` sets both).
    cfg.dsv41_pass_rows = 32768;
    cfg.pinned_prefill_chunk = 32768;
    try t.expectEqualSlices(usize, &.{17_638}, Loop.cuts(&cfg, 17_638, &buf));
    try t.expectEqualSlices(usize, &.{ 20_224, 19_776 }, Loop.cuts(&cfg, 40_000, &buf));
    cfg.dsv41_pass_rows = 8192;
    cfg.pinned_prefill_chunk = 8192;
    try t.expectEqualSlices(usize, &.{ 5888, 5888, 5862 }, Loop.cuts(&cfg, 17_638, &buf));
    // A pin under the old cap (a tight box, an explicit narrower `--prefill-chunk`) keeps today's cut.
    cfg.dsv41_pass_rows = 4096;
    cfg.pinned_prefill_chunk = 4096;
    try t.expectEqual(@as(?usize, null), cfg.passPrefillChunk(4096, 17_638));
    // The forward's own config copy holds no pass (no pin was taken on it): the switch and the request's width decide.
    const copy = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    {
        const was = env.override(.MLX_SERVE_DSV41_PF_PASS_SIZE, "1");
        defer was.restore();
        try t.expectEqual(@as(?usize, 17_638), copy.passPrefillChunk(32768, 17_638));
        try t.expectEqual(@as(?usize, null), copy.passPrefillChunk(0, 17_638));
    }
    {
        const was = env.override(.MLX_SERVE_DSV41_PF_PASS_SIZE, "0");
        defer was.restore();
        try t.expectEqual(@as(?usize, null), copy.passPrefillChunk(32768, 17_638));
    }
}
