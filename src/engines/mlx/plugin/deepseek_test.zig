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
const world = @import("../../../sdk/testing/world.zig");

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

test "dsv41 prefill chunk: the pass plan pins the widest whole-span pass the ladder's share bills (PF_IDX_PASS's queries included), up to the context, the explicit width over it, and the ladder's pin where not even 8192 rows fit" {
    const t = std.testing;
    const env = @import("../../../env.zig");
    const pc = @import("../admission/prefill_chunk.zig");
    const span = @import("../../../plugins/deepseek_v41/deepseek_v41.zig").SPAN;
    var cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    const was = env.override(.MLX_SERVE_DSV41_PF_PASS_SIZE, "1");
    defer was.restore();
    const was_idx = env.override(.MLX_SERVE_DSV41_PF_IDX_PASS, "1");
    defer was_idx.restore();
    // The memory-sized width, no working-set cap (PF_PASS_CEIL's test holds the cap).
    const no_cap = env.override(.MLX_SERVE_DSV41_PF_PASS_CEIL, "0");
    defer no_cap.restore();
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
    // PF_IDX_PASS's whole-pass index queries are billed too, and the pinned pass above fits with them.
    env.set(.MLX_SERVE_DSV41_PF_IDX_PASS, "0");
    const no_idx = dsv41PrefillMemoryNeeded(&cfg, wide, wide);
    env.set(.MLX_SERVE_DSV41_PF_IDX_PASS, "1");
    try t.expect(dsv41PrefillMemoryNeeded(&cfg, wide, wide) > no_idx);
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

/// One release share for PF_PASS_CEIL's test: the context, the ladder's share and the rows the plan pins under the
/// cap, as e0ea875c and 75d33cb3 pinned them on the M5 Ultra 512 GiB (39,680 rows over the memory-sized width).
const KneeCase = struct { ctx: u32, cap: u64, rows: u32 };

/// The release config's widest pass for every case under the working-set cap `knee`, and for every share from 8 to
/// 64 GiB at 1M rows of context: the memory-sized width, or `knee_rows` where that is narrower.
fn kneePlanHolds(cfg: *model.ModelConfig, cases: []const KneeCase, knee: u64, knee_rows: u32) bool {
    const pc = @import("../admission/prefill_chunk.zig");
    var held = true;
    for (cases) |cs| {
        cfg.ctx_override = cs.ctx;
        held = held and pc.dsv41PassRowsUnder(cfg, 16, cs.cap, knee, 0, 8192) == cs.rows;
    }
    cfg.ctx_override = 1 << 20;
    var cap: u64 = 8 << 30;
    while (cap <= 64 << 30) : (cap += 256 << 20) {
        const wide = pc.dsv41PassRowsUnder(cfg, 16, cap, 0, 0, 8192);
        held = held and pc.dsv41PassRowsUnder(cfg, 16, cap, knee, 0, 8192) == @min(wide, knee_rows);
    }
    return held;
}

test "dsv41 prefill chunk: PF_PASS_CEIL caps a memory-sized pass's working set at PASS_KNEE_BYTES, the bill of the 39,680 rows measured as the 1M knee on the M5 Ultra 512 GiB: the release shares at 16K, 256K and 1M plan as before, 128K, 256K and 512K prompts run in 4, 7 and 14 passes, a smaller share keeps its memory-sized width, a larger one stops at the cap, the explicit width is never capped; a cap 1,024 rows high fails" {
    const t = std.testing;
    const env = @import("../../../env.zig");
    const pc = @import("../admission/prefill_chunk.zig");
    const ds = @import("../../../plugins/deepseek_v41/deepseek_v41.zig");
    var cfg = try model.parseConfigFromJson(t.allocator, model.dsv41_release_config);
    // The levers at the defaults the knee was measured with.
    const was = env.override(.MLX_SERVE_DSV41_PF_PASS_SIZE, "1");
    defer was.restore();
    const was_idx = env.override(.MLX_SERVE_DSV41_PF_IDX_PASS, "1");
    defer was_idx.restore();
    const was_int8 = env.override(.MLX_SERVE_DSV41_PF_INT8, "0");
    defer was_int8.restore();
    const gib: u64 = 1 << 30;
    const mb: u64 = 1 << 20;
    const knee = ds.PASS_KNEE_BYTES;
    const knee_rows: u32 = 39_680;
    const Bill = struct {
        fn at(c: *const model.ModelConfig, rows: u64) u64 {
            return dsv41PrefillMemoryNeeded(c, rows, rows);
        }
    };
    // On by default: unset takes the cap, =0 none.
    {
        const unset = env.override(.MLX_SERVE_DSV41_PF_PASS_CEIL, null);
        defer unset.restore();
        try t.expectEqual(knee, ds.passKneeBytes());
    }
    {
        const zero = env.override(.MLX_SERVE_DSV41_PF_PASS_CEIL, "0");
        defer zero.restore();
        try t.expectEqual(@as(u64, 0), ds.passKneeBytes());
    }
    // Under the cap's 39,680 rows a 128K, 256K or 512K prompt runs in 4, 7 or 14 passes, where the formats'
    // memory-sized widths (NVFP4 44,544 and the MXFP4 view 47,616 at 512000 MiB) cut 3, 6 and 12 passes of 43,776 rows.
    for ([_]usize{ 131_072, 262_144, 524_288 }, [_]usize{ 32_768, 37_632, 37_632 }) |rows, want| {
        try t.expectEqual(want, ds.passRows(rows, knee_rows));
        try t.expectEqual(@as(usize, 43_776), ds.passRows(rows, 47_616));
    }
    // A smaller share keeps its memory-sized width, under the cap.
    cfg.ctx_override = 1 << 20;
    const small = pc.dsv41PassRowsUnder(&cfg, 16, 16 * gib, 0, 0, 8192);
    try t.expect(small > 8192 and Bill.at(&cfg, small) <= knee);
    try t.expectEqual(small, pc.dsv41PassRows(&cfg, 16, 16 * gib, 0, 8192));
    // A larger share (a 1 TB box's, or past it) stops at the cap: the widest whole-span pass whose bill fits it.
    for ([_]u64{ 64 * gib, 4096 * gib }) |cap| {
        const big = pc.dsv41PassRows(&cfg, 16, cap, 0, 8192);
        try t.expect(big < pc.dsv41PassRowsUnder(&cfg, 16, cap, 0, 0, 8192) and big % ds.SPAN == 0);
        try t.expect(Bill.at(&cfg, big) <= knee and Bill.at(&cfg, big + ds.SPAN) > knee);
        // The explicit width is never capped.
        try t.expectEqual(@as(u32, 47_360), pc.dsv41PassRows(&cfg, 16, cap, 47_360, 8192));
    }
    // Where the knee was measured (the matrix units), the cap is the bill of its 39,680 rows, and the release shares
    // (16K, 256K, the NVFP4 and the MXFP4 view's at 1M, and a wider one) pin what the row ceiling pinned there.
    if (!world.nax()) return;
    try t.expect(Bill.at(&cfg, knee_rows) <= knee and Bill.at(&cfg, knee_rows + ds.SPAN) > knee);
    const cases = [_]KneeCase{
        .{ .ctx = 32_768, .cap = 21_099 * mb, .rows = 32_768 },
        .{ .ctx = 1 << 18, .cap = 25_869 * mb, .rows = 39_680 },
        .{ .ctx = 1 << 20, .cap = 21_728 * mb, .rows = 39_424 },
        .{ .ctx = 1 << 20, .cap = 25_869 * mb, .rows = 39_680 },
        .{ .ctx = 1 << 20, .cap = 30 * gib, .rows = 39_680 },
    };
    try t.expect(kneePlanHolds(&cfg, &cases, knee, knee_rows));
    // The 16K standard test is one pass.
    cfg.ctx_override = 32_768;
    try t.expectEqual(@as(usize, 17_638), ds.passRows(17_638, pc.dsv41PassRows(&cfg, 16, 21_099 * mb, 0, 8192)));
    // The plant: a cap 1,024 rows high.
    try t.expect(!kneePlanHolds(&cfg, &cases, Bill.at(&cfg, knee_rows + 1024), knee_rows));
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

test "plugin bill parity: all five built-ins with non-zero bill terms match legacy ModelConfig" {
    const t = std.testing;
    const sdk = @import("../../../sdk/root.zig");
    const Binding = @import("config.zig").Binding;
    const slotFixedKvBytes = @import("../admission/bills.zig").slotFixedKvBytes;

    const cases = [_]struct { table: *const abi.ModelTable, json: []const u8 }{
        .{ .table = &@import("../../../plugins/deepseek_v4/plugin.zig").table, .json = config.release_config },
        .{ .table = &@import("../../../plugins/deepseek_v41/plugin.zig").table, .json = model.dsv41_release_config },
        .{ .table = &@import("../../../plugins/glm5_next/plugin.zig").table, .json = @embedFile("../../../fixtures/model-configs/glm5_next-mlx-community-4bit.json") },
        .{ .table = &@import("../../../plugins/bailing_hybrid/plugin.zig").table, .json = @embedFile("../../../fixtures/model-configs/bailing_hybrid-ling-3.0-flash-oq4e-mtp.json") },
        .{ .table = &@import("../../../plugins/qwen4_exp/plugin.zig").table, .json = @embedFile("../../../fixtures/model-configs/qwen4_exp-iq-mlx-4.7bpw.json") },
    };

    for (cases) |tc| {
        var legacy = try model.parseConfigFromJson(t.allocator, tc.json);
        defer legacy.deinit(t.allocator);
        const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = tc.json }} };
        const dir = mem.packDir();
        var binding: Binding = .{ .table = tc.table };
        try t.expect(tc.table.parse(&dir, .of(tc.json), .{}, null, true, &binding.parsed).isOk());
        defer tc.table.free_parsed(&binding.parsed);
        binding.readBillTerms();

        var bound = legacy;
        bound.plugin = &binding;
        try t.expectEqual(legacy.kvBytesPerToken(), bound.kvBytesPerToken());
        for ([_]u64{ 16, 8 }) |kv_bits| {
            for ([_]u64{ 4096, 8192 }) |chunk| {
                try t.expectEqual(slotFixedKvBytes(&legacy, kv_bits, chunk), slotFixedKvBytes(&bound, kv_bits, chunk));
            }
        }
        for ([_]u64{ 1, 100, 641, 4096, 65536, 1 << 20 }) |seq| {
            try t.expectEqual(legacy.prefillAttnKeys(seq), bound.prefillAttnKeys(seq));
        }
    }
}
