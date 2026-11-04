//! qwen4_exp's QSA score sheet upkeep: the block dump (`QWEN4_DUMP_QSA_BLOCKS`), and the bank, bill and fallback tests
//! of the trunk that feeds the SDK's score kernel.

const std = @import("std");
const env = @import("../../../env.zig");
const crt = @import("../../../sdk/util/crt.zig");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const history = @import("history.zig");
const score = @import("../../../sdk/blocks/attention/sparse_block/score.zig");
const select = @import("../../../sdk/blocks/attention/sparse_block/select.zig");
const Blocks = @import("../../../sdk/blocks/attention/sparse_block/params.zig").Blocks;
const testing = std.testing;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("../trunk.zig").Trunk;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verify_qmm_nax_vars = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const ssmFreeQsaState = @import("history.zig").ssmFreeQsaState;
const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT = score.QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
const qsaPooledRowContiguous = score.qsaPooledRowContiguous;
const qsaPrefillTransientBytes = score.qsaPrefillTransientBytes;
const qsaScoreFused = score.qsaScoreFused;
const qsaScoreFusedActive = score.qsaScoreFusedActive;
const qsaScoreFusedActiveFor = score.qsaScoreFusedActiveFor;
const qsaScoreFusedArm = score.qsaScoreFusedArm;
const qsaScoreFusedEligibleFrom = score.qsaScoreFusedEligibleFrom;
const qsaScoreRowsPerChunk = score.qsaScoreRowsPerChunk;
const qsaScoreSheet = score.qsaScoreSheet;
const qsaScoreSheetComposed = score.qsaScoreSheetComposed;
const qsaSelectTopBlocks = select.qsaSelectTopBlocks;

/// Test seam for the dump path; null reads `QWEN4_DUMP_QSA_BLOCKS`.
pub var qsa_dump_blocks_path_override: ?[]const u8 = null;
var qsa_dump_blocks_cached: ?[]const u8 = null;
var qsa_dump_blocks_read: bool = false;

fn qsaDumpBlocksPath() ?[]const u8 {
    if (qsa_dump_blocks_path_override) |p| {
        if (p.len == 0 or p[0] == '0') return null;
        return p;
    }
    if (qsa_dump_blocks_read) return qsa_dump_blocks_cached;
    qsa_dump_blocks_read = true;
    const raw = env.get(.QWEN4_DUMP_QSA_BLOCKS) orelse return null;
    if (!env.onValue(raw)) return null;
    const s = std.mem.sliceTo(raw, 0);
    qsa_dump_blocks_cached = s;
    return s;
}

pub fn qsaDumpQsaBlocks(layer: u32, row0: c_int, rows: c_int, kb: c_int, blocks: mlx.mlx_array) void {
    const path = qsaDumpBlocksPath() orelse return;
    if (blocks.ctx == null) return;
    if (mlx.mlx_array_eval(blocks) != 0) return;
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    const s = mlx.gpuStream();
    if (mlx.mlx_contiguous(&contig, blocks, false, s) != 0) return;
    if (mlx.mlx_array_eval(contig) != 0) return;
    const data = mlx.mlx_array_data_int32(contig) orelse return;
    const n = mlx.mlx_array_size(contig);
    var path_z: [1024]u8 = undefined;
    if (path.len == 0 or path.len >= path_z.len) return;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;
    const fd = crt.openWrite(@ptrCast(&path_z), false);
    if (fd < 0) return;
    defer crt.close(fd);
    var hdr: [128]u8 = undefined;
    const header = std.fmt.bufPrint(&hdr, "layer={d} row0={d} rows={d} kb={d}\n", .{ layer, row0, rows, kb }) catch return;
    _ = crt.write(fd, header);
    var buf: [32]u8 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const piece = if (i == 0)
            std.fmt.bufPrint(&buf, "{d}", .{data[i]}) catch return
        else
            std.fmt.bufPrint(&buf, " {d}", .{data[i]}) catch return;
        _ = crt.write(fd, piece);
    }
    _ = crt.write(fd, "\n");
}

test "qsa score bank: incremental column append matches a from-scratch rebuild" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Transformer = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    t.qsa_score_bank_builds = 0;

    const hd: c_int = 16;
    const batch: c_int = 1;
    var prng = std.Random.DefaultPrng.init(0x5C0E_BA11);
    const rnd = prng.random();

    var scratch: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&scratch);
    var incr: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&incr);

    const before = history.qsa_score_incremental_appends;
    var nb: c_int = 0;
    var step_i: usize = 0;
    while (step_i < 24) : (step_i += 1) {
        const add: c_int = if (step_i == 0) 5 else 1;
        const new3 = try attn256RandBf16(rnd, &[_]c_int{ batch, add, hd }, s);
        defer _ = mlx.mlx_array_free(new3);
        try t.qsaAppendPooled(&scratch, new3, nb);
        try t.qsaAppendPooled(&incr, new3, nb);
        nb += add;
        _ = try t.qsaScoreBank(&incr, batch, nb, hd);
    }
    try testing.expectEqual(before + 23, history.qsa_score_incremental_appends);
    try testing.expect(mlx.getShape(incr.qsa_score_buf)[3] > nb);
    const bank_inc = try t.qsaScoreBank(&incr, batch, nb, hd);
    const bank_scratch = try t.qsaScoreBank(&scratch, batch, nb, hd);
    try testing.expectEqual(@as(c_int, hd), mlx.getShape(bank_inc)[2]);
    try testing.expectEqual(nb, mlx.getShape(bank_inc)[3]);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(bank_inc, bank_scratch, s));
    {
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_reshape(&k_rope, scratch.qsa_pooled, &[_]c_int{ batch, 1, nb, hd }, 4, s));
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, k_rope, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
        var k32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32);
        try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(bank_inc, k32, s));
    }
    try testing.expect(history.qsa_score_incremental_appends > before);
}

test "qsa score fused: the bill follows the predicate" {
    const saved_fused = score.qsa_score_fused_override;
    const saved_nax = verify_qmm_nax_vars.vqmm_nax_probe_override;
    const saved_sheet = score.qsa_score_sheet_mb_override;
    defer {
        score.qsa_score_fused_override = saved_fused;
        verify_qmm_nax_vars.vqmm_nax_probe_override = saved_nax;
        score.qsa_score_sheet_mb_override = saved_sheet;
    }
    score.qsa_score_sheet_mb_override = QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    const n_idx: u64 = 4;
    const fwd: u64 = 4096;
    const nb: u64 = 262144;
    const kv = nb * 4;
    const ratio: u64 = 4;

    verify_qmm_nax_vars.vqmm_nax_probe_override = false;
    score.qsa_score_fused_override = true;
    const composed_rows = qsaScoreRowsPerChunk(n_idx, nb, fwd, 128);
    const composed_bill = qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128);
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    const composed_bank = cfg.qsaScoreBankBytesPerToken();
    try testing.expectEqual(@as(u64, 64), composed_rows);
    try testing.expectEqual(@as(u64, 1_536), composed_bank);

    score.qsa_score_fused_override = false;
    try testing.expectEqual(composed_rows, qsaScoreRowsPerChunk(n_idx, nb, fwd, 128));
    try testing.expectEqual(composed_bill, qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128));
    try testing.expectEqual(composed_bank, cfg.qsaScoreBankBytesPerToken());

    score.qsa_score_fused_override = true;
    verify_qmm_nax_vars.vqmm_nax_probe_override = null;
    qsaScoreFusedArm();
    if (qsaScoreFusedActive()) {
        const fused_rows = qsaScoreRowsPerChunk(n_idx, nb, fwd, 128);
        try testing.expectEqual(composed_rows * 4, fused_rows);
        try testing.expect(fused_rows >= 128);
        const fused_bill = qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128);
        try testing.expectEqual(fused_rows * nb * 4 + fwd * @min(nb, 512) * 4, fused_bill);
        try testing.expectEqual(@as(u64, 0), cfg.qsaScoreBankBytesPerToken());
        const saved_sel = select.qsa_select_kernel_override;
        defer select.qsa_select_kernel_override = saved_sel;
        select.qsa_select_kernel_override = false;
        try testing.expectEqual(fused_rows * nb * 4 * 4 + fwd * @min(nb, 512) * 4, qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128));
    }
}
test "qsa score fused: no bank is built under the fused arm" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    xfm.qsa_score_bank_builds = 0;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;
    entry.qsa_reserve_rows = 256;
    var prng = std.Random.DefaultPrng.init(0xBA11);
    const hd: c_int = 128;
    const nb: c_int = 32;
    const pooled = try attn256RandBf16(prng.random(), &[_]c_int{ 1, nb, hd }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try xfm.qsaAppendPooled(&entry, pooled, 0);
    const saved_fused = score.qsa_score_fused_override;
    const saved_nax = verify_qmm_nax_vars.vqmm_nax_probe_override;
    defer {
        score.qsa_score_fused_override = saved_fused;
        verify_qmm_nax_vars.vqmm_nax_probe_override = saved_nax;
    }
    score.qsa_score_fused_override = false;
    verify_qmm_nax_vars.vqmm_nax_probe_override = false;
    _ = try xfm.qsaScoreOperand(&entry, 1, nb, hd, false);
    try testing.expect(entry.qsa_score_bank.ctx != null);
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    entry.qsa_score_bank = .{ .ctx = null };
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    score.qsa_score_fused_override = true;
    verify_qmm_nax_vars.vqmm_nax_probe_override = null;
    qsaScoreFusedArm();
    if (qsaScoreFusedActive()) {
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, hd, true);
        try testing.expect(entry.qsa_score_bank.ctx == null);
    }
}

test "qsa score fused: unsupported indexer geometry falls back to the composed chain" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    xfm.qsa_score_bank_builds = 0;
    const saved_fused = score.qsa_score_fused_override;
    const saved_nax = verify_qmm_nax_vars.vqmm_nax_probe_override;
    const saved_sheet = score.qsa_score_sheet_mb_override;
    defer {
        score.qsa_score_fused_override = saved_fused;
        verify_qmm_nax_vars.vqmm_nax_probe_override = saved_nax;
        score.qsa_score_sheet_mb_override = saved_sheet;
    }
    score.qsa_score_sheet_mb_override = QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    score.qsa_score_fused_override = true;

    const fwd: u64 = 4096;
    const kv: u64 = 262144 * 4;
    const ratio: u64 = 4;
    score.qsa_score_fused_override = false;
    const composed_n2 = qsaPrefillTransientBytes(2, fwd, kv, ratio, 128);
    const composed_hd64 = qsaPrefillTransientBytes(4, fwd, kv, ratio, 64);
    const composed_rows_n2 = qsaScoreRowsPerChunk(2, kv / ratio, fwd, 128);
    const composed_rows_hd64 = qsaScoreRowsPerChunk(4, kv / ratio, fwd, 64);
    score.qsa_score_fused_override = true;
    try testing.expect(!qsaScoreFusedActiveFor(1, 2, 128));
    try testing.expect(!qsaScoreFusedActiveFor(1, 4, 64));
    try testing.expectEqual(composed_n2, qsaPrefillTransientBytes(2, fwd, kv, ratio, 128));
    try testing.expectEqual(composed_hd64, qsaPrefillTransientBytes(4, fwd, kv, ratio, 64));
    try testing.expectEqual(composed_rows_n2, qsaScoreRowsPerChunk(2, kv / ratio, fwd, 128));
    try testing.expectEqual(composed_rows_hd64, qsaScoreRowsPerChunk(4, kv / ratio, fwd, 64));

    var cfg2 = model_mod.ModelConfig{};
    cfg2.num_hidden_layers = 48;
    cfg2.full_attention_interval = 4;
    cfg2.indexer_budget = 2048;
    cfg2.indexer_n_heads = 2;
    cfg2.indexer_head_dim = 128;
    cfg2.indexer_compress_ratio = 4;
    const bank2 = cfg2.qsaScoreBankBytesPerToken();
    try testing.expect(bank2 > 0);
    var cfg64 = cfg2;
    cfg64.indexer_n_heads = 4;
    cfg64.indexer_head_dim = 64;
    try testing.expect(cfg64.qsaScoreBankBytesPerToken() > 0);

    var prng = std.Random.DefaultPrng.init(0x6E01);
    const rnd = prng.random();
    const nb: c_int = 32;
    const rows: c_int = 4;
    {
        var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        defer ssmFreeQsaState(&entry);
        entry.qsa_ratio = 4;
        entry.qsa_reserve_rows = 256;
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try xfm.qsaAppendPooled(&entry, pooled, 0);
        const fused = qsaScoreFusedActiveFor(1, 2, 128);
        try testing.expect(!fused);
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, 128, fused);
        try testing.expect(entry.qsa_score_bank.ctx != null);
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q);
        const sheet = try qsaScoreSheet(s, q, pooled, entry.qsa_score_bank, fused);
        defer _ = mlx.mlx_array_free(sheet);
        const want = try qsaScoreSheetComposed(s, q, entry.qsa_score_bank);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(sheet, want, s));
    }
    {
        var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        defer ssmFreeQsaState(&entry);
        entry.qsa_ratio = 4;
        entry.qsa_reserve_rows = 256;
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 64 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try xfm.qsaAppendPooled(&entry, pooled, 0);
        const fused = qsaScoreFusedActiveFor(1, 4, 64);
        try testing.expect(!fused);
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, 64, fused);
        try testing.expect(entry.qsa_score_bank.ctx != null);
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 64 }, s);
        defer _ = mlx.mlx_array_free(q);
        const sheet = try qsaScoreSheet(s, q, pooled, entry.qsa_score_bank, fused);
        defer _ = mlx.mlx_array_free(sheet);
        const want = try qsaScoreSheetComposed(s, q, entry.qsa_score_bank);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(sheet, want, s));
    }
    {
        var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        defer ssmFreeQsaState(&entry);
        entry.qsa_ratio = 4;
        entry.qsa_reserve_rows = 256;
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try xfm.qsaAppendPooled(&entry, pooled, 0);
        try mlx.check(mlx.mlx_array_eval(pooled));
        const q_bf = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q_bf);
        var q32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q32);
        try mlx.check(mlx.mlx_astype(&q32, q_bf, .float32, s));
        const fused = qsaScoreFusedEligibleFrom(q32, entry.qsa_pooled) and
            qsaPooledRowContiguous(entry.qsa_pooled) and mlx.streamIsGpu(s);
        try testing.expect(!fused);
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, 128, fused);
        try testing.expect(entry.qsa_score_bank.ctx != null);
        const sheet = try qsaScoreSheet(s, q32, pooled, entry.qsa_score_bank, fused);
        defer _ = mlx.mlx_array_free(sheet);
        const want = try qsaScoreSheetComposed(s, q32, entry.qsa_score_bank);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(sheet, want, s));
    }
}

test "qsa score fused: dump hook writes a parseable line" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    var file_buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&file_buf, "{s}/qsa-blocks.txt", .{path_buf[0..root_len]});
    qsa_dump_blocks_path_override = path;
    defer qsa_dump_blocks_path_override = null;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    score.qsa_score_fused_override = true;
    defer score.qsa_score_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0xD00B);
    const rnd = prng.random();
    const nb: c_int = 64;
    const rows: c_int = 2;
    const k: c_int = 8;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    const fused = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    blocks4.selectBounds(bounds_host, nb * 4, nb, true);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const ids = (try qsaSelectTopBlocks(s, fused, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(ids);
    qsaDumpQsaBlocks(3, 12, rows, k, ids);
    const body = try tmp.dir.readFileAlloc(std.testing.io, "qsa-blocks.txt", ta, .limited(1 << 20));
    defer ta.free(body);
    try testing.expect(std.mem.startsWith(u8, body, "layer=3 row0=12 rows=2 kb=8\n"));
    try testing.expect(std.mem.indexOfScalar(u8, body, ' ') != null);
}

/// The block geometry of the dump test.
const blocks4: Blocks = .{ .size = 4, .top_k = 512 };
