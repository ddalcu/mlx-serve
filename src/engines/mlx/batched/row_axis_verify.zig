//! engines/mlx/batched/row_axis_verify.
//! Owns VerifyLaps, VerifyRollback, forwardRowAxisVerify, verifyLaps, takeJoinedVerifyLogits, expectRowsByteEqual, and 9 more.

const std = @import("std");
const modules = @import("../modules/call.zig");
const env = @import("../../../env.zig");
const distributed_mod = @import("../cluster/distributed.zig");
const mlx = @import("../../../mlx/ffi.zig");
const KVQuantConfig = @import("../cache/kv_quant.zig").KVQuantConfig;
const model_mod = @import("../../../sdk/config/model.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util_mod = @import("../../../sdk/util/io_util.zig");
const gather_qmv_rows = @import("../../../sdk/blocks/moe/gather_qmv_rows.zig");
const layered = @import("layered.zig");
const intree = @import("../plugin/intree.zig");
const mtp = @import("../../../plugins/qwen4_exp/mtp.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const Qwen4TestSlot = @import("../../../plugins/qwen4_exp/forward.zig").Qwen4TestSlot;
const Qwen4Trunk = @import("../../../plugins/qwen4_exp/trunk.zig").Trunk;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const qsaArraysAllEqual = @import("../../../sdk/testing/parity.zig").qsaArraysAllEqual;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const mtpHeadRowsFit = @import("../../../sdk/blocks/linear/verify_projection.zig").mtpHeadRowsFit;
const qwen4ReadF32 = @import("../../../sdk/testing/parity.zig").qwen4ReadF32;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const KVCacheEntry = @import("../kv/cache.zig").KVCacheEntry;
const freeKVEntry = @import("../kv/cache.zig").freeKVEntry;
const newEmptyKVEntry = @import("../kv/cache.zig").newEmptyKVEntry;
const testing = @import("../kv/cache.zig").testing;
const SSMCacheEntry = @import("../ssm/entry.zig").SSMCacheEntry;
const SSMCacheEntrySnapshot = @import("../ssm/entry.zig").SSMCacheEntrySnapshot;
const ssmFreeSpecCapture = @import("../ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../ssm/entry.zig").ssmRollbackFromCapture;
const ssmSnapshot = @import("../ssm/entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("../ssm/entry.zig").ssmSnapshotDeinit;
const nextSsmMemberGen = @import("../ssm/group.zig").nextSsmMemberGen;
const Transformer = @import("../transformer.zig").Transformer;
const deinit = @import("../transformer.zig").Transformer.deinit;
const MAX_BATCH_ROWS = @import("layered.zig").MAX_BATCH_ROWS;

pub const VerifyLaps = struct {
    build_ns: u64 = 0,
    ple_ns: u64 = 0,
    ple_sync_ns: u64 = 0,
};

pub const VerifyRollback = struct {
    const KvMark = struct { offset: usize, initialized: bool };

    allocator: std.mem.Allocator,
    /// The trunk's per-forward QSA scratch key, bumped on restore.
    fwd_gen: *u64,
    ctx: *ForwardCtx,
    kv_marks: []KvMark,
    cache_step: usize,
    ssm: []SSMCacheEntrySnapshot,
    offset: usize,

    pub fn capture(allocator: std.mem.Allocator, fwd_gen: *u64, ctx: *ForwardCtx) !@This() {
        const marks = try allocator.alloc(KvMark, ctx.cache.entries.len);
        errdefer allocator.free(marks);
        for (ctx.cache.entries, marks) |entry, *mark| mark.* = .{ .offset = entry.offset, .initialized = entry.initialized };
        const entries = ctx.ssm_entries orelse &.{};
        const states = try allocator.alloc(SSMCacheEntrySnapshot, entries.len);
        for (entries, states) |*entry, *state| state.* = ssmSnapshot(entry);
        return .{ .allocator = allocator, .fwd_gen = fwd_gen, .ctx = ctx, .kv_marks = marks, .cache_step = ctx.cache.step, .ssm = states, .offset = ctx.moe_seq_offset.* };
    }

    pub fn restore(self: *@This()) void {
        for (self.ctx.cache.entries, self.kv_marks) |*current, saved| {
            if (!saved.initialized) {
                freeKVEntry(current);
                current.* = newEmptyKVEntry();
            } else {
                // As with KVCache.restore, the next update rebuilds views from the retained prefix.
                inline for (.{ "key_view", "value_view", "key_scales_view", "key_biases_view", "value_scales_view", "value_biases_view" }) |field| {
                    _ = mlx.mlx_array_free(@field(current, field));
                    @field(current, field) = mlx.mlx_array_new();
                }
            }
            current.offset = saved.offset;
        }
        self.ctx.cache.step = self.cache_step;
        self.ctx.moe_seq_offset.* = self.offset;
        if (self.ctx.ssm_entries) |entries| {
            for (entries, self.ssm) |*entry, *saved| {
                _ = mlx.mlx_array_free(entry.conv_state);
                _ = mlx.mlx_array_free(entry.ssm_state);
                ssmFreeSpecCapture(entry);
                ssmFreeQsaState(entry);
                entry.conv_state = saved.conv_state;
                entry.ssm_state = saved.ssm_state;
                entry.aux_state = saved.aux_state;
                entry.qsa_pooled = saved.qsa_pooled;
                saved.conv_state = .{ .ctx = null };
                saved.ssm_state = .{ .ctx = null };
                saved.aux_state = .{ .ctx = null };
                saved.qsa_pooled = .{ .ctx = null };
                entry.initialized = saved.initialized;
                entry.qsa_ratio = saved.qsa_ratio;
                entry.qsa_hist_rows = saved.qsa_rows;
                entry.ple_prev = saved.ple_prev;
                entry.ple_prev_valid = saved.ple_prev_valid;
                if (entry.aux_state.ctx != null) entry.qsa_key_rows = mlx.getShape(entry.aux_state)[1];
                if (entry.qsa_pooled.ctx != null) entry.qsa_pooled_blocks = mlx.getShape(entry.qsa_pooled)[1];
            }
        }
        self.ctx.ssm_member_gen = nextSsmMemberGen();
        self.fwd_gen.* +%= 1;
    }

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.kv_marks);
        for (self.ssm) |*state| ssmSnapshotDeinit(state);
        self.allocator.free(self.ssm);
    }
};

pub fn forwardRowAxisVerify(
    self: *Transformer,
    token_rows: []const mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    out_logits: []mlx.mlx_array,
    out_last: []mlx.mlx_array,
    out_all: []mlx.mlx_array,
    row_ns: ?[]u64,
    rollback_out: ?[]VerifyRollback,
) !void {
    std.debug.assert(token_rows.len == ctxs.len);
    std.debug.assert(out_logits.len >= token_rows.len and out_last.len >= token_rows.len and out_all.len >= token_rows.len);
    // qwen4_exp's grouped verify runs on its trunk module (a single row keeps the plugin forward below), owed PLE work
    // settled first.
    if (token_rows.len > 1) if (intree.inner(self)) |t| {
        try self.flushDeferredPleGroup(ctxs);
        return modules.call("qwen4_row_axis_verify", .{ t, token_rows, ctxs, out_logits, out_last, out_all, row_ns, rollback_out });
    };
    const outer_group_rows = verify_projection.verify_group_rows;
    verify_projection.verify_group_rows = token_rows.len;
    defer verify_projection.verify_group_rows = outer_group_rows;
    if (token_rows.len > 1 and !layered.row_axis_verify_logged) {
        layered.row_axis_verify_logged = true;
        log.info("[batched] row-axis mtp verify engaged (slots={d}, width={d})\n", .{ token_rows.len, mlx.getShape(token_rows[0])[1] });
    }
    if (row_ns) |ns| @memset(ns[0..token_rows.len], 0);
    self.verify_laps = .{};
    var lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    var built: usize = 0;
    var touched: usize = 0;
    errdefer {
        for (0..built) |i| {
            _ = mlx.mlx_array_free(out_logits[i]);
            _ = mlx.mlx_array_free(out_last[i]);
            _ = mlx.mlx_array_free(out_all[i]);
        }
        for (0..touched) |i| {
            self.discardDeferredPle(ctxs[i]);
            if (ctxs[i].ssm_entries) |entries| for (entries) |*e| ssmFreeSpecCapture(e);
        }
    }
    for (token_rows, ctxs, 0..) |row, ctx, i| {
        touched = i + 1;
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        // Per-position SSM capture lets a partial accept roll back without a re-forward; a served model's deferred
        // host work keeps out of the graph build.
        ctx.capture_ssm_seq = ctx.ssm_entries != null;
        ctx.ple_defer = true;
        const logits = self.forwardWithCaptureAll(ctx, row, &last, &all) catch |e| {
            ctx.ple_defer = false;
            ctx.capture_ssm_seq = false;
            _ = mlx.mlx_array_free(last);
            _ = mlx.mlx_array_free(all);
            return e;
        };
        ctx.ple_defer = false;
        ctx.capture_ssm_seq = false;
        out_logits[i] = logits;
        out_last[i] = last;
        out_all[i] = all;
        built = i + 1;
        const row_build_ns = lap.read();
        self.verify_laps.build_ns += row_build_ns;
        lap.reset();
        if (row_ns) |ns| ns[i] += row_build_ns;
    }
    // Settles the deferred host work after every row's build, before the first eval downstream of it.
    try self.flushDeferredPleGroup(ctxs[0..token_rows.len]);
    const ple_ns = lap.read();
    self.verify_laps.ple_ns = ple_ns;
    if (row_ns) |ns| ns[0] += ple_ns;
}

/// The laps of the last grouped verify: qwen4_exp's trunk keeps its own.
pub fn verifyLaps(self: *Transformer) *VerifyLaps {
    if (intree.trunk(self)) |t| return &t.verify_laps;
    return &self.verify_laps;
}

/// Hand the joined verify block to the caller (null when the projection ran per row; only qwen4_exp's grouped verify
/// joins one).
pub fn takeJoinedVerifyLogits(self: *Transformer) mlx.mlx_array {
    const t = intree.trunk(self) orelse return .{ .ctx = null };
    return t.takeJoinedVerifyLogits();
}

// ── tests ──

pub fn expectRowsByteEqual(alloc: std.mem.Allocator, s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, tag: []const u8) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    if (ok) return;
    const ha = try qwen4ReadF32(alloc, a, s);
    defer alloc.free(ha);
    const hb = try qwen4ReadF32(alloc, b, s);
    defer alloc.free(hb);
    var max_diff: f64 = 0;
    var first: usize = std.math.maxInt(usize);
    var n_diff: usize = 0;
    for (ha, hb, 0..) |x, y, i| {
        if (x == y) continue;
        n_diff += 1;
        if (first == std.math.maxInt(usize)) first = i;
        const d = @abs(@as(f64, x) - @as(f64, y));
        if (d > max_diff) max_diff = d;
    }
    std.debug.print("[mtp verify] {s}: NOT byte-equal — {d}/{d} elements differ, first at {d} ({d} vs {d}), max |diff| {d}\n", .{ tag, n_diff, ha.len, first, ha[first], hb[first], max_diff });
    return error.RowNotByteEqual;
}

pub const L10VerifySlots = struct {
    solo: [2]*Qwen4TestSlot,
    group: [2]*Qwen4TestSlot,

    pub fn init(alloc: std.mem.Allocator, xfm: anytype, n_layers: u32, prompts: [2][]const i32) !L10VerifySlots {
        var out: L10VerifySlots = undefined;
        for (0..2) |i| {
            out.solo[i] = try Qwen4TestSlot.init(alloc, n_layers);
            out.group[i] = try Qwen4TestSlot.init(alloc, n_layers);
            _ = mlx.mlx_array_free(try out.solo[i].forward(xfm, prompts[i]));
            _ = mlx.mlx_array_free(try out.group[i].forward(xfm, prompts[i]));
        }
        return out;
    }
    pub fn deinit(self: *L10VerifySlots, alloc: std.mem.Allocator) void {
        for (0..2) |i| {
            self.solo[i].deinit(alloc);
            self.group[i].deinit(alloc);
        }
    }
};

/// The solo MTP verify: capture armed, PLE deferred, one `forwardWithCaptureAll`, flush
/// after. `forwardRowAxisVerify` at one row must match it.
pub fn logicalSsmCapture(s: mlx.mlx_stream, entry: *const SSMCacheEntry) !mlx.mlx_array {
    const shape = mlx.getShape(entry.spec_state_seq);
    if (shape.len != 5 or shape[0] < 1) return error.SpecCaptureShape;
    var final = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(final);
    try mlx.check(mlx.mlx_expand_dims(&final, entry.ssm_state, 0, s));
    if (shape[0] == 1) return final;
    defer _ = mlx.mlx_array_free(final);
    var prefix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prefix);
    var stop: [5]c_int = undefined;
    @memcpy(&stop, shape);
    stop[0] -= 1;
    const start: [5]c_int = @splat(0);
    const stride: [5]c_int = @splat(1);
    try mlx.check(mlx.mlx_slice(&prefix, entry.spec_state_seq, &start, 5, &stop, 5, &stride, 5, s));
    return Transformer.concatAxis0(s, &.{ prefix, final });
}

test "row-axis verify: every row of a group verify is byte-identical to its solo verify" {
    const model_dir = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Qwen4Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.prepare();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..320], ids[64..232] };
    const indexed_before = verify_projection.mtp_verify_indexed_input_calls;
    const reduce_before = verify_projection.mtp_verify_expert_reduce_calls;
    const expert_before = verify_projection.mtp_verify_expert_pairs_calls;
    const attn_before = verify_projection.mtp_verify_attn_rows_calls;
    const lmhead_before = verify_projection.mtp_verify_lmhead_rows_calls;
    const gdn_before = verify_projection.mtp_verify_gdn_rows_calls;
    const shared_gate_before = verify_projection.mtp_verify_shared_gate_rows_calls;
    const grouped_before = mtp.mtp_verify_moe_group_graphs;

    // A solo verify of up to 8 rows takes the MoE rows arm, a joined group past 8 the sorted
    // verify kernels, which reduce differently: rows match only while both sides run one arm.
    const old_rows_arm = gather_qmv_rows.moe_verify_rows_override;
    defer gather_qmv_rows.moe_verify_rows_override = old_rows_arm;
    const Pass = struct { rows_arm: bool, shapes: []const [2]usize };
    const passes = [_]Pass{
        .{ .rows_arm = false, .shapes = &.{ .{ 2, 2 }, .{ 5, 5 }, .{ 2, 3 }, .{ 3, 2 }, .{ 9, 9 }, .{ 2, 9 }, .{ 9, 3 } } },
        .{ .rows_arm = true, .shapes = &.{ .{ 2, 2 }, .{ 2, 3 }, .{ 3, 2 } } },
    };
    var group_calls: u64 = 0;
    for (passes) |pass| for (pass.shapes) |S_row| {
        gather_qmv_rows.moe_verify_rows_override = pass.rows_arm;
        group_calls += 1;
        var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
        defer slots.deinit(allocator);

        var rows: [2]mlx.mlx_array = undefined;
        defer for (rows) |r| {
            _ = mlx.mlx_array_free(r);
        };
        for (0..2) |i| {
            const shape = [_]c_int{ 1, @intCast(S_row[i]) };
            rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S_row[i]].ptr), &shape, 2, .int32);
        }

        var solo_logits: [2]mlx.mlx_array = undefined;
        var solo_last: [2]mlx.mlx_array = undefined;
        var solo_all: [2]mlx.mlx_array = undefined;
        defer for (0..2) |i| {
            _ = mlx.mlx_array_free(solo_logits[i]);
            _ = mlx.mlx_array_free(solo_last[i]);
            _ = mlx.mlx_array_free(solo_all[i]);
        };
        for (0..2) |i| {
            const one_row = [_]mlx.mlx_array{rows[i]};
            const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
            try xfm.forwardRowAxisVerify(&one_row, &one_ctx, solo_logits[i .. i + 1], solo_last[i .. i + 1], solo_all[i .. i + 1], null, null);
        }

        var grp_logits: [2]mlx.mlx_array = undefined;
        var grp_last: [2]mlx.mlx_array = undefined;
        var grp_all: [2]mlx.mlx_array = undefined;
        defer for (0..2) |i| {
            _ = mlx.mlx_array_free(grp_logits[i]);
            _ = mlx.mlx_array_free(grp_last[i]);
            _ = mlx.mlx_array_free(grp_all[i]);
        };
        const grp_ctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };
        try xfm.forwardRowAxisVerify(&rows, &grp_ctxs, &grp_logits, &grp_last, &grp_all, null, null);

        for (0..2) |i| {
            try expectRowsByteEqual(allocator, s, solo_logits[i], grp_logits[i], "logits");
            try expectRowsByteEqual(allocator, s, solo_all[i], grp_all[i], "hidden_all");
            try expectRowsByteEqual(allocator, s, solo_last[i], grp_last[i], "hidden_last");
            try testing.expectEqual(prompts[i].len + S_row[i], slots.group[i].off);
            try testing.expectEqual(slots.solo[i].off, slots.group[i].off);
        }
        std.debug.print("[mtp verify] S={any} rows_arm={}: rows equal (logits, hidden_all, hidden_last)\n", .{ S_row, pass.rows_arm });
    };
    try testing.expectEqual(group_calls, mtp.mtp_verify_moe_group_graphs - grouped_before);
    try testing.expect(verify_projection.mtp_verify_indexed_input_calls > indexed_before);
    try testing.expect(verify_projection.mtp_verify_expert_reduce_calls > reduce_before);
    try testing.expect(verify_projection.mtp_verify_expert_pairs_calls > expert_before);
    try testing.expect(verify_projection.mtp_verify_gdn_rows_calls > gdn_before);
    try testing.expect(verify_projection.mtp_verify_shared_gate_rows_calls > shared_gate_before);
    try testing.expect(verify_projection.mtp_verify_attn_rows_calls > attn_before);
    try testing.expect(verify_projection.mtp_verify_lmhead_rows_calls > lmhead_before);
    try testing.expect(mtp.mtp_verify_moe_group_last_ops > 0);
    std.debug.print("[qwen4-verify-ops] grouped calls={d} last_ops={d}\n", .{ mtp.mtp_verify_moe_group_graphs - grouped_before, mtp.mtp_verify_moe_group_last_ops });
}

fn rollbackRowToAccepted(sl: *Qwen4TestSlot, accepted: u32, verify_len: u32, off_snap: usize, step_snap: usize, s: mlx.mlx_stream) !void {
    const accepted_len: usize = 1 + @as(usize, accepted);
    try sl.cache.truncate(off_snap + accepted_len, s);
    sl.cache.step = step_snap;
    // Full acceptance keeps the dedicated final state; the capture tail is unwritten.
    if (accepted_len < verify_len) for (sl.entries) |*e| {
        try ssmRollbackFromCapture(e, accepted, verify_len, s);
    };
    sl.off = off_snap + accepted_len;
    for (sl.entries) |*e| ssmFreeSpecCapture(e);
}

test "rollback fixture preserves the separate final state at full acceptance" {
    const allocator = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]u32{ 0, 1, 2 }) |accepted| {
        const slot = try Qwen4TestSlot.init(allocator, 1);
        defer slot.deinit(allocator);
        const entry = &slot.entries[0];
        _ = mlx.mlx_array_free(entry.ssm_state);
        entry.ssm_state = mlx.mlx_array_new_data(&[_]f32{33}, &.{ 1, 1, 1, 1 }, 4, .float32);
        entry.spec_state_seq = mlx.mlx_array_new_data(&[_]f32{ 11, 22, -999 }, &.{ 3, 1, 1, 1, 1 }, 5, .float32);
        try rollbackRowToAccepted(slot, accepted, 3, 0, 0, s);
        try mlx.check(mlx.mlx_array_eval(entry.ssm_state));
        const values = mlx.mlx_array_data_float32(entry.ssm_state) orelse return error.Unreadable;
        try testing.expectEqual(@as(f32, @floatFromInt((accepted + 1) * 11)), values[0]);
    }
}

fn expectSlotStateEqual(alloc: std.mem.Allocator, s: mlx.mlx_stream, a: *Qwen4TestSlot, b: *Qwen4TestSlot, row: usize) !void {
    try testing.expectEqual(a.off, b.off);
    try testing.expectEqual(a.cache.step, b.cache.step);
    try testing.expectEqual(a.entries.len, b.entries.len);
    var buf: [96]u8 = undefined;
    for (a.entries, b.entries, 0..) |*ea, *eb, l| {
        try testing.expectEqual(ea.initialized, eb.initialized);
        try testing.expectEqual(ea.ple_prev_valid, eb.ple_prev_valid);
        try testing.expectEqualSlices(u32, &ea.ple_prev, &eb.ple_prev);
        try testing.expectEqual(a.cache.entries[l].offset, b.cache.entries[l].offset);
        inline for (.{ "conv_state", "ssm_state", "qsa_pooled", "aux_state" }) |field| {
            const xa = @field(ea, field);
            const xb = @field(eb, field);
            try testing.expectEqual(xa.ctx == null, xb.ctx == null);
            if (xa.ctx != null) {
                const tag = try std.fmt.bufPrint(&buf, "row {d} layer {d} {s}", .{ row, l, field });
                try expectRowsByteEqual(alloc, s, xa, xb, tag);
            }
        }
    }
    const ma = a.head.meta();
    const mb = b.head.meta();
    try testing.expectEqual(ma.stash_len, mb.stash_len);
    try testing.expectEqual(ma.origin, mb.origin);
    try testing.expectEqual(ma.cache_step, mb.cache_step);
    try testing.expectEqual(a.head.cache.entries[0].offset, b.head.cache.entries[0].offset);
    try testing.expectEqual(a.head.entry.initialized, b.head.entry.initialized);
    try testing.expectEqualSlices(u32, &a.head.entry.ple_prev, &b.head.entry.ple_prev);
}

test "per-row rollback: different accept counts across rows restore exactly the solo states" {
    const model_dir = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Qwen4Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.prepare();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..320], ids[64..232] };

    const S: usize = 5;
    const accepts = [2]u32{ 0, 3 };
    var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
    defer slots.deinit(allocator);

    var rows: [2]mlx.mlx_array = undefined;
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (0..2) |i| {
        const shape = [_]c_int{ 1, @intCast(S) };
        rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S].ptr), &shape, 2, .int32);
    }

    var off_snap: [2]usize = undefined;
    var step_snap: [2]usize = undefined;
    for (0..2) |i| {
        off_snap[i] = slots.solo[i].off;
        step_snap[i] = slots.solo[i].cache.step;
        try testing.expectEqual(off_snap[i], slots.group[i].off);
        try testing.expectEqual(step_snap[i], slots.group[i].cache.step);
    }

    var solo_logits: [2]mlx.mlx_array = undefined;
    var solo_last: [2]mlx.mlx_array = undefined;
    var solo_all: [2]mlx.mlx_array = undefined;
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(solo_logits[i]);
        _ = mlx.mlx_array_free(solo_last[i]);
        _ = mlx.mlx_array_free(solo_all[i]);
    };
    for (0..2) |i| {
        const one_row = [_]mlx.mlx_array{rows[i]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, solo_logits[i .. i + 1], solo_last[i .. i + 1], solo_all[i .. i + 1], null, null);
    }

    var grp_logits: [2]mlx.mlx_array = undefined;
    var grp_last: [2]mlx.mlx_array = undefined;
    var grp_all: [2]mlx.mlx_array = undefined;
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(grp_logits[i]);
        _ = mlx.mlx_array_free(grp_last[i]);
        _ = mlx.mlx_array_free(grp_all[i]);
    };
    const grp_ctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };
    try xfm.forwardRowAxisVerify(&rows, &grp_ctxs, &grp_logits, &grp_last, &grp_all, null, null);

    for (0..2) |i| {
        const a_start = [_]c_int{ 0, @intCast(accepts[i]), 0 };
        const a_stop = [_]c_int{ 1, @as(c_int, @intCast(accepts[i])) + 1, mlx.getShape(solo_all[i])[2] };
        const stride = [_]c_int{ 1, 1, 1 };
        var h_solo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_solo);
        var h_grp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_grp);
        try mlx.check(mlx.mlx_slice(&h_solo, solo_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try mlx.check(mlx.mlx_slice(&h_grp, grp_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try expectRowsByteEqual(allocator, s, h_solo, h_grp, "next-round hidden");
    }

    for (0..2) |i| {
        try rollbackRowToAccepted(slots.solo[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try rollbackRowToAccepted(slots.group[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try testing.expectEqual(off_snap[i] + 1 + accepts[i], slots.group[i].off);
        try expectSlotStateEqual(allocator, s, slots.solo[i], slots.group[i], i);
    }

    for (0..2) |i| {
        const next = [_]i32{ids[500 + i]};
        const la = try slots.solo[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(la);
        const lb = try slots.group[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(lb);
        try expectRowsByteEqual(allocator, s, la, lb, "next-token logits");
    }
    std.debug.print("[mtp verify] rollback at accepts {any}: every row's restored state and next token are byte-equal\n", .{accepts});
}

test "a tensor-parallel rank 0 never takes the grouped verify, whose MoE skips the exchange" {
    const model_dir = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Qwen4Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.prepare();

    var ids: [256]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..96], ids[64..160] };
    var slots = try L10VerifySlots.init(allocator, &xfm, config.num_hidden_layers, prompts);
    defer slots.deinit(allocator);
    var rows: [2]mlx.mlx_array = undefined;
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    const shape = [_]c_int{ 1, 1 };
    for (0..2) |i| rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[200 + i * 8 ..][0..1].ptr), &shape, 2, .int32);
    const ctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };

    if (!xfm.qwen4VerifyRowsEligible(&rows, &ctxs)) return error.SkipZigTest;
    var peer: distributed_mod.Peer = undefined;
    xfm.dist_tp = &peer;
    defer xfm.dist_tp = null;
    try testing.expect(!xfm.qwen4VerifyRowsEligible(&rows, &ctxs));
}

test "group round: a row's own argmax and budget decide it, and nothing else does" {
    const gen_mod = @import("../generate.zig");
    const model_dir = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Qwen4Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.prepare();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [2][]const i32{ ids[0..288], ids[64..200] };
    const S_row = [2]usize{ 5, 3 };

    var slots = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts);
    defer slots.deinit(allocator);
    var rows: [2]mlx.mlx_array = undefined;
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var drafts: [2][8]u32 = undefined;
    for (0..2) |i| {
        const shape = [_]c_int{ 1, @intCast(S_row[i]) };
        const row_ids = ids[400 + i * 32 ..][0..S_row[i]];
        rows[i] = mlx.mlx_array_new_data(@ptrCast(row_ids.ptr), &shape, 2, .int32);
        for (row_ids[1..], 0..) |t, k| drafts[i][k] = @intCast(t);
    }

    var so: [2]mlx.mlx_array = undefined;
    var sl: [2]mlx.mlx_array = undefined;
    var sa: [2]mlx.mlx_array = undefined;
    var go: [2]mlx.mlx_array = undefined;
    var gl: [2]mlx.mlx_array = undefined;
    var ga: [2]mlx.mlx_array = undefined;
    defer for (0..2) |i| {
        _ = mlx.mlx_array_free(so[i]);
        _ = mlx.mlx_array_free(sl[i]);
        _ = mlx.mlx_array_free(sa[i]);
        _ = mlx.mlx_array_free(go[i]);
        _ = mlx.mlx_array_free(gl[i]);
        _ = mlx.mlx_array_free(ga[i]);
    };
    for (0..2) |i| {
        const one_row = [_]mlx.mlx_array{rows[i]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, so[i .. i + 1], sl[i .. i + 1], sa[i .. i + 1], null, null);
    }
    const gctxs = [_]*ForwardCtx{ &slots.group[0].ctx, &slots.group[1].ctx };
    try xfm.forwardRowAxisVerify(&rows, &gctxs, &go, &gl, &ga, null, null);

    var solo_am: [2][]i32 = undefined;
    var grp_am: [2][]i32 = undefined;
    defer for (0..2) |i| {
        allocator.free(solo_am[i]);
        allocator.free(grp_am[i]);
    };
    for ([_]*[2]mlx.mlx_array{ &so, &go }, [_]*[2][]i32{ &solo_am, &grp_am }) |logits, out| {
        for (0..2) |i| {
            var am = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(am);
            try mlx.check(mlx.mlx_argmax_axis(&am, logits[i], 2, false, s));
            try mlx.check(mlx.mlx_array_eval(am));
            const n = mlx.mlx_array_size(am);
            const src = mlx.mlx_array_data_int32(am) orelse return error.Unreadable;
            const buf = try allocator.alloc(i32, n);
            @memcpy(buf, src[0..n]);
            out[i] = buf;
        }
    }
    for (0..2) |i| try testing.expectEqualSlices(i32, solo_am[i], grp_am[i]);

    const accept = gen_mod.Generator.mtpGreedyVerdict;
    var base: [2]gen_mod.Generator.MtpRowVerdict = undefined;
    for (0..2) |i| {
        const solo_v = accept(solo_am[i], drafts[i][0 .. S_row[i] - 1], 0, 400);
        const grp_v = accept(grp_am[i], drafts[i][0 .. S_row[i] - 1], 0, 400);
        try testing.expectEqual(solo_v.accepted, grp_v.accepted);
        try testing.expectEqual(solo_v.next_pending, grp_v.next_pending);
        base[i] = grp_v;
    }

    const clipped = accept(grp_am[0], drafts[0][0 .. S_row[0] - 1], 399, 400);
    try testing.expectEqual(@as(u32, 0), clipped.accepted);
    try testing.expectEqual(@as(u32, @intCast(grp_am[0][0])), clipped.next_pending);
    try testing.expectEqual(@as(u32, 1), gen_mod.Generator.mtpRetained(clipped.accepted));
    const row1_again = accept(grp_am[1], drafts[1][0 .. S_row[1] - 1], 0, 400);
    try testing.expectEqual(base[1].accepted, row1_again.accepted);
    try testing.expectEqual(base[1].next_pending, row1_again.next_pending);

    const retained = [2]u32{ gen_mod.Generator.mtpRetained(base[0].accepted), gen_mod.Generator.mtpRetained(base[1].accepted) };
    for (0..2) |i| try testing.expect(retained[i] >= 1 and retained[i] <= S_row[i]);
    std.debug.print("[mtp verify] group round: accepts {any}, retained {any}, row argmax ids equal solo on both rows\n", .{ .{ base[0].accepted, base[1].accepted }, retained });
}

test "head row coverage declines virtual widths before any stacked dispatch" {
    try testing.expect(mtpHeadRowsFit(&.{1}, 4));
    try testing.expect(mtpHeadRowsFit(&.{ 1, 1 }, 4));
    try testing.expect(mtpHeadRowsFit(&.{ 1, 1, 1, 1 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 1, 1, 1, 1, 1 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 1, 1, 1, 1, 1, 1, 1, 1 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 2, 3 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 2, 2 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{}, 4));
}

test "N=4 rollback: rejects, partial accepts and full accept restore solo states" {
    const model_dir = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Qwen4Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.prepare();
    const n_layers = config.num_hidden_layers;

    var ids: [640]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const prompts = [4][]const i32{ ids[0..320], ids[64..232], ids[16..128], ids[128..384] };

    const S: usize = 5;
    const accepts = [4]u32{ 0, 3, 1, 4 };
    var left = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts[0..2].*);
    defer left.deinit(allocator);
    var right = try L10VerifySlots.init(allocator, &xfm, n_layers, prompts[2..4].*);
    defer right.deinit(allocator);
    const slots = .{
        .solo = [_]*Qwen4TestSlot{ left.solo[0], left.solo[1], right.solo[0], right.solo[1] },
        .group = [_]*Qwen4TestSlot{ left.group[0], left.group[1], right.group[0], right.group[1] },
    };

    var rows: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (0..4) |i| {
        const shape = [_]c_int{ 1, @intCast(S) };
        rows[i] = mlx.mlx_array_new_data(@ptrCast(ids[400 + i * 32 ..][0..S].ptr), &shape, 2, .int32);
    }

    var off_snap: [4]usize = undefined;
    var step_snap: [4]usize = undefined;
    for (0..4) |i| {
        off_snap[i] = slots.solo[i].off;
        step_snap[i] = slots.solo[i].cache.step;
        try testing.expectEqual(off_snap[i], slots.group[i].off);
        try testing.expectEqual(step_snap[i], slots.group[i].cache.step);
    }

    var solo_logits: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var solo_last: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var solo_all: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (0..4) |i| {
        _ = mlx.mlx_array_free(solo_logits[i]);
        _ = mlx.mlx_array_free(solo_last[i]);
        _ = mlx.mlx_array_free(solo_all[i]);
    };
    for (0..4) |i| {
        const one_row = [_]mlx.mlx_array{rows[i]};
        const one_ctx = [_]*ForwardCtx{&slots.solo[i].ctx};
        try xfm.forwardRowAxisVerify(&one_row, &one_ctx, solo_logits[i .. i + 1], solo_last[i .. i + 1], solo_all[i .. i + 1], null, null);
    }

    var grp_logits: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var grp_last: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    var grp_all: [4]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (0..4) |i| {
        _ = mlx.mlx_array_free(grp_logits[i]);
        _ = mlx.mlx_array_free(grp_last[i]);
        _ = mlx.mlx_array_free(grp_all[i]);
    };
    var grp_ctxs: [4]*ForwardCtx = undefined;
    for (0..4) |i| grp_ctxs[i] = &slots.group[i].ctx;
    try xfm.forwardRowAxisVerify(&rows, &grp_ctxs, &grp_logits, &grp_last, &grp_all, null, null);

    for (0..4) |i| {
        const a_start = [_]c_int{ 0, @intCast(accepts[i]), 0 };
        const a_stop = [_]c_int{ 1, @as(c_int, @intCast(accepts[i])) + 1, mlx.getShape(solo_all[i])[2] };
        const stride = [_]c_int{ 1, 1, 1 };
        var h_solo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_solo);
        var h_grp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_grp);
        try mlx.check(mlx.mlx_slice(&h_solo, solo_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try mlx.check(mlx.mlx_slice(&h_grp, grp_all[i], &a_start, 3, &a_stop, 3, &stride, 3, s));
        try expectRowsByteEqual(allocator, s, h_solo, h_grp, "next-round hidden");
    }

    for (0..4) |i| {
        try rollbackRowToAccepted(slots.solo[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try rollbackRowToAccepted(slots.group[i], accepts[i], @intCast(S), off_snap[i], step_snap[i], s);
        try testing.expectEqual(off_snap[i] + 1 + accepts[i], slots.group[i].off);
        try expectSlotStateEqual(allocator, s, slots.solo[i], slots.group[i], i);
    }

    for (0..4) |i| {
        const next = [_]i32{ids[500 + i]};
        const la = try slots.solo[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(la);
        const lb = try slots.group[i].forward(&xfm, &next);
        defer _ = mlx.mlx_array_free(lb);
        try expectRowsByteEqual(allocator, s, la, lb, "next-token logits");
    }
    std.debug.print("[mtp verify] rollback at accepts {any}: every row's restored state and next token are byte-equal\n", .{accepts});
}

test "logical capture includes every defined state and the separate final state" {
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    const captures = [_]mlx.mlx_array{
        mlx.mlx_array_new_data(&[_]f32{ 11, 22, 999 }, &[_]c_int{ 3, 1, 1, 1, 1 }, 5, .float32),
        mlx.mlx_array_new_data(&[_]f32{ 11, 22, -999 }, &[_]c_int{ 3, 1, 1, 1, 1 }, 5, .float32),
        mlx.mlx_array_new_data(&[_]f32{ 11, 23, 999 }, &[_]c_int{ 3, 1, 1, 1, 1 }, 5, .float32),
    };
    defer for (captures) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    const final = mlx.mlx_array_new_data(&[_]f32{33}, &[_]c_int{ 1, 1, 1, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(final);
    var entries: [3]SSMCacheEntry = undefined;
    for (&entries, captures) |*entry, capture| entry.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = final, .initialized = true, .spec_state_seq = capture };
    const a = try logicalSsmCapture(stream, &entries[0]);
    defer _ = mlx.mlx_array_free(a);
    const b = try logicalSsmCapture(stream, &entries[1]);
    defer _ = mlx.mlx_array_free(b);
    const c = try logicalSsmCapture(stream, &entries[2]);
    defer _ = mlx.mlx_array_free(c);
    try testing.expect(try qsaArraysAllEqual(a, b, stream));
    try testing.expect(!try qsaArraysAllEqual(a, c, stream));
    const changed_final = mlx.mlx_array_new_data(&[_]f32{34}, &[_]c_int{ 1, 1, 1, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(changed_final);
    entries[1].ssm_state = changed_final;
    const d = try logicalSsmCapture(stream, &entries[1]);
    defer _ = mlx.mlx_array_free(d);
    try testing.expect(!try qsaArraysAllEqual(a, d, stream));
}

test "verify rollback permits in-place KV append and restores its valid prefix" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const stream = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(allocator, 1, KVQuantConfig.affine(8));
    defer cache.deinit();
    var prng = std.Random.DefaultPrng.init(618);
    const prefix = try attn256RandBf16(prng.random(), &.{ 1, 1, 8, 64 }, stream);
    defer _ = mlx.mlx_array_free(prefix);
    const tail = try attn256RandBf16(prng.random(), &.{ 1, 1, 4, 64 }, stream);
    defer _ = mlx.mlx_array_free(tail);
    var initial_view = try cache.update(0, prefix, prefix, stream, 0);
    initial_view.deinit();
    const Eval = struct {
        fn buffers(entry: *const KVCacheEntry) !void {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| _ = mlx.mlx_vector_array_append_value(vec, @field(entry, field));
            try mlx.check(mlx.mlx_eval(vec));
        }
    };
    try Eval.buffers(&cache.entries[0]);
    const before = mlx.mlx_array_data_uint32(cache.entries[0].keys) orelse return error.KvUnreadable;
    const address = @intFromPtr(before);
    var expected: [128]u32 = undefined;
    @memcpy(&expected, before[0..expected.len]);
    var offset: usize = 8;
    var ctx = ForwardCtx{ .cache = &cache, .moe_seq_offset = &offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var fwd_gen: u64 = 0;
    var rollback = try VerifyRollback.capture(allocator, &fwd_gen, &ctx);
    defer rollback.deinit();
    var appended = try cache.update(0, tail, tail, stream, 0);
    appended.deinit();
    offset = 12;
    try Eval.buffers(&cache.entries[0]);
    const after = mlx.mlx_array_data_uint32(cache.entries[0].keys) orelse return error.KvUnreadable;
    try testing.expectEqual(address, @intFromPtr(after));
    rollback.restore();
    try testing.expectEqual(@as(usize, 8), offset);
    try testing.expectEqual(@as(usize, 8), cache.step);
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);
    const restored = mlx.mlx_array_data_uint32(cache.entries[0].keys) orelse return error.KvUnreadable;
    try testing.expectEqualSlices(u32, &expected, restored[0..expected.len]);
    var resumed = try cache.update(0, tail, tail, stream, 0);
    defer resumed.deinit();
    try Eval.buffers(&cache.entries[0]);
    try testing.expectEqual(@as(usize, 12), cache.entries[0].offset);
    const again = mlx.mlx_array_data_uint32(cache.entries[0].keys) orelse return error.KvUnreadable;
    try testing.expectEqualSlices(u32, &expected, again[0..expected.len]);
}

fn verifyRollbackEvalStorage(entry: *const KVCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const value = @field(entry, field);
        if (value.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, value);
    }
    try mlx.check(mlx.mlx_eval(vec));
}

fn verifyRollbackStorageAddress(value: mlx.mlx_array) !usize {
    return switch (mlx.mlx_array_dtype(value)) {
        .bfloat16 => @intFromPtr(mlx.mlx_array_data_bfloat16(value) orelse return error.KvUnreadable),
        .uint32 => @intFromPtr(mlx.mlx_array_data_uint32(value) orelse return error.KvUnreadable),
        else => error.KvUnreadable,
    };
}

fn verifyRollbackValidPrefixEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.initialized, b.initialized);
    try testing.expectEqual(a.offset, b.offset);
    if (!a.initialized) return;
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const av = @field(a, field);
        const bv = @field(b, field);
        try testing.expectEqual(av.ctx == null, bv.ctx == null);
        if (av.ctx != null) {
            const ash = mlx.getShape(av);
            const bsh = mlx.getShape(bv);
            var aslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(aslice);
            var bslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bslice);
            try mlx.check(mlx.mlx_slice(&aslice, av, &.{ 0, 0, 0, 0 }, 4, &.{ ash[0], ash[1], @intCast(a.offset), ash[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_slice(&bslice, bv, &.{ 0, 0, 0, 0 }, 4, &.{ bsh[0], bsh[1], @intCast(b.offset), bsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try testing.expect(try qsaArraysAllEqual(aslice, bslice, s));
        }
    }
}

fn verifyRollbackKvCase(config: KVQuantConfig, prefix_len: usize, force_growth: bool) !void {
    const allocator = testing.allocator;
    const stream = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(allocator, 1, config);
    defer cache.deinit();
    var reference = try KVCache.initWithConfig(allocator, 1, config);
    defer reference.deinit();
    var prng = std.Random.DefaultPrng.init(1911 + prefix_len + @intFromBool(force_growth));
    if (prefix_len != 0) {
        const prefix = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(prefix_len), 64 }, stream);
        defer _ = mlx.mlx_array_free(prefix);
        var cv = try cache.update(0, prefix, prefix, stream, 0);
        cv.deinit();
        var rv = try reference.update(0, prefix, prefix, stream, 0);
        rv.deinit();
        try verifyRollbackEvalStorage(&cache.entries[0]);
        try verifyRollbackEvalStorage(&reference.entries[0]);
    }
    const tail_len = if (force_growth and prefix_len != 0)
        KVCache.bufferCapacity(cache.entries[0].keys) - cache.entries[0].offset + 1
    else
        4;
    const tail = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(tail_len), 64 }, stream);
    defer _ = mlx.mlx_array_free(tail);
    var addresses: [6]usize = @splat(0);
    if (prefix_len != 0 and !force_growth) {
        inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }, 0..) |field, i| {
            const value = @field(cache.entries[0], field);
            if (value.ctx != null) addresses[i] = try verifyRollbackStorageAddress(value);
        }
    }
    var offset = prefix_len;
    var ctx = ForwardCtx{ .cache = &cache, .moe_seq_offset = &offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var fwd_gen: u64 = 0;
    var rollback = try VerifyRollback.capture(allocator, &fwd_gen, &ctx);
    defer rollback.deinit();
    var appended = try cache.update(0, tail, tail, stream, 0);
    appended.deinit();
    offset += tail_len;
    try verifyRollbackEvalStorage(&cache.entries[0]);
    if (prefix_len != 0 and !force_growth) {
        inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }, 0..) |field, i| {
            const value = @field(cache.entries[0], field);
            if (value.ctx != null) try testing.expectEqual(addresses[i], try verifyRollbackStorageAddress(value));
        }
    }
    rollback.restore();
    try testing.expectEqual(prefix_len, offset);
    try testing.expectEqual(reference.step, cache.step);
    try verifyRollbackValidPrefixEqual(&cache.entries[0], &reference.entries[0], stream);
    var resumed = try cache.update(0, tail, tail, stream, 0);
    resumed.deinit();
    var expected = try reference.update(0, tail, tail, stream, 0);
    expected.deinit();
    try verifyRollbackEvalStorage(&cache.entries[0]);
    try verifyRollbackEvalStorage(&reference.entries[0]);
    try testing.expectEqual(reference.step, cache.step);
    try verifyRollbackValidPrefixEqual(&cache.entries[0], &reference.entries[0], stream);
}

test "offset rollback preserves dense affine growth and cold initialization" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    inline for (.{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |config| {
        try verifyRollbackKvCase(config, 8, false);
        try verifyRollbackKvCase(config, 8, true);
        try verifyRollbackKvCase(config, 0, false);
    }
}
