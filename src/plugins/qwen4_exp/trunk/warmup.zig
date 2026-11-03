//! The MTP verify warm-up: every draft width solo, every grouped-verify row total and the head's own step JIT their
//! Metal pipelines at load, not inside the first measured round.
//! copied_from: engines/mlx/warmup.zig (`warmupSpecVerify` and its slots).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const KVQuantConfig = @import("../../../engines/mlx/cache/kv_quant.zig").KVQuantConfig;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const testing = @import("std").testing;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const nextSsmMemberGen = @import("../../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmFreeQsaState = @import("../qsa/history.zig").ssmFreeQsaState;
const Qwen4MtpProject = @import("../mtp.zig").Qwen4MtpProject;
const Trunk = @import("../trunk.zig").Trunk;

const MAX_WARM_SEQ: usize = MAX_WIDTH_WARM + 1;
const MAX_WIDTH_WARM: usize = 8;

const Slot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,

    fn init(t: *Trunk, kv_config: KVQuantConfig) !*Slot {
        const alloc = t.allocator;
        const n_layers = t.config.num_hidden_layers;
        const sl = try alloc.create(Slot);
        errdefer alloc.destroy(sl);
        var cache = try KVCache.initWithConfig(alloc, n_layers, kv_config);
        errdefer cache.deinit();
        sl.* = .{ .cache = cache, .entries = try alloc.alloc(SSMCacheEntry, n_layers) };
        for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        sl.ctx = .{
            .cache = &sl.cache,
            .moe_seq_offset = &sl.off,
            .ssm_entries = sl.entries,
            .ssm_member_gen = nextSsmMemberGen(),
            .capture_hidden = null,
            .vision_embeddings = null,
        };
        return sl;
    }

    fn deinit(sl: *Slot, alloc: std.mem.Allocator) void {
        for (sl.entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        alloc.free(sl.entries);
        sl.cache.deinit();
        alloc.destroy(sl);
    }
};

fn verifyAt(t: *Trunk, slots: []const *Slot, seq: usize) !void {
    var ids: [MAX_BATCH_ROWS][MAX_WARM_SEQ]i32 = undefined;
    var rows: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var ctxs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    defer for (rows[0..slots.len]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (slots, 0..) |sl, i| {
        rows[i] = specWarmIds(ids[i][0..seq]);
        ctxs[i] = &sl.ctx;
    }
    var logits: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var last: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var all: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    try t.forwardRowAxisVerify(rows[0..slots.len], ctxs[0..slots.len], logits[0..slots.len], last[0..slots.len], all[0..slots.len], null, null);
    specWarmEvalFree(logits[0..slots.len]);
    specWarmEvalFree(last[0..slots.len]);
    specWarmEvalFree(all[0..slots.len]);
}

pub fn warmupSpecVerify(self: *Trunk, max_width: u32, kv_config: KVQuantConfig) !void {
    if (self.qwen4_mtp == null) return;
    var lap = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    const width_cap: usize = @min(@max(max_width, 1), MAX_WIDTH_WARM);
    const group_cap: usize = if (verifyQmmNaxAvailable()) 16 else 7;
    const max_rows: usize = @min(specWarmGroupRows(group_cap), MAX_BATCH_ROWS);

    var slots: [MAX_BATCH_ROWS]*Slot = undefined;
    var n_slots: usize = 0;
    defer {
        for (slots[0..n_slots]) |sl| {
            self.ssmGroupRelease(&sl.ctx) catch {};
            sl.deinit(self.allocator);
        }
        _ = mlx.mlx_clear_cache();
    }
    var head_in = mlx.mlx_array_new();
    defer if (head_in.ctx != null) {
        _ = mlx.mlx_array_free(head_in);
    };
    while (n_slots < max_rows) : (n_slots += 1) {
        slots[n_slots] = try Slot.init(self, kv_config);
        var prompt_ids: [8]i32 = undefined;
        const prompt = specWarmIds(&prompt_ids);
        defer _ = mlx.mlx_array_free(prompt);
        var stream = mlx.mlx_array_new();
        slots[n_slots].ctx.capture_stream_all = if (n_slots == 0) &stream else null;
        const logits = try self.forwardWith(&slots[n_slots].ctx, prompt);
        slots[n_slots].ctx.capture_stream_all = null;
        if (n_slots == 0 and stream.ctx != null) {
            const sh = mlx.getShape(stream);
            try mlx.check(mlx.mlx_slice(&head_in, stream, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        }
        specWarmEvalFree(&[_]mlx.mlx_array{ logits, stream });
    }

    var w: usize = 1;
    while (w <= width_cap) : (w += 1) try verifyAt(self, slots[0..1], w + 1);

    var n: usize = 2;
    var group_seq_max: usize = 0;
    while (n <= n_slots) : (n += 1) {
        const per_row_cap = specWarmGroupSeq(group_cap, n, width_cap) - 1;
        group_seq_max = @max(group_seq_max, per_row_cap + 1);
        var seq: usize = 2;
        while (seq <= per_row_cap + 1) : (seq += 1) try verifyAt(self, slots[0..n], seq);
    }
    if (head_in.ctx != null) {
        var st = try self.qwen4MtpStateNew();
        defer st.deinit();
        var id_buf: [1]i32 = undefined;
        const id_arr = specWarmIds(&id_buf);
        defer _ = mlx.mlx_array_free(id_arr);
        for ([_]Qwen4MtpProject{ .mixed_last_row, .none }, 0..) |project, i| {
            const out = try self.qwen4MtpForwardOn(&st, head_in, id_arr, @intCast(1 + i), null, project);
            specWarmEvalFree(&[_]mlx.mlx_array{ out.logits, out.stream, out.mixed });
        }
    }
    log.info("[spec-warmup] verify widths 1..{d} solo, group rows 2..{d} at up to {d} tokens each, head width 1 ({d} ms).\n", .{ width_cap, n_slots, group_seq_max, lap.read() / std.time.ns_per_ms });
}

fn specWarmIds(buf: []i32) mlx.mlx_array {
    for (buf, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % 40000);
    const shape = [_]c_int{ 1, @intCast(buf.len) };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 2, .int32);
}

fn specWarmEvalFree(arrays: []const mlx.mlx_array) void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, a);
    }
    _ = mlx.mlx_eval(vec);
    for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    }
}

// Mirror of `scheduler.mtpSubGroupSize`: the slots one row-axis tick can carry at this row
// budget (NAX wide lane 16, stock lane 7) and the drafts each slot gets.
fn specWarmGroupRows(group_cap: usize) usize {
    return if (group_cap >= 16) 4 else 3;
}

fn specWarmGroupSeq(group_cap: usize, rows: usize, width_cap: usize) usize {
    return @min(group_cap / rows - 1, width_cap) + 1;
}

test "spec warm-up covers exactly the group shapes the scheduler can build" {
    try testing.expectEqual(@as(usize, 4), specWarmGroupRows(16));
    try testing.expectEqual(@as(usize, 3), specWarmGroupRows(7));
    try testing.expectEqual(@as(usize, 7), specWarmGroupSeq(16, 2, 6));
    try testing.expectEqual(@as(usize, 5), specWarmGroupSeq(16, 3, 6));
    try testing.expectEqual(@as(usize, 4), specWarmGroupSeq(16, 4, 6));
    try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(7, 2, 6));
    try testing.expectEqual(@as(usize, 2), specWarmGroupSeq(7, 3, 6));
    try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(16, 2, 2));
}
