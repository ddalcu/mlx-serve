//! Which QSA attention arm served each full-attention call (`QsaArm`, `QsaArmCounts`, decline reasons): the
//! per-request `[qsa-arms]` meter.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const Trunk = @import("../trunk.zig").Trunk;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const OneShotBits = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").OneShotBits;
const qsaWidthBucket = @import("../../../sdk/blocks/attention/sparse_block/one_shot.zig").widthBucket;

// ── QSA arm meters ──
//
// A single process-global `engaged` bool is not a meter: it fires on the
// first request of a session and then says nothing about the width or the
// context length that is actually slow, and a DECLINE says nothing at all —
// so a gather that silently hands every 62k decode step back to the dense
// mask (a full kv8 dequant per layer per step) reads exactly like a healthy
// run. These three pieces make the arm that ran observable: one-shot bits
// per (arm, width bucket) for engagement, one-shot bits per reason for
// declines, and a per-request tally.

/// Which arm served one qwen4_exp full-attention call.
pub const QsaArm = enum(u2) { mask = 0, decode_gather = 1, verify_gather = 2, prefill_gather = 3 };

/// Why a gather arm handed the call back to the dense mask.
pub const QsaDeclineReason = enum(u5) {
    disabled,
    width,
    geometry,
    dtype,
    kv_floor,
    no_win,
    quant_layout,
    gather_shape,

    pub fn text(self: QsaDeclineReason) []const u8 {
        return switch (self) {
            .disabled => "kill switch",
            .width => "query width not this arm's",
            .geometry => "geometry",
            .dtype => "dtype",
            .kv_floor => "kv below the arm's floor",
            .no_win => "no sparsity win (gathered rows >= kv)",
            .quant_layout => "quant triple layout",
            .gather_shape => "gathered slab shape",
        };
    }
};

/// Decline reporter threaded through one gather call: `no(reason)` logs at
/// most once per (arm, reason) for the process and hands the caller the
/// decline. `s`/`kv` are filled in as soon as the guards learn them (-1 =
/// not yet known at that guard).
pub const QsaDecline = struct {
    bits: *OneShotBits,
    arm: []const u8,
    s: c_int = -1,
    kv: c_int = -1,

    pub fn no(self: *QsaDecline, r: QsaDeclineReason) ?mlx.mlx_array {
        if (self.bits.take(@backingInt(r))) {
            log.info(
                "[{s}] declined: {s} (S={d} kv={d}) — the dense [S, kv] mask arm serves this call\n",
                .{ self.arm, r.text(), self.s, self.kv },
            );
        }
        return null;
    }
};

/// Per-request tally of which arm served each full-attention call. One
/// increment per attention layer; read and reset where the request's
/// `[qsa-arms]` line is emitted. It lives on the REQUEST's `ForwardCtx`
/// (the Generator's own), never on the shared Trunk: qwen4 text slots
/// batch-decode on one Trunk and a cold prefill interleaves with other
/// slots' decode ticks, so a Trunk-level tally prints the union of
/// every concurrent slot and leaves the next finisher an empty one.
pub const QsaArmCounts = struct {
    mask: u32 = 0,
    decode: u32 = 0,
    verify: u32 = 0,
    prefill: u32 = 0,

    pub fn note(self: *QsaArmCounts, arm: QsaArm) void {
        switch (arm) {
            .mask => self.mask +|= 1,
            .decode_gather => self.decode +|= 1,
            .verify_gather => self.verify +|= 1,
            .prefill_gather => self.prefill +|= 1,
        }
    }
    /// A gather served the call: which gather is the query WIDTH's business.
    pub fn noteGather(self: *QsaArmCounts, seq_len: c_int) void {
        self.note(switch (qsaWidthBucket(seq_len)) {
            0 => .decode_gather,
            1 => .verify_gather,
            else => .prefill_gather,
        });
    }
    pub fn add(self: *QsaArmCounts, o: QsaArmCounts) void {
        self.mask +|= o.mask;
        self.decode +|= o.decode;
        self.verify +|= o.verify;
        self.prefill +|= o.prefill;
    }
    pub fn total(self: QsaArmCounts) u32 {
        return self.mask +| self.decode +| self.verify +| self.prefill;
    }
    /// Read and clear this request's tally. Null when nothing was counted
    /// (not a QSA arch, or below the sparse budget) so a caller never prints
    /// a meaningless `qsa=mask 0/0`.
    pub fn take(self: *QsaArmCounts) ?QsaArmCounts {
        const c = self.*;
        self.reset();
        if (c.total() == 0) return null;
        return c;
    }
    /// The arm that served the most calls; null when nothing was counted.
    /// Ties break toward the CHEAPEST arm that could have run so a tie never
    /// reads as "the mask arm is fine".
    pub fn majority(self: QsaArmCounts) ?QsaArm {
        if (self.total() == 0) return null;
        var best: QsaArm = .mask;
        var best_n: u32 = self.mask;
        if (self.decode >= best_n) {
            best = .decode_gather;
            best_n = self.decode;
        }
        if (self.verify >= best_n) {
            best = .verify_gather;
            best_n = self.verify;
        }
        if (self.prefill >= best_n) {
            best = .prefill_gather;
            best_n = self.prefill;
        }
        return best;
    }
    pub fn reset(self: *QsaArmCounts) void {
        self.* = .{};
    }
};

/// One batched call serves every slot in the group with the SAME arm (the
/// stacked-mask SDPA), so it counts once on each participating slot's own
/// tally — a single shared counter would report the group as one request.
pub fn noteQsaArmForSlots(slots: []const *ForwardCtx, arm: QsaArm) void {
    for (slots) |slot_ctx| slot_ctx.qsa_arms.note(arm);
}

/// Engagement meters, one-shot per (arm, width bucket): bit = arm*4 + bucket.
pub var qsa_engaged_bits: OneShotBits = .{};
pub var qsa_decode_decline_bits: OneShotBits = .{};
pub var qsa_dec_engaged_bits: OneShotBits = .{};
pub var qsa_verify_decline_bits: OneShotBits = .{};

pub fn qsaEngagedBit(arm: QsaArm, seq_len: c_int) u5 {
    return @as(u5, @backingInt(arm)) * 4 + @as(u5, qsaWidthBucket(seq_len));
}

// ── tests ──

test "qsa arm meters: one-shot bits, width buckets, and the per-request tally" {
    // The meters exist because a process-global `engaged` bool cannot answer
    // "which arm served THIS request at THIS context length". Each property
    // it needs is pinned here: a bit fires once, decode/verify/prefill get
    // SEPARATE engagement bits (so a 16k decode engagement cannot vouch for a
    // 62k verify call), and an empty tally reports nothing rather than a
    // misleading `qsa=mask`.
    var bits: OneShotBits = .{};
    try std.testing.expect(bits.take(3));
    try std.testing.expect(!bits.take(3));
    try std.testing.expect(bits.take(4));
    try std.testing.expect(!bits.take(4));
    bits.reset();
    try std.testing.expect(bits.take(3));

    try std.testing.expectEqual(@as(u2, 0), qsaWidthBucket(1));
    try std.testing.expectEqual(@as(u2, 1), qsaWidthBucket(2));
    try std.testing.expectEqual(@as(u2, 1), qsaWidthBucket(FUSED256_MIN_Q_LEN - 1));
    try std.testing.expectEqual(@as(u2, 2), qsaWidthBucket(FUSED256_MIN_Q_LEN));
    try std.testing.expectEqual(@as(u2, 2), qsaWidthBucket(4096));

    // Every (arm, bucket) pair owns a distinct bit — a collision would let one
    // arm's engagement silence another's.
    var seen = std.AutoHashMap(u5, void).init(std.testing.allocator);
    defer seen.deinit();
    for ([_]QsaArm{ .mask, .decode_gather, .verify_gather, .prefill_gather }) |arm| {
        for ([_]c_int{ 1, 5, 64 }) |s| {
            const bit = qsaEngagedBit(arm, s);
            try std.testing.expect(!seen.contains(bit));
            try seen.put(bit, {});
        }
    }

    var counts: QsaArmCounts = .{};
    try std.testing.expect(counts.majority() == null);
    try std.testing.expectEqual(@as(u32, 0), counts.total());
    counts.note(.mask);
    counts.note(.mask);
    counts.note(.decode_gather);
    try std.testing.expectEqual(@as(u32, 3), counts.total());
    try std.testing.expectEqual(QsaArm.mask, counts.majority().?);
    counts.note(.decode_gather);
    // Tie breaks toward the gather: a tie must never read as "the mask is fine".
    try std.testing.expectEqual(QsaArm.decode_gather, counts.majority().?);
    counts.reset();
    try std.testing.expectEqual(@as(u32, 0), counts.total());

    // Every reason carries distinct, non-empty text (the log's whole value).
    var texts = std.StringHashMap(void).init(std.testing.allocator);
    defer texts.deinit();
    inline for (@typeInfo(QsaDeclineReason).@"enum".field_values) |v| {
        const t = (@as(QsaDeclineReason, @fromBackingInt(@intCast(v)))).text();
        try std.testing.expect(t.len > 0);
        try std.testing.expect(!texts.contains(t));
        try texts.put(t, {});
        // Reasons index OneShotBits directly — they must fit its 32 bits.
        try std.testing.expect(v < 32);
    }
}

test "qsa arm tally is owned by the request: two ForwardCtx owners never mix" {
    // The `[qsa-arms]` line is a PER-REQUEST meter, so its tally belongs to
    // the seam that logs it (the Generator's own ForwardCtx), never to the
    // shared Trunk. qwen4 text slots batch-decode on ONE Trunk
    // (`forwardMoeBatchedDecode`) and a cold prefill interleaves with other
    // slots' decode ticks (`scheduler.interleaveDecodeTick`), so a
    // Trunk-level counter printed the union of every concurrent slot
    // since the last reset and left the next finisher a near-empty tally — a
    // short chat claimed `qsa=verify_gather` while the 40k chat later logged
    // `qsa=mask`. Two owners here stand in for two concurrent slots.
    //
    // `cache` / `moe_seq_offset` are never read on this path — the tally is
    // pure bookkeeping on the ctx — so the ctxs are built as shells.
    var cache_a: KVCache = undefined;
    var cache_b: KVCache = undefined;
    var off_a: usize = 0;
    var off_b: usize = 0;
    var a: ForwardCtx = .{ .cache = &cache_a, .moe_seq_offset = &off_a, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var b: ForwardCtx = .{ .cache = &cache_b, .moe_seq_offset = &off_b, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };

    // A serves a decode-width gather and a prefill-width gather; B serves the
    // dense mask arm in between (the interleave the shared counter merged).
    a.qsa_arms.noteGather(1);
    b.qsa_arms.note(.mask);
    a.qsa_arms.noteGather(FUSED256_MIN_Q_LEN);
    b.qsa_arms.note(.mask);
    a.qsa_arms.noteGather(4);

    try std.testing.expectEqual(@as(u32, 3), a.qsa_arms.total());
    try std.testing.expectEqual(@as(u32, 0), a.qsa_arms.mask);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.decode);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.verify);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.prefill);
    try std.testing.expectEqual(@as(u32, 2), b.qsa_arms.total());
    try std.testing.expectEqual(@as(u32, 2), b.qsa_arms.mask);

    // A finishing request takes its OWN tally; B's survives untouched (the
    // shared field reset whichever slot finished first).
    const taken = a.qsa_arms.take().?;
    try std.testing.expectEqual(@as(u32, 3), taken.total());
    try std.testing.expect(a.qsa_arms.take() == null);
    try std.testing.expectEqual(@as(u32, 2), b.qsa_arms.total());
    try std.testing.expectEqual(QsaArm.mask, b.qsa_arms.take().?.majority().?);

    // One batched call serves the whole group with the SAME arm: it counts
    // once on every participating slot, not once on a shared counter.
    const slots = [_]*ForwardCtx{ &a, &b };
    noteQsaArmForSlots(&slots, .mask);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.mask);
    try std.testing.expectEqual(@as(u32, 1), b.qsa_arms.mask);
}

test "the qsa arm tally lives on the ForwardCtx, never the shared Trunk" {
    try std.testing.expect(@hasField(ForwardCtx, "qsa_arms"));
    try std.testing.expect(!@hasField(Trunk, "qsa_arms"));
    try std.testing.expect(!@hasDecl(Trunk, "qsaArmsTake"));
}
