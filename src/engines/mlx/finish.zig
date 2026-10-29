//! Finishing a slot: its terminator, outcome metrics and the short-generation log.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const metrics_mod = @import("../../host/metrics/metrics.zig");
const log = @import("../../sdk/util/log.zig");
const mlx_stream = @import("modules/stream.zig");
const testing = std.testing;

const publishHotCacheResidency = @import("cache/budget.zig").publishHotCacheResidency;
const commitSlotIfApplicable = @import("commit.zig").commitSlotIfApplicable;
const Scheduler = @import("scheduler.zig").Scheduler;
const Slot = @import("slot.zig").Slot;
const specTickMode = @import("spec/wiring.zig").specTickMode;
const plugin_settle = @import("plugin/settle.zig");

/// Both contexts name the same plugin slot. Clear them together before either can be destroyed.
pub fn releasePluginSlot(slot: *Slot) void {
    const xfm = slot.model.transformer orelse return;
    if (!xfm.ownsModuleDecodeState()) return;
    if (slot.legacy_gen) |*g| {
        xfm.discardDeferredPle(&g.ctx);
        const id = g.ctx.plugin_slot;
        plugin_settle.release(xfm, &g.ctx);
        if (slot.ctx.plugin_slot == id) slot.ctx.plugin_slot = 0;
    }
    xfm.discardDeferredPle(&slot.ctx);
    plugin_settle.release(xfm, &slot.ctx);
}

/// The terminator a finishing slot publishes, and the only caller of `Slot.markFinished`.
/// `latched` is a peek (`mlx.peekErrorName`): a decode forward that failed can still hand
/// back a plausible EOS, and the request must not finish 200 on it. The tick wrapper still
/// owns the latch and fails the rest of a batched group with it.
pub fn publishSlotTerminator(slot: anytype, reason: []const u8, latched: ?[]const u8) void {
    if (latched) |name| {
        slot.markError(name);
        return;
    }
    slot.markFinished(reason);
}

/// Decoded bytes of the answer that ride the `[short-gen]` line.
const SHORT_GEN_TEXT_CAP = 200;

/// `emitted` (what the client got) and `realized` (what the generator appended) diverge only on a block path.
fn formatShortGen(
    out: []u8,
    reason: []const u8,
    emitted: u32,
    ids: []const u32,
    text: []const u8,
    path: []const u8,
) []const u8 {
    var n: usize = 0;
    n += (std.fmt.bufPrint(out[n..], "[short-gen] reason={s} emitted={d} realized={d} ids=[", .{ reason, emitted, ids.len }) catch return out[0..n]).len;
    for (ids, 0..) |id, i| {
        const sep: []const u8 = if (i == 0) "" else ",";
        n += (std.fmt.bufPrint(out[n..], "{s}{d}", .{ sep, id }) catch break).len;
    }
    n += (std.fmt.bufPrint(out[n..], "] bytes=\"", .{}) catch return out[0..n]).len;
    var cap = @min(text.len, SHORT_GEN_TEXT_CAP);
    // A token is a BPE fragment: never cut inside a multi-byte sequence.
    while (cap > 0 and cap < text.len and (text[cap] & 0xC0) == 0x80) cap -= 1;
    for (text[0..cap]) |c| {
        var one = [_]u8{if (c < 0x20) ' ' else c};
        const esc: []const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => one[0..1],
        };
        if (n + esc.len > out.len) break;
        @memcpy(out[n..][0..esc.len], esc);
        n += esc.len;
    }
    n += (std.fmt.bufPrint(out[n..], "\" path={s}", .{path}) catch return out[0..n]).len;
    return out[0..n];
}

/// A one-word answer and a block-path handling bug look identical until the ids are named.
fn logShortGen(slot: *Slot, reason: []const u8) void {
    const gen = if (slot.legacy_gen) |*g| g else return;
    const mode = specTickMode(
        slot.enable_mtp,
        gen.mtp != null,
        slot.enable_drafter,
        gen.drafter != null,
        gen.dflash != null,
        slot.enable_pld,
        gen.pld_enabled,
        gen.dspark_enabled,
    );
    const ids = gen.generated_ids.items;
    const decoded: ?[]u8 = if (slot.model.tokenizer) |tok|
        (tok.decode(slot.allocator, ids, false) catch null)
    else
        null;
    defer if (decoded) |d| slot.allocator.free(d);
    var buf: [1024]u8 = undefined;
    log.info("{s}\n", .{formatShortGen(
        &buf,
        reason,
        slot.completion_tokens,
        ids,
        decoded orelse "",
        if (mode == .regular) "serial" else @tagName(mode),
    )});
}

/// Outcome of a slot that reached `finishSlot`. A latched MLX failure turns the finish into an error.
pub fn finishOutcome(reason: []const u8, latched: ?[]const u8) metrics_mod.Outcome {
    if (latched != null) return .failed;
    if (std.mem.eql(u8, reason, "cancelled")) return .cancelled;
    return .success;
}

/// Outcome of a slot seen by the cleanup drain without `finishSlot` having recorded it, read only
/// from state the inference thread set. `cancelled` is never consulted: `Scheduler.complete` sets it
/// on every completion, normal ones included.
fn cleanupOutcome(slot: anytype) metrics_mod.Outcome {
    if (slot.error_code != null) return .failed;
    if (slot.finished) return finishOutcome(slot.finish_reason, null);
    return .cancelled;
}

/// Count a slot's outcome once. The first path to reach a slot wins.
pub fn recordSlotEnd(metrics: ?*metrics_mod.Metrics, slot: anytype, outcome: metrics_mod.Outcome) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    slot.metrics_recorded = true;
    m.recordRequest(
        outcome,
        slot.first_token_ns,
        slot.prefill_ns,
        slot.decode_ns,
        slot.prompt_tokens,
        slot.completion_tokens,
        slot.cached_tokens,
    );
}

/// The cleanup drain's count: a slot refused before its first forward is a rejection, anything
/// `finishSlot` already counted is skipped.
pub fn recordSlotCleanup(metrics: ?*metrics_mod.Metrics, slot: anytype) void {
    const m = metrics orelse return;
    if (slot.metrics_recorded) return;
    if (slot.error_code) |name| if (std.mem.eql(u8, name, "PrefillDoesNotFit")) {
        slot.metrics_recorded = true;
        m.recordRejected();
        return;
    };
    recordSlotEnd(m, slot, cleanupOutcome(slot));
}

pub fn finishSlot(sch: *Scheduler, slot: *Slot, reason: []const u8) void {
    // Emit the `[spec-stats]` summary (no-op for non-speculative slots).
    // The legacy generate() path logs this itself; scheduler-driven slots
    // finalize here instead.
    if (slot.legacy_gen) |*g| {
        g.logSpecStats();
        // `[qsa-arms]` rides the same seam: the SERVE path finalizes here, so
        // wiring it only beside generate.zig's own logSpecStats() calls (the
        // legacy/CLI path) makes it dead on every served request.
        g.logQsaArms();
        g.persistRoundCost();
        // The cache commit below snapshots the n-gram history: settle the pipelined step's owed advance first.
        g.xfm.flushDeferredPle(&g.ctx) catch g.xfm.discardDeferredPle(&g.ctx);
    }
    const latched: ?[]const u8 = mlx.peekErrorName();
    if (latched) |name| {
        log.err("[scheduler] finish suppressed: the last forward failed ({s}) but produced a \"{s}\" — failing this request rather than answering 200 with what Metal never wrote\n", .{ name, reason });
    }
    if (slot.completion_tokens <= 2) logShortGen(slot, reason);
    commitSlotIfApplicable(sch, slot);
    releasePluginSlot(slot);
    publishHotCacheResidency(sch);
    // Restore by move: a slot that ended without committing still holds its checkout, and
    // the record now describes bytes that die with `slot.cache`. Above every early return.
    if (slot.model.prefix_cache) |hc| hc.releaseCheckout(@intFromPtr(slot), reason);
    // SSD flush runs AFTER markFinished so the client never waits on the
    // chunk-append — but everything it needs must be captured BEFORE the
    // broadcast: the conn thread may complete()+free the slot immediately.
    // The prefix cache and stream live on the registry-owned LoadedModel,
    // which outlives the slot (unload also runs on this thread).
    const hc_opt: ?*prefix_cache_mod.HotPrefixCache =
        if (slot.model.prefix_cache) |p| p else null;
    const stream_opt: ?mlx.mlx_stream =
        if (slot.model.transformer) |x| x.s else null;
    // Record per-request metrics while slot fields are still live (before
    // markFinished broadcasts — the conn thread may complete()+free the slot
    // immediately after). Off the per-token path; a null sink (metrics off) is
    // a single per-request branch. real_ttft = first_token_ns (queue+prefill,
    // captured exactly at prefill completion); recordRequest derives
    // e2e = first_token_ns + decode_ns.
    recordSlotEnd(sch.metrics, slot, finishOutcome(reason, latched));
    // mlx-stream: the arch's request end (GLM-5.3's decode line), before the terminator.
    if (slot.model.transformer) |x| if (x.mlx_stream_ext) |m| mlx_stream.end(m);
    publishSlotTerminator(slot, reason, latched);
    if (hc_opt) |hc| {
        if (stream_opt) |s| {
            hc.flushPendingDisk(s);
            // After the flush, so this turn's entry is the MRU one and any entry spilled
            // already has a complete copy to spill into.
            hc.spillIdleEntries(s);
            publishHotCacheResidency(sch);
        }
    }
    // Return this turn's transients to the OS. The per-`CACHE_CLEAR_INTERVAL`
    // clear inside `Generator.advanceStep` can't cover the tail of a turn, and
    // MLX parks freed buffers in a size-keyed pool rather than releasing them —
    // so without this a short turn hands everything it stranded to the next one
    // and the process footprint ratchets across a session (issue #110).
    _ = mlx.mlx_clear_cache();
}

// ── tests ──

test "a finish over a latched MLX failure ends the request as an ERROR, never a 200" {
    const Stub = struct {
        finished: ?[]const u8 = null,
        errored: ?[]const u8 = null,
        fn markFinished(self: *@This(), reason: []const u8) void {
            self.finished = reason;
        }
        fn markError(self: *@This(), name: []const u8) void {
            self.errored = name;
        }
    };

    var clean = Stub{};
    publishSlotTerminator(&clean, "stop", null);
    try testing.expectEqualStrings("stop", clean.finished.?);
    try testing.expect(clean.errored == null);

    // Latched: no "stop" is published at all.
    var poisoned = Stub{};
    publishSlotTerminator(&poisoned, "stop", "OutOfMemory");
    try testing.expect(poisoned.finished == null);
    try testing.expectEqualStrings("OutOfMemory", poisoned.errored.?);
    try testing.expect(Slot.errorNameIsMemory(poisoned.errored.?));

    var shape = Stub{};
    publishSlotTerminator(&shape, "length", "MlxFailure");
    try testing.expect(shape.finished == null);
    try testing.expectEqualStrings("MlxFailure", shape.errored.?);
    try testing.expect(!Slot.errorNameIsMemory(shape.errored.?));
}

const OutcomeStub = struct {
    finished: bool = false,
    error_code: ?[]const u8 = null,
    finish_reason: []const u8 = "",
    cancelled: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    metrics_recorded: bool = false,
    first_token_ns: u64 = 5_000_000,
    prefill_ns: u64 = 4_000_000,
    decode_ns: u64 = 20_000_000,
    prompt_tokens: u32 = 100,
    completion_tokens: u32 = 10,
    cached_tokens: u32 = 0,

    /// `finishSlot` on the inference thread: records, then publishes the terminator.
    fn finish(self: *OutcomeStub, m: ?*metrics_mod.Metrics, reason: []const u8) void {
        recordSlotEnd(m, self, finishOutcome(reason, null));
        self.finished = true;
        self.finish_reason = reason;
    }
    /// `Slot.markError` mid-decode: no finishSlot follows.
    fn fail(self: *OutcomeStub, name: []const u8) void {
        if (self.error_code != null or self.finished) return;
        self.error_code = name;
    }
    /// The connection thread's `Scheduler.complete`: sets `cancelled` on EVERY completion.
    fn complete(self: *OutcomeStub) void {
        self.cancelled.store(true, .release);
    }
};

fn expectOutcomes(m: *const metrics_mod.Metrics, success: u64, cancelled: u64, failed: u64, rejected: u64) !void {
    try testing.expectEqual(success, m.requests_success_total.load());
    try testing.expectEqual(cancelled, m.requests_cancelled_total.load());
    try testing.expectEqual(failed, m.requests_failed_total.load());
    try testing.expectEqual(rejected, m.requests_rejected_total.load());
}

test "outcome row 1: a disconnect while the request waits in pending counts cancelled once" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "outcome row 2: a disconnect during prefill is counted once, not again by the drain" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.complete();
    s.finish(&m, "cancelled");
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "outcome row 3: a disconnect during decode never reaches finishSlot and the drain counts it" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.completion_tokens = 7;
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 1, 0, 0);
    // A cancelled request never feeds the success histograms.
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
}

test "outcome row 4: a normal completion is success once even though complete() sets cancelled" {
    var m = metrics_mod.Metrics.init();
    for ([_][]const u8{ "stop", "length" }) |reason| {
        var s = OutcomeStub{};
        s.finish(&m, reason);
        s.complete();
        recordSlotCleanup(&m, &s);
    }
    try expectOutcomes(&m, 2, 0, 0, 0);
    try testing.expectEqual(@as(u64, 20), m.generation_tokens_total.load());
}

test "outcome row 5: a mid-decode generation error counts failed and leaves the histograms alone" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("MlxFailure");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
    try testing.expectEqual(@as(u64, 0), m.ttft_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.e2e_latency_ns.count.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), m.generation_tokens_total.load());
}

test "outcome row 5b: a finish over a latched MLX failure counts failed, not success" {
    try testing.expectEqual(metrics_mod.Outcome.failed, finishOutcome("stop", "OutOfMemory"));
    try testing.expectEqual(metrics_mod.Outcome.success, finishOutcome("stop", null));
    try testing.expectEqual(metrics_mod.Outcome.cancelled, finishOutcome("cancelled", null));
}

test "outcome row 6: an error followed by a client disconnect is one failure, never two outcomes" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("MlxFailure");
    recordSlotCleanup(&m, &s);
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 1, 0);
}

test "outcome row 7: a slot refused before its first forward is a rejection, not a failure" {
    var m = metrics_mod.Metrics.init();
    var s = OutcomeStub{};
    s.fail("PrefillDoesNotFit");
    s.complete();
    recordSlotCleanup(&m, &s);
    try expectOutcomes(&m, 0, 0, 0, 1);
}

test "outcome row 9: metrics off does no work and leaves the slot untouched" {
    var s = OutcomeStub{};
    s.complete();
    recordSlotCleanup(null, &s);
    s.finish(null, "stop");
    try testing.expect(!s.metrics_recorded);
}

test "[short-gen] carries both token counts, the ids and what they decode to" {
    // Bar: the line carries emitted vs realized and survives quotes, newlines and a 300-byte answer.
    var buf: [1024]u8 = undefined;
    const ids = [_]u32{ 151645, 17 };
    try testing.expectEqualStrings(
        "[short-gen] reason=stop emitted=0 realized=2 ids=[151645,17] bytes=\"hi\" path=mtp",
        formatShortGen(&buf, "stop", 0, &ids, "hi", "mtp"),
    );

    const esc = formatShortGen(&buf, "length", 2, &.{}, "a\"b\nc", "serial");
    try testing.expectEqualStrings(
        "[short-gen] reason=length emitted=2 realized=0 ids=[] bytes=\"a\\\"b\\nc\" path=serial",
        esc,
    );

    var long: [300]u8 = undefined;
    @memset(&long, 'x');
    const capped = formatShortGen(&buf, "stop", 1, &.{}, &long, "pld");
    try testing.expectEqual(@as(usize, SHORT_GEN_TEXT_CAP), std.mem.count(u8, capped, "x"));
}
