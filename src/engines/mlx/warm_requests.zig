//! The requests a model serves itself on every load. A kernel compiles at its first use, and the variant a forward
//! selects follows the prompt's size class, the cache it lands on, the speculative mode and every switch, flag,
//! setting and pack format the load read: the bare forwards of `warmup.zig` reach a fraction of them, and the rest
//! would compile inside the first prompts someone sends. Metal's shader cache is the one record of what is compiled:
//! a request whose kernels it holds compiles nothing, so the plan costs only its own forwards on a load that matches an
//! earlier one, and nothing else is kept to decide whether a load owes it (no marker: a key of the load's inputs is a
//! copy of the deciders, and it missed them). By default the model reports ready first and serves the width sweep and
//! a few short requests while nothing else is asked of the server (`tick`); under `--warmup-full` it serves them, a
//! ladder of prompt sizes and a prompt at its widest pass before it reports ready (`run`), which is what a benchmark
//! wants. macOS only (`wanted`).

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const plugin_driver = @import("plugin/driver.zig");
const dsparkEnabled = @import("spec/dspark.zig").dsparkEnabled;
const LoadedModel = @import("../../host/registry/registry.zig").LoadedModel;
const scheduler_mod = @import("scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = @import("slot.zig").Slot;
const chatPass = @import("step.zig").chatPass;
const drainSideQueues = @import("runtime/loop.zig").drainSideQueues;

/// A prompt of `len` synthetic ids and `gen` tokens generated from it. The ids are the start of sequence `base`,
/// but for the last `own`: a request with some lands on what the one before it left in the cache, and forks.
/// `rounds`: the request times a native drafter's rounds (`ROUNDS`).
const Request = struct { base: u32, len: u32, own: u32 = 0, gen: u32, sampled: bool = false, rounds: bool = false };

/// The request a load times a native drafter's rounds with (`rounds.Timing`, `Generator.timeRounds`): a context past
/// the sliding window, sampled at temperature 1 under a fixed seed, and room for every round to keep its block.
const ROUNDS: Request = .{ .base = 3, .len = 1280, .gen = 2048, .rounds = true };

/// Past the row counts the prefill kernels switch on.
const LONG: u32 = 256;
/// One full draft block of the widest speculator and a round after it.
const GEN: u32 = 20;

pub const Kind = enum { minimal, full };

/// A short chat, a long prompt, then that prompt with other, longer endings: a fork restores what the long prompt
/// left in the cache and prefills its tail as a chunk of its own. Short prompts only: a ready model serves these.
const MINIMAL = [_]Request{
    .{ .base = 1, .len = 24, .gen = GEN },
    .{ .base = 2, .len = LONG, .gen = 1 },
} ++ forks(2, LONG);
/// Then a ladder of prompts, one per size class the prefill kernels are built for and the last one past a chunk, two
/// of them forked the same way: a turn on a long context selects other variants than one on a short context.
const FULL = MINIMAL ++ ladder;
const ladder = blk: {
    var rungs: [LADDER.len + FORKED.len * FORK_GROWTH.len]Request = undefined;
    var n: usize = 0;
    for (LADDER) |len| {
        rungs[n] = .{ .base = len, .len = len, .gen = 1 };
        n += 1;
        if (std.mem.indexOfScalar(u32, &FORKED, len) == null) continue;
        for (forks(len, len)) |f| {
            rungs[n] = f;
            n += 1;
        }
    }
    break :blk rungs;
};
const LADDER = [_]u32{ 768, 1536, 2048, 3072, 5120, 10240 };
const FORKED = [_]u32{ 3072, 10240 };
/// A prompt's state restores `KEPT_SHORT` rows short of its end, and a tail runs as 32-row blocks, a remainder and
/// its last row. Each fork drops some of its prompt's last ids and grows past it: the first leaves a remainder of 6
/// and generates; the rest leave 1, 2 and 3, the remainders with kernel variants of their own, and drop enough that
/// no fork before them left a restore point inside what they share with it.
const KEPT_SHORT: u32 = 31;
const FORK_GROWTH = [_]u32{ 8, 3, 4, 5 };
const FORK_DROP = [_]u32{ 8, 30, 30, 30 };
const LONGEST: u32 = LADDER[LADDER.len - 1] + FORK_GROWTH[0];

/// `len` ids of sequence `base` with other, longer endings.
fn forks(base: u32, len: u32) [FORK_GROWTH.len]Request {
    var out: [FORK_GROWTH.len]Request = undefined;
    for (FORK_GROWTH, FORK_DROP, &out, 0..) |grow, drop, *r, i|
        r.* = .{ .base = base, .len = len + grow, .own = drop + grow, .gen = if (i == 0) GEN else 1, .sampled = i == 0 };
    return out;
}
/// Longer than any one request takes with every kernel still to compile; past it the load goes on without the rest.
/// The plan has no time budget of its own: a slower or hotter machine serves every rung, the widest last.
const LIMIT_NS: u64 = 120 * std.time.ns_per_s;
/// How long a real request waits behind an idle warm request before the warm-up gives way for good.
const WAITED_ON_NS: u64 = 15 * std.time.ns_per_s;

/// Ordinary vocabulary ids, clear of the special ids at either end of a vocabulary. `salt` names the sequence.
pub fn promptIds(buf: []u32, vocab: u32, salt: u32) void {
    const low: u32 = if (vocab > 2000) 1000 else 0;
    const span = @max(@min(vocab, 20_000), low + 1) - low;
    for (buf, 0..) |*id, i| {
        const h = (@as(u32, @truncate(i)) +% 1) *% 2654435761 +% salt *% 40503;
        id.* = low + (h >> 8) % span;
    }
}

fn fill(buf: []u32, vocab: u32, r: Request) []const u32 {
    const ids = buf[0..r.len];
    promptIds(ids, vocab, r.base);
    promptIds(ids[r.len - r.own ..], vocab, r.base +% 0x9e3779b9 *% r.own);
    return ids;
}

/// Whether this load serves warm requests: on macOS, where Metal compiles a kernel at its first use and keeps it in the
/// per-user shader cache. A phone's first load cannot afford the ladder, and no other backend compiles this way.
pub fn wanted() bool {
    return comptime builtin.os.tag == .macos;
}

/// The prompt at the load's widest pass (`ModelConfig.pinned_prefill_chunk`, the width the load pinned from memory, or
/// the explicit `--prefill-chunk`), one row short of the context `cap` leaves (0 = no limit): its pass's kernels are
/// built for that width, and no ladder rung reaches it on a load that pins wider than the ladder. Null where the pin is
/// no wider than the ladder's widest prompt.
fn widestRung(pinned: u32, cap: u32) ?Request {
    const len = if (cap != 0) @min(pinned, cap -| 1) else pinned;
    if (len <= LONGEST) return null;
    return .{ .base = len, .len = len, .gen = 1 };
}

/// Serve the full plan on the inference thread, inside the load: `FULL`, then the prompt at the load's widest pass
/// (`widestRung`). The requests run against a RAM twin of the model's prefix cache and count in no metric, so the model
/// comes up as if nothing had been asked of it. A shutdown, a failed request or the context cap stops it; nothing is
/// kept either way, so the next load serves the whole plan again.
pub fn run(sch: *Scheduler, entry: *LoadedModel) void {
    const cfg = entry.config orelse return;
    const cap = promptLimit(cfg.contextCap(), cfg.ctx_override, sch.gguf_ctx_size);
    const widest = widestRung(cfg.pinned_prefill_chunk, cap);
    const n = FULL.len + @intFromBool(widest != null);
    // A rung past the context is no part of this load's plan: no request can reach it.
    var total: usize = 0;
    for (0..n) |i| total += @intFromBool(fits(planAt(i, widest), cap));
    log.info("Warm requests: compiling this load's kernels before it reports ready ({d} requests, widest pass {d} rows).\n", .{ total, if (widest) |w| w.len else 0 });
    const sw = io_util.Stopwatch.init(sch.io);
    var twin = twinOf(entry);
    defer if (twin) |*t| t.deinit();

    const buf = sch.allocator.alloc(u32, @max(LONGEST, if (widest) |w| w.len else 0)) catch {
        log.warn("Warm requests: no memory for their prompts; the first prompts compile the kernels.\n", .{});
        return;
    };
    defer sch.allocator.free(buf);
    var done: usize = 0;
    for (0..n) |i| {
        const r = planAt(i, widest);
        if (!fits(r, cap)) continue;
        serve(sch, entry, if (twin) |*t| t else null, fill(buf, cfg.vocab_size, r), r) catch |err| {
            if (err == error.Shutdown) return;
            log.warn("Warm request {d} of {d} failed ({s}); the first prompts compile what is left.\n", .{ done + 1, total, @errorName(err) });
            break;
        };
        done += 1;
    }
    if (done == total) {
        log.info("Warm requests complete ({d} of {d}, {d} ms).\n", .{ done, total, sw.read() / std.time.ns_per_ms });
    } else {
        log.warn("Warm requests stopped at {d} of {d} ({d} ms).\n", .{ done, total, sw.read() / std.time.ns_per_ms });
    }
}

/// Request `i` of the full plan: `FULL`, then the widest pass's prompt.
fn planAt(i: usize, widest: ?Request) Request {
    return if (i < FULL.len) FULL[i] else widest.?;
}

/// Whether `r` fits the context `cap` leaves (0 = no limit).
fn fits(r: Request, cap: u32) bool {
    return cap == 0 or r.len + r.gen <= cap;
}

/// Whether `entry` serves a native drafter whose rounds no request has been asked to time, where rounds run at all.
pub fn owesRounds(entry: *const LoadedModel) bool {
    const xfm = entry.transformer orelse return false;
    return plugin_driver.nativeStages(xfm) > 0 and !xfm.plugin.?.ladder_asked and dsparkEnabled();
}

/// Time `entry`'s native drafter's rounds now, inside the load (`--warmup-full`: whoever asks next is served on them).
pub fn timeRounds(sch: *Scheduler, entry: *LoadedModel) void {
    if (!owesRounds(entry)) return;
    const cfg = entry.config orelse return;
    if (!holds(sch, cfg, ROUNDS)) return;
    var twin = twinOf(entry);
    defer if (twin) |*t| t.deinit();
    var buf: [ROUNDS.len]u32 = undefined;
    serve(sch, entry, if (twin) |*t| t else null, fill(&buf, cfg.vocab_size, ROUNDS), ROUNDS) catch |err|
        log.warn("Timing the drafter's rounds failed ({s}); it stays off.\n", .{@errorName(err)});
}

/// An empty cache with the retention rules of the model's own, for its warm requests.
fn twinOf(entry: *const LoadedModel) ?prefix_cache_mod.HotPrefixCache {
    var twin = (entry.prefix_cache orelse return null).ramTwin();
    // The fork needs the long prompt kept; the ladder's prompts are worth nothing once prefilled.
    twin.max_entries = @min(twin.max_entries, 2);
    return twin;
}

/// The longest prompt this load serves: what the model resolves, under the context the operator set for the model or
/// the process. 0 = no limit.
fn promptLimit(model_cap: u32, model_ctx: u32, launch_ctx: u32) u32 {
    const set = if (model_ctx != 0) model_ctx else launch_ctx;
    if (set == 0) return model_cap;
    return if (model_cap == 0) set else @min(model_cap, set);
}

/// Whether the context this load serves holds `r`. A drafter whose timing request it cannot hold stays off.
fn holds(sch: *const Scheduler, cfg: anytype, r: Request) bool {
    const cap = promptLimit(cfg.contextCap(), cfg.ctx_override, sch.gguf_ctx_size);
    if (cap == 0 or r.len + r.gen <= cap) return true;
    if (r.rounds) log.warn("[rounds] the context holds under {d} tokens: the drafter's rounds are not timed and it stays off\n", .{r.len + r.gen});
    return false;
}

fn submit(sch: *Scheduler, entry: *LoadedModel, cache: ?*prefix_cache_mod.HotPrefixCache, ids: []const u32, r: Request) !*Slot {
    const cfg = entry.config.?;
    // The request-level defaults of `host/api/request_options.zig`; the engine picks one mode from them.
    const sidecar = entry.dflash != null or (entry.drafter != null and !cfg.isMoe() and !cfg.has_hybrid_layers);
    const slot = try scheduler_mod.submitNow(sch, .{
        .model = entry,
        .prefix_cache = cache,
        .prompt_ids = ids,
        .sampling = if (r.rounds) .{ .temperature = 1.0, .seed = 1 } else if (r.sampled) .{ .temperature = 0.7, .top_p = 0.95, .top_k = 20 } else .{ .temperature = 0 },
        .eos_token_ids = &.{},
        .max_tokens = r.gen,
        .enable_thinking = r.sampled,
        .enable_pld = true,
        .enable_drafter = sidecar,
        .drafter = if (sidecar) entry.drafter else null,
        .dflash = if (sidecar) entry.dflash else null,
        .drafter_block_size = entry.drafter_block_size,
        .enable_mtp = true,
        .mtp = entry.mtp,
        .mtp_depth = entry.mtp_depth,
    });
    slot.metrics_recorded = true;
    slot.warm = true;
    slot.times_rounds = r.rounds;
    if (r.rounds) entry.transformer.?.plugin.?.ladder_asked = true;
    return slot;
}

/// Give the request's slot back and free it here, on the thread that owns its arrays.
fn release(sch: *Scheduler, slot: *Slot) void {
    scheduler_mod.complete(sch, slot);
    while (drainSideQueues(sch)) {}
}

const Outcome = enum { running, ended, failed, timed_out };

fn outcome(slot: *Slot, asked: io_util.Stopwatch, limit_ns: u64) Outcome {
    while (slot.waitNextTimeout(0)) |next| switch (next) {
        .token => {},
        .done => return .ended,
        .err => return .failed,
    };
    return if (asked.read() > limit_ns) .timed_out else .running;
}

fn serve(sch: *Scheduler, entry: *LoadedModel, cache: ?*prefix_cache_mod.HotPrefixCache, ids: []const u32, r: Request) !void {
    const slot = try submit(sch, entry, cache, ids, r);
    defer release(sch, slot);
    const sw = io_util.Stopwatch.init(sch.io);
    while (true) {
        switch (outcome(slot, sw, LIMIT_NS)) {
            .running => {},
            .ended => return,
            .failed => return error.WarmRequestFailed,
            .timed_out => return error.WarmRequestTimedOut,
        }
        if (chatPass(sch, .yield) == .shutdown) return error.Shutdown;
    }
}

/// The warm-up a ready model still owes: the verify sweep, then `MINIMAL`, a step at a time while nothing else is
/// asked of the server (`tick`). It never holds its model in the registry, so the model stays as evictable as any
/// idle one: whoever unloads it drops the warm-up first (`forget`).
pub const Idle = struct {
    entry: *LoadedModel,
    verify_rows: u32,
    /// How many of `MINIMAL` it serves: none on a model that cannot chat, or a load that served them before ready.
    requests: u32,
    /// Then `ROUNDS`: the model's native drafter is not timed yet.
    rounds: bool,
    twin: ?prefix_cache_mod.HotPrefixCache,
    /// The sweep's widths done, then the requests ended.
    step: u32 = 0,
    slot: ?*Slot = null,
    asked: io_util.Stopwatch = undefined,
    /// Since a real request began to wait behind `slot`.
    waited_on: ?io_util.Stopwatch = null,
    spent_ns: u64 = 0,
};

/// `entry`, just loaded, owes the minimal warm-up (every load does, whatever an earlier one served) and the timing of
/// its native drafter's rounds, if it has one. `verify_rows` 0 and `chat` false: the timing alone. Inference thread.
pub fn owe(sch: *Scheduler, entry: *LoadedModel, verify_rows: u32, chat: bool) void {
    const w = sch.allocator.create(Idle) catch return;
    sch.warm_queue.append(sch.allocator, w) catch return sch.allocator.destroy(w);
    w.* = .{ .entry = entry, .verify_rows = verify_rows, .requests = if (chat) MINIMAL.len else 0, .rounds = owesRounds(entry), .twin = twinOf(entry) };
}

/// Idle warm-ups that served their whole plan, and those cut short (a failed or slow request, a request given way to,
/// the context cap, the model leaving, shutdown): a test's view of how they ended. Neither is read by any load.
pub var completed: std.atomic.Value(u32) = .init(0);
pub var cut: std.atomic.Value(u32) = .init(0);

/// Requests in flight beside the warm one, on this thread's engine. Caller holds `queue_mu`.
fn othersLocked(sch: *const Scheduler) usize {
    return (sch.in_flight -| sch.warm_in_flight) -| sch.offloaded.items.len;
}

/// A warm-up can move: its request ended, or nothing else is asked of this thread. Caller holds `queue_mu`.
pub fn owesStepLocked(sch: *const Scheduler) bool {
    return sch.warm_queue.items.len > 0 and othersLocked(sch) == 0;
}

/// The warm request that has not ended: a pass admits it alone and nothing beside it (`chatPass`), so a real request
/// never decodes in its company. Inference thread.
pub fn liveSlot(sch: *const Scheduler) ?*Slot {
    if (sch.warm_queue.items.len == 0) return null;
    const slot = sch.warm_queue.items[0].slot orelse return null;
    return if (slot.finished or slot.error_code != null or slot.cancelled.load(.acquire)) null else slot;
}

const Step = union(enum) { wait, sweep: u32, request: u32, done, abort };

/// What an idle warm-up does now. A step starts only while its model is ready and nothing else is asked of the
/// server; a request in flight is waited for, not cut (it is a few hundred tokens), unless the model is leaving.
fn nextStep(step: u32, widths: u32, requests: u32, ready: bool, in_flight: bool, idle: bool) Step {
    if (!ready) return .abort;
    if (in_flight or !idle) return .wait;
    if (step < widths) return .{ .sweep = step + 1 };
    if (step - widths < requests) return .{ .request = step - widths };
    return .done;
}

fn readyNow(sch: *Scheduler, entry: *const LoadedModel) bool {
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    return entry.state == .ready;
}

/// One step of the first warm-up owed, on the inference thread between its passes.
pub fn tick(sch: *Scheduler) void {
    if (sch.warm_queue.items.len == 0) return;
    const w = sch.warm_queue.items[0];
    sch.queue_mu.lockUncancelable(sch.io);
    const others = othersLocked(sch);
    const idle = others == 0 and sch.load_queue.items.len + sch.unload_queue.items.len + sch.gen_queue.items.len == 0;
    sch.queue_mu.unlock(sch.io);
    const slot = w.slot orelse {
        if (!idle) return;
        return start(sch, w);
    };
    if (others == 0) w.waited_on = null else if (w.waited_on == null) w.waited_on = io_util.Stopwatch.init(sch.io);
    const gave_way = if (w.waited_on) |sw| sw.read() > WAITED_ON_NS else false;
    const now = outcome(slot, w.asked, LIMIT_NS);
    switch (if (now == .running and gave_way) Outcome.timed_out else now) {
        .running => {},
        .ended => {
            endRequest(sch, w);
            w.step += 1;
        },
        .failed, .timed_out => |why| {
            const how = if (why == .failed) "failed" else "took too long";
            if (slot.times_rounds)
                log.warn("[rounds] the round timing {s}: the drafter stays off until the model loads again\n", .{how})
            else
                log.warn("Warm request {d} of {d} {s}; the first prompts compile what is left.\n", .{ w.step - w.verify_rows + 1, w.requests, how });
            end(sch, w, false);
        },
    }
}

/// Start the next step of `w`: nothing of its own is in flight and nothing else is asked of the server.
fn start(sch: *Scheduler, w: *Idle) void {
    switch (nextStep(w.step, w.verify_rows, w.requests + @intFromBool(w.rounds), readyNow(sch, w.entry), false, true)) {
        .wait => {},
        .abort => end(sch, w, false),
        .done => end(sch, w, true),
        .sweep => |width| {
            const sw = io_util.Stopwatch.init(sch.io);
            // Best effort: a width that fails compiles in the round that first verifies it, and must fail no request.
            const had_error = mlx.errorPending();
            if (w.entry.transformer) |xfm| xfm.sweepWidth(width) catch {};
            mlx.dropLatchedErrorUnless(had_error);
            w.spent_ns += sw.read();
            w.step += 1;
        },
        .request => |i| {
            const cfg = w.entry.config.?;
            const r = if (i < w.requests) MINIMAL[i] else ROUNDS;
            // Past the context: no request can reach it, so it is no part of this load's plan.
            if (!holds(sch, cfg, r)) {
                w.step += 1;
                return;
            }
            var buf: [@max(LONG + FORK_GROWTH[0], ROUNDS.len)]u32 = undefined;
            sch.queue_mu.lockUncancelable(sch.io);
            sch.warm_in_flight = 1;
            sch.queue_mu.unlock(sch.io);
            w.slot = submit(sch, w.entry, if (w.twin) |*t| t else null, fill(&buf, cfg.vocab_size, r), r) catch |err| {
                log.warn("Warm request {d} of {d} failed ({s}); the first prompts compile what is left.\n", .{ i + 1, w.requests, @errorName(err) });
                return end(sch, w, false);
            };
            w.asked = io_util.Stopwatch.init(sch.io);
            w.waited_on = null;
        },
    }
}

/// Give back the request in flight, if any.
fn endRequest(sch: *Scheduler, w: *Idle) void {
    if (w.slot) |slot| {
        w.spent_ns += w.asked.read();
        release(sch, slot);
        w.slot = null;
    }
    sch.queue_mu.lockUncancelable(sch.io);
    sch.warm_in_flight = 0;
    sch.queue_mu.unlock(sch.io);
}

/// A warm-up is over: its whole plan served (`complete`), or cut short. Nothing is kept either way: the next load owes
/// its own.
fn end(sch: *Scheduler, w: *Idle, complete: bool) void {
    endRequest(sch, w);
    if (w.twin) |*t| t.deinit();
    const served = @min(w.step -| w.verify_rows, w.requests);
    if (complete) {
        _ = completed.fetchAdd(1, .monotonic);
        log.info("Warm requests complete ({d} of {d}, {d} ms).\n", .{ served, w.requests, w.spent_ns / std.time.ns_per_ms });
    } else {
        _ = cut.fetchAdd(1, .monotonic);
        if (w.step > 0) log.warn("Warm requests stopped at {d} of {d} ({d} ms).\n", .{ served, w.requests, w.spent_ns / std.time.ns_per_ms });
    }
    _ = sch.warm_queue.orderedRemove(std.mem.indexOfScalar(*Idle, sch.warm_queue.items, w).?);
    sch.allocator.destroy(w);
}

/// `entry` is about to be unloaded (every unload runs on this thread, between passes): its warm-up goes first, the
/// request in flight and the cache with it.
pub fn forget(sch: *Scheduler, entry: *const LoadedModel) void {
    for (sch.warm_queue.items) |w| if (w.entry == entry) return end(sch, w, false);
}

/// The inference thread is leaving: every warm-up owed gives back its request and its cache.
pub fn stop(sch: *Scheduler) void {
    while (sch.warm_queue.items.len > 0) end(sch, sch.warm_queue.items[0], false);
    sch.warm_queue.deinit(sch.allocator);
}

// ── tests ──

const testing = std.testing;

test "promptIds: ordinary ids inside the vocabulary, the same sequence for the same salt" {
    var a: [LONG]u32 = undefined;
    var b: [LONG]u32 = undefined;
    promptIds(&a, 248_320, 7);
    promptIds(&b, 248_320, 7);
    try testing.expectEqualSlices(u32, &a, &b);
    var distinct = std.AutoHashMap(u32, void).init(testing.allocator);
    defer distinct.deinit();
    for (a) |id| {
        try testing.expect(id >= 1000 and id < 20_000);
        try distinct.put(id, {});
    }
    // A prompt of one repeated id would route every row alike and score as an echo.
    try testing.expect(distinct.count() > LONG * 9 / 10);
    promptIds(&b, 248_320, 8);
    try testing.expect(!std.mem.eql(u32, &a, &b));
    // A fixture-sized vocabulary still gets ids it holds.
    promptIds(&a, 64, 7);
    for (a) |id| try testing.expect(id < 64);
}

test "a fork restores its prompt and nothing after it, one fork for each remainder of a 32-row block" {
    var a: [LONGEST]u32 = undefined;
    var b: [LONGEST]u32 = undefined;
    var remainders: [4]usize = @splat(0);
    for (FULL, 0..) |r, i| {
        if (r.own == 0) continue;
        var whole = i - 1;
        while (FULL[whole].own != 0) whole -= 1;
        const base = FULL[whole];
        const forked = fill(&b, 50_000, r);
        const shared = std.mem.indexOfDiff(u32, fill(&a, 50_000, base), forked).?;
        // What the prompt left can be restored inside what the fork shares with it.
        try testing.expect(base.len - KEPT_SHORT <= shared);
        // What an earlier fork of the same prompt left cannot: it would shorten this fork's tail.
        for (FULL[whole + 1 .. i]) |earlier| {
            try testing.expect(std.mem.indexOfDiff(u32, fill(&a, 50_000, earlier), forked).? == shared);
            try testing.expect(earlier.len - KEPT_SHORT > shared);
        }
        // The tail past the restore point, less the row forwarded last.
        const rows = r.len - (base.len - KEPT_SHORT) - 1;
        try testing.expect(rows > 32 and rows % 32 != 0);
        remainders[if (rows % 32 > 3) 3 else rows % 32 - 1] += 1;
    }
    for (remainders) |n| try testing.expectEqual(1 + FORKED.len, n);
}

test "the ladder's prompts share nothing with each other, so each prefills whole" {
    var a: [LONGEST]u32 = undefined;
    var b: [LONGEST]u32 = undefined;
    var rungs: usize = 0;
    for (FULL, 0..) |r, i| {
        if (r.own != 0) continue;
        rungs += 1;
        const mine = fill(&a, 50_000, r);
        for (FULL[0..i]) |earlier| {
            if (earlier.own != 0) continue;
            try testing.expect(std.mem.indexOfDiff(u32, mine, fill(&b, 50_000, earlier)).? < 4);
        }
    }
    try testing.expectEqual(2 + LADDER.len, rungs);
    try testing.expect(std.sort.isSorted(u32, &LADDER, {}, std.sort.asc(u32)));
    for (FORKED) |len| try testing.expect(std.mem.indexOfScalar(u32, &LADDER, len) != null);
}

test "promptLimit: the ladder stops at the context the operator set, the model's own setting first" {
    try testing.expectEqual(@as(u32, 0), promptLimit(0, 0, 0));
    try testing.expectEqual(@as(u32, 131_072), promptLimit(131_072, 0, 0));
    try testing.expectEqual(@as(u32, 4096), promptLimit(131_072, 0, 4096));
    try testing.expectEqual(@as(u32, 8192), promptLimit(131_072, 8192, 4096));
    try testing.expectEqual(@as(u32, 2048), promptLimit(2048, 8192, 0));
    try testing.expectEqual(@as(u32, 4096), promptLimit(0, 0, 4096));
}

test "widestRung: a load that pins a pass wider than the ladder serves a prompt at that pass, one row short of the context" {
    // DeepSeek-V4.1 on a 512 GB machine at --ctx-size 32768 pins a 32,768-row pass: the prompt is 32,767 rows.
    try testing.expectEqual(@as(u32, 32_767), widestRung(32_768, 32_768).?.len);
    // A 1M context pins its pass from memory (44,544 rows), well inside the context.
    try testing.expectEqual(@as(u32, 44_544), widestRung(44_544, 1_048_576).?.len);
    try testing.expectEqual(@as(u32, 44_544), widestRung(44_544, 0).?.len);
    // The rung fits the context the plan checks.
    try testing.expect(fits(widestRung(32_768, 32_768).?, 32_768));
    // No rung where the ladder already reaches the pin, or no pin was made.
    try testing.expect(widestRung(8192, 32_768) == null);
    try testing.expect(widestRung(LONGEST, 0) == null);
    try testing.expect(widestRung(0, 0) == null);
    try testing.expectEqual(@as(u32, LONGEST + 1), widestRung(LONGEST + 1, 0).?.len);
}

test "the full plan: FULL, then the widest pass's prompt last; a rung past the context is skipped, never a stop" {
    const w = widestRung(32_768, 32_768).?;
    for (FULL, 0..) |r, i| try testing.expectEqual(r, planAt(i, w));
    try testing.expectEqual(w, planAt(FULL.len, w));
    try testing.expect(fits(planAt(FULL.len, w), 32_768));
    // Under a 4,096-token context the 5,120 and 10,240 rungs and their forks cannot be asked for; what fits still runs.
    var fit: usize = 0;
    for (FULL) |r| fit += @intFromBool(fits(r, 4096));
    try testing.expect(fit > MINIMAL.len and fit < FULL.len);
    try testing.expect(fits(.{ .base = 3072, .len = 3072, .gen = 1 }, 4096));
}

test "the minimal plan is short prompts only, and the full plan starts with it" {
    for (MINIMAL) |r| try testing.expect(r.len <= LONG + 8 and r.gen <= GEN);
    try testing.expect(FULL.len > MINIMAL.len);
    for (MINIMAL, FULL[0..MINIMAL.len]) |a, b| try testing.expectEqual(a, b);
    for (FULL[MINIMAL.len..]) |r| try testing.expect(r.len > LONG + 8);
}

test "nextStep: the sweep's widths, then the requests, one at a time and only while the server is idle" {
    // widths 1..2, then requests 0..2, then done.
    const want = [_]Step{ .{ .sweep = 1 }, .{ .sweep = 2 }, .{ .request = 0 }, .{ .request = 1 }, .{ .request = 2 }, .done };
    for (want, 0..) |step, i| try testing.expectEqual(step, nextStep(@intCast(i), 2, 3, true, false, true));
    // A model that cannot chat owes the sweep alone.
    try testing.expectEqual(Step.done, nextStep(2, 2, 0, true, false, true));
    // Nothing starts beside a real request or before its own request ended.
    try testing.expectEqual(Step.wait, nextStep(0, 2, 3, true, false, false));
    try testing.expectEqual(Step.wait, nextStep(3, 2, 3, true, true, true));
    // A model that is leaving ends it at once, request in flight or not.
    try testing.expectEqual(Step.abort, nextStep(0, 2, 3, false, false, true));
    try testing.expectEqual(Step.abort, nextStep(3, 2, 3, false, true, false));
}

/// A scheduler with one ready model that owes an idle warm-up of no widths and no requests.
const IdleFixture = struct {
    reg: *@import("../../host/registry/registry.zig").ModelRegistry,
    sch: Scheduler = undefined,
    entry: *LoadedModel = undefined,
    /// `completed` and `cut` when the fixture began.
    done0: u32 = 0,
    cut0: u32 = 0,

    fn init(f: *IdleFixture) !void {
        f.reg = try .init(testing.allocator, testing.io, null, 4, 0, null);
        f.entry = try f.reg.registerStub("a", "/a", 1);
        f.entry.state = .ready;
        f.entry.prefix_cache = null;
        f.done0 = completed.load(.monotonic);
        f.cut0 = cut.load(.monotonic);
        f.sch.allocator = testing.allocator;
        f.sch.io = testing.io;
        f.sch.registry = f.reg;
        f.sch.queue_mu = .init;
        f.sch.warm_queue = .empty;
        f.sch.warm_in_flight = 0;
        f.sch.in_flight = 0;
        inline for (.{ "offloaded", "load_queue", "unload_queue", "gen_queue" }) |name| @field(f.sch, name) = .empty;
        owe(&f.sch, f.entry, 0, false);
    }

    fn deinit(f: *IdleFixture) void {
        stop(&f.sch);
        f.reg.deinit();
    }

    /// Warm-ups this fixture saw complete, and cut short.
    fn ended(f: *const IdleFixture) [2]u32 {
        return .{ completed.load(.monotonic) - f.done0, cut.load(.monotonic) - f.cut0 };
    }
};

test "an idle warm-up waits beside a real request without holding its model, then completes" {
    var f: IdleFixture = .{ .reg = undefined };
    try f.init();
    defer f.deinit();
    // Beside a real request it waits, and the thread may park: that request's end wakes it.
    f.sch.in_flight = 1;
    try testing.expect(!owesStepLocked(&f.sch));
    tick(&f.sch);
    try testing.expectEqual(@as(usize, 1), f.sch.warm_queue.items.len);
    try testing.expectEqual([2]u32{ 0, 0 }, f.ended());
    // Waiting, it pins nothing: a load that needs the room, or the idle sweep, can still unload the model.
    try testing.expectEqual(@as(u32, 0), f.entry.refcount.load(.acquire));
    try testing.expect(f.reg.pickLruEvictable("", false) == f.entry);
    f.sch.in_flight = 0;
    try testing.expect(owesStepLocked(&f.sch));
    tick(&f.sch);
    try testing.expectEqual(@as(usize, 0), f.sch.warm_queue.items.len);
    try testing.expectEqual([2]u32{ 1, 0 }, f.ended());
    try testing.expect(!owesStepLocked(&f.sch));
    // Every load owes its own warm-up: the same model loaded again queues the whole plan, whatever this one served.
    owe(&f.sch, f.entry, 9, true);
    try testing.expectEqual(@as(usize, 1), f.sch.warm_queue.items.len);
    try testing.expectEqual(@as(u32, 9), f.sch.warm_queue.items[0].verify_rows);
    try testing.expectEqual(@as(u32, MINIMAL.len), f.sch.warm_queue.items[0].requests);
}

test "forget: a model being unloaded takes its warm-up with it, cut short, wherever it stands in the queue" {
    var f: IdleFixture = .{ .reg = undefined };
    try f.init();
    defer f.deinit();
    const other = try f.reg.registerStub("b", "/b", 1);
    other.state = .ready;
    owe(&f.sch, other, 0, false);
    forget(&f.sch, other);
    try testing.expectEqual(@as(usize, 1), f.sch.warm_queue.items.len);
    try testing.expect(f.sch.warm_queue.items[0].entry == f.entry);
    forget(&f.sch, f.entry);
    try testing.expectEqual(@as(usize, 0), f.sch.warm_queue.items.len);
    try testing.expectEqual([2]u32{ 0, 2 }, f.ended());
}

test "an idle warm-up starts nothing on a model that is leaving, and claims nothing" {
    var f: IdleFixture = .{ .reg = undefined };
    try f.init();
    defer f.deinit();
    f.entry.state = .evicting;
    tick(&f.sch);
    try testing.expectEqual(@as(usize, 0), f.sch.warm_queue.items.len);
    try testing.expectEqual([2]u32{ 0, 1 }, f.ended());
}

test "stop: a warm-up still owed at shutdown is dropped, cut short; the partial plan claims nothing a later load reads" {
    var f: IdleFixture = .{ .reg = undefined };
    try f.init();
    defer f.deinit();
    stop(&f.sch);
    f.sch.warm_queue = .empty;
    try testing.expectEqual([2]u32{ 0, 1 }, f.ended());
    // The next load of the same model owes the whole plan again: nothing of the cut warm-up survives it.
    owe(&f.sch, f.entry, 0, true);
    try testing.expectEqual(@as(u32, MINIMAL.len), f.sch.warm_queue.items[0].requests);
    try testing.expectEqual(@as(u32, 0), f.sch.warm_queue.items[0].step);
}
