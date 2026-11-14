//! The requests a model serves itself the first time a build loads it on a machine. A kernel compiles at its first
//! use, and the variant a forward selects follows the prompt's size class, the cache it lands on and the speculative
//! mode: the bare forwards of `warmup.zig` reach a fraction of them, and the rest compiled inside the first prompts
//! someone sent. Metal keeps what it compiled, so later loads owe nothing. By default the model reports ready first
//! and serves a few short ones while nothing else is asked of the server (`tick`); under `--warmup-full` it serves
//! them and a ladder of prompt sizes before it reports ready (`run`), which is what a benchmark wants.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const crt = @import("../../sdk/util/crt.zig");
const round_cost = @import("spec/round_cost.zig");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const prefix_cache_mod = @import("cache/prefix_cache.zig");
const LoadedModel = @import("../../host/registry/registry.zig").LoadedModel;
const scheduler_mod = @import("scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = @import("slot.zig").Slot;
const chatPass = @import("step.zig").chatPass;
const drainSideQueues = @import("runtime/loop.zig").drainSideQueues;

/// A prompt of `len` synthetic ids and `gen` tokens generated from it. The ids are the start of sequence `base`,
/// but for the last `own`: a request with some lands on what the one before it left in the cache, and forks.
const Request = struct { base: u32, len: u32, own: u32 = 0, gen: u32, sampled: bool = false };

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
/// A rung starts only inside this much of the ladder's time: a slow machine stops short of the widest prompts.
const LADDER_BUDGET_NS: u64 = 10 * std.time.ns_per_s;

/// Longer than any of them takes with every kernel still to compile; past it the load goes on without the rest.
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

/// What a load's kernels depend on beside the build: the pack, and what speculates on it.
pub const Subject = struct {
    model_path: []const u8,
    model_type: []const u8,
    /// The widest block a speculative round verifies (`warmup.zig`).
    verify_rows: u32,
    kv_bits: u32,
};

extern "c" fn confstr(name: c_int, buf: ?[*]u8, len: usize) usize;
const CS_DARWIN_USER_CACHE_DIR: c_int = 65538;

/// The per-user cache directory, where Metal keeps its compiled shaders: a marker there lives as long as they do.
/// macOS only: a phone's first load cannot afford the ladder, and no other backend keeps what it compiles here.
fn cacheDir(buf: []u8) ?[]const u8 {
    if (comptime builtin.os.tag != .macos) return null;
    const n = confstr(CS_DARWIN_USER_CACHE_DIR, buf.ptr, buf.len);
    return if (n > 1 and n <= buf.len) buf[0 .. n - 1] else null;
}

/// The marker of one build having served one plan to one subject under one OS build, in `dir` (which ends in a
/// slash).
pub fn markerPath(buf: []u8, dir: []const u8, build: []const u8, os: []const u8, s: Subject, kind: Kind) ?[:0]const u8 {
    var h = std.hash.Fnv1a_64.init();
    for ([_][]const u8{ build, os, s.model_path, s.model_type }) |part| {
        h.update(part);
        h.update("\x00");
    }
    h.update(std.mem.asBytes(&[_]u32{ s.verify_rows, s.kv_bits, FULL.len, LONGEST }));
    return std.fmt.bufPrintSentinel(buf, "{s}mlx-serve-warm/{x:0>16}.{s}", .{ dir, h.final(), @tagName(kind) }, 0) catch null;
}

/// The marker to write once this load has served the warm requests of `kind`, or null when it owes none: this build
/// already served them for this subject on this machine, or nothing here compiles Metal kernels.
pub fn owed(buf: []u8, kind: Kind, s: Subject) ?[:0]const u8 {
    if (comptime builtin.os.tag != .macos) return null;
    var dir_buf: [1024]u8 = undefined;
    const dir = cacheDir(&dir_buf) orelse return null;
    var os_buf: [64]u8 = undefined;
    var os_len: usize = os_buf.len;
    const os = if (std.c.sysctlbyname("kern.osversion", &os_buf, &os_len, null, 0) == 0) os_buf[0..os_len] else "";
    var exe_buf: [4096]u8 = undefined;
    var exe_len: u32 = exe_buf.len;
    const exe = if (std.c._NSGetExecutablePath(&exe_buf, &exe_len) == 0) std.mem.sliceTo(&exe_buf, 0) else "";
    // A build is its id and where it runs from: a copy under another bundle id has another shader cache.
    var build_buf: [4096 + 64]u8 = undefined;
    const build = std.fmt.bufPrint(&build_buf, "{s}\x00{s}", .{ round_cost.engineBuildId(), exe }) catch return null;
    return owedIn(buf, dir, build, os, kind, s);
}

/// The full plan holds the minimal one, so its marker answers for both.
fn owedIn(buf: []u8, dir: []const u8, build: []const u8, os: []const u8, kind: Kind, s: Subject) ?[:0]const u8 {
    const full = markerPath(buf, dir, build, os, s, .full) orelse return null;
    if (served(full)) return null;
    if (kind == .full) return full;
    const path = markerPath(buf, dir, build, os, s, .minimal) orelse return null;
    return if (served(path)) null else path;
}

fn served(path: [:0]const u8) bool {
    const fd = crt.openRead(path.ptr);
    if (fd < 0) return false;
    crt.close(fd);
    return true;
}

/// This build has served its warm requests for the marker's subject.
pub fn remember(path: [:0]const u8) void {
    var dir_buf: [1100]u8 = undefined;
    const dir = std.fmt.bufPrintSentinel(&dir_buf, "{s}", .{std.fs.path.dirname(path) orelse return}, 0) catch return;
    crt.mkdir(dir.ptr);
    const fd = crt.openWrite(path.ptr, true);
    if (fd >= 0) crt.close(fd);
}

/// Serve the full plan on the inference thread, inside the load. The requests run against a RAM twin of the model's
/// prefix cache and count in no metric, so the model comes up as if nothing had been asked of it. False when a
/// shutdown cut the plan short: the next load still owes it.
pub fn run(sch: *Scheduler, entry: *LoadedModel) bool {
    const cfg = entry.config orelse return true;
    log.info("First load of this build with this model: compiling its kernels before it reports ready.\n", .{});
    const sw = io_util.Stopwatch.init(sch.io);
    var twin = twinOf(entry);
    defer if (twin) |*t| t.deinit();

    var buf: [LONGEST]u32 = undefined;
    var done: usize = 0;
    var ladder_sw: ?io_util.Stopwatch = null;
    const cap = promptLimit(cfg.contextCap(), cfg.ctx_override, sch.gguf_ctx_size);
    for (FULL, 0..) |r, i| {
        // A fork runs if its rung did; the budget is asked before a rung.
        if (i >= MINIMAL.len and r.own == 0) {
            if (ladder_sw == null) ladder_sw = io_util.Stopwatch.init(sch.io);
            if (ladder_sw.?.read() > LADDER_BUDGET_NS) break;
        }
        if (cap != 0 and r.len + r.gen > cap) break;
        serve(sch, entry, if (twin) |*t| t else null, fill(&buf, cfg.vocab_size, r), r) catch |err| {
            if (err == error.Shutdown) return false;
            log.warn("Warm request {d} of {d} failed ({s}); the first prompts compile what is left.\n", .{ done + 1, FULL.len, @errorName(err) });
            break;
        };
        done += 1;
    }
    log.info("Warm requests complete ({d} of {d}, {d} ms).\n", .{ done, FULL.len, sw.read() / std.time.ns_per_ms });
    return true;
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

fn submit(sch: *Scheduler, entry: *LoadedModel, cache: ?*prefix_cache_mod.HotPrefixCache, ids: []const u32, r: Request) !*Slot {
    const cfg = entry.config.?;
    // The request-level defaults of `host/api/request_options.zig`; the engine picks one mode from them.
    const sidecar = entry.dflash != null or (entry.drafter != null and !cfg.isMoe() and !cfg.has_hybrid_layers);
    const slot = try scheduler_mod.submitNow(sch, .{
        .model = entry,
        .prefix_cache = cache,
        .prompt_ids = ids,
        .sampling = if (r.sampled) .{ .temperature = 0.7, .top_p = 0.95, .top_k = 20 } else .{ .temperature = 0 },
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
    marker_buf: [1280]u8 = undefined,
    marker_len: usize,
    verify_rows: u32,
    /// How many of `MINIMAL` it serves: none on a model that cannot chat.
    requests: u32,
    twin: ?prefix_cache_mod.HotPrefixCache,
    /// The sweep's widths done, then the requests ended.
    step: u32 = 0,
    slot: ?*Slot = null,
    asked: io_util.Stopwatch = undefined,
    /// Since a real request began to wait behind `slot`.
    waited_on: ?io_util.Stopwatch = null,
    spent_ns: u64 = 0,
};

/// `entry`, just loaded, owes the minimal warm-up; `marker` is written once it has served it. Inference thread.
pub fn owe(sch: *Scheduler, entry: *LoadedModel, marker: []const u8, verify_rows: u32, chat: bool) void {
    const w = sch.allocator.create(Idle) catch return;
    sch.warm_queue.append(sch.allocator, w) catch return sch.allocator.destroy(w);
    w.* = .{ .entry = entry, .marker_len = marker.len, .verify_rows = verify_rows, .requests = if (chat) MINIMAL.len else 0, .twin = twinOf(entry) };
    @memcpy(w.marker_buf[0..marker.len], marker);
    w.marker_buf[marker.len] = 0;
}

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
            log.warn("Warm request {d} of {d} {s}; the first prompts compile what is left.\n", .{ w.step - w.verify_rows + 1, w.requests, if (why == .failed) "failed" else "took too long" });
            end(sch, w, true);
        },
    }
}

/// Start the next step of `w`: nothing of its own is in flight and nothing else is asked of the server.
fn start(sch: *Scheduler, w: *Idle) void {
    switch (nextStep(w.step, w.verify_rows, w.requests, readyNow(sch, w.entry), false, true)) {
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
            const r = MINIMAL[i];
            const cap = promptLimit(cfg.contextCap(), cfg.ctx_override, sch.gguf_ctx_size);
            if (cap != 0 and r.len + r.gen > cap) return end(sch, w, true);
            var buf: [LONG + FORK_GROWTH[0]]u32 = undefined;
            sch.queue_mu.lockUncancelable(sch.io);
            sch.warm_in_flight = 1;
            sch.queue_mu.unlock(sch.io);
            w.slot = submit(sch, w.entry, if (w.twin) |*t| t else null, fill(&buf, cfg.vocab_size, r), r) catch |err| {
                log.warn("Warm request {d} of {d} failed ({s}); the first prompts compile what is left.\n", .{ i + 1, w.requests, @errorName(err) });
                return end(sch, w, true);
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

/// A warm-up is over: served (`mark` it, so no later load owes it) or cut short by its model leaving.
fn end(sch: *Scheduler, w: *Idle, mark: bool) void {
    endRequest(sch, w);
    if (w.twin) |*t| t.deinit();
    if (mark) {
        remember(w.marker_buf[0..w.marker_len :0]);
        log.info("Warm requests complete ({d} of {d}, {d} ms).\n", .{ @min(w.step -| w.verify_rows, w.requests), w.requests, w.spent_ns / std.time.ns_per_ms });
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

test "markerPath: one marker a build, an OS build and a subject" {
    const s: Subject = .{ .model_path = "/models/a", .model_type = "qwen3_5", .verify_rows = 17, .kv_bits = 0 };
    var bufs: [6][256]u8 = undefined;
    const base = markerPath(&bufs[0], "/cache/", "build-1", "27A1", s, .full).?;
    try testing.expect(std.mem.startsWith(u8, base, "/cache/mlx-serve-warm/"));
    try testing.expectEqualStrings(base, markerPath(&bufs[1], "/cache/", "build-1", "27A1", s, .full).?);
    var other = s;
    other.model_path = "/models/b";
    try testing.expect(!std.mem.eql(u8, base, markerPath(&bufs[2], "/cache/", "build-1", "27A1", other, .full).?));
    other = s;
    other.verify_rows = 9;
    try testing.expect(!std.mem.eql(u8, base, markerPath(&bufs[3], "/cache/", "build-1", "27A1", other, .full).?));
    try testing.expect(!std.mem.eql(u8, base, markerPath(&bufs[4], "/cache/", "build-2", "27A1", s, .full).?));
    try testing.expect(!std.mem.eql(u8, base, markerPath(&bufs[5], "/cache/", "build-1", "27B2", s, .full).?));
    try testing.expect(!std.mem.eql(u8, base, markerPath(&bufs[5], "/cache/", "build-1", "27A1", s, .minimal).?));
}

test "remember: a marker written is a load that owes nothing" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    const n = try tmp.dir.realPathFile(testing.io, ".", &dir_buf);
    dir_buf[n] = '/';
    var buf: [1024]u8 = undefined;
    const path = markerPath(&buf, dir_buf[0 .. n + 1], "build", "os", .{ .model_path = "/m", .model_type = "t", .verify_rows = 9, .kv_bits = 0 }, .full).?;
    try testing.expect(!served(path));
    remember(path);
    try testing.expect(served(path));
}

test "owed: the full plan's marker answers for the minimal one, the minimal one's for itself alone" {
    var tmps: [2]testing.TmpDir = .{ testing.tmpDir(.{}), testing.tmpDir(.{}) };
    defer for (&tmps) |*t| t.cleanup();
    const s: Subject = .{ .model_path = "/m", .model_type = "t", .verify_rows = 9, .kv_bits = 0 };
    var buf: [1024]u8 = undefined;
    for (tmps, [_]Kind{ .minimal, .full }) |tmp, first| {
        var dir_buf: [512]u8 = undefined;
        const n = try tmp.dir.realPathFile(testing.io, ".", &dir_buf);
        dir_buf[n] = '/';
        const dir = dir_buf[0 .. n + 1];
        try testing.expect(owedIn(&buf, dir, "build", "os", .full, s) != null);
        remember(owedIn(&buf, dir, "build", "os", first, s).?);
        try testing.expect(owedIn(&buf, dir, "build", "os", .minimal, s) == null);
        try testing.expectEqual(first == .minimal, owedIn(&buf, dir, "build", "os", .full, s) != null);
    }
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
    tmp: testing.TmpDir,
    reg: *@import("../../host/registry/registry.zig").ModelRegistry,
    sch: Scheduler = undefined,
    entry: *LoadedModel = undefined,
    marker_buf: [1024]u8 = undefined,
    marker: [:0]const u8 = undefined,

    fn init(f: *IdleFixture) !void {
        f.tmp = testing.tmpDir(.{});
        f.reg = try .init(testing.allocator, testing.io, null, 4, 0, null);
        f.entry = try f.reg.registerStub("a", "/a", 1);
        f.entry.state = .ready;
        f.entry.prefix_cache = null;
        var dir_buf: [512]u8 = undefined;
        const n = try f.tmp.dir.realPathFile(testing.io, ".", &dir_buf);
        dir_buf[n] = '/';
        f.marker = markerPath(&f.marker_buf, dir_buf[0 .. n + 1], "build", "os", .{ .model_path = "/a", .model_type = "t", .verify_rows = 0, .kv_bits = 0 }, .minimal).?;
        f.sch.allocator = testing.allocator;
        f.sch.io = testing.io;
        f.sch.registry = f.reg;
        f.sch.queue_mu = .init;
        f.sch.warm_queue = .empty;
        f.sch.warm_in_flight = 0;
        f.sch.in_flight = 0;
        inline for (.{ "offloaded", "load_queue", "unload_queue", "gen_queue" }) |name| @field(f.sch, name) = .empty;
        owe(&f.sch, f.entry, f.marker, 0, false);
    }

    fn deinit(f: *IdleFixture) void {
        stop(&f.sch);
        f.reg.deinit();
        f.tmp.cleanup();
    }
};

test "an idle warm-up waits beside a real request without holding its model, then marks the build" {
    var f: IdleFixture = .{ .tmp = undefined, .reg = undefined };
    try f.init();
    defer f.deinit();
    // Beside a real request it waits, and the thread may park: that request's end wakes it.
    f.sch.in_flight = 1;
    try testing.expect(!owesStepLocked(&f.sch));
    tick(&f.sch);
    try testing.expectEqual(@as(usize, 1), f.sch.warm_queue.items.len);
    try testing.expect(!served(f.marker));
    // Waiting, it pins nothing: a load that needs the room, or the idle sweep, can still unload the model.
    try testing.expectEqual(@as(u32, 0), f.entry.refcount.load(.acquire));
    try testing.expect(f.reg.pickLruEvictable("", false) == f.entry);
    f.sch.in_flight = 0;
    try testing.expect(owesStepLocked(&f.sch));
    tick(&f.sch);
    try testing.expectEqual(@as(usize, 0), f.sch.warm_queue.items.len);
    try testing.expect(served(f.marker));
    try testing.expect(!owesStepLocked(&f.sch));
}

test "forget: a model being unloaded takes its warm-up with it, unmarked, wherever it stands in the queue" {
    var f: IdleFixture = .{ .tmp = undefined, .reg = undefined };
    try f.init();
    defer f.deinit();
    const other = try f.reg.registerStub("b", "/b", 1);
    other.state = .ready;
    owe(&f.sch, other, f.marker, 0, false);
    forget(&f.sch, other);
    try testing.expectEqual(@as(usize, 1), f.sch.warm_queue.items.len);
    try testing.expect(f.sch.warm_queue.items[0].entry == f.entry);
    forget(&f.sch, f.entry);
    try testing.expectEqual(@as(usize, 0), f.sch.warm_queue.items.len);
    try testing.expect(!served(f.marker));
}

test "an idle warm-up starts nothing on a model that is leaving" {
    var f: IdleFixture = .{ .tmp = undefined, .reg = undefined };
    try f.init();
    defer f.deinit();
    f.entry.state = .evicting;
    tick(&f.sch);
    try testing.expectEqual(@as(usize, 0), f.sch.warm_queue.items.len);
    try testing.expect(!served(f.marker));
}

test "stop: a warm-up still owed at shutdown is dropped unmarked" {
    var f: IdleFixture = .{ .tmp = undefined, .reg = undefined };
    try f.init();
    defer f.deinit();
    stop(&f.sch);
    f.sch.warm_queue = .empty;
    try testing.expect(!served(f.marker));
}
