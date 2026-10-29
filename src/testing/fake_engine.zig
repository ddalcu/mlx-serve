//! A deterministic engine table with no model and no MLX, one per threading mode: a request's reply echoes its
//! prompt's ids, then counts up from 1000. A prefix cache keeps every prompt a request lets it keep (and an SSD tier
//! with `"ssd"` in the settings: one per process, `wipeDisk` clears it); a request drafts unless it asks for
//! `no_drafts`. The broken variants (`Flaw`) are the contract and exactness suites' planted controls.

const std = @import("std");
const abi = @import("../abi/engine.zig");
const contract = @import("../abi/contract_test.zig");
const exactness = @import("../abi/exactness_test.zig");
const conformance = @import("../sdk/testing/conformance.zig");

const io = std.testing.io;
const gpa = std.testing.allocator;

/// Declared on, so the suite checks them; every other request-shaped cap is off, so it checks the refusals.
pub const caps: abi.Caps = .{ .penalties = true, .logit_bias = true, .logprobs = true, .batches = true, .drafts = true, .prefix_cache = true };
pub const max_concurrent = 8;
pub const eos = [_]u32{2};
pub const first_count: u32 = 1000;

pub const Flaw = enum {
    none,
    double_finished,
    ignores_stop,
    overshoots_max_tokens,
    accepts_undeclared,
    locks_across_sink,
    drops_on_close,
    // exactness: a token moves when drafted / resumed / in company; a reference that is not plain or not cold
    drafted_drifts,
    resumed_drifts,
    concurrent_drifts,
    ignores_no_drafts,
    ignores_cold,
    forgets_on_close,
};

pub const own_thread = Fake(.own_thread, .none).engine;
pub const host_gpu_thread = Fake(.host_gpu_thread, .none).engine;

/// The fake's SSD tier: the prompts a closed `"ssd"` instance kept, read by the next one.
var disk: std.ArrayList([]u32) = .empty;
var disk_mu: std.Io.Mutex = .init;

pub fn wipeDisk() void {
    for (disk.items) |p| gpa.free(p);
    disk.clearAndFree(gpa);
}

/// The k-th token of a reply to `prompt`.
pub fn tokenAt(prompt: []const u32, k: u64) u32 {
    return if (k < prompt.len) prompt[@intCast(k)] else first_count + @as(u32, @intCast(k - prompt.len));
}

pub fn Fake(comptime threading: abi.Threading, comptime flaw: Flaw) type {
    return struct {
        pub const engine: abi.Engine = .{
            .probe = probe,
            .load_bytes = loadBytes,
            .open = open,
            .close = close,
            .submit = submit,
            .cancel = cancel,
            .step = if (threading == .host_gpu_thread) step else null,
            .status = status,
            .memory = memory,
            .report_json = null,
            .trim = null,
            .media_rows = null,
            .embed = null,
            .decide = null,
            .tokenize = null,
            .token_bytes = null,
            .chat_template = null,
            .render = null,
        };

        /// Tokens one round commits: rounds of several tokens exercise the `max_tokens` cap mid-round.
        const round_len: u32 = if (threading == .host_gpu_thread) 3 else 2;
        const max_top = 4;

        const Req = struct {
            id: u64,
            prompt: []u32,
            eos: []u32,
            max_tokens: u32,
            logprobs: u8,
            drafts: bool,
            cached: u32 = 0,
            sink: abi.Sink,
            cancelled: std.atomic.Value(bool) = .init(false),
            // Engine thread only:
            emitted: u64 = 0,
            prefilled: bool = false,
            finished: bool = false,

            fn finish(r: *Req, reason: @FieldType(abi.Event, "reason")) void {
                const stats: abi.Stats = .{ .rounds = r.emitted, .drafted = if (r.drafts) r.emitted else 0 };
                const ev: abi.Event = .{ .kind = .finished, .reason = reason, .stats = &stats };
                _ = r.sink.event(r.sink.ctx, r.id, &ev);
                if (flaw == .double_finished) _ = r.sink.event(r.sink.ctx, r.id, &ev);
                r.finished = true;
            }
        };

        const State = struct {
            mu: std.Io.Mutex = .init,
            cond: std.Io.Condition = .init,
            reqs: std.ArrayList(*Req) = .empty,
            kept: std.ArrayList([]u32) = .empty,
            ssd: bool = false,
            closing: bool = false,
            peak: u64 = 0,
            worker: ?std.Thread = null,

            fn bytes(st: *const State) u64 {
                var n: u64 = 0;
                for (st.reqs.items) |r| n += @sizeOf(Req) + 4 * (r.prompt.len + r.eos.len);
                return n;
            }
        };

        fn probe(_: abi.Str, config_json: abi.Str, out: *abi.Verdict) callconv(.c) bool {
            if (std.mem.indexOf(u8, abi.items(u8, config_json), "\"fake\"") != null) return true;
            out.* = .{ .code = .unsupported, .why = abi.span(u8, "not a fake model: config.json names another model_type") };
            return false;
        }

        fn loadBytes(_: *const abi.Open, bytes: *u64) callconv(.c) abi.Verdict {
            bytes.* = 1 << 20;
            return .{};
        }

        fn open(o: *const abi.Open, info: *abi.Info, out: *abi.Verdict) callconv(.c) ?*anyopaque {
            const st = gpa.create(State) catch {
                out.* = .{ .code = .failed, .why = abi.span(u8, "out of memory") };
                return null;
            };
            st.* = .{ .ssd = std.mem.indexOf(u8, abi.items(u8, o.settings_json), "\"ssd\"") != null };
            if (st.ssd) {
                disk_mu.lockUncancelable(io);
                defer disk_mu.unlock(io);
                for (disk.items) |p| keep(st, p) catch {};
            }
            if (threading == .own_thread) {
                st.worker = std.Thread.spawn(.{}, work, .{st}) catch {
                    gpa.destroy(st);
                    out.* = .{ .code = .failed, .why = abi.span(u8, "could not start the engine thread") };
                    return null;
                };
            }
            info.* = .{
                .name = abi.span(u8, "fake"),
                .version = abi.span(u8, "1"),
                .caps = caps,
                .threading = threading,
                .context_window = 4096,
                .max_concurrent = max_concurrent,
                .embedding_dims = 0,
                .eos = abi.span(u32, &eos),
                .startup = abi.span(u8, "fake engine: no model, no GPU"),
            };
            return st;
        }

        fn close(e: *anyopaque) callconv(.c) void {
            const st: *State = @ptrCast(@alignCast(e));
            st.mu.lockUncancelable(io);
            st.closing = true;
            if (flaw != .drops_on_close) for (st.reqs.items) |r| r.cancelled.store(true, .release);
            st.cond.broadcast(io);
            st.mu.unlock(io);
            if (st.worker) |t| t.join() else while (step(st, 0)) {}
            for (st.reqs.items) |r| free(r); // only `drops_on_close` leaves any
            st.reqs.deinit(gpa);
            if (st.ssd and flaw != .forgets_on_close) {
                disk_mu.lockUncancelable(io);
                defer disk_mu.unlock(io);
                wipeDisk();
                disk = st.kept;
            } else {
                for (st.kept.items) |p| gpa.free(p);
                st.kept.deinit(gpa);
            }
            gpa.destroy(st);
        }

        fn submit(e: *anyopaque, id: u64, r: *const abi.Request, sink: abi.Sink) callconv(.c) abi.Verdict {
            const st: *State = @ptrCast(@alignCast(e));
            if (flaw != .accepts_undeclared) if (abi.refusal(caps, abi.needs(r))) |v| return v;
            const req = gpa.create(Req) catch return .{ .code = .failed, .why = abi.span(u8, "out of memory") };
            const drafts = !r.no_drafts or flaw == .ignores_no_drafts;
            req.* = .{ .id = id, .prompt = &.{}, .eos = &.{}, .max_tokens = r.max_tokens, .logprobs = r.logprobs, .drafts = drafts, .sink = sink };
            req.prompt = gpa.dupe(u32, abi.items(u32, r.prompt)) catch return refuseOom(req);
            req.eos = gpa.dupe(u32, abi.items(u32, r.eos)) catch return refuseOom(req);
            st.mu.lockUncancelable(io);
            defer st.mu.unlock(io);
            if (st.reqs.items.len >= max_concurrent) {
                free(req);
                return .{ .code = .busy, .why = abi.span(u8, "the fake engine runs 8 requests at most") };
            }
            st.reqs.append(gpa, req) catch return refuseOom(req);
            st.peak = @max(st.peak, st.bytes());
            // A restore always leaves the last prompt token to forward.
            if (!r.cache.cold or flaw == .ignores_cold) for (st.kept.items) |p| {
                req.cached = @max(req.cached, @as(u32, @intCast(@min(prefixLen(p, req.prompt), req.prompt.len -| 1))));
            };
            if (r.cache.keep) keep(st, req.prompt) catch {};
            st.cond.broadcast(io);
            return .{};
        }

        fn keep(st: *State, prompt: []const u32) !void {
            const p = try gpa.dupe(u32, prompt);
            errdefer gpa.free(p);
            try st.kept.append(gpa, p);
        }

        fn prefixLen(a: []const u32, b: []const u32) usize {
            return std.mem.indexOfDiff(u32, a, b) orelse a.len;
        }

        fn refuseOom(req: *Req) abi.Verdict {
            free(req);
            return .{ .code = .failed, .why = abi.span(u8, "out of memory") };
        }

        fn free(r: *Req) void {
            gpa.free(r.prompt);
            gpa.free(r.eos);
            gpa.destroy(r);
        }

        fn cancel(e: *anyopaque, id: u64) callconv(.c) void {
            const st: *State = @ptrCast(@alignCast(e));
            st.mu.lockUncancelable(io);
            defer st.mu.unlock(io);
            for (st.reqs.items) |r| if (r.id == id) r.cancelled.store(true, .release);
        }

        fn status(e: *anyopaque, out: *abi.Status) callconv(.c) void {
            const st: *State = @ptrCast(@alignCast(e));
            st.mu.lockUncancelable(io);
            defer st.mu.unlock(io);
            var running: u32 = 0;
            for (st.reqs.items) |r| running += @intFromBool(@atomicLoad(bool, &r.prefilled, .monotonic));
            out.* = .{ .running = running, .waiting = @as(u32, @intCast(st.reqs.items.len)) - running, .decode_tps = 100, .prefill_tps = 1000 };
        }

        fn memory(e: *anyopaque, out: *abi.Memory) callconv(.c) void {
            const st: *State = @ptrCast(@alignCast(e));
            st.mu.lockUncancelable(io);
            defer st.mu.unlock(io);
            out.* = .{ .active = st.bytes(), .cache = 0, .peak = st.peak };
        }

        /// One round for every live request; false when there was none. The host's GPU thread calls it, and
        /// so does `close` for this mode; the own-thread mode's worker loops on it.
        fn step(e: *anyopaque, _: u64) callconv(.c) bool {
            const st: *State = @ptrCast(@alignCast(e));
            var live: [max_concurrent]*Req = undefined;
            st.mu.lockUncancelable(io);
            if (flaw == .drops_on_close and st.closing) {
                st.mu.unlock(io);
                return false;
            }
            const n = st.reqs.items.len;
            @memcpy(live[0..n], st.reqs.items);
            if (flaw != .locks_across_sink) st.mu.unlock(io);
            for (live[0..n]) |r| round(r, n);
            if (flaw == .locks_across_sink) st.mu.unlock(io);
            st.mu.lockUncancelable(io);
            defer st.mu.unlock(io);
            var i: usize = 0;
            while (i < st.reqs.items.len) {
                const r = st.reqs.items[i];
                if (r.finished) {
                    _ = st.reqs.swapRemove(i);
                    free(r);
                } else i += 1;
            }
            return n > 0;
        }

        fn work(st: *State) void {
            while (true) {
                st.mu.lockUncancelable(io);
                while (st.reqs.items.len == 0 and !st.closing) st.cond.waitUncancelable(io, &st.mu);
                const idle = st.reqs.items.len == 0;
                const exit = st.closing and (idle or flaw == .drops_on_close);
                st.mu.unlock(io);
                if (exit) return;
                _ = step(st, 0);
            }
        }

        fn round(r: *Req, crowd: usize) void {
            if (r.cancelled.load(.acquire)) return r.finish(.cancelled);
            if (!r.prefilled) {
                @atomicStore(bool, &r.prefilled, true, .monotonic);
                const ev: abi.Event = .{ .kind = .prefilled, .cached = r.cached };
                if (r.sink.event(r.sink.ctx, r.id, &ev) == .stop) r.finish(.stop);
                return;
            }
            var toks: [round_len]u32 = undefined;
            var lps: [round_len]abi.Logprob = undefined;
            var tops: [round_len][max_top]abi.Bias = undefined;
            const room = r.max_tokens - @min(r.max_tokens, r.emitted);
            const want: u64 = if (flaw == .overshoots_max_tokens) round_len else @min(round_len, room);
            var n: usize = 0;
            var hit_eos = false;
            while (n < want) : (n += 1) {
                const at = r.emitted + n;
                const drift = at == r.prompt.len + 1 and switch (flaw) {
                    .drafted_drifts => r.drafts,
                    .resumed_drifts => r.cached > 0,
                    .concurrent_drifts => crowd > 1,
                    else => false,
                };
                const t = tokenAt(r.prompt, at) + @intFromBool(drift);
                if (std.mem.indexOfScalar(u32, r.eos, t) != null) {
                    hit_eos = true;
                    break;
                }
                toks[n] = t;
                const k = @min(r.logprobs, max_top);
                for (tops[n][0..k], 0..) |*b, j| b.* = .{ .token = t + @as(u32, @intCast(j)), .bias = -0.25 * @as(f32, @floatFromInt(j + 1)) };
                lps[n] = .{ .token = t, .logprob = -0.25, .top = abi.span(abi.Bias, tops[n][0..k]) };
            }
            var action: abi.Action = .go;
            if (n > 0) {
                const ev: abi.Event = .{
                    .kind = .tokens,
                    .tokens = abi.span(u32, toks[0..n]),
                    .logprobs = if (r.logprobs > 0) abi.span(abi.Logprob, lps[0..n]) else .{},
                };
                action = r.sink.event(r.sink.ctx, r.id, &ev);
                r.emitted += n;
            }
            if (hit_eos or (action == .stop and flaw != .ignores_stop)) return r.finish(.stop);
            if (r.emitted >= r.max_tokens) return r.finish(.length);
        }
    };
}

// ── the contract suite, against both fakes and the planted controls

const fixture: contract.Fixture = .{
    .model_dir = "/fake",
    .model_type = "fake",
    .config_json = "{\"model_type\":\"fake\"}",
    .prompt = &.{ 5, 6, 7 },
    .timeout_ms = 5_000,
    .gate_ms = 300,
};

test "fake engine: the own-thread table passes the contract" {
    try contract.run(&own_thread, fixture);
}

test "fake engine: the host-GPU-thread table passes the contract" {
    try contract.run(&host_gpu_thread, fixture);
}

test "fake engine: the contract catches every planted flaw, by name, in both threading modes" {
    const cases = .{
        .{ Flaw.double_finished, error.EventAfterFinished },
        .{ Flaw.ignores_stop, error.StopIgnored },
        .{ Flaw.overshoots_max_tokens, error.MaxTokensExceeded },
        .{ Flaw.accepts_undeclared, error.UndeclaredCapAccepted },
        .{ Flaw.locks_across_sink, error.StatusBlocked },
        .{ Flaw.drops_on_close, error.CloseDroppedFinished },
    };
    inline for (.{ abi.Threading.own_thread, abi.Threading.host_gpu_thread }) |mode| {
        inline for (cases) |c| {
            const broken = Fake(mode, c[0]).engine;
            try std.testing.expectError(c[1], contract.run(&broken, fixture));
        }
    }
}

// ── the exactness contracts, against both fakes and their planted controls

const exact_fixture: exactness.Fixture = .{
    .model_dir = "/fake",
    .model_type = "fake",
    .config_json = "{\"model_type\":\"fake\"}",
    .settings_json = "{\"ssd\":true}",
    .prompts = &.{ &.{ 5, 6, 7 }, &.{ 8, 9 }, &.{ 5, 6, 11, 12 } },
    .ssd = true,
    .timeout_ms = 5_000,
    .gate_ms = 300,
};

const all: exactness.Contracts = .{ .drafted = true, .resumed = true, .concurrent = true };

test "fake engine: both tables hold the three exactness contracts" {
    defer wipeDisk();
    try exactness.expect(&own_thread, exact_fixture, all);
    try exactness.expect(&host_gpu_thread, exact_fixture, all);
}

test "fake engine: the exactness check catches every planted break, by name, in both threading modes" {
    defer wipeDisk();
    const cases = .{
        .{ Flaw.drafted_drifts, error.DraftedDiffers },
        .{ Flaw.resumed_drifts, error.ResumedDiffers },
        .{ Flaw.concurrent_drifts, error.ConcurrentDiffers },
        .{ Flaw.ignores_no_drafts, error.PlainDrafted },
        .{ Flaw.ignores_cold, error.UnexpectedCacheHit },
        .{ Flaw.forgets_on_close, error.NotResumedFromDisk },
    };
    inline for (.{ abi.Threading.own_thread, abi.Threading.host_gpu_thread }) |mode| {
        inline for (cases) |c| {
            wipeDisk();
            const broken = Fake(mode, c[0]).engine;
            try std.testing.expectError(c[1], exactness.expect(&broken, exact_fixture, all));
        }
    }
}

test "conformance: only the contracts a plugin declares run on the engine serving it" {
    defer wipeDisk();
    const drifts = Fake(.host_gpu_thread, .concurrent_drifts).engine;
    try conformance.expectExactness(&drifts, exact_fixture, .{ .exact_drafted = true, .exact_resumed = true });
    try std.testing.expectError(error.ConcurrentDiffers, conformance.expectExactness(&drifts, exact_fixture, .{ .exact_concurrent = true }));
}

test "fake engine: a reply echoes the prompt, then counts" {
    try std.testing.expectEqual(@as(u32, 6), tokenAt(&.{ 5, 6 }, 1));
    try std.testing.expectEqual(@as(u32, 1001), tokenAt(&.{ 5, 6 }, 3));
}

test {
    _ = abi;
    _ = contract;
    _ = exactness;
}
