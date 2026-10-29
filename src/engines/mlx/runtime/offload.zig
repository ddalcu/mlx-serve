//! A slot served by an engine beside MLX: its request handed to the table, and the engine's events written back into
//! the slot from the engine's thread (`Sink.event`), as the inference thread writes an MLX slot.

const std = @import("std");
const abi = @import("../../../abi/engine.zig");
const log = @import("../../../sdk/util/log.zig");
const served_mod = @import("../../../host/engines/served.zig");
const Served = served_mod.Served;
const Kind = served_mod.Kind;
const refusal = @import("../../../host/engines/refusal.zig");
const request_mod = @import("../../../sdk/request/request.zig");
const finish = @import("../finish.zig");
const Slot = @import("../slot.zig").Slot;
const SubmitParams = @import("../scheduler/params.zig").SubmitParams;

/// What the request asks that the engine does not declare (`abi.Caps` field name), or null when it serves it.
pub fn missingCap(s: *const Served, p: SubmitParams) ?[]const u8 {
    return abi.missing(s.caps(), without(needs(p.sampling, p.logprobs_n, p.prompt_logprobs), ignored(s.kind)));
}

/// What a GGUF engine has always accepted and ignored: a grammar (`response_format`). Its request is served
/// unconstrained, named in one log line, never refused.
fn ignored(kind: Kind) abi.Caps {
    return .{ .grammar = kind == .ds4 or kind == .llama };
}

fn without(a: abi.Caps, b: abi.Caps) abi.Caps {
    return @bitCast(@as(u64, @bitCast(a)) & ~@as(u64, @bitCast(b)));
}

fn needs(sampling: request_mod.SamplingParams, logprobs_n: u32, prompt_logprobs: bool) abi.Caps {
    return .{
        .penalties = sampling.penalized(),
        .logprobs = logprobs_n > 0,
        .grammar = sampling.constraint != null,
        .think_budget = sampling.think_bound != null,
        .tool_gate = sampling.tool_force != null,
        .prompt_logprobs = prompt_logprobs,
    };
}

/// The omitted `min_p` per engine: ds4's sampler has always drawn with 0.05.
fn defaultMinP(kind: Kind) f32 {
    return if (kind == .ds4) 0.05 else 0;
}

/// Hands `slot` to its model's engine; the caller holds the scheduler's `queue_mu` and has put the slot on its
/// `offloaded` list. The slot's `in_pass` is held from here to its `finished` event.
pub fn submit(slot: *Slot, s: *Served) !void {
    const sp = slot.sampling;
    // An engine that owns its vocabulary owns its stop set; the request's own EOS joins it.
    var eos: std.ArrayList(u32) = .empty;
    defer eos.deinit(slot.allocator);
    try eos.appendSlice(slot.allocator, slot.eos_token_ids);
    if (s.caps().owns_tokenizer) try eos.appendSlice(slot.allocator, abi.items(u32, s.info.eos));
    const grammar: abi.Grammar = .{ .kind = .json_schema, .text = .{} };
    const tool_gate: abi.ToolGate = .{ .opener = .{}, .names = .{} };
    const think: abi.Think = .{ .budget = 0, .open = .{}, .opened_in_prompt = false, .close = .{}, .end = abi.no_token };
    const ask_grammar = sp.constraint != null and !ignored(s.kind).grammar;
    if (sp.constraint != null and !ask_grammar) log.info("[{s}] response_format ignored: the engine has no grammar sampler\n", .{@tagName(s.kind)});
    const r: abi.Request = .{
        .prompt = abi.span(u32, slot.full_prompt),
        .max_tokens = slot.max_tokens,
        .sampling = .{
            .temperature = sp.temperature,
            .top_p = sp.top_p,
            .top_k = sp.top_k,
            .min_p = sp.min_p orelse defaultMinP(s.kind),
            .seed = sp.seed orelse 0,
            .seeded = sp.seed != null,
            .repeat_penalty = sp.repeat_penalty,
            .frequency_penalty = sp.frequency_penalty,
            .presence_penalty = sp.presence_penalty,
        },
        .eos = abi.span(u32, eos.items),
        .logprobs = @intCast(@min(slot.logprobs_n, std.math.maxInt(u8))),
        .grammar = if (ask_grammar) &grammar else null,
        .think = if (sp.think_bound != null) &think else null,
        .tool_gate = if (sp.tool_force != null) &tool_gate else null,
        .prompt_logprobs = slot.prompt_logprobs,
    };
    _ = slot.in_pass.fetchAdd(1, .acq_rel);
    _ = s.gauges.prefilling.fetchAdd(1, .monotonic);
    const v = s.table.submit(s.handle, id(slot), &r, .{ .ctx = slot, .event = onEvent });
    if (v.code == .ok) return;
    _ = s.gauges.prefilling.fetchSub(1, .monotonic);
    _ = slot.in_pass.fetchSub(1, .release);
    log.err("[{s}] request refused: {s}\n", .{ @tagName(s.kind), abi.items(u8, v.why) });
    return if (v.code == .unsupported) error.EngineUnsupported else error.EngineBusy;
}

/// Asks the engine to end `slot` (a queued one finishes alone, a running one after its round).
pub fn cancel(slot: *Slot) void {
    const s = slot.model.engine orelse return;
    s.table.cancel(s.handle, id(slot));
}

fn id(slot: *const Slot) u64 {
    return @intFromPtr(slot);
}

/// The engine's thread: write the event into the slot. Never takes `queue_mu`.
fn onEvent(ctx: *anyopaque, _: u64, ev: *const abi.Event) callconv(.c) abi.Action {
    const slot: *Slot = @ptrCast(@alignCast(ctx));
    const s = slot.model.engine.?;
    switch (ev.kind) {
        .prefilled => {
            _ = s.gauges.prefilling.fetchSub(1, .monotonic);
            slot.prompt_tokens = @intCast(slot.full_prompt.len);
            slot.cached_tokens = ev.cached;
            slot.first_token_ns = @intCast(slot.request_start_ts.untilNow(slot.io, .boot).nanoseconds);
            slot.state = .decoding;
            slot.emitPrefilled();
        },
        .audio => slot.emitAudio(abi.items(u32, ev.codes), abi.items(f32, ev.samples), ev.sample_rate),
        .tokens => {
            const lps = abi.items(abi.Logprob, ev.logprobs);
            for (abi.items(u32, ev.tokens), 0..) |t, i| {
                slot.pushTokenWithLogprob(t, if (i < lps.len) logprob(slot.allocator, lps[i]) else null);
                if (t != 0) slot.was_pad_only = false;
                slot.completion_tokens += 1;
            }
            _ = s.gauges.generated.fetchAdd(ev.tokens.len, .monotonic);
        },
        .finished => {
            finished(s, slot, ev);
            return .go;
        },
    }
    return if (slot.cancelled.load(.acquire)) .stop else .go;
}

/// A logprob entry as the MLX path publishes it; its top list is the slot's to free.
fn logprob(gpa: std.mem.Allocator, lp: abi.Logprob) ?request_mod.LogprobResult {
    const top = abi.items(abi.Bias, lp.top);
    const out = gpa.alloc(@import("../sampling/logprobs.zig").TokenLogprob, top.len) catch return null;
    for (top, out) |b, *o| o.* = .{ .token_id = b.token, .logprob = b.bias };
    return .{ .token_logprob = lp.logprob, .top_logprobs = out };
}

/// The slot's terminator and its metrics; the slot's `in_pass` goes last, after which the slot may be freed.
fn finished(s: *Served, slot: *Slot, ev: *const abi.Event) void {
    defer _ = slot.in_pass.fetchSub(1, .release);
    if (slot.state != .decoding) _ = s.gauges.prefilling.fetchSub(1, .monotonic);
    if (ev.stats) |st| {
        slot.prefill_ns = st.prefill_ns;
        slot.decode_ns = st.decode_ns;
        logSpec(s.kind, st);
    }
    _ = s.gauges.generated.fetchSub(slot.completion_tokens, .monotonic);
    if (ev.reason == .failed) {
        const why = abi.items(u8, ev.verdict.why);
        log.err("[{s}] request failed: {s}\n", .{ @tagName(s.kind), why });
        finish.recordSlotEnd(s.metrics, slot, .failed);
        slot.markError(why);
        return;
    }
    const reason: []const u8 = if (slot.cancelled.load(.acquire)) "cancelled" else switch (ev.reason) {
        .stop => "stop",
        .length => "length",
        .cancelled, .failed => "cancelled",
    };
    finish.recordSlotEnd(s.metrics, slot, finish.finishOutcome(reason, null));
    finish.publishSlotTerminator(slot, reason, null);
}

/// The speculative summary each engine has always logged at a request's end.
fn logSpec(kind: Kind, st: *const abi.Stats) void {
    if (st.rounds == 0) return;
    switch (kind) {
        .llama => log.info("  [spec-stats] mode=llama-mtp attempts={d} accepts={d} drafted={d} per_draft_pct={d:.1}%\n", .{
            st.rounds, st.accepted, st.drafted, 100.0 * @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(@max(st.drafted, 1))),
        }),
        .ane => log.info("[ane] spec: {d} cycles, {d}% of drafts accepted, {d:.2} tokens per call\n", .{
            st.rounds, 100 * st.accepted / @max(st.drafted, 1), @as(f64, @floatFromInt(st.accepted + st.rounds)) / @as(f64, @floatFromInt(st.rounds)),
        }),
        .ds4, .mlx => {},
    }
}

// ── tests ──

const testing = std.testing;

test "missingCap: plain sampling is served; a grammar, a penalty, logprobs, a forced tool call or prompt logprobs names its cap" {
    const plain: request_mod.SamplingParams = .{ .temperature = 0.7, .top_p = 0.9, .top_k = 20 };
    try testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(needs(plain, 0, false))));
    try testing.expectEqualStrings("penalties", abi.missing(.{}, needs(.{ .repeat_penalty = 1.1 }, 0, false)).?);
    try testing.expectEqualStrings("grammar", abi.missing(.{}, needs(.{ .constraint = @ptrFromInt(8) }, 0, false)).?);
    try testing.expectEqualStrings("tool_gate", abi.missing(.{}, needs(.{ .tool_force = @ptrFromInt(8) }, 0, false)).?);
    try testing.expectEqualStrings("logprobs", abi.missing(.{ .penalties = true }, needs(.{}, 5, false)).?);
    try testing.expectEqualStrings("prompt_logprobs", abi.missing(.{ .logprobs = true }, needs(.{}, 0, true)).?);
    try testing.expect(abi.missing(.{ .logprobs = true, .penalties = true }, needs(.{ .presence_penalty = 0.5 }, 2, false)) == null);
}

test "missingCap: a GGUF engine accepts a grammar and ignores it, as it always has; the Neural Engine refuses it" {
    const grammar: request_mod.SamplingParams = .{ .constraint = @ptrFromInt(8) };
    var s: Served = undefined;
    s.info.caps = .{ .owns_tokenizer = true, .penalties = true, .logprobs = true };
    inline for (.{ .ds4, .llama }) |kind| {
        s.kind = kind;
        try testing.expect(missingCap(&s, .{ .prompt_ids = &.{1}, .sampling = grammar, .eos_token_ids = &.{}, .max_tokens = 1, .model = undefined }) == null);
    }
    s.kind = .ane;
    try testing.expectEqualStrings("grammar", missingCap(&s, .{ .prompt_ids = &.{1}, .sampling = grammar, .eos_token_ids = &.{}, .max_tokens = 1, .model = undefined }).?);
    // Penalties and logprobs are served where declared: no 400 for what 26.10.3 answered.
    s.kind = .llama;
    try testing.expect(missingCap(&s, .{ .prompt_ids = &.{1}, .sampling = .{ .presence_penalty = 0.5, .frequency_penalty = 0.2 }, .eos_token_ids = &.{}, .max_tokens = 1, .logprobs_n = 2, .model = undefined }) == null);
}

test "defaultMinP: ds4 keeps its own 0.05 floor; the others draw without one" {
    try testing.expectEqual(@as(f32, 0.05), defaultMinP(.ds4));
    try testing.expectEqual(@as(f32, 0), defaultMinP(.llama));
    try testing.expectEqual(@as(f32, 0), defaultMinP(.ane));
}

test "submit: the engine's events land in the slot from its thread — prefilled, the tokens, then the terminator" {
    const fake = @import("../../../testing/fake_engine.zig");
    const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
    const LoadedModel = @import("../../../host/registry/registry.zig").LoadedModel;
    const Quiet = struct {
        fn log(_: *anyopaque, _: u8, _: abi.Str) callconv(.c) void {}
        fn wake(_: *anyopaque) callconv(.c) void {}
    };
    var progress: @import("../../../host/registry/load_progress.zig").Progress = .{};
    var gauges: served_mod.Gauges = .{};
    var s: Served = .{ .gpa = testing.allocator, .kind = .llama, .table = &fake.own_thread, .handle = undefined, .info = undefined, .host = undefined, .progress = &progress, .gauges = &gauges, .metrics = null };
    s.host = .{ .ctx = &s, .log = Quiet.log, .wake = Quiet.wake };
    const o: abi.Open = .{ .model_dir = .{}, .model_type = .{}, .config_json = .{}, .context = 0, .max_concurrent = 1, .memory_budget = 0, .settings_json = .{}, .options_json = .{}, .host = &s.host };
    var v: abi.Verdict = .{};
    s.handle = fake.own_thread.open(&o, &s.info, &v) orelse return error.Open;
    defer fake.own_thread.close(s.handle);

    var lm: LoadedModel = undefined;
    lm.engine = &s;
    lm.transformer = null;
    var cfg: ModelConfig = .{};
    const slot = try Slot.init(testing.allocator, testing.io, &cfg, .{ .model = &lm, .prompt_ids = &.{ 5, 6, 7 }, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 5 }, .dense);
    defer slot.deinit();
    try submit(slot, &s);
    var got: [8]u32 = undefined;
    var n: usize = 0;
    while (true) switch (slot.waitNext()) {
        .token => |t| {
            got[n] = t;
            n += 1;
        },
        .done => break,
        .err => return error.SlotFailed,
    };
    // The fake echoes the prompt, then counts from 1000; `max_tokens` cuts it.
    try testing.expectEqualSlices(u32, &.{ 5, 6, 7, 1000, 1001 }, got[0..n]);
    try testing.expectEqualStrings("length", slot.finish_reason);
    try testing.expectEqual(@as(u32, 3), slot.prompt_tokens);
    try testing.expectEqual(@as(u32, 5), slot.completion_tokens);
    while (slot.in_pass.load(.acquire) != 0) testing.io.sleep(.fromMilliseconds(1), .awake) catch {};
    try testing.expectEqual(@as(u64, 0), gauges.prefilling.load(.monotonic) + gauges.generated.load(.monotonic));
}
