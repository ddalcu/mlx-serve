//! The one interface between the host and every engine (ours included). Extern layouts, C calling convention.
//!
//! Rules: slices cross as pointer and length (UTF-8, never NUL-terminated); memory is freed by whoever allocated
//! it; a struct passed on its own starts with `size` and grows only at the end; callbacks run on the engine's
//! thread, return at once and never call back into the engine; every fallible call returns a `Verdict` whose
//! `why` is static text; `Caps` describe the opened model, never a static table.

const std = @import("std");

pub const version: u32 = 1;

const common = @import("common.zig");
pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;

/// What this opened model does beyond the required core (sampling, seed, EOS, cancel). Returned by `open`; the
/// host refuses by name a request that needs what is not declared.
pub const Caps = packed struct(u64) {
    penalties: bool = false,
    logit_bias: bool = false,
    logprobs: bool = false,
    grammar: bool = false, // JSON schema, JSON object, regex, choice, EBNF
    think_budget: bool = false, // closes a thought at its budget, at decode
    tool_gate: bool = false, // tool_choice "required" or a named function, enforced at decode
    images: bool = false,
    video: bool = false,
    audio: bool = false,
    embed: bool = false,
    decide: bool = false, // typed decisions (laya, kev)
    owns_tokenizer: bool = false, // a GGUF engine that carries its own vocabulary
    owns_template: bool = false, // ... and renders its own chat prompt
    batches: bool = false, // reported in /props and /v1/models
    drafts: bool = false,
    prefix_cache: bool = false,
    distributed: bool = false, // reserved: a model served across several Macs
    prompt_logprobs: bool = false, // logprobs of the prompt's own tokens
    _: u46 = 0,
};

/// MLX engines run on the host's one GPU thread (one MLX caller per process); engines with their own GPU runtime
/// run their own thread.
pub const Threading = enum(u8) { own_thread, host_gpu_thread };

pub const Host = extern struct {
    size: u32 = @sizeOf(Host),
    ctx: *anyopaque,
    log: *const fn (ctx: *anyopaque, level: u8, msg: Str) callconv(.c) void,
    wake: *const fn (ctx: *anyopaque) callconv(.c) void, // a host_gpu_thread engine has work without a submit
    progress: ?*const fn (ctx: *anyopaque, what: Progress, n: i64) callconv(.c) void = null,
};

/// What the host shows while it waits on an engine, reported as deltas from any of the engine's threads.
pub const Progress = enum(u8) {
    load_total, // `open`: units of work the load does (a Neural Engine build: its programs)
    load_done,
    load_slow, // a unit missed its cache: this load takes minutes (a first compile)
    prefill_todo, // prompt tokens the engine's requests will feed; taken back when a prefill ends
    prefill_done,
};

pub const Open = extern struct {
    size: u32 = @sizeOf(Open),
    model_dir: Str,
    model_type: Str,
    config_json: Str, // config.json, read once by the host
    context: u32, // the model's ctx_size; 0 = the engine's own
    max_concurrent: u32,
    memory_budget: u64, // what the engine may hold: GPU ceiling less memory in use, OS reserve out
    settings_json: Str, // launch flags + Model Settings that shape a load (kv_quant, mtp, drafter, caches, cluster)
    options_json: Str, // Model Settings `engine_options`, verbatim
    host: *const Host, // lives until `close`
};

pub const Info = extern struct {
    size: u32 = @sizeOf(Info),
    name: Str,
    version: Str,
    caps: Caps,
    threading: Threading,
    context_window: u32, // what the host advertises: prompt + reply
    max_concurrent: u32,
    embedding_dims: u32,
    eos: Span(u32), // the engine's stop set; the host merges generation_config's
    startup: Str, // one line the host logs at load (the memory plan, the cluster)
    vocab_size: u32 = 0, // Caps.owns_tokenizer: the ids a prompt may carry
};

pub const Sampling = extern struct {
    temperature: f32 = 1,
    top_p: f32 = 1,
    top_k: u32 = 0,
    min_p: f32 = 0,
    seed: u64 = 0,
    seeded: bool = false,
    repeat_penalty: f32 = 1, // Caps.penalties
    frequency_penalty: f32 = 0,
    presence_penalty: f32 = 0,
    logit_bias: Span(Bias) = .{}, // Caps.logit_bias
};
pub const Bias = extern struct { token: u32, bias: f32 };

pub const Grammar = extern struct {
    size: u32 = @sizeOf(Grammar),
    kind: enum(u8) { json_object, json_schema, regex, choice, ebnf },
    text: Str,
    after: u32 = no_token, // with thinking on, the constraint starts after this token
};

pub const Think = extern struct {
    size: u32 = @sizeOf(Think),
    budget: u32, // reply tokens allowed inside the thought
    open: Span(u32), // the model's own opener (`<think>`, `<|channel>thought`, ` to=self<|message|>`)
    opened_in_prompt: bool, // the rendered prompt already ended inside a thought: count from token 0
    close: Span(u32), // the early-stop line + closer, committed as one forward
    end: u32, // the token that ends a thought
};

pub const ToolGate = extern struct {
    size: u32 = @sizeOf(ToolGate),
    opener: Span(u32), // the dialect's call opener, committed after the thought
    names: Span(Str), // offered names; empty = any
    think_end: u32 = no_token,
};

pub const Media = extern struct {
    kind: enum(u8) { image, video_frame, audio },
    at: u32, // where its placeholder run starts in the prompt
    width: u32,
    height: u32,
    data: Span(u8), // RGB8 pixels or PCM f32, decoded by the host
    hash: u64, // pixel hash: the encoder cache and the prefix cache key on it
};

pub const CacheHint = extern struct {
    history_len: u32 = 0, // where the rendered history ends: worth keeping for the next turn
    key: u64 = 0, // the workload key for fair eviction
    keep: bool = true, // false: do not retain this request's state (the host sets it after a tool-call reply)
    cold: bool = false, // true: read no kept state (the exactness contracts' cold reference)
};

/// Valid only during `submit`: the engine copies what it keeps.
pub const Request = extern struct {
    size: u32 = @sizeOf(Request),
    prompt: Span(u32),
    media: Span(Media) = .{}, // Caps.images / video / audio
    max_tokens: u32,
    sampling: Sampling = .{},
    eos: Span(u32) = .{}, // empty: the request ignores EOS
    logprobs: u8 = 0, // top-n per token, 0 = off (Caps.logprobs)
    grammar: ?*const Grammar = null,
    think: ?*const Think = null,
    tool_gate: ?*const ToolGate = null,
    cache: CacheHint = .{},
    background: bool = false,
    options_json: Str = .{}, // enable_mtp, enable_drafter, enable_batch_mtp, kv_quant, draft depths...
    no_drafts: bool = false, // no speculative round of any kind: the plain reference of the exactness contracts
    prompt_logprobs: bool = false, // Caps.prompt_logprobs
};

pub const Logprob = extern struct { token: u32, logprob: f32, top: Span(Bias) }; // top: (token, logprob) pairs

pub const Stats = extern struct {
    size: u32 = @sizeOf(Stats),
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    spec_mode: Str = .{},
    telemetry_json: Str = .{},
};

/// Per request: at most one `prefilled`, then `tokens`*, then exactly one `finished`, last. A request that ends
/// before its prefill completes (cancelled, failed) gets `finished` alone.
pub const Event = extern struct {
    size: u32 = @sizeOf(Event),
    kind: enum(u8) { prefilled, tokens, finished, audio },
    cached: u32 = 0, // prefilled: prompt tokens restored from a kept prefix
    tokens: Span(u32) = .{}, // tokens: committed ids in order, valid during the call
    logprobs: Span(Logprob) = .{}, // tokens: one per id when asked for
    reason: enum(u8) { stop, length, cancelled, failed } = .stop, // finished
    stats: ?*const Stats = null, // finished
    verdict: Verdict = .{}, // finished(failed): the named error
    prompt_tokens: u32 = 0, // prefilled: the prompt as the engine counted it
    prompt_logprobs: Span(Logprob) = .{}, // prefilled: the prompt's own logprobs when asked for
    constraint_at: u32 = std.math.maxInt(u32), // tokens: the token index where a constrained payload begins
    constraint_offset: u32 = 0, // tokens: its byte offset inside that token
    finish_details: Str = .{}, // finished: a static cause beside the reason (`repetition_loop`)
    loop_trim_start: u64 = std.math.maxInt(u64), // finished: where a loop-stopped span begins in the generated tokens
    refused_needed: u64 = 0, // finished(failed): bytes the engine's admission needed ...
    refused_available: u64 = 0, // ... and had, when it refused the prefill
    codes: Span(u32) = .{}, // audio: spoken frames' codes, frame after frame, valid during the call
    samples: Span(f32) = .{}, // audio: the mono samples those frames made final
    sample_rate: u32 = 0, // audio
};

/// Runs on the engine's thread; returns at once. `stop` ends the request as `finished(stop)` after this round:
/// the host decodes the round's tail there and judges stop strings, loop-stop and disconnect.
pub const Action = enum(u8) { go, stop };

pub const Sink = extern struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, id: u64, ev: *const Event) callconv(.c) Action,
};

pub const Status = extern struct { size: u32 = @sizeOf(Status), running: u32 = 0, waiting: u32 = 0, decode_tps: f64 = 0, prefill_tps: f64 = 0 };
pub const Memory = extern struct { size: u32 = @sizeOf(Memory), active: u64 = 0, cache: u64 = 0, peak: u64 = 0 };

pub const Engine = extern struct {
    size: u32 = @sizeOf(Engine),
    version: u32 = version,

    // ── lifecycle
    probe: *const fn (model_dir: Str, config_json: Str, out: *Verdict) callconv(.c) bool,
    load_bytes: *const fn (o: *const Open, bytes: *u64) callconv(.c) Verdict, // the preflight's bill
    open: *const fn (o: *const Open, info: *Info, out: *Verdict) callconv(.c) ?*anyopaque,
    close: *const fn (e: *anyopaque) callconv(.c) void, // cancels every request; each still gets `finished`

    // ── generation
    submit: *const fn (e: *anyopaque, id: u64, r: *const Request, sink: Sink) callconv(.c) Verdict,
    cancel: *const fn (e: *anyopaque, id: u64) callconv(.c) void,
    step: ?*const fn (e: *anyopaque, until_ns: u64) callconv(.c) bool, // host_gpu_thread only: true = did work

    // ── state the host reads (any thread; never touches the GPU)
    status: *const fn (e: *anyopaque, out: *Status) callconv(.c) void,
    memory: *const fn (e: *anyopaque, out: *Memory) callconv(.c) void,
    report_json: ?*const fn (e: *anyopaque, what: enum(u8) { props, metrics }, out: *Str) callconv(.c) void,
    trim: ?*const fn (e: *anyopaque, want_bytes: u64) callconv(.c) u64, // drop caches under memory pressure

    // ── optional entry points, each behind its Caps bit
    media_rows: ?*const fn (e: *anyopaque, m: *const Media) callconv(.c) u32, // placeholder run length
    embed: ?*const fn (e: *anyopaque, inputs: Span(Span(u32)), out: [*]f32) callconv(.c) Verdict,
    decide: ?*const fn (e: *anyopaque, request_json: Str, out_json: *Str) callconv(.c) Verdict,
    tokenize: ?*const fn (e: *anyopaque, text: Str, specials: bool, out: [*]u32, cap: usize) callconv(.c) usize,
    token_bytes: ?*const fn (e: *anyopaque, token: u32) callconv(.c) Str, // engine-owned, stable
    chat_template: ?*const fn (e: *anyopaque) callconv(.c) Str,
    render: ?*const fn (e: *anyopaque, chat_json: Str, out: [*]u32, cap: usize) callconv(.c) usize,

    // ── a host_gpu_thread engine's memory side
    admission: ?*const Admission = null,
};

pub const no_token: u32 = 0xffff_ffff;

/// Each engine exports exactly this symbol.
pub const entry_symbol = "mlx_serve_engine_v1";

/// What an engine on the host's GPU thread prices against memory: the context it advertises and whether a prompt fits.
/// `model` is the engine's own per-model object (ours: its `ModelConfig`).
pub const Admission = extern struct {
    size: u32 = @sizeOf(Admission),
    context: *const fn (model: *const anyopaque) callconv(.c) u32, // prompt + reply, as advertised
    pin_context: *const fn (model: *anyopaque, announce: bool) callconv(.c) u32, // freezes it; `announce` logs how
    max_safe_context: *const fn (model: *const anyopaque) callconv(.c) u32, // the largest that fits now, no headroom
    cache_budget: *const fn () callconv(.c) u64, // the resolved prefix-cache byte budget; 0 = none
    /// true admits; false refuses, `out` holding what was compared. `keep` names the model no idle unload may take;
    /// `kv_bits`: 0 dense, 4 or 8, `kv_configured` = the model's own scheme.
    admit: *const fn (model: *const anyopaque, keep: Str, prompt: Span(u32), max_tokens: u32, kv_bits: u8, mtp: bool, media_bytes: u64, out: *Bill) callconv(.c) bool,
};

pub const kv_configured: u8 = 0xff;

/// A prompt's memory bill and the headroom it was compared against, in bytes.
pub const Bill = extern struct {
    size: u32 = @sizeOf(Bill),
    needed: u64 = 0,
    available: u64 = 0,
    /// Bytes the hot prefix cache is holding. Reported in a refusal, never credited (see `reclaimable`).
    evictable: u64 = 0,
    /// The part of `evictable` an eviction can prove it will return: residency minus the
    /// largest entry (a restore pins at most one).
    reclaimable: u64 = 0,
    /// The prefill width this bill was taken at, so a refusal can name it.
    chunk: u64 = 0,

    pub fn fits(self: Bill) bool {
        return self.needed <= self.available;
    }

    /// The one credit the decision and both sentences read: `reclaimable`, never `evictable`
    /// (a refusal once quoted the slot's own checked-out prefix as "evicted").
    pub fn evictionCredit(self: Bill) u64 {
        return self.reclaimable;
    }

    /// Fits once the cache has given back what it provably can. Credits `reclaimable`, not
    /// `evictable`: a 383k prompt was admitted on one entry it then restored from and refused
    /// after the eviction pass returned nothing.
    pub fn fitsAfterEviction(self: Bill) bool {
        return self.needed <= self.available +| self.evictionCredit();
    }
};

// ── helpers (Zig side only; nothing here crosses the interface)

pub fn span(comptime T: type, s: []const T) Span(T) {
    return .{ .ptr = s.ptr, .len = s.len };
}

pub fn items(comptime T: type, s: Span(T)) []const T {
    return if (s.ptr) |p| p[0..s.len] else &.{};
}

/// The capabilities a request asks for. A field at its neutral value asks for nothing.
pub fn needs(r: *const Request) Caps {
    var c: Caps = .{
        .penalties = r.sampling.repeat_penalty != 1 or r.sampling.frequency_penalty != 0 or r.sampling.presence_penalty != 0,
        .logit_bias = r.sampling.logit_bias.len > 0,
        .logprobs = r.logprobs > 0,
        .grammar = r.grammar != null,
        .think_budget = r.think != null,
        .tool_gate = r.tool_gate != null,
        .prompt_logprobs = r.prompt_logprobs,
    };
    for (items(Media, r.media)) |m| switch (m.kind) {
        .image => c.images = true,
        .video_frame => c.video = true,
        .audio => c.audio = true,
    };
    return c;
}

/// The first capability `need` asks that `have` lacks, by its `Caps` field name; null when served.
pub fn missing(have: Caps, need: Caps) ?[]const u8 {
    const info = @typeInfo(Caps).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        if (T == bool and @field(need, name) and !@field(have, name)) return name;
    }
    return null;
}

/// The refusal for the first capability `need` asks that `have` lacks, naming its `Caps` field; null when served.
pub fn refusal(have: Caps, need: Caps) ?Verdict {
    const info = @typeInfo(Caps).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        if (T != bool) continue;
        if (@field(need, name) and !@field(have, name)) {
            const why = "Caps." ++ name ++ " is not declared by this engine for this model";
            return .{ .code = .unsupported, .why = span(u8, why) };
        }
    }
    return null;
}

// ── layout: what an engine built against version 1 reads at each offset

const testing = std.testing;

fn expectLayout(comptime T: type, size: usize, offsets: anytype) !void {
    try testing.expectEqual(size, @sizeOf(T));
    inline for (@typeInfo(@TypeOf(offsets)).@"struct".field_names) |name| {
        try testing.expectEqual(@as(usize, @field(offsets, name)), @offsetOf(T, name));
    }
}

test "engine abi: extern layouts are pinned (64-bit)" {
    if (@sizeOf(usize) != 8) return error.SkipZigTest;
    try expectLayout(Str, 16, .{ .ptr = 0, .len = 8 });
    try expectLayout(Verdict, 24, .{ .code = 0, .why = 8 });
    try expectLayout(Host, 40, .{ .size = 0, .ctx = 8, .log = 16, .wake = 24, .progress = 32 });
    try expectLayout(Open, 112, .{ .size = 0, .model_dir = 8, .model_type = 24, .config_json = 40, .context = 56, .max_concurrent = 60, .memory_budget = 64, .settings_json = 72, .options_json = 88, .host = 104 });
    try expectLayout(Info, 104, .{ .size = 0, .name = 8, .version = 24, .caps = 40, .threading = 48, .context_window = 52, .max_concurrent = 56, .embedding_dims = 60, .eos = 64, .startup = 80, .vocab_size = 96 });
    try expectLayout(Sampling, 56, .{ .temperature = 0, .top_p = 4, .top_k = 8, .min_p = 12, .seed = 16, .seeded = 24, .repeat_penalty = 28, .frequency_penalty = 32, .presence_penalty = 36, .logit_bias = 40 });
    try expectLayout(Bias, 8, .{ .token = 0, .bias = 4 });
    try expectLayout(Grammar, 32, .{ .size = 0, .kind = 4, .text = 8, .after = 24 });
    try expectLayout(Think, 56, .{ .size = 0, .budget = 4, .open = 8, .opened_in_prompt = 24, .close = 32, .end = 48 });
    try expectLayout(ToolGate, 48, .{ .size = 0, .opener = 8, .names = 24, .think_end = 40 });
    try expectLayout(Media, 40, .{ .kind = 0, .at = 4, .width = 8, .height = 12, .data = 16, .hash = 32 });
    try expectLayout(CacheHint, 24, .{ .history_len = 0, .key = 8, .keep = 16, .cold = 17 });
    try expectLayout(Request, 208, .{ .size = 0, .prompt = 8, .media = 24, .max_tokens = 40, .sampling = 48, .eos = 104, .logprobs = 120, .grammar = 128, .think = 136, .tool_gate = 144, .cache = 152, .background = 176, .options_json = 184, .no_drafts = 200, .prompt_logprobs = 201 });
    try expectLayout(Logprob, 24, .{ .token = 0, .logprob = 4, .top = 8 });
    try expectLayout(Stats, 80, .{ .size = 0, .rounds = 8, .drafted = 16, .accepted = 24, .prefill_ns = 32, .decode_ns = 40, .spec_mode = 48, .telemetry_json = 64 });
    try expectLayout(Event, 200, .{ .size = 0, .kind = 4, .cached = 8, .tokens = 16, .logprobs = 32, .reason = 48, .stats = 56, .verdict = 64, .prompt_tokens = 88, .prompt_logprobs = 96, .constraint_at = 112, .constraint_offset = 116, .finish_details = 120, .loop_trim_start = 136, .refused_needed = 144, .refused_available = 152, .codes = 160, .samples = 176, .sample_rate = 192 });
    try expectLayout(Sink, 16, .{ .ctx = 0, .event = 8 });
    try expectLayout(Status, 32, .{ .size = 0, .running = 4, .waiting = 8, .decode_tps = 16, .prefill_tps = 24 });
    try expectLayout(Memory, 32, .{ .size = 0, .active = 8, .cache = 16, .peak = 24 });
    try expectLayout(Admission, 48, .{ .size = 0, .context = 8, .pin_context = 16, .max_safe_context = 24, .cache_budget = 32, .admit = 40 });
    try expectLayout(Bill, 48, .{ .size = 0, .needed = 8, .available = 16, .evictable = 24, .reclaimable = 32, .chunk = 40 });
    try expectLayout(Engine, 160, .{ .size = 0, .version = 4, .probe = 8, .load_bytes = 16, .open = 24, .close = 32, .submit = 40, .cancel = 48, .step = 56, .status = 64, .memory = 72, .report_json = 80, .trim = 88, .media_rows = 96, .embed = 104, .decide = 112, .tokenize = 120, .token_bytes = 128, .chat_template = 136, .render = 144, .admission = 152 });
}

test "engine abi: every struct carrying `size` has it first, as a u32 defaulting to its own size" {
    inline for (.{ Host, Open, Info, Grammar, Think, ToolGate, Request, Stats, Event, Status, Memory, Engine, Admission, Bill }) |T| {
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
        try testing.expectEqual(u32, @FieldType(T, "size"));
        const d = @typeInfo(T).@"struct".field_attrs[0].defaultValue(u32) orelse return error.NoSizeDefault;
        try testing.expectEqual(@as(u32, @sizeOf(T)), d);
    }
}

test "engine abi: Caps bit positions are pinned" {
    const bits = .{ "penalties", "logit_bias", "logprobs", "grammar", "think_budget", "tool_gate", "images", "video", "audio", "embed", "decide", "owns_tokenizer", "owns_template", "batches", "drafts", "prefix_cache", "distributed", "prompt_logprobs" };
    inline for (bits, 0..) |name, i| try testing.expectEqual(@as(u16, i), @bitOffsetOf(Caps, name));
    try testing.expectEqual(@as(u64, 1 << 2), @as(u64, @bitCast(Caps{ .logprobs = true })));
}

test "engine abi: enum tags and constants are pinned" {
    try testing.expectEqual(@as(u8, 1), @intFromEnum(Threading.host_gpu_thread));
    try testing.expectEqual(@as(u8, 4), @intFromEnum(@FieldType(Verdict, "code").failed));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(@FieldType(Event, "kind").finished));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(@FieldType(Event, "reason").failed));
    try testing.expectEqual(@as(u8, 4), @intFromEnum(@FieldType(Grammar, "kind").ebnf));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(@FieldType(Media, "kind").audio));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(Action.stop));
    try testing.expectEqual(@as(u8, 4), @intFromEnum(Progress.prefill_done));
    try testing.expectEqualStrings("mlx_serve_engine_v1", entry_symbol);
    try testing.expectEqual(@as(u32, 1), version);
}

test "engine abi: a field at its neutral value asks for nothing" {
    const prompt = [_]u32{1};
    try testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(needs(&.{ .prompt = span(u32, &prompt), .max_tokens = 1 }))));
    const g: Grammar = .{ .kind = .json_object, .text = .{} };
    const media = [_]Media{.{ .kind = .audio, .at = 0, .width = 0, .height = 0, .data = .{}, .hash = 0 }};
    const n = needs(&.{ .prompt = span(u32, &prompt), .max_tokens = 1, .grammar = &g, .logprobs = 2, .sampling = .{ .presence_penalty = 0.5 }, .media = span(Media, &media) });
    try testing.expect(n.grammar and n.logprobs and n.penalties and n.audio);
    try testing.expect(!n.images and !n.logit_bias and !n.think_budget and !n.prompt_logprobs);
    try testing.expect(needs(&.{ .prompt = span(u32, &prompt), .max_tokens = 1, .prompt_logprobs = true }).prompt_logprobs);
}

test "engine abi: a refusal names the first missing Caps field, statically" {
    try testing.expect(refusal(.{ .grammar = true }, .{ .grammar = true }) == null);
    const v = refusal(.{ .logprobs = true }, .{ .logprobs = true, .tool_gate = true }).?;
    try testing.expectEqual(@FieldType(Verdict, "code").unsupported, v.code);
    try testing.expectEqualStrings("Caps.tool_gate is not declared by this engine for this model", items(u8, v.why));
    try testing.expectEqualStrings("tool_gate", missing(.{ .logprobs = true }, .{ .logprobs = true, .tool_gate = true }).?);
    try testing.expect(missing(.{ .grammar = true }, .{ .grammar = true }) == null);
}
