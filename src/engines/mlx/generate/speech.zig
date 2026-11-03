//! Speech output (`Parsed.speech`, `SpeechHooks`): mlx-vlm's lfm3_audio `stream_slots` over the engine's cache. Each
//! slot forwards one row: a text token, or a spoken frame (`frame_token` with the frame's rows as `Forward.embeds`)
//! the plugin draws from the trunk's last hidden row. Text slots sample with the request's sampling over the last 32
//! text tokens, frames with the speech sampling over the last `repetition_window` frames. Frames go to the codec in
//! chunks and out as `.audio` events beside their `frame_token`.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const Generator = @import("generator.zig").Generator;
const applyPenalties = @import("../sampling/filters.zig").applyPenalties;
const sampleTokenLazy = @import("../sampling/lazy.zig").sampleTokenLazy;

const request = @import("../../../sdk/request/speech.zig");
pub const Mode = request.Mode;
pub const Request = request.Request;

/// The reference's text repetition window.
const TEXT_WINDOW = 32;
/// Frames per codec push: 320 ms of speech, each push re-decoding the codec's left context.
pub const CODEC_CHUNK = 4;

/// `stream_slots`'s mode switching, one call per slot.
pub const Machine = struct {
    f: abi.SpeechFacts,
    mode: Mode,
    slots_left: u32,
    speaking: bool = false,
    text_done: bool = false,

    pub fn init(f: abi.SpeechFacts, last_prompt_id: u32, explicit: ?Mode) Machine {
        const mode = explicit orelse if (last_prompt_id == f.audio_start or last_prompt_id == f.frame_token)
            Mode.audio
        else if (last_prompt_id == f.mixed_start) Mode.interleaved else Mode.text;
        return .{ .f = f, .mode = mode, .slots_left = f.interleave_text };
    }

    pub fn wantsFrame(m: *const Machine) bool {
        return m.mode == .audio or (m.mode == .interleaved and m.speaking);
    }

    pub const Slot = union(enum) { frame: struct { eos: bool }, text: struct { token: u32, eos: bool } };

    /// Applies one slot's outcome; true when it ends the turn.
    pub fn after(m: *Machine, s: Slot) bool {
        const interleaving = m.mode == .interleaved;
        var finished = false;
        switch (s) {
            .frame => |fr| if (fr.eos) {
                m.mode = .text;
            },
            .text => |t| {
                if (t.token == m.f.text_end) {
                    // Ends the text of an interleaved block, else the turn.
                    if (interleaving) {
                        m.text_done = true;
                        m.speaking = true;
                    } else finished = true;
                } else if (t.eos) {
                    finished = true;
                } else if (interleaving) {
                    // Markers are plain text inside a block.
                } else if (t.token == m.f.audio_start) {
                    m.mode = .audio;
                } else if (t.token == m.f.mixed_start) {
                    m.mode = .interleaved;
                    m.slots_left = m.f.interleave_text;
                    m.speaking = false;
                    m.text_done = false;
                }
            },
        }
        if (interleaving and !m.text_done) {
            m.slots_left -= 1;
            if (m.slots_left == 0) {
                m.speaking = !m.speaking;
                m.slots_left = if (m.speaking) m.f.interleave_audio else m.f.interleave_text;
            }
        }
        return finished;
    }
};

/// A speech request's state beside its generator: `gen.pending_logits` and `gen.last_hidden` are the last forwarded
/// row's, nothing sampled from them yet.
pub const State = struct {
    a: std.mem.Allocator,
    hooks: *const abi.SpeechHooks,
    model: *anyopaque,
    machine: Machine,
    how: abi.SpeechSampling,
    order: []const u32,
    /// Text sampling: the request's, with the speech markers' suppression.
    mask: mlx.mlx_array,
    text_window: std.ArrayList(u32) = .empty,
    /// Spoken frames' codes, frame after frame (the penalty reads the last `repetition_window`).
    frames: std.ArrayList(u32) = .empty,
    /// Frames already pushed through the codec.
    pushed: usize = 0,
    /// The stream frames go into; null until the next frame after an end-of-speech push (a segment of its own).
    codec: abi.Handle = null,
    /// The codec holds samples it has not released: a push since the last one with `finish`.
    open: bool = false,
    samples: std.ArrayList(f32) = .empty,
    drawn: u64 = 0,

    /// `reserved` = the model's reserved-token mask (null = none); the markers a turn emits are allowed through it.
    pub fn init(a: std.mem.Allocator, hooks: *const abi.SpeechHooks, model: *anyopaque, f: abi.SpeechFacts, placeholder_audio: u32, vocab: usize, reserved: ?mlx.mlx_array, last_prompt_id: u32, req: Request, seed: u64, s: mlx.mlx_stream) !State {
        try checkFacts(f);
        const allow = [_]u32{ f.audio_start, f.mixed_start, f.text_end };
        var deny: [4]u32 = .{ placeholder_audio, f.frame_token, 0, 0 };
        // An explicit text mode keeps the reply in text, as the reference does.
        if (req.mode == .text) deny[2..4].* = .{ f.mixed_start, f.audio_start };
        const mask = try textMask(a, vocab, reserved, &allow, &deny, s);
        errdefer _ = mlx.mlx_array_free(mask);
        const st: State = .{
            .a = a,
            .hooks = hooks,
            .model = model,
            .machine = Machine.init(f, last_prompt_id, req.mode),
            .how = .{
                .temperature = req.temperature orelse f.temperature,
                .top_k = req.top_k orelse f.top_k,
                .repetition_penalty = req.repetition_penalty orelse f.repetition_penalty,
                .key = seed,
            },
            .order = req.order,
            .mask = mask,
        };
        return st;
    }

    pub fn deinit(st: *State) void {
        if (st.codec != null) st.hooks.codec_close(st.model, st.codec);
        _ = mlx.mlx_array_free(st.mask);
        st.text_window.deinit(st.a);
        st.frames.deinit(st.a);
        st.samples.deinit(st.a);
    }

    /// What one slot produced: its token, and when a codec push ran, the codes and samples it covered (valid until
    /// the next call).
    pub const Out = struct {
        token: u32,
        codes: []const u32 = &.{},
        samples: []const f32 = &.{},
        finished: bool = false,
        /// `token` is the stop that ended the turn: not output, as a text turn's.
        stop: bool = false,
    };

    /// Draws one slot from the last row, and unless it ends the turn (or the budget), forwards it.
    pub fn next(st: *State, gen: *Generator) !Out {
        const f = st.machine.f;
        const k = f.codebooks;
        var out: Out = undefined;
        var rows: mlx.mlx_array = .{ .ctx = null };
        defer if (rows.ctx != null) {
            _ = mlx.mlx_array_free(rows);
        };
        if (st.machine.wantsFrame()) {
            var codes: [MAX_CODEBOOKS]u32 = undefined;
            const window = @min(st.frames.items.len, @as(usize, f.repetition_window) * k);
            var how = st.how;
            how.key = st.how.key +% st.drawn *% 0x9E3779B97F4A7C15;
            if (st.order.len >= (st.drawn + 1) * k) how.order = .of(st.order[st.drawn * k ..][0..k]);
            st.drawn += 1;
            try check(st.hooks.frame(st.model, .{ .ctx = gen.last_hidden.ctx }, &how, .of(st.frames.items[st.frames.items.len - window ..]), &codes));
            const eos = codes[0] == f.eos_code;
            out = .{ .token = f.frame_token, .finished = st.machine.after(.{ .frame = .{ .eos = eos } }) };
            if (!eos) try st.frames.appendSlice(st.a, codes[0..k]);
            var rows_abi: abi.Array = .{};
            try check(st.hooks.frame_rows(st.model, .of(codes[0..k]), &rows_abi));
            rows = .{ .ctx = rows_abi.ctx };
            // The speech ends at its end-of-speech frame: the codec's tail goes out with it.
            if (eos or st.frames.items.len / k - st.pushed >= CODEC_CHUNK) try st.push(&out, eos);
        } else {
            const token = try st.sampleText(gen);
            const eos = for (gen.eos_token_ids) |e| {
                if (e == token) break true;
            } else false;
            out = .{ .token = token, .finished = st.machine.after(.{ .text = .{ .token = token, .eos = eos } }) };
            out.stop = out.finished;
            try st.text_window.append(st.a, token);
            if (st.text_window.items.len > TEXT_WINDOW) _ = st.text_window.orderedRemove(0);
        }
        gen.advanceStep(1);
        if (out.finished or gen.step >= gen.max_tokens) {
            gen.finish_reason = if (out.finished) "stop" else "length";
            out.finished = true;
            try st.finishTurn(&out);
        } else try forward(gen, out.token, rows);
        return out;
    }

    /// The turn ended: the codec releases what it still holds.
    fn finishTurn(st: *State, out: *Out) !void {
        if (st.open or st.frames.items.len / st.machine.f.codebooks > st.pushed) try st.push(out, true);
    }

    /// Pushes the frames the codec has not seen (`finish` = and its tail) into `out`'s codes and samples. A finish
    /// closes the stream: speech after it starts a new one, never overlapping samples already sent.
    fn push(st: *State, out: *Out, finish: bool) !void {
        const f = st.machine.f;
        if (st.codec == null) try check(st.hooks.codec_open(st.model, &st.codec));
        const codes = st.frames.items[st.pushed * f.codebooks ..];
        const frames = codes.len / f.codebooks;
        try st.samples.resize(st.a, (frames + 1) * f.frame_samples);
        var len: usize = 0;
        try check(st.hooks.codec_push(st.model, st.codec, .of(codes), finish, st.samples.items.ptr, st.samples.items.len, &len));
        st.pushed += frames;
        st.open = !finish;
        if (finish) {
            st.hooks.codec_close(st.model, st.codec);
            st.codec = null;
        }
        out.codes = codes;
        out.samples = st.samples.items[0..len];
    }

    fn sampleText(st: *State, gen: *Generator) !u32 {
        var sampling = gen.sampling;
        sampling.suppress_mask = st.mask;
        defer gen.sampling.draw +%= 1;
        var pen = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen);
        const logits = if (sampling.penalized() and st.text_window.items.len > 0) blk: {
            try applyPenalties(&pen, gen.pending_logits, st.text_window.items, sampling, gen.xfm.s);
            break :blk pen;
        } else gen.pending_logits;
        const tok = sampleTokenLazy(logits, sampling, gen.xfm.s);
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_array_eval(tok));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, tok));
        return @intCast(v);
    }
};

/// Forwards one row (`token`, its embedding replaced by `rows` when set): its logits and hidden row become pending.
fn forward(gen: *Generator, token: u32, rows: mlx.mlx_array) !void {
    const id: i32 = @intCast(token);
    const ids = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    gen.ctx.step_embeds = if (rows.ctx != null) rows else null;
    defer gen.ctx.step_embeds = null;
    var hidden = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(hidden);
    const logits = try gen.xfm.forwardWithCapture(&gen.ctx, ids, &hidden);
    if (gen.has_pending_logits) _ = mlx.mlx_array_free(gen.pending_logits);
    if (gen.has_last_hidden) _ = mlx.mlx_array_free(gen.last_hidden);
    gen.pending_logits = logits;
    gen.has_pending_logits = true;
    gen.last_hidden = hidden;
    gen.has_last_hidden = true;
}

/// The text slots' mask: `reserved` less `allow`, plus `deny` (0 = unset).
fn textMask(a: std.mem.Allocator, vocab: usize, reserved: ?mlx.mlx_array, allow: []const u32, deny: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const keep = try a.alloc(bool, vocab);
    defer a.free(keep);
    const add = try a.alloc(bool, vocab);
    defer a.free(add);
    @memset(keep, true);
    @memset(add, false);
    for (allow) |id| if (id < vocab) {
        keep[id] = false;
    };
    for (deny) |id| if (id != 0 and id < vocab) {
        add[id] = true;
    };
    const shape = [_]c_int{@intCast(vocab)};
    const keep_a = mlx.mlx_array_new_data(keep.ptr, &shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(keep_a);
    const add_a = mlx.mlx_array_new_data(add.ptr, &shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(add_a);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (reserved) |r| {
        var kept = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kept);
        try mlx.check(mlx.mlx_logical_and(&kept, r, keep_a, s));
        try mlx.check(mlx.mlx_logical_or(&out, kept, add_a, s));
    } else try mlx.check(mlx.mlx_array_set(&out, add_a));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The most codebooks a frame may hold: the host draws a frame into a buffer this size.
const MAX_CODEBOOKS = 64;

/// An outside plugin's speech facts are a promise the host's buffers rely on.
fn checkFacts(f: abi.SpeechFacts) !void {
    if (f.codebooks == 0 or f.codebooks > MAX_CODEBOOKS) {
        log.err("[speech] {d} codebooks: the host draws at most {d}\n", .{ f.codebooks, MAX_CODEBOOKS });
        return error.SpeechFactsOutOfRange;
    }
}

fn check(v: abi.Verdict) !void {
    if (v.isOk()) return;
    log.err("[speech] {s}\n", .{v.why.slice()});
    return error.SpeechHookFailed;
}

// ── tests ──

const testing = std.testing;

const facts: abi.SpeechFacts = .{
    .codebooks = 12,
    .codebook_size = 2017,
    .eos_code = 2016,
    .frame_token = 32,
    .audio_start = 12,
    .mixed_start = 15,
    .text_end = 14,
    .interleave_text = 6,
    .interleave_audio = 12,
};

test "speech machine: the prompt's last token picks the mode; explicit wins" {
    try testing.expectEqual(Mode.audio, Machine.init(facts, 12, null).mode);
    try testing.expectEqual(Mode.audio, Machine.init(facts, 32, null).mode);
    try testing.expectEqual(Mode.interleaved, Machine.init(facts, 15, null).mode);
    try testing.expectEqual(Mode.text, Machine.init(facts, 7, null).mode);
    try testing.expectEqual(Mode.text, Machine.init(facts, 12, .text).mode);
}

test "speech machine: interleaved runs 6 text slots, 12 frames, again; text_end ends the text, an EOS frame the speech" {
    var m = Machine.init(facts, 15, null);
    var kinds: std.ArrayList(u8) = .empty;
    defer kinds.deinit(testing.allocator);
    for (0..18) |_| {
        const frame = m.wantsFrame();
        try kinds.append(testing.allocator, if (frame) 'a' else 't');
        try testing.expect(!m.after(if (frame) .{ .frame = .{ .eos = false } } else .{ .text = .{ .token = 100, .eos = false } }));
    }
    try testing.expectEqualStrings("tttttt" ++ "aaaaaaaaaaaa", kinds.items);
    // Two text slots, then text_end: speech until its end-of-speech frame, then text to the turn's end.
    try testing.expect(!m.wantsFrame());
    try testing.expect(!m.after(.{ .text = .{ .token = 100, .eos = false } }));
    try testing.expect(!m.after(.{ .text = .{ .token = 14, .eos = true } }));
    for (0..30) |_| {
        try testing.expect(m.wantsFrame());
        try testing.expect(!m.after(.{ .frame = .{ .eos = false } }));
    }
    try testing.expect(!m.after(.{ .frame = .{ .eos = true } }));
    try testing.expect(!m.wantsFrame());
    try testing.expectEqual(Mode.text, m.mode);
    try testing.expect(m.after(.{ .text = .{ .token = 14, .eos = true } }));
}

test "speech machine: in text a marker switches the mode, inside a block it is text; an EOS id ends the turn" {
    var m = Machine.init(facts, 7, null);
    try testing.expect(!m.after(.{ .text = .{ .token = 12, .eos = false } }));
    try testing.expectEqual(Mode.audio, m.mode);
    try testing.expect(m.wantsFrame());
    var b = Machine.init(facts, 7, null);
    try testing.expect(!b.after(.{ .text = .{ .token = 15, .eos = false } }));
    try testing.expectEqual(Mode.interleaved, b.mode);
    try testing.expect(!b.after(.{ .text = .{ .token = 12, .eos = false } }));
    try testing.expectEqual(Mode.interleaved, b.mode);
    var c = Machine.init(facts, 7, null);
    try testing.expect(c.after(.{ .text = .{ .token = 99, .eos = true } }));
}

test "speech: LFM3-Audio's TTS and interleaved-chat recipes speak the reference's text, every frame through the codec (LFM3_AUDIO_MODEL, LFM3_SPEECH_FIXTURE)" {
    // tests/dump_lfm3_fixtures.py --speech: greedy text and speech over a seeded reveal order.
    const pack = std.c.getenv("LFM3_AUDIO_MODEL") orelse return error.SkipZigTest;
    const fx = std.c.getenv("LFM3_SPEECH_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const model_mod = @import("../../../sdk/config/model.zig");
    const driver = @import("../plugin/driver.zig");
    var cfg = try model_mod.parseConfig(io, a, std.mem.span(pack));
    defer cfg.deinit(a);
    const read = struct {
        fn f(al: std.mem.Allocator, rio: std.Io, dir: [*:0]const u8, name: []const u8) ![]u32 {
            var buf: [1024]u8 = undefined;
            const bytes = try std.Io.Dir.cwd().readFileAlloc(rio, try std.fmt.bufPrint(&buf, "{s}/{s}", .{ std.mem.span(dir), name }), al, .limited(1 << 24));
            defer al.free(bytes);
            return al.dupe(u32, std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(bytes))));
        }
    }.f;
    const order = try read(a, io, fx, "order.u32");
    defer a.free(order);
    var tok_dummy: @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer = undefined;
    for ([_][]const u8{ "tts", "chat" }) |name| {
        // A fresh model per recipe: its conv layers' state is the model's, not the generator's.
        var xfm = try driver.load(io, a, &cfg, std.mem.span(pack), mlx.gpuStream());
        defer xfm.deinit();
        var nb: [64]u8 = undefined;
        const prompt = try read(a, io, fx, try std.fmt.bufPrint(&nb, "{s}_prompt.u32", .{name}));
        defer a.free(prompt);
        const want_tokens = try read(a, io, fx, try std.fmt.bufPrint(&nb, "{s}_tokens.u32", .{name}));
        defer a.free(want_tokens);
        const want_codes = try read(a, io, fx, try std.fmt.bufPrint(&nb, "{s}_codes.u32", .{name}));
        defer a.free(want_codes);

        var gen = try Generator.initWithOptions(io, a, &xfm, &tok_dummy, prompt, 96, .{ .temperature = 0.0 }, &.{ 124900, 125014 }, .{
            .speech = .{ .temperature = 0.0, .repetition_penalty = 1.0, .order = order },
        });
        defer gen.deinit(a);
        var tokens: std.ArrayList(u32) = .empty;
        defer tokens.deinit(a);
        var samples: usize = 0;
        while (true) {
            const out = try gen.speech.?.next(&gen);
            try tokens.append(a, out.token);
            samples += out.samples.len;
            if (out.finished) break;
        }
        const st = gen.speech.?;
        std.debug.print("[speech] {s}: {d} slots ({d} want), {d} frames ({d} want), {d} samples\n", .{ name, tokens.items.len, want_tokens.len, st.frames.items.len / 12, want_codes.len / 12, samples });
        // The text slots in order; where the speech ends follows its codes.
        var want_text: std.ArrayList(u32) = .empty;
        defer want_text.deinit(a);
        var our_text: std.ArrayList(u32) = .empty;
        defer our_text.deinit(a);
        for (want_tokens) |t| if (t != 125032) try want_text.append(a, t);
        for (tokens.items) |t| if (t != 125032) try our_text.append(a, t);
        try testing.expectEqualSlices(u32, want_text.items, our_text.items);
        // Greedy codes sit on exact bf16 ties (the reference's own f32 depth decoder picks otherwise), and a flipped
        // code feeds every later pass and frame: agreement is logged, the depth decoder's exactness is the tiny proof's.
        var same: usize = 0;
        const n = @min(want_codes.len, st.frames.items.len);
        for (want_codes[0..n], st.frames.items[0..n]) |w, o| same += @intFromBool(w == o);
        std.debug.print("[speech] {s}: {d} of {d} codes equal\n", .{ name, same, n });
        // Every spoken frame reached the codec: 1920 samples a frame.
        try testing.expectEqual(st.frames.items.len / 12 * 1920, samples);
    }
}

/// A codec that records each push (frames, finish) and each stream it opened; it releases nothing.
const FakeCodec = struct {
    var pushes: [8][2]usize = undefined;
    var n: usize = 0;
    var opened: usize = 0;
    fn open(_: *anyopaque, out: *abi.Handle) callconv(.c) abi.Verdict {
        opened += 1;
        out.* = @ptrFromInt(opened);
        return .ok;
    }
    fn push(_: *anyopaque, _: abi.Handle, codes: abi.Span(u32), finish: bool, _: [*]f32, _: usize, len: *usize) callconv(.c) abi.Verdict {
        pushes[n] = .{ codes.len / facts.codebooks, @intFromBool(finish) };
        n += 1;
        len.* = 0;
        return .ok;
    }
    fn close(_: *anyopaque, _: abi.Handle) callconv(.c) void {}
    const hooks: abi.SpeechHooks = .{ .frame = undefined, .frame_rows = undefined, .codec_open = open, .codec_push = push, .codec_close = close };
};

test "speech codec: a turn ending right after a regular push still releases the codec's tail; speech after an end-of-speech frame opens a new stream" {
    FakeCodec.n = 0;
    FakeCodec.opened = 0;
    var dummy: u8 = 0;
    var st: State = .{ .a = testing.allocator, .hooks = &FakeCodec.hooks, .model = &dummy, .machine = Machine.init(facts, 15, null), .how = .{}, .order = &.{}, .mask = .{ .ctx = null } };
    defer {
        st.frames.deinit(st.a);
        st.samples.deinit(st.a);
    }
    var out: State.Out = .{ .token = 0 };
    try st.frames.appendNTimes(st.a, 1, 4 * facts.codebooks);
    try st.push(&out, false);
    try st.finishTurn(&out);
    try testing.expectEqual(@as(usize, 2), FakeCodec.n);
    try testing.expectEqual([2]usize{ 0, 1 }, FakeCodec.pushes[1]);

    // A second segment after the end-of-speech push gets a stream of its own, never the closed one's overlap.
    try st.frames.appendNTimes(st.a, 1, 2 * facts.codebooks);
    try st.push(&out, false);
    try testing.expectEqual(@as(usize, 2), FakeCodec.opened);
    try st.finishTurn(&out);
    try testing.expectEqual([2]usize{ 0, 1 }, FakeCodec.pushes[3]);
}

test "speech facts: a plugin's codebook count must fit the frame the host draws into" {
    try checkFacts(facts);
    var many = facts;
    many.codebooks = MAX_CODEBOOKS + 1;
    try testing.expectError(error.SpeechFactsOutOfRange, checkFacts(many));
    many.codebooks = 0;
    try testing.expectError(error.SpeechFactsOutOfRange, checkFacts(many));
}

