//! MiniMax Music 3 on the media table: `/v1/audio/music-generations` text2music (caption + lyrics, or instrumental),
//! 44.1 kHz stereo WAV. Tempo and key ride the caption; ACE-Step-only fields are named refusals.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const music3 = @import("music3.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .audio,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    e: *music3.Engine,
    wav: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const e = music3.Engine.load(io, gpa, ctx.model_dir.slice()) catch {
        out.* = Verdict.refuse(.failed, "music3: the pack does not load");
        return null;
    };
    const self = gpa.create(Engine) catch {
        e.deinit();
        out.* = Verdict.refuse(.failed, "music3: out of memory");
        return null;
    };
    self.* = .{ .e = e };
    return self;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const self: *Engine = @ptrCast(@alignCast(ep));
    gpa.free(self.wav);
    self.e.deinit();
    gpa.destroy(self);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// A request's fields in the order the endpoint has always named them; `caption` is owned by `gpa` (it carries the
/// tempo/key/no-vocals facts), the rest borrow the body.
const Parsed = struct { req: music3.MusicRequest, caption: []u8 };

fn parse(b: *const job.Body) union(enum) { ok: Parsed, bad: Verdict } {
    inline for (.{ "timesignature", "vocal_language" }) |f| if (b.get(f)) |v| if (v == .string or v == .integer)
        return .{ .bad = bad("'" ++ f ++ "' is an ACE-Step field; MiniMax Music 3 has no documented equivalent (put it in 'prompt' yourself)") };
    const prompt = b.str("prompt") orelse return .{ .bad = bad("missing 'prompt' (style/genre/mood description)") };
    if (prompt.len == 0) return .{ .bad = bad("empty 'prompt'") };
    const instrumental = b.flag("instrumental");
    const lyrics = b.str("lyrics") orelse "";
    if (job.instrumentalConflicts(instrumental, lyrics)) return .{ .bad = bad("'instrumental' is true but 'lyrics' is non-empty — send one or the other") };
    const none: []const u8 = "";
    if ((b.str("ref_audio") orelse none).len > 0) return .{ .bad = bad("'ref_audio' is not supported by this model (MiniMax Music 3 takes no reference audio) — it is an ACE-Step field") };
    if ((b.str("src_audio") orelse none).len > 0) return .{ .bad = bad("'src_audio' is not supported by this model (MiniMax Music 3 has no cover / complete mode) — it is an ACE-Step field") };
    if (b.str("task")) |t| if (!std.mem.eql(u8, t, "text2music"))
        return .{ .bad = bad("'task' is not supported by this model (MiniMax Music 3 is text2music only) — cover / complete are ACE-Step tasks") };
    if (!instrumental and std.mem.trim(u8, lyrics, " \t\r\n").len == 0)
        return .{ .bad = bad("missing 'lyrics' (MiniMax Music 3 is lyric-conditioned; structure tags like [verse] go on their own lines, or send \"instrumental\": true)") };
    var bpm: ?u32 = null;
    if (b.get("bpm")) |v| if (v == .integer) {
        if (v.integer < 30 or v.integer > 300) return .{ .bad = bad("'bpm' must be in [30,300]") };
        bpm = @intCast(v.integer);
    };
    const duration = b.uint("duration_seconds") orelse 60;
    if (duration < music3.MIN_DURATION_S or duration > music3.MAX_DURATION_S) return .{ .bad = bad("'duration_seconds' must be in [1,360]") };
    var steps: u32 = music3.DEFAULT_STEPS;
    if (b.get("steps")) |v| if (v == .integer) {
        if (v.integer < 4 or v.integer > 100) return .{ .bad = bad("'steps' must be in [4,100]") };
        steps = @intCast(v.integer);
    };
    const caption = music3.captionWithFacts(gpa, prompt, bpm, b.str("keyscale") orelse "", instrumental) catch
        return .{ .bad = job.failure(error.OutOfMemory, "music generation failed") };
    return .{ .ok = .{ .caption = caption, .req = .{
        .caption = caption,
        .lyrics = music3.resolveLyrics(instrumental, lyrics),
        .duration_s = @intCast(duration),
        .seed = b.uint("seed") orelse 42,
        .steps = steps,
    } } };
}

fn step(ep: *anyopaque, r: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const self: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, r.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    const p = switch (parse(&body)) {
        .ok => |q| q,
        .bad => |v| return v,
    };
    defer gpa.free(p.caption);
    log.info("[music3] generating {d}s steps={d} seed={d} lyrics={d}ch instrumental={}\n", .{ p.req.duration_s, p.req.steps, p.req.seed, p.req.lyrics.len, body.flag("instrumental") });
    const out = self.e.generateWav(gpa, p.req, job.progressOf(progress)) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        error.PromptTooLong => bad("assembled prompt exceeds 5000 tokens"),
        else => job.failure(err, "music generation failed"),
    };
    gpa.free(self.wav);
    self.wav = out;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const self: *Engine = @ptrCast(@alignCast(ep));
    if (self.wav.len == 0) return Verdict.refuse(.failed, "music3: no finished job to encode");
    out.* = .of(self.wav);
    return .ok;
}

/// A tiny random Music 3: a one-layer Qwen3 at 32 over the real vocabulary layout (semantic codes past
/// `CODE_OFFSET`, a 4-bit head so the pruned-head path runs), a one-layer depth decoder, a one-block DiT at 32 with
/// a 64-wide condition, and a vocoder at 16 → 4 channels on the published strides. Three AR frames at most.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const h = 32; // LM / depth hidden
    const sem = 64; // semantic codes
    const vocab: u64 = @as(u64, @intCast(music3.CODE_OFFSET)) + sem;
    const av = 64; // audio codes per book
    const dh = 32; // DiT hidden
    const cond = 64;

    var lm = T.init(a, 41);
    try lm.rand("model.embed_tokens.weight", .{}, &.{ vocab, h }, .BF16, 0.5);
    try lm.words("lm_head.weight", .{}, &.{ vocab, h * 4 / 32 });
    try lm.rand("lm_head.scales", .{}, &.{ vocab, 1 }, .BF16, 0.02);
    try lm.rand("lm_head.biases", .{}, &.{ vocab, 1 }, .BF16, 0.02);
    try lm.fill("model.norm.weight", .{}, &.{h}, .BF16, 1);
    try lm.fill("model.layers.0.input_layernorm.weight", .{}, &.{h}, .BF16, 1);
    try lm.fill("model.layers.0.post_attention_layernorm.weight", .{}, &.{h}, .BF16, 1);
    try lm.fill("model.layers.0.self_attn.q_norm.weight", .{}, &.{16}, .BF16, 1);
    try lm.fill("model.layers.0.self_attn.k_norm.weight", .{}, &.{16}, .BF16, 1);
    inline for (.{ .{ "self_attn.q_proj", 32, h }, .{ "self_attn.k_proj", 16, h }, .{ "self_attn.v_proj", 16, h }, .{ "self_attn.o_proj", h, 32 }, .{ "mlp.gate_proj", 64, h }, .{ "mlp.up_proj", 64, h }, .{ "mlp.down_proj", h, 64 } }) |l|
        try lm.rand("model.layers.0." ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.2);

    var dd = T.init(a, 42);
    try dd.rand("pos_embedding.weight", .{}, &.{ 16, h }, .BF16, 0.2);
    try dd.rand("audio_embeddings.weight", .{}, &.{ 7 * av, h }, .BF16, 0.5);
    for (0..7) |i| try dd.rand("audio_heads.{d}.weight", .{i}, &.{ av, h }, .BF16, 0.5);
    try dd.rand("projection.weight", .{}, &.{ h, h }, .BF16, 0.2);
    try dd.fill("norm.weight", .{}, &.{h}, .BF16, 1);
    try dd.fill("layers.0.input_layernorm.weight", .{}, &.{h}, .BF16, 1);
    try dd.fill("layers.0.post_attention_layernorm.weight", .{}, &.{h}, .BF16, 1);
    inline for (.{ .{ "attn.to_q", h, h }, .{ "attn.to_k", h, h }, .{ "attn.to_v", h, h }, .{ "attn.to_out", h, h }, .{ "gate_proj", 64, h }, .{ "up_proj", 64, h }, .{ "down_proj", h, 64 } }) |l|
        try dd.rand("layers.0." ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.2);

    var dit = T.init(a, 43);
    const ch = 2 * 128 + cond;
    try dit.rand("preprocess_conv.weight", .{}, &.{ ch, ch, 1 }, .F32, 0.02);
    try dit.rand("postprocess_conv.weight", .{}, &.{ 128, 128, 1 }, .F32, 0.02);
    try dit.rand("proj_in.weight", .{}, &.{ dh, ch }, .BF16, 0.1);
    try dit.rand("proj_out.weight", .{}, &.{ 128, dh }, .F32, 0.1);
    try dit.rand("time_proj.weight", .{}, &.{ 128, 1 }, .F32, 1);
    try dit.rand("time_embed.linear_1.weight", .{}, &.{ dh, 256 }, .F32, 0.1);
    try dit.rand("time_embed.linear_1.bias", .{}, &.{dh}, .F32, 0.1);
    try dit.rand("time_embed.linear_2.weight", .{}, &.{ dh, dh }, .BF16, 0.1);
    try dit.rand("time_embed.linear_2.bias", .{}, &.{dh}, .F32, 0.1);
    inline for (.{ "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0" }) |n|
        try dit.rand("transformer_blocks.0." ++ n ++ ".weight", .{}, &.{ dh, dh }, .BF16, 0.2);
    try dit.rand("transformer_blocks.0.ff_in.weight", .{}, &.{ 128, dh }, .BF16, 0.2);
    try dit.rand("transformer_blocks.0.ff_in.bias", .{}, &.{128}, .F32, 0.1);
    try dit.rand("transformer_blocks.0.ff_out.weight", .{}, &.{ dh, 64 }, .BF16, 0.2);
    try dit.rand("transformer_blocks.0.ff_out.bias", .{}, &.{dh}, .F32, 0.1);
    inline for (.{ "norm1", "norm2" }) |n| {
        try dit.fill("transformer_blocks.0." ++ n ++ ".weight", .{}, &.{dh}, .F32, 1);
        try dit.fill("transformer_blocks.0." ++ n ++ ".bias", .{}, &.{dh}, .F32, 0);
    }

    var ce = T.init(a, 44);
    try ce.fill("layer_scale", .{}, &.{1}, .F32, 1);
    try ce.rand("layer_weight_logits", .{}, &.{8}, .F32, 1);
    try ce.rand("proj.weight", .{}, &.{ cond, h, 3 }, .F32, 0.1);
    try ce.rand("proj.bias", .{}, &.{cond}, .F32, 0.1);

    // Vocoder (PyTorch layouts, weight-normed): 64 → 16, then 16 → 8 → 8 → 4 → 4 over strides 8, 8, 4, 2.
    var voc = T.init(a, 45);
    try voc.rand("dec_in_proj.weight", .{}, &.{ 16, 64, 1 }, .F32, 0.1);
    try voc.rand("dec_in_proj.bias", .{}, &.{16}, .F32, 0.1);
    try wnConv(&voc, "conv_in", .{}, 16, 16, 7);
    const chs = [_]u64{ 16, 8, 8, 4, 4 };
    for ([_]u64{ 8, 8, 4, 2 }, 0..) |stride, bi| {
        try alpha(&voc, "blocks.{d}.snake1", .{bi}, chs[bi]);
        try voc.fill("blocks.{d}.conv_t1.weight_g", .{bi}, &.{ chs[bi], 1, 1 }, .F32, 1);
        try voc.rand("blocks.{d}.conv_t1.weight_v", .{bi}, &.{ chs[bi], chs[bi + 1], 2 * stride }, .F32, 0.3);
        try voc.rand("blocks.{d}.conv_t1.bias", .{bi}, &.{chs[bi + 1]}, .F32, 0.1);
        for (1..4) |u| {
            try alpha(&voc, "blocks.{d}.res_unit{d}.snake1", .{ bi, u }, chs[bi + 1]);
            try alpha(&voc, "blocks.{d}.res_unit{d}.snake2", .{ bi, u }, chs[bi + 1]);
            try wnConv(&voc, "blocks.{d}.res_unit{d}.conv1", .{ bi, u }, chs[bi + 1], chs[bi + 1], 7);
            try wnConv(&voc, "blocks.{d}.res_unit{d}.conv2", .{ bi, u }, chs[bi + 1], chs[bi + 1], 1);
        }
    }
    try alpha(&voc, "snake_out", .{}, 4);
    try wnConv(&voc, "conv_out", .{}, 1, 4, 7);

    const specials = [_]sdk.byte_tokenizer.Special{
        .{ .id = 151643, .content = "<|endoftext|>" },     .{ .id = 151644, .content = "<|im_start|>" },
        .{ .id = 151645, .content = "<|im_end|>" },        .{ .id = 151669, .content = "<|audio_start|>" },
        .{ .id = 151671, .content = "<|caption_start|>" }, .{ .id = 151672, .content = "<|caption_end|>" },
        .{ .id = 151673, .content = "<|lyrics_start|>" },  .{ .id = 151674, .content = "<|lyrics_end|>" },
    };

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"minimax_music3","sample_rate":44100,"max_frames":3,"max_text_tokens":5000,"num_codebooks":8,
            \\ "semantic_vocab_size":64,"audio_vocab_size":64,
            \\ "language_model":{"hidden_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
            \\   "head_dim":16,"vocab_size":151739,"rms_norm_eps":1e-06,"rope_theta":1000000},
            \\ "rvq_depth_decoder":{"hidden_size":32,"num_attention_heads":2,"num_layers":1},
            \\ "transformer":{"num_layers":1,"num_attention_heads":2,"attention_head_dim":16,"hidden_size":32,
            \\   "in_channels":128,"condition_dim":64,"fourier_embedding_dim":256,"rotary_dim":16}}
            },
            .{ .name = "language_model.safetensors", .bytes = try lm.bytes() },
            .{ .name = "rvq_depth_decoder.safetensors", .bytes = try dd.bytes() },
            .{ .name = "transformer.safetensors", .bytes = try dit.bytes() },
            .{ .name = "condition_encoder.safetensors", .bytes = try ce.bytes() },
            .{ .name = "vocoder.safetensors", .bytes = try voc.bytes() },
            .{ .name = "tokenizer/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
        }),
        .body = "{\"prompt\":\"lofi piano\",\"instrumental\":true,\"duration_seconds\":1,\"steps\":4,\"seed\":3}",
    };
}

fn alpha(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, c: u64) !void {
    try t.rand(p ++ ".alpha", args, &.{ 1, c, 1 }, .F32, 1);
}

/// A weight-normed PyTorch Conv1d (`weight_g` [out,1,1], `weight_v` [out,in,k], bias).
fn wnConv(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, out: u64, in: u64, k: u64) !void {
    try t.fill(p ++ ".weight_g", args, &.{ out, 1, 1 }, .F32, 1);
    try t.rand(p ++ ".weight_v", args, &.{ out, in, k }, .F32, 0.3);
    try t.rand(p ++ ".bias", args, &.{out}, .F32, 0.1);
}

// ── tests ──

const testing = std.testing;

test "music3 plugin: a request is validated before any forward, by name" {
    inline for (.{
        .{ "{\"prompt\":\"p\",\"timesignature\":\"4/4\"}", "'timesignature' is an ACE-Step field; MiniMax Music 3 has no documented equivalent (put it in 'prompt' yourself)" },
        .{ "{}", "missing 'prompt' (style/genre/mood description)" },
        .{ "{\"prompt\":\"p\",\"instrumental\":true,\"lyrics\":\"la\"}", "'instrumental' is true but 'lyrics' is non-empty — send one or the other" },
        .{ "{\"prompt\":\"p\",\"task\":\"cover\",\"lyrics\":\"la\"}", "'task' is not supported by this model (MiniMax Music 3 is text2music only) — cover / complete are ACE-Step tasks" },
        .{ "{\"prompt\":\"p\"}", "missing 'lyrics' (MiniMax Music 3 is lyric-conditioned; structure tags like [verse] go on their own lines, or send \"instrumental\": true)" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"la\",\"duration_seconds\":0}", "'duration_seconds' must be in [1,360]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"la\",\"steps\":2}", "'steps' must be in [4,100]" },
    }) |c| {
        var b = job.Body.parse(testing.allocator, c[0]).?;
        defer b.deinit();
        try testing.expectEqualStrings(c[1], parse(&b).bad.why.slice());
    }
}

test "music3 plugin: a seeded song on the tiny pack is a deterministic stereo WAV" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
