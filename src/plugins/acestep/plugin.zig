//! ACE-Step v1.5 on the media table: `/v1/audio/music-generations` text2music, cover and complete (the task is the
//! DiT's context stream + its instruction line), an optional reference clip for style/timbre; 48 kHz stereo WAV.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const wav = @import("../../sdk/media/wav.zig");
const mlx = @import("../../mlx/ffi.zig");
const acestep = @import("acestep.zig");
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
    e: *acestep.Engine,
    wav: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const e = acestep.Engine.load(io, gpa, ctx.model_dir.slice(), if (ctx.machine) |m| m.low_mem else false) catch {
        out.* = Verdict.refuse(.failed, "acestep: the pack does not load");
        return null;
    };
    const self = gpa.create(Engine) catch {
        e.deinit();
        out.* = Verdict.refuse(.failed, "acestep: out of memory");
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

/// A clip as `[1,N,2]` f32 stereo at the engine's rate (owned), or the refusal naming the field.
const Clip = union(enum) { none, arr: mlx.mlx_array, bad: Verdict };

/// `src_audio`: the cover / complete source, full length (10-600 s).
fn sourceClip(b: *const job.Body, task: acestep.Task, rate: u32) Clip {
    const b64 = b.str("src_audio") orelse return .none;
    if (b64.len == 0) return .none;
    if (task == .text2music) return .{ .bad = bad("'src_audio' needs task \"cover\" or \"complete\" (text2music takes 'ref_audio' for style)") };
    const bytes = job.base64Decode(gpa, b64) catch return .{ .bad = bad("src_audio: invalid base64") };
    defer gpa.free(bytes);
    const dec = wav.decode(gpa, bytes) catch return .{ .bad = bad("src_audio: expected a PCM16/PCM24/float32 WAV") };
    defer gpa.free(dec.pcm);
    const secs = (dec.pcm.len / dec.channels) / dec.sample_rate;
    if (secs < acestep.MIN_DURATION_S) return .{ .bad = bad("src_audio: clip too short (needs at least 10 s)") };
    if (secs > acestep.MAX_DURATION_S) return .{ .bad = bad("src_audio: clip longer than 600 s") };
    return stereoAt(dec, rate, false);
}

/// `ref_audio`: a style/timbre clip (at least 1 s), windowed to the 30 s the timbre encoder sees.
fn referenceClip(b: *const job.Body, rate: u32) Clip {
    const b64 = b.str("ref_audio") orelse return .none;
    if (b64.len == 0) return .none;
    const bytes = job.base64Decode(gpa, b64) catch return .{ .bad = bad("ref_audio: invalid base64") };
    defer gpa.free(bytes);
    const dec = wav.decode(gpa, bytes) catch return .{ .bad = bad("ref_audio: expected a PCM16/PCM24/float32 WAV") };
    defer gpa.free(dec.pcm);
    if (dec.pcm.len / dec.channels < dec.sample_rate) return .{ .bad = bad("ref_audio: clip too short (needs at least 1 s)") };
    return stereoAt(dec, rate, true);
}

fn stereoAt(dec: wav.Decoded, rate: u32, window: bool) Clip {
    const stereo = wav.toStereoInterleaved(gpa, dec.pcm, dec.channels) catch return .{ .bad = job.failure(error.OutOfMemory, "music generation failed") };
    defer gpa.free(stereo);
    const resampled = wav.resampleLinear(gpa, stereo, 2, dec.sample_rate, rate) catch return .{ .bad = job.failure(error.OutOfMemory, "music generation failed") };
    defer gpa.free(resampled);
    const pcm = if (window) acestep.referenceWindow(gpa, resampled, rate) catch return .{ .bad = job.failure(error.OutOfMemory, "music generation failed") } else resampled;
    defer if (window) gpa.free(pcm);
    const n: c_int = @intCast(pcm.len / 2);
    return .{ .arr = mlx.mlx_array_new_data(pcm.ptr, &[_]c_int{ 1, n, 2 }, 3, .float32) };
}

/// A music request's fields, validated in the order the endpoint has always named them (strings borrow the body;
/// `track_classes` is owned by `gpa`).
fn parse(b: *const job.Body, fsq_available: bool) union(enum) { ok: acestep.MusicRequest, bad: Verdict } {
    const prompt = b.str("prompt") orelse return .{ .bad = bad("missing 'prompt' (style/genre/mood description)") };
    if (prompt.len == 0) return .{ .bad = bad("empty 'prompt'") };
    const lyrics = b.str("lyrics") orelse "";
    const instrumental = b.flag("instrumental");
    if (job.instrumentalConflicts(instrumental, lyrics)) return .{ .bad = bad("'instrumental' is true but 'lyrics' is non-empty — send one or the other") };
    var req: acestep.MusicRequest = .{
        .caption = prompt,
        .lyrics = acestep.resolveLyrics(instrumental, lyrics),
        .language = b.str("vocal_language") orelse "en",
        .keyscale = b.str("keyscale") orelse "",
        .timesignature = b.str("timesignature") orelse "",
        .seed = b.uint("seed") orelse 42,
    };
    if (b.get("bpm")) |v| if (v == .integer) {
        if (v.integer < 30 or v.integer > 300) return .{ .bad = bad("'bpm' must be in [30,300]") };
        req.bpm = @intCast(v.integer);
    };
    const duration = b.uint("duration_seconds") orelse 60;
    if (duration < acestep.MIN_DURATION_S or duration > acestep.MAX_DURATION_S) return .{ .bad = bad("'duration_seconds' must be in [10,600]") };
    req.duration_s = @intCast(duration);
    if (b.str("task")) |t| req.task = std.meta.stringToEnum(acestep.Task, t) orelse return .{ .bad = bad("'task' must be one of text2music, cover, complete") };
    const cs = b.float("cover_strength");
    const cn = b.float("cover_noise_strength");
    req.cover_strength = @floatCast(cs orelse 1.0);
    req.cover_noise_strength = @floatCast(cn orelse 0.0);
    if (req.cover_strength < 0.0 or req.cover_strength > 1.0 or req.cover_noise_strength < 0.0 or req.cover_noise_strength > 1.0)
        return .{ .bad = bad("'cover_strength' and 'cover_noise_strength' must be in [0,1]") };
    if (req.task != .cover and (cs != null or cn != null)) return .{ .bad = bad("'cover_strength' / 'cover_noise_strength' only apply to task \"cover\"") };
    if (b.get("track_classes")) |v| if (v == .array) {
        if (req.task != .complete) return .{ .bad = bad("'track_classes' only applies to task \"complete\"") };
        var names: std.ArrayList([]const u8) = .empty;
        defer names.deinit(gpa);
        for (v.array.items) |it| {
            if (it != .string) return .{ .bad = bad("'track_classes' must be an array of strings") };
            names.append(gpa, it.string) catch return .{ .bad = job.failure(error.OutOfMemory, "music generation failed") };
        }
        req.track_classes = acestep.joinTrackClasses(gpa, names.items) catch
            return .{ .bad = bad("'track_classes' entries must be from: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals") };
    };
    if (req.task == .cover and !fsq_available)
        return .{ .bad = bad("task \"cover\" needs fsq.safetensors beside model.safetensors (this pack predates cover mode — re-download it, or fetch fsq.safetensors from the HF mirror into the model folder)") };
    return .{ .ok = req };
}

fn step(ep: *anyopaque, r: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const self: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, r.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var req = switch (parse(&body, self.e.fsqAvailable())) {
        .ok => |q| q,
        .bad => |v| return v,
    };
    defer if (req.track_classes.len > 0) gpa.free(req.track_classes);
    switch (sourceClip(&body, req.task, self.e.cfg.sample_rate)) {
        .none => {},
        .arr => |x| req.src_audio = x,
        .bad => |v| return v,
    }
    defer if (req.src_audio) |x| {
        _ = mlx.mlx_array_free(x);
    };
    if (req.task != .text2music and req.src_audio == null) return bad("task \"cover\" / \"complete\" needs 'src_audio' (base64 WAV of the source track)");
    switch (referenceClip(&body, self.e.cfg.sample_rate)) {
        .none => {},
        .arr => |x| req.ref_audio = x,
        .bad => |v| return v,
    }
    defer if (req.ref_audio) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const out = self.e.generateWav(gpa, req, job.progressOf(progress)) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        else => job.failure(err, "music generation failed"),
    };
    gpa.free(self.wav);
    self.wav = out;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const self: *Engine = @ptrCast(@alignCast(ep));
    if (self.wav.len == 0) return Verdict.refuse(.failed, "acestep: no finished job to encode");
    out.* = .of(self.wav);
    return .ok;
}

/// A tiny random ACE-Step: DiT hidden 64 x 2 layers, condition encoders at 32 (one layer each), a one-layer Qwen3
/// text encoder over a byte-level tokenizer, and an Oobleck decoder at 64 → 4 channels on the published strides.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const d = 64; // DiT hidden
    const hd = 32; // head_dim (DiT and encoders)
    const enc = 32;
    const th = 32; // text hidden
    const ac = 64; // acoustic latent
    var t = T.init(a, 31);
    try lin(&t, "decoder.condition_embedder", d, enc, true);
    for (0..2) |li| {
        inline for (.{ "self_attn", "cross_attn" }) |at| {
            try t.fill("decoder.layers.{d}." ++ at ++ "_norm.weight", .{li}, &.{d}, .BF16, 1);
            try linF(&t, "decoder.layers.{d}." ++ at ++ ".q_proj", .{li}, 2 * hd, d);
            try linF(&t, "decoder.layers.{d}." ++ at ++ ".k_proj", .{li}, hd, d);
            try linF(&t, "decoder.layers.{d}." ++ at ++ ".v_proj", .{li}, hd, d);
            try linF(&t, "decoder.layers.{d}." ++ at ++ ".o_proj", .{li}, d, 2 * hd);
            try t.fill("decoder.layers.{d}." ++ at ++ ".q_norm.weight", .{li}, &.{hd}, .BF16, 1);
            try t.fill("decoder.layers.{d}." ++ at ++ ".k_norm.weight", .{li}, &.{hd}, .BF16, 1);
        }
        try t.fill("decoder.layers.{d}.mlp_norm.weight", .{li}, &.{d}, .BF16, 1);
        try linF(&t, "decoder.layers.{d}.mlp.gate_proj", .{li}, 128, d);
        try linF(&t, "decoder.layers.{d}.mlp.up_proj", .{li}, 128, d);
        try linF(&t, "decoder.layers.{d}.mlp.down_proj", .{li}, d, 128);
        try t.rand("decoder.layers.{d}.scale_shift_table", .{li}, &.{ 1, 6, d }, .BF16, 0.1);
    }
    try t.fill("decoder.norm_out.weight", .{}, &.{d}, .BF16, 1);
    try t.rand("decoder.proj_in.weight", .{}, &.{ d, 2, 3 * ac }, .BF16, 0.1);
    try t.rand("decoder.proj_in.bias", .{}, &.{d}, .BF16, 0.1);
    try t.rand("decoder.proj_out.weight", .{}, &.{ ac, 2, d }, .BF16, 0.1);
    try t.rand("decoder.proj_out.bias", .{}, &.{ac}, .BF16, 0.1);
    try t.rand("decoder.scale_shift_table", .{}, &.{ 1, 2, d }, .BF16, 0.1);
    inline for (.{ "time_embed", "time_embed_r" }) |te| {
        try lin(&t, "decoder." ++ te ++ ".linear_1", d, 256, true);
        try lin(&t, "decoder." ++ te ++ ".linear_2", d, d, true);
        try lin(&t, "decoder." ++ te ++ ".time_proj", 6 * d, d, true);
    }
    inline for (.{ "lyric_encoder", "timbre_encoder" }, .{ th, ac }) |e, in| {
        try lin(&t, "encoder." ++ e ++ ".embed_tokens", enc, in, true);
        try encLayer(&t, "encoder." ++ e ++ ".layers.0", enc, hd);
        try t.fill("encoder." ++ e ++ ".norm.weight", .{}, &.{enc}, .BF16, 1);
    }
    try lin(&t, "encoder.text_projector", enc, th, false);
    try t.rand("encoder.timbre_encoder.special_token", .{}, &.{ 1, 1, enc }, .BF16, 0.5);
    try t.rand("null_condition_emb", .{}, &.{ 1, 1, enc }, .BF16, 0.5);
    try t.rand("silence_latent", .{}, &.{ 1, 300, ac }, .BF16, 0.5);

    // Text encoder: Qwen3, one layer at 32 (2 heads x 16, 1 KV head).
    var te = T.init(a, 32);
    try te.rand("embed_tokens.weight", .{}, &.{ 320, th }, .BF16, 0.5);
    try te.fill("norm.weight", .{}, &.{th}, .BF16, 1);
    try encLayer(&te, "layers.0", th, 16);

    // VAE decoder: channels 64, 32, 16, 8, 4, 4 over strides 10, 6, 4, 4, 2 (bf16 convs, f32 Snake).
    var v = T.init(a, 33);
    const ch = [_]u64{ 64, 32, 16, 8, 4, 4 };
    const strides = [_]u64{ 10, 6, 4, 4, 2 };
    try v.rand("decoder.conv1.weight", .{}, &.{ ch[0], 7, ac }, .BF16, 0.05);
    try v.rand("decoder.conv1.bias", .{}, &.{ch[0]}, .BF16, 0.1);
    for (strides, 0..) |s, bi| {
        try snake(&v, "decoder.block.{d}.snake1", .{bi}, ch[bi]);
        try v.rand("decoder.block.{d}.conv_t1.weight", .{bi}, &.{ ch[bi + 1], 2 * s, ch[bi] }, .BF16, 0.2);
        try v.rand("decoder.block.{d}.conv_t1.bias", .{bi}, &.{ch[bi + 1]}, .BF16, 0.1);
        for (1..4) |u| {
            try snake(&v, "decoder.block.{d}.res_unit{d}.snake1", .{ bi, u }, ch[bi + 1]);
            try v.rand("decoder.block.{d}.res_unit{d}.conv1.weight", .{ bi, u }, &.{ ch[bi + 1], 7, ch[bi + 1] }, .BF16, 0.2);
            try v.rand("decoder.block.{d}.res_unit{d}.conv1.bias", .{ bi, u }, &.{ch[bi + 1]}, .BF16, 0.1);
            try snake(&v, "decoder.block.{d}.res_unit{d}.snake2", .{ bi, u }, ch[bi + 1]);
            try v.rand("decoder.block.{d}.res_unit{d}.conv2.weight", .{ bi, u }, &.{ ch[bi + 1], 1, ch[bi + 1] }, .BF16, 0.2);
            try v.rand("decoder.block.{d}.res_unit{d}.conv2.bias", .{ bi, u }, &.{ch[bi + 1]}, .BF16, 0.1);
        }
    }
    try snake(&v, "decoder.snake1", .{}, ch[5]);
    try v.rand("decoder.conv2.weight", .{}, &.{ 2, 7, ch[5] }, .BF16, 0.2);
    try v.rand("decoder.conv2.bias", .{}, &.{2}, .BF16, 0.1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"acestep","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
            \\ "head_dim":32,"intermediate_size":128,"encoder_hidden_size":32,"encoder_intermediate_size":64,
            \\ "encoder_num_attention_heads":2,"encoder_num_key_value_heads":1,"num_lyric_encoder_hidden_layers":1,
            \\ "num_timbre_encoder_hidden_layers":1,"text_hidden_dim":32,"audio_acoustic_hidden_dim":64,"in_channels":192,
            \\ "patch_size":2,"sliding_window":16,"timbre_fix_frame":30,"sample_rate":48000,"vae_hop":1920}
            },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "vae.safetensors", .bytes = try v.bytes() },
            .{ .name = "text_encoder/config.json", .bytes =
            \\{"model_type":"qwen3","hidden_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
            \\ "head_dim":16,"rms_norm_eps":1e-06,"rope_theta":1000000.0}
            },
            .{ .name = "text_encoder/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{.{ .id = 300, .content = "<|endoftext|>" }}) },
            .{ .name = "text_encoder/model.safetensors", .bytes = try te.bytes() },
        }),
        .body = "{\"prompt\":\"lofi piano\",\"instrumental\":true,\"duration_seconds\":10,\"seed\":5}",
    };
}

fn lin(t: *sdk.tensors.Tensors, comptime p: []const u8, out: u64, in: u64, bias: bool) !void {
    try t.rand(p ++ ".weight", .{}, &.{ out, in }, .BF16, 0.1);
    if (bias) try t.rand(p ++ ".bias", .{}, &.{out}, .BF16, 0.1);
}

fn linF(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.rand(p ++ ".weight", args, &.{ out, in }, .BF16, 0.1);
}

fn snake(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, ch: u64) !void {
    try t.rand(p ++ ".alpha", args, &.{ch}, .F32, 0.3);
    try t.rand(p ++ ".beta", args, &.{ch}, .F32, 0.3);
}

/// A pre-norm Qwen3-shaped encoder layer at width `w` (2 heads x `hd`, 1 KV head, MLP 2w).
fn encLayer(t: *sdk.tensors.Tensors, comptime p: []const u8, w: u64, hd: u64) !void {
    try t.fill(p ++ ".input_layernorm.weight", .{}, &.{w}, .BF16, 1);
    try t.fill(p ++ ".post_attention_layernorm.weight", .{}, &.{w}, .BF16, 1);
    try lin(t, p ++ ".self_attn.q_proj", 2 * hd, w, false);
    try lin(t, p ++ ".self_attn.k_proj", hd, w, false);
    try lin(t, p ++ ".self_attn.v_proj", hd, w, false);
    try lin(t, p ++ ".self_attn.o_proj", w, 2 * hd, false);
    try t.fill(p ++ ".self_attn.q_norm.weight", .{}, &.{hd}, .BF16, 1);
    try t.fill(p ++ ".self_attn.k_norm.weight", .{}, &.{hd}, .BF16, 1);
    try lin(t, p ++ ".mlp.gate_proj", 2 * w, w, false);
    try lin(t, p ++ ".mlp.up_proj", 2 * w, w, false);
    try lin(t, p ++ ".mlp.down_proj", w, 2 * w, false);
}

// ── tests ──

const testing = std.testing;

test "acestep plugin: a request is validated before any forward, by name" {
    inline for (.{
        .{ "{}", "missing 'prompt' (style/genre/mood description)" },
        .{ "{\"prompt\":\"p\",\"instrumental\":true,\"lyrics\":\"la\"}", "'instrumental' is true but 'lyrics' is non-empty — send one or the other" },
        .{ "{\"prompt\":\"p\",\"bpm\":10}", "'bpm' must be in [30,300]" },
        .{ "{\"prompt\":\"p\",\"duration_seconds\":5}", "'duration_seconds' must be in [10,600]" },
        .{ "{\"prompt\":\"p\",\"task\":\"remix\"}", "'task' must be one of text2music, cover, complete" },
        .{ "{\"prompt\":\"p\",\"cover_strength\":0.5}", "'cover_strength' / 'cover_noise_strength' only apply to task \"cover\"" },
        .{ "{\"prompt\":\"p\",\"track_classes\":[\"drums\"]}", "'track_classes' only applies to task \"complete\"" },
        .{ "{\"prompt\":\"p\",\"task\":\"complete\",\"track_classes\":[\"kazoo\"]}", "'track_classes' entries must be from: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals" },
        .{ "{\"prompt\":\"p\",\"task\":\"cover\"}", "task \"cover\" needs fsq.safetensors beside model.safetensors (this pack predates cover mode — re-download it, or fetch fsq.safetensors from the HF mirror into the model folder)" },
    }) |c| {
        var b = job.Body.parse(testing.allocator, c[0]).?;
        defer b.deinit();
        try testing.expectEqualStrings(c[1], parse(&b, false).bad.why.slice());
    }
}

test "acestep plugin: a seeded song on the tiny pack is a deterministic stereo WAV; a source needs its task" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("task \"cover\" / \"complete\" needs 'src_audio' (base64 WAV of the source track)", (try pack.run("{\"prompt\":\"p\",\"task\":\"complete\"}")).refused.why.slice());
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
