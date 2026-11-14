//! Qwen3-TTS on the media table: `/v1/audio/speech` from text, with zero-shot cloning from a reference WAV on the
//! `-Base` checkpoints (ECAPA speaker encoder) and a `warm_only` pre-embed. The result is a 24 kHz mono WAV, or the
//! warm-up's JSON.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const log = @import("../../sdk/util/log.zig");
const wav = @import("../../sdk/media/wav.zig");
const mlx = @import("../../mlx/ffi.zig");
const tts = @import("tts.zig");
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

/// Frames one request may generate (12.5 per second: ~164 s).
const max_frames = 2048;

const Engine = struct {
    synth: tts.Synthesizer,
    /// The last job's result: a WAV, or the warm-up's JSON (static).
    out: []const u8 = &.{},
    owned: bool = false,

    fn setOut(e: *Engine, bytes: []const u8, owned: bool) void {
        if (e.owned) gpa.free(e.out);
        e.out = bytes;
        e.owned = owned;
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    var synth = tts.Synthesizer.load(io, gpa, .{ .ctx = ctx.stream.ctx }, ctx.model_dir.slice()) catch {
        out.* = Verdict.refuse(.failed, "qwen3_tts: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        synth.deinit();
        out.* = Verdict.refuse(.failed, "qwen3_tts: out of memory");
        return null;
    };
    e.* = .{ .synth = synth };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.setOut(&.{}, false);
    e.synth.deinit();
    gpa.destroy(e);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

const no_voices = "this model has no built-in voices: leave 'voice' out (a Base pack clones 'ref_audio', a VoiceDesign pack takes 'instructions')";
const unknown_voice = "'voice' must be one of this model's voices";
const no_instructions = "this model takes no 'instructions' (VoiceDesign and 1.7B CustomVoice checkpoints do)";
const needs_instructions = "a VoiceDesign model needs 'instructions' describing the voice";

/// `ref_audio` as samples: a base64 24 kHz mono PCM16 WAV (the app normalizes it). Null when absent or empty.
fn refSamples(b: *const job.Body) union(enum) { none, samples: []f32, bad_base64, bad_wav } {
    const b64 = b.str("ref_audio") orelse return .none;
    if (b64.len == 0) return .none;
    const bytes = job.base64Decode(gpa, b64) catch return .bad_base64;
    defer gpa.free(bytes);
    return .{ .samples = wav.decodePcm16Mono(gpa, bytes) catch return .bad_wav };
}

/// `language`: absent or `auto` (null) lets the model infer it from the text; otherwise a name the checkpoint declares.
fn languageOf(e: *const Engine, b: *const job.Body) error{UnsupportedLanguage}!?i32 {
    const v = b.get("language") orelse return null;
    if (v == .null) return null;
    if (v != .string) return error.UnsupportedLanguage;
    const name = v.string;
    if (std.ascii.eqlIgnoreCase(name, "auto")) return null;
    return e.synth.model.cfg.languageId(name) orelse error.UnsupportedLanguage;
}

/// `voice` (or `speaker`): a CustomVoice checkpoint's built-in voice, required there; elsewhere only absent, empty or
/// `default`, since no other pack has voices to pick.
fn voiceOf(cfg: *const tts.TtsConfig, b: *const job.Body) union(enum) { voice: ?*const tts.TtsConfig.Speaker, no_voices, unknown: []const u8 } {
    const v = b.get("voice") orelse b.get("speaker") orelse std.json.Value{ .null = {} };
    const name: []const u8 = switch (v) {
        .null => "",
        .string => |n| n,
        else => "?",
    };
    if (cfg.kind == .custom_voice) return if (cfg.speaker(name)) |sp| .{ .voice = sp } else .{ .unknown = name };
    return if (name.len == 0 or std.mem.eql(u8, name, "default")) .{ .voice = null } else .no_voices;
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    if (body.flag("warm_only")) return warm(e, &body);

    const text = body.str("input") orelse body.str("text") orelse return bad("missing 'input'");
    if (text.len == 0) return bad("empty 'input'");
    // A reference that cannot be used is refused, never answered in a different voice.
    var ref: ?[]f32 = null;
    defer if (ref) |r| gpa.free(r);
    switch (refSamples(&body)) {
        .none => {},
        .samples => |smp| if (e.synth.supportsCloning()) {
            ref = smp;
        } else {
            gpa.free(smp);
            return bad("ref_audio requires a model with a speaker encoder");
        },
        .bad_base64 => return bad("ref_audio is not valid base64"),
        .bad_wav => return bad("ref_audio is not a decodable WAV"),
    }
    const language = languageOf(e, &body) catch return bad("language is not one this model supports (auto, or a name in its codec_language_id)");
    const cfg = &e.synth.model.cfg;
    const speaker = switch (voiceOf(cfg, &body)) {
        .voice => |sp| sp,
        .no_voices => return bad(no_voices),
        .unknown => |name| {
            var names: [tts.TtsConfig.max_speakers * 26]u8 = undefined;
            var w = std.Io.Writer.fixed(&names);
            for (cfg.speakers[0..cfg.n_speakers], 0..) |*sp, i| w.print("{s}{s}", .{ if (i == 0) "" else ", ", sp.name[0..sp.len] }) catch break;
            return req.refuseFmt(.unsupported, unknown_voice, "'voice' must be one of this model's voices ({s}), got '{s}'", .{ w.buffered(), name });
        },
    };
    const instruct: ?[]const u8 = if (body.get("instructions")) |v| switch (v) {
        .null => null,
        .string => |text_| if (text_.len == 0) null else text_,
        else => return bad("'instructions' must be a string"),
    } else null;
    if (instruct != null and !cfg.takesInstruct()) return bad(no_instructions);
    if (instruct == null and cfg.kind == .voice_design) return bad(needs_instructions);
    const out = e.synth.synthesizeWav(text, max_frames, job.progressOf(progress), .{
        .ref_audio = ref,
        .speaker = speaker,
        .instruct = instruct,
        .language = language,
        .seed = body.uint("seed") orelse 0,
    }) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        else => job.failure(err, "synthesis failed"),
    };
    e.setOut(out, true);
    return .ok;
}

/// Embeds and caches the clone voice without synthesizing, so the first sentence of a voice session skips the cold
/// ECAPA forward. Misuse is refused by name, never a silent no-op.
fn warm(e: *Engine, body: *const job.Body) Verdict {
    if (!e.synth.supportsCloning()) return bad("this model has no speaker encoder; nothing to warm");
    const samples = switch (refSamples(body)) {
        .none => return bad("warm_only requires ref_audio"),
        .samples => |smp| smp,
        .bad_base64 => return bad("ref_audio is not valid base64"),
        .bad_wav => return bad("ref_audio is not a decodable WAV"),
    };
    defer gpa.free(samples);
    const was_cached = e.synth.warmSpeaker(samples) catch |err| return job.failure(err, "speaker embedding failed");
    e.setOut(if (was_cached) "{\"warmed\":true,\"cache\":\"hit\"}" else "{\"warmed\":true,\"cache\":\"miss\"}", false);
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.out.len == 0) return Verdict.refuse(.failed, "qwen3_tts: no finished job to encode");
    out.* = .of(e.out);
    return .ok;
}

/// A tiny random Qwen3-TTS: one talker layer and one predictor layer at hidden 32, a byte-level tokenizer, and a
/// speech-tokenizer decoder at latent 16 / decoder 32 with the published upsampling (1920 samples a frame). The
/// codec head's bias makes EOS likely within a few frames, so a seeded job stays short.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    var t = T.init(a, 21);
    const h = 32; // talker = predictor hidden (no small_to_mtp projection)
    const th = 16; // text hidden
    const codec_vocab = 3072;
    try t.rand("talker.model.text_embedding.weight", .{}, &.{ 151936, th }, .BF16, 0.5);
    try t.rand("talker.model.codec_embedding.weight", .{}, &.{ codec_vocab, h }, .BF16, 0.5);
    try lin(&t, "talker.text_projection.linear_fc1", th, th, true);
    try lin(&t, "talker.text_projection.linear_fc2", h, th, true);
    try t.fill("talker.model.norm.weight", .{}, &.{h}, .BF16, 1);
    try t.rand("talker.codec_head.weight", .{}, &.{ codec_vocab, h }, .BF16, 0.02);
    var bias: [codec_vocab]f32 = @splat(0);
    bias[2150] = 2.5; // codec_eos
    try t.values("talker.codec_head.bias", .{}, &.{codec_vocab}, .BF16, &bias);
    try qwenLayer(&t, "talker.model.layers.0", h);
    try t.fill("talker.code_predictor.model.norm.weight", .{}, &.{h}, .BF16, 1);
    try qwenLayer(&t, "talker.code_predictor.model.layers.0", h);
    for (0..15) |i| {
        try t.rand("talker.code_predictor.model.codec_embedding.{d}.weight", .{i}, &.{ 2048, h }, .BF16, 0.5);
        try t.rand("talker.code_predictor.lm_head.{d}.weight", .{i}, &.{ 2048, h }, .BF16, 0.1);
    }

    // Speech tokenizer: latent 16, codebook 16 (RVQ dim 8), decoder 32 → 16 → 8 → 4 → 2.
    var c = T.init(a, 22);
    const lat = 16;
    const cbd = 16;
    const hd = 16; // pre-transformer hidden
    const dd = 32;
    inline for (.{ "rvq_first", "rvq_rest" }, .{ 1, 15 }) |q, n| {
        for (0..n) |i| {
            try c.rand("decoder.quantizer." ++ q ++ ".vq.layers.{d}._codebook.embedding_sum", .{i}, &.{ 2048, cbd / 2 }, .F32, 1);
            try c.fill("decoder.quantizer." ++ q ++ ".vq.layers.{d}._codebook.cluster_usage", .{i}, &.{2048}, .F32, 1);
        }
        try c.rand("decoder.quantizer." ++ q ++ ".output_proj.weight", .{}, &.{ cbd, cbd / 2, 1 }, .F32, 0.3);
    }
    try conv(&c, "decoder.pre_conv.conv", lat, cbd, 3);
    try lin(&c, "decoder.pre_transformer.input_proj", hd, lat, true);
    try lin(&c, "decoder.pre_transformer.output_proj", lat, hd, true);
    try c.fill("decoder.pre_transformer.norm.weight", .{}, &.{hd}, .F32, 1);
    {
        const p = "decoder.pre_transformer.layers.0.";
        try c.fill(p ++ "input_layernorm.weight", .{}, &.{hd}, .F32, 1);
        try c.fill(p ++ "post_attention_layernorm.weight", .{}, &.{hd}, .F32, 1);
        inline for (.{ "q", "k", "v" }) |n| try c.rand(p ++ "self_attn." ++ n ++ "_proj.weight", .{}, &.{ 16, hd }, .F32, 0.2);
        try c.rand(p ++ "self_attn.o_proj.weight", .{}, &.{ hd, 16 }, .F32, 0.2);
        try c.fill(p ++ "self_attn_layer_scale.scale", .{}, &.{hd}, .F32, 0.01);
        try c.fill(p ++ "mlp_layer_scale.scale", .{}, &.{hd}, .F32, 0.01);
        try c.rand(p ++ "mlp.gate_proj.weight", .{}, &.{ 32, hd }, .F32, 0.2);
        try c.rand(p ++ "mlp.up_proj.weight", .{}, &.{ 32, hd }, .F32, 0.2);
        try c.rand(p ++ "mlp.down_proj.weight", .{}, &.{ hd, 32 }, .F32, 0.2);
    }
    for (0..2) |i| {
        try c.rand("decoder.upsample.{d}.0.conv.weight", .{i}, &.{ lat, lat, 2 }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.0.conv.bias", .{i}, &.{lat}, .F32, 0.1);
        try c.rand("decoder.upsample.{d}.1.dwconv.conv.weight", .{i}, &.{ lat, 1, 7 }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.1.dwconv.conv.bias", .{i}, &.{lat}, .F32, 0.1);
        try c.fill("decoder.upsample.{d}.1.norm.weight", .{i}, &.{lat}, .F32, 1);
        try c.fill("decoder.upsample.{d}.1.norm.bias", .{i}, &.{lat}, .F32, 0);
        try c.rand("decoder.upsample.{d}.1.pwconv1.weight", .{i}, &.{ 4 * lat, lat }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.1.pwconv1.bias", .{i}, &.{4 * lat}, .F32, 0.1);
        try c.rand("decoder.upsample.{d}.1.pwconv2.weight", .{i}, &.{ lat, 4 * lat }, .F32, 0.2);
        try c.rand("decoder.upsample.{d}.1.pwconv2.bias", .{i}, &.{lat}, .F32, 0.1);
        try c.fill("decoder.upsample.{d}.1.gamma", .{i}, &.{lat}, .F32, 0.1);
    }
    try conv(&c, "decoder.decoder.0.conv", dd, lat, 7);
    const rates = [_]u64{ 8, 5, 4, 3 };
    for (rates, 1..) |r, bi| {
        const in = @as(u64, dd) >> @intCast(bi - 1);
        const out = @as(u64, dd) >> @intCast(bi);
        try snake(&c, "decoder.decoder.{d}.block.0", .{bi}, in);
        try c.rand("decoder.decoder.{d}.block.1.conv.weight", .{bi}, &.{ in, out, 2 * r }, .F32, 0.2);
        try c.rand("decoder.decoder.{d}.block.1.conv.bias", .{bi}, &.{out}, .F32, 0.1);
        for (2..5) |u| {
            try snake(&c, "decoder.decoder.{d}.block.{d}.act1", .{ bi, u }, out);
            try c.rand("decoder.decoder.{d}.block.{d}.conv1.conv.weight", .{ bi, u }, &.{ out, out, 7 }, .F32, 0.2);
            try c.rand("decoder.decoder.{d}.block.{d}.conv1.conv.bias", .{ bi, u }, &.{out}, .F32, 0.1);
            try snake(&c, "decoder.decoder.{d}.block.{d}.act2", .{ bi, u }, out);
            try c.rand("decoder.decoder.{d}.block.{d}.conv2.conv.weight", .{ bi, u }, &.{ out, out, 1 }, .F32, 0.2);
            try c.rand("decoder.decoder.{d}.block.{d}.conv2.conv.bias", .{ bi, u }, &.{out}, .F32, 0.1);
        }
    }
    try snake(&c, "decoder.decoder.5", .{}, dd >> 4);
    try conv(&c, "decoder.decoder.6.conv", 1, dd >> 4, 7);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"qwen3_tts","talker_config":{"hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,
            \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"vocab_size":3072,"text_vocab_size":151936,
            \\ "code_predictor_config":{"hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,"num_attention_heads":2,
            \\ "num_key_value_heads":1,"head_dim":16,"num_code_groups":16},
            \\ "codec_think_id":2154,"codec_language_id":{"english":2050,"portuguese":2071,"beijing_dialect":2074}}}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{}) },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "speech_tokenizer/config.json", .bytes =
            \\{"decoder_config":{"latent_dim":16,"codebook_dim":16,"decoder_dim":32,"hidden_size":16,"num_hidden_layers":1,
            \\ "num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"intermediate_size":32,"num_quantizers":16,
            \\ "num_semantic_quantizers":1,"upsample_rates":[8,5,4,3]}}
            },
            .{ .name = "speech_tokenizer/model.safetensors", .bytes = try c.bytes() },
        }),
        .body = "{\"input\":\"hi there\"}",
    };
}

fn lin(t: *sdk.tensors.Tensors, comptime p: []const u8, out: u64, in: u64, bias: bool) !void {
    try t.rand(p ++ ".weight", .{}, &.{ out, in }, .BF16, 0.2);
    if (bias) try t.rand(p ++ ".bias", .{}, &.{out}, .BF16, 0.1);
}

fn conv(t: *sdk.tensors.Tensors, comptime p: []const u8, out: u64, in: u64, k: u64) !void {
    try t.rand(p ++ ".weight", .{}, &.{ out, in, k }, .F32, 0.2);
    try t.rand(p ++ ".bias", .{}, &.{out}, .F32, 0.1);
}

fn snake(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, ch: u64) !void {
    try t.rand(p ++ ".alpha", args, &.{ch}, .F32, 0.3);
    try t.rand(p ++ ".beta", args, &.{ch}, .F32, 0.3);
}

/// A Qwen3 decoder layer at hidden `h` (2 heads x 16, 1 KV head, MLP 64), bf16.
fn qwenLayer(t: *sdk.tensors.Tensors, comptime p: []const u8, h: u64) !void {
    try t.fill(p ++ ".input_layernorm.weight", .{}, &.{h}, .BF16, 1);
    try t.fill(p ++ ".post_attention_layernorm.weight", .{}, &.{h}, .BF16, 1);
    try lin(t, p ++ ".self_attn.q_proj", 32, h, false);
    try lin(t, p ++ ".self_attn.k_proj", 16, h, false);
    try lin(t, p ++ ".self_attn.v_proj", 16, h, false);
    try lin(t, p ++ ".self_attn.o_proj", h, 32, false);
    try t.fill(p ++ ".self_attn.q_norm.weight", .{}, &.{16}, .BF16, 1);
    try t.fill(p ++ ".self_attn.k_norm.weight", .{}, &.{16}, .BF16, 1);
    try lin(t, p ++ ".mlp.gate_proj", 64, h, false);
    try lin(t, p ++ ".mlp.up_proj", 64, h, false);
    try lin(t, p ++ ".mlp.down_proj", h, 64, false);
}

// ── tests ──

const testing = std.testing;

test "qwen3_tts plugin: a seeded job on the tiny pack is a deterministic WAV; misuse is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expect(first.len > 44);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("this model has no speaker encoder; nothing to warm", (try pack.run("{\"warm_only\":true}")).refused.why.slice());
    try testing.expectEqualStrings("missing 'input'", (try pack.run("{}")).refused.why.slice());
}

test "qwen3_tts plugin: an unusable ref_audio is refused by name, not synthesized in the plain voice" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try MediaPack.open(al);
    defer pack.close();
    const std_b64 = std.base64.standard.Encoder;
    const tone = try wav.encodePcm16Mono(al, &[_]f32{ 0, 0.25, -0.25, 0 }, 24000);
    const good = try al.alloc(u8, std_b64.calcSize(tone.len));
    _ = std_b64.encode(good, tone);
    const junk = try al.alloc(u8, std_b64.calcSize(8));
    _ = std_b64.encode(junk, "notawave");
    const cases = [_]struct { ref: []const u8, why: []const u8 }{
        .{ .ref = "%%%", .why = "ref_audio is not valid base64" },
        .{ .ref = junk, .why = "ref_audio is not a decodable WAV" },
        .{ .ref = good, .why = "ref_audio requires a model with a speaker encoder" },
    };
    for (cases) |c| {
        const body = try std.fmt.allocPrint(al, "{{\"input\":\"hi\",\"ref_audio\":\"{s}\"}}", .{c.ref});
        try testing.expectEqualStrings(c.why, (try pack.run(body)).refused.why.slice());
    }
    try testing.expectEqualStrings("RIFF", (try pack.run("{\"input\":\"hi\",\"ref_audio\":\"\"}")).ok[0..4]);
}

test "qwen3_tts plugin: a named language conditions the codec prefix; auto is the plain path; others are refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try MediaPack.open(al);
    defer pack.close();
    const plain = try al.dupe(u8, (try pack.run("{\"input\":\"hi there\"}")).ok);
    try testing.expectEqualSlices(u8, plain, (try pack.run("{\"input\":\"hi there\",\"language\":\"Auto\"}")).ok);
    const pt = (try pack.run("{\"input\":\"hi there\",\"language\":\"Portuguese\"}")).ok;
    try testing.expectEqualStrings("RIFF", pt[0..4]);
    try testing.expect(!std.mem.eql(u8, plain, pt));
    const why = "language is not one this model supports (auto, or a name in its codec_language_id)";
    try testing.expectEqualStrings(why, (try pack.run("{\"input\":\"hi\",\"language\":\"klingon\"}")).refused.why.slice());
    try testing.expectEqualStrings(why, (try pack.run("{\"input\":\"hi\",\"language\":\"beijing_dialect\"}")).refused.why.slice());
    try testing.expectEqualStrings(why, (try pack.run("{\"input\":\"hi\",\"language\":2071}")).refused.why.slice());
}

/// The tiny pack as another checkpoint kind: `top` joins the config's root keys, `talker` its `talker_config`.
fn openAs(a: std.mem.Allocator, comptime top: []const u8, comptime talker: []const u8) !MediaPack {
    var fx = try conformanceFixture(a);
    const files = try a.dupe(sdk.fixture.File, fx.files);
    for (files) |*f| if (std.mem.eql(u8, f.name, "config.json")) {
        const rooted = try std.mem.replaceOwned(u8, a, f.bytes, "{\"model_type\":\"qwen3_tts\",", "{\"model_type\":\"qwen3_tts\"," ++ top);
        f.bytes = try std.mem.replaceOwned(u8, a, rooted, "\"talker_config\":{", "\"talker_config\":{" ++ talker);
    };
    fx.files = files;
    return MediaPack.openFixture(a, fx);
}

const custom_voice_talker = "\"spk_id\":{\"ryan\":2861,\"dylan\":2878},\"spk_is_dialect\":{\"ryan\":false,\"dylan\":\"beijing_dialect\"},";

fn wavOf(a: std.mem.Allocator, pack: *MediaPack, body: []const u8) ![]const u8 {
    const out = try pack.run(body);
    if (out == .refused) {
        std.debug.print("refused: {s} ({s})\n", .{ out.refused.why.slice(), body });
        return error.Refused;
    }
    try testing.expectEqualStrings("RIFF", out.ok[0..4]);
    return a.dupe(u8, out.ok);
}

fn whyOf(pack: *MediaPack, body: []const u8) ![]const u8 {
    const out = try pack.run(body);
    try testing.expect(out == .refused);
    return out.refused.why.slice();
}

test "qwen3_tts plugin: a seed picks the draw" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try MediaPack.open(al);
    defer pack.close();
    const zero = try wavOf(al, &pack, "{\"input\":\"hi there\"}");
    try testing.expectEqualSlices(u8, zero, try wavOf(al, &pack, "{\"input\":\"hi there\",\"seed\":0}"));
    const seeded = try wavOf(al, &pack, "{\"input\":\"hi there\",\"seed\":7}");
    try testing.expect(!std.mem.eql(u8, zero, seeded));
    try testing.expectEqualSlices(u8, seeded, try wavOf(al, &pack, "{\"input\":\"hi there\",\"seed\":7}"));
}

test "qwen3_tts plugin: a Base pack refuses a voice name and instructions by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try MediaPack.open(al);
    defer pack.close();
    const plain = try wavOf(al, &pack, "{\"input\":\"hi there\"}");
    try testing.expectEqualSlices(u8, plain, try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"default\"}"));
    try testing.expectEqualSlices(u8, plain, try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"\",\"instructions\":\"\"}"));
    for ([_][]const u8{ "{\"input\":\"hi\",\"voice\":\"ryan\"}", "{\"input\":\"hi\",\"speaker\":\"ryan\"}", "{\"input\":\"hi\",\"voice\":7}" }) |body|
        try testing.expectEqualStrings(no_voices, try whyOf(&pack, body));
    try testing.expectEqualStrings(no_instructions, try whyOf(&pack, "{\"input\":\"hi\",\"instructions\":\"Speak slowly.\"}"));
}

test "qwen3_tts plugin: a CustomVoice pack speaks in the named voice, styled by instructions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try openAs(al, "\"tts_model_type\":\"custom_voice\",\"tts_model_size\":\"1b7\",", custom_voice_talker);
    defer pack.close();
    const ryan = try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"Ryan\"}");
    try testing.expectEqualSlices(u8, ryan, try wavOf(al, &pack, "{\"input\":\"hi there\",\"speaker\":\"ryan\"}"));
    const dylan = try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"dylan\"}");
    try testing.expect(!std.mem.eql(u8, ryan, dylan));
    const angry = try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"ryan\",\"instructions\":\"Speak very angrily and loudly.\"}");
    try testing.expect(!std.mem.eql(u8, ryan, angry));
    // A dialect voice speaks its dialect under auto; a named language is kept.
    try testing.expect(!std.mem.eql(u8, dylan, try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"dylan\",\"language\":\"english\"}")));

    try testing.expectEqualStrings(unknown_voice, try whyOf(&pack, "{\"input\":\"hi\"}"));
    try testing.expectEqualStrings(unknown_voice, try whyOf(&pack, "{\"input\":\"hi\",\"voice\":\"alloy\"}"));
    try testing.expectEqualStrings("'instructions' must be a string", try whyOf(&pack, "{\"input\":\"hi\",\"voice\":\"ryan\",\"instructions\":3}"));
}

test "qwen3_tts plugin: a 0.6B CustomVoice pack refuses instructions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try openAs(al, "\"tts_model_type\":\"custom_voice\",\"tts_model_size\":\"0b6\",", custom_voice_talker);
    defer pack.close();
    _ = try wavOf(al, &pack, "{\"input\":\"hi there\",\"voice\":\"ryan\"}");
    try testing.expectEqualStrings(no_instructions, try whyOf(&pack, "{\"input\":\"hi\",\"voice\":\"ryan\",\"instructions\":\"Speak slowly.\"}"));
}

test "qwen3_tts plugin: a VoiceDesign pack speaks in the described voice and needs the description" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();
    var pack = try openAs(al, "\"tts_model_type\":\"voice_design\",\"tts_model_size\":\"1b7\",", "\"spk_id\":{},\"spk_is_dialect\":{},");
    defer pack.close();
    const old = try wavOf(al, &pack, "{\"input\":\"hi there\",\"instructions\":\"A deep, gravelly voice of an elderly man.\"}");
    const young = try wavOf(al, &pack, "{\"input\":\"hi there\",\"instructions\":\"A bright, energetic young woman.\"}");
    try testing.expect(!std.mem.eql(u8, old, young));
    try testing.expectEqualStrings(needs_instructions, try whyOf(&pack, "{\"input\":\"hi\"}"));
    try testing.expectEqualStrings(needs_instructions, try whyOf(&pack, "{\"input\":\"hi\",\"instructions\":\"\"}"));
    try testing.expectEqualStrings(no_voices, try whyOf(&pack, "{\"input\":\"hi\",\"voice\":\"ryan\",\"instructions\":\"A calm man.\"}"));
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
