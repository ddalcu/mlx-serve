//! NVIDIA Nemotron 3.5 Transcribe (`nemo_speechlm`, mlx-audio's pack layout): the claim, the trunk's parse through the
//! nemotron_h plugin (the NeMo `llm_config`), and the perception module's geometry. MLX-free.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const json = @import("../../sdk/config/json.zig");
const nh = @import("../nemotron_h/config.zig");
const ctc_align = @import("ctc_align.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "nemotron_transcribe",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

/// NVIDIA's name and mlx-audio's: both write the same tensor names.
pub const model_types = [_][]const u8{ "nemo_speechlm", "nemotron_transcribe" };

pub const claims = sdk.claimsByType(&model_types);

pub const freeParsed = nh.freeParsed;

/// What `prepare` sees of the clip before a load: the checkpoint's sample rate, mel hop and stacking factor, which
/// `parse` insists on.
pub const SAMPLE_RATE: u32 = 16000;
pub const HOP: u32 = 160;
pub const STACK: u32 = 8;

/// `<|audio|>` is id `vocab_size`, past the embedding table, so the trunk's gather reads this reserved token
/// (`<SPECIAL_999>`) in its rows instead; a pack too small for it takes its last id.
pub const SPLICE_ID: u32 = 999;

pub fn spliceId(vocab: u32) u32 {
    return @min(SPLICE_ID, vocab -| 1);
}

/// One feature-stacking Transformer encoder.
pub const Encoder = struct {
    d_model: u32,
    heads: u32,
    layers: u32,
    ff: u32,
    qk_norm: bool,
    rope_base: f32,
    rotary_fraction: f32,
};

/// The speaker cache's compression policy and the streaming windows (encoder rows unless noted).
pub const Streaming = struct {
    /// Mel frames per chunk is `chunk * stack`; the ASR and speaker encoders read `left`/`right` rows of context.
    chunk: u32 = 500,
    left: u32 = 50,
    right: u32 = 50,
    fifo: u32 = 40,
    cache: u32 = 188,
    period: u32 = 300,
    silence_per_speaker: u32 = 1,
    score_threshold: f32 = 0.25,
    latest_boost: f32 = 0.05,
    strong_boost_rate: f32 = 0.75,
    weak_boost_rate: f32 = 1.5,
    min_positive_rate: f32 = 0.5,
};

/// The CTC timestamp head (`ctc_timestamps.decoder_config`): the ASR rows upsampled `upsample` x by a learned
/// transposed convolution, a Transformer bridge scaled by `residual` and added back, a pointwise projection onto
/// `classes` tokens plus the blank (the last column).
pub const Ctc = struct {
    enc: Encoder,
    /// NVIDIA's aligner constants, the checkpoint's `ctc_timestamps.alignment_config` where it names them.
    timing: ctc_align.Params = .{},
    classes: u32,
    upsample: u32,
    residual: f32,
};

pub const Audio = struct {
    rate: u32,
    mels: u32,
    n_fft: u32,
    win: u32,
    hop: u32,
    preemphasis: f32,
    stack: u32,
    speakers: u32,
    /// Speaker activity is thresholded (`threshold`) before it is fused; else the probabilities go in as they are.
    thresholded: bool,
    threshold: f32,
    kernel_scale: f32,
    out_dim: u32,
    asr: Encoder,
    spk: Encoder,
    stream: Streaming,
    /// Null when the pack has no head or one the port does not read.
    ctc: ?Ctc = null,
};

const Error = error{ Invalid, Unsupported };

fn num(o: json.Object, key: []const u8, default: f64) Error!f64 {
    return (json.floatField(o, key) catch return error.Invalid) orelse default;
}

fn uint(o: json.Object, key: []const u8, default: u32) Error!u32 {
    return (json.intField(o, key) catch return error.Invalid) orelse default;
}

fn expectString(o: json.Object, key: []const u8, want: []const u8) Error!void {
    const v = json.field(o, key) orelse return;
    if (v == .null) return;
    if (json.string(v)) |s| if (std.mem.eql(u8, s, want)) return;
    return error.Unsupported;
}

fn encoder(o: json.Object) Error!Encoder {
    try expectString(o, "subsampling", "feature_stacking");
    try expectString(o, "self_attention_model", "rope");
    try expectString(o, "attn_mode", "full");
    if (json.boolean(json.field(o, "qkv_bias")) orelse false) return error.Unsupported;
    if (json.boolean(json.field(o, "xscaling")) orelse false) return error.Unsupported;
    if (!(json.boolean(json.field(o, "pre_block_norm")) orelse true)) return error.Unsupported;
    const e: Encoder = .{
        .d_model = try uint(o, "d_model", 0),
        .heads = try uint(o, "n_heads", 0),
        .layers = try uint(o, "n_layers", 0),
        .ff = @intFromFloat(try num(o, "d_model", 0) * try num(o, "ff_expansion", 4)),
        .qk_norm = json.boolean(json.field(o, "qk_norm")) orelse false,
        .rope_base = @floatCast(try num(o, "rope_base", 10000)),
        .rotary_fraction = @floatCast(try num(o, "rotary_fraction", 1)),
    };
    if (e.d_model == 0 or e.heads == 0 or e.d_model % e.heads != 0 or e.layers == 0) return error.Invalid;
    return e;
}

/// The CTC head as the config states it; null for a head the port does not run.
fn parseCtc(root: json.Object, asr_width: u32) ?Ctc {
    const d = json.object(json.field(json.object(json.field(root, "ctc_timestamps")) orelse return null, "decoder_config")) orelse return null;
    const ok = (json.boolean(json.field(d, "use_transformer")) orelse false) and (json.boolean(json.field(d, "residual")) orelse false);
    if (!ok or (json.boolean(json.field(d, "learnable_residual_scale")) orelse false)) return null;
    expectString(d, "upsampling_mode", "conv_transpose") catch return null;
    const width = uint(d, "feat_in", 0) catch return null;
    const heads = uint(d, "n_heads", 0) catch return null;
    const layers = uint(d, "n_layers", 0) catch return null;
    const upsample = uint(d, "upsampling_factor", 1) catch return null;
    const classes = uint(d, "num_classes", 0) catch return null;
    if (width != asr_width or heads == 0 or width % heads != 0 or layers == 0 or upsample == 0 or classes == 0) return null;
    const e: Encoder = .{
        .d_model = width,
        .heads = heads,
        .layers = layers,
        .ff = @intFromFloat(@as(f64, @floatFromInt(width)) * (num(d, "ff_expansion", 4) catch return null)),
        .qk_norm = json.boolean(json.field(d, "qk_norm")) orelse false,
        .rope_base = @floatCast(num(d, "rope_base", 10000) catch return null),
        .rotary_fraction = @floatCast(num(d, "rotary_fraction", 1) catch return null),
    };
    expectString(d, "self_attention_model", "rope") catch return null;
    expectString(d, "attn_mode", "full") catch return null;
    if (json.boolean(json.field(d, "qkv_bias")) orelse false) return null;
    if (json.boolean(json.field(d, "xscaling")) orelse false) return null;
    if (!(json.boolean(json.field(d, "pre_block_norm")) orelse true)) return null;
    var params: ctc_align.Params = .{ .max_word_frames = 12 * upsample };
    if (json.object(json.field(json.object(json.field(root, "ctc_timestamps")).?, "alignment_config"))) |ac| if (json.object(json.field(ac, "parameters"))) |ps| {
        params.speaker_logprob_weight = @floatCast(num(ps, "speaker_logprob_weight", params.speaker_logprob_weight) catch return null);
        params.speaker_activity_threshold = @floatCast(num(ps, "speaker_activity_threshold", params.speaker_activity_threshold) catch return null);
        params.support_threshold = @floatCast(num(ps, "support_activity_threshold", params.support_threshold) catch return null);
        params.maximum_token_len = num(ps, "maximum_token_len", params.maximum_token_len) catch return null;
        params.max_word_frames = (uint(ps, "max_word_frames", params.max_word_frames / upsample) catch return null) * upsample;
    };
    return .{ .enc = e, .timing = params, .classes = classes, .upsample = upsample, .residual = @floatCast(num(d, "residual_scale", 1) catch return null) };
}

/// The perception module as the checkpoint's own config states it (`perception`, `pe_encoder_config`).
pub fn parseAudio(root: json.Object) Error!Audio {
    const perception = json.object(json.field(root, "perception")) orelse return error.Invalid;
    const pe = json.object(json.field(root, "pe_encoder_config")) orelse return error.Invalid;
    const pre = json.object(json.field(perception, "preprocessor")) orelse return error.Invalid;
    const asr = json.object(json.field(pe, "asr_encoder_cfg")) orelse return error.Invalid;
    const dm = json.object(json.field(pe, "diarization_model_cfg")) orelse return error.Invalid;
    const spk = json.object(json.field(dm, "encoder")) orelse return error.Invalid;
    const sm = json.object(json.field(dm, "sortformer_modules")) orelse return error.Invalid;

    try expectString(pre, "window", "hann");
    if (json.field(pre, "normalize")) |v| if (v != .null and !std.mem.eql(u8, json.string(v) orelse "", "NA")) return error.Unsupported;
    if (!(json.boolean(json.field(pre, "log")) orelse true) or (try uint(pre, "frame_splicing", 1)) != 1) return error.Unsupported;
    try expectString(pe, "asr_normalize_type", "per_feature");
    if (json.int(json.field(pe, "speaker_feature_config_version")) != 1) return error.Unsupported;
    if (!(json.boolean(json.field(dm, "high_resolution")) orelse false)) return error.Unsupported;

    const rate = try uint(pre, "sample_rate", 16000);
    const hop: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(rate)) * try num(pre, "window_stride", 0.01)));
    const win: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(rate)) * try num(pre, "window_size", 0.025)));
    const mode = json.string(json.field(pe, "speaker_feature_mode")) orelse return error.Unsupported;
    const thresholded = std.mem.eql(u8, mode, "thresholded");
    if (!thresholded and !std.mem.eql(u8, mode, "continuous")) return error.Unsupported;

    var st: Streaming = .{
        .chunk = try uint(pe, "online_inference_length", 500),
        .left = try uint(pe, "chunk_left_context", 50),
        .right = try uint(pe, "chunk_right_context", 50),
        .fifo = try uint(pe, "diar_fifo_len", 40),
        .cache = try uint(pe, "diar_spkcache_len", 188),
        .period = try uint(pe, "diar_spkcache_update_period", 300),
        .silence_per_speaker = try uint(sm, "spkcache_sil_frames_per_spk", 1),
    };
    st.score_threshold = @floatCast(try num(sm, "pred_score_threshold", st.score_threshold));
    st.latest_boost = @floatCast(try num(sm, "scores_boost_latest", st.latest_boost));
    st.strong_boost_rate = @floatCast(try num(sm, "strong_boost_rate", st.strong_boost_rate));
    st.weak_boost_rate = @floatCast(try num(sm, "weak_boost_rate", st.weak_boost_rate));
    st.min_positive_rate = @floatCast(try num(sm, "min_pos_scores_rate", st.min_positive_rate));

    var a: Audio = .{
        .rate = rate,
        .mels = try uint(pre, "features", 128),
        .n_fft = try uint(pre, "n_fft", 512),
        .win = win,
        .hop = hop,
        .preemphasis = @floatCast(try num(pre, "preemph", 0.97)),
        .stack = try uint(asr, "subsampling_factor", 8),
        .speakers = try uint(sm, "num_spks", 8),
        .thresholded = thresholded,
        .threshold = @floatCast(try num(pe, "speaker_activity_threshold", 0.5)),
        .kernel_scale = @floatCast(try num(pe, "spk_kernel_scale", 1.0)),
        .out_dim = try uint(perception, "output_dim", 0),
        .asr = try encoder(asr),
        .spk = try encoder(spk),
        .stream = st,
    };
    a.ctc = parseCtc(root, a.asr.d_model);
    if (a.stack == 0 or a.stack != (try uint(spk, "subsampling_factor", 8)) or a.hop == 0 or a.win > a.n_fft or a.speakers == 0 or
        a.out_dim == 0 or st.chunk == 0 or st.cache < (1 + st.silence_per_speaker) * a.speakers)
        return error.Invalid;
    return a;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    const v = nh.parse(dir, config_json, settings_json, span, stub, out);
    if (!v.isOk()) return v;
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return refuse(out, .failed, "config.json is not a JSON object");
    const audio = parseAudio(root) catch |e| return refuse(out, if (e == error.Unsupported) .unsupported else .failed, "the perception config");
    if (audio.rate != SAMPLE_RATE or audio.hop != HOP or audio.stack != STACK) return refuse(out, .unsupported, "a sample rate, hop or stacking factor other than 16 kHz / 160 / 8");
    if (audio.out_dim != out.shell.hidden_size) return refuse(out, .failed, "the perception output is not the trunk's width");
    // mlx-audio's config drops `audio_token_id` and keeps the locator tag: `<|audio|>` is the id past the table.
    const id: u32 = json.int(json.field(root, "audio_token_id")) orelse
        (if (json.string(json.field(root, "audio_locator_tag")) != null) out.shell.vocab_size else return refuse(out, .failed, "no audio_token_id"));
    if (out.shell.vocab_size < 2 or id == spliceId(out.shell.vocab_size)) return refuse(out, .failed, "no room for the splice stand-in id");
    out.caps.audio = true;
    out.placeholder_ids.audio = id;
    out.weights.prefix = abi.Str.of("llm.backbone");
    return .ok;
}

fn refuse(out: *abi.Parsed, code: abi.Verdict.Code, comptime why: []const u8) abi.Verdict {
    nh.freeParsed(out);
    return .{ .code = code, .why = abi.Str.of("nemotron_transcribe: " ++ why) };
}

// ── tests ──

const testing = std.testing;

/// The checkpoint's config, abridged to the keys the port reads.
pub const sample_json =
    \\{"model_type":"nemo_speechlm","audio_token_id":131072,"eos_token_id":[11],
    \\ "quantization":{"group_size":128,"bits":8,"mode":"affine"},
    \\ "llm_config":{"model_type":"nemotron_h","hidden_size":2688,"intermediate_size":1856,"num_attention_heads":32,
    \\  "num_key_value_heads":2,"head_dim":128,"vocab_size":131072,"mamba_num_heads":64,"mamba_head_dim":64,
    \\  "layers_block_type":["mamba","moe","attention","moe"],"n_routed_experts":128,"num_experts_per_tok":6,
    \\  "moe_intermediate_size":1856,"moe_shared_expert_intermediate_size":3712,"routed_scaling_factor":2.5,
    \\  "layer_norm_epsilon":1e-5,"time_step_limit":[0.0,{"__float__":"Infinity"}]},
    \\ "ctc_timestamps":{"decoder_config":{"feat_in":1280,"num_classes":23862,"use_transformer":true,"n_heads":16,"n_layers":2,
    \\  "qk_norm":true,"ff_expansion":0.5,"pre_block_norm":true,"self_attention_model":"rope","rope_base":10000.0,
    \\  "rotary_fraction":0.5,"attn_mode":"full","residual":true,"residual_scale":0.1,"upsampling_factor":2,
    \\  "upsampling_mode":"conv_transpose"}},
    \\ "perception":{"output_dim":2688,"preprocessor":{"sample_rate":16000,"normalize":null,"window_size":0.025,
    \\  "window_stride":0.01,"window":"hann","features":128,"n_fft":512,"log":true,"frame_splicing":1}},
    \\ "pe_encoder_config":{"asr_encoder_type":"transformer","asr_normalize_type":"per_feature",
    \\  "speaker_feature_config_version":1,"speaker_feature_mode":"thresholded","speaker_activity_threshold":0.5,
    \\  "spk_kernel_scale":1.0,
    \\  "asr_encoder_cfg":{"feat_in":128,"d_model":1280,"n_heads":10,"n_layers":32,"qkv_bias":false,"qk_norm":true,
    \\   "ff_expansion":4,"pre_block_norm":true,"subsampling_factor":8,"attn_mode":"full","self_attention_model":"rope",
    \\   "rope_base":10000.0,"rotary_fraction":0.5},
    \\  "diarization_model_cfg":{"high_resolution":true,
    \\   "sortformer_modules":{"num_spks":8,"spkcache_sil_frames_per_spk":1,"pred_score_threshold":0.25,
    \\    "scores_boost_latest":0.05,"strong_boost_rate":0.75,"weak_boost_rate":1.5,"min_pos_scores_rate":0.5},
    \\   "encoder":{"feat_in":128,"n_layers":31,"d_model":512,"n_heads":8,"subsampling":"feature_stacking",
    \\    "subsampling_factor":8,"ff_expansion":4.0,"self_attention_model":"rope","xscaling":false,"qkv_bias":false,
    \\    "qk_norm":false,"pre_block_norm":true,"attn_mode":"full"}}}}
;

test "nemotron_transcribe config: the NeMo trunk parses through nemotron_h, audio rows ride the tokenizer's id" {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = sample_json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(sample_json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 2688), p.shell.hidden_size);
    try testing.expectEqual(@as(u8, 8), p.shell.quant.bits);
    try testing.expect(p.caps.audio and p.caps.moe);
    try testing.expectEqual(@as(u32, 131072), p.placeholder_ids.audio);
    try testing.expectEqualStrings("llm.backbone", p.weights.prefix.slice());
}

test "nemotron_transcribe config: mlx-audio's config names no audio id, its locator tag puts `<|audio|>` past the table" {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = sample_json }} };
    const dir = mem.packDir();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const theirs = try std.mem.replaceOwned(u8, arena.allocator(), sample_json, "\"audio_token_id\":131072", "\"audio_locator_tag\":\"<|audio|>\",\"nothing\":0");
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(theirs), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 131072), p.placeholder_ids.audio);
}

test "nemotron_transcribe config: the splice stand-in is an id inside the table, never the audio id" {
    try testing.expectEqual(@as(u32, 999), spliceId(131072));
    try testing.expectEqual(@as(u32, 63), spliceId(64));
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = sample_json }} };
    const dir = mem.packDir();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const clash = try std.mem.replaceOwned(u8, arena.allocator(), sample_json, "\"audio_token_id\":131072", "\"audio_token_id\":999");
    var p: abi.Parsed = .{};
    const v = parse(&dir, abi.Str.of(clash), .{}, null, true, &p);
    try testing.expect(!v.isOk());
    try testing.expect(std.mem.indexOf(u8, v.why.slice(), "splice stand-in") != null);
}

test "nemotron_transcribe config: the perception geometry and the streaming defaults" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = try parseAudio(try json.parseRoot(arena.allocator(), sample_json));
    try testing.expectEqual(@as(u32, 160), a.hop);
    try testing.expectEqual(@as(u32, 400), a.win);
    try testing.expect(a.thresholded and a.asr.qk_norm and !a.spk.qk_norm);
    try testing.expectEqual(@as(u32, 5120), a.asr.ff);
    try testing.expectEqual(@as(u32, 188), a.stream.cache);
    try testing.expectEqual(@as(u32, 8), a.speakers);
    const ctc = a.ctc.?;
    try testing.expectEqual(@as(u32, 640), ctc.enc.ff);
    try testing.expectEqual(@as(u32, 23862), ctc.classes);
    try testing.expectEqual(@as(u32, 2), ctc.upsample);
    try testing.expectApproxEqAbs(@as(f32, 0.1), ctc.residual, 1e-6);
}

test "nemotron_transcribe config: a head the port does not run leaves the pack without timestamps, not refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const bad = try std.mem.replaceOwned(u8, arena.allocator(), sample_json, "\"conv_transpose\"", "\"nearest\"");
    try testing.expect((try parseAudio(try json.parseRoot(arena.allocator(), bad))).ctc == null);
}

test "nemotron_transcribe config: a speaker feature mode or an encoder the port lacks is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const bad = try std.mem.replaceOwned(u8, arena.allocator(), sample_json, "\"thresholded\"", "\"learned\"");
    try testing.expectError(error.Unsupported, parseAudio(try json.parseRoot(arena.allocator(), bad)));
    const rot = try std.mem.replaceOwned(u8, arena.allocator(), sample_json, "\"self_attention_model\":\"rope\"", "\"self_attention_model\":\"abs\"");
    try testing.expectError(error.Unsupported, parseAudio(try json.parseRoot(arena.allocator(), rot)));
}
