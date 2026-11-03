//! LFM3 packs (`lfm3`, `lfm3_vl`, `lfm3_audio`, `lfm3_omni`): the claim and the parse into `Parsed` plus this plugin's own fields. MLX-free. The
//! trunk's fields read the text section first (`text_config`, then lfm3_audio's `lfm`), then the root, per field.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const json = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "lfm3",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{ "LiquidAI/LFM3-1.7B-MLX-8bit", "LiquidAI/LFM3-VL-2B-MLX-8bit", "LiquidAI/LFM3-Audio-2B-MLX-8bit" },
};

pub const model_types = [_][]const u8{ "lfm3", "lfm3_vl", "lfm3_audio", "lfm3_omni" };

pub const claims = @import("../../sdk/plugin.zig").claimsByType(&model_types);

/// The gated conv serves no batched rows: slots decode one at a time.
pub const batches_decode = false;

pub const max_layers = 128;

pub const LayerKind = enum(u8) { conv, full, sliding };

/// One attention layer type's rotary embedding (`rope_parameters.<type>`).
pub const Rope = struct {
    theta: f32,
    /// Rotated frequency pairs; the rest of the head's `head_dim / 2` pairs stay unrotated (the zero tail).
    angles: u32,
};

pub const max_vision_layers = 64;

/// LFM3-VL's tower (`vision_config`, `lfm3_vl_vision`), its projector and its processor's bounds.
pub const Vision = struct {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    patch: u32 = 16,
    merge: u32 = 2,
    temporal: u32 = 2,
    conv_kernel: u32 = 3,
    eps: f32 = 1e-5,
    /// The axial RoPE's base.
    theta: f32 = 1e6,
    /// Attention layers; the rest are gated 2-D convs.
    attention: [max_vision_layers]bool = @splat(false),
    proj_hidden: u32 = 4096,
    proj_eps: f32 = 1e-6,
    /// `processor_config.json`'s pixel bounds: an image's, and a whole clip's (every frame together).
    image_min_pixels: u32 = 65536,
    image_max_pixels: u32 = 16777216,
    video_min_pixels: u32 = 4096,
    video_max_pixels: u32 = 25165824,
    /// The processor's sampling rate: a client's frames that name none were sampled at it.
    fps: f32 = 2.0,
};

pub const max_audio_layers = 64;

/// The speech encoder (lfm3_audio's `encoder`, Omni's `audio_encoder_config`): log-mel bins, the downsampler's
/// channels, then a windowed LFM2 trunk at `hidden`; the projector maps it to the text width.
pub const Audio = struct {
    mel_bins: u32 = 128,
    conv: u32 = 256,
    hidden: u32 = 768,
    /// Keys an attention layer reads back (itself included).
    window: u32 = 32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    conv_kernel: u32 = 3,
    eps: f32 = 1e-5,
    theta: f32 = 1e6,
    attention: [max_audio_layers]bool = @splat(false),
};

pub const max_schedule = 8;
pub const max_depth_layers = 8;

/// Speech output (lfm3_audio, Omni): a frame of `codebooks` codes per `<|audio_out|>` slot from the depth decoder,
/// MaskGIT over `schedule`; the slot markers; the interleave pattern; the speech sampling defaults.
pub const Speech = struct {
    codebooks: u32 = 12,
    codebook_size: u32 = 2017,
    eos_code: u32 = 2016,
    schedule: [max_schedule]u32 = @splat(0),
    schedule_len: u32 = 0,
    interleave_text: u32 = 6,
    interleave_audio: u32 = 12,
    audio_out: u32 = 125032,
    audio_start: u32 = 125012,
    text_end: u32 = 125014,
    mixed_start: u32 = 125015,
    depth_attention: [max_depth_layers]bool = @splat(true),
    depth_layers: u32 = 0,
    depth_eps: f32 = 1e-5,
    depth_theta: f32 = 1e6,
    /// `generation_config.json`'s speech sampling (top_k 0 = every code).
    temperature: f32 = 1.0,
    top_k: u32 = 128,
    repetition_penalty: f32 = 1.1,
    repetition_window: u32 = 4096,
    /// The codec beside the pack (`audio_tokenizer/config.json`); 0 = none, the model does not speak.
    sample_rate: u32 = 0,
    frame_samples: u32 = 0,

    pub fn maskgit(s: *const Speech) []const u32 {
        return s.schedule[0..s.schedule_len];
    }
};

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    norm_eps: f32,
    max_position_embeddings: u32,
    conv_kernel: u32 = 7,
    sliding_window: u32 = 256,
    full_rope: Rope,
    sliding_rope: Rope,
    tie_word_embeddings: bool = true,
    vision: ?Vision = null,
    audio: ?Audio = null,
    speech: ?Speech = null,
    layers: [max_layers]LayerKind = @splat(.full),
    quant: abi.Quant = .{},
    /// The quant format's `open` block.
    block_json: []const u8 = "{}",
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "lfm3: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "lfm3: " ++ why);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return invalid("config.json is not a JSON object");
    const type_name = json.string(json.field(root, "model_type")) orelse return unsupported("no model_type");
    // Omni is both: the VL tower and the speech stack around one trunk.
    const omni = std.mem.eql(u8, type_name, "lfm3_omni");
    const vl = omni or std.mem.eql(u8, type_name, "lfm3_vl");
    const speech = omni or std.mem.eql(u8, type_name, "lfm3_audio");
    if (!vl and !speech and !std.mem.eql(u8, type_name, "lfm3")) return unsupported("another model_type");
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    const text = json.object(json.field(root, "text_config")) orelse json.object(json.field(root, "lfm"));
    var c = readConfig(.{ .text = text, .root = root }) catch |e| return switch (e) {
        error.Unsupported => unsupported("a shape this plugin does not serve"),
        error.Invalid => invalid("a config value out of range"),
    };
    if (vl) {
        var v = readVision(root) catch |e| return switch (e) {
            error.Unsupported => unsupported("a vision tower this plugin does not serve"),
            error.Invalid => invalid("vision_config"),
        };
        readProcessor(arena.allocator(), dir, &v) catch return invalid("processor_config.json");
        c.vision = v;
    }

    if (speech) {
        c.audio = readAudio(root) catch |e| return switch (e) {
            error.Unsupported => unsupported("a speech encoder this plugin does not serve"),
            error.Invalid => invalid("encoder"),
        };
        var sp = readSpeech(root) catch |e| return switch (e) {
            error.Unsupported => unsupported("a depth decoder this plugin does not serve"),
            error.Invalid => invalid("the speech output config"),
        };
        readSpeechSampling(arena.allocator(), dir, &sp) catch return invalid("generation_config.json");
        readCodec(arena.allocator(), dir, &sp) catch return invalid("audio_tokenizer/config.json");
        c.speech = sp;
    }

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = batches_decode, .exact_resumed = true, .dflash_capture = true, .images = vl, .video = vl, .audio = c.audio != null } };
    // lfm3_audio's config names none of its ids: mlx-vlm's defaults are the tokenizer's.
    if (c.audio != null) p.placeholder_ids.audio = json.int(json.field(root, "audio_token_id")) orelse 125031;
    if (c.speech) |sp| if (sp.frame_samples != 0) {
        p.placeholder_ids.speech = sp.audio_out;
        p.speech = .{
            .codebooks = sp.codebooks,
            .codebook_size = sp.codebook_size,
            .eos_code = sp.eos_code,
            .frame_token = sp.audio_out,
            .audio_start = sp.audio_start,
            .mixed_start = sp.mixed_start,
            .text_end = sp.text_end,
            .interleave_text = sp.interleave_text,
            .interleave_audio = sp.interleave_audio,
            .sample_rate = sp.sample_rate,
            .frame_samples = sp.frame_samples,
            .temperature = sp.temperature,
            .top_k = sp.top_k,
            .repetition_penalty = sp.repetition_penalty,
            .repetition_window = sp.repetition_window,
            // A card fact, not a config one: LFM3-Audio documents tools with a spoken reply, Omni answers them in text.
            .tools = !omni,
        };
    };
    // Field by field: Omni's audio and speech ids are already set. Omni's config names no vision start/end: the
    // tokenizer's, as LFM3-VL's config names them.
    if (vl) {
        const ph = &p.placeholder_ids;
        ph.image = json.int(json.field(root, "image_token_id")) orelse return invalid("image_token_id");
        ph.video = json.int(json.field(root, "video_token_id")) orelse return invalid("video_token_id");
        ph.vision_start = json.int(json.field(root, "vision_start_token_id")) orelse 125009;
        ph.vision_end = json.int(json.field(root, "vision_end_token_id")) orelse 125010;
    }
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.max_position_embeddings,
        .quant = c.quant,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(if (omni) "lfm3_omni" else if (vl) "lfm3_vl" else if (speech) "lfm3_audio" else "lfm3"),
    };
    if (json.field(root, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = json.int(v) orelse return invalid("bos_token_id");
    };
    json.eos(root, s) catch return invalid("eos_token_id");
    // The conv windows are the recurrent state the prefix cache snapshots.
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .bf16 };
    // mlx-vlm's layout (its `sanitize` nests a flat `model.*` pack there too); the load probes the others.
    p.weights.prefix = abi.Str.of("language_model.model");
    p.chat.tool_dialect = .qwen_xml;
    p.chat.reasoning_dialect = .think_tags;
    // LiquidAI's card: reasoning on by default; turning it off "can be inconsistent". The speech models' templates
    // open no think block (a client prefills one).
    if (c.audio == null) p.chat.thinking_default = .on;
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return invalid("out of memory");
    c.block_json = json.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return invalid("out of memory");
    };
    cfg_out.* = c;
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

const ReadError = error{ Unsupported, Invalid };

/// The text section, then the root, per field.
const Src = struct {
    text: ?json.Object,
    root: json.Object,

    fn get(s: Src, key: []const u8) ?json.Value {
        if (s.text) |t| if (json.field(t, key)) |v| return v;
        return json.field(s.root, key);
    }
    fn int(s: Src, key: []const u8) ReadError!?u32 {
        const v = s.get(key) orelse return null;
        return json.int(v) orelse error.Invalid;
    }
};

fn readConfig(src: Src) ReadError!Config {
    var c: Config = .{
        .vocab_size = (try src.int("vocab_size")) orelse return error.Invalid,
        .hidden_size = (try src.int("hidden_size")) orelse return error.Invalid,
        .intermediate_size = (try src.int("intermediate_size")) orelse return error.Invalid,
        .num_hidden_layers = (try src.int("num_hidden_layers")) orelse return error.Invalid,
        .num_attention_heads = (try src.int("num_attention_heads")) orelse return error.Invalid,
        .num_key_value_heads = (try src.int("num_key_value_heads")) orelse return error.Invalid,
        .head_dim = 0,
        .norm_eps = @floatCast(json.float(src.get("norm_eps")) orelse 1e-5),
        .max_position_embeddings = (try src.int("max_position_embeddings")) orelse 262144,
        .full_rope = .{ .theta = 1e6, .angles = 0 },
        .sliding_rope = .{ .theta = 1e4, .angles = 0 },
    };
    if (c.num_attention_heads == 0 or c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers) return error.Unsupported;
    c.head_dim = (try src.int("head_dim")) orelse c.hidden_size / c.num_attention_heads;
    if (try src.int("conv_L_cache")) |k| c.conv_kernel = k;
    if (try src.int("sliding_window")) |w| c.sliding_window = w;
    if (json.boolean(src.get("tie_word_embeddings"))) |t| c.tie_word_embeddings = t;
    // mlx-vlm's LFM3 serves the sandwich norms and a bias-free conv; no released pack ships another.
    if (json.boolean(src.get("use_sandwich_norm")) == false or json.boolean(src.get("conv_bias")) == true) return error.Unsupported;

    const types = src.get("layer_types") orelse return error.Unsupported;
    if (types != .array or types.array.items.len != c.num_hidden_layers) return error.Unsupported;
    for (types.array.items, 0..) |item, i| {
        const t = json.string(item) orelse return error.Invalid;
        c.layers[i] = if (std.mem.eql(u8, t, "conv")) .conv else if (std.mem.eql(u8, t, "full_attention")) .full else if (std.mem.eql(u8, t, "sliding_attention")) .sliding else return error.Unsupported;
    }
    c.full_rope = try ropeOf(src.get("rope_parameters"), "full_attention", c.head_dim, .{ .theta = 1e6, .factor = 0.5 });
    c.sliding_rope = try ropeOf(src.get("rope_parameters"), "sliding_attention", c.head_dim, .{ .theta = 1e4, .factor = 1.0 });

    c.quant = json.quantOf(src.root, "quantization", false) catch |e| return if (e == error.Unsupported) error.Unsupported else error.Invalid;
    if (c.vocab_size == 0 or c.hidden_size == 0 or c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim % 2 != 0 or c.conv_kernel < 2 or c.intermediate_size == 0 or c.sliding_window == 0)
        return error.Unsupported;
    return c;
}

/// `vision_config` and the projector as mlx-vlm's `lfm3_vl` reads them; a shape it does not build is refused.
fn readVision(root: json.Object) ReadError!Vision {
    const vc = json.object(json.field(root, "vision_config")) orelse return error.Unsupported;
    var v: Vision = .{
        .hidden = (try json.intField(vc, "hidden_size")) orelse return error.Invalid,
        .intermediate = (try json.intField(vc, "intermediate_size")) orelse return error.Invalid,
        .layers = (try json.intField(vc, "num_hidden_layers")) orelse return error.Invalid,
        .heads = (try json.intField(vc, "num_attention_heads")) orelse return error.Invalid,
        .kv_heads = (try json.intField(vc, "num_key_value_heads")) orelse return error.Invalid,
    };
    if (try json.intField(vc, "patch_size")) |x| v.patch = x;
    if (try json.intField(vc, "spatial_merge_size")) |x| v.merge = x;
    if (try json.intField(vc, "temporal_patch_size")) |x| v.temporal = x;
    if (try json.intField(vc, "conv_kernel_size")) |x| v.conv_kernel = x;
    if (try json.floatField(vc, "norm_eps")) |x| v.eps = @floatCast(x);
    if (json.object(json.field(vc, "rope_parameters"))) |rp| if (json.float(json.field(rp, "rope_theta"))) |t| {
        v.theta = @floatCast(t);
    };
    if (try json.intField(root, "projector_hidden_size")) |x| v.proj_hidden = x;
    if (try json.floatField(root, "projector_layer_norm_eps")) |x| v.proj_eps = @floatCast(x);
    if ((try json.intField(vc, "in_channels") orelse 3) != 3) return error.Unsupported;
    // The projector folds one merge block into a row: the two factors are one in every release.
    if ((try json.intField(root, "downsample_factor") orelse v.merge) != v.merge) return error.Unsupported;
    if (json.boolean(json.field(root, "projector_bias")) == false) return error.Unsupported;
    if (json.string(json.field(root, "projector_hidden_act"))) |act| if (!std.mem.eql(u8, act, "gelu")) return error.Unsupported;
    if (v.layers == 0 or v.layers > max_vision_layers or v.heads == 0 or v.kv_heads == 0 or v.heads % v.kv_heads != 0 or
        v.hidden % v.heads != 0 or (v.hidden / v.heads) % 4 != 0 or v.patch == 0 or v.merge == 0 or v.temporal == 0 or
        v.conv_kernel % 2 == 0 or v.intermediate == 0 or v.proj_hidden == 0)
        return error.Unsupported;
    const types = json.field(vc, "layer_types") orelse return error.Unsupported;
    if (types != .array or types.array.items.len != v.layers) return error.Unsupported;
    for (types.array.items, 0..) |item, i| {
        const t = json.string(item) orelse return error.Invalid;
        v.attention[i] = if (std.mem.eql(u8, t, "full_attention")) true else if (std.mem.eql(u8, t, "conv")) false else return error.Unsupported;
    }
    return v;
}

/// The speech encoder in either layout: lfm3_audio's `encoder` (`dim_in`, `dim_conv`, `dim_out`, its trunk under
/// `lfm`) or Omni's `audio_encoder_config` (`num_mel_bins`, `conv_channels`, `hidden_size`, `text_config`).
fn readAudio(root: json.Object) ReadError!Audio {
    const omni = json.object(json.field(root, "audio_encoder_config"));
    const enc = omni orelse json.object(json.field(root, "encoder")) orelse return error.Unsupported;
    const trunk = json.object(json.field(enc, if (omni != null) "text_config" else "lfm")) orelse return error.Unsupported;
    var a: Audio = .{
        .layers = (try json.intField(trunk, "num_hidden_layers")) orelse return error.Invalid,
        .heads = (try json.intField(trunk, "num_attention_heads")) orelse return error.Invalid,
        .kv_heads = (try json.intField(trunk, "num_key_value_heads")) orelse return error.Invalid,
    };
    if (try json.intField(enc, if (omni != null) "num_mel_bins" else "dim_in")) |x| a.mel_bins = x;
    if (try json.intField(enc, if (omni != null) "conv_channels" else "dim_conv")) |x| a.conv = x;
    if (try json.intField(enc, if (omni != null) "hidden_size" else "dim_out")) |x| a.hidden = x;
    if (try json.intField(enc, "sliding_window")) |x| a.window = x;
    if (try json.intField(trunk, "conv_L_cache")) |x| a.conv_kernel = x;
    if (try json.floatField(trunk, "norm_eps")) |x| a.eps = @floatCast(x);
    a.theta = @floatCast(try json.ropeTheta(trunk, 1e6));
    if ((try json.intField(trunk, "hidden_size") orelse a.hidden) != a.hidden) return error.Invalid;
    // The reference frames 16 kHz clips into 128 bins; the downsampler folds 8 frames per row.
    if (a.mel_bins != 128 or a.layers == 0 or a.layers > max_audio_layers or a.heads == 0 or a.kv_heads == 0 or
        a.heads % a.kv_heads != 0 or a.hidden % a.heads != 0 or a.conv == 0 or a.conv_kernel < 2 or a.window == 0)
        return error.Unsupported;
    const types = json.field(trunk, "layer_types") orelse return error.Unsupported;
    if (types != .array or types.array.items.len != a.layers) return error.Unsupported;
    for (types.array.items, 0..) |item, i| {
        const t = json.string(item) orelse return error.Invalid;
        a.attention[i] = if (std.mem.eql(u8, t, "full_attention")) true else if (std.mem.eql(u8, t, "conv")) false else return error.Unsupported;
    }
    return a;
}

/// Speech output in either layout: lfm3_audio's (`codebooks`, `audio_vocab_size`, `maskgit_steps`, `depthformer`,
/// the ids mlx-vlm's defaults) or Omni's (`num_codebooks`, `codebook_size`, `audio_eos_code`, `maskgit_schedule`,
/// `depth_config`, the ids named).
fn readSpeech(root: json.Object) ReadError!Speech {
    var s: Speech = .{};
    const depth = json.object(json.field(root, "depth_config")) orelse json.object(json.field(root, "depthformer")) orelse return error.Unsupported;
    if (try json.intField(root, "num_codebooks") orelse try json.intField(root, "codebooks")) |x| s.codebooks = x;
    if (try json.intField(root, "codebook_size") orelse try json.intField(root, "audio_vocab_size")) |x| s.codebook_size = x;
    s.eos_code = (try json.intField(root, "audio_eos_code")) orelse s.codebook_size - 1;
    for ([_]struct { []const u8, *u32 }{
        .{ "interleaved_n_text", &s.interleave_text }, .{ "interleaved_n_audio", &s.interleave_audio }, .{ "audio_out_token_id", &s.audio_out },
        .{ "audio_start_token_id", &s.audio_start }, .{ "text_end_token_id", &s.text_end }, .{ "mixed_start_token_id", &s.mixed_start },
    }) |f| if (try json.intField(root, f[0])) |x| {
        f[1].* = x;
    };
    if (json.field(root, "maskgit_schedule")) |v| {
        if (v != .array or v.array.items.len == 0 or v.array.items.len > max_schedule) return error.Unsupported;
        for (v.array.items, 0..) |item, i| s.schedule[i] = json.int(item) orelse return error.Invalid;
        s.schedule_len = @intCast(v.array.items.len);
    } else {
        const steps = (try json.intField(root, "maskgit_steps")) orelse 3;
        if (steps == 0 or steps > max_schedule or steps > s.codebooks) return error.Unsupported;
        for (s.schedule[0..steps]) |*x| x.* = s.codebooks / steps;
        s.schedule[steps - 1] += s.codebooks % steps;
        s.schedule_len = steps;
    }
    var total: u32 = 0;
    for (s.maskgit()) |x| total += x;
    s.depth_layers = (try json.intField(depth, "num_hidden_layers")) orelse return error.Invalid;
    if (try json.floatField(depth, "norm_eps")) |x| s.depth_eps = @floatCast(x);
    s.depth_theta = @floatCast(try json.ropeTheta(depth, 1e6));
    if (json.field(depth, "layer_types")) |lt| if (lt == .array and lt.array.items.len == s.depth_layers) for (lt.array.items, 0..) |item, i| {
        s.depth_attention[i] = std.mem.eql(u8, json.string(item) orelse return error.Invalid, "full_attention");
    };
    if (total != s.codebooks or s.codebooks == 0 or s.codebooks > 32 or s.codebook_size < 2 or s.eos_code >= s.codebook_size or
        s.depth_layers == 0 or s.depth_layers > max_depth_layers or s.interleave_text == 0 or s.interleave_audio == 0)
        return error.Unsupported;
    // The depth trunk attends over its slots: a conv layer there is a shape no release ships.
    for (s.depth_attention[0..s.depth_layers]) |att| if (!att) return error.Unsupported;
    return s;
}

/// `generation_config.json`'s speech sampling (`audio_top_k: null` = every code).
fn readSpeechSampling(arena: std.mem.Allocator, dir: *const abi.PackDir, s: *Speech) !void {
    const buf = try arena.alloc(u8, 1 << 16);
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("generation_config.json", buf, &why) orelse return;
    const root = json.parseRoot(arena, bytes) catch return error.Invalid;
    if (try json.floatField(root, "audio_temperature")) |x| s.temperature = @floatCast(x);
    // A JSON null here is a value (every code), not an absent key.
    if (root.get("audio_top_k")) |v| s.top_k = if (v == .null) 0 else json.int(v) orelse return error.Invalid;
    if (try json.floatField(root, "audio_repetition_penalty")) |x| s.repetition_penalty = @floatCast(x);
    if (try json.intField(root, "audio_repetition_window")) |x| s.repetition_window = x;
}

/// The codec's output rate and samples per frame (`stft_frame_stack` STFT hops); absent = the pack does not speak.
fn readCodec(arena: std.mem.Allocator, dir: *const abi.PackDir, s: *Speech) !void {
    const buf = try arena.alloc(u8, 1 << 15);
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("audio_tokenizer/config.json", buf, &why) orelse return;
    const root = json.parseRoot(arena, bytes) catch return error.Invalid;
    const rate = (try json.intField(root, "sampling_rate")) orelse 24000;
    const stack = (try json.intField(root, "stft_frame_stack")) orelse 6;
    const hop = (try json.intField(root, "stft_hop_length")) orelse 320;
    if (rate == 0 or stack == 0 or hop == 0) return error.Invalid;
    s.sample_rate = rate;
    s.frame_samples = stack * hop;
}

/// The image and video processors' pixel bounds (`size.shortest_edge` / `longest_edge`) and the video rate.
fn readProcessor(arena: std.mem.Allocator, dir: *const abi.PackDir, v: *Vision) !void {
    const buf = try arena.alloc(u8, 1 << 20);
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("processor_config.json", buf, &why) orelse return;
    const root = json.parseRoot(arena, bytes) catch return error.Invalid;
    if (json.object(json.field(root, "image_processor"))) |ip| if (json.object(json.field(ip, "size"))) |sz| {
        if (try json.intField(sz, "shortest_edge")) |x| v.image_min_pixels = x;
        if (try json.intField(sz, "longest_edge")) |x| v.image_max_pixels = x;
    };
    if (json.object(json.field(root, "video_processor"))) |vp| {
        if (json.object(json.field(vp, "size"))) |sz| {
            if (try json.intField(sz, "shortest_edge")) |x| v.video_min_pixels = x;
            if (try json.intField(sz, "longest_edge")) |x| v.video_max_pixels = x;
        }
        if (try json.floatField(vp, "fps")) |f| if (f > 0) {
            v.fps = @floatCast(f);
        };
    }
}

/// `rope_parameters.<layer_type>` as mlx-vlm reads it: default RoPE only, `int(head_dim * factor) / 2` pairs rotated.
fn ropeOf(params: ?json.Value, key: []const u8, head_dim: u32, default: struct { theta: f64, factor: f64 }) ReadError!Rope {
    const p = if (params) |v| (json.object(v) orelse return error.Invalid) else null;
    const o = if (p) |obj| json.object(json.field(obj, key)) else null;
    var theta = default.theta;
    var factor = if (o == null) default.factor else 1.0;
    if (o) |obj| {
        if (json.string(json.field(obj, "rope_type"))) |t| if (!std.mem.eql(u8, t, "default")) return error.Unsupported;
        if (json.float(json.field(obj, "rope_theta"))) |t| theta = t;
        if (json.float(json.field(obj, "partial_rotary_factor"))) |f| factor = f;
    }
    if (!(theta > 0) or !(factor > 0 and factor <= 1)) return error.Invalid;
    const angles: u32 = @intFromFloat(@floor(@as(f64, @floatFromInt(head_dim)) * factor));
    if (angles / 2 == 0) return error.Invalid;
    return .{ .theta = @floatCast(theta), .angles = angles / 2 };
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        gpa.free(c.block_json);
        gpa.destroy(c);
    }
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

/// Conv layers keep a window, full layers all their keys, sliding layers `sliding_window` keys.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.num_hidden_layers, cap);
    for (out[0..n], c.layers[0..n]) |*o, k| o.* = switch (k) {
        .conv => .{ .kind = .conv, .window = c.conv_kernel },
        .full => .{ .kind = .full, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim },
        .sliding => .{ .kind = .sliding, .window = c.sliding_window, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim },
    };
    return n;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

fn parseJson(js: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = js }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(js), .{}, null, true, out);
}

/// LiquidAI/LFM3-1.7B-MLX-8bit's config.json, trimmed to the keys the parse reads.
pub const lfm3_1_7b =
    \\{"model_type":"lfm3","bos_token_id":124894,"conv_L_cache":7,"conv_bias":false,"eos_token_id":[124900],
    \\ "head_dim":128,"hidden_size":2048,"intermediate_size":7680,"layer_types":["conv","conv","full_attention","conv",
    \\ "sliding_attention","conv","full_attention","conv","sliding_attention","conv","full_attention","conv",
    \\ "sliding_attention","conv","full_attention","conv","sliding_attention","conv","full_attention","conv",
    \\ "sliding_attention","conv","full_attention","conv"],"max_position_embeddings":262144,"norm_eps":1e-05,
    \\ "num_attention_heads":16,"num_hidden_layers":24,"num_key_value_heads":4,
    \\ "quantization":{"group_size":64,"bits":8,"mode":"affine"},
    \\ "quantization_config":{"group_size":64,"bits":8,"mode":"affine"},
    \\ "rope_parameters":{"full_attention":{"partial_rotary_factor":0.5,"rope_theta":1000000.0,"rope_type":"default"},
    \\ "sliding_attention":{"rope_theta":10000,"rope_type":"default"}},"sliding_window":256,"tie_word_embeddings":true,
    \\ "use_sandwich_norm":true,"vision_config":{},"vocab_size":128000}
;

test "lfm3 config: LFM3-1.7B parses into its shell, layer kinds, ropes and cache layout" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(lfm3_1_7b, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 128), p.shell.head_dim);
    try testing.expectEqual(@as(i64, 124894), p.shell.bos);
    try testing.expectEqualSlices(u32, &.{124900}, p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqual(@as(u8, 8), p.shell.quant.bits);
    try testing.expectEqual(LayerKind.full, c.layers[2]);
    try testing.expectEqual(LayerKind.sliding, c.layers[4]);
    try testing.expectEqual(Rope{ .theta = 1e6, .angles = 32 }, c.full_rope);
    try testing.expectEqual(Rope{ .theta = 1e4, .angles = 64 }, c.sliding_rope);
    try testing.expect(p.caps.exact_resumed and !p.caps.batches_decode);
    try testing.expectEqual(abi.ToolDialect.qwen_xml, p.chat.tool_dialect);
    try testing.expectEqual(abi.ThinkingDefault.on, p.chat.thinking_default);
    var lay: [24]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 24), layout(&p, null, &lay, lay.len));
    try testing.expectEqual(abi.LayerKind.conv, lay[0].kind);
    try testing.expectEqual(@as(u32, 7), lay[0].window);
    try testing.expectEqual(abi.LayerKind.full, lay[2].kind);
    try testing.expectEqual(abi.LayerKind.sliding, lay[4].kind);
    try testing.expectEqual(@as(u32, 256), lay[4].window);
}

/// LiquidAI/LFM3-VL-2B-MLX-8bit's config.json, trimmed to the keys the parse reads.
pub const lfm3_vl_2b =
    \\{"model_type":"lfm3_vl","bos_token_id":124894,"eos_token_id":124900,"image_token_id":125030,"video_token_id":125029,
    \\ "vision_start_token_id":125009,"vision_end_token_id":125010,"downsample_factor":2,"projector_bias":true,
    \\ "projector_hidden_act":"gelu","projector_hidden_size":4096,"projector_layer_norm_eps":1e-06,
    \\ "quantization":{"group_size":64,"bits":8,"mode":"affine"},
    \\ "text_config":{"conv_L_cache":7,"head_dim":128,"hidden_size":2048,"intermediate_size":7680,"layer_types":["conv",
    \\ "conv","full_attention","conv","sliding_attention","conv","full_attention","conv","sliding_attention","conv",
    \\ "full_attention","conv","sliding_attention","conv","full_attention","conv","sliding_attention","conv","full_attention",
    \\ "conv","sliding_attention","conv","full_attention","conv"],"max_position_embeddings":128000,"model_type":"lfm3",
    \\ "norm_eps":1e-05,"num_attention_heads":16,"num_hidden_layers":24,"num_key_value_heads":4,"sliding_window":256,
    \\ "tie_word_embeddings":true,"use_sandwich_norm":true,"vocab_size":128000},
    \\ "vision_config":{"conv_kernel_size":3,"hidden_size":1024,"in_channels":3,"intermediate_size":2560,"layer_types":["conv",
    \\ "conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention","conv","full_attention",
    \\ "conv","full_attention","conv"],"model_type":"lfm3_vl_vision","norm_eps":1e-05,"num_attention_heads":16,
    \\ "num_hidden_layers":14,"num_key_value_heads":8,"patch_size":16,"rope_parameters":{"rope_theta":1000000.0,
    \\ "rope_type":"axial"},"spatial_merge_size":2,"temporal_patch_size":2}}
;

test "lfm3 config: LFM3-VL-2B parses its trunk, tower, projector, processor bounds and placeholders" {
    var p: abi.Parsed = .{};
    const proc =
        \\{"image_processor":{"size":{"shortest_edge":65536,"longest_edge":16777216}},
        \\ "video_processor":{"fps":2.0,"size":{"shortest_edge":4096,"longest_edge":25165824}}}
    ;
    const mem: fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = lfm3_vl_2b }, .{ .name = "processor_config.json", .bytes = proc } } };
    const dir = mem.packDir();
    try testing.expect(parse(&dir, abi.Str.of(lfm3_vl_2b), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const v = of(&p).vision.?;
    try testing.expectEqual(@as(u32, 2048), p.shell.hidden_size);
    try testing.expectEqualStrings("lfm3_vl", p.shell.model_type.slice());
    try testing.expect(p.caps.images and p.caps.video and !p.caps.audio);
    try testing.expectEqual(@as(u32, 125030), p.placeholder_ids.image);
    try testing.expectEqual(@as(u32, 125029), p.placeholder_ids.video);
    try testing.expectEqual(@as(u32, 125009), p.placeholder_ids.vision_start);
    try testing.expectEqual(@as(u32, 1024), v.hidden);
    try testing.expectEqual(@as(u32, 8), v.kv_heads);
    try testing.expect(!v.attention[0] and v.attention[2] and !v.attention[13]);
    try testing.expectEqual(@as(f32, 1e6), v.theta);
    try testing.expectEqual(@as(u32, 65536), v.image_min_pixels);
    try testing.expectEqual(@as(u32, 25165824), v.video_max_pixels);
    try testing.expectEqual(@as(f32, 2.0), v.fps);
    // The text model ships an empty `vision_config`: no tower.
    var t: abi.Parsed = .{};
    try testing.expect(parseJson(lfm3_1_7b, &t).isOk());
    defer freeParsed(&t);
    try testing.expect(of(&t).vision == null and !t.caps.images);
}

test "lfm3 config: fields nested under text_config or lfm win over the root, per field" {
    var p: abi.Parsed = .{};
    const nested =
        \\{"model_type":"lfm3","vocab_size":10,"lfm":{"vocab_size":300,"hidden_size":64,"intermediate_size":96,
        \\ "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"layer_types":["conv","sliding_attention"]}}
    ;
    try testing.expect(parseJson(nested, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 300), p.shell.vocab_size);
    try testing.expectEqual(@as(u32, 16), p.shell.head_dim);
    try testing.expectEqual(Rope{ .theta = 1e6, .angles = 4 }, of(&p).full_rope);
}

test "lfm3 config: shapes mlx-vlm's LFM3 does not serve are refused by name" {
    var p: abi.Parsed = .{};
    const base = "\"model_type\":\"lfm3\",\"vocab_size\":300,\"hidden_size\":64,\"intermediate_size\":96,\"num_hidden_layers\":1,\"num_attention_heads\":4,\"num_key_value_heads\":2";
    const cases = [_][]const u8{
        "{" ++ base ++ ",\"layer_types\":[\"mamba\"]}",
        "{" ++ base ++ "}",
        "{" ++ base ++ ",\"layer_types\":[\"conv\"],\"use_sandwich_norm\":false}",
        "{" ++ base ++ ",\"layer_types\":[\"conv\"],\"conv_bias\":true}",
        "{" ++ base ++ ",\"layer_types\":[\"conv\"],\"rope_parameters\":{\"full_attention\":{\"rope_type\":\"yarn\"}}}",
    };
    for (cases) |js| try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(js, &p).code);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson("{\"model_type\":\"lfm2\"}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("[1]", &p).code);
    try testing.expect(p.private == null);
}

test "lfm3 config: LFM3-Audio-2B's encoder, ids and caps; its template opens no think block, so no default" {
    var p: abi.Parsed = .{};
    const js = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/LiquidAI_LFM3-Audio-2B-MLX-8bit.json");
    try testing.expect(parseJson(js, &p).isOk());
    defer freeParsed(&p);
    const a = of(&p).audio.?;
    try testing.expectEqualStrings("lfm3_audio", p.shell.model_type.slice());
    try testing.expectEqual(@as(u32, 2048), p.shell.hidden_size);
    try testing.expect(p.caps.audio and !p.caps.images);
    try testing.expectEqual(@as(u32, 125031), p.placeholder_ids.audio);
    try testing.expectEqual(@as(u32, 17), a.layers);
    try testing.expectEqual(@as(u32, 768), a.hidden);
    try testing.expectEqual(@as(u32, 256), a.conv);
    try testing.expectEqual(@as(u32, 32), a.window);
    try testing.expect(!a.attention[0] and a.attention[2] and a.attention[16]);
    try testing.expectEqual(abi.ThinkingDefault.unset, p.chat.thinking_default);
}

test "lfm3 config: LFM3-Omni-2B sees, hears and speaks, with its own speech sampling; no thinking default" {
    const js = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/LiquidAI_LFM3-Omni-2B-MLX-8bit.json");
    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "config.json", .bytes = js },
        .{ .name = "audio_tokenizer/config.json", .bytes = "{\"sampling_rate\":24000,\"stft_frame_stack\":6,\"stft_hop_length\":320}" },
        .{ .name = "generation_config.json", .bytes = "{\"audio_temperature\":0.4,\"audio_top_k\":null,\"audio_repetition_penalty\":1.2,\"audio_repetition_window\":64}" },
    } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(js), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqualStrings("lfm3_omni", p.shell.model_type.slice());
    try testing.expect(p.caps.images and p.caps.video and p.caps.audio);
    try testing.expect(p.placeholder_ids.image != 0 and p.placeholder_ids.audio != 0 and p.placeholder_ids.speech == 125032);
    try testing.expectEqual(@as(u32, 125009), p.placeholder_ids.vision_start);
    try testing.expectEqual(@as(u32, 125010), p.placeholder_ids.vision_end);
    try testing.expectEqual(@as(u32, 12), p.speech.codebooks);
    try testing.expectEqual(@as(u32, 64), p.speech.repetition_window);
    try testing.expectEqual(@as(u32, 0), p.speech.top_k);
    try testing.expectApproxEqAbs(@as(f32, 0.4), p.speech.temperature, 1e-6);
    // Its card answers a request with tools in text; under the speech recipe it called a tool for a joke.
    try testing.expect(!p.speech.tools);
    try testing.expectEqual(abi.ThinkingDefault.unset, p.chat.thinking_default);
    try testing.expectEqual(@as(u32, 768), of(&p).audio.?.hidden);
}

test "lfm3 config: LFM3-Audio-2B speaks only beside its codec, with the card's speech sampling" {
    const js = @embedFile("../../engines/mlx/plugin/fixtures/mlx_lm/LiquidAI_LFM3-Audio-2B-MLX-8bit.json");
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(js, &p).isOk());
    try testing.expectEqual(@as(u32, 0), p.speech.codebooks);
    try testing.expectEqual(@as(u32, 0), p.placeholder_ids.speech);
    freeParsed(&p);

    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "config.json", .bytes = js },
        .{ .name = "audio_tokenizer/config.json", .bytes = "{\"sampling_rate\":24000,\"stft_frame_stack\":6,\"stft_hop_length\":320}" },
        .{ .name = "generation_config.json", .bytes = "{\"audio_temperature\":0.9,\"audio_top_k\":null,\"audio_repetition_penalty\":1.1,\"audio_repetition_window\":4096}" },
    } };
    const dir = mem.packDir();
    try testing.expect(parse(&dir, abi.Str.of(js), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const s = p.speech;
    try testing.expectEqual(@as(u32, 12), s.codebooks);
    try testing.expectEqual(@as(u32, 2017), s.codebook_size);
    try testing.expectEqual(@as(u32, 2016), s.eos_code);
    try testing.expectEqual(@as(u32, 125032), s.frame_token);
    try testing.expectEqual(s.frame_token, p.placeholder_ids.speech);
    try testing.expectEqual(@as(u32, 1920), s.frame_samples);
    try testing.expectEqual(@as(u32, 6), s.interleave_text);
    try testing.expectEqual(@as(u32, 12), s.interleave_audio);
    // A null top-k is every code, not the default.
    try testing.expectEqual(@as(u32, 0), s.top_k);
    try testing.expectApproxEqAbs(@as(f32, 0.9), s.temperature, 1e-6);
    try testing.expect(s.tools);
    try testing.expectEqualSlices(u32, &.{ 4, 4, 4 }, of(&p).speech.?.maskgit());
}
