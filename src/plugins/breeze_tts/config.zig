//! Breeze-TTS-2's geometry, claims, caps and bills, MLX-free: a `breeze_tts` config.json, complete once the Mimi
//! audio_tokenizer lands; the speech route; backbone, depth decoder, text encoder and codec all resident.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");
const fields = @import("../../sdk/media/config_fields.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "breeze_tts",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Breeze-TTS-2-mlx"},
};

pub const model_types = [_][]const u8{"breeze_tts"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// The Mimi codec lives in `audio_tokenizer/`; a partial pull can miss the subdir.
pub const required_file = "audio_tokenizer/model.safetensors";

/// The pack's geometry and ids, read from `config.json`. Defaults are the
/// published Breeze-TTS-2 values; an omitted field keeps them, a present one
/// must be parseable and inside the shapes the engine implements.
pub const Config = struct {
    // token ids (fixed by the published pack, read from the pack's config)
    audio_token_id: u32 = 262144,
    audio_eos_id: u32 = 262145,
    speaker_base: u32 = 262146, // [S0]..[S9]
    ins_bos_id: u32 = 262156,
    ins_eos_id: u32 = 262157,
    eoi_token_id: u32 = 256000, // TE row zeroed at gather
    te_vocab: u32 = 262158,

    // codebooks: 16 (1 semantic + 15 acoustic); the engine's frame loop is 16 wide
    num_codebooks: u32 = 16,
    // audio-token vocab per codebook: codes [0, codebook_size) + pad + eos
    audio_vocab_size: u32 = 2051, // CB_VOCAB
    codebook_pad_token_id: u32 = 2050, // CB_PAD (lm_head row 2050)
    codebook_size: u32 = 2048, // mimi codebook_size, ids >= this never sample

    // text encoder (T5Gemma2)
    te_hidden: u32 = 1152,
    te_heads: u32 = 4,
    te_kv: u32 = 1,
    te_head_dim: u32 = 256,
    te_inter: u32 = 6912,
    te_layers: u32 = 26,
    te_window: u32 = 512,
    te_theta_full: f64 = 1_000_000,
    te_theta_slide: f64 = 10_000,
    te_factor_full: f64 = 8.0,
    te_rms_eps: f32 = 1e-6,

    // backbone (Qwen3)
    b_hidden: u32 = 2048,
    b_heads: u32 = 16,
    b_kv: u32 = 8,
    b_q_dim: u32 = 2048,
    b_kv_dim: u32 = 1024,
    b_head_dim: u32 = 128,
    b_inter: u32 = 6144,
    b_layers: u32 = 28,
    b_theta: f64 = 1_000_000,
    b_rms_eps: f32 = 1e-6,

    // depth decoder (llama-100M flavor)
    d_hidden: u32 = 1024,
    d_heads: u32 = 8,
    d_kv: u32 = 2,
    d_head_dim: u32 = 128,
    d_inter: u32 = 8192,
    d_layers: u32 = 12,
    d_theta: f64 = 500_000,
    d_rms_eps: f32 = 1e-5,
    d_num_codebooks: u32 = 16,

    // the position window both decoders share
    max_seq: u32 = 2048,

    pub inline fn cbVocab(self: *const Config) u32 {
        return self.audio_vocab_size;
    }
    pub inline fn cbLm(self: *const Config) u32 {
        return self.audio_vocab_size + 1;
    }
    pub inline fn cbStride(self: *const Config) u32 {
        return self.audio_vocab_size;
    }
    pub inline fn cbEos(self: *const Config) u32 {
        return self.audio_vocab_size - 1;
    }

    /// The frame loop is 16 codebooks wide; every vocab size the rows carry
    /// comes from the tensors, so a smaller pack parses. `audio_vocab_size`
    /// feeds the 3-D depth head's table and the audio-table stride, both read
    /// at their tensor shapes.
    pub fn supportedLayout(self: *const Config) bool {
        return self.num_codebooks == 16;
    }
};

/// Read the geometry from `config.json`. A value outside the layout the engine
/// implements is refused by name, never computed wrong.
pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) (error{ BadPackConfig } || error{ UnsupportedBreezeGeometry })!Config {
    var cfg = Config{};
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return error.BadPackConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadPackConfig;
    const o = parsed.value.object;

    try fields.fill(Config, .{
        .{ "audio_token_id", "audio_token_id" },
        .{ "audio_eos_token_id", "audio_eos_id" },
        .{ "audio_vocab_size", "audio_vocab_size" },
        .{ "codebook_pad_token_id", "codebook_pad_token_id" },
        .{ "num_codebooks", "num_codebooks" },
        .{ "text_vocab_size", "te_vocab" },
        .{ "max_position_embeddings", "max_seq" },
    }, o, &cfg);
    // mimi's codebook_size ships in the codec section; fall back to vocab−2
    // (codes + pad + eos) so a pack stating only audio_vocab_size parses.
    if (o.get("codec_config")) |cc| if (cc == .object) {
        try fields.fill(Config, .{.{ "codebook_size", "codebook_size" }}, cc.object, &cfg);
    };
    if (o.get("codebook_size")) |v| {
        if (v == .integer and v.integer > 0) cfg.codebook_size = @intCast(v.integer);
    } else {
        cfg.codebook_size = cfg.audio_vocab_size - 2;
    }

    const te = try fields.section(o, "text_encoder_config");
    if (te.get("eoi_token_index")) |v| {
        if (v == .integer and v.integer >= 0) cfg.eoi_token_id = @intCast(v.integer);
    }
    try fields.fill(Config, .{
        .{ "hidden_size", "te_hidden" },
        .{ "num_attention_heads", "te_heads" },
        .{ "num_key_value_heads", "te_kv" },
        .{ "head_dim", "te_head_dim" },
        .{ "intermediate_size", "te_inter" },
        .{ "num_hidden_layers", "te_layers" },
        .{ "sliding_window", "te_window" },
        .{ "rms_norm_eps", "te_rms_eps" },
    }, te, &cfg);
    if (te.get("rope_parameters")) |rp| if (rp == .object) {
        if (rp.object.get("full_attention")) |fa| if (fa == .object) {
            try fields.fill(Config, .{
                .{ "rope_theta", "te_theta_full" },
                .{ "factor", "te_factor_full" },
            }, fa.object, &cfg);
        };
        if (rp.object.get("sliding_attention")) |sa| if (sa == .object) {
            try fields.fill(Config, .{.{ "rope_theta", "te_theta_slide" }}, sa.object, &cfg);
        };
    };

    const bb = try fields.section(o, "backbone_config");
    try fields.fill(Config, .{
        .{ "hidden_size", "b_hidden" },
        .{ "num_attention_heads", "b_heads" },
        .{ "num_key_value_heads", "b_kv" },
        .{ "head_dim", "b_head_dim" },
        .{ "intermediate_size", "b_inter" },
        .{ "num_hidden_layers", "b_layers" },
        .{ "rope_theta", "b_theta" },
        .{ "rms_norm_eps", "b_rms_eps" },
    }, bb, &cfg);

    const dd = try fields.section(o, "depth_decoder_config");
    try fields.fill(Config, .{
        .{ "hidden_size", "d_hidden" },
        .{ "num_attention_heads", "d_heads" },
        .{ "num_key_value_heads", "d_kv" },
        .{ "head_dim", "d_head_dim" },
        .{ "intermediate_size", "d_inter" },
        .{ "num_hidden_layers", "d_layers" },
        .{ "rope_theta", "d_theta" },
        .{ "rms_norm_eps", "d_rms_eps" },
        .{ "num_codebooks", "d_num_codebooks" },
    }, dd, &cfg);

    if (o.get("text_encoder_special_tokens_config")) |st| if (st == .object) {
        if (st.object.get("token_ids")) |ti| if (ti == .object) {
            if (ti.object.get("<ins_bos>")) |v| { if (v == .integer and v.integer >= 0) cfg.ins_bos_id = @intCast(v.integer); }
            if (ti.object.get("<ins_eos>")) |v| { if (v == .integer and v.integer >= 0) cfg.ins_eos_id = @intCast(v.integer); }
            if (ti.object.get("[S0]")) |v| { if (v == .integer and v.integer >= 0) cfg.speaker_base = @intCast(v.integer); }
        };
    };

    if (!cfg.supportedLayout()) return error.UnsupportedBreezeGeometry;
    if (@rem(cfg.d_hidden, cfg.d_head_dim) != 0 or @rem(cfg.te_heads, cfg.te_kv) != 0 or
        @rem(cfg.b_heads, cfg.b_kv) != 0 or @rem(cfg.d_heads, cfg.d_kv) != 0 or
        cfg.codebook_size >= cfg.audio_vocab_size) return error.BadPackConfig;
    return cfg;
}

/// Parse the pack's `config.json` through `dir` (the MLX-free read the host allows).
pub fn readPackConfig(dir: *const abi.PackDir) ?Config {
    var buf: [256 << 10]u8 = undefined;
    var why: abi.Verdict = .{};
    const txt = dir.readFile("config.json", &buf, &why) orelse return null;
    var fba_buf: [512 << 10]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    return parseConfig(fba.allocator(), txt) catch null;
}

/// What a request holds above the resident weights: the frame-loop host rows
/// (cond, stream, span at the window, depth span, the codec decode's latent +
/// waveform activations over the frame budget). The frame budget comes from
/// the body's `text` length and its `max_frames` cap; the engine refuses a
/// request past the window at step time, this keeps admission honest.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m };
    const hints = parseBodyHints(body.slice()) orelse BodyHints{ .max_frames = 1500, .text_len = 2048 };
    const hf: u64 = hints.max_frames;
    const ht: u64 = hints.text_len;
    const window: u64 = 2048;
    const seq: u64 = @min(window, ht + hf);
    const frames: u64 = @min(@as(u64, 1500), hf);
    // Backbone rows ride stream + span (2 × f32[2048]); the depth span is
    // (16+1) rows at f32[1024] per frame; the codec decode holds a latent
    // chain (f32[1024] per frame) plus the waveform (2 × f32[1920]).
    const seq_bytes = seq * @as(u64, 16384); // stream + span: 2 × f32[2048]
    const depth_bytes = frames * @as(u64, 69632); // (16+1) × f32[1024]
    const codec_bytes = frames * @as(u64, 19456); // latent + waveform chain
    const row_bytes = seq_bytes + depth_bytes + codec_bytes;
    // Margin for the TE cond buffer, evalHost staging and allocator slack.
    return row_bytes + (row_bytes / 4) + (128 << 20);
}

const BodyHints = struct { max_frames: u64, text_len: u64 };

/// A cheap read of the body's `max_frames` and text length; null if the JSON
/// is unparseable (the engine refuses it by name inside `step`; the bill
/// defaults to the worst-case frame budget).
fn parseBodyHints(body: []const u8) ?BodyHints {
    if (body.len == 0) return null;
    var fba_buf: [256 << 10]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const v = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), body, .{}) catch return null;
    if (v != .object) return null;
    const o = v.object;
    const text = o.get("text") orelse o.get("input") orelse return null;
    if (text != .string) return null;
    var max_frames: u64 = 1500;
    if (o.get("max_frames")) |f| { if (f == .integer and f.integer > 0) max_frames = @min(1500, @as(u64, @intCast(f.integer))); }
    return .{ .max_frames = max_frames, .text_len = text.string.len };
}

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .speech = true } };
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "");
}

// ── tests ──

const testing = std.testing;

const real_pack = \\{"model_type":"breeze","audio_token_id":262144,"audio_eos_token_id":262145,
 \\ "audio_vocab_size":2051,"codebook_pad_token_id":2050,"num_codebooks":16,
 \\ "text_vocab_size":262158,"max_position_embeddings":2048,
 \\ "text_encoder_config":{"hidden_size":1152,"num_attention_heads":4,"num_key_value_heads":1,
 \\   "head_dim":256,"intermediate_size":6912,"num_hidden_layers":26,"sliding_window":512,
 \\   "rms_norm_eps":1e-05,
 \\   "rope_parameters":{"full_attention":{"rope_theta":1000000,"factor":8.0},
 \\     "sliding_attention":{"rope_theta":10000}}},
 \\ "backbone_config":{"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":8,
 \\   "head_dim":128,"intermediate_size":6144,"num_hidden_layers":28,"rope_theta":1000000,
 \\   "rms_norm_eps":1e-06},
 \\ "depth_decoder_config":{"hidden_size":1024,"num_attention_heads":8,"num_key_value_heads":2,
 \\   "head_dim":128,"intermediate_size":8192,"num_hidden_layers":12,"rope_theta":500000,
 \\   "rms_norm_eps":1e-05,"num_codebooks":16},
 \\ "text_encoder_special_tokens_config":{"token_ids":{"<ins_bos>":262156,"<ins_eos>":262157,"[S0]":262146}}}
;

test "parseConfig: the published pack's config.json reads as-is" {
    const cfg = try parseConfig(testing.allocator, real_pack);
    try testing.expectEqual(@as(u32, 28), cfg.b_layers);
    try testing.expectEqual(@as(u32, 26), cfg.te_layers);
    try testing.expectEqual(@as(u32, 12), cfg.d_layers);
    try testing.expectEqual(@as(u32, 2048), cfg.b_hidden);
    try testing.expectEqual(@as(u32, 1152), cfg.te_hidden);
    try testing.expectEqual(@as(f64, 500000), cfg.d_theta);
    try testing.expectEqual(@as(u32, 262156), cfg.ins_bos_id);
    try testing.expectEqual(@as(u32, 262146), cfg.speaker_base);
    try testing.expectEqual(@as(u32, 2051), cfg.cbVocab());
    try testing.expectEqual(@as(u32, 2052), cfg.cbLm());
    try testing.expect(cfg.supportedLayout());
}

test "parseConfig: a fixture pack states only what it shrinks" {
    const cfg = try parseConfig(testing.allocator,
        \\{"model_type":"breeze","audio_vocab_size":35,"num_codebooks":16,
        \\ "text_encoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":1,
        \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"sliding_window":8,"rms_norm_eps":1e-05,
        \\   "rope_parameters":{"full_attention":{"rope_theta":1000000,"factor":8.0},"sliding_attention":{"rope_theta":10000}}},
        \\ "backbone_config":{"hidden_size":128,"num_attention_heads":4,"num_key_value_heads":2,
        \\   "head_dim":32,"intermediate_size":256,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-06},
        \\ "depth_decoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,
        \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-05,"num_codebooks":16},
        \\ "text_vocab_size":200,"max_position_embeddings":64,
        \\ "eoi_token_index":100}
    );
    try testing.expectEqual(@as(u32, 1), cfg.b_layers);
    try testing.expectEqual(@as(u32, 128), cfg.b_hidden);
    try testing.expectEqual(@as(u32, 35), cfg.cbVocab());
    try testing.expectEqual(@as(u32, 36), cfg.cbLm());
}

test "parseConfig: an unsupported codebook count is refused by name" {
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator,
        \\{"model_type":"breeze","num_codebooks":8,"audio_vocab_size":2051}
    ));
    try testing.expectError(error.BadPackConfig, parseConfig(testing.allocator,
        \\{"model_type":"breeze","num_codebooks":16,"audio_vocab_size":1027,"codebook_size":2048}
    ));
}

fn noDir() abi.PackDir {
    const V = abi.Verdict{};
    return .{
        .ctx = null,
        .exists = struct { fn f(_: ?*anyopaque, _: abi.Str) callconv(.c) bool { return false; } }.f,
        .read = struct { fn f(_: ?*anyopaque, _: abi.Str, _: [*]u8, _: usize, _: *usize) callconv(.c) abi.Verdict { return V; } }.f,
        .safetensors_header = struct { fn f(_: ?*anyopaque, _: abi.Str, _: [*]u8, _: usize, _: *usize) callconv(.c) abi.Verdict { return V; } }.f,
        .list = struct { fn f(_: ?*anyopaque, _: abi.Str, _: [*]u8, _: usize, _: *usize) callconv(.c) abi.Verdict { return V; } }.f,
        .file_bytes = struct { fn f(_: ?*anyopaque, _: abi.Str) callconv(.c) u64 { return 0; } }.f,
    };
}

test "billRequest: a tiny request bills frames+window; the worst case stays under 2 GiB" {
    const mib = 1 << 20;
    const dir = noDir();
    const tiny = billRequest(&dir, &.{}, abi.Str.of("{\"text\":\"hi\",\"max_frames\":9}"));
    try testing.expect(tiny > 0);
    const worst = billRequest(&dir, &.{}, abi.Str.of("{\"text\":\"x\",\"max_frames\":1500}"));
    // the worst case (a full 2048-row span + 1500-frame decode) sits under a
    // GiB and far under what admission can grant beside the load bill
    try testing.expect(worst > tiny);
    try testing.expect(worst < 1024 * mib);
    try testing.expect(tiny < 200 * mib);
    // an unparseable body still bills a full-frame worst case
    try testing.expect(billRequest(&dir, &.{}, abi.Str.of("{oops")) >= worst);
}
