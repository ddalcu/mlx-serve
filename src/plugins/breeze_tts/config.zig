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
/// Config defaults are the published Breeze-TTS-2 geometry.
const SENT: u32 = std.math.maxInt(u32);

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
    /// `full_attention` layers in the `layer_types` ladder (published pack: 5/11/17/23)
    te_full_layers: [16]u32 = .{ 5, 11, 17, 23, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT, SENT },
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
    /// Runaway frame ceiling (generation max_new_tokens is a sampling default,
    /// not a model limit; 1500 matches the reference bridge at 12.5 Hz).
    max_frames: u32 = 1500,

    pub inline fn numCodebooks(self: *const Config) usize {
        return @intCast(self.num_codebooks);
    }
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
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return error.BadPackConfig;
    defer parsed.deinit();
    return parseFromValue(parsed.value);
}

/// `rope_scaling` on the two qwen stacks (the engine ropes with plain theta).
/// A missing or null arm is plain theta and loads. A declared arm must be the
/// published llama3 arm EXACTLY — factor 32 with `original_max_position_
/// embeddings` at the section's published value (depth decoder 16, root 1024)
/// — which is the shape the sampled oracle reproduces at plain theta (within
/// the depth span positions never pass 16/16; the oracle shows early eos on
/// the scaled table). ANY other arm (yarn, linear, a different factor, an
/// original_max off the published value) is a rotation this engine does not
/// implement and is refused BY NAME. The TE's `rope_parameters` are separate:
/// read into `te_theta_*`/`te_factor_full` above, implemented as a position
/// scale, and their factors checked to the published values.
fn checkRopeScaling(v: ?std.json.Value, comptime published_ompe: i64) (error{ BadPackConfig } || error{ UnsupportedBreezeGeometry })!void {
    const s = v orelse return;
    if (s == .null) return;
    if (s != .object) return error.BadPackConfig;
    const o = s.object;
    const rt = o.get("rope_type") orelse return error.UnsupportedBreezeGeometry;
    if (!(rt == .string and std.mem.eql(u8, rt.string, "llama3"))) return error.UnsupportedBreezeGeometry;
    const factor = o.get("factor") orelse return error.UnsupportedBreezeGeometry;
    const f: f64 = switch (factor) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => return error.UnsupportedBreezeGeometry,
    };
    if (f != 32.0) return error.UnsupportedBreezeGeometry;
    const ompe = o.get("original_max_position_embeddings") orelse return error.UnsupportedBreezeGeometry;
    if (ompe != .integer or ompe.integer != published_ompe) return error.UnsupportedBreezeGeometry;
}

fn parseFromValue(value: std.json.Value) (error{ BadPackConfig } || error{ UnsupportedBreezeGeometry })!Config {
    var cfg = Config{};
    if (value != .object) return error.BadPackConfig;
    const o = value.object;

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
    if (te.get("layer_types")) |lt| if (lt == .array) {
        var n: usize = 0;
        for (lt.array.items, 0..) |v, i| {
            if (v != .string) return error.BadPackConfig;
            if (std.mem.eql(u8, v.string, "full_attention")) {
                if (n == cfg.te_full_layers.len - 1) return error.UnsupportedBreezeGeometry;
                cfg.te_full_layers[n] = @intCast(i);
                n += 1;
            }
        }
        cfg.te_full_layers[n] = SENT;
        @memset(cfg.te_full_layers[n + 1 ..], SENT);
    } else return error.BadPackConfig;
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

    // Each stack's OWN arm governs that stack (mlx-lm's apply_rope_scaling
    // rewrites per section). The depth decoder's arm must be the published
    // llama3 arm exactly (original_max 16: inert across the 32-token depth
    // span, plain theta matches the oracle); the backbone's is null, and the
    // published root block (llama3, factor 32, original_max 1024) must be
    // that exact shape too — the engine ropes both with plain theta and the
    // sampled oracle reproduces it. A missing arm declares nothing and loads.
    // The TE's `rope_parameters` are separate: the arms it implements are the
    // theta pair plus the published full-attention linear factor; any other
    // declared rotation is refused by name, never decoded wrong.
    try checkRopeScaling(dd.get("rope_scaling"), 16);
    try checkRopeScaling(o.get("rope_scaling"), 1024);
    if (te.get("rope_parameters")) |rp| if (rp == .object) {
        // the TE's implemented arms: a full-attention block declares the
        // published linear factor (a block that declares none is plain
        // theta, the same arm the defaults carry) and a sliding block
        // declares plain theta only.
        const want: f64 = 8.0; // the published full-attention linear factor
        if (rp.object.get("full_attention")) |fa| if (fa == .object) {
            if (fa.object.get("factor")) |f| {
                const ok = (f == .float and f.float == want) or (f == .integer and @as(f64, @floatFromInt(f.integer)) == want);
                if (!ok) return error.UnsupportedBreezeGeometry;
            } else {
                if (want != 1.0) return error.UnsupportedBreezeGeometry;
            }
        } else return error.UnsupportedBreezeGeometry;
        if (rp.object.get("sliding_attention")) |sa| if (sa == .object) {
            if (sa.object.get("factor")) |f| {
                const ok = (f == .float and f.float == 1.0) or (f == .integer and f.integer == 1);
                if (!ok) return error.UnsupportedBreezeGeometry;
            }
        };
    };

    if (!cfg.supportedLayout()) return error.UnsupportedBreezeGeometry;
    if (@rem(cfg.d_hidden, cfg.d_head_dim) != 0 or @rem(cfg.te_heads, cfg.te_kv) != 0 or
        @rem(cfg.b_heads, cfg.b_kv) != 0 or @rem(cfg.d_heads, cfg.d_kv) != 0 or
        cfg.codebook_size >= cfg.audio_vocab_size) return error.BadPackConfig;
    return cfg;
}

/// Read and parse `model_dir/config.json` (the engine's own file read; a
/// missing file is the published geometry, a broken one is refused by name).
pub fn readPackConfig(io: std.Io, alloc: std.mem.Allocator, model_dir: []const u8) !Config {
    var cfg = try readRootConfig(io, alloc, model_dir);
    // generation_config's max_new_tokens (750 = 60 s) is a sampling default,
    // not a model limit; the bridge's 1500 keeps speech under the window.
    const gp = std.fmt.allocPrint(alloc, "{s}/generation_config.json", .{model_dir}) catch return cfg;
    defer alloc.free(gp);
    if (std.Io.Dir.openFileAbsolute(io, gp, .{})) |f| {
        defer f.close(io);
        var gbuf: [4096]u8 = undefined;
        var grs = f.reader(io, &gbuf);
        const gtxt = grs.interface.allocRemaining(alloc, .limited(1 << 16)) catch return cfg;
        defer alloc.free(gtxt);
        var gparsed = std.json.parseFromSlice(std.json.Value, alloc, gtxt, .{}) catch return cfg;
        defer gparsed.deinit();
        if (gparsed.value == .object) {
            var tmp = cfg;
            fields.fill(Config, .{.{ "max_new_tokens", "max_frames" }}, gparsed.value.object, &tmp) catch {};
            cfg = tmp;
        }
    } else |_| {}
    return cfg;
}

/// `config.json` alone (what `billRequest` reads): missing = published defaults.
pub fn readRootConfig(io: std.Io, alloc: std.mem.Allocator, model_dir: []const u8) !Config {
    const path = try std.fmt.allocPrint(alloc, "{s}/config.json", .{model_dir});
    defer alloc.free(path);
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return Config{};
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const txt = try rs.interface.allocRemaining(alloc, .limited(4 << 20));
    defer alloc.free(txt);
    return parseConfig(alloc, txt);
}

/// The host's view of the same read, through the PackDir seam.
fn readDirConfig(dir: *const abi.PackDir) !Config {
    var buf: [1 << 20]u8 = undefined;
    var len: usize = 0;
    const v = dir.read(dir.ctx, abi.Str.of("config.json"), &buf, buf.len, &len);
    if (v.code != .ok) return error.BadPackConfig;
    const gpa = std.heap.c_allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, buf[0..len], .{});
    defer parsed.deinit();
    return parseFromValue(parsed.value);
}

/// What a request holds above the resident weights: the frame-loop host rows
/// (cond, stream, span at the window, depth span, the codec decode's latent +
/// waveform activations over the frame budget). The frame budget comes from
/// the body's `text` length and its `max_frames` cap; the engine refuses a
/// request past the window at step time, this keeps admission honest.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = m;
    // Config reads go through the host's PackDir seam; the engine reads the
    // same file directly at load (MLX code may not import the host).
    const cfg = readDirConfig(dir) catch Config{};
    const hints = parseBodyHints(body.slice(), cfg.max_frames) orelse
        BodyHints{ .max_frames = cfg.max_frames, .text_len = cfg.max_seq };
    const hf: u64 = hints.max_frames;
    const ht: u64 = hints.text_len;
    const window: u64 = cfg.max_seq;
    const seq: u64 = @min(window, ht + hf);
    const frames: u64 = @min(@as(u64, cfg.max_frames), hf);
    const bh: u64 = cfg.b_hidden;
    const dh: u64 = cfg.d_hidden;
    const spp: u64 = 1920; // codec samples per frame
    // Backbone rows ride stream + span (2 × f32[b_hidden]); the depth span is
    // (num_codebooks+1) rows at f32[d_hidden] per frame; the codec decode holds
    // a latent chain (f32[1024] per frame) plus the waveform (2 × f32[spp]).
    const seq_bytes = seq * (bh * 8); // stream + span: 2 × f32[b_hidden]
    const depth_bytes = frames * ((@as(u64, cfg.num_codebooks) + 1) * dh * 4);
    const codec_bytes = frames * (4096 + spp * 8); // latent + waveform chain
    const row_bytes = seq_bytes + depth_bytes + codec_bytes;
    // Margin for the TE cond buffer, evalHost staging and allocator slack.
    return row_bytes + (row_bytes / 4) + (128 << 20);
}

const BodyHints = struct { max_frames: u64, text_len: u64 };

/// A cheap read of the body's `max_frames` and text length; null if the JSON
/// is unparseable (the engine refuses it by name inside `step`; the bill
/// defaults to the worst-case frame budget).
fn parseBodyHints(body: []const u8, default_max: u32) ?BodyHints {
    if (body.len == 0) return null;
    var fba_buf: [256 << 10]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const v = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), body, .{}) catch return null;
    if (v != .object) return null;
    const o = v.object;
    const text = o.get("text") orelse o.get("input") orelse return null;
    if (text != .string) return null;
    var max_frames: u64 = default_max;
    if (o.get("max_frames")) |f| {
        if (f == .integer and f.integer > 0) max_frames = @min(default_max, @as(u64, @intCast(f.integer)));
    }
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
 \\   "rms_norm_eps":1e-05,"num_codebooks":16,
 \\   "rope_scaling":{"rope_type":"llama3","factor":32.0,"original_max_position_embeddings":16,
 \\     "low_freq_factor":0.001953125,"high_freq_factor":0.0078125}},
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

test "parseConfig: rope_scaling — the published inert arms load, any other is refused by name" {
    const base =
        \\{"model_type":"breeze","audio_vocab_size":35,"num_codebooks":16,
        \\ "text_encoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":1,
        \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"sliding_window":8,"rms_norm_eps":1e-05},
        \\ "backbone_config":{"hidden_size":128,"num_attention_heads":4,"num_key_value_heads":2,
        \\   "head_dim":32,"intermediate_size":256,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-06},
        \\ "depth_decoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,
        \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-05}}
    ;
    const withRoot = struct {
        fn f(comptime tail: []const u8) []const u8 {
            return base[0 .. base.len - 1] ++ tail;
        }
    }.f;
    // absent (the string above) and explicit null load on plain theta.
    _ = try parseConfig(testing.allocator, base);
    _ = try parseConfig(testing.allocator, withRoot(
        \\,"rope_scaling":null}
    ));
    // the published depth-decoder arm: llama3, factor 32, original_max 16.
    _ = try parseConfig(testing.allocator,
        \\{"model_type":"breeze","audio_vocab_size":35,"num_codebooks":16,
        \\ "text_encoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":1,
        \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"sliding_window":8,"rms_norm_eps":1e-05},
        \\ "backbone_config":{"hidden_size":128,"num_attention_heads":4,"num_key_value_heads":2,
        \\   "head_dim":32,"intermediate_size":256,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-06},
        \\ "depth_decoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,
        \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-05,
        \\   "rope_scaling":{"rope_type":"llama3","factor":32.0,"original_max_position_embeddings":16}}}
    );
    // any other declared scaling is refused, never decoded wrong.
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator, withRoot(
        \\,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":4096}}
    )));
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator, withRoot(
        \\,"rope_scaling":{"rope_type":"linear","factor":4.0}}
    )));
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator, withRoot(
        \\,"rope_scaling":{"rope_type":"llama3","factor":8.0,"original_max_position_embeddings":16}}
    )));
    // the root section's published arm is llama3/32/1024 exactly.
    _ = try parseConfig(testing.allocator, withRoot(
        \\,"rope_scaling":{"rope_type":"llama3","factor":32.0,"original_max_position_embeddings":1024}}
    ));
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator, withRoot(
        \\,"rope_scaling":{"rope_type":"llama3","factor":32.0,"original_max_position_embeddings":16}}
    )));
    const te_json = struct {
        fn f(comptime factor: []const u8) []const u8 {
            const head =
                \\{"model_type":"breeze","audio_vocab_size":35,"num_codebooks":16,
                \\ "text_encoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":1,
                \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"sliding_window":8,"rms_norm_eps":1e-05,
                \\   "rope_parameters":{"full_attention":{"rope_theta":1000000,"factor":
            ;
            const tail =
                \\},"sliding_attention":{"rope_theta":10000}}},
                \\ "backbone_config":{"hidden_size":128,"num_attention_heads":4,"num_key_value_heads":2,
                \\   "head_dim":32,"intermediate_size":256,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-06},
                \\ "depth_decoder_config":{"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,
                \\   "head_dim":16,"intermediate_size":128,"num_hidden_layers":1,"rope_theta":10000,"rms_norm_eps":1e-05}}
            ;
            return head ++ factor ++ tail;
        }
    }.f;
    // the TE's full-attention factor must be the published 8.0.
    _ = try parseConfig(testing.allocator, te_json("8.0"));
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator, te_json("4.0")));
    try testing.expectError(error.UnsupportedBreezeGeometry, parseConfig(testing.allocator, te_json("16.0")));
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
