//! YuE2-3B (m-a-p/YuE2-3B) — lyrics + style → 48 kHz stereo song (`.audio`, `yue2`).
//!
//! Loads the ahmadw/YuE2-3B-MLX layout (bf16, 8-bit or mixed 4/8-bit): `config.json`,
//! `model.safetensors` (checkpoint names, `time_embedder.fc1/fc2`), `vae.safetensors`
//! (decoder only, weight norm folded, MLX conv layout), `vae_config.json`, `qwen.tiktoken`,
//! `yue2_generation_config.json`. The math follows yue2_infer's native protocol:
//!
//!   style + lyrics → [ABC score, AR, `cot` melody|full] → semantic codec ids (AR, CFG)
//!   → acoustic latents (NAR flow matching, 32 midpoint steps, per context chunk)
//!   → Oobleck VAE decoder (tiled, f32) → WAV.
//!
//! One Qwen3-shaped trunk carries two weight sets per layer (AR for text + codec ids, NAR
//! for the latent canvas); the NAR queries attend over the AR path's K/V of the whole prefix.
//! Sampling runs on the host (the candidate set is 32k ids), seeded per phase.
//! Parity: env-gated `YUE2_*` oracles fed by tests/dump_yue2_fixtures.py.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const tok_mod = @import("tokenizer.zig");
const wav_mod = @import("wav.zig");
const sse = @import("gen_sse.zig");
const Scope = @import("mlx_scope.zig").Scope;

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const Weights = model_mod.Weights;

pub const SAMPLE_RATE: u32 = 48000;
/// Codec frames per second (one semantic id = 1920 samples).
pub const FRAME_RATE: u32 = 25;
pub const MIN_SECONDS: u32 = 5;
pub const MAX_SECONDS: u32 = 360;
pub const CONTEXT: u32 = 24576;
pub const DEFAULT_STEPS: u32 = 32;
pub const MAX_STEPS: u32 = 100;
pub const MAX_CFG: f32 = 20;

const EOD: u32 = 151643;
const ABC_START: u32 = 151847;
const ABC_END: u32 = 151848;
const MUSIC_START: u32 = 151851;
const MUSIC_END: u32 = 151852;
const CODEC_OFFSET: u32 = 151853;
const CODEC_SIZE: u32 = 32768;
const VOCAB: u32 = 184704;
const LATENT_DIM: c_int = 64;
const ABC_MAX_TOKENS: u32 = 4096;

pub const Cot = enum {
    off,
    melody,
    full,

    pub fn parse(s: []const u8) ?Cot {
        return std.meta.stringToEnum(Cot, s);
    }

    fn instruction(self: Cot) []const u8 {
        return switch (self) {
            .off => "Generate music with codec tokens from the given conditions.",
            .melody => "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
            .full => "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
        };
    }

    /// Classifier-free guidance when the request names none.
    fn guidance(self: Cot) f32 {
        return if (self == .off) 1.01 else 1.0;
    }
};

pub const Sampling = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 0.95,
    top_k: u32 = 100,
    repetition_penalty: f32 = 1.2,
    penalty_window: u32 = 50,
    min_tokens: u32 = 200,
    max_tokens: u32 = 9000,

    fn valid(self: Sampling) bool {
        return std.math.isFinite(self.temperature) and self.temperature >= 0 and self.temperature <= 5 and
            std.math.isFinite(self.top_p) and self.top_p > 0 and self.top_p <= 1 and self.top_k >= 1 and
            std.math.isFinite(self.repetition_penalty) and self.repetition_penalty > 0 and
            self.penalty_window >= 1 and self.penalty_window <= 100 and
            self.min_tokens <= self.max_tokens and self.max_tokens >= 1;
    }
};

const ABC_SAMPLING = Sampling{ .temperature = 0.7, .top_p = 0.9, .top_k = 30, .repetition_penalty = 1.005, .penalty_window = 100, .min_tokens = 32, .max_tokens = ABC_MAX_TOKENS };

/// `yue2_generation_config.json`.
const GenFile = struct {
    abc: Sampling = ABC_SAMPLING,
    semantic: Sampling = .{},
    ode_steps: u32 = DEFAULT_STEPS,
};

pub const Cfg = struct {
    hidden: c_int,
    layers: u32,
    heads: c_int,
    kv_heads: c_int,
    hd: c_int,
    ffn: u32,
    eps: f32,
    rope_theta: f32,
    max_latent_frames: c_int,
};

const ConfigFile = struct {
    hidden_size: c_int,
    num_hidden_layers: u32,
    num_attention_heads: c_int,
    num_key_value_heads: c_int,
    head_dim: c_int,
    intermediate_size: u32,
    vocab_size: u32,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1e6,
    latent_dim: c_int,
    max_latent_frames: c_int,
    timestep_shift: f32 = 1.0,
};

/// A config this engine cannot serve is a NAMED load error, never a mis-shaped forward.
pub fn parseConfig(a: std.mem.Allocator, bytes: []const u8) !Cfg {
    var parsed = std.json.parseFromSlice(ConfigFile, a, bytes, .{ .ignore_unknown_fields = true }) catch return error.Yue2ConfigInvalid;
    defer parsed.deinit();
    const c = parsed.value;
    if (c.vocab_size != VOCAB or c.latent_dim != LATENT_DIM or c.timestep_shift != 1.0) return error.Yue2ConfigUnsupported;
    if (c.hidden_size <= 0 or c.num_hidden_layers == 0 or c.head_dim <= 0 or c.num_key_value_heads <= 0 or
        c.intermediate_size == 0 or c.num_attention_heads <= 0 or @rem(c.num_attention_heads, c.num_key_value_heads) != 0 or c.max_latent_frames <= 0)
        return error.Yue2ConfigInvalid;
    return .{
        .hidden = c.hidden_size,
        .layers = c.num_hidden_layers,
        .heads = c.num_attention_heads,
        .kv_heads = c.num_key_value_heads,
        .hd = c.head_dim,
        .ffn = c.intermediate_size,
        .eps = c.rms_norm_eps,
        .rope_theta = c.rope_theta,
        .max_latent_frames = c.max_latent_frames,
    };
}

const VaeFile = struct {
    decoder_config: struct {
        strides: []const c_int,
        latent_dim: c_int,
        out_channels: c_int = 2,
    },
    sample_rate: u32 = SAMPLE_RATE,
    decode_core_frames: u32 = 1024,
    decode_halo_frames: u32 = 16,
};

/// Samples a decoder turns `frames` latents into: each transposed conv (kernel 2s, padding ceil(s/2))
/// yields L*s + s - 2*ceil(s/2), so odd strides lose a sample per stage.
pub fn decodedLen(strides: []const c_int, frames: usize) usize {
    var n: usize = frames;
    var i = strides.len;
    while (i > 0) {
        i -= 1;
        const s: usize = @intCast(strides[i]);
        n = n * s + s - 2 * ((s + 1) / 2);
    }
    return n;
}

fn hopOf(strides: []const c_int) usize {
    var h: usize = 1;
    for (strides) |s| h *= @intCast(s);
    return h;
}

/// Fixed-length context chunks of codec frames the NAR solves independently.
pub fn chunkSize(prefix_len: usize) usize {
    const room = (@as(usize, CONTEXT) -| prefix_len -| 3) / 2;
    return @min(room, CONTEXT);
}

fn flowTime(step: u32, steps: u32) f32 {
    return 1.0 - @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(steps));
}

/// The sigmoid-domain time the model embeds: logit(t) clamped to ±20.
pub fn logitTime(t: f32) f32 {
    if (t <= 0) return -20;
    if (t >= 1) return 20;
    return std.math.clamp(@log(t / (1 - t)), -20, 20);
}

// ════════════════════════════════════════════════════════════════════════
// Prompt protocol
// ════════════════════════════════════════════════════════════════════════

/// The chat-less request text the AR path reads.
pub fn promptText(a: std.mem.Allocator, cot: Cot, style: []const u8, lyrics: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}\n[Tags]\n{s}\n[Lyrics]\n{s}\n", .{ cot.instruction(), style, lyrics });
}

/// `[EOD] ++ text ++ ABC section ++ MUSIC_START`; with no ABC ids yet the section stays open
/// (`ABC_START` only) for the planning phase to continue.
pub fn assemblePrefix(a: std.mem.Allocator, text_ids: []const u32, cot: Cot, abc: ?[]const u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    try out.append(a, EOD);
    try out.appendSlice(a, text_ids);
    try out.append(a, ABC_START);
    if (cot == .off) {
        try out.appendSlice(a, &.{ ABC_END, MUSIC_START });
    } else if (abc) |ids| {
        try out.appendSlice(a, ids);
        try out.appendSlice(a, &.{ ABC_END, MUSIC_START });
    }
    return out.toOwnedSlice(a);
}

/// The unconditional CFG branch: the instruction alone, with the same score.
pub fn assembleNegative(a: std.mem.Allocator, instruction_ids: []const u32, cot: Cot, abc: []const u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    try out.append(a, EOD);
    try out.appendSlice(a, instruction_ids);
    if (cot != .off) {
        try out.append(a, ABC_START);
        try out.appendSlice(a, abc);
        try out.append(a, ABC_END);
    }
    try out.append(a, MUSIC_START);
    return out.toOwnedSlice(a);
}

// ════════════════════════════════════════════════════════════════════════
// Host sampling
// ════════════════════════════════════════════════════════════════════════

pub const Phase = enum {
    abc,
    semantic,

    fn end(self: Phase) u32 {
        return if (self == .abc) ABC_END else MUSIC_END;
    }
    fn lo(self: Phase) usize {
        return if (self == .abc) 0 else CODEC_OFFSET;
    }
    fn hi(self: Phase) usize {
        return if (self == .abc) EOD else CODEC_OFFSET + CODEC_SIZE;
    }
};

const Cand = struct {
    id: u32,
    v: f32,
    fn before(_: void, a: Cand, b: Cand) bool {
        return if (a.v != b.v) a.v > b.v else a.id < b.id;
    }
};

/// One draw from the reference's `distribution`: only the phase's ids and its end marker are
/// candidates; the rest are -inf and never touched. `scores` is the full-vocab logit row.
pub const Sampler = struct {
    a: std.mem.Allocator,
    prng: std.Random.DefaultPrng,
    top: std.ArrayList(Cand) = .empty,
    keep: std.ArrayList(Cand) = .empty,

    pub fn init(a: std.mem.Allocator, seed: u64) Sampler {
        return .{ .a = a, .prng = std.Random.DefaultPrng.init(seed) };
    }
    pub fn deinit(self: *Sampler) void {
        self.top.deinit(self.a);
        self.keep.deinit(self.a);
    }

    pub fn pick(self: *Sampler, scores: []f32, s: Sampling, history: []const u32, step: u32, phase: Phase, top_p_floor: usize) !u32 {
        const end = phase.end();
        // End marker is off the table until `min_tokens`; the window penalty reads the raw scores.
        if (step < s.min_tokens) scores[end] = -std.math.inf(f32);
        if (s.repetition_penalty != 1.0) penalize(scores, history[history.len -| s.penalty_window..], s.repetition_penalty);
        if (s.temperature == 0) return argmax(scores, phase);
        if (s.temperature != 1) {
            for (phase.lo()..phase.hi()) |i| scores[i] /= s.temperature;
            scores[end] /= s.temperature;
        }

        // Top-k by threshold: ties at the k-th score all survive, like the reference.
        const k: u32 = @intCast(@min(s.top_k, phase.hi() - phase.lo() + 1));
        self.top.clearRetainingCapacity();
        try self.top.ensureTotalCapacity(self.a, k + 1);
        for (phase.lo()..phase.hi()) |i| try self.offer(.{ .id = @intCast(i), .v = scores[i] }, k);
        try self.offer(.{ .id = end, .v = scores[end] }, k);
        const threshold = self.top.items[self.top.items.len - 1].v;

        self.keep.clearRetainingCapacity();
        for (phase.lo()..phase.hi()) |i| if (scores[i] >= threshold) try self.keep.append(self.a, .{ .id = @intCast(i), .v = scores[i] });
        if (scores[end] >= threshold) try self.keep.append(self.a, .{ .id = end, .v = scores[end] });
        const c = self.keep.items;
        std.mem.sort(Cand, c, {}, Cand.before);

        // Softmax over the survivors (every other id has probability 0), then nucleus.
        var total: f64 = 0;
        const best = c[0].v;
        for (c) |*x| {
            x.v = @floatCast(@exp(@as(f64, x.v - best)));
            total += x.v;
        }
        var cum: f64 = 0;
        var kept: f64 = 0;
        for (c, 0..) |x, i| {
            const p = x.v / total;
            const removed = s.top_p < 1 and i >= top_p_floor and cum > s.top_p;
            cum += p;
            if (removed) {
                c[i].v = 0;
            } else kept += x.v;
        }
        var r = self.prng.random().float(f64) * kept;
        for (c) |x| {
            if (x.v == 0) continue;
            r -= x.v;
            if (r < 0) return x.id;
        }
        return c[0].id;
    }

    /// Keep the k best candidates, sorted descending.
    fn offer(self: *Sampler, x: Cand, k: u32) !void {
        const t = &self.top;
        if (t.items.len == k and !Cand.before({}, x, t.items[k - 1])) return;
        var i = t.items.len;
        if (i == k) i -= 1 else t.appendAssumeCapacity(x);
        while (i > 0 and Cand.before({}, x, t.items[i - 1])) : (i -= 1) t.items[i] = t.items[i - 1];
        t.items[i] = x;
    }
};

fn penalize(scores: []f32, window: []const u32, penalty: f32) void {
    for (window, 0..) |id, i| {
        if (std.mem.indexOfScalar(u32, window[0..i], id) != null) continue;
        var freq: f32 = 0;
        for (window) |o| freq += @floatFromInt(@intFromBool(o == id));
        const alpha = std.math.pow(f32, penalty, freq);
        scores[id] = if (scores[id] < 0) scores[id] * alpha else scores[id] / alpha;
    }
}

fn argmax(scores: []const f32, phase: Phase) u32 {
    var best: u32 = phase.end();
    for (phase.lo()..phase.hi()) |i| {
        if (scores[i] > scores[best] or (scores[i] == scores[best] and i < best)) best = @intCast(i);
    }
    return best;
}

// ════════════════════════════════════════════════════════════════════════
// Weights
// ════════════════════════════════════════════════════════════════════════

/// A projection resolved once at load: affine-quantized (width solved from the packed
/// geometry, never a literal) or dense, with an optional bias. Handles are borrowed from `Weights`.
const Linear = struct {
    w: A,
    scales: ?A = null,
    biases: A = .{ .ctx = null },
    bias: ?A = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    fn forward(self: *const Linear, sc: *Scope, x: A) !A {
        const y = if (self.scales) |sca|
            try sc.qmm(x, self.w, sca, self.biases, self.group_size, self.bits)
        else
            try sc.matmul(x, try sc.transpose(self.w, &.{ 1, 0 }));
        return if (self.bias) |b| sc.add(y, b) else y;
    }
};

/// One of a layer's two weight sets.
const Path = struct {
    ln: A,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    qn: A,
    kn: A,
    mlp_ln: A,
    gate: Linear,
    up: Linear,
    down: Linear,
};

const Layer = struct { ar: Path, nar: Path };

/// A layer's K/V for the NAR queries to attend over (owned handles).
const ArKv = struct { k: A, v: A };

fn getW(w: *const Weights, comptime fmt: []const u8, args: anytype) !A {
    var buf: [160]u8 = undefined;
    const key = try std.fmt.bufPrint(&buf, fmt, args);
    return w.get(key) orelse {
        log.err("[yue2] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingWeight;
    };
}

fn linear(w: *const Weights, comptime fmt: []const u8, args: anytype, in_dim: u32) !Linear {
    var buf: [160]u8 = undefined;
    const prefix = try std.fmt.bufPrint(&buf, fmt, args);
    var l = Linear{ .w = try getW(w, "{s}.weight", .{prefix}) };
    if (w.get(try std.fmt.bufPrint(buf[prefix.len..], "{s}.scales", .{prefix}))) |sca| {
        l.scales = sca;
        l.biases = try getW(w, "{s}.biases", .{prefix});
        const qp = transformer_mod.affineParamsFromGeometry(l.w, sca, in_dim) orelse {
            log.err("[yue2] unsolvable quant geometry for {s}\n", .{prefix});
            return error.BadQuantGeometry;
        };
        l.bits = qp.bits;
        l.group_size = qp.group_size;
    }
    l.bias = w.get(try std.fmt.bufPrint(buf[prefix.len..], "{s}.bias", .{prefix}));
    return l;
}

fn resolvePath(w: *const Weights, li: usize, comptime ln: []const u8, comptime attn: []const u8, comptime mlp_ln: []const u8, comptime mlp: []const u8, hidden: u32, q_out: u32, ffn_in: u32) !Path {
    const p = "model.layers.{d}.";
    return .{
        .ln = try getW(w, p ++ ln ++ ".weight", .{li}),
        .q = try linear(w, p ++ attn ++ ".q_proj", .{li}, hidden),
        .k = try linear(w, p ++ attn ++ ".k_proj", .{li}, hidden),
        .v = try linear(w, p ++ attn ++ ".v_proj", .{li}, hidden),
        .o = try linear(w, p ++ attn ++ ".o_proj", .{li}, q_out),
        .qn = try getW(w, p ++ attn ++ ".q_norm.weight", .{li}),
        .kn = try getW(w, p ++ attn ++ ".k_norm.weight", .{li}),
        .mlp_ln = try getW(w, p ++ mlp_ln ++ ".weight", .{li}),
        .gate = try linear(w, p ++ mlp ++ ".gate_proj", .{li}, hidden),
        .up = try linear(w, p ++ mlp ++ ".up_proj", .{li}, hidden),
        .down = try linear(w, p ++ mlp ++ ".down_proj", .{li}, ffn_in),
    };
}

/// Evaluate every array once so no weight stays a lazy file read.
fn evalAll(w: *const Weights) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var it = w.map.valueIterator();
    while (it.next()) |v| _ = mlx.mlx_vector_array_append_value(vec, v.*);
    try mlx.check(mlx.mlx_eval(vec));
}

// ════════════════════════════════════════════════════════════════════════
// Engine
// ════════════════════════════════════════════════════════════════════════

/// Per-layer K/V buffers for one AR sequence, written in place.
const Kv = struct {
    ks: []A,
    vs: []A,
    len: c_int = 0,

    fn init(a: std.mem.Allocator, sc: *Scope, cfg: Cfg, cap: usize) !Kv {
        const ks = try a.alloc(A, cfg.layers);
        errdefer a.free(ks);
        const vs = try a.alloc(A, cfg.layers);
        errdefer a.free(vs);
        const shape = [_]c_int{ 1, cfg.kv_heads, @intCast(cap), cfg.hd };
        for (ks, vs) |*k, *v| {
            k.* = sc.out(try sc.zeros(&shape, .bfloat16));
            v.* = sc.out(try sc.zeros(&shape, .bfloat16));
        }
        return .{ .ks = ks, .vs = vs };
    }

    fn deinit(self: *Kv, a: std.mem.Allocator) void {
        for (self.ks) |k| _ = mlx.mlx_array_free(k);
        for (self.vs) |v| _ = mlx.mlx_array_free(v);
        a.free(self.ks);
        a.free(self.vs);
    }

    /// Write [1,KV,T,hd] at `len`; the old buffer is released so the update donates in place.
    fn write(self: *Kv, sc: *Scope, buf: *A, new: A) !void {
        const sh = mlx.getShape(buf.*);
        const t = mlx.getShape(new)[2];
        const start = [_]c_int{ 0, 0, self.len, 0 };
        const stop = [_]c_int{ sh[0], sh[1], self.len + t, sh[3] };
        const o = sc.out(try sc.sliceUpdate(buf.*, new, &start, &stop));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = o;
    }
};

pub const Request = struct {
    style: []const u8,
    lyrics: []const u8,
    cot: Cot = .full,
    /// An external ABC score replaces the planning phase (cot melody|full).
    abc: ?[]const u8 = null,
    seed: u64 = 831001,
    cfg_scale: ?f32 = null,
    steps: ?u32 = null,
    /// Longest song, in codec frames (25 per second); the model may stop earlier.
    max_frames: u32 = 9000,
};

pub const Song = struct {
    wav: []u8,
    /// The score the song was rendered from (owned), null for cot=off.
    abc: ?[]u8,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: Cfg,
    gen: GenFile,
    vae_parsed: std.json.Parsed(VaeFile),
    w: Weights,
    vw: Weights,
    tok: tok_mod.Tokenizer,
    layers: []Layer,
    embed: A,
    norm: A,
    lm_head: Linear,
    llm2vae: Linear,
    vae2llm: Linear,
    time_fc1: Linear,
    time_fc2: Linear,
    pos_embed: A,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();

        const cfg_bytes = try readFile(io, allocator, model_dir, "config.json");
        defer allocator.free(cfg_bytes);
        self.cfg = try parseConfig(allocator, cfg_bytes);

        self.gen = .{};
        if (readFile(io, allocator, model_dir, "yue2_generation_config.json")) |gb| {
            defer allocator.free(gb);
            const p = std.json.parseFromSlice(GenFile, allocator, gb, .{ .ignore_unknown_fields = true }) catch return error.Yue2ConfigInvalid;
            defer p.deinit();
            self.gen = p.value;
            if (!self.gen.abc.valid() or !self.gen.semantic.valid() or self.gen.ode_steps < 1) return error.Yue2ConfigInvalid;
        } else |_| {}

        const vae_bytes = try readFile(io, allocator, model_dir, "vae_config.json");
        defer allocator.free(vae_bytes);
        self.vae_parsed = std.json.parseFromSlice(VaeFile, allocator, vae_bytes, .{ .ignore_unknown_fields = true }) catch return error.Yue2ConfigInvalid;
        errdefer self.vae_parsed.deinit();
        const vd = self.vae_parsed.value.decoder_config;
        if (vd.latent_dim != LATENT_DIM or vd.strides.len == 0 or self.vae_parsed.value.sample_rate != SAMPLE_RATE or
            hopOf(vd.strides) != SAMPLE_RATE / FRAME_RATE or self.vae_parsed.value.decode_core_frames == 0)
            return error.Yue2ConfigUnsupported;

        self.vw = try model_mod.loadWeightsFile(allocator, model_dir, "vae.safetensors");
        errdefer self.vw.deinit();
        self.w = try model_mod.loadWeightsFile(allocator, model_dir, "model.safetensors");
        errdefer self.w.deinit();
        try evalAll(&self.vw);
        try evalAll(&self.w);

        const tok_path = try std.fmt.allocPrint(allocator, "{s}/qwen.tiktoken", .{model_dir});
        defer allocator.free(tok_path);
        self.tok = try tok_mod.loadTiktoken(io, allocator, tok_path);
        errdefer self.tok.deinit();

        self.layers = try allocator.alloc(Layer, self.cfg.layers);
        errdefer allocator.free(self.layers);
        try self.resolve();
        log.info("[yue2] engine ready ({d} + {d} tensors, {d} layers x2 paths, vae {d} stages)\n", .{ self.w.count(), self.vw.count(), self.cfg.layers, vd.strides.len });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.allocator.free(self.layers);
        self.tok.deinit();
        self.w.deinit();
        self.vw.deinit();
        self.vae_parsed.deinit();
        self.allocator.destroy(self);
    }

    fn resolve(self: *Engine) !void {
        const w = &self.w;
        const c = self.cfg;
        const hidden: u32 = @intCast(c.hidden);
        const q_out: u32 = @intCast(c.heads * c.hd);
        for (self.layers, 0..) |*l, i| {
            l.ar = try resolvePath(w, i, "input_layernorm", "self_attn", "post_attention_layernorm", "mlp", hidden, q_out, c.ffn);
            l.nar = try resolvePath(w, i, "nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp", hidden, q_out, c.ffn);
        }
        self.embed = try getW(w, "model.embed_tokens.weight", .{});
        self.norm = try getW(w, "model.norm.weight", .{});
        self.lm_head = try linear(w, "lm_head", .{}, hidden);
        self.llm2vae = try linear(w, "llm2vae", .{}, hidden);
        self.vae2llm = try linear(w, "vae2llm", .{}, LATENT_DIM);
        self.time_fc1 = try linear(w, "time_embedder.fc1", .{}, TIME_FREQ);
        self.time_fc2 = try linear(w, "time_embedder.fc2", .{}, hidden);
        self.pos_embed = try getW(w, "latent_pos_embed.pe", .{});
    }

    // ── tokenizer ────────────────────────────────────────────────────────

    /// The text ids of a request (owned), checked against the context before any forward.
    pub fn textIds(self: *const Engine, a: std.mem.Allocator, text: []const u8) ![]u32 {
        return self.tok.encode(a, text);
    }

    /// Can a request with this text hold `max_frames` of song (and its score) inside the context?
    pub fn fits(self: *const Engine, a: std.mem.Allocator, req: Request) !bool {
        const text = try promptText(a, req.cot, req.style, req.lyrics);
        defer a.free(text);
        const ids = try self.textIds(a, text);
        defer a.free(ids);
        const abc_room: usize = if (req.cot == .off) 0 else if (req.abc) |s| s.len else ABC_MAX_TOKENS;
        return ids.len + 4 + abc_room + req.max_frames <= CONTEXT;
    }

    // ── AR path ──────────────────────────────────────────────────────────

    fn qkv(self: *const Engine, sc: *Scope, p: *const Path, x: A, offset: c_int) !struct { q: A, k: A, v: A } {
        const c = self.cfg;
        const h = try sc.rms(x, p.ln, c.eps);
        const q = try sc.rms(try sc.heads(try p.q.forward(sc, h), c.heads, c.hd), p.qn, c.eps);
        const k = try sc.rms(try sc.heads(try p.k.forward(sc, h), c.kv_heads, c.hd), p.kn, c.eps);
        return .{
            .q = try sc.ropeAt(q, c.hd, c.rope_theta, offset),
            .k = try sc.ropeAt(k, c.hd, c.rope_theta, offset),
            .v = try sc.heads(try p.v.forward(sc, h), c.kv_heads, c.hd),
        };
    }

    /// x + o(attn), then + the path's SwiGLU MLP.
    fn finish(self: *const Engine, sc: *Scope, p: *const Path, x: A, attn: A) !A {
        const h = try sc.add(x, try p.o.forward(sc, try sc.merge(attn)));
        const m = try sc.rms(h, p.mlp_ln, self.cfg.eps);
        const gu = try sc.mul(try sc.silu(try p.gate.forward(sc, m)), try p.up.forward(sc, m));
        return sc.add(h, try p.down.forward(sc, gu));
    }

    fn scale(self: *const Engine) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.hd)));
    }

    fn embedIds(self: *const Engine, sc: *Scope, ids: []const u32) !A {
        const i32s = try sc.a.alloc(i32, ids.len);
        defer sc.a.free(i32s);
        for (ids, i32s) |id, *o| o.* = @intCast(id);
        const rows = try sc.take(self.embed, try sc.fromI32(i32s, &.{@intCast(ids.len)}));
        return sc.reshape(rows, &.{ 1, @intCast(ids.len), self.cfg.hidden });
    }

    /// Append `ids` to `kv` and return the last position's logits [V] (bf16, lazy).
    fn arForward(self: *Engine, a: std.mem.Allocator, ids: []const u32, kv: *Kv) !A {
        var sc0 = Scope.init(a, self.s);
        defer sc0.deinit();
        var x = sc0.out(try self.embedIds(&sc0, ids));
        defer _ = mlx.mlx_array_free(x);
        const t: c_int = @intCast(ids.len);
        const prefill = t > 1;
        for (self.layers, 0..) |*l, li| {
            var sc = Scope.init(a, self.s);
            defer sc.deinit();
            const r = try self.qkv(&sc, &l.ar, x, kv.len);
            try kv.write(&sc, &kv.ks[li], r.k);
            try kv.write(&sc, &kv.vs[li], r.v);
            const kview = try sc.slice(kv.ks[li], 2, 0, kv.len + t);
            const vview = try sc.slice(kv.vs[li], 2, 0, kv.len + t);
            const attn = try sc.sdpaMode(r.q, kview, vview, self.scale(), if (prefill) "causal" else "");
            const next = sc.out(try self.finish(&sc, &l.ar, x, attn));
            _ = mlx.mlx_array_free(x);
            x = next;
            if (prefill) try mlx.check(mlx.mlx_array_eval(x));
        }
        kv.len += t;
        const last = try sc0.rms(try sc0.slice(x, 1, t - 1, t), self.norm, self.cfg.eps);
        return sc0.out(try sc0.reshape(try self.lm_head.forward(&sc0, last), &.{@intCast(VOCAB)}));
    }

    const Generated = struct { ids: []u32, truncated: bool };

    /// Sample one phase to its end marker or `s.max_tokens`. CFG runs the negative branch on
    /// its own cache: `uncond + g * (cond - uncond)` in the logits' dtype.
    fn generateTokens(self: *Engine, a: std.mem.Allocator, prefix: []const u32, negative: ?[]const u32, cfg_scale: f32, s: Sampling, phase: Phase, seed: u64, top_p_floor: usize, progress: ?sse.Progress) !Generated {
        const stage = @tagName(phase);
        var sc = Scope.init(a, self.s);
        defer sc.deinit();
        var kvs: [2]Kv = undefined;
        const n_kv: usize = if (negative != null) 2 else 1;
        const lens = [2]usize{ prefix.len, if (negative) |n| n.len else 0 };
        var made: usize = 0;
        defer for (kvs[0..made]) |*k| k.deinit(a);
        for (0..n_kv) |i| {
            kvs[i] = try Kv.init(a, &sc, self.cfg, lens[i] + s.max_tokens);
            made += 1;
        }
        var sampler = Sampler.init(a, seed);
        defer sampler.deinit();
        var history: std.ArrayList(u32) = .empty;
        errdefer history.deinit(a);
        const host = try a.alloc(f32, VOCAB);
        defer a.free(host);

        var feed: [2][]const u32 = .{ prefix, negative orelse &.{} };
        var eos = false;
        var step: u32 = 0;
        while (step < s.max_tokens) : (step += 1) {
            if (progress) |p| {
                if (p.boundary()) return error.Cancelled;
                if (step % FRAME_RATE == 0) p.emit(stage, step, 0);
            }
            var step_sc = Scope.init(a, self.s);
            defer step_sc.deinit();
            const cond = try step_sc.keep(try self.arForward(a, feed[0], &kvs[0]));
            var logits = cond;
            if (n_kv == 2) {
                const uncond = try step_sc.keep(try self.arForward(a, feed[1], &kvs[1]));
                logits = try step_sc.add(uncond, try step_sc.mulLike(try step_sc.sub(cond, uncond), cfg_scale));
            }
            const f = try step_sc.astype(logits, .float32);
            try mlx.check(mlx.mlx_array_eval(f));
            @memcpy(host, (mlx.mlx_array_data_float32(f) orelse return error.NoData)[0..VOCAB]);
            const token = try sampler.pick(host, s, history.items, step, phase, top_p_floor);
            if (token == phase.end()) {
                eos = true;
                break;
            }
            try history.append(a, token);
            feed = .{ history.items[history.items.len - 1 ..], history.items[history.items.len - 1 ..] };
        }
        if (progress) |p| p.emit(stage, step, 0);
        return .{ .ids = try history.toOwnedSlice(a), .truncated = !eos };
    }

    // ── NAR path ─────────────────────────────────────────────────────────

    /// The AR path over `ids` once (causal), keeping each layer's K/V for the NAR queries.
    fn arPrefillKv(self: *Engine, a: std.mem.Allocator, ids: []const u32, out: []ArKv) !usize {
        var sc0 = Scope.init(a, self.s);
        defer sc0.deinit();
        var x = sc0.out(try self.embedIds(&sc0, ids));
        defer _ = mlx.mlx_array_free(x);
        var made: usize = 0;
        errdefer for (out[0..made]) |kv| {
            _ = mlx.mlx_array_free(kv.k);
            _ = mlx.mlx_array_free(kv.v);
        };
        for (self.layers, out) |*l, *slot| {
            var sc = Scope.init(a, self.s);
            defer sc.deinit();
            const r = try self.qkv(&sc, &l.ar, x, 0);
            const attn = try sc.sdpaMode(r.q, r.k, r.v, self.scale(), "causal");
            const next = sc.out(try self.finish(&sc, &l.ar, x, attn));
            slot.* = .{ .k = sc.out(r.k), .v = sc.out(r.v) };
            made += 1;
            _ = mlx.mlx_array_free(x);
            x = next;
            try mlx.check(mlx.mlx_array_eval(x));
            try mlx.check(mlx.mlx_array_eval(slot.k));
            try mlx.check(mlx.mlx_array_eval(slot.v));
        }
        return made;
    }

    /// Flow-matching velocity [T,64] for `state` [T,64] at logit-time `raw_t`, as one bidirectional
    /// canvas of LATENT_START + frames + LATENT_END rows after the AR prefix.
    fn velocity(self: *Engine, a: std.mem.Allocator, state: A, raw_t: f32, ar_kv: []const ArKv, ar_len: c_int) !A {
        var sc0 = Scope.init(a, self.s);
        defer sc0.deinit();
        const frames = mlx.getShape(state)[0];
        const n = frames + 2;
        // t is rounded through the state's dtype, as the reference's weak-typed scalar is.
        const t = try sc0.sigmoid(try sc0.astype(try sc0.scalar(raw_t), mlx.mlx_array_dtype(state)));
        const x0 = try self.vae2llm.forward(&sc0, try sc0.reshape(try sc0.padAxis(state, 0, 1, 1), &.{ 1, n, LATENT_DIM }));
        const temb = try self.timeEmbedding(&sc0, t, mlx.mlx_array_dtype(state));
        var x = sc0.out(try sc0.add(try sc0.add(x0, temb), try sc0.slice(self.pos_embed, 0, 0, n)));
        defer _ = mlx.mlx_array_free(x);
        for (self.layers, ar_kv) |*l, kv| {
            var sc = Scope.init(a, self.s);
            defer sc.deinit();
            const r = try self.qkv(&sc, &l.nar, x, ar_len);
            const attn = try sc.sdpaMode(r.q, try sc.concat(&.{ kv.k, r.k }, 2), try sc.concat(&.{ kv.v, r.v }, 2), self.scale(), "");
            const next = sc.out(try self.finish(&sc, &l.nar, x, attn));
            _ = mlx.mlx_array_free(x);
            x = next;
            try mlx.check(mlx.mlx_array_eval(x));
        }
        const v = try self.llm2vae.forward(&sc0, try sc0.rms(x, self.norm, self.cfg.eps));
        return sc0.out(try sc0.reshape(try sc0.slice(v, 1, 1, n - 1), &.{ frames, LATENT_DIM }));
    }

    /// Sinusoidal embedding of the scalar `t` → MLP → [1,1,hidden].
    fn timeEmbedding(self: *Engine, sc: *Scope, t: A, dt: mlx.mlx_dtype) !A {
        var freqs: [TIME_FREQ / 2]f32 = undefined;
        for (&freqs, 0..) |*f, i| f.* = @exp(-@log(@as(f32, 10000)) * @as(f32, @floatFromInt(i)) / (TIME_FREQ / 2));
        const args = try sc.mul(try sc.reshape(try sc.astype(t, .float32), &.{ 1, 1 }), try sc.fromF32(&freqs, &.{ 1, TIME_FREQ / 2 }));
        const emb = try sc.astype(try sc.concat(&.{ try sc.cos(args), try sc.sin(args) }, -1), dt);
        const h = try self.time_fc2.forward(sc, try sc.silu(try self.time_fc1.forward(sc, emb)));
        return sc.reshape(h, &.{ 1, 1, self.cfg.hidden });
    }

    /// Latents [frames,64] f32: midpoint ODE from noise at t=1 to 0, one context chunk at a time.
    fn synthesize(self: *Engine, a: std.mem.Allocator, prefix: []const u32, codec: []const u32, seed: u64, steps: u32, progress: ?sse.Progress) !A {
        var sc = Scope.init(a, self.s);
        defer sc.deinit();
        const key = try sc.keep(blk: {
            var k = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_key(&k, seed));
            break :blk k;
        });
        const frames: c_int = @intCast(codec.len);
        var noise = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&noise, &[_]c_int{ frames, LATENT_DIM }, 2, .float32, 0, 1, key, self.s));
        _ = try sc.keep(noise);

        const size = chunkSize(prefix.len);
        if (codec.len == 0 or size == 0) return error.NoAcousticContext;
        const n_chunks = (codec.len + size - 1) / size;
        var pieces = try a.alloc(A, n_chunks);
        defer a.free(pieces);
        var made: usize = 0;
        defer for (pieces[0..made]) |p| {
            _ = mlx.mlx_array_free(p);
        };

        const dt = 1.0 / @as(f32, @floatFromInt(steps));
        var start: usize = 0;
        while (start < codec.len) : (start += size) {
            const stop = @min(start + size, codec.len);
            const ar_ids = try a.alloc(u32, prefix.len + (stop - start) + 1);
            defer a.free(ar_ids);
            @memcpy(ar_ids[0..prefix.len], prefix);
            for (codec[start..stop], ar_ids[prefix.len..][0 .. stop - start]) |c, *o| o.* = c + CODEC_OFFSET;
            ar_ids[ar_ids.len - 1] = MUSIC_END;

            const ar_kv = try a.alloc(ArKv, self.cfg.layers);
            defer a.free(ar_kv);
            const built = try self.arPrefillKv(a, ar_ids, ar_kv);
            defer for (ar_kv[0..built]) |kv| {
                _ = mlx.mlx_array_free(kv.k);
                _ = mlx.mlx_array_free(kv.v);
            };
            _ = mlx.mlx_clear_cache();

            var state = sc.out(try sc.astype(try sc.slice(noise, 0, @intCast(start), @intCast(stop)), .bfloat16));
            defer _ = mlx.mlx_array_free(state);
            const ar_len: c_int = @intCast(ar_ids.len);
            for (0..steps) |si| {
                if (progress) |p| {
                    if (p.boundary()) return error.Cancelled;
                    p.emit("nar", @intCast((start / size) * steps + si), @intCast(n_chunks * steps));
                }
                var st = Scope.init(a, self.s);
                defer st.deinit();
                const t = flowTime(@intCast(si), steps);
                const v1 = try st.keep(try self.velocity(a, state, logitTime(t), ar_kv[0..built], ar_len));
                const mid = try st.sub(state, try st.mulLike(v1, dt / 2));
                const v2 = try st.keep(try self.velocity(a, mid, logitTime(t - dt / 2), ar_kv[0..built], ar_len));
                const next = st.out(try st.sub(state, try st.mulLike(v2, dt)));
                try mlx.check(mlx.mlx_array_eval(next));
                _ = mlx.mlx_array_free(state);
                state = next;
            }
            pieces[made] = sc.out(try sc.astype(state, .float32));
            made += 1;
        }
        return sc.out(try sc.concat(pieces[0..made], 0));
    }

    // ── VAE ──────────────────────────────────────────────────────────────

    fn vaeW(self: *const Engine, comptime fmt: []const u8, args: anytype) !A {
        return getW(&self.vw, fmt, args);
    }

    /// x + (1/exp(beta)) sin^2(exp(alpha) x); both parameters are stored as logs. The temporaries
    /// live in their own scope: at a 1056-frame tile each is half a gigabyte.
    fn snake(self: *const Engine, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype) !A {
        var t = Scope.init(sc.a, sc.s);
        defer t.deinit();
        const alpha = try t.exp(try self.vaeW(fmt ++ ".alpha", args));
        const beta = try t.exp(try self.vaeW(fmt ++ ".beta", args));
        const sn = try t.sin(try t.mul(x, alpha));
        return sc.keep(t.out(try t.add(x, try t.div(try t.mul(sn, sn), try t.addS(beta, 1e-9)))));
    }

    fn conv(self: *const Engine, sc: *Scope, x: A, comptime fmt: []const u8, args: anytype, padding: c_int, dilation: c_int) !A {
        const y = try sc.conv1d(x, try self.vaeW(fmt ++ ".weight", args), 1, padding, dilation);
        var buf: [160]u8 = undefined;
        const bias = self.vw.get(try std.fmt.bufPrint(&buf, fmt ++ ".bias", args)) orelse return y;
        return sc.add(y, bias);
    }

    fn resUnit(self: *const Engine, sc: *Scope, x: A, block: usize, unit: usize, dilation: c_int) !A {
        var t = Scope.init(sc.a, sc.s);
        defer t.deinit();
        const a1 = try self.snake(&t, x, "layers.{d}.layers.{d}.layers.0", .{ block, unit });
        const c1 = try self.conv(&t, a1, "layers.{d}.layers.{d}.layers.1", .{ block, unit }, 3 * dilation, dilation);
        const a2 = try self.snake(&t, c1, "layers.{d}.layers.{d}.layers.2", .{ block, unit });
        return sc.keep(t.out(try t.add(x, try self.conv(&t, a2, "layers.{d}.layers.{d}.layers.3", .{ block, unit }, 0, 1))));
    }

    /// Latents [1,L,64] f32 → audio [1,decodedLen(L),2] f32. One scope and one eval per
    /// stage keep the transient at a single stage's activations.
    fn decodeWindow(self: *const Engine, a: std.mem.Allocator, z: A) !A {
        const d = self.vae_parsed.value.decoder_config;
        var sc0 = Scope.init(a, self.s);
        defer sc0.deinit();
        var h = sc0.out(try self.conv(&sc0, z, "layers.0", .{}, 3, 1));
        defer _ = mlx.mlx_array_free(h);
        const n = d.strides.len;
        for (0..n) |bi| {
            var sc = Scope.init(a, self.s);
            defer sc.deinit();
            const stride = d.strides[n - 1 - bi];
            const li = bi + 1;
            var y = try self.snake(&sc, h, "layers.{d}.layers.0", .{li});
            const up = try sc.convT1d(y, try self.vaeW("layers.{d}.layers.1.weight", .{li}), stride, @divTrunc(stride + 1, 2));
            y = try sc.add(up, try self.vaeW("layers.{d}.layers.1.bias", .{li}));
            y = try self.resUnit(&sc, y, li, 2, 1);
            y = try self.resUnit(&sc, y, li, 3, 3);
            y = try self.resUnit(&sc, y, li, 4, 9);
            const next = sc.out(y);
            try mlx.check(mlx.mlx_array_eval(next));
            _ = mlx.mlx_array_free(h);
            h = next;
        }
        const a1 = try self.snake(&sc0, h, "layers.{d}", .{n + 1});
        return sc0.out(try self.conv(&sc0, a1, "layers.{d}", .{n + 2}, 3, 1));
    }

    /// Latents [frames,64] → interleaved stereo f32 (owned): cores decoded with a halo of context
    /// on each side, then cropped exactly, so tiles join without a crossfade.
    fn decodeLatents(self: *const Engine, a: std.mem.Allocator, latents: A, progress: ?sse.Progress) ![]f32 {
        const d = self.vae_parsed.value.decoder_config;
        const vf = self.vae_parsed.value;
        const frames: usize = @intCast(mlx.getShape(latents)[0]);
        const hop = hopOf(d.strides);
        const total = decodedLen(d.strides, frames);
        const out = try a.alloc(f32, total * 2);
        errdefer a.free(out);
        const core: usize = vf.decode_core_frames;
        const n_tiles = (frames + core - 1) / core;
        var tile: usize = 0;
        var start: usize = 0;
        while (start < frames) : (start += core) {
            if (progress) |p| {
                if (p.boundary()) return error.Cancelled;
                p.emit("decode", @intCast(tile), @intCast(n_tiles));
            }
            tile += 1;
            const end = @min(frames, start + core);
            const left = start -| vf.decode_halo_frames;
            const right = @min(frames, end + vf.decode_halo_frames);
            var sc = Scope.init(a, self.s);
            defer sc.deinit();
            const z = try sc.reshape(try sc.slice(latents, 0, @intCast(left), @intCast(right)), &.{ 1, @intCast(right - left), LATENT_DIM });
            const audio = try self.decodeWindow(a, z);
            defer _ = mlx.mlx_array_free(audio);
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_contiguous(&flat, audio, false, self.s));
            try mlx.check(mlx.mlx_array_eval(flat));
            const data = mlx.mlx_array_data_float32(flat) orelse return error.NoData;
            const crop = (start - left) * hop;
            const take = @min(end * hop, total) - start * hop;
            @memcpy(out[start * hop * 2 ..][0 .. take * 2], data[crop * 2 ..][0 .. take * 2]);
        }
        if (progress) |p| p.emit("decode", @intCast(n_tiles), @intCast(n_tiles));
        return out;
    }

    // ── song ─────────────────────────────────────────────────────────────

    pub fn generate(self: *Engine, allocator: std.mem.Allocator, req: Request, progress: ?sse.Progress) !Song {
        const steps = req.steps orelse self.gen.ode_steps;
        const text = try promptText(allocator, req.cot, req.style, req.lyrics);
        defer allocator.free(text);
        const text_ids = try self.textIds(allocator, text);
        defer allocator.free(text_ids);
        // The reference's cot=off arithmetic keeps the top 3 through top-p, every other path the top 1.
        const floor: usize = if (req.cot == .off) 3 else 1;

        var abc_ids: []u32 = &.{};
        var abc_text: ?[]u8 = null;
        errdefer if (abc_text) |t| allocator.free(t);
        defer allocator.free(abc_ids);
        if (req.cot != .off) {
            if (req.abc) |score| {
                abc_ids = try self.tok.encode(allocator, score);
                abc_text = try allocator.dupe(u8, score);
            } else {
                const open = try assemblePrefix(allocator, text_ids, req.cot, null);
                defer allocator.free(open);
                if (open.len + ABC_MAX_TOKENS > CONTEXT) return error.PromptTooLong;
                log.info("[yue2] planning the ABC score ({s}, prefix {d})\n", .{ @tagName(req.cot), open.len });
                const g = try self.generateTokens(allocator, open, null, 1.0, self.gen.abc, .abc, req.seed, floor, progress);
                abc_ids = g.ids;
                abc_text = try self.tok.decode(allocator, abc_ids, false);
                if (g.truncated) log.info("[yue2] ABC hit max_tokens\n", .{});
            }
        }

        const prefix = try assemblePrefix(allocator, text_ids, req.cot, abc_ids);
        defer allocator.free(prefix);
        var samp = self.gen.semantic;
        samp.max_tokens = @min(samp.max_tokens, req.max_frames);
        samp.min_tokens = @min(samp.min_tokens, samp.max_tokens);
        if (prefix.len + samp.max_tokens > CONTEXT) return error.PromptTooLong;
        const guidance = req.cfg_scale orelse req.cot.guidance();
        var negative: ?[]u32 = null;
        defer if (negative) |n| allocator.free(n);
        if (guidance != 1) {
            const instruction_ids = try self.textIds(allocator, req.cot.instruction());
            defer allocator.free(instruction_ids);
            negative = try assembleNegative(allocator, instruction_ids, req.cot, abc_ids);
        }

        log.info("[yue2] semantic: prefix {d} tokens, cfg {d:.2}, up to {d} frames\n", .{ prefix.len, guidance, samp.max_tokens });
        const sem = try self.generateTokens(allocator, prefix, negative, guidance, samp, .semantic, req.seed, floor, progress);
        defer allocator.free(sem.ids);
        if (sem.ids.len == 0) return error.NoAudioFrames;
        if (sem.truncated) log.info("[yue2] semantic hit max_tokens\n", .{});
        const codec = try allocator.alloc(u32, sem.ids.len);
        defer allocator.free(codec);
        for (sem.ids, codec) |id, *c| c.* = id - CODEC_OFFSET;
        _ = mlx.mlx_clear_cache();

        log.info("[yue2] nar: {d} frames ({d:.1}s), {d} midpoint steps\n", .{ codec.len, @as(f32, @floatFromInt(codec.len)) / FRAME_RATE, steps });
        const latents = try self.synthesize(allocator, prefix, codec, req.seed, steps, progress);
        defer _ = mlx.mlx_array_free(latents);
        _ = mlx.mlx_clear_cache();

        const pcm = try self.decodeLatents(allocator, latents, progress);
        defer allocator.free(pcm);
        const wav = try wav_mod.encodePcm16(allocator, pcm, SAMPLE_RATE, 2);
        return .{ .wav = wav, .abc = abc_text };
    }
};

const TIME_FREQ = 256;

fn readFile(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    return rs.interface.allocRemaining(a, .limited(16 * 1024 * 1024));
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first, then env-gated oracles (YUE2_TEST_MODEL + YUE2_*,
// fed by tests/dump_yue2_fixtures.py).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "yue2 prefix: the ABC section is open, closed, or skipped per cot" {
    const a = testing.allocator;
    const text = [_]u32{ 11, 12 };
    const abc = [_]u32{ 21, 22, 23 };
    const open = try assemblePrefix(a, &text, .full, null);
    defer a.free(open);
    try testing.expectEqualSlices(u32, &.{ EOD, 11, 12, ABC_START }, open);
    const closed = try assemblePrefix(a, &text, .melody, &abc);
    defer a.free(closed);
    try testing.expectEqualSlices(u32, &.{ EOD, 11, 12, ABC_START, 21, 22, 23, ABC_END, MUSIC_START }, closed);
    // cot=off never plans: an empty section, whatever ids ride along.
    const off = try assemblePrefix(a, &text, .off, &abc);
    defer a.free(off);
    try testing.expectEqualSlices(u32, &.{ EOD, 11, 12, ABC_START, ABC_END, MUSIC_START }, off);
}

test "yue2 negative prefix: the instruction alone, with the same score" {
    const a = testing.allocator;
    const ins = [_]u32{ 5, 6 };
    const abc = [_]u32{ 21, 22 };
    const full = try assembleNegative(a, &ins, .full, &abc);
    defer a.free(full);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, 6, ABC_START, 21, 22, ABC_END, MUSIC_START }, full);
    const off = try assembleNegative(a, &ins, .off, &.{});
    defer a.free(off);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, 6, MUSIC_START }, off);
}

test "yue2 request text names the instruction, tags and lyrics" {
    const a = testing.allocator;
    const t = try promptText(a, .off, "pop", "la");
    defer a.free(t);
    try testing.expectEqualStrings("Generate music with codec tokens from the given conditions.\n[Tags]\npop\n[Lyrics]\nla\n", t);
    try testing.expectEqual(@as(?Cot, .melody), Cot.parse("melody"));
    try testing.expectEqual(@as(?Cot, null), Cot.parse("loud"));
}

test "yue2 decoder length: odd strides lose a sample per stage, 1920 per frame otherwise" {
    const strides = [_]c_int{ 2, 2, 4, 4, 5, 6 };
    try testing.expectEqual(@as(usize, 1920 * 1 - 64), decodedLen(&strides, 1));
    try testing.expectEqual(@as(usize, 1920 * 125 - 64), decodedLen(&strides, 125));
    try testing.expectEqual(@as(usize, 1920), hopOf(&strides));
    try testing.expectEqual(@as(usize, 2 * 7), decodedLen(&.{2}, 7));
}

test "yue2 NAR chunks: half the room left after the prefix, never past the context" {
    try testing.expectEqual(@as(usize, (24576 - 59 - 3) / 2), chunkSize(59));
    try testing.expectEqual(@as(usize, 0), chunkSize(24576));
    try testing.expectEqual(@as(usize, 0), chunkSize(100000));
}

test "yue2 logit time: the model's sigmoid input, clamped" {
    try testing.expectApproxEqAbs(@as(f32, 0), logitTime(0.5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0986123), logitTime(0.75), 1e-6);
    try testing.expectEqual(@as(f32, 20), logitTime(1.0));
    try testing.expectEqual(@as(f32, -20), logitTime(0));
    try testing.expectEqual(@as(f32, 1.0), flowTime(0, 32));
    try testing.expectEqual(@as(f32, 0.5), flowTime(16, 32));
}

test "yue2 config: the shipped shape parses, other shapes are named errors" {
    const a = testing.allocator;
    const ok =
        \\{"hidden_size":2048,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
        \\ "intermediate_size":6144,"vocab_size":184704,"rms_norm_eps":1e-6,"rope_theta":1000000,"latent_dim":64,
        \\ "max_latent_frames":24576,"timestep_shift":1.0,"quantization":{"bits":8}}
    ;
    const c = try parseConfig(a, ok);
    try testing.expectEqual(@as(u32, 28), c.layers);
    try testing.expectEqual(@as(u32, 6144), c.ffn);
    const bad_vocab = try std.mem.replaceOwned(u8, a, ok, "184704", "151936");
    defer a.free(bad_vocab);
    try testing.expectError(error.Yue2ConfigUnsupported, parseConfig(a, bad_vocab));
    const shifted = try std.mem.replaceOwned(u8, a, ok, "\"timestep_shift\":1.0", "\"timestep_shift\":3.0");
    defer a.free(shifted);
    try testing.expectError(error.Yue2ConfigUnsupported, parseConfig(a, shifted));
    const ragged = try std.mem.replaceOwned(u8, a, ok, "\"num_key_value_heads\":8", "\"num_key_value_heads\":5");
    defer a.free(ragged);
    try testing.expectError(error.Yue2ConfigInvalid, parseConfig(a, ragged));
    try testing.expectError(error.Yue2ConfigInvalid, parseConfig(a, "{}"));
}

fn flatScores(a: std.mem.Allocator, fill: f32) ![]f32 {
    const s = try a.alloc(f32, VOCAB);
    @memset(s, fill);
    return s;
}

test "yue2 sampler: temperature 0 is the argmax of the phase's ids, never an id outside it" {
    const a = testing.allocator;
    const scores = try flatScores(a, -1);
    defer a.free(scores);
    scores[EOD + 5] = 100; // outside the ABC phase's text ids
    scores[777] = 3;
    var smp = Sampler.init(a, 1);
    defer smp.deinit();
    const s = Sampling{ .temperature = 0, .repetition_penalty = 1, .min_tokens = 0 };
    try testing.expectEqual(@as(u32, 777), try smp.pick(scores, s, &.{}, 0, .abc, 1));
}

test "yue2 sampler: the end marker waits for min_tokens, then can win" {
    const a = testing.allocator;
    var smp = Sampler.init(a, 1);
    defer smp.deinit();
    const s = Sampling{ .temperature = 0, .repetition_penalty = 1, .min_tokens = 10 };
    const early = try flatScores(a, -1);
    defer a.free(early);
    early[MUSIC_END] = 50;
    early[CODEC_OFFSET + 9] = 1;
    try testing.expectEqual(@as(u32, CODEC_OFFSET + 9), try smp.pick(early, s, &.{}, 9, .semantic, 1));
    const late = try flatScores(a, -1);
    defer a.free(late);
    late[MUSIC_END] = 50;
    late[CODEC_OFFSET + 9] = 1;
    try testing.expectEqual(MUSIC_END, try smp.pick(late, s, &.{}, 10, .semantic, 1));
}

test "yue2 sampler: the window penalty moves a repeated id by penalty^count" {
    const a = testing.allocator;
    var smp = Sampler.init(a, 1);
    defer smp.deinit();
    const s = Sampling{ .temperature = 0, .repetition_penalty = 2, .penalty_window = 4, .min_tokens = 0 };
    const hist = [_]u32{ CODEC_OFFSET + 1, CODEC_OFFSET + 1 };
    const scores = try flatScores(a, -9);
    defer a.free(scores);
    scores[CODEC_OFFSET + 1] = 3.9; // /4 -> 0.975
    scores[CODEC_OFFSET + 2] = 1;
    try testing.expectEqual(@as(u32, CODEC_OFFSET + 2), try smp.pick(scores, s, &hist, 0, .semantic, 1));
    try testing.expectApproxEqAbs(@as(f32, 0.975), scores[CODEC_OFFSET + 1], 1e-6);
    // Outside the window the history no longer counts.
    const old = [_]u32{ CODEC_OFFSET + 1, 9, 9, 9, 9 };
    scores[CODEC_OFFSET + 1] = 3.9;
    try testing.expectEqual(@as(u32, CODEC_OFFSET + 1), try smp.pick(scores, s, &old, 0, .semantic, 1));
}

test "yue2 sampler: top-k and the nucleus bound the draw, a seed repeats it" {
    const a = testing.allocator;
    const s = Sampling{ .temperature = 1, .top_p = 0.5, .top_k = 3, .repetition_penalty = 1, .min_tokens = 0 };
    var seen = [_]u32{ 0, 0, 0, 0 };
    var prev: ?u32 = null;
    for (0..200) |seed| {
        const scores = try flatScores(a, -20);
        defer a.free(scores);
        // Probabilities ~ 0.64 / 0.24 / 0.09 / ...: top_p 0.5 keeps only the first.
        scores[100] = 2;
        scores[101] = 1;
        scores[102] = 0;
        scores[103] = -1;
        var smp = Sampler.init(a, seed);
        defer smp.deinit();
        const t = try smp.pick(scores, s, &.{}, 0, .abc, 1);
        seen[@min(t - 100, 3)] += 1;
        // The same seed draws the same id.
        const again = try flatScores(a, -20);
        defer a.free(again);
        again[100] = 2;
        again[101] = 1;
        again[102] = 0;
        again[103] = -1;
        var smp2 = Sampler.init(a, seed);
        defer smp2.deinit();
        try testing.expectEqual(t, try smp2.pick(again, s, &.{}, 0, .abc, 1));
        prev = t;
    }
    try testing.expectEqual(@as(u32, 200), seen[0]);
    // Without the nucleus the top-3 all show up, and the fourth never does.
    var wide = [_]u32{ 0, 0, 0, 0 };
    for (0..400) |seed| {
        const scores = try flatScores(a, -20);
        defer a.free(scores);
        scores[100] = 2;
        scores[101] = 1;
        scores[102] = 0;
        scores[103] = -1;
        var smp = Sampler.init(a, seed);
        defer smp.deinit();
        const t = try smp.pick(scores, .{ .temperature = 1, .top_p = 1, .top_k = 3, .repetition_penalty = 1, .min_tokens = 0 }, &.{}, 0, .abc, 1);
        wide[@min(t - 100, 3)] += 1;
    }
    try testing.expect(wide[0] > 0 and wide[1] > 0 and wide[2] > 0);
    try testing.expectEqual(@as(u32, 0), wide[3]);
}

test "yue2 sampler: top-k keeps every id tied with the k-th score" {
    const a = testing.allocator;
    const scores = try flatScores(a, -20);
    defer a.free(scores);
    scores[10] = 1;
    scores[11] = 1;
    scores[12] = 1;
    var smp = Sampler.init(a, 3);
    defer smp.deinit();
    _ = try smp.pick(scores, .{ .temperature = 1, .top_p = 1, .top_k = 1, .repetition_penalty = 1, .min_tokens = 0 }, &.{}, 0, .abc, 1);
    try testing.expectEqual(@as(usize, 3), smp.keep.items.len);
}

test "yue2 sampling bounds: a nonsense knob is not valid" {
    try testing.expect((Sampling{}).valid());
    try testing.expect(ABC_SAMPLING.valid());
    try testing.expect(!(Sampling{ .top_p = 0 }).valid());
    try testing.expect(!(Sampling{ .temperature = -1 }).valid());
    try testing.expect(!(Sampling{ .penalty_window = 0 }).valid());
    try testing.expect(!(Sampling{ .min_tokens = 10, .max_tokens = 5 }).valid());
    try testing.expect(!(Sampling{ .top_k = 0 }).valid());
}

fn envPath(env: [*:0]const u8) ![]const u8 {
    return std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
}

fn readRaw(comptime T: type, a: std.mem.Allocator, env: [*:0]const u8) ![]T {
    const path = try envPath(env);
    const io = std.Io.Threaded.global_single_threaded.io();
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1 << 30));
    defer a.free(bytes);
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

fn testEngine() !*Engine {
    return Engine.load(std.Io.Threaded.global_single_threaded.io(), testing.allocator, std.mem.span(std.c.getenv("YUE2_TEST_MODEL") orelse return error.SkipZigTest));
}

fn toIds(a: std.mem.Allocator, raw: []const i32) ![]u32 {
    const out = try a.alloc(u32, raw.len);
    for (raw, out) |r, *o| o.* = @intCast(r);
    return out;
}

/// A logit row matters through its softmax, which a constant shift (bf16 rounding moves the whole
/// background by an ulp) does not change: same argmax and a small KL from the reference.
fn expectLogits(arr: A, ref: []const f32, label: []const u8) !void {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var f32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32a);
    const s = mlx.mlx_default_gpu_stream_new();
    try mlx.check(mlx.mlx_astype(&f32a, arr, .float32, s));
    try mlx.check(mlx.mlx_contiguous(&flat, f32a, false, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    try testing.expectEqual(ref.len, mlx.mlx_array_size(flat));
    const d = mlx.mlx_array_data_float32(flat) orelse return error.NoData;
    var lse_r: f64 = 0;
    var lse_m: f64 = 0;
    var top_r: usize = 0;
    var top_m: usize = 0;
    for (ref, 0..) |r, i| {
        try testing.expect(std.math.isFinite(d[i]));
        lse_r += @exp(@as(f64, r));
        lse_m += @exp(@as(f64, d[i]));
        if (r > ref[top_r]) top_r = i;
        if (d[i] > d[top_m]) top_m = i;
    }
    lse_r = @log(lse_r);
    lse_m = @log(lse_m);
    var kl: f64 = 0;
    for (ref, 0..) |r, i| {
        const lp = @as(f64, r) - lse_r;
        kl += @exp(lp) * (lp - (@as(f64, d[i]) - lse_m));
    }
    std.debug.print("[yue2-test] {s}: KL {e:.2} argmax ref {d} mine {d}\n", .{ label, kl, top_r, top_m });
    try testing.expectEqual(top_r, top_m);
    try testing.expect(kl < 2e-3);
}

/// Cosine AND scale: a concatenated tensor can match in direction while off in norm.
fn expectParity(arr: A, ref: []const f32, min_cos: f64, label: []const u8) !void {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    var f32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32a);
    const s = mlx.mlx_default_gpu_stream_new();
    try mlx.check(mlx.mlx_astype(&f32a, arr, .float32, s));
    try mlx.check(mlx.mlx_contiguous(&flat, f32a, false, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    try testing.expectEqual(ref.len, mlx.mlx_array_size(flat));
    const d = mlx.mlx_array_data_float32(flat) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (ref, 0..) |r, i| {
        try testing.expect(std.math.isFinite(d[i]));
        dot += @as(f64, d[i]) * r;
        na += @as(f64, d[i]) * d[i];
        nb += @as(f64, r) * r;
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    const rms = @sqrt(na / nb);
    std.debug.print("[yue2-test] {s}: cos {d:.6} rms_ratio {d:.4}\n", .{ label, cos, rms });
    try testing.expect(cos >= min_cos);
    try testing.expect(rms > 0.97 and rms < 1.03);
}

test "yue2 tokenizer: the tiktoken ranks encode like the reference" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const io = std.Io.Threaded.global_single_threaded.io();
    const text = try readFile(io, a, std.fs.path.dirname(try envPath("YUE2_TOK_TEXT")).?, std.fs.path.basename(try envPath("YUE2_TOK_TEXT")));
    defer a.free(text);
    const want = try readRaw(i32, a, "YUE2_TOK_IDS");
    defer a.free(want);
    const got = try e.textIds(a, text);
    defer a.free(got);
    const want_ids = try toIds(a, want);
    defer a.free(want_ids);
    try testing.expectEqualSlices(u32, want_ids, got);
}

test "yue2 AR path: prefill and a cached decode step match the reference logits" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const raw = try readRaw(i32, a, "YUE2_AR_PREFIX");
    defer a.free(raw);
    const prefix = try toIds(a, raw);
    defer a.free(prefix);
    const nxt_raw = try readRaw(i32, a, "YUE2_AR_NEXT");
    defer a.free(nxt_raw);
    const ref0 = try readRaw(f32, a, "YUE2_AR_LOGITS0");
    defer a.free(ref0);
    const ref1 = try readRaw(f32, a, "YUE2_AR_LOGITS1");
    defer a.free(ref1);

    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    var kv = try Kv.init(a, &sc, e.cfg, prefix.len + 4);
    defer kv.deinit(a);
    const l0 = try sc.keep(try e.arForward(a, prefix, &kv));
    try expectLogits(l0, ref0, "ar prefill logits");
    const l1 = try sc.keep(try e.arForward(a, &.{@intCast(nxt_raw[0])}, &kv));
    try expectLogits(l1, ref1, "ar cached decode logits");
}

test "yue2 NAR path: velocity and the seeded 4-step solve match the reference" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const ar_raw = try readRaw(i32, a, "YUE2_NAR_AR");
    defer a.free(ar_raw);
    const ar_ids = try toIds(a, ar_raw);
    defer a.free(ar_ids);
    const state_f = try readRaw(f32, a, "YUE2_NAR_STATE");
    defer a.free(state_f);
    const ref_v = try readRaw(f32, a, "YUE2_NAR_V");
    defer a.free(ref_v);

    const ar_kv = try a.alloc(ArKv, e.cfg.layers);
    defer a.free(ar_kv);
    const built = try e.arPrefillKv(a, ar_ids, ar_kv);
    defer for (ar_kv[0..built]) |kv| {
        _ = mlx.mlx_array_free(kv.k);
        _ = mlx.mlx_array_free(kv.v);
    };
    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    const state = try sc.astype(try sc.fromF32(state_f, &.{ 24, LATENT_DIM }), .bfloat16);
    const v = try sc.keep(try e.velocity(a, state, 1.0986123, ar_kv[0..built], @intCast(ar_ids.len)));
    try expectParity(v, ref_v, 0.998, "nar velocity");

    const plen = try readRaw(i32, a, "YUE2_NAR_PREFIX_LEN");
    defer a.free(plen);
    const codec_raw = try readRaw(i32, a, "YUE2_NAR_CODEC");
    defer a.free(codec_raw);
    const codec = try toIds(a, codec_raw);
    defer a.free(codec);
    const ref_lat = try readRaw(f32, a, "YUE2_NAR_LAT");
    defer a.free(ref_lat);
    const lat = try sc.keep(try e.synthesize(a, ar_ids[0..@intCast(plen[0])], codec, 7, 4, null));
    try expectParity(lat, ref_lat, 0.995, "nar 4-step latents");
}

test "yue2 VAE: two tiles of random latents decode like the reference" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const lat_f = try readRaw(f32, a, "YUE2_VAE_LAT");
    defer a.free(lat_f);
    const ref = try readRaw(f32, a, "YUE2_VAE_AUDIO");
    defer a.free(ref);
    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    const z = try sc.fromF32(lat_f, &.{ @intCast(lat_f.len / @as(usize, LATENT_DIM)), LATENT_DIM });
    const pcm = try e.decodeLatents(a, z, null);
    defer a.free(pcm);
    try testing.expectEqual(ref.len, pcm.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (ref, pcm) |r, p| {
        const rc: f64 = std.math.clamp(r, -1, 1);
        const pc: f64 = std.math.clamp(p, -1, 1);
        dot += rc * pc;
        na += pc * pc;
        nb += rc * rc;
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    std.debug.print("[yue2-test] vae tiles: cos {d:.6} rms_ratio {d:.4}\n", .{ cos, @sqrt(na / nb) });
    // This build's NAX f32 convs round to TF32: a uniform 0.4% gain, error 40 dB under the signal.
    // `MLX_ENABLE_TF32=0` makes the same decode match to cos 1.000000.
    try testing.expect(cos > 0.9999);
    try testing.expectApproxEqRel(@as(f64, 1), @sqrt(na / nb), 0.01);
}
