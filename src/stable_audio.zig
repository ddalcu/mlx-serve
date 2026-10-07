//! Stable Audio 3 (small) — text-to-audio (`.audio` modality, `stable_audio3`).
//!
//! Loads the OFFICIAL `stabilityai/stable-audio-3-small-*` repo as published:
//! `model.safetensors` (PyTorch names, f32 DiT + SAME-S codec + conditioner),
//! `model_config.json`, and T5Gemma in `t5gemma-b-b-ul2/` (bf16, HF layout).
//! The math follows Stability's own MLX port (stable-audio-3
//! `optimized/mlx/models/defs`, MIT):
//!
//!   prompt → T5Gemma encoder (Gemma-2 encoder, bidirectional, softcap 50)
//!   → 256 rows (pad rows = the learned padding embedding) + a seconds row
//!   → 20-layer DiT (64 memory tokens, AdaLN, cross-attn, local-add cond)
//!   → ping-pong rectified-flow sampler (LogSNR-shifted schedule, no CFG)
//!   → SAME-S decoder (6 differential-attention blocks over 34-row chunks,
//!     the second half midpoint-shifted) → unpatch 256 → 44.1 kHz stereo.
//!
//! Every activation is f32: the decoder's differential attention cancels
//! catastrophically in f16. Weights stay in their stored dtype.
//! Parity: env-gated `SA3_*` oracles fed by tests/dump_stable_audio_fixtures.py.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const wav_mod = @import("wav.zig");
const sse = @import("gen_sse.zig");

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const Weights = model_mod.Weights;

pub const SAMPLE_RATE: u32 = 44100;
pub const SAMPLES_PER_LATENT: u32 = 4096;
pub const DEFAULT_STEPS: u32 = 8;
pub const MAX_STEPS: u32 = 50;
/// The small models' longest generation (the paper's limit for Small).
pub const MAX_SECONDS: f32 = 120;
/// Silence sampled past the request, then trimmed (the reference `generate()`'s
/// `duration_padding_sec`): the model trained with silence after every clip,
/// never on one that ends at the sequence edge.
const HEADROOM_SECONDS: f64 = 6;
/// `model_config.json` `sample_size`: the longest sequence the checkpoint samples.
const SAMPLE_SIZE: u32 = 5_292_032;

const LATENT: c_int = 256;
const COND: c_int = 768;
const MAX_TOKENS: usize = 256;

// T5Gemma t5gemma-b-b-ul2 encoder.
const T5_LAYERS = 12;
const T5_HEADS: c_int = 12;
const T5_HD: c_int = 64;
const T5_EPS: f32 = 1e-6;
const T5_SOFTCAP: f32 = 50;

// SAME-S decoder.
const DEC_DIM: c_int = 768;
const DEC_HEADS: c_int = 12;
const DEC_BLOCKS = 6;
const DEC_STRIDE: c_int = 16; // patches per latent
const DEC_SUB: c_int = DEC_STRIDE + 1; // the latent row + its 16 new tokens
const DEC_CHUNK: c_int = 2 * DEC_SUB; // 34-row attention chunks
const DEC_PATCH: c_int = 256; // samples per channel per patch
const ROPE_DIMS: c_int = 32;

pub const Cfg = struct {
    embed: c_int = 1024,
    depth: u32 = 20,
    heads: c_int = 16,
    memory: c_int = 64,
};

/// `model_config.json` (stable-audio-tools layout). Medium's DiT uses
/// differential attention and the SAME-L codec — refused by name.
pub fn parseConfig(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Cfg {
    const path = try std.fmt.allocPrint(a, "{s}/model_config.json", .{model_dir});
    defer a.free(path);
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [8192]u8 = undefined;
    var rs = file.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
    defer a.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    return cfgFromJson(parsed.value);
}

fn cfgFromJson(root: std.json.Value) !Cfg {
    const dc = jsonPath(root, &.{ "model", "diffusion", "config" }) orelse return error.StableAudioConfigInvalid;
    if (jsonPath(dc, &.{ "attn_kwargs", "differential" })) |d| if (d == .bool and d.bool) return error.StableAudioMediumUnsupported;
    var cfg = Cfg{};
    cfg.embed = jsonInt(dc, "embed_dim") orelse cfg.embed;
    cfg.depth = @intCast(jsonInt(dc, "depth") orelse @as(c_int, @intCast(cfg.depth)));
    cfg.heads = jsonInt(dc, "num_heads") orelse cfg.heads;
    cfg.memory = jsonInt(dc, "num_memory_tokens") orelse cfg.memory;
    // Head width must be whole and hold the 32 rotary dims (the DiT's @divExact).
    if (@rem(cfg.embed, cfg.heads) != 0 or @divTrunc(cfg.embed, cfg.heads) < ROPE_DIMS) return error.StableAudioConfigInvalid;
    return cfg;
}

fn jsonPath(v: std.json.Value, keys: []const []const u8) ?std.json.Value {
    var cur = v;
    for (keys) |k| {
        if (cur != .object) return null;
        cur = cur.object.get(k) orelse return null;
    }
    return cur;
}

fn jsonInt(v: std.json.Value, key: []const u8) ?c_int {
    const x = jsonPath(v, &.{key}) orelse return null;
    return if (x == .integer and x.integer > 0 and x.integer < std.math.maxInt(c_int)) @intCast(x.integer) else null;
}

/// The `seconds_total` training saw for a clip this long (`ceil(samples / rate)`):
/// a fraction is a condition it never saw, and the output is clipped noise.
pub fn trainedSeconds(seconds: f32) f32 {
    return @max(1, @ceil(seconds));
}

/// Latent frames sampled for a duration: the request plus the headroom, rounded
/// up to an even count (the encoder's 2-latent alignment), capped at `SAMPLE_SIZE`.
pub fn latentCount(seconds: f32) u32 {
    const samples = @floor((@as(f64, seconds) + HEADROOM_SECONDS) * SAMPLE_RATE);
    const n: u32 = @intFromFloat(@ceil(samples / SAMPLES_PER_LATENT));
    return @min(n + n % 2, SAMPLE_SIZE / SAMPLES_PER_LATENT);
}

/// The ping-pong schedule: linspace(1, 0) warped through LogSNR space
/// (anchor -6.2, end 2.0); endpoints exact.
pub fn schedule(out: []f32) void {
    const n = out.len - 1;
    for (out, 0..) |*o, i| {
        const t: f32 = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        const logsnr = 2.0 - t * (2.0 - -6.2);
        o.* = if (i == 0) 1.0 else if (i == n) 0.0 else 1.0 / (1.0 + @exp(logsnr));
    }
}

/// stable-audio-tools ExpoFourierFeatures over 256 dims: [cos | sin] of
/// t·2π·f, f log-spaced in [0.5, 10000]. f32 throughout, as the reference.
fn expoFeatures(t: f32, out: *[256]f32) void {
    const lmin: f32 = @log(@as(f32, 0.5));
    const lspan: f32 = @floatCast(@log(@as(f64, 10000.0)) - @log(@as(f64, 0.5)));
    for (0..128) |i| {
        const ramp: f32 = @as(f32, @floatFromInt(i)) / 127.0;
        const f: f32 = @exp(ramp * lspan + lmin) * 2.0 * std.math.pi;
        const arg = t * f;
        out[i] = @cos(arg);
        out[128 + i] = @sin(arg);
    }
}

/// One SAME-S decode call over `latents[start..start+len]`, keeping output
/// latent slots [keep_from, keep_from+keep_len). `pad_last` appends a copy of
/// the last latent (odd lengths under 7 have no even window).
pub const Window = struct { start: u32, len: u32, keep_from: u32, keep_len: u32, pad_last: bool = false };

/// The reference dispatch (`sa3_mlx.py` + `decode_chunked`): every window is
/// an EVEN run of REAL latents (34-row chunks need T·17 ≡ 0 mod 34), with
/// `ovl` latents of context each side; never zero padding.
pub fn decodePlan(a: std.mem.Allocator, t: u32) ![]Window {
    var out: std.ArrayList(Window) = .empty;
    errdefer out.deinit(a);
    if (t <= 12 and t % 2 == 0) {
        try out.append(a, .{ .start = 0, .len = t, .keep_from = 0, .keep_len = t });
    } else if (t <= 6) {
        try out.append(a, .{ .start = 0, .len = t, .keep_from = 0, .keep_len = t, .pad_last = true });
    } else {
        const chunk: u32 = if (t > 12) 8 else 2;
        const ovl: u32 = 2;
        const kernel = chunk + 2 * ovl;
        try out.append(a, .{ .start = 0, .len = kernel, .keep_from = 0, .keep_len = chunk + ovl });
        var i: u32 = chunk + ovl;
        while (i + chunk + ovl <= t) : (i += chunk)
            try out.append(a, .{ .start = i - ovl, .len = kernel, .keep_from = ovl, .keep_len = chunk });
        if (i < t) try out.append(a, .{ .start = t - kernel, .len = kernel, .keep_from = kernel - (t - i), .keep_len = t - i });
    }
    return out.toOwnedSlice(a);
}

// ════════════════════════════════════════════════════════════════════════
// Op scope: every intermediate a forward makes, freed together.
// ════════════════════════════════════════════════════════════════════════

const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,

    fn init(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }
    fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }
    fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }
    /// An op's output handle `o`, after the op returned `rc`.
    fn res(self: *Scope, rc: c_int, o: *const A) !A {
        if (rc != 0) {
            _ = mlx.mlx_array_free(o.*);
            return error.MlxError;
        }
        return self.keep(o.*);
    }
    /// A new handle on `x` that outlives the scope (caller frees).
    fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }

    fn add(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_add(&o, x, y, sc.s), &o);
    }
    fn sub(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_subtract(&o, x, y, sc.s), &o);
    }
    fn mul(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_multiply(&o, x, y, sc.s), &o);
    }
    fn scalar(sc: *Scope, v: f32) !A {
        return sc.keep(mlx.mlx_array_new_float(v));
    }
    fn addS(sc: *Scope, x: A, v: f32) !A {
        return sc.add(x, try sc.scalar(v));
    }
    fn mulS(sc: *Scope, x: A, v: f32) !A {
        return sc.mul(x, try sc.scalar(v));
    }
    fn matmul(sc: *Scope, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_matmul(&o, x, y, sc.s), &o);
    }
    fn tanh(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_tanh(&o, x, sc.s), &o);
    }
    fn sigmoid(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_sigmoid(&o, x, sc.s), &o);
    }
    fn silu(sc: *Scope, x: A) !A {
        return sc.mul(x, try sc.sigmoid(x));
    }
    fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    /// x[..., start:stop, ...] on one axis.
    fn slice(sc: *Scope, x: A, axis: usize, start: c_int, stop: c_int) !A {
        const sh = mlx.getShape(x);
        var lo: [8]c_int = @splat(0);
        var hi: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(hi[0..sh.len], sh);
        lo[axis] = start;
        hi[axis] = stop;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &lo, sh.len, &hi, sh.len, &st, sh.len, sc.s), &o);
    }
    fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (xs) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    /// x @ w^T (+ b) — PyTorch Linear over the last axis.
    fn lin(sc: *Scope, x: A, w: A, b: ?A) !A {
        const y = try sc.matmul(x, try sc.transpose(w, &.{ 1, 0 }));
        return if (b) |bb| sc.add(y, bb) else y;
    }
    fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, x, w, eps, sc.s), &o);
    }
    fn rope(sc: *Scope, x: A, dims: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rope(&o, x, dims, false, mlx.mlx_optional_float.some(10000.0), 1.0, 0, .{ .ctx = null }, sc.s), &o);
    }
    fn sdpa(sc: *Scope, q: A, k: A, v: A, scale: f32) !A {
        var o = mlx.mlx_array_new();
        const none = A{ .ctx = null };
        return sc.res(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", none, none, false, sc.s), &o);
    }
    /// [B,L,H*D] → [B,H,L,D]
    fn heads(sc: *Scope, x: A, h: c_int, d: c_int) !A {
        const sh = mlx.getShape(x);
        return sc.transpose(try sc.reshape(x, &.{ sh[0], sh[1], h, d }), &.{ 0, 2, 1, 3 });
    }
    /// [B,H,L,D] → [B,L,H*D]
    fn merge(sc: *Scope, x: A) !A {
        const sh = mlx.getShape(x);
        return sc.reshape(try sc.transpose(x, &.{ 0, 2, 1, 3 }), &.{ sh[0], sh[2], sh[1] * sh[3] });
    }
    fn fromF32(sc: *Scope, data: []const f32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
    }
};

// ════════════════════════════════════════════════════════════════════════
// Engine
// ════════════════════════════════════════════════════════════════════════

const T5Layer = struct { pre_attn: A, post_attn: A, pre_ff: A, post_ff: A, q: A, k: A, v: A, o: A, gate: A, up: A, down: A };

const DitLayer = struct {
    pre_norm: A,
    qkv: A,
    self_out: A,
    self_qn: A,
    self_kn: A,
    cross_norm: A,
    cross_q: A,
    cross_kv: A,
    cross_out: A,
    cross_qn: A,
    cross_kn: A,
    ff_norm: A,
    ff_in: A,
    ff_in_b: A,
    ff_out: A,
    ff_out_b: A,
    ssg: A,
    /// to_local_embed(0): inpaint conditioning is zeros for text-to-audio, so
    /// the MLP collapses to one [embed] vector (load-time, owned).
    local: A,
};

const DyT = struct { alpha: A, gamma: A, beta: A };

const DecBlock = struct { pre: DyT, ff_norm: DyT, qn: DyT, kn: DyT, qkv: A, out: A, glu: A, glu_b: A, po: A, po_b: A };

pub const Request = struct {
    prompt: []const u8,
    seconds: f32,
    steps: u32 = DEFAULT_STEPS,
    seed: u64 = 0,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: Cfg,
    w: Weights,
    t5w: Weights,
    tok: tok_mod.Tokenizer,
    /// Load-time derived arrays (owned): T5 `1 + w` norms, the fused
    /// weight-norm conv, the 1x1 conv matrices, DiT local vectors.
    owned: std.ArrayList(A),
    t5: [T5_LAYERS]T5Layer,
    t5_norm: A,
    t5_embed: A,
    dit: []DitLayer,
    pre_conv: A,
    post_conv: A,
    dec: [DEC_BLOCKS]DecBlock,
    dec_map: A,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.cfg = try parseConfig(io, allocator, model_dir);
        self.owned = .empty;
        errdefer self.freeOwned();

        self.w = try loadFiltered(allocator, model_dir, "model.safetensors", &.{ "model.model.", "conditioner.conditioners.", "pretransform.model.decoder.", "pretransform.model.bottleneck.running_std" });
        errdefer self.w.deinit();
        self.t5w = try loadFiltered(allocator, model_dir, "t5gemma-b-b-ul2/model.safetensors", &.{"model.encoder."});
        errdefer self.t5w.deinit();
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/t5gemma-b-b-ul2", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        errdefer self.tok.deinit();

        self.dit = try allocator.alloc(DitLayer, self.cfg.depth);
        errdefer allocator.free(self.dit);
        try self.resolve();
        log.info("[sa3] engine ready ({d} + {d} tensors, DiT {d}x{d})\n", .{ self.w.count(), self.t5w.count(), self.cfg.depth, self.cfg.embed });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.freeOwned();
        self.allocator.free(self.dit);
        self.w.deinit();
        self.t5w.deinit();
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    fn freeOwned(self: *Engine) void {
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.allocator);
    }

    fn get(w: *const Weights, comptime fmt: []const u8, args: anytype) !A {
        var buf: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, fmt, args);
        return w.get(key) orelse {
            log.err("[sa3] MISSING WEIGHT: {s}\n", .{key});
            return error.MissingWeight;
        };
    }

    /// Materialize a load-time derived array into `owned`.
    fn own(self: *Engine, x: A) !A {
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_contiguous(&o, x, false, self.s));
        try mlx.check(mlx.mlx_array_eval(o));
        try self.owned.append(self.allocator, o);
        return o;
    }

    fn resolve(self: *Engine) !void {
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const w = &self.w;
        const t = &self.t5w;

        self.t5_embed = try get(t, "model.encoder.embed_tokens.weight", .{});
        self.t5_norm = try self.own(try sc.addS(try sc.astype(try get(t, "model.encoder.norm.weight", .{}), .float32), 1));
        for (&self.t5, 0..) |*l, i| {
            const n = struct {
                fn f(e: *Engine, c: *Scope, ww: *const Weights, li: usize, comptime name: []const u8) !A {
                    return e.own(try c.addS(try c.astype(try get(ww, "model.encoder.layers.{d}." ++ name ++ ".weight", .{li}), .float32), 1));
                }
            }.f;
            l.pre_attn = try n(self, &sc, t, i, "pre_self_attn_layernorm");
            l.post_attn = try n(self, &sc, t, i, "post_self_attn_layernorm");
            l.pre_ff = try n(self, &sc, t, i, "pre_feedforward_layernorm");
            l.post_ff = try n(self, &sc, t, i, "post_feedforward_layernorm");
            l.q = try get(t, "model.encoder.layers.{d}.self_attn.q_proj.weight", .{i});
            l.k = try get(t, "model.encoder.layers.{d}.self_attn.k_proj.weight", .{i});
            l.v = try get(t, "model.encoder.layers.{d}.self_attn.v_proj.weight", .{i});
            l.o = try get(t, "model.encoder.layers.{d}.self_attn.o_proj.weight", .{i});
            l.gate = try get(t, "model.encoder.layers.{d}.mlp.gate_proj.weight", .{i});
            l.up = try get(t, "model.encoder.layers.{d}.mlp.up_proj.weight", .{i});
            l.down = try get(t, "model.encoder.layers.{d}.mlp.down_proj.weight", .{i});
        }

        const p = "model.model.transformer.layers.{d}.";
        for (self.dit, 0..) |*l, i| {
            l.pre_norm = try get(w, p ++ "pre_norm.gamma", .{i});
            l.qkv = try get(w, p ++ "self_attn.to_qkv.weight", .{i});
            l.self_out = try get(w, p ++ "self_attn.to_out.weight", .{i});
            l.self_qn = try get(w, p ++ "self_attn.q_norm.gamma", .{i});
            l.self_kn = try get(w, p ++ "self_attn.k_norm.gamma", .{i});
            l.cross_norm = try get(w, p ++ "cross_attend_norm.gamma", .{i});
            l.cross_q = try get(w, p ++ "cross_attn.to_q.weight", .{i});
            l.cross_kv = try get(w, p ++ "cross_attn.to_kv.weight", .{i});
            l.cross_out = try get(w, p ++ "cross_attn.to_out.weight", .{i});
            l.cross_qn = try get(w, p ++ "cross_attn.q_norm.gamma", .{i});
            l.cross_kn = try get(w, p ++ "cross_attn.k_norm.gamma", .{i});
            l.ff_norm = try get(w, p ++ "ff_norm.gamma", .{i});
            l.ff_in = try get(w, p ++ "ff.ff.0.proj.weight", .{i});
            l.ff_in_b = try get(w, p ++ "ff.ff.0.proj.bias", .{i});
            l.ff_out = try get(w, p ++ "ff.ff.2.weight", .{i});
            l.ff_out_b = try get(w, p ++ "ff.ff.2.bias", .{i});
            l.ssg = try get(w, p ++ "to_scale_shift_gate", .{i});
            const b0 = try get(w, p ++ "to_local_embed.0.bias", .{i});
            l.local = try self.own(try sc.lin(try sc.silu(b0), try get(w, p ++ "to_local_embed.2.weight", .{i}), try get(w, p ++ "to_local_embed.2.bias", .{i})));
        }
        const c = LATENT;
        self.pre_conv = try self.own(try sc.reshape(try get(w, "model.model.preprocess_conv.weight", .{}), &.{ c, c }));
        self.post_conv = try self.own(try sc.reshape(try get(w, "model.model.postprocess_conv.weight", .{}), &.{ c, c }));

        const d = "pretransform.model.decoder.layers.3.";
        for (&self.dec, 0..) |*b, i| {
            const norm = struct {
                fn f(ww: *const Weights, bi: usize, comptime name: []const u8) !DyT {
                    return .{
                        .alpha = try get(ww, d ++ "transformers.{d}." ++ name ++ ".alpha", .{bi}),
                        .gamma = try get(ww, d ++ "transformers.{d}." ++ name ++ ".gamma", .{bi}),
                        .beta = try get(ww, d ++ "transformers.{d}." ++ name ++ ".beta", .{bi}),
                    };
                }
            }.f;
            b.pre = try norm(w, i, "pre_norm");
            b.ff_norm = try norm(w, i, "ff_norm");
            b.qn = try norm(w, i, "self_attn.q_norm");
            b.kn = try norm(w, i, "self_attn.k_norm");
            b.qkv = try get(w, d ++ "transformers.{d}.self_attn.to_qkv.weight", .{i});
            b.out = try get(w, d ++ "transformers.{d}.self_attn.to_out.weight", .{i});
            b.glu = try get(w, d ++ "transformers.{d}.ff.ff.0.proj.weight", .{i});
            b.glu_b = try get(w, d ++ "transformers.{d}.ff.ff.0.proj.bias", .{i});
            b.po = try get(w, d ++ "transformers.{d}.ff.ff.2.weight", .{i});
            b.po_b = try get(w, d ++ "transformers.{d}.ff.ff.2.bias", .{i});
        }
        // WNConv1d: w = g · v / ‖v‖ per output channel, then MLX's [out, k, in].
        const g = try get(w, d ++ "mapping.weight_g", .{});
        const v = try get(w, d ++ "mapping.weight_v", .{});
        var n1 = mlx.mlx_array_new();
        const sq = try sc.mul(v, v);
        const r1 = try sc.res(mlx.mlx_sum_axis(&n1, sq, 1, true, sc.s), &n1);
        var n2 = mlx.mlx_array_new();
        const r2 = try sc.res(mlx.mlx_sum_axis(&n2, r1, 2, true, sc.s), &n2);
        var nrm = mlx.mlx_array_new();
        const norm = try sc.res(mlx.mlx_sqrt(&nrm, r2, sc.s), &nrm);
        var dv = mlx.mlx_array_new();
        const fused = try sc.res(mlx.mlx_divide(&dv, try sc.mul(g, v), norm, sc.s), &dv);
        self.dec_map = try self.own(try sc.transpose(fused, &.{ 0, 2, 1 }));
    }

    // ── conditioning ────────────────────────────────────────────────────

    /// T5Gemma token ids (no BOS/EOS — the HF tokenizer_config), capped at 256.
    pub fn tokenize(self: *const Engine, a: std.mem.Allocator, prompt: []const u8) ![]i32 {
        const ids = try self.tok.encode(a, prompt);
        defer a.free(ids);
        const n = @min(ids.len, MAX_TOKENS);
        const out = try a.alloc(i32, n);
        for (out, ids[0..n]) |*o, id| o.* = @intCast(id);
        return out;
    }

    /// T5Gemma encoder over the real tokens only → [1, n, 768] f32. Pad keys are
    /// masked out in the reference, so the unpadded forward is the same math.
    pub fn encodeText(self: *Engine, sc: *Scope, ids: []const i32) !A {
        const n: c_int = @intCast(ids.len);
        const ids_a = try sc.keep(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, n }, 2, .int32));
        var e = mlx.mlx_array_new();
        const emb = try sc.res(mlx.mlx_take_axis(&e, self.t5_embed, ids_a, 0, sc.s), &e);
        var x = try sc.mulS(try sc.astype(emb, .float32), @sqrt(@as(f32, @floatFromInt(COND))));
        for (&self.t5) |*l| {
            var h = try sc.rms(x, l.pre_attn, T5_EPS);
            const q = try sc.rope(try sc.heads(try sc.lin(h, l.q, null), T5_HEADS, T5_HD), T5_HD);
            const k = try sc.rope(try sc.heads(try sc.lin(h, l.k, null), T5_HEADS, T5_HD), T5_HD);
            const v = try sc.heads(try sc.lin(h, l.v, null), T5_HEADS, T5_HD);
            var scores = try sc.mulS(try sc.matmul(q, try sc.transpose(k, &.{ 0, 1, 3, 2 })), 1.0 / @sqrt(@as(f32, @floatFromInt(T5_HD))));
            scores = try sc.mulS(try sc.tanh(try sc.mulS(scores, 1.0 / T5_SOFTCAP)), T5_SOFTCAP);
            var pr = mlx.mlx_array_new();
            const probs = try sc.res(mlx.mlx_softmax_axis(&pr, scores, -1, true, sc.s), &pr);
            h = try sc.lin(try sc.merge(try sc.matmul(probs, v)), l.o, null);
            x = try sc.add(x, try sc.rms(h, l.post_attn, T5_EPS));
            h = try sc.rms(x, l.pre_ff, T5_EPS);
            h = try sc.lin(try sc.mul(try geluTanh(sc, try sc.lin(h, l.gate, null)), try sc.lin(h, l.up, null)), l.down, null);
            x = try sc.add(x, try sc.rms(h, l.post_ff, T5_EPS));
        }
        return sc.rms(x, self.t5_norm, T5_EPS);
    }

    pub const Cond = struct { cross: A, global: A };

    /// Cross-attention rows [1, 257, 768] (prompt rows, learned padding to 256,
    /// the seconds row) and the global condition [1, 768] (the seconds row).
    pub fn condition(self: *Engine, sc: *Scope, ids: []const i32, seconds: f32) !Cond {
        const cp = "conditioner.conditioners.";
        var feats: [256]f32 = undefined;
        expoFeatures(std.math.clamp(seconds, 0, 384) / 384.0, &feats);
        const f = try sc.fromF32(&feats, &.{ 1, 256 });
        const sec = try sc.lin(f, try get(&self.w, cp ++ "seconds_total.embedder.embedding.1.weight", .{}), try get(&self.w, cp ++ "seconds_total.embedder.embedding.1.bias", .{}));
        const sec3 = try sc.reshape(sec, &.{ 1, 1, COND });
        const pad_n: c_int = @intCast(MAX_TOKENS - ids.len);
        var parts: [3]A = undefined;
        var np: usize = 0;
        if (ids.len > 0) {
            parts[np] = try self.encodeText(sc, ids);
            np += 1;
        }
        if (pad_n > 0) {
            const pe = try sc.reshape(try get(&self.w, cp ++ "prompt.padding_embedding", .{}), &.{ 1, 1, COND });
            parts[np] = try sc.broadcast(pe, &.{ 1, pad_n, COND });
            np += 1;
        }
        parts[np] = sec3;
        np += 1;
        return .{ .cross = try sc.concat(parts[0..np], 1), .global = sec };
    }

    // ── DiT ─────────────────────────────────────────────────────────────

    /// Everything constant across sampler steps: cross-attention K/V per
    /// layer, the projected global condition, the local-add row mask.
    const Prep = struct { k: []A, v: []A, global: A, local_mask: A };

    /// The returned arrays live in `sc`, evaluated; their intermediates do not.
    /// `k`/`v` are owned by the caller (free with `sc.a`).
    fn prepare(self: *Engine, sc: *Scope, cond: Cond, t_lat: c_int) !Prep {
        const w = &self.w;
        const k = try sc.a.alloc(A, self.dit.len);
        errdefer sc.a.free(k);
        const v = try sc.a.alloc(A, self.dit.len);
        errdefer sc.a.free(v);
        var ps = Scope.init(sc.a, sc.s);
        defer ps.deinit();
        const ctx = try ps.lin(try ps.silu(try ps.lin(cond.cross, try get(w, "model.model.to_cond_embed.0.weight", .{}), null)), try get(w, "model.model.to_cond_embed.2.weight", .{}), null);
        const e = self.cfg.embed;
        const hd = @divExact(e, self.cfg.heads);
        const outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        for (self.dit, 0..) |*l, i| {
            const kv = try ps.lin(ctx, l.cross_kv, null);
            k[i] = try sc.keep(ps.out(try ps.rms(try ps.heads(try ps.slice(kv, 2, 0, e), self.cfg.heads, hd), l.cross_kn, 1e-6)));
            v[i] = try sc.keep(ps.out(try ps.heads(try ps.slice(kv, 2, e, 2 * e), self.cfg.heads, hd)));
            _ = mlx.mlx_vector_array_append_value(outs, k[i]);
            _ = mlx.mlx_vector_array_append_value(outs, v[i]);
        }
        const global = try sc.keep(ps.out(try ps.lin(try ps.silu(try ps.lin(cond.global, try get(w, "model.model.to_global_embed.0.weight", .{}), null)), try get(w, "model.model.to_global_embed.2.weight", .{}), null)));
        var z = mlx.mlx_array_new();
        const zeros = try ps.res(mlx.mlx_zeros(&z, &[_]c_int{ 1, self.cfg.memory, 1 }, 3, .float32, ps.s), &z);
        var o = mlx.mlx_array_new();
        const ones = try ps.res(mlx.mlx_ones(&o, &[_]c_int{ 1, t_lat, 1 }, 3, .float32, ps.s), &o);
        const mask = try sc.keep(ps.out(try ps.concat(&.{ zeros, ones }, 1)));
        _ = mlx.mlx_vector_array_append_value(outs, global);
        _ = mlx.mlx_vector_array_append_value(outs, mask);
        try mlx.check(mlx.mlx_eval(outs));
        return .{ .k = k, .v = v, .global = global, .local_mask = mask };
    }

    /// Velocity at noise level `t` for latents x [1, T, 256] (channels last).
    fn ditForward(self: *Engine, sc: *Scope, prep: Prep, x: A, t: f32) !A {
        const w = &self.w;
        const e = self.cfg.embed;
        const hd = @divExact(e, self.cfg.heads);
        var feats: [256]f32 = undefined;
        expoFeatures(t, &feats);
        const tf = try sc.fromF32(&feats, &.{ 1, 256 });
        const te = try sc.lin(try sc.silu(try sc.lin(tf, try get(w, "model.model.to_timestep_embed.0.weight", .{}), try get(w, "model.model.to_timestep_embed.0.bias", .{}))), try get(w, "model.model.to_timestep_embed.2.weight", .{}), try get(w, "model.model.to_timestep_embed.2.bias", .{}));
        const ge = try sc.add(prep.global, te);
        const tp = "model.model.transformer.";
        const g = try sc.lin(try sc.silu(try sc.lin(ge, try get(w, tp ++ "global_cond_embedder.0.weight", .{}), try get(w, tp ++ "global_cond_embedder.0.bias", .{}))), try get(w, tp ++ "global_cond_embedder.2.weight", .{}), try get(w, tp ++ "global_cond_embedder.2.bias", .{}));

        const xp = try sc.add(x, try sc.lin(x, self.pre_conv, null));
        const mem = try sc.reshape(try get(w, tp ++ "memory_tokens", .{}), &.{ 1, self.cfg.memory, e });
        var h = try sc.concat(&.{ mem, try sc.lin(xp, try get(w, tp ++ "project_in.weight", .{}), null) }, 1);

        for (self.dit, 0..) |*l, i| {
            const ss = try sc.reshape(try sc.add(l.ssg, g), &.{ 1, 6, e });
            const scale_self = try sc.addS(try sc.slice(ss, 1, 0, 1), 1);
            const shift_self = try sc.slice(ss, 1, 1, 2);
            const gate_self = try sc.sigmoid(try sc.sub(try sc.scalar(1), try sc.slice(ss, 1, 2, 3)));
            const scale_ff = try sc.addS(try sc.slice(ss, 1, 3, 4), 1);
            const shift_ff = try sc.slice(ss, 1, 4, 5);
            const gate_ff = try sc.sigmoid(try sc.sub(try sc.scalar(1), try sc.slice(ss, 1, 5, 6)));

            var a = try sc.add(try sc.mul(try sc.rms(h, l.pre_norm, 1e-5), scale_self), shift_self);
            const qkv = try sc.lin(a, l.qkv, null);
            const q = try sc.rope(try sc.rms(try sc.heads(try sc.slice(qkv, 2, 0, e), self.cfg.heads, hd), l.self_qn, 1e-6), ROPE_DIMS);
            const k = try sc.rope(try sc.rms(try sc.heads(try sc.slice(qkv, 2, e, 2 * e), self.cfg.heads, hd), l.self_kn, 1e-6), ROPE_DIMS);
            const v = try sc.heads(try sc.slice(qkv, 2, 2 * e, 3 * e), self.cfg.heads, hd);
            a = try sc.lin(try sc.merge(try sc.sdpa(q, k, v, 1.0 / @sqrt(@as(f32, @floatFromInt(hd))))), l.self_out, null);
            h = try sc.add(h, try sc.mul(a, gate_self));

            const cq = try sc.rms(try sc.heads(try sc.lin(try sc.rms(h, l.cross_norm, 1e-5), l.cross_q, null), self.cfg.heads, hd), l.cross_qn, 1e-6);
            h = try sc.add(h, try sc.lin(try sc.merge(try sc.sdpa(cq, prep.k[i], prep.v[i], 1.0 / @sqrt(@as(f32, @floatFromInt(hd))))), l.cross_out, null));
            h = try sc.add(h, try sc.mul(prep.local_mask, l.local));

            a = try sc.add(try sc.mul(try sc.rms(h, l.ff_norm, 1e-5), scale_ff), shift_ff);
            const ff = try sc.lin(a, l.ff_in, l.ff_in_b);
            const inner = @divExact(mlx.getShape(ff)[2], 2);
            a = try sc.lin(try sc.mul(try sc.slice(ff, 2, 0, inner), try sc.silu(try sc.slice(ff, 2, inner, 2 * inner))), l.ff_out, l.ff_out_b);
            h = try sc.add(h, try sc.mul(a, gate_ff));
        }
        const hl = mlx.getShape(h)[1];
        const o = try sc.lin(try sc.slice(h, 1, self.cfg.memory, hl), try get(w, tp ++ "project_out.weight", .{}), null);
        return sc.add(o, try sc.lin(o, self.post_conv, null));
    }

    /// Seeded ping-pong sampling → latents [1, T, 256] (caller frees). Noise is
    /// drawn channels-first like the reference so a seed means the same sound.
    pub fn sample(self: *Engine, cond: Cond, t_lat: u32, steps: u32, seed: u64, progress: ?sse.Progress) !A {
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const prep = try self.prepare(&sc, cond, @intCast(t_lat));
        defer self.allocator.free(prep.k);
        defer self.allocator.free(prep.v);
        const sigmas = try self.allocator.alloc(f32, steps + 1);
        defer self.allocator.free(sigmas);
        schedule(sigmas);

        var k_init = mlx.mlx_array_new();
        var x = try self.noise(&sc, try sc.res(mlx.mlx_random_key(&k_init, seed), &k_init), t_lat);
        var k0 = mlx.mlx_array_new();
        var key = try sc.res(mlx.mlx_random_key(&k0, seed +% 1), &k0);
        for (0..steps) |i| {
            // The step's intermediates are released BEFORE the eval: a live
            // handle pins its buffer, so the whole step would stay resident.
            const next = blk: {
                var step = Scope.init(self.allocator, self.s);
                defer step.deinit();
                const v = try self.ditForward(&step, prep, x, sigmas[i]);
                const den = try step.sub(x, try step.mulS(v, sigmas[i]));
                const tn = sigmas[i + 1];
                if (i + 1 == steps or tn == 0) break :blk step.out(den);
                var k1 = mlx.mlx_array_new();
                var k2 = mlx.mlx_array_new();
                const rc = mlx.mlx_random_split(&k1, &k2, key, self.s);
                key = sc.keep(k1) catch |err| {
                    _ = mlx.mlx_array_free(k2);
                    return err;
                };
                const sub = try sc.keep(k2);
                try mlx.check(rc);
                break :blk step.out(try step.add(try step.mulS(den, 1 - tn), try step.mulS(try self.noise(&step, sub, t_lat), tn)));
            };
            x = try sc.keep(next);
            try mlx.check(mlx.mlx_array_eval(x));
            if (progress) |p| {
                p.emit("diffuse", @intCast(i + 1), steps);
                if (p.boundary()) return error.Cancelled;
            }
        }
        return sc.out(x);
    }

    fn noise(_: *Engine, sc: *Scope, key: A, t_lat: u32) !A {
        var n = mlx.mlx_array_new();
        const z = try sc.res(mlx.mlx_random_normal(&n, &[_]c_int{ 1, LATENT, @intCast(t_lat) }, 3, .float32, 0, 1, key, sc.s), &n);
        return sc.transpose(z, &.{ 0, 2, 1 });
    }

    // ── SAME-S decoder ──────────────────────────────────────────────────

    fn dyt(sc: *Scope, x: A, p: DyT) !A {
        return sc.add(try sc.mul(p.gamma, try sc.tanh(try sc.mul(p.alpha, x))), p.beta);
    }

    fn decBlock(sc: *Scope, x: A, b: *const DecBlock) !A {
        const hd = @divExact(DEC_DIM, DEC_HEADS);
        const qkv = try sc.lin(try dyt(sc, x, b.pre), b.qkv, null);
        var hh: [5]A = undefined;
        for (&hh, 0..) |*o, j| o.* = try sc.heads(try sc.slice(qkv, 2, @as(c_int, @intCast(j)) * DEC_DIM, @as(c_int, @intCast(j + 1)) * DEC_DIM), DEC_HEADS, hd);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const q = try sc.rope(try dyt(sc, hh[0], b.qn), ROPE_DIMS);
        const k = try sc.rope(try dyt(sc, hh[1], b.kn), ROPE_DIMS);
        const qd = try sc.rope(try dyt(sc, hh[3], b.qn), ROPE_DIMS);
        const kd = try sc.rope(try dyt(sc, hh[4], b.kn), ROPE_DIMS);
        const att = try sc.sub(try sc.sdpa(q, k, hh[2], scale), try sc.sdpa(qd, kd, hh[2], scale));
        const y = try sc.add(x, try sc.lin(try sc.merge(att), b.out, null));
        const ff = try sc.lin(try dyt(sc, y, b.ff_norm), b.glu, b.glu_b);
        const inner = @divExact(mlx.getShape(ff)[2], 2);
        return sc.add(y, try sc.lin(try sc.mul(try sc.slice(ff, 2, 0, inner), try sc.silu(try sc.slice(ff, 2, inner, 2 * inner))), b.po, b.po_b));
    }

    /// One decoder call: latents [1, n, 256] (n even) → patches [1, n·16, 512].
    fn decodeWindow(self: *Engine, sc: *Scope, lat: A) !A {
        const w = &self.w;
        const dp = "pretransform.model.decoder.layers.";
        const n = mlx.getShape(lat)[1];
        const rows = n * DEC_SUB;
        var x = try sc.mul(lat, try get(w, "pretransform.model.bottleneck.running_std", .{}));
        x = try sc.lin(x, try get(w, dp ++ "1.weight", .{}), try get(w, dp ++ "1.bias", .{}));
        const nt = try sc.broadcast(try sc.reshape(try get(w, dp ++ "3.new_tokens", .{}), &.{ 1, 1, 1, DEC_DIM }), &.{ 1, n, DEC_STRIDE, DEC_DIM });
        x = try sc.reshape(try sc.concat(&.{ try sc.reshape(x, &.{ 1, n, 1, DEC_DIM }), nt }, 2), &.{ @divExact(rows, DEC_CHUNK), DEC_CHUNK, DEC_DIM });
        for (self.dec[0..3]) |*b| x = try decBlock(sc, x, b);
        // Second half: shifted by half a chunk, edges padded with their own rows.
        x = try sc.reshape(x, &.{ 1, rows, DEC_DIM });
        const half = @divExact(DEC_CHUNK, 2);
        x = try sc.concat(&.{ try sc.slice(x, 1, 0, half), x, try sc.slice(x, 1, rows - half, rows) }, 1);
        x = try sc.reshape(x, &.{ @divExact(rows + DEC_CHUNK, DEC_CHUNK), DEC_CHUNK, DEC_DIM });
        for (self.dec[3..]) |*b| x = try decBlock(sc, x, b);
        x = try sc.slice(try sc.reshape(x, &.{ 1, rows + DEC_CHUNK, DEC_DIM }), 1, half, half + rows);
        x = try sc.slice(try sc.reshape(x, &.{ n, DEC_SUB, DEC_DIM }), 1, 1, DEC_SUB);
        x = try sc.reshape(x, &.{ 1, n * DEC_STRIDE, DEC_DIM });
        var c = mlx.mlx_array_new();
        const conv = try sc.res(mlx.mlx_conv1d(&c, x, self.dec_map, 1, 1, 1, 1, sc.s), &c);
        return sc.add(conv, try get(w, dp ++ "3.mapping.bias", .{}));
    }

    /// latents [1, T, 256] → patches [1, T·16, 512] through the reference's
    /// window plan (`decodePlan`).
    pub fn decode(self: *Engine, latents: A) !A {
        const t_lat: u32 = @intCast(mlx.getShape(latents)[1]);
        const plan = try decodePlan(self.allocator, t_lat);
        defer self.allocator.free(plan);
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const pieces = try self.allocator.alloc(A, plan.len);
        defer self.allocator.free(pieces);
        for (plan, pieces) |win, *piece| {
            const kept = blk: {
                var ws = Scope.init(self.allocator, self.s);
                defer ws.deinit();
                var lat = try ws.slice(latents, 1, @intCast(win.start), @intCast(win.start + win.len));
                if (win.pad_last) lat = try ws.concat(&.{ lat, try ws.slice(lat, 1, @intCast(win.len - 1), @intCast(win.len)) }, 1);
                const p = try self.decodeWindow(&ws, lat);
                break :blk ws.out(try ws.slice(p, 1, @intCast(win.keep_from * 16), @intCast((win.keep_from + win.keep_len) * 16)));
            };
            piece.* = try sc.keep(kept);
            try mlx.check(mlx.mlx_array_eval(piece.*));
        }
        return sc.out(try sc.concat(pieces, 1));
    }

    /// prompt → 44.1 kHz stereo PCM16 WAV (owned).
    pub fn generateWav(self: *Engine, allocator: std.mem.Allocator, req: Request, progress: ?sse.Progress) ![]u8 {
        const secs = trainedSeconds(req.seconds);
        const t_lat = latentCount(secs);
        const ids = try self.tokenize(allocator, req.prompt);
        defer allocator.free(ids);
        log.info("[sa3] {d:.2}s -> {d} latents, {d} prompt tokens, steps={d}, seed={d}\n", .{ req.seconds, t_lat, ids.len, req.steps, req.seed });

        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        // The text encoder's intermediates go before sampling starts.
        const cond = blk: {
            var cs = Scope.init(self.allocator, self.s);
            defer cs.deinit();
            const c = try self.condition(&cs, ids, secs);
            break :blk Cond{ .cross = try sc.keep(cs.out(c.cross)), .global = try sc.keep(cs.out(c.global)) };
        };
        const outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        _ = mlx.mlx_vector_array_append_value(outs, cond.cross);
        _ = mlx.mlx_vector_array_append_value(outs, cond.global);
        try mlx.check(mlx.mlx_eval(outs));
        if (progress) |p| p.emit("encode", 1, 1);

        const lat = try sc.keep(try self.sample(cond, t_lat, req.steps, req.seed, progress));
        if (progress) |p| p.emit("decode", 0, 1);
        const patches = try sc.keep(try self.decode(lat));
        const samples: usize = @intFromFloat(@round(@as(f64, req.seconds) * SAMPLE_RATE));
        const pcm = try unpatch(self.allocator, self.s, patches, samples);
        defer self.allocator.free(pcm);
        if (progress) |p| p.emit("decode", 1, 1);
        return wav_mod.encodePcm16(allocator, pcm, SAMPLE_RATE, 2);
    }
};

/// patches [1, L, 512] → interleaved stereo f32 (owned), trimmed to `samples`
/// frames: channel c, patch l, offset h is sample l·256 + h of channel c.
fn unpatch(a: std.mem.Allocator, s: S, patches: A, samples: usize) ![]f32 {
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const l = mlx.getShape(patches)[1];
    const st = try sc.reshape(try sc.transpose(try sc.reshape(patches, &.{ l, 2, DEC_PATCH }), &.{ 0, 2, 1 }), &.{ l * DEC_PATCH, 2 });
    var c = mlx.mlx_array_new();
    const cont = try sc.res(mlx.mlx_contiguous(&c, st, false, s), &c);
    try mlx.check(mlx.mlx_array_eval(cont));
    const data = mlx.mlx_array_data_float32(cont) orelse return error.NoData;
    const n = @min(samples * 2, mlx.mlx_array_size(cont));
    return a.dupe(f32, data[0..n]);
}

/// tanh-approximated GELU (Gemma's `gelu_pytorch_tanh`).
fn geluTanh(sc: *Scope, x: A) !A {
    const x3 = try sc.mul(try sc.mul(x, x), x);
    const inner = try sc.mulS(try sc.add(x, try sc.mulS(x3, 0.044715)), @sqrt(2.0 / std.math.pi));
    return sc.mul(try sc.mulS(x, 0.5), try sc.addS(try sc.tanh(inner), 1));
}

/// Load one safetensors file, keeping only keys under `prefixes`, materialized.
fn loadFiltered(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8, prefixes: []const []const u8) !Weights {
    var w = try model_mod.loadWeightsFile(allocator, model_dir, file);
    errdefer w.deinit();
    var drop: std.ArrayList([]const u8) = .empty;
    defer drop.deinit(allocator);
    const keep = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(keep);
    var it = w.map.iterator();
    while (it.next()) |e| {
        const wanted = for (prefixes) |p| {
            if (std.mem.startsWith(u8, e.key_ptr.*, p)) break true;
        } else false;
        if (wanted) _ = mlx.mlx_vector_array_append_value(keep, e.value_ptr.*) else try drop.append(allocator, e.key_ptr.*);
    }
    for (drop.items) |k| {
        const kv = w.map.fetchRemove(k).?;
        _ = mlx.mlx_array_free(kv.value);
        allocator.free(kv.key);
    }
    try mlx.check(mlx.mlx_eval(keep));
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first, then env-gated oracles (SA3_TEST_MODEL + SA3_*,
// fed by tests/dump_stable_audio_fixtures.py).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sa3 schedule: 8 steps match the reference's LogSNR-shifted linspace" {
    var s: [9]f32 = undefined;
    schedule(&s);
    const ref = [_]f32{ 1.0, 0.9943755865097046, 0.9844802618026733, 0.957912266254425, 0.8909031748771667, 0.7455465793609619, 0.5124973654747009, 0.27388501167297363, 0.0 };
    for (ref, s) |r, v| try testing.expectApproxEqAbs(r, v, 1e-6);
}

test "sa3 trainedSeconds: whole seconds rounded up, at least 1" {
    try testing.expectEqual(@as(f32, 1), trainedSeconds(0.01));
    try testing.expectEqual(@as(f32, 1), trainedSeconds(0.9));
    try testing.expectEqual(@as(f32, 1), trainedSeconds(1));
    try testing.expectEqual(@as(f32, 2), trainedSeconds(1.5));
    try testing.expectEqual(@as(f32, 120), trainedSeconds(119.2));
}

test "sa3 latentCount: the request plus 6 s of headroom, even, capped at sample_size" {
    try testing.expectEqual(@as(u32, 76), latentCount(1));
    try testing.expectEqual(@as(u32, 120), latentCount(5));
    try testing.expectEqual(@as(u32, 388), latentCount(30));
    try testing.expectEqual(@as(u32, 1292), latentCount(115));
    try testing.expectEqual(@as(u32, 1292), latentCount(120));
}

test "sa3 decodePlan: even windows of real latents that tile every length exactly once" {
    for (1..400) |tu| {
        const t: u32 = @intCast(tu);
        const plan = try decodePlan(testing.allocator, t);
        defer testing.allocator.free(plan);
        var covered: u32 = 0;
        for (plan) |w| {
            try testing.expect(w.start + w.len <= t);
            try testing.expect(w.keep_from + w.keep_len <= w.len);
            try testing.expectEqual(covered, w.start + w.keep_from);
            try testing.expectEqual(@as(u32, 0), (w.len + @intFromBool(w.pad_last)) % 2);
            covered += w.keep_len;
        }
        try testing.expectEqual(t, covered);
    }
    // 5 s: the reference's chunk 8 / overlap 2 dispatch.
    const plan = try decodePlan(testing.allocator, 54);
    defer testing.allocator.free(plan);
    try testing.expectEqual(Window{ .start = 0, .len = 12, .keep_from = 0, .keep_len = 10 }, plan[0]);
    try testing.expectEqual(Window{ .start = 8, .len = 12, .keep_from = 2, .keep_len = 8 }, plan[1]);
    try testing.expectEqual(Window{ .start = 42, .len = 12, .keep_from = 8, .keep_len = 4 }, plan[plan.len - 1]);
}

test "sa3 config: a geometry that does not divide into heads is a named error" {
    const a = testing.allocator;
    const ok = try std.json.parseFromSlice(std.json.Value, a, "{\"model\":{\"diffusion\":{\"config\":{\"embed_dim\":1024,\"num_heads\":16,\"depth\":20}}}}", .{});
    defer ok.deinit();
    try testing.expectEqual(Cfg{}, try cfgFromJson(ok.value));
    for ([_][]const u8{
        "{\"model\":{\"diffusion\":{\"config\":{\"embed_dim\":1000,\"num_heads\":16}}}}",
        "{\"model\":{\"diffusion\":{\"config\":{\"embed_dim\":1024,\"num_heads\":3}}}}",
        "{\"model\":{}}",
    }) |src| {
        const p = try std.json.parseFromSlice(std.json.Value, a, src, .{});
        defer p.deinit();
        try testing.expectError(error.StableAudioConfigInvalid, cfgFromJson(p.value));
    }
    const med = try std.json.parseFromSlice(std.json.Value, a, "{\"model\":{\"diffusion\":{\"config\":{\"attn_kwargs\":{\"differential\":true}}}}}", .{});
    defer med.deinit();
    try testing.expectError(error.StableAudioMediumUnsupported, cfgFromJson(med.value));
}

test "sa3 unpatch: patch row l, channel c, offset h is interleaved sample (l*256+h, c), trimmed" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    const l = 2;
    var data: [l * 512]f32 = undefined;
    for (0..l) |li| for (0..2) |c| for (0..256) |h| {
        data[li * 512 + c * 256 + h] = @floatFromInt(c * 10000 + li * 256 + h);
    };
    var sc = Scope.init(a, s);
    defer sc.deinit();
    const pcm = try unpatch(a, s, try sc.fromF32(&data, &.{ 1, l, 512 }), 300);
    defer a.free(pcm);
    try testing.expectEqual(@as(usize, 600), pcm.len);
    for (0..300) |i| for (0..2) |c| {
        try testing.expectEqual(@as(f32, @floatFromInt(c * 10000 + i)), pcm[i * 2 + c]);
    };
}

fn readRaw(comptime T: type, a: std.mem.Allocator, env: [*:0]const u8) ![]T {
    const path = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
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

/// Cosine of `arr` (any layout) against a row-major f32 reference.
fn cosineTo(arr: A, ref: []const f32, s: S) !f64 {
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    var c = mlx.mlx_array_new();
    const cont = try sc.res(mlx.mlx_contiguous(&c, try sc.astype(arr, .float32), false, s), &c);
    try mlx.check(mlx.mlx_array_eval(cont));
    try testing.expectEqual(ref.len, mlx.mlx_array_size(cont));
    const d = mlx.mlx_array_data_float32(cont) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (ref, 0..) |r, i| {
        try testing.expect(std.math.isFinite(d[i]));
        dot += @as(f64, d[i]) * r;
        na += @as(f64, d[i]) * d[i];
        nb += @as(f64, r) * r;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

fn testEngine() !*Engine {
    const dir = std.mem.span(std.c.getenv("SA3_TEST_MODEL") orelse return error.SkipZigTest);
    return Engine.load(std.Io.Threaded.global_single_threaded.io(), testing.allocator, dir);
}

const ORACLE_PROMPT = "Dog barking next to a waterfall";

test "sa3 oracle: prompt tokens, T5Gemma rows and conditioning match the reference" {
    const a = testing.allocator;
    const tok = try readRaw(i32, a, "SA3_TOK");
    defer a.free(tok);
    const t5 = try readRaw(f32, a, "SA3_T5");
    defer a.free(t5);
    const cross = try readRaw(f32, a, "SA3_CROSS");
    defer a.free(cross);
    const glob = try readRaw(f32, a, "SA3_GLOBAL");
    defer a.free(glob);
    var e = try testEngine();
    defer e.deinit();

    const ids = try e.tokenize(a, ORACLE_PROMPT);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, tok, ids);

    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    const c_t5 = try cosineTo(try e.encodeText(&sc, ids), t5, e.s);
    const cond = try e.condition(&sc, ids, 5.0);
    const c_cross = try cosineTo(cond.cross, cross, e.s);
    const c_glob = try cosineTo(cond.global, glob, e.s);
    std.debug.print("[sa3 oracle] t5 cos={d:.6} cross cos={d:.6} global cos={d:.6}\n", .{ c_t5, c_cross, c_glob });
    try testing.expect(c_t5 > 0.999);
    try testing.expect(c_cross > 0.999);
    try testing.expect(c_glob > 0.99999);
}

test "sa3: a 30 s sample + decode holds no more than one step's working set" {
    var e = try testEngine();
    defer e.deinit();
    const a = testing.allocator;
    const ids = try e.tokenize(a, ORACLE_PROMPT);
    defer a.free(ids);
    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    const cond = try e.condition(&sc, ids, 30);
    try mlx.check(mlx.mlx_array_eval(cond.cross));
    var base: usize = 0;
    _ = mlx.mlx_get_active_memory(&base);
    _ = mlx.mlx_reset_peak_memory();
    const lat = try sc.keep(try e.sample(cond, latentCount(30), DEFAULT_STEPS, 1, null));
    _ = try sc.keep(try e.decode(lat));
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("[sa3] 30 s working set {d} MB\n", .{(peak -| base) >> 20});
    try testing.expect(peak -| base < 1 << 30);
}

/// The reference's channels-first [1, 256, T] as our [1, T, 256].
fn latentsFromRef(sc: *Scope, data: []const f32) !A {
    const t: c_int = @intCast(data.len / @as(usize, LATENT));
    return sc.transpose(try sc.fromF32(data, &.{ 1, LATENT, t }), &.{ 0, 2, 1 });
}

test "sa3 oracle: one DiT velocity, the seeded 8-step sample and the SAME-S decode match the reference" {
    const a = testing.allocator;
    const cross = try readRaw(f32, a, "SA3_CROSS");
    defer a.free(cross);
    const glob = try readRaw(f32, a, "SA3_GLOBAL");
    defer a.free(glob);
    const xr = try readRaw(f32, a, "SA3_X");
    defer a.free(xr);
    const vr = try readRaw(f32, a, "SA3_DIT");
    defer a.free(vr);
    const latr = try readRaw(f32, a, "SA3_LAT");
    defer a.free(latr);
    const patchr = try readRaw(f32, a, "SA3_PATCH");
    defer a.free(patchr);
    var e = try testEngine();
    defer e.deinit();

    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    const cond = Engine.Cond{ .cross = try sc.fromF32(cross, &.{ 1, 257, COND }), .global = try sc.fromF32(glob, &.{ 1, COND }) };
    const t_lat: u32 = @intCast(xr.len / @as(usize, LATENT));
    const prep = try e.prepare(&sc, cond, @intCast(t_lat));
    defer a.free(prep.k);
    defer a.free(prep.v);
    const v = try e.ditForward(&sc, prep, try latentsFromRef(&sc, xr), 0.7);
    const c_dit = try cosineTo(try sc.transpose(v, &.{ 0, 2, 1 }), vr, e.s);

    const lat = try sc.keep(try e.sample(cond, t_lat, DEFAULT_STEPS, 42, null));
    const c_lat = try cosineTo(try sc.transpose(lat, &.{ 0, 2, 1 }), latr, e.s);

    const patches = try sc.keep(try e.decode(try latentsFromRef(&sc, latr)));
    const c_dec = try cosineTo(try sc.transpose(patches, &.{ 0, 2, 1 }), patchr, e.s);
    std.debug.print("[sa3 oracle] dit cos={d:.6} sample cos={d:.6} decode cos={d:.6}\n", .{ c_dit, c_lat, c_dec });
    try testing.expect(c_dit > 0.999);
    try testing.expect(c_lat > 0.99);
    try testing.expect(c_dec > 0.999);
}
