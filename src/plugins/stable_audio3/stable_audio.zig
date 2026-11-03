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
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const wav_mod = @import("../../sdk/media/wav.zig");
const sse = @import("../../sdk/media/progress.zig");

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

const ops = @import("../../mlx/ops.zig");
const gelu = @import("../../sdk/blocks/mlp/gelu.zig");
const Graph = ops.Graph;

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
    /// Load-time derived arrays: T5 `1 + w` norms, the fused weight-norm conv,
    /// the 1x1 conv matrices, DiT local vectors.
    owned: Graph,
    t5: [T5_LAYERS]T5Layer,
    t5_norm: A,
    t5_embed: A,
    dit: []DitLayer,
    pre_conv: A,
    post_conv: A,
    dec: [DEC_BLOCKS]DecBlock,
    dec_map: A,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        return loadOn(io, allocator, model_dir, mlx.mlx_default_gpu_stream_new());
    }

    /// `load` on stream `s` (the media table's `LoadCtx.stream`).
    pub fn loadOn(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;
        self.cfg = try parseConfig(io, allocator, model_dir);
        self.owned = .init(allocator, self.s);
        errdefer self.owned.deinit();

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
        self.owned.deinit();
        self.allocator.free(self.dit);
        self.w.deinit();
        self.t5w.deinit();
        self.tok.deinit();
        self.allocator.destroy(self);
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
        const o = try self.owned.contiguous(x);
        try ops.eval(o);
        return o;
    }

    fn resolve(self: *Engine) !void {
        var g = Graph.init(self.allocator, self.s);
        defer g.deinit();
        const w = &self.w;
        const t = &self.t5w;

        self.t5_embed = try get(t, "model.encoder.embed_tokens.weight", .{});
        self.t5_norm = try self.own(try g.addScalar(try g.astype(try get(t, "model.encoder.norm.weight", .{}), .float32), 1));
        for (&self.t5, 0..) |*l, i| {
            const n = struct {
                fn f(e: *Engine, c: *Graph, ww: *const Weights, li: usize, comptime name: []const u8) !A {
                    return e.own(try c.addScalar(try c.astype(try get(ww, "model.encoder.layers.{d}." ++ name ++ ".weight", .{li}), .float32), 1));
                }
            }.f;
            l.pre_attn = try n(self, &g, t, i, "pre_self_attn_layernorm");
            l.post_attn = try n(self, &g, t, i, "post_self_attn_layernorm");
            l.pre_ff = try n(self, &g, t, i, "pre_feedforward_layernorm");
            l.post_ff = try n(self, &g, t, i, "post_feedforward_layernorm");
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
            l.local = try self.own(try g.linear(try g.silu(b0), try get(w, p ++ "to_local_embed.2.weight", .{i}), try get(w, p ++ "to_local_embed.2.bias", .{i})));
        }
        const c = LATENT;
        self.pre_conv = try self.own(try g.reshape(try get(w, "model.model.preprocess_conv.weight", .{}), &.{ c, c }));
        self.post_conv = try self.own(try g.reshape(try get(w, "model.model.postprocess_conv.weight", .{}), &.{ c, c }));

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
        const wg = try get(w, d ++ "mapping.weight_g", .{});
        const v = try get(w, d ++ "mapping.weight_v", .{});
        const norm = try g.sqrt(try g.sum(try g.sum(try g.mul(v, v), 1, true), 2, true));
        self.dec_map = try self.own(try g.transpose(try g.div(try g.mul(wg, v), norm), &.{ 0, 2, 1 }));
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
    pub fn encodeText(self: *Engine, parent: *Graph, ids: []const i32) !A {
        var g = parent.scope();
        defer g.deinit();
        const n: c_int = @intCast(ids.len);
        const emb = try g.take(self.t5_embed, try g.fromSlice(i32, ids, &.{ 1, n }), 0);
        var x = try g.mulScalar(try g.astype(emb, .float32), @sqrt(@as(f32, @floatFromInt(COND))));
        for (&self.t5) |*l| {
            var h = try g.rmsNorm(x, l.pre_attn, T5_EPS);
            const q = try g.rope(try g.splitHeads(try g.linear(h, l.q, null), T5_HEADS, T5_HD), T5_HD, 10000, 0);
            const k = try g.rope(try g.splitHeads(try g.linear(h, l.k, null), T5_HEADS, T5_HD), T5_HD, 10000, 0);
            const v = try g.splitHeads(try g.linear(h, l.v, null), T5_HEADS, T5_HD);
            var scores = try g.mulScalar(try g.matmul(q, try g.transpose(k, &.{ 0, 1, 3, 2 })), 1.0 / @sqrt(@as(f32, @floatFromInt(T5_HD))));
            scores = try g.mulScalar(try g.tanh(try g.mulScalar(scores, 1.0 / T5_SOFTCAP)), T5_SOFTCAP);
            h = try g.linear(try g.mergeHeads(try g.matmul(try g.softmax(scores, -1, true), v)), l.o, null);
            x = try g.add(x, try g.rmsNorm(h, l.post_attn, T5_EPS));
            h = try g.rmsNorm(x, l.pre_ff, T5_EPS);
            h = try g.linear(try g.mul(try geluTanh(&g, try g.linear(h, l.gate, null)), try g.linear(h, l.up, null)), l.down, null);
            x = try g.add(x, try g.rmsNorm(h, l.post_ff, T5_EPS));
        }
        return parent.keep(try g.rmsNorm(x, self.t5_norm, T5_EPS));
    }

    pub const Cond = struct { cross: A, global: A };

    /// Cross-attention rows [1, 257, 768] (prompt rows, learned padding to 256,
    /// the seconds row) and the global condition [1, 768] (the seconds row).
    pub fn condition(self: *Engine, g: *Graph, ids: []const i32, seconds: f32) !Cond {
        const cp = "conditioner.conditioners.";
        var feats: [256]f32 = undefined;
        expoFeatures(std.math.clamp(seconds, 0, 384) / 384.0, &feats);
        const f = try g.fromSlice(f32, &feats, &.{ 1, 256 });
        const sec = try g.linear(f, try get(&self.w, cp ++ "seconds_total.embedder.embedding.1.weight", .{}), try get(&self.w, cp ++ "seconds_total.embedder.embedding.1.bias", .{}));
        const sec3 = try g.reshape(sec, &.{ 1, 1, COND });
        const pad_n: c_int = @intCast(MAX_TOKENS - ids.len);
        var parts: [3]A = undefined;
        var np: usize = 0;
        if (ids.len > 0) {
            parts[np] = try self.encodeText(g, ids);
            np += 1;
        }
        if (pad_n > 0) {
            const pe = try g.reshape(try get(&self.w, cp ++ "prompt.padding_embedding", .{}), &.{ 1, 1, COND });
            parts[np] = try g.broadcast(pe, &.{ 1, pad_n, COND });
            np += 1;
        }
        parts[np] = sec3;
        np += 1;
        return .{ .cross = try g.concat(parts[0..np], 1), .global = sec };
    }

    // ── DiT ─────────────────────────────────────────────────────────────

    /// Everything constant across sampler steps: cross-attention K/V per
    /// layer, the projected global condition, the local-add row mask.
    const Prep = struct { k: []A, v: []A, global: A, local_mask: A };

    /// The returned arrays live in `g`, evaluated; their intermediates do not.
    /// `k`/`v` are owned by the caller (free with `g.a`).
    fn prepare(self: *Engine, g: *Graph, cond: Cond, t_lat: c_int) !Prep {
        const w = &self.w;
        const n = self.dit.len;
        const k = try g.a.alloc(A, n);
        errdefer g.a.free(k);
        const v = try g.a.alloc(A, n);
        errdefer g.a.free(v);
        const outs = try g.a.alloc(A, 2 * n + 2);
        defer g.a.free(outs);
        var ps = g.scope();
        defer ps.deinit();
        const ctx = try ps.linear(try ps.silu(try ps.linear(cond.cross, try get(w, "model.model.to_cond_embed.0.weight", .{}), null)), try get(w, "model.model.to_cond_embed.2.weight", .{}), null);
        const e = self.cfg.embed;
        const hd = @divExact(e, self.cfg.heads);
        for (self.dit, 0..) |*l, i| {
            const kv = try ps.linear(ctx, l.cross_kv, null);
            k[i] = try g.keep(try ps.rmsNorm(try ps.splitHeads(try ps.sliceAxis(kv, 2, 0, e), self.cfg.heads, hd), l.cross_kn, 1e-6));
            v[i] = try g.keep(try ps.splitHeads(try ps.sliceAxis(kv, 2, e, 2 * e), self.cfg.heads, hd));
            outs[2 * i] = k[i];
            outs[2 * i + 1] = v[i];
        }
        const global = try g.keep(try ps.linear(try ps.silu(try ps.linear(cond.global, try get(w, "model.model.to_global_embed.0.weight", .{}), null)), try get(w, "model.model.to_global_embed.2.weight", .{}), null));
        const zeros = try ps.zeros(&.{ 1, self.cfg.memory, 1 }, .float32);
        const ones = try ps.ones(&.{ 1, t_lat, 1 }, .float32);
        const mask = try g.keep(try ps.concat(&.{ zeros, ones }, 1));
        outs[2 * n] = global;
        outs[2 * n + 1] = mask;
        try ops.evalMany(outs);
        return .{ .k = k, .v = v, .global = global, .local_mask = mask };
    }

    /// Velocity at noise level `t` for latents x [1, T, 256] (channels last).
    fn ditForward(self: *Engine, parent: *Graph, prep: Prep, x: A, t: f32) !A {
        var g = parent.scope();
        defer g.deinit();
        const w = &self.w;
        const e = self.cfg.embed;
        const hd = @divExact(e, self.cfg.heads);
        var feats: [256]f32 = undefined;
        expoFeatures(t, &feats);
        const tf = try g.fromSlice(f32, &feats, &.{ 1, 256 });
        const te = try g.linear(try g.silu(try g.linear(tf, try get(w, "model.model.to_timestep_embed.0.weight", .{}), try get(w, "model.model.to_timestep_embed.0.bias", .{}))), try get(w, "model.model.to_timestep_embed.2.weight", .{}), try get(w, "model.model.to_timestep_embed.2.bias", .{}));
        const ge = try g.add(prep.global, te);
        const tp = "model.model.transformer.";
        const cg = try g.linear(try g.silu(try g.linear(ge, try get(w, tp ++ "global_cond_embedder.0.weight", .{}), try get(w, tp ++ "global_cond_embedder.0.bias", .{}))), try get(w, tp ++ "global_cond_embedder.2.weight", .{}), try get(w, tp ++ "global_cond_embedder.2.bias", .{}));

        const xp = try g.add(x, try g.linear(x, self.pre_conv, null));
        const mem = try g.reshape(try get(w, tp ++ "memory_tokens", .{}), &.{ 1, self.cfg.memory, e });
        var h = try g.concat(&.{ mem, try g.linear(xp, try get(w, tp ++ "project_in.weight", .{}), null) }, 1);

        const attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (self.dit, 0..) |*l, i| {
            const ss = try g.reshape(try g.add(l.ssg, cg), &.{ 1, 6, e });
            const scale_self = try g.addScalar(try g.sliceAxis(ss, 1, 0, 1), 1);
            const shift_self = try g.sliceAxis(ss, 1, 1, 2);
            const gate_self = try g.sigmoid(try g.sub(try g.scalar(1, .float32), try g.sliceAxis(ss, 1, 2, 3)));
            const scale_ff = try g.addScalar(try g.sliceAxis(ss, 1, 3, 4), 1);
            const shift_ff = try g.sliceAxis(ss, 1, 4, 5);
            const gate_ff = try g.sigmoid(try g.sub(try g.scalar(1, .float32), try g.sliceAxis(ss, 1, 5, 6)));

            var a = try g.add(try g.mul(try g.rmsNorm(h, l.pre_norm, 1e-5), scale_self), shift_self);
            const qkv = try g.linear(a, l.qkv, null);
            const q = try g.rope(try g.rmsNorm(try g.splitHeads(try g.sliceAxis(qkv, 2, 0, e), self.cfg.heads, hd), l.self_qn, 1e-6), ROPE_DIMS, 10000, 0);
            const k = try g.rope(try g.rmsNorm(try g.splitHeads(try g.sliceAxis(qkv, 2, e, 2 * e), self.cfg.heads, hd), l.self_kn, 1e-6), ROPE_DIMS, 10000, 0);
            const v = try g.splitHeads(try g.sliceAxis(qkv, 2, 2 * e, 3 * e), self.cfg.heads, hd);
            a = try g.linear(try g.mergeHeads(try g.sdpa(q, k, v, attn_scale, .none)), l.self_out, null);
            h = try g.add(h, try g.mul(a, gate_self));

            const cq = try g.rmsNorm(try g.splitHeads(try g.linear(try g.rmsNorm(h, l.cross_norm, 1e-5), l.cross_q, null), self.cfg.heads, hd), l.cross_qn, 1e-6);
            h = try g.add(h, try g.linear(try g.mergeHeads(try g.sdpa(cq, prep.k[i], prep.v[i], attn_scale, .none)), l.cross_out, null));
            h = try g.add(h, try g.mul(prep.local_mask, l.local));

            a = try g.add(try g.mul(try g.rmsNorm(h, l.ff_norm, 1e-5), scale_ff), shift_ff);
            const ff = try g.linear(a, l.ff_in, l.ff_in_b);
            const inner = @divExact(ops.shape(ff)[2], 2);
            a = try g.linear(try g.mul(try g.sliceAxis(ff, 2, 0, inner), try g.silu(try g.sliceAxis(ff, 2, inner, 2 * inner))), l.ff_out, l.ff_out_b);
            h = try g.add(h, try g.mul(a, gate_ff));
        }
        const hl = ops.shape(h)[1];
        const o = try g.linear(try g.sliceAxis(h, 1, self.cfg.memory, hl), try get(w, tp ++ "project_out.weight", .{}), null);
        return parent.keep(try g.add(o, try g.linear(o, self.post_conv, null)));
    }

    /// Seeded ping-pong sampling → latents [1, T, 256]. Noise is drawn
    /// channels-first like the reference so a seed means the same sound.
    pub fn sample(self: *Engine, parent: *Graph, cond: Cond, t_lat: u32, steps: u32, seed: u64, progress: ?sse.Progress) !A {
        var g = parent.scope();
        defer g.deinit();
        const prep = try self.prepare(&g, cond, @intCast(t_lat));
        defer g.a.free(prep.k);
        defer g.a.free(prep.v);
        const sigmas = try g.a.alloc(f32, steps + 1);
        defer g.a.free(sigmas);
        schedule(sigmas);

        var x = try self.noise(&g, try g.call(mlx.mlx_random_key, .{seed}), t_lat);
        var key = try g.call(mlx.mlx_random_key, .{seed +% 1});
        for (0..steps) |i| {
            // The step's intermediates are released BEFORE the eval: a live
            // handle pins its buffer, so the whole step would stay resident.
            x = blk: {
                var step = g.scope();
                defer step.deinit();
                const v = try self.ditForward(&step, prep, x, sigmas[i]);
                const den = try step.sub(x, try step.mulScalar(v, sigmas[i]));
                const tn = sigmas[i + 1];
                if (i + 1 == steps or tn == 0) break :blk try g.keep(den);
                const keys = try g.splitKey(key);
                key = keys[0];
                break :blk try g.keep(try step.add(try step.mulScalar(den, 1 - tn), try step.mulScalar(try self.noise(&step, keys[1], t_lat), tn)));
            };
            try ops.eval(x);
            if (progress) |p| {
                p.emit("diffuse", @intCast(i + 1), steps);
                if (p.boundary()) return error.Cancelled;
            }
        }
        return parent.keep(x);
    }

    fn noise(_: *Engine, parent: *Graph, key: A, t_lat: u32) !A {
        var g = parent.scope();
        defer g.deinit();
        const z = try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, LATENT, @intCast(t_lat) }, 3, .float32, 0, 1, key });
        return parent.keep(try g.transpose(z, &.{ 0, 2, 1 }));
    }

    // ── SAME-S decoder ──────────────────────────────────────────────────

    fn dyt(parent: *Graph, x: A, p: DyT) !A {
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(try g.add(try g.mul(p.gamma, try g.tanh(try g.mul(p.alpha, x))), p.beta));
    }

    fn decBlock(parent: *Graph, x: A, b: *const DecBlock) !A {
        var g = parent.scope();
        defer g.deinit();
        const hd = @divExact(DEC_DIM, DEC_HEADS);
        const qkv = try g.linear(try dyt(&g, x, b.pre), b.qkv, null);
        var hh: [5]A = undefined;
        for (&hh, 0..) |*o, j| o.* = try g.splitHeads(try g.sliceAxis(qkv, 2, @as(c_int, @intCast(j)) * DEC_DIM, @as(c_int, @intCast(j + 1)) * DEC_DIM), DEC_HEADS, hd);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const q = try g.rope(try dyt(&g, hh[0], b.qn), ROPE_DIMS, 10000, 0);
        const k = try g.rope(try dyt(&g, hh[1], b.kn), ROPE_DIMS, 10000, 0);
        const qd = try g.rope(try dyt(&g, hh[3], b.qn), ROPE_DIMS, 10000, 0);
        const kd = try g.rope(try dyt(&g, hh[4], b.kn), ROPE_DIMS, 10000, 0);
        const att = try g.sub(try g.sdpa(q, k, hh[2], scale, .none), try g.sdpa(qd, kd, hh[2], scale, .none));
        const y = try g.add(x, try g.linear(try g.mergeHeads(att), b.out, null));
        const ff = try g.linear(try dyt(&g, y, b.ff_norm), b.glu, b.glu_b);
        const inner = @divExact(ops.shape(ff)[2], 2);
        return parent.keep(try g.add(y, try g.linear(try g.mul(try g.sliceAxis(ff, 2, 0, inner), try g.silu(try g.sliceAxis(ff, 2, inner, 2 * inner))), b.po, b.po_b)));
    }

    /// One decoder call: latents [1, n, 256] (n even) → patches [1, n·16, 512].
    fn decodeWindow(self: *Engine, parent: *Graph, lat: A) !A {
        var g = parent.scope();
        defer g.deinit();
        const w = &self.w;
        const dp = "pretransform.model.decoder.layers.";
        const n = ops.shape(lat)[1];
        const rows = n * DEC_SUB;
        var x = try g.mul(lat, try get(w, "pretransform.model.bottleneck.running_std", .{}));
        x = try g.linear(x, try get(w, dp ++ "1.weight", .{}), try get(w, dp ++ "1.bias", .{}));
        const nt = try g.broadcast(try g.reshape(try get(w, dp ++ "3.new_tokens", .{}), &.{ 1, 1, 1, DEC_DIM }), &.{ 1, n, DEC_STRIDE, DEC_DIM });
        x = try g.reshape(try g.concat(&.{ try g.reshape(x, &.{ 1, n, 1, DEC_DIM }), nt }, 2), &.{ @divExact(rows, DEC_CHUNK), DEC_CHUNK, DEC_DIM });
        for (self.dec[0..3]) |*b| x = try decBlock(&g, x, b);
        // Second half: shifted by half a chunk, edges padded with their own rows.
        x = try g.reshape(x, &.{ 1, rows, DEC_DIM });
        const half = @divExact(DEC_CHUNK, 2);
        x = try g.concat(&.{ try g.sliceAxis(x, 1, 0, half), x, try g.sliceAxis(x, 1, rows - half, rows) }, 1);
        x = try g.reshape(x, &.{ @divExact(rows + DEC_CHUNK, DEC_CHUNK), DEC_CHUNK, DEC_DIM });
        for (self.dec[3..]) |*b| x = try decBlock(&g, x, b);
        x = try g.sliceAxis(try g.reshape(x, &.{ 1, rows + DEC_CHUNK, DEC_DIM }), 1, half, half + rows);
        x = try g.sliceAxis(try g.reshape(x, &.{ n, DEC_SUB, DEC_DIM }), 1, 1, DEC_SUB);
        x = try g.reshape(x, &.{ 1, n * DEC_STRIDE, DEC_DIM });
        return parent.keep(try g.add(try g.conv1d(x, self.dec_map, 1, 1, 1, 1), try get(w, dp ++ "3.mapping.bias", .{})));
    }

    /// latents [1, T, 256] → patches [1, T·16, 512] through the reference's
    /// window plan (`decodePlan`).
    pub fn decode(self: *Engine, parent: *Graph, latents: A) !A {
        const t_lat: u32 = @intCast(ops.shape(latents)[1]);
        var g = parent.scope();
        defer g.deinit();
        const plan = try decodePlan(g.a, t_lat);
        defer g.a.free(plan);
        const pieces = try g.a.alloc(A, plan.len);
        defer g.a.free(pieces);
        for (plan, pieces) |win, *piece| {
            piece.* = blk: {
                var ws = g.scope();
                defer ws.deinit();
                var lat = try ws.sliceAxis(latents, 1, @intCast(win.start), @intCast(win.start + win.len));
                if (win.pad_last) lat = try ws.concat(&.{ lat, try ws.sliceAxis(lat, 1, @intCast(win.len - 1), @intCast(win.len)) }, 1);
                const p = try self.decodeWindow(&ws, lat);
                break :blk try g.keep(try ws.sliceAxis(p, 1, @intCast(win.keep_from * 16), @intCast((win.keep_from + win.keep_len) * 16)));
            };
            try ops.eval(piece.*);
        }
        return parent.keep(try g.concat(pieces, 1));
    }

    /// prompt → 44.1 kHz stereo PCM16 WAV (owned).
    pub fn generateWav(self: *Engine, allocator: std.mem.Allocator, req: Request, progress: ?sse.Progress) ![]u8 {
        const secs = trainedSeconds(req.seconds);
        const t_lat = latentCount(secs);
        const ids = try self.tokenize(allocator, req.prompt);
        defer allocator.free(ids);
        log.info("[sa3] {d:.2}s -> {d} latents, {d} prompt tokens, steps={d}, seed={d}\n", .{ req.seconds, t_lat, ids.len, req.steps, req.seed });

        var g = Graph.init(self.allocator, self.s);
        defer g.deinit();
        // The text encoder's intermediates go before sampling starts.
        const cond = blk: {
            var cs = g.scope();
            defer cs.deinit();
            const c = try self.condition(&cs, ids, secs);
            break :blk Cond{ .cross = try g.keep(c.cross), .global = try g.keep(c.global) };
        };
        try ops.evalMany(&.{ cond.cross, cond.global });
        if (progress) |p| p.emit("encode", 1, 1);

        const lat = try self.sample(&g, cond, t_lat, req.steps, req.seed, progress);
        if (progress) |p| p.emit("decode", 0, 1);
        const patches = try self.decode(&g, lat);
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
    var g = Graph.init(a, s);
    defer g.deinit();
    const l = ops.shape(patches)[1];
    const st = try g.reshape(try g.transpose(try g.reshape(patches, &.{ l, 2, DEC_PATCH }), &.{ 0, 2, 1 }), &.{ l * DEC_PATCH, 2 });
    const cont = try g.contiguous(st);
    try ops.eval(cont);
    const data = mlx.mlx_array_data_float32(cont) orelse return error.NoData;
    const n = @min(samples * 2, mlx.mlx_array_size(cont));
    return a.dupe(f32, data[0..n]);
}

/// tanh-approximated GELU (Gemma's `gelu_pytorch_tanh`), its constants in the operand's dtype.
fn geluTanh(parent: *Graph, x: A) !A {
    return gelu.geluTanh(parent, x, ops.dtype(x), gelu.SQRT_2_PI_LOW);
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

// ── tests ──
// Hermetic first, then env-gated oracles (SA3_TEST_MODEL + SA3_*, fed by tests/dump_stable_audio_fixtures.py).

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
    var g = Graph.init(a, s);
    defer g.deinit();
    const pcm = try unpatch(a, s, try g.fromSlice(f32, &data, &.{ 1, l, 512 }), 300);
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
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const cont = try g.contiguous(try g.astype(arr, .float32));
    try ops.eval(cont);
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

    var g = Graph.init(a, e.s);
    defer g.deinit();
    const c_t5 = try cosineTo(try e.encodeText(&g, ids), t5, e.s);
    const cond = try e.condition(&g, ids, 5.0);
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
    var g = Graph.init(a, e.s);
    defer g.deinit();
    const cond = try e.condition(&g, ids, 30);
    try ops.eval(cond.cross);
    var base: usize = 0;
    _ = mlx.mlx_get_active_memory(&base);
    _ = mlx.mlx_reset_peak_memory();
    const lat = try e.sample(&g, cond, latentCount(30), DEFAULT_STEPS, 1, null);
    _ = try e.decode(&g, lat);
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("[sa3] 30 s working set {d} MB\n", .{(peak -| base) >> 20});
    try testing.expect(peak -| base < 1 << 30);
}

/// The reference's channels-first [1, 256, T] as our [1, T, 256].
fn latentsFromRef(g: *Graph, data: []const f32) !A {
    const t: c_int = @intCast(data.len / @as(usize, LATENT));
    return g.transpose(try g.fromSlice(f32, data, &.{ 1, LATENT, t }), &.{ 0, 2, 1 });
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

    var g = Graph.init(a, e.s);
    defer g.deinit();
    const cond = Engine.Cond{ .cross = try g.fromSlice(f32, cross, &.{ 1, 257, COND }), .global = try g.fromSlice(f32, glob, &.{ 1, COND }) };
    const t_lat: u32 = @intCast(xr.len / @as(usize, LATENT));
    const prep = try e.prepare(&g, cond, @intCast(t_lat));
    defer a.free(prep.k);
    defer a.free(prep.v);
    const v = try e.ditForward(&g, prep, try latentsFromRef(&g, xr), 0.7);
    const c_dit = try cosineTo(try g.transpose(v, &.{ 0, 2, 1 }), vr, e.s);

    const lat = try e.sample(&g, cond, t_lat, DEFAULT_STEPS, 42, null);
    const c_lat = try cosineTo(try g.transpose(lat, &.{ 0, 2, 1 }), latr, e.s);

    const patches = try e.decode(&g, try latentsFromRef(&g, latr));
    const c_dec = try cosineTo(try g.transpose(patches, &.{ 0, 2, 1 }), patchr, e.s);
    std.debug.print("[sa3 oracle] dit cos={d:.6} sample cos={d:.6} decode cos={d:.6}\n", .{ c_dit, c_lat, c_dec });
    try testing.expect(c_dit > 0.999);
    try testing.expect(c_lat > 0.99);
    try testing.expect(c_dec > 0.999);
}

// The Scope forwards the Graph port must equal bit for bit (they go when Scope does).

const Scope = @import("../../mlx/scope.zig").Scope;

const legacy = struct {
    const Derived = struct { t5_norm: A, t5: [T5_LAYERS][4]A, local: [8]A, pre_conv: A, post_conv: A, dec_map: A };

    fn own(e: *Engine, owned: *std.ArrayList(A), x: A) !A {
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_contiguous(&o, x, false, e.s));
        try mlx.check(mlx.mlx_array_eval(o));
        try owned.append(e.allocator, o);
        return o;
    }

    fn resolve(e: *Engine, owned: *std.ArrayList(A), out: *Derived) !void {
        const get = Engine.get;
        var sc = Scope.init(e.allocator, e.s);
        defer sc.deinit();
        const w = &e.w;
        const t = &e.t5w;
        out.t5_norm = try own(e, owned, try sc.addS(try sc.astype(try get(t, "model.encoder.norm.weight", .{}), .float32), 1));
        for (&out.t5, 0..) |*l, i| {
            const n = struct {
                fn f(ee: *Engine, ow: *std.ArrayList(A), c: *Scope, ww: *const Weights, li: usize, comptime name: []const u8) !A {
                    return own(ee, ow, try c.addS(try c.astype(try Engine.get(ww, "model.encoder.layers.{d}." ++ name ++ ".weight", .{li}), .float32), 1));
                }
            }.f;
            l[0] = try n(e, owned, &sc, t, i, "pre_self_attn_layernorm");
            l[1] = try n(e, owned, &sc, t, i, "post_self_attn_layernorm");
            l[2] = try n(e, owned, &sc, t, i, "pre_feedforward_layernorm");
            l[3] = try n(e, owned, &sc, t, i, "post_feedforward_layernorm");
        }
        const p = "model.model.transformer.layers.{d}.";
        for (e.dit, 0..) |_, i| {
            const b0 = try get(w, p ++ "to_local_embed.0.bias", .{i});
            out.local[i] = try own(e, owned, try sc.lin(try sc.silu(b0), try get(w, p ++ "to_local_embed.2.weight", .{i}), try get(w, p ++ "to_local_embed.2.bias", .{i})));
        }
        const c = LATENT;
        out.pre_conv = try own(e, owned, try sc.reshape(try get(w, "model.model.preprocess_conv.weight", .{}), &.{ c, c }));
        out.post_conv = try own(e, owned, try sc.reshape(try get(w, "model.model.postprocess_conv.weight", .{}), &.{ c, c }));
        const d = "pretransform.model.decoder.layers.3.";
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
        out.dec_map = try own(e, owned, try sc.transpose(fused, &.{ 0, 2, 1 }));
    }

    fn encodeText(self: *Engine, sc: *Scope, ids: []const i32) !A {
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
            h = try sc.lin(try sc.mul(try geluTanhScope(sc, try sc.lin(h, l.gate, null)), try sc.lin(h, l.up, null)), l.down, null);
            x = try sc.add(x, try sc.rms(h, l.post_ff, T5_EPS));
        }
        return sc.rms(x, self.t5_norm, T5_EPS);
    }

    fn condition(self: *Engine, sc: *Scope, ids: []const i32, seconds: f32) !Engine.Cond {
        const get = Engine.get;
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
            parts[np] = try encodeText(self, sc, ids);
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

    fn prepare(self: *Engine, sc: *Scope, cond: Engine.Cond, t_lat: c_int) !Engine.Prep {
        const get = Engine.get;
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

    fn ditForward(self: *Engine, sc: *Scope, prep: Engine.Prep, x: A, t: f32) !A {
        const get = Engine.get;
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

    fn sample(self: *Engine, cond: Engine.Cond, t_lat: u32, steps: u32, seed: u64) !A {
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const prep = try prepare(self, &sc, cond, @intCast(t_lat));
        defer self.allocator.free(prep.k);
        defer self.allocator.free(prep.v);
        const sigmas = try self.allocator.alloc(f32, steps + 1);
        defer self.allocator.free(sigmas);
        schedule(sigmas);
        var k_init = mlx.mlx_array_new();
        var x = try noise(&sc, try sc.res(mlx.mlx_random_key(&k_init, seed), &k_init), t_lat);
        var k0 = mlx.mlx_array_new();
        var key = try sc.res(mlx.mlx_random_key(&k0, seed +% 1), &k0);
        for (0..steps) |i| {
            const next = blk: {
                var step = Scope.init(self.allocator, self.s);
                defer step.deinit();
                const v = try ditForward(self, &step, prep, x, sigmas[i]);
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
                break :blk step.out(try step.add(try step.mulS(den, 1 - tn), try step.mulS(try noise(&step, sub, t_lat), tn)));
            };
            x = try sc.keep(next);
            try mlx.check(mlx.mlx_array_eval(x));
        }
        return sc.out(x);
    }

    fn noise(sc: *Scope, key: A, t_lat: u32) !A {
        var n = mlx.mlx_array_new();
        const z = try sc.res(mlx.mlx_random_normal(&n, &[_]c_int{ 1, LATENT, @intCast(t_lat) }, 3, .float32, 0, 1, key, sc.s), &n);
        return sc.transpose(z, &.{ 0, 2, 1 });
    }

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

    fn decodeWindow(self: *Engine, sc: *Scope, lat: A) !A {
        const get = Engine.get;
        const w = &self.w;
        const dp = "pretransform.model.decoder.layers.";
        const n = mlx.getShape(lat)[1];
        const rows = n * DEC_SUB;
        var x = try sc.mul(lat, try get(w, "pretransform.model.bottleneck.running_std", .{}));
        x = try sc.lin(x, try get(w, dp ++ "1.weight", .{}), try get(w, dp ++ "1.bias", .{}));
        const nt = try sc.broadcast(try sc.reshape(try get(w, dp ++ "3.new_tokens", .{}), &.{ 1, 1, 1, DEC_DIM }), &.{ 1, n, DEC_STRIDE, DEC_DIM });
        x = try sc.reshape(try sc.concat(&.{ try sc.reshape(x, &.{ 1, n, 1, DEC_DIM }), nt }, 2), &.{ @divExact(rows, DEC_CHUNK), DEC_CHUNK, DEC_DIM });
        for (self.dec[0..3]) |*b| x = try decBlock(sc, x, b);
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

    fn decode(self: *Engine, latents: A) !A {
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
                const p = try decodeWindow(self, &ws, lat);
                break :blk ws.out(try ws.slice(p, 1, @intCast(win.keep_from * 16), @intCast((win.keep_from + win.keep_len) * 16)));
            };
            piece.* = try sc.keep(kept);
            try mlx.check(mlx.mlx_array_eval(piece.*));
        }
        return sc.out(try sc.concat(pieces, 1));
    }

    fn geluTanhScope(sc: *Scope, x: A) !A {
        const x3 = try sc.mul(try sc.mul(x, x), x);
        const inner = try sc.mulS(try sc.add(x, try sc.mulS(x3, 0.044715)), @sqrt(2.0 / std.math.pi));
        return sc.mul(try sc.mulS(x, 0.5), try sc.addS(try sc.tanh(inner), 1));
    }
};

const expectBitEqual = ops.expectBitEqual;

/// A tiny random-weight engine (one random layer shared by every layer of a stack) on the CPU stream.
const Fixture = struct {
    e: *Engine,
    src: Graph,
    rng: u64 = 1,

    fn init(a: std.mem.Allocator) !Fixture {
        const s = mlx.mlx_default_cpu_stream_new();
        const e = try a.create(Engine);
        e.allocator = a;
        e.s = s;
        e.cfg = .{ .embed = 64, .depth = 2, .heads = 2, .memory = 4 };
        e.w = Weights.init(a);
        e.t5w = Weights.init(a);
        e.owned = .init(a, s);
        e.dit = try a.alloc(DitLayer, e.cfg.depth);
        var fx = Fixture{ .e = e, .src = .init(a, s) };
        try fx.fill();
        try e.resolve();
        return fx;
    }

    fn deinit(fx: *Fixture) void {
        const e = fx.e;
        e.owned.deinit();
        e.allocator.free(e.dit);
        e.w.deinit();
        e.t5w.deinit();
        fx.src.deinit();
        e.allocator.destroy(e);
    }

    fn rand(fx: *Fixture, sh: []const c_int, dt: mlx.mlx_dtype, scale: f32) !A {
        fx.rng += 1;
        const key = try fx.src.call(mlx.mlx_random_key, .{fx.rng});
        const x = try fx.src.call(mlx.mlx_random_normal, .{ sh.ptr, sh.len, mlx.mlx_dtype.float32, 0, scale, key });
        const y = try fx.src.astype(x, dt);
        try ops.eval(y);
        return y;
    }

    fn put(fx: *Fixture, w: *Weights, key: []const u8, x: A) !void {
        try w.map.put(try fx.e.allocator.dupe(u8, key), fx.src.escape(x));
    }

    fn putN(fx: *Fixture, w: *Weights, comptime fmt: []const u8, n: usize, x: A) !void {
        for (0..n) |i| {
            var buf: [160]u8 = undefined;
            try fx.put(w, try std.fmt.bufPrint(&buf, fmt, .{i}), x);
        }
    }

    fn fill(fx: *Fixture) !void {
        const e = fx.e;
        const E = e.cfg.embed;
        const hd = @divExact(E, e.cfg.heads);
        const F = 16;
        const bf = mlx.mlx_dtype.bfloat16;
        const f = mlx.mlx_dtype.float32;
        // T5Gemma (bf16, the published dtype).
        const t = &e.t5w;
        try fx.put(t, "model.encoder.embed_tokens.weight", try fx.rand(&.{ 32, COND }, bf, 1));
        try fx.put(t, "model.encoder.norm.weight", try fx.rand(&.{COND}, bf, 0.1));
        const tl = "model.encoder.layers.{d}.";
        inline for (.{ "pre_self_attn_layernorm", "post_self_attn_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" }) |nm|
            try fx.putN(t, tl ++ nm ++ ".weight", T5_LAYERS, try fx.rand(&.{COND}, bf, 0.1));
        inline for (.{ "q", "k", "v", "o" }) |nm|
            try fx.putN(t, tl ++ "self_attn." ++ nm ++ "_proj.weight", T5_LAYERS, try fx.rand(&.{ COND, COND }, bf, 0.04));
        try fx.putN(t, tl ++ "mlp.gate_proj.weight", T5_LAYERS, try fx.rand(&.{ F, COND }, bf, 0.04));
        try fx.putN(t, tl ++ "mlp.up_proj.weight", T5_LAYERS, try fx.rand(&.{ F, COND }, bf, 0.04));
        try fx.putN(t, tl ++ "mlp.down_proj.weight", T5_LAYERS, try fx.rand(&.{ COND, F }, bf, 0.2));
        // Conditioner + DiT (f32).
        const w = &e.w;
        const cp = "conditioner.conditioners.";
        try fx.put(w, cp ++ "seconds_total.embedder.embedding.1.weight", try fx.rand(&.{ COND, 256 }, f, 0.06));
        try fx.put(w, cp ++ "seconds_total.embedder.embedding.1.bias", try fx.rand(&.{COND}, f, 0.1));
        try fx.put(w, cp ++ "prompt.padding_embedding", try fx.rand(&.{COND}, f, 1));
        const mm = "model.model.";
        try fx.put(w, mm ++ "to_cond_embed.0.weight", try fx.rand(&.{ E, COND }, f, 0.04));
        try fx.put(w, mm ++ "to_cond_embed.2.weight", try fx.rand(&.{ E, E }, f, 0.1));
        try fx.put(w, mm ++ "to_global_embed.0.weight", try fx.rand(&.{ E, COND }, f, 0.04));
        try fx.put(w, mm ++ "to_global_embed.2.weight", try fx.rand(&.{ E, E }, f, 0.1));
        try fx.put(w, mm ++ "to_timestep_embed.0.weight", try fx.rand(&.{ E, 256 }, f, 0.06));
        try fx.put(w, mm ++ "to_timestep_embed.0.bias", try fx.rand(&.{E}, f, 0.1));
        try fx.put(w, mm ++ "to_timestep_embed.2.weight", try fx.rand(&.{ E, E }, f, 0.1));
        try fx.put(w, mm ++ "to_timestep_embed.2.bias", try fx.rand(&.{E}, f, 0.1));
        try fx.put(w, mm ++ "preprocess_conv.weight", try fx.rand(&.{ LATENT, LATENT, 1 }, f, 0.06));
        try fx.put(w, mm ++ "postprocess_conv.weight", try fx.rand(&.{ LATENT, LATENT, 1 }, f, 0.06));
        const tp = mm ++ "transformer.";
        try fx.put(w, tp ++ "global_cond_embedder.0.weight", try fx.rand(&.{ E, E }, f, 0.1));
        try fx.put(w, tp ++ "global_cond_embedder.0.bias", try fx.rand(&.{E}, f, 0.1));
        try fx.put(w, tp ++ "global_cond_embedder.2.weight", try fx.rand(&.{ 6 * E, E }, f, 0.1));
        try fx.put(w, tp ++ "global_cond_embedder.2.bias", try fx.rand(&.{6 * E}, f, 0.1));
        try fx.put(w, tp ++ "memory_tokens", try fx.rand(&.{ e.cfg.memory, E }, f, 1));
        try fx.put(w, tp ++ "project_in.weight", try fx.rand(&.{ E, LATENT }, f, 0.06));
        try fx.put(w, tp ++ "project_out.weight", try fx.rand(&.{ LATENT, E }, f, 0.1));
        const n = e.cfg.depth;
        const dl = tp ++ "layers.{d}.";
        inline for (.{ "pre_norm.gamma", "cross_attend_norm.gamma", "ff_norm.gamma" }) |nm| try fx.putN(w, dl ++ nm, n, try fx.rand(&.{E}, f, 1));
        inline for (.{ "self_attn.q_norm.gamma", "self_attn.k_norm.gamma", "cross_attn.q_norm.gamma", "cross_attn.k_norm.gamma" }) |nm| try fx.putN(w, dl ++ nm, n, try fx.rand(&.{hd}, f, 1));
        try fx.putN(w, dl ++ "self_attn.to_qkv.weight", n, try fx.rand(&.{ 3 * E, E }, f, 0.1));
        try fx.putN(w, dl ++ "self_attn.to_out.weight", n, try fx.rand(&.{ E, E }, f, 0.1));
        try fx.putN(w, dl ++ "cross_attn.to_q.weight", n, try fx.rand(&.{ E, E }, f, 0.1));
        try fx.putN(w, dl ++ "cross_attn.to_kv.weight", n, try fx.rand(&.{ 2 * E, E }, f, 0.1));
        try fx.putN(w, dl ++ "cross_attn.to_out.weight", n, try fx.rand(&.{ E, E }, f, 0.1));
        try fx.putN(w, dl ++ "ff.ff.0.proj.weight", n, try fx.rand(&.{ 2 * F, E }, f, 0.1));
        try fx.putN(w, dl ++ "ff.ff.0.proj.bias", n, try fx.rand(&.{2 * F}, f, 0.1));
        try fx.putN(w, dl ++ "ff.ff.2.weight", n, try fx.rand(&.{ E, F }, f, 0.2));
        try fx.putN(w, dl ++ "ff.ff.2.bias", n, try fx.rand(&.{E}, f, 0.1));
        try fx.putN(w, dl ++ "to_scale_shift_gate", n, try fx.rand(&.{6 * E}, f, 0.1));
        try fx.putN(w, dl ++ "to_local_embed.0.bias", n, try fx.rand(&.{8}, f, 1));
        try fx.putN(w, dl ++ "to_local_embed.2.weight", n, try fx.rand(&.{ E, 8 }, f, 0.3));
        try fx.putN(w, dl ++ "to_local_embed.2.bias", n, try fx.rand(&.{E}, f, 0.1));
        // SAME-S decoder (f32).
        try fx.put(w, "pretransform.model.bottleneck.running_std", try fx.rand(&.{1}, f, 1));
        const dp = "pretransform.model.decoder.layers.";
        try fx.put(w, dp ++ "1.weight", try fx.rand(&.{ DEC_DIM, LATENT }, f, 0.06));
        try fx.put(w, dp ++ "1.bias", try fx.rand(&.{DEC_DIM}, f, 0.1));
        try fx.put(w, dp ++ "3.new_tokens", try fx.rand(&.{DEC_DIM}, f, 1));
        try fx.put(w, dp ++ "3.mapping.weight_g", try fx.rand(&.{ 512, 1, 1 }, f, 1));
        try fx.put(w, dp ++ "3.mapping.weight_v", try fx.rand(&.{ 512, DEC_DIM, 3 }, f, 1));
        try fx.put(w, dp ++ "3.mapping.bias", try fx.rand(&.{512}, f, 0.1));
        const bl = dp ++ "3.transformers.{d}.";
        inline for (.{ "pre_norm", "ff_norm" }) |nm| {
            try fx.putN(w, bl ++ nm ++ ".alpha", DEC_BLOCKS, try fx.rand(&.{1}, f, 1));
            try fx.putN(w, bl ++ nm ++ ".gamma", DEC_BLOCKS, try fx.rand(&.{DEC_DIM}, f, 1));
            try fx.putN(w, bl ++ nm ++ ".beta", DEC_BLOCKS, try fx.rand(&.{DEC_DIM}, f, 0.1));
        }
        inline for (.{ "self_attn.q_norm", "self_attn.k_norm" }) |nm| {
            try fx.putN(w, bl ++ nm ++ ".alpha", DEC_BLOCKS, try fx.rand(&.{1}, f, 1));
            try fx.putN(w, bl ++ nm ++ ".gamma", DEC_BLOCKS, try fx.rand(&.{@divExact(DEC_DIM, DEC_HEADS)}, f, 1));
            try fx.putN(w, bl ++ nm ++ ".beta", DEC_BLOCKS, try fx.rand(&.{@divExact(DEC_DIM, DEC_HEADS)}, f, 0.1));
        }
        try fx.putN(w, bl ++ "self_attn.to_qkv.weight", DEC_BLOCKS, try fx.rand(&.{ 5 * DEC_DIM, DEC_DIM }, f, 0.04));
        try fx.putN(w, bl ++ "self_attn.to_out.weight", DEC_BLOCKS, try fx.rand(&.{ DEC_DIM, DEC_DIM }, f, 0.04));
        try fx.putN(w, bl ++ "ff.ff.0.proj.weight", DEC_BLOCKS, try fx.rand(&.{ 2 * F, DEC_DIM }, f, 0.04));
        try fx.putN(w, bl ++ "ff.ff.0.proj.bias", DEC_BLOCKS, try fx.rand(&.{2 * F}, f, 0.1));
        try fx.putN(w, bl ++ "ff.ff.2.weight", DEC_BLOCKS, try fx.rand(&.{ DEC_DIM, F }, f, 0.2));
        try fx.putN(w, bl ++ "ff.ff.2.bias", DEC_BLOCKS, try fx.rand(&.{DEC_DIM}, f, 0.1));
    }
};

test "sa3 Graph port: load-time derivations and every forward are bit-identical to the Scope code" {
    const a = testing.allocator;
    var fx = try Fixture.init(a);
    defer fx.deinit();
    const e = fx.e;

    var owned: std.ArrayList(A) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(a);
    }
    var old: legacy.Derived = undefined;
    try legacy.resolve(e, &owned, &old);
    try expectBitEqual(e.t5_norm, old.t5_norm);
    for (e.t5, old.t5) |l, o| for ([_]A{ l.pre_attn, l.post_attn, l.pre_ff, l.post_ff }, o) |x, y| try expectBitEqual(x, y);
    for (e.dit, old.local[0..e.dit.len]) |l, o| try expectBitEqual(l.local, o);
    try expectBitEqual(e.pre_conv, old.pre_conv);
    try expectBitEqual(e.post_conv, old.post_conv);
    try expectBitEqual(e.dec_map, old.dec_map);

    var g = Graph.init(a, e.s);
    defer g.deinit();
    var sc = Scope.init(a, e.s);
    defer sc.deinit();
    const ids = [_]i32{ 5, 17, 3, 9 };
    try expectBitEqual(try e.encodeText(&g, &ids), try legacy.encodeText(e, &sc, &ids));
    const cond = try e.condition(&g, &ids, 5.0);
    const cond_old = try legacy.condition(e, &sc, &ids, 5.0);
    try expectBitEqual(cond.cross, cond_old.cross);
    try expectBitEqual(cond.global, cond_old.global);

    const t_lat: c_int = 6;
    const prep = try e.prepare(&g, cond, t_lat);
    defer a.free(prep.k);
    defer a.free(prep.v);
    const prep_old = try legacy.prepare(e, &sc, cond, t_lat);
    defer a.free(prep_old.k);
    defer a.free(prep_old.v);
    for (prep.k, prep_old.k, prep.v, prep_old.v) |k, ko, v, vo| {
        try expectBitEqual(k, ko);
        try expectBitEqual(v, vo);
    }
    try expectBitEqual(prep.global, prep_old.global);
    try expectBitEqual(prep.local_mask, prep_old.local_mask);
    const x = try fx.rand(&.{ 1, t_lat, LATENT }, .float32, 1);
    try expectBitEqual(try e.ditForward(&g, prep, x, 0.7), try legacy.ditForward(e, &sc, prep, x, 0.7));

    const lat = try e.sample(&g, cond, @intCast(t_lat), 3, 11, null);
    const lat_old = try sc.keep(try legacy.sample(e, cond, @intCast(t_lat), 3, 11));
    try expectBitEqual(lat, lat_old);

    // 3 latents pad the last one; 14 run the overlapped multi-window plan.
    for ([_]c_int{ 3, 14 }) |n| {
        const z = try fx.rand(&.{ 1, n, LATENT }, .float32, 1);
        try expectBitEqual(try e.decode(&g, z), try sc.keep(try legacy.decode(e, z)));
    }
}
