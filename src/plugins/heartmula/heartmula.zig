//! HeartMuLa-oss-3B (lyrics + style tags → 48 kHz stereo song, `.audio`, `heartmula`), a port of heartlib.
//!
//!   tags + lyrics → text rows (+ one reference-audio slot) → Llama-3.2-3B backbone (batch 2 for CFG) → codebook 0
//!   → a 3-layer depth decoder (head_dim 384, KV reset per frame) → codebooks 1..7, one 80 ms frame at a time
//!   → HeartCodec: RVQ dequant → a two-stack AdaLN flow DiT, 10 Euler steps in 29.76 s windows continued in context
//!   → a Snake-free PReLU conv decoder per stereo band → crossfaded WAV.
//!
//! Pack: scripts/convert_heartmula_weights.py (mlx-audio's CSM names; q/k rows stay torchtune-INTERLEAVED, so every
//! RoPE here is `traditional`). LM bf16 or affine; the codec DiT bf16 with f32 solver state, the scalar decoder f32.
//! Parity: env-gated `HEARTMULA_*` oracles fed by tests/dump_heartmula_fixtures.py.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const tok_load = @import("../../sdk/tokenizer/load.zig");
const wav_mod = @import("../../sdk/media/wav.zig");
const sse = @import("../../sdk/media/progress.zig");
const params = @import("../../sdk/quant/params.zig");
const embedding = @import("../../sdk/blocks/embedding.zig");
const rope_scaled = @import("../../sdk/blocks/rope/scaled.zig");
const ops = @import("../../mlx/ops.zig");
const scope_mod = @import("../../mlx/scope.zig");
const Graph = ops.Graph;
const Linear = scope_mod.Linear;
const getW = scope_mod.getW;
const linear = scope_mod.linear;

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const Weights = model_mod.Weights;

pub const FRAME_MS: u32 = 80;
pub const MIN_SECONDS: u32 = 5;
pub const MAX_SECONDS: u32 = 360;
/// The model ends its own songs: an omitted length is the longest cap.
pub const DEFAULT_SECONDS: u32 = MAX_SECONDS;
pub const DEFAULT_CFG: f32 = 3.0;
pub const DEFAULT_TOP_K: u32 = 50;
pub const DEFAULT_STEPS: u32 = 10;
const CODEC_GUIDANCE: f32 = 1.25;
/// heartlib's decode window (duration 29.76 s): codes, latents and samples per window, the hop and the overlap,
/// as its own `int(duration * rate)` truncations give them.
const WIN_CODES: usize = 372;
const HOP_CODES: usize = 320;
const OVLP_CODES: usize = WIN_CODES - HOP_CODES;
const WIN_LATENTS: usize = 2 * WIN_CODES;
const CTX_LATENTS: usize = 2 * OVLP_CODES;
const WIN_SAMPLES: usize = 1428480;
const HOP_SAMPLES: usize = 1228800;
const OVLP_SAMPLES: usize = WIN_SAMPLES - HOP_SAMPLES;
/// Frames between host reads of the codes (the EOS check); the frames between ride the GPU unsynced.
const EOS_POLL: usize = 8;

// ════════════════════════════════════════════════════════════════════════
// Config
// ════════════════════════════════════════════════════════════════════════

const RopeFile = struct { factor: f64, low_freq_factor: f64 = 1, high_freq_factor: f64 = 4, original_max_position_embeddings: f64 = 8192 };

const StackFile = struct {
    num_hidden_layers: u32,
    hidden_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f64 = 500000,
    rope_scaling: ?RopeFile = null,
};

const CodecFile = struct {
    dim: u32,
    codebook_size: u32,
    codebook_dim: u32,
    num_quantizers: u32,
    attention_head_dim: u32,
    in_channels: u32,
    num_attention_heads: u32,
    num_layers: u32,
    num_layers_2: u32,
    out_channels: u32,
    sample_rate: u32,
    num_samples: u32,
    upsample_factors: []const u32,
    upsample_kernel_sizes: []const u32,
    latent_hidden_dim: u32,
    delay_kernel_size: u32,
    init_channel: u32,
    res_kernel_size: u32,
    default_kernel_size: u32,
};

pub const Cfg = struct {
    audio_num_codebooks: u32,
    audio_vocab_size: u32,
    text_vocab_size: u32,
    backbone: StackFile,
    decoder: StackFile,
    codec: CodecFile,
    text_bos_id: u32,
    text_eos_id: u32,
    audio_eos_id: u32,
};

fn stackOk(s: StackFile) bool {
    return s.num_hidden_layers > 0 and s.num_key_value_heads > 0 and s.num_attention_heads % s.num_key_value_heads == 0 and
        s.head_dim * s.num_attention_heads == s.hidden_size and s.head_dim % 2 == 0 and s.max_position_embeddings > 0;
}

/// A config this engine cannot serve is a NAMED load error: the decode windows assume 12.5 Hz codes and 25 Hz
/// latents split into two bands, and the causal transposed convs a kernel of twice their stride.
pub fn parseConfig(a: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Cfg) {
    const p = std.json.parseFromSlice(Cfg, a, bytes, .{ .ignore_unknown_fields = true }) catch return error.HeartMulaConfigInvalid;
    errdefer p.deinit();
    const c = p.value;
    const k = c.codec;
    var hop: u64 = k.num_samples;
    for (k.upsample_factors) |f| hop *= f;
    const ok = stackOk(c.backbone) and stackOk(c.decoder) and c.backbone.hidden_size == c.decoder.hidden_size and
        c.audio_num_codebooks >= 2 and k.num_quantizers == c.audio_num_codebooks and c.audio_eos_id < c.audio_vocab_size and
        c.text_bos_id < c.text_vocab_size and c.text_eos_id < c.text_vocab_size and k.codebook_size > 0 and
        k.in_channels == 2 * k.out_channels + k.dim and k.out_channels == 2 * k.latent_hidden_dim and
        k.upsample_factors.len > 0 and k.upsample_factors.len == k.upsample_kernel_sizes.len and
        hop * 25 == k.sample_rate and k.attention_head_dim % 2 == 0 and k.num_attention_heads > 0;
    if (!ok) return error.HeartMulaConfigUnsupported;
    for (k.upsample_factors, k.upsample_kernel_sizes) |f, ks| if (ks != 2 * f) return error.HeartMulaConfigUnsupported;
    return p;
}

// ════════════════════════════════════════════════════════════════════════
// Prompt byte contract (pipelines/music_generation.py `preprocess`)
// ════════════════════════════════════════════════════════════════════════

/// Python `str.lower()` over the scripts with a simple case mapping (ASCII, Latin-1, Latin Extended-A, Greek,
/// Cyrillic); every other code point passes through (CJK has no case). Invalid UTF-8 is copied byte for byte.
pub fn pyLower(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        const n = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const cp = if (i + n <= text.len) std.unicode.utf8Decode(text[i .. i + n]) catch null else null;
        const c = cp orelse {
            try out.append(a, text[i]);
            i += 1;
            continue;
        };
        const l: u21 = switch (c) {
            'A'...'Z', 0xC0...0xD6, 0xD8...0xDE, 0x391...0x3A1, 0x3A3...0x3AB, 0x410...0x42F => c + 32,
            0x100...0x137, 0x14A...0x177 => c | 1,
            0x139...0x148 => if (c % 2 == 1) c + 1 else c,
            0x400...0x40F => c + 80,
            else => c,
        };
        var buf: [4]u8 = undefined;
        try out.appendSlice(a, buf[0 .. std.unicode.utf8Encode(l, &buf) catch unreachable]);
        i += n;
    }
    return out.toOwnedSlice(a);
}

pub const Prompt = struct {
    /// The text column of every prompt row (the reference-audio slot holds 0).
    ids: []u32,
    /// The reference-audio slot's row.
    muq: usize,
};

/// Tags as training wrote them: comma-separated with no space around a comma (heartlib's README), empties dropped.
pub fn joinTags(a: std.mem.Allocator, tags: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, tags, ',');
    while (it.next()) |raw| {
        const t = std.mem.trim(u8, raw, " \t\r\n");
        if (t.len == 0) continue;
        if (out.items.len > 0) try out.append(a, ',');
        try out.appendSlice(a, t);
    }
    return out.toOwnedSlice(a);
}

/// `<tag>tags</tag>` and the lyrics, each lowercased and wrapped in BOS/EOS, around one reference-audio slot.
pub fn buildPrompt(a: std.mem.Allocator, tok: *const tok_mod.Tokenizer, tags: []const u8, lyrics: []const u8, bos: u32, eos: u32) !Prompt {
    const joined = try joinTags(a, tags);
    defer a.free(joined);
    const lt = try pyLower(a, joined);
    defer a.free(lt);
    const open = !std.mem.startsWith(u8, lt, "<tag>");
    const close = !std.mem.endsWith(u8, lt, "</tag>");
    const wrapped = try std.fmt.allocPrint(a, "{s}{s}{s}", .{ if (open) "<tag>" else "", lt, if (close) "</tag>" else "" });
    defer a.free(wrapped);
    const ll = try pyLower(a, lyrics);
    defer a.free(ll);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    try appendWrapped(a, &ids, tok, wrapped, bos, eos);
    const muq = ids.items.len;
    try ids.append(a, 0);
    try appendWrapped(a, &ids, tok, ll, bos, eos);
    return .{ .ids = try ids.toOwnedSlice(a), .muq = muq };
}

fn appendWrapped(a: std.mem.Allocator, ids: *std.ArrayList(u32), tok: *const tok_mod.Tokenizer, text: []const u8, bos: u32, eos: u32) !void {
    const t = try tok.encode(a, text);
    defer a.free(t);
    if (t.len == 0 or t[0] != bos) try ids.append(a, bos);
    try ids.appendSlice(a, t);
    if (t.len == 0 or t[t.len - 1] != eos) try ids.append(a, eos);
}

/// The reference's loop: one frame from the prompt, then `max_ms / 80` more at most.
pub fn maxFrames(seconds: u32) u32 {
    return 1 + seconds * 1000 / FRAME_MS;
}

// ════════════════════════════════════════════════════════════════════════
// Decode windows (heartcodec/models/decoder.py `detokenize`)
// ════════════════════════════════════════════════════════════════════════

/// The code frames the windows read: fewer than one window repeat to fill it, then the sequence repeats to the next
/// whole hop past the overlap (the reference's `cat([codes, codes])` doubling is a periodic extension).
pub fn padFrames(a: std.mem.Allocator, frames: []const [8]i32) ![][8]i32 {
    const n = frames.len;
    const first = @max(n, WIN_CODES);
    const rest = first - OVLP_CODES;
    const total = if (rest % HOP_CODES == 0) first else (rest + HOP_CODES - 1) / HOP_CODES * HOP_CODES + OVLP_CODES;
    const out = try a.alloc([8]i32, total);
    for (out, 0..) |*o, i| {
        const j = i % first;
        o.* = frames[if (n < WIN_CODES) j % n else j];
    }
    return out;
}

/// Window starts over `padded` code frames: `range(0, padded - hop + 1, hop)`.
pub fn windowCount(padded: usize) usize {
    return (padded - HOP_CODES) / HOP_CODES + 1;
}

/// Samples kept from `frames` code frames: `int(frames / 12.5 * sample_rate)`, in the reference's float arithmetic.
pub fn songSamples(frames: usize, sample_rate: u32) usize {
    return @intFromFloat(@as(f64, @floatFromInt(frames)) / 12.5 * @as(f64, @floatFromInt(sample_rate)));
}

/// `np.linspace(0, 1, OVLP_SAMPLES)[i]`: the incoming window's crossfade weight at overlap sample `i`.
pub fn fadeIn(i: usize) f64 {
    if (i == OVLP_SAMPLES - 1) return 1.0;
    return @as(f64, @floatFromInt(i)) * (1.0 / @as(f64, @floatFromInt(OVLP_SAMPLES - 1)));
}

/// Appends one decoded window (`[2][WIN_SAMPLES]` planar) to the planar song, crossfading its overlap.
fn appendWindow(a: std.mem.Allocator, song: *[2]std.ArrayList(f32), win: []const f32) !void {
    for (song, 0..) |*ch, c| {
        const w = win[c * WIN_SAMPLES ..][0..WIN_SAMPLES];
        if (ch.items.len == 0) {
            try ch.appendSlice(a, w);
            continue;
        }
        const tail = ch.items[ch.items.len - OVLP_SAMPLES ..];
        for (tail, w[0..OVLP_SAMPLES], 0..) |*o, x, i| {
            const f = fadeIn(i);
            o.* = @floatCast(@as(f64, o.*) * (1 - f) + @as(f64, x) * f);
        }
        try ch.appendSlice(a, w[OVLP_SAMPLES..]);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Weights
// ════════════════════════════════════════════════════════════════════════

fn project(parent: *Graph, l: *const Linear, x: A) !A {
    var g = parent.scope();
    defer g.deinit();
    const y = if (l.scales) |sca|
        try g.qmm(x, l.w, sca, l.biases, @intCast(l.group_size), @intCast(l.bits))
    else
        try g.matmul(x, try g.transpose(l.w, &.{ 1, 0 }));
    return parent.keep(if (l.bias) |b| try g.add(y, b) else y);
}

/// A gather-read table: dense, or affine-quantized and read through `gatherQuantizedRows`.
const Table = struct {
    l: Linear,

    fn rows(t: *const Table, parent: *Graph, ids: A) !A {
        const sca = t.l.scales orelse return parent.take(t.l.w, ids, 0);
        return parent.adopt(try embedding.gatherQuantizedRows(parent.s, t.l.w, sca, t.l.biases, ids, t.l.group_size, t.l.bits, .affine, .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }));
    }
};

const Layer = struct {
    ln: A,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    post_ln: A,
    gate: Linear,
    up: Linear,
    down: Linear,
};

const Stack = struct {
    c: StackFile,
    layers: []Layer,
    norm: A,
    /// llama3-scaled `mlx_fast_rope` denominators, or null for the plain base.
    freqs: ?A,

    fn load(a: std.mem.Allocator, w: *const Weights, comptime p: []const u8, c: StackFile) !Stack {
        const layers = try a.alloc(Layer, c.num_hidden_layers);
        errdefer a.free(layers);
        const h = c.hidden_size;
        for (layers, 0..) |*l, i| l.* = .{
            .ln = try getW(w, p ++ ".layers.{d}.input_layernorm.weight", .{i}),
            .q = try linear(w, p ++ ".layers.{d}.self_attn.q_proj", .{i}, h),
            .k = try linear(w, p ++ ".layers.{d}.self_attn.k_proj", .{i}, h),
            .v = try linear(w, p ++ ".layers.{d}.self_attn.v_proj", .{i}, h),
            .o = try linear(w, p ++ ".layers.{d}.self_attn.o_proj", .{i}, h),
            .post_ln = try getW(w, p ++ ".layers.{d}.post_attention_layernorm.weight", .{i}),
            .gate = try linear(w, p ++ ".layers.{d}.mlp.gate_proj", .{i}, h),
            .up = try linear(w, p ++ ".layers.{d}.mlp.up_proj", .{i}, h),
            .down = try linear(w, p ++ ".layers.{d}.mlp.down_proj", .{i}, c.intermediate_size),
        };
        const norm = try getW(w, p ++ ".norm.weight", .{});
        const freqs = if (c.rope_scaling) |r|
            try rope_scaled.freqs(.{ .llama3 = .{ .factor = r.factor, .low_freq_factor = r.low_freq_factor, .high_freq_factor = r.high_freq_factor, .original_max_position_embeddings = r.original_max_position_embeddings } }, c.head_dim, c.rope_theta)
        else
            null;
        return .{ .c = c, .layers = layers, .norm = norm, .freqs = freqs };
    }

    fn deinit(self: *Stack, a: std.mem.Allocator) void {
        if (self.freqs) |f| _ = mlx.mlx_array_free(f);
        a.free(self.layers);
    }

    fn rope(self: *const Stack, g: *Graph, x: A, offset: c_int) !A {
        const hd: c_int = @intCast(self.c.head_dim);
        if (self.freqs) |f| return g.call(mlx.mlx_fast_rope, .{ x, hd, true, mlx.mlx_optional_float.none(), 1.0, offset, f });
        return g.call(mlx.mlx_fast_rope, .{ x, hd, true, mlx.mlx_optional_float.some(@floatCast(self.c.rope_theta)), 1.0, offset, A{} });
    }

    /// `x` [rows, T, D] appended to `kv`; the last position's post-norm hidden [rows, D].
    fn forward(self: *const Stack, parent: *Graph, x: A, kv: *Kv) !A {
        var g0 = parent.scope();
        defer g0.deinit();
        const c = self.c;
        const sh = ops.shape(x);
        const t = sh[1];
        const heads: c_int = @intCast(c.num_attention_heads);
        const kvh: c_int = @intCast(c.num_key_value_heads);
        const hd: c_int = @intCast(c.head_dim);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        var h = x;
        for (self.layers, 0..) |*l, li| {
            var g = g0.scope();
            defer g.deinit();
            const n = try g.rmsNorm(h, l.ln, c.rms_norm_eps);
            const q = try self.rope(&g, try g.splitHeads(try project(&g, &l.q, n), heads, hd), kv.len);
            try kv.write(li, try self.rope(&g, try g.splitHeads(try project(&g, &l.k, n), kvh, hd), kv.len), try g.splitHeads(try project(&g, &l.v, n), kvh, hd));
            const kview = try g.sliceAxis(kv.ks[li], 2, 0, kv.len + t);
            const vview = try g.sliceAxis(kv.vs[li], 2, 0, kv.len + t);
            const attn = try g.sdpa(q, kview, vview, scale, if (t > 1) .causal else .none);
            const h1 = try g.add(h, try project(&g, &l.o, try g.mergeHeads(attn)));
            const m = try g.rmsNorm(h1, l.post_ln, c.rms_norm_eps);
            const next = try g0.keep(try g.add(h1, try project(&g, &l.down, try g.mul(try g.silu(try project(&g, &l.gate, m)), try project(&g, &l.up, m)))));
            if (li > 0) g0.drop(h);
            h = next;
        }
        kv.len += t;
        const last = try g0.reshape(try g0.sliceAxis(h, 1, t - 1, t), &.{ sh[0], @intCast(c.hidden_size) });
        return parent.keep(try g0.rmsNorm(last, self.norm, c.rms_norm_eps));
    }
};

/// Per-layer K/V buffers [rows, kv_heads, cap, head_dim], written in place.
const Kv = struct {
    g: Graph,
    ks: []A,
    vs: []A,
    len: c_int = 0,

    fn init(a: std.mem.Allocator, s: S, c: StackFile, rows: c_int, cap: usize) !Kv {
        const ks = try a.alloc(A, c.num_hidden_layers);
        errdefer a.free(ks);
        const vs = try a.alloc(A, c.num_hidden_layers);
        errdefer a.free(vs);
        var g = Graph.init(a, s);
        errdefer g.deinit();
        const shape = [_]c_int{ rows, @intCast(c.num_key_value_heads), @intCast(cap), @intCast(c.head_dim) };
        for (ks, vs) |*k, *v| {
            k.* = try g.zeros(&shape, .bfloat16);
            v.* = try g.zeros(&shape, .bfloat16);
        }
        return .{ .g = g, .ks = ks, .vs = vs };
    }

    fn deinit(self: *Kv) void {
        self.g.deinit();
        self.g.a.free(self.ks);
        self.g.a.free(self.vs);
    }

    fn write(self: *Kv, li: usize, k: A, v: A) !void {
        inline for (.{ &self.ks[li], &self.vs[li] }, .{ k, v }) |buf, new| {
            const sh = ops.shape(buf.*);
            const t = ops.shape(new)[2];
            const o = try self.g.sliceUpdate(buf.*, try self.g.astype(new, .bfloat16), &.{ 0, 0, self.len, 0 }, &.{ sh[0], sh[1], self.len + t, sh[3] });
            self.g.drop(buf.*);
            buf.* = o;
        }
    }
};

// ════════════════════════════════════════════════════════════════════════
// Sampling
// ════════════════════════════════════════════════════════════════════════

pub const Sampling = struct {
    temperature: f32 = 1.0,
    top_k: u32 = DEFAULT_TOP_K,
    cfg: f32 = DEFAULT_CFG,
    seed: u64 = 0,
    /// argmax instead of a draw (the parity replay).
    greedy: bool = false,
    draws: u64 = 0,

    fn key(self: *Sampling, g: *Graph) !A {
        self.draws += 1;
        return g.call(mlx.mlx_random_key, .{self.seed +% self.draws *% 0x9E3779B97F4A7C15});
    }
};

/// `uncond + (cond - uncond) * cfg` over f32 logits [rows, V] → [1, V].
fn guide(parent: *Graph, logits: A, cfg: f32) !A {
    if (ops.shape(logits)[0] == 1) return logits;
    var g = parent.scope();
    defer g.deinit();
    const v = ops.shape(logits)[1];
    const c = try g.slice(logits, &.{ 0, 0 }, &.{ 1, v });
    const u = try g.slice(logits, &.{ 1, 0 }, &.{ 2, v });
    return parent.keep(try g.add(u, try g.mulScalar(try g.sub(c, u), cfg)));
}

/// heartlib `sample_topk`: logits / T, everything below the k-th largest to -inf, one categorical draw → [1] i32.
fn sampleTopK(parent: *Graph, logits: A, smp: *Sampling) !A {
    var g = parent.scope();
    defer g.deinit();
    if (smp.greedy) return parent.keep(try g.astype(try g.argmax(logits, -1, false), .int32));
    const x = try g.div(logits, try g.scalarLike(logits, smp.temperature));
    const k: c_int = @intCast(@min(smp.top_k, @as(u32, @intCast(ops.shape(x)[1]))));
    const floor = try g.call(mlx.mlx_min_axis, .{ try g.call(mlx.mlx_topk_axis, .{ x, k, -1 }), -1, true });
    const masked = try g.where(try g.less(x, floor), try g.scalarLike(x, -std.math.inf(f32)), x);
    const pick = try g.call(mlx.mlx_random_categorical, .{ masked, -1, try smp.key(&g) });
    return parent.keep(try g.astype(pick, .int32));
}

// ════════════════════════════════════════════════════════════════════════
// Codec (HeartCodec decode path)
// ════════════════════════════════════════════════════════════════════════

const Block = struct {
    attn_norm: A,
    mlp_norm: A,
    table: A,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
};

const ProjectLayer = struct {
    conv_w: A,
    conv_b: A,
    k: u32,
    lin: Linear,

    fn load(g: *Graph, w: *const Weights, comptime p: []const u8, dt: mlx.mlx_dtype) !ProjectLayer {
        const cw = try getW(w, p ++ ".ffn_1.weight", .{});
        const k: u32 = @intCast(ops.shape(cw)[1]);
        return .{ .conv_w = try g.astype(cw, dt), .conv_b = try g.astype(try getW(w, p ++ ".ffn_1.bias", .{}), dt), .k = k, .lin = try linearAs(g, w, p ++ ".ffn_2", dt) };
    }

    /// Conv1d (same padding) · k^-1/2, then a Linear.
    fn forward(self: *const ProjectLayer, parent: *Graph, x: A) !A {
        var g = parent.scope();
        defer g.deinit();
        const y = try g.add(try g.conv1d(x, self.conv_w, 1, @intCast(self.k / 2), 1, 1), self.conv_b);
        const scaled = try g.mulScalar(y, 1.0 / @sqrt(@as(f32, @floatFromInt(self.k))));
        return parent.keep(try project(&g, &self.lin, scaled));
    }
};

/// A dense Linear with its weight and bias in `dt` (held by `g`).
fn linearAs(g: *Graph, w: *const Weights, comptime p: []const u8, dt: mlx.mlx_dtype) !Linear {
    var l = try linear(w, p, .{}, 0);
    l.w = try g.astype(l.w, dt);
    if (l.bias) |b| l.bias = try g.astype(b, dt);
    return l;
}

const AdaLn = struct {
    l1: Linear,
    l2: Linear,
    mod: Linear,
};

const DitStack = struct {
    blocks: []Block,
    ada: AdaLn,
    table: A,
    heads: c_int,
    hd: c_int,
};

const Codec = struct {
    keep: Graph,
    c: CodecFile,
    dt: mlx.mlx_dtype,
    codebooks: []A,
    project_out: Linear,
    cond_emb: Linear,
    proj_in: ProjectLayer,
    connection: ProjectLayer,
    proj_out: ProjectLayer,
    stacks: [2]DitStack,

    fn load(a: std.mem.Allocator, s: S, w: *const Weights, c: CodecFile) !Codec {
        const dt: mlx.mlx_dtype = .bfloat16;
        var self: Codec = undefined;
        self.keep = Graph.init(a, s);
        errdefer self.keep.deinit();
        self.c = c;
        self.dt = dt;
        const g = &self.keep;
        self.codebooks = try a.alloc(A, c.num_quantizers);
        errdefer a.free(self.codebooks);
        for (self.codebooks, 0..) |*cb, i| cb.* = try getW(w, "flow_matching.vq_embed.layers.{d}.codebook", .{i});
        self.project_out = try linear(w, "flow_matching.vq_embed.project_out", .{}, 0);
        self.cond_emb = try linear(w, "flow_matching.cond_feature_emb", .{}, 0);
        const e = "flow_matching.estimator.";
        self.proj_in = try ProjectLayer.load(g, w, e ++ "proj_in", dt);
        self.connection = try ProjectLayer.load(g, w, e ++ "connection_proj", dt);
        self.proj_out = try ProjectLayer.load(g, w, e ++ "proj_out", dt);
        const counts = [2]u32{ c.num_layers, c.num_layers_2 };
        var made: usize = 0;
        errdefer for (self.stacks[0..made]) |st| a.free(st.blocks);
        inline for (.{ "transformer_blocks", "transformer_blocks_2" }, .{ "adaln_single", "adaln_single_2" }, .{ "scale_shift_table", "scale_shift_table_2" }, 0..) |bp, ap, tp, si| {
            const blocks = try a.alloc(Block, counts[si]);
            errdefer a.free(blocks);
            for (blocks, 0..) |*b, i| b.* = .{
                .attn_norm = try g.astype(try getW(w, e ++ bp ++ ".{d}.attn_norm.weight", .{i}), dt),
                .mlp_norm = try g.astype(try getW(w, e ++ bp ++ ".{d}.mlp_norm.weight", .{i}), dt),
                .table = try g.astype(try getW(w, e ++ bp ++ ".{d}.scale_shift_table", .{i}), dt),
                .q = try linear(w, e ++ bp ++ ".{d}.attn.q_proj", .{i}, 0),
                .k = try linear(w, e ++ bp ++ ".{d}.attn.k_proj", .{i}, 0),
                .v = try linear(w, e ++ bp ++ ".{d}.attn.v_proj", .{i}, 0),
                .o = try linear(w, e ++ bp ++ ".{d}.attn.o_proj", .{i}, 0),
                .gate = try linear(w, e ++ bp ++ ".{d}.mlp.gate", .{i}, 0),
                .up = try linear(w, e ++ bp ++ ".{d}.mlp.up", .{i}, 0),
                .down = try linear(w, e ++ bp ++ ".{d}.mlp.down", .{i}, 0),
            };
            self.stacks[si] = .{
                .blocks = blocks,
                .ada = .{
                    .l1 = try linearAs(g, w, e ++ ap ++ ".emb.timestep_embedder.linear_1", dt),
                    .l2 = try linearAs(g, w, e ++ ap ++ ".emb.timestep_embedder.linear_2", dt),
                    .mod = try linearAs(g, w, e ++ ap ++ ".linear", dt),
                },
                .table = try g.astype(try getW(w, e ++ tp, .{}), dt),
                .heads = @intCast(c.num_attention_heads),
                .hd = @intCast(c.attention_head_dim * (si + 1)),
            };
            made += 1;
        }
        return self;
    }

    fn deinit(self: *Codec, a: std.mem.Allocator) void {
        for (self.stacks) |st| a.free(st.blocks);
        a.free(self.codebooks);
        self.keep.deinit();
    }

    /// RVQ dequant + `cond_feature_emb` + nearest ×2: codes `[n][Q]` → [1, 2n, dim] f32.
    fn condition(self: *const Codec, parent: *Graph, frames: []const [8]i32) !A {
        var g = parent.scope();
        defer g.deinit();
        const n: c_int = @intCast(frames.len);
        const q = self.c.num_quantizers;
        const idx = try g.a.alloc(i32, frames.len * q);
        defer g.a.free(idx);
        const top: i32 = @intCast(self.c.codebook_size - 1);
        for (0..q) |qi| for (frames, 0..) |f, t| {
            idx[qi * frames.len + t] = std.math.clamp(f[qi], 0, top);
        };
        const ids = try g.fromSlice(i32, idx, &.{ @intCast(q), n });
        var sum: ?A = null;
        for (self.codebooks, 0..) |cb, qi| {
            const rows = try g.take(cb, try g.reshape(try g.sliceAxis(ids, 0, @intCast(qi), @intCast(qi + 1)), &.{n}), 0);
            sum = if (sum) |s| try g.add(s, rows) else rows;
        }
        const emb = try project(&g, &self.cond_emb, try project(&g, &self.project_out, sum.?));
        return parent.keep(try g.reshape(try g.repeat(emb, 2, 0), &.{ 1, 2 * n, @intCast(self.c.dim) }));
    }

    /// PixArt-α timestep embedding through one stack's AdaLN-single: (embedded [1, D], modulation [1, 6D]).
    fn timestep(self: *const Codec, parent: *Graph, st: *const DitStack, t: f32) ![2]A {
        var g = parent.scope();
        defer g.deinit();
        var freqs: [512]f32 = undefined;
        for (freqs[0..256], 0..) |*f, i| f.* = t * @exp(-@log(@as(f32, 10000)) * @as(f32, @floatFromInt(i)) / 256.0) * 1000;
        const args = try g.fromSlice(f32, freqs[0..256], &.{ 1, 256 });
        const emb = try g.astype(try g.concat(&.{ try g.cos(args), try g.sin(args) }, -1), self.dt);
        const e = try project(&g, &st.ada.l2, try g.silu(try project(&g, &st.ada.l1, emb)));
        return .{ try parent.keep(e), try parent.keep(try project(&g, &st.ada.mod, try g.silu(e))) };
    }

    fn block(parent: *Graph, st: *const DitStack, b: *const Block, x: A, mod: A) !A {
        var g = parent.scope();
        defer g.deinit();
        const d = ops.shape(x)[2];
        const m = try g.add(try g.expandDims(b.table, 0), try g.reshape(mod, &.{ 1, 6, d }));
        var parts: [6]A = undefined;
        for (&parts, 0..) |*p, i| p.* = try g.sliceAxis(m, 1, @intCast(i), @intCast(i + 1));
        const one = try g.scalarLike(x, 1);
        const n1 = try g.add(try g.mul(try g.rmsNorm(x, b.attn_norm, 1e-6), try g.add(one, parts[1])), parts[0]);
        const rope = struct {
            fn f(gg: *Graph, y: A, hd: c_int) !A {
                return gg.call(mlx.mlx_fast_rope, .{ y, hd, true, mlx.mlx_optional_float.some(10000), 1.0, 0, A{} });
            }
        }.f;
        const q = try rope(&g, try g.splitHeads(try project(&g, &b.q, n1), st.heads, st.hd), st.hd);
        const k = try rope(&g, try g.splitHeads(try project(&g, &b.k, n1), st.heads, st.hd), st.hd);
        const v = try g.splitHeads(try project(&g, &b.v, n1), st.heads, st.hd);
        const attn = try project(&g, &b.o, try g.mergeHeads(try g.sdpa(q, k, v, 1.0 / @sqrt(@as(f32, @floatFromInt(st.hd))), .none)));
        const x1 = try g.add(x, try g.mul(parts[2], attn));
        const n2 = try g.add(try g.mul(try g.rmsNorm(x1, b.mlp_norm, 1e-6), try g.add(one, parts[4])), parts[3]);
        const mlp = try project(&g, &b.down, try g.mul(try g.silu(try project(&g, &b.gate, n2)), try project(&g, &b.up, n2)));
        return parent.keep(try g.add(x1, try g.mul(parts[5], mlp)));
    }

    /// One stack's blocks, a scope (and an eval) per block, then the final modulated LayerNorm.
    fn runStack(self: *const Codec, parent: *Graph, si: usize, x_in: A, t: f32) !A {
        var g = parent.scope();
        defer g.deinit();
        const st = &self.stacks[si];
        const te = try self.timestep(&g, st, t);
        var x = x_in;
        for (st.blocks, 0..) |*b, i| {
            const next = try block(&g, st, b, x, te[1]);
            try ops.eval(next);
            if (i > 0) g.drop(x);
            x = next;
        }
        const ss = try g.add(try g.expandDims(st.table, 0), try g.expandDims(te[0], 1));
        const shift = try g.sliceAxis(ss, 1, 0, 1);
        const scale = try g.sliceAxis(ss, 1, 1, 2);
        const n = try g.call(mlx.mlx_fast_layer_norm, .{ x, A{}, A{}, @as(f32, 1e-6) });
        return parent.keep(try g.add(try g.mul(n, try g.add(try g.scalarLike(n, 1), scale)), shift));
    }

    /// The estimator: [B, T, in_channels] at flow time `t` → velocity [B, T, out_channels] f32.
    fn velocity(self: *const Codec, parent: *Graph, x: A, t: f32) !A {
        var g = parent.scope();
        defer g.deinit();
        const xin = try g.astype(x, self.dt);
        const s1 = try self.runStack(&g, 0, try self.proj_in.forward(&g, xin), t);
        const s2 = try self.runStack(&g, 1, try self.connection.forward(&g, try g.concat(&.{ xin, s1 }, -1)), t);
        return parent.keep(try g.astype(try self.proj_out.forward(&g, s2), .float32));
    }

    /// heartlib `solve_euler` + the in-context write-back: noise/ctx [1, L, out], mu [1, L, dim] → latents [1, L, out].
    fn solve(self: *const Codec, parent: *Graph, noise: A, ctx: A, ic: c_int, mu: A, steps: u32, guidance: f32, progress: ?sse.Progress, step_base: u32, step_total: u32) !A {
        var g = parent.scope();
        defer g.deinit();
        const sh = ops.shape(noise);
        var x = noise;
        var t: f32 = 0;
        var dt: f32 = 1.0 / @as(f32, @floatFromInt(steps));
        const zeros = try g.zeros(ops.shape(mu), .float32);
        for (1..steps + 1) |step| {
            if (progress) |p| {
                if (p.boundary()) return error.Cancelled;
                p.emit("nar", step_base + @as(u32, @intCast(step - 1)), step_total);
            }
            var sg = g.scope();
            defer sg.deinit();
            var xs = x;
            if (ic > 0) {
                const blend = try sg.add(try sg.mulScalar(try sg.sliceAxis(noise, 1, 0, ic), 1 - (1 - 1e-6) * t), try sg.mulScalar(try sg.sliceAxis(ctx, 1, 0, ic), t));
                xs = try sg.sliceUpdate(x, blend, &.{ 0, 0, 0 }, &.{ sh[0], ic, sh[2] });
            }
            const v = if (guidance > 1.0) blk: {
                const both = try sg.concat(&.{ try sg.concat(&.{ xs, ctx, zeros }, -1), try sg.concat(&.{ xs, ctx, mu }, -1) }, 0);
                const vv = try self.velocity(&sg, both, t);
                const u = try sg.sliceAxis(vv, 0, 0, 1);
                break :blk try sg.add(u, try sg.mulScalar(try sg.sub(try sg.sliceAxis(vv, 0, 1, 2), u), guidance));
            } else try self.velocity(&sg, try sg.concat(&.{ xs, ctx, mu }, -1), t);
            const next = try g.keep(try sg.add(xs, try sg.mulScalar(v, dt)));
            try ops.eval(next);
            if (step > 1) g.drop(x);
            x = next;
            t += dt;
            if (step < steps) dt = @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(steps)) - t;
        }
        if (ic > 0) x = try g.sliceUpdate(x, try g.sliceAxis(ctx, 1, 0, ic), &.{ 0, 0, 0 }, &.{ sh[0], ic, sh[2] });
        return parent.keep(x);
    }
};

// ── scalar decoder (sq_codec.ScalarModel.decode), f32 channels-last ──

fn prelu(g: *Graph, x: A, a: A) !A {
    const zero = try g.scalarLike(x, 0);
    return g.add(try g.maximum(x, zero), try g.mul(a, try g.minimum(x, zero)));
}

const Scalar = struct {
    w: *const Weights,

    fn conv(self: Scalar, parent: *Graph, x: A, comptime fmt: []const u8, args: anytype, causal_pad: c_int, pad: c_int, dilation: c_int) !A {
        var g = parent.scope();
        defer g.deinit();
        const xin = if (causal_pad > 0) try g.padAxis(x, 1, causal_pad, 0) else x;
        const y = try g.conv1d(xin, try getW(self.w, fmt ++ ".weight", args), 1, pad, dilation, 1);
        return parent.keep(try g.add(y, try getW(self.w, fmt ++ ".bias", args)));
    }

    /// Evaluates `next`, frees `prev` and hands the freed buffers back (the decoder's activations reach ~1 GB).
    fn advance(g: *Graph, prev: A, next: A) !A {
        try ops.eval(next);
        g.drop(prev);
        _ = mlx.mlx_clear_cache();
        return next;
    }

    /// Latents [bands, L, hidden] f32 → audio [bands, L · hop] f32.
    fn decode(self: Scalar, parent: *Graph, z: A, c: CodecFile) !A {
        var g = parent.scope();
        defer g.deinit();
        const p = "scalar_model.decoder.{d}";
        const q = try g.div(try g.call(mlx.mlx_round, .{ try g.mulScalar(z, 9), @as(c_int, 0) }), try g.scalarLike(z, 9));
        var h = try self.conv(&g, q, p, .{0}, 0, @intCast(c.delay_kernel_size / 2), 1);
        const k: c_int = @intCast(c.res_kernel_size);
        for (c.upsample_factors, 1..) |f, bi| {
            const stride: c_int = @intCast(f);
            {
                var sg = g.scope();
                defer sg.deinit();
                const up = try sg.convTranspose1d(h, try getW(self.w, p ++ ".up_conv.layer.weight", .{bi}), stride, 0, 1, 0, 1);
                const len = ops.shape(up)[1];
                const y = try sg.add(try sg.sliceAxis(up, 1, 0, len - stride), try getW(self.w, p ++ ".up_conv.layer.bias", .{bi}));
                h = try advance(&g, h, try g.keep(y));
            }
            for ([_]c_int{ 1, 3, 5, 7, 9 }, 0..) |d, ui| {
                var sg = g.scope();
                defer sg.deinit();
                const u = p ++ ".convs.{d}";
                const c1 = try prelu(&sg, try self.conv(&sg, h, u ++ ".conv1", .{ bi, ui }, d * (k - 1), 0, d), try getW(self.w, u ++ ".activation1.weight", .{ bi, ui }));
                const c2 = try prelu(&sg, try self.conv(&sg, c1, u ++ ".conv2", .{ bi, ui }, 0, 0, 1), try getW(self.w, u ++ ".activation2.weight", .{ bi, ui }));
                h = try advance(&g, h, try g.keep(try sg.add(h, c2)));
            }
        }
        const n = c.upsample_factors.len + 1;
        const dk: c_int = @intCast(c.default_kernel_size);
        {
            var sg = g.scope();
            defer sg.deinit();
            const rep = try sg.repeat(h, @intCast(c.num_samples), 1);
            const y = try prelu(&sg, try self.conv(&sg, rep, p ++ ".conv", .{n}, dk - 1, 0, 1), try getW(self.w, p ++ ".activation.weight", .{n}));
            h = try advance(&g, h, try g.keep(y));
        }
        const out = try self.conv(&g, h, p, .{n + 1}, dk - 1, 0, 1);
        const sh = ops.shape(out);
        return parent.keep(try g.reshape(out, &.{ sh[0], sh[1] }));
    }
};

// ════════════════════════════════════════════════════════════════════════
// Engine
// ════════════════════════════════════════════════════════════════════════

pub const Request = struct {
    tags: []const u8,
    lyrics: []const u8,
    seconds: u32 = DEFAULT_SECONDS,
    sampling: Sampling = .{},
    steps: u32 = DEFAULT_STEPS,
};

pub const Engine = struct {
    a: std.mem.Allocator,
    s: S,
    parsed: std.json.Parsed(Cfg),
    cfg: Cfg,
    w: Weights,
    cw: Weights,
    tok: tok_mod.Tokenizer,
    backbone: Stack,
    decoder: Stack,
    text: Table,
    audio: Table,
    uncond: A,
    muq_bias: A,
    projection: Linear,
    c0_head: Linear,
    heads: []Linear,
    codec: Codec,

    pub fn load(io: std.Io, a: std.mem.Allocator, dir: []const u8, s: S) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        self.a = a;
        self.s = s;
        const bytes = try readFile(io, a, dir, "config.json");
        defer a.free(bytes);
        self.parsed = try parseConfig(a, bytes);
        errdefer self.parsed.deinit();
        const c = self.parsed.value;
        self.cfg = c;
        self.w = try model_mod.loadWeightsFile(a, dir, "model.safetensors");
        errdefer self.w.deinit();
        self.cw = try model_mod.loadWeightsFile(a, dir, "codec.safetensors");
        errdefer self.cw.deinit();
        try scope_mod.evalAll(&self.w);
        try scope_mod.evalAll(&self.cw);
        self.tok = try tok_load.loadTokenizer(io, a, dir);
        errdefer self.tok.deinit();

        const w = &self.w;
        const d = c.backbone.hidden_size;
        self.backbone = try Stack.load(a, w, "backbone", c.backbone);
        errdefer self.backbone.deinit(a);
        self.decoder = try Stack.load(a, w, "decoder", c.decoder);
        errdefer self.decoder.deinit(a);
        self.text = .{ .l = try linear(w, "text_embeddings", .{}, d) };
        self.audio = .{ .l = try linear(w, "audio_embeddings", .{}, d) };
        self.uncond = try getW(w, "unconditional_text_embedding.weight", .{});
        self.muq_bias = try getW(w, "muq_linear.bias", .{});
        self.projection = try linear(w, "projection", .{}, d);
        self.c0_head = try linear(w, "codebook0_head", .{}, d);
        self.heads = try a.alloc(Linear, c.audio_num_codebooks - 1);
        errdefer a.free(self.heads);
        for (self.heads, 0..) |*h, i| h.* = try linear(w, "audio_head.{d}", .{i}, d);
        self.codec = try Codec.load(a, s, &self.cw, c.codec);
        log.info("[heartmula] engine ready ({d}-layer backbone, {d}-layer depth decoder, codec {d}+{d} blocks)\n", .{ c.backbone.num_hidden_layers, c.decoder.num_hidden_layers, c.codec.num_layers, c.codec.num_layers_2 });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        const a = self.a;
        self.codec.deinit(a);
        a.free(self.heads);
        self.decoder.deinit(a);
        self.backbone.deinit(a);
        self.tok.deinit();
        self.cw.deinit();
        self.w.deinit();
        self.parsed.deinit();
        a.destroy(self);
    }

    pub fn prompt(self: *const Engine, a: std.mem.Allocator, tags: []const u8, lyrics: []const u8) !Prompt {
        return buildPrompt(a, &self.tok, tags, lyrics, self.cfg.text_bos_id, self.cfg.text_eos_id);
    }

    /// Does the prompt plus `seconds` of frames fit the backbone's positions?
    pub fn fits(self: *const Engine, p: Prompt, seconds: u32) bool {
        return p.ids.len + maxFrames(seconds) - 1 <= self.cfg.backbone.max_position_embeddings;
    }

    fn dim(self: *const Engine) c_int {
        return @intCast(self.cfg.backbone.hidden_size);
    }

    /// The prompt rows [rows, P, D]: the conditional row's text embeddings with the reference-audio slot set to
    /// `muq_linear(0)` (its bias); the unconditional row is `unconditional_text_embedding` everywhere, slot included.
    fn promptEmbeds(self: *const Engine, parent: *Graph, p: Prompt, rows: c_int) !A {
        var g = parent.scope();
        defer g.deinit();
        const n: c_int = @intCast(p.ids.len);
        const ids = try g.a.alloc(i32, p.ids.len);
        defer g.a.free(ids);
        for (p.ids, ids) |x, *o| o.* = @intCast(x);
        const text = try self.text.rows(&g, try g.fromSlice(i32, ids, &.{n}));
        const muq: c_int = @intCast(p.muq);
        const bias = try g.astype(try g.reshape(self.muq_bias, &.{ 1, self.dim() }), ops.dtype(text));
        const cond = try g.reshape(try g.sliceUpdate(text, bias, &.{ muq, 0 }, &.{ muq + 1, self.dim() }), &.{ 1, n, self.dim() });
        if (rows == 1) return parent.keep(cond);
        const un = try g.broadcast(try g.reshape(try g.astype(self.uncond, ops.dtype(text)), &.{ 1, 1, self.dim() }), &.{ 1, n, self.dim() });
        return parent.keep(try g.concat(&.{ cond, un }, 0));
    }

    /// Codebook `cb`'s embedding of `code` [1] → [1, D].
    fn audioRow(self: *const Engine, parent: *Graph, code: A, cb: u32) !A {
        var g = parent.scope();
        defer g.deinit();
        const off = try g.fromSlice(i32, &.{@intCast(cb * self.cfg.audio_vocab_size)}, &.{1});
        return parent.keep(try self.audio.rows(&g, try g.add(code, off)));
    }

    /// The next backbone input [rows, 1, D]: the sum of the frame's codebook embeddings (the text column is masked).
    fn frameEmbed(self: *const Engine, parent: *Graph, codes: A, rows: c_int) !A {
        var g = parent.scope();
        defer g.deinit();
        const nq = self.cfg.audio_num_codebooks;
        const offs = try g.a.alloc(i32, nq);
        defer g.a.free(offs);
        for (offs, 0..) |*o, i| o.* = @intCast(i * self.cfg.audio_vocab_size);
        const rws = try self.audio.rows(&g, try g.add(codes, try g.fromSlice(i32, offs, &.{@intCast(nq)})));
        const sum = try g.reshape(try g.sum(rws, 0, false), &.{ 1, 1, self.dim() });
        return parent.keep(try g.broadcast(sum, &.{ rows, 1, self.dim() }));
    }

    /// A projected [rows, T, D] decoder input from rows [n, D] (n = rows, or 1 broadcast).
    fn depthIn(self: *const Engine, parent: *Graph, x: A, rows: c_int) !A {
        var g = parent.scope();
        defer g.deinit();
        const sh = ops.shape(x);
        const p = try project(&g, &self.projection, try g.reshape(x, &.{ sh[0], 1, self.dim() }));
        return parent.keep(if (sh[0] == rows) p else try g.broadcast(p, &.{ rows, 1, self.dim() }));
    }

    pub const Replay = struct {
        /// The reference frame's codes, fed back instead of ours.
        force: []const i32,
        /// Our argmax matched the forced code.
        agree: *u32,
        /// Every guided logit row of the frame, kept in `taps`'s Graph.
        taps: ?*std.ArrayList(A) = null,
        tap_graph: ?*Graph = null,
    };

    /// One frame from backbone input `h` [rows, T, D]: its codes [Q] i32 (lazy).
    fn frame(self: *const Engine, parent: *Graph, h: A, bkv: *Kv, dkv: *Kv, smp: *Sampling, replay: ?Replay) !A {
        var g = parent.scope();
        defer g.deinit();
        const rows = ops.shape(h)[0];
        const last = try self.backbone.forward(&g, h, bkv);
        var codes: [8]A = undefined;
        const nq = self.cfg.audio_num_codebooks;
        dkv.len = 0;
        var cb: u32 = 0;
        var dh = last;
        while (cb < nq) : (cb += 1) {
            const lin = if (cb == 0) &self.c0_head else &self.heads[cb - 1];
            const logits = try guide(&g, try g.astype(try project(&g, lin, dh), .float32), smp.cfg);
            if (replay) |r| if (r.taps) |t| try t.append(r.tap_graph.?.a, try r.tap_graph.?.keep(logits));
            var code = try sampleTopK(&g, logits, smp);
            if (replay) |r| {
                try ops.eval(code);
                var ours: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&ours, code));
                if (ours == r.force[cb]) r.agree.* += 1;
                code = try g.fromSlice(i32, r.force[cb .. cb + 1], &.{1});
            }
            codes[cb] = code;
            if (cb + 1 == nq) break;
            const e = try self.depthIn(&g, try self.audioRow(&g, code, cb), rows);
            const x = if (cb == 0) try g.concat(&.{ try self.depthIn(&g, last, rows), e }, 1) else e;
            dh = try self.decoder.forward(&g, x, dkv);
        }
        return parent.keep(try g.concat(codes[0..nq], 0));
    }

    /// The AR stage: code frames until an EOS frame or `max_frames` (owned, `[n][8]`).
    pub fn generateFrames(self: *Engine, a: std.mem.Allocator, p: Prompt, max_frames: u32, smp: *Sampling, progress: ?sse.Progress) ![][8]i32 {
        if (self.cfg.audio_num_codebooks != 8) return error.HeartMulaConfigUnsupported;
        const rows: c_int = if (smp.cfg > 1.0) 2 else 1;
        var g = Graph.init(a, self.s);
        defer g.deinit();
        var bkv = try Kv.init(a, self.s, self.cfg.backbone, rows, p.ids.len + max_frames);
        defer bkv.deinit();
        var dkv = try Kv.init(a, self.s, self.cfg.decoder, rows, self.cfg.audio_num_codebooks);
        defer dkv.deinit();

        var out: std.ArrayList([8]i32) = .empty;
        errdefer out.deinit(a);
        var pending: std.ArrayList(A) = .empty;
        defer pending.deinit(a);
        var h = try self.promptEmbeds(&g, p, rows);
        var done = false;
        var i: u32 = 0;
        while (i < max_frames and !done) : (i += 1) {
            if (progress) |pr| {
                if (pr.boundary()) return error.Cancelled;
                if (i % 25 == 0) pr.emit("song", i * FRAME_MS / 1000, 0);
            }
            const codes = try self.frame(&g, h, &bkv, &dkv, smp, null);
            const next = try self.frameEmbed(&g, codes, rows);
            try ops.asyncMany(&.{ codes, next });
            g.drop(h);
            h = next;
            try pending.append(a, codes);
            if (pending.items.len == EOS_POLL or i + 1 == max_frames) {
                for (pending.items) |pc| {
                    const host = try readCodes(pc);
                    // the prompt's frame is never checked (the reference appends it unconditionally)
                    if (out.items.len > 0 and endsSong(host, self.cfg.audio_eos_id)) {
                        done = true;
                        break;
                    }
                    try out.append(a, host);
                }
                for (pending.items) |pc| g.drop(pc);
                pending.clearRetainingCapacity();
            }
        }
        if (progress) |pr| pr.emit("song", @intCast(out.items.len * FRAME_MS / 1000), 0);
        return out.toOwnedSlice(a);
    }

    /// The codec stage: code frames → planar stereo f32 `[2][samples]` (owned).
    pub fn decodeFrames(self: *Engine, a: std.mem.Allocator, frames: []const [8]i32, steps: u32, seed: u64, progress: ?sse.Progress) ![2][]f32 {
        const c = self.cfg.codec;
        const padded = try padFrames(a, frames);
        defer a.free(padded);
        const n_win = windowCount(padded.len);
        var song: [2]std.ArrayList(f32) = .{ .empty, .empty };
        defer for (&song) |*ch| ch.deinit(a);
        const latent: c_int = @intCast(c.out_channels);
        const band: c_int = @intCast(c.latent_hidden_dim);
        const lw: c_int = @intCast(WIN_LATENTS);
        var prev_tail: ?A = null;
        var keep = Graph.init(a, self.s);
        defer keep.deinit();
        for (0..n_win) |wi| {
            var g = Graph.init(a, self.s);
            defer g.deinit();
            defer _ = mlx.mlx_clear_cache();
            const mu = try self.codec.condition(&g, padded[wi * HOP_CODES ..][0..WIN_CODES]);
            const key = try g.call(mlx.mlx_random_key, .{seed ^ 0x5EED_C0DE +% wi});
            const noise = try g.call(mlx.mlx_random_normal, .{ &[_]c_int{ 1, lw, latent }, 3, .float32, @as(f32, 0), @as(f32, 1), key });
            var ctx = try g.zeros(&.{ 1, lw, latent }, .float32);
            var ic: c_int = 0;
            if (prev_tail) |t| {
                ic = @intCast(CTX_LATENTS);
                ctx = try g.sliceUpdate(ctx, t, &.{ 0, 0, 0 }, &.{ 1, ic, latent });
            }
            const lat = try self.codec.solve(&g, noise, ctx, ic, mu, steps, CODEC_GUIDANCE, progress, @intCast(wi * steps), @intCast(n_win * steps));
            try ops.eval(lat);
            if (prev_tail) |t| keep.drop(t);
            prev_tail = try keep.keep(try g.contiguous(try g.sliceAxis(lat, 1, lw - @as(c_int, @intCast(CTX_LATENTS)), lw)));
            try ops.eval(prev_tail.?);
            const z = try g.transpose(try g.reshape(lat, &.{ lw, 2, band }), &.{ 1, 0, 2 });
            const pcm = try g.contiguous(try (Scalar{ .w = &self.cw }).decode(&g, z, c));
            try ops.eval(pcm);
            const total: usize = @intCast(ops.shape(pcm)[1]);
            if (total < WIN_SAMPLES) return error.HeartMulaShortWindow;
            const data = mlx.mlx_array_data_float32(pcm) orelse return error.NoData;
            const win = try a.alloc(f32, 2 * WIN_SAMPLES);
            defer a.free(win);
            for (0..2) |ch| @memcpy(win[ch * WIN_SAMPLES ..][0..WIN_SAMPLES], data[ch * total ..][0..WIN_SAMPLES]);
            try appendWindow(a, &song, win);
        }
        const keep_n = @min(songSamples(frames.len, c.sample_rate), song[0].items.len);
        var out: [2][]f32 = undefined;
        out[0] = try a.dupe(f32, song[0].items[0..keep_n]);
        errdefer a.free(out[0]);
        out[1] = try a.dupe(f32, song[1].items[0..keep_n]);
        return out;
    }

    /// A whole song → WAV bytes (owned).
    pub fn generate(self: *Engine, a: std.mem.Allocator, req: Request, progress: ?sse.Progress) ![]u8 {
        const p = try self.prompt(a, req.tags, req.lyrics);
        defer a.free(p.ids);
        if (!self.fits(p, req.seconds)) return error.PromptTooLong;
        var smp = req.sampling;
        const frames = try self.generateFrames(a, p, maxFrames(req.seconds), &smp, progress);
        defer a.free(frames);
        log.info("[heartmula] {d} frames ({d:.1}s); codec {d} steps\n", .{ frames.len, @as(f32, @floatFromInt(frames.len)) * FRAME_MS / 1000, req.steps });
        _ = mlx.mlx_clear_cache();
        const pcm = try self.decodeFrames(a, frames, req.steps, req.sampling.seed, progress);
        defer for (pcm) |ch| a.free(ch);
        const inter = try a.alloc(f32, pcm[0].len * 2);
        defer a.free(inter);
        for (pcm[0], pcm[1], 0..) |l, r, i| {
            inter[2 * i] = l;
            inter[2 * i + 1] = r;
        }
        return wav_mod.encodePcm16(a, inter, self.cfg.codec.sample_rate, 2);
    }
};

fn readCodes(x: A) ![8]i32 {
    try ops.eval(x);
    const d = mlx.mlx_array_data_int32(x) orelse return error.NoData;
    return d[0..8].*;
}

/// A frame ends the song when any codebook reaches the audio EOS id.
pub fn endsSong(codes: [8]i32, eos: u32) bool {
    for (codes) |c| if (c >= @as(i32, @intCast(eos))) return true;
    return false;
}

fn readFile(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(a, .limited(16 << 20));
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "heartmula windows: a short song repeats to one window, a long one to the next whole hop" {
    const a = testing.allocator;
    var src: [5][8]i32 = undefined;
    for (&src, 0..) |*f, i| f.* = @splat(@intCast(i));
    const p = try padFrames(a, &src);
    defer a.free(p);
    try testing.expectEqual(WIN_CODES, p.len);
    try testing.expectEqual(@as(i32, 1), p[371][0]); // 371 % 5
    try testing.expectEqual(@as(usize, 1), windowCount(p.len));

    const long = try a.alloc([8]i32, 400);
    defer a.free(long);
    for (long, 0..) |*f, i| f.* = @splat(@intCast(i));
    const q = try padFrames(a, long);
    defer a.free(q);
    try testing.expectEqual(@as(usize, 692), q.len); // ceil(348 / 320) * 320 + 52
    try testing.expectEqual(@as(i32, 0), q[400][0]); // the periodic extension starts over
    try testing.expectEqual(@as(usize, 2), windowCount(q.len));
}

test "heartmula windows: the kept length is the reference's float truncation" {
    try testing.expectEqual(@as(usize, 111359), songSamples(29, 48000)); // int(29 / 12.5 * 48000), not 29 * 3840
    try testing.expectEqual(@as(usize, 3840 * 30), songSamples(30, 48000));
    try testing.expectEqual(@as(u32, 1501), maxFrames(120));
}

test "heartmula crossfade: the overlap ramps from the old window to the new one" {
    try testing.expectEqual(@as(f64, 0), fadeIn(0));
    try testing.expectEqual(@as(f64, 1), fadeIn(OVLP_SAMPLES - 1));
    const a = testing.allocator;
    var song: [2]std.ArrayList(f32) = .{ .empty, .empty };
    defer for (&song) |*ch| ch.deinit(a);
    const ones = try a.alloc(f32, 2 * WIN_SAMPLES);
    defer a.free(ones);
    @memset(ones, 1);
    try appendWindow(a, &song, ones);
    @memset(ones, 3);
    try appendWindow(a, &song, ones);
    try testing.expectEqual(WIN_SAMPLES + HOP_SAMPLES, song[0].items.len);
    try testing.expectEqual(@as(f32, 1), song[1].items[HOP_SAMPLES]);
    try testing.expectEqual(@as(f32, 3), song[1].items[WIN_SAMPLES - 1]);
    try testing.expectApproxEqAbs(@as(f32, 2), song[0].items[HOP_SAMPLES + OVLP_SAMPLES / 2], 1e-4);
}

test "heartmula lower: Python's str.lower on the cased scripts, CJK untouched" {
    const a = testing.allocator;
    const got = try pyLower(a, "Pop, CAFÉ Ŀ Ωmega ДОМ 月光 [Verse]");
    defer a.free(got);
    try testing.expectEqualStrings("pop, café ŀ ωmega дом 月光 [verse]", got);
}

test "heartmula tags: joined the way training wrote them, empties dropped" {
    const a = testing.allocator;
    const got = try joinTags(a, " Pop, Piano ,, female vocal ,\tupbeat ");
    defer a.free(got);
    try testing.expectEqualStrings("Pop,Piano,female vocal,upbeat", got);
}

test "heartmula EOS: any codebook at the EOS id ends the song" {
    try testing.expect(!endsSong(@splat(8192), 8193));
    var f: [8]i32 = @splat(3);
    f[5] = 8195;
    try testing.expect(endsSong(f, 8193));
}

test "heartmula config: the shipped shape parses, other shapes are named errors" {
    const a = testing.allocator;
    const good = shipped_config;
    const p = try parseConfig(a, good);
    p.deinit();
    try testing.expectError(error.HeartMulaConfigInvalid, parseConfig(a, "{}"));
    const bad = try std.mem.replaceOwned(u8, a, good, "\"upsample_kernel_sizes\": [10, 8, 8, 8, 6]", "\"upsample_kernel_sizes\": [10, 8, 8, 8, 7]");
    defer a.free(bad);
    try testing.expectError(error.HeartMulaConfigUnsupported, parseConfig(a, bad));
    const bad2 = try std.mem.replaceOwned(u8, a, good, "\"head_dim\": 384", "\"head_dim\": 256");
    defer a.free(bad2);
    try testing.expectError(error.HeartMulaConfigUnsupported, parseConfig(a, bad2));
}

const shipped_config =
    \\{"model_type": "heartmula", "audio_num_codebooks": 8, "audio_vocab_size": 8197, "text_vocab_size": 128256,
    \\ "backbone": {"num_hidden_layers": 28, "hidden_size": 3072, "num_attention_heads": 24, "num_key_value_heads": 8,
    \\   "head_dim": 128, "intermediate_size": 8192, "max_position_embeddings": 8192, "rope_theta": 500000.0,
    \\   "rms_norm_eps": 1e-05, "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
    \\   "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}},
    \\ "decoder": {"num_hidden_layers": 3, "hidden_size": 3072, "num_attention_heads": 8, "num_key_value_heads": 4,
    \\   "head_dim": 384, "intermediate_size": 8192, "max_position_embeddings": 2048, "rope_theta": 500000.0},
    \\ "codec": {"dim": 512, "codebook_size": 8192, "codebook_dim": 32, "num_quantizers": 8, "attention_head_dim": 64,
    \\   "in_channels": 1024, "num_attention_heads": 24, "num_layers": 24, "num_layers_2": 6, "out_channels": 256,
    \\   "sample_rate": 48000, "num_samples": 2, "upsample_factors": [5, 4, 4, 4, 3],
    \\   "upsample_kernel_sizes": [10, 8, 8, 8, 6], "latent_hidden_dim": 128, "delay_kernel_size": 5,
    \\   "init_channel": 64, "res_kernel_size": 7, "default_kernel_size": 7},
    \\ "text_bos_id": 128000, "text_eos_id": 128001, "audio_eos_id": 8193, "empty_id": 0}
;

// ── env-gated oracles: HEARTMULA_TEST_MODEL = the bf16 pack, HEARTMULA_FIXTURES = tests/dump_heartmula_fixtures.py ──

fn fixturePath(a: std.mem.Allocator, name: []const u8) ![]u8 {
    const dir = std.mem.span(std.c.getenv("HEARTMULA_FIXTURES") orelse return error.SkipZigTest);
    return std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
}

fn readFixture(comptime T: type, a: std.mem.Allocator, name: []const u8) ![]T {
    const dir = std.mem.span(std.c.getenv("HEARTMULA_FIXTURES") orelse return error.SkipZigTest);
    const bytes = try readFile(std.Io.Threaded.global_single_threaded.io(), a, dir, name);
    defer a.free(bytes);
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

fn testEngine() !*Engine {
    const dir = std.mem.span(std.c.getenv("HEARTMULA_TEST_MODEL") orelse return error.SkipZigTest);
    if (std.c.getenv("HEARTMULA_FIXTURES") == null) return error.SkipZigTest;
    return Engine.load(std.Io.Threaded.global_single_threaded.io(), testing.allocator, dir, mlx.mlx_default_gpu_stream_new());
}

/// Cosine AND scale (cosine cannot see a scale error).
fn expectParity(arr: A, ref: []const f32, min_cos: f64, label: []const u8) !void {
    return expectParityBand(arr, ref, min_cos, 0.01, label);
}

fn expectParityBand(arr: A, ref: []const f32, min_cos: f64, band: f64, label: []const u8) !void {
    var g = Graph.init(testing.allocator, mlx.mlx_default_gpu_stream_new());
    defer g.deinit();
    const flat = try g.contiguous(try g.astype(arr, .float32));
    try ops.eval(flat);
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
    std.debug.print("[heartmula-test] {s}: cos {d:.6} rms_ratio {d:.4}\n", .{ label, cos, rms });
    try testing.expect(cos >= min_cos);
    try testing.expect(@abs(rms - 1) < band);
}

fn argmaxOf(x: []const f32) usize {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return best;
}

test "heartmula oracle: the prompt ids are byte-exact" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(std.c.getenv("HEARTMULA_FIXTURES").?);
    for ([_][]const u8{ "a", "b", "c" }) |k| {
        const tn = try std.fmt.allocPrint(a, "tags_{s}.txt", .{k});
        defer a.free(tn);
        const ln = try std.fmt.allocPrint(a, "lyrics_{s}.txt", .{k});
        defer a.free(ln);
        const iname = try std.fmt.allocPrint(a, "ids_{s}.i32", .{k});
        defer a.free(iname);
        const tags = try readFile(io, a, dir, tn);
        defer a.free(tags);
        const lyrics = try readFile(io, a, dir, ln);
        defer a.free(lyrics);
        const want = try readFixture(i32, a, iname);
        defer a.free(want);
        const p = try e.prompt(a, tags, lyrics);
        defer a.free(p.ids);
        try testing.expectEqual(want.len, p.ids.len);
        for (want, p.ids) |x, y| try testing.expectEqual(@as(u32, @intCast(x)), y);
    }
}

fn samplePrompt(a: std.mem.Allocator, e: *const Engine) !Prompt {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(std.c.getenv("HEARTMULA_FIXTURES").?);
    const tags = try readFile(io, a, dir, "tags_a.txt");
    defer a.free(tags);
    const lyrics = try readFile(io, a, dir, "lyrics_a.txt");
    defer a.free(lyrics);
    return e.prompt(a, tags, lyrics);
}

/// Teacher-forces the first `n` reference frames (greedy picks compared, then replaced); the codes that agreed.
fn replayFrames(e: *Engine, g: *Graph, p: Prompt, codes: []const i32, n: usize, taps: ?*std.ArrayList(A)) !u32 {
    const a = g.a;
    var bkv = try Kv.init(a, e.s, e.cfg.backbone, 2, p.ids.len + n);
    defer bkv.deinit();
    var dkv = try Kv.init(a, e.s, e.cfg.decoder, 2, 8);
    defer dkv.deinit();
    var smp: Sampling = .{ .greedy = true };
    var agree: u32 = 0;
    var fg = g.scope();
    defer fg.deinit();
    var h = try e.promptEmbeds(&fg, p, 2);
    for (0..n) |fi| {
        var sg = fg.scope();
        defer sg.deinit();
        const out = try e.frame(&sg, h, &bkv, &dkv, &smp, .{ .force = codes[fi * 8 ..][0..8], .agree = &agree, .taps = taps, .tap_graph = g });
        const next = try fg.keep(try e.frameEmbed(&sg, out, 2));
        try ops.eval(next);
        fg.drop(h);
        h = next;
    }
    return agree;
}

test "heartmula oracle: prefill logits and 4 frames of guided logits match the reference" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const p = try samplePrompt(a, e);
    defer a.free(p.ids);
    const c0 = try readFixture(f32, a, "c0_prefill.f32");
    defer a.free(c0);
    const guided_ref = try readFixture(f32, a, "guided.f32");
    defer a.free(guided_ref);
    const codes = try readFixture(i32, a, "codes.i32");
    defer a.free(codes);
    const v = e.cfg.audio_vocab_size;
    var g = Graph.init(a, e.s);
    defer g.deinit();
    {
        var pkv = try Kv.init(a, e.s, e.cfg.backbone, 2, p.ids.len);
        defer pkv.deinit();
        var sg = g.scope();
        defer sg.deinit();
        const last = try e.backbone.forward(&sg, try e.promptEmbeds(&sg, p, 2), &pkv);
        try expectParity(try project(&sg, &e.c0_head, last), c0, 0.999, "c0 logits after the prompt (both rows)");
    }
    var taps: std.ArrayList(A) = .empty;
    defer taps.deinit(a);
    _ = try replayFrames(e, &g, p, codes, 4, &taps);
    for (taps.items, 0..) |t, i| {
        const ref = guided_ref[i * v ..][0..v];
        var label_buf: [64]u8 = undefined;
        // the guide 3c - 2u amplifies bf16 noise ~5x: a 3% scale band, the raw logits keep 1%
        try expectParityBand(t, ref, 0.999, 0.03, try std.fmt.bufPrint(&label_buf, "guided logits frame {d} codebook {d}", .{ i / 8, i % 8 }));
        const flat = try g.contiguous(t);
        try ops.eval(flat);
        // the reference's pick is ours, or a near-tie with it
        const d = mlx.mlx_array_data_float32(flat).?[0..v];
        try testing.expect(d[argmaxOf(d)] - d[argmaxOf(ref)] < 0.5);
    }
}

test "heartmula oracle: the greedy replay agrees with the fp32 reference as often as PyTorch bf16 does" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    const p = try samplePrompt(a, e);
    defer a.free(p.ids);
    const codes = try readFixture(i32, a, "codes.i32");
    defer a.free(codes);
    var g = Graph.init(a, e.s);
    defer g.deinit();
    const agree = try replayFrames(e, &g, p, codes, codes.len / 8, null);
    const rate = @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(codes.len));
    std.debug.print("[heartmula-test] greedy replay: {d}/{d} codes agree ({d:.1}%)\n", .{ agree, codes.len, rate * 100 });
    // PyTorch bf16 on MPS scores 90.6% on the same 200 frames (`dump_heartmula_fixtures.py --replay-bf16`)
    try testing.expect(rate >= 0.85);
}

test "heartmula oracle: RVQ condition, one DiT velocity, a 3-step in-context solve and the scalar decoder" {
    const a = testing.allocator;
    const e = try testEngine();
    defer e.deinit();
    var g = Graph.init(a, e.s);
    defer g.deinit();

    const rc = try readFixture(i32, a, "rvq_codes.i32");
    defer a.free(rc);
    const n = rc.len / 8;
    const frames = try a.alloc([8]i32, n);
    defer a.free(frames);
    for (frames, 0..) |*f, t| for (f, 0..) |*x, q| {
        x.* = rc[q * n + t];
    };
    const rvq = try readFixture(f32, a, "rvq_out.f32");
    defer a.free(rvq);
    try expectParity(try e.codec.condition(&g, frames), rvq, 0.99999, "rvq condition");

    const din = try readFixture(f32, a, "dit_in.f32");
    defer a.free(din);
    const dout = try readFixture(f32, a, "dit_out.f32");
    defer a.free(dout);
    try expectParity(try e.codec.velocity(&g, try g.fromSlice(f32, din, &.{ 2, 96, 1024 }), 0.3), dout, 0.999, "dit velocity t=0.3");

    var arrs: [4]A = undefined;
    for ([_][]const u8{ "eu_noise.f32", "eu_ctx.f32", "eu_mu.f32" }, [_]c_int{ 256, 256, 512 }, 0..) |name, w, i| {
        const d = try readFixture(f32, a, name);
        defer a.free(d);
        arrs[i] = try g.fromSlice(f32, d, &.{ 1, 96, w });
    }
    const eu = try readFixture(f32, a, "eu_out.f32");
    defer a.free(eu);
    arrs[3] = try e.codec.solve(&g, arrs[0], arrs[1], 24, arrs[2], 3, 1.25, null, 0, 3);
    try expectParity(arrs[3], eu, 0.999, "3-step euler with 24 in-context rows");

    const sq_in = try readFixture(f32, a, "sq_in.f32");
    defer a.free(sq_in);
    const sq_out = try readFixture(f32, a, "sq_out.f32");
    defer a.free(sq_out);
    // NAX-enabled MLX rounds f32 convs to TF32: cosine plus a 1% scale band, as yue2's VAE.
    try expectParity(try (Scalar{ .w = &e.cw }).decode(&g, try g.fromSlice(f32, sq_in, &.{ 2, 24, 128 }), e.cfg.codec), sq_out, 0.999, "scalar decoder");
}
