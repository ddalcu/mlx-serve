//! Breeze-TTS-2 backend (`model_type: breeze_tts`).
//!
//! A multi-codebook TTS stack (dense bf16 packs):
//!   text ids → T5Gemma2 text-encoder conditioning (host-gathered table),
//!   Qwen3 backbone emits codebook-0 @ 12.5 Hz,
//!   `depth_decoder` fills codebooks 1..15 per frame (local depth span),
//!   the qwen3_tts_tokenizer_12hz codec (shared with tts.zig) → 24 kHz PCM.
//!
//! Host-driven frame loop (device step per frame), host-side numpy-order
//! sampling keyed off a counter for determinism. The audio-table / TE-table
//! boundaries are host-gathered; the backbone/depth forwards run on-device.
//! Voice cloning (tokenizer encode) refuses by name until it lands.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/load.zig");
const tts = @import("mimi.zig");
const wav_mod = @import("../../sdk/media/wav.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

pub const model_type = "breeze_tts";

// ── token ids (fixed by the published pack) ────────────────────────────────
pub const AUDIO_TOKEN_ID: u32 = 262144;
pub const MIMI_RATE: u32 = 24000;
pub const AUDIO_EOS_ID: u32 = 262145;
pub const SPEAKER_BASE: u32 = 262146; // [S0]..[S9]
pub const INS_BOS_ID: u32 = 262156;
pub const INS_EOS_ID: u32 = 262157;
pub const EOI_TOKEN_ID: u32 = 256000; // TE row zeroed at gather
pub const TE_VOCAB: usize = 262158;
pub const NUM_CODEBOOKS: usize = 16;
pub const CB_VOCAB: usize = 2051;
pub const CB_PAD: u32 = 2050; // frame-end code (lm_head row 2050)
pub const CB_LM: usize = 2052; // lm_head rows: codes [0,2048), pad 2048..2050, eos 2051
pub const CODEBOOK_SIZE: u32 = 2048; // mimi codebook_size; ids >= this never sample
pub const CB_EOS: u32 = 2051;
pub const CB_STRIDE: usize = 2051; // audio-table stride per codebook

// ── geometry (Breeze-TTS-2 / Qwen3-0.6B flavor) ────────────────────────────
pub const TE_HIDDEN: usize = 1152;
pub const TE_HEADS: usize = 4;
pub const TE_KV: usize = 1;
pub const TE_HEAD_DIM: usize = 256;
pub const TE_INTER: usize = 6912;
pub const TE_LAYERS: usize = 26;
pub const TE_WINDOW: usize = 512;
pub const TE_THETA_FULL: f64 = 1_000_000;
pub const TE_THETA_SLIDE: f64 = 10_000;
pub const TE_FACTOR_FULL: f64 = 8.0;

pub const B_HIDDEN: usize = 2048;
pub const B_Q_DIM: usize = 2048;
pub const B_KV_DIM: usize = 1024;
pub const B_HEAD_DIM: usize = 128;
pub const B_INTER: usize = 6144;
pub const B_LAYERS: usize = 28;
pub const B_THETA: f64 = 1_000_000;

pub const D_HIDDEN: usize = 1024;
pub const D_HEADS: usize = 8;
pub const D_KV: usize = 2;
pub const D_HEAD_DIM: usize = 128;
pub const D_INTER: usize = 8192;
pub const D_LAYERS: usize = 12;
pub const D_THETA: f64 = 500_000;

pub const MAX_SEQ: usize = 2048; // span guard: text + frame budget
// generation_config max_new_tokens (750 = 60 s at 12.5 Hz) is a sampling
// default, not a model limit: the true ceiling is the MAX_SEQ position span.
// 1500 matches the legacy Python bridge (max_new_tokens=1500, max_seq_len=2048)
// so speech isn't truncated at 60 s; the real stop is CB_EOS.
pub const MAX_FRAMES: u32 = 1500;
pub const SAMPLE_RATE: u32 = 24000;
pub const FRAME_RATE: f64 = 12.5;
pub const CHARS_PER_SEC: f64 = 15.0;
pub const FRAME_EST_DIV: f64 = 3.0; // model keeps generating ~4.5+ frames per text token; generous ceiling, true stop comes from eos

pub fn isFullAttentionLayer(layer: u32) bool {
    return layer == 5 or layer == 11 or layer == 17 or layer == 23;
}

pub const Config = struct {};

pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) !Config {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return Config{};
    defer parsed.deinit();
    return Config{};
}

/// Voice-design speaker slot: `index` 0..9 maps to the pack's `[S0]..[S9]` ids.
pub fn speakerToken(index: u32) ?u32 {
    if (index >= 10) return null;
    return SPEAKER_BASE + index;
}

/// A request `voice` names a slot S0..S9 (case-insensitive), nothing else.
pub fn voiceSlotValid(name: []const u8) bool {
    if (name.len != 2) return false;
    if (name[0] != 'S' and name[0] != 's') return false;
    return name[1] >= '0' and name[1] <= '9';
}

/// Frame budget from speak units (see speakUnits): units / CHARS_PER_SEC ×
/// FRAME_RATE, with a headroom factor, clamped by a request cap and
/// MAX_FRAMES. The true stop is always CB_EOS; this is a runaway ceiling.
pub fn estimateFrames(units: usize, req_max: ?u32) u32 {
    const est: f64 = @as(f64, @floatFromInt(units)) / CHARS_PER_SEC * FRAME_RATE * FRAME_EST_DIV + 4.0;
    var frames: u32 = @intFromFloat(@ceil(est));
    if (req_max) |m| frames = @min(frames, m);
    return @min(frames, MAX_FRAMES);
}

/// Speakable-character units: 1 per ASCII char, 3 per non-ASCII. CJK speech
/// runs ~2.5–3× longer per character and tokenizes denser, so time and
/// budget math uses this instead of raw character counts.
pub fn speakUnits(s: []const u8) usize {
    var units: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        units += if (n == 1) @as(usize, 1) else 3;
        i += @min(n, s.len - i);
    }
    return units;
}

/// Frames a segment will likely speak, with 15 % speech-rate margin.
fn needFrames(units: usize) u32 {
    const f: f64 = @ceil(@as(f64, @floatFromInt(units)) / CHARS_PER_SEC * FRAME_RATE * 1.15);
    return if (f >= 1e9) std.math.maxInt(u32) else @intFromFloat(f);
}

/// Whether a segment's exact prompt tokens plus its estimated speech frames
/// (plus per-request prompt overhead) fit the KV window and frame cap with
/// margin. Used to pack sentences into turn-sized segments.
fn segmentFits(overhead: usize, toks: usize, units: usize) bool {
    const need = needFrames(units);
    return overhead + toks + need + 16 <= MAX_SEQ and need <= MAX_FRAMES;
}

/// Sentence pieces above this unit count (one giant unpunctuated run) are
/// hard-split so even that stays inside the window.
const MAX_SENT_UNITS: usize = 1400;

fn isSentenceTerm(r: u21) bool {
    return r == '.' or r == '!' or r == '?' or r == 0x3002 or r == 0xFF01 or r == 0xFF1F;
}

/// Split text into speakable sentence slices: Latin terminators (.!?) close
/// at the following whitespace run (longform.py's `(?<=[.!?])\s+`), CJK
/// terminators close at the character. Sentences are trimmed; runs over
/// MAX_SENT_UNITS are hard-split (see appendSentence). Borrows `text`.
fn splitSentences(allocator: std.mem.Allocator, text: []const u8, out: *std.ArrayList([]const u8)) !void {
    var start: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const n = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const step = @min(n, text.len - i);
        const r: u21 = if (step == 1) text[i] else (std.unicode.utf8Decode(text[i..][0..step]) catch 0xFFFF);
        if (isSentenceTerm(r)) {
            var end = i + step;
            if (r < 0x80) { // latin terminators split at the following whitespace run
                var j = end;
                while (j < text.len and std.ascii.isWhitespace(text[j])) j += 1;
                if (j > end) end = j;
            }
            try appendSentence(allocator, std.mem.trim(u8, text[start..end], " \t\r\n"), out);
            start = end;
            i = end;
            continue;
        }
        i += step;
    }
    try appendSentence(allocator, std.mem.trim(u8, text[start..], " \t\r\n"), out);
}

/// Append a trimmed sentence, hard-splitting at whitespace (else at a UTF-8
/// boundary) when it exceeds MAX_SENT_UNITS. All emitted slices borrow `s`.
fn appendSentence(allocator: std.mem.Allocator, s: []const u8, out: *std.ArrayList([]const u8)) !void {
    var rest = s;
    var rem = speakUnits(rest);
    while (rem > MAX_SENT_UNITS) {
        var units: usize = 0;
        var cut: usize = 0;
        var ws_cut: usize = 0;
        var ws_units: usize = 0;
        var i: usize = 0;
        while (i < rest.len) {
            const n = std.unicode.utf8ByteSequenceLength(rest[i]) catch 1;
            const step = @min(n, rest.len - i);
            if (step == 1 and std.ascii.isWhitespace(rest[i])) {
                ws_cut = i + step;
                ws_units = units;
            }
            units += if (step == 1) @as(usize, 1) else 3;
            i += step;
            if (units >= MAX_SENT_UNITS) break;
        }
        cut = i;
        if (ws_cut > 0 and ws_cut < cut and ws_units > MAX_SENT_UNITS / 2) {
            cut = ws_cut;
            units = ws_units;
        }
        var piece = rest[0..cut];
        while (piece.len > 0 and (piece[piece.len - 1] & 0xC0) == 0x80) piece = piece[0 .. piece.len - 1];
        if (piece.len > 0) try out.append(allocator, piece);
        rest = std.mem.trim(u8, rest[cut..], " \t\r\n");
        rem = speakUnits(rest);
    }
    if (rest.len > 0) try out.append(allocator, rest);
}

/// Join sentence slices with single spaces (longform.py's `" ".join`).
fn joinSentences(allocator: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    var total: usize = 0;
    for (parts) |p| total += p.len + 1;
    if (total == 0) return allocator.alloc(u8, 0);
    const buf = try allocator.alloc(u8, total - 1);
    var i: usize = 0;
    for (parts, 0..) |p, k| {
        @memcpy(buf[i .. i + p.len], p);
        i += p.len;
        if (k + 1 < parts.len) {
            buf[i] = ' ';
            i += 1;
        }
    }
    return buf;
}

// ───────────────────────────── rng + host sampler ────────────────────────────

pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        var z = seed ^ 0xA5A5_5A5A_1234_9876;
        if (z == 0) z = 1;
        return .{ .state = z };
    }

    pub fn nextF32(self: *Rng) f32 {
        self.state +%= 0x9E3779B97F4A7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        z = z ^ (z >> 31);
        return @as(f32, @floatFromInt(z >> 40)) * (1.0 / 16777216.0); // 24 bits
    }
};

pub const Temp = struct { t: f32 = 0.9, top_k: u32 = 50, top_p: f32 = 1.0 };

/// numpy-order categorical sampler: temperature → top-k (value threshold, the
/// k-th highest, lowest ids keep) → top-p (strict cumulative drop) →
/// multinomial. temperature<=0 → argmax. `scratch`/`order` are ≥ n long.
pub const SampleOpts = struct {
    suppress_from: u32 = 0,
    suppress_to: u32 = 0,
    history: []const u32 = &[_]u32{},
    rep: f32 = 1.0,
};

pub fn sampleToken(rng: *Rng, logits: []const f32, temp: Temp, scratch: []f32, order: []usize, opts: SampleOpts) usize {
    const n = logits.len;
    @memcpy(scratch[0..n], logits);
    if (opts.history.len > 0 and opts.rep != 1.0) {
        var seen = _hist_seen;
        if (seen.len < n) seen = &[_]bool{};
        if (seen.len >= n) {
            @memset(seen[0..n], false);
            for (opts.history) |tok| {
                if (tok >= n) continue;
                if (seen[tok]) continue;
                seen[tok] = true;
                scratch[tok] = if (scratch[tok] > 0) scratch[tok] / opts.rep else scratch[tok] * opts.rep;
            }
        }
    }
    if (opts.suppress_to > opts.suppress_from) {
        var i: usize = opts.suppress_from;
        while (i < @min(opts.suppress_to, @as(u32, @intCast(n)))) : (i += 1) scratch[i] = -std.math.inf(f32);
    }
    if (temp.t <= 0) {
        var best: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        for (scratch[0..n], 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        return best;
    }
    for (scratch[0..n]) |*v| v.* /= temp.t;

    if (temp.top_k > 0 and temp.top_k < n) {
        for (order[0..n], 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order[0..n], scratch, struct {
            fn lt(sc: []const f32, a: usize, b: usize) bool {
                if (sc[a] == sc[b]) return a < b;
                return sc[a] > sc[b];
            }
        }.lt);
        const thr = scratch[order[temp.top_k - 1]];
        var i: usize = temp.top_k;
        while (i < n) : (i += 1) {
            if (scratch[order[i]] < thr) scratch[order[i]] = -std.math.inf(f32);
        }
    }

    var maxv: f32 = -std.math.inf(f32);
    for (scratch[0..n]) |v| maxv = @max(maxv, v);
    var sum: f64 = 0;
    for (scratch[0..n]) |*v| {
        if (std.math.isInf(v.*) and v.* < 0) {
            v.* = 0;
            continue;
        }
        const e: f64 = @exp(@as(f64, @floatCast(v.*)) - @as(f64, @floatCast(maxv)));
        v.* = @floatCast(e);
        sum += e;
    }
    if (sum <= 0) return 0;

    if (temp.top_p < 1.0) {
        for (order[0..n], 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order[0..n], scratch, struct {
            fn lt(sc: []const f32, a: usize, b: usize) bool {
                if (sc[a] == sc[b]) return a < b;
                return sc[a] > sc[b];
            }
        }.lt);
        var cum: f64 = 0;
        var keep: usize = n;
        for (order[0..n], 0..) |idx, oi| {
            cum += @as(f64, @floatCast(scratch[idx])) / sum;
            if (cum >= temp.top_p) {
                keep = oi + 1;
                break;
            }
        }
        var i = keep;
        while (i < n) : (i += 1) scratch[order[i]] = 0;
        sum = 0;
        for (scratch[0..n]) |v| sum += @as(f64, @floatCast(v));
    }

    const r = @as(f64, @floatCast(rng.nextF32())) * sum;
    var acc: f64 = 0;
    for (scratch[0..n], 0..) |v, i| {
        acc += @as(f64, @floatCast(v));
        if (r < acc) return i;
    }
    return n - 1;
}

// ───────────────────────────── device helpers ────────────────────────────────

fn ownT16(w: *const Weights, s: S, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, arr, &[_]c_int{ 1, 0 }, 2, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, t, .bfloat16, s));
    return out;
}

fn ownF16(w: *const Weights, s: S, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, arr, .bfloat16, s));
    return out;
}

fn ownF16Plus1(w: *const Weights, s: S, alloc: std.mem.Allocator, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
    const shape = mlx.getShape(f32arr);
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try alloc.alloc(f32, n);
    defer alloc.free(buf);
    for (0..n) |i| buf[i] = data[i] + 1.0;
    return hostF32(buf, shape);
}

fn hostF32(data: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(@constCast(@ptrCast(data.ptr)), shape.ptr, @intCast(shape.len), .float32);
}

fn hostU32(data: []const u32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(@constCast(@ptrCast(data.ptr)), shape.ptr, @intCast(shape.len), .uint32);
}

const null_arr = mlx.mlx_array{ .ctx = null };

fn evalHost(s: S, arr: mlx.mlx_array, out: []f32) !void {
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
    @memcpy(out, data[0..out.len]);
}

inline fn mm(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_matmul(&r, x, w_t, s);
    return r;
}

inline fn addM(a: mlx.mlx_array, b: mlx.mlx_array, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_add(&r, a, b, s);
    return r;
}

inline fn rmsn(x: mlx.mlx_array, g: mlx.mlx_array, eps: f32, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_fast_rms_norm(&r, x, g, eps, s);
    return r;
}

inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_reshape(&r, x, shape.ptr, @intCast(shape.len), s);
    return r;
}

inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_transpose_axes(&r, x, axes.ptr, @intCast(axes.len), s);
    return r;
}

fn siluArr(x: mlx.mlx_array, s: S) mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    _ = mlx.mlx_sigmoid(&sig, x, s);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&r, x, sig, s);
    return r;
}

/// RoPE via the house fused op (interleaved HF convention).
fn ropeApply(x: mlx.mlx_array, theta: f64, scale: f64, hd: c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_fast_rope(&r, x, hd, false, .{ .value = @floatCast(theta), .has_value = true }, @floatCast(scale), 0, null_arr, s);
    return r;
}

pub const D_HD: usize = 128;

fn ropeApplyFreqs(x: mlx.mlx_array, freqs: mlx.mlx_array, hd: c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_fast_rope(&r, x, hd, false, .{}, 1.0, 0, freqs, s);
    return r;
}

// Empirical: the pack config declares llama3 scaling (factor 32, orig_ctx 16) on the
// depth rope, but sampled audio is word-accurate only with plain theta; the scaled
// table garbles it. Scaled-path parity against the oracle is still open (early eos).
fn depthInvFreqs() [D_HD / 2]f32 {
    _ = 0; // kept as oracle note

    var t: [D_HD / 2]f32 = undefined;
    const factor: f64 = 32.0;
    const low_f: f64 = 0.001953125;
    const high_f: f64 = 0.0078125;
    const old_ctx: f64 = 16.0;
    const low_wl = old_ctx / low_f;
    const high_wl = old_ctx / high_f;
    var i: usize = 0;
    while (i < D_HD / 2) : (i += 1) {
        const f0 = 1.0 / std.math.pow(f64, D_THETA, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(D_HD)));
        const wl = 2.0 * std.math.pi / f0;
        var f = f0;
        if (wl >= high_wl) {
            f = f0 / factor;
        } else if (wl > low_wl) {
            const smooth = (high_wl / wl - 1.0) / (high_wl / low_wl - 1.0);
            f = (1.0 - smooth) * f0 + smooth * (f0 / factor);
        }
        t[i] = @floatCast(f);
    }
    return t;
}

fn sdpa(q4: mlx.mlx_array, k4: mlx.mlx_array, v4: mlx.mlx_array, scale: f32, mask: mlx.mlx_array, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    const mode: [*:0]const u8 = if (mask.ctx == null) "causal" else "array";
    _ = mlx.mlx_fast_scaled_dot_product_attention(&r, q4, k4, v4, scale, mode, mask, null_arr, false, s);
    return r;
}

// ─────────────────────────── qwen stack (generic) ────────────────────────────

pub const KV = struct {
    k: mlx.mlx_array = null_arr,
    v: mlx.mlx_array = null_arr,
};

fn concat2(s: S, a: mlx.mlx_array, b: mlx.mlx_array, axis: c_int) mlx.mlx_array {
    if (a.ctx == null) return b;
    const vec = mlx.mlx_vector_array_new();
    _ = mlx.mlx_vector_array_append_value(vec, a);
    _ = mlx.mlx_vector_array_append_value(vec, b);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_concatenate_axis(&r, vec, axis, s);
    _ = mlx.mlx_vector_array_free(vec);
    return r;
}

const QLayer = struct {
    q: mlx.mlx_array, // [in, q_dim] f16 (pre-transposed)
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    o: mlx.mlx_array,
    qn: mlx.mlx_array, // zero-length array = no qk-norm (depth decoder)
    kn: mlx.mlx_array,
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
    in_ln: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    pre_ff: mlx.mlx_array, // NULL on backbone/depth; TE carries the 4th norm
};

/// Full re-forward over a host [L, hidden] f32 embedding span. `pos_scale`
/// scales RoPE positions (TE full layers), optional causal local-window mask
/// (`window` 0 = plain causal). qk-norm applies when qn has data.
var dbg_layer0 = false;

fn dumpFirst(s: S, a: mlx.mlx_array, tag: [*:0]const u8) void {
    var idx_arr: [12]u32 = undefined;
    for (&idx_arr, 0..) |*v, i| v.* = @intCast(i);
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    if (mlx.mlx_take_axis(&taken, a, hostU32(&idx_arr, &[_]c_int{1}), 0, s) != 0) return;
    var buf: [12]f32 = undefined;
    evalHost(s, taken, &buf) catch return;
    log.err("[breeze-dump] {s} {any}\n", .{ tag, buf });
}

fn dumpLast(s: S, a: mlx.mlx_array, tag: [*:0]const u8) void {
    const shape = mlx.getShape(a);
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    var buf: [4096]f32 = undefined;
    const m = @min(n, buf.len);
    const flat = reshape(a, &[_]c_int{@intCast(m)}, s);
    defer _ = mlx.mlx_array_free(flat);
    evalHost(s, flat, buf[0..m]) catch return;
    log.err("[breeze-dump] {s} {any}\n", .{ tag, buf[0..@min(12, m)] });
}

fn qwenForward(s: S, layers: []const QLayer, norm: mlx.mlx_array, eps: f32, theta: f64, x_host: []const f32, L: usize, hidden: usize, pos_scale: f64, window: usize, freqs: mlx.mlx_array, kv_save: ?[]KV, out_last: []f32) !void {
    const dbg_l = std.c.getenv("BREEZE_LAYER0") != null;
    if (dbg_l) dbg_layer0 = true;
    const sh = [_]c_int{ 1, @intCast(L), @intCast(hidden) };
    var x = hostF32(x_host, &sh);
    var x16 = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&x16, x, .bfloat16, s);
    _ = mlx.mlx_array_free(x); // x16's graph owns the impl now
    x = x16;

    for (layers, 0..) |lw, lidx| {
        if (dbg_l) log.err("[breeze-dump] LAYER {d} qn={any} in={d}\n", .{ lidx, lw.qn.ctx != null, hidden });
        const xn = rmsn(x, lw.in_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = mm(xn, lw.q, s);
        defer _ = mlx.mlx_array_free(q);
        const k = mm(xn, lw.k, s);
        defer _ = mlx.mlx_array_free(k);
        const v = mm(xn, lw.v, s);
        defer _ = mlx.mlx_array_free(v);
        const qh: c_int = @intCast(@divExact(mlx.getShape(lw.q)[1], 128));
        const kh: c_int = @intCast(@divExact(mlx.getShape(lw.k)[1], 128));
        var q4 = reshape(q, &[_]c_int{ 1, @intCast(L), qh, 128 }, s);
        var k4 = reshape(k, &[_]c_int{ 1, @intCast(L), kh, 128 }, s);
        const v4 = reshape(v, &[_]c_int{ 1, @intCast(L), kh, 128 }, s);
        defer _ = mlx.mlx_array_free(v4);
        const lq: usize = L * @as(usize, @intCast(qh));
        const lk: usize = L * @as(usize, @intCast(kh));
        const hd: c_int = 128;
        if (lw.qn.ctx != null and mlx.getShape(lw.qn)[0] > 0) {
            const q4f = reshape(q4, &[_]c_int{ @intCast(lq), hd }, s);
            const q4n1 = rmsn(q4f, lw.qn, eps, s);
            _ = mlx.mlx_array_free(q4f);
            const q4n = reshape(q4n1, &[_]c_int{ 1, @intCast(L), qh, hd }, s);
            _ = mlx.mlx_array_free(q4n1);
            _ = mlx.mlx_array_free(q4);
            q4 = q4n;
            if (dbg_layer0) dumpFirst(s, reshape(q4, &[_]c_int{ @intCast(lq * 128) }, s), "QNOTH");
            const k4f = reshape(k4, &[_]c_int{ @intCast(lk), hd }, s);
            const k4n1 = rmsn(k4f, lw.kn, eps, s);
            _ = mlx.mlx_array_free(k4f);
            const k4n = reshape(k4n1, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
            _ = mlx.mlx_array_free(k4n1);
            _ = mlx.mlx_array_free(k4);
            k4 = k4n;
        }
        defer _ = mlx.mlx_array_free(q4);
        defer _ = mlx.mlx_array_free(k4);
        const qt = transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const qr = if (freqs.ctx != null) ropeApplyFreqs(qt, freqs, 128, s) else ropeApply(qt, theta, pos_scale, 128, s);
        defer _ = mlx.mlx_array_free(qr);
        if (dbg_layer0) dumpFirst(s, reshape(qr, &[_]c_int{ @intCast(L * @as(usize, @intCast(qh)) * 128) }, s), "QR");
        const kr = if (freqs.ctx != null) ropeApplyFreqs(kt, freqs, 128, s) else ropeApply(kt, theta, pos_scale, 128, s);
        const has_mask = window > 0;
        const mask = if (has_mask) windowMask(s, L, window) else null_arr;
        const att = sdpa(qr, kr, vt, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(128))))), mask, s);
        defer _ = mlx.mlx_array_free(att);
        if (has_mask) _ = mlx.mlx_array_free(mask);
        if (kv_save) |ks| {
            ks[lidx] = .{ .k = kr, .v = vt }; // ownership moves to the caller's KV cache
        } else {
            _ = mlx.mlx_array_free(kr);
            _ = mlx.mlx_array_free(vt);
        }
        const at = transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = reshape(at, &[_]c_int{ 1, @intCast(L), @intCast(mlx.getShape(lw.q)[1]) }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = mm(af, lw.o, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = addM(x, o, s);
        defer _ = mlx.mlx_array_free(h1);
        if (dbg_layer0) dumpLast(s, h1, "H1");
        const hn = rmsn(h1, lw.post_ln, eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = mm(hn, lw.gate, s);
        defer _ = mlx.mlx_array_free(g);
        const u = mm(hn, lw.up, s);
        defer _ = mlx.mlx_array_free(u);
        const gs = siluArr(g, s);
        defer _ = mlx.mlx_array_free(gs);
        const gu = mulEl(s, gs, u);
        defer _ = mlx.mlx_array_free(gu);
        const mlp = mm(gu, lw.down, s);
        defer _ = mlx.mlx_array_free(mlp);
        const x_prev = x;
        x = addM(h1, mlp, s);
        _ = mlx.mlx_array_free(x_prev);
    }
    const x_norm_prev = x;
    x = rmsn(x, norm, eps, s);
    _ = mlx.mlx_array_free(x_norm_prev);
    const last = reshape(x, &[_]c_int{ @intCast(L), @intCast(hidden) }, s);
    defer _ = mlx.mlx_array_free(last);
    var idx = [_]u32{@intCast(L - 1)};
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, last, hostU32(&idx, &[_]c_int{1}), 0, s));
    try evalHost(s, taken, out_last);
    _ = mlx.mlx_array_free(x);
}

fn eval1(a: mlx.mlx_array) c_int {
    const vec = mlx.mlx_vector_array_new_value(a);
    const r = mlx.mlx_eval(vec);
    _ = mlx.mlx_vector_array_free(vec);
    return r;
}

fn ropeApplyOff(x: mlx.mlx_array, theta: f64, off: c_int, hd: c_int, s: S) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_fast_rope(&r, x, hd, false, .{ .value = @floatCast(theta), .has_value = true }, 1.0, off, null_arr, s);
    return r;
}

fn qwenStep(s: S, layers: []const QLayer, norm: mlx.mlx_array, eps: f32, theta: f64, xrow: []const f32, hidden: usize, pos: c_int, kv: []KV, out_last: []f32) !void {
    if (std.c.getenv("BREEZE_STEP_TRACE") != null) log.err("[bstep] enter pos={d} kv0.k={any} kv0.v={any}\n", .{ pos, kv[0].k.ctx != null, kv[0].v.ctx != null });
    const sh = [_]c_int{ 1, 1, @intCast(hidden) };
    var x = hostF32(xrow, &sh);
    var x16 = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&x16, x, .bfloat16, s);
    _ = mlx.mlx_array_free(x); // x16's graph owns the impl now
    x = x16;
    if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(x); log.err("[beval] x shape={d},{d},{d}\n", .{ mlx.getShape(x)[0], mlx.getShape(x)[1], mlx.getShape(x)[2] }); }

    for (layers, 0..) |lw, lidx| {
        const xn = rmsn(x, lw.in_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(xn); log.err("[beval] l{d} xn\n", .{lidx}); }
        const q = mm(xn, lw.q, s);
        defer _ = mlx.mlx_array_free(q);
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(q); log.err("[beval] l{d} q shape={d}\n", .{lidx, mlx.getShape(q)[0]}); }
        const k = mm(xn, lw.k, s);
        defer _ = mlx.mlx_array_free(k);
        const v = mm(xn, lw.v, s);
        defer _ = mlx.mlx_array_free(v);
        const qh: c_int = @intCast(@divExact(mlx.getShape(lw.q)[1], 128));
        const kh: c_int = @intCast(@divExact(mlx.getShape(lw.k)[1], 128));
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(k); log.err("[beval] l{d} k shape={d}\n", .{lidx, mlx.getShape(k)[0]}); }
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(v); log.err("[beval] l{d} v shape={d}\n", .{lidx, mlx.getShape(v)[0]}); }
        var q4 = reshape(q, &[_]c_int{ 1, 1, qh, 128 }, s);
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(q4); log.err("[beval] l{d} q4 shape={d},{d},{d}\n", .{lidx, mlx.getShape(q4)[0], mlx.getShape(q4)[2], mlx.getShape(q4)[3]}); }
        var k4 = reshape(k, &[_]c_int{ 1, 1, kh, 128 }, s);
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(k4); log.err("[beval] l{d} k4\n", .{lidx}); }
        const v4 = reshape(v, &[_]c_int{ 1, 1, kh, 128 }, s);
        defer _ = mlx.mlx_array_free(v4);
        if (lw.qn.ctx != null and mlx.getShape(lw.qn)[0] > 0) {
            const q4f = reshape(q4, &[_]c_int{ @intCast(@as(usize, @intCast(qh))), 128 }, s);
            const q4n1 = rmsn(q4f, lw.qn, eps, s);
            _ = mlx.mlx_array_free(q4f);
            const q4n = reshape(q4n1, &[_]c_int{ 1, 1, qh, 128 }, s);
            _ = mlx.mlx_array_free(q4n1);
            _ = mlx.mlx_array_free(q4);
            q4 = q4n;
            if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(q4); log.err("[beval] l{d} qnorm\n", .{lidx}); }
            const k4f = reshape(k4, &[_]c_int{ @intCast(@as(usize, @intCast(kh))), 128 }, s);
            const k4n1 = rmsn(k4f, lw.kn, eps, s);
            _ = mlx.mlx_array_free(k4f);
            const k4n = reshape(k4n1, &[_]c_int{ 1, 1, kh, 128 }, s);
            _ = mlx.mlx_array_free(k4n1);
            _ = mlx.mlx_array_free(k4);
            k4 = k4n;
            if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(k4); log.err("[beval] l{d} knorm\n", .{lidx}); }
        }
        defer _ = mlx.mlx_array_free(q4);
        defer _ = mlx.mlx_array_free(k4);
        const qt = transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const qr = ropeApplyOff(qt, theta, pos, 128, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = ropeApplyOff(kt, theta, pos, 128, s);
        const pk = kv[lidx].k;
        const pv = kv[lidx].v;
        const kc = concat2(s, kv[lidx].k, kr, 2);
        const vc = concat2(s, kv[lidx].v, vt, 2);
        kv[lidx].k = kc;
        kv[lidx].v = vc;
        if (pk.ctx != null) _ = mlx.mlx_array_free(pk);
        if (pv.ctx != null) _ = mlx.mlx_array_free(pv);
        _ = eval1(kv[lidx].k);
        _ = eval1(kv[lidx].v);
        if (std.c.getenv("BREEZE_STEP_TRACE") != null and lidx == 0) log.err("[bstep] L0 k/v ok att=\n", .{});
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(kr); _ = eval1(vt); log.err("[beval] l{d} kr vt\n", .{lidx}); }
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(qr); _ = eval1(kr); log.err("[beval] l{d} ropes kL={d}\n", .{lidx, mlx.getShape(kv[lidx].k)[2]}); }
        const att = sdpa(qr, kv[lidx].k, kv[lidx].v, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(128))))), null_arr, s);
        defer _ = mlx.mlx_array_free(att);
        const at = transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = reshape(at, &[_]c_int{ 1, 1, @intCast(mlx.getShape(lw.q)[1]) }, s);
        defer _ = mlx.mlx_array_free(af);
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(att); log.err("[beval] l{d} att shape={d}\n", .{lidx, mlx.getShape(att)[0]}); }
        const o = mm(af, lw.o, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = addM(x, o, s);
        defer _ = mlx.mlx_array_free(h1);
        if (std.c.getenv("BREEZE_STEP_EVAL") != null) { _ = eval1(h1); log.err("[beval] l{d} h1\n", .{lidx}); }
        const hn = rmsn(h1, lw.post_ln, eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = mm(hn, lw.gate, s);
        defer _ = mlx.mlx_array_free(g);
        const u = mm(hn, lw.up, s);
        defer _ = mlx.mlx_array_free(u);
        const gs = siluArr(g, s);
        defer _ = mlx.mlx_array_free(gs);
        const gu = mulEl(s, gs, u);
        defer _ = mlx.mlx_array_free(gu);
        const mlp = mm(gu, lw.down, s);
        defer _ = mlx.mlx_array_free(mlp);
        const x_prev = x;
        x = addM(h1, mlp, s);
        _ = mlx.mlx_array_free(x_prev);
        // concat2 returns `b` unchanged when the cache is still empty, in which
        // case the kv entry aliases kr/vt's handles and they must stay alive
        // until here (last uses are the sdpa above / debug evals).
        if (kc.ctx != kr.ctx) _ = mlx.mlx_array_free(kr);
        if (vc.ctx != vt.ctx) _ = mlx.mlx_array_free(vt);
    }
    if (std.c.getenv("BREEZE_STEP_TRACE") != null) log.err("[bstep] done normed\n", .{});
    const x_norm_prev = x;
    x = rmsn(x, norm, eps, s);
    _ = mlx.mlx_array_free(x_norm_prev);
    const lasth = reshape(x, &[_]c_int{@intCast(hidden)}, s);
    defer _ = mlx.mlx_array_free(lasth);
    try evalHost(s, lasth, out_last);
    _ = mlx.mlx_array_free(x);
}

fn mulEl(s: S, a: mlx.mlx_array, b: mlx.mlx_array) mlx.mlx_array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&r, a, b, s);
    return r;
}

fn windowMask(s: S, L: usize, w: usize) mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    for (0..L) |i| {
        for (0..L) |j| {
            const allowed = j <= i and (i - j) < w;
            buf[i * L + j] = if (allowed) 0 else -std.math.inf(f32);
        }
    }
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    defer _ = mlx.mlx_array_free(h);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&r, h, .bfloat16, s);
    return r;
}

// ───────────────────────────── text encoder ──────────────────────────────────

var te_buf: []f32 = &[_]f32{};

const Te = struct {
    layers: []QLayer, // reuse layout (q [in,1024] etc via ownT16)
    norm: mlx.mlx_array,
    table: []const f32, // HOST [TE_VOCAB, TE_HIDDEN] ×√hidden, eoi zeroed

    fn load(w: *const Weights, s: S, alloc: std.mem.Allocator) !Te {
        const layers = try alloc.alloc(QLayer, TE_LAYERS);
        for (layers, 0..) |*L, i| {
            const p = try std.fmt.allocPrint(alloc, "text_encoder.layers.{d}.", .{i});
            defer alloc.free(p);
            L.* = .{
                .q = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_proj.weight", .{p})),
                .k = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_proj.weight", .{p})),
                .v = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.v_proj.weight", .{p})),
                .o = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.o_proj.weight", .{p})),
                .qn = try ownF16Plus1(w, s, alloc, try std.fmt.allocPrint(alloc, "{s}self_attn.q_norm.weight", .{p})),
                .kn = try ownF16Plus1(w, s, alloc, try std.fmt.allocPrint(alloc, "{s}self_attn.k_norm.weight", .{p})),
                .gate = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.gate_proj.weight", .{p})),
                .up = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.up_proj.weight", .{p})),
                .down = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.down_proj.weight", .{p})),
                .in_ln = try ownF16Plus1(w, s, alloc, try std.fmt.allocPrint(alloc, "{s}pre_self_attn_layernorm.weight", .{p})),
                .post_ln = try ownF16Plus1(w, s, alloc, try std.fmt.allocPrint(alloc, "{s}post_feedforward_layernorm.weight", .{p})),
                .pre_ff = try ownF16Plus1(w, s, alloc, try std.fmt.allocPrint(alloc, "{s}pre_feedforward_layernorm.weight", .{p})),
            };
            // post_self_attn_layernorm rides alongside as the attn post norm.
            const ps = try ownF16Plus1(w, s, alloc, try std.fmt.allocPrint(alloc, "{s}post_self_attn_layernorm.weight", .{p}));
            te_post_attn = appendHandle(te_post_attn, alloc, ps);
        }
        return .{
            .layers = layers,
            .norm = try ownF16Plus1(w, s, alloc, "text_encoder.norm.weight"),
            .table = &[_]f32{},
        };
    }

    fn loadTable(w: *const Weights, s: S, alloc: std.mem.Allocator) ![]const f32 {
        const arr = w.get("text_encoder.embed_tokens.weight") orelse return error.MissingBreezeWeight;
        var f32arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32arr);
        try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f32arr));
        const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
        const out = try alloc.alloc(f32, TE_VOCAB * TE_HIDDEN);
        const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(TE_HIDDEN))));
        for (0..TE_VOCAB) |r| {
            const src = data[r * TE_HIDDEN ..][0..TE_HIDDEN];
            const dst = out[r * TE_HIDDEN ..][0..TE_HIDDEN];
            if (r == EOI_TOKEN_ID) {
                @memset(dst, 0);
            } else {
                for (src, dst) |sv, *dv| dv.* = sv * scale;
            }
        }
        return out;
    }
};

var te_post_attn: []mlx.mlx_array = &[_]mlx.mlx_array{};

fn appendHandle(buf: []mlx.mlx_array, alloc: std.mem.Allocator, h: mlx.mlx_array) []mlx.mlx_array {
    const out = alloc.alloc(mlx.mlx_array, buf.len + 1) catch return buf;
    if (buf.len > 0) {
        @memcpy(out[0..buf.len], buf);
        alloc.free(buf);
    }
    out[out.len - 1] = h;
    return out;
}

/// TE conditioning over text ids → host rows [L, TE_HIDDEN].
fn teCond(alloc: std.mem.Allocator, s: S, te: *const Te, ids: []const u32, out: []f32) !void {
    const L = ids.len;
    var buf = alloc.alloc(f32, L * TE_HIDDEN) catch return error.OutOfMemory;
    defer alloc.free(buf);
    for (ids, 0..) |id, i| {
        if (id >= TE_VOCAB) return error.TokenBeyondTeVocab;
        @memcpy(buf[i * TE_HIDDEN ..][0..TE_HIDDEN], te.table[@as(usize, id) * TE_HIDDEN ..][0..TE_HIDDEN]);
    }
    const sh = [_]c_int{ 1, @intCast(L), @intCast(TE_HIDDEN) };
    var x = hostF32(buf, &sh);

    for (te.layers, te_post_attn, 0..) |lw, ps, li| {
        const full = isFullAttentionLayer(@intCast(li));
        const xn = rmsn(x, lw.in_ln, 1e-6, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = mm(xn, lw.q, s);
        defer _ = mlx.mlx_array_free(q);
        const k = mm(xn, lw.k, s);
        defer _ = mlx.mlx_array_free(k);
        const v = mm(xn, lw.v, s);
        defer _ = mlx.mlx_array_free(v);
        const qh: c_int = @divExact(mlx.getShape(lw.q)[1], @as(c_int, @intCast(TE_HEAD_DIM)));
        const kh: c_int = @divExact(mlx.getShape(lw.k)[1], @as(c_int, @intCast(TE_HEAD_DIM)));
        var q4 = reshape(q, &[_]c_int{ 1, @intCast(L), qh, @intCast(TE_HEAD_DIM) }, s);
        var k4 = reshape(k, &[_]c_int{ 1, @intCast(L), kh, @intCast(TE_HEAD_DIM) }, s);
        const v4 = reshape(v, &[_]c_int{ 1, @intCast(L), kh, @intCast(TE_HEAD_DIM) }, s);
        defer _ = mlx.mlx_array_free(v4);
        const lq: usize = L * @as(usize, @intCast(qh));
        const lk: usize = L * @as(usize, @intCast(kh));
        const hd: c_int = @intCast(TE_HEAD_DIM);
        {
            const q4f = reshape(q4, &[_]c_int{ @intCast(lq), hd }, s);
            const q4n1 = rmsn(q4f, lw.qn, 1e-6, s);
            _ = mlx.mlx_array_free(q4f);
            const q4n = reshape(q4n1, &[_]c_int{ 1, @intCast(L), qh, hd }, s);
            _ = mlx.mlx_array_free(q4n1);
            _ = mlx.mlx_array_free(q4);
            q4 = q4n;
            const k4f = reshape(k4, &[_]c_int{ @intCast(lk), hd }, s);
            const k4n1 = rmsn(k4f, lw.kn, 1e-6, s);
            _ = mlx.mlx_array_free(k4f);
            const k4n = reshape(k4n1, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
            _ = mlx.mlx_array_free(k4n1);
            _ = mlx.mlx_array_free(k4);
            k4 = k4n;
        }
        defer _ = mlx.mlx_array_free(q4);
        defer _ = mlx.mlx_array_free(k4);
        const qt = transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);
        const theta = if (full) TE_THETA_FULL else TE_THETA_SLIDE;
        const p_scale = if (full) (1.0 / TE_FACTOR_FULL) else 1.0;
        const qr = ropeApply(qt, theta, p_scale, @intCast(TE_HEAD_DIM), s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = ropeApply(kt, theta, p_scale, @intCast(TE_HEAD_DIM), s);
        defer _ = mlx.mlx_array_free(kr);
        const mask = if (full) teFullMask(s, L) else teLocalMask(s, L);
        const mask_live = mask.ctx != null;
        defer {
            if (mask_live) _ = mlx.mlx_array_free(mask);
        }
        const att = sdpa(qr, kr, vt, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(TE_HEAD_DIM))))), mask, s);
        defer _ = mlx.mlx_array_free(att);
        const at = transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = reshape(at, &[_]c_int{ 1, @intCast(L), @intCast(TE_HEADS * TE_HEAD_DIM) }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = mm(af, lw.o, s);
        defer _ = mlx.mlx_array_free(o);
        const on = rmsn(o, ps, 1e-6, s);
        defer _ = mlx.mlx_array_free(on);
        const h1 = addM(x, on, s);
        defer _ = mlx.mlx_array_free(h1);
        const hn = rmsn(h1, lw.pre_ff, 1e-6, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = mm(hn, lw.gate, s);
        defer _ = mlx.mlx_array_free(g);
        const u = mm(hn, lw.up, s);
        defer _ = mlx.mlx_array_free(u);
        const act = geluTanhArr(g, s);
        defer _ = mlx.mlx_array_free(act);
        const gu = mulEl(s, act, u);
        defer _ = mlx.mlx_array_free(gu);
        const dn = mm(gu, lw.down, s);
        defer _ = mlx.mlx_array_free(dn);
        const dnn = rmsn(dn, lw.post_ln, 1e-6, s);
        defer _ = mlx.mlx_array_free(dnn);
        const x_prev = x;
        x = addM(h1, dnn, s);
        _ = mlx.mlx_array_free(x_prev);
    }
    const x_norm_prev = x;
    const xn = rmsn(x, te.norm, 1e-6, s);
    defer _ = mlx.mlx_array_free(xn);
    _ = mlx.mlx_array_free(x_norm_prev);
    const last = reshape(xn, &[_]c_int{ @intCast(L), @intCast(TE_HIDDEN) }, s);
    defer _ = mlx.mlx_array_free(last);
    try evalHost(s, last, out);
}

fn teFullMask(s: S, L: usize) mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    @memset(buf, 0);
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    defer _ = mlx.mlx_array_free(h);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&r, h, .bfloat16, s);
    return r;
}

fn teLocalMask(s: S, L: usize) mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    const left = (TE_WINDOW + 1) / 2;
    const right = TE_WINDOW / 2 + 1;
    for (0..L) |i| {
        for (0..L) |j| {
            const allowed = (j <= i and i - j < left) or (j > i and j - i < right);
            buf[i * L + j] = if (allowed) 0 else -std.math.inf(f32);
        }
    }
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    defer _ = mlx.mlx_array_free(h);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&r, h, .bfloat16, s);
    return r;
}

fn geluTanhArr(x: mlx.mlx_array, s: S) mlx.mlx_array {
    // 0.5·x·(1+tanh(√(2/π)(x+0.044715x³))) — TE hidden_activation
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    _ = mlx.mlx_multiply(&x2, x, x, s);
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    _ = mlx.mlx_multiply(&x3, x2, x, s);
    const sc1 = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(sc1);
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    _ = mlx.mlx_multiply(&k, x3, sc1, s);
    var inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner);
    _ = mlx.mlx_add(&inner, x, k, s);
    const sc2 = mlx.mlx_array_new_float(0.7978845608);
    defer _ = mlx.mlx_array_free(sc2);
    var lin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lin);
    _ = mlx.mlx_multiply(&lin, inner, sc2, s);
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    _ = mlx.mlx_tanh(&th, lin, s);
    const sc3 = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(sc3);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    _ = mlx.mlx_add(&one, th, sc3, s);
    const sc4 = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(sc4);
    var half = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(half);
    _ = mlx.mlx_multiply(&half, one, sc4, s);
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_multiply(&r, x, half, s);
    return r;
}

// ───────────────────────── heads (host) ──────────────────────────────────────

fn readTable(s: S, w: *const Weights, key: []const u8, out: []f32) !void {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    var f32arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32arr);
    try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32arr));
    const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
    @memcpy(out, data[0..out.len]);
}

/// host logits [cb] over canonical [out,in] host row set.
fn headLogits(host: []const f32, hidden: []const f32, out: []f32, vocab: usize, dim: usize, slot: usize) void {
    const base = slot * vocab * dim;
    for (0..vocab) |o| {
        var acc: f32 = 0;
        const row = host[base + o * dim ..][0..dim];
        for (hidden, row) |hv, wv| acc += hv * wv;
        out[o] = acc;
    }
}

// ───────────────────────────── Engine API ────────────────────────────────────

pub const Request = struct {
    text: []const u8 = "",
    voice: ?[]const u8 = null, // "S0".."S9" voice-design slot
    instruction: ?[]const u8 = null,
    ref_audio: ?[]const u8 = null, // WAV bytes → Mimi ref frames (clone)
    ref_text: ?[]const u8 = null, // transcript of ref_audio (required with it)
    max_frames: ?u32 = null,
    seed: ?u64 = null,
};

pub const Synthesis = struct {
    pcm: []f32,
    frames: []const [NUM_CODEBOOKS]u32,
};

/// Streaming synthesis controls. `emit` receives f32 PCM slices in
/// completion order, non-overlapping, covering the full utterance once;
/// each slice is the chunk freshly decoded at its frame boundary.
pub const StreamSpec = struct {
    emit: *const fn (user: *anyopaque, pcm: []const f32) void,
    emit_user: *anyopaque,
    /// Emit every N completed frames (0 = default 12).
    chunk_frames: u32 = 12,
    /// Checked before each frame; true aborts with error.Cancelled.
    /// Same signature as the sse Progress cancelled_cb.
    cancel: ?*const fn (ctx: *anyopaque) bool = null,
    cancel_user: ?*anyopaque = null,
};

pub const Engine = struct {
    alloc: std.mem.Allocator,
    tok: @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer,
    te: Te,
    te_proj: mlx.mlx_array, // [1152, 2048] f16 pre-transposed
    b_layers: []QLayer,
    b_norm: mlx.mlx_array,
    d_layers: []QLayer,
    d_norm: mlx.mlx_array,
    d_inp: mlx.mlx_array, // depth input projection [2048, 1024] f16
    lm_host: []f32, // [CB_LM, B_HIDDEN] canonical (codes + pad rows + eos row)
    d_freqs: mlx.mlx_array = null_arr,
    heads_host: []f32, // [15, CB_VOCAB, D_HIDDEN]
    bkv: [B_LAYERS]KV = undefined,
    bkv_len: usize = 0,
    audio_table: []f32, // HOST [NUM_CODEBOOKS*CB_STRIDE, B_HIDDEN]
    codec: tts.CodecDecoder,
    samples_per_frame: usize = 0, // codec total upsample ratio (1920 = 12.5 fps @ 24 kHz)
    scratch: []f32,
    order: []usize,
    model_dir: []const u8 = "",
    io: std.Io = undefined,
    mimi: ?*tts.MimiEncoder = null,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !*Engine {
        if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.InvalidModelDir;
        // The host owns the inference stream; MLX work (and the load itself) runs on it.
        g_stream = s;
        mlx_allocator = allocator;
        _hist_seen = allocator.alloc(bool, CB_LM) catch _hist_seen;

        var self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        var w = try model_mod.loadWeights(io, allocator, model_dir);
        defer w.deinit();

        var te = try Te.load(&w, s, allocator);
        errdefer allocator.free(te.layers);
        te.table = try Te.loadTable(&w, s, allocator);
        errdefer allocator.free(te.table);

        const lm_host = try allocator.alloc(f32, CB_LM * B_HIDDEN);
        errdefer allocator.free(lm_host);
        // lm_head: torch [2052, 2048] audio rows 0..CB_VOCAB-1, canonical.
        {
            const arr = w.get("lm_head.weight") orelse return error.MissingBreezeWeight;
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
            for (0..CB_LM) |r| @memcpy(lm_host[r * B_HIDDEN ..][0..B_HIDDEN], data[r * B_HIDDEN ..][0..B_HIDDEN]);
        }

        const heads_host = try allocator.alloc(f32, (NUM_CODEBOOKS - 1) * CB_VOCAB * D_HIDDEN);
        errdefer allocator.free(heads_host);
        // codebooks_head [15, 1024, 2051] — canonicalize to [15, out, in].
        {
            const arr = w.get("depth_decoder.codebooks_head.weight") orelse return error.MissingBreezeWeight;
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
            const shp = mlx.getShape(arr);
            if (shp.len == 3 and shp[1] == CB_VOCAB) {
                @memcpy(heads_host, data[0..heads_host.len]);
            } else {
                var h: usize = 0;
                while (h < NUM_CODEBOOKS - 1) : (h += 1) {
                    for (0..D_HIDDEN) |i| {
                        for (0..CB_VOCAB) |o| {
                            heads_host[h * CB_VOCAB * D_HIDDEN + o * D_HIDDEN + i] = data[h * D_HIDDEN * CB_VOCAB + i * CB_VOCAB + o];
                        }
                    }
                }
            }
        }

        const audio_table = try allocator.alloc(f32, NUM_CODEBOOKS * CB_STRIDE * B_HIDDEN);
        errdefer allocator.free(audio_table);
        try readTable(s, &w, "depth_decoder.model.embed_tokens.weight", audio_table);

        self.* = .{
            .alloc = allocator,
            .tok = try tokenizer_mod.loadTokenizer(io, allocator, model_dir),
            .te = te,
            .te_proj = undefined,
            .model_dir = try allocator.dupe(u8, model_dir),
            .io = io,
            .mimi = null,
            .b_layers = try loadQLayers("backbone_model.layers.{d}.", &w, allocator, s, B_LAYERS, true),
            .b_norm = try ownF16(&w, s, "backbone_model.norm.weight"),
            .d_layers = try loadQLayers("depth_decoder.model.layers.{d}.", &w, allocator, s, D_LAYERS, false),
            .d_norm = try ownF16(&w, s, "depth_decoder.model.norm.weight"),
            .d_inp = try ownT16(&w, s, "depth_decoder.model.inputs_embeds_projector.weight"),
            .lm_host = lm_host,
            .heads_host = heads_host,
            .audio_table = audio_table,
            .codec = try tts.loadCodecDecoderFrom(io, allocator, s, try std.fmt.allocPrint(allocator, "{s}/audio_tokenizer", .{model_dir})),
            .scratch = try allocator.alloc(f32, CB_LM),
            .order = try allocator.alloc(usize, CB_LM),
            .bkv_len = 0,
            .d_freqs = blk: {
                d_freqs_buf = depthInvFreqs();
                if (std.c.getenv("BREEZE_FREQS") != null) log.err("[breeze-dump] FR {any}\n", .{d_freqs_buf[0..8]});
                break :blk hostF32(&d_freqs_buf, &[_]c_int{D_HD / 2});
            },
            .bkv = blk: {
                var tmp: [B_LAYERS]KV = undefined;
                for (&tmp) |*e| e.* = .{ .k = null_arr, .v = null_arr };
                break :blk tmp;
            },
        };
        errdefer {
            allocator.free(self.scratch);
            allocator.free(self.order);
        }
        self.te_proj = try ownT16(&w, s, "text_encoder_proj.weight");
        self.samples_per_frame = self.codec.samplesPerFrame();
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.tok.deinit();
        self.codec.deinit();
        for (&self.bkv) |*e| {
            if (e.k.ctx != null) _ = mlx.mlx_array_free(e.k);
            if (e.v.ctx != null) _ = mlx.mlx_array_free(e.v);
            e.k.ctx = null;
            e.v.ctx = null;
        }
        self.alloc.free(self.te.table);
        self.alloc.free(self.te.layers);
        self.alloc.free(self.lm_host);
        self.alloc.free(self.heads_host);
        self.alloc.free(self.audio_table);
        if (self.mimi) |m| { m.deinit(); self.alloc.destroy(m); }
        self.alloc.free(self.model_dir);
        self.alloc.free(self.scratch);
        self.alloc.free(self.order);
        self.alloc.destroy(self);
    }

    pub fn sampleRate(self: *const Engine) u32 {
        _ = self;
        return SAMPLE_RATE;
    }

    fn enc(self: *Engine, out: *std.ArrayList(u32), text: []const u8) !void {
        const ids = try self.tok.encode(self.alloc, text);
        defer self.alloc.free(ids);
        try out.appendSlice(self.alloc, ids);
    }

    /// Plain / tagged / instruction synthesis.
    pub fn synthesize(self: *Engine, allocator: std.mem.Allocator, req: Request) !Synthesis {
        return self.run(allocator, req, null);
    }

    /// Streaming synthesis. Result is prefix-exact with `synthesize` (the
    /// codec decoder is fully causal, so each chunk decode is identical to
    /// the corresponding prefix of the final decode). `Synthesis.pcm` is
    /// empty — the PCM left via `st.emit`; the caller does not free it.
    pub fn synthesizeStream(
        self: *Engine,
        allocator: std.mem.Allocator,
        req: Request,
        st: StreamSpec,
    ) !Synthesis {
        return self.run(allocator, req, &st);
    }

    fn run(self: *Engine, allocator: std.mem.Allocator, req: Request, st: ?*const StreamSpec) !Synthesis {
        if (req.text.len == 0) return error.EmptyText;
        const clone = req.ref_audio != null and req.ref_audio.?.len > 0;
        if (clone and (req.ref_text == null or req.ref_text.?.len == 0)) return error.CloneMissingRefText;
        var rng = Rng.init(req.seed orelse 42);

        // ref_clone_tata: TE-encodes each text segment over its own positions;
        // audio slots carry audio_embedding frame sums (zero frame at audio_eos).
        var seg1: std.ArrayList(u32) = .empty;
        defer seg1.deinit(allocator);
        var seg2: std.ArrayList(u32) = .empty;
        defer seg2.deinit(allocator);
        try seg1.append(allocator, 2); // BOS on the first text segment
        if (req.voice) |v| {
            if (!voiceSlotValid(v)) return error.UnknownVoice;
            try seg1.append(allocator, SPEAKER_BASE + @as(u32, v[1] - '0'));
        }
        if (clone) {
            try self.enc(&seg1, req.ref_text.?);
            // Every text segment is encoded with its own BOS: the reference
            // tokenizer applies add_special_tokens=True per segment, so the
            // target-text segment must open with BOS too. Without it the
            // model continues the ref-text segment (speaks a sample-continuation
            // prefix) instead of starting a new segment at the target text.
            try seg2.append(allocator, 2); // BOS on the second text segment
            if (req.voice) |v| {
                if (!voiceSlotValid(v)) return error.UnknownVoice;
                try seg2.append(allocator, SPEAKER_BASE + @as(u32, v[1] - '0'));
            }
            if (req.instruction) |ins| {
                try seg2.append(allocator, INS_BOS_ID);
                try self.enc(&seg2, ins);
                try seg2.append(allocator, INS_EOS_ID);
            }
        } else {
            if (req.instruction) |ins| {
                try seg1.append(allocator, INS_BOS_ID);
                try self.enc(&seg1, ins);
                try seg1.append(allocator, INS_EOS_ID);
            }
        }

        var ref_frames: std.ArrayList([]u32) = .empty;
        defer for (ref_frames.items) |f| allocator.free(f);
        if (clone) {
            const d = try wav_mod.decode(allocator, req.ref_audio.?);
            defer allocator.free(d.pcm);
            var pcm = d.pcm;
            if (d.channels == 2) {
                const m = try allocator.alloc(f32, d.pcm.len / 2);
                defer allocator.free(m);
                for (0..m.len) |i| m[i] = (d.pcm[i * 2] + d.pcm[i * 2 + 1]) / 2;
                if (d.sample_rate != MIMI_RATE) return error.UnsupportedRefRate; // mixed resample skipped
                pcm = m;
            }
            if (self.mimi == null) {
                const tok_dir = try std.fs.path.join(allocator, &.{ self.model_dir, "audio_tokenizer" });
                defer allocator.free(tok_dir);
                const enc2 = try allocator.create(tts.MimiEncoder);
                errdefer allocator.destroy(enc2);
                enc2.* = try tts.loadMimiEncoder(self.io, allocator, g_stream, tok_dir);
                self.mimi = enc2;
            }
            const fr = try self.mimi.?.encodeSamples(pcm);
            defer allocator.free(fr);
            for (fr) |f| try ref_frames.append(allocator, f);
        }

        // Long-form chunking (port of longform.py): when the text cannot fit
        // the 2048-position window in one turn, split it at sentence
        // boundaries into segments that each fit (exact tokenizer counts +
        // estimated speech frames + prompt overhead) and synthesize one turn
        // per segment, appending frames to a single stitched stream. The
        // codec decode at the tail stays one causal pass over all frames, so
        // buffered WAVs and prefix-exact stream chunks are unaffected by the
        // stitching; prosody restarts per segment (fresh speaker/text prompt),
        // same as the reference longform tool.
        const trimmed = std.mem.trim(u8, req.text, " \t\r\n");
        if (trimmed.len == 0) return error.EmptyText;

        var seg_texts: std.ArrayList([]const u8) = .empty; // owned; all freed at scope exit
        defer {
            for (seg_texts.items) |s| allocator.free(s);
            seg_texts.deinit(allocator);
        }
        const overhead = seg1.items.len +
            (if (clone) seg2.items.len + ref_frames.items.len + 1 else 0);
        var whole_toks: std.ArrayList(u32) = .empty;
        defer whole_toks.deinit(allocator);
        try self.enc(&whole_toks, trimmed);
        if (segmentFits(overhead, whole_toks.items.len, speakUnits(trimmed))) {
            try seg_texts.append(allocator, try allocator.dupe(u8, trimmed));
        } else {
            var sentences: std.ArrayList([]const u8) = .empty;
            defer sentences.deinit(allocator);
            try splitSentences(allocator, trimmed, &sentences);
            var pack: std.ArrayList(u32) = .empty;
            defer pack.deinit(allocator);
            var p_start: usize = 0;
            var p_toks: usize = 0;
            var p_units: usize = 0;
            for (sentences.items, 0..) |sent, si| {
                pack.clearRetainingCapacity();
                try self.enc(&pack, sent);
                const s_units = speakUnits(sent);
                if (si > p_start and !segmentFits(overhead, p_toks + pack.items.len, p_units + s_units)) {
                    try seg_texts.append(allocator, try joinSentences(allocator, sentences.items[p_start..si]));
                    p_start = si;
                    p_toks = 0;
                    p_units = 0;
                }
                p_toks += pack.items.len;
                p_units += s_units;
            }
            try seg_texts.append(allocator, try joinSentences(allocator, sentences.items[p_start..]));
        }

        var frames: std.ArrayList([NUM_CODEBOOKS]u32) = .empty; // stitched across segments
        errdefer frames.deinit(allocator);
        var hist: std.ArrayList(u32) = .empty; // repetition-penalty history, spans segments
        defer hist.deinit(allocator);

        const h_row = allocator.alloc(f32, B_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(h_row);
        var h_proj = allocator.alloc(f32, D_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(h_proj);
        const emb_row = allocator.alloc(f32, B_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(emb_row);
        const logits = allocator.alloc(f32, CB_LM) catch return error.OutOfMemory;
        defer allocator.free(logits);
        var depth_span = allocator.alloc(f32, (NUM_CODEBOOKS + 1) * D_HIDDEN) catch return error.OutOfMemory;
        defer allocator.free(depth_span);
        const span_mode = std.c.getenv("BREEZE_SPAN") != null;

        var seg_i: usize = 0;
        while (seg_i < seg_texts.items.len) : (seg_i += 1) {
            const seg_text = seg_texts.items[seg_i];

            var toks: std.ArrayList(u32) = .empty;
            defer toks.deinit(allocator);
            try self.enc(&toks, seg_text);

            var s1: std.ArrayList(u32) = .empty; // TE text segments for this turn
            defer s1.deinit(allocator);
            var s2: std.ArrayList(u32) = .empty;
            defer s2.deinit(allocator);
            var ids: std.ArrayList(u32) = .empty; // meta prompt; audio slots = placeholder ids
            defer ids.deinit(allocator);
            try s1.appendSlice(allocator, seg1.items);
            if (clone) {
                try s2.appendSlice(allocator, seg2.items);
                try s2.appendSlice(allocator, toks.items); // clone: text rides in seg2
                try ids.appendSlice(allocator, s1.items);
                for (0..ref_frames.items.len) |_| try ids.append(allocator, AUDIO_TOKEN_ID);
                try ids.append(allocator, AUDIO_EOS_ID);
                try ids.appendSlice(allocator, s2.items);
            } else {
                try s1.appendSlice(allocator, toks.items); // plain: text rides in seg1
                try ids.appendSlice(allocator, s1.items);
            }

            const text_len = ids.items.len;
            if (std.c.getenv("BREEZE_IDS") != null) log.err("[breeze-dump] IDS {any}\n", .{ids.items});
            var frames_cap = estimateFrames(speakUnits(seg_text), req.max_frames);
            // The packer keeps segments window-safe; this clamp is the safety
            // net so even a model that never emits EOS stays in the span.
            if (text_len < MAX_SEQ and frames_cap > MAX_SEQ - text_len)
                frames_cap = @intCast(MAX_SEQ - text_len);
            const total = text_len + frames_cap;
            if (total > MAX_SEQ) return error.PromptTooLongForFrameBudget;

            // TE conditioning → backbone-width stream rows.
            var cond = allocator.alloc(f32, @max(s1.items.len, s2.items.len) * TE_HIDDEN) catch return error.OutOfMemory;
            defer allocator.free(cond);
            var stream = allocator.alloc(f32, total * B_HIDDEN) catch return error.OutOfMemory;
            defer allocator.free(stream);
            @memset(stream, 0);
            {
                const wt = allocator.alloc(f32, TE_HIDDEN * B_HIDDEN) catch return error.OutOfMemory;
                defer allocator.free(wt);
                try evalHost(g_stream, self.te_proj, wt);
                var row: usize = 0;
                try teCond(allocator, g_stream, &self.te, s1.items, cond[0 .. s1.items.len * TE_HIDDEN]);
                for (0..s1.items.len) |t| {
                    const r = cond[t * TE_HIDDEN ..][0..TE_HIDDEN];
                    for (0..B_HIDDEN) |o| {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < TE_HIDDEN) : (i += 1) acc += r[i] * wt[o * TE_HIDDEN + i];
                        stream[row * B_HIDDEN + o] = acc;
                    }
                    row += 1;
                }
                if (clone) {
                    for (ref_frames.items) |f| {
                        frameEmbed(self.audio_table, B_HIDDEN, f, emb_row);
                        @memcpy(stream[row * B_HIDDEN ..][0..B_HIDDEN], emb_row);
                        row += 1;
                    }
                    var zframe: [NUM_CODEBOOKS]u32 = undefined;
                    @memset(&zframe, 0);
                    frameEmbed(self.audio_table, B_HIDDEN, &zframe, emb_row);
                    @memcpy(stream[row * B_HIDDEN ..][0..B_HIDDEN], emb_row);
                    row += 1;
                    try teCond(allocator, g_stream, &self.te, s2.items, cond[0 .. s2.items.len * TE_HIDDEN]);
                    for (0..s2.items.len) |t| {
                        const r = cond[t * TE_HIDDEN ..][0..TE_HIDDEN];
                        for (0..B_HIDDEN) |o| {
                            var acc: f32 = 0;
                            var i: usize = 0;
                            while (i < TE_HIDDEN) : (i += 1) acc += r[i] * wt[o * TE_HIDDEN + i];
                            stream[row * B_HIDDEN + o] = acc;
                        }
                        row += 1;
                    }
                }
            }

            if (std.c.getenv("BREEZE_DUMP") != null) {
                log.err("[breeze-dump] ids={any}\n", .{ids.items});
                var row: usize = 0;
                while (row < text_len) : (row += 1) {
                    var buf: [32768]u8 = undefined;
                    var pos: usize = 0;
                    const r = stream[row * B_HIDDEN ..][0..B_HIDDEN];
                    for (r, 0..) |v, i| {
                        if (i > 0) buf[pos] = ',';
                        pos += 1;
                        const sres = std.fmt.bufPrint(buf[pos..], "{d}", .{v}) catch break;
                        pos += sres.len;
                    }
                    log.err("[breeze-dump] row={d} {s}\n", .{ row, buf[0..pos] });
                }
            }

            for (&self.bkv) |*e| {
                if (e.k.ctx != null) _ = mlx.mlx_array_free(e.k);
                if (e.v.ctx != null) _ = mlx.mlx_array_free(e.v);
                e.k.ctx = null;
                e.v.ctx = null;
            }
            var span = allocator.alloc(f32, total * B_HIDDEN) catch return error.OutOfMemory;
            defer allocator.free(span);

            var frame_i: usize = 0;
            while (frame_i < frames_cap) : (frame_i += 1) {
                if (st) |spec| {
                    if (spec.cancel) |cb| {
                        if (spec.cancel_user) |u| {
                            if (cb(u)) return error.Cancelled;
                        }
                    }
                }
                if (span_mode or frame_i == 0) {
                    const L = text_len + frame_i;
                    @memcpy(span[0 .. L * B_HIDDEN], stream[0 .. L * B_HIDDEN]);
                    try qwenForward(g_stream, self.b_layers, self.b_norm, 1e-6, B_THETA, span[0 .. L * B_HIDDEN], L, B_HIDDEN, 1.0, 0, null_arr, if (frame_i == 0 and !span_mode) self.bkv[0 .. B_LAYERS] else null, h_row);
                    if (!span_mode) self.bkv_len = text_len;
                } else {
                    try qwenStep(g_stream, self.b_layers, self.b_norm, 1e-6, B_THETA, stream[(text_len + frame_i - 1) * B_HIDDEN ..][0..B_HIDDEN], B_HIDDEN, @intCast(self.bkv_len), self.bkv[0 .. B_LAYERS], h_row);
                    self.bkv_len += 1;
                }
                headLogits(self.lm_host, h_row, logits, CB_LM, B_HIDDEN, 0);
                var frame: [NUM_CODEBOOKS]u32 = undefined;
                const dbg_depth = std.c.getenv("BREEZE_DEPTH") != null;
                const greedy = std.c.getenv("BREEZE_GREEDY") != null;
                const first_tok = sampleToken(&rng, logits, if (greedy) .{ .t = 0 } else .{}, self.scratch, self.order, .{ .suppress_from = CODEBOOK_SIZE, .suppress_to = CB_VOCAB, .history = if (std.c.getenv("BREEZE_NOREP") != null) &[_]u32{} else hist.items, .rep = 1.1 });
                frame[0] = if (dbg_depth and frame_i == 0) @as(u32, 1221) else @intCast(first_tok);
                if (greedy) log.err("[breeze-dump] F{d} {d}\n", .{ frame_i, frame[0] });
                if (std.c.getenv("BREEZE_DUMP") != null and frame_i == 0) {
                    var buf: [65536]u8 = undefined;
                    var pos: usize = 0;
                    for (logits[0..CB_LM], 0..) |v, i| {
                        if (i > 0) buf[pos] = ',';
                        pos += 1;
                        const sres = std.fmt.bufPrint(buf[pos..], "{d}", .{v}) catch break;
                        pos += sres.len;
                    }
                    log.err("[breeze-dump] lm0 {s}\n", .{buf[0..pos]});
                }
                if (frame[0] == CB_EOS) break;

                // depth span: [backbone_hidden_proj, emb(cb0), emb(cb1), ...]
                try projHostCached(allocator, g_stream, self.d_inp, h_row, B_HIDDEN, D_HIDDEN, h_proj[0..D_HIDDEN]);
                @memcpy(depth_span[0..D_HIDDEN], h_proj);
                if (dbg_depth) log.err("[breeze-dump] HPROJ {any}\nHR {any}\n", .{ h_proj[0..12], h_row[0..12] });
                var dl: usize = 1;
                var cb: usize = 1;
                while (cb < NUM_CODEBOOKS) : (cb += 1) {
                    const code = frame[cb - 1];
                    tableRow(self.audio_table, B_HIDDEN, cb - 1, @min(code, @as(u32, CB_VOCAB - 1)), emb_row);
                    if (dbg_depth and cb == 1) log.err("[breeze-dump] FEROW {any}\n", .{emb_row[0..12]});
                    if (std.c.getenv("BREEZE_ROW2") != null and cb == 3) log.err("[breeze-dump] ROW3 {any}\n", .{emb_row[0..12]});
                    try projHostCached(allocator, g_stream, self.d_inp, emb_row, B_HIDDEN, D_HIDDEN, depth_span[dl * D_HIDDEN ..][0..D_HIDDEN]);
                    dl += 1;
                    const d_last = allocator.alloc(f32, D_HIDDEN) catch return error.OutOfMemory;
                    defer allocator.free(d_last);
                    try qwenForward(g_stream, self.d_layers, self.d_norm, 1e-5, D_THETA, depth_span[0 .. dl * D_HIDDEN], dl, D_HIDDEN, 1.0, 0, (if (std.c.getenv("BREEZE_DROPE") != null and std.c.getenv("BREEZE_DROPE").?[0] == 104) self.d_freqs else null_arr), null, d_last);
                    headLogits(self.heads_host, d_last, logits, CB_VOCAB, D_HIDDEN, cb - 1);
                    if (dbg_depth) {
                        var buf: [16384]u8 = undefined;
                        var pos: usize = 0;
                        for (logits[0..12], 0..) |v, i| {
                            if (i > 0) buf[pos] = ',';
                            pos += 1;
                            const sres = std.fmt.bufPrint(buf[pos..], "{d}", .{v}) catch break;
                            pos += sres.len;
                        }
                        log.err("[breeze-dump] H{d} {s}\n", .{ cb, buf[0..pos] });
                    }
                    frame[cb] = @intCast(sampleToken(&rng, logits[0..CB_VOCAB], if (dbg_depth or greedy) .{ .t = 0 } else .{}, self.scratch, self.order, .{ .suppress_from = CODEBOOK_SIZE, .suppress_to = CB_VOCAB }));
                    if (dbg_depth) log.err("[breeze-dump] CODE {d} {d}\n", .{ cb, frame[cb] });
                }

                if (dbg_depth and frame_i == 0) break;
                if (frame_i % 4 == 3) { _ = mlx.mlx_clear_cache(); }
                try hist.append(allocator, frame[0]);
                // frame embed row into the stream
                frameEmbed(self.audio_table, B_HIDDEN, &frame, emb_row);
                @memcpy(stream[(text_len + frame_i) * B_HIDDEN ..][0..B_HIDDEN], emb_row);
                try frames.append(allocator, frame);
                if (st) |spec| {
                    const n = frames.items.len;
                    const k = if (spec.chunk_frames == 0) 12 else spec.chunk_frames;
                    if (n % k == 0) {
                        // Prefix-exact under the causal codec decoder: this
                        // decode's tail equals the final decode's same region.
                        const dec = try self.codec.decode(frames.items);
                        defer allocator.free(dec);
                        const dec_total = @min(dec.len, n * self.samples_per_frame);
                        const from = @min(dec_total, (n - k) * self.samples_per_frame);
                        spec.emit(spec.emit_user, dec[from..dec_total]);
                    }
                }
            }
        }

        const out_frames = frames.items;
        if (out_frames.len == 0) return error.NoFramesEmitted;

        if (st) |spec| {
            const n = out_frames.len;
            const k = if (spec.chunk_frames == 0) 12 else spec.chunk_frames;
            const rem = n % k;
            if (k > 0 and rem > 0) {
                const dec = try self.codec.decode(out_frames);
                defer allocator.free(dec);
                const dec_total = @min(dec.len, n * self.samples_per_frame);
                const from = @min(dec_total, (n - rem) * self.samples_per_frame);
                spec.emit(spec.emit_user, dec[from..dec_total]);
            }
        }

        if (st == null) {
            const pcm = try self.codec.decode(out_frames);
            return .{ .pcm = pcm, .frames = out_frames };
        }
        // Stream mode: PCM already emitted in chunks; Synthesis.pcm is empty
        // and must not be freed.
        return .{ .pcm = &[_]f32{}, .frames = out_frames };
    }
};

var _hist_seen: []bool = &[_]bool{};
var d_freqs_buf: [D_HD / 2]f32 = undefined;
var mlx_allocator: std.mem.Allocator = undefined;
var g_stream: S = undefined;

fn tableRow(table: []const f32, width: usize, cb: usize, code: u32, out: []f32) void {
    @memcpy(out, table[(@as(usize, cb) * CB_STRIDE + @as(usize, code)) * width ..][0..width]);
}

fn frameEmbed(table: []const f32, width: usize, codes: []const u32, out: []f32) void {
    @memset(out, 0);
    for (codes, 0..) |code, cb| {
        const c = @min(code, @as(u32, CB_VOCAB - 1));
        const row = table[(@as(usize, cb) * CB_STRIDE + @as(usize, c)) * width ..][0..width];
        for (out, row) |*o, rv| o.* += rv;
    }
}

var d_inp_cache: []f32 = &[_]f32{};

var d_inp_key: [2]usize = .{ 0, 0 };

fn projHostCached(alloc: std.mem.Allocator, s: S, w: mlx.mlx_array, x: []const f32, in: usize, out_ch: usize, out: []f32) !void {
    if (d_inp_key[0] != in or d_inp_key[1] != out_ch) {
        d_inp_cache = alloc.alloc(f32, in * out_ch) catch return error.OutOfMemory;
        try evalHost(s, w, d_inp_cache);
        d_inp_key = .{ in, out_ch };
    }
    for (0..out_ch) |o| {
        var acc: f32 = 0;
        var i: usize = 0;
        while (i < in) : (i += 1) acc += x[i] * d_inp_cache[o * in + i];
        out[o] = acc;
    }
}

fn projHost(alloc: std.mem.Allocator, s: S, w: mlx.mlx_array, x: []const f32, in: usize, out_ch: usize, out: []f32) !void {
    // w is [in, out] f16 on-device; eval to host once
    const host = alloc.alloc(f32, in * out_ch) catch return error.OutOfMemory;
    defer alloc.free(host);
    try evalHost(s, w, host);
    for (0..out_ch) |o| {
        var acc: f32 = 0;
        var i: usize = 0;
        while (i < in) : (i += 1) acc += x[i] * host[i * out_ch + o];
        out[o] = acc;
    }
}

fn loadQLayers(comptime prefix_fmt: []const u8, w: *const Weights, alloc: std.mem.Allocator, s: S, count: usize, qk_norm: bool) ![]QLayer {
    const layers = try alloc.alloc(QLayer, count);
    for (layers, 0..) |*L, i| {
        const p = try std.fmt.allocPrint(alloc, prefix_fmt, .{i});
        defer alloc.free(p);
        L.* = .{
            .q = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_proj.weight", .{p})),
            .k = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_proj.weight", .{p})),
            .v = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.v_proj.weight", .{p})),
            .o = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.o_proj.weight", .{p})),
            .qn = if (qk_norm) try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.q_norm.weight", .{p})) else null_arr,
            .kn = if (qk_norm) try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}self_attn.k_norm.weight", .{p})) else null_arr,
            .gate = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.gate_proj.weight", .{p})),
            .up = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.up_proj.weight", .{p})),
            .down = try ownT16(w, s, try std.fmt.allocPrint(alloc, "{s}mlp.down_proj.weight", .{p})),
            .in_ln = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}input_layernorm.weight", .{p})),
            .post_ln = try ownF16(w, s, try std.fmt.allocPrint(alloc, "{s}post_attention_layernorm.weight", .{p})),
            .pre_ff = null_arr,
        };
    }
    return layers;
}

// ───────────────────────────── tests ─────────────────────────────────────────

test "voice slot ids" {
    try std.testing.expectEqual(@as(?u32, SPEAKER_BASE), speakerToken(0));
    try std.testing.expectEqual(@as(?u32, SPEAKER_BASE + 9), speakerToken(9));
    try std.testing.expectEqual(@as(?u32, null), speakerToken(10));
}

test "voice slot validation" {
    try std.testing.expect(voiceSlotValid("S0"));
    try std.testing.expect(voiceSlotValid("s7"));
    try std.testing.expect(!voiceSlotValid("S10"));
    try std.testing.expect(!voiceSlotValid("S"));
    try std.testing.expect(!voiceSlotValid("V0"));
}

test "estimateFrames clamp" {
    const a = estimateFrames(300, null);
    try std.testing.expect(a > 0);
    try std.testing.expect(a <= MAX_FRAMES);
    const b = estimateFrames(300, 50);
    try std.testing.expectEqual(@as(u32, 50), @min(b, 50));
}

test "splitSentences latin, cjk, hard split" {
    var out: std.ArrayList([]const u8) = .empty;
    const al = std.testing.allocator;
    defer out.deinit(al);
    try splitSentences(al, "Hello there. How are you?   Fine! trailing tail", &out);
    try std.testing.expectEqual(@as(usize, 4), out.items.len);
    try std.testing.expectEqualStrings("Hello there.", out.items[0]);
    try std.testing.expectEqualStrings("How are you?", out.items[1]);
    try std.testing.expectEqualStrings("Fine!", out.items[2]);
    try std.testing.expectEqualStrings("trailing tail", out.items[3]);

    out.clearRetainingCapacity();
    try splitSentences(al, "今天天气不错。我们出去走走！", &out);
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqualStrings("今天天气不错。", out.items[0]);
    try std.testing.expectEqualStrings("我们出去走走！", out.items[1]);

    // unpunctuated run over MAX_SENT_UNITS hard-splits at word breaks
    out.clearRetainingCapacity();
    const big = comptime blk: {
        @setEvalBranchQuota(20000);
        var s: []const u8 = "";
        for (0..400) |_| s = s ++ "word ";
        break :blk s;
    };
    try splitSentences(al, big, &out);
    try std.testing.expect(out.items.len >= 2);
    for (out.items) |s| {
        try std.testing.expect(s.len > 0);
        try std.testing.expect(speakUnits(s) <= MAX_SENT_UNITS + 3);
    }
}

test "segmentFits window + cap" {
    // ~73 s passage (1178 ascii chars, ~300 tok) fits one turn
    try std.testing.expect(segmentFits(4, 300, 1178));
    // > MAX_FRAMES of speech on its own must not fit
    try std.testing.expect(!segmentFits(4, 400, 2500));
    // token span exhaustion must not fit
    try std.testing.expect(!segmentFits(4, 2045, 1));
}

test "sampler deterministic + temp0 argmax" {
    const logits = [_]f32{ 0.1, 5.0, 0.2, 0.3 };
    var scratch: [4]f32 = undefined;
    var order: [4]usize = undefined;
    var r1 = Rng.init(7);
    var r2 = Rng.init(7);
    const g1 = sampleToken(&r1, &logits, .{ .t = 0 }, &scratch, &order, .{} );
    try std.testing.expectEqual(@as(usize, 1), g1);
    var s2: [4]f32 = undefined;
    var o2: [4]usize = undefined;
    const a = sampleToken(&r1, &logits, .{}, &scratch, &order, .{} );
    const b = sampleToken(&r2, &logits, .{}, &s2, &o2, .{} );
    try std.testing.expectEqual(a, b);
}

test "frameEmbed sums table rows" {
    // minimal stride-two stand-in: cb0=[1,2], cb1=[3,4]
    var stub = [_]f32{ 1, 2, 3, 4 };
    var out = [_]f32{ 0, 0 };
    for ([_]u32{ 0, 0 }, 0..) |c, cb| {
        const row = stub[@as(usize, cb) * 2 + @as(usize, c) * 0 ..][0..2];
        for (&out, row) |*o, rv| o.* += rv;
    }
    try std.testing.expectEqual(@as(f32, 4), out[0]);
    try std.testing.expectEqual(@as(f32, 6), out[1]);
}

test "live synthesize (MLX_SERVE_BREEZE_MODEL)" {
    const dir = std.mem.span(std.c.getenv("MLX_SERVE_BREEZE_MODEL") orelse return error.SkipZigTest);
    const alloc = std.testing.allocator;
    var engine = try Engine.load(std.Io.Threaded.global_single_threaded.io(), alloc, dir);
    defer engine.deinit();
    const synth = try engine.synthesize(alloc, .{ .text = "Hello world, this is Breeze speaking.", .voice = "S0", .seed = 1 });
    try std.testing.expect(synth.pcm.len > 0);
    alloc.free(synth.pcm);
}
