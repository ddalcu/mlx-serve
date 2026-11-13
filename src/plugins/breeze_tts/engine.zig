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
const tts = @import("../../sdk/media/mimi_codec.zig");
const wav_mod = @import("../../sdk/media/wav.zig");
const os_mutex = @import("../../sdk/util/os_mutex.zig");
const env = @import("../../env.zig");
const residual_norm = @import("../../sdk/blocks/norms/residual_norm.zig");
const fused_activations = @import("../../sdk/blocks/mlp/fused_activations.zig");
const qk_norm_rope = @import("../../sdk/blocks/rope/qk_norm_rope.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

pub const model_type = "breeze_tts";

pub const cfg_mod = @import("config.zig");
pub const Config = cfg_mod.Config;

// Geometry and token ids come from the pack's config.json (cfg_mod.Config); the
// defaults below ride with the published Breeze-TTS-2 pack. A fixture override
// is a smaller config.json, not an env lever.

pub const MIMI_RATE: u32 = 24000;

// Frame-rate estimation constants (how many frames a text is likely to speak).
const FRAME_RATE: f64 = 12.5;
const CHARS_PER_SEC: f64 = 15.0;
const FRAME_EST_DIV: f64 = 3.0; // the model keeps generating ~4.5+ frames per text token; the true stop is eos

/// Voice-design speaker slot: `index` 0..9 maps to the pack's `[S0]..[S9]` ids.
pub fn speakerToken(base: u32, index: u32) ?u32 {
    if (index >= 10) return null;
    return base + index;
}

/// A request `voice` names a slot S0..S9 (case-insensitive), nothing else.
pub fn voiceSlotValid(name: []const u8) bool {
    if (name.len != 2) return false;
    if (name[0] != 'S' and name[0] != 's') return false;
    return name[1] >= '0' and name[1] <= '9';
}

/// Frame budget from speak units (see speakUnits): units / CHARS_PER_SEC ×
/// FRAME_RATE, with a headroom factor, clamped by a request cap and the
/// config's frame ceiling. The true stop is always the codebook EOS; this is
/// a runaway ceiling.
pub fn estimateFrames(units: usize, req_max: ?u32, max_frames: u32) u32 {
    const est: f64 = @as(f64, @floatFromInt(units)) / CHARS_PER_SEC * FRAME_RATE * FRAME_EST_DIV + 4.0;
    var frames: u32 = @intFromFloat(@ceil(est));
    if (req_max) |m| frames = @min(frames, m);
    return @min(frames, max_frames);
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
fn segmentFits(overhead: usize, toks: usize, units: usize, max_seq: usize, max_frames: u32) bool {
    const need = needFrames(units);
    return overhead + toks + need + 16 <= max_seq and need <= max_frames;
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

/// A host buffer MLX owns: `mlx_array_new_data_managed_payload` runs the dtor
/// when the array's last reference dies. The run is page-aligned so the dtor
/// can free it straight back to the page allocator; `assertDrained` is the
/// bar that every such run is gone by teardown.
const ManagedF32 = struct {
    run: ?*anyopaque,
    len: usize,

    fn freeRun(p: ?*anyopaque) callconv(.c) void {
        const h: *ManagedF32 = @ptrCast(@alignCast(p));
        const run = @as([*]u8, @ptrCast(h.run))[0..h.len];
        std.heap.page_allocator.rawFree(run, .@"16", @returnAddress());
    }

    var live: usize = 0;

    fn countUp() void {
        runLock().lock();
        defer runLock().unlock();
        live += 1;
    }

    /// Debug builds: every owned run must be gone at teardown.
    pub fn assertDrained() void {
        if (comptime std.debug.runtime_safety) {
            runLock().lock();
            defer runLock().unlock();
            std.debug.assert(live == 0);
        }
    }
};

fn runLock() *os_mutex.Mutex {
    const m = struct {
        var l: os_mutex.Mutex = .{};
    };
    return &m.l;
}

fn ownF16Plus1(w: *const Weights, s: S, alloc: std.mem.Allocator, key: []const u8) !mlx.mlx_array {
    _ = alloc;
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
    const hdr_bytes = std.mem.alignForward(usize, @sizeOf(ManagedF32), @alignOf(f32));
    const total = hdr_bytes + n * @sizeOf(f32);
    const al: std.mem.Alignment = .@"16";
    const run = std.heap.page_allocator.alignedAlloc(u8, al, total) catch return error.OutOfMemory;
    errdefer std.heap.page_allocator.rawFree(run, al, @returnAddress());
    const hdr: *ManagedF32 = @ptrCast(@alignCast(run.ptr));
    hdr.* = .{ .run = run.ptr, .len = total };
    const payload: [*]f32 = @alignCast(@ptrCast(run.ptr[hdr_bytes..]));
    for (0..n) |i| payload[i] = data[i] + 1.0;
    const out = mlx.mlx_array_new_data_managed_payload(
        @ptrCast(payload),
        shape.ptr,
        @intCast(shape.len),
        .float32,
        hdr,
        ManagedF32.freeRun,
    );
    if (out.ctx == null) return error.MLXOpFailed;
    ManagedF32.countUp();
    return out;
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

inline fn mm(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&r, x, w_t, s));
    return r;
}

inline fn addM(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&r, a, b, s));
    return r;
}

inline fn rmsn(x: mlx.mlx_array, g: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&r, x, g, eps, s));
    return r;
}

inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, @intCast(shape.len), s));
    return r;
}

inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&r, x, axes.ptr, @intCast(axes.len), s));
    return r;
}

fn siluArr(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&r, x, sig, s));
    return r;
}

/// RoPE via the house fused op (interleaved HF convention).
fn ropeApply(x: mlx.mlx_array, theta: f64, scale: f64, hd: c_int, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&r, x, hd, false, .{ .value = @floatCast(theta), .has_value = true }, @floatCast(scale), 0, null_arr, s));
    return r;
}

fn ropeApplyFreqs(x: mlx.mlx_array, freqs: mlx.mlx_array, hd: c_int, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&r, x, hd, false, .{}, 1.0, 0, freqs, s));
    return r;
}

// The depth decoder's rope: the pack declares llama3 scaling (factor 32,
// orig_ctx 16) and plain theta is what the sampled oracle reproduces, so this
// engine ropes with plain theta. config.zig reads `rope_scaling` and refuses
// any OTHER declared arm by name — the reason lives there, not as a comment.

fn sdpa(q4: mlx.mlx_array, k4: mlx.mlx_array, v4: mlx.mlx_array, scale: f32, mask: mlx.mlx_array, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    const mode: [*:0]const u8 = if (mask.ctx == null) "causal" else "array";
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&r, q4, k4, v4, scale, mode, mask, null_arr, false, s));
    return r;
}

// ─────────────────────────── qwen stack (generic) ────────────────────────────

pub const KV = struct {
    k: mlx.mlx_array = null_arr,
    v: mlx.mlx_array = null_arr,
};

fn concat2(s: S, a: mlx.mlx_array, b: mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    if (a.ctx == null) return b;
    const vec = mlx.mlx_vector_array_new();
    _ = mlx.mlx_vector_array_append_value(vec, a);
    _ = mlx.mlx_vector_array_append_value(vec, b);
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, s));
    _ = mlx.mlx_vector_array_free(vec);
    return r;
}

const QLayer = struct {
    q_heads: usize, // from the pack config
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f64,
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
    eps_arr: mlx.mlx_array = null_arr, // 0-d f32 for the sdk fused add-norm / qk kernels

    fn freeHandles(l: *QLayer) void {
        inline for (.{ l.q, l.k, l.v, l.o, l.qn, l.kn, l.gate, l.up, l.down, l.in_ln, l.post_ln, l.pre_ff, l.eps_arr }) |x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
        }
        l.* = std.mem.zeroes(QLayer);
    }
};

/// Full re-forward over a host [L, hidden] f32 embedding span. `pos_scale`
/// scales RoPE positions (TE full layers), optional causal local-window mask
/// (`window` 0 = plain causal). qk-norm applies when qn has data.
fn qwenForward(s: S, layers: []const QLayer, norm: mlx.mlx_array, x_host: []const f32, L: usize, hidden: usize, pos_scale: f64, window: usize, freqs: mlx.mlx_array, kv_save: ?[]KV, out_last: []f32) !void {
    if (layers.len == 0) return;
    const eps = layers[0].eps;
    const theta = layers[0].theta;
    const sh = [_]c_int{ 1, @intCast(L), @intCast(hidden) };
    var x = hostF32(x_host, &sh);
    var x16 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x16, x, .bfloat16, s));
    _ = mlx.mlx_array_free(x); // x16's graph owns the impl now
    x = x16;

    for (layers, 0..) |lw, lidx| {
        const xn = try rmsn(x, lw.in_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try mm(xn, lw.q, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try mm(xn, lw.k, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try mm(xn, lw.v, s);
        defer _ = mlx.mlx_array_free(v);
        const hd: c_int = @intCast(lw.head_dim);
        const qh: c_int = @intCast(@divExact(mlx.getShape(lw.q)[1], @as(c_int, hd)));
        const kh: c_int = @intCast(@divExact(mlx.getShape(lw.k)[1], @as(c_int, hd)));
        var q4 = try reshape(q, &[_]c_int{ 1, @intCast(L), qh, hd }, s);
        var k4 = try reshape(k, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
        const v4 = try reshape(v, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const lq: usize = L * @as(usize, @intCast(qh));
        const lk: usize = L * @as(usize, @intCast(kh));
        if (lw.qn.ctx != null and mlx.getShape(lw.qn)[0] > 0) {
            const q4f = try reshape(q4, &[_]c_int{ @intCast(lq), hd }, s);
            const q4n1 = try rmsn(q4f, lw.qn, eps, s);
            _ = mlx.mlx_array_free(q4f);
            const q4n = try reshape(q4n1, &[_]c_int{ 1, @intCast(L), qh, hd }, s);
            _ = mlx.mlx_array_free(q4n1);
            _ = mlx.mlx_array_free(q4);
            q4 = q4n;
            const k4f = try reshape(k4, &[_]c_int{ @intCast(lk), hd }, s);
            const k4n1 = try rmsn(k4f, lw.kn, eps, s);
            _ = mlx.mlx_array_free(k4f);
            const k4n = try reshape(k4n1, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
            _ = mlx.mlx_array_free(k4n1);
            _ = mlx.mlx_array_free(k4);
            k4 = k4n;
        }
        defer _ = mlx.mlx_array_free(q4);
        defer _ = mlx.mlx_array_free(k4);
        const qt = try transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = try transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        const qr = if (freqs.ctx != null) try ropeApplyFreqs(qt, freqs, hd, s) else try ropeApply(qt, theta, pos_scale, hd, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = if (freqs.ctx != null) try ropeApplyFreqs(kt, freqs, hd, s) else try ropeApply(kt, theta, pos_scale, hd, s);
        const has_mask = window > 0;
        const mask = if (has_mask) try windowMask(s, L, window) else null_arr;
        const att = try sdpa(qr, kr, vt, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(lw.head_dim))))), mask, s);
        defer _ = mlx.mlx_array_free(att);
        if (has_mask) _ = mlx.mlx_array_free(mask);
        if (kv_save) |ks| {
            ks[lidx] = .{ .k = kr, .v = vt }; // ownership moves to the caller's KV cache
        } else {
            _ = mlx.mlx_array_free(kr);
            _ = mlx.mlx_array_free(vt);
        }
        const at = try transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, @intCast(L), @intCast(mlx.getShape(lw.q)[1]) }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try mm(af, lw.o, s);
        defer _ = mlx.mlx_array_free(o);
        const ar = blk: {
            if (lw.eps_arr.ctx != null) {
                if (try residual_norm.fusedAddRmsNorm(s, x, o, lw.post_ln, lw.eps_arr)) |r| break :blk r;
            }
            const h1n = try addM(x, o, s);
            break :blk residual_norm.AddNormResult{ .sum = h1n, .normed = try rmsn(h1n, lw.post_ln, eps, s) };
        };
        const h1 = ar.sum;
        defer _ = mlx.mlx_array_free(h1);
        const hn = ar.normed;
        defer _ = mlx.mlx_array_free(hn);
        const g = try mm(hn, lw.gate, s);
        defer _ = mlx.mlx_array_free(g);
        const u = try mm(hn, lw.up, s);
        defer _ = mlx.mlx_array_free(u);
        const gu = if (try fused_activations.fusedSwiGLU(s, g, u)) |sg| sg else blk: {
            const gs = try siluArr(g, s);
            break :blk try mulEl(s, gs, u);
        };
        defer _ = mlx.mlx_array_free(gu);
        const mlp = try mm(gu, lw.down, s);
        defer _ = mlx.mlx_array_free(mlp);
        const x_prev = x;
        x = try addM(h1, mlp, s);
        _ = mlx.mlx_array_free(x_prev);
    }
    const x_norm_prev = x;
    x = try rmsn(x, norm, eps, s);
    _ = mlx.mlx_array_free(x_norm_prev);
    const last = try reshape(x, &[_]c_int{ @intCast(L), @intCast(hidden) }, s);
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
    defer _ = mlx.mlx_vector_array_free(vec);
    return mlx.mlx_eval(vec);
}

/// A deferred eval whose failure surfaces: an empty k/v concat is only read
/// after this point, so a failed command buffer must not slide into the sdpa.
fn evalChecked(a: mlx.mlx_array) !void {
    if (eval1(a) != 0) return error.MlxError;
}

fn ropeApplyOff(x: mlx.mlx_array, theta: f64, off: c_int, hd: c_int, s: S) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&r, x, hd, false, .{ .value = @floatCast(theta), .has_value = true }, 1.0, off, null_arr, s));
    return r;
}

fn qwenStep(s: S, layers: []const QLayer, norm: mlx.mlx_array, xrow: []const f32, hidden: usize, pos: c_int, kv: []KV, out_last: []f32) !void {
    if (layers.len == 0) return;
    const eps = layers[0].eps;
    const theta = layers[0].theta;
    const sh = [_]c_int{ 1, 1, @intCast(hidden) };
    var x = hostF32(xrow, &sh);
    var x16 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x16, x, .bfloat16, s));
    _ = mlx.mlx_array_free(x); // x16's graph owns the impl now
    x = x16;

    for (layers, 0..) |lw, lidx| {
        const xn = try rmsn(x, lw.in_ln, eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try mm(xn, lw.q, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try mm(xn, lw.k, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try mm(xn, lw.v, s);
        defer _ = mlx.mlx_array_free(v);
        const hd: c_int = @intCast(lw.head_dim);
        const qh: c_int = @intCast(@divExact(mlx.getShape(lw.q)[1], @as(c_int, hd)));
        const kh: c_int = @intCast(@divExact(mlx.getShape(lw.k)[1], @as(c_int, hd)));
        const v4 = try reshape(v, &[_]c_int{ 1, 1, kh, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s); // freed at the loop tail: kv may alias it
        // The sdk fused decode kernel: q/k per-head RMSNorm + RoPE + transpose
        // in one dispatch (bit-identical to the chain below; qwen3_tts's shape),
        // composed chain when it declines.
        var qr = null_arr;
        var kr = null_arr;
        var fused = false;
        if (lw.qn.ctx != null and mlx.getShape(lw.qn)[0] > 0) {
            const angles = try qk_norm_rope.ropeAngleRow(s, hd, .{ .value = @floatCast(theta), .has_value = true }, null_arr, pos, std.heap.c_allocator);
            defer _ = mlx.mlx_array_free(angles);
            if (try qk_norm_rope.fusedQkNormRope(s, q, k, lw.qn, lw.kn, angles, null, lw.eps_arr, qh, kh, hd)) |pair| {
                qr = pair[0];
                kr = pair[1];
                fused = true;
            }
        }
        if (!fused) {
            var q4 = try reshape(q, &[_]c_int{ 1, 1, qh, hd }, s);
            var k4 = try reshape(k, &[_]c_int{ 1, 1, kh, hd }, s);
            if (lw.qn.ctx != null and mlx.getShape(lw.qn)[0] > 0) {
                const q4f = try reshape(q4, &[_]c_int{ @intCast(@as(usize, @intCast(qh))), hd }, s);
                const q4n1 = try rmsn(q4f, lw.qn, eps, s);
                _ = mlx.mlx_array_free(q4f);
                _ = mlx.mlx_array_free(q4);
                q4 = try reshape(q4n1, &[_]c_int{ 1, 1, qh, hd }, s);
                _ = mlx.mlx_array_free(q4n1);
                const k4f = try reshape(k4, &[_]c_int{ @intCast(@as(usize, @intCast(kh))), hd }, s);
                const k4n1 = try rmsn(k4f, lw.kn, eps, s);
                _ = mlx.mlx_array_free(k4f);
                _ = mlx.mlx_array_free(k4);
                k4 = try reshape(k4n1, &[_]c_int{ 1, 1, kh, hd }, s);
                _ = mlx.mlx_array_free(k4n1);
            }
            const qt = try transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
            defer _ = mlx.mlx_array_free(qt);
            const kt = try transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
            defer _ = mlx.mlx_array_free(kt);
            qr = try ropeApplyOff(qt, theta, pos, hd, s);
            kr = try ropeApplyOff(kt, theta, pos, hd, s);
            _ = mlx.mlx_array_free(q4);
            _ = mlx.mlx_array_free(k4);
        }
        const pk = kv[lidx].k;
        const pv = kv[lidx].v;
        const kc = try concat2(s, kv[lidx].k, kr, 2);
        const vc = try concat2(s, kv[lidx].v, vt, 2);
        kv[lidx].k = kc;
        kv[lidx].v = vc;
        if (pk.ctx != null) _ = mlx.mlx_array_free(pk);
        if (pv.ctx != null) _ = mlx.mlx_array_free(pv);
        try evalChecked(kv[lidx].k);
        try evalChecked(kv[lidx].v);
        const att = try sdpa(qr, kv[lidx].k, kv[lidx].v, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(lw.head_dim))))), null_arr, s);
        defer _ = mlx.mlx_array_free(att);
        const at = try transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, 1, @intCast(mlx.getShape(lw.q)[1]) }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try mm(af, lw.o, s);
        defer _ = mlx.mlx_array_free(o);
        const ar = blk: {
            if (lw.eps_arr.ctx != null) {
                if (try residual_norm.fusedAddRmsNorm(s, x, o, lw.post_ln, lw.eps_arr)) |r| break :blk r;
            }
            const h1n = try addM(x, o, s);
            break :blk residual_norm.AddNormResult{ .sum = h1n, .normed = try rmsn(h1n, lw.post_ln, eps, s) };
        };
        const h1 = ar.sum;
        defer _ = mlx.mlx_array_free(h1);
        const hn = ar.normed;
        defer _ = mlx.mlx_array_free(hn);
        const g = try mm(hn, lw.gate, s);
        defer _ = mlx.mlx_array_free(g);
        const u = try mm(hn, lw.up, s);
        defer _ = mlx.mlx_array_free(u);
        const gu = if (try fused_activations.fusedSwiGLU(s, g, u)) |sg| sg else blk: {
            const gs = try siluArr(g, s);
            break :blk try mulEl(s, gs, u);
        };
        defer _ = mlx.mlx_array_free(gu);
        const mlp = try mm(gu, lw.down, s);
        defer _ = mlx.mlx_array_free(mlp);
        const x_prev = x;
        x = try addM(h1, mlp, s);
        _ = mlx.mlx_array_free(x_prev);
        // concat2 returns `b` unchanged when the cache is still empty, in which
        // case the kv entry aliases kr/vt's handles and they must stay alive
        // until here (last uses are the sdpa above / debug evals).
        if (kc.ctx != kr.ctx) _ = mlx.mlx_array_free(kr);
        if (vc.ctx != vt.ctx) _ = mlx.mlx_array_free(vt);
    }
    const x_norm_prev = x;
    x = try rmsn(x, norm, eps, s);
    _ = mlx.mlx_array_free(x_norm_prev);
    const lasth = try reshape(x, &[_]c_int{@intCast(hidden)}, s);
    defer _ = mlx.mlx_array_free(lasth);
    try evalHost(s, lasth, out_last);
    _ = mlx.mlx_array_free(x);
}

fn mulEl(s: S, a: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&r, a, b, s));
    return r;
}

fn windowMask(s: S, L: usize, w: usize) !mlx.mlx_array {
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
    try mlx.check(mlx.mlx_astype(&r, h, .bfloat16, s));
    return r;
}

// ───────────────────────────── text encoder ──────────────────────────────────

var te_buf: []f32 = &[_]f32{};

const Te = struct {
    layers: []QLayer, // reuse layout (q [in,1024] etc via ownT16)
    norm: mlx.mlx_array,
    table: []const f32, // HOST [te_vocab, te_hidden] ×√hidden, eoi zeroed
    hidden: usize,
    vocab: usize,
    head_dim: usize,
    window: usize,
    theta_full: f64,
    theta_slide: f64,
    factor_full: f64,
    eps: f32,
    full_layers: [16]u32,

    /// TE layer `i` attends globally when its `layer_types` entry is
    /// `full_attention` (the published pack: 5/11/17/23 of 26).
    fn isFullLayer(te: *const Te, i: usize) bool {
        const n: u32 = @intCast(i);
        for (te.full_layers) |f| if (f == n) return true;
        return false;
    }

    fn load(cfg: *const Config, w: *const Weights, s: S, alloc: std.mem.Allocator) !Te {
        const layers = try alloc.alloc(QLayer, cfg.te_layers);
        for (layers, 0..) |*L, i| {
            const p = try std.fmt.allocPrint(alloc, "text_encoder.layers.{d}.", .{i});
            defer alloc.free(p);
            var k: [256]u8 = undefined;
            L.* = .{
                .q_heads = cfg.te_heads,
                .kv_heads = cfg.te_kv,
                .head_dim = cfg.te_head_dim,
                .eps = cfg.te_rms_eps,
                .theta = cfg.te_theta_slide,
                .q = try ownT16(w, s, layerKey(&k, "{s}self_attn.q_proj.weight", p)),
                .k = try ownT16(w, s, layerKey(&k, "{s}self_attn.k_proj.weight", p)),
                .v = try ownT16(w, s, layerKey(&k, "{s}self_attn.v_proj.weight", p)),
                .o = try ownT16(w, s, layerKey(&k, "{s}self_attn.o_proj.weight", p)),
                .qn = try ownF16Plus1(w, s, alloc, layerKey(&k, "{s}self_attn.q_norm.weight", p)),
                .kn = try ownF16Plus1(w, s, alloc, layerKey(&k, "{s}self_attn.k_norm.weight", p)),
                .gate = try ownT16(w, s, layerKey(&k, "{s}mlp.gate_proj.weight", p)),
                .up = try ownT16(w, s, layerKey(&k, "{s}mlp.up_proj.weight", p)),
                .down = try ownT16(w, s, layerKey(&k, "{s}mlp.down_proj.weight", p)),
                .in_ln = try ownF16Plus1(w, s, alloc, layerKey(&k, "{s}pre_self_attn_layernorm.weight", p)),
                .post_ln = try ownF16Plus1(w, s, alloc, layerKey(&k, "{s}post_feedforward_layernorm.weight", p)),
                .pre_ff = try ownF16Plus1(w, s, alloc, layerKey(&k, "{s}pre_feedforward_layernorm.weight", p)),
                .eps_arr = mlx.mlx_array_new_float(cfg.te_rms_eps),
            };
            // post_self_attn_layernorm rides alongside as the attn post norm.
            const ps = try ownF16Plus1(w, s, alloc, layerKey(&k, "{s}post_self_attn_layernorm.weight", p));
            te_post_attn = appendHandle(te_post_attn, alloc, ps);
        }
        return .{
            .layers = layers,
            .norm = try ownF16Plus1(w, s, alloc, "text_encoder.norm.weight"),
            .table = &[_]f32{},
            .hidden = cfg.te_hidden,
            .vocab = cfg.te_vocab,
            .head_dim = cfg.te_head_dim,
            .window = cfg.te_window,
            .theta_full = cfg.te_theta_full,
            .theta_slide = cfg.te_theta_slide,
            .factor_full = cfg.te_factor_full,
            .eps = cfg.te_rms_eps,
            .full_layers = cfg.te_full_layers,
        };
    }

    fn loadTable(cfg: *const Config, w: *const Weights, s: S, alloc: std.mem.Allocator) ![]const f32 {
        const arr = w.get("text_encoder.embed_tokens.weight") orelse return error.MissingBreezeWeight;
        var f32arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32arr);
        try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f32arr));
        const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
        const n: usize = @intCast(cfg.te_vocab);
        const h: usize = @intCast(cfg.te_hidden);
        const out = try alloc.alloc(f32, n * h);
        const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(h))));
        for (0..n) |r| {
            const src = data[r * h ..][0..h];
            const dst = out[r * h ..][0..h];
            if (r == cfg.eoi_token_id) {
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
    var buf = alloc.alloc(f32, L * te.hidden) catch return error.OutOfMemory;
    defer alloc.free(buf);
    for (ids, 0..) |id, i| {
        if (id >= te.vocab) return error.TokenBeyondTeVocab;
        @memcpy(buf[i * te.hidden ..][0..te.hidden], te.table[@as(usize, id) * te.hidden ..][0..te.hidden]);
    }
    const sh = [_]c_int{ 1, @intCast(L), @intCast(te.hidden) };
    var x = hostF32(buf, &sh);

    for (te.layers, te_post_attn, 0..) |lw, ps, li| {
        const full = te.isFullLayer(li);
        const xn = try rmsn(x, lw.in_ln, te.eps, s);
        defer _ = mlx.mlx_array_free(xn);
        const q = try mm(xn, lw.q, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try mm(xn, lw.k, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try mm(xn, lw.v, s);
        defer _ = mlx.mlx_array_free(v);
        const qh: c_int = @divExact(mlx.getShape(lw.q)[1], @as(c_int, @intCast(te.head_dim)));
        const kh: c_int = @divExact(mlx.getShape(lw.k)[1], @as(c_int, @intCast(te.head_dim)));
        const hd: c_int = @intCast(te.head_dim);
        const v4 = try reshape(v, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(vt);
        const theta = if (full) te.theta_full else te.theta_slide;
        const p_scale = if (full) (1.0 / te.factor_full) else 1.0;
        // TE's rope carries a per-layer position scale (YaRN-style on the full
        // layers) the fused QK kernel has no arm for, so the chain stays
        // composed here; the fused decode kernel covers the frame loop.
        var q4 = try reshape(q, &[_]c_int{ 1, @intCast(L), qh, hd }, s);
        var k4 = try reshape(k, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        defer _ = mlx.mlx_array_free(k4);
        {
            const q4f = try reshape(q4, &[_]c_int{ @intCast(L * @as(usize, @intCast(qh))), hd }, s);
            const q4n1 = try rmsn(q4f, lw.qn, te.eps, s);
            _ = mlx.mlx_array_free(q4f);
            const q4n = try reshape(q4n1, &[_]c_int{ 1, @intCast(L), qh, hd }, s);
            _ = mlx.mlx_array_free(q4n1);
            _ = mlx.mlx_array_free(q4);
            q4 = q4n;
            const k4f = try reshape(k4, &[_]c_int{ @intCast(L * @as(usize, @intCast(kh))), hd }, s);
            const k4n1 = try rmsn(k4f, lw.kn, te.eps, s);
            _ = mlx.mlx_array_free(k4f);
            const k4n = try reshape(k4n1, &[_]c_int{ 1, @intCast(L), kh, hd }, s);
            _ = mlx.mlx_array_free(k4n1);
            _ = mlx.mlx_array_free(k4);
            k4 = k4n;
        }
        const qt = try transpose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(qt);
        const kt = try transpose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const qr = try ropeApply(qt, theta, p_scale, @intCast(te.head_dim), s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try ropeApply(kt, theta, p_scale, @intCast(te.head_dim), s);
        defer _ = mlx.mlx_array_free(kr);
        const mask = if (full) try teFullMask(s, L) else try teLocalMask(s, L, te.window);
        const mask_live = mask.ctx != null;
        defer {
            if (mask_live) _ = mlx.mlx_array_free(mask);
        }
        const att = try sdpa(qr, kr, vt, @as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(te.head_dim))))), mask, s);
        defer _ = mlx.mlx_array_free(att);
        const at = try transpose(att, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(at);
        const af = try reshape(at, &[_]c_int{ 1, @intCast(L), @intCast(te.layers[0].q_heads * te.head_dim) }, s);
        defer _ = mlx.mlx_array_free(af);
        const o = try mm(af, lw.o, s);
        defer _ = mlx.mlx_array_free(o);
        const on = try rmsn(o, ps, te.eps, s);
        defer _ = mlx.mlx_array_free(on);
        const h1 = try addM(x, on, s);
        defer _ = mlx.mlx_array_free(h1);
        const hn = try rmsn(h1, lw.pre_ff, te.eps, s);
        defer _ = mlx.mlx_array_free(hn);
        const g = try mm(hn, lw.gate, s);
        defer _ = mlx.mlx_array_free(g);
        const u = try mm(hn, lw.up, s);
        defer _ = mlx.mlx_array_free(u);
        const act = try geluTanhArr(g, s);
        defer _ = mlx.mlx_array_free(act);
        const gu = try mulEl(s, act, u);
        defer _ = mlx.mlx_array_free(gu);
        const dn = try mm(gu, lw.down, s);
        defer _ = mlx.mlx_array_free(dn);
        const dnn = try rmsn(dn, lw.post_ln, te.eps, s);
        defer _ = mlx.mlx_array_free(dnn);
        const x_prev = x;
        x = try addM(h1, dnn, s);
        _ = mlx.mlx_array_free(x_prev);
    }
    const x_norm_prev = x;
    const xn = try rmsn(x, te.norm, te.eps, s);
    defer _ = mlx.mlx_array_free(xn);
    _ = mlx.mlx_array_free(x_norm_prev);
    const last = try reshape(xn, &[_]c_int{ @intCast(L), @intCast(te.hidden) }, s);
    defer _ = mlx.mlx_array_free(last);
    try evalHost(s, last, out);
}

fn teFullMask(s: S, L: usize) !mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    @memset(buf, 0);
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    defer _ = mlx.mlx_array_free(h);
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&r, h, .bfloat16, s));
    return r;
}

fn teLocalMask(s: S, L: usize, window: usize) !mlx.mlx_array {
    const buf = std.heap.page_allocator.alloc(f32, L * L) catch return null_arr;
    defer std.heap.page_allocator.free(buf);
    const left = (window + 1) / 2;
    const right = window / 2 + 1;
    for (0..L) |i| {
        for (0..L) |j| {
            const allowed = (j <= i and i - j < left) or (j > i and j - i < right);
            buf[i * L + j] = if (allowed) 0 else -std.math.inf(f32);
        }
    }
    const h = hostF32(buf, &[_]c_int{ 1, 1, @intCast(L), @intCast(L) });
    defer _ = mlx.mlx_array_free(h);
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&r, h, .bfloat16, s));
    return r;
}

fn geluTanhArr(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    // 0.5·x·(1+tanh(√(2/π)(x+0.044715x³))) — TE hidden_activation
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_multiply(&x2, x, x, s));
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_multiply(&x3, x2, x, s));
    const sc1 = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(sc1);
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_multiply(&k, x3, sc1, s));
    var inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner);
    try mlx.check(mlx.mlx_add(&inner, x, k, s));
    const sc2 = mlx.mlx_array_new_float(0.7978845608);
    defer _ = mlx.mlx_array_free(sc2);
    var lin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lin);
    try mlx.check(mlx.mlx_multiply(&lin, inner, sc2, s));
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, lin, s));
    const sc3 = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(sc3);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_add(&one, th, sc3, s));
    const sc4 = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(sc4);
    var half = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(half);
    try mlx.check(mlx.mlx_multiply(&half, one, sc4, s));
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&r, x, half, s));
    return r;
}

// ───────────────────────── heads (host) ──────────────────────────────────────

fn readTable(s: S, w: *const Weights, key: []const u8, out: []f32, expect_rows: usize, expect_cols: usize) !void {
    const arr = w.get(key) orelse {
        log.err("[breeze] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingBreezeWeight;
    };
    const shp = mlx.getShape(arr);
    if (shp.len != 2 or @as(usize, @intCast(shp[0])) != expect_rows or @as(usize, @intCast(shp[1])) != expect_cols) {
        log.err("[breeze] {s} is {d}x{d}, the config asks for {d}x{d}\n", .{ key, shp[0], shp[1], expect_rows, expect_cols });
        return error.BadPackConfig;
    }
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
    /// Generated frames, each `[num_codebooks]` wide: views into `backing`,
    /// the flat buffer the frame loop grew. The host frees the row views and
    /// their buffer together (`freeSynthesisFrames`) when the job's WAV lands.
    frames: []const []const u32,
    backing: []const u32,
};

/// Free a `Synthesis`'s frame storage: the row-view slice and the flat
/// buffer they point into.
pub fn freeSynthesisFrames(alloc: std.mem.Allocator, syn: Synthesis) void {
    alloc.free(@constCast(syn.frames));
    alloc.free(@constCast(syn.backing));
}

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
    cfg: Config,
    tok: @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer,
    te: Te,
    te_proj: mlx.mlx_array, // [1152, 2048] f16 pre-transposed
    b_layers: []QLayer,
    b_norm: mlx.mlx_array,
    d_layers: []QLayer,
    d_norm: mlx.mlx_array,
    d_inp: mlx.mlx_array, // depth input projection [2048, 1024] f16
    lm_host: []f32, // [cbLm, b_hidden] canonical (codes + pad rows + eos row)
    heads_host: []f32, // [num_codebooks-1, cbVocab, d_hidden]
    bkv: []KV,
    bkv_len: usize = 0,
    audio_table: []f32, // HOST [num_codebooks*cbStride, b_hidden]
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

        const cfg = cfg_mod.readPackConfig(io, allocator, model_dir) catch |e| {
            log.err("[breeze] {s} config.json refused: {s}\n", .{ model_dir, @errorName(e) });
            return e;
        };
        const cb_lm = @as(usize, cfg.cbLm());
        if (_hist_seen.len < cb_lm) {
            if (_hist_seen.len > 0) allocator.free(_hist_seen);
            _hist_seen = allocator.alloc(bool, cb_lm) catch _hist_seen;
        }

        var self = try allocator.create(Engine);
        errdefer allocator.destroy(self);

        var w = try model_mod.loadWeights(io, allocator, model_dir);
        defer w.deinit();

        var te = try Te.load(&cfg, &w, s, allocator);
        errdefer allocator.free(te.layers);
        te.table = try Te.loadTable(&cfg, &w, s, allocator);
        errdefer allocator.free(te.table);

        const lm_host = try allocator.alloc(f32, cb_lm * cfg.b_hidden);
        errdefer allocator.free(lm_host);
        // lm_head: torch [cbLm, hidden] audio rows 0..cbVocab-1, canonical.
        {
            const arr = w.get("lm_head.weight") orelse return error.MissingBreezeWeight;
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
            const lh: usize = @intCast(mlx.getShape(arr)[1]);
            if (lh != cfg.b_hidden) return error.BadPackConfig;
            for (0..cb_lm) |r| @memcpy(lm_host[r * lh ..][0..lh], data[r * lh ..][0..lh]);
        }

        const n_cb: usize = @intCast(cfg.num_codebooks);
        const cbv: usize = @intCast(cfg.cbVocab());
        const heads_host = try allocator.alloc(f32, (n_cb - 1) * cbv * cfg.d_hidden);
        errdefer allocator.free(heads_host);
        // codebooks_head [15, hidden, vocab] — canonicalize to [15, out, in].
        {
            const arr = w.get("depth_decoder.codebooks_head.weight") orelse return error.MissingBreezeWeight;
            var f32arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f32arr);
            try mlx.check(mlx.mlx_astype(&f32arr, arr, .float32, s));
            try mlx.check(mlx.mlx_array_eval(f32arr));
            const data = mlx.mlx_array_data_float32(f32arr) orelse return error.NoData;
            const shp = mlx.getShape(arr);
            if (shp.len != 3 or @as(usize, @intCast(shp[0])) != n_cb - 1 or
                (@as(usize, @intCast(shp[1])) != cbv and @as(usize, @intCast(shp[2])) != cbv)) return error.BadPackConfig;
            if (shp[1] == cbv) {
                @memcpy(heads_host, data[0..heads_host.len]);
            } else {
                const dh: usize = @intCast(shp[1]);
                var h: usize = 0;
                while (h < n_cb - 1) : (h += 1) {
                    for (0..dh) |i| {
                        for (0..cbv) |o| {
                            heads_host[h * cbv * dh + o * dh + i] = data[h * dh * cbv + i * cbv + o];
                        }
                    }
                }
            }
        }

        const audio_table = try allocator.alloc(f32, n_cb * cbv * cfg.b_hidden);
        errdefer allocator.free(audio_table);
        try readTable(s, &w, "depth_decoder.model.embed_tokens.weight", audio_table, n_cb * cbv, cfg.b_hidden);

        self.* = .{
            .alloc = allocator,
            .cfg = cfg,
            .tok = try tokenizer_mod.loadTokenizer(io, allocator, model_dir),
            .te = te,
            .te_proj = undefined,
            .model_dir = try allocator.dupe(u8, model_dir),
            .io = io,
            .mimi = null,
            .b_layers = try loadQLayers("backbone_model.layers.{d}.", &w, allocator, s, .{
                .count = cfg.b_layers, .qk_norm = true, .q_heads = cfg.b_heads, .kv_heads = cfg.b_kv,
                .head_dim = cfg.b_head_dim, .eps = cfg.b_rms_eps, .theta = cfg.b_theta,
            }),
            .b_norm = try ownF16(&w, s, "backbone_model.norm.weight"),
            .d_layers = try loadQLayers("depth_decoder.model.layers.{d}.", &w, allocator, s, .{
                .count = cfg.d_layers, .q_heads = cfg.d_heads, .kv_heads = cfg.d_kv,
                .head_dim = cfg.d_head_dim, .eps = cfg.d_rms_eps, .theta = cfg.d_theta,
            }),
            .d_norm = try ownF16(&w, s, "depth_decoder.model.norm.weight"),
            .d_inp = try ownT16(&w, s, "depth_decoder.model.inputs_embeds_projector.weight"),
            .lm_host = lm_host,
            .heads_host = heads_host,
            .audio_table = audio_table,
            .codec = blk: {
                const tok_dir = try std.fmt.allocPrint(allocator, "{s}/audio_tokenizer", .{model_dir});
                defer allocator.free(tok_dir);
                break :blk tts.loadCodecDecoderFrom(io, allocator, s, tok_dir) catch |e| {
                    allocator.free(tok_dir); // the defer above would double-free on the error path
                    return e;
                };
            },
            .scratch = try allocator.alloc(f32, cb_lm),
            .order = try allocator.alloc(usize, cb_lm),
            .bkv_len = 0,
            .bkv = blk: {
                const tmp = allocator.alloc(KV, @as(usize, cfg.b_layers)) catch return error.OutOfMemory;
                for (tmp) |*e| e.* = .{ .k = null_arr, .v = null_arr };
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
        const a = self.alloc;
        self.tok.deinit();
        self.codec.deinit();
        for (self.bkv) |*e| {
            if (e.k.ctx != null) _ = mlx.mlx_array_free(e.k);
            if (e.v.ctx != null) _ = mlx.mlx_array_free(e.v);
            e.k.ctx = null;
            e.v.ctx = null;
        }
        a.free(self.bkv);
        a.free(self.te.table);
        for (self.te.layers) |*l| l.freeHandles();
        a.free(self.te.layers);
        for (self.b_layers) |*l| l.freeHandles();
        a.free(self.b_layers);
        for (self.d_layers) |*l| l.freeHandles();
        a.free(self.d_layers);
        _ = mlx.mlx_array_free(self.b_norm);
        _ = mlx.mlx_array_free(self.d_norm);
        _ = mlx.mlx_array_free(self.d_inp);
        _ = mlx.mlx_array_free(self.te_proj);
        for (te_post_attn) |ps| _ = mlx.mlx_array_free(ps);
        a.free(te_post_attn);
        te_post_attn = &[_]mlx.mlx_array{};
        if (d_inp_cache.len > 0) a.free(d_inp_cache);
        d_inp_cache = &[_]f32{};
        d_inp_key = .{ 0, 0 };
        if (_hist_seen.len > 0) a.free(_hist_seen);
        _hist_seen = &[_]bool{};
        a.free(self.lm_host);
        a.free(self.heads_host);
        a.free(self.audio_table);
        if (self.mimi) |m| { m.deinit(); a.destroy(m); }
        a.free(self.model_dir);
        a.free(self.scratch);
        a.free(self.order);
        a.destroy(self);
    }

    pub fn sampleRate(self: *const Engine) u32 {
        return self.codec.cfg.sampling_rate;
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
            try seg1.append(allocator, self.cfg.speaker_base + @as(u32, v[1] - '0'));
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
                try seg2.append(allocator, self.cfg.speaker_base + @as(u32, v[1] - '0'));
            }
            if (req.instruction) |ins| {
                try seg2.append(allocator, self.cfg.ins_bos_id);
                try self.enc(&seg2, ins);
                try seg2.append(allocator, self.cfg.ins_eos_id);
            }
        } else {
            if (req.instruction) |ins| {
                try seg1.append(allocator, self.cfg.ins_bos_id);
                try self.enc(&seg1, ins);
                try seg1.append(allocator, self.cfg.ins_eos_id);
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
                if (d.sample_rate != self.sampleRate()) return error.UnsupportedRefRate; // resample skipped
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
        if (segmentFits(overhead, whole_toks.items.len, speakUnits(trimmed), self.cfg.max_seq, self.cfg.max_frames)) {
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
                if (si > p_start and !segmentFits(overhead, p_toks + pack.items.len, p_units + s_units, self.cfg.max_seq, self.cfg.max_frames)) {
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

        const bh: usize = self.cfg.b_hidden;
        const dh: usize = self.cfg.d_hidden;
        const th: usize = self.cfg.te_hidden;
        const n_cb: usize = self.cfg.numCodebooks();
        const cbv: usize = @intCast(self.cfg.cbVocab());
        const cb_lm: usize = @intCast(self.cfg.cbLm());
        const stride: usize = @intCast(self.cfg.cbStride());
        const cb_eos: u32 = self.cfg.cbEos();
        const max_seq: usize = @intCast(self.cfg.max_seq);
        const max_code = self.cfg.codebook_size;

        var frames: std.ArrayListUnmanaged(u32) = .empty; // stitched across segments; frame t is items[t*n_cb..][0..n_cb]
        defer frames.deinit(allocator);
        var hist: std.ArrayList(u32) = .empty; // repetition-penalty history, spans segments
        defer hist.deinit(allocator);

        const h_row = allocator.alloc(f32, bh) catch return error.OutOfMemory;
        defer allocator.free(h_row);
        var h_proj = allocator.alloc(f32, dh) catch return error.OutOfMemory;
        defer allocator.free(h_proj);
        const emb_row = allocator.alloc(f32, bh) catch return error.OutOfMemory;
        defer allocator.free(emb_row);
        const logits = allocator.alloc(f32, cb_lm) catch return error.OutOfMemory;
        defer allocator.free(logits);
        const frame_row = allocator.alloc(u32, n_cb) catch return error.OutOfMemory;
        defer allocator.free(frame_row);
        @memset(frame_row, 0);
        var depth_span = allocator.alloc(f32, (n_cb + 1) * dh) catch return error.OutOfMemory;
        defer allocator.free(depth_span);

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
                for (0..ref_frames.items.len) |_| try ids.append(allocator, self.cfg.audio_token_id);
                try ids.append(allocator, self.cfg.audio_eos_id);
                try ids.appendSlice(allocator, s2.items);
            } else {
                try s1.appendSlice(allocator, toks.items); // plain: text rides in seg1
                try ids.appendSlice(allocator, s1.items);
            }

            const text_len = ids.items.len;
            var frames_cap = estimateFrames(speakUnits(seg_text), req.max_frames, self.cfg.max_frames);
            // The packer keeps segments window-safe; this clamp is the safety
            // net so even a model that never emits EOS stays in the span.
            if (text_len < max_seq and frames_cap > max_seq - text_len)
                frames_cap = @intCast(max_seq - text_len);
            const total = text_len + frames_cap;
            if (total > max_seq) return error.PromptTooLongForFrameBudget;

            // TE conditioning → backbone-width stream rows.
            var cond = allocator.alloc(f32, @max(s1.items.len, s2.items.len) * th) catch return error.OutOfMemory;
            defer allocator.free(cond);
            var stream = allocator.alloc(f32, total * bh) catch return error.OutOfMemory;
            defer allocator.free(stream);
            @memset(stream, 0);
            {
                const wt = allocator.alloc(f32, th * bh) catch return error.OutOfMemory;
                defer allocator.free(wt);
                try evalHost(g_stream, self.te_proj, wt);
                var row: usize = 0;
                try teCond(allocator, g_stream, &self.te, s1.items, cond[0 .. s1.items.len * th]);
                for (0..s1.items.len) |t| {
                    const r = cond[t * th ..][0..th];
                    for (0..bh) |o| {
                        var acc: f32 = 0;
                        var i: usize = 0;
                        while (i < th) : (i += 1) acc += r[i] * wt[o * th + i];
                        stream[row * bh + o] = acc;
                    }
                    row += 1;
                }
                if (clone) {
                    for (ref_frames.items) |f| {
                        frameEmbed(self.audio_table, bh, stride, f, max_code, emb_row);
                        @memcpy(stream[row * bh ..][0..bh], emb_row);
                        row += 1;
                    }
                    const zframe = frame_row;
                    @memset(zframe, 0);
                    frameEmbed(self.audio_table, bh, stride, zframe, max_code, emb_row);
                    @memcpy(stream[row * bh ..][0..bh], emb_row);
                    row += 1;
                    try teCond(allocator, g_stream, &self.te, s2.items, cond[0 .. s2.items.len * th]);
                    for (0..s2.items.len) |t| {
                        const r = cond[t * th ..][0..th];
                        for (0..bh) |o| {
                            var acc: f32 = 0;
                            var i: usize = 0;
                            while (i < th) : (i += 1) acc += r[i] * wt[o * th + i];
                            stream[row * bh + o] = acc;
                        }
                        row += 1;
                    }
                }
            }

            for (self.bkv) |*e| {
                if (e.k.ctx != null) _ = mlx.mlx_array_free(e.k);
                if (e.v.ctx != null) _ = mlx.mlx_array_free(e.v);
                e.k.ctx = null;
                e.v.ctx = null;
            }
            var span = allocator.alloc(f32, total * bh) catch return error.OutOfMemory;
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
                if (frame_i == 0) {
                    const L = text_len + frame_i;
                    @memcpy(span[0 .. L * bh], stream[0 .. L * bh]);
                    try qwenForward(g_stream, self.b_layers, self.b_norm, span[0 .. L * bh], L, bh, 1.0, 0, null_arr, self.bkv, h_row);
                    self.bkv_len = text_len;
                } else {
                    try qwenStep(g_stream, self.b_layers, self.b_norm, stream[(text_len + frame_i - 1) * bh ..][0..bh], bh, @intCast(self.bkv_len), self.bkv, h_row);
                    self.bkv_len += 1;
                }
                headLogits(self.lm_host, h_row, logits, cb_lm, bh, 0);
                const frame = try allocator.alloc(u32, n_cb);
                errdefer allocator.free(frame); // freed after it folds into `frames`
                frame[0] = @intCast(sampleToken(&rng, logits, .{}, self.scratch, self.order, .{ .suppress_from = max_code, .suppress_to = @intCast(cbv), .history = hist.items, .rep = 1.1 }));
                if (frame[0] == cb_eos) {
                    allocator.free(frame);
                    break;
                }

                // depth span: [backbone_hidden_proj, emb(cb0), emb(cb1), ...]
                try projHostCached(allocator, g_stream, self.d_inp, h_row, bh, dh, h_proj[0..dh]);
                @memcpy(depth_span[0..dh], h_proj);
                var dl: usize = 1;
                var cb: usize = 1;
                while (cb < n_cb) : (cb += 1) {
                    const code = frame[cb - 1];
                    tableRow(self.audio_table, bh, stride, cb - 1, @min(code, @as(u32, @intCast(cbv - 1))), emb_row);
                    try projHostCached(allocator, g_stream, self.d_inp, emb_row, bh, dh, depth_span[dl * dh ..][0..dh]);
                    dl += 1;
                    const d_last = allocator.alloc(f32, dh) catch return error.OutOfMemory;
                    defer allocator.free(d_last);
                    try qwenForward(g_stream, self.d_layers, self.d_norm, depth_span[0 .. dl * dh], dl, dh, 1.0, 0, null_arr, null, d_last);
                    headLogits(self.heads_host, d_last, logits, cbv, dh, cb - 1);
                    frame[cb] = @intCast(sampleToken(&rng, logits[0..cbv], .{}, self.scratch, self.order, .{ .suppress_from = max_code, .suppress_to = @intCast(cbv) }));
                }
                if (frame_i % 4 == 3) { _ = mlx.mlx_clear_cache(); }
                try hist.append(allocator, frame[0]);
                // frame embed row into the stream
                frameEmbed(self.audio_table, bh, stride, frame, max_code, emb_row);
                @memcpy(stream[(text_len + frame_i) * bh ..][0..bh], emb_row);
                try frames.appendSlice(allocator, frame);
                allocator.free(frame);
                if (st) |spec| {
                    const n = frames.items.len / n_cb;
                    const k = if (spec.chunk_frames == 0) 12 else spec.chunk_frames;
                    if (n % k == 0) {
                        // Prefix-exact under the causal codec decoder: this
                        // decode's tail equals the final decode's same region.
                        const views = allocator.alloc([]const u32, n) catch return error.OutOfMemory;
                        defer allocator.free(views);
                        for (0..n) |t| views[t] = frames.items[t * n_cb ..][0..n_cb];
                        const dec = try self.codec.decodeSlice(views);
                        defer allocator.free(dec);
                        const dec_total = @min(dec.len, n * self.samples_per_frame);
                        const from = @min(dec_total, (n - k) * self.samples_per_frame);
                        spec.emit(spec.emit_user, dec[from..dec_total]);
                    }
                }
            }
        }

        // The flat frame buffer the list grew into rides straight into the
        // Synthesis; the row views are borrowed slices of it. The caller
        // frees both through `freeSynthesisFrames`.
        const n_total = frames.items.len / n_cb;
        if (n_total == 0) return error.NoFramesEmitted;
        const backing = frames.toOwnedSlice(allocator) catch return error.OutOfMemory;
        errdefer allocator.free(backing);
        const out_frames = allocator.alloc([]const u32, n_total) catch return error.OutOfMemory;
        errdefer allocator.free(out_frames);
        for (0..n_total) |t| out_frames[t] = backing[t * n_cb ..][0..n_cb];

        if (st) |spec| {
            const n = out_frames.len;
            const k = if (spec.chunk_frames == 0) 12 else spec.chunk_frames;
            const rem = n % k;
            if (k > 0 and rem > 0) {
                const dec = try self.codec.decodeSlice(out_frames);
                defer allocator.free(dec);
                const dec_total = @min(dec.len, n * self.samples_per_frame);
                const from = @min(dec_total, (n - rem) * self.samples_per_frame);
                spec.emit(spec.emit_user, dec[from..dec_total]);
            }
        }

        if (st == null) {
            const pcm = try self.codec.decodeSlice(out_frames);
            return .{ .pcm = pcm, .frames = out_frames, .backing = backing };
        }
        // Stream mode: PCM already emitted in chunks; Synthesis.pcm is empty
        // and must not be freed.
        return .{ .pcm = &[_]f32{}, .frames = out_frames, .backing = backing };
    }
};

var _hist_seen: []bool = &[_]bool{};
var mlx_allocator: std.mem.Allocator = undefined;
var g_stream: S = undefined;

fn tableRow(table: []const f32, width: usize, stride: usize, cb: usize, code: u32, out: []f32) void {
    @memcpy(out, table[(@as(usize, cb) * stride + @as(usize, code)) * width ..][0..width]);
}

fn frameEmbed(table: []const f32, width: usize, stride: usize, codes: []const u32, max_code: u32, out: []f32) void {
    @memset(out, 0);
    for (codes, 0..) |code, cb| {
        const c = @min(code, max_code);
        const row = table[(@as(usize, cb) * stride + @as(usize, c)) * width ..][0..width];
        for (out, row) |*o, rv| o.* += rv;
    }
}

var d_inp_cache: []f32 = &[_]f32{};

var d_inp_key: [2]usize = .{ 0, 0 };

fn projHostCached(alloc: std.mem.Allocator, s: S, w: mlx.mlx_array, x: []const f32, in: usize, out_ch: usize, out: []f32) !void {
    if (d_inp_key[0] != in or d_inp_key[1] != out_ch) {
        if (d_inp_cache.len > 0) alloc.free(d_inp_cache);
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

/// One decoder stack's shape, read from the pack config by `Engine.load`.
const LayerDims = struct {
    count: usize,
    qk_norm: bool = false,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f64,
};

/// One weight key for layer `i`, formatted through `buf` so the caller frees
/// ONE string per layer instead of one per tensor.
fn layerKey(buf: *[256]u8, comptime fmt: []const u8, p: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, fmt, .{p}) catch unreachable;
}

fn loadQLayers(comptime prefix_fmt: []const u8, w: *const Weights, alloc: std.mem.Allocator, s: S, d: LayerDims) ![]QLayer {
    const layers = try alloc.alloc(QLayer, d.count);
    errdefer alloc.free(layers);
    for (layers, 0..) |*L, i| {
        const p = try std.fmt.allocPrint(alloc, prefix_fmt, .{i});
        defer alloc.free(p);
        var k: [256]u8 = undefined;
        L.* = .{
            .q_heads = d.q_heads,
            .kv_heads = d.kv_heads,
            .head_dim = d.head_dim,
            .eps = d.eps,
            .theta = d.theta,
            .q = try ownT16(w, s, layerKey(&k, "{s}self_attn.q_proj.weight", p)),
            .k = try ownT16(w, s, layerKey(&k, "{s}self_attn.k_proj.weight", p)),
            .v = try ownT16(w, s, layerKey(&k, "{s}self_attn.v_proj.weight", p)),
            .o = try ownT16(w, s, layerKey(&k, "{s}self_attn.o_proj.weight", p)),
            .qn = if (d.qk_norm) try ownF16(w, s, layerKey(&k, "{s}self_attn.q_norm.weight", p)) else null_arr,
            .kn = if (d.qk_norm) try ownF16(w, s, layerKey(&k, "{s}self_attn.k_norm.weight", p)) else null_arr,
            .gate = try ownT16(w, s, layerKey(&k, "{s}mlp.gate_proj.weight", p)),
            .up = try ownT16(w, s, layerKey(&k, "{s}mlp.up_proj.weight", p)),
            .down = try ownT16(w, s, layerKey(&k, "{s}mlp.down_proj.weight", p)),
            .in_ln = try ownF16(w, s, layerKey(&k, "{s}input_layernorm.weight", p)),
            .post_ln = try ownF16(w, s, layerKey(&k, "{s}post_attention_layernorm.weight", p)),
            .pre_ff = null_arr,
            .eps_arr = mlx.mlx_array_new_float(d.eps),
        };
    }
    return layers;
}

// ───────────────────────────── tests ─────────────────────────────────────────

test "voice slot ids" {
    const base: u32 = 262146;
    try std.testing.expectEqual(@as(?u32, base), speakerToken(base, 0));
    try std.testing.expectEqual(@as(?u32, base + 9), speakerToken(base, 9));
    try std.testing.expectEqual(@as(?u32, null), speakerToken(base, 10));
}

test "voice slot validation" {
    try std.testing.expect(voiceSlotValid("S0"));
    try std.testing.expect(voiceSlotValid("s7"));
    try std.testing.expect(!voiceSlotValid("S10"));
    try std.testing.expect(!voiceSlotValid("S"));
    try std.testing.expect(!voiceSlotValid("V0"));
}

test "estimateFrames clamp" {
    const a = estimateFrames(300, null, 1500);
    try std.testing.expect(a > 0);
    try std.testing.expect(a <= 1500);
    const b = estimateFrames(300, 50, 1500);
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
    try std.testing.expect(segmentFits(4, 300, 1178, 2048, 1500));
    // > frame cap of speech on its own must not fit
    try std.testing.expect(!segmentFits(4, 400, 2500, 2048, 1500));
    // token span exhaustion must not fit
    try std.testing.expect(!segmentFits(4, 2045, 1, 2048, 1500));
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

test "live synthesize (BREEZE_MODEL)" {
    const dir = env.span(.BREEZE_MODEL) orelse return error.SkipZigTest;
    const alloc = std.testing.allocator;
    // Standalone gated test: no host, so this test owns its stream.
    var engine = try Engine.load(std.Io.Threaded.global_single_threaded.io(), alloc, mlx.mlx_default_gpu_stream_new(), dir);
    defer engine.deinit();
    const synth = try engine.synthesize(alloc, .{ .text = "Hello world, this is Breeze speaking.", .voice = "S0", .seed = 1 });
    try std.testing.expect(synth.pcm.len > 0);
    defer freeSynthesisFrames(alloc, synth);
    alloc.free(synth.pcm);
}

// The shared sdk codec's encoder parity bar, kept here with the fixtures. Clip + oracle are the
// first frames of a real utterance (u16-LE codes from qwen_tts's tokenizer on the same pack);
// MIMI_ENCODE still names a real pack, the encoder is real-sized.
test "mimi encoder parity vs oracle (MIMI_ENCODE)" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const tok_dir = env.span(.MIMI_ENCODE) orelse return error.SkipZigTest;
    const wav_path = env.span(.MIMI_WAV) orelse "src/plugins/breeze_tts/fixtures/mimi_clip.wav";
    const oracle_path = env.span(.MIMI_ORACLE) orelse "src/plugins/breeze_tts/fixtures/mimi_codes.bin";

    const alloc = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    const wav_bytes = try tts.readBytes(io, alloc, wav_path, 1 << 30);
    defer alloc.free(wav_bytes);
    const d = try wav_mod.decode(alloc, wav_bytes);
    defer alloc.free(d.pcm);

    var enc = try tts.loadMimiEncoder(io, alloc, s, tok_dir);
    defer enc.deinit();
    const frames = try enc.encodeSamples(d.pcm);
    defer {
        for (frames) |f| alloc.free(f);
        alloc.free(frames);
    }

    const oracle = try tts.readBytes(io, alloc, oracle_path, 1 << 20);
    defer alloc.free(oracle);
    const n_frames = oracle.len / (16 * 2);
    try std.testing.expectEqual(n_frames, frames.len);
    // Frames t>=1 must be byte-exact; frame 0 rides the left-pad edge where torch/MLX
    // accumulation order can flip near-tie codebook picks (≤ 16 cells).
    var mism: usize = 0;
    var mism_t0: usize = 0;
    for (frames, 0..) |row, t| {
        for (0..16) |q| {
            const ov = @as(u16, oracle[t * 32 + q * 2]) | (@as(u16, oracle[t * 32 + q * 2 + 1]) << 8);
            if (ov != row[q]) {
                if (t == 0) mism_t0 += 1 else mism += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mism);
    try std.testing.expectEqual(@as(u32, 259), frames[0][0]);
}
