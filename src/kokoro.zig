//! Kokoro-82M TTS — Zig + mlx-c port of hexgrad/Kokoro-82M (StyleTTS 2 + iSTFTNet).
//!
//! NON-AUTOREGRESSIVE: one forward pass per utterance, no sampling loop. That is
//! the whole reason an 82M model beats a 600M autoregressive one on both speed
//! and quality — do not "optimize" it into a decode loop.
//!
//! Pipeline (mirrors `KModel.forward_with_tokens`):
//!   phoneme ids [1,T]
//!     → ALBERT (12 layers, SHARED weights, hidden 768) → [1,T,768]
//!     → bert_encoder Linear(768→512) → d_en [1,T,512]
//!     → DurationEncoder (3× [BiLSTM(640→512), AdaLayerNorm]) → d [1,T,640]
//!     → BiLSTM + duration_proj → per-phoneme durations [T]
//!     → EXPAND by duration (see `expandIndices`) → en [1,F,640], asr [1,F,512]
//!     → F0Ntrain (BiLSTM + 3× AdainResBlk1d, block 1 upsamples ×2) → F0/N [1,2F]
//!     → Decoder + iSTFTNet Generator → waveform [F*600] @ 24 kHz
//!
//! A voice is NOT one vector — it is a `[510, 1, 256]` TABLE indexed by the
//! phoneme count of the utterance (`pack[len(ps)-1]` in the reference). Pick
//! the row, then dims [128:] drive prosody+duration and dims [:128] drive the
//! decoder. Indexing with a fixed row instead of the real length is a silent
//! quality regression, not a crash.
//!
//! Voices BLEND by plain mean over whole packs — the reference spells a blend
//! as a comma-separated name (`"af_bella,af_jessica"`), which `/v1/audio/speech`
//! mirrors. That is the supported way to make a new voice, not a fine-tune.
//!
//! Layout conventions (match `ltx_audio.zig`, whose conv helpers this reuses):
//!   - activations are NLC `[B, L, C]`, MLX-native;
//!   - conv weights stay in PyTorch layout and are transposed at USE;
//!   - weight-norm is PRE-FOLDED by `tests/convert_kokoro_weights.py`
//!     (`weight = g · v/‖v‖`), so there are no `.weight_g`/`.weight_v` here.
//!
//! Numerics: f32 end to end. The whole model is 82M (~330 MB f32), so there is
//! nothing to win from bf16, and the iSTFT head is precision-sensitive the same
//! way ACE-Step's Snake/encode path is.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const S = mlx.mlx_stream;

// ════════════════════════════════════════════════════════════════════════
// Config
// ════════════════════════════════════════════════════════════════════════

/// Kokoro `config.json`. Defaults are the published v1.0 values, so a minimal
/// config still loads (same tolerance as `tts.parseConfig`).
pub const Config = struct {
    // Trunk.
    n_token: u32 = 178,
    hidden_dim: u32 = 512,
    style_dim: u32 = 128,
    n_layer: u32 = 3,
    max_dur: u32 = 50,
    max_conv_dim: u32 = 512,
    dim_in: u32 = 64,
    n_mels: u32 = 80,
    text_encoder_kernel_size: u32 = 5,

    // PLBERT (ALBERT — one layer's weights, applied `num_hidden_layers` times).
    bert_hidden: u32 = 768,
    bert_heads: u32 = 12,
    bert_inter: u32 = 2048,
    bert_layers: u32 = 12,
    bert_max_pos: u32 = 512,

    // iSTFTNet generator.
    upsample_rates: [2]u32 = .{ 10, 6 },
    upsample_kernel_sizes: [2]u32 = .{ 20, 12 },
    upsample_initial_channel: u32 = 512,
    gen_istft_n_fft: u32 = 20,
    gen_istft_hop_size: u32 = 5,

    sample_rate: u32 = 24000,

    /// Samples of audio produced per input frame: ∏upsample_rates × hop.
    /// 10 × 6 × 5 = 300, and F0/N run at 2× the frame rate, so one phoneme
    /// frame is 600 samples (25 ms).
    pub fn samplesPerFrame(self: Config) u32 {
        return self.upsample_rates[0] * self.upsample_rates[1] * self.gen_istft_hop_size;
    }
};

pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) !Config {
    var cfg = Config{};
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return cfg;
    defer parsed.deinit();
    if (parsed.value != .object) return cfg;
    const obj = parsed.value.object;

    setU32(&cfg.n_token, obj, "n_token");
    setU32(&cfg.hidden_dim, obj, "hidden_dim");
    setU32(&cfg.style_dim, obj, "style_dim");
    setU32(&cfg.n_layer, obj, "n_layer");
    setU32(&cfg.max_dur, obj, "max_dur");
    setU32(&cfg.max_conv_dim, obj, "max_conv_dim");
    setU32(&cfg.dim_in, obj, "dim_in");
    setU32(&cfg.n_mels, obj, "n_mels");
    setU32(&cfg.text_encoder_kernel_size, obj, "text_encoder_kernel_size");

    if (obj.get("plbert")) |v| if (v == .object) {
        const p = v.object;
        setU32(&cfg.bert_hidden, p, "hidden_size");
        setU32(&cfg.bert_heads, p, "num_attention_heads");
        setU32(&cfg.bert_inter, p, "intermediate_size");
        setU32(&cfg.bert_layers, p, "num_hidden_layers");
        setU32(&cfg.bert_max_pos, p, "max_position_embeddings");
    };

    if (obj.get("istftnet")) |v| if (v == .object) {
        const g = v.object;
        setU32(&cfg.upsample_initial_channel, g, "upsample_initial_channel");
        setU32(&cfg.gen_istft_n_fft, g, "gen_istft_n_fft");
        setU32(&cfg.gen_istft_hop_size, g, "gen_istft_hop_size");
        setU32Pair(&cfg.upsample_rates, g, "upsample_rates");
        setU32Pair(&cfg.upsample_kernel_sizes, g, "upsample_kernel_sizes");
    };

    return cfg;
}

fn setU32(dst: *u32, obj: std.json.ObjectMap, key: []const u8) void {
    if (obj.get(key)) |v| if (v == .integer and v.integer >= 0) {
        dst.* = @intCast(v.integer);
    };
}

fn setU32Pair(dst: *[2]u32, obj: std.json.ObjectMap, key: []const u8) void {
    const v = obj.get(key) orelse return;
    if (v != .array or v.array.items.len != 2) return;
    for (v.array.items, 0..) |it, i| {
        if (it != .integer or it.integer < 0) return;
        dst[i] = @intCast(it.integer);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Phoneme vocab
// ════════════════════════════════════════════════════════════════════════

/// Kokoro's phoneme table, parsed from `config.json`'s `vocab` object. Keys are
/// IPA symbols, most of them MULTI-BYTE UTF-8 (ɑ, ʃ, ˈ …), so lookup is by
/// codepoint slice, never by byte.
pub const Vocab = struct {
    map: std.StringHashMapUnmanaged(u16) = .{},
    allocator: std.mem.Allocator,
    /// Backing store for the key slices (owned).
    keys: std.ArrayListUnmanaged([]u8) = .empty,

    pub fn parse(allocator: std.mem.Allocator, json_text: []const u8) !Vocab {
        var self = Vocab{ .allocator = allocator };
        errdefer self.deinit();

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadKokoroConfig;
        const v = parsed.value.object.get("vocab") orelse return error.BadKokoroConfig;
        if (v != .object) return error.BadKokoroConfig;

        var it = v.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .integer) continue;
            const id: u16 = @intCast(e.value_ptr.integer);
            const key = try allocator.dupe(u8, e.key_ptr.*);
            try self.keys.append(allocator, key);
            try self.map.put(allocator, key, id);
        }
        return self;
    }

    pub fn deinit(self: *Vocab) void {
        for (self.keys.items) |k| self.allocator.free(k);
        self.keys.deinit(self.allocator);
        self.map.deinit(self.allocator);
    }

    pub fn get(self: *const Vocab, symbol: []const u8) ?u16 {
        return self.map.get(symbol);
    }

    /// Encode an IPA phoneme string to input ids, wrapped in the boundary token
    /// 0 at both ends (`KModel.forward`). Unknown symbols are DROPPED, matching
    /// the reference's `filter(lambda i: i is not None, ...)` — a symbol we
    /// cannot say is better skipped than voiced as garbage.
    pub fn encode(self: *const Vocab, allocator: std.mem.Allocator, phonemes: []const u8) ![]i32 {
        var ids: std.ArrayListUnmanaged(i32) = .empty;
        errdefer ids.deinit(allocator);
        try ids.append(allocator, 0);

        var i: usize = 0;
        while (i < phonemes.len) {
            const len = std.unicode.utf8ByteSequenceLength(phonemes[i]) catch 1;
            const end = @min(i + len, phonemes.len);
            if (self.get(phonemes[i..end])) |id| try ids.append(allocator, id);
            i = end;
        }

        try ids.append(allocator, 0);
        return ids.toOwnedSlice(allocator);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Duration → frame expansion
//
// The reference builds a [T, F] one-hot alignment matrix `pred_aln_trg` and
// does `d.transpose(-1,-2) @ pred_aln_trg`. That matmul only ever COPIES each
// phoneme column `dur[t]` times, so we take along the time axis with the same
// index vector instead: identical result, O(F) instead of O(T·F·640).
// ════════════════════════════════════════════════════════════════════════

/// `torch.round` is round-half-to-EVEN; Zig's `@round` is half-away-from-zero.
/// Durations land on exact .5 often enough (they are sums of 50 sigmoids) that
/// the difference is an audible one-frame drift against the reference.
fn roundHalfToEven(x: f32) f32 {
    const away = @round(x);
    if (@abs(x - @trunc(x)) != 0.5) return away;
    return if (@mod(away, 2.0) == 0.0) away else away - std.math.sign(x);
}

/// Per-phoneme frame counts from the duration head's raw logits.
/// `logits` is row-major `[T, max_dur]`; the reference is
/// `round(sigmoid(logits).sum(-1) / speed).clamp(min=1)`.
pub fn predictedDurations(
    allocator: std.mem.Allocator,
    logits: []const f32,
    n_tokens: usize,
    max_dur: usize,
    speed: f32,
) ![]u32 {
    std.debug.assert(logits.len == n_tokens * max_dur);
    const out = try allocator.alloc(u32, n_tokens);
    errdefer allocator.free(out);
    for (0..n_tokens) |t| {
        var sum: f32 = 0;
        for (logits[t * max_dur ..][0..max_dur]) |z| sum += 1.0 / (1.0 + @exp(-z));
        const rounded = roundHalfToEven(sum / speed);
        out[t] = if (rounded < 1) 1 else @intFromFloat(rounded);
    }
    return out;
}

/// `repeat_interleave(arange(T), durations)` — the frame→phoneme index vector.
/// Length is the total frame count.
pub fn expandIndices(allocator: std.mem.Allocator, durations: []const u32) ![]i32 {
    var total: usize = 0;
    for (durations) |d| total += d;

    const out = try allocator.alloc(i32, total);
    errdefer allocator.free(out);
    var k: usize = 0;
    for (durations, 0..) |d, t| {
        for (0..d) |_| {
            out[k] = @intCast(t);
            k += 1;
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Windowing
// ════════════════════════════════════════════════════════════════════════

/// `torch.hann_window(n, periodic=true)` = 0.5·(1 − cos(2πk/n)).
/// PERIODIC, not symmetric — `ltx_audio.zig`'s vocoder framing uses the
/// symmetric form and the two are not interchangeable.
pub fn hannPeriodic(allocator: std.mem.Allocator, n: usize) ![]f32 {
    const w = try allocator.alloc(f32, n);
    const nf: f32 = @floatFromInt(n);
    for (0..n) |k| {
        const kf: f32 = @floatFromInt(k);
        w[k] = 0.5 * (1.0 - @cos(2.0 * std.math.pi * kf / nf));
    }
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// Bidirectional LSTM
//
// Kokoro has SIX of them (TextEncoder.lstm, DurationEncoder ×3,
// ProsodyPredictor.lstm, ProsodyPredictor.shared) and nothing else in this
// repo needs one, so it lives here rather than in a shared module.
//
// Batch is always 1 and there is never padding (one utterance, no bucketing),
// so the reference's `pack_padded_sequence`/`masked_fill_` are all no-ops and
// the port carries no masking. If batching ever arrives, that stops being true.
// ════════════════════════════════════════════════════════════════════════

/// Small mlx helpers, local so this file reads without hopping modules.
fn mm(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, a, b, s));
    return out;
}

fn add2(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, a, b, s));
    return out;
}

fn mul2(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, a, b, s));
    return out;
}

fn sigmoid1(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&out, a, s));
    return out;
}

fn tanh1(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&out, a, s));
    return out;
}

fn transpose2(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, s));
    return out;
}

/// Slice `[start, stop)` along `axis`, keeping every other axis whole.
fn sliceAxis(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    var lo = try std.heap.c_allocator.alloc(c_int, shape.len);
    defer std.heap.c_allocator.free(lo);
    var hi = try std.heap.c_allocator.alloc(c_int, shape.len);
    defer std.heap.c_allocator.free(hi);
    var st = try std.heap.c_allocator.alloc(c_int, shape.len);
    defer std.heap.c_allocator.free(st);
    for (shape, 0..) |d, i| {
        lo[i] = 0;
        hi[i] = d;
        st[i] = 1;
    }
    lo[axis] = start;
    hi[axis] = stop;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, lo.ptr, lo.len, hi.ptr, hi.len, st.ptr, st.len, s));
    return out;
}

/// One LSTM direction. Weights are PyTorch `nn.LSTM` layout: `w_ih` is
/// `[4H, in]`, `w_hh` is `[4H, H]`, gates ordered i, f, g, o.
pub const LstmDir = struct {
    w_ih: mlx.mlx_array,
    w_hh: mlx.mlx_array,
    /// `b_ih + b_hh` pre-summed at load — PyTorch applies both and they are
    /// both constants, so carrying two is pure per-step work for no effect.
    bias: mlx.mlx_array,
};

pub const BiLstm = struct {
    fwd: LstmDir,
    rev: LstmDir,
    hidden: usize,

    /// `x` is `[1, T, in]`; returns `[1, T, 2H]` (forward ‖ reverse), matching
    /// `nn.LSTM(..., batch_first=True, bidirectional=True)`.
    pub fn forward(self: *const BiLstm, allocator: std.mem.Allocator, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const shape = mlx.getShape(x);
        const t_len: usize = @intCast(shape[1]);

        const f = try self.runDir(allocator, &self.fwd, x, t_len, false, s);
        defer _ = mlx.mlx_array_free(f);
        const r = try self.runDir(allocator, &self.rev, x, t_len, true, s);
        defer _ = mlx.mlx_array_free(r);

        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, f);
        _ = mlx.mlx_vector_array_append_value(vec, r);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
        return out;
    }

    /// The input projection `x @ w_ih^T + bias` is computed for ALL timesteps in
    /// ONE matmul; only the recurrent `h @ w_hh^T` has to be sequential. That is
    /// the difference between one big GEMM and T tiny ones.
    fn runDir(
        self: *const BiLstm,
        allocator: std.mem.Allocator,
        dir: *const LstmDir,
        x: mlx.mlx_array,
        t_len: usize,
        reverse: bool,
        s: S,
    ) !mlx.mlx_array {
        const h_dim = self.hidden;

        const w_ih_t = try transpose2(dir.w_ih, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(w_ih_t);
        const proj = try mm(x, w_ih_t, s); // [1, T, 4H]
        defer _ = mlx.mlx_array_free(proj);
        const xp = try add2(proj, dir.bias, s);
        defer _ = mlx.mlx_array_free(xp);

        const w_hh_t = try transpose2(dir.w_hh, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(w_hh_t);

        var h = try zerosF32(&[_]c_int{ 1, @intCast(h_dim) }, s);
        defer _ = mlx.mlx_array_free(h);
        var c = try zerosF32(&[_]c_int{ 1, @intCast(h_dim) }, s);
        defer _ = mlx.mlx_array_free(c);

        const steps = try allocator.alloc(mlx.mlx_array, t_len);
        defer allocator.free(steps);

        for (0..t_len) |step| {
            const t = if (reverse) t_len - 1 - step else step;

            const xt_3 = try sliceAxis(xp, 1, @intCast(t), @intCast(t + 1), s); // [1,1,4H]
            defer _ = mlx.mlx_array_free(xt_3);
            const xt = try reshape2(xt_3, &[_]c_int{ 1, @intCast(4 * h_dim) }, s);
            defer _ = mlx.mlx_array_free(xt);

            const rec = try mm(h, w_hh_t, s); // [1, 4H]
            defer _ = mlx.mlx_array_free(rec);
            const gates = try add2(xt, rec, s);
            defer _ = mlx.mlx_array_free(gates);

            // PyTorch gate order: i, f, g, o.
            const hi: c_int = @intCast(h_dim);
            const g_i = try sliceAxis(gates, 1, 0, hi, s);
            defer _ = mlx.mlx_array_free(g_i);
            const g_f = try sliceAxis(gates, 1, hi, 2 * hi, s);
            defer _ = mlx.mlx_array_free(g_f);
            const g_g = try sliceAxis(gates, 1, 2 * hi, 3 * hi, s);
            defer _ = mlx.mlx_array_free(g_g);
            const g_o = try sliceAxis(gates, 1, 3 * hi, 4 * hi, s);
            defer _ = mlx.mlx_array_free(g_o);

            const i_t = try sigmoid1(g_i, s);
            defer _ = mlx.mlx_array_free(i_t);
            const f_t = try sigmoid1(g_f, s);
            defer _ = mlx.mlx_array_free(f_t);
            const g_t = try tanh1(g_g, s);
            defer _ = mlx.mlx_array_free(g_t);
            const o_t = try sigmoid1(g_o, s);
            defer _ = mlx.mlx_array_free(o_t);

            const fc = try mul2(f_t, c, s);
            defer _ = mlx.mlx_array_free(fc);
            const ig = try mul2(i_t, g_t, s);
            defer _ = mlx.mlx_array_free(ig);
            const c_new = try add2(fc, ig, s);
            _ = mlx.mlx_array_free(c);
            c = c_new;

            const ct = try tanh1(c, s);
            defer _ = mlx.mlx_array_free(ct);
            const h_new = try mul2(o_t, ct, s);
            _ = mlx.mlx_array_free(h);
            h = h_new;

            // Store at the ORIGINAL time index so the reverse pass comes back
            // in forward order — concatenating a reversed sequence is the
            // classic silent bug here.
            steps[t] = try reshape2(h, &[_]c_int{ 1, 1, @intCast(h_dim) }, s);
        }

        defer {
            for (steps) |a| _ = mlx.mlx_array_free(a);
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (steps) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
        return out;
    }
};

fn zerosF32(shape: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&out, shape.ptr, shape.len, .float32, s));
    return out;
}

fn reshape2(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// STFT / iSTFT head (n_fft = 20, hop = 5, periodic Hann)
//
// The generator needs a FORWARD transform (on the harmonic source signal) and
// an INVERSE one (final waveform). Both are tiny — 20-point — so this is
// framing + rfft/irfft, not a heavyweight kernel.
// ════════════════════════════════════════════════════════════════════════

/// Frame a `[1, N]` signal into `[1, frames, n_fft]` with centre padding
/// (reflect), matching `torch.stft(..., center=True)`.
pub fn frameSignal(allocator: std.mem.Allocator, samples: []const f32, n_fft: usize, hop: usize) !struct { data: []f32, frames: usize } {
    const pad = n_fft / 2;
    const padded_len = samples.len + 2 * pad;
    const padded = try allocator.alloc(f32, padded_len);
    defer allocator.free(padded);

    // Reflect padding: torch reflects WITHOUT repeating the edge sample.
    for (0..pad) |i| padded[i] = samples[@min(pad - i, samples.len - 1)];
    @memcpy(padded[pad .. pad + samples.len], samples);
    for (0..pad) |i| {
        const src = samples.len - 2 - i;
        padded[pad + samples.len + i] = samples[if (src < samples.len) src else 0];
    }

    const frames = if (padded_len < n_fft) 0 else (padded_len - n_fft) / hop + 1;
    const data = try allocator.alloc(f32, frames * n_fft);
    errdefer allocator.free(data);
    for (0..frames) |f| {
        @memcpy(data[f * n_fft ..][0..n_fft], padded[f * hop ..][0..n_fft]);
    }
    return .{ .data = data, .frames = frames };
}

/// Overlap-add reconstruction with window-squared normalisation — the inverse
/// of `frameSignal` under the same window, i.e. `torch.istft(center=True)`.
pub fn overlapAdd(allocator: std.mem.Allocator, frames_data: []const f32, frames: usize, n_fft: usize, hop: usize, window: []const f32) ![]f32 {
    const padded_len = (frames - 1) * hop + n_fft;
    const acc = try allocator.alloc(f32, padded_len);
    defer allocator.free(acc);
    const wsum = try allocator.alloc(f32, padded_len);
    defer allocator.free(wsum);
    @memset(acc, 0);
    @memset(wsum, 0);

    for (0..frames) |f| {
        const off = f * hop;
        for (0..n_fft) |k| {
            acc[off + k] += frames_data[f * n_fft + k] * window[k];
            wsum[off + k] += window[k] * window[k];
        }
    }

    const pad = n_fft / 2;
    const out_len = padded_len - 2 * pad;
    const out = try allocator.alloc(f32, out_len);
    errdefer allocator.free(out);
    for (0..out_len) |i| {
        const w = wsum[pad + i];
        // torch guards the same way; below this the frame coverage is degenerate.
        out[i] = if (w > 1e-11) acc[pad + i] / w else 0;
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Elementwise / shape helpers
//
// All activations are NLC `[B, T, C]`. The reference is channel-first
// `[B, C, T]`; every transpose in it that only exists to satisfy that layout is
// simply ABSENT here. Where a reference transpose pair cancels (AdaLayerNorm
// does `transpose(-1,-2)` then `transpose(1,-1)`, a net identity on rank 3),
// this port does nothing rather than reproducing the round trip.
// ════════════════════════════════════════════════════════════════════════

fn scalar(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}

/// A second OWNING handle on the same array. Lets a branch that sometimes
/// derives a new array and sometimes passes the input through have one
/// uniform `free` on the way out, instead of a conditional-ownership flag —
/// the shape that leaks (see the sentinel-by-content rule in CLAUDE.md).
fn retain(x: mlx.mlx_array) mlx.mlx_array {
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&out, x);
    return out;
}

fn mulS(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = scalar(v);
    defer _ = mlx.mlx_array_free(c);
    return mul2(x, c, s);
}

fn addS(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = scalar(v);
    defer _ = mlx.mlx_array_free(c);
    return add2(x, c, s);
}

fn sub2(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&out, a, b, s));
    return out;
}

fn div2(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&out, a, b, s));
    return out;
}

fn sin1(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sin(&out, a, s));
    return out;
}

fn exp1(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_exp(&out, a, s));
    return out;
}

fn square1(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_square(&out, a, s));
    return out;
}

fn rsqrt1(a: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_rsqrt(&out, a, s));
    return out;
}

fn takeAxis(x: mlx.mlx_array, idx: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&out, x, idx, axis, s));
    return out;
}

fn concat2(a: mlx.mlx_array, b: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, a);
    _ = mlx.mlx_vector_array_append_value(vec, b);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, s));
    return out;
}

/// `nn.LeakyReLU(slope)` = max(x, slope·x). Kokoro uses THREE different slopes
/// and they are not interchangeable: 0.2 inside AdainResBlk1d, 0.1 in the
/// generator's upsample loop, and torch's 0.01 default at the final activation
/// before `conv_post`.
fn leakyRelu(x: mlx.mlx_array, slope: f32, s: S) !mlx.mlx_array {
    const scaled = try mulS(x, slope, s);
    defer _ = mlx.mlx_array_free(scaled);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_maximum(&out, x, scaled, s));
    return out;
}

/// `gelu_new` (the tanh approximation) — AlbertConfig's default `hidden_act`.
/// NOT the erf-exact gelu; they differ enough to matter over 12 layers.
fn geluNew(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const x3 = blk: {
        const sq = try square1(x, s);
        defer _ = mlx.mlx_array_free(sq);
        break :blk try mul2(sq, x, s);
    };
    defer _ = mlx.mlx_array_free(x3);
    const inner = blk: {
        const t = try mulS(x3, 0.044715, s);
        defer _ = mlx.mlx_array_free(t);
        const sum = try add2(x, t, s);
        defer _ = mlx.mlx_array_free(sum);
        break :blk try mulS(sum, std.math.sqrt(2.0 / std.math.pi), s);
    };
    defer _ = mlx.mlx_array_free(inner);
    const th = try tanh1(inner, s);
    defer _ = mlx.mlx_array_free(th);
    const one_plus = try addS(th, 1.0, s);
    defer _ = mlx.mlx_array_free(one_plus);
    const prod = try mul2(x, one_plus, s);
    defer _ = mlx.mlx_array_free(prod);
    return mulS(prod, 0.5, s);
}

/// Linear layer over NLC input: `x @ w^T + b`, `w` in PyTorch `[out, in]`.
fn linear(x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    const wt = try transpose2(w, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(wt);
    const y = try mm(x, wt, s);
    if (b) |bias| {
        defer _ = mlx.mlx_array_free(y);
        return add2(y, bias, s);
    }
    return y;
}

/// LayerNorm over the LAST axis with optional affine.
fn layerNorm(x: mlx.mlx_array, w: ?mlx.mlx_array, b: ?mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    const wv = w orelse mlx.mlx_array_new();
    const bv = b orelse mlx.mlx_array_new();
    defer {
        if (w == null) _ = mlx.mlx_array_free(wv);
        if (b == null) _ = mlx.mlx_array_free(bv);
    }
    try mlx.check(mlx.mlx_fast_layer_norm(&out, x, wv, bv, eps, s));
    return out;
}

/// Normalize over the TIME axis per channel — `nn.InstanceNorm1d` with the
/// affine turned off. The checkpoint declares `affine=True` but ships no
/// `norm.weight`/`norm.bias`: `KModel` loads with `strict=False`, so those stay
/// at their 1/0 init. There is nothing to apply, and inventing something here
/// silently changes the voice.
fn instanceNorm(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, x, 1, true, s));
    var variance = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(variance);
    try mlx.check(mlx.mlx_var_axis(&variance, x, 1, true, 0, s));

    const centered = try sub2(x, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const eps_var = try addS(variance, 1e-5, s);
    defer _ = mlx.mlx_array_free(eps_var);
    const inv = try rsqrt1(eps_var, s);
    defer _ = mlx.mlx_array_free(inv);
    return mul2(centered, inv, s);
}

/// Split `fc(style)` into the (gamma, beta) pair both adaptive norms use.
/// `fc` maps style_dim → 2·C; the reference chunks on the CHANNEL axis, which
/// in NLC means the two halves of the last axis.
fn styleGammaBeta(style: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !struct { gamma: mlx.mlx_array, beta: mlx.mlx_array } {
    const h = try linear(style, w, b, s); // [1, 2C]
    defer _ = mlx.mlx_array_free(h);
    const two_c = mlx.getShape(h)[1];
    const c = @divExact(two_c, 2);

    const g_flat = try sliceAxis(h, 1, 0, c, s);
    defer _ = mlx.mlx_array_free(g_flat);
    const b_flat = try sliceAxis(h, 1, c, two_c, s);
    defer _ = mlx.mlx_array_free(b_flat);
    // → [1, 1, C] so it broadcasts across time.
    return .{
        .gamma = try reshape2(g_flat, &[_]c_int{ 1, 1, c }, s),
        .beta = try reshape2(b_flat, &[_]c_int{ 1, 1, c }, s),
    };
}

/// `(1 + gamma) · normalized + beta`, shared by AdaLayerNorm and AdaIN1d.
fn applyStyle(norm: mlx.mlx_array, gamma: mlx.mlx_array, beta: mlx.mlx_array, s: S) !mlx.mlx_array {
    const gp1 = try addS(gamma, 1.0, s);
    defer _ = mlx.mlx_array_free(gp1);
    const scaled = try mul2(gp1, norm, s);
    defer _ = mlx.mlx_array_free(scaled);
    return add2(scaled, beta, s);
}

// ════════════════════════════════════════════════════════════════════════
// Weight access
// ════════════════════════════════════════════════════════════════════════

// `loadComponent` (safetensors → map) lives in ltx_video; the NLC conv
// primitives live in ltx_audio. Both are reused rather than duplicated.
const ltx = @import("ltx_video.zig");
const ltxa = @import("ltx_audio.zig");

/// A missing key is a BUG (bad conversion, wrong checkpoint), never something
/// to paper over with a default — an absent conv silently produces quiet,
/// wrong audio. Errors name the key so the converter is the obvious suspect.
fn need(comp: *const ltx.Component, key: []const u8) !mlx.mlx_array {
    return comp.get(key) orelse {
        log.err("[kokoro] missing weight: {s}\n", .{key});
        return error.MissingKokoroWeight;
    };
}

/// `need` with a formatted key. Callers pass a scratch buffer so this stays
/// allocation-free on the hot path.
fn needf(comp: *const ltx.Component, buf: []u8, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const key = try std.fmt.bufPrint(buf, fmt, args);
    return need(comp, key);
}

fn hasf(comp: *const ltx.Component, buf: []u8, comptime fmt: []const u8, args: anytype) bool {
    const key = std.fmt.bufPrint(buf, fmt, args) catch return false;
    return comp.get(key) != null;
}

/// Load one `nn.LSTM` direction from the PyTorch parameter quartet. The two
/// biases are summed here: torch applies both and both are constants, so
/// carrying them separately is per-step work for no effect.
fn loadLstmDir(comp: *const ltx.Component, buf: []u8, prefix: []const u8, suffix: []const u8, s: S) !LstmDir {
    const b_ih = try needf(comp, buf, "{s}.bias_ih_l0{s}", .{ prefix, suffix });
    const b_hh = try needf(comp, buf, "{s}.bias_hh_l0{s}", .{ prefix, suffix });
    return .{
        .w_ih = try needf(comp, buf, "{s}.weight_ih_l0{s}", .{ prefix, suffix }),
        .w_hh = try needf(comp, buf, "{s}.weight_hh_l0{s}", .{ prefix, suffix }),
        .bias = try add2(b_ih, b_hh, s),
    };
}

fn loadBiLstm(comp: *const ltx.Component, buf: []u8, prefix: []const u8, s: S) !BiLstm {
    const fwd = try loadLstmDir(comp, buf, prefix, "", s);
    const rev = try loadLstmDir(comp, buf, prefix, "_reverse", s);
    // hidden = 4H rows / 4.
    const hidden: usize = @intCast(@divExact(mlx.getShape(fwd.w_hh)[0], 4));
    return .{ .fwd = fwd, .rev = rev, .hidden = hidden };
}

// ════════════════════════════════════════════════════════════════════════
// ALBERT (PLBERT)
//
// ALBERT shares ONE layer's weights across all 12 layers — the checkpoint has
// a single `albert_layer_groups.0.albert_layers.0.*` and we run it 12 times.
// Embeddings are 128-wide and projected up to 768 by
// `encoder.embedding_hidden_mapping_in`; that factorization IS the architecture,
// not a quirk to normalize away.
//
// The attention mask is all-ones (one utterance, no padding), so this port has
// no mask at all.
// ════════════════════════════════════════════════════════════════════════

const ALBERT_LAYER = "bert.encoder.albert_layer_groups.0.albert_layers.0";
const ALBERT_EPS: f32 = 1e-12; // AlbertConfig.layer_norm_eps

fn albertForward(
    comp: *const ltx.Component,
    cfg: Config,
    input_ids: mlx.mlx_array,
    s: S,
) !mlx.mlx_array {
    var buf: [256]u8 = undefined;
    const t_len = mlx.getShape(input_ids)[1];

    // ── embeddings: word + position + token_type(0) ──
    const word = try takeAxis(try need(comp, "bert.embeddings.word_embeddings.weight"), input_ids, 0, s);
    defer _ = mlx.mlx_array_free(word);

    const pos_all = try need(comp, "bert.embeddings.position_embeddings.weight");
    const pos = try sliceAxis(pos_all, 0, 0, t_len, s);
    defer _ = mlx.mlx_array_free(pos);

    const tt_all = try need(comp, "bert.embeddings.token_type_embeddings.weight");
    const tt = try sliceAxis(tt_all, 0, 0, 1, s);
    defer _ = mlx.mlx_array_free(tt);

    var emb = blk: {
        const wp = try add2(word, pos, s);
        defer _ = mlx.mlx_array_free(wp);
        break :blk try add2(wp, tt, s);
    };
    {
        const normed = try layerNorm(
            emb,
            try need(comp, "bert.embeddings.LayerNorm.weight"),
            try need(comp, "bert.embeddings.LayerNorm.bias"),
            ALBERT_EPS,
            s,
        );
        _ = mlx.mlx_array_free(emb);
        emb = normed;
    }
    defer _ = mlx.mlx_array_free(emb);

    // ── 128 → 768 ──
    var h = try linear(
        emb,
        try need(comp, "bert.encoder.embedding_hidden_mapping_in.weight"),
        try need(comp, "bert.encoder.embedding_hidden_mapping_in.bias"),
        s,
    );
    errdefer _ = mlx.mlx_array_free(h);

    const n_heads: c_int = @intCast(cfg.bert_heads);
    const hidden: c_int = @intCast(cfg.bert_hidden);
    const head_dim = @divExact(hidden, n_heads);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..cfg.bert_layers) |_| {
        // ── self-attention ──
        const q = try linear(h, try needf(comp, &buf, "{s}.attention.query.weight", .{ALBERT_LAYER}), try needf(comp, &buf, "{s}.attention.query.bias", .{ALBERT_LAYER}), s);
        defer _ = mlx.mlx_array_free(q);
        const k = try linear(h, try needf(comp, &buf, "{s}.attention.key.weight", .{ALBERT_LAYER}), try needf(comp, &buf, "{s}.attention.key.bias", .{ALBERT_LAYER}), s);
        defer _ = mlx.mlx_array_free(k);
        const v = try linear(h, try needf(comp, &buf, "{s}.attention.value.weight", .{ALBERT_LAYER}), try needf(comp, &buf, "{s}.attention.value.bias", .{ALBERT_LAYER}), s);
        defer _ = mlx.mlx_array_free(v);

        const ctx = try attention(q, k, v, t_len, n_heads, head_dim, scale, s);
        defer _ = mlx.mlx_array_free(ctx);

        const proj = try linear(ctx, try needf(comp, &buf, "{s}.attention.dense.weight", .{ALBERT_LAYER}), try needf(comp, &buf, "{s}.attention.dense.bias", .{ALBERT_LAYER}), s);
        defer _ = mlx.mlx_array_free(proj);

        // ALBERT norms the RESIDUAL SUM, not the projection alone.
        const attn_out = blk: {
            const sum = try add2(h, proj, s);
            defer _ = mlx.mlx_array_free(sum);
            break :blk try layerNorm(
                sum,
                try needf(comp, &buf, "{s}.attention.LayerNorm.weight", .{ALBERT_LAYER}),
                try needf(comp, &buf, "{s}.attention.LayerNorm.bias", .{ALBERT_LAYER}),
                ALBERT_EPS,
                s,
            );
        };
        defer _ = mlx.mlx_array_free(attn_out);

        // ── feed-forward ──
        const ff = blk: {
            const up = try linear(attn_out, try needf(comp, &buf, "{s}.ffn.weight", .{ALBERT_LAYER}), try needf(comp, &buf, "{s}.ffn.bias", .{ALBERT_LAYER}), s);
            defer _ = mlx.mlx_array_free(up);
            const act = try geluNew(up, s);
            defer _ = mlx.mlx_array_free(act);
            break :blk try linear(act, try needf(comp, &buf, "{s}.ffn_output.weight", .{ALBERT_LAYER}), try needf(comp, &buf, "{s}.ffn_output.bias", .{ALBERT_LAYER}), s);
        };
        defer _ = mlx.mlx_array_free(ff);

        const next = blk: {
            const sum = try add2(ff, attn_out, s);
            defer _ = mlx.mlx_array_free(sum);
            break :blk try layerNorm(
                sum,
                try needf(comp, &buf, "{s}.full_layer_layer_norm.weight", .{ALBERT_LAYER}),
                try needf(comp, &buf, "{s}.full_layer_layer_norm.bias", .{ALBERT_LAYER}),
                ALBERT_EPS,
                s,
            );
        };
        _ = mlx.mlx_array_free(h);
        h = next;
    }
    return h;
}

/// Plain multi-head attention, no mask. `q`/`k`/`v` are `[1, T, H·D]`.
fn attention(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, t_len: c_int, heads: c_int, head_dim: c_int, scale: f32, s: S) !mlx.mlx_array {
    const shape = [_]c_int{ 1, t_len, heads, head_dim };
    const perm = [_]c_int{ 0, 2, 1, 3 };

    const qh = try reshapeTranspose(q, &shape, &perm, s);
    defer _ = mlx.mlx_array_free(qh);
    const kh = try reshapeTranspose(k, &shape, &perm, s);
    defer _ = mlx.mlx_array_free(kh);
    const vh = try reshapeTranspose(v, &shape, &perm, s);
    defer _ = mlx.mlx_array_free(vh);

    const kt = try transpose2(kh, &[_]c_int{ 0, 1, 3, 2 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const raw = try mm(qh, kt, s);
    defer _ = mlx.mlx_array_free(raw);
    const scaled = try mulS(raw, scale, s);
    defer _ = mlx.mlx_array_free(scaled);

    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, scaled, 3, true, s));

    const ctx = try mm(probs, vh, s);
    defer _ = mlx.mlx_array_free(ctx);
    const back = try transpose2(ctx, &perm, s);
    defer _ = mlx.mlx_array_free(back);
    return reshape2(back, &[_]c_int{ 1, t_len, heads * head_dim }, s);
}

fn reshapeTranspose(x: mlx.mlx_array, shape: []const c_int, perm: []const c_int, s: S) !mlx.mlx_array {
    const r = try reshape2(x, shape, s);
    defer _ = mlx.mlx_array_free(r);
    return transpose2(r, perm, s);
}

// ════════════════════════════════════════════════════════════════════════
// AdaIN residual blocks
// ════════════════════════════════════════════════════════════════════════

/// Snake1D's dilation schedule. Uniform `[[1,3,5],[1,3,5],[1,3,5]]` in every
/// published config; kernel sizes are read off the weights instead of being
/// hardcoded, so only this stays a constant.
const RESBLOCK_DILATIONS = [3]c_int{ 1, 3, 5 };

/// `get_padding(k, d)` from the reference: keeps the length unchanged.
fn convPadding(k: c_int, d: c_int) c_int {
    return @divTrunc(k * d - d, 2);
}

/// DEPTHWISE transposed conv (`groups == C`). MLX wants `[C_out, k, C_in/g]`,
/// so a depthwise PyTorch weight `[C, 1, k]` transposes {0,2,1} — NOT the {1,2,0}
/// that `ltx_audio.convTranspose1d` applies, which is only correct at groups=1.
/// Feeding a depthwise weight through the groups=1 helper is a shape error at
/// best and silently mixed channels at worst.
fn convTransposeDepthwise(x: mlx.mlx_array, w_pt: mlx.mlx_array, b: ?mlx.mlx_array, stride: c_int, padding: c_int, out_padding: c_int, s: S) !mlx.mlx_array {
    const groups = mlx.getShape(w_pt)[0];
    const w = try transpose2(w_pt, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(w);
    return ltxa.convTranspose1dMlx(x, w, b, stride, padding, out_padding, groups, s);
}

/// AdaIN1d: instance-normalize over time, then style-modulate.
fn adaIn(comp: *const ltx.Component, buf: []u8, prefix: []const u8, x: mlx.mlx_array, style: mlx.mlx_array, s: S) !mlx.mlx_array {
    const gb = try styleGammaBeta(
        style,
        try needf(comp, buf, "{s}.fc.weight", .{prefix}),
        try needf(comp, buf, "{s}.fc.bias", .{prefix}),
        s,
    );
    defer _ = mlx.mlx_array_free(gb.gamma);
    defer _ = mlx.mlx_array_free(gb.beta);
    const norm = try instanceNorm(x, s);
    defer _ = mlx.mlx_array_free(norm);
    return applyStyle(norm, gb.gamma, gb.beta, s);
}

/// `AdainResBlk1d` — the block used by ProsodyPredictor's F0/N stacks and the
/// Decoder. Shape facts are PROBED from the checkpoint rather than passed in:
/// `pool` present ⇒ this block upsamples ×2, `conv1x1` present ⇒ learned
/// shortcut. That is the `hy3ExpertContainer` convention and it means a block
/// whose geometry changes upstream cannot silently take the wrong branch.
fn adainResBlk1d(comp: *const ltx.Component, buf: []u8, prefix: []const u8, x: mlx.mlx_array, style: mlx.mlx_array, s: S) !mlx.mlx_array {
    var kb: [256]u8 = undefined;
    const upsamples = hasf(comp, &kb, "{s}.pool.weight", .{prefix});
    const learned_sc = hasf(comp, &kb, "{s}.conv1x1.weight", .{prefix});

    // ── residual ──
    var r = blk: {
        var pfx: [256]u8 = undefined;
        const p = try std.fmt.bufPrint(&pfx, "{s}.norm1", .{prefix});
        break :blk try adaIn(comp, buf, p, x, style, s);
    };
    {
        const act = try leakyRelu(r, 0.2, s);
        _ = mlx.mlx_array_free(r);
        r = act;
    }
    if (upsamples) {
        const pooled = try convTransposeDepthwise(
            r,
            try needf(comp, buf, "{s}.pool.weight", .{prefix}),
            try needf(comp, buf, "{s}.pool.bias", .{prefix}),
            2,
            1,
            1,
            s,
        );
        _ = mlx.mlx_array_free(r);
        r = pooled;
    }
    {
        const c1 = try ltxa.conv1d(
            r,
            try needf(comp, buf, "{s}.conv1.weight", .{prefix}),
            try needf(comp, buf, "{s}.conv1.bias", .{prefix}),
            1,
            1,
            1,
            1,
            s,
        );
        _ = mlx.mlx_array_free(r);
        r = c1;
    }
    {
        var pfx: [256]u8 = undefined;
        const p = try std.fmt.bufPrint(&pfx, "{s}.norm2", .{prefix});
        const n2 = try adaIn(comp, buf, p, r, style, s);
        _ = mlx.mlx_array_free(r);
        r = n2;
    }
    {
        const act = try leakyRelu(r, 0.2, s);
        _ = mlx.mlx_array_free(r);
        r = act;
    }
    {
        const c2 = try ltxa.conv1d(
            r,
            try needf(comp, buf, "{s}.conv2.weight", .{prefix}),
            try needf(comp, buf, "{s}.conv2.bias", .{prefix}),
            1,
            1,
            1,
            1,
            s,
        );
        _ = mlx.mlx_array_free(r);
        r = c2;
    }
    defer _ = mlx.mlx_array_free(r);

    // ── shortcut ──
    var sc: mlx.mlx_array = undefined;
    if (upsamples) {
        // nearest ×2 — repeat each timestep, which is exactly F.interpolate's
        // nearest mode at an integer scale.
        var up = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&up, x, 2, 1, s));
        sc = up;
    } else {
        sc = retain(x);
    }
    if (learned_sc) {
        // conv1x1 carries NO bias.
        const projected = try ltxa.conv1d(sc, try needf(comp, buf, "{s}.conv1x1.weight", .{prefix}), null, 1, 0, 1, 1, s);
        _ = mlx.mlx_array_free(sc);
        sc = projected;
    }
    defer _ = mlx.mlx_array_free(sc);

    const sum = try add2(r, sc, s);
    defer _ = mlx.mlx_array_free(sum);
    return mulS(sum, std.math.sqrt1_2, s);
}

/// `AdaINResBlock1` — the generator's residual block: three (AdaIN → Snake1D →
/// dilated conv → AdaIN → Snake1D → conv) stages, each added back to the input.
fn adainResBlock1(comp: *const ltx.Component, buf: []u8, prefix: []const u8, x_in: mlx.mlx_array, style: mlx.mlx_array, s: S) !mlx.mlx_array {
    var x = retain(x_in);
    errdefer _ = mlx.mlx_array_free(x);

    for (0..3) |j| {
        var pfx: [256]u8 = undefined;

        const w1 = try needf(comp, buf, "{s}.convs1.{d}.weight", .{ prefix, j });
        const k1 = mlx.getShape(w1)[2];
        const d1 = RESBLOCK_DILATIONS[j];

        var xt = blk: {
            const p = try std.fmt.bufPrint(&pfx, "{s}.adain1.{d}", .{ prefix, j });
            break :blk try adaIn(comp, buf, p, x, style, s);
        };
        {
            const sn = try snake1d(xt, try needf(comp, buf, "{s}.alpha1.{d}", .{ prefix, j }), s);
            _ = mlx.mlx_array_free(xt);
            xt = sn;
        }
        {
            const c = try ltxa.conv1d(xt, w1, try needf(comp, buf, "{s}.convs1.{d}.bias", .{ prefix, j }), 1, convPadding(k1, d1), d1, 1, s);
            _ = mlx.mlx_array_free(xt);
            xt = c;
        }
        {
            const p = try std.fmt.bufPrint(&pfx, "{s}.adain2.{d}", .{ prefix, j });
            const n = try adaIn(comp, buf, p, xt, style, s);
            _ = mlx.mlx_array_free(xt);
            xt = n;
        }
        {
            const sn = try snake1d(xt, try needf(comp, buf, "{s}.alpha2.{d}", .{ prefix, j }), s);
            _ = mlx.mlx_array_free(xt);
            xt = sn;
        }
        {
            const w2 = try needf(comp, buf, "{s}.convs2.{d}.weight", .{ prefix, j });
            const k2 = mlx.getShape(w2)[2];
            const c = try ltxa.conv1d(xt, w2, try needf(comp, buf, "{s}.convs2.{d}.bias", .{ prefix, j }), 1, convPadding(k2, 1), 1, 1, s);
            _ = mlx.mlx_array_free(xt);
            xt = c;
        }
        defer _ = mlx.mlx_array_free(xt);

        const sum = try add2(xt, x, s);
        _ = mlx.mlx_array_free(x);
        x = sum;
    }
    return x;
}

/// Snake1D: `x + (1/a)·sin²(a·x)`. `alpha` is stored `[1, C, 1]` (channel-first)
/// and reshaped to `[1, 1, C]` for NLC broadcasting.
fn snake1d(x: mlx.mlx_array, alpha_pt: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c = mlx.getShape(alpha_pt)[1];
    const a = try reshape2(alpha_pt, &[_]c_int{ 1, 1, c }, s);
    defer _ = mlx.mlx_array_free(a);

    const ax = try mul2(a, x, s);
    defer _ = mlx.mlx_array_free(ax);
    const sn = try sin1(ax, s);
    defer _ = mlx.mlx_array_free(sn);
    const sq = try square1(sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const scaled = try div2(sq, a, s);
    defer _ = mlx.mlx_array_free(scaled);
    return add2(x, scaled, s);
}

// ════════════════════════════════════════════════════════════════════════
// TextEncoder / DurationEncoder / ProsodyPredictor
// ════════════════════════════════════════════════════════════════════════

/// `TextEncoder`: embedding → 3×(conv, LayerNorm, LeakyReLU) → BiLSTM.
/// The custom LayerNorm in `modules.py` stores `.gamma`/`.beta`, NOT
/// `.weight`/`.bias`, and normalizes over CHANNELS.
fn textEncoderForward(comp: *const ltx.Component, allocator: std.mem.Allocator, cfg: Config, input_ids: mlx.mlx_array, s: S) !mlx.mlx_array {
    var buf: [256]u8 = undefined;

    var x = try takeAxis(try need(comp, "text_encoder.embedding.weight"), input_ids, 0, s);
    errdefer _ = mlx.mlx_array_free(x);

    const k: c_int = @intCast(cfg.text_encoder_kernel_size);
    const pad = @divTrunc(k - 1, 2);
    for (0..cfg.n_layer) |i| {
        {
            const c = try ltxa.conv1d(
                x,
                try needf(comp, &buf, "text_encoder.cnn.{d}.0.weight", .{i}),
                try needf(comp, &buf, "text_encoder.cnn.{d}.0.bias", .{i}),
                1,
                pad,
                1,
                1,
                s,
            );
            _ = mlx.mlx_array_free(x);
            x = c;
        }
        {
            const n = try layerNorm(
                x,
                try needf(comp, &buf, "text_encoder.cnn.{d}.1.gamma", .{i}),
                try needf(comp, &buf, "text_encoder.cnn.{d}.1.beta", .{i}),
                1e-5,
                s,
            );
            _ = mlx.mlx_array_free(x);
            x = n;
        }
        {
            const a = try leakyRelu(x, 0.2, s);
            _ = mlx.mlx_array_free(x);
            x = a;
        }
    }

    const lstm = try loadBiLstm(comp, &buf, "text_encoder.lstm", s);
    defer _ = mlx.mlx_array_free(lstm.fwd.bias);
    defer _ = mlx.mlx_array_free(lstm.rev.bias);
    const out = try lstm.forward(allocator, x, s);
    _ = mlx.mlx_array_free(x);
    return out;
}

/// `DurationEncoder`: alternating BiLSTM / AdaLayerNorm, re-concatenating the
/// style vector after every norm. `lstms.{0,2,4}` are the LSTMs and
/// `lstms.{1,3,5}` the norms — one flat ModuleList holding two kinds of thing.
fn durationEncoderForward(comp: *const ltx.Component, allocator: std.mem.Allocator, cfg: Config, d_en: mlx.mlx_array, style: mlx.mlx_array, s: S) !mlx.mlx_array {
    var buf: [256]u8 = undefined;
    const t_len = mlx.getShape(d_en)[1];

    // style [1, 128] → [1, T, 128] so it can ride alongside the features.
    const style_t = blk: {
        const c = mlx.getShape(style)[1];
        const r = try reshape2(style, &[_]c_int{ 1, 1, c }, s);
        defer _ = mlx.mlx_array_free(r);
        var rep = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&rep, r, t_len, 1, s));
        break :blk rep;
    };
    defer _ = mlx.mlx_array_free(style_t);

    var x = try concat2(d_en, style_t, 2, s);
    errdefer _ = mlx.mlx_array_free(x);

    for (0..cfg.n_layer) |i| {
        // ── BiLSTM at index 2i ──
        {
            var pfx: [64]u8 = undefined;
            const p = try std.fmt.bufPrint(&pfx, "predictor.text_encoder.lstms.{d}", .{2 * i});
            const lstm = try loadBiLstm(comp, &buf, p, s);
            defer _ = mlx.mlx_array_free(lstm.fwd.bias);
            defer _ = mlx.mlx_array_free(lstm.rev.bias);
            const y = try lstm.forward(allocator, x, s);
            _ = mlx.mlx_array_free(x);
            x = y;
        }
        // ── AdaLayerNorm at index 2i+1, then re-attach the style ──
        {
            const gb = try styleGammaBeta(
                style,
                try needf(comp, &buf, "predictor.text_encoder.lstms.{d}.fc.weight", .{2 * i + 1}),
                try needf(comp, &buf, "predictor.text_encoder.lstms.{d}.fc.bias", .{2 * i + 1}),
                s,
            );
            defer _ = mlx.mlx_array_free(gb.gamma);
            defer _ = mlx.mlx_array_free(gb.beta);
            // AdaLayerNorm normalizes with NO affine of its own.
            const normed = try layerNorm(x, null, null, 1e-5, s);
            defer _ = mlx.mlx_array_free(normed);
            const styled = try applyStyle(normed, gb.gamma, gb.beta, s);
            _ = mlx.mlx_array_free(x);
            defer _ = mlx.mlx_array_free(styled);
            x = try concat2(styled, style_t, 2, s);
        }
    }
    return x;
}

/// `F0Ntrain`: one shared BiLSTM feeding two independent 3-block stacks. Block
/// 1 of each stack upsamples ×2, so F0/N come out at twice the frame rate.
fn f0NtrainForward(
    comp: *const ltx.Component,
    allocator: std.mem.Allocator,
    en: mlx.mlx_array,
    style: mlx.mlx_array,
    s: S,
) !struct { f0: mlx.mlx_array, n: mlx.mlx_array } {
    var buf: [256]u8 = undefined;

    const shared = try loadBiLstm(comp, &buf, "predictor.shared", s);
    defer _ = mlx.mlx_array_free(shared.fwd.bias);
    defer _ = mlx.mlx_array_free(shared.rev.bias);
    const h = try shared.forward(allocator, en, s);
    defer _ = mlx.mlx_array_free(h);

    const f0 = try f0NBranch(comp, &buf, "predictor.F0", "predictor.F0_proj", h, style, s);
    errdefer _ = mlx.mlx_array_free(f0);
    const n = try f0NBranch(comp, &buf, "predictor.N", "predictor.N_proj", h, style, s);
    return .{ .f0 = f0, .n = n };
}

/// One F0 or N branch: 3 AdainResBlk1d then a 1×1 projection to a single
/// channel. Returns `[1, 2F]` — the channel axis is squeezed out.
fn f0NBranch(comp: *const ltx.Component, buf: []u8, stack: []const u8, proj: []const u8, h: mlx.mlx_array, style: mlx.mlx_array, s: S) !mlx.mlx_array {
    var x = retain(h);
    errdefer _ = mlx.mlx_array_free(x);
    for (0..3) |i| {
        var pfx: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&pfx, "{s}.{d}", .{ stack, i });
        const y = try adainResBlk1d(comp, buf, p, x, style, s);
        _ = mlx.mlx_array_free(x);
        x = y;
    }
    defer _ = mlx.mlx_array_free(x);

    const projected = try ltxa.conv1d(
        x,
        try needf(comp, buf, "{s}.weight", .{proj}),
        try needf(comp, buf, "{s}.bias", .{proj}),
        1,
        0,
        1,
        1,
        s,
    );
    defer _ = mlx.mlx_array_free(projected);
    // [1, L, 1] → [1, L]
    const len = mlx.getShape(projected)[1];
    return reshape2(projected, &[_]c_int{ 1, len }, s);
}

// ════════════════════════════════════════════════════════════════════════
// iSTFTNet generator
// ════════════════════════════════════════════════════════════════════════

const HARMONICS: c_int = 9; // harmonic_num 8 + fundamental
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;

/// Linear-interpolating upsample along time by an integer factor, matching
/// `F.interpolate(mode='linear', align_corners=False)`: output i samples input
/// position `(i + 0.5)/scale − 0.5`, clamped at the ends.
fn linearUpsample(allocator: std.mem.Allocator, x: mlx.mlx_array, scale: usize, s: S) !mlx.mlx_array {
    const n: usize = @intCast(mlx.getShape(x)[1]);
    const out_len = n * scale;

    const lo = try allocator.alloc(i32, out_len);
    defer allocator.free(lo);
    const hi = try allocator.alloc(i32, out_len);
    defer allocator.free(hi);
    const wt = try allocator.alloc(f32, out_len);
    defer allocator.free(wt);

    for (0..out_len) |i| {
        const fi: f64 = @floatFromInt(i);
        const sc: f64 = @floatFromInt(scale);
        var p = (fi + 0.5) / sc - 0.5;
        if (p < 0) p = 0;
        const max_p: f64 = @floatFromInt(n - 1);
        if (p > max_p) p = max_p;
        const l: usize = @intFromFloat(@floor(p));
        const h = @min(l + 1, n - 1);
        lo[i] = @intCast(l);
        hi[i] = @intCast(h);
        wt[i] = @floatCast(p - @floor(p));
    }

    const lo_a = mlx.mlx_array_new_data(lo.ptr, &[_]c_int{@intCast(out_len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_data(hi.ptr, &[_]c_int{@intCast(out_len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(hi_a);
    const w_a = mlx.mlx_array_new_data(wt.ptr, &[_]c_int{ 1, @intCast(out_len), 1 }, 3, .float32);
    defer _ = mlx.mlx_array_free(w_a);

    const a = try takeAxis(x, lo_a, 1, s);
    defer _ = mlx.mlx_array_free(a);
    const b = try takeAxis(x, hi_a, 1, s);
    defer _ = mlx.mlx_array_free(b);

    const diff = try sub2(b, a, s);
    defer _ = mlx.mlx_array_free(diff);
    const scaled = try mul2(diff, w_a, s);
    defer _ = mlx.mlx_array_free(scaled);
    return add2(a, scaled, s);
}

/// SineGen + SourceModuleHnNSF, collapsed to the frame rate where the
/// reference works at the sample rate.
///
/// The reference upsamples F0 to the sample rate with NEAREST, then
/// linear-DOWNsamples the phase by exactly that factor before the cumsum. Over
/// a piecewise-constant signal that round trip is the identity, so we cumsum
/// straight from the per-frame values: mathematically the same, and it skips
/// building two sample-rate tensors.
///
/// STOCHASTIC: random initial phase per harmonic plus additive noise, so the
/// caller must pass a seed if it wants a reproducible waveform.
fn harmonicSource(
    allocator: std.mem.Allocator,
    comp: *const ltx.Component,
    f0_frame: mlx.mlx_array, // [1, 2F]
    upsample: usize, // 300
    sample_rate: f32,
    seed: u64,
    s: S,
) !mlx.mlx_array {
    var buf: [256]u8 = undefined;
    const n_frames = mlx.getShape(f0_frame)[1];

    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));

    // f0 per harmonic: [1, 2F, 1] × [1..9]
    const f0_c = try reshape2(f0_frame, &[_]c_int{ 1, n_frames, 1 }, s);
    defer _ = mlx.mlx_array_free(f0_c);

    var mult_data: [9]f32 = undefined;
    for (0..9) |i| mult_data[i] = @floatFromInt(i + 1);
    const mult = mlx.mlx_array_new_data(&mult_data, &[_]c_int{ 1, 1, HARMONICS }, 3, .float32);
    defer _ = mlx.mlx_array_free(mult);

    const fn_ = try mul2(f0_c, mult, s);
    defer _ = mlx.mlx_array_free(fn_);

    // rad = (f/sr) mod 1
    const rad = blk: {
        const div = try mulS(fn_, 1.0 / sample_rate, s);
        defer _ = mlx.mlx_array_free(div);
        const fl = try floorArr(div, s);
        defer _ = mlx.mlx_array_free(fl);
        break :blk try sub2(div, fl, s);
    };
    defer _ = mlx.mlx_array_free(rad);

    // NO random initial phase — and that is not an omission.
    //
    // The reference adds `rand_ini` to rad_values at SAMPLE 0, then
    // linear-downsamples by 1/300 with align_corners=False, which samples
    // position (j+0.5)·300 − 0.5 = 149.5 for the first output. That is the
    // MIDDLE of block 0, so the offset at sample 0 is never read and the
    // downsampled result is bit-identical with and without it (verified: max
    // abs diff 0.0). Because this port collapses that round trip and works at
    // the frame rate directly, adding rand_ini HERE would apply a random phase
    // offset to harmonics 2..9 that the reference silently throws away —
    // measured as a drop from ~0.99 to 0.48 waveform cosine against the
    // reference. Reproducing an upstream no-op faithfully means not doing it.

    // phase = cumsum(rad) · 2π, then linear-upsample by `upsample` with the
    // reference's pre-multiplication by the same factor.
    var phase = blk: {
        var cs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_cumsum(&cs, rad, 1, false, true, s));
        defer _ = mlx.mlx_array_free(cs);
        break :blk try mulS(cs, 2.0 * std.math.pi * @as(f32, @floatFromInt(upsample)), s);
    };
    {
        const up = try linearUpsample(allocator, phase, upsample, s);
        _ = mlx.mlx_array_free(phase);
        phase = up;
    }
    defer _ = mlx.mlx_array_free(phase);

    const sines = try sin1(phase, s);
    defer _ = mlx.mlx_array_free(sines);

    // Voicing gate at the SAMPLE rate — f0 is piecewise constant, so repeating
    // the per-frame decision is exact.
    const uv = blk: {
        const thr = scalar(VOICED_THRESHOLD);
        defer _ = mlx.mlx_array_free(thr);
        var gt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gt);
        try mlx.check(mlx.mlx_greater(&gt, f0_c, thr, s));
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, gt, .float32, s));
        var rep = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_repeat_axis(&rep, f, @intCast(upsample), 1, s));
        break :blk rep;
    };
    defer _ = mlx.mlx_array_free(uv);

    const total_len = mlx.getShape(sines)[1];
    var noise = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(noise);
    try mlx.check(mlx.mlx_random_normal(&noise, &[_]c_int{ 1, total_len, HARMONICS }, 3, .float32, 0.0, 1.0, key, s));

    // noise_amp = uv·0.003 + (1−uv)·(0.1/3)
    const noise_amp = blk: {
        const voiced = try mulS(uv, NOISE_STD, s);
        defer _ = mlx.mlx_array_free(voiced);
        const inv = blk2: {
            const neg = try mulS(uv, -1.0, s);
            defer _ = mlx.mlx_array_free(neg);
            break :blk2 try addS(neg, 1.0, s);
        };
        defer _ = mlx.mlx_array_free(inv);
        const unvoiced = try mulS(inv, SINE_AMP / 3.0, s);
        defer _ = mlx.mlx_array_free(unvoiced);
        break :blk try add2(voiced, unvoiced, s);
    };
    defer _ = mlx.mlx_array_free(noise_amp);

    // sine_waves = sines·0.1·uv + noise_amp·noise
    const waves = blk: {
        const amped = try mulS(sines, SINE_AMP, s);
        defer _ = mlx.mlx_array_free(amped);
        const gated = try mul2(amped, uv, s);
        defer _ = mlx.mlx_array_free(gated);
        const nz = try mul2(noise_amp, noise, s);
        defer _ = mlx.mlx_array_free(nz);
        break :blk try add2(gated, nz, s);
    };
    defer _ = mlx.mlx_array_free(waves);

    // Merge the harmonics: Linear(9 → 1) then tanh.
    const merged = try linear(
        waves,
        try needf(comp, &buf, "decoder.generator.m_source.l_linear.weight", .{}),
        try needf(comp, &buf, "decoder.generator.m_source.l_linear.bias", .{}),
        s,
    );
    defer _ = mlx.mlx_array_free(merged);
    const t = try tanh1(merged, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape2(t, &[_]c_int{ 1, total_len }, s);
}

fn floorArr(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_floor(&out, x, s));
    return out;
}

/// Forward STFT of a `[1, L]` real signal → magnitude and phase, each
/// `[1, frames, bins]`. Framing/windowing happen on the host (the signal is
/// already materialized), the rfft on the stream.
fn stftMagPhase(allocator: std.mem.Allocator, samples: []const f32, n_fft: usize, hop: usize, window: []const f32, s: S) !struct { mag: mlx.mlx_array, phase: mlx.mlx_array, frames: usize } {
    const framed = try frameSignal(allocator, samples, n_fft, hop);
    defer allocator.free(framed.data);
    for (0..framed.frames) |f| {
        for (0..n_fft) |k| framed.data[f * n_fft + k] *= window[k];
    }

    const x = mlx.mlx_array_new_data(framed.data.ptr, &[_]c_int{ 1, @intCast(framed.frames), @intCast(n_fft) }, 3, .float32);
    defer _ = mlx.mlx_array_free(x);

    var spec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(spec);
    try mlx.check(mlx.mlx_fft_rfft(&spec, x, @intCast(n_fft), 2, mlx.MLX_FFT_NORM_BACKWARD, s));

    var re = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(re);
    try mlx.check(mlx.mlx_real(&re, spec, s));
    var im = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(im);
    try mlx.check(mlx.mlx_imag(&im, spec, s));

    const mag = blk: {
        const r2 = try square1(re, s);
        defer _ = mlx.mlx_array_free(r2);
        const im2 = try square1(im, s);
        defer _ = mlx.mlx_array_free(im2);
        const sum = try add2(r2, im2, s);
        defer _ = mlx.mlx_array_free(sum);
        var sq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sqrt(&sq, sum, s));
        break :blk sq;
    };
    errdefer _ = mlx.mlx_array_free(mag);

    var phase = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_arctan2(&phase, im, re, s));

    return .{ .mag = mag, .phase = phase, .frames = framed.frames };
}

/// Inverse STFT from magnitude and phase `[1, frames, bins]` → `[]f32`.
/// The overlap-add runs on the host because it needs the window-squared
/// normalisation, which has no clean vectorised MLX form at this size.
fn istft(allocator: std.mem.Allocator, mag: mlx.mlx_array, phase: mlx.mlx_array, n_fft: usize, hop: usize, window: []const f32, s: S) ![]f32 {
    const cos_p = blk: {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_cos(&c, phase, s));
        break :blk c;
    };
    defer _ = mlx.mlx_array_free(cos_p);
    const sin_p = try sin1(phase, s);
    defer _ = mlx.mlx_array_free(sin_p);

    const re = try mul2(mag, cos_p, s);
    defer _ = mlx.mlx_array_free(re);
    const im = try mul2(mag, sin_p, s);
    defer _ = mlx.mlx_array_free(im);

    // No mlx-c op builds a complex array from two real ones, so multiply the
    // imaginary part by a complex unit and add.
    const imag_unit = mlx.mlx_array_new_complex(0.0, 1.0);
    defer _ = mlx.mlx_array_free(imag_unit);
    const im_c = try mul2(im, imag_unit, s);
    defer _ = mlx.mlx_array_free(im_c);
    const cplx = try add2(re, im_c, s);
    defer _ = mlx.mlx_array_free(cplx);

    var frames_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(frames_arr);
    try mlx.check(mlx.mlx_fft_irfft(&frames_arr, cplx, @intCast(n_fft), 2, mlx.MLX_FFT_NORM_BACKWARD, s));

    _ = mlx.mlx_array_eval(frames_arr);
    const shape = mlx.getShape(frames_arr);
    const n_frames: usize = @intCast(shape[1]);
    const data = mlx.mlx_array_data_float32(frames_arr) orelse return error.KokoroIstftReadFailed;

    return overlapAdd(allocator, data[0 .. n_frames * n_fft], n_frames, n_fft, hop, window);
}

/// The iSTFTNet generator: two upsample stages, each mixing in a harmonic
/// source branch and averaging three residual blocks, then an iSTFT head.
fn generatorForward(
    comp: *const ltx.Component,
    allocator: std.mem.Allocator,
    cfg: Config,
    x_in: mlx.mlx_array, // [1, 2F, 512]
    style: mlx.mlx_array,
    f0_curve: mlx.mlx_array, // [1, 2F]
    seed: u64,
    s: S,
) ![]f32 {
    var buf: [256]u8 = undefined;
    const n_fft: usize = cfg.gen_istft_n_fft;
    const hop: usize = cfg.gen_istft_hop_size;

    const window = try hannPeriodic(allocator, n_fft);
    defer allocator.free(window);

    // ── harmonic source → its own STFT, used as a per-stage conditioning signal ──
    const har_wave = try harmonicSource(
        allocator,
        comp,
        f0_curve,
        cfg.samplesPerFrame(),
        @floatFromInt(cfg.sample_rate),
        seed,
        s,
    );
    defer _ = mlx.mlx_array_free(har_wave);

    _ = mlx.mlx_array_eval(har_wave);
    const har_len: usize = @intCast(mlx.getShape(har_wave)[1]);
    const har_data = mlx.mlx_array_data_float32(har_wave) orelse return error.KokoroSourceReadFailed;

    const har = blk: {
        const mp = try stftMagPhase(allocator, har_data[0..har_len], n_fft, hop, window, s);
        defer _ = mlx.mlx_array_free(mp.mag);
        defer _ = mlx.mlx_array_free(mp.phase);
        // cat([magnitude, phase]) on the channel axis → [1, frames, 22]
        break :blk try concat2(mp.mag, mp.phase, 2, s);
    };
    defer _ = mlx.mlx_array_free(har);

    var x = retain(x_in);
    errdefer _ = mlx.mlx_array_free(x);

    const n_up = cfg.upsample_rates.len;
    for (0..n_up) |i| {
        {
            const a = try leakyRelu(x, 0.1, s);
            _ = mlx.mlx_array_free(x);
            x = a;
        }

        // Source branch for this stage.
        const w_nc = try needf(comp, &buf, "decoder.generator.noise_convs.{d}.weight", .{i});
        const k_nc = mlx.getShape(w_nc)[2];
        const stride_f0: c_int = if (i + 1 < n_up) @intCast(cfg.upsample_rates[i + 1]) else 1;
        const pad_nc: c_int = if (i + 1 < n_up) @divTrunc(stride_f0 + 1, 2) else 0;
        _ = k_nc;

        var x_source = try ltxa.conv1d(
            har,
            w_nc,
            try needf(comp, &buf, "decoder.generator.noise_convs.{d}.bias", .{i}),
            stride_f0,
            pad_nc,
            1,
            1,
            s,
        );
        {
            var pfx: [128]u8 = undefined;
            const p = try std.fmt.bufPrint(&pfx, "decoder.generator.noise_res.{d}", .{i});
            const y = try adainResBlock1(comp, &buf, p, x_source, style, s);
            _ = mlx.mlx_array_free(x_source);
            x_source = y;
        }
        defer _ = mlx.mlx_array_free(x_source);

        // Upsample the trunk. padding = (k − stride)/2 per the reference.
        {
            const w_up = try needf(comp, &buf, "decoder.generator.ups.{d}.weight", .{i});
            const k_up = mlx.getShape(w_up)[2];
            const stride: c_int = @intCast(cfg.upsample_rates[i]);
            const up = try ltxa.convTranspose1d(
                x,
                w_up,
                try needf(comp, &buf, "decoder.generator.ups.{d}.bias", .{i}),
                stride,
                @divTrunc(k_up - stride, 2),
                0,
                1,
                s,
            );
            _ = mlx.mlx_array_free(x);
            x = up;
        }

        // Only the LAST stage reflection-pads (1, 0) along time, which is what
        // lines the trunk up with the source branch's frame count.
        if (i == n_up - 1) {
            const padded = try reflectPadTimeFront(x, s);
            _ = mlx.mlx_array_free(x);
            x = padded;
        }

        {
            const sum = try add2(x, x_source, s);
            _ = mlx.mlx_array_free(x);
            x = sum;
        }

        // Average the three residual blocks for this stage.
        {
            var acc: ?mlx.mlx_array = null;
            for (0..3) |j| {
                var pfx: [128]u8 = undefined;
                const p = try std.fmt.bufPrint(&pfx, "decoder.generator.resblocks.{d}", .{i * 3 + j});
                const y = try adainResBlock1(comp, &buf, p, x, style, s);
                if (acc) |a| {
                    defer _ = mlx.mlx_array_free(y);
                    defer _ = mlx.mlx_array_free(a);
                    acc = try add2(a, y, s);
                } else {
                    acc = y;
                }
            }
            const avg = try mulS(acc.?, 1.0 / 3.0, s);
            _ = mlx.mlx_array_free(acc.?);
            _ = mlx.mlx_array_free(x);
            x = avg;
        }
    }

    // Final activation uses torch's DEFAULT slope (0.01), not the 0.1 above.
    {
        const a = try leakyRelu(x, 0.01, s);
        _ = mlx.mlx_array_free(x);
        x = a;
    }
    {
        const w_post = try need(comp, "decoder.generator.conv_post.weight");
        const k_post = mlx.getShape(w_post)[2];
        const post = try ltxa.conv1d(x, w_post, try need(comp, "decoder.generator.conv_post.bias"), 1, @divTrunc(k_post - 1, 2), 1, 1, s);
        _ = mlx.mlx_array_free(x);
        x = post;
    }
    defer _ = mlx.mlx_array_free(x);

    // Channels split into magnitude (exp) and phase (sin).
    const bins: c_int = @intCast(n_fft / 2 + 1);
    const mag = blk: {
        const sl = try sliceAxis(x, 2, 0, bins, s);
        defer _ = mlx.mlx_array_free(sl);
        break :blk try exp1(sl, s);
    };
    defer _ = mlx.mlx_array_free(mag);
    const phase = blk: {
        const sl = try sliceAxis(x, 2, bins, 2 * bins, s);
        defer _ = mlx.mlx_array_free(sl);
        break :blk try sin1(sl, s);
    };
    defer _ = mlx.mlx_array_free(phase);

    return istft(allocator, mag, phase, n_fft, hop, window, s);
}

/// `nn.ReflectionPad1d((1, 0))` on the TIME axis of an NLC tensor: prepend the
/// value at index 1 (reflection excludes the edge sample itself).
fn reflectPadTimeFront(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const t = mlx.getShape(x)[1];
    const src: c_int = if (t > 1) 1 else 0;
    const head = try sliceAxis(x, 1, src, src + 1, s);
    defer _ = mlx.mlx_array_free(head);
    return concat2(head, x, 1, s);
}

// ════════════════════════════════════════════════════════════════════════
// Decoder
// ════════════════════════════════════════════════════════════════════════

fn decoderForward(
    comp: *const ltx.Component,
    allocator: std.mem.Allocator,
    cfg: Config,
    asr: mlx.mlx_array, // [1, F, 512]
    f0_curve: mlx.mlx_array, // [1, 2F]
    n_curve: mlx.mlx_array, // [1, 2F]
    style: mlx.mlx_array, // [1, 128]
    seed: u64,
    s: S,
) ![]f32 {
    var buf: [256]u8 = undefined;

    // F0/N enter at 2× the frame rate and are halved by a stride-2 conv.
    const f0 = try curveConv(comp, &buf, "decoder.F0_conv", f0_curve, s);
    defer _ = mlx.mlx_array_free(f0);
    const n = try curveConv(comp, &buf, "decoder.N_conv", n_curve, s);
    defer _ = mlx.mlx_array_free(n);

    var x = blk: {
        const with_f0 = try concat2(asr, f0, 2, s);
        defer _ = mlx.mlx_array_free(with_f0);
        break :blk try concat2(with_f0, n, 2, s);
    };
    {
        const enc = try adainResBlk1d(comp, &buf, "decoder.encode", x, style, s);
        _ = mlx.mlx_array_free(x);
        x = enc;
    }
    errdefer _ = mlx.mlx_array_free(x);

    const asr_res = try ltxa.conv1d(
        asr,
        try need(comp, "decoder.asr_res.0.weight"),
        try need(comp, "decoder.asr_res.0.bias"),
        1,
        0,
        1,
        1,
        s,
    );
    defer _ = mlx.mlx_array_free(asr_res);

    // Every decode block re-attaches (asr_res, F0, N); the last one upsamples,
    // after which the residuals no longer line up and are dropped.
    var i: usize = 0;
    while (true) : (i += 1) {
        var pfx: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&pfx, "decoder.decode.{d}", .{i});
        if (!hasf(comp, &buf, "{s}.conv1.weight", .{p})) break;

        const upsamples = hasf(comp, &buf, "{s}.pool.weight", .{p});
        {
            const a = try concat2(x, asr_res, 2, s);
            defer _ = mlx.mlx_array_free(a);
            const b = try concat2(a, f0, 2, s);
            defer _ = mlx.mlx_array_free(b);
            const c = try concat2(b, n, 2, s);
            _ = mlx.mlx_array_free(x);
            x = c;
        }
        const y = try adainResBlk1d(comp, &buf, p, x, style, s);
        _ = mlx.mlx_array_free(x);
        x = y;
        if (upsamples) break;
    }
    defer _ = mlx.mlx_array_free(x);

    return generatorForward(comp, allocator, cfg, x, style, f0_curve, seed, s);
}

/// The stride-2 `F0_conv`/`N_conv`: `[1, 2F]` → `[1, F, 1]`.
fn curveConv(comp: *const ltx.Component, buf: []u8, prefix: []const u8, curve: mlx.mlx_array, s: S) !mlx.mlx_array {
    const len = mlx.getShape(curve)[1];
    const x = try reshape2(curve, &[_]c_int{ 1, len, 1 }, s);
    defer _ = mlx.mlx_array_free(x);
    return ltxa.conv1d(
        x,
        try needf(comp, buf, "{s}.weight", .{prefix}),
        try needf(comp, buf, "{s}.bias", .{prefix}),
        2,
        1,
        1,
        1,
        s,
    );
}

// ════════════════════════════════════════════════════════════════════════
// Model
// ════════════════════════════════════════════════════════════════════════

pub const DEFAULT_VOICE = "af_heart";

/// Evaluate and copy an array's f32 contents to host memory.
fn copyToHost(a: std.mem.Allocator, arr: mlx.mlx_array, s: S) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    _ = mlx.mlx_array_eval(f);
    var n: c_int = 1;
    for (mlx.getShape(arr)) |d| n *= d;
    const src = mlx.mlx_array_data_float32(f) orelse return error.KokoroReadFailed;
    return a.dupe(f32, src[0..@intCast(n)]);
}

fn readWholeFile(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(a, .limited(16 * 1024 * 1024));
}

pub const Model = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    vocab: Vocab,
    trunk: ltx.Component,
    voices: ltx.Component,
    stream: S,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !Model {
        var cfg_path_buf: [1024]u8 = undefined;
        const cfg_path = try std.fmt.bufPrint(&cfg_path_buf, "{s}/config.json", .{model_dir});
        const cfg_text = try readWholeFile(io, allocator, cfg_path);
        defer allocator.free(cfg_text);

        const cfg = try parseConfig(allocator, cfg_text);
        var vocab = try Vocab.parse(allocator, cfg_text);
        errdefer vocab.deinit();

        // safetensors READS run on the CPU stream — mlx's Load op has no GPU
        // implementation and dies with an uncatchable "[Load::eval_gpu] Not
        // implemented". Compute still uses the caller's (GPU) stream.
        const load_s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(load_s);

        const trunk_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{model_dir}, 0);
        defer allocator.free(trunk_path);
        var trunk = try ltx.loadComponent(allocator, trunk_path, load_s);
        errdefer trunk.deinit();

        const voices_path = try std.fmt.allocPrintSentinel(allocator, "{s}/voices.safetensors", .{model_dir}, 0);
        defer allocator.free(voices_path);
        var voices = try ltx.loadComponent(allocator, voices_path, load_s);
        errdefer voices.deinit();

        log.info("[kokoro] ready — {d} tensors, {d} voices, {d} Hz\n", .{ trunk.count(), voices.count(), cfg.sample_rate });
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .vocab = vocab,
            .trunk = trunk,
            .voices = voices,
            .stream = s,
        };
    }

    pub fn deinit(self: *Model) void {
        self.vocab.deinit();
        self.trunk.deinit();
        self.voices.deinit();
    }

    pub fn hasVoice(self: *const Model, name: []const u8) bool {
        return self.voices.get(name) != null;
    }

    /// Resolve a voice spec to the `[1, 256]` style row for an utterance of
    /// `n_phonemes` phonemes.
    ///
    /// A spec may name SEVERAL voices comma-separated (`"af_bella,af_sky"`),
    /// which the reference averages — that is the supported way to make a new
    /// voice. The row index is the phoneme count, NOT a fixed row: each pack is
    /// a `[510, 1, 256]` table and picking the wrong row degrades prosody
    /// silently.
    pub fn styleFor(self: *const Model, spec: []const u8, n_phonemes: usize) !mlx.mlx_array {
        var acc: ?mlx.mlx_array = null;
        errdefer {
            if (acc) |a| _ = mlx.mlx_array_free(a);
        }
        var count: usize = 0;

        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |raw| {
            const name = std.mem.trim(u8, raw, " \t");
            if (name.len == 0) continue;
            const pack = self.voices.get(name) orelse {
                log.err("[kokoro] unknown voice: {s}\n", .{name});
                return error.UnknownKokoroVoice;
            };

            const rows: usize = @intCast(mlx.getShape(pack)[0]);
            const idx: c_int = @intCast(@min(n_phonemes -| 1, rows - 1));
            const row = try sliceAxis(pack, 0, idx, idx + 1, self.stream); // [1,1,256]
            defer _ = mlx.mlx_array_free(row);
            const flat = try reshape2(row, &[_]c_int{ 1, 256 }, self.stream);

            if (acc) |a| {
                defer _ = mlx.mlx_array_free(a);
                defer _ = mlx.mlx_array_free(flat);
                acc = try add2(a, flat, self.stream);
            } else {
                acc = flat;
            }
            count += 1;
        }

        if (count == 0) return error.UnknownKokoroVoice;
        if (count == 1) return acc.?;
        defer _ = mlx.mlx_array_free(acc.?);
        return mulS(acc.?, 1.0 / @as(f32, @floatFromInt(count)), self.stream);
    }

    /// Intermediates captured for the parity oracles. Every field is optional
    /// and OWNED BY THE CALLER once filled. Nothing here is on the normal path:
    /// with a null trace `synthesize` allocates none of it.
    pub const Trace = struct {
        durations: ?[]u32 = null,
        f0: ?[]f32 = null,
        n: ?[]f32 = null,
        asr: ?[]f32 = null,

        pub fn deinit(self: *Trace, a: std.mem.Allocator) void {
            if (self.durations) |v| a.free(v);
            if (self.f0) |v| a.free(v);
            if (self.n) |v| a.free(v);
            if (self.asr) |v| a.free(v);
        }
    };

    /// Synthesize from an IPA PHONEME string (not plain text — the phonemizer
    /// lives in `kokoro_g2p.zig`). Returns f32 samples at `cfg.sample_rate`.
    pub fn synthesize(self: *Model, phonemes: []const u8, voice: []const u8, speed: f32, seed: u64) ![]f32 {
        return self.synthesizeTraced(phonemes, voice, speed, seed, null);
    }

    pub fn synthesizeTraced(self: *Model, phonemes: []const u8, voice: []const u8, speed: f32, seed: u64, trace: ?*Trace) ![]f32 {
        const a = self.allocator;
        const s = self.stream;

        const ids = try self.vocab.encode(a, phonemes);
        defer a.free(ids);
        // The two boundary zeros are not phonemes; the voice table is indexed by
        // the real phoneme count.
        if (ids.len <= 2) return error.EmptyKokoroInput;
        const n_phonemes = ids.len - 2;

        const max_ctx = self.cfg.bert_max_pos;
        if (ids.len > max_ctx) {
            log.err("[kokoro] {d} tokens exceeds the {d}-token context\n", .{ ids.len, max_ctx });
            return error.KokoroInputTooLong;
        }

        const input_ids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
        defer _ = mlx.mlx_array_free(input_ids);

        const ref_s = try self.styleFor(voice, n_phonemes);
        defer _ = mlx.mlx_array_free(ref_s);
        // dims [128:] drive prosody + duration, dims [:128] drive the decoder.
        const s_pred = try sliceAxis(ref_s, 1, 128, 256, s);
        defer _ = mlx.mlx_array_free(s_pred);
        const s_dec = try sliceAxis(ref_s, 1, 0, 128, s);
        defer _ = mlx.mlx_array_free(s_dec);

        // ── prosody trunk ──
        const bert_out = try albertForward(&self.trunk, self.cfg, input_ids, s);
        defer _ = mlx.mlx_array_free(bert_out);

        const d_en = try linear(bert_out, try need(&self.trunk, "bert_encoder.weight"), try need(&self.trunk, "bert_encoder.bias"), s);
        defer _ = mlx.mlx_array_free(d_en);

        const d = try durationEncoderForward(&self.trunk, a, self.cfg, d_en, s_pred, s);
        defer _ = mlx.mlx_array_free(d);

        // ── durations ──
        const dur_logits = blk: {
            var buf: [128]u8 = undefined;
            const lstm = try loadBiLstm(&self.trunk, &buf, "predictor.lstm", s);
            defer _ = mlx.mlx_array_free(lstm.fwd.bias);
            defer _ = mlx.mlx_array_free(lstm.rev.bias);
            const h = try lstm.forward(a, d, s);
            defer _ = mlx.mlx_array_free(h);
            break :blk try linear(
                h,
                try need(&self.trunk, "predictor.duration_proj.linear_layer.weight"),
                try need(&self.trunk, "predictor.duration_proj.linear_layer.bias"),
                s,
            );
        };
        defer _ = mlx.mlx_array_free(dur_logits);

        _ = mlx.mlx_array_eval(dur_logits);
        const logit_data = mlx.mlx_array_data_float32(dur_logits) orelse return error.KokoroDurationReadFailed;
        const durations = try predictedDurations(a, logit_data[0 .. ids.len * self.cfg.max_dur], ids.len, self.cfg.max_dur, speed);
        defer a.free(durations);

        if (trace) |t| t.durations = try a.dupe(u32, durations);

        const indices = try expandIndices(a, durations);
        defer a.free(indices);
        if (indices.len == 0) return error.EmptyKokoroInput;

        const idx_arr = mlx.mlx_array_new_data(indices.ptr, &[_]c_int{@intCast(indices.len)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx_arr);

        // ── expand to frames ──
        const en = try takeAxis(d, idx_arr, 1, s);
        defer _ = mlx.mlx_array_free(en);

        const curves = try f0NtrainForward(&self.trunk, a, en, s_pred, s);
        defer _ = mlx.mlx_array_free(curves.f0);
        defer _ = mlx.mlx_array_free(curves.n);

        if (trace) |t| {
            t.f0 = try copyToHost(a, curves.f0, s);
            t.n = try copyToHost(a, curves.n, s);
        }

        const t_en = try textEncoderForward(&self.trunk, a, self.cfg, input_ids, s);
        defer _ = mlx.mlx_array_free(t_en);
        const asr = try takeAxis(t_en, idx_arr, 1, s);
        defer _ = mlx.mlx_array_free(asr);
        if (trace) |t| t.asr = try copyToHost(a, asr, s);

        return decoderForward(&self.trunk, a, self.cfg, asr, curves.f0, curves.n, s_dec, seed, s);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Engine — the `gen.AudioBackend` arm
// ════════════════════════════════════════════════════════════════════════

const g2p = @import("kokoro_g2p.zig");
const wav = @import("wav.zig");

/// Model + phonemizer, so `/v1/audio/speech` can take TEXT. Owned by
/// `gen.AudioEngine`.
pub const Engine = struct {
    allocator: std.mem.Allocator,
    model: Model,
    phonemizer: g2p.Phonemizer,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.model = try Model.load(io, allocator, model_dir, s);
        errdefer self.model.deinit();
        self.phonemizer = try g2p.Phonemizer.load(io, allocator, model_dir);
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.phonemizer.deinit();
        self.model.deinit();
        self.allocator.destroy(self);
    }

    pub fn sampleRate(self: *const Engine) u32 {
        return self.model.cfg.sample_rate;
    }

    pub fn hasVoice(self: *const Engine, spec: []const u8) bool {
        // A blend is valid only if EVERY named voice is.
        var it = std.mem.splitScalar(u8, spec, ',');
        var any = false;
        while (it.next()) |raw| {
            const name = std.mem.trim(u8, raw, " \t");
            if (name.len == 0) continue;
            if (!self.model.hasVoice(name)) return false;
            any = true;
        }
        return any;
    }

    /// Text → 24 kHz mono WAV bytes (caller frees).
    pub fn synthesizeWav(self: *Engine, text: []const u8, voice: []const u8, speed: f32, seed: u64) ![]u8 {
        const a = self.allocator;
        const phonemes = try self.phonemizer.phonemize(a, text);
        defer a.free(phonemes);
        log.debug("[kokoro] \"{s}\" → {s}\n", .{ text, phonemes });

        const samples = try self.model.synthesize(phonemes, voice, speed, seed);
        defer a.free(samples);
        return wav.encodePcm16Mono(a, samples, self.model.cfg.sample_rate);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "kokoro: config defaults survive an empty object and samplesPerFrame is 300" {
    const cfg = try parseConfig(testing.allocator, "{}");
    try testing.expectEqual(@as(u32, 178), cfg.n_token);
    try testing.expectEqual(@as(u32, 512), cfg.hidden_dim);
    try testing.expectEqual(@as(u32, 128), cfg.style_dim);
    // 10 × 6 × 5 — one phoneme frame is 600 samples once F0/N's ×2 is applied.
    try testing.expectEqual(@as(u32, 300), cfg.samplesPerFrame());
}

test "kokoro: config parses the published v1.0 shape" {
    const json =
        \\{"n_token":178,"hidden_dim":512,"style_dim":128,"n_layer":3,"max_dur":50,
        \\ "plbert":{"hidden_size":768,"num_attention_heads":12,"intermediate_size":2048,
        \\           "num_hidden_layers":12,"max_position_embeddings":512},
        \\ "istftnet":{"upsample_rates":[10,6],"upsample_kernel_sizes":[20,12],
        \\             "upsample_initial_channel":512,"gen_istft_n_fft":20,
        \\             "gen_istft_hop_size":5}}
    ;
    const cfg = try parseConfig(testing.allocator, json);
    try testing.expectEqual(@as(u32, 768), cfg.bert_hidden);
    try testing.expectEqual(@as(u32, 12), cfg.bert_layers);
    try testing.expectEqual(@as(u32, 2048), cfg.bert_inter);
    try testing.expectEqual([2]u32{ 10, 6 }, cfg.upsample_rates);
    try testing.expectEqual([2]u32{ 20, 12 }, cfg.upsample_kernel_sizes);
    try testing.expectEqual(@as(u32, 20), cfg.gen_istft_n_fft);
    try testing.expectEqual(@as(u32, 5), cfg.gen_istft_hop_size);
}

test "kokoro: vocab encode wraps in boundary zeros and handles multi-byte IPA" {
    // A cut-down config carrying the real ids for these symbols.
    const json =
        \\{"vocab":{"h":50,"ə":83,"l":54,"ˈ":156,"o":57}}
    ;
    var vocab = try Vocab.parse(testing.allocator, json);
    defer vocab.deinit();

    // "həlˈo" — ə and ˈ are 2-byte UTF-8, so a byte-wise splitter would miss them.
    const ids = try vocab.encode(testing.allocator, "həlˈo");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 50, 83, 54, 156, 57, 0 }, ids);
}

test "kokoro: vocab encode drops unknown symbols rather than failing" {
    const json =
        \\{"vocab":{"h":50,"o":57}}
    ;
    var vocab = try Vocab.parse(testing.allocator, json);
    defer vocab.deinit();

    const ids = try vocab.encode(testing.allocator, "hZoQ");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 50, 57, 0 }, ids);
}

test "kokoro: expandIndices is repeat_interleave over the duration vector" {
    const idx = try expandIndices(testing.allocator, &[_]u32{ 2, 1, 3 });
    defer testing.allocator.free(idx);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 1, 2, 2, 2 }, idx);
}

test "kokoro: expandIndices on an empty duration vector yields no frames" {
    const idx = try expandIndices(testing.allocator, &[_]u32{});
    defer testing.allocator.free(idx);
    try testing.expectEqual(@as(usize, 0), idx.len);
}

test "kokoro: predictedDurations sums sigmoids, clamps to >= 1, and scales by speed" {
    // max_dur = 4. Row 0: four large logits → sigmoid≈1 each → ≈4 frames.
    // Row 1: four very negative logits → ≈0 → must CLAMP to 1, never 0
    //        (a 0-frame phoneme would vanish from the alignment).
    const logits = [_]f32{
        20, 20, 20, 20,
        -20, -20, -20, -20,
    };
    const dur = try predictedDurations(testing.allocator, &logits, 2, 4, 1.0);
    defer testing.allocator.free(dur);
    try testing.expectEqualSlices(u32, &[_]u32{ 4, 1 }, dur);

    // speed 2.0 halves the frame count (faster speech = fewer frames).
    const fast = try predictedDurations(testing.allocator, &logits, 2, 4, 2.0);
    defer testing.allocator.free(fast);
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 1 }, fast);
}

test "kokoro: duration rounding is half-to-even, matching torch.round" {
    // torch.round(2.5) == 2, NOT 3. Zig's @round gives 3, which drifts the
    // alignment by a frame against the reference.
    try testing.expectEqual(@as(f32, 2.0), roundHalfToEven(2.5));
    try testing.expectEqual(@as(f32, 4.0), roundHalfToEven(3.5));
    try testing.expectEqual(@as(f32, 2.0), roundHalfToEven(1.5));
    // Non-ties are unchanged.
    try testing.expectEqual(@as(f32, 3.0), roundHalfToEven(2.6));
    try testing.expectEqual(@as(f32, 2.0), roundHalfToEven(2.4));
}

// ── STFT framing ────────────────────────────────────────────────────────

test "kokoro: frame → window → overlap-add reconstructs the signal (COLA)" {
    // Perfect reconstruction is the whole contract of the iSTFT head: analysis
    // windows, synthesis windows again, divides by Σw². If padding or the frame
    // count is off by one this drifts or clips the tail, and the only symptom
    // downstream is audio that sounds subtly wrong.
    const n_fft: usize = 20;
    const hop: usize = 5;
    const a = testing.allocator;

    const w = try hannPeriodic(a, n_fft);
    defer a.free(w);

    var sig: [200]f32 = undefined;
    for (&sig, 0..) |*v, i| {
        const fi: f32 = @floatFromInt(i);
        v.* = @sin(fi * 0.3) * 0.7 + @cos(fi * 0.05);
    }

    const framed = try frameSignal(a, &sig, n_fft, hop);
    defer a.free(framed.data);

    // Analysis window (the engine applies this before the rfft).
    for (0..framed.frames) |f| {
        for (0..n_fft) |k| framed.data[f * n_fft + k] *= w[k];
    }

    const rec = try overlapAdd(a, framed.data, framed.frames, n_fft, hop, w);
    defer a.free(rec);

    try testing.expectEqual(sig.len, rec.len);
    for (sig, rec) |want, got| try testing.expectApproxEqAbs(want, got, 1e-4);
}

test "kokoro: frame count follows the centre-padded torch.stft convention" {
    const a = testing.allocator;
    const silence: [100]f32 = @splat(0);
    const framed = try frameSignal(a, &silence, 20, 5);
    defer a.free(framed.data);
    // centre-padded length 100 + 20 = 120 → (120 - 20)/5 + 1 = 21
    try testing.expectEqual(@as(usize, 21), framed.frames);
    try testing.expectEqual(@as(usize, 21 * 20), framed.data.len);
}

// ── BiLSTM ──────────────────────────────────────────────────────────────

fn testArray(data: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

fn readAll(arr: mlx.mlx_array, s: S) []const f32 {
    _ = mlx.mlx_array_eval(arr);
    var f = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    var n: c_int = 1;
    for (mlx.getShape(arr)) |d| n *= d;
    return mlx.mlx_array_data_float32(f).?[0..@intCast(n)];
}

test "kokoro: BiLSTM with zeroed recurrence matches the closed-form gate math" {
    // Killing w_hh removes the recurrence, so every timestep reduces to
    //   h = sigmoid(b_o) · tanh(sigmoid(b_i) · tanh(b_g))
    // which is checkable by hand. This pins the PyTorch gate ORDER (i,f,g,o) —
    // get it wrong and the model still runs, just sounds wrong.
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;

    const h_dim: usize = 2;
    const in_dim: usize = 1;

    const w_ih: [4 * h_dim * in_dim]f32 = @splat(0);
    const w_hh: [4 * h_dim * h_dim]f32 = @splat(0);
    // bias laid out i,i,f,f,g,g,o,o. The FORGET bias is slammed shut (-20 →
    // sigmoid ≈ 2e-9): zeroing w_hh alone does NOT make h constant, because c
    // still accumulates through the forget gate (c_t = f·c_{t-1} + i·g). With
    // f ≈ 0 every step is independent and h is the same at every t.
    const bias = [_]f32{ 0.5, 0.5, -20.0, -20.0, 1.0, 1.0, 0.25, 0.25 };

    const dir = LstmDir{
        .w_ih = testArray(&w_ih, &[_]c_int{ 4 * h_dim, in_dim }),
        .w_hh = testArray(&w_hh, &[_]c_int{ 4 * h_dim, h_dim }),
        .bias = testArray(&bias, &[_]c_int{4 * h_dim}),
    };
    defer _ = mlx.mlx_array_free(dir.w_ih);
    defer _ = mlx.mlx_array_free(dir.w_hh);
    defer _ = mlx.mlx_array_free(dir.bias);

    const lstm = BiLstm{ .fwd = dir, .rev = dir, .hidden = h_dim };
    const x_data = [_]f32{ 1.0, 2.0, 3.0 };
    const x = testArray(&x_data, &[_]c_int{ 1, 3, in_dim });
    defer _ = mlx.mlx_array_free(x);

    const y = try lstm.forward(a, x, s);
    defer _ = mlx.mlx_array_free(y);

    const sh = mlx.getShape(y);
    try testing.expectEqual(@as(c_int, 1), sh[0]);
    try testing.expectEqual(@as(c_int, 3), sh[1]);
    try testing.expectEqual(@as(c_int, 2 * h_dim), sh[2]); // bidirectional concat

    const sig_i: f32 = 1.0 / (1.0 + @exp(@as(f32, -0.5)));
    const sig_o: f32 = 1.0 / (1.0 + @exp(@as(f32, -0.25)));
    const want: f32 = sig_o * std.math.tanh(sig_i * std.math.tanh(@as(f32, 1.0)));

    const got = readAll(y, s);
    // With no recurrence every timestep and both directions agree.
    for (got) |v| try testing.expectApproxEqAbs(want, v, 1e-5);
}

test "kokoro: BiLSTM reverse half reads the sequence backwards" {
    // The forward half at t=0 has seen only x[0]; the reverse half at t=0 has
    // seen the WHOLE sequence. Feeding a sequence that is zero everywhere
    // except the last step separates them: forward t=0 stays at its bias-only
    // value, reverse t=0 does not. This is the test that catches a reverse pass
    // stored in the wrong order (which otherwise produces plausible audio).
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;

    const h_dim: usize = 1;
    const in_dim: usize = 1;
    // Route input into every gate so a nonzero x visibly moves h.
    const w_ih = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const w_hh = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const bias = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const dir = LstmDir{
        .w_ih = testArray(&w_ih, &[_]c_int{ 4 * h_dim, in_dim }),
        .w_hh = testArray(&w_hh, &[_]c_int{ 4 * h_dim, h_dim }),
        .bias = testArray(&bias, &[_]c_int{4 * h_dim}),
    };
    defer _ = mlx.mlx_array_free(dir.w_ih);
    defer _ = mlx.mlx_array_free(dir.w_hh);
    defer _ = mlx.mlx_array_free(dir.bias);

    const lstm = BiLstm{ .fwd = dir, .rev = dir, .hidden = h_dim };
    const x_data = [_]f32{ 0.0, 0.0, 1.0 };
    const x = testArray(&x_data, &[_]c_int{ 1, 3, in_dim });
    defer _ = mlx.mlx_array_free(x);

    const y = try lstm.forward(a, x, s);
    defer _ = mlx.mlx_array_free(y);
    const got = readAll(y, s); // [t0_fwd, t0_rev, t1_fwd, t1_rev, t2_fwd, t2_rev]

    // Forward at t=0 and t=1 has seen only zeros → h stays 0.
    try testing.expectApproxEqAbs(@as(f32, 0.0), got[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), got[2], 1e-6);
    // THE direction assertion: reverse at t=0 has already consumed x[2]=1, so
    // it is nonzero. A reverse pass that secretly ran forwards would read 0
    // here and every other number in this test would still look plausible.
    try testing.expect(@abs(got[1]) > 1e-3);
    try testing.expect(@abs(got[3]) > 1e-3);
    // Forward reaches x[2] only at the last step; both halves agree there
    // because each has then consumed exactly the one nonzero input.
    try testing.expectApproxEqAbs(got[4], got[5], 1e-6);
    try testing.expect(@abs(got[4]) > 1e-3);
}

// ── Live model (env-gated) ──────────────────────────────────────────────
//
// `KOKORO_TEST_MODEL=<dir> zig build test -Dtest-filter=kokoro`
// Weights come from `tests/convert_kokoro_weights.py`.

fn liveModelDir() ?[]const u8 {
    const p = std.c.getenv("KOKORO_TEST_MODEL") orelse return null;
    return std.mem.span(p);
}

fn testIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

test "kokoro: live synthesis produces well-formed audio" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();

    // "hello world" in Kokoro's IPA inventory.
    const phonemes = "həlˈoʊ wˈɜːld";
    const audio = try model.synthesize(phonemes, DEFAULT_VOICE, 1.0, 0);
    defer a.free(audio);

    // Length is exactly 600 samples per phoneme frame — the whole pipeline is
    // deterministic in length, so a mismatch means a stride/padding bug.
    try testing.expect(audio.len > 0);
    try testing.expectEqual(@as(usize, 0), audio.len % 600);

    var peak: f32 = 0;
    var energy: f64 = 0;
    for (audio) |v| {
        try testing.expect(std.math.isFinite(v));
        peak = @max(peak, @abs(v));
        energy += @as(f64, v) * @as(f64, v);
    }
    const rms = @sqrt(energy / @as(f64, @floatFromInt(audio.len)));

    // Speech, not silence and not a blown-up waveform. A wrong AdaIN or a
    // dropped weight-norm fold typically lands outside one of these.
    try testing.expect(peak > 0.01);
    try testing.expect(peak < 10.0);
    try testing.expect(rms > 0.001);
    std.debug.print("\n[kokoro] {d} samples ({d:.2}s) peak={d:.3} rms={d:.4}\n", .{
        audio.len,
        @as(f32, @floatFromInt(audio.len)) / 24000.0,
        peak,
        rms,
    });
}

test "kokoro: live voice blending averages the packs" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();

    const one = try model.styleFor("af_bella", 10);
    defer _ = mlx.mlx_array_free(one);
    const other = try model.styleFor("af_sky", 10);
    defer _ = mlx.mlx_array_free(other);
    const blend = try model.styleFor("af_bella,af_sky", 10);
    defer _ = mlx.mlx_array_free(blend);

    const d1 = readAll(one, s);
    const d2 = readAll(other, s);
    const db = readAll(blend, s);
    try testing.expectEqual(@as(usize, 256), db.len);
    for (0..256) |i| {
        try testing.expectApproxEqAbs((d1[i] + d2[i]) / 2.0, db[i], 1e-5);
    }
    // The blend must be a real third voice, not a no-op alias of either input.
    var differs = false;
    for (0..256) |i| {
        if (@abs(db[i] - d1[i]) > 1e-4) differs = true;
    }
    try testing.expect(differs);

    try testing.expectError(error.UnknownKokoroVoice, model.styleFor("no_such_voice", 10));
}

// ── Parity oracles (env-gated) ──────────────────────────────────────────
//
// Fixtures from `tests/dump_kokoro_fixtures.py`. Run with BOTH:
//   KOKORO_TEST_MODEL=<converted dir> KOKORO_FIXTURES=<dump dir>

fn fixturesDir() ?[]const u8 {
    const p = std.c.getenv("KOKORO_FIXTURES") orelse return null;
    return std.mem.span(p);
}

fn readFixtureF32(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const bytes = try readWholeFile(testIo(), a, path);
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn readFixtureI32(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]i32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const bytes = try readWholeFile(testIo(), a, path);
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(i32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn cosine(x: []const f32, y: []const f32) f64 {
    var dot: f64 = 0;
    var nx: f64 = 0;
    var ny: f64 = 0;
    for (x, y) |a, b| {
        dot += @as(f64, a) * @as(f64, b);
        nx += @as(f64, a) * @as(f64, a);
        ny += @as(f64, b) * @as(f64, b);
    }
    if (nx == 0 or ny == 0) return 0;
    return dot / (@sqrt(nx) * @sqrt(ny));
}

/// The reference phoneme string the dump script defaults to.
const ORACLE_PHONEMES = "həlˈoʊ wˈɜːld";

test "kokoro oracle: durations match the reference EXACTLY" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const fx = fixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();

    var trace = Model.Trace{};
    defer trace.deinit(a);
    const audio = try model.synthesizeTraced(ORACLE_PHONEMES, "af_heart", 1.0, 0, &trace);
    defer a.free(audio);

    const want = try readFixtureI32(a, fx, "durations.i32");
    defer a.free(want);
    const got = trace.durations.?;

    // Durations are fully deterministic — integer frame counts out of ALBERT,
    // the DurationEncoder, the duration BiLSTM and duration_proj. Exact
    // equality here means the entire prosody trunk is correct, including the
    // half-to-even rounding. A cosine check would hide an off-by-one frame.
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectEqual(@as(u32, @intCast(w)), g);
}

test "kokoro oracle: F0, noise and asr match the reference" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const fx = fixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();

    var trace = Model.Trace{};
    defer trace.deinit(a);
    const audio = try model.synthesizeTraced(ORACLE_PHONEMES, "af_heart", 1.0, 0, &trace);
    defer a.free(audio);

    const f0_ref = try readFixtureF32(a, fx, "f0.f32");
    defer a.free(f0_ref);
    const n_ref = try readFixtureF32(a, fx, "n.f32");
    defer a.free(n_ref);
    const asr_ref = try readFixtureF32(a, fx, "asr.f32");
    defer a.free(asr_ref);

    try testing.expectEqual(f0_ref.len, trace.f0.?.len);
    try testing.expectEqual(n_ref.len, trace.n.?.len);
    try testing.expectEqual(asr_ref.len, trace.asr.?.len);

    const cos_f0 = cosine(f0_ref, trace.f0.?);
    const cos_n = cosine(n_ref, trace.n.?);
    const cos_asr = cosine(asr_ref, trace.asr.?);
    std.debug.print("\n[kokoro oracle] f0={d:.6} n={d:.6} asr={d:.6}\n", .{ cos_f0, cos_n, cos_asr });

    // Deterministic paths — these are pure forward passes with no sampling, so
    // the bar is parity, not similarity.
    try testing.expect(cos_asr > 0.999);
    try testing.expect(cos_f0 > 0.999);
    try testing.expect(cos_n > 0.999);
}

test "kokoro oracle: generated audio tracks the reference" {
    const dir = liveModelDir() orelse return error.SkipZigTest;
    const fx = fixturesDir() orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var model = try Model.load(testIo(), a, dir, s);
    defer model.deinit();

    const audio = try model.synthesize(ORACLE_PHONEMES, "af_heart", 1.0, 0);
    defer a.free(audio);

    const ref = try readFixtureF32(a, fx, "audio.f32");
    defer a.free(ref);

    // Sample count is fully determined by the durations, so this stays exact
    // even though the waveform itself is stochastic.
    try testing.expectEqual(ref.len, audio.len);

    const cos = cosine(ref, audio);
    std.debug.print("[kokoro oracle] audio={d:.6}\n", .{cos});
    // The bar is the REFERENCE'S OWN NOISE FLOOR, measured, not guessed: the
    // torch model against itself at three different seeds scores 0.9941–0.9960,
    // because the only surviving randomness is SineGen's additive noise (the
    // random initial phase is discarded by a downsample — see `harmonicSource`).
    // So "it's stochastic" excuses a couple of thousandths, nothing more. An
    // earlier build scored 0.477 here and that was a real bug, not PRNG drift.
    try testing.expect(cos > 0.98);
}

test "kokoro: periodic Hann window matches torch.hann_window(n, periodic=true)" {
    const w = try hannPeriodic(testing.allocator, 20);
    defer testing.allocator.free(w);
    try testing.expectEqual(@as(usize, 20), w.len);
    // Periodic: w[0] == 0 and the window does NOT return to 0 at the last tap
    // (that is the symmetric form). torch.hann_window(20)[19] ≈ 0.0245.
    try testing.expectApproxEqAbs(@as(f32, 0.0), w[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.024472), w[19], 1e-5);
    // Peak at the midpoint.
    try testing.expectApproxEqAbs(@as(f32, 1.0), w[10], 1e-6);
}
