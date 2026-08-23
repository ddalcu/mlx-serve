//! ACE-Step v1.5 XL Turbo — native text-to-music engine (`.audio` modality,
//! second backend beside Qwen3-TTS in gen.AudioBackend).
//!
//! Pipeline (music-dossier.md is the BINDING reference; verified against the
//! upstream repo's own MLX implementation acestep/models/mlx/*):
//!   prompt/lyrics strings → Qwen3-Embedding-0.6B (full forward for the prompt,
//!   embed-table lookup for lyrics) → condition encoder (lyric 8L + timbre 4L
//!   CLS + text projector, packed [lyric|timbre|text]) → 32-layer DiT (AdaLN,
//!   bidirectional alternating sliding-128/full self-attn, cross-attn K/V
//!   computed ONCE per generation) → 8-step Euler flow-match (shift=3 schedule)
//!   + DCW haar "double" correction → AutoencoderOobleck VAE decoder (chunked)
//!   → 48 kHz stereo WAV, peak-normalized to −1 dBFS.
//!
//! Weights arrive PRE-CONVERTED by tests/convert_acestep_weights.py:
//!   model.safetensors  DiT + condition encoder + silence_latent (bf16 / 8-bit)
//!   vae.safetensors    Oobleck VAE, weight-norm fused, MLX conv layouts
//!   fsq.safetensors    FSQ audio tokenizer + detokenizer (cover mode; loaded
//!                      on first use, absent on packs that predate it)
//!   text_encoder/      Qwen3-Embedding-0.6B verbatim (bf16, standard qwen3)
//!
//! Tasks: text2music (context = [silence | ones]), complete (= vocal2bgm: the
//! source clip's VAE latent as context, "Complete the input track…"), cover
//! (the source latent round-tripped through pooler → FSQ → detokenizer, the
//! "Generate audio semantic tokens…" instruction). Same DiT forward for all
//! three — the task is the CONTEXT stream + the instruction line.
//! The converter already drops the Sequential indices (proj_in.1 → proj_in) and
//! swaps conv layouts — this engine does STANDARD reshapes only.
//!
//! Numerics: latents/sampler state fp32; transformer compute bf16; Snake
//! activations f32 (exp headroom, α/β stored f32). Parity: env-gated
//! `ACESTEP_*` cos oracles fed by tests/dump_acestep_fixtures.py.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const tok_mod = @import("tokenizer.zig");
const wav_mod = @import("wav.zig");
const sse = @import("gen_sse.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

// ── architecture facts (converted config.json mirrors these; the checkpoint
// family has exactly one member so they double as defaults) ─────────────────
pub const Cfg = struct {
    hidden: u32 = 2560,
    layers: u32 = 32,
    heads: u32 = 32,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    intermediate: u32 = 9728,
    enc_hidden: u32 = 2048,
    enc_intermediate: u32 = 6144,
    enc_heads: u32 = 16,
    enc_kv_heads: u32 = 8,
    lyric_layers: u32 = 8,
    timbre_layers: u32 = 4,
    text_hidden: u32 = 1024,
    acoustic_dim: u32 = 64,
    in_channels: u32 = 192,
    patch_size: u32 = 2,
    sliding_window: u32 = 128,
    rope_theta: f32 = 1_000_000.0,
    eps: f32 = 1e-6,
    timbre_fix_frame: u32 = 750,
    sample_rate: u32 = 48000,
    vae_hop: u32 = 1920, // 2*4*4*6*10 → latents at exactly 25 Hz
    // FSQ audio tokenizer (cover mode): 5 latent frames pool to one token.
    pool_window: u32 = 5,
    pooler_layers: u32 = 2,
    fsq_levels: [6]f32 = .{ 8, 8, 8, 5, 5, 5 },
};

// Qwen3-Embedding-0.6B (text_encoder/, standard qwen3 — keys are unprefixed:
// embed_tokens.weight / layers.N.* / norm.weight).
const QwenCfg = struct {
    hidden: u32 = 1024,
    layers: u32 = 28,
    heads: u32 = 16,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
};

pub const NUM_STEPS: u32 = 8; // turbo is distillation-fixed
pub const SHIFT: f32 = 3.0; // turbo UI default (yields the SHIFT_TIMESTEPS[3.0] table)
const DCW_LOW: f32 = 0.05; // DCW "double" defaults (reference GenerationParams)
const DCW_HIGH: f32 = 0.02;
const NORMALIZE_DB: f32 = -1.0; // peak-normalize target
pub const MIN_DURATION_S: u32 = 10;
pub const MAX_DURATION_S: u32 = 600;
const VAE_CHUNK_FRAMES: usize = 512; // latent frames per decode window (~20 s)
const VAE_CHUNK_FRAMES_LOWMEM: usize = 128; // ~5 s — decode is the pipeline's peak
const VAE_OVERLAP_FRAMES: usize = 64; // reference tiled-decode overlap

/// Decode window for the memory budget. Low-mem (iOS jetsam ceilings, small
/// Macs) trades a few extra overlap re-decodes for a ~4× smaller working set
/// in the upsampling stack — identical audio in the cores either way.
fn vaeChunkFrames(low_mem: bool) usize {
    return if (low_mem) VAE_CHUNK_FRAMES_LOWMEM else VAE_CHUNK_FRAMES;
}
const MAX_PROMPT_TOKENS: usize = 256; // reference truncation limits
const MAX_LYRIC_TOKENS: usize = 2048;
const VAE_ENC_CHUNK_FRAMES: usize = 750; // 30 s encode windows (f32 activations)
pub const FSQ_FILE = "fsq.safetensors";

// ════════════════════════════════════════════════════════════════════════
// Pure helpers (schedule, prompt formatting, normalization) — hermetic tests.
// ════════════════════════════════════════════════════════════════════════

test "low-mem shrinks the VAE decode window but never below the overlap" {
    try std.testing.expectEqual(VAE_CHUNK_FRAMES, vaeChunkFrames(false));
    try std.testing.expectEqual(VAE_CHUNK_FRAMES_LOWMEM, vaeChunkFrames(true));
    // Redundant halo (≤ overlap each side) must not dominate the useful core;
    // core ≥ overlap keeps the re-decode factor ≤ 3× worst case.
    try std.testing.expect(vaeChunkFrames(true) >= VAE_OVERLAP_FRAMES);
    try std.testing.expect(vaeChunkFrames(false) >= VAE_OVERLAP_FRAMES);
}

/// Flow-match timestep schedule: t_i = 1 − i/N through the shift transform
/// t ← shift·t / (1 + (shift−1)·t). N=8, shift=3 reproduces the torch model's
/// SHIFT_TIMESTEPS[3.0] lookup table exactly.
pub fn timestepSchedule(buf: []f32, shift: f32) void {
    const n = buf.len;
    for (0..n) |i| {
        var t: f32 = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        if (shift != 1.0) t = shift * t / (1.0 + (shift - 1.0) * t);
        buf[i] = t;
    }
}

/// Index of the schedule entry nearest `t` (cover_noise_strength start point;
/// ties resolve to the EARLIER step, like Python's `min` over the list).
pub fn nearestStep(sched: []const f32, t: f32) usize {
    var best: usize = 0;
    for (sched, 0..) |v, i| {
        if (@abs(v - t) < @abs(sched[best] - t)) best = i;
    }
    return best;
}

/// Latent frame count for a duration: exactly 25 frames/s (48000/1920).
pub fn latentFrames(duration_s: u32) u32 {
    return duration_s * 25;
}

/// Metas block (music-dossier.md §2.4). `bpm`/`keyscale`/`timesignature`
/// default to "N/A"; duration is always present ("- duration: N seconds\n").
pub fn formatMetaString(a: std.mem.Allocator, bpm: ?u32, keyscale: []const u8, timesignature: []const u8, duration_s: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    if (bpm) |b| {
        try out.print(a, "- bpm: {d}\n", .{b});
    } else {
        try out.appendSlice(a, "- bpm: N/A\n");
    }
    const ts = std.mem.trim(u8, timesignature, " \t");
    try out.print(a, "- timesignature: {s}\n", .{if (ts.len == 0) "N/A" else ts});
    const ks = std.mem.trim(u8, keyscale, " \t");
    try out.print(a, "- keyscale: {s}\n", .{if (ks.len == 0) "N/A" else ks});
    try out.print(a, "- duration: {d} seconds\n", .{duration_s});
    return out.toOwnedSlice(a);
}

/// SFT_GEN_PROMPT. The metas string already ends with '\n', so `<|endoftext|>`
/// follows the duration line. `instruction` comes from `taskInstruction`.
pub fn formatPrompt(a: std.mem.Allocator, instruction: []const u8, caption: []const u8, metas: []const u8) ![]u8 {
    return std.fmt.allocPrint(
        a,
        "# Instruction\n{s}\n\n# Caption\n{s}\n\n# Metas\n{s}<|endoftext|>\n",
        .{ instruction, caption, metas },
    );
}

pub const Task = enum { text2music, cover, complete };

pub const TEXT2MUSIC_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:";
pub const COVER_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:";

/// The instruction line is part of the TASK (constants.py TASK_INSTRUCTIONS via
/// task_utils.generate_instruction). `track_classes` is the `A | B` list from
/// `joinTrackClasses` for `complete`; empty = the default spelling.
pub fn taskInstruction(a: std.mem.Allocator, task: Task, track_classes: []const u8) ![]u8 {
    return switch (task) {
        .text2music => a.dupe(u8, TEXT2MUSIC_INSTRUCTION),
        .cover => a.dupe(u8, COVER_INSTRUCTION),
        .complete => if (track_classes.len == 0)
            a.dupe(u8, "Complete the input track:")
        else
            std.fmt.allocPrint(a, "Complete the input track with {s}:", .{track_classes}),
    };
}

/// The instrument vocabulary the checkpoint was trained on (constants.py
/// TRACK_NAMES). Anything else is a named 400, not an in-prompt typo.
pub const TRACK_NAMES = [_][]const u8{ "woodwinds", "brass", "fx", "synth", "strings", "percussion", "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals" };

/// `["bass","drums"]` → "BASS | DRUMS" (case-insensitive match, upper-cased,
/// ` | `-joined, request order kept — task_utils.generate_instruction).
pub fn joinTrackClasses(a: std.mem.Allocator, names: []const []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    for (names, 0..) |raw, i| {
        const name = std.mem.trim(u8, raw, " \t\r\n");
        var known = false;
        for (TRACK_NAMES) |t| {
            if (std.ascii.eqlIgnoreCase(name, t)) known = true;
        }
        if (!known) return error.UnknownTrackClass;
        if (i > 0) try out.appendSlice(a, " | ");
        for (name) |c| try out.append(a, std.ascii.toUpper(c));
    }
    return out.toOwnedSlice(a);
}

/// The reference marker for a wordless track. ACE-Step has had this since day
/// one as the empty-lyrics default; `resolveLyrics` is the explicit spelling.
pub const INSTRUMENTAL_LYRICS = "[Instrumental]";

/// The lyrics the engine should condition on. `instrumental: true` is the
/// EXPLICIT form of what an empty lyric block already meant here, so it
/// resolves to the same marker rather than opening a second code path.
pub fn resolveLyrics(instrumental: bool, lyrics: []const u8) []const u8 {
    return if (instrumental) INSTRUMENTAL_LYRICS else lyrics;
}

/// Lyric block. Empty lyrics default to the reference "[Instrumental]" marker.
pub fn formatLyrics(a: std.mem.Allocator, language: []const u8, lyrics: []const u8) ![]u8 {
    const lang = if (language.len == 0) "en" else language;
    const body = if (std.mem.trim(u8, lyrics, " \t\r\n").len == 0) INSTRUMENTAL_LYRICS else lyrics;
    return std.fmt.allocPrint(a, "# Languages\n{s}\n\n# Lyric\n{s}<|endoftext|>", .{ lang, body });
}

/// Peak-normalize in place to `target_db` dBFS (reference normalize_audio:
/// always rescales when a nonzero peak exists, up or down).
pub fn peakNormalize(samples: []f32, target_db: f32) void {
    var peak: f32 = 0;
    for (samples) |v| peak = @max(peak, @abs(v));
    if (peak <= 0) return;
    const gain = std.math.pow(f32, 10.0, target_db / 20.0) / peak;
    for (samples) |*v| v.* *= gain;
}

// ════════════════════════════════════════════════════════════════════════
// mlx micro-helpers
// ════════════════════════════════════════════════════════════════════════

fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
fn addA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, x, y, s));
    return o;
}
fn subA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, x, y, s));
    return o;
}
fn mulA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, y, s));
    return o;
}
fn mulScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    return mulA(x, c, s);
}
fn sliceA(x: mlx.mlx_array, start: []const c_int, stop: []const c_int, strides: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, s));
    return o;
}
fn concat2(x: mlx.mlx_array, y: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, x);
    _ = mlx.mlx_vector_array_append_value(vec, y);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}
/// Plain RMSNorm (Qwen3 style: x_normed * w, no +1).
fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
/// [B,T,H*hd] → [B,H,T,hd]
fn splitHeads(x: mlx.mlx_array, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const x4 = try reshape(x, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(x4);
    return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
}
/// [B,H,T,hd] → [B,T,H*hd]
fn mergeHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const t = try transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, s);
}
fn ropeInPlace(x: mlx.mlx_array, hd: c_int, theta: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&o, x, hd, false, mlx.mlx_optional_float.some(theta), 1.0, 0, .{ .ctx = null }, s));
    return o;
}
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: ?mlx.mlx_array, mode: [*:0]const u8, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, mask orelse null_a, null_a, s));
    return o;
}

fn getW(w: *const Weights, key: []const u8) !mlx.mlx_array {
    return w.get(key) orelse {
        log.err("[acestep] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingAceStepWeight;
    };
}

/// Linear over the Weights map: quantized (scales/biases present; bits +
/// group_size inferred from packed geometry against x's inner dim — the
/// MixedLinear rule) or dense bf16 (lazy-transpose matmul). Adds `.bias`
/// when present. `x` must be bf16.
fn lin(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
    defer a.free(sk);
    const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
    defer a.free(bk);
    const ak = try std.fmt.allocPrint(a, "{s}.bias", .{prefix});
    defer a.free(ak);

    const xsh = mlx.getShape(x);
    const in_features: u32 = @intCast(xsh[xsh.len - 1]);

    var o = mlx.mlx_array_new();
    if (w.get(sk)) |scales| {
        const wq = try getW(w, wk);
        const qb = try getW(w, bk);
        const w_cols: u32 = @intCast(mlx.getShape(wq)[1]);
        const s_cols: u32 = @intCast(mlx.getShape(scales)[1]);
        const bits: u32 = @divExact(32 * w_cols, in_features);
        const gs: u32 = @divExact(in_features, s_cols);
        try mlx.check(mlx.mlx_quantized_matmul(&o, x, wq, scales, qb, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
    } else {
        const wd = try getW(w, wk);
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        const axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wt, wd, &axes, 2, s));
        try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
    }
    if (w.get(ak)) |bias| {
        const r = try addA(o, bias, s);
        _ = mlx.mlx_array_free(o);
        o = r;
    }
    return o;
}

/// Bidirectional band mask for sliding-window layers: additive [1,1,T,T] bf16,
/// 0 where |i−j| ≤ window else −1e9 (matches the reference MLX mask).
fn slidingBandMask(t_len: c_int, window: c_int, s: S) !mlx.mlx_array {
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0.0, @floatFromInt(t_len), 1.0, .float32, s));
    const col = try reshape(ar, &[_]c_int{ t_len, 1 }, s);
    defer _ = mlx.mlx_array_free(col);
    const row = try reshape(ar, &[_]c_int{ 1, t_len }, s);
    defer _ = mlx.mlx_array_free(row);
    const diff = try subA(col, row, s);
    defer _ = mlx.mlx_array_free(diff);
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, diff, s));
    const wlim = mlx.mlx_array_new_float(@floatFromInt(window));
    defer _ = mlx.mlx_array_free(wlim);
    var inside = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inside);
    try mlx.check(mlx.mlx_less_equal(&inside, ad, wlim, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    const neg = mlx.mlx_array_new_float(-1e9);
    defer _ = mlx.mlx_array_free(neg);
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_where(&m, inside, zero, neg, s));
    const m4 = try reshape(m, &[_]c_int{ 1, 1, t_len, t_len }, s);
    defer _ = mlx.mlx_array_free(m4);
    return astype(m4, .bfloat16, s);
}

// ════════════════════════════════════════════════════════════════════════
// Qwen3-Embedding-0.6B text encoder (causal, standard qwen3; ltx gemmaCapture
// pattern: per-call key lookups over a resident Weights map).
// ════════════════════════════════════════════════════════════════════════

/// Embedding-table lookup: ids → [1,T,1024] bf16. This IS the whole lyric
/// path (the reference calls text_encoder.embed_tokens only for lyrics).
fn qwenEmbedLookup(w: *const Weights, ids: []const i32, s: S) !mlx.mlx_array {
    const table = try getW(w, "embed_tokens.weight");
    const id_shape = [_]c_int{@intCast(ids.len)};
    const ids_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_take_axis(&rows, table, ids_arr, 0, s));
    const hidden: c_int = mlx.getShape(table)[1];
    return reshape(rows, &[_]c_int{ 1, @intCast(ids.len), hidden }, s);
}

/// Full causal forward → post-final-norm hidden states [1,T,1024] bf16
/// (= HF last_hidden_state; the reference passes no attention mask).
fn qwenForward(w: *const Weights, allocator: std.mem.Allocator, ids: []const i32, s: S) !mlx.mlx_array {
    const cfg = QwenCfg{};
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    var h = try qwenEmbedLookup(w, ids, s);
    const nh: c_int = @intCast(cfg.heads);
    const nkv: c_int = @intCast(cfg.kv_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    var li: u32 = 0;
    while (li < cfg.layers) : (li += 1) {
        const pfx = try std.fmt.allocPrint(a, "layers.{d}", .{li});
        const in_ln = try getW(w, try std.fmt.allocPrint(a, "{s}.input_layernorm.weight", .{pfx}));
        const x = try rmsNorm(h, in_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(x);

        const q = try lin(w, a, x, try std.fmt.allocPrint(a, "{s}.self_attn.q_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(q);
        const k = try lin(w, a, x, try std.fmt.allocPrint(a, "{s}.self_attn.k_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(k);
        const v = try lin(w, a, x, try std.fmt.allocPrint(a, "{s}.self_attn.v_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(v);

        const qh = try splitHeads(q, nh, hd, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k, nkv, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v, nkv, hd, s);
        defer _ = mlx.mlx_array_free(vh);

        const qn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn.q_norm.weight", .{pfx}));
        const qn = try rmsNorm(qh, qn_w, cfg.eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const kn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn.k_norm.weight", .{pfx}));
        const kn = try rmsNorm(kh, kn_w, cfg.eps, s);
        defer _ = mlx.mlx_array_free(kn);

        const qr = try ropeInPlace(qn, hd, cfg.rope_theta, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try ropeInPlace(kn, hd, cfg.rope_theta, s);
        defer _ = mlx.mlx_array_free(kr);

        const attn = try sdpa(qr, kr, vh, scale, null, "causal", s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try lin(w, a, merged, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(h, o, s);
        _ = mlx.mlx_array_free(h);

        const pa_ln = try getW(w, try std.fmt.allocPrint(a, "{s}.post_attention_layernorm.weight", .{pfx}));
        const xm = try rmsNorm(h1, pa_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(xm);
        const gate = try lin(w, a, xm, try std.fmt.allocPrint(a, "{s}.mlp.gate_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(gate);
        const up = try lin(w, a, xm, try std.fmt.allocPrint(a, "{s}.mlp.up_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(up);
        const gact = try silu(gate, s);
        defer _ = mlx.mlx_array_free(gact);
        const gu = try mulA(gact, up, s);
        defer _ = mlx.mlx_array_free(gu);
        const down = try lin(w, a, gu, try std.fmt.allocPrint(a, "{s}.mlp.down_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(down);
        h = try addA(h1, down, s);
        _ = mlx.mlx_array_free(h1);

        _ = mlx.mlx_array_eval(h); // bound the lazy graph
        _ = arena_inst.reset(.retain_capacity);
    }

    const norm_w = try getW(w, "norm.weight");
    const out = try rmsNorm(h, norm_w, cfg.eps, s);
    _ = mlx.mlx_array_free(h);
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Condition encoder: lyric encoder (8 layers) + timbre encoder (4 layers,
// CLS pooling) + text projector, packed [lyric | timbre | text].
// Encoder layers are BIDIRECTIONAL pre-norm blocks; sliding layers (even
// index) apply the |i−j|≤128 band mask, full layers are unmasked (batch=1
// server traffic carries no padding).
// ════════════════════════════════════════════════════════════════════════

fn encoderLayer(e: *const Engine, a: std.mem.Allocator, w: *const Weights, prefix: []const u8, h_in: mlx.mlx_array, sliding_mask: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    const cfg = e.cfg;
    const nh: c_int = @intCast(cfg.enc_heads);
    const nkv: c_int = @intCast(cfg.enc_kv_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));

    const in_ln = try getW(w, try std.fmt.allocPrint(a, "{s}.input_layernorm.weight", .{prefix}));
    const x = try rmsNorm(h_in, in_ln, cfg.eps, s);
    defer _ = mlx.mlx_array_free(x);

    const q = try lin(w, a, x, try std.fmt.allocPrint(a, "{s}.self_attn.q_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(q);
    const k = try lin(w, a, x, try std.fmt.allocPrint(a, "{s}.self_attn.k_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(k);
    const v = try lin(w, a, x, try std.fmt.allocPrint(a, "{s}.self_attn.v_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(v);

    const qh = try splitHeads(q, nh, hd, s);
    defer _ = mlx.mlx_array_free(qh);
    const kh = try splitHeads(k, nkv, hd, s);
    defer _ = mlx.mlx_array_free(kh);
    const vh = try splitHeads(v, nkv, hd, s);
    defer _ = mlx.mlx_array_free(vh);

    const qn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn.q_norm.weight", .{prefix}));
    const qn = try rmsNorm(qh, qn_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const kn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn.k_norm.weight", .{prefix}));
    const kn = try rmsNorm(kh, kn_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(kn);

    const qr = try ropeInPlace(qn, hd, cfg.rope_theta, s);
    defer _ = mlx.mlx_array_free(qr);
    const kr = try ropeInPlace(kn, hd, cfg.rope_theta, s);
    defer _ = mlx.mlx_array_free(kr);

    const attn = if (sliding_mask) |m|
        try sdpa(qr, kr, vh, scale, m, "array", s)
    else
        try sdpa(qr, kr, vh, scale, null, "", s);
    defer _ = mlx.mlx_array_free(attn);
    const merged = try mergeHeads(attn, s);
    defer _ = mlx.mlx_array_free(merged);
    const o = try lin(w, a, merged, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(h_in, o, s);
    defer _ = mlx.mlx_array_free(h1);

    const pa_ln = try getW(w, try std.fmt.allocPrint(a, "{s}.post_attention_layernorm.weight", .{prefix}));
    const xm = try rmsNorm(h1, pa_ln, cfg.eps, s);
    defer _ = mlx.mlx_array_free(xm);
    const gate = try lin(w, a, xm, try std.fmt.allocPrint(a, "{s}.mlp.gate_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try lin(w, a, xm, try std.fmt.allocPrint(a, "{s}.mlp.up_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(up);
    const gact = try silu(gate, s);
    defer _ = mlx.mlx_array_free(gact);
    const gu = try mulA(gact, up, s);
    defer _ = mlx.mlx_array_free(gu);
    const down = try lin(w, a, gu, try std.fmt.allocPrint(a, "{s}.mlp.down_proj", .{prefix}), s);
    defer _ = mlx.mlx_array_free(down);
    return addA(h1, down, s);
}

/// Run an encoder stack (`encoder.lyric_encoder` / `encoder.timbre_encoder` /
/// the FSQ pooler + detokenizer) over pre-embedded input [B,T,2048]; returns
/// the FINAL-NORMED sequence. B>1 = independent windows: RoPE positions and
/// the band mask restart per batch row, which is exactly what the pooler's
/// `(b t) p c` regrouping wants.
fn encoderStack(e: *const Engine, allocator: std.mem.Allocator, w: *const Weights, base: []const u8, input: mlx.mlx_array, n_layers: u32, s: S) !mlx.mlx_array {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    const t_len: c_int = mlx.getShape(input)[1];
    const band = try slidingBandMask(t_len, @intCast(e.cfg.sliding_window), s);
    defer _ = mlx.mlx_array_free(band);

    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&h, input));
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        const pfx = try std.fmt.allocPrint(a, "{s}.layers.{d}", .{ base, li });
        // Even index = sliding_attention, odd = full_attention (config order).
        const mask: ?mlx.mlx_array = if (li % 2 == 0) band else null;
        const nh = try encoderLayer(e, a, w, pfx, h, mask, s);
        _ = mlx.mlx_array_free(h);
        h = nh;
        _ = mlx.mlx_array_eval(h);
        _ = arena_inst.reset(.retain_capacity);
    }
    const norm_key = try std.fmt.allocPrint(a, "{s}.norm.weight", .{base});
    const norm_w = try getW(w, norm_key);
    const out = try rmsNorm(h, norm_w, e.cfg.eps, s);
    _ = mlx.mlx_array_free(h);
    return out;
}

/// Full conditioning build: text hidden [1,Tt,1024] + lyric embeds [1,Tl,1024]
/// + timbre latents [1,Ttb,64] → packed encoder states [1, Tl+1+Tt, 2048] bf16.
fn buildConditioning(e: *const Engine, allocator: std.mem.Allocator, text_hidden: mlx.mlx_array, lyric_embeds: mlx.mlx_array, timbre_latents: mlx.mlx_array, s: S) !mlx.mlx_array {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    // Text: projector only (no encoder stack).
    const text_p = try lin(&e.w, a, text_hidden, "encoder.text_projector", s);
    defer _ = mlx.mlx_array_free(text_p);

    // Lyrics: embed_tokens Linear → 8-layer stack.
    const lyr_in = try lin(&e.w, a, lyric_embeds, "encoder.lyric_encoder.embed_tokens", s);
    defer _ = mlx.mlx_array_free(lyr_in);
    const lyr = try encoderStack(e, allocator, &e.w, "encoder.lyric_encoder", lyr_in, e.cfg.lyric_layers, s);
    defer _ = mlx.mlx_array_free(lyr);

    // Timbre: embed → prepend CLS → 4-layer stack → CLS output.
    const tim_in = try lin(&e.w, a, timbre_latents, "encoder.timbre_encoder.embed_tokens", s);
    defer _ = mlx.mlx_array_free(tim_in);
    const cls = try getW(&e.w, "encoder.timbre_encoder.special_token");
    const tim_seq = try concat2(cls, tim_in, 1, s);
    defer _ = mlx.mlx_array_free(tim_seq);
    const tim_out = try encoderStack(e, allocator, &e.w, "encoder.timbre_encoder", tim_seq, e.cfg.timbre_layers, s);
    defer _ = mlx.mlx_array_free(tim_out);
    const enc_h: c_int = @intCast(e.cfg.enc_hidden);
    const tim_cls = try sliceA(tim_out, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, 1, enc_h }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(tim_cls);

    // pack_sequences order (batch=1, all-valid): [lyric | timbre | text].
    const lt = try concat2(lyr, tim_cls, 1, s);
    defer _ = mlx.mlx_array_free(lt);
    return concat2(lt, text_p, 1, s);
}

// ════════════════════════════════════════════════════════════════════════
// DiT decoder
// ════════════════════════════════════════════════════════════════════════

/// TimestepEmbedding: sinusoidal(256, cos-first, t×1000) → linear_1 → SiLU →
/// linear_2 = temb [1,D]; time_proj(SiLU(temb)) → [1,6,D]. Both bf16.
fn timestepEmbed(e: *const Engine, a: std.mem.Allocator, name: []const u8, t_val: f32, s: S) !struct { temb: mlx.mlx_array, proj: mlx.mlx_array } {
    var sin_buf: [256]f32 = undefined;
    const half = 128;
    const ts = t_val * 1000.0;
    for (0..half) |i| {
        const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        const arg = ts * freq;
        sin_buf[i] = @cos(arg);
        sin_buf[half + i] = @sin(arg);
    }
    const sshape = [_]c_int{ 1, 256 };
    const sin_f32 = mlx.mlx_array_new_data(&sin_buf, &sshape, 2, .float32);
    defer _ = mlx.mlx_array_free(sin_f32);
    const sin_bf = try astype(sin_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sin_bf);

    const l1 = try lin(&e.w, a, sin_bf, try std.fmt.allocPrint(a, "decoder.{s}.linear_1", .{name}), s);
    defer _ = mlx.mlx_array_free(l1);
    const a1 = try silu(l1, s);
    defer _ = mlx.mlx_array_free(a1);
    const temb = try lin(&e.w, a, a1, try std.fmt.allocPrint(a, "decoder.{s}.linear_2", .{name}), s);
    errdefer _ = mlx.mlx_array_free(temb);
    const a2 = try silu(temb, s);
    defer _ = mlx.mlx_array_free(a2);
    const proj_flat = try lin(&e.w, a, a2, try std.fmt.allocPrint(a, "decoder.{s}.time_proj", .{name}), s);
    defer _ = mlx.mlx_array_free(proj_flat);
    const d: c_int = @intCast(e.cfg.hidden);
    const proj = try reshape(proj_flat, &[_]c_int{ 1, 6, d }, s);
    return .{ .temb = temb, .proj = proj };
}

/// Per-generation cross-attention K/V (computed once from the projected
/// conditioning, reused for all 8 steps — the EncoderDecoderCache pattern).
const CrossKv = struct {
    ks: []mlx.mlx_array,
    vs: []mlx.mlx_array,

    fn deinit(self: *CrossKv, allocator: std.mem.Allocator) void {
        for (self.ks) |k| _ = mlx.mlx_array_free(k);
        for (self.vs) |v| _ = mlx.mlx_array_free(v);
        allocator.free(self.ks);
        allocator.free(self.vs);
    }
};

/// Precompute per-layer cross K/V from the condition_embedder-projected
/// states `cond` [1,L,2560].
fn buildCrossKv(e: *const Engine, allocator: std.mem.Allocator, cond: mlx.mlx_array, s: S) !CrossKv {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const nkv: c_int = @intCast(e.cfg.kv_heads);
    const hd: c_int = @intCast(e.cfg.head_dim);

    var ks = try allocator.alloc(mlx.mlx_array, e.cfg.layers);
    var vs = try allocator.alloc(mlx.mlx_array, e.cfg.layers);
    var done: usize = 0;
    errdefer {
        for (0..done) |i| {
            _ = mlx.mlx_array_free(ks[i]);
            _ = mlx.mlx_array_free(vs[i]);
        }
        allocator.free(ks);
        allocator.free(vs);
    }
    var li: u32 = 0;
    while (li < e.cfg.layers) : (li += 1) {
        const pfx = try std.fmt.allocPrint(a, "decoder.layers.{d}.cross_attn", .{li});
        const k = try lin(&e.w, a, cond, try std.fmt.allocPrint(a, "{s}.k_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(k);
        const kh = try splitHeads(k, nkv, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const kn_w = try getW(&e.w, try std.fmt.allocPrint(a, "{s}.k_norm.weight", .{pfx}));
        ks[li] = try rmsNorm(kh, kn_w, e.cfg.eps, s);
        const v = try lin(&e.w, a, cond, try std.fmt.allocPrint(a, "{s}.v_proj", .{pfx}), s);
        defer _ = mlx.mlx_array_free(v);
        vs[li] = try splitHeads(v, nkv, hd, s);
        done += 1;
        _ = mlx.mlx_array_eval(ks[li]);
        _ = arena_inst.reset(.retain_capacity);
    }
    return .{ .ks = ks, .vs = vs };
}

/// One DiT layer. `proj6` is the shared [1,6,D] timestep projection; each layer
/// adds its own scale_shift_table. AdaLN wraps self-attn and MLP; cross-attn is
/// a plain residual (no AdaLN, no RoPE).
fn ditLayer(e: *const Engine, a: std.mem.Allocator, li: u32, h_in: mlx.mlx_array, proj6: mlx.mlx_array, sliding_mask: ?mlx.mlx_array, cross_k: mlx.mlx_array, cross_v: mlx.mlx_array, s: S) !mlx.mlx_array {
    const w = &e.w;
    const cfg = e.cfg;
    const nh: c_int = @intCast(cfg.heads);
    const nkv: c_int = @intCast(cfg.kv_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const d: c_int = @intCast(cfg.hidden);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const pfx = try std.fmt.allocPrint(a, "decoder.layers.{d}", .{li});

    // mod = scale_shift_table + proj6 → 6× [1,1,D]
    const tbl = try getW(w, try std.fmt.allocPrint(a, "{s}.scale_shift_table", .{pfx}));
    const mod6 = try addA(tbl, proj6, s);
    defer _ = mlx.mlx_array_free(mod6);
    var chunks: [6]mlx.mlx_array = undefined;
    for (0..6) |ci| {
        chunks[ci] = try sliceA(mod6, &[_]c_int{ 0, @intCast(ci), 0 }, &[_]c_int{ 1, @intCast(ci + 1), d }, &[_]c_int{ 1, 1, 1 }, s);
    }
    defer for (0..6) |ci| {
        _ = mlx.mlx_array_free(chunks[ci]);
    };
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);

    // ── self-attention with AdaLN ──
    const sa_norm_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn_norm.weight", .{pfx}));
    const n0 = try rmsNorm(h_in, sa_norm_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(n0);
    const sc1 = try addA(chunks[1], one, s); // 1 + scale_msa
    defer _ = mlx.mlx_array_free(sc1);
    const n1 = try mulA(n0, sc1, s);
    defer _ = mlx.mlx_array_free(n1);
    const nx = try addA(n1, chunks[0], s); // + shift_msa
    defer _ = mlx.mlx_array_free(nx);

    const q = try lin(w, a, nx, try std.fmt.allocPrint(a, "{s}.self_attn.q_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(q);
    const k = try lin(w, a, nx, try std.fmt.allocPrint(a, "{s}.self_attn.k_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(k);
    const v = try lin(w, a, nx, try std.fmt.allocPrint(a, "{s}.self_attn.v_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(v);
    const qh = try splitHeads(q, nh, hd, s);
    defer _ = mlx.mlx_array_free(qh);
    const kh = try splitHeads(k, nkv, hd, s);
    defer _ = mlx.mlx_array_free(kh);
    const vh = try splitHeads(v, nkv, hd, s);
    defer _ = mlx.mlx_array_free(vh);
    const qn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn.q_norm.weight", .{pfx}));
    const qn = try rmsNorm(qh, qn_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const kn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.self_attn.k_norm.weight", .{pfx}));
    const kn = try rmsNorm(kh, kn_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(kn);
    const qr = try ropeInPlace(qn, hd, cfg.rope_theta, s);
    defer _ = mlx.mlx_array_free(qr);
    const kr = try ropeInPlace(kn, hd, cfg.rope_theta, s);
    defer _ = mlx.mlx_array_free(kr);
    const attn = if (sliding_mask) |m|
        try sdpa(qr, kr, vh, scale, m, "array", s)
    else
        try sdpa(qr, kr, vh, scale, null, "", s);
    defer _ = mlx.mlx_array_free(attn);
    const merged = try mergeHeads(attn, s);
    defer _ = mlx.mlx_array_free(merged);
    const sa_out = try lin(w, a, merged, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(sa_out);
    const gated = try mulA(sa_out, chunks[2], s); // × gate_msa
    defer _ = mlx.mlx_array_free(gated);
    const h1 = try addA(h_in, gated, s);
    defer _ = mlx.mlx_array_free(h1);

    // ── cross-attention (plain residual) ──
    const ca_norm_w = try getW(w, try std.fmt.allocPrint(a, "{s}.cross_attn_norm.weight", .{pfx}));
    const cn = try rmsNorm(h1, ca_norm_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(cn);
    const cq = try lin(w, a, cn, try std.fmt.allocPrint(a, "{s}.cross_attn.q_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(cq);
    const cqh = try splitHeads(cq, nh, hd, s);
    defer _ = mlx.mlx_array_free(cqh);
    const cqn_w = try getW(w, try std.fmt.allocPrint(a, "{s}.cross_attn.q_norm.weight", .{pfx}));
    const cqn = try rmsNorm(cqh, cqn_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(cqn);
    const cattn = try sdpa(cqn, cross_k, cross_v, scale, null, "", s);
    defer _ = mlx.mlx_array_free(cattn);
    const cmerged = try mergeHeads(cattn, s);
    defer _ = mlx.mlx_array_free(cmerged);
    const ca_out = try lin(w, a, cmerged, try std.fmt.allocPrint(a, "{s}.cross_attn.o_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(ca_out);
    const h2 = try addA(h1, ca_out, s);
    defer _ = mlx.mlx_array_free(h2);

    // ── MLP with AdaLN ──
    const mlp_norm_w = try getW(w, try std.fmt.allocPrint(a, "{s}.mlp_norm.weight", .{pfx}));
    const m0 = try rmsNorm(h2, mlp_norm_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(m0);
    const sc2 = try addA(chunks[4], one, s); // 1 + c_scale_msa
    defer _ = mlx.mlx_array_free(sc2);
    const m1 = try mulA(m0, sc2, s);
    defer _ = mlx.mlx_array_free(m1);
    const mx_in = try addA(m1, chunks[3], s); // + c_shift_msa
    defer _ = mlx.mlx_array_free(mx_in);
    const gate = try lin(w, a, mx_in, try std.fmt.allocPrint(a, "{s}.mlp.gate_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try lin(w, a, mx_in, try std.fmt.allocPrint(a, "{s}.mlp.up_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(up);
    const gact = try silu(gate, s);
    defer _ = mlx.mlx_array_free(gact);
    const gu = try mulA(gact, up, s);
    defer _ = mlx.mlx_array_free(gu);
    const down = try lin(w, a, gu, try std.fmt.allocPrint(a, "{s}.mlp.down_proj", .{pfx}), s);
    defer _ = mlx.mlx_array_free(down);
    const fgated = try mulA(down, chunks[5], s); // × c_gate_msa
    defer _ = mlx.mlx_array_free(fgated);
    return addA(h2, fgated, s);
}

/// Full DiT forward: xt [1,T,64] f32 + context [1,T,128] f32 → velocity
/// [1,T,64] f32. `cross` carries the per-generation cross K/V; `t_r − t = 0`
/// at inference so time_embed_r always sees 0.0 (its MLP output is nonzero).
fn ditForward(e: *const Engine, allocator: std.mem.Allocator, xt_f32: mlx.mlx_array, t_val: f32, ctx_f32: mlx.mlx_array, cross: *const CrossKv, s: S) !mlx.mlx_array {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const cfg = e.cfg;
    const d: c_int = @intCast(cfg.hidden);

    // Timesteps: temb = temb_t + temb_r, proj = proj_t + proj_r.
    const te_t = try timestepEmbed(e, a, "time_embed", t_val, s);
    defer _ = mlx.mlx_array_free(te_t.temb);
    defer _ = mlx.mlx_array_free(te_t.proj);
    const te_r = try timestepEmbed(e, a, "time_embed_r", 0.0, s);
    defer _ = mlx.mlx_array_free(te_r.temb);
    defer _ = mlx.mlx_array_free(te_r.proj);
    const temb = try addA(te_t.temb, te_r.temb, s);
    defer _ = mlx.mlx_array_free(temb);
    const proj6 = try addA(te_t.proj, te_r.proj, s);
    defer _ = mlx.mlx_array_free(proj6);

    // Input: concat context + noisy latents → [1,T,192] bf16, pad to patch.
    const cat = try concat2(ctx_f32, xt_f32, 2, s);
    defer _ = mlx.mlx_array_free(cat);
    const cat_bf = try astype(cat, .bfloat16, s);
    defer _ = mlx.mlx_array_free(cat_bf);
    const orig_t: c_int = mlx.getShape(cat_bf)[1];
    var padded = cat_bf;
    var padded_owned = false;
    defer if (padded_owned) {
        _ = mlx.mlx_array_free(padded);
    };
    if (@mod(orig_t, @as(c_int, @intCast(cfg.patch_size))) != 0) {
        const axes = [_]c_int{1};
        const lo = [_]c_int{0};
        const hi = [_]c_int{@as(c_int, @intCast(cfg.patch_size)) - @mod(orig_t, @as(c_int, @intCast(cfg.patch_size)))};
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var p = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(&p, cat_bf, &axes, 1, &lo, 1, &hi, 1, zero, "constant", s));
        padded = p;
        padded_owned = true;
    }

    // proj_in: Conv1d k=2 s=2 (192→2560), + bias.
    const pi_w = try getW(&e.w, "decoder.proj_in.weight");
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&h, padded, pi_w, @intCast(cfg.patch_size), 0, 1, 1, s));
    const pi_b = try getW(&e.w, "decoder.proj_in.bias");
    {
        const hb = try addA(h, pi_b, s);
        _ = mlx.mlx_array_free(h);
        h = hb;
    }

    const tp: c_int = mlx.getShape(h)[1];
    const band = try slidingBandMask(tp, @intCast(cfg.sliding_window), s);
    defer _ = mlx.mlx_array_free(band);

    var li: u32 = 0;
    while (li < cfg.layers) : (li += 1) {
        const mask: ?mlx.mlx_array = if (li % 2 == 0) band else null;
        const nh = try ditLayer(e, a, li, h, proj6, mask, cross.ks[li], cross.vs[li], s);
        _ = mlx.mlx_array_free(h);
        h = nh;
        _ = mlx.mlx_array_eval(h);
        _ = arena_inst.reset(.retain_capacity);
    }

    // Output AdaLN uses temb (2-entry table), then de-patchify.
    const out_tbl = try getW(&e.w, "decoder.scale_shift_table");
    const temb3 = try reshape(temb, &[_]c_int{ 1, 1, d }, s);
    defer _ = mlx.mlx_array_free(temb3);
    const mod2 = try addA(out_tbl, temb3, s);
    defer _ = mlx.mlx_array_free(mod2);
    const shift = try sliceA(mod2, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, 1, d }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(shift);
    const scale2 = try sliceA(mod2, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ 1, 2, d }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(scale2);
    const no_w = try getW(&e.w, "decoder.norm_out.weight");
    const hn = try rmsNorm(h, no_w, cfg.eps, s);
    _ = mlx.mlx_array_free(h);
    defer _ = mlx.mlx_array_free(hn);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const sc1 = try addA(scale2, one, s);
    defer _ = mlx.mlx_array_free(sc1);
    const hm = try mulA(hn, sc1, s);
    defer _ = mlx.mlx_array_free(hm);
    const hs = try addA(hm, shift, s);
    defer _ = mlx.mlx_array_free(hs);

    // proj_out: ConvTranspose1d k=2 s=2 (2560→64) + bias, crop, cast f32.
    const po_w = try getW(&e.w, "decoder.proj_out.weight");
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv_transpose1d(&out, hs, po_w, @intCast(cfg.patch_size), 0, 1, 0, 1, s));
    defer _ = mlx.mlx_array_free(out);
    const po_b = try getW(&e.w, "decoder.proj_out.bias");
    const outb = try addA(out, po_b, s);
    defer _ = mlx.mlx_array_free(outb);
    const ad: c_int = @intCast(cfg.acoustic_dim);
    const cropped = try sliceA(outb, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, orig_t, ad }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(cropped);
    return astype(cropped, .float32, s);
}

// ════════════════════════════════════════════════════════════════════════
// DCW — Differential Correction in Wavelet domain (haar, "double" mode; the
// pipeline's shipping default). Single-level Haar DWT along T with zero-pad
// on odd lengths; low band pushed by t·0.05, high band by (1−t)·0.02.
// ════════════════════════════════════════════════════════════════════════

const HaarBands = struct { low: mlx.mlx_array, high: mlx.mlx_array };

fn haarDwt(x: mlx.mlx_array, s: S) !HaarBands {
    const sh = mlx.getShape(x); // [B,T,C]
    var t_len = sh[1];
    var src = x;
    var src_owned = false;
    defer if (src_owned) {
        _ = mlx.mlx_array_free(src);
    };
    if (@mod(t_len, 2) == 1) {
        const axes = [_]c_int{1};
        const lo = [_]c_int{0};
        const hi = [_]c_int{1};
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var p = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(&p, x, &axes, 1, &lo, 1, &hi, 1, zero, "constant", s));
        src = p;
        src_owned = true;
        t_len += 1;
    }
    const even = try sliceA(src, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ sh[0], t_len, sh[2] }, &[_]c_int{ 1, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(even);
    const odd = try sliceA(src, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ sh[0], t_len, sh[2] }, &[_]c_int{ 1, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(odd);
    const inv = 1.0 / @sqrt(@as(f32, 2.0));
    const sum = try addA(even, odd, s);
    defer _ = mlx.mlx_array_free(sum);
    const diff = try subA(even, odd, s);
    defer _ = mlx.mlx_array_free(diff);
    return .{ .low = try mulScalar(sum, inv, s), .high = try mulScalar(diff, inv, s) };
}

fn haarIdwt(low: mlx.mlx_array, high: mlx.mlx_array, out_t: c_int, s: S) !mlx.mlx_array {
    const inv = 1.0 / @sqrt(@as(f32, 2.0));
    const sum = try addA(low, high, s);
    defer _ = mlx.mlx_array_free(sum);
    const diff = try subA(low, high, s);
    defer _ = mlx.mlx_array_free(diff);
    const even = try mulScalar(sum, inv, s);
    defer _ = mlx.mlx_array_free(even);
    const odd = try mulScalar(diff, inv, s);
    defer _ = mlx.mlx_array_free(odd);
    // interleave: [B,Th,C] × 2 → [B,Th,2,C] → [B,2Th,C] → crop
    var ee = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ee);
    try mlx.check(mlx.mlx_expand_dims(&ee, even, 2, s));
    var oe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(oe);
    try mlx.check(mlx.mlx_expand_dims(&oe, odd, 2, s));
    const st = try concat2(ee, oe, 2, s);
    defer _ = mlx.mlx_array_free(st);
    const sh = mlx.getShape(st); // [B,Th,2,C]
    const flat = try reshape(st, &[_]c_int{ sh[0], sh[1] * 2, sh[3] }, s);
    defer _ = mlx.mlx_array_free(flat);
    return sliceA(flat, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ sh[0], out_t, sh[3] }, &[_]c_int{ 1, 1, 1 }, s);
}

/// x_next ← DCW(x_next, denoised, t): per-band push away from the predicted
/// clean sample. Identity when both effective scalers are 0.
fn applyDcwDouble(x_next: mlx.mlx_array, denoised: mlx.mlx_array, t_curr: f32, s: S) !mlx.mlx_array {
    const low_s = t_curr * DCW_LOW;
    const high_s = (1.0 - t_curr) * DCW_HIGH;
    const out_t = mlx.getShape(x_next)[1];
    const xb = try haarDwt(x_next, s);
    defer _ = mlx.mlx_array_free(xb.low);
    defer _ = mlx.mlx_array_free(xb.high);
    const yb = try haarDwt(denoised, s);
    defer _ = mlx.mlx_array_free(yb.low);
    defer _ = mlx.mlx_array_free(yb.high);

    var xl = xb.low;
    var xl_owned = false;
    defer if (xl_owned) {
        _ = mlx.mlx_array_free(xl);
    };
    if (low_s != 0.0) {
        const d = try subA(xb.low, yb.low, s);
        defer _ = mlx.mlx_array_free(d);
        const ds = try mulScalar(d, low_s, s);
        defer _ = mlx.mlx_array_free(ds);
        xl = try addA(xb.low, ds, s);
        xl_owned = true;
    }
    var xh = xb.high;
    var xh_owned = false;
    defer if (xh_owned) {
        _ = mlx.mlx_array_free(xh);
    };
    if (high_s != 0.0) {
        const d = try subA(xb.high, yb.high, s);
        defer _ = mlx.mlx_array_free(d);
        const ds = try mulScalar(d, high_s, s);
        defer _ = mlx.mlx_array_free(ds);
        xh = try addA(xb.high, ds, s);
        xh_owned = true;
    }
    return haarIdwt(xl, xh, out_t, s);
}

// ════════════════════════════════════════════════════════════════════════
// AutoencoderOobleck VAE (Snake activations f32; convs bf16).
// ════════════════════════════════════════════════════════════════════════

const VAE_STRIDES_DOWN = [_]u32{ 2, 4, 4, 6, 10 };
const VAE_CM = [_]u32{ 1, 1, 2, 4, 8, 16 }; // [1] ++ channel_multiples

/// Snake: x + (1/exp(β))·sin²(exp(α)·x), computed f32 (exp headroom), result
/// cast back to x's dtype. α/β stored [C] f32 by the converter.
fn snake(e: *const Engine, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, s: S) !mlx.mlx_array {
    const alpha = try getW(&e.vae_w, try std.fmt.allocPrint(a, "{s}.alpha", .{prefix}));
    const beta = try getW(&e.vae_w, try std.fmt.allocPrint(a, "{s}.beta", .{prefix}));
    const in_dtype = mlx.mlx_array_dtype(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var ea = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ea);
    try mlx.check(mlx.mlx_exp(&ea, alpha, s));
    var eb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eb);
    try mlx.check(mlx.mlx_exp(&eb, beta, s));
    const ax = try mulA(xf, ea, s);
    defer _ = mlx.mlx_array_free(ax);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, ax, s));
    const sq = try mulA(sn, sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps_c = mlx.mlx_array_new_float(1e-9);
    defer _ = mlx.mlx_array_free(eps_c);
    const beps = try addA(eb, eps_c, s);
    defer _ = mlx.mlx_array_free(beps);
    var recip = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(recip);
    try mlx.check(mlx.mlx_divide(&recip, sq, beps, s));
    const out_f = try addA(xf, recip, s);
    defer _ = mlx.mlx_array_free(out_f);
    return astype(out_f, in_dtype, s);
}

fn vaeConv(e: *const Engine, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, stride: c_int, padding: c_int, dilation: c_int, s: S) !mlx.mlx_array {
    const w = try getW(&e.vae_w, try std.fmt.allocPrint(a, "{s}.weight", .{prefix}));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&o, x, w, stride, padding, dilation, 1, s));
    if (e.vae_w.get(try std.fmt.allocPrint(a, "{s}.bias", .{prefix}))) |b| {
        const ob = try addA(o, b, s);
        _ = mlx.mlx_array_free(o);
        o = ob;
    }
    return o;
}

fn vaeConvT(e: *const Engine, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, stride: c_int, padding: c_int, s: S) !mlx.mlx_array {
    const w = try getW(&e.vae_w, try std.fmt.allocPrint(a, "{s}.weight", .{prefix}));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv_transpose1d(&o, x, w, stride, padding, 1, 0, 1, s));
    if (e.vae_w.get(try std.fmt.allocPrint(a, "{s}.bias", .{prefix}))) |b| {
        const ob = try addA(o, b, s);
        _ = mlx.mlx_array_free(o);
        o = ob;
    }
    return o;
}

/// Residual unit: snake1 → conv1 (k7, dilated) → snake2 → conv2 (k1) → +x.
fn vaeResUnit(e: *const Engine, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, dilation: c_int, s: S) !mlx.mlx_array {
    const pad = @divTrunc((7 - 1) * dilation, 2);
    const s1 = try snake(e, a, x, try std.fmt.allocPrint(a, "{s}.snake1", .{prefix}), s);
    defer _ = mlx.mlx_array_free(s1);
    const c1 = try vaeConv(e, a, s1, try std.fmt.allocPrint(a, "{s}.conv1", .{prefix}), 1, pad, dilation, s);
    defer _ = mlx.mlx_array_free(c1);
    const s2 = try snake(e, a, c1, try std.fmt.allocPrint(a, "{s}.snake2", .{prefix}), s);
    defer _ = mlx.mlx_array_free(s2);
    const c2 = try vaeConv(e, a, s2, try std.fmt.allocPrint(a, "{s}.conv2", .{prefix}), 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(c2);
    return addA(x, c2, s);
}

/// Decode one latent window [1,Tw,64] bf16 → audio [1,Tw*1920,2] bf16.
fn vaeDecodeWindow(e: *const Engine, allocator: std.mem.Allocator, latents: mlx.mlx_array, s: S) !mlx.mlx_array {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    var h = try vaeConv(e, a, latents, "decoder.conv1", 1, 3, 1, s);
    // 5 upsample blocks, strides reversed([2,4,4,6,10]) = [10,6,4,4,2].
    var bi: usize = 0;
    while (bi < 5) : (bi += 1) {
        const stride: c_int = @intCast(VAE_STRIDES_DOWN[4 - bi]);
        const pad: c_int = @intCast((VAE_STRIDES_DOWN[4 - bi] + 1) / 2);
        const pfx = try std.fmt.allocPrint(a, "decoder.block.{d}", .{bi});
        const sn0 = try snake(e, a, h, try std.fmt.allocPrint(a, "{s}.snake1", .{pfx}), s);
        _ = mlx.mlx_array_free(h);
        const up = try vaeConvT(e, a, sn0, try std.fmt.allocPrint(a, "{s}.conv_t1", .{pfx}), stride, pad, s);
        _ = mlx.mlx_array_free(sn0);
        const r1 = try vaeResUnit(e, a, up, try std.fmt.allocPrint(a, "{s}.res_unit1", .{pfx}), 1, s);
        _ = mlx.mlx_array_free(up);
        const r2 = try vaeResUnit(e, a, r1, try std.fmt.allocPrint(a, "{s}.res_unit2", .{pfx}), 3, s);
        _ = mlx.mlx_array_free(r1);
        const r3 = try vaeResUnit(e, a, r2, try std.fmt.allocPrint(a, "{s}.res_unit3", .{pfx}), 9, s);
        _ = mlx.mlx_array_free(r2);
        h = r3;
        _ = mlx.mlx_array_eval(h);
        _ = arena_inst.reset(.retain_capacity);
    }
    const sn = try snake(e, a, h, "decoder.snake1", s);
    _ = mlx.mlx_array_free(h);
    const out = try vaeConv(e, a, sn, "decoder.conv2", 1, 3, 1, s);
    _ = mlx.mlx_array_free(sn);
    return out;
}

/// Chunked decode: latents [1,T,64] f32 → interleaved stereo f32 samples
/// (owned). Overlap-window strategy mirrors the reference tiled decode:
/// decode [core−ov .. core+ov], trim the padded sides, concat cores.
fn vaeDecodeChunked(e: *const Engine, allocator: std.mem.Allocator, latents: mlx.mlx_array, progress: ?sse.Progress, s: S) ![]f32 {
    const sh = mlx.getShape(latents);
    const total: usize = @intCast(sh[1]);
    const hop: usize = @intCast(e.cfg.vae_hop);
    const lat_bf = try astype(latents, .bfloat16, s);
    defer _ = mlx.mlx_array_free(lat_bf);

    var out = try allocator.alloc(f32, total * hop * 2);
    errdefer allocator.free(out);
    var write_frame: usize = 0;

    const chunk_frames = vaeChunkFrames(e.low_mem);
    const n_chunks = std.math.divCeil(usize, total, chunk_frames) catch unreachable;
    var ci: usize = 0;
    while (ci < n_chunks) : (ci += 1) {
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("decode", @intCast(ci), @intCast(n_chunks));
        }
        const core_start = ci * chunk_frames;
        const core_end = @min(core_start + chunk_frames, total);
        const win_start = core_start -| VAE_OVERLAP_FRAMES;
        const win_end = @min(core_end + VAE_OVERLAP_FRAMES, total);

        const win = try sliceA(lat_bf, &[_]c_int{ 0, @intCast(win_start), 0 }, &[_]c_int{ 1, @intCast(win_end), sh[2] }, &[_]c_int{ 1, 1, 1 }, s);
        defer _ = mlx.mlx_array_free(win);
        const audio = try vaeDecodeWindow(e, allocator, win, s);
        defer _ = mlx.mlx_array_free(audio);
        const af = try astype(audio, .float32, s);
        defer _ = mlx.mlx_array_free(af);
        var ac = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ac);
        try mlx.check(mlx.mlx_contiguous(&ac, af, false, s));
        _ = mlx.mlx_array_eval(ac);
        const data = mlx.mlx_array_data_float32(ac) orelse return error.NoData;

        // audio is [1, Twin*hop, 2] channels-last → already interleaved.
        const trim_start = (core_start - win_start) * hop;
        const core_frames = (core_end - core_start) * hop;
        const src = data[trim_start * 2 .. (trim_start + core_frames) * 2];
        @memcpy(out[write_frame * 2 .. (write_frame + core_frames / hop * hop) * 2], src);
        write_frame += core_frames;
        // Return each window's upsampling buffers before decoding the next.
        if (e.low_mem) _ = mlx.mlx_clear_cache();
    }
    if (progress) |p| p.emit("decode", @intCast(n_chunks), @intCast(n_chunks));
    std.debug.assert(write_frame == total * hop);
    return out;
}

/// VAE encoder → latent MEAN [1,T,64] f32 (M3 reference-audio path + oracle 6).
/// Input audio [1,N,2] channels-last F32 (callers pass decoded samples
/// directly — a bf16 round-trip of the raw SIGNAL alone costs cos 0.9999 →
/// 0.958 on the latent mean; 8 mantissa bits are not enough for audio).
/// Compute also stays f32 end-to-end; reference clips are short
/// (≤ timbre_fix_frame), so f32 residency is cheap.
pub fn vaeEncodeMean(e: *const Engine, allocator: std.mem.Allocator, audio: mlx.mlx_array, s: S) !mlx.mlx_array {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    const audio_f = try astype(audio, .float32, s);
    defer _ = mlx.mlx_array_free(audio_f);
    var h = try vaeConv(e, a, audio_f, "encoder.conv1", 1, 3, 1, s);
    var bi: usize = 0;
    while (bi < 5) : (bi += 1) {
        const stride: c_int = @intCast(VAE_STRIDES_DOWN[bi]);
        const pad: c_int = @intCast((VAE_STRIDES_DOWN[bi] + 1) / 2);
        const pfx = try std.fmt.allocPrint(a, "encoder.block.{d}", .{bi});
        const r1 = try vaeResUnit(e, a, h, try std.fmt.allocPrint(a, "{s}.res_unit1", .{pfx}), 1, s);
        _ = mlx.mlx_array_free(h);
        const r2 = try vaeResUnit(e, a, r1, try std.fmt.allocPrint(a, "{s}.res_unit2", .{pfx}), 3, s);
        _ = mlx.mlx_array_free(r1);
        const r3 = try vaeResUnit(e, a, r2, try std.fmt.allocPrint(a, "{s}.res_unit3", .{pfx}), 9, s);
        _ = mlx.mlx_array_free(r2);
        const sn0 = try snake(e, a, r3, try std.fmt.allocPrint(a, "{s}.snake1", .{pfx}), s);
        _ = mlx.mlx_array_free(r3);
        const dn = try vaeConv(e, a, sn0, try std.fmt.allocPrint(a, "{s}.conv1", .{pfx}), stride, pad, 1, s);
        _ = mlx.mlx_array_free(sn0);
        h = dn;
        _ = mlx.mlx_array_eval(h);
        _ = arena_inst.reset(.retain_capacity);
    }
    const sn = try snake(e, a, h, "encoder.snake1", s);
    _ = mlx.mlx_array_free(h);
    const out = try vaeConv(e, a, sn, "encoder.conv2", 1, 1, 1, s);
    _ = mlx.mlx_array_free(sn);
    defer _ = mlx.mlx_array_free(out);
    // Diagonal Gaussian: first half = mean, second half = log-scale.
    const osh = mlx.getShape(out); // [1,T,128]
    const half = @divExact(osh[2], 2);
    return sliceA(out, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, osh[1], half }, &[_]c_int{ 1, 1, 1 }, s);
}

/// Full-length encode (the cover/complete SOURCE): `[1, frames*hop, 2]` f32 →
/// latent mean `[1, frames, 64]` f32 in 30 s windows — one untiled pass at
/// 600 s holds 128 ch × 28.8M samples f32 (15 GB) per activation. Windows
/// start on latent boundaries so every strided conv lands where the untiled
/// pass would; the 64-frame halo (≫ the encoder's ~12-frame receptive field)
/// absorbs the zero-padded edges. The decode's overlap strategy, mirrored.
pub fn vaeEncodeMeanChunked(e: *const Engine, allocator: std.mem.Allocator, audio: mlx.mlx_array, progress: ?sse.Progress, s: S) !mlx.mlx_array {
    const hop: usize = @intCast(e.cfg.vae_hop);
    const total: usize = @as(usize, @intCast(mlx.getShape(audio)[1])) / hop;
    if (total <= VAE_ENC_CHUNK_FRAMES) return vaeEncodeMean(e, allocator, audio, s);
    const n_chunks = std.math.divCeil(usize, total, VAE_ENC_CHUNK_FRAMES) catch unreachable;
    var acc: ?mlx.mlx_array = null;
    errdefer if (acc) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var ci: usize = 0;
    while (ci < n_chunks) : (ci += 1) {
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("encode", @intCast(ci), @intCast(n_chunks));
        }
        const core_start = ci * VAE_ENC_CHUNK_FRAMES;
        const core_end = @min(core_start + VAE_ENC_CHUNK_FRAMES, total);
        const win_start = core_start -| VAE_OVERLAP_FRAMES;
        const win_end = @min(core_end + VAE_OVERLAP_FRAMES, total);
        const win = try sliceA(audio, &[_]c_int{ 0, @intCast(win_start * hop), 0 }, &[_]c_int{ 1, @intCast(win_end * hop), 2 }, &[_]c_int{ 1, 1, 1 }, s);
        defer _ = mlx.mlx_array_free(win);
        const m = try vaeEncodeMean(e, allocator, win, s);
        defer _ = mlx.mlx_array_free(m);
        const core = try sliceA(m, &[_]c_int{ 0, @intCast(core_start - win_start), 0 }, &[_]c_int{ 1, @intCast(core_end - win_start), 64 }, &[_]c_int{ 1, 1, 1 }, s);
        defer _ = mlx.mlx_array_free(core);
        if (acc) |prev| {
            const next = try concat2(prev, core, 1, s);
            _ = mlx.mlx_array_free(prev);
            acc = next;
        } else {
            var own = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&own, core));
            acc = own;
        }
        _ = mlx.mlx_array_eval(acc.?);
        if (e.low_mem) _ = mlx.mlx_clear_cache();
    }
    return acc.?;
}

// ════════════════════════════════════════════════════════════════════════
// FSQ audio tokenizer + detokenizer (cover mode). Reference: AceStepAudioTokenizer
// (acoustic proj → AttentionPooler → ResidualFSQ, 1 quantizer so scale = 1) and
// AudioTokenDetokenizer. vector-quantize-pytorch's ResidualFSQ SOFT-CLAMPS its
// input, z' = c·tanh(z/c) with c = L/(L−1) per dim (`soft_clamp_input_value`,
// defaulted from the levels), then builds its FSQ with preserve_symmetry +
// bound_hard_clamp, so the grid is
//   QL(z') = 2/(L−1) · ⌊(L−1)(clamp(z',−1,1)+1)/2 + 0.5⌋ − 1
// — NOT the tanh `bound()` of a bare FSQ, and not the hard clamp alone either
// (that reading flipped the reference's level at z=−0.787, L=5). Quantization
// runs f32 (the reference forces it; a bf16 ulp at a cell edge flips a code).
// ════════════════════════════════════════════════════════════════════════

fn levelsArray(levels: []const f32) mlx.mlx_array {
    const sh = [_]c_int{@intCast(levels.len)};
    return mlx.mlx_array_new_data(levels.ptr, &sh, 1, .float32);
}

/// The ResidualFSQ input soft clamp, per dim: c·tanh(z/c), c = L/(L−1).
fn fsqSoftClamp(z: f32, level: f32) f32 {
    const c = level / (level - 1.0);
    return c * std.math.tanh(z / c);
}

/// z [..., d] f32 → codes on the symmetric grid, same shape (values in [−1, 1]).
fn fsqQuantize(z: mlx.mlx_array, levels: []const f32, s: S) !mlx.mlx_array {
    const lv = levelsArray(levels);
    defer _ = mlx.mlx_array_free(lv);
    const neg1 = mlx.mlx_array_new_float(-1.0);
    defer _ = mlx.mlx_array_free(neg1);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const lm1 = try subA(lv, one, s);
    defer _ = mlx.mlx_array_free(lm1);
    // soft clamp: c·tanh(z/c)
    var cl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cl);
    try mlx.check(mlx.mlx_divide(&cl, lv, lm1, s));
    var zdc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zdc);
    try mlx.check(mlx.mlx_divide(&zdc, z, cl, s));
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, zdc, s));
    const zs = try mulA(th, cl, s);
    defer _ = mlx.mlx_array_free(zs);
    var zc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zc);
    try mlx.check(mlx.mlx_clip(&zc, zs, neg1, one, s));
    const zp1 = try addA(zc, one, s);
    defer _ = mlx.mlx_array_free(zp1);
    const prod = try mulA(zp1, lm1, s);
    defer _ = mlx.mlx_array_free(prod);
    const halved = try mulScalar(prod, 0.5, s);
    defer _ = mlx.mlx_array_free(halved);
    const shifted = try addA(halved, half, s);
    defer _ = mlx.mlx_array_free(shifted);
    var fl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fl);
    try mlx.check(mlx.mlx_floor(&fl, shifted, s));
    var scale = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scale);
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    try mlx.check(mlx.mlx_divide(&scale, two, lm1, s));
    const scaled = try mulA(fl, scale, s);
    defer _ = mlx.mlx_array_free(scaled);
    return subA(scaled, one, s);
}

/// codes [..., d] → codebook indices [...] int32 (FSQ.codes_to_indices:
/// per-level index (code+1)(L−1)/2, mixed-radix with basis cumprod([1] ++ L[:-1])).
fn fsqIndices(codes: mlx.mlx_array, levels: []const f32, s: S) !mlx.mlx_array {
    var basis_buf: [16]f32 = undefined;
    var b: f32 = 1.0;
    for (levels, 0..) |l, i| {
        basis_buf[i] = b;
        b *= l;
    }
    const basis = levelsArray(basis_buf[0..levels.len]);
    defer _ = mlx.mlx_array_free(basis);
    const lv = levelsArray(levels);
    defer _ = mlx.mlx_array_free(lv);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const lm1 = try subA(lv, one, s);
    defer _ = mlx.mlx_array_free(lm1);
    const cp1 = try addA(codes, one, s);
    defer _ = mlx.mlx_array_free(cp1);
    const lvl = try mulA(cp1, lm1, s);
    defer _ = mlx.mlx_array_free(lvl);
    const lvl_h = try mulScalar(lvl, 0.5, s);
    defer _ = mlx.mlx_array_free(lvl_h);
    const weighted = try mulA(lvl_h, basis, s);
    defer _ = mlx.mlx_array_free(weighted);
    var summed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(summed);
    try mlx.check(mlx.mlx_sum_axis(&summed, weighted, -1, false, s));
    var rounded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rounded);
    try mlx.check(mlx.mlx_round(&rounded, summed, 0, s));
    return astype(rounded, .int32, s);
}

const AudioTokens = struct {
    hints: mlx.mlx_array,
    indices: mlx.mlx_array,
    /// Pre-quantization values [1,T5,6] f32 (the parity oracle's per-dim probe).
    z: mlx.mlx_array,
    fn deinit(self: *const AudioTokens) void {
        _ = mlx.mlx_array_free(self.hints);
        _ = mlx.mlx_array_free(self.indices);
        _ = mlx.mlx_array_free(self.z);
    }
};

/// Source latents [1,T,64] → 5 Hz FSQ hints [1,⌈T/5⌉,2048] f32 + codebook
/// indices [1,⌈T/5⌉] i32 (`AceStepConditionGenerationModel.tokenize`: pad T
/// to ×5 with the silence latent, acoustic proj, CLS-prepended 6-token windows
/// through the pooler, RMSNorm, CLS row → ResidualFSQ).
fn audioTokenize(self: *Engine, allocator: std.mem.Allocator, latents: mlx.mlx_array, s: S) !AudioTokens {
    const fw = try self.ensureFsq();
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const cfg = self.cfg;
    const p: c_int = @intCast(cfg.pool_window);
    const ad: c_int = @intCast(cfg.acoustic_dim);
    const enc_h: c_int = @intCast(cfg.enc_hidden);

    // f32 activations through the pooler: the grid below is a floor, and bf16
    // attention moved a pre-quantization value 0.07 cells — enough to flip a
    // code the fp32 reference did not (both 8-bit and dense-bf16 weights
    // flipped the SAME dim). The stacks are tiny (T/5 windows of 6 tokens).
    const lat_f = try astype(latents, .float32, s);
    defer _ = mlx.mlx_array_free(lat_f);
    const t_len: c_int = mlx.getShape(lat_f)[1];
    var x = lat_f;
    var x_owned = false;
    defer if (x_owned) {
        _ = mlx.mlx_array_free(x);
    };
    if (@mod(t_len, p) != 0) {
        const pad_bf = try self.silenceSlice(@intCast(p - @mod(t_len, p)), s);
        defer _ = mlx.mlx_array_free(pad_bf);
        const pad = try astype(pad_bf, .float32, s);
        defer _ = mlx.mlx_array_free(pad);
        x = try concat2(lat_f, pad, 1, s);
        x_owned = true;
    }
    const t5: c_int = @divExact(mlx.getShape(x)[1], p);
    const windows = try reshape(x, &[_]c_int{ t5, p, ad }, s);
    defer _ = mlx.mlx_array_free(windows);
    const proj = try lin(fw, a, windows, "tokenizer.audio_acoustic_proj", s);
    defer _ = mlx.mlx_array_free(proj);
    const emb = try lin(fw, a, proj, "tokenizer.attention_pooler.embed_tokens", s);
    defer _ = mlx.mlx_array_free(emb);
    const cls_w = try getW(fw, "tokenizer.attention_pooler.special_token");
    const cls = try astype(cls_w, .float32, s);
    defer _ = mlx.mlx_array_free(cls);
    var cls_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cls_b);
    try mlx.check(mlx.mlx_broadcast_to(&cls_b, cls, &[_]c_int{ t5, 1, enc_h }, 3, s));
    const seq = try concat2(cls_b, emb, 1, s);
    defer _ = mlx.mlx_array_free(seq);
    const pooled = try encoderStack(self, allocator, fw, "tokenizer.attention_pooler", seq, cfg.pooler_layers, s);
    defer _ = mlx.mlx_array_free(pooled);
    const cls_out = try sliceA(pooled, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ t5, 1, enc_h }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(cls_out);
    const cls_f = try reshape(cls_out, &[_]c_int{ 1, t5, enc_h }, s);
    defer _ = mlx.mlx_array_free(cls_f);
    const z = try lin(fw, a, cls_f, "tokenizer.quantizer.project_in", s);
    errdefer _ = mlx.mlx_array_free(z);
    const codes = try fsqQuantize(z, &cfg.fsq_levels, s);
    defer _ = mlx.mlx_array_free(codes);
    const indices = try fsqIndices(codes, &cfg.fsq_levels, s);
    errdefer _ = mlx.mlx_array_free(indices);
    const hints = try lin(fw, a, codes, "tokenizer.quantizer.project_out", s); // f32
    errdefer _ = mlx.mlx_array_free(hints);
    _ = mlx.mlx_array_eval(hints);
    _ = mlx.mlx_array_eval(indices);
    _ = mlx.mlx_array_eval(z);
    return .{ .hints = hints, .indices = indices, .z = z };
}

/// 5 Hz hints [1,T5,2048] → 25 Hz latents [1,T5·5,64] f32
/// (AudioTokenDetokenizer: embed, ×5 + per-slot special tokens, 5-token
/// windows through the detokenizer stack, RMSNorm, proj_out).
fn audioDetokenize(self: *Engine, allocator: std.mem.Allocator, hints: mlx.mlx_array, s: S) !mlx.mlx_array {
    const fw = try self.ensureFsq();
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const cfg = self.cfg;
    const p: c_int = @intCast(cfg.pool_window);
    const enc_h: c_int = @intCast(cfg.enc_hidden);
    const t5: c_int = mlx.getShape(hints)[1];

    const hints_f = try astype(hints, .float32, s);
    defer _ = mlx.mlx_array_free(hints_f);
    const emb = try lin(fw, a, hints_f, "detokenizer.embed_tokens", s);
    defer _ = mlx.mlx_array_free(emb);
    const emb3 = try reshape(emb, &[_]c_int{ t5, 1, enc_h }, s);
    defer _ = mlx.mlx_array_free(emb3);
    const special_w = try getW(fw, "detokenizer.special_tokens"); // [1,5,2048]
    const special = try astype(special_w, .float32, s);
    defer _ = mlx.mlx_array_free(special);
    const x = try addA(emb3, special, s); // broadcast → [t5,5,2048]
    defer _ = mlx.mlx_array_free(x);
    const out = try encoderStack(self, allocator, fw, "detokenizer", x, cfg.pooler_layers, s);
    defer _ = mlx.mlx_array_free(out);
    const po = try lin(fw, a, out, "detokenizer.proj_out", s); // [t5,5,64]
    defer _ = mlx.mlx_array_free(po);
    const flat = try reshape(po, &[_]c_int{ 1, t5 * p, @intCast(cfg.acoustic_dim) }, s);
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, .float32, s);
}

// ════════════════════════════════════════════════════════════════════════
// Engine
// ════════════════════════════════════════════════════════════════════════

pub const MusicRequest = struct {
    caption: []const u8,
    lyrics: []const u8 = "",
    language: []const u8 = "en",
    bpm: ?u32 = null,
    keyscale: []const u8 = "",
    timesignature: []const u8 = "",
    duration_s: u32 = 60,
    seed: u64 = 0,
    /// Reference clip for style/timbre (#259): decoded samples `[1,N,2]` f32
    /// channels-last at 48 kHz, already windowed by `referenceWindow`. Fills
    /// the condition encoder's timbre slot INSTEAD of the silence latent —
    /// never both. Borrowed; the caller frees it.
    ref_audio: ?mlx.mlx_array = null,
    task: Task = .text2music,
    /// cover / complete source: decoded samples `[1,N,2]` f32 48 kHz, FULL
    /// length — its VAE latent IS the context stream and decides the track
    /// length (`duration_s` is ignored). Borrowed.
    src_audio: ?mlx.mlx_array = null,
    /// cover: fraction of the denoise steps that see the source codes (1 =
    /// all); the remaining steps switch to plain text2music conditioning
    /// (upstream `audio_cover_strength`).
    cover_strength: f32 = 1.0,
    /// cover: start from `t·noise + (1−t)·source latent` at the schedule step
    /// nearest `1 − strength` instead of pure noise (upstream
    /// `cover_noise_strength`; 0 = off). Blends the RAW latent, not the hints.
    cover_noise_strength: f32 = 0.0,
    /// complete: the `A | B` instrument list from `joinTrackClasses`.
    track_classes: []const u8 = "",
};

/// Reference-clip window the timbre encoder sees: always 30 s of stereo 48 kHz
/// (= `timbre_fix_frame` 750 latents). The reference (`process_reference_audio`)
/// tiles a shorter clip up to 30 s and then takes one 10 s segment from each
/// third at a RANDOM offset; ours takes the start of each third so the same
/// request is reproducible. Input/output are interleaved stereo.
pub const REF_WINDOW_S: usize = 30;
pub const REF_SEGMENT_S: usize = 10;
pub fn referenceWindow(allocator: std.mem.Allocator, stereo: []const f32, sample_rate: u32) ![]f32 {
    const sr: usize = sample_rate;
    const target = REF_WINDOW_S * sr * 2;
    if (stereo.len == 0) return error.EmptyReference;
    if (stereo.len < target) {
        const out = try allocator.alloc(f32, target);
        var i: usize = 0;
        while (i < target) : (i += 1) out[i] = stereo[i % stereo.len];
        return out;
    }
    const seg = REF_SEGMENT_S * sr * 2;
    const frames = stereo.len / 2;
    const third = (frames / 3) * 2; // interleaved samples per third (frame-aligned)
    const out = try allocator.alloc(f32, target);
    for (0..3) |k| {
        const start = k * third;
        @memcpy(out[k * seg .. (k + 1) * seg], stereo[start .. start + seg]);
    }
    return out;
}

test "acestep referenceWindow is always 30 s: tiles short clips, samples each third of long ones" {
    const a = testing.allocator;
    const sr: u32 = 100; // tiny rate keeps the test cheap; the math is rate-agnostic
    // 40 s clip whose value encodes its second: each third contributes its first 10 s.
    var long: [40 * 100 * 2]f32 = undefined;
    for (&long, 0..) |*v, i| v.* = @floatFromInt(i / 200);
    const w = try referenceWindow(a, &long, sr);
    defer a.free(w);
    try testing.expectEqual(@as(usize, 30 * 100 * 2), w.len);
    try testing.expectEqual(@as(f32, 0), w[0]);
    try testing.expectEqual(@as(f32, 9), w[10 * 200 - 1]);
    try testing.expectEqual(@as(f32, 13), w[10 * 200]); // second third starts at 13.33 s
    try testing.expectEqual(@as(f32, 26), w[20 * 200]); // last third starts at 26.66 s
    // 4 s clip tiles.
    var short: [4 * 100 * 2]f32 = undefined;
    for (&short, 0..) |*v, i| v.* = @floatFromInt(i);
    const t = try referenceWindow(a, &short, sr);
    defer a.free(t);
    try testing.expectEqual(@as(usize, 30 * 100 * 2), t.len);
    try testing.expectEqual(@as(f32, 0), t[800]);
    try testing.expectEqual(@as(f32, 399), t[t.len - 1]); // 6000 % 800
    try testing.expectError(error.EmptyReference, referenceWindow(a, &[_]f32{}, sr));
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    cfg: Cfg,
    w: Weights, // model.safetensors (DiT + condition encoder + silence_latent)
    vae_w: Weights, // vae.safetensors
    fsq_w: ?Weights, // fsq.safetensors (cover mode), loaded on first use
    model_dir: []u8,
    te_w: ?Weights, // text_encoder/ (Qwen3-Embedding-0.6B); null while phased out (low_mem)
    te_dir: []u8, // for per-request text-encoder reloads in low_mem
    tok: tok_mod.Tokenizer,
    /// Phone jetsam ceilings: load the 1.2 GB bf16 text encoder per request
    /// (freed after conditioning) and decode the VAE in small windows. The
    /// 4-bit model + resident TE + decode peak got the app SIGKILLed on an
    /// 8 GB iPhone 16 Pro (2026-07-06).
    low_mem: bool,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, low_mem: bool) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.io = io;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.cfg = Cfg{}; // single-member family; converted config mirrors these
        self.low_mem = low_mem;

        self.w = try loadFileWeights(allocator, model_dir, "model.safetensors");
        errdefer self.w.deinit();
        if (self.w.get("silence_latent") == null) return error.MissingAceStepWeight;
        self.vae_w = try loadFileWeights(allocator, model_dir, "vae.safetensors");
        errdefer self.vae_w.deinit();
        self.fsq_w = null;
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);

        self.te_dir = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{model_dir});
        errdefer allocator.free(self.te_dir);
        if (low_mem) {
            self.te_w = null;
            log.info("[acestep] low-mem mode: text encoder loads per request\n", .{});
        } else {
            self.te_w = try model_mod.loadWeights(io, allocator, self.te_dir);
        }
        errdefer if (self.te_w) |*t| t.deinit();
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, self.te_dir);
        log.info("[acestep] engine ready (model {d} + vae {d} + text_encoder {d} tensors, fsq {s})\n", .{ self.w.count(), self.vae_w.count(), if (self.te_w) |t| t.count() else 0, if (self.fsqAvailable()) "present" else "absent" });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.w.deinit();
        self.vae_w.deinit();
        if (self.fsq_w) |*f| f.deinit();
        self.allocator.free(self.model_dir);
        if (self.te_w) |*t| t.deinit();
        self.allocator.free(self.te_dir);
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    /// Whether the pack carries the cover-mode tokenizer (`fsq.safetensors`).
    /// A stat, not a load: the app fetches the file into a pack that predates
    /// it while the engine is resident, so the answer can change.
    pub fn fsqAvailable(self: *const Engine) bool {
        if (self.fsq_w != null) return true;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ self.model_dir, FSQ_FILE }) catch return false;
        std.Io.Dir.accessAbsolute(self.io, path, .{}) catch return false;
        return true;
    }

    /// FSQ weights, loaded on first cover request (`error.PackLacksFsq` when the
    /// pack predates cover mode — the handler turns that into a named 400).
    fn ensureFsq(self: *Engine) !*const Weights {
        if (self.fsq_w == null) {
            if (!self.fsqAvailable()) return error.PackLacksFsq;
            self.fsq_w = try loadFileWeights(self.allocator, self.model_dir, FSQ_FILE);
        }
        return &self.fsq_w.?;
    }

    /// Text-encoder weights, loading them on demand in low_mem mode.
    fn ensureTextEncoder(self: *Engine) !*const Weights {
        if (self.te_w == null) {
            log.info("[acestep] low-mem: loading text encoder for this request\n", .{});
            self.te_w = try model_mod.loadWeights(self.io, self.allocator, self.te_dir);
        }
        return &self.te_w.?;
    }

    /// Drop the text-encoder weights + MLX buffer cache. Callers must have
    /// EVALUATED everything that depends on them first — MLX laziness would
    /// otherwise pin the weight buffers through the graph anyway.
    fn releaseTextEncoder(self: *Engine) void {
        if (self.te_w) |*t| {
            t.deinit();
            self.te_w = null;
        }
        _ = mlx.mlx_clear_cache();
        log.info("[acestep] low-mem: text encoder freed after encode\n", .{});
    }

    /// silence_latent slice [1,frames,64] (bf16). Tiles when frames exceed the
    /// stored 15000 (600 s) — matches the reference `_get_silence_latent_slice`.
    fn silenceSlice(self: *const Engine, frames: u32, s: S) !mlx.mlx_array {
        const sil = try getW(&self.w, "silence_latent");
        const sh = mlx.getShape(sil);
        const avail: u32 = @intCast(sh[1]);
        if (frames <= avail) {
            return sliceA(sil, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, @intCast(frames), sh[2] }, &[_]c_int{ 1, 1, 1 }, s);
        }
        // Tile (duration is clamped to 600 s upstream, so this is belt+braces).
        var acc = try sliceA(sil, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, sh[1], sh[2] }, &[_]c_int{ 1, 1, 1 }, s);
        var have: u32 = avail;
        while (have < frames) : (have += avail) {
            const take: u32 = @min(avail, frames - have);
            const part = try sliceA(sil, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, @intCast(take), sh[2] }, &[_]c_int{ 1, 1, 1 }, s);
            defer _ = mlx.mlx_array_free(part);
            const next = try concat2(acc, part, 1, s);
            _ = mlx.mlx_array_free(acc);
            acc = next;
        }
        return acc;
    }

    /// Reference audio `[1,N,2]` f32 → timbre latents `[1,≤timbre_fix_frame,64]`
    /// bf16 (`refer_audio_acoustic_hidden_states_packed`): the VAE latent MEAN
    /// (the reference samples the posterior; the mean keeps a seed reproducible),
    /// cropped to the timbre window.
    fn refTimbreLatents(self: *const Engine, allocator: std.mem.Allocator, audio: mlx.mlx_array, s: S) !mlx.mlx_array {
        const mean = try vaeEncodeMean(self, allocator, audio, s); // [1,T,64] f32
        defer _ = mlx.mlx_array_free(mean);
        const sh = mlx.getShape(mean);
        const t: c_int = @min(sh[1], @as(c_int, @intCast(self.cfg.timbre_fix_frame)));
        const crop = try sliceA(mean, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, t, sh[2] }, &[_]c_int{ 1, 1, 1 }, s);
        defer _ = mlx.mlx_array_free(crop);
        const out = try astype(crop, .bfloat16, s);
        _ = mlx.mlx_array_eval(out);
        const n_samp = mlx.getShape(audio)[1];
        log.info("[acestep] reference audio: {d} latent frames ({d:.1}s)\n", .{ t, @as(f32, @floatFromInt(n_samp)) / @as(f32, @floatFromInt(self.cfg.sample_rate)) });
        return out;
    }

    /// Source clip `[1,N,2]` f32 → VAE latent mean `[1,T,64]` f32 with
    /// T = ⌊N/1920⌋ capped at 600 s (cropped, never tiled — the track IS the
    /// source's length). Shorter than the 10 s floor is the handler's 400.
    fn srcLatents(self: *const Engine, allocator: std.mem.Allocator, audio: mlx.mlx_array, progress: ?sse.Progress, s: S) !mlx.mlx_array {
        const hop: c_int = @intCast(self.cfg.vae_hop);
        const n: c_int = mlx.getShape(audio)[1];
        const frames: c_int = @min(@divTrunc(n, hop), @as(c_int, @intCast(latentFrames(MAX_DURATION_S))));
        if (frames < @as(c_int, @intCast(latentFrames(MIN_DURATION_S)))) return error.SourceTooShort;
        const cropped = try sliceA(audio, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, frames * hop, 2 }, &[_]c_int{ 1, 1, 1 }, s);
        defer _ = mlx.mlx_array_free(cropped);
        const mean = try vaeEncodeMeanChunked(self, allocator, cropped, progress, s);
        _ = mlx.mlx_array_eval(mean);
        return mean;
    }

    /// Tokenize with the reference truncation cap; returns owned i32 ids.
    /// The Qwen3-Embedding tokenizer.json carries a TemplateProcessing
    /// post-processor that appends `<|endoftext|>` after EVERY sequence (on
    /// top of any literal one in the text) — our BPE port doesn't run
    /// post-processors, so append it here (truncation reserves its slot,
    /// mirroring HF's num_special_tokens_to_add accounting).
    fn tokenize(self: *const Engine, allocator: std.mem.Allocator, text: []const u8, max_tokens: usize) ![]i32 {
        const ids_u = try self.tok.encode(allocator, text);
        defer allocator.free(ids_u);
        const eos = self.tok.specialTokenId("<|endoftext|>");
        const cap = if (eos != null) max_tokens - 1 else max_tokens;
        const n = @min(ids_u.len, cap);
        const ids = try allocator.alloc(i32, n + @intFromBool(eos != null));
        for (0..n) |i| ids[i] = @intCast(ids_u[i]);
        if (eos) |e| ids[n] = @intCast(e);
        return ids;
    }

    /// One condition set: packed encoder states → projected → per-layer cross K/V.
    fn conditionSet(self: *Engine, allocator: std.mem.Allocator, cond2048: mlx.mlx_array, s: S) !CrossKv {
        var arena_inst = std.heap.ArenaAllocator.init(allocator);
        defer arena_inst.deinit();
        const cond2560 = try lin(&self.w, arena_inst.allocator(), cond2048, "decoder.condition_embedder", s);
        defer _ = mlx.mlx_array_free(cond2560);
        return buildCrossKv(self, allocator, cond2560, s);
    }

    /// `[src | ones]` context stream [1,T,128] f32 from a [1,T,64] source.
    fn contextLatents(self: *const Engine, src: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(src);
        const src_f = try astype(src, .float32, s);
        defer _ = mlx.mlx_array_free(src_f);
        const ones_sh = [_]c_int{ 1, sh[1], @intCast(self.cfg.acoustic_dim) };
        var chunk_ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(chunk_ones);
        try mlx.check(mlx.mlx_ones(&chunk_ones, &ones_sh, 3, .float32, s));
        return concat2(src_f, chunk_ones, 2, s);
    }

    /// Text prompt → Qwen3-Embedding hidden states for one instruction.
    fn encodeTextPrompt(self: *Engine, allocator: std.mem.Allocator, te: *const Weights, instruction: []const u8, req: MusicRequest, metas: []const u8, s: S) !mlx.mlx_array {
        const prompt = try formatPrompt(allocator, instruction, req.caption, metas);
        defer allocator.free(prompt);
        const text_ids = try self.tokenize(allocator, prompt, MAX_PROMPT_TOKENS);
        defer allocator.free(text_ids);
        return qwenForward(te, allocator, text_ids, s);
    }

    /// prompt/lyrics (+ source clip for cover/complete) → 48 kHz stereo PCM16
    /// WAV bytes (owned).
    pub fn generateWav(self: *Engine, allocator: std.mem.Allocator, req: MusicRequest, progress: ?sse.Progress) ![]u8 {
        const s = self.s;
        if (progress) |p| p.emit("encode", 0, 1);

        // ── source latents decide the length on cover/complete ──
        var src_lat: ?mlx.mlx_array = null;
        defer if (src_lat) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var frames: u32 = undefined;
        var duration: u32 = undefined;
        if (req.src_audio) |audio| {
            if (req.task == .text2music) return error.SourceWithoutTask;
            src_lat = try self.srcLatents(allocator, audio, progress, s);
            frames = @intCast(mlx.getShape(src_lat.?)[1]);
            duration = frames / 25;
            log.info("[acestep] {s}: source {d:.1}s -> {d} latent frames (duration_seconds ignored), seed={d}\n", .{ @tagName(req.task), @as(f32, @floatFromInt(mlx.getShape(audio)[1])) / @as(f32, @floatFromInt(self.cfg.sample_rate)), frames, req.seed });
        } else {
            if (req.task != .text2music) return error.MissingSourceAudio;
            duration = std.math.clamp(req.duration_s, MIN_DURATION_S, MAX_DURATION_S);
            frames = latentFrames(duration);
            log.info("[acestep] text2music: {d}s ({d} frames), seed={d}\n", .{ duration, frames, req.seed });
        }

        // ── conditioning strings → token ids ──
        const metas = try formatMetaString(allocator, req.bpm, req.keyscale, req.timesignature, duration);
        defer allocator.free(metas);
        const instruction = try taskInstruction(allocator, req.task, req.track_classes);
        defer allocator.free(instruction);
        const lyric_text = try formatLyrics(allocator, req.language, req.lyrics);
        defer allocator.free(lyric_text);
        const lyric_ids = try self.tokenize(allocator, lyric_text, MAX_LYRIC_TOKENS);
        defer allocator.free(lyric_ids);

        // ── text encoder (full forward) + lyric embeds (table lookup) ──
        const te = try self.ensureTextEncoder();
        const text_hidden = try self.encodeTextPrompt(allocator, te, instruction, req, metas, s);
        defer _ = mlx.mlx_array_free(text_hidden);
        const lyric_embeds = try qwenEmbedLookup(te, lyric_ids, s);
        defer _ = mlx.mlx_array_free(lyric_embeds);
        // cover_strength < 1: the tail of the schedule runs the text2music
        // condition set (its instruction, silence context) — encode it NOW,
        // while the text encoder is resident.
        const two_sets = req.task == .cover and req.cover_strength < 1.0;
        const text_hidden2: ?mlx.mlx_array = if (two_sets) try self.encodeTextPrompt(allocator, te, TEXT2MUSIC_INSTRUCTION, req, metas, s) else null;
        defer if (text_hidden2) |x| {
            _ = mlx.mlx_array_free(x);
        };
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
        }

        // ── condition encoder: timbre = the reference clip's VAE mean OR the
        // silence latent (text2music) — one slot, never both ──
        const timbre = if (req.ref_audio) |ref|
            try self.refTimbreLatents(allocator, ref, s)
        else
            try self.silenceSlice(self.cfg.timbre_fix_frame, s);
        defer _ = mlx.mlx_array_free(timbre);
        const cond2048 = try buildConditioning(self, allocator, text_hidden, lyric_embeds, timbre, s);
        defer _ = mlx.mlx_array_free(cond2048);
        const cond2048_2: ?mlx.mlx_array = if (text_hidden2) |th2| try buildConditioning(self, allocator, th2, lyric_embeds, timbre, s) else null;
        defer if (cond2048_2) |x| {
            _ = mlx.mlx_array_free(x);
        };
        // Phased text encoder (the FLUX low-mem pattern): materialize the
        // conditioning NOW — laziness would pin the TE weights through the
        // graph — then return its 1.2 GB before the denoise loop.
        if (self.low_mem) {
            _ = mlx.mlx_array_eval(cond2048);
            if (cond2048_2) |c2| _ = mlx.mlx_array_eval(c2);
            self.releaseTextEncoder();
        }
        // Project once, build the per-layer cross K/V once (per set).
        var cross = try self.conditionSet(allocator, cond2048, s);
        defer cross.deinit(allocator);
        var cross2: ?CrossKv = if (cond2048_2) |c2| try self.conditionSet(allocator, c2, s) else null;
        defer if (cross2) |*c| c.deinit(allocator);
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("encode", 1, 1);
        }

        // ── context latents [1,T,128]: the task IS the context stream ──
        const ctx = switch (req.task) {
            .text2music => blk: {
                const sil = try self.silenceSlice(frames, s);
                defer _ = mlx.mlx_array_free(sil);
                break :blk try self.contextLatents(sil, s);
            },
            .complete => blk: {
                log.info("[acestep] complete: {d} source frames as context, classes={s}\n", .{ frames, if (req.track_classes.len == 0) "(default)" else req.track_classes });
                break :blk try self.contextLatents(src_lat.?, s);
            },
            .cover => blk: {
                const toks = try audioTokenize(self, allocator, src_lat.?, s);
                defer toks.deinit();
                const det = try audioDetokenize(self, allocator, toks.hints, s);
                defer _ = mlx.mlx_array_free(det);
                const crop = try sliceA(det, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, @intCast(frames), @intCast(self.cfg.acoustic_dim) }, &[_]c_int{ 1, 1, 1 }, s);
                defer _ = mlx.mlx_array_free(crop);
                log.info("[acestep] cover: source {d} frames -> {d} codes (strength={d:.2}, noise_strength={d:.2})\n", .{ frames, mlx.getShape(toks.hints)[1], req.cover_strength, req.cover_noise_strength });
                break :blk try self.contextLatents(crop, s);
            },
        };
        defer _ = mlx.mlx_array_free(ctx);
        const ctx2: ?mlx.mlx_array = if (two_sets) blk: {
            const sil = try self.silenceSlice(frames, s);
            defer _ = mlx.mlx_array_free(sil);
            break :blk try self.contextLatents(sil, s);
        } else null;
        defer if (ctx2) |x| {
            _ = mlx.mlx_array_free(x);
        };

        // ── seeded noise ──
        const lat_sh = [_]c_int{ 1, @intCast(frames), @intCast(self.cfg.acoustic_dim) };
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, req.seed));
        var xt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&xt, &lat_sh, 3, .float32, 0.0, 1.0, key, s));
        defer _ = mlx.mlx_array_free(xt);

        // ── 8-step Euler flow-match + DCW ──
        var sched: [NUM_STEPS]f32 = undefined;
        timestepSchedule(&sched, SHIFT);
        var start_step: usize = 0;
        if (req.task == .cover and req.cover_noise_strength > 0.0) {
            // Start part-way down the schedule from a noise/source blend.
            start_step = nearestStep(&sched, 1.0 - req.cover_noise_strength);
            const t0 = sched[start_step];
            const noisy = try mulScalar(xt, t0, s);
            defer _ = mlx.mlx_array_free(noisy);
            const srcw = try mulScalar(src_lat.?, 1.0 - t0, s);
            defer _ = mlx.mlx_array_free(srcw);
            const blend = try addA(noisy, srcw, s);
            _ = mlx.mlx_array_free(xt);
            xt = blend;
            log.info("[acestep] cover: noise_strength={d:.2} -> start at t={d:.3} ({d} steps remain)\n", .{ req.cover_noise_strength, t0, NUM_STEPS - start_step });
        }
        const remaining = NUM_STEPS - start_step;
        const cover_steps: usize = if (req.task == .cover) @intFromFloat(@as(f32, @floatFromInt(remaining)) * req.cover_strength) else remaining;
        var switched = false;
        var cur_ctx = ctx;
        var cur_cross: *const CrossKv = &cross;
        for (start_step..NUM_STEPS) |step| {
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("diffuse", @intCast(step - start_step), @intCast(remaining));
            }
            if (two_sets and !switched and step - start_step >= cover_steps) {
                switched = true;
                cur_ctx = ctx2.?;
                cur_cross = &cross2.?;
                log.info("[acestep] cover: switching to text2music conditioning at step {d}/{d}\n", .{ step - start_step, remaining });
            }
            const t_curr = sched[step];
            const vt = try ditForward(self, allocator, xt, t_curr, cur_ctx, cur_cross, s);
            defer _ = mlx.mlx_array_free(vt);

            const step_size = if (step == NUM_STEPS - 1) t_curr else t_curr - sched[step + 1];
            const dv = try mulScalar(vt, step_size, s);
            defer _ = mlx.mlx_array_free(dv);
            const x_next = try subA(xt, dv, s);
            defer _ = mlx.mlx_array_free(x_next);

            // denoised = x_before − v·t (uses the PRE-update latent)
            const vtt = try mulScalar(vt, t_curr, s);
            defer _ = mlx.mlx_array_free(vtt);
            const denoised = try subA(xt, vtt, s);
            defer _ = mlx.mlx_array_free(denoised);

            const corrected = try applyDcwDouble(x_next, denoised, t_curr, s);
            _ = mlx.mlx_array_free(xt);
            xt = corrected;
            _ = mlx.mlx_array_eval(xt);
        }
        if (progress) |p| p.emit("diffuse", @intCast(remaining), @intCast(remaining));

        // ── VAE decode → normalize → WAV ──
        // Decode is the pipeline's memory peak; start it from a drained cache.
        if (self.low_mem) _ = mlx.mlx_clear_cache();
        const samples = try vaeDecodeChunked(self, allocator, xt, progress, s);
        defer allocator.free(samples);
        peakNormalize(samples, NORMALIZE_DB);
        return wav_mod.encodePcm16(allocator, samples, self.cfg.sample_rate, 2);
    }
};

/// Load ONE safetensors file into a Weights map (CPU stream; the iterator's +1
/// reference is transferred into the map — the model.zig pattern).
fn loadFileWeights(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, cpu_s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = try allocator.dupe(u8, std.mem.span(key.?));
        errdefer allocator.free(owned_key);
        try w.map.put(owned_key, value);
    }
    log.info("[acestep] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first (no weights), then env-gated cos oracles fed by
// tests/dump_acestep_fixtures.py (ACESTEP_*, mirrors HY3D_*).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "acestep timestep schedule: shift=3 N=8 reproduces SHIFT_TIMESTEPS[3.0]" {
    var buf: [8]f32 = undefined;
    timestepSchedule(&buf, 3.0);
    const expected = [_]f32{ 1.0, 0.9545454545, 0.9, 0.8333333333, 0.75, 0.6428571429, 0.5, 0.3 };
    for (0..8) |i| try testing.expectApproxEqAbs(expected[i], buf[i], 1e-6);
}

test "acestep timestep schedule: shift=1 is plain linspace" {
    var buf: [8]f32 = undefined;
    timestepSchedule(&buf, 1.0);
    const expected = [_]f32{ 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125 };
    for (0..8) |i| try testing.expectApproxEqAbs(expected[i], buf[i], 1e-7);
}

test "acestep latent frames: exactly 25 per second" {
    try testing.expectEqual(@as(u32, 250), latentFrames(10));
    try testing.expectEqual(@as(u32, 15000), latentFrames(600));
}

test "acestep prompt format matches the reference SFT_GEN_PROMPT byte-exactly" {
    const a = testing.allocator;
    const metas = try formatMetaString(a, null, "", "", 10);
    defer a.free(metas);
    try testing.expectEqualStrings(
        "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 10 seconds\n",
        metas,
    );
    const prompt = try formatPrompt(a, TEXT2MUSIC_INSTRUCTION, "upbeat synthwave with driving bass, dreamy pads and a catchy lead melody", metas);
    defer a.free(prompt);
    try testing.expectEqualStrings(
        "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n" ++
            "# Caption\nupbeat synthwave with driving bass, dreamy pads and a catchy lead melody\n\n" ++
            "# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 10 seconds\n<|endoftext|>\n",
        prompt,
    );
}

test "acestep task instruction: the instruction line is the task (constants.py TASK_INSTRUCTIONS)" {
    const a = testing.allocator;
    const t2m = try taskInstruction(a, .text2music, "");
    defer a.free(t2m);
    try testing.expectEqualStrings("Fill the audio semantic mask based on the given conditions:", t2m);
    const cov = try taskInstruction(a, .cover, "");
    defer a.free(cov);
    try testing.expectEqualStrings("Generate audio semantic tokens based on the given conditions:", cov);
    const dflt = try taskInstruction(a, .complete, "");
    defer a.free(dflt);
    try testing.expectEqualStrings("Complete the input track:", dflt);
    const classes = try joinTrackClasses(a, &.{ "bass", " Drums", "backing_vocals" });
    defer a.free(classes);
    try testing.expectEqualStrings("BASS | DRUMS | BACKING_VOCALS", classes);
    const comp = try taskInstruction(a, .complete, classes);
    defer a.free(comp);
    try testing.expectEqualStrings("Complete the input track with BASS | DRUMS | BACKING_VOCALS:", comp);
    try testing.expectError(error.UnknownTrackClass, joinTrackClasses(a, &.{ "bass", "kazoo" }));
    // The prompt carries it verbatim.
    const p = try formatPrompt(a, comp, "lo-fi", "- duration: 10 seconds\n");
    defer a.free(p);
    try testing.expectEqualStrings("# Instruction\nComplete the input track with BASS | DRUMS | BACKING_VOCALS:\n\n# Caption\nlo-fi\n\n# Metas\n- duration: 10 seconds\n<|endoftext|>\n", p);
}

test "acestep nearestStep: cover_noise_strength picks the schedule entry nearest 1-strength" {
    var sched: [NUM_STEPS]f32 = undefined;
    timestepSchedule(&sched, SHIFT); // 1, .9545, .9, .8333, .75, .6429, .5, .3
    try testing.expectEqual(@as(usize, 0), nearestStep(&sched, 1.0));
    try testing.expectEqual(@as(usize, 7), nearestStep(&sched, 0.0));
    try testing.expectEqual(@as(usize, 6), nearestStep(&sched, 0.45)); // |0.5-0.45| < |0.3-0.45|
    try testing.expectEqual(@as(usize, 2), nearestStep(&sched, 0.9));
}

test "acestep FSQ grid: soft clamp + symmetry-preserving floor grid + mixed-radix indices" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    // levels [8,5], z' = c·tanh(z/c):
    //   L=8 (c=8/7): z=0.3 → 0.2933 → ⌊7·1.2933/2+0.5⌋=5 → 5·2/7−1=0.428571; z=1.7 → 1.032 → clamp 1 → 1.0
    //   L=5 (c=1.25): z=0.3 → 0.2944 → ⌊4·1.2944/2+0.5⌋=3 → 0.5; z=−1 → −0.830 → ⌊0.84⌋=0 → −1
    //   L=5: z=−0.78662 → −0.6971 → ⌊1.106⌋=1 → −0.5 (the hard clamp ALONE floors 0.927 → −1: the
    //   reference's code for this exact value is level 1 — oracle 8 caught it)
    const levels = [_]f32{ 8, 5 };
    const zv = [_]f32{ 0.3, 0.3, 1.7, -1.0, 0.0, -0.78662 };
    const sh = [_]c_int{ 1, 3, 2 };
    const z = mlx.mlx_array_new_data(&zv, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(z);
    const codes = try fsqQuantize(z, &levels, s);
    defer _ = mlx.mlx_array_free(codes);
    var cc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cc);
    try mlx.check(mlx.mlx_contiguous(&cc, codes, false, s));
    _ = mlx.mlx_array_eval(cc);
    const d = mlx.mlx_array_data_float32(cc) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 5.0 * 2.0 / 7.0 - 1.0), d[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), d[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), d[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), d[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.5), d[5], 1e-6);
    // indices: level idx (5,3) → 5 + 3·8 = 29; (7,0) → 7; (4,1) → 4 + 8 = 12 (z=0 at L=8 is level 4)
    const idx = try fsqIndices(codes, &levels, s);
    defer _ = mlx.mlx_array_free(idx);
    var ic = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ic);
    try mlx.check(mlx.mlx_contiguous(&ic, idx, false, s));
    _ = mlx.mlx_array_eval(ic);
    const di = mlx.mlx_array_data_int32(ic) orelse return error.NoData;
    try testing.expectEqual(@as(i32, 29), di[0]);
    try testing.expectEqual(@as(i32, 7), di[1]);
    try testing.expectEqual(@as(i32, 12), di[2]);
}

test "acestep meta string with explicit bpm/keyscale/timesignature" {
    const a = testing.allocator;
    const metas = try formatMetaString(a, 128, "F# minor", "4", 60);
    defer a.free(metas);
    try testing.expectEqualStrings(
        "- bpm: 128\n- timesignature: 4\n- keyscale: F# minor\n- duration: 60 seconds\n",
        metas,
    );
}

test "acestep lyric format: empty lyrics become [Instrumental]" {
    const a = testing.allocator;
    const l1 = try formatLyrics(a, "en", "");
    defer a.free(l1);
    try testing.expectEqualStrings("# Languages\nen\n\n# Lyric\n[Instrumental]<|endoftext|>", l1);
    const l2 = try formatLyrics(a, "", "la la la");
    defer a.free(l2);
    try testing.expectEqualStrings("# Languages\nen\n\n# Lyric\nla la la<|endoftext|>", l2);
}

test "acestep peak normalize hits -1 dBFS and skips silence" {
    var buf = [_]f32{ 0.1, -0.4, 0.2 };
    peakNormalize(&buf, -1.0);
    const target = std.math.pow(f32, 10.0, -1.0 / 20.0);
    try testing.expectApproxEqAbs(target, @abs(buf[1]), 1e-6);
    try testing.expectApproxEqAbs(target * 0.25, buf[0], 1e-6);
    var quiet = [_]f32{ 0.0, 0.0 };
    peakNormalize(&quiet, -1.0); // must not divide by zero
    try testing.expectEqual(@as(f32, 0.0), quiet[0]);
}

test "acestep haar DWT/IDWT round-trips exactly (DCW identity at zero scalers)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    const vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, -6.0 };
    const sh = [_]c_int{ 1, 6, 1 };
    const x = mlx.mlx_array_new_data(&vals, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    const bands = try haarDwt(x, s);
    defer _ = mlx.mlx_array_free(bands.low);
    defer _ = mlx.mlx_array_free(bands.high);
    const rec = try haarIdwt(bands.low, bands.high, 6, s);
    defer _ = mlx.mlx_array_free(rec);
    var rc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rc);
    try mlx.check(mlx.mlx_contiguous(&rc, rec, false, s));
    _ = mlx.mlx_array_eval(rc);
    const d = mlx.mlx_array_data_float32(rc) orelse return error.NoData;
    for (0..6) |i| try testing.expectApproxEqAbs(vals[i], d[i], 1e-5);
}

test "acestep DCW double: hand-computed low/high band push" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    // T=2, C=1: x=[2,0] → low=√2, high=√2; y=[0,0] → 0,0. t=0.5:
    // low' = √2(1+0.5·0.05) = √2·1.025; high' = √2(1+0.5·0.02) = √2·1.01
    // rec = [(low'+high')/√2, (low'−high')/√2] = [2.035, 0.015]
    const xv = [_]f32{ 2.0, 0.0 };
    const yv = [_]f32{ 0.0, 0.0 };
    const sh = [_]c_int{ 1, 2, 1 };
    const x = mlx.mlx_array_new_data(&xv, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    const y = mlx.mlx_array_new_data(&yv, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(y);
    const out = try applyDcwDouble(x, y, 0.5, s);
    defer _ = mlx.mlx_array_free(out);
    var oc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(oc);
    try mlx.check(mlx.mlx_contiguous(&oc, out, false, s));
    _ = mlx.mlx_array_eval(oc);
    const d = mlx.mlx_array_data_float32(oc) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 2.035), d[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.015), d[1], 1e-5);
}

test "acestep sliding band mask zeroes |i-j|<=w and blocks beyond" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    const m = try slidingBandMask(5, 2, s);
    defer _ = mlx.mlx_array_free(m);
    const mf = try astype(m, .float32, s);
    defer _ = mlx.mlx_array_free(mf);
    var mc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mc);
    try mlx.check(mlx.mlx_contiguous(&mc, mf, false, s));
    _ = mlx.mlx_array_eval(mc);
    const d = mlx.mlx_array_data_float32(mc) orelse return error.NoData;
    // row 0: j=0,1,2 inside; j=3,4 blocked
    try testing.expectEqual(@as(f32, 0.0), d[0]);
    try testing.expectEqual(@as(f32, 0.0), d[2]);
    try testing.expect(d[3] < -1e8);
    // row 4: j=2..4 inside, j=0,1 blocked (bidirectional symmetry)
    try testing.expect(d[4 * 5 + 0] < -1e8);
    try testing.expectEqual(@as(f32, 0.0), d[4 * 5 + 2]);
}

// ── oracle helpers ──

fn readF32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn readI32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]i32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(i32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn cosine(data: []const f32, ref: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..data.len) |i| {
        dot += @as(f64, data[i]) * ref[i];
        na += @as(f64, data[i]) * data[i];
        nb += @as(f64, ref[i]) * ref[i];
    }
    return dot / (std.math.sqrt(na) * std.math.sqrt(nb));
}

fn arrayCosine(arr: mlx.mlx_array, ref: []const f32, s: S) !f64 {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    var fc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fc);
    try mlx.check(mlx.mlx_contiguous(&fc, f, false, s));
    _ = mlx.mlx_array_eval(fc);
    const n: usize = @intCast(mlx.mlx_array_size(fc));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(fc) orelse return error.NoData;
    return cosine(d[0..n], ref);
}

fn testEngine(io: std.Io, a: std.mem.Allocator) !*Engine {
    const model_dir = std.mem.span(std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest);
    return Engine.load(io, a, model_dir, false);
}

// Oracle 1a: tokenizer + prompt formatting. The dump script writes the token
// ids the REFERENCE tokenizer produced from the exact reference strings; our
// formatter + tokenizer must reproduce them (byte-level conditioning parity).
test "acestep oracle: prompt tokenization matches reference ids" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const ids_p = std.mem.span(std.c.getenv("ACESTEP_TEXT_IDS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ref_ids = try readI32(io, a, ids_p);
    defer a.free(ref_ids);

    var e = try testEngine(io, a);
    defer e.deinit();
    const metas = try formatMetaString(a, null, "", "", 10);
    defer a.free(metas);
    const prompt = try formatPrompt(a, TEXT2MUSIC_INSTRUCTION, "upbeat synthwave with driving bass, dreamy pads and a catchy lead melody", metas);
    defer a.free(prompt);
    const ids = try e.tokenize(a, prompt, MAX_PROMPT_TOKENS);
    defer a.free(ids);
    try testing.expectEqualSlices(i32, ref_ids, ids);
}

// Oracle 1b: Qwen3-Embedding full forward → last_hidden_state.
test "acestep oracle: text encoder hidden states match reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const ids_p = std.mem.span(std.c.getenv("ACESTEP_TEXT_IDS") orelse return error.SkipZigTest);
    const hid_p = std.mem.span(std.c.getenv("ACESTEP_TEXT_HIDDEN") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p);
    defer a.free(ids);
    const ref = try readF32(io, a, hid_p);
    defer a.free(ref);
    var e = try testEngine(io, a);
    defer e.deinit();
    const hidden = try qwenForward(&e.te_w.?, a, ids, e.s);
    defer _ = mlx.mlx_array_free(hidden);
    const corr = try arrayCosine(hidden, ref, e.s);
    std.debug.print("[acestep-text] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.999);
}

// Oracle 2: condition encoder → packed [lyric|timbre|text] states.
test "acestep oracle: condition encoder matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const th_p = std.mem.span(std.c.getenv("ACESTEP_TEXT_HIDDEN") orelse return error.SkipZigTest);
    const tl_s = std.mem.span(std.c.getenv("ACESTEP_TEXT_LEN") orelse return error.SkipZigTest);
    const le_p = std.mem.span(std.c.getenv("ACESTEP_LYRIC_EMBEDS") orelse return error.SkipZigTest);
    const ll_s = std.mem.span(std.c.getenv("ACESTEP_LYRIC_LEN") orelse return error.SkipZigTest);
    const cond_p = std.mem.span(std.c.getenv("ACESTEP_COND") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const text_len = try std.fmt.parseInt(c_int, tl_s, 10);
    const lyric_len = try std.fmt.parseInt(c_int, ll_s, 10);
    const th = try readF32(io, a, th_p);
    defer a.free(th);
    const le = try readF32(io, a, le_p);
    defer a.free(le);
    const ref = try readF32(io, a, cond_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const th_sh = [_]c_int{ 1, text_len, 1024 };
    const th_f32 = mlx.mlx_array_new_data(th.ptr, &th_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(th_f32);
    const th_bf = try astype(th_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(th_bf);
    const le_sh = [_]c_int{ 1, lyric_len, 1024 };
    const le_f32 = mlx.mlx_array_new_data(le.ptr, &le_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(le_f32);
    const le_bf = try astype(le_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(le_bf);
    const timbre = try e.silenceSlice(e.cfg.timbre_fix_frame, s);
    defer _ = mlx.mlx_array_free(timbre);
    const cond = try buildConditioning(e, a, th_bf, le_bf, timbre, s);
    defer _ = mlx.mlx_array_free(cond);
    const corr = try arrayCosine(cond, ref, s);
    std.debug.print("[acestep-cond] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.999);
}

// Oracle 7: condition encoder with a REAL reference latent in the timbre slot
// (the #259 path). Reuses the oracle-6 audio: VAE mean → timbre → packed states.
test "acestep oracle: condition encoder with reference audio matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const th_p = std.mem.span(std.c.getenv("ACESTEP_TEXT_HIDDEN") orelse return error.SkipZigTest);
    const tl_s = std.mem.span(std.c.getenv("ACESTEP_TEXT_LEN") orelse return error.SkipZigTest);
    const le_p = std.mem.span(std.c.getenv("ACESTEP_LYRIC_EMBEDS") orelse return error.SkipZigTest);
    const ll_s = std.mem.span(std.c.getenv("ACESTEP_LYRIC_LEN") orelse return error.SkipZigTest);
    const audio_p = std.mem.span(std.c.getenv("ACESTEP_VAEENC_AUDIO") orelse return error.SkipZigTest);
    const cond_p = std.mem.span(std.c.getenv("ACESTEP_REFCOND") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const text_len = try std.fmt.parseInt(c_int, tl_s, 10);
    const lyric_len = try std.fmt.parseInt(c_int, ll_s, 10);
    const th = try readF32(io, a, th_p);
    defer a.free(th);
    const le = try readF32(io, a, le_p);
    defer a.free(le);
    const audio_d = try readF32(io, a, audio_p);
    defer a.free(audio_d);
    const ref = try readF32(io, a, cond_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const th_f32 = mlx.mlx_array_new_data(th.ptr, &[_]c_int{ 1, text_len, 1024 }, 3, .float32);
    defer _ = mlx.mlx_array_free(th_f32);
    const th_bf = try astype(th_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(th_bf);
    const le_f32 = mlx.mlx_array_new_data(le.ptr, &[_]c_int{ 1, lyric_len, 1024 }, 3, .float32);
    defer _ = mlx.mlx_array_free(le_f32);
    const le_bf = try astype(le_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(le_bf);
    // audio fixture is [1,2,N] channels-first → [1,N,2]
    const n_samp: c_int = @intCast(audio_d.len / 2);
    const audio_cf = mlx.mlx_array_new_data(audio_d.ptr, &[_]c_int{ 1, 2, n_samp }, 3, .float32);
    defer _ = mlx.mlx_array_free(audio_cf);
    var audio_nlc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(audio_nlc);
    try mlx.check(mlx.mlx_transpose_axes(&audio_nlc, audio_cf, &[_]c_int{ 0, 2, 1 }, 3, s));
    const timbre = try e.refTimbreLatents(a, audio_nlc, s);
    defer _ = mlx.mlx_array_free(timbre);
    const cond = try buildConditioning(e, a, th_bf, le_bf, timbre, s);
    defer _ = mlx.mlx_array_free(cond);
    const corr = try arrayCosine(cond, ref, s);
    // The timbre token is ONE row of the pack (index lyric_len), so the
    // whole-tensor cosine is dominated by the 85 rows that never see the
    // clip — the silence and reference fixtures differ ONLY on that row
    // (row cos 0.075 between them). Pin the row itself.
    const row_i: usize = @intCast(lyric_len);
    const row = try sliceA(cond, &[_]c_int{ 0, lyric_len, 0 }, &[_]c_int{ 1, lyric_len + 1, 2048 }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(row);
    const row_corr = try arrayCosine(row, ref[row_i * 2048 .. (row_i + 1) * 2048], s);
    std.debug.print("[acestep-refcond] corr={d:.6} timbre_row_corr={d:.6}\n", .{ corr, row_corr });
    try testing.expect(corr > 0.999);
    try testing.expect(row_corr > 0.999);
}

// Oracle 3: one DiT velocity at t=1.0 from injected noise + reference cond.
test "acestep oracle: DiT velocity matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const cond_p = std.mem.span(std.c.getenv("ACESTEP_COND") orelse return error.SkipZigTest);
    const noise_p = std.mem.span(std.c.getenv("ACESTEP_DIT_NOISE") orelse return error.SkipZigTest);
    const v1_p = std.mem.span(std.c.getenv("ACESTEP_DIT_V1") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const cond_d = try readF32(io, a, cond_p);
    defer a.free(cond_d);
    const noise_d = try readF32(io, a, noise_p);
    defer a.free(noise_d);
    const ref = try readF32(io, a, v1_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const frames: c_int = @intCast(noise_d.len / 64);
    const cond_len: c_int = @intCast(cond_d.len / 2048);
    const c_sh = [_]c_int{ 1, cond_len, 2048 };
    const cond_f32 = mlx.mlx_array_new_data(cond_d.ptr, &c_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(cond_f32);
    const cond_bf = try astype(cond_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(cond_bf);
    var arena_inst = std.heap.ArenaAllocator.init(a);
    defer arena_inst.deinit();
    const cond2560 = try lin(&e.w, arena_inst.allocator(), cond_bf, "decoder.condition_embedder", s);
    defer _ = mlx.mlx_array_free(cond2560);
    var cross = try buildCrossKv(e, a, cond2560, s);
    defer cross.deinit(a);

    const n_sh = [_]c_int{ 1, frames, 64 };
    const noise = mlx.mlx_array_new_data(noise_d.ptr, &n_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(noise);
    const src = try e.silenceSlice(@intCast(frames), s);
    defer _ = mlx.mlx_array_free(src);
    const src_f = try astype(src, .float32, s);
    defer _ = mlx.mlx_array_free(src_f);
    var ones_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_a);
    try mlx.check(mlx.mlx_ones(&ones_a, &n_sh, 3, .float32, s));
    const ctx = try concat2(src_f, ones_a, 2, s);
    defer _ = mlx.mlx_array_free(ctx);

    const v1 = try ditForward(e, a, noise, 1.0, ctx, &cross, s);
    defer _ = mlx.mlx_array_free(v1);
    const corr = try arrayCosine(v1, ref, s);
    std.debug.print("[acestep-dit] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.995);
}

// Oracle 4: full 8-step euler + DCW from oracle 3's noise → final latents.
test "acestep oracle: e2e denoise matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const cond_p = std.mem.span(std.c.getenv("ACESTEP_COND") orelse return error.SkipZigTest);
    const noise_p = std.mem.span(std.c.getenv("ACESTEP_DIT_NOISE") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("ACESTEP_E2E_LATENTS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const cond_d = try readF32(io, a, cond_p);
    defer a.free(cond_d);
    const noise_d = try readF32(io, a, noise_p);
    defer a.free(noise_d);
    const ref = try readF32(io, a, out_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const frames: c_int = @intCast(noise_d.len / 64);
    const cond_len: c_int = @intCast(cond_d.len / 2048);
    const c_sh = [_]c_int{ 1, cond_len, 2048 };
    const cond_f32 = mlx.mlx_array_new_data(cond_d.ptr, &c_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(cond_f32);
    const cond_bf = try astype(cond_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(cond_bf);
    var arena_inst = std.heap.ArenaAllocator.init(a);
    defer arena_inst.deinit();
    const cond2560 = try lin(&e.w, arena_inst.allocator(), cond_bf, "decoder.condition_embedder", s);
    defer _ = mlx.mlx_array_free(cond2560);
    var cross = try buildCrossKv(e, a, cond2560, s);
    defer cross.deinit(a);

    const n_sh = [_]c_int{ 1, frames, 64 };
    var xt = mlx.mlx_array_new_data(noise_d.ptr, &n_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(xt);
    const src = try e.silenceSlice(@intCast(frames), s);
    defer _ = mlx.mlx_array_free(src);
    const src_f = try astype(src, .float32, s);
    defer _ = mlx.mlx_array_free(src_f);
    var ones_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_a);
    try mlx.check(mlx.mlx_ones(&ones_a, &n_sh, 3, .float32, s));
    const ctx = try concat2(src_f, ones_a, 2, s);
    defer _ = mlx.mlx_array_free(ctx);

    var sched: [NUM_STEPS]f32 = undefined;
    timestepSchedule(&sched, SHIFT);
    for (0..NUM_STEPS) |step| {
        const t_curr = sched[step];
        const vt = try ditForward(e, a, xt, t_curr, ctx, &cross, s);
        defer _ = mlx.mlx_array_free(vt);
        const step_size = if (step == NUM_STEPS - 1) t_curr else t_curr - sched[step + 1];
        const dv = try mulScalar(vt, step_size, s);
        defer _ = mlx.mlx_array_free(dv);
        const x_next = try subA(xt, dv, s);
        defer _ = mlx.mlx_array_free(x_next);
        const vtt = try mulScalar(vt, t_curr, s);
        defer _ = mlx.mlx_array_free(vtt);
        const denoised = try subA(xt, vtt, s);
        defer _ = mlx.mlx_array_free(denoised);
        const corrected = try applyDcwDouble(x_next, denoised, t_curr, s);
        _ = mlx.mlx_array_free(xt);
        xt = corrected;
        _ = mlx.mlx_array_eval(xt);
    }
    const corr = try arrayCosine(xt, ref, s);
    std.debug.print("[acestep-e2e] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.99);
}

// Oracle 5: VAE decode. Reference dumps [1,64,T] (torch channel-first);
// transpose to our NLC layout before decoding.
test "acestep oracle: VAE decode matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const lat_p = std.mem.span(std.c.getenv("ACESTEP_VAEDEC_LAT") orelse return error.SkipZigTest);
    const wav_p = std.mem.span(std.c.getenv("ACESTEP_VAEDEC_WAV") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat_d = try readF32(io, a, lat_p);
    defer a.free(lat_d);
    const ref = try readF32(io, a, wav_p); // [1,2,N] channel-first
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const frames: c_int = @intCast(lat_d.len / 64);
    const cf_sh = [_]c_int{ 1, 64, frames };
    const lat_cf = mlx.mlx_array_new_data(lat_d.ptr, &cf_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(lat_cf);
    const lat_nlc = try transpose(lat_cf, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(lat_nlc);
    const lat_bf = try astype(lat_nlc, .bfloat16, s);
    defer _ = mlx.mlx_array_free(lat_bf);
    const audio = try vaeDecodeWindow(e, a, lat_bf, s); // [1,N,2] NLC
    defer _ = mlx.mlx_array_free(audio);
    // reference is [1,2,N] — transpose ours to channel-first for comparison
    const audio_cf = try transpose(audio, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(audio_cf);
    const corr = try arrayCosine(audio_cf, ref, s);
    std.debug.print("[acestep-vaedec] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.99);
}

// Oracle 6: VAE encode mean (M3 reference-audio prep). Reference audio is
// [1,2,N] channel-first, latent mean [1,64,T].
test "acestep oracle: VAE encode mean matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const audio_p = std.mem.span(std.c.getenv("ACESTEP_VAEENC_AUDIO") orelse return error.SkipZigTest);
    const mean_p = std.mem.span(std.c.getenv("ACESTEP_VAEENC_MEAN") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const audio_d = try readF32(io, a, audio_p);
    defer a.free(audio_d);
    const ref = try readF32(io, a, mean_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const n_samp: c_int = @intCast(audio_d.len / 2);
    const cf_sh = [_]c_int{ 1, 2, n_samp };
    const audio_cf = mlx.mlx_array_new_data(audio_d.ptr, &cf_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(audio_cf);
    const audio_nlc = try transpose(audio_cf, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(audio_nlc);
    // f32 audio in — the engine's own decode path produces f32 samples, and a
    // bf16 round-trip of the INPUT signal alone costs cos 0.9999 → 0.958.
    const mean = try vaeEncodeMean(e, a, audio_nlc, s); // [1,T,64] NLC
    defer _ = mlx.mlx_array_free(mean);
    const mean_cf = try transpose(mean, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(mean_cf);
    const corr = try arrayCosine(mean_cf, ref, s);
    std.debug.print("[acestep-vaeenc] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.999);
}

/// FSQ boundary distance of a pre-quantization value, in cells: how far
/// `(L−1)(clamp(z)+1)/2 + 0.5` sits from the nearest floor boundary (0 = on
/// the edge, 0.5 = dead centre).
fn fsqBoundaryDistance(z: f32, level: f32) f32 {
    const u = (level - 1.0) * (std.math.clamp(fsqSoftClamp(z, level), -1.0, 1.0) + 1.0) / 2.0 + 0.5;
    return @abs(u - @round(u));
}

/// Mixed-radix digit `d` of a codebook index (basis cumprod([1] ++ L[:-1])).
fn fsqDigit(index: i32, levels: []const f32, d: usize) i32 {
    var idx = index;
    for (0..d) |i| idx = @divTrunc(idx, @as(i32, @intFromFloat(levels[i])));
    return @mod(idx, @as(i32, @intFromFloat(levels[d])));
}

test "acestep FSQ boundary distance + digits" {
    // L=8, z=0.3 → soft 0.29330 → u=5.0266 → 0.027 from the floor boundary;
    // z=1.7 → soft 1.032 → clamps → u=7.5 → centre.
    try testing.expectApproxEqAbs(@as(f32, 0.0266), fsqBoundaryDistance(0.3, 8), 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.5), fsqBoundaryDistance(1.7, 8), 1e-5);
    const lv = [_]f32{ 8, 5 };
    try testing.expectEqual(@as(i32, 5), fsqDigit(29, &lv, 0));
    try testing.expectEqual(@as(i32, 3), fsqDigit(29, &lv, 1));
}

// Chunked encode == untiled encode. A cover source longer than one window
// (30 s) is the ONLY path through `vaeEncodeMeanChunked`; every short test
// clip fits one window and never exercises the halo/offset arithmetic.
test "acestep chunked VAE encode matches the untiled pass on a multi-window clip" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    // 40 s stereo chirp + tone = 1000 frames → two windows with a halo seam at 750.
    const n: usize = 1000 * 1920;
    const buf = try a.alloc(f32, n * 2);
    defer a.free(buf);
    for (0..n) |i| {
        const t = @as(f32, @floatFromInt(i)) / 48000.0;
        buf[i * 2] = 0.5 * @sin(2.0 * std.math.pi * (110.0 + 20.0 * t) * t);
        buf[i * 2 + 1] = 0.4 * @sin(2.0 * std.math.pi * 330.0 * t) * @sin(0.5 * t);
    }
    const audio = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(n), 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(audio);
    const whole = try vaeEncodeMean(e, a, audio, s);
    defer _ = mlx.mlx_array_free(whole);
    const chunked = try vaeEncodeMeanChunked(e, a, audio, null, s);
    defer _ = mlx.mlx_array_free(chunked);
    try testing.expectEqual(mlx.getShape(whole)[1], mlx.getShape(chunked)[1]);
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, whole, false, s));
    _ = mlx.mlx_array_eval(wc);
    const nw: usize = @intCast(mlx.mlx_array_size(wc));
    const wd = mlx.mlx_array_data_float32(wc) orelse return error.NoData;
    const corr = try arrayCosine(chunked, wd[0..nw], s);
    std.debug.print("[acestep-chunked-enc] frames={d} corr={d:.6}\n", .{ mlx.getShape(whole)[1], corr });
    try testing.expect(corr > 0.9999);
}

// Oracle 8: FSQ tokenizer. Reference: oracle-6 latent mean [1,64,50] → 5 Hz
// hints [1,10,2048] + codebook indices [1,10]. The bar is NEAR-TIE acquittal
// (the MTP rule): a code may differ from the fp32 reference ONLY on a grid
// dimension whose reference pre-quantization value sits within 0.05 cells of
// a floor boundary. Measured 2026-08-22: dense-bf16 FSQ weights + f32
// activations are EXACT (0/10, max |Δz| 2e-5); the 8-bit pack moves z by up
// to 0.039 and flips 3/10 codes, all at ≤ 0.012 cells. A wrong grid formula
// (the hard clamp without the soft clamp) or a wrong RoPE/mask/window layout
// flips dims sitting 0.07–0.5 cells deep, which is how the soft clamp was found.
test "acestep oracle: FSQ tokenizer hints + indices match reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const mean_p = std.mem.span(std.c.getenv("ACESTEP_VAEENC_MEAN") orelse return error.SkipZigTest);
    const hints_p = std.mem.span(std.c.getenv("ACESTEP_TOK_HINTS") orelse return error.SkipZigTest);
    const idx_p = std.mem.span(std.c.getenv("ACESTEP_TOK_INDICES") orelse return error.SkipZigTest);
    const z_p = std.mem.span(std.c.getenv("ACESTEP_TOK_Z") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const mean_d = try readF32(io, a, mean_p); // [1,64,T] channel-first
    defer a.free(mean_d);
    const ref_h = try readF32(io, a, hints_p);
    defer a.free(ref_h);
    const ref_i = try readI32(io, a, idx_p);
    defer a.free(ref_i);
    const ref_z = try readF32(io, a, z_p); // [T5,6] reference pre-quantization values
    defer a.free(ref_z);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const frames: c_int = @intCast(mean_d.len / 64);
    const mean_cf = mlx.mlx_array_new_data(mean_d.ptr, &[_]c_int{ 1, 64, frames }, 3, .float32);
    defer _ = mlx.mlx_array_free(mean_cf);
    const mean_nlc = try transpose(mean_cf, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(mean_nlc);
    const toks = try audioTokenize(e, a, mean_nlc, s);
    defer toks.deinit();
    var zc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zc);
    try mlx.check(mlx.mlx_contiguous(&zc, toks.z, false, s));
    _ = mlx.mlx_array_eval(zc);
    const dz = mlx.mlx_array_data_float32(zc) orelse return error.NoData;
    var ic = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ic);
    try mlx.check(mlx.mlx_contiguous(&ic, toks.indices, false, s));
    _ = mlx.mlx_array_eval(ic);
    const n_idx: usize = @intCast(mlx.mlx_array_size(ic));
    const di = mlx.mlx_array_data_int32(ic) orelse return error.NoData;
    const levels = &e.cfg.fsq_levels;
    var mismatches: usize = 0;
    var deep_flips: usize = 0;
    var worst_dist: f32 = 0;
    for (0..n_idx) |i| {
        if (di[i] == ref_i[i]) continue;
        mismatches += 1;
        for (0..levels.len) |d| {
            if (fsqDigit(di[i], levels, d) == fsqDigit(ref_i[i], levels, d)) continue;
            const zr = ref_z[i * levels.len + d];
            const dist = fsqBoundaryDistance(zr, levels[d]);
            std.debug.print("[acestep-tok] flip tok {d} dim {d}: ref z={d:.5} ours z={d:.5} (L={d}, boundary dist {d:.4} cells)\n", .{ i, d, zr, dz[i * levels.len + d], levels[d], dist });
            worst_dist = @max(worst_dist, dist);
            if (dist >= 0.05) deep_flips += 1;
        }
    }
    var max_dz: f32 = 0;
    for (0..ref_z.len) |k| max_dz = @max(max_dz, @abs(dz[k] - ref_z[k]));
    std.debug.print("[acestep-tok] max |z - z_ref| over all dims = {d:.5}\n", .{max_dz});
    const corr = try arrayCosine(toks.hints, ref_h, s);
    std.debug.print("[acestep-tok] corr={d:.6} codes={d} mismatched={d} deep_flips={d} worst_flip_dist={d:.4}\n", .{ corr, n_idx, mismatches, deep_flips, worst_dist });
    try testing.expectEqual(ref_i.len, n_idx);
    try testing.expectEqual(@as(usize, 0), deep_flips);
    try testing.expect(corr > 0.99);
}

// Oracle 9: detokenizer. Hints → 25 Hz latents [1,50,64]. It is CONCATENATED
// into the DiT's context stream, so the bar is cosine AND rms ratio (the
// cosine-cannot-see-scale rule).
test "acestep oracle: FSQ detokenizer matches reference (cos + scale)" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const hints_p = std.mem.span(std.c.getenv("ACESTEP_TOK_HINTS") orelse return error.SkipZigTest);
    const det_p = std.mem.span(std.c.getenv("ACESTEP_DETOK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const hints_d = try readF32(io, a, hints_p);
    defer a.free(hints_d);
    const ref = try readF32(io, a, det_p); // [1,T,64]
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const t5: c_int = @intCast(hints_d.len / 2048);
    const h_f32 = mlx.mlx_array_new_data(hints_d.ptr, &[_]c_int{ 1, t5, 2048 }, 3, .float32);
    defer _ = mlx.mlx_array_free(h_f32);
    const det = try audioDetokenize(e, a, h_f32, s);
    defer _ = mlx.mlx_array_free(det);
    const corr = try arrayCosine(det, ref, s);
    var dc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dc);
    try mlx.check(mlx.mlx_contiguous(&dc, det, false, s));
    _ = mlx.mlx_array_eval(dc);
    const d = mlx.mlx_array_data_float32(dc) orelse return error.NoData;
    var ss_ours: f64 = 0;
    var ss_ref: f64 = 0;
    for (0..ref.len) |i| {
        ss_ours += @as(f64, d[i]) * d[i];
        ss_ref += @as(f64, ref[i]) * ref[i];
    }
    const rms_ratio = std.math.sqrt(ss_ours / ss_ref);
    std.debug.print("[acestep-detok] corr={d:.6} rms_ratio={d:.4}\n", .{ corr, rms_ratio });
    try testing.expect(corr > 0.999);
    try testing.expect(rms_ratio > 0.97 and rms_ratio < 1.03);
}

// Oracle 10: the cover DiT velocity — context [detok | ones] + the COVER
// instruction's conditioning, one forward at t=1.0 from injected noise.
test "acestep oracle: cover DiT velocity matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const cond_p = std.mem.span(std.c.getenv("ACESTEP_COVER_COND") orelse return error.SkipZigTest);
    const det_p = std.mem.span(std.c.getenv("ACESTEP_DETOK") orelse return error.SkipZigTest);
    const noise_p = std.mem.span(std.c.getenv("ACESTEP_COVER_NOISE") orelse return error.SkipZigTest);
    const v1_p = std.mem.span(std.c.getenv("ACESTEP_COVER_V1") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const cond_d = try readF32(io, a, cond_p);
    defer a.free(cond_d);
    const det_d = try readF32(io, a, det_p);
    defer a.free(det_d);
    const noise_d = try readF32(io, a, noise_p);
    defer a.free(noise_d);
    const ref = try readF32(io, a, v1_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const frames: c_int = @intCast(noise_d.len / 64);
    const cond_len: c_int = @intCast(cond_d.len / 2048);
    const cond_f32 = mlx.mlx_array_new_data(cond_d.ptr, &[_]c_int{ 1, cond_len, 2048 }, 3, .float32);
    defer _ = mlx.mlx_array_free(cond_f32);
    const cond_bf = try astype(cond_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(cond_bf);
    var cross = try e.conditionSet(a, cond_bf, s);
    defer cross.deinit(a);
    const noise = mlx.mlx_array_new_data(noise_d.ptr, &[_]c_int{ 1, frames, 64 }, 3, .float32);
    defer _ = mlx.mlx_array_free(noise);
    const det = mlx.mlx_array_new_data(det_d.ptr, &[_]c_int{ 1, frames, 64 }, 3, .float32);
    defer _ = mlx.mlx_array_free(det);
    const ctx = try e.contextLatents(det, s);
    defer _ = mlx.mlx_array_free(ctx);
    const v1 = try ditForward(e, a, noise, 1.0, ctx, &cross, s);
    defer _ = mlx.mlx_array_free(v1);
    const corr = try arrayCosine(v1, ref, s);
    std.debug.print("[acestep-cover-dit] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.995);
}

// Oracle 11: the WHOLE cover pipeline on a real clip — reference source latent
// → OUR tokenize → detokenize → context, OUR condition set from the reference
// cover text hidden + the fixture lyrics → 8-step Euler + DCW from injected
// noise, against the reference's final latents. Also reports how close each
// cover sits to the SOURCE latent — the "does a cover follow the song" number.
test "acestep oracle: cover e2e denoise matches reference" {
    _ = std.c.getenv("ACESTEP_TEST_MODEL") orelse return error.SkipZigTest;
    const src_p = std.mem.span(std.c.getenv("ACESTEP_COVER_SRC") orelse return error.SkipZigTest);
    const th_p = std.mem.span(std.c.getenv("ACESTEP_COVER_E2E_TEXT_HIDDEN") orelse return error.SkipZigTest);
    const le_p = std.mem.span(std.c.getenv("ACESTEP_LYRIC_EMBEDS") orelse return error.SkipZigTest);
    const noise_p = std.mem.span(std.c.getenv("ACESTEP_COVER_E2E_NOISE") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("ACESTEP_COVER_E2E_LATENTS") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_d = try readF32(io, a, src_p); // [1,T,64]
    defer a.free(src_d);
    const th = try readF32(io, a, th_p);
    defer a.free(th);
    const le = try readF32(io, a, le_p);
    defer a.free(le);
    const noise_d = try readF32(io, a, noise_p);
    defer a.free(noise_d);
    const ref = try readF32(io, a, out_p);
    defer a.free(ref);

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;
    const frames: c_int = @intCast(src_d.len / 64);
    const text_len: c_int = @intCast(th.len / 1024);
    const lyric_len: c_int = @intCast(le.len / 1024);
    const th_f32 = mlx.mlx_array_new_data(th.ptr, &[_]c_int{ 1, text_len, 1024 }, 3, .float32);
    defer _ = mlx.mlx_array_free(th_f32);
    const th_bf = try astype(th_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(th_bf);
    const le_f32 = mlx.mlx_array_new_data(le.ptr, &[_]c_int{ 1, lyric_len, 1024 }, 3, .float32);
    defer _ = mlx.mlx_array_free(le_f32);
    const le_bf = try astype(le_f32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(le_bf);
    const timbre = try e.silenceSlice(e.cfg.timbre_fix_frame, s);
    defer _ = mlx.mlx_array_free(timbre);
    const cond = try buildConditioning(e, a, th_bf, le_bf, timbre, s);
    defer _ = mlx.mlx_array_free(cond);
    var cross = try e.conditionSet(a, cond, s);
    defer cross.deinit(a);

    const src = mlx.mlx_array_new_data(src_d.ptr, &[_]c_int{ 1, frames, 64 }, 3, .float32);
    defer _ = mlx.mlx_array_free(src);
    const toks = try audioTokenize(e, a, src, s);
    defer toks.deinit();
    const det = try audioDetokenize(e, a, toks.hints, s);
    defer _ = mlx.mlx_array_free(det);
    const crop = try sliceA(det, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, frames, 64 }, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(crop);
    const ctx = try e.contextLatents(crop, s);
    defer _ = mlx.mlx_array_free(ctx);

    var xt = mlx.mlx_array_new_data(noise_d.ptr, &[_]c_int{ 1, frames, 64 }, 3, .float32);
    defer _ = mlx.mlx_array_free(xt);
    var sched: [NUM_STEPS]f32 = undefined;
    timestepSchedule(&sched, SHIFT);
    for (0..NUM_STEPS) |step| {
        const t_curr = sched[step];
        const vt = try ditForward(e, a, xt, t_curr, ctx, &cross, s);
        defer _ = mlx.mlx_array_free(vt);
        const step_size = if (step == NUM_STEPS - 1) t_curr else t_curr - sched[step + 1];
        const dv = try mulScalar(vt, step_size, s);
        defer _ = mlx.mlx_array_free(dv);
        const x_next = try subA(xt, dv, s);
        defer _ = mlx.mlx_array_free(x_next);
        const vtt = try mulScalar(vt, t_curr, s);
        defer _ = mlx.mlx_array_free(vtt);
        const denoised = try subA(xt, vtt, s);
        defer _ = mlx.mlx_array_free(denoised);
        const corrected = try applyDcwDouble(x_next, denoised, t_curr, s);
        _ = mlx.mlx_array_free(xt);
        xt = corrected;
        _ = mlx.mlx_array_eval(xt);
    }
    const corr = try arrayCosine(xt, ref, s);
    const ours_vs_src = try arrayCosine(xt, src_d, s);
    const ref_vs_src = cosine(ref, src_d);
    const det_vs_src = try arrayCosine(crop, src_d, s);
    std.debug.print("[acestep-cover-e2e] frames={d} corr(ours,ref)={d:.6} cos(ours,src)={d:.4} cos(ref,src)={d:.4} cos(detok,src)={d:.4}\n", .{ frames, corr, ours_vs_src, ref_vs_src, det_vs_src });
    try testing.expect(corr > 0.95);
}

test "acestep instrumental flag reaches the same reference marker as empty lyrics" {
    const a = std.testing.allocator;
    // `instrumental: true` is the EXPLICIT spelling of what an empty lyric
    // block already meant here, so the two must assemble byte-identically —
    // otherwise the flag would be a second, silently different code path.
    const flagged = try formatLyrics(a, "en", resolveLyrics(true, ""));
    defer a.free(flagged);
    const empty = try formatLyrics(a, "en", "");
    defer a.free(empty);
    try std.testing.expectEqualStrings(empty, flagged);
    try std.testing.expectEqualStrings("# Languages\nen\n\n# Lyric\n[Instrumental]<|endoftext|>", flagged);
    try std.testing.expectEqualStrings("[verse]\nhi", resolveLyrics(false, "[verse]\nhi"));
}
