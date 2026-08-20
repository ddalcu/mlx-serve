//! MiniMax Music 3 — hierarchical autoregressive text-to-music engine
//! (`.audio` modality, `music3` arm of gen.AudioBackend).
//!
//! NOT an ACE-Step variant: the LLM's HIDDEN STATES (not tokens) drive a
//! flow-matching decoder. Pipeline (minimax_music_engine_plan.md; reference =
//! diffusers minimax-music3-integration @ dafe3733, ~/claude-tmp/music3/ref):
//!   caption+lyrics → byte-contract prompt → Qwen3-8B global LLM (AR at 25 Hz,
//!   batch 2 for CFG 1.5, top-k 50) emits one semantic code per frame; a 0.6B
//!   depth decoder autoregressively samples 7 residual codes per frame; the 8
//!   per-frame hidden states (1 global + 7 depth) feed a condition encoder
//!   (softmax mix → Conv1d 4096→2048 → nearest resample 25→86.13 Hz) that
//!   conditions a 2.4B flow-matching DiT (timestep as a PREPENDED TOKEN, not
//!   AdaLN; reversed SwiGLU; partial RoPE dims 32 theta 10000), denoised in
//!   200-frame windows (hop 100, per-step overlap blend, CFG 1.7 vs a ZERO
//!   condition), then a Snake/DAC Flow-VAE vocoder → 44.1 kHz stereo WAV.
//!
//! Pack: scripts/convert_music3_weights.py — affine group-64 quantized linears
//! (packed uint32 + bf16 scales/biases, widths solved per weight from
//! geometry, NEVER a literal), gather tables + vocoder + condition encoder
//! dense. The engine owes the structural transforms the converter left out:
//! vocoder weight-norm fusion, PT→MLX conv axis swaps, Snake alpha reshape.
//!
//! Numerics: LLM/depth/DiT compute bf16 (f32 DiT norm/bias params narrowed to
//! bf16 at load — the reference pipeline loads the whole transformer bf16);
//! latents/sampler state f32; condition encoder + vocoder f32 (VAE-class).
//! Parity: env-gated MUSIC3_* oracles fed by tests/dump_music3_fixtures.py.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const tok_mod = @import("tokenizer.zig");
const wav_mod = @import("wav.zig");
const sse = @import("gen_sse.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

// ── checkpoint contract (single-member family; converted config mirrors) ────
pub const Cfg = struct {
    // global LLM (Qwen3 8B)
    lm_layers: u32 = 36,
    lm_hidden: u32 = 4096,
    lm_heads: u32 = 32,
    lm_kv_heads: u32 = 8,
    lm_head_dim: u32 = 128,
    lm_vocab: u32 = 200000,
    lm_rope_theta: f32 = 1_000_000.0,
    eps: f32 = 1e-6,
    // depth decoder (0.6B, 4L, hd 256)
    dd_layers: u32 = 4,
    dd_heads: u32 = 16,
    // pipeline
    num_codebooks: u32 = 8,
    audio_vocab: u32 = 1024,
    semantic_vocab: u32 = 16384,
    frame_rate: u32 = 25,
    max_frames_cap: u32 = 9000,
    max_text_tokens: u32 = 5000,
    sample_rate: u32 = 44100,
    // flow DiT
    dit_layers: u32 = 36,
    dit_hidden: u32 = 2048,
    dit_heads: u32 = 32,
    dit_head_dim: u32 = 64,
    dit_rotary: u32 = 32,
    dit_in_ch: u32 = 128,
    dit_cond: u32 = 2048,
    dit_fourier: u32 = 256,
    dit_rope_theta: f32 = 10000.0,
};

pub const AUDIO_END: i32 = 151670;
pub const AUDIO_CFG_TOKEN: i32 = 151654;
pub const CODE_OFFSET: i32 = 151675;
const CFG_AR: f32 = 1.5;
const CFG_DIT: f32 = 1.7;
const TOP_K: c_int = 50;
const NEG_MASK: f32 = -1e9;
const CHUNK_FRAMES: u32 = 200;
const CHUNK_HOP: u32 = 100;
const OVERLAP_LATENT: u32 = 172;
const CROP_LEFT_LATENT: u32 = 86;
const CROP_RIGHT_LATENT: u32 = 344 - 86;
const VOC_HOP: u32 = 512; // samples per latent frame (8*8*4*2)
/// The lyric block an instrumental request sends.
///
/// `is_instrumental` is a convenience on MiniMax's HOSTED api — the open
/// weights have no such parameter (their own end-to-end script,
/// `scripts/end_to_end/minimax_ttm_test.py`, posts nothing but lyrics text),
/// and the checkpoint's lyric vocabulary is plain BPE with no special token
/// for it. So the flag can only become a lyric TAG, and the tag has to be one
/// the model was trained on. MiniMaxAI/MiniMax-Music3's model card lists the
/// section tags verbatim: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`,
/// `[Post-Chorus]`, `[Bridge]`, `[Instrumental]`, `[Solo]`, `[Outro]`.
///
/// The two MiniMax sources DISAGREE on the spelling, and neither documents a
/// "whole track has no vocals" tag at all — both list theirs among SECTION
/// markers, beside `[Bridge]` and `[Solo]`:
///   - the open-weights model card (our exact weights): `[Instrumental]`
///   - the hosted api reference: `[Inst]`, with `is_instrumental` a SEPARATE
///     boolean, which means their product does preprocessing we cannot see.
/// So this is a best-effort mapping, not a verified contract. `[Instrumental]`
/// wins the default because it is the card for the checkpoint we actually run
/// — and it is ALSO ACE-Step's existing marker, so one spelling serves both
/// backends. `MLX_SERVE_MUSIC3_INST_TAG` overrides it so the alternative can be
/// A/B'd by ear without a rebuild; the honest test is listening, not a fixture.
pub const INSTRUMENTAL_LYRICS = "[Instrumental]";

/// The marker actually sent, honoring the env override.
pub fn instrumentalMarker() []const u8 {
    const raw = std.c.getenv("MLX_SERVE_MUSIC3_INST_TAG") orelse return markerFromEnvValue(null);
    return markerFromEnvValue(std.mem.span(raw));
}

/// Split out so the override's three arms are testable without an environment.
/// `none` sends an EMPTY lyric body — the hosted api makes `lyrics` optional
/// under `is_instrumental`, and `[start]` alone is the closest this prompt
/// template gets to sending nothing. Unset or blank keeps the default: an
/// override that emptied the block by accident would read as a broken flag.
pub fn markerFromEnvValue(raw: ?[]const u8) []const u8 {
    const v = std.mem.trim(u8, raw orelse return INSTRUMENTAL_LYRICS, " \t\r\n");
    if (v.len == 0) return INSTRUMENTAL_LYRICS;
    if (std.ascii.eqlIgnoreCase(v, "none")) return "";
    return v;
}

/// The clause appended to an instrumental request's caption.
///
/// MiniMax's api marks `prompt` REQUIRED for an instrumental track while making
/// `lyrics` optional, which says the CAPTION carries the no-vocals intent and
/// the lyric tag was never meant to carry it alone. Measured 2026-08-18:
/// `[Instrumental]` by itself sang no words but still produced vocal texture —
/// exactly what a SECTION tag would do.
pub const INSTRUMENTAL_CAPTION_CLAUSE = "Instrumental only: no vocals, no singing, no lyrics.";

/// Only guards against stacking OUR OWN clause twice.
///
/// This used to fuzzy-match "instrumental" / "no vocal" / "without vocal" /
/// "no singing" anywhere in the caption, on the theory that appending on top
/// repeats the user back at the model. That was wrong, and it silently
/// disabled the feature for exactly the people using it: a real caption opens
/// "Instrumental ambient field-recording piece, freely paced, no vocals" —
/// where "Instrumental" is the GENRE, not an instruction — so the guard
/// matched, the clause never fired, and every take was secretly a tag-only
/// take. Measured on a live session 2026-08-19: `caption_facts=false` on nine
/// consecutive runs the user believed were testing the clause.
///
/// A short clause repeated near a user's own wording is harmless, and emphasis
/// may even help. Duplicating our exact sentence is the only real waste.
pub fn captionMentionsNoVocals(caption: []const u8) bool {
    return captionContainsAny(caption, &.{INSTRUMENTAL_CAPTION_CLAUSE});
}

/// Default ON. `MLX_SERVE_MUSIC3_INST_CAPTION=0` sends the tag alone, which is
/// the arm that was measured to leave vocal texture in.
pub fn instrumentalCaptionEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_MUSIC3_INST_CAPTION") orelse return true;
    return raw[0] != '0';
}

fn captionContainsAny(caption: []const u8, needles: []const []const u8) bool {
    for (needles) |needle| {
        if (caption.len < needle.len) continue;
        var i: usize = 0;
        while (i + needle.len <= caption.len) : (i += 1) {
            if (std.ascii.eqlIgnoreCase(caption[i .. i + needle.len], needle)) return true;
        }
    }
    return false;
}

/// The caption, plus the facts this request carries that the caption does not
/// already state.
///
/// Music 3 has no `bpm` or `keyscale` request field — but it is NOT missing
/// tempo and key. MiniMaxAI/MiniMax-Music3's card lists both under Global
/// Metadata and its own example caption reads
/// `Genre: acoustic pop. BPM: 96. Key: C major.`, so these are caption TEXT
/// here where they are conditioning fields on ACE-Step. Same knobs, different
/// channel — which is why the pane can offer them on both engines.
///
/// Every clause is skipped when the user already said it: appending on top
/// repeats them back to the model and spends caption budget. Order is fixed so
/// a seed stays reproducible across runs. Returns an OWNED copy always, so the
/// caller frees exactly one thing whether or not anything was added.
pub fn captionWithFacts(
    a: std.mem.Allocator,
    caption: []const u8,
    bpm: ?u32,
    keyscale: []const u8,
    instrumental: bool,
) ![]u8 {
    var facts: std.ArrayList(u8) = .empty;
    defer facts.deinit(a);

    if (instrumental and !captionMentionsNoVocals(caption))
        try facts.appendSlice(a, INSTRUMENTAL_CAPTION_CLAUSE);
    if (bpm) |b| {
        if (!captionContainsAny(caption, &.{ "bpm", "beats per minute" })) {
            if (facts.items.len != 0) try facts.append(a, ' ');
            var nb: [32]u8 = undefined;
            try facts.appendSlice(a, std.fmt.bufPrint(&nb, "BPM: {d}.", .{b}) catch unreachable);
        }
    }
    const key = std.mem.trim(u8, keyscale, " \t\r\n");
    if (key.len != 0 and !captionContainsAny(caption, &.{ "key:", " major", " minor" })) {
        if (facts.items.len != 0) try facts.append(a, ' ');
        try facts.appendSlice(a, "Key: ");
        try facts.appendSlice(a, key);
        try facts.append(a, '.');
    }

    if (facts.items.len == 0) return a.dupe(u8, caption);
    return std.fmt.allocPrint(a, "{s}\n{s}", .{ caption, facts.items });
}

/// The lyrics the engine should condition on. An instrumental request replaces
/// whatever the client sent with the canonical marker; everything else passes
/// through untouched. Empty lyrics WITHOUT the flag stay empty — the handler
/// 400s on those, so a client that simply forgot the field never gets a
/// wordless track it did not ask for.
pub fn resolveLyrics(instrumental: bool, lyrics: []const u8) []const u8 {
    return if (instrumental) instrumentalMarker() else lyrics;
}

pub const DEFAULT_STEPS: u32 = 30;
pub const MIN_DURATION_S: u32 = 1;
pub const MAX_DURATION_S: u32 = 360; // 9000 frames / 25 Hz

// ════════════════════════════════════════════════════════════════════════
// Pure helpers — hermetic tests at the bottom.
// ════════════════════════════════════════════════════════════════════════

/// Latent frames for `frames` LLM frames: int(frames * 44100/24000 * 960/512)
/// = frames*441/128 truncated, floor 1. Also the 25 Hz → 86.13 Hz resampler
/// ratio.
pub fn latentLen(frames: u32) u32 {
    return @max(1, (frames * 441) / 128);
}

/// Denoise window starts: [0] when everything fits one window, else a
/// 100-frame hop stopping before the last 100 (python range(0, F-100, 100)).
pub fn chunkStarts(a: std.mem.Allocator, frames: u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    if (frames <= CHUNK_FRAMES) {
        try out.append(a, 0);
    } else {
        var start: u32 = 0;
        while (start < frames - CHUNK_HOP) : (start += CHUNK_HOP) try out.append(a, start);
    }
    return out.toOwnedSlice(a);
}

/// Flow time for Euler step i of n: sigmas linspace(1, 1/n, n) inverted by the
/// scheduler (invert_sigmas) to t_i = i/n ascending, 0 = pure noise; every
/// step advances by exactly 1/n toward the appended terminal sigma 1.0.
pub fn flowTime(i: u32, n: u32) f32 {
    return @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
}

/// Nearest-neighbour resample index (torch F.interpolate "nearest"):
/// src = floor(dst * in / out).
pub fn nearestIdx(dst: u32, in_len: u32, out_len: u32) u32 {
    return @intCast((@as(u64, dst) * in_len) / out_len);
}

/// CFG prompt row: everything except the first token and the LAST TWO
/// (python ids[1:-2] = <|audio_cfg|>).
pub fn buildUncondIds(ids: []const i32, out: []i32) void {
    @memcpy(out, ids);
    if (ids.len < 4) return;
    for (1..ids.len - 2) |i| out[i] = AUDIO_CFG_TOKEN;
}

// ════════════════════════════════════════════════════════════════════════
// Prompt assembly — byte contract with the reference (_clean_caption /
// _normalize_lyrics in encoders.py). Hand-rolled scanners replicating the
// exact regex semantics; table-tested against outputs of the real Python.
// Unicode caveat: tag lowercasing is ASCII-only (structural tags are ASCII).
// ════════════════════════════════════════════════════════════════════════

fn isPyWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}

/// `<\|([^|]*)\|>` → "k is v" (two ws-split parts) or the stripped inner.
/// On a failed match the scan advances one char past '<' (regex retry).
fn rewriteSpecialTags(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        const open = std.mem.indexOfPos(u8, text, i, "<|") orelse {
            try out.appendSlice(a, text[i..]);
            break;
        };
        var k = open + 2;
        while (k < text.len and text[k] != '|') k += 1;
        if (k + 1 < text.len and text[k] == '|' and text[k + 1] == '>') {
            try out.appendSlice(a, text[i..open]);
            const inner = std.mem.trim(u8, text[open + 2 .. k], " \t\n\r\x0b\x0c");
            var split: ?usize = null;
            for (inner, 0..) |c, ci| {
                if (isPyWs(c)) {
                    split = ci;
                    break;
                }
            }
            if (split) |sp| {
                var rest = sp;
                while (rest < inner.len and isPyWs(inner[rest])) rest += 1;
                try out.appendSlice(a, inner[0..sp]);
                try out.appendSlice(a, " is ");
                try out.appendSlice(a, inner[rest..]);
            } else {
                try out.appendSlice(a, inner);
            }
            i = k + 2;
        } else {
            try out.append(a, '<');
            i = open + 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// `^\s{0,3}#{1,6}\s+` — 0-3 leading ws, 1-6 hashes, ≥1 ws (run consumed).
fn stripHeading(line: []const u8) []const u8 {
    var n: usize = 0;
    while (n < line.len and n < 4 and isPyWs(line[n])) n += 1;
    if (n > 3 or n >= line.len or line[n] != '#') return line;
    var h: usize = 0;
    while (n + h < line.len and line[n + h] == '#') h += 1;
    if (h > 6) return line;
    var j = n + h;
    if (j >= line.len or !isPyWs(line[j])) return line;
    while (j < line.len and isPyWs(line[j])) j += 1;
    return line[j..];
}

/// `^\s*[*+-]\s+` then `^\s*\*\s+` (the reference applies both).
fn stripBullet(line: []const u8, star_only: bool) []const u8 {
    var n: usize = 0;
    while (n < line.len and isPyWs(line[n])) n += 1;
    if (n >= line.len) return line;
    const c = line[n];
    const is_bullet = if (star_only) c == '*' else (c == '*' or c == '+' or c == '-');
    if (!is_bullet) return line;
    var j = n + 1;
    if (j >= line.len or !isPyWs(line[j])) return line;
    while (j < line.len and isPyWs(line[j])) j += 1;
    return line[j..];
}

/// One global pass of `\*\*([^*]+)\*\*` → content. Returns null when nothing
/// matched (fixpoint reached).
fn boldPass(a: std.mem.Allocator, line: []const u8) !?[]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var matched = false;
    var i: usize = 0;
    var p: usize = 0;
    while (p + 1 < line.len) {
        if (line[p] != '*' or line[p + 1] != '*') {
            p += 1;
            continue;
        }
        var k = p + 2;
        while (k < line.len and line[k] != '*') k += 1;
        if (k > p + 2 and k + 1 < line.len and line[k] == '*' and line[k + 1] == '*') {
            try out.appendSlice(a, line[i..p]);
            try out.appendSlice(a, line[p + 2 .. k]);
            i = k + 2;
            p = k + 2;
            matched = true;
        } else {
            p += 1;
        }
    }
    if (!matched) {
        out.deinit(a);
        return null;
    }
    try out.appendSlice(a, line[i..]);
    return try out.toOwnedSlice(a);
}

/// `(?<!\*)\*([^*\n]+)\*(?!\*)` → content, one global pass.
fn italicPass(a: std.mem.Allocator, line: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    var p: usize = 0;
    while (p < line.len) {
        if (line[p] != '*' or (p > 0 and line[p - 1] == '*')) {
            p += 1;
            continue;
        }
        var k = p + 1;
        while (k < line.len and line[k] != '*') k += 1;
        if (k > p + 1 and k < line.len and (k + 1 >= line.len or line[k + 1] != '*')) {
            try out.appendSlice(a, line[i..p]);
            try out.appendSlice(a, line[p + 1 .. k]);
            i = k + 1;
            p = k + 1;
        } else {
            p += 1;
        }
    }
    try out.appendSlice(a, line[i..]);
    return out.toOwnedSlice(a);
}

fn rstripWs(line: []const u8) []const u8 {
    var e = line.len;
    while (e > 0 and isPyWs(line[e - 1])) e -= 1;
    return line[0..e];
}

/// `^\s*[-*_]{3,}\s*$` MULTILINE → "". `\s` crosses newlines, so a match can
/// absorb a whitespace-only line before the rule and the newline after it;
/// the trailing `\s*$` greedily ends at the LAST newline inside its ws run.
fn removeRules(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var copy_from: usize = 0;
    var p: usize = 0; // candidate match start (a line start)
    while (p < text.len) {
        if (p >= copy_from) {
            var q = p;
            while (q < text.len and isPyWs(text[q])) q += 1;
            var r = q;
            while (r < text.len and (text[r] == '-' or text[r] == '*' or text[r] == '_')) r += 1;
            if (r - q >= 3) {
                var e = r;
                var last_nl: ?usize = null;
                while (e < text.len and isPyWs(text[e])) : (e += 1) {
                    if (text[e] == '\n') last_nl = e;
                }
                const match_end: ?usize = if (e == text.len)
                    e
                else if (last_nl) |nl|
                    nl
                else if (text[r] == '\n' or r == text.len)
                    r
                else
                    null;
                if (match_end) |me| {
                    try out.appendSlice(a, text[copy_from..p]);
                    copy_from = me;
                }
            }
        }
        // advance to the next line start
        const nl = std.mem.indexOfScalarPos(u8, text, p, '\n') orelse break;
        p = nl + 1;
    }
    if (copy_from < text.len) try out.appendSlice(a, text[copy_from..]);
    return out.toOwnedSlice(a);
}

fn replaceAll(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, text, i, from)) |at| {
        try out.appendSlice(a, text[i..at]);
        try out.appendSlice(a, to);
        i = at + from.len;
    }
    try out.appendSlice(a, text[i..]);
    return out.toOwnedSlice(a);
}

/// `\n{2,}` → "\n".
fn collapseNewlines(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        try out.append(a, text[i]);
        if (text[i] == '\n') {
            var j = i + 1;
            while (j < text.len and text[j] == '\n') j += 1;
            i = j;
        } else {
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// Python str.splitlines boundaries (ASCII subset: \n, \r, \r\n, \v, \f).
fn isLineBreak(c: u8) bool {
    return c == '\n' or c == '\r' or c == 0x0b or c == 0x0c;
}

/// The reference's per-line caption pass: heading → bullet → star-bullet →
/// bold fixpoint → italic → rstrip, lines rejoined with '\n'.
fn cleanCaptionLines(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    var first = true;
    while (i <= text.len) {
        var j = i;
        while (j < text.len and !isLineBreak(text[j])) j += 1;
        if (i == text.len and !first) break; // trailing break: splitlines drops the empty tail
        const raw = text[i..j];
        if (!first) try out.append(a, '\n');
        first = false;

        var line: []const u8 = stripBullet(stripBullet(stripHeading(raw), false), true);
        var owned: ?[]u8 = null;
        defer if (owned) |o| a.free(o);
        while (std.mem.indexOf(u8, line, "**") != null) {
            const next = try boldPass(a, line) orelse break;
            if (owned) |o| a.free(o);
            owned = next;
            line = next;
        }
        const italics = try italicPass(a, line);
        defer a.free(italics);
        try out.appendSlice(a, rstripWs(italics));

        if (j >= text.len) break;
        i = if (text[j] == '\r' and j + 1 < text.len and text[j + 1] == '\n') j + 2 else j + 1;
    }
    return out.toOwnedSlice(a);
}

pub fn cleanCaption(a: std.mem.Allocator, caption: []const u8) ![]u8 {
    const tagged = try rewriteSpecialTags(a, caption);
    defer a.free(tagged);
    const lines = try cleanCaptionLines(a, tagged);
    defer a.free(lines);
    const ruled = try removeRules(a, lines);
    defer a.free(ruled);
    const no_dot = try replaceAll(a, ruled, "\xe2\x80\xa2 ", "");
    defer a.free(no_dot);
    const no_indent = try replaceAll(a, no_dot, "    ", "");
    defer a.free(no_indent);
    return collapseNewlines(a, no_indent);
}

/// `\[([^\]]+)\]` → lowercased inner (ASCII).
fn lowercaseTags(a: std.mem.Allocator, text: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] != '[') {
            try out.append(a, text[i]);
            i += 1;
            continue;
        }
        var k = i + 1;
        while (k < text.len and text[k] != ']') k += 1;
        if (k < text.len and k > i + 1) {
            try out.append(a, '[');
            for (text[i + 1 .. k]) |c| try out.append(a, std.ascii.toLower(c));
            try out.append(a, ']');
            i = k + 1;
        } else {
            try out.append(a, '[');
            i += 1;
        }
    }
    return out.toOwnedSlice(a);
}

/// A line starting with `[tag]` runs keeps ONLY the tags (text on the line is
/// dropped): `^[ \t]*((?:\[[^\]]+\][ \t]*)+)` → group(1).strip().
fn keepLeadingTags(line: []const u8) []const u8 {
    var n: usize = 0;
    while (n < line.len and (line[n] == ' ' or line[n] == '\t')) n += 1;
    var p = n;
    var last_close: ?usize = null;
    while (p < line.len and line[p] == '[') {
        var k = p + 1;
        while (k < line.len and line[k] != ']') k += 1;
        if (k >= line.len or k == p + 1) break; // no close / empty content
        last_close = k;
        p = k + 1;
        while (p < line.len and (line[p] == ' ' or line[p] == '\t')) p += 1;
    }
    if (last_close) |lc| return line[n .. lc + 1];
    return line;
}

pub fn normalizeLyrics(a: std.mem.Allocator, lyrics: []const u8) ![]u8 {
    var joined: std.ArrayList(u8) = .empty;
    defer joined.deinit(a);
    var it = std.mem.splitScalar(u8, lyrics, '\n');
    var first = true;
    while (it.next()) |line| {
        if (!first) try joined.append(a, '\n');
        first = false;
        try joined.appendSlice(a, keepLeadingTags(line));
    }
    const r1 = try replaceAll(a, joined.items, "] ", "]\n");
    defer a.free(r1);
    const r2 = try replaceAll(a, r1, " [", "\n[");
    defer a.free(r2);
    const r3 = try replaceAll(a, r2, " ^ ", "\n");
    defer a.free(r3);
    const lowered = try lowercaseTags(a, r3);
    defer a.free(lowered);
    return std.fmt.allocPrint(a, "[start]\n{s}", .{lowered});
}

/// The checkpoint's prompt template. NO chat template; whitespace is a token
/// contract — the fixture pins the assembled bytes.
pub fn assemblePrompt(a: std.mem.Allocator, caption: []const u8, lyrics: []const u8) ![]u8 {
    const cap = try cleanCaption(a, caption);
    defer a.free(cap);
    const lyr = try normalizeLyrics(a, lyrics);
    defer a.free(lyr);
    return std.fmt.allocPrint(
        a,
        "<|im_start|><|caption_start|>{s}<|caption_end|><|lyrics_start|>{s}<|lyrics_end|><|im_end|><|audio_start|>",
        .{ cap, lyr },
    );
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
fn divA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, x, y, s));
    return o;
}
/// Scalar op in x's OWN dtype (a bare f32 scalar array would promote bf16
/// operands to f32 — the MageFlow scalarLike rule).
fn scalarLike(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    return astype(c, mlx.mlx_array_dtype(x), s);
}
fn mulScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = try scalarLike(x, v, s);
    defer _ = mlx.mlx_array_free(c);
    return mulA(x, c, s);
}
fn sliceA(x: mlx.mlx_array, start: []const c_int, stop: []const c_int, s: S) !mlx.mlx_array {
    var strides: [8]c_int = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, start.ptr, start.len, stop.ptr, stop.len, strides[0..start.len].ptr, start.len, s));
    return o;
}
fn sliceUpdateA(dst: mlx.mlx_array, src: mlx.mlx_array, start: []const c_int, stop: []const c_int, s: S) !mlx.mlx_array {
    var strides: [8]c_int = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice_update(&o, dst, src, start.ptr, start.len, stop.ptr, stop.len, strides[0..start.len].ptr, start.len, s));
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
fn concatN(items: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (items) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
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
fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}
fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, eps, s));
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
fn ropeAt(x: mlx.mlx_array, dims: c_int, theta: f32, offset: c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&o, x, dims, false, mlx.mlx_optional_float.some(theta), 1.0, offset, .{ .ctx = null }, s));
    return o;
}
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mode: [*:0]const u8, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, null_a, null_a, s));
    return o;
}
fn zerosA(shape: []const c_int, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, s));
    return o;
}
fn broadcastTo(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, s));
    return o;
}
fn i32Arr(vals: []const i32) mlx.mlx_array {
    const sh = [_]c_int{@intCast(vals.len)};
    return mlx.mlx_array_new_data(vals.ptr, &sh, 1, .int32);
}
fn takeRows(table: mlx.mlx_array, ids: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&o, table, ids, 0, s));
    return o;
}
/// Materialize a slice/transpose that outlives its parent (the sliceContig
/// rule): contiguous + eval breaks the graph edge to the parent buffer.
fn materialize(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    _ = mlx.mlx_array_eval(o);
    return o;
}
fn evalA(x: mlx.mlx_array) void {
    _ = mlx.mlx_array_eval(x);
}
fn readScalarI32(arr: mlx.mlx_array) !i32 {
    evalA(arr);
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, arr));
    return v;
}

fn getW(w: *const Weights, key: []const u8) !mlx.mlx_array {
    return w.get(key) orelse {
        log.err("[music3] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingMusic3Weight;
    };
}

/// Linear over a Weights map: quantized (sibling `.scales`; width solved from
/// packed geometry via `affineParamsFromGeometry` — never a literal) or dense
/// (lazy-transpose matmul). Adds `.bias` when present.
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
        const qp = transformer_mod.affineParamsFromGeometry(wq, scales, in_features) orelse {
            log.err("[music3] unsolvable quant geometry for {s}\n", .{prefix});
            return error.BadQuantGeometry;
        };
        try mlx.check(mlx.mlx_quantized_matmul(&o, x, wq, scales, qb, true, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", s));
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

/// One linear's handles + quant params resolved ONCE at load (the per-frame
/// allocPrint + hashmap + geometry-solve tax in `lin` was real CPU time).
/// Handles are BORROWED from the owning Weights map — freed only via the map.
const QLin = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array = .{ .ctx = null },
    bi: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array = .{ .ctx = null },
    bits: u32 = 0,
    gs: u32 = 0,

    /// Same ops as `lin` (bit-identical by construction).
    fn forward(self: *const QLin, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        if (self.sc.ctx != null) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w, self.sc, self.bi, true, mlx.mlx_optional_int.some(@intCast(self.gs)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            const axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wt, self.w, &axes, 2, s));
            try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
        }
        if (self.bias.ctx != null) {
            const r = try addA(o, self.bias, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
};

fn rowsOf(w: mlx.mlx_array) u32 {
    return @intCast(mlx.getShape(w)[0]);
}

/// Resolve `prefix`.{weight,scales,biases,bias} once. `in_features` is the
/// contraction width — quant params cannot be solved from geometry alone
/// (bits/group_size are ambiguous without it).
fn resolveQLin(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32) !QLin {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
    defer a.free(sk);
    const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
    defer a.free(bk);
    const ak = try std.fmt.allocPrint(a, "{s}.bias", .{prefix});
    defer a.free(ak);

    var q = QLin{ .w = try getW(w, wk) };
    if (w.get(sk)) |scales| {
        q.sc = scales;
        q.bi = try getW(w, bk);
        const qp = transformer_mod.affineParamsFromGeometry(q.w, scales, in_features) orelse {
            log.err("[music3] unsolvable quant geometry for {s}\n", .{prefix});
            return error.BadQuantGeometry;
        };
        q.bits = qp.bits;
        q.gs = qp.group_size;
    }
    if (w.get(ak)) |bias| q.bias = bias;
    return q;
}

const LmLayerW = struct {
    in_ln: mlx.mlx_array,
    pa_ln: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    gate: QLin,
    up: QLin,
    down: QLin,
};

const DepthLayerW = struct {
    in_ln: mlx.mlx_array,
    pa_ln: mlx.mlx_array,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    gate: QLin,
    up: QLin,
    down: QLin,
};

const DitBlockW = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    ff_in: QLin,
    ff_out: QLin,
};

fn conv1dW(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, stride: c_int, padding: c_int, dilation: c_int, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const weight = try getW(w, wk);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&o, x, weight, stride, padding, dilation, 1, s));
    const bk = try std.fmt.allocPrint(a, "{s}.bias", .{prefix});
    defer a.free(bk);
    if (w.get(bk)) |b| {
        const r = try addA(o, b, s);
        _ = mlx.mlx_array_free(o);
        o = r;
    }
    return o;
}

fn convT1dW(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, stride: c_int, padding: c_int, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const weight = try getW(w, wk);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv_transpose1d(&o, x, weight, stride, padding, 1, 0, 1, s));
    const bk = try std.fmt.allocPrint(a, "{s}.bias", .{prefix});
    defer a.free(bk);
    if (w.get(bk)) |b| {
        const r = try addA(o, b, s);
        _ = mlx.mlx_array_free(o);
        o = r;
    }
    return o;
}

/// Deterministic per-draw keys: seed advances by a golden-ratio stride per
/// draw (semantic + each depth code + each window's noise).
const Sampler = struct {
    seed: u64,
    ctr: u64 = 0,
    greedy: bool = false,

    fn nextKey(self: *Sampler) mlx.mlx_array {
        var k = mlx.mlx_array_new();
        _ = mlx.mlx_random_key(&k, self.seed +% self.ctr *% 0x9E3779B97F4A7C15);
        self.ctr += 1;
        return k;
    }
};

/// Top-k sample (reference `_sample_top_k`): keep the k largest logits,
/// everything else → -1e9 (softmax-zero), then categorical (or argmax in
/// greedy mode). `mlx_topk` values are unsorted, so the floor is their min.
fn sampleTopK(logits: mlx.mlx_array, smp: *Sampler, s: S) !mlx.mlx_array {
    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_topk_axis(&top, logits, TOP_K, -1, s));
    var floor = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(floor);
    try mlx.check(mlx.mlx_min_axis(&floor, top, -1, true, s));
    var below = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(below);
    try mlx.check(mlx.mlx_less(&below, logits, floor, s));
    const neg = try scalarLike(logits, NEG_MASK, s);
    defer _ = mlx.mlx_array_free(neg);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, below, neg, logits, s));

    var pick = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pick);
    if (smp.greedy) {
        try mlx.check(mlx.mlx_argmax_axis(&pick, masked, -1, false, s));
    } else {
        const key = smp.nextKey();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_categorical(&pick, masked, -1, key, s));
    }
    return astype(pick, .int32, s);
}

// ════════════════════════════════════════════════════════════════════════
// Global LLM (Qwen3 8B, batch 2) with a preallocated KV cache.
// ════════════════════════════════════════════════════════════════════════

const LmKv = struct {
    ks: []mlx.mlx_array,
    vs: []mlx.mlx_array,
    len: c_int = 0,
    cap: c_int,

    fn init(allocator: std.mem.Allocator, cfg: Cfg, cap: u32, s: S) !LmKv {
        return initShape(allocator, cfg.lm_layers, cfg.lm_kv_heads, cap, cfg.lm_head_dim, s);
    }

    fn initShape(allocator: std.mem.Allocator, layers: u32, kv_heads: u32, cap: u32, head_dim: u32, s: S) !LmKv {
        const ks = try allocator.alloc(mlx.mlx_array, layers);
        errdefer allocator.free(ks);
        const vs = try allocator.alloc(mlx.mlx_array, layers);
        errdefer allocator.free(vs);
        const shape = [_]c_int{ 2, @intCast(kv_heads), @intCast(cap), @intCast(head_dim) };
        for (ks, vs) |*k, *v| {
            k.* = try zerosA(&shape, .bfloat16, s);
            v.* = try zerosA(&shape, .bfloat16, s);
        }
        return .{ .ks = ks, .vs = vs, .cap = @intCast(cap) };
    }

    fn deinit(self: *LmKv, allocator: std.mem.Allocator) void {
        for (self.ks) |k| _ = mlx.mlx_array_free(k);
        for (self.vs) |v| _ = mlx.mlx_array_free(v);
        allocator.free(self.ks);
        allocator.free(self.vs);
    }

    /// Write `new` [2,KV,T,hd] at position `len` (buffer donated in place on
    /// the free-after-update pattern) and return the readable [0..len+T) view.
    fn append(self: *LmKv, li: usize, is_k: bool, new: mlx.mlx_array, s: S) !void {
        const buf = if (is_k) &self.ks[li] else &self.vs[li];
        const sh = mlx.getShape(buf.*);
        const t = mlx.getShape(new)[2];
        const start = [_]c_int{ 0, 0, self.len, 0 };
        const stop = [_]c_int{ sh[0], sh[1], self.len + t, sh[3] };
        const updated = try sliceUpdateA(buf.*, new, &start, &stop, s);
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }

    fn view(self: *const LmKv, li: usize, is_k: bool, upto: c_int, s: S) !mlx.mlx_array {
        const buf = if (is_k) self.ks[li] else self.vs[li];
        const sh = mlx.getShape(buf);
        return sliceA(buf, &[_]c_int{ 0, 0, 0, 0 }, &[_]c_int{ sh[0], sh[1], upto, sh[3] }, s);
    }
};

/// Forward [2,T,4096] embeddings through the 36 layers, appending to the KV
/// cache; returns the LAST position's post-norm hidden [2,4096] bf16
/// (= HF last_hidden_state[:, -1]).
fn lmForward(e: *const Engine, embeds: mlx.mlx_array, kv: *LmKv) !mlx.mlx_array {
    const cfg = e.cfg;
    const s = e.s;

    const t_len: c_int = mlx.getShape(embeds)[1];
    const nh: c_int = @intCast(cfg.lm_heads);
    const nkv: c_int = @intCast(cfg.lm_kv_heads);
    const hd: c_int = @intCast(cfg.lm_head_dim);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.lm_head_dim)));
    const prefill = t_len > 1;

    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&h, embeds));
    for (e.lm_lw, 0..) |*lw, li| {
        const x = try rmsNorm(h, lw.in_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(x);

        const q = try lw.q.forward(x, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try lw.k.forward(x, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try lw.v.forward(x, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try splitHeads(q, nh, hd, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k, nkv, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v, nkv, hd, s);
        defer _ = mlx.mlx_array_free(vh);

        const qn = try rmsNorm(qh, lw.q_norm, cfg.eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const kn = try rmsNorm(kh, lw.k_norm, cfg.eps, s);
        defer _ = mlx.mlx_array_free(kn);

        const qr = try ropeAt(qn, hd, cfg.lm_rope_theta, kv.len, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try ropeAt(kn, hd, cfg.lm_rope_theta, kv.len, s);
        defer _ = mlx.mlx_array_free(kr);

        try kv.append(li, true, kr, s);
        try kv.append(li, false, vh, s);
        const kview = try kv.view(li, true, kv.len + t_len, s);
        defer _ = mlx.mlx_array_free(kview);
        const vview = try kv.view(li, false, kv.len + t_len, s);
        defer _ = mlx.mlx_array_free(vview);

        const attn = try sdpa(qr, kview, vview, scale, if (prefill) "causal" else "", s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try lw.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(h, o, s);
        _ = mlx.mlx_array_free(h);

        const xm = try rmsNorm(h1, lw.pa_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(xm);
        const gate = try lw.gate.forward(xm, s);
        defer _ = mlx.mlx_array_free(gate);
        const up = try lw.up.forward(xm, s);
        defer _ = mlx.mlx_array_free(up);
        const gact = try silu(gate, s);
        defer _ = mlx.mlx_array_free(gact);
        const gu = try mulA(gact, up, s);
        defer _ = mlx.mlx_array_free(gu);
        const down = try lw.down.forward(gu, s);
        defer _ = mlx.mlx_array_free(down);
        h = try addA(h1, down, s);
        _ = mlx.mlx_array_free(h1);

        if (prefill) evalA(h);
    }
    kv.len += t_len;

    const last = try sliceA(h, &[_]c_int{ 0, t_len - 1, 0 }, &[_]c_int{ 2, t_len, @intCast(cfg.lm_hidden) }, s);
    _ = mlx.mlx_array_free(h);
    defer _ = mlx.mlx_array_free(last);
    const norm_w = try getW(&e.lm_w, "model.norm.weight");
    const normed = try rmsNorm(last, norm_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(normed);
    return reshape(normed, &[_]c_int{ 2, @intCast(cfg.lm_hidden) }, s);
}

/// lm_head(last_hidden) as f32 [2,V] (full vocab — the parity-oracle path;
/// the hot loop uses the pruned head below when built).
fn lmHeadLogits(e: *const Engine, allocator: std.mem.Allocator, last_hidden: mlx.mlx_array) !mlx.mlx_array {
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const logits = try lin(&e.lm_w, arena_inst.allocator(), last_hidden, "lm_head", e.s);
    defer _ = mlx.mlx_array_free(logits);
    return astype(logits, .float32, e.s);
}

/// Pruned-head row index → vocab id. Rows are the allowed-output set in
/// order: [code 0 .. code sem_vocab-1, <|audio_end|>].
pub fn prunedRowToVocab(idx: i32, sem_vocab: i32) i32 {
    return if (idx < sem_vocab) idx + CODE_OFFSET else AUDIO_END;
}

/// The semantic head sliced to the 16385 rows a frame can legally emit
/// (semantic codes + <|audio_end|>). Rows of a per-row-group affine-quantized
/// weight slice bit-exact (the gatherQuantizedRows precedent), so the pruned
/// matmul IS the full matmul restricted to the allowed set — the additive
/// vocab mask disappears. ~67 MB beside the full head (which stays in the
/// map for the parity oracle).
const PrunedHead = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    gs: u32,
    rows: u32,

    fn deinit(self: *PrunedHead) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.sc);
        _ = mlx.mlx_array_free(self.bi);
    }
};

/// Slice rows [lo,hi) ++ row `end_row` out of a [V, cols] tensor, materialized
/// (slice-born weights into quantized_matmul rule).
fn pruneHeadRows(src: mlx.mlx_array, lo: c_int, hi: c_int, end_row: c_int, s: S) !mlx.mlx_array {
    const cols = mlx.getShape(src)[1];
    const code = try sliceA(src, &[_]c_int{ lo, 0 }, &[_]c_int{ hi, cols }, s);
    defer _ = mlx.mlx_array_free(code);
    const end = try sliceA(src, &[_]c_int{ end_row, 0 }, &[_]c_int{ end_row + 1, cols }, s);
    defer _ = mlx.mlx_array_free(end);
    const cat = try concat2(code, end, 0, s);
    defer _ = mlx.mlx_array_free(cat);
    return materialize(cat, s);
}

/// Build the pruned head at load. Null (= full-vocab + mask path) when the
/// kill switch MLX_SERVE_MUSIC3_LMHEAD_PRUNE=0 is set or the head is not
/// affine-quantized.
fn buildPrunedHead(cfg: Cfg, lm_w: *const Weights, s: S) !?PrunedHead {
    if (std.c.getenv("MLX_SERVE_MUSIC3_LMHEAD_PRUNE")) |v| {
        if (v[0] == '0') return null;
    }
    const wq = lm_w.get("lm_head.weight") orelse return null;
    const sc = lm_w.get("lm_head.scales") orelse return null;
    const bi = lm_w.get("lm_head.biases") orelse return null;
    const qp = transformer_mod.affineParamsFromGeometry(wq, sc, cfg.lm_hidden) orelse return null;
    const lo: c_int = CODE_OFFSET;
    const hi: c_int = CODE_OFFSET + @as(c_int, @intCast(cfg.semantic_vocab));
    const pw = try pruneHeadRows(wq, lo, hi, AUDIO_END, s);
    errdefer _ = mlx.mlx_array_free(pw);
    const psc = try pruneHeadRows(sc, lo, hi, AUDIO_END, s);
    errdefer _ = mlx.mlx_array_free(psc);
    const pbi = try pruneHeadRows(bi, lo, hi, AUDIO_END, s);
    const rows = cfg.semantic_vocab + 1;
    log.info("[music3] lm-head prune engaged ({d} rows)\n", .{rows});
    return .{ .w = pw, .sc = psc, .bi = pbi, .bits = qp.bits, .gs = qp.group_size, .rows = rows };
}

/// MLX_SERVE_MUSIC3_DEPTH_BITS=<2..6>: re-quantize the depth-decoder LAYER
/// linears from the 8-bit pack at load (the dflash load-time-requant
/// precedent). Experiment, default OFF — UNLIKE dflash this is a QUALITY
/// lever (depth codes are output, not verified drafts): adopt only if greedy
/// replay agreement vs the 8-bit arm AND listening both hold.
fn maybeRequantDepth(w: *Weights, cfg: Cfg, s: S) !void {
    const raw = std.c.getenv("MLX_SERVE_MUSIC3_DEPTH_BITS") orelse return;
    const bits = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch return;
    if (bits == 8) return; // the pack width — nothing to do
    switch (bits) {
        2, 3, 4, 5, 6 => {},
        else => return,
    }

    var bases: std.ArrayList([]u8) = .empty;
    defer {
        for (bases.items) |b| w.allocator.free(b);
        bases.deinit(w.allocator);
    }
    var it = w.map.keyIterator();
    while (it.next()) |kp| {
        const key = kp.*;
        if (!std.mem.startsWith(u8, key, "layers.")) continue;
        if (!std.mem.endsWith(u8, key, ".weight")) continue;
        try bases.append(w.allocator, try w.allocator.dupe(u8, key[0 .. key.len - ".weight".len]));
    }

    var count: u32 = 0;
    for (bases.items) |base| {
        var kb1: [256]u8 = undefined;
        const wk = try std.fmt.bufPrint(&kb1, "{s}.weight", .{base});
        var kb2: [256]u8 = undefined;
        const sk = try std.fmt.bufPrint(&kb2, "{s}.scales", .{base});
        var kb3: [256]u8 = undefined;
        const bk = try std.fmt.bufPrint(&kb3, "{s}.biases", .{base});
        const wq = try getW(w, wk);
        const sc = w.get(sk) orelse continue; // norms + dense stay
        const bi = try getW(w, bk);
        // in_features = contraction width: lm_hidden everywhere except
        // down_proj (rows of the sibling up_proj).
        const in_features: u32 = blk: {
            if (std.mem.endsWith(u8, base, "down_proj")) {
                var kb4: [256]u8 = undefined;
                const upk = try std.fmt.bufPrint(&kb4, "{s}up_proj.weight", .{base[0 .. base.len - "down_proj".len]});
                break :blk rowsOf(try getW(w, upk));
            }
            break :blk cfg.lm_hidden;
        };
        const qp = transformer_mod.affineParamsFromGeometry(wq, sc, in_features) orelse continue;
        if (qp.bits <= bits) continue; // only ever narrow
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_dequantize(&dense, wq, sc, bi, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", .{}, mlx.mlx_optional_dtype{}, s));
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
        var nw = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(nw);
        var nsc = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(nsc);
        var nbi = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(nbi);
        try mlx.check(mlx.mlx_vector_array_get(&nw, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&nsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&nbi, triple, 2));
        evalA(nw);
        evalA(nsc);
        evalA(nbi);
        try putWeight(w, wk, nw);
        try putWeight(w, sk, nsc);
        try putWeight(w, bk, nbi);
        count += 1;
    }
    log.info("[music3] depth requant engaged: {d}-bit ({d} weights)\n", .{ bits, count });
}

/// Pruned-head logits [2, rows] f32.
fn lmHeadLogitsPruned(e: *const Engine, ph: *const PrunedHead, last_hidden: mlx.mlx_array) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(&o, last_hidden, ph.w, ph.sc, ph.bi, true, mlx.mlx_optional_int.some(@intCast(ph.gs)), mlx.mlx_optional_int.some(@intCast(ph.bits)), "affine", e.s));
    defer _ = mlx.mlx_array_free(o);
    return astype(o, .float32, e.s);
}

/// Additive vocab mask [V] f32: 0 on the semantic-code range + <|audio_end|>,
/// -1e9 everywhere else (finite, so CFG arithmetic can't make NaN — the
/// reference re-masks three times to fix that up; we never produce it).
fn buildVocabMask(e: *const Engine, allocator: std.mem.Allocator) !mlx.mlx_array {
    const v = e.cfg.lm_vocab;
    const buf = try allocator.alloc(f32, v);
    defer allocator.free(buf);
    @memset(buf, NEG_MASK);
    const lo: usize = @intCast(CODE_OFFSET);
    for (lo..lo + e.cfg.semantic_vocab) |i| buf[i] = 0;
    buf[@intCast(AUDIO_END)] = 0;
    const sh = [_]c_int{@intCast(v)};
    return mlx.mlx_array_new_data(buf.ptr, &sh, 1, .float32);
}

/// CFG-guided semantic logits [1,V]: mask both rows, guide at 1.5, restrict
/// to the CONDITIONAL row's top-50 (threshold from cond, strictly-less mask,
/// ties kept — reference order).
fn guidedSemanticLogits(e: *const Engine, logits_f32: mlx.mlx_array, mask: ?mlx.mlx_array) !mlx.mlx_array {
    const s = e.s;
    const v: c_int = mlx.getShape(logits_f32)[1]; // full vocab OR pruned rows
    const c_raw = try sliceA(logits_f32, &[_]c_int{ 0, 0 }, &[_]c_int{ 1, v }, s);
    defer _ = mlx.mlx_array_free(c_raw);
    const u_raw = try sliceA(logits_f32, &[_]c_int{ 1, 0 }, &[_]c_int{ 2, v }, s);
    defer _ = mlx.mlx_array_free(u_raw);
    // Pruned head: every row is legal, the mask is the identity — skip it.
    var cond = mlx.mlx_array_new();
    if (mask) |m| try mlx.check(mlx.mlx_add(&cond, c_raw, m, s)) else try mlx.check(mlx.mlx_array_set(&cond, c_raw));
    defer _ = mlx.mlx_array_free(cond);
    var uncond = mlx.mlx_array_new();
    if (mask) |m| try mlx.check(mlx.mlx_add(&uncond, u_raw, m, s)) else try mlx.check(mlx.mlx_array_set(&uncond, u_raw));
    defer _ = mlx.mlx_array_free(uncond);

    const diff = try subA(cond, uncond, s);
    defer _ = mlx.mlx_array_free(diff);
    const scaled = try mulScalar(diff, CFG_AR, s);
    defer _ = mlx.mlx_array_free(scaled);
    const guided = try addA(uncond, scaled, s);
    defer _ = mlx.mlx_array_free(guided);

    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_topk_axis(&top, cond, TOP_K, -1, s));
    var floor = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(floor);
    try mlx.check(mlx.mlx_min_axis(&floor, top, -1, true, s));
    var below = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(below);
    try mlx.check(mlx.mlx_less(&below, cond, floor, s));
    const neg = try scalarLike(guided, NEG_MASK, s);
    defer _ = mlx.mlx_array_free(neg);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&out, below, neg, guided, s));
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Depth decoder (4L causal, 16 heads hd 256, absolute pos embedding).
// ════════════════════════════════════════════════════════════════════════

/// Depth forward. `kv == null` is the reference shape: full re-forward over
/// the ≤8-position sequence. With `kv`, `seq` is only the NEW tail — cached
/// K/V cover the prefix, and the absolute position for the pos-embedding rows
/// is `kv.len` (no RoPE, no GQA here, so the cache is trivially correct).
/// Returns the LAST position's post-norm hidden [2,4096] bf16.
fn depthForward(e: *const Engine, seq: mlx.mlx_array, kv: ?*LmKv) !mlx.mlx_array {
    const cfg = e.cfg;
    const s = e.s;

    const n: c_int = mlx.getShape(seq)[1];
    const dim: c_int = @intCast(cfg.lm_hidden);
    const nh: c_int = @intCast(cfg.dd_heads);
    const hd = @divExact(dim, nh); // 256
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(@as(u32, @intCast(hd)))));

    const pos_off: c_int = if (kv) |c| c.len else 0;
    const pos_table = try getW(&e.dd_w, "pos_embedding.weight");
    const pos = try sliceA(pos_table, &[_]c_int{ pos_off, 0 }, &[_]c_int{ pos_off + n, dim }, s);
    defer _ = mlx.mlx_array_free(pos);
    var h = try addA(seq, pos, s);

    for (e.dd_lw, 0..) |*lw, li| {
        const x = try rmsNorm(h, lw.in_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(x);
        const q = try lw.q.forward(x, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try lw.k.forward(x, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try lw.v.forward(x, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try splitHeads(q, nh, hd, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k, nh, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v, nh, hd, s);
        defer _ = mlx.mlx_array_free(vh);
        const attn = blk: {
            if (kv) |c| {
                try c.append(li, true, kh, s);
                try c.append(li, false, vh, s);
                const kview = try c.view(li, true, c.len + n, s);
                defer _ = mlx.mlx_array_free(kview);
                const vview = try c.view(li, false, c.len + n, s);
                defer _ = mlx.mlx_array_free(vview);
                break :blk try sdpa(qh, kview, vview, scale, if (n > 1) "causal" else "", s);
            }
            break :blk try sdpa(qh, kh, vh, scale, "causal", s);
        };
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try lw.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(h, o, s);
        _ = mlx.mlx_array_free(h);

        const xm = try rmsNorm(h1, lw.pa_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(xm);
        const gate = try lw.gate.forward(xm, s);
        defer _ = mlx.mlx_array_free(gate);
        const up = try lw.up.forward(xm, s);
        defer _ = mlx.mlx_array_free(up);
        const gact = try silu(gate, s);
        defer _ = mlx.mlx_array_free(gact);
        const gu = try mulA(gact, up, s);
        defer _ = mlx.mlx_array_free(gu);
        const down = try lw.down.forward(gu, s);
        defer _ = mlx.mlx_array_free(down);
        h = try addA(h1, down, s);
        _ = mlx.mlx_array_free(h1);
    }
    if (kv) |c| c.len += n;

    const last = try sliceA(h, &[_]c_int{ 0, n - 1, 0 }, &[_]c_int{ 2, n, dim }, s);
    _ = mlx.mlx_array_free(h);
    defer _ = mlx.mlx_array_free(last);
    const norm_w = try getW(&e.dd_w, "norm.weight");
    const normed = try rmsNorm(last, norm_w, cfg.eps, s);
    defer _ = mlx.mlx_array_free(normed);
    return reshape(normed, &[_]c_int{ 2, dim }, s);
}

/// projection() over rows [1|2, 4096] → [2,1,4096] (broadcast when needed).
fn depthProjectRow(e: *const Engine, row: mlx.mlx_array) !mlx.mlx_array {
    const p = try e.dd_proj.forward(row, e.s);
    defer _ = mlx.mlx_array_free(p);
    const dim: c_int = @intCast(e.cfg.lm_hidden);
    const sh = mlx.getShape(p);
    const p3 = try reshape(p, &[_]c_int{ sh[0], 1, dim }, e.s);
    if (sh[0] == 2) return p3;
    defer _ = mlx.mlx_array_free(p3);
    return broadcastTo(p3, &[_]c_int{ 2, 1, dim }, e.s);
}

const DepthResult = struct {
    /// c1..c7 as [1] i32 GPU arrays (never read to host on the hot path).
    codes: [7]mlx.mlx_array,
    /// concat of the 7 conditional-row hiddens: [1, 7*4096] bf16.
    depth_hidden: mlx.mlx_array,

    fn deinit(self: *DepthResult) void {
        for (self.codes) |c| _ = mlx.mlx_array_free(c);
        _ = mlx.mlx_array_free(self.depth_hidden);
    }
};

/// One frame's residual codes c1..c7 (reference `_generate_depth_codes`).
/// `force` (oracle replay) supplies the codes; the sampler's pick is still
/// computed and compared so the test can count agreement.
fn generateDepthCodes(
    e: *const Engine,
    allocator: std.mem.Allocator,
    last_hidden: mlx.mlx_array,
    sem_plus_offset: mlx.mlx_array,
    smp: *Sampler,
    force: ?[]const i32,
    agree: ?*u32,
    dkv: ?*LmKv,
) !DepthResult {
    const s = e.s;
    const dim: c_int = @intCast(e.cfg.lm_hidden);

    var seq: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (seq.items) |x| _ = mlx.mlx_array_free(x);
        seq.deinit(allocator);
    }
    try seq.append(allocator, try depthProjectRow(e, last_hidden));
    {
        const embed_table = try getW(&e.lm_w, "model.embed_tokens.weight");
        const row = try takeRows(embed_table, sem_plus_offset, s);
        defer _ = mlx.mlx_array_free(row);
        try seq.append(allocator, try depthProjectRow(e, row));
    }

    var codes: [7]mlx.mlx_array = undefined;
    var hidden_parts: [7]mlx.mlx_array = undefined;
    var done: usize = 0;
    errdefer for (0..done) |i| {
        _ = mlx.mlx_array_free(codes[i]);
        _ = mlx.mlx_array_free(hidden_parts[i]);
    };

    if (dkv) |c| c.len = 0; // fresh cache per frame (buffer reused)
    var index: u32 = 1;
    while (index < e.cfg.num_codebooks) : (index += 1) {
        // Cached path: forward only the not-yet-fed tail (2 rows on the first
        // step, the single new projected embed after).
        const h_last = blk: {
            if (dkv) |c| {
                const fresh = seq.items[@intCast(c.len)..];
                if (fresh.len == 1) break :blk try depthForward(e, fresh[0], c);
                const cat = try concatN(fresh, 1, s);
                defer _ = mlx.mlx_array_free(cat);
                break :blk try depthForward(e, cat, c);
            }
            const cat = try concatN(seq.items, 1, s);
            defer _ = mlx.mlx_array_free(cat);
            break :blk try depthForward(e, cat, null);
        }; // [2,4096]
        defer _ = mlx.mlx_array_free(h_last);
        const cond_h = try sliceA(h_last, &[_]c_int{ 0, 0 }, &[_]c_int{ 1, dim }, s);
        hidden_parts[index - 1] = cond_h;

        const logits_bf = try e.dd_heads[index - 1].forward(h_last, s);
        defer _ = mlx.mlx_array_free(logits_bf);
        const logits = try astype(logits_bf, .float32, s);
        defer _ = mlx.mlx_array_free(logits);
        const av: c_int = @intCast(e.cfg.audio_vocab);
        const c_row = try sliceA(logits, &[_]c_int{ 0, 0 }, &[_]c_int{ 1, av }, s);
        defer _ = mlx.mlx_array_free(c_row);
        const u_row = try sliceA(logits, &[_]c_int{ 1, 0 }, &[_]c_int{ 2, av }, s);
        defer _ = mlx.mlx_array_free(u_row);
        const diff = try subA(c_row, u_row, s);
        defer _ = mlx.mlx_array_free(diff);
        const scaled = try mulScalar(diff, CFG_AR, s);
        defer _ = mlx.mlx_array_free(scaled);
        const guided = try addA(u_row, scaled, s);
        defer _ = mlx.mlx_array_free(guided);

        var code = try sampleTopK(guided, smp, s);
        if (force) |f| {
            if (agree) |ag| {
                const ours = try readScalarI32(code);
                if (ours == f[index - 1]) ag.* += 1;
            }
            _ = mlx.mlx_array_free(code);
            code = i32Arr(f[index - 1 .. index]);
        }
        codes[index - 1] = code;
        done = index;

        if (index < e.cfg.num_codebooks - 1) {
            const bank_off = i32Arr(&[_]i32{@intCast((index - 1) * e.cfg.audio_vocab)});
            defer _ = mlx.mlx_array_free(bank_off);
            const shifted = try addA(code, bank_off, s);
            defer _ = mlx.mlx_array_free(shifted);
            const table = try getW(&e.dd_w, "audio_embeddings.weight");
            const row = try takeRows(table, shifted, s);
            defer _ = mlx.mlx_array_free(row);
            try seq.append(allocator, try depthProjectRow(e, row));
        }
    }

    const depth_hidden = try concatN(hidden_parts[0..7], 1, s);
    for (hidden_parts) |p| _ = mlx.mlx_array_free(p);
    return .{ .codes = codes, .depth_hidden = depth_hidden };
}

/// Per-frame feedback embedding (`_embed_audio_frame`): semantic row +
/// summed residual-bank rows, scaled by num_codebooks^-0.5 (load-bearing).
/// Rows are identical across the CFG batch, so compute once and broadcast.
fn embedAudioFrame(e: *const Engine, sem_plus_offset: mlx.mlx_array, codes: *const [7]mlx.mlx_array) !mlx.mlx_array {
    const s = e.s;
    const dim: c_int = @intCast(e.cfg.lm_hidden);
    const embed_table = try getW(&e.lm_w, "model.embed_tokens.weight");
    var acc = try takeRows(embed_table, sem_plus_offset, s); // [1,4096]
    const bank = try getW(&e.dd_w, "audio_embeddings.weight");
    for (codes, 0..) |code, i| {
        const bank_off = i32Arr(&[_]i32{@intCast(i * e.cfg.audio_vocab)});
        defer _ = mlx.mlx_array_free(bank_off);
        const shifted = try addA(code, bank_off, s);
        defer _ = mlx.mlx_array_free(shifted);
        const row = try takeRows(bank, shifted, s);
        defer _ = mlx.mlx_array_free(row);
        const next = try addA(acc, row, s);
        _ = mlx.mlx_array_free(acc);
        acc = next;
    }
    defer _ = mlx.mlx_array_free(acc);
    const scaled = try mulScalar(acc, 1.0 / @sqrt(@as(f32, @floatFromInt(e.cfg.num_codebooks))), s);
    defer _ = mlx.mlx_array_free(scaled);
    const r3 = try reshape(scaled, &[_]c_int{ 1, 1, dim }, s);
    defer _ = mlx.mlx_array_free(r3);
    return broadcastTo(r3, &[_]c_int{ 2, 1, dim }, s);
}

// ════════════════════════════════════════════════════════════════════════
// Condition encoder: softmax-mixed 8 hidden streams → Conv1d 4096→2048 →
// nearest resample onto the latent timeline. All f32 (dense checkpoint).
// ════════════════════════════════════════════════════════════════════════

/// `hiddens` [F, 32768] bf16 (frame-buffer rows) → condition [1, L, 2048] f32.
fn condEncode(e: *const Engine, allocator: std.mem.Allocator, hiddens: mlx.mlx_array) !mlx.mlx_array {
    const s = e.s;
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const frames: c_int = mlx.getShape(hiddens)[0];
    const dim: c_int = @intCast(e.cfg.lm_hidden);
    const nl: c_int = @intCast(e.cfg.num_codebooks);

    const hf = try astype(hiddens, .float32, s);
    defer _ = mlx.mlx_array_free(hf);
    const h4 = try reshape(hf, &[_]c_int{ 1, frames, nl, dim }, s);
    defer _ = mlx.mlx_array_free(h4);
    const weighted = try mulA(h4, e.mix, s); // mix [1,1,8,1] f32 (softmax × layer_scale)
    defer _ = mlx.mlx_array_free(weighted);
    var mixed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mixed);
    try mlx.check(mlx.mlx_sum_axis(&mixed, weighted, 2, false, s)); // [1,F,4096]

    const conv = try conv1dW(&e.ce_w, a, mixed, "proj", 1, 1, 1, s); // [1,F,2048]
    defer _ = mlx.mlx_array_free(conv);

    const out_len = latentLen(@intCast(frames));
    const idx = try allocator.alloc(i32, out_len);
    defer allocator.free(idx);
    for (idx, 0..) |*v, j| v.* = @intCast(nearestIdx(@intCast(j), @intCast(frames), out_len));
    const idx_arr = i32Arr(idx);
    defer _ = mlx.mlx_array_free(idx_arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&out, conv, idx_arr, 1, s));
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Flow-matching DiT. Timestep is a PREPENDED TOKEN (plain LayerNorms, no
// AdaLN — do not reach for acestep's); SwiGLU order REVERSED
// (gate_states * silu(gate)); partial RoPE dims 32 theta 10000 with the temb
// token at position 0; k=1 convs applied as residuals.
// ════════════════════════════════════════════════════════════════════════

/// Batch-agnostic DiT forward: latents [B,L,128] f32 + condition [B,L,2048]
/// f32 at flow time `t` → velocity [B,L,128] f32.
fn ditForward(e: *const Engine, allocator: std.mem.Allocator, lats: mlx.mlx_array, t: f32, conds: mlx.mlx_array) !mlx.mlx_array {
    const cfg = e.cfg;
    const s = e.s;
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const w = &e.dit_w;
    const batch: c_int = mlx.getShape(lats)[0];
    const l_len: c_int = mlx.getShape(lats)[1];
    const in_ch: c_int = @intCast(cfg.dit_in_ch);
    const dim: c_int = @intCast(cfg.dit_hidden);

    // [latent | zeros(in_ch) | condition] on channels — the zeros block is
    // upstream's shape, not a bug to fix.
    const zeros_ch = try zerosA(&[_]c_int{ batch, l_len, in_ch }, .float32, s);
    defer _ = mlx.mlx_array_free(zeros_ch);
    const stacked = try concatN(&[_]mlx.mlx_array{ lats, zeros_ch, conds }, 2, s);
    defer _ = mlx.mlx_array_free(stacked);

    const pre = try conv1dW(w, a, stacked, "preprocess_conv", 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(pre);
    const pre_res = try addA(pre, stacked, s);
    defer _ = mlx.mlx_array_free(pre_res);
    const xb = try astype(pre_res, .bfloat16, s);
    defer _ = mlx.mlx_array_free(xb);
    const proj = try e.dit_proj_in.forward(xb, s); // [2,L,2048] bf16
    defer _ = mlx.mlx_array_free(proj);

    // temb: Fourier features (host, trained weight) → linear_1 (f32) → SiLU →
    // linear_2 (quantized bf16); one extra token prepended to both rows.
    const half = cfg.dit_fourier / 2;
    var fbuf: [256]f32 = undefined;
    for (0..half) |i| {
        const ang = 2.0 * std.math.pi * t * e.fourier_w[i];
        fbuf[i] = @cos(ang);
        fbuf[half + i] = @sin(ang);
    }
    const fsh = [_]c_int{ 1, @intCast(cfg.dit_fourier) };
    const fourier = mlx.mlx_array_new_data(&fbuf, &fsh, 2, .float32);
    defer _ = mlx.mlx_array_free(fourier);
    const t1 = try e.dit_time1.forward(fourier, s);
    defer _ = mlx.mlx_array_free(t1);
    const t1a = try silu(t1, s);
    defer _ = mlx.mlx_array_free(t1a);
    const t1b = try astype(t1a, .bfloat16, s);
    defer _ = mlx.mlx_array_free(t1b);
    const temb = try e.dit_time2.forward(t1b, s); // [1,2048] bf16
    defer _ = mlx.mlx_array_free(temb);
    const temb3 = try reshape(temb, &[_]c_int{ 1, 1, dim }, s);
    defer _ = mlx.mlx_array_free(temb3);
    const temb2 = try broadcastTo(temb3, &[_]c_int{ batch, 1, dim }, s);
    defer _ = mlx.mlx_array_free(temb2);

    var h = try concat2(temb2, proj, 1, s); // [B,L+1,2048]

    const nh: c_int = @intCast(cfg.dit_heads);
    const hd: c_int = @intCast(cfg.dit_head_dim);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.dit_head_dim)));
    const ff: c_int = 8192;
    for (e.dit_bw, 0..) |*bw, li| {
        const n1 = try layerNorm(h, bw.n1w, bw.n1b, 1e-5, s);
        defer _ = mlx.mlx_array_free(n1);
        const q = try bw.q.forward(n1, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try bw.k.forward(n1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try bw.v.forward(n1, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try splitHeads(q, nh, hd, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k, nh, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v, nh, hd, s);
        defer _ = mlx.mlx_array_free(vh);
        const qr = try ropeAt(qh, @intCast(cfg.dit_rotary), cfg.dit_rope_theta, 0, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try ropeAt(kh, @intCast(cfg.dit_rotary), cfg.dit_rope_theta, 0, s);
        defer _ = mlx.mlx_array_free(kr);
        const attn = try sdpa(qr, kr, vh, scale, "", s); // bidirectional
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try bw.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(h, o, s);
        _ = mlx.mlx_array_free(h);

        const n2 = try layerNorm(h1, bw.n2w, bw.n2b, 1e-5, s);
        defer _ = mlx.mlx_array_free(n2);
        const ffin = try bw.ff_in.forward(n2, s); // [B,L+1,16384]
        defer _ = mlx.mlx_array_free(ffin);
        const t_all: c_int = mlx.getShape(ffin)[1];
        // REVERSED SwiGLU: first chunk = value states, second chunk = gate.
        const gate_states = try sliceA(ffin, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ batch, t_all, ff }, s);
        defer _ = mlx.mlx_array_free(gate_states);
        const gate = try sliceA(ffin, &[_]c_int{ 0, 0, ff }, &[_]c_int{ batch, t_all, 2 * ff }, s);
        defer _ = mlx.mlx_array_free(gate);
        const gact = try silu(gate, s);
        defer _ = mlx.mlx_array_free(gact);
        const prod = try mulA(gate_states, gact, s);
        defer _ = mlx.mlx_array_free(prod);
        const ffout = try bw.ff_out.forward(prod, s);
        defer _ = mlx.mlx_array_free(ffout);
        const h2 = try addA(h1, ffout, s);
        _ = mlx.mlx_array_free(h1);
        h = h2;

        if (li % 8 == 7) evalA(h);
    }
    defer _ = mlx.mlx_array_free(h);

    const body = try sliceA(h, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ batch, l_len + 1, dim }, s); // drop temb token
    defer _ = mlx.mlx_array_free(body);
    const body_f = try astype(body, .float32, s);
    defer _ = mlx.mlx_array_free(body_f);
    const out = try e.dit_proj_out.forward(body_f, s); // dense f32 [B,L,128]
    defer _ = mlx.mlx_array_free(out);
    const post = try conv1dW(w, a, out, "postprocess_conv", 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(post);
    return addA(post, out, s);
}

/// One CFG-guided velocity: batch 2 through the DiT — the unconditional row
/// conditions on ZEROS (not an empty prompt), guidance 1.7 (NOT the AR
/// stage's 1.5). latents/condition [1,L,*] f32 → guided velocity [1,L,128].
fn ditVelocity(e: *const Engine, allocator: std.mem.Allocator, latents: mlx.mlx_array, t: f32, cond: mlx.mlx_array) !mlx.mlx_array {
    const s = e.s;
    const l_len: c_int = mlx.getShape(latents)[1];
    const in_ch: c_int = @intCast(e.cfg.dit_in_ch);
    const lat2 = try concat2(latents, latents, 0, s);
    defer _ = mlx.mlx_array_free(lat2);
    const cond_zero = try zerosA(&[_]c_int{ 1, l_len, @intCast(e.cfg.dit_cond) }, .float32, s);
    defer _ = mlx.mlx_array_free(cond_zero);
    const cond2 = try concat2(cond, cond_zero, 0, s);
    defer _ = mlx.mlx_array_free(cond2);
    const vel2 = try ditForward(e, allocator, lat2, t, cond2);
    defer _ = mlx.mlx_array_free(vel2);

    const vc = try sliceA(vel2, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, l_len, in_ch }, s);
    defer _ = mlx.mlx_array_free(vc);
    const vu = try sliceA(vel2, &[_]c_int{ 1, 0, 0 }, &[_]c_int{ 2, l_len, in_ch }, s);
    defer _ = mlx.mlx_array_free(vu);
    const diff = try subA(vc, vu, s);
    defer _ = mlx.mlx_array_free(diff);
    const scaled = try mulScalar(diff, CFG_DIT, s);
    defer _ = mlx.mlx_array_free(scaled);
    return addA(vu, scaled, s);
}

// ════════════════════════════════════════════════════════════════════════
// Vocoder (Flow-VAE / DAC decoder). Weight norm fused + conv axes swapped at
// LOAD; Snake here is alpha-only (NO beta, NO exp — do not copy ACE-Step's).
// All f32.
// ════════════════════════════════════════════════════════════════════════

/// x + 1/(alpha + 1e-9) * sin(alpha*x)^2, alpha [C] over NLC channels-last.
fn snakeA(e: *const Engine, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, s: S) !mlx.mlx_array {
    const alpha = try getW(&e.voc_w, try std.fmt.allocPrint(a, "{s}.alpha", .{prefix}));
    const ax = try mulA(x, alpha, s);
    defer _ = mlx.mlx_array_free(ax);
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, ax, s));
    const sq = try mulA(sn, sn, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps_c = try scalarLike(alpha, 1e-9, s);
    defer _ = mlx.mlx_array_free(eps_c);
    const aeps = try addA(alpha, eps_c, s);
    defer _ = mlx.mlx_array_free(aeps);
    const frac = try divA(sq, aeps, s);
    defer _ = mlx.mlx_array_free(frac);
    return addA(x, frac, s);
}

fn vocResUnit(e: *const Engine, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, dilation: c_int, s: S) !mlx.mlx_array {
    const pad = @divTrunc((7 - 1) * dilation, 2);
    const s1 = try snakeA(e, a, x, try std.fmt.allocPrint(a, "{s}.snake1", .{prefix}), s);
    defer _ = mlx.mlx_array_free(s1);
    const c1 = try conv1dW(&e.voc_w, a, s1, try std.fmt.allocPrint(a, "{s}.conv1", .{prefix}), 1, pad, dilation, s);
    defer _ = mlx.mlx_array_free(c1);
    const s2 = try snakeA(e, a, c1, try std.fmt.allocPrint(a, "{s}.snake2", .{prefix}), s);
    defer _ = mlx.mlx_array_free(s2);
    const c2 = try conv1dW(&e.voc_w, a, s2, try std.fmt.allocPrint(a, "{s}.conv2", .{prefix}), 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(c2);
    return addA(x, c2, s);
}

const VOC_STRIDES = [_]c_int{ 8, 8, 4, 2 };

/// latents [1,L,128] f32 → stereo wave [2, L*512, 1] f32 (row 0 = left).
/// The 128 channels FOLD into two 64-channel streams decoded as a batch.
fn vocodeWindow(e: *const Engine, allocator: std.mem.Allocator, latents: mlx.mlx_array) !mlx.mlx_array {
    const s = e.s;
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const l_len: c_int = mlx.getShape(latents)[1];

    const ch_a = try sliceA(latents, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, l_len, 64 }, s);
    defer _ = mlx.mlx_array_free(ch_a);
    const ch_b = try sliceA(latents, &[_]c_int{ 0, 0, 64 }, &[_]c_int{ 1, l_len, 128 }, s);
    defer _ = mlx.mlx_array_free(ch_b);
    const folded = try concat2(ch_a, ch_b, 0, s); // [2,L,64]
    defer _ = mlx.mlx_array_free(folded);

    const pin = try conv1dW(&e.voc_w, a, folded, "dec_in_proj", 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(pin);
    var h = try conv1dW(&e.voc_w, a, pin, "conv_in", 1, 3, 1, s);

    for (VOC_STRIDES, 0..) |stride, bi| {
        const pfx = try std.fmt.allocPrint(a, "blocks.{d}", .{bi});
        const sn = try snakeA(e, a, h, try std.fmt.allocPrint(a, "{s}.snake1", .{pfx}), s);
        _ = mlx.mlx_array_free(h);
        const pad = @divTrunc(stride, 2) + @mod(stride, 2); // ceil(stride/2)
        const up = try convT1dW(&e.voc_w, a, sn, try std.fmt.allocPrint(a, "{s}.conv_t1", .{pfx}), stride, pad, s);
        _ = mlx.mlx_array_free(sn);
        const r1 = try vocResUnit(e, a, up, try std.fmt.allocPrint(a, "{s}.res_unit1", .{pfx}), 1, s);
        _ = mlx.mlx_array_free(up);
        const r2 = try vocResUnit(e, a, r1, try std.fmt.allocPrint(a, "{s}.res_unit2", .{pfx}), 3, s);
        _ = mlx.mlx_array_free(r1);
        const r3 = try vocResUnit(e, a, r2, try std.fmt.allocPrint(a, "{s}.res_unit3", .{pfx}), 9, s);
        _ = mlx.mlx_array_free(r2);
        h = r3;
        evalA(h);
        _ = arena_inst.reset(.retain_capacity);
    }

    const sn_out = try snakeA(e, a, h, "snake_out", s);
    _ = mlx.mlx_array_free(h);
    defer _ = mlx.mlx_array_free(sn_out);
    const conv_out = try conv1dW(&e.voc_w, a, sn_out, "conv_out", 1, 3, 1, s);
    defer _ = mlx.mlx_array_free(conv_out);
    var wave = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&wave, conv_out, s));
    return wave;
}

// ════════════════════════════════════════════════════════════════════════
// Engine
// ════════════════════════════════════════════════════════════════════════

/// Monotonic lap clock over std.Io (this Zig nightly has no std.time.Timer).
const LapClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,
    fn init(io: std.Io) LapClock {
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }
    /// µs since the previous lap (or init).
    fn lapUs(self: *LapClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d / 1_000;
    }
};

/// MUSIC3_COST_PROBE: per-stage laps with eval barriers. The barriers change
/// the schedule, so probe numbers are for ATTRIBUTION; A/B arms must both run
/// with the probe on (symmetric overhead).
fn costProbeEnabled() bool {
    return std.c.getenv("MUSIC3_COST_PROBE") != null;
}

/// Depth-decoder KV cache (default ON). MLX_SERVE_MUSIC3_DEPTH_KV=0 keeps
/// the naive full-re-forward path — it IS the reference shape.
fn depthKvEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_MUSIC3_DEPTH_KV") orelse return true;
    return raw[0] != '0';
}

pub const MusicRequest = struct {
    caption: []const u8,
    lyrics: []const u8 = "",
    duration_s: u32 = 60,
    seed: u64 = 0,
    steps: u32 = DEFAULT_STEPS,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    cfg: Cfg,
    lm_w: Weights,
    dd_w: Weights,
    dit_w: Weights,
    ce_w: Weights,
    voc_w: Weights,
    tok: tok_mod.Tokenizer,
    /// softmax(layer_weight_logits) × layer_scale, [1,1,8,1] f32 (load-time).
    mix: mlx.mlx_array,
    /// time_proj.weight [128] read to host at load (Fourier features are
    /// computed host-side per timestep).
    fourier_w: [128]f32,
    /// Sliced 16385-row semantic head (null = kill-switched or dense head).
    pruned_head: ?PrunedHead,
    /// Per-layer handles resolved once at load (borrowed from the maps).
    lm_lw: []LmLayerW,
    dd_lw: []DepthLayerW,
    dd_heads: [7]QLin,
    dd_proj: QLin,
    dit_bw: []DitBlockW,
    dit_proj_in: QLin,
    dit_proj_out: QLin,
    dit_time1: QLin,
    dit_time2: QLin,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.io = io;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.cfg = Cfg{}; // single-member family; converted config mirrors these

        self.lm_w = try loadFileWeights(allocator, model_dir, "language_model.safetensors");
        errdefer self.lm_w.deinit();
        self.dd_w = try loadFileWeights(allocator, model_dir, "rvq_depth_decoder.safetensors");
        errdefer self.dd_w.deinit();
        self.dit_w = try loadFileWeights(allocator, model_dir, "transformer.safetensors");
        errdefer self.dit_w.deinit();
        self.ce_w = try loadFileWeights(allocator, model_dir, "condition_encoder.safetensors");
        errdefer self.ce_w.deinit();
        self.voc_w = try loadFileWeights(allocator, model_dir, "vocoder.safetensors");
        errdefer self.voc_w.deinit();

        try fixupDit(&self.dit_w, self.s);
        try fuseVocoder(&self.voc_w, self.s);
        try swapConvAxes(&self.ce_w, "proj.weight", self.s);
        self.mix = try buildMix(&self.ce_w, self.s);
        errdefer _ = mlx.mlx_array_free(self.mix);
        try readFourier(&self.dit_w, &self.fourier_w);
        try maybeRequantDepth(&self.dd_w, self.cfg, self.s);
        self.pruned_head = try buildPrunedHead(self.cfg, &self.lm_w, self.s);
        errdefer if (self.pruned_head) |*ph| ph.deinit();
        try self.buildResolved();
        errdefer {
            self.allocator.free(self.lm_lw);
            self.allocator.free(self.dd_lw);
            self.allocator.free(self.dit_bw);
        }

        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        log.info("[music3] engine ready (lm {d} + depth {d} + dit {d} + cond {d} + vocoder {d} tensors)\n", .{
            self.lm_w.count(), self.dd_w.count(), self.dit_w.count(), self.ce_w.count(), self.voc_w.count(),
        });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.lm_w.deinit();
        self.dd_w.deinit();
        self.dit_w.deinit();
        self.ce_w.deinit();
        self.voc_w.deinit();
        _ = mlx.mlx_array_free(self.mix);
        if (self.pruned_head) |*ph| ph.deinit();
        self.allocator.free(self.lm_lw);
        self.allocator.free(self.dd_lw);
        self.allocator.free(self.dit_bw);
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    /// Resolve every hot-path linear/norm handle once (key strings live in a
    /// throwaway arena; the handles are borrowed from the maps).
    fn buildResolved(self: *Engine) !void {
        const cfg = self.cfg;
        var arena_inst = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_inst.deinit();
        const a = arena_inst.allocator();
        const hid = cfg.lm_hidden;

        self.lm_lw = try self.allocator.alloc(LmLayerW, cfg.lm_layers);
        errdefer self.allocator.free(self.lm_lw);
        for (self.lm_lw, 0..) |*lw, li| {
            const pfx = try std.fmt.allocPrint(a, "model.layers.{d}", .{li});
            lw.in_ln = try getW(&self.lm_w, try std.fmt.allocPrint(a, "{s}.input_layernorm.weight", .{pfx}));
            lw.pa_ln = try getW(&self.lm_w, try std.fmt.allocPrint(a, "{s}.post_attention_layernorm.weight", .{pfx}));
            lw.q_norm = try getW(&self.lm_w, try std.fmt.allocPrint(a, "{s}.self_attn.q_norm.weight", .{pfx}));
            lw.k_norm = try getW(&self.lm_w, try std.fmt.allocPrint(a, "{s}.self_attn.k_norm.weight", .{pfx}));
            lw.q = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.self_attn.q_proj", .{pfx}), hid);
            lw.k = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.self_attn.k_proj", .{pfx}), hid);
            lw.v = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.self_attn.v_proj", .{pfx}), hid);
            lw.o = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{pfx}), cfg.lm_heads * cfg.lm_head_dim);
            lw.gate = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.mlp.gate_proj", .{pfx}), hid);
            lw.up = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.mlp.up_proj", .{pfx}), hid);
            lw.down = try resolveQLin(&self.lm_w, a, try std.fmt.allocPrint(a, "{s}.mlp.down_proj", .{pfx}), rowsOf(lw.up.w));
        }

        self.dd_lw = try self.allocator.alloc(DepthLayerW, cfg.dd_layers);
        errdefer self.allocator.free(self.dd_lw);
        for (self.dd_lw, 0..) |*lw, li| {
            const pfx = try std.fmt.allocPrint(a, "layers.{d}", .{li});
            lw.in_ln = try getW(&self.dd_w, try std.fmt.allocPrint(a, "{s}.input_layernorm.weight", .{pfx}));
            lw.pa_ln = try getW(&self.dd_w, try std.fmt.allocPrint(a, "{s}.post_attention_layernorm.weight", .{pfx}));
            lw.q = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_q", .{pfx}), hid);
            lw.k = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_k", .{pfx}), hid);
            lw.v = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_v", .{pfx}), hid);
            lw.o = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_out", .{pfx}), hid);
            lw.gate = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.gate_proj", .{pfx}), hid);
            lw.up = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.up_proj", .{pfx}), hid);
            lw.down = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "{s}.down_proj", .{pfx}), rowsOf(lw.up.w));
        }
        for (&self.dd_heads, 0..) |*hq, i| {
            hq.* = try resolveQLin(&self.dd_w, a, try std.fmt.allocPrint(a, "audio_heads.{d}", .{i}), hid);
        }
        self.dd_proj = try resolveQLin(&self.dd_w, a, "projection", hid);

        self.dit_bw = try self.allocator.alloc(DitBlockW, cfg.dit_layers);
        errdefer self.allocator.free(self.dit_bw);
        for (self.dit_bw, 0..) |*bw, li| {
            const pfx = try std.fmt.allocPrint(a, "transformer_blocks.{d}", .{li});
            bw.n1w = try getW(&self.dit_w, try std.fmt.allocPrint(a, "{s}.norm1.weight", .{pfx}));
            bw.n1b = try getW(&self.dit_w, try std.fmt.allocPrint(a, "{s}.norm1.bias", .{pfx}));
            bw.n2w = try getW(&self.dit_w, try std.fmt.allocPrint(a, "{s}.norm2.weight", .{pfx}));
            bw.n2b = try getW(&self.dit_w, try std.fmt.allocPrint(a, "{s}.norm2.bias", .{pfx}));
            bw.q = try resolveQLin(&self.dit_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_q", .{pfx}), cfg.dit_hidden);
            bw.k = try resolveQLin(&self.dit_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_k", .{pfx}), cfg.dit_hidden);
            bw.v = try resolveQLin(&self.dit_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_v", .{pfx}), cfg.dit_hidden);
            bw.o = try resolveQLin(&self.dit_w, a, try std.fmt.allocPrint(a, "{s}.attn.to_out.0", .{pfx}), cfg.dit_heads * cfg.dit_head_dim);
            bw.ff_in = try resolveQLin(&self.dit_w, a, try std.fmt.allocPrint(a, "{s}.ff_in", .{pfx}), cfg.dit_hidden);
            bw.ff_out = try resolveQLin(&self.dit_w, a, try std.fmt.allocPrint(a, "{s}.ff_out", .{pfx}), rowsOf(bw.ff_in.w) / 2);
        }
        self.dit_proj_in = try resolveQLin(&self.dit_w, a, "proj_in", cfg.dit_in_ch * 2 + cfg.dit_cond);
        self.dit_time1 = try resolveQLin(&self.dit_w, a, "time_embed.linear_1", cfg.dit_fourier);
        self.dit_time2 = try resolveQLin(&self.dit_w, a, "time_embed.linear_2", rowsOf(self.dit_time1.w));
        self.dit_proj_out = try resolveQLin(&self.dit_w, a, "proj_out", cfg.dit_hidden);
    }

    /// Assemble + tokenize the prompt pair. `ids`/`uncond` are owned by the
    /// caller. Over max_text_tokens is a hard error (reference contract).
    pub fn tokenizePrompt(self: *const Engine, allocator: std.mem.Allocator, caption: []const u8, lyrics: []const u8) !struct { ids: []i32, uncond: []i32 } {
        const prompt = try assemblePrompt(allocator, caption, lyrics);
        defer allocator.free(prompt);
        const ids_u = try self.tok.encode(allocator, prompt);
        defer allocator.free(ids_u);
        if (ids_u.len > self.cfg.max_text_tokens) return error.PromptTooLong;
        const ids = try allocator.alloc(i32, ids_u.len);
        errdefer allocator.free(ids);
        for (ids_u, 0..) |u, i| ids[i] = @intCast(u);
        const uncond = try allocator.alloc(i32, ids.len);
        errdefer allocator.free(uncond);
        buildUncondIds(ids, uncond);
        return .{ .ids = ids, .uncond = uncond };
    }

    /// caption/lyrics → 44.1 kHz stereo PCM16 WAV bytes (owned).
    pub fn generateWav(self: *Engine, allocator: std.mem.Allocator, req: MusicRequest, progress: ?sse.Progress) ![]u8 {
        const duration = std.math.clamp(req.duration_s, MIN_DURATION_S, MAX_DURATION_S);
        const max_frames = @min(duration * self.cfg.frame_rate, self.cfg.max_frames_cap);
        const steps = if (req.steps == 0) DEFAULT_STEPS else req.steps;
        log.info("[music3] text2music: {d}s (≤{d} frames), steps={d}, seed={d}\n", .{ duration, max_frames, steps, req.seed });

        const toks = try self.tokenizePrompt(allocator, req.caption, req.lyrics);
        defer allocator.free(toks.ids);
        defer allocator.free(toks.uncond);

        var smp = Sampler{ .seed = req.seed };
        const ar = try self.runArStage(allocator, toks.ids, toks.uncond, .{ .max_frames = max_frames }, &smp, progress);
        defer _ = mlx.mlx_array_free(ar.frame_buf);
        if (ar.emitted == 0) return error.NoAudioFrames;
        _ = mlx.mlx_clear_cache();

        const samples = try self.denoiseAndVocode(allocator, ar.frame_buf, ar.emitted, steps, &smp, progress);
        defer allocator.free(samples);
        return wav_mod.encodePcm16(allocator, samples, self.cfg.sample_rate, 2);
    }

    pub const ArOptions = struct {
        max_frames: u32,
        greedy: bool = false,
        /// Oracle replay: flattened [n_frames*8] codes (c0..c7 per frame,
        /// frame 0 included) that OVERRIDE sampling; the engine's own picks
        /// are still counted against them in `agree`.
        force_codes: ?[]const i32 = null,
    };
    pub const ArResult = struct {
        /// [max_frames, 32768] bf16; rows 0..emitted hold the condition
        /// stream (global hidden ++ 7 depth hiddens per frame).
        frame_buf: mlx.mlx_array,
        emitted: u32,
        agree: u32 = 0,
        forced_total: u32 = 0,
    };

    /// The nested AR stage (reference SemanticGenerationStep): max_frames+1
    /// iterations, frame 0 only advances state past <|audio_start|>.
    pub fn runArStage(
        self: *Engine,
        allocator: std.mem.Allocator,
        ids: []const i32,
        uncond: []const i32,
        opts: ArOptions,
        smp: *Sampler,
        progress: ?sse.Progress,
    ) !ArResult {
        const s = self.s;
        const cfg = self.cfg;
        if (progress) |p| p.emit("prefill", 0, 1);
        const clk = std.Io.Timestamp.now(self.io, .boot);

        var kv = try LmKv.init(allocator, cfg, @intCast(ids.len + opts.max_frames + 2), s);
        defer kv.deinit(allocator);

        // [2,T] ids → embeds [2,T,4096]
        const both = try allocator.alloc(i32, ids.len * 2);
        defer allocator.free(both);
        @memcpy(both[0..ids.len], ids);
        @memcpy(both[ids.len..], uncond);
        const id_shape = [_]c_int{ 2, @intCast(ids.len) };
        const id_arr = mlx.mlx_array_new_data(both.ptr, &id_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        const embed_table = try getW(&self.lm_w, "model.embed_tokens.weight");
        const embeds = try takeRows(embed_table, id_arr, s);
        defer _ = mlx.mlx_array_free(embeds);

        var last_hidden = try lmForward(self, embeds, &kv);
        defer _ = mlx.mlx_array_free(last_hidden);
        evalA(last_hidden);
        const prefill_ms = @as(u64, @intCast(clk.untilNow(self.io, .boot).nanoseconds)) / 1_000_000;
        log.info("[music3] prefill {d} tokens x2 in {d} ms\n", .{ ids.len, prefill_ms });
        if (progress) |p| p.emit("prefill", 1, 1);

        const mask: ?mlx.mlx_array = if (self.pruned_head == null) try buildVocabMask(self, allocator) else null;
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };

        var dkv: ?LmKv = if (depthKvEnabled())
            try LmKv.initShape(allocator, cfg.dd_layers, cfg.dd_heads, cfg.num_codebooks, @divExact(cfg.lm_hidden, cfg.dd_heads), s)
        else
            null;
        defer if (dkv) |*c| c.deinit(allocator);
        if (dkv != null) log.info("[music3] depth-kv engaged\n", .{});

        const hidden_w: c_int = @intCast(cfg.num_codebooks * cfg.lm_hidden);
        var frame_buf = try zerosA(&[_]c_int{ @intCast(opts.max_frames), hidden_w }, .bfloat16, s);
        errdefer _ = mlx.mlx_array_free(frame_buf);

        var result = ArResult{ .frame_buf = frame_buf, .emitted = 0 };
        const probe = costProbeEnabled();
        var pclk = LapClock.init(self.io);
        var prof = [_]u64{ 0, 0, 0, 0 }; // µs: head+sample | depth | feed+book | lm decode
        var prof_iters: u64 = 0;
        var frame: u32 = 0;
        var frame_smp = smp;
        var greedy_smp = Sampler{ .seed = smp.seed, .greedy = true };
        if (opts.greedy) frame_smp = &greedy_smp;

        while (frame <= opts.max_frames) : (frame += 1) {
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                if (frame % 25 == 0) p.emit("frames", result.emitted, opts.max_frames);
            }
            if (probe) _ = pclk.lapUs();
            const logits = if (self.pruned_head) |*ph|
                try lmHeadLogitsPruned(self, ph, last_hidden)
            else
                try lmHeadLogits(self, allocator, last_hidden);
            defer _ = mlx.mlx_array_free(logits);
            const guided = try guidedSemanticLogits(self, logits, mask);
            defer _ = mlx.mlx_array_free(guided);

            var sem_val: i32 = undefined;
            var force_frame: ?[]const i32 = null;
            if (opts.force_codes) |fc| {
                if ((frame + 1) * 8 <= fc.len) force_frame = fc[frame * 8 .. frame * 8 + 8];
            }
            {
                const pick = try sampleTopK(guided, frame_smp, s);
                defer _ = mlx.mlx_array_free(pick);
                var picked = try readScalarI32(pick);
                if (self.pruned_head != null) picked = prunedRowToVocab(picked, @intCast(cfg.semantic_vocab));
                if (force_frame) |f| {
                    result.forced_total += 1;
                    if (picked == f[0] + CODE_OFFSET) result.agree += 1;
                    sem_val = f[0] + CODE_OFFSET;
                } else {
                    sem_val = picked;
                }
            }
            if (probe) prof[0] += pclk.lapUs();
            if (sem_val == AUDIO_END) {
                log.info("[music3] <|audio_end|> at frame {d}\n", .{frame});
                break;
            }
            const sem_arr = i32Arr(&[_]i32{sem_val});
            defer _ = mlx.mlx_array_free(sem_arr);

            var agree_ptr: ?*u32 = null;
            if (force_frame != null) {
                result.forced_total += 7;
                agree_ptr = &result.agree;
            }
            var depth = try generateDepthCodes(self, allocator, last_hidden, sem_arr, frame_smp, if (force_frame) |f| f[1..8] else null, agree_ptr, if (dkv) |*c| c else null);
            defer depth.deinit();
            if (probe) {
                evalA(depth.depth_hidden);
                evalA(depth.codes[6]);
                prof[1] += pclk.lapUs();
            }

            if (frame > 0) {
                const dim: c_int = @intCast(cfg.lm_hidden);
                const cond_h = try sliceA(last_hidden, &[_]c_int{ 0, 0 }, &[_]c_int{ 1, dim }, s);
                defer _ = mlx.mlx_array_free(cond_h);
                const row = try concat2(cond_h, depth.depth_hidden, 1, s); // [1,32768]
                defer _ = mlx.mlx_array_free(row);
                const row_bf = try astype(row, .bfloat16, s);
                defer _ = mlx.mlx_array_free(row_bf);
                const at: c_int = @intCast(result.emitted);
                const updated = try sliceUpdateA(frame_buf, row_bf, &[_]c_int{ at, 0 }, &[_]c_int{ at + 1, hidden_w }, s);
                _ = mlx.mlx_array_free(frame_buf);
                frame_buf = updated;
                result.frame_buf = frame_buf;
                // A per-frame blocking eval here was a SECOND sync after the
                // token read; every 32 frames bounds the lazy slice_update
                // chain (shapes are constant, the pool stays flat).
                if (result.emitted % 32 == 31) evalA(frame_buf);
                result.emitted += 1;
                if (result.emitted >= opts.max_frames) break;
            }

            const feedback = try embedAudioFrame(self, sem_arr, &depth.codes);
            defer _ = mlx.mlx_array_free(feedback);
            if (probe) {
                evalA(feedback);
                prof[2] += pclk.lapUs();
            }
            const next_hidden = try lmForward(self, feedback, &kv);
            _ = mlx.mlx_array_free(last_hidden);
            last_hidden = next_hidden;
            if (probe) {
                evalA(last_hidden);
                prof[3] += pclk.lapUs();
                prof_iters += 1;
            }
        }
        evalA(frame_buf);
        if (result.emitted > 0) {
            const total_ms = @as(u64, @intCast(clk.untilNow(self.io, .boot).nanoseconds)) / 1_000_000;
            const ar_ms = total_ms -| prefill_ms;
            const ms_per = @as(f64, @floatFromInt(ar_ms)) / @as(f64, @floatFromInt(result.emitted));
            log.info("[music3] AR stage: {d} frames in {d} ms ({d:.1} ms/frame)\n", .{ result.emitted, ar_ms, ms_per });
        }
        if (probe and prof_iters > 0) {
            const fi = @as(f64, @floatFromInt(prof_iters));
            log.info("[music3-prof] per-frame: lm {d:.2} ms | head+sample {d:.2} ms | depth {d:.2} ms | feed+book {d:.2} ms ({d} iters)\n", .{
                @as(f64, @floatFromInt(prof[3])) / fi / 1000.0,
                @as(f64, @floatFromInt(prof[0])) / fi / 1000.0,
                @as(f64, @floatFromInt(prof[1])) / fi / 1000.0,
                @as(f64, @floatFromInt(prof[2])) / fi / 1000.0,
                prof_iters,
            });
        }
        if (progress) |p| p.emit("frames", result.emitted, opts.max_frames);
        return result;
    }

    /// Chunked flow-match denoise + vocode: 200-frame windows, hop 100, the
    /// previous window's trailing latents blended into the overlap at EVERY
    /// Euler step, waveform crops tiling the song (reference denoise.py /
    /// decoders.py). Returns interleaved stereo f32 samples (owned).
    fn denoiseAndVocode(
        self: *Engine,
        allocator: std.mem.Allocator,
        frame_buf: mlx.mlx_array,
        frames: u32,
        steps: u32,
        smp: *Sampler,
        progress: ?sse.Progress,
    ) ![]f32 {
        const s = self.s;
        const starts = try chunkStarts(allocator, frames);
        defer allocator.free(starts);
        const n_chunks: u32 = @intCast(starts.len);
        const hidden_w: c_int = @intCast(self.cfg.num_codebooks * self.cfg.lm_hidden);

        var samples: std.ArrayList(f32) = .empty;
        errdefer samples.deinit(allocator);

        const probe = costProbeEnabled();
        var pclk = LapClock.init(self.io);
        var prev_latent: ?mlx.mlx_array = null; // [1, ≤172, 128] f32
        var prev_cond: ?mlx.mlx_array = null;
        defer if (prev_latent) |p| {
            _ = mlx.mlx_array_free(p);
        };
        defer if (prev_cond) |p| {
            _ = mlx.mlx_array_free(p);
        };

        for (starts, 0..) |start, k| {
            const end = @min(start + CHUNK_FRAMES, frames);
            log.info("[music3] denoise window {d}/{d}: frames {d}..{d}\n", .{ k + 1, n_chunks, start, end });
            if (probe) _ = pclk.lapUs();
            const win = try sliceA(frame_buf, &[_]c_int{ @intCast(start), 0 }, &[_]c_int{ @intCast(end), hidden_w }, s);
            defer _ = mlx.mlx_array_free(win);
            var cond = try condEncode(self, allocator, win); // [1,L,2048] f32
            defer _ = mlx.mlx_array_free(cond);
            const l_len: c_int = mlx.getShape(cond)[1];

            var overlap: c_int = 0;
            if (prev_latent) |pl| {
                overlap = @min(mlx.getShape(pl)[1], l_len);
                const pc_slice = try sliceA(prev_cond.?, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 2048 }, s);
                defer _ = mlx.mlx_array_free(pc_slice);
                const spliced = try sliceUpdateA(cond, pc_slice, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 2048 }, s);
                _ = mlx.mlx_array_free(cond);
                cond = spliced;
            }

            const lat_shape = [_]c_int{ 1, l_len, 128 };
            const key = smp.nextKey();
            defer _ = mlx.mlx_array_free(key);
            var latents = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&latents, &lat_shape, 3, .float32, 0.0, 1.0, key, s));
            defer _ = mlx.mlx_array_free(latents);

            var noise_prompt: ?mlx.mlx_array = null;
            defer if (noise_prompt) |np| {
                _ = mlx.mlx_array_free(np);
            };
            if (overlap > 0) {
                const np_view = try sliceA(latents, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 128 }, s);
                defer _ = mlx.mlx_array_free(np_view);
                noise_prompt = try materialize(np_view, s);
            }

            var i: u32 = 0;
            while (i < steps) : (i += 1) {
                if (progress) |p| {
                    if (p.cancelled()) return error.Cancelled;
                    p.emit("diffuse", @intCast(k * steps + i), n_chunks * steps);
                }
                const t = flowTime(i, steps);
                if (overlap > 0) {
                    // latents[:overlap] = (1-(1-1e-6)t)·noise + t·prev
                    const pl_slice = try sliceA(prev_latent.?, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 128 }, s);
                    defer _ = mlx.mlx_array_free(pl_slice);
                    const n_term = try mulScalar(noise_prompt.?, 1.0 - (1.0 - 1e-6) * t, s);
                    defer _ = mlx.mlx_array_free(n_term);
                    const p_term = try mulScalar(pl_slice, t, s);
                    defer _ = mlx.mlx_array_free(p_term);
                    const blend = try addA(n_term, p_term, s);
                    defer _ = mlx.mlx_array_free(blend);
                    const upd = try sliceUpdateA(latents, blend, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 128 }, s);
                    _ = mlx.mlx_array_free(latents);
                    latents = upd;
                }
                const vel = try ditVelocity(self, allocator, latents, t, cond);
                defer _ = mlx.mlx_array_free(vel);
                const dv = try mulScalar(vel, 1.0 / @as(f32, @floatFromInt(steps)), s);
                defer _ = mlx.mlx_array_free(dv);
                const next = try addA(latents, dv, s);
                _ = mlx.mlx_array_free(latents);
                latents = next;
                evalA(latents);
            }
            if (probe) log.info("[music3-prof] window {d}: denoise {d} ms ({d} steps)\n", .{ k + 1, pclk.lapUs() / 1000, steps });

            if (overlap > 0) {
                const pl_slice = try sliceA(prev_latent.?, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 128 }, s);
                defer _ = mlx.mlx_array_free(pl_slice);
                const upd = try sliceUpdateA(latents, pl_slice, &[_]c_int{ 0, 0, 0 }, &[_]c_int{ 1, overlap, 128 }, s);
                _ = mlx.mlx_array_free(latents);
                latents = upd;
            }

            // carry latents[L-344 : L-172] (+ matching condition) to the next window
            const ov2: c_int = @intCast(2 * OVERLAP_LATENT);
            const ov1: c_int = @intCast(OVERLAP_LATENT);
            const c_start: c_int = @max(0, l_len - ov2);
            const c_end: c_int = @max(c_start, l_len - ov1);
            if (prev_latent) |p| _ = mlx.mlx_array_free(p);
            if (prev_cond) |p| _ = mlx.mlx_array_free(p);
            {
                const pl_view = try sliceA(latents, &[_]c_int{ 0, c_start, 0 }, &[_]c_int{ 1, c_end, 128 }, s);
                defer _ = mlx.mlx_array_free(pl_view);
                prev_latent = try materialize(pl_view, s);
                const pc_view = try sliceA(cond, &[_]c_int{ 0, c_start, 0 }, &[_]c_int{ 1, c_end, 2048 }, s);
                defer _ = mlx.mlx_array_free(pc_view);
                prev_cond = try materialize(pc_view, s);
            }

            // vocode this window, crop the overlap spans, append
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("decode", @intCast(k), n_chunks);
            }
            const wave = try vocodeWindow(self, allocator, latents); // [2,N,1] f32
            defer _ = mlx.mlx_array_free(wave);
            var wc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wc);
            try mlx.check(mlx.mlx_contiguous(&wc, wave, false, s));
            evalA(wc);
            if (probe) log.info("[music3-prof] window {d}: vocode {d} ms\n", .{ k + 1, pclk.lapUs() / 1000 });
            const n_samp: usize = @intCast(mlx.getShape(wc)[1]);
            const data = mlx.mlx_array_data_float32(wc) orelse return error.NoData;
            const left_crop: usize = if (k == 0) 0 else @as(usize, CROP_LEFT_LATENT) * VOC_HOP;
            const right_crop: usize = if (k == starts.len - 1) 0 else @as(usize, CROP_RIGHT_LATENT) * VOC_HOP;
            const keep_start = left_crop;
            const keep_end = n_samp - right_crop;
            var si: usize = keep_start;
            while (si < keep_end) : (si += 1) {
                const l = std.math.clamp(data[si], -1.0, 1.0);
                const r = std.math.clamp(data[n_samp + si], -1.0, 1.0);
                try samples.append(allocator, l);
                try samples.append(allocator, r);
            }
            _ = mlx.mlx_clear_cache();
        }
        if (progress) |p| p.emit("decode", n_chunks, n_chunks);
        return samples.toOwnedSlice(allocator);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Load-time weight fixups
// ════════════════════════════════════════════════════════════════════════

fn putWeight(w: *Weights, key: []const u8, arr: mlx.mlx_array) !void {
    if (w.map.getPtr(key)) |p| {
        _ = mlx.mlx_array_free(p.*);
        p.* = arr;
        return;
    }
    const owned = try w.allocator.dupe(u8, key);
    errdefer w.allocator.free(owned);
    try w.map.put(owned, arr);
}

fn removeWeight(w: *Weights, key: []const u8) void {
    if (w.map.fetchRemove(key)) |kv| {
        _ = mlx.mlx_array_free(kv.value);
        w.allocator.free(kv.key);
    }
}

/// PT Conv1d [out,in,K] → MLX [out,K,in], materialized (the original entry
/// is freed).
fn swapConvAxes(w: *Weights, key: []const u8, s: S) !void {
    const orig = try getW(w, key);
    const t = try transpose(orig, &[_]c_int{ 0, 2, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const m = try materialize(t, s);
    try putWeight(w, key, m);
}

/// DiT fixups: conv axis swaps + narrow the f32 1-D block params to bf16
/// (the reference pipeline loads the whole transformer bf16; f32 biases
/// would silently promote the residual stream — the f16-narrow rule).
fn fixupDit(w: *Weights, s: S) !void {
    try swapConvAxes(w, "preprocess_conv.weight", s);
    try swapConvAxes(w, "postprocess_conv.weight", s);

    var to_narrow: std.ArrayList([]const u8) = .empty;
    defer to_narrow.deinit(w.allocator);
    var it = w.map.keyIterator();
    while (it.next()) |kp| {
        const key = kp.*;
        const narrow = (std.mem.startsWith(u8, key, "transformer_blocks.") and
            (std.mem.endsWith(u8, key, ".norm1.weight") or std.mem.endsWith(u8, key, ".norm1.bias") or
                std.mem.endsWith(u8, key, ".norm2.weight") or std.mem.endsWith(u8, key, ".norm2.bias") or
                std.mem.endsWith(u8, key, ".ff_in.bias") or std.mem.endsWith(u8, key, ".ff_out.bias"))) or
            std.mem.eql(u8, key, "time_embed.linear_2.bias");
        if (narrow) try to_narrow.append(w.allocator, key);
    }
    for (to_narrow.items) |key| {
        const orig = try getW(w, key);
        if (mlx.mlx_array_dtype(orig) == .bfloat16) continue;
        const bf = try astype(orig, .bfloat16, s);
        defer _ = mlx.mlx_array_free(bf);
        const m = try materialize(bf, s);
        try putWeight(w, key, m);
    }
}

/// Vocoder fixups: weight_norm fusion (w = g·v/||v||, norm over all dims but
/// 0), PT→MLX conv axis swaps (ConvTranspose1d is [in,out,K] — identified by
/// NAME: only `conv_t1` is transposed here; the bias-length heuristic is
/// ambiguous on the square res-unit convs), Snake alpha [1,C,1] → [C].
fn fuseVocoder(w: *Weights, s: S) !void {
    var bases: std.ArrayList([]u8) = .empty;
    defer {
        for (bases.items) |b| w.allocator.free(b);
        bases.deinit(w.allocator);
    }
    var alphas: std.ArrayList([]u8) = .empty;
    defer {
        for (alphas.items) |b| w.allocator.free(b);
        alphas.deinit(w.allocator);
    }
    var it = w.map.keyIterator();
    while (it.next()) |kp| {
        const key = kp.*;
        if (std.mem.endsWith(u8, key, ".weight_g")) {
            try bases.append(w.allocator, try w.allocator.dupe(u8, key[0 .. key.len - ".weight_g".len]));
        } else if (std.mem.endsWith(u8, key, ".alpha")) {
            try alphas.append(w.allocator, try w.allocator.dupe(u8, key));
        }
    }

    var keybuf: [256]u8 = undefined;
    for (bases.items) |base| {
        const gk = try std.fmt.bufPrint(&keybuf, "{s}.weight_g", .{base});
        const g = try getW(w, gk);
        var vkbuf: [256]u8 = undefined;
        const vk = try std.fmt.bufPrint(&vkbuf, "{s}.weight_v", .{base});
        const v = try getW(w, vk);

        const sq = try mulA(v, v, s);
        defer _ = mlx.mlx_array_free(sq);
        var s2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s2);
        try mlx.check(mlx.mlx_sum_axis(&s2, sq, 2, true, s));
        var s1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s1);
        try mlx.check(mlx.mlx_sum_axis(&s1, s2, 1, true, s));
        var norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(norm);
        try mlx.check(mlx.mlx_sqrt(&norm, s1, s));
        const gv = try mulA(g, v, s);
        defer _ = mlx.mlx_array_free(gv);
        const fused = try divA(gv, norm, s);
        defer _ = mlx.mlx_array_free(fused);

        const is_tconv = std.mem.endsWith(u8, base, "conv_t1");
        const perm: []const c_int = if (is_tconv) &[_]c_int{ 1, 2, 0 } else &[_]c_int{ 0, 2, 1 };
        const t = try transpose(fused, perm, s);
        defer _ = mlx.mlx_array_free(t);
        const m = try materialize(t, s);
        var wkbuf: [256]u8 = undefined;
        const wk = try std.fmt.bufPrint(&wkbuf, "{s}.weight", .{base});
        try putWeight(w, wk, m);
        removeWeight(w, gk);
        removeWeight(w, vk);
    }

    try swapConvAxes(w, "dec_in_proj.weight", s);

    for (alphas.items) |key| {
        const orig = try getW(w, key);
        const sh = mlx.getShape(orig);
        if (sh.len != 3) continue;
        const flat = try reshape(orig, &[_]c_int{sh[1]}, s);
        defer _ = mlx.mlx_array_free(flat);
        const m = try materialize(flat, s);
        try putWeight(w, key, m);
    }
}

/// Read an f32 tensor's data (loaded contiguous on the CPU stream).
fn readF32Data(arr: mlx.mlx_array, out: []f32) !void {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    const cpu = mlx.mlx_default_cpu_stream_new();
    const f = try astype(arr, .float32, cpu);
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, cpu));
    evalA(c);
    const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    @memcpy(out, d[0..out.len]);
}

/// softmax(layer_weight_logits) × layer_scale, precomputed to [1,1,8,1] f32.
fn buildMix(w: *Weights, s: S) !mlx.mlx_array {
    _ = s;
    var logits: [8]f32 = undefined;
    try readF32Data(try getW(w, "layer_weight_logits"), &logits);
    var scale: [1]f32 = undefined;
    try readF32Data(try getW(w, "layer_scale"), &scale);
    var mx: f32 = logits[0];
    for (logits) |v| mx = @max(mx, v);
    var sum: f32 = 0;
    var out: [8]f32 = undefined;
    for (logits, 0..) |v, i| {
        out[i] = @exp(v - mx);
        sum += out[i];
    }
    for (&out) |*v| v.* = v.* / sum * scale[0];
    const sh = [_]c_int{ 1, 1, 8, 1 };
    return mlx.mlx_array_new_data(&out, &sh, 4, .float32);
}

fn readFourier(w: *Weights, out: *[128]f32) !void {
    try readF32Data(try getW(w, "time_proj.weight"), out);
}

/// Load ONE safetensors file into a Weights map (CPU stream; iterator +1
/// transferred into the map — the model.zig pattern).
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
    log.info("[music3] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first (no weights), then env-gated load/cost/oracle
// (MUSIC3_TEST_MODEL + MUSIC3_FIXTURES, fed by tests/dump_music3_fixtures.py).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "music3 latent length: frames*441/128 truncated, floor 1" {
    try testing.expectEqual(@as(u32, 689), latentLen(200));
    try testing.expectEqual(@as(u32, 41), latentLen(12));
    try testing.expectEqual(@as(u32, 3), latentLen(1));
    try testing.expectEqual(@as(u32, 1), latentLen(0));
    try testing.expectEqual(@as(u32, 31007), latentLen(9000));
}

test "music3 chunk starts: python range(0, F-100, 100) semantics" {
    const a = testing.allocator;
    const one = try chunkStarts(a, 200);
    defer a.free(one);
    try testing.expectEqualSlices(u32, &[_]u32{0}, one);
    const two = try chunkStarts(a, 201);
    defer a.free(two);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 100 }, two);
    const four = try chunkStarts(a, 450);
    defer a.free(four);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 100, 200, 300 }, four);
}

test "music3 flow time: uniform ascending grid, 0 = noise" {
    try testing.expectApproxEqAbs(@as(f32, 0.0), flowTime(0, 30), 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 0.5), flowTime(15, 30), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 29.0 / 30.0), flowTime(29, 30), 1e-7);
}

test "music3 nearest resample indices are floor(dst*in/out) and monotone" {
    try testing.expectEqual(@as(u32, 0), nearestIdx(0, 12, 41));
    try testing.expectEqual(@as(u32, 11), nearestIdx(40, 12, 41));
    var prev: u32 = 0;
    for (0..689) |j| {
        const v = nearestIdx(@intCast(j), 200, 689);
        try testing.expect(v >= prev and v < 200);
        prev = v;
    }
}

test "music3 pruned head row maps to code offset then audio_end" {
    try testing.expectEqual(CODE_OFFSET, prunedRowToVocab(0, 16384));
    try testing.expectEqual(CODE_OFFSET + 16383, prunedRowToVocab(16383, 16384));
    try testing.expectEqual(AUDIO_END, prunedRowToVocab(16384, 16384));
}

test "music3 uncond ids: [1:-2] become <|audio_cfg|>" {
    const ids = [_]i32{ 100, 1, 2, 3, 4, 200, 300 };
    var out: [7]i32 = undefined;
    buildUncondIds(&ids, &out);
    try testing.expectEqualSlices(i32, &[_]i32{ 100, AUDIO_CFG_TOKEN, AUDIO_CFG_TOKEN, AUDIO_CFG_TOKEN, AUDIO_CFG_TOKEN, 200, 300 }, &out);
}

// Expected strings generated by executing the REFERENCE Python regex logic
// (encoders.py verbatim) over each input — the byte contract.
test "music3 clean caption matches the reference python byte-for-byte" {
    const a = testing.allocator;
    const cases = [_][2][]const u8{
        .{ "plain caption", "plain caption" },
        .{ "Upbeat **synthwave** with driving bass and dreamy pads\n<|bpm 120|>", "Upbeat synthwave with driving bass and dreamy pads\nbpm is 120" },
        .{ "## Heading here\n- bullet one\n* bullet two\n+ bullet three", "Heading here\nbullet one\nbullet two\nbullet three" },
        .{ "*italic* and **bold** and ***both***", "italic and bold and both" },
        .{ "a\n\n\n\nb", "a\nb" },
        .{ "rule:\n---\nafter", "rule:\nafter" },
        .{ "\xe2\x80\xa2 dotted    indented", "dottedindented" },
        .{ "<|key|> <|k v w|> <|  spaced key  |>", "key k is v w spaced is key" },
        .{ "####### seven hashes stay", "####### seven hashes stay" },
        .{ "  \n---\nX", "\nX" }, // ws-only line absorbed by the rule match
        .{ "*a**b*", "*a**b*" }, // italic lookarounds refuse
        .{ "<|a<|b|>", "<|ab" }, // tag scanner retry semantics
        .{ "   * deep bullet\n    #tab", "deep bullet\n#tab" },
        .{ "**unclosed bold", "**unclosed bold" },
        .{ "----", "" },
        .{ "mid --- dash", "mid --- dash" },
    };
    for (cases) |case| {
        const got = try cleanCaption(a, case[0]);
        defer a.free(got);
        try testing.expectEqualStrings(case[1], got);
    }
}

test "music3 normalize lyrics matches the reference python byte-for-byte" {
    const a = testing.allocator;
    const cases = [_][2][]const u8{
        .{ "hello world", "[start]\nhello world" },
        .{ "[Verse] ignored text\nneon lights across the bay [Chorus]\nwe run all night ^ we never stay", "[start]\n[verse]\nneon lights across the bay\n[chorus]\nwe run all night\nwe never stay" },
        .{ "[A][B] [C]  tail\nplain", "[start]\n[a][b]\n[c]\nplain" },
        .{ "  [Tag]\t[TWO]  ", "[start]\n[tag]\t[two]" },
        .{ "[]\nempty stays", "[start]\n[]\nempty stays" },
        .{ "x [Y] z", "[start]\nx\n[y]\nz" },
        .{ "a ^ b ^ c", "[start]\na\nb\nc" },
        .{ "[UPPER lower MiXeD]", "[start]\n[upper lower mixed]" },
    };
    for (cases) |case| {
        const got = try normalizeLyrics(a, case[0]);
        defer a.free(got);
        try testing.expectEqualStrings(case[1], got);
    }
}

test "music3 assembled prompt is the exact reference template" {
    const a = testing.allocator;
    const prompt = try assemblePrompt(
        a,
        "Upbeat **synthwave** with driving bass and dreamy pads\n<|bpm 120|>",
        "[Verse] ignored text\nneon lights across the bay [Chorus]\nwe run all night ^ we never stay",
    );
    defer a.free(prompt);
    try testing.expectEqualStrings(
        "<|im_start|><|caption_start|>Upbeat synthwave with driving bass and dreamy pads\nbpm is 120<|caption_end|>" ++
            "<|lyrics_start|>[start]\n[verse]\nneon lights across the bay\n[chorus]\nwe run all night\nwe never stay<|lyrics_end|>" ++
            "<|im_end|><|audio_start|>",
        prompt,
    );
}

// ── env-gated: load + oracles ──

fn readRawF32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
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

fn readRawI32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]i32 {
    const raw = try readRawF32(io, a, dir, name);
    return @as([]i32, @ptrCast(raw));
}

fn cosineF64(x: []const f32, y: []const f32) f64 {
    var dot: f64 = 0;
    var nx: f64 = 0;
    var ny: f64 = 0;
    for (x, y) |a, b| {
        dot += @as(f64, a) * b;
        nx += @as(f64, a) * a;
        ny += @as(f64, b) * b;
    }
    return dot / (@sqrt(nx) * @sqrt(ny) + 1e-30);
}

fn rmsRatio(x: []const f32, y: []const f32) f64 {
    var sx: f64 = 0;
    var sy: f64 = 0;
    for (x) |a| sx += @as(f64, a) * a;
    for (y) |b| sy += @as(f64, b) * b;
    return @sqrt(sx / @as(f64, @floatFromInt(x.len))) / (@sqrt(sy / @as(f64, @floatFromInt(y.len))) + 1e-30);
}

/// cos AND rms_ratio vs a fixture — a cosine alone cannot see a scale error.
fn assertParity(arr: mlx.mlx_array, ref: []const f32, label: []const u8, min_cos: f64, rms_tol: f64, s: S) !void {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    evalA(c);
    const n: usize = @intCast(mlx.mlx_array_size(c));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    const cos = cosineF64(d[0..n], ref);
    const rr = rmsRatio(d[0..n], ref);
    std.debug.print("[music3-{s}] cos={d:.6} rms_ratio={d:.4}\n", .{ label, cos, rr });
    try testing.expect(cos > min_cos);
    try testing.expect(rr > 1.0 - rms_tol and rr < 1.0 + rms_tol);
}

fn testEngine(io: std.Io, a: std.mem.Allocator) !*Engine {
    const model_dir = std.mem.span(std.c.getenv("MUSIC3_TEST_MODEL") orelse return error.SkipZigTest);
    return Engine.load(io, a, model_dir);
}

fn fixturesDir() ![]const u8 {
    return std.mem.span(std.c.getenv("MUSIC3_FIXTURES") orelse return error.SkipZigTest);
}

test "music3 load: shapes match the checkpoint contract" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var e = try testEngine(io, a);
    defer e.deinit();

    const emb = try getW(&e.lm_w, "model.embed_tokens.weight");
    try testing.expectEqualSlices(c_int, &[_]c_int{ 200000, 4096 }, mlx.getShape(emb));
    // vocoder fused + swapped: conv_in [1536,1024,7] → [1536,7,1024]
    const ci = try getW(&e.voc_w, "conv_in.weight");
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1536, 7, 1024 }, mlx.getShape(ci));
    try testing.expect(e.voc_w.get("conv_in.weight_g") == null);
    // conv_t1 [1536,768,16] (in,out,K) → [768,16,1536]
    const ct = try getW(&e.voc_w, "blocks.0.conv_t1.weight");
    try testing.expectEqualSlices(c_int, &[_]c_int{ 768, 16, 1536 }, mlx.getShape(ct));
    // snake alpha flattened
    const al = try getW(&e.voc_w, "blocks.0.snake1.alpha");
    try testing.expectEqualSlices(c_int, &[_]c_int{1536}, mlx.getShape(al));
    // condition-encoder conv swapped
    const pj = try getW(&e.ce_w, "proj.weight");
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2048, 3, 4096 }, mlx.getShape(pj));
    // DiT convs swapped, norm params narrowed
    const pc = try getW(&e.dit_w, "preprocess_conv.weight");
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2304, 1, 2304 }, mlx.getShape(pc));
    const n1 = try getW(&e.dit_w, "transformer_blocks.0.norm1.weight");
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(n1));
    // pruned head: 16385 rows sliced from the full head, same packed geometry
    // (kill switch MLX_SERVE_MUSIC3_LMHEAD_PRUNE=0 must leave it unbuilt)
    const prune_killed = if (std.c.getenv("MLX_SERVE_MUSIC3_LMHEAD_PRUNE")) |v| v[0] == '0' else false;
    if (prune_killed) {
        try testing.expect(e.pruned_head == null);
    } else {
        const full = try getW(&e.lm_w, "lm_head.weight");
        const ph = e.pruned_head orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(u32, 16385), ph.rows);
        try testing.expectEqual(mlx.getShape(full)[1], mlx.getShape(ph.w)[1]);
        try testing.expectEqualSlices(c_int, &[_]c_int{ 16385, mlx.getShape(full)[1] }, mlx.getShape(ph.w));
    }
}

test "music3 oracle: prompt token ids byte-exact" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const fix = try fixturesDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ref_ids = try readRawI32(io, a, fix, "text_ids.i32.raw");
    defer a.free(ref_ids);
    const ref_uncond = try readRawI32(io, a, fix, "uncond_ids.i32.raw");
    defer a.free(ref_uncond);

    var e = try testEngine(io, a);
    defer e.deinit();
    const toks = try e.tokenizePrompt(
        a,
        "Upbeat **synthwave** with driving bass and dreamy pads\n<|bpm 120|>",
        "[Verse] ignored text\nneon lights across the bay [Chorus]\nwe run all night ^ we never stay",
    );
    defer a.free(toks.ids);
    defer a.free(toks.uncond);
    try testing.expectEqualSlices(i32, ref_ids, toks.ids);
    try testing.expectEqualSlices(i32, ref_uncond, toks.uncond);
}

test "music3 oracle: prefill last_hidden + logits match reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const fix = try fixturesDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readRawI32(io, a, fix, "text_ids.i32.raw");
    defer a.free(ids);
    const uncond = try readRawI32(io, a, fix, "uncond_ids.i32.raw");
    defer a.free(uncond);
    const ref_hidden = try readRawF32(io, a, fix, "last_hidden.f32.raw");
    defer a.free(ref_hidden);
    const ref_logits = try readRawF32(io, a, fix, "logits0.f32.raw");
    defer a.free(ref_logits);

    var e = try testEngine(io, a);
    defer e.deinit();
    var kv = try LmKv.init(a, e.cfg, @intCast(ids.len + 4), e.s);
    defer kv.deinit(a);
    const both = try a.alloc(i32, ids.len * 2);
    defer a.free(both);
    @memcpy(both[0..ids.len], ids);
    @memcpy(both[ids.len..], uncond);
    const id_shape = [_]c_int{ 2, @intCast(ids.len) };
    const id_arr = mlx.mlx_array_new_data(both.ptr, &id_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(id_arr);
    const table = try getW(&e.lm_w, "model.embed_tokens.weight");
    const embeds = try takeRows(table, id_arr, e.s);
    defer _ = mlx.mlx_array_free(embeds);
    const last = try lmForward(e, embeds, &kv);
    defer _ = mlx.mlx_array_free(last);
    try assertParity(last, ref_hidden, "prefill", 0.98, 0.1, e.s);
    const logits = try lmHeadLogits(e, a, last);
    defer _ = mlx.mlx_array_free(logits);
    try assertParity(logits, ref_logits, "logits", 0.98, 0.1, e.s);
}

test "music3 oracle: greedy AR replay tracks reference codes + hiddens" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const fix = try fixturesDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readRawI32(io, a, fix, "text_ids.i32.raw");
    defer a.free(ids);
    const uncond = try readRawI32(io, a, fix, "uncond_ids.i32.raw");
    defer a.free(uncond);
    const codes = try readRawI32(io, a, fix, "ar_codes.i32.raw");
    defer a.free(codes);
    const ref_hiddens = try readRawF32(io, a, fix, "ar_hiddens.f32.raw");
    defer a.free(ref_hiddens);
    const n_frames: u32 = @intCast(ref_hiddens.len / 32768);
    try testing.expectEqual(codes.len, (n_frames + 1) * 8); // frame 0 included

    var e = try testEngine(io, a);
    defer e.deinit();
    var smp = Sampler{ .seed = 0, .greedy = true };
    const ar = try e.runArStage(a, ids, uncond, .{
        .max_frames = n_frames,
        .greedy = true,
        .force_codes = codes,
    }, &smp, null);
    defer _ = mlx.mlx_array_free(ar.frame_buf);
    try testing.expectEqual(n_frames, ar.emitted);
    const pct = @as(f64, @floatFromInt(ar.agree)) / @as(f64, @floatFromInt(ar.forced_total));
    std.debug.print("[music3-ar] greedy agreement {d}/{d} ({d:.3})\n", .{ ar.agree, ar.forced_total, pct });
    try testing.expect(pct > 0.85);
    try assertParity(ar.frame_buf, ref_hiddens, "ar-hiddens", 0.98, 0.1, e.s);
}

test "music3 oracle: condition encoder matches reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const fix = try fixturesDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const hiddens = try readRawF32(io, a, fix, "ar_hiddens.f32.raw");
    defer a.free(hiddens);
    const ref_cond = try readRawF32(io, a, fix, "cond_out.f32.raw");
    defer a.free(ref_cond);
    const n_frames: c_int = @intCast(hiddens.len / 32768);

    var e = try testEngine(io, a);
    defer e.deinit();
    const sh = [_]c_int{ n_frames, 32768 };
    const h_f32 = mlx.mlx_array_new_data(hiddens.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(h_f32);
    const h_bf = try astype(h_f32, .bfloat16, e.s); // frame buffer's storage dtype
    defer _ = mlx.mlx_array_free(h_bf);
    const cond = try condEncode(e, a, h_bf);
    defer _ = mlx.mlx_array_free(cond);
    try assertParity(cond, ref_cond, "cond", 0.99, 0.05, e.s);
}

test "music3 oracle: DiT velocity matches reference at t=0 and t=0.5" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const fix = try fixturesDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat_cf = try readRawF32(io, a, fix, "dit_lat.f32.raw"); // [1,128,L] channel-first
    defer a.free(lat_cf);
    const cond_d = try readRawF32(io, a, fix, "cond_out.f32.raw");
    defer a.free(cond_d);
    const l_len: c_int = @intCast(lat_cf.len / 128);

    var e = try testEngine(io, a);
    defer e.deinit();
    const cf_sh = [_]c_int{ 1, 128, l_len };
    const lat0 = mlx.mlx_array_new_data(lat_cf.ptr, &cf_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(lat0);
    const lat = try transpose(lat0, &[_]c_int{ 0, 2, 1 }, e.s);
    defer _ = mlx.mlx_array_free(lat);
    const c_sh = [_]c_int{ 1, l_len, 2048 };
    const cond = mlx.mlx_array_new_data(cond_d.ptr, &c_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(cond);

    // The fixture dumps the plain single-branch forward (batch 1, the given
    // condition) — compare against ditForward directly; CFG is pure math on
    // top and is covered hermetically.
    for ([_]struct { name: []const u8, t: f32 }{
        .{ .name = "dit_v_t0.f32.raw", .t = 0.0 },
        .{ .name = "dit_v_t05.f32.raw", .t = 0.5 },
    }) |case| {
        const ref_cf = try readRawF32(io, a, fix, case.name);
        defer a.free(ref_cf);
        const vel = try ditForward(e, a, lat, case.t, cond);
        defer _ = mlx.mlx_array_free(vel);
        const vel_cf = try transpose(vel, &[_]c_int{ 0, 2, 1 }, e.s);
        defer _ = mlx.mlx_array_free(vel_cf);
        try assertParity(vel_cf, ref_cf, case.name, 0.98, 0.1, e.s);
    }
}

test "music3 oracle: vocoder waveform matches reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const fix = try fixturesDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat_cf = try readRawF32(io, a, fix, "dit_lat.f32.raw");
    defer a.free(lat_cf);
    const ref_wav = try readRawF32(io, a, fix, "voc_wav.f32.raw"); // [1,2,N]
    defer a.free(ref_wav);
    const l_len: c_int = @intCast(lat_cf.len / 128);

    var e = try testEngine(io, a);
    defer e.deinit();
    const cf_sh = [_]c_int{ 1, 128, l_len };
    const lat0 = mlx.mlx_array_new_data(lat_cf.ptr, &cf_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(lat0);
    const lat = try transpose(lat0, &[_]c_int{ 0, 2, 1 }, e.s);
    defer _ = mlx.mlx_array_free(lat);
    const wave = try vocodeWindow(e, a, lat); // [2,N,1]
    defer _ = mlx.mlx_array_free(wave);
    const n: c_int = mlx.getShape(wave)[1];
    const flat = try reshape(wave, &[_]c_int{ 1, 2, n }, e.s);
    defer _ = mlx.mlx_array_free(flat);
    try assertParity(flat, ref_wav, "vocoder", 0.99, 0.05, e.s);
}

test "music3 wav dump harness (env-gated, fixed seed)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const out_path = std.mem.span(std.c.getenv("MUSIC3_WAV_OUT") orelse return error.SkipZigTest);
    if (out_path.len == 0 or out_path[0] != '/') return error.SkipZigTest;
    log.enableStderr();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var e = try testEngine(io, a);
    defer e.deinit();
    const wav = try e.generateWav(a, .{
        .caption = "an upbeat synthwave track with driving bass",
        .lyrics = "[verse]\nneon lights",
        .duration_s = 8,
        .seed = 7,
        .steps = 8,
    }, null);
    defer a.free(wav);
    const dir_path = std.fs.path.dirname(out_path) orelse return error.SkipZigTest;
    var d = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer d.close(io);
    var fh = try d.createFile(io, std.fs.path.basename(out_path), .{});
    defer fh.close(io);
    var wbuf: [4096]u8 = undefined;
    var w = fh.writer(io, &wbuf);
    try w.interface.writeAll(wav);
    try w.interface.flush();
    std.debug.print("[music3-wav] wrote {d} bytes to {s}\n", .{ wav.len, out_path });
}

test "music3 frame cost probe (env-gated, informational)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    _ = std.c.getenv("MUSIC3_COST_PROBE") orelse return error.SkipZigTest;
    log.enableStderr(); // the ms/frame line IS this harness's output
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var e = try testEngine(io, a);
    defer e.deinit();
    const toks = try e.tokenizePrompt(a, "an upbeat synthwave track with driving bass", "[verse]\nneon lights");
    defer a.free(toks.ids);
    defer a.free(toks.uncond);
    var smp = Sampler{ .seed = 0 };
    const frames: u32 = blk: {
        const raw = std.c.getenv("MUSIC3_COST_FRAMES") orelse break :blk 100;
        break :blk std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 100;
    };
    const ar = try e.runArStage(a, toks.ids, toks.uncond, .{ .max_frames = frames }, &smp, null);
    defer _ = mlx.mlx_array_free(ar.frame_buf);
    try testing.expect(ar.emitted > 0);
}

test "music3 instrumental marker survives lyric normalization as the bare [inst] tag" {
    const a = std.testing.allocator;
    const norm = try normalizeLyrics(a, INSTRUMENTAL_LYRICS);
    defer a.free(norm);
    // `[start]` is unconditional; the marker must arrive as the single
    // lowercased tag the checkpoint was trained on, with no stray words.
    try std.testing.expectEqualStrings("[start]\n[instrumental]", norm);
}

test "music3 instrumental lyrics resolve from the flag, never from an empty string" {
    // The flag is the ONLY way in: an empty `lyrics` stays empty (the handler
    // 400s on it) so a client that forgot the field cannot silently get an
    // instrumental it did not ask for.
    try std.testing.expectEqualStrings(instrumentalMarker(), resolveLyrics(true, ""));
    try std.testing.expectEqualStrings(instrumentalMarker(), resolveLyrics(true, "   \n\t "));
    // Unset in the test environment, so the override resolves to the default.
    try std.testing.expectEqualStrings(INSTRUMENTAL_LYRICS, instrumentalMarker());
    try std.testing.expectEqualStrings("", resolveLyrics(false, ""));
    try std.testing.expectEqualStrings("[verse]\nhello", resolveLyrics(false, "[verse]\nhello"));
}

test "music3 instrumental caption: the clause is appended only when the user did not already say it" {
    // MiniMax's own api marks `prompt` REQUIRED for an instrumental track while
    // making `lyrics` optional — which says the CAPTION is where the no-vocals
    // intent lives, and the lyric tag alone was never meant to carry it. Live
    // 2026-08-18: `[Instrumental]` alone sang no words but still produced vocal
    // texture, exactly what a SECTION tag would do.
    try std.testing.expect(!captionMentionsNoVocals("upbeat synthwave with driving bass"));
    // A caption whose GENRE is instrumental music must still get the clause —
    // the old fuzzy guard matched these and silently disabled the feature.
    try std.testing.expect(!captionMentionsNoVocals("Instrumental ambient field-recording piece, no vocals"));
    try std.testing.expect(!captionMentionsNoVocals("an instrumental piece"));
    try std.testing.expect(!captionMentionsNoVocals("lo-fi piano, no vocals"));
    // Only our own sentence, so a re-render cannot stack it twice.
    try std.testing.expect(captionMentionsNoVocals("lo-fi. " ++ INSTRUMENTAL_CAPTION_CLAUSE));
}

test "music3 instrumental tag override: a spelling, or 'none' for a bare lyric block" {
    // The two MiniMax sources disagree on the tag and neither documents a
    // track-level one, so the spelling is a lever, not a contract. `none` is
    // the third arm: the hosted api says lyrics are NOT REQUIRED under
    // `is_instrumental`, and the closest local equivalent to sending nothing is
    // an EMPTY lyric body — `[start]` on its own.
    try std.testing.expectEqualStrings("", markerFromEnvValue("none"));
    try std.testing.expectEqualStrings("", markerFromEnvValue("NONE"));
    try std.testing.expectEqualStrings("[Inst]", markerFromEnvValue("[Inst]"));
    // Unset or blank keeps the default rather than silently emptying the block.
    try std.testing.expectEqualStrings(INSTRUMENTAL_LYRICS, markerFromEnvValue(null));
    try std.testing.expectEqualStrings(INSTRUMENTAL_LYRICS, markerFromEnvValue("   "));
}

test "music3 instrumental with an empty marker still assembles a legal lyric block" {
    const a = std.testing.allocator;
    // The `none` arm must not produce a malformed prompt: `[start]` is
    // unconditional in normalizeLyrics, so an empty body is just that alone.
    const norm = try normalizeLyrics(a, "");
    defer a.free(norm);
    try std.testing.expectEqualStrings("[start]\n", norm);
}

test "music3 caption facts: bpm and key are appended in MiniMax's own spelling" {
    const a = std.testing.allocator;
    // Music 3 has no `bpm` field, but it DOES support tempo and key — the model
    // card lists them under Global Metadata and its own example caption reads
    // "Genre: acoustic pop. BPM: 96. Key: C major." So the fields are not
    // inapplicable here, they are carried as caption TEXT.
    const c1 = try captionWithFacts(a, "acoustic pop", 96, "C major", false);
    defer a.free(c1);
    try std.testing.expectEqualStrings("acoustic pop\nBPM: 96. Key: C major.", c1);

    // Each fact is independent.
    const c2 = try captionWithFacts(a, "lo-fi", 84, "", false);
    defer a.free(c2);
    try std.testing.expectEqualStrings("lo-fi\nBPM: 84.", c2);
    const c3 = try captionWithFacts(a, "lo-fi", null, "F minor", false);
    defer a.free(c3);
    try std.testing.expectEqualStrings("lo-fi\nKey: F minor.", c3);

    // Nothing to add leaves the caption byte-identical, so a plain request is
    // unchanged from before this existed.
    const c4 = try captionWithFacts(a, "lo-fi", null, "", false);
    defer a.free(c4);
    try std.testing.expectEqualStrings("lo-fi", c4);
}

test "music3 caption facts: the user's own words always win" {
    const a = std.testing.allocator;
    // Appending on top of what they already wrote repeats them back at the
    // model and spends caption budget doing it.
    const c1 = try captionWithFacts(a, "acoustic pop at 96 bpm", 120, "", false);
    defer a.free(c1);
    try std.testing.expectEqualStrings("acoustic pop at 96 bpm", c1);
    const c2 = try captionWithFacts(a, "something in C major", null, "F minor", false);
    defer a.free(c2);
    try std.testing.expectEqualStrings("something in C major", c2);
    // ...but "instrumental" in the GENRE is not the user asking us to skip the
    // clause, so it still fires alongside the tempo fact.
    const c3 = try captionWithFacts(a, "an instrumental piece", 96, "", true);
    defer a.free(c3);
    try std.testing.expectEqualStrings(
        "an instrumental piece\nInstrumental only: no vocals, no singing, no lyrics. BPM: 96.", c3);
}

test "music3 caption facts: instrumental clause leads, then bpm, then key" {
    const a = std.testing.allocator;
    // Order is a contract only in that it is STABLE — a caption that reshuffles
    // between runs makes seeds non-reproducible for no reason.
    const c = try captionWithFacts(a, "ambient", 70, "A minor", true);
    defer a.free(c);
    try std.testing.expectEqualStrings(
        "ambient\nInstrumental only: no vocals, no singing, no lyrics. BPM: 70. Key: A minor.", c);
}
