//! SamplingParams and the reserved-token suppress mask.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../../../sdk/tokenizer/tokenizer.zig");
const log = @import("../../../sdk/util/log.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const Constraint = @import("../grammar/constraint.zig").Constraint;
const ThinkBound = @import("../../../sdk/request/forced.zig").ThinkBound;
const ToolForce = @import("../../../sdk/request/forced.zig").ToolForce;
const probAt = @import("spec_sampling.zig").probAt;
const probsAtLastPos = @import("spec_sampling.zig").probsAtLastPos;
const sampleToken = @import("sampler.zig").sampleToken;
const sampleTokenLazy = @import("lazy.zig").sampleTokenLazy;
const samplerTestReadFlat = @import("filters.zig").samplerTestReadFlat;
const verifyArgmax = @import("../spec/verify.zig").verifyArgmax;


pub const SamplingParams = @import("../../../sdk/request/sampling.zig").SamplingParams;

/// Build the `[vocab]` bool suppression mask (true = never sample) on the
/// host, once per model load. Caller owns the returned array.
///
/// Two disjoint sources of "never sample this row":
///   - `ids`: the reserved specials (`tokenizer.reservedOutputIds`).
///   - `[defined_vocab, vocab)`: the checkpoint's PADDING rows. A config's
///     `vocab_size` is a padded matrix dimension, not a vocabulary — the
///     tokenizer's highest defined id + 1 is. The rows in between decode to
///     nothing, so a sample from one emits nothing while consuming a step and
///     poisoning the KV; they were the one class of unsampleable row this
///     mask never covered.
/// `defined_vocab == 0` or `>= vocab` means "no padding known" and adds
/// nothing — a mask that suppressed the whole vocab is `-inf` everywhere,
/// which argmaxes to id 0 (the all-false-mask class, from the other side).
pub fn buildSuppressMask(ids: []const u32, defined_vocab: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const alloc = std.heap.page_allocator;
    const buf = try alloc.alloc(bool, vocab);
    defer alloc.free(buf);
    fillSuppressMask(buf, ids, defined_vocab);
    const shape = [_]c_int{@intCast(vocab)};
    return mlx.mlx_array_new_data(buf.ptr, &shape, 1, .bool_);
}

/// The host half of `buildSuppressMask`, pure so the row set is testable
/// without a GPU. `buf.len` is the mask's vocab.
pub fn fillSuppressMask(buf: []bool, ids: []const u32, defined_vocab: usize) void {
    @memset(buf, false);
    for (ids) |id| {
        if (id < buf.len) buf[id] = true;
    }
    if (defined_vocab > 0 and defined_vocab < buf.len) {
        @memset(buf[defined_vocab..], true);
    }
}

/// Derive + install the reserved-token suppression mask on a freshly-built
/// Transformer (both the serve load path and the CLI run path call this).
/// The id set is `tokenizer.reservedOutputIds`: `special: true` added tokens
/// minus EOS/stop ids minus template-emitted markers. Never fails a load — any error logs and
/// leaves the mask null (suppression off), and engagement is one-shot-logged
/// so a silent no-op is visible (the silent-fallback class).
pub fn installSuppressMask(xfm: *Transformer, tok: *const Tokenizer, chat_template: []const u8, eos_ids: []const u32) void {
    const ids = tokenizer_mod.reservedOutputIds(xfm.allocator, tok.flagged_specials, chat_template, eos_ids) catch |err| {
        log.warn("[suppress] derivation failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    // The mask's length is the LOGITS dim, not the embedding table's:
    // inkling slices its lm_head to `unpadded_vocab_size`, and a mask sized
    // to the padded vocab would fail the `where` broadcast on every sample.
    const logits_dim: usize = if (xfm.config.unpadded_vocab_size > 0)
        xfm.config.unpadded_vocab_size
    else
        xfm.config.vocab_size;
    // Padding rows past the tokenizer's last defined id. A checkpoint that
    // DECLARES `unpadded_vocab_size` has already had those rows sliced off the
    // lm_head, so `logits_dim` IS the real width there — applying the
    // tokenizer's cut on top would be a second, different trim of a vocab that
    // is already trimmed. One trim, whichever the checkpoint provides.
    const defined_vocab: usize = if (xfm.config.unpadded_vocab_size > 0) 0 else tok.definedVocabSize();
    const pad_rows: usize = if (defined_vocab > 0 and defined_vocab < logits_dim)
        logits_dim - defined_vocab
    else
        0;
    if (ids.len == 0 and pad_rows == 0) return;
    xfm.suppress_mask = buildSuppressMask(ids, defined_vocab, logits_dim, xfm.s) catch |err| {
        log.warn("[suppress] mask build failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    log.info(
        "[suppress] {d} of {d} flagged specials masked from sampling (template + eos exempt); {d} padding rows past id {d} of {d}\n",
        .{ ids.len, tok.flagged_specials.len, pad_rows, defined_vocab, logits_dim },
    );
}

/// Sorted, disjoint. U+2E80-U+DFFF is one span on purpose: every lead byte
/// E3-ED is blocked whole, Yi and the rare Latin/Cyrillic extensions there
/// included.
const cjk_ranges = [_][2]u21{
    .{ 0x1100, 0x11FF },   .{ 0x2E80, 0xDFFF },   .{ 0xF900, 0xFAFF },   .{ 0xFE10, 0xFE1F },
    .{ 0xFE30, 0xFE6F },   .{ 0xFF00, 0xFFEF },   .{ 0x16FE0, 0x16FFF }, .{ 0x1AFF0, 0x1B16F },
    .{ 0x1F200, 0x1F2FF }, .{ 0x20000, 0x3FFFF },
};

fn cjkSpan(lo: u21, hi: u21) bool {
    for (cjk_ranges) |r| {
        if (lo >= r[0] and hi <= r[1]) return true;
    }
    return false;
}

/// A masked model spells CJK from byte tokens: block any prefix whose every completion is CJK, plus a
/// bare F0/EF (plane-2 Han, compatibility ideographs; emoji and VS16 never need it). E1/E2 alone stay
/// open for quotes, dashes and ⚠, so Jamo and radicals spelled that way are not caught.
pub fn cjkBytes(b: []const u8) bool {
    var i: usize = 0;
    while (i < b.len) : (i += 1) {
        const len: usize = switch (b[i]) {
            0xC2...0xDF => 2,
            0xE0...0xEF => 3,
            0xF0...0xF4 => 4,
            else => continue,
        };
        var cp: u21 = b[i] & (@as(u8, 0x7F) >> @intCast(len));
        var n: usize = 1;
        while (n < len and i + n < b.len and (b[i + n] & 0xC0) == 0x80) : (n += 1) {
            cp = (cp << 6) | (b[i + n] & 0x3F);
        }
        if (n == len) {
            if (cjkSpan(cp, cp)) return true;
        } else if (i + n == b.len) {
            if (n == 1 and (b[i] == 0xF0 or b[i] == 0xEF)) return true;
            const open: u5 = @intCast(6 * (len - n));
            const lo = cp << open;
            if (cjkSpan(lo, lo | ((@as(u21, 1) << open) - 1))) return true;
        }
        i += n - 1;
    }
    return false;
}

/// Marks every id whose decoded bytes `cjkBytes`. Specials stay open: EOS and
/// the template markers must remain sampleable.
pub fn fillCjkMask(buf: []bool, tok: *const Tokenizer) !usize {
    const alloc = std.heap.page_allocator;
    var specials = try std.DynamicBitSetUnmanaged.initEmpty(alloc, buf.len);
    defer specials.deinit(alloc);
    var sit = tok.special_tokens.valueIterator();
    while (sit.next()) |id| {
        if (id.* < buf.len) specials.set(id.*);
    }
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    var n: usize = 0;
    var one: [1]u8 = undefined;
    var it = tok.id_to_token.iterator();
    while (it.next()) |e| {
        const id = e.key_ptr.*;
        if (id >= buf.len or specials.isSet(id)) continue;
        const bytes: []const u8 = if (byteFallbackPiece(tok, e.value_ptr.*)) |b| blk: {
            one[0] = b;
            break :blk &one;
        } else blk: {
            _ = arena.reset(.retain_capacity);
            break :blk tok.decode(arena.allocator(), &[_]u32{id}, false) catch continue;
        };
        if (cjkBytes(bytes)) {
            buf[id] = true;
            n += 1;
        }
    }
    return n;
}

/// A SentencePiece `<0xNN>` piece decodes alone to U+FFFD; the mask needs its raw byte.
fn byteFallbackPiece(tok: *const Tokenizer, piece: []const u8) ?u8 {
    if (!tok.byte_fallback_decode or piece.len != 6 or !std.mem.startsWith(u8, piece, "<0x") or piece[5] != '>') return null;
    return std.fmt.parseInt(u8, piece[3..5], 16) catch null;
}

/// Never fails a load: without the mask a `suppress_cjk` request gets a 400.
pub fn installCjkMask(xfm: *Transformer, tok: *const Tokenizer) void {
    const logits_dim: usize = if (xfm.config.unpadded_vocab_size > 0)
        xfm.config.unpadded_vocab_size
    else
        xfm.config.vocab_size;
    const alloc = std.heap.page_allocator;
    const buf = alloc.alloc(bool, logits_dim) catch {
        log.warn("[suppress] CJK mask allocation failed; suppress_cjk off\n", .{});
        return;
    };
    defer alloc.free(buf);
    @memset(buf, false);
    const n = fillCjkMask(buf, tok) catch |err| {
        log.warn("[suppress] CJK mask derivation failed ({s}); suppress_cjk off\n", .{@errorName(err)});
        return;
    };
    if (n == 0) {
        log.warn("[suppress] no CJK token in this vocabulary; suppress_cjk off\n", .{});
        return;
    }
    const shape = [_]c_int{@intCast(logits_dim)};
    const cjk = mlx.mlx_array_new_data(buf.ptr, &shape, 1, .bool_);
    if (xfm.suppress_mask) |reserved| {
        defer _ = mlx.mlx_array_free(cjk);
        var merged = mlx.mlx_array_new();
        mlx.check(mlx.mlx_logical_or(&merged, reserved, cjk, xfm.s)) catch |err| {
            _ = mlx.mlx_array_free(merged);
            log.warn("[suppress] CJK mask merge failed ({s}); suppress_cjk off\n", .{@errorName(err)});
            return;
        };
        mlx.check(mlx.mlx_array_eval(merged)) catch |err| {
            _ = mlx.mlx_array_free(merged);
            log.warn("[suppress] CJK mask merge failed ({s}); suppress_cjk off\n", .{@errorName(err)});
            return;
        };
        xfm.suppress_mask_cjk = merged;
    } else {
        xfm.suppress_mask_cjk = cjk;
    }
    log.info("[suppress] suppress_cjk ready: {d} CJK tokens of {d}\n", .{ n, logits_dim });
}

/// `out = where(mask, -inf, logits)` — the masked lanes get EXACTLY -inf
/// (never an additive -inf, whose 0×-inf/NaN edge the parity rules exist
/// for). `[V]` broadcasts over both `[1, V]` and `[1, 1, V]` logits.
pub fn applySuppressMask(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(out, mask, neg_inf, logits, s));
}

/// A scalar in the array's own dtype, so a masked bf16 block is not widened to
/// f32 by its own `-inf`.
pub fn scalarLike(v: f32, like: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(like), s));
    return out;
}

// ── tests ──


// ── Tests ──

const testing = std.testing;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

pub fn evalLazyToken(lazy: mlx.mlx_array) !u32 {
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    return @intCast(val);
}

test "fillSuppressMask: padding rows past the tokenizer's last defined id are suppressed" {
    // A config's `vocab_size` is a padded matrix dimension, not a vocabulary.
    // qwen4_exp declares 248320 while the tokenizer defines 248044 + 33 =
    // 248077 ids: 243 rows that decode to NOTHING, carry whatever the
    // initializer left, and were sampleable — a drawn one emits no text while
    // consuming a step and entering the KV. Same mask, second row class.
    var buf: [16]bool = undefined;
    fillSuppressMask(&buf, &[_]u32{3}, 12);
    for (buf, 0..) |v, i| {
        const want = (i == 3) or (i >= 12);
        try testing.expectEqual(want, v);
    }

    // No padding known (0) and a fully-defined vocab (== len) both add
    // nothing: a mask over the WHOLE vocab is -inf everywhere, and argmax
    // over that row returns id 0 — the all-false-mask class from the other
    // side. The reserved id must still be the only true.
    for ([_]usize{ 0, 16, 99 }) |defined| {
        fillSuppressMask(&buf, &[_]u32{3}, defined);
        for (buf, 0..) |v, i| try testing.expectEqual(i == 3, v);
    }
}

test "cjkBytes: CJK code points, CJK-only prefixes and a bare F0/EF are caught, E1/E2 alone stay open" {
    for ([_][]const u8{
        "北京",
        " 的",
        "。",
        "，",
        "（",
        "한국",
        "カタカナ",
        "ｶ",
        "⼀",
        "\u{20000}",
        "🈚",
        "\xe4",
        "\xea",
        "\xed",
        "\xe2\xbc",
        "\xef\xbc",
        "\xf0\xa0",
        "ab\xe5\x8c",
        "\xf0\x9f\x88",
        "\xf0",
        "a\xef",
    }) |t| try testing.expect(cjkBytes(t));
    for ([_][]const u8{
        "",
        "„Grüße“ – ok…",
        "Běijīng",
        "€",
        "\u{FE0F}",
        "\u{FFFD}",
        "😀",
        "\xe2",
        "\xe1",
        "\xf0\x9f",
        "\xef\xb8",
        "\xbc\x8c",
    }) |t| try testing.expect(!cjkBytes(t));
}

test "fillCjkMask: CJK tokens are marked, specials and everything else stay open" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    try tok.id_to_token.put(0, "a");
    try tok.id_to_token.put(1, "北");
    try tok.id_to_token.put(2, "„");
    try tok.id_to_token.put(3, "<|北|>");
    try tok.special_tokens.put(try allocator.dupe(u8, "<|北|>"), 3);
    var buf: [5]bool = @splat(false);
    try testing.expectEqual(@as(usize, 1), try fillCjkMask(&buf, &tok));
    try testing.expectEqualSlices(bool, &.{ false, true, false, false, false }, &buf);
}

test "fillCjkMask: byte-level pieces are judged by their raw bytes" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    for (tok.byte_to_unicode, 0..) |cp, b| try tok.unicode_to_byte.put(cp, @intCast(b));
    const raw = [_][]const u8{ "\xe4", "\xe2", "\xef\xbc\x8c", "\xe2\x80\x9e", "\xef\xb8\x8f" };
    var pieces: [raw.len][]u8 = undefined;
    for (raw, 0..) |bytes, id| {
        var piece: std.ArrayList(u8) = .empty;
        for (bytes) |b| {
            var enc: [4]u8 = undefined;
            const n = try std.unicode.utf8Encode(tok.byte_to_unicode[b], &enc);
            try piece.appendSlice(allocator, enc[0..n]);
        }
        pieces[id] = try piece.toOwnedSlice(allocator);
        try tok.id_to_token.put(@intCast(id), pieces[id]);
    }
    defer for (pieces) |piece| allocator.free(piece);
    var buf: [raw.len]bool = @splat(false);
    try testing.expectEqual(@as(usize, 2), try fillCjkMask(&buf, &tok));
    try testing.expectEqualSlices(bool, &.{ true, false, true, false, false }, &buf);
}

test "Tokenizer.definedVocabSize: highest defined id + 1, 0 when nothing is defined" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 0), tok.definedVocabSize());
    // Added specials sit ABOVE the base vocab and are not contiguous with it,
    // so the answer is the maximum key, never the entry count.
    try tok.id_to_token.put(0, "a");
    try tok.id_to_token.put(1, "b");
    try tok.id_to_token.put(11, "<|special|>");
    try testing.expectEqual(@as(usize, 12), tok.definedVocabSize());
}

test "suppress_mask: a suppressed id is unreachable from both samplers, everything else stays" {
    // A collapsed distribution can rank `<|fim_hole|>` (a reserved FIM marker) in the
    // top-5 at every degenerate position, and greedy DREW it live. A reserved
    // marker in chat output is always a bug, so the sampler carries a
    // model-lifetime suppression mask. Contract pinned here: with the mask,
    // neither sampler can return a suppressed id on any of its paths (greedy
    // fast path, greedy general, categorical, sync sampleToken); with a null
    // mask the same logits DO pick it (red-on-revert); and logprobs keep
    // reporting the MODEL's raw distribution — the suppressed id still ranks
    // where the model put it, because the field reports the model while the
    // mask is sampling policy.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const V: usize = 8;
    const mask = try buildSuppressMask(&[_]u32{3}, 0, V, s);
    defer _ = mlx.mlx_array_free(mask);

    // id 3 (suppressed) dominates, id 5 is the runner-up, the rest are so far
    // down that after masking the categorical underflows to exactly id 5.
    var data = [_]f32{ -1000, -1000, -1000, 100, -1000, 50, -1000, -1000 };

    // Greedy fast path: [1, 1, V].
    const shape3 = [_]c_int{ 1, 1, V };
    const logits3 = mlx.mlx_array_new_data(&data, &shape3, 3, .float32);
    defer _ = mlx.mlx_array_free(logits3);
    const masked_sp = SamplingParams{ .temperature = 0.0, .suppress_mask = mask };
    const open_sp = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_sp, s)));
    try testing.expectEqual(@as(u32, 3), try evalLazyToken(sampleTokenLazy(logits3, open_sp, s)));

    // Categorical path at temp 1.0 (exercises the seq_len==1 reshape arm and
    // the sampler proper; the masked lane's probability is exactly 0 and the
    // -1000 lanes underflow, so the draw is deterministic).
    const masked_t1 = SamplingParams{ .temperature = 1.0, .suppress_mask = mask };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_t1, s)));

    // Sync sampler (logprobs/penalty path): same policy, and the reported
    // distribution stays the model's own — rank 1 is the SUPPRESSED id.
    const r = try sampleToken(allocator, logits3, masked_sp, null, 2, s);
    const lp = r.logprob_result orelse return error.NoLogprobs;
    defer allocator.free(lp.top_logprobs);
    try testing.expectEqual(@as(u32, 5), r.token_id);
    try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);

    // Stochastic-verify filters: a suppressed draft's acceptance probability
    // must be exactly 0 (always rejected), and the residual it corrects from
    // can never re-draw it.
    const probs = try probsAtLastPos(logits3, masked_t1, s);
    defer _ = mlx.mlx_array_free(probs);
    try testing.expectEqual(@as(f32, 0.0), try probAt(probs, 3, s));
    try testing.expect(try probAt(probs, 5, s) > 0.99);
}

test "no block decoder commits an argmax over a suppressed id" {
    // A row whose raw argmax is a padding id past the defined vocabulary, plus a
    // flagged special: both must lose to the best legal column. Run at the block
    // widths the committing decoders verify at (MTP and the gemma drafter `1 + m`,
    // DFlash a whole block), so a per-position axis bug cannot hide. WHICH decoders
    // are covered is a compile-time property: every commit site takes a
    // `CommittedArgmax`, and only `verifyArgmax` returns one.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 512;
    const defined: usize = 500;
    const flagged = [_]u32{7};
    for ([_]usize{ 2, 3, 5 }) |rows| try suppressedArgmaxCase(allocator, s, v, defined, &flagged, rows);
}

fn suppressedArgmaxCase(allocator: std.mem.Allocator, s: mlx.mlx_stream, v: usize, defined: usize, flagged: []const u32, rows: usize) !void {
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    @memset(host, 0.0);
    for (0..rows) |r| {
        const row = host[r * v ..][0..v];
        row[505] = 9.0; // padding row: highest of all
        row[7] = 8.0; // flagged special: second
        row[100 + r] = 7.0; // the best LEGAL id, different per position
    }
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const f32_block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f32_block);
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
    const mask = try buildSuppressMask(flagged, defined, v, s);
    defer _ = mlx.mlx_array_free(mask);

    var raw = try verifyArgmax(block, null, s);
    defer raw.deinit();
    const raw_ids = try samplerTestReadFlat(allocator, raw.lazy(), rows, s);
    defer allocator.free(raw_ids);
    for (raw_ids) |id| try testing.expectEqual(@as(f32, 505), id);

    var masked = try verifyArgmax(block, mask, s);
    defer masked.deinit();
    const ids = try samplerTestReadFlat(allocator, masked.lazy(), rows, s);
    defer allocator.free(ids);
    for (ids, 0..) |id, r| try testing.expectEqual(@as(f32, @floatFromInt(100 + r)), id);
}
