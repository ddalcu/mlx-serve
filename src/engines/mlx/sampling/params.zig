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
