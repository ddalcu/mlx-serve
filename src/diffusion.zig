//! DiffusionGemma block-diffusion generation.
//!
//! Algorithm (faithful to HF transformers `generation_diffusion_gemma.py`
//! and mlx-vlm's `stream_diffusion_generate`):
//!
//!   1. ENCODE the prompt causally (forwardWith + use_encoder_scalars +
//!      skip_lm_head) — fills the KV cache like any autoregressive prefill.
//!   2. Fill a canvas (≤ config.canvas_length tokens) with UNIFORM RANDOM
//!      vocab tokens (no mask token in this scheme).
//!   3. Denoise up to `max_denoising_steps` times: each step re-forwards the
//!      whole canvas through the BIDIRECTIONAL decoder
//!      (forwardDiffusionDecoder), scales logits by a linear temperature
//!      schedule (t_max → t_min as steps count down), samples a candidate
//!      canvas, and ACCEPTS the lowest-entropy positions under the
//!      entropy-bound rule; rejected positions are re-randomized. The next
//!      step is conditioned on this step's softmax via soft embeddings
//!      (self-conditioning).
//!   4. Early-stop when the argmax canvas is stable across
//!      `stability_threshold` steps AND mean token entropy <
//!      `confidence_threshold`.
//!   5. COMMIT the argmax canvas, then causally re-encode it (encoder pass)
//!      to extend the KV cache; repeat from 2 until EOS or max_tokens.
//!
//! The scheduler drives one canvas per decode tick (runDiffusionDecodeTick)
//! and pushes committed tokens through the normal slot machinery, so
//! streaming arrives block-wise on every HTTP surface.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");

const Transformer = transformer_mod.Transformer;
const ForwardCtx = transformer_mod.ForwardCtx;

/// Variable canvas sizing (mlx-vlm): a short completion shouldn't pay for a
/// full 256-token canvas. The canvas is min(model_canvas, max(remaining,
/// MIN_CANVAS)) — large enough to place an EOS, small enough to be cheap.
pub const MIN_CANVAS_LENGTH: u32 = 64;

/// Encoder prefill chunk size (matches the AR path's default prefill stride).
const PREFILL_CHUNK: usize = 2048;

/// Linear logits-temperature schedule. `cur_step` counts DOWN
/// (max_steps .. 1), so the first denoising step is the hottest (T = t_max)
/// and the last lands just above t_min — never exactly t_min.
pub fn linearTemperature(cur_step: u32, max_steps: u32, t_min: f32, t_max: f32) f32 {
    const frac = @as(f32, @floatFromInt(cur_step)) / @as(f32, @floatFromInt(max_steps));
    return t_min + (t_max - t_min) * frac;
}

/// Per-position Shannon entropy (nats) over the vocab axis of fp32 logits
/// [..., V] → [...]. Computed via the numerically-stable
/// log_probs = logits − logsumexp chain (same as the reference).
pub fn tokenEntropy(logits_f32: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits_f32, -1, true, s));
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_subtract(&log_probs, logits_f32, lse, s));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_exp(&probs, log_probs, s));
    var plogp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(plogp);
    try mlx.check(mlx.mlx_multiply(&plogp, probs, log_probs, s));
    var summed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(summed);
    try mlx.check(mlx.mlx_sum_axis(&summed, plogp, -1, false, s));
    var ent = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_negative(&ent, summed, s));
    return ent;
}

/// EntropyBoundSampler acceptance mask. Sort positions by entropy ascending;
/// accept the sorted prefix while `cumsum(H) − cummax(H) ≤ entropy_bound`
/// (cummax of an ascending sort is the running prefix max, i.e. the
/// position's own entropy — so the criterion is "the rest of the accepted
/// set's joint entropy stays under the bound"). The single lowest-entropy
/// position always qualifies (cum − max = 0), so every step accepts ≥ 1.
/// `entropy` is [B, L] fp32; returns a bool mask of the same shape.
pub fn entropyTransferMask(entropy: mlx.mlx_array, entropy_bound: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var sorted_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_idx);
    try mlx.check(mlx.mlx_argsort_axis(&sorted_idx, entropy, -1, s));
    var sorted_ent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_ent);
    try mlx.check(mlx.mlx_take_along_axis(&sorted_ent, entropy, sorted_idx, -1, s));
    var cum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cum);
    try mlx.check(mlx.mlx_cumsum(&cum, sorted_ent, -1, false, true, s));
    var cmax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cmax);
    try mlx.check(mlx.mlx_cummax(&cmax, sorted_ent, -1, false, true, s));
    var excess = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(excess);
    try mlx.check(mlx.mlx_subtract(&excess, cum, cmax, s));
    const bound_scalar = mlx.mlx_array_new_float(entropy_bound);
    defer _ = mlx.mlx_array_free(bound_scalar);
    var sel_sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_sorted);
    try mlx.check(mlx.mlx_less_equal(&sel_sorted, excess, bound_scalar, s));

    // Scatter back to original position order.
    const ent_shape = mlx.getShape(entropy);
    var zeros_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros_mask);
    try mlx.check(mlx.mlx_zeros(&zeros_mask, ent_shape.ptr, ent_shape.len, .bool_, s));
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros_mask, sorted_idx, sel_sorted, -1, s));
    return mask;
}

/// Uniform-random canvas [1, len] in [0, vocab) — the "noise" of uniform
/// state diffusion (canvas init and per-step renoise of rejected positions).
pub fn randomCanvas(len: c_int, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const low = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(low);
    const high = mlx.mlx_array_new_int(vocab);
    defer _ = mlx.mlx_array_free(high);
    const shape = [_]c_int{ 1, len };
    var canvas = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_randint(&canvas, low, high, &shape, 2, .int32, .{ .ctx = null }, s));
    return canvas;
}

/// Schedule-scaled fp32 logits: astype(fp32) / temperature.
pub fn scaleLogits(logits: mlx.mlx_array, temperature: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var f32_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_logits);
    try mlx.check(mlx.mlx_astype(&f32_logits, logits, .float32, s));
    const t_scalar = mlx.mlx_array_new_float(temperature);
    defer _ = mlx.mlx_array_free(t_scalar);
    var scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&scaled, f32_logits, t_scalar, s));
    return scaled;
}

/// Argmax over the vocab axis → int32 token ids [B, L].
pub fn argmaxTokens(processed: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, processed, -1, false, s));
    var ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&ids, am, .int32, s));
    return ids;
}

/// Denoiser candidate canvas. `user_temp < 0.01` → greedy argmax (mlx-vlm's
/// temp-0 path); otherwise a per-position categorical sample of
/// processed/user_temp (user_temp 1.0 ≡ the HF reference's multinomial of
/// the schedule-scaled logits).
pub fn sampleTokens(processed: mlx.mlx_array, user_temp: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    if (user_temp < 0.01) return argmaxTokens(processed, s);
    var logits = processed;
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (user_temp != 1.0) {
        const t_scalar = mlx.mlx_array_new_float(user_temp);
        defer _ = mlx.mlx_array_free(t_scalar);
        try mlx.check(mlx.mlx_divide(&scaled, logits, t_scalar, s));
        logits = scaled;
    }
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, logits, -1, .{ .ctx = null }, s));
    var ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&ids, sampled, .int32, s));
    return ids;
}

/// Elementwise select — accepted positions take `a`, the rest keep `b`.
pub fn whereSelect(mask: mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&out, mask, a, b, s));
    return out;
}

/// All-elements equality between two same-shaped arrays. SYNCS the stream
/// (item read) — callers batch this with the per-step eval point.
pub fn arraysEqualAll(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, a, b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var result: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&result, all));
    return result;
}

/// Mean of a float array as a host scalar. SYNCS the stream.
pub fn meanScalar(arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var flat_mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_mean);
    try mlx.check(mlx.mlx_mean(&flat_mean, arr, false, s));
    try mlx.check(mlx.mlx_array_eval(flat_mean));
    var result: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&result, flat_mean));
    return result;
}

/// Self-conditioning soft embeddings: softmax(processed) @ embed_table,
/// scaled by √hidden (the same embed scale the canvas token embeddings get).
/// `emb_table` is the DENSE [V, H] table (dequantized once per request);
/// `emb_scale` is the transformer's bf16 √hidden scalar (null → unscaled).
pub fn softEmbeddings(
    processed_f32: mlx.mlx_array,
    emb_table: mlx.mlx_array,
    emb_scale: ?mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, processed_f32, -1, true, s));
    var probs_cast = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs_cast);
    try mlx.check(mlx.mlx_astype(&probs_cast, probs, mlx.mlx_array_dtype(emb_table), s));
    var soft = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&soft, probs_cast, emb_table, s));
    if (emb_scale) |es| {
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, soft, es, s));
        _ = mlx.mlx_array_free(soft);
        return scaled;
    }
    return soft;
}

/// One committed canvas block.
pub const CanvasResult = struct {
    /// Committed (argmax) canvas token ids; caller owns.
    tokens: []u32,
    /// Denoising steps this canvas used (≤ max_denoising_steps).
    steps: u32,
};

/// Per-request block-diffusion driver. Owns the dequantized embedding table
/// (for self-conditioning) and the pending-encode buffer; borrows the
/// transformer and the slot's ForwardCtx (whose KV cache holds the
/// committed-context state between canvases).
pub const Runner = struct {
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    ctx: *ForwardCtx,
    emb_table: mlx.mlx_array,
    user_temp: f32,
    max_tokens: u32,
    committed: u32 = 0,
    canvas_index: u32 = 0,
    total_steps: u32 = 0,
    /// Committed canvas awaiting causal re-encode at the next tick (owned).
    pending_encode: ?[]u32 = null,
    /// Cooperative cancellation — checked once per denoising step.
    cancel_flag: ?*const std.atomic.Value(bool) = null,

    pub fn init(
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        ctx: *ForwardCtx,
        user_temp: f32,
        max_tokens: u32,
    ) !Runner {
        const emb_table = try xfm.dequantizedEmbedding();
        return .{
            .allocator = allocator,
            .xfm = xfm,
            .ctx = ctx,
            .emb_table = emb_table,
            .user_temp = user_temp,
            .max_tokens = max_tokens,
        };
    }

    pub fn deinit(self: *Runner) void {
        _ = mlx.mlx_array_free(self.emb_table);
        if (self.pending_encode) |p| self.allocator.free(p);
        self.pending_encode = null;
    }

    /// Causal encoder pass over `ids`, chunked. Fills the KV cache and
    /// advances ctx.moe_seq_offset; output hidden is discarded.
    fn encodeTokens(self: *Runner, ids: []const u32) !void {
        const prev_enc = self.ctx.use_encoder_scalars;
        const prev_skip = self.ctx.skip_lm_head;
        self.ctx.use_encoder_scalars = true;
        self.ctx.skip_lm_head = true;
        defer {
            self.ctx.use_encoder_scalars = prev_enc;
            self.ctx.skip_lm_head = prev_skip;
        }

        const ids_i32 = try self.allocator.alloc(i32, ids.len);
        defer self.allocator.free(ids_i32);
        for (ids, 0..) |t, i| ids_i32[i] = @intCast(t);

        var pos: usize = 0;
        while (pos < ids.len) {
            const chunk_len = @min(PREFILL_CHUNK, ids.len - pos);
            const chunk_shape = [_]c_int{ 1, @intCast(chunk_len) };
            const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(chunk_input);
            const hidden = try self.xfm.forwardWith(self.ctx, chunk_input);
            try mlx.check(mlx.mlx_array_eval(hidden));
            _ = mlx.mlx_array_free(hidden);
            pos += chunk_len;
        }
    }

    /// Prompt prefill (encoder). Call once before the first nextCanvas.
    pub fn prefill(self: *Runner, prompt_ids: []const u32) !void {
        try self.encodeTokens(prompt_ids);
    }

    /// Denoise and commit one canvas. Encodes the previously committed
    /// canvas first (deferred so an EOS-terminated request never pays for
    /// an encode it doesn't need). Returns null when max_tokens is reached.
    pub fn nextCanvas(self: *Runner, allocator: std.mem.Allocator) !?CanvasResult {
        if (self.committed >= self.max_tokens) return null;
        const cfg = &self.xfm.config;

        if (self.pending_encode) |ids| {
            try self.encodeTokens(ids);
            self.allocator.free(ids);
            self.pending_encode = null;
        }

        const remaining: u32 = self.max_tokens - self.committed;
        const min_canvas = @min(MIN_CANVAS_LENGTH, cfg.canvas_length);
        const canvas_len: u32 = @min(cfg.canvas_length, @max(remaining, min_canvas));
        const vocab: c_int = @intCast(cfg.vocab_size);
        self.canvas_index += 1;

        var canvas = try randomCanvas(@intCast(canvas_len), vocab, self.xfm.s);
        defer _ = mlx.mlx_array_free(canvas);
        var sc_embeds: ?mlx.mlx_array = null;
        defer if (sc_embeds) |sc| {
            _ = mlx.mlx_array_free(sc);
        };
        var argmax_canvas: ?mlx.mlx_array = null;
        defer if (argmax_canvas) |am| {
            _ = mlx.mlx_array_free(am);
        };
        // Stability history: the last `stability_threshold` argmax canvases.
        const stability: usize = @max(1, cfg.diffusion_stability_threshold);
        const history = try self.allocator.alloc(?mlx.mlx_array, stability);
        defer {
            for (history) |h| if (h) |arr| {
                _ = mlx.mlx_array_free(arr);
            };
            self.allocator.free(history);
        }
        for (history) |*h| h.* = null;

        const max_steps = cfg.diffusion_max_steps;
        var steps_used: u32 = 0;
        var cur_step: u32 = max_steps;
        while (cur_step >= 1) : (cur_step -= 1) {
            if (self.cancel_flag) |cf| {
                if (cf.load(.acquire)) return error.Cancelled;
            }
            steps_used += 1;

            const logits = try self.xfm.forwardDiffusionDecoder(self.ctx, canvas, sc_embeds);
            defer _ = mlx.mlx_array_free(logits);
            const temp = linearTemperature(cur_step, max_steps, cfg.diffusion_t_min, cfg.diffusion_t_max);
            const processed = try scaleLogits(logits, temp, self.xfm.s);
            defer _ = mlx.mlx_array_free(processed);

            const new_argmax = try argmaxTokens(processed, self.xfm.s);
            if (argmax_canvas) |old| {
                _ = mlx.mlx_array_free(old);
            }
            argmax_canvas = new_argmax;

            // Last step commits the argmax directly — no sampler work.
            if (cur_step == 1) break;

            const denoiser = try sampleTokens(processed, self.user_temp, self.xfm.s);
            defer _ = mlx.mlx_array_free(denoiser);
            const entropy = try tokenEntropy(processed, self.xfm.s);
            defer _ = mlx.mlx_array_free(entropy);
            const accept_mask = try entropyTransferMask(entropy, cfg.diffusion_entropy_bound, self.xfm.s);
            defer _ = mlx.mlx_array_free(accept_mask);

            // accepted = where(mask, denoiser, canvas); rejected → renoise
            const accepted = try whereSelect(accept_mask, denoiser, canvas, self.xfm.s);
            defer _ = mlx.mlx_array_free(accepted);
            const noise = try randomCanvas(@intCast(canvas_len), vocab, self.xfm.s);
            defer _ = mlx.mlx_array_free(noise);
            const new_canvas = try whereSelect(accept_mask, accepted, noise, self.xfm.s);
            _ = mlx.mlx_array_free(canvas);
            canvas = new_canvas;
            try mlx.check(mlx.mlx_array_eval(canvas));

            if (std.c.getenv("MLX_SERVE_DIFFUSION_TRACE") != null) {
                const me = try meanScalar(entropy, self.xfm.s);
                var mask_i32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(mask_i32);
                mlx.check(mlx.mlx_astype(&mask_i32, accept_mask, .float32, self.xfm.s)) catch {};
                const acc_frac = meanScalar(mask_i32, self.xfm.s) catch 0;
                log.info("[diffusion-trace] step={d} mean_ent={d:.4} accepted={d:.0}\n", .{ max_steps - cur_step + 1, me, acc_frac * @as(f32, @floatFromInt(canvas_len)) });
            }

            // Stable & confident early stop. Stability first (cheap int
            // compare); entropy mean only when the argmax has settled.
            var stable = true;
            for (history) |h| {
                if (h) |prev| {
                    if (!try arraysEqualAll(new_argmax, prev, self.xfm.s)) {
                        stable = false;
                        break;
                    }
                } else stable = false;
            }
            // Roll history (drop oldest, push current).
            if (history[0]) |oldest| {
                _ = mlx.mlx_array_free(oldest);
            }
            for (0..stability - 1) |i| history[i] = history[i + 1];
            var current_copy = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&current_copy, new_argmax));
            history[stability - 1] = current_copy;

            if (stable) {
                const mean_ent = try meanScalar(entropy, self.xfm.s);
                if (mean_ent < cfg.diffusion_confidence_threshold) break;
            }

            // Self-conditioning for the next step.
            const next_sc = try softEmbeddings(processed, self.emb_table, self.xfm.emb_scale, self.xfm.s);
            if (sc_embeds) |old| {
                _ = mlx.mlx_array_free(old);
            }
            sc_embeds = next_sc;
            try mlx.check(mlx.mlx_array_eval(sc_embeds.?));
        }

        // Commit the ARGMAX canvas (not the sampled/accepted one).
        const committed_arr = argmax_canvas.?;
        try mlx.check(mlx.mlx_array_eval(committed_arr));
        const data = mlx.mlx_array_data_int32(committed_arr) orelse return error.DiffusionCommitReadFailed;
        const tokens = try allocator.alloc(u32, canvas_len);
        errdefer allocator.free(tokens);
        for (tokens, 0..) |*t, i| t.* = @intCast(@max(0, data[i]));

        self.committed += canvas_len;
        self.total_steps += steps_used;
        self.pending_encode = try self.allocator.dupe(u32, tokens);
        log.debug("[diffusion] committed head: {any}\n", .{tokens[0..@min(tokens.len, 24)]});
        log.info(
            "[diffusion] canvas={d} len={d} steps={d}/{d} (tokens/forward={d:.1})\n",
            .{ self.canvas_index, canvas_len, steps_used, max_steps, @as(f32, @floatFromInt(canvas_len)) / @as(f32, @floatFromInt(steps_used)) },
        );
        return .{ .tokens = tokens, .steps = steps_used };
    }
};

// ── Tests ──

const testing = std.testing;

test "diffusion linearTemperature endpoints and direction" {
    // First step (cur_step = max) is the hottest: exactly t_max.
    try testing.expectApproxEqAbs(@as(f32, 0.8), linearTemperature(48, 48, 0.4, 0.8), 0.0001);
    // Last step is t_min + span/48 — never exactly t_min.
    try testing.expectApproxEqAbs(@as(f32, 0.4 + 0.4 / 48.0), linearTemperature(1, 48, 0.4, 0.8), 0.0001);
    // Monotonic in cur_step.
    try testing.expect(linearTemperature(24, 48, 0.4, 0.8) < linearTemperature(25, 48, 0.4, 0.8));
}

test "diffusion tokenEntropy matches hand-computed values" {
    const s = mlx.gpuStream();
    // Two positions over a 4-token vocab:
    //   pos 0: uniform logits → H = ln(4) ≈ 1.386294
    //   pos 1: one-hot-ish (large gap) → H ≈ 0
    var host = [_]f32{
        0.0,  0.0,  0.0,  0.0,
        20.0, 0.0,  0.0,  0.0,
    };
    const shape = [_]c_int{ 1, 2, 4 };
    const logits = mlx.mlx_array_new_data(&host, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const ent = try tokenEntropy(logits, s);
    defer _ = mlx.mlx_array_free(ent);
    try mlx.check(mlx.mlx_array_eval(ent));
    const data = mlx.mlx_array_data_float32(ent) orelse return error.TestUnexpectedNullData;
    try testing.expectApproxEqAbs(@as(f32, 1.3862944), data[0], 0.0001);
    try testing.expect(data[1] < 0.001);
}

test "diffusion entropyTransferMask accepts low-entropy prefix under the bound" {
    const s = mlx.gpuStream();
    // Entropies (original order): [0.5, 0.001, 0.002, 0.3]
    // Ascending sort: [0.001, 0.002, 0.3, 0.5] at positions [1, 2, 3, 0].
    // cum  = [0.001, 0.003, 0.303, 0.803]
    // cmax = [0.001, 0.002, 0.3,   0.5  ]
    // cum−cmax = [0, 0.001, 0.003, 0.303] ≤ 0.1 → [T, T, T, F]
    // → accept positions {1, 2, 3}; reject position 0.
    var host = [_]f32{ 0.5, 0.001, 0.002, 0.3 };
    const shape = [_]c_int{ 1, 4 };
    const ent = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(ent);

    const mask = try entropyTransferMask(ent, 0.1, s);
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_array_eval(mask));
    const data = mlx.mlx_array_data_bool(mask) orelse return error.TestUnexpectedNullData;
    try testing.expectEqual(false, data[0]);
    try testing.expectEqual(true, data[1]);
    try testing.expectEqual(true, data[2]);
    try testing.expectEqual(true, data[3]);
}

test "diffusion entropyTransferMask always accepts at least the lowest-entropy position" {
    const s = mlx.gpuStream();
    // All positions high-entropy: only the sorted-first survives
    // (cum − cmax = 0 ≤ bound by construction).
    var host = [_]f32{ 5.0, 4.0, 6.0 };
    const shape = [_]c_int{ 1, 3 };
    const ent = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(ent);

    const mask = try entropyTransferMask(ent, 0.1, s);
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_array_eval(mask));
    const data = mlx.mlx_array_data_bool(mask) orelse return error.TestUnexpectedNullData;
    try testing.expectEqual(false, data[0]);
    try testing.expectEqual(true, data[1]); // lowest entropy (4.0)
    try testing.expectEqual(false, data[2]);
}

test "diffusion randomCanvas stays in [0, vocab)" {
    const s = mlx.gpuStream();
    const canvas = try randomCanvas(64, 100, s);
    defer _ = mlx.mlx_array_free(canvas);
    try mlx.check(mlx.mlx_array_eval(canvas));
    const data = mlx.mlx_array_data_int32(canvas) orelse return error.TestUnexpectedNullData;
    for (0..64) |i| {
        try testing.expect(data[i] >= 0);
        try testing.expect(data[i] < 100);
    }
}

test "diffusion sampleTokens greedy path equals argmax" {
    const s = mlx.gpuStream();
    var host = [_]f32{
        0.0, 9.0, 1.0,
        7.0, 0.0, 1.0,
    };
    const shape = [_]c_int{ 1, 2, 3 };
    const logits = mlx.mlx_array_new_data(&host, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const ids = try sampleTokens(logits, 0.0, s);
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_array_eval(ids));
    const data = mlx.mlx_array_data_int32(ids) orelse return error.TestUnexpectedNullData;
    try testing.expectEqual(@as(i32, 1), data[0]);
    try testing.expectEqual(@as(i32, 0), data[1]);
}

test "diffusion arraysEqualAll and whereSelect semantics" {
    const s = mlx.gpuStream();
    var a_host = [_]i32{ 1, 2, 3 };
    var b_host = [_]i32{ 1, 9, 3 };
    var m_host = [_]bool{ true, false, true };
    const shape = [_]c_int{ 1, 3 };
    const a = mlx.mlx_array_new_data(&a_host, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(a);
    const b = mlx.mlx_array_new_data(&b_host, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(b);
    const m = mlx.mlx_array_new_data(&m_host, &shape, 2, .bool_);
    defer _ = mlx.mlx_array_free(m);

    try testing.expect(try arraysEqualAll(a, a, s));
    try testing.expect(!try arraysEqualAll(a, b, s));

    const sel = try whereSelect(m, a, b, s);
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_array_eval(sel));
    const data = mlx.mlx_array_data_int32(sel) orelse return error.TestUnexpectedNullData;
    try testing.expectEqual(@as(i32, 1), data[0]);
    try testing.expectEqual(@as(i32, 9), data[1]);
    try testing.expectEqual(@as(i32, 3), data[2]);
}

test "diffusion softEmbeddings is a probability-weighted embedding average" {
    const s = mlx.gpuStream();
    // Vocab 2, hidden 2, table rows [1,0] and [0,1]; logits force ~one-hot
    // on token 1 at pos 0 and uniform at pos 1 → soft embeds ≈ [0,1] and
    // [0.5,0.5].
    var logits_host = [_]f32{
        -30.0, 30.0,
        0.0,   0.0,
    };
    const lshape = [_]c_int{ 1, 2, 2 };
    const logits = mlx.mlx_array_new_data(&logits_host, &lshape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var table_host = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const tshape = [_]c_int{ 2, 2 };
    const table = mlx.mlx_array_new_data(&table_host, &tshape, 2, .float32);
    defer _ = mlx.mlx_array_free(table);

    const soft = try softEmbeddings(logits, table, null, s);
    defer _ = mlx.mlx_array_free(soft);
    try mlx.check(mlx.mlx_array_eval(soft));
    const data = mlx.mlx_array_data_float32(soft) orelse return error.TestUnexpectedNullData;
    try testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), data[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.5), data[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.5), data[3], 0.001);
}

test "diffusion: live converged-canvas self-consistency vs mlx-vlm (DIFFUSION_TEST_MODEL)" {
    // The property block-diffusion convergence depends on: given the real
    // chat prompt and the canvas the REFERENCE implementation converged to,
    // one decoder forward must reproduce that canvas as its argmax.
    // Reference (mlx_vlm 0.6.3, mlx-community/diffusiongemma-26B-A4B-it-4bit):
    //   without self-conditioning: 63/64 (one INT4 near-tie flip at pos 7)
    //   with    self-conditioning: 64/64
    // Logit-value diffs on RANDOM canvases are deliberately NOT asserted —
    // garbage input puts the MoE router on knife-edge ties that amplify
    // harmless kernel-order noise into large logit deltas.
    // Gated on DIFFUSION_TEST_MODEL (path to the checkpoint); skips when
    // unset so CI stays green.
    const raw = std.c.getenv("DIFFUSION_TEST_MODEL") orelse return error.SkipZigTest;
    const dir = std.mem.sliceTo(raw, 0);
    if (dir.len == 0) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    const config = try model_mod.parseConfig(io, alloc, dir);
    try testing.expect(config.isDiffusion());
    var weights = try model_mod.loadWeights(io, alloc, dir);
    defer weights.deinit();
    var xfm = try Transformer.init(io, alloc, config, &weights);
    defer xfm.deinit();

    var ctx = xfm.defaultCtx();

    // Causal encoder pass over the rendered chat prompt
    // "<bos><|turn>user\nSay hello in exactly five words.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>\n".
    ctx.use_encoder_scalars = true;
    ctx.skip_lm_head = true;
    const chat_prompt = [_]i32{ 2, 105, 2364, 107, 37889, 29104, 528, 7121, 3493, 4171, 236761, 106, 107, 105, 4368, 107, 100, 45518, 107, 101, 107 };
    const cp_shape = [_]c_int{ 1, chat_prompt.len };
    const cp_arr = mlx.mlx_array_new_data(&chat_prompt, &cp_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(cp_arr);
    const enc_hidden = try xfm.forwardWith(&ctx, cp_arr);
    try mlx.check(mlx.mlx_array_eval(enc_hidden));
    _ = mlx.mlx_array_free(enc_hidden);
    ctx.use_encoder_scalars = false;
    ctx.skip_lm_head = false;

    // The canvas the reference loop committed for this prompt
    // ("<|channel>thought\n<channel|>Hello to you, friend.<turn|><eos>…").
    var conv_host: [64]i32 = .{1} ** 64;
    const content = [_]i32{ 100, 45518, 107, 101, 9259, 531, 611, 236764, 4389, 236761, 106 };
    @memcpy(conv_host[0..content.len], &content);
    const cv_shape = [_]c_int{ 1, 64 };
    const conv_canvas = mlx.mlx_array_new_data(&conv_host, &cv_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(conv_canvas);

    // (1) Without self-conditioning: bidirectional decoder + cache read.
    {
        const lg = try xfm.forwardDiffusionDecoder(&ctx, conv_canvas, null);
        defer _ = mlx.mlx_array_free(lg);
        const am = try argmaxTokens(lg, xfm.s);
        defer _ = mlx.mlx_array_free(am);
        try mlx.check(mlx.mlx_array_eval(am));
        const ad = mlx.mlx_array_data_int32(am) orelse return error.TestUnexpectedNullData;
        var match: u32 = 0;
        for (0..64) |i| {
            if (ad[i] == conv_host[i]) match += 1;
        }
        std.debug.print("\nself-consistency (no sc): {d}/64\n", .{match});
        // Reference scores 63/64; allow a few INT4 near-tie flips.
        try testing.expect(match >= 58);
    }

    // (2) With self-conditioning built exactly the way the Runner does:
    // softEmbeddings over near-one-hot logits spiked at the canvas tokens.
    // Exercises dequantizedEmbedding + softEmbeddings + the SelfConditioning
    // module end to end. Reference scores 64/64.
    {
        const emb_table = try xfm.dequantizedEmbedding();
        defer _ = mlx.mlx_array_free(emb_table);
        var onehot_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onehot_logits);
        const ol_shape = [_]c_int{ 1, 64, @intCast(xfm.config.vocab_size) };
        try mlx.check(mlx.mlx_zeros(&onehot_logits, &ol_shape, 3, .float32, xfm.s));
        var idx3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idx3);
        const idx_shape = [_]c_int{ 1, 64, 1 };
        try mlx.check(mlx.mlx_reshape(&idx3, conv_canvas, &idx_shape, 3, xfm.s));
        const thirty = mlx.mlx_array_new_float(30.0);
        defer _ = mlx.mlx_array_free(thirty);
        var spiked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(spiked);
        try mlx.check(mlx.mlx_put_along_axis(&spiked, onehot_logits, idx3, thirty, -1, xfm.s));
        const sc_sig = try softEmbeddings(spiked, emb_table, xfm.emb_scale, xfm.s);
        defer _ = mlx.mlx_array_free(sc_sig);

        const lg = try xfm.forwardDiffusionDecoder(&ctx, conv_canvas, sc_sig);
        defer _ = mlx.mlx_array_free(lg);
        const am = try argmaxTokens(lg, xfm.s);
        defer _ = mlx.mlx_array_free(am);
        try mlx.check(mlx.mlx_array_eval(am));
        const ad = mlx.mlx_array_data_int32(am) orelse return error.TestUnexpectedNullData;
        var match: u32 = 0;
        for (0..64) |i| {
            if (ad[i] == conv_host[i]) match += 1;
        }
        std.debug.print("self-consistency (sc): {d}/64\n", .{match});
        try testing.expect(match >= 62);
    }
}

test "diffusion meanScalar" {
    const s = mlx.gpuStream();
    var host = [_]f32{ 1.0, 2.0, 3.0, 6.0 };
    const shape = [_]c_int{ 1, 4 };
    const arr = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);
    try testing.expectApproxEqAbs(@as(f32, 3.0), try meanScalar(arr, s), 0.0001);
}
