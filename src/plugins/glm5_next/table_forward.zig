//! GLM-5.3-Flash's forward behind the model table: the [B, S, hc, D] hyper-connection stream through KDA and DSA
//! sublayers and the MoE, every slot's KV and recurrent state through `CacheOps`. Prefill, decode, verify (rows
//! capture their KDA states) and batched rows (each slot's state advanced by its own call, the MLP shared).
//! copied_from: plugins/glm5_next/forward.zig (`forwardGlm5With`, `glmSublayer`, `glmBranch`, `glmBranchSlots`,
//! `glmKdaSlots`, `glmDsaAttn`, `forwardGlmBatchedDecode`), the ForwardCtx reads replaced by `Slots`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const glm5 = @import("glm5_next.zig");
const slots_mod = @import("slots.zig");
const trunk = @import("trunk.zig");
const kda = @import("kda.zig");
const kda_recurrence = @import("../../sdk/blocks/kda/recurrence.zig");
const spliceVisionRows = @import("../../sdk/vision/splice.zig").spliceVisionRows;
const axisView = @import("../../mlx/array.zig").axisView;
const concatAxis1 = @import("../../mlx/array.zig").concatAxis1;
const glm_moe = @import("moe.zig");
const dsa_prompt = @import("dsa_prompt.zig");
const dsa_append = @import("dsa_append.zig");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");

const Array = mlx.mlx_array;
const Slots = slots_mod.Slots;
const Model = trunk.Model;
const Layer = trunk.Layer;

/// The rows a batched tick carries per trunk forward (the MoE decode kernels' row limit).
pub const MOE_DECODE_BATCH_MAX = 8;

/// One forward's inputs beyond the token ids.
pub const Step = struct {
    slots: Slots,
    /// A batched tick: `ids` holds one next token per slot as rows of one window.
    batched: bool = false,
    /// Per-row KDA state capture (a speculative verify).
    capture: bool = false,
    /// Layers whose stream mean is published (DFlash), and where.
    capture_ids: []const u32 = &.{},
    capture_out: []Array = &.{},
    /// Media rows spliced at the image placeholders of a prompt chunk, from row `media_offset`.
    media: Array = .{ .ctx = null },
    media_offset: usize = 0,
    /// Evaluate the stream (and the recurrent states) every this many layers of a wide forward; 0 = never.
    eval_every: u32 = 0,
};

/// The final-normed hidden `[1, S, H]`; the head is the caller's.
pub fn hidden(m: *Model, st: *const Step, token_ids: Array) !Array {
    const cfg = m.c;
    const s = m.s;
    // MLX_SERVE_GLM_DSA_TAILS: a window's DSA layers' indexer tail rows read first, in a command buffer of their own
    // (each layer's cache append then reuses its buffer), dropped at the forward's end.
    var dsa_layers: [dsa_append.trunk_layers]u32 = undefined;
    try dsa_append.begin(s, st.slots, st.batched, mlx.getShape(token_ids), dsa_append.trunkDsaLayers(m, &dsa_layers));
    defer dsa_append.end();

    var emb = try m.embedding(token_ids);
    {
        errdefer _ = mlx.mlx_array_free(emb);
        // Prompt rows only; the splice frees the embedding it replaces.
        if (st.media.ctx != null and mlx.getShape(emb)[1] > 1 and cfg.image_token_id != 0)
            emb = try spliceVisionRows(emb, token_ids, st.media, cfg.image_token_id, 0, 0, st.media_offset, s);
    }
    defer _ = mlx.mlx_array_free(emb);
    const x_shape = mlx.getShape(emb);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1 and !st.batched;
    var stream = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream);
    {
        var e4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e4);
        try mlx.check(mlx.mlx_expand_dims(&e4, emb, 2, m.s));
        try mlx.check(mlx.mlx_broadcast_to(&stream, e4, &[_]c_int{ batch, seq_len, glm5.HC, x_shape[2] }, 4, m.s));
    }

    const n_layers = cfg.num_hidden_layers;
    var deferred: ?glm5.Deferred = null;
    defer if (deferred) |*df| df.deinit();
    for (m.layers[0..n_layers], 0..) |*lw, li| {
        for ([_]bool{ true, false }) |attn| {
            try sublayer(m, st, &stream, &deferred, lw, attn, li, batch, seq_len, is_prefill);
        }
        // DFlash taps the MEAN of the layer's hyper-connection streams (SGLang's `hc_contract`).
        for (st.capture_ids, st.capture_out) |cid, *slot| if (cid == li) {
            if (deferred) |*df| {
                _ = mlx.mlx_array_free(stream);
                stream = try df.materialize(m.s);
                df.deinit();
                deferred = null;
            }
            var mean = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mean);
            try mlx.check(mlx.mlx_mean_axis(&mean, stream, 2, false, m.s));
            _ = mlx.mlx_array_set(slot, mean);
        };
        if (is_prefill and st.eval_every > 0 and ((li + 1) % st.eval_every == 0 or li + 1 == n_layers)) {
            try st.slots.evalWithAsync(stream, @intCast(li + 1), asyncEval());
        }
    }
    if (deferred) |*df| {
        _ = mlx.mlx_array_free(stream);
        stream = try df.materialize(m.s);
    }
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, stream, 2, false, m.s));
    return m.rmsNorm(mean, m.final_norm);
}

/// A split prompt forward's spans (`Forward.split_rows`): span A, `a` rows from position `pos_a`, then span B, `b` rows
/// from `pos_b` = `pos_a + a`.
pub const Spans = struct { a: u32, b: u32, pos_a: u32, pos_b: u32 };

/// Why a split prompt forward is refused, before any cache is touched (the host then runs two forwards).
pub const SplitRefusal = error{ GlmSplitMode, GlmSplitSlots, GlmSplitCapture, GlmSplitMedia, GlmSplitStates, GlmSplitRows, GlmSplitShortSpan, GlmSplitRoutedPath };

/// The spans of a split forward of `f` whose span A is its first `rows_a` rows, or the refusal by name (MLX-free).
/// Served: one slot's prompt rows (host tokens, or the engine's ids) without captures or media rows, a cache for span
/// A's states (`has_states`), and two spans of more than `glm5.HC_PRE_MAX_ROWS` rows each: at that many rows or fewer a
/// sublayer's hyper-connection takes `hcPre`'s one-dispatch arm and defers its expand (`sublayer`), at more it collapses
/// and expands in place, so both spans take the arm their own forwards take. `routed_source`: every MoE layer runs both
/// spans' routed experts in one call of the streamed bank's source (`splitServes`).
pub fn splitSpans(f: *const abi.Forward, rows_a: u32, has_states: bool, routed_source: bool) SplitRefusal!Spans {
    if (f.mode != .prefill) return error.GlmSplitMode;
    if (f.caches.len != 1 or f.positions.len != 1) return error.GlmSplitSlots;
    if (f.capture_layers.len > 0) return error.GlmSplitCapture;
    if (f.media.ctx != null) return error.GlmSplitMedia;
    if (!has_states) return error.GlmSplitStates;
    if (rows_a >= f.rows or (f.ids.ctx == null and f.tokens.len != f.rows)) return error.GlmSplitRows;
    const b = f.rows - rows_a;
    const pos_a = f.positions.slice()[0];
    const pos_b = std.math.add(u32, pos_a, rows_a) catch return error.GlmSplitRows;
    const short: u32 = glm5.HC_PRE_MAX_ROWS;
    if (rows_a <= short or b <= short) return error.GlmSplitShortSpan;
    if (!routed_source) return error.GlmSplitRoutedPath;
    return .{ .a = rows_a, .b = b, .pos_a = pos_a, .pos_b = pos_b };
}

/// Whether every MoE layer of `m` runs a split forward's spans of `rows_a` and `rows_b` rows in one call of the
/// streamed bank's source (`glm_moe.splitServes`), checked before the forward runs any layer.
pub fn splitServes(m: *const Model, rows_a: u32, rows_b: u32) bool {
    const ra = std.math.cast(c_int, rows_a) orelse return false;
    const rb = std.math.cast(c_int, rows_b) orelse return false;
    var any = false;
    for (m.layers[0..m.c.num_hidden_layers]) |*lw| switch (lw.mlp) {
        .moe => |*mw| {
            if (!glm_moe.splitServes(mw, ra, rb)) return false;
            any = true;
        },
        .dense => {},
    };
    return any;
}

/// A split prompt forward's two spans on one slot's caches (`Forward.split_rows`): span A's rows (`st_a`, `ids_a`), then
/// span B's (`st_b`, `ids_b`, positioned from A's end); each span's final-normed hidden `[1, S, H]` is what `hidden`
/// returns for that span's own forward. Layer by layer: span A's attention sublayer, span A's recurrent states of the
/// layer written to `states` (a KDA layer's `slots.prompt_state_slots`: the states the host snapshots at A's end) before
/// span B's attention sublayer reads and replaces them in the slot's cache, then both spans' MLP sublayers, a MoE
/// layer's routed experts in one streamed call over both spans' rows (`mlpSublayerSplit`), then each span's evaluation
/// at `hidden`'s cadence. Each layer's KV and states are the layer's own (`CacheOps`, per layer), so A then B at every
/// layer leaves the caches as A's forward then B's. Every op of a span reads that span's rows and those caches only, but
/// the routed call, whose rows the bank's source runs each by itself. The caller refuses what this does not serve
/// (`splitSpans`, `splitServes`).
pub fn hiddenSplit(m: *Model, st_a: *const Step, st_b: *const Step, ids_a: Array, ids_b: Array, states: abi.Handle) ![2]Array {
    const cfg = m.c;
    const steps = [2]*const Step{ st_a, st_b };
    const ids = [2]Array{ ids_a, ids_b };
    for (steps) |st| if (st.batched or st.capture or st.capture_ids.len > 0 or st.media.ctx != null) return error.GlmSplitStep;
    var streams: [2]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (streams) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var batch: [2]c_int = undefined;
    var seq_len: [2]c_int = undefined;
    // Each span's [B, S, HC, D] stream, as `hidden` builds it.
    for (ids, &streams, &batch, &seq_len) |token_ids, *stream, *bt, *sl| {
        const emb = try m.embedding(token_ids);
        defer _ = mlx.mlx_array_free(emb);
        const x_shape = mlx.getShape(emb);
        bt.* = x_shape[0];
        sl.* = x_shape[1];
        if (sl.* < 2) return error.GlmSplitShortSpan;
        var e4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e4);
        try mlx.check(mlx.mlx_expand_dims(&e4, emb, 2, m.s));
        try mlx.check(mlx.mlx_broadcast_to(stream, e4, &[_]c_int{ bt.*, sl.*, glm5.HC, x_shape[2] }, 4, m.s));
    }
    const n_layers = cfg.num_hidden_layers;
    var deferred: [2]?glm5.Deferred = .{ null, null };
    defer for (&deferred) |*d| if (d.*) |*df| df.deinit();
    for (m.layers[0..n_layers], 0..) |*lw, li| {
        try sublayer(m, st_a, &streams[0], &deferred[0], lw, true, li, batch[0], seq_len[0], true);
        if (lw.mixer == .kda) for (slots_mod.prompt_state_slots) |which| try st_a.slots.copyState(0, @intCast(li), which, states);
        try sublayer(m, st_b, &streams[1], &deferred[1], lw, true, li, batch[1], seq_len[1], true);
        try mlpSublayerSplit(m, &streams, &deferred, lw);
        for (steps, streams) |st, stream| {
            if (st.eval_every > 0 and ((li + 1) % st.eval_every == 0 or li + 1 == n_layers)) {
                try st.slots.evalWithAsync(stream, @intCast(li + 1), asyncEval());
            }
        }
    }
    var out: [2]Array = undefined;
    var made: usize = 0;
    errdefer for (out[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&streams, &deferred, &out) |*stream, *d, *o| {
        if (d.*) |*df| {
            _ = mlx.mlx_array_free(stream.*);
            stream.* = try df.materialize(m.s);
        }
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_mean_axis(&mean, stream.*, 2, false, m.s));
        o.* = try m.rmsNorm(mean, m.final_norm);
        made += 1;
    }
    return out;
}

/// Both spans' MLP sublayers of a split forward, each inside its hyper-connection as `sublayer` runs it (`attn` false),
/// but a MoE layer's routed experts, which run in one call over both spans' rows (`glm_moe.forwardSplit`); a dense
/// layer's MLP runs per span. `sublayer`'s one-dispatch arm (`hcPre`, its expand deferred: spans of
/// `glm5.HC_PRE_MAX_ROWS` rows or fewer) is not taken here: a split refuses such spans (`splitSpans`).
fn mlpSublayerSplit(m: *Model, streams: *[2]Array, deferred: *[2]?glm5.Deferred, lw: *const Layer) !void {
    const cfg = m.c;
    const w = lw.ffn_hc.?;
    const norm_w = lw.post_norm;
    var cs: [2]glm5.Collapsed = undefined;
    var made: usize = 0;
    defer for (cs[0..made]) |*c| c.deinit();
    var normed: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (normed) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (streams, deferred, &cs, &normed) |*stream, *d, *c, *nrm| {
        if (glm5.hcRowsServes(m.s, mlx.getShape(stream.*), mlx.mlx_array_dtype(stream.*), w.fn_rows, norm_w)) return error.GlmSplitShortSpan;
        if (d.*) |*df| {
            _ = mlx.mlx_array_free(stream.*);
            stream.* = try df.materialize(m.s);
            df.deinit();
            d.* = null;
        }
        c.* = try glm5.collapse(m.s, stream.*, w.fn_rows, w.scale, w.base, cfg.dsv4_hc_sinkhorn_iters, cfg.dsv4_hc_eps, cfg.rms_norm_eps);
        made += 1;
        nrm.* = try m.rmsNorm(c.x, norm_w);
    }
    var outs: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (outs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    switch (lw.mlp) {
        .moe => |*mw| outs = try glm_moe.forwardSplit(mw, m.s, normed[0], normed[1]),
        .dense => |*dw| for (normed, &outs) |x, *o| {
            o.* = try m.denseMlp(dw, x);
        },
    }
    for (streams, &cs, outs) |*stream, *c, o| {
        const res = try glm5.expand(m.s, o, stream.*, c);
        _ = mlx.mlx_array_free(stream.*);
        stream.* = res;
    }
}

/// One sublayer (attention when `attn`, else the MLP) inside its hyper-connection. For one token the expand stays
/// in `deferred` (then `stream` is stale until the forward materializes it).
fn sublayer(m: *Model, st: *const Step, stream: *Array, deferred: *?glm5.Deferred, lw: *const Layer, attn: bool, li: usize, batch: c_int, seq_len: c_int, is_prefill: bool) !void {
    const cfg = m.c;
    const w = if (attn) lw.attn_hc.? else lw.ffn_hc.?;
    const norm_w = if (attn) lw.input_norm else lw.post_norm;
    const one = glm5.hcRowsServes(m.s, mlx.getShape(stream.*), mlx.mlx_array_dtype(stream.*), w.fn_rows, norm_w);
    if (!one) if (deferred.*) |*df| {
        _ = mlx.mlx_array_free(stream.*);
        stream.* = try df.materialize(m.s);
        df.deinit();
        deferred.* = null;
    };
    if (one) {
        var pre = try glm5.hcPre(m.s, if (deferred.* == null) stream.* else null, if (deferred.*) |*df| df else null, w.fn_rows, w.scale, w.base, norm_w, cfg.dsv4_hc_sinkhorn_iters, cfg.dsv4_hc_eps, cfg.rms_norm_eps, cfg.rms_norm_eps);
        defer pre.deinit();
        const out = try branch(m, st, pre.normed, lw, attn, li, batch, seq_len, is_prefill);
        defer _ = mlx.mlx_array_free(out);
        if (deferred.*) |*df| {
            _ = mlx.mlx_array_set(stream, pre.h);
            df.deinit();
        }
        deferred.* = pre.defer_(out, stream.*);
        return;
    }
    var c = try glm5.collapse(m.s, stream.*, w.fn_rows, w.scale, w.base, cfg.dsv4_hc_sinkhorn_iters, cfg.dsv4_hc_eps, cfg.rms_norm_eps);
    defer c.deinit();
    const normed = try m.rmsNorm(c.x, norm_w);
    defer _ = mlx.mlx_array_free(normed);
    const out = try branch(m, st, normed, lw, attn, li, batch, seq_len, is_prefill);
    defer _ = mlx.mlx_array_free(out);
    const res = try glm5.expand(m.s, out, stream.*, &c);
    _ = mlx.mlx_array_free(stream.*);
    stream.* = res;
}

fn branch(m: *Model, st: *const Step, normed: Array, lw: *const Layer, attn: bool, li: usize, batch: c_int, seq_len: c_int, is_prefill: bool) !Array {
    if (st.batched) return branchSlots(m, st.slots, normed, lw, attn, li);
    return if (attn) switch (lw.mixer) {
        .kda => |*la| try slots_mod.kda(m, st.slots, 0, normed, la, @intCast(li), batch, seq_len, is_prefill, st.capture),
        .dsa => |*d| try dsaAttn(m, st.slots, 0, normed, d, @intCast(li), batch, seq_len),
        .mla => error.GlmLayerIsMla,
    } else switch (lw.mlp) {
        .moe => |*mw| try glm_moe.forward(mw, m.s, normed),
        .dense => |*dw| try m.denseMlp(dw, normed),
    };
}

/// One batched tick's branch: `normed` `[1, N, H]` holds N slots' next-token rows. The MLP and the projections read
/// their weights once for all N rows; a slot's own state (its KDA recurrence, its DSA cache) is advanced by its call.
fn branchSlots(m: *Model, sl: Slots, normed: Array, lw: *const Layer, attn: bool, li: usize) !Array {
    if (!attn) return switch (lw.mlp) {
        .moe => |*mw| try glm_moe.forward(mw, m.s, normed),
        .dense => |*dw| try m.denseMlp(dw, normed),
    };
    switch (lw.mixer) {
        .kda => |*la| return kdaSlots(m, sl, normed, la, @intCast(li)),
        .mla => return error.GlmLayerIsMla,
        .dsa => |*d| {
            var rows: [MOE_DECODE_BATCH_MAX]Array = undefined;
            var built: usize = 0;
            defer for (rows[0..built]) |r| {
                _ = mlx.mlx_array_free(r);
            };
            for (0..sl.len()) |i| {
                const row = try axisView(m.s, normed, 1, i);
                defer _ = mlx.mlx_array_free(row);
                rows[i] = try dsaAttn(m, sl, i, row, d, @intCast(li), 1, 1);
                built += 1;
            }
            return concatAxis1(m.s, rows[0..sl.len()]);
        },
    }
}

/// KDA over N slots' rows: one joined input projection and one output projection, the fused one-token step per
/// slot against that slot's own conv and recurrent state. A layer whose projection is not joined runs per slot.
fn kdaSlots(m: *Model, sl: Slots, x: Array, la: *const trunk.Kda, layer: u32) !Array {
    const cfg = m.c;
    const heads: c_int = @intCast(cfg.linear_num_value_heads);
    var rows: [MOE_DECODE_BATCH_MAX]Array = undefined;
    var built: usize = 0;
    defer for (rows[0..built]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    const joined = la.in.built() and la.a_log_h.ctx != null and la.a2.w.ctx != null and la.z2.w.ctx != null;
    // A GPU that cannot launch the fused step declines at the first slot, and a slot with no state yet steps from
    // zeros: the per-slot path below serves both.
    if (joined) joined_path: {
        for (0..sl.len()) |i| {
            if (sl.state(i, layer, .ssm).ctx == null or sl.state(i, layer, .conv).ctx == null) break :joined_path;
        }
        const all = try m.mm(&la.in.lin, x);
        defer _ = mlx.mlx_array_free(all);
        const w = la.in.widths;
        const fb = kda.low(m, &la.a2);
        const gb = kda.low(m, &la.z2);
        for (0..sl.len()) |i| {
            const proj = try axisView(m.s, all, 1, i);
            defer _ = mlx.mlx_array_free(proj);
            const r = (try kda_recurrence.decodeStep(m.s, proj, w[0], w[0] + w[1], w[0] + w[1] + w[2], heads, sl.state(i, layer, .conv), la.conv_w, la.a_log_h, la.dt_f32, sl.state(i, layer, .ssm), la.norm_w, fb, gb, cfg.kda_gate_lower_bound, cfg.rms_norm_eps, false)) orelse if (i == 0) break :joined_path else return error.GlmBatchedKdaDeclined;
            defer _ = mlx.mlx_array_free(r.conv_state);
            defer _ = mlx.mlx_array_free(r.state);
            try sl.setState(i, layer, .conv, r.conv_state);
            try sl.setState(i, layer, .ssm, r.state);
            rows[i] = r.y;
            built += 1;
        }
        const y = try concatAxis1(m.s, rows[0..sl.len()]);
        defer _ = mlx.mlx_array_free(y);
        return m.mm(&la.out, y);
    }
    for (0..sl.len()) |i| {
        const row = try axisView(m.s, x, 1, i);
        defer _ = mlx.mlx_array_free(row);
        rows[i] = try slots_mod.kda(m, sl, i, row, la, layer, 1, 1, false, false);
        built += 1;
    }
    return concatAxis1(m.s, rows[0..sl.len()]);
}

/// DeepSeek sparse attention on slot `i`. The cache holds, per token, the latent (`k`) and the indexer's key and
/// pool gate (`v`). Queries attend in the latent (absorbed MLA) over the tokens the indexer picks; when every query
/// picks every earlier token the attention is plain causal and prefill runs it on lifted keys.
pub fn dsaAttn(m: *const Model, sl: Slots, slot: usize, x: Array, d: *const trunk.Dsa, layer: u32, batch: c_int, seq_len: c_int) !Array {
    const cfg = m.c;
    const s = m.s;
    const offset: usize = sl.positions[slot];
    const heads: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.dsa_head_dim);
    const r: c_int = @intCast(cfg.dsa_kv_lora_rank);
    const ihd: c_int = @intCast(cfg.dsa_index_head_dim);
    const ih: c_int = @intCast(cfg.dsa_index_heads);

    const q_a = try m.mm(&d.q_a, x);
    defer _ = mlx.mlx_array_free(q_a);
    const kv_a = try m.mm(&d.kv_a, x);
    defer _ = mlx.mlx_array_free(kv_a);
    const q_resid = try m.rmsNorm(q_a, d.q_a_norm);
    defer _ = mlx.mlx_array_free(q_resid);
    const q_flat = try m.mm(&d.q_b, q_resid);
    defer _ = mlx.mlx_array_free(q_flat);
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    try mlx.check(mlx.mlx_reshape(&q4, q_flat, &[_]c_int{ batch, seq_len, heads, hd }, 4, s));
    var qh = mlx.mlx_array_new(); // [B, H, S, hd]
    defer _ = mlx.mlx_array_free(qh);
    try mlx.check(mlx.mlx_transpose_axes(&qh, q4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));

    const lat = try m.rmsNorm(kv_a, d.kv_a_norm);
    defer _ = mlx.mlx_array_free(lat);
    var lat4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lat4);
    try mlx.check(mlx.mlx_reshape(&lat4, lat, &[_]c_int{ batch, 1, seq_len, r }, 4, s));

    // Indexer key (LayerNorm) and pool gate (f32 projection, stored in the key's dtype).
    const ik = try m.mm(&d.idx_wk, x);
    defer _ = mlx.mlx_array_free(ik);
    var ikn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ikn);
    try mlx.check(mlx.mlx_fast_layer_norm(&ikn, ik, d.idx_k_norm_w, d.idx_k_norm_b, 1e-6, s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_matmul(&gate, xf, d.idx_gate_t, s));
    var gate_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_c);
    try mlx.check(mlx.mlx_astype(&gate_c, gate, mlx.mlx_array_dtype(ikn), s));
    var idx_new = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_new);
    {
        const parts = [_]Array{ ikn, gate_c };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, -1, s));
        try mlx.check(mlx.mlx_reshape(&idx_new, cat, &[_]c_int{ batch, 1, seq_len, 2 * ihd }, 4, s));
    }

    // A row that completes a pool stores the pool's key (`glm5.withPooledKeys`).
    const idx_rows = blk: {
        var rows3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows3);
        try mlx.check(mlx.mlx_reshape(&rows3, idx_new, &[_]c_int{ batch, seq_len, 2 * ihd }, 3, s));
        const kpool: c_int = @intCast(cfg.dsa_index_kpool);
        const n_prev: c_int = @min(@as(c_int, @intCast(offset)), kpool - 1);
        var prev = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prev);
        // MLX_SERVE_GLM_DSA_TAILS: the rows the forward read at its start (null: the cache view's).
        const early: ?Array = if (n_prev > 0) dsa_append.take(sl, slot, layer, n_prev) else null;
        if (early) |tail_rows| {
            _ = mlx.mlx_array_set(&prev, tail_rows);
        } else if (n_prev > 0) {
            var cached = try sl.view(slot, layer, s);
            defer cached.deinit();
            const vsh = mlx.getShape(cached.v);
            var tail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail);
            try mlx.check(mlx.mlx_slice(&tail, cached.v, &[_]c_int{ 0, 0, vsh[2] - n_prev, 0 }, 4, &[_]c_int{ vsh[0], vsh[1], vsh[2], vsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_reshape(&prev, tail, &[_]c_int{ batch, n_prev, 2 * ihd }, 3, s));
        } else _ = mlx.mlx_array_set(&prev, rows3);
        const pooled = try glm5.withPooledKeys(s, prev, rows3, d.idx_ape, @intCast(offset), .{ .topk = @intCast(cfg.dsa_index_topk), .kpool = kpool, .heads = ih, .head_dim = ihd });
        defer _ = mlx.mlx_array_free(pooled);
        var r4 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&r4, pooled, &[_]c_int{ batch, 1, seq_len, 2 * ihd }, 4, s));
        break :blk r4;
    };
    defer _ = mlx.mlx_array_free(idx_rows);
    var view = try sl.append(slot, layer, lat4, idx_rows, s);
    defer view.deinit();
    const t: c_int = @intCast(offset + @as(usize, @intCast(seq_len)));
    const ix = glm5.Indexer{ .topk = @intCast(cfg.dsa_index_topk), .kpool = @intCast(cfg.dsa_index_kpool), .heads = ih, .head_dim = ihd };
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const none = Array{ .ctx = null };

    // A row at a position below `dense_end` reads every earlier token (all its pools fit the budget): plain causal
    // attention. Prefill runs those rows on lifted keys and the rest through the indexer; decode and verify rows
    // stay in the latent.
    const dense_end: c_int = (@divTrunc(ix.topk, ix.kpool) + 1) * ix.kpool - 1;
    const dense_rows: c_int = std.math.clamp(dense_end - @as(c_int, @intCast(offset)), 0, seq_len);
    var parts: [2]Array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (parts) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    if (seq_len > glm5.HC_PRE_MAX_ROWS and dense_rows > 0) {
        var kv_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kv_rows);
        try mlx.check(mlx.mlx_slice(&kv_rows, view.k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, 1, @as(c_int, @intCast(offset)) + dense_rows, r }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const keys = try m.headMm(kv_rows, &d.embed_q, false);
        defer _ = mlx.mlx_array_free(keys);
        const values = try m.headMm(kv_rows, &d.unembed, true);
        defer _ = mlx.mlx_array_free(values);
        var q_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rows);
        try mlx.check(mlx.mlx_slice(&q_rows, qh, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ batch, heads, dense_rows, hd }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        parts[0] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&parts[0], q_rows, keys, values, scale, "causal", none, none, false, s));
    }
    const sparse_from: c_int = if (seq_len > glm5.HC_PRE_MAX_ROWS) dense_rows else 0;
    if (sparse_from < seq_len) {
        const rows = seq_len - sparse_from;
        const qh_s = try rowsFrom(qh, 2, sparse_from, s);
        defer _ = mlx.mlx_array_free(qh_s);
        // One query: the per-head kernel applies the attention scale as it rounds.
        var q_scale: ?f32 = scale;
        const q_abs = blk: {
            if (rows == 1 and d.embed_q.s.ctx != null) {
                if (d.embed_q.mode == .affine) if (try glm5.headQmv(s, qh_s, d.embed_q.w, d.embed_q.s, d.embed_q.b, d.embed_q.bits, d.embed_q.group_size, scale)) |y| {
                    q_scale = null;
                    break :blk y;
                };
            }
            break :blk try m.headMm(qh_s, &d.embed_q, true);
        };
        defer _ = mlx.mlx_array_free(q_abs);
        var o_lat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o_lat);
        if (glm5.selectsAll(t, ix) and rows <= glm5.HC_PRE_MAX_ROWS and batch == 1) {
            var keys = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(keys);
            try mlx.check(mlx.mlx_reshape(&keys, view.k, &[_]c_int{ 1, t, r }, 3, s));
            _ = mlx.mlx_array_free(o_lat);
            if (rows == 1) {
                o_lat = try glm5.latentAttendFolded(s, q_abs, keys, null, q_scale);
            } else {
                // A verify window folds its rows into the head axis under a per-row causal mask: MLX's sdpa
                // declines the 64:1 GQA factor and reads the keys per head.
                var qf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(qf);
                try mlx.check(mlx.mlx_reshape(&qf, q_abs, &[_]c_int{ 1, heads * rows, 1, r }, 4, s));
                const valid = try glm5.latentCausalMask(s, heads, rows, t);
                defer _ = mlx.mlx_array_free(valid);
                const of = try glm5.latentAttendFolded(s, qf, keys, valid, q_scale);
                defer _ = mlx.mlx_array_free(of);
                o_lat = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&o_lat, of, &[_]c_int{ 1, heads, rows, r }, 4, s));
            }
        } else if (glm5.selectsAll(t, ix)) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_lat, q_abs, view.k, view.k, q_scale orelse 1.0, if (rows > 1) "causal" else "", none, none, false, s));
        } else {
            const qr_s = try rowsFrom(q_resid, 1, sparse_from, s);
            defer _ = mlx.mlx_array_free(qr_s);
            const q_idx_flat = try m.mm(&d.idx_wq_b, qr_s);
            defer _ = mlx.mlx_array_free(q_idx_flat);
            var q_idx = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_idx);
            try mlx.check(mlx.mlx_reshape(&q_idx, q_idx_flat, &[_]c_int{ batch, rows, ih, ihd }, 4, s));
            const x_s = try rowsFrom(x, 1, sparse_from, s);
            defer _ = mlx.mlx_array_free(x_s);
            const wp = try m.mm(&d.idx_wproj, x_s);
            defer _ = mlx.mlx_array_free(wp);
            var wpf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wpf);
            try mlx.check(mlx.mlx_astype(&wpf, wp, .float32, s));
            const hscale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ih))));
            defer _ = mlx.mlx_array_free(hscale);
            var wts = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wts);
            try mlx.check(mlx.mlx_multiply(&wts, wpf, hscale, s));
            var idx_cache = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(idx_cache);
            try mlx.check(mlx.mlx_reshape(&idx_cache, view.v, &[_]c_int{ batch, t, 2 * ihd }, 3, s));
            const sel = try glm5.selectTokens(s, q_idx, wts, idx_cache, ix);
            defer _ = mlx.mlx_array_free(sel);
            const nax = if (rows > 8) try glm5.sparseLatentNax(s, q_abs, view.k, sel, q_scale orelse 1.0) else null;
            const lat_out = nax orelse try glm5.sparseLatentAttention(s, q_abs, view.k, sel, q_scale, 64);
            _ = mlx.mlx_array_free(o_lat);
            o_lat = lat_out;
        }
        parts[1] = try m.headMm(o_lat, &d.unembed, true);
    }
    var o = mlx.mlx_array_new(); // [B, H, S, hd]; [B, S, H, hd] where `rows_first`
    defer _ = mlx.mlx_array_free(o);
    // MLX_SERVE_GLM_DSA_PROMPT_INDEXER: both parts in the output's row order in one copy (no transpose below).
    var rows_first = false;
    if (parts[0].ctx != null and parts[1].ctx != null) {
        rows_first = try dsa_prompt.assemble(&o, s, parts[0], parts[1]);
        if (!rows_first) {
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
        }
    } else {
        try mlx.check(mlx.mlx_array_set(&o, if (parts[0].ctx != null) parts[0] else parts[1]));
    }
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    if (rows_first) {
        try mlx.check(mlx.mlx_array_set(&ot, o));
    } else try mlx.check(mlx.mlx_transpose_axes(&ot, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, ot, &[_]c_int{ batch, seq_len, heads * hd }, 3, s));
    const of = try glm5.withDependency(flat, view.v);
    defer _ = mlx.mlx_array_free(of);
    return m.mm(&d.o, of);
}

/// `a` from index `from` on along `axis`, a new reference.
fn rowsFrom(a: Array, axis: usize, from: c_int, s: mlx.mlx_stream) !Array {
    const sh = mlx.getShape(a);
    var lo: [4]c_int = @splat(0);
    var hi: [4]c_int = undefined;
    for (sh, 0..) |v, i| hi[i] = v;
    lo[axis] = from;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &lo, sh.len, &hi, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, s));
    return out;
}

// ── a prompt forward's evals ──

var async_eval_said = false;

/// MLX_SERVE_GLM_PROMPT_ASYNC_EVAL: a prompt forward's cadence eval hands its arrays to the GPU without the host's wait
/// (the same arrays, kernels and order; a tail's buffers can outlive the next sublayer's start).
fn asyncEval() bool {
    const on = if (env.get(.MLX_SERVE_GLM_PROMPT_ASYNC_EVAL)) |v| v[0] != '0' else true;
    if (on and !async_eval_said) {
        async_eval_said = true;
        log.info("[glm5_next] prompt async eval engaged: the cadence's evals without the host's wait\n", .{});
    }
    return on;
}

// ── tests ──

/// A prompt forward of `toks` from position `pos[0]` on the one cache `cache[0]`.
fn promptForward(toks: []const u32, pos: *const [1]u32, cache: *const [1]abi.Handle) abi.Forward {
    return .{ .mode = .prefill, .rows = @intCast(toks.len), .tokens = abi.Span(u32).of(toks), .positions = abi.Span(u32).of(pos), .caches = abi.Span(abi.Handle).of(cache) };
}

test "glm5_next split: a split prompt forward is refused by name unless each span takes its own forward's arms; else its spans, span B from span A's end" {
    const t = std.testing;
    var toks: [71]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(i);
    const pos = [1]u32{100};
    const cache = [1]abi.Handle{@ptrFromInt(0x1000)};
    const f = promptForward(&toks, &pos, &cache);
    // 40 + 31 rows (a 16K prompt's held-back span), span B at 140; then both spans one row past hcPre's
    try t.expectEqual(Spans{ .a = 40, .b = 31, .pos_a = 100, .pos_b = 140 }, try splitSpans(&f, 40, true, true));
    const f34 = promptForward(toks[0..34], &pos, &cache);
    try t.expectEqual(Spans{ .a = 17, .b = 17, .pos_a = 100, .pos_b = 117 }, try splitSpans(&f34, 17, true, true));
    // the engine's device ids in place of host tokens
    var ids = f;
    ids.tokens = .{};
    ids.ids = .{ .ctx = @ptrFromInt(0x2000) };
    try t.expectEqual(Spans{ .a = 40, .b = 31, .pos_a = 100, .pos_b = 140 }, try splitSpans(&ids, 40, true, true));
    // not a prompt
    for ([_]abi.Mode{ .decode, .verify, .batched_rows }) |mode| {
        var x = f;
        x.mode = mode;
        try t.expectError(error.GlmSplitMode, splitSpans(&x, 40, true, true));
    }
    // two slots
    const caches2 = [2]abi.Handle{ @ptrFromInt(0x1000), @ptrFromInt(0x3000) };
    var two = f;
    two.caches = abi.Span(abi.Handle).of(&caches2);
    try t.expectError(error.GlmSplitSlots, splitSpans(&two, 40, true, true));
    // DFlash captures, media rows, no cache for span A's states
    const layers = [_]u32{10};
    var cap = f;
    cap.capture_layers = abi.Span(u32).of(&layers);
    try t.expectError(error.GlmSplitCapture, splitSpans(&cap, 40, true, true));
    var media = f;
    media.media = .{ .ctx = @ptrFromInt(0x4000) };
    try t.expectError(error.GlmSplitMedia, splitSpans(&media, 40, true, true));
    try t.expectError(error.GlmSplitStates, splitSpans(&f, 40, false, true));
    // no span B, host tokens other than the rows, span B past the last position
    try t.expectError(error.GlmSplitRows, splitSpans(&f, 71, true, true));
    var less = f;
    less.tokens = abi.Span(u32).of(toks[0..70]);
    try t.expectError(error.GlmSplitRows, splitSpans(&less, 40, true, true));
    const pos_hi = [1]u32{std.math.maxInt(u32) - 10};
    const hi = promptForward(&toks, &pos_hi, &cache);
    try t.expectError(error.GlmSplitRows, splitSpans(&hi, 40, true, true));
    // a span of 16 rows or fewer: its sublayers would take hcPre's deferred arm
    try t.expectError(error.GlmSplitShortSpan, splitSpans(&f, 16, true, true));
    const f56 = promptForward(toks[0..56], &pos, &cache);
    try t.expectError(error.GlmSplitShortSpan, splitSpans(&f56, 40, true, true));
    // a MoE layer off the streamed source arm
    try t.expectError(error.GlmSplitRoutedPath, splitSpans(&f, 40, true, false));
}
