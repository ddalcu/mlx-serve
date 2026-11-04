//! GLM-5.3-Flash's forward behind the model table: the [B, S, hc, D] hyper-connection stream through KDA and DSA
//! sublayers and the MoE, every slot's KV and recurrent state through `CacheOps`. Prefill, decode, verify (rows
//! capture their KDA states) and batched rows (each slot's state advanced by its own call, the MLP shared).
//! copied_from: plugins/glm5_next/forward.zig (`forwardGlm5With`, `glmSublayer`, `glmBranch`, `glmBranchSlots`,
//! `glmKdaSlots`, `glmDsaAttn`, `forwardGlmBatchedDecode`), the ForwardCtx reads replaced by `Slots`.

const std = @import("std");
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
            try st.slots.evalWith(stream, @intCast(li + 1));
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
        if (n_prev > 0) {
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
    var o = mlx.mlx_array_new(); // [B, H, S, hd]
    defer _ = mlx.mlx_array_free(o);
    if (parts[0].ctx != null and parts[1].ctx != null) {
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&o, if (parts[0].ctx != null) parts[0] else parts[1]));
    }
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    try mlx.check(mlx.mlx_transpose_axes(&ot, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
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
