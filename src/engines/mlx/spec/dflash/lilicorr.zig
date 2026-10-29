//! LiLiCorr correlation head for the Bolt drafter.
//!
//! The trunk is an ordinary DFlash block. This module turns that block's
//! logits and final-norm hiddens into a greedy candidate chain, matching
//! NVIDIA Model Optimizer's `LiLiCorrHead` and the greedy serving walk:
//! top-k ids per slot, a 5-feature LayerNorm MLP, attention layers over
//! the slot×rank lattice, then L2-normalized factor heads. Slot 0 is scored
//! against the anchor context feature; later slots are scored against the
//! previous slot's chosen candidate.

const std = @import("std");
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../../sdk/util/log.zig");
const linear = @import("../../../../plugins/glm5_next/vision.zig").linear;
const scalarOf = @import("../../transformer.zig").scalarOf;
const model_mod = @import("../../../../sdk/config/model.zig");

const Weights = model_mod.Weights;

pub const LoadCfg = struct {
    top_k: u32,
    hidden: u32,
    trunk: u32,
    heads: u32,
    slots: u32,
    layers: u32,
    factor: u32,
    vector_eps: f32,
    logit_scale: f32,
    rms_eps: f32,
};

const Layer = struct {
    attn_norm: mlx.mlx_array = .{ .ctx = null },
    mlp_norm: mlx.mlx_array = .{ .ctx = null },
    in_w: mlx.mlx_array = .{ .ctx = null },
    in_b: mlx.mlx_array = .{ .ctx = null },
    out_w: mlx.mlx_array = .{ .ctx = null },
    out_b: mlx.mlx_array = .{ .ctx = null },
    up_w: mlx.mlx_array = .{ .ctx = null },
    up_b: mlx.mlx_array = .{ .ctx = null },
    down_w: mlx.mlx_array = .{ .ctx = null },
    down_b: mlx.mlx_array = .{ .ctx = null },

    fn deinit(self: *Layer) void {
        _ = mlx.mlx_array_free(self.attn_norm);
        _ = mlx.mlx_array_free(self.mlp_norm);
        _ = mlx.mlx_array_free(self.in_w);
        _ = mlx.mlx_array_free(self.in_b);
        _ = mlx.mlx_array_free(self.out_w);
        _ = mlx.mlx_array_free(self.out_b);
        _ = mlx.mlx_array_free(self.up_w);
        _ = mlx.mlx_array_free(self.up_b);
        _ = mlx.mlx_array_free(self.down_w);
        _ = mlx.mlx_array_free(self.down_b);
    }

    fn appendEval(self: *const Layer, vec: mlx.mlx_vector_array) void {
        for ([_]mlx.mlx_array{
            self.attn_norm, self.mlp_norm, self.in_w,   self.in_b,   self.out_w, self.out_b,
            self.up_w,      self.up_b,     self.down_w, self.down_b,
        }) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    }
};

pub const Head = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    top_k: u32,
    hidden: u32,
    trunk: u32,
    heads: u32,
    slots: u32,
    factor: u32,
    vector_eps: f32,
    logit_scale: f32,
    rms_eps: f32,

    context_w: mlx.mlx_array = .{ .ctx = null },
    context_b: mlx.mlx_array = .{ .ctx = null },
    token_w: mlx.mlx_array = .{ .ctx = null },
    token_b: mlx.mlx_array = .{ .ctx = null },
    pass_w: mlx.mlx_array = .{ .ctx = null },
    pass_b: mlx.mlx_array = .{ .ctx = null },
    feature_norm_w: mlx.mlx_array = .{ .ctx = null },
    feature_norm_b: mlx.mlx_array = .{ .ctx = null },
    feature_up_w: mlx.mlx_array = .{ .ctx = null },
    feature_up_b: mlx.mlx_array = .{ .ctx = null },
    feature_down_w: mlx.mlx_array = .{ .ctx = null },
    feature_down_b: mlx.mlx_array = .{ .ctx = null },
    slot_emb: mlx.mlx_array = .{ .ctx = null },
    rank_emb: mlx.mlx_array = .{ .ctx = null },
    attn_bias: mlx.mlx_array = .{ .ctx = null },
    layers: []Layer = &.{},
    output_norm: mlx.mlx_array = .{ .ctx = null },
    anchor_norm: mlx.mlx_array = .{ .ctx = null },
    factor_w: mlx.mlx_array = .{ .ctx = null },
    factor_b: mlx.mlx_array = .{ .ctx = null },
    in_w: mlx.mlx_array = .{ .ctx = null },
    in_b: mlx.mlx_array = .{ .ctx = null },
    out_w: mlx.mlx_array = .{ .ctx = null },
    out_b: mlx.mlx_array = .{ .ctx = null },
    anchor_out_w: mlx.mlx_array = .{ .ctx = null },
    anchor_out_b: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *Head) void {
        const mats = [_]mlx.mlx_array{
            self.context_w,    self.context_b,    self.token_w,        self.token_b,
            self.pass_w,       self.pass_b,       self.feature_norm_w, self.feature_norm_b,
            self.feature_up_w, self.feature_up_b, self.feature_down_w, self.feature_down_b,
            self.slot_emb,     self.rank_emb,     self.attn_bias,      self.output_norm,
            self.anchor_norm,  self.factor_w,     self.factor_b,       self.in_w,
            self.in_b,         self.out_w,        self.out_b,          self.anchor_out_w,
            self.anchor_out_b,
        };
        for (mats) |a| _ = mlx.mlx_array_free(a);
        for (self.layers) |*lw| lw.deinit();
        self.allocator.free(self.layers);
    }

    pub fn appendEval(self: *const Head, vec: mlx.mlx_vector_array) void {
        const mats = [_]mlx.mlx_array{
            self.context_w,    self.context_b,    self.token_w,        self.token_b,
            self.pass_w,       self.pass_b,       self.feature_norm_w, self.feature_norm_b,
            self.feature_up_w, self.feature_up_b, self.feature_down_w, self.feature_down_b,
            self.slot_emb,     self.rank_emb,     self.attn_bias,      self.output_norm,
            self.anchor_norm,  self.factor_w,     self.factor_b,       self.in_w,
            self.in_b,         self.out_w,        self.out_b,          self.anchor_out_w,
            self.anchor_out_b,
        };
        for (mats) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
        for (self.layers) |*lw| lw.appendEval(vec);
    }

    /// Greedy draft ids, length `n_out` (at most `slots`). Caller frees.
    /// `anchor` is the encoded context feature of the token before this block,
    /// `[1, 1, trunk]` or `[trunk]`. `pass_hidden` is the drafter's post-final-norm
    /// block `[1, block, trunk]` (row 0 is the noise anchor and is skipped).
    /// `draft_logits` is `[1, slots, vocab]`, the lm_head of those same rows.
    pub fn draft(
        self: *const Head,
        allocator: std.mem.Allocator,
        embed: Embed,
        anchor: mlx.mlx_array,
        pass_hidden: mlx.mlx_array,
        draft_logits: mlx.mlx_array,
        n_out: u32,
    ) ![]u32 {
        const s = self.s;
        const slots: usize = self.slots;
        const k: usize = self.top_k;
        if (n_out == 0 or n_out > slots) return error.InvalidLilicorrDraft;
        var candidates = try candidateLattice(allocator, draft_logits, slots, k, s);
        defer candidates.deinit(allocator);
        const sorted_ids = candidates.ids;
        const features = candidates.features;

        var scores = try self.score(embed, anchor, pass_hidden, sorted_ids, features);
        defer scores.deinit();
        const first = scores.first;
        const pair = scores.pair;
        var first_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(first_f);
        try mlx.check(mlx.mlx_astype(&first_f, first, .float32, s));
        var pair_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pair_f);
        if (slots > 1) {
            try mlx.check(mlx.mlx_astype(&pair_f, pair, .float32, s));
        }
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, first_f);
            if (slots > 1) _ = mlx.mlx_vector_array_append_value(ev, pair_f);
            try mlx.check(mlx.mlx_eval(ev));
        }
        const first_data = mlx.mlx_array_data_float32(first_f) orelse return error.MlxArrayDataNull;
        const pair_data: []const f32 = if (slots > 1) blk: {
            const p = mlx.mlx_array_data_float32(pair_f) orelse return error.MlxArrayDataNull;
            break :blk p[0 .. (slots - 1) * k * k];
        } else &.{};
        const out_ids = try allocator.alloc(u32, n_out);
        errdefer allocator.free(out_ids);
        greedyPick(sorted_ids, first_data[0..k], pair_data, slots, k, n_out, out_ids);
        log.debug("[lilicorr] drafted {d} tokens from a {d}x{d} lattice\n", .{ n_out, slots, k });
        return out_ids;
    }

    const Scores = struct {
        first: mlx.mlx_array,
        pair: mlx.mlx_array,

        fn deinit(self: *Scores) void {
            _ = mlx.mlx_array_free(self.first);
            _ = mlx.mlx_array_free(self.pair);
        }
    };

    fn score(self: *const Head, embed: Embed, anchor: mlx.mlx_array, pass_hidden: mlx.mlx_array, sorted_ids: []const i32, features: []const f32) !Scores {
        const s = self.s;
        const slots: usize = self.slots;
        const k: usize = self.top_k;
        const nodes = slots * k;
        const feat_shape = [_]c_int{ 1, @intCast(nodes), 5 };
        const feat_f = mlx.mlx_array_new_data(features.ptr, &feat_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(feat_f);
        var feat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(feat);
        try mlx.check(mlx.mlx_astype(&feat, feat_f, mlx.mlx_array_dtype(self.feature_up_w), s));
        var feat_n = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(feat_n);
        try mlx.check(mlx.mlx_fast_layer_norm(&feat_n, feat, self.feature_norm_w, self.feature_norm_b, 1e-5, s));
        const feat_up = try linear(feat_n, .{ .w = self.feature_up_w, .bias = self.feature_up_b }, s);
        defer _ = mlx.mlx_array_free(feat_up);
        const feat_act = try silu(feat_up, s);
        defer _ = mlx.mlx_array_free(feat_act);
        const feat_h = try linear(feat_act, .{ .w = self.feature_down_w, .bias = self.feature_down_b }, s);
        defer _ = mlx.mlx_array_free(feat_h);

        const id_shape = [_]c_int{ 1, @intCast(nodes) };
        const id_arr = mlx.mlx_array_new_data(sorted_ids.ptr, &id_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        const tok = try embed.call(embed.ctx, id_arr);
        defer _ = mlx.mlx_array_free(tok);
        const tok_h = try linear(tok, .{ .w = self.token_w, .bias = self.token_b }, s);
        defer _ = mlx.mlx_array_free(tok_h);

        var pass_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pass_rows);
        {
            const ph = mlx.getShape(pass_hidden);
            if (ph.len != 3 or ph[1] < @as(c_int, @intCast(slots)) + 1) return error.InvalidLilicorrDraft;
            const start = [_]c_int{ 0, 1, 0 };
            const stop = [_]c_int{ 1, 1 + @as(c_int, @intCast(slots)), ph[2] };
            const step = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&pass_rows, pass_hidden, &start, 3, &stop, 3, &step, 3, s));
        }
        const pass_h = try linear(pass_rows, .{ .w = self.pass_w, .bias = self.pass_b }, s);
        defer _ = mlx.mlx_array_free(pass_h);

        const pass_rep = try repeatSlots(pass_h, @intCast(slots), @intCast(k), @intCast(self.hidden), s);
        defer _ = mlx.mlx_array_free(pass_rep);

        const pos = try slotRank(self.slot_emb, self.rank_emb, @intCast(slots), @intCast(k), @intCast(self.hidden), s);
        defer _ = mlx.mlx_array_free(pos);

        var x = try add4(tok_h, pass_rep, feat_h, pos, s);
        defer _ = mlx.mlx_array_free(x);

        const anchor_h = try projectAnchor(anchor, self.context_w, self.context_b, s);
        defer _ = mlx.mlx_array_free(anchor_h);

        for (self.layers) |*lw| {
            const nxt = try self.layer(x, lw);
            _ = mlx.mlx_array_free(x);
            x = nxt;
        }

        const x_n = try rms(x, self.output_norm, self.rms_eps, s);
        _ = mlx.mlx_array_free(x);
        x = x_n;

        const anchor_n = try rms(anchor_h, self.anchor_norm, self.rms_eps, s);
        defer _ = mlx.mlx_array_free(anchor_n);

        var anchor_nodes = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(anchor_nodes);
        try mlx.check(mlx.mlx_broadcast_to(&anchor_nodes, anchor_n, &[_]c_int{ 1, @intCast(nodes), @intCast(self.hidden) }, 3, s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_multiply(&prod, x, anchor_nodes, s));
        var factor_in = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(factor_in);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, x);
            _ = mlx.mlx_vector_array_append_value(vec, anchor_nodes);
            _ = mlx.mlx_vector_array_append_value(vec, prod);
            try mlx.check(mlx.mlx_concatenate_axis(&factor_in, vec, 2, s));
        }
        const factor_lin = try linear(factor_in, .{ .w = self.factor_w, .bias = self.factor_b }, s);
        defer _ = mlx.mlx_array_free(factor_lin);

        const factors = try silu(factor_lin, s);
        defer _ = mlx.mlx_array_free(factors);

        const in_raw = try linear(factors, .{ .w = self.in_w, .bias = self.in_b }, s);
        defer _ = mlx.mlx_array_free(in_raw);
        const in_n = try l2(in_raw, self.vector_eps, s);
        defer _ = mlx.mlx_array_free(in_n);
        const out_raw = try linear(factors, .{ .w = self.out_w, .bias = self.out_b }, s);
        defer _ = mlx.mlx_array_free(out_raw);
        const out_n = try l2(out_raw, self.vector_eps, s);
        defer _ = mlx.mlx_array_free(out_n);
        const anc_raw = try linear(anchor_n, .{ .w = self.anchor_out_w, .bias = self.anchor_out_b }, s);
        defer _ = mlx.mlx_array_free(anc_raw);
        const anc_n = try l2(anc_raw, self.vector_eps, s);
        defer _ = mlx.mlx_array_free(anc_n);

        const lattice = [_]c_int{ @intCast(slots), @intCast(k), @intCast(self.factor) };
        var in_lat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(in_lat);
        try mlx.check(mlx.mlx_reshape(&in_lat, in_n, &lattice, 3, s));
        var out_lat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_lat);
        try mlx.check(mlx.mlx_reshape(&out_lat, out_n, &lattice, 3, s));

        var first = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(first);
        {
            var in0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(in0);
            try mlx.check(mlx.mlx_slice(&in0, in_lat, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, @intCast(k), @intCast(self.factor) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var in0_2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(in0_2);
            try mlx.check(mlx.mlx_reshape(&in0_2, in0, &[_]c_int{ @intCast(k), @intCast(self.factor) }, 2, s));
            var anc_col = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(anc_col);
            try mlx.check(mlx.mlx_reshape(&anc_col, anc_n, &[_]c_int{ @intCast(self.factor), 1 }, 2, s));
            var dots = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dots);
            try mlx.check(mlx.mlx_matmul(&dots, in0_2, anc_col, s));
            const scale = try scalarOf(self.logit_scale, mlx.mlx_array_dtype(dots), s);
            defer _ = mlx.mlx_array_free(scale);
            try mlx.check(mlx.mlx_multiply(&first, dots, scale, s));
        }
        var pair = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(pair);
        if (slots > 1) {
            var prev = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(prev);
            var next = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(next);
            const pair_n: c_int = @intCast(slots - 1);
            try mlx.check(mlx.mlx_slice(&prev, out_lat, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ pair_n, @intCast(k), @intCast(self.factor) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            try mlx.check(mlx.mlx_slice(&next, in_lat, &[_]c_int{ 1, 0, 0 }, 3, &[_]c_int{ @intCast(slots), @intCast(k), @intCast(self.factor) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var next_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(next_t);
            try mlx.check(mlx.mlx_transpose_axes(&next_t, next, &[_]c_int{ 0, 2, 1 }, 3, s));
            var dots = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dots);
            try mlx.check(mlx.mlx_matmul(&dots, prev, next_t, s));
            const scale = try scalarOf(self.logit_scale, mlx.mlx_array_dtype(dots), s);
            defer _ = mlx.mlx_array_free(scale);
            try mlx.check(mlx.mlx_multiply(&pair, dots, scale, s));
        } else {
            _ = mlx.mlx_array_free(pair);
            pair = mlx.mlx_array_new_float(0);
        }
        return .{ .first = first, .pair = pair };
    }

    fn layer(self: *const Head, x: mlx.mlx_array, lw: *const Layer) !mlx.mlx_array {
        const s = self.s;
        const nodes: c_int = @intCast(self.slots * self.top_k);
        const hd: c_int = @intCast(self.hidden / self.heads);
        const normed = try rms(x, lw.attn_norm, self.rms_eps, s);
        defer _ = mlx.mlx_array_free(normed);
        const qkv = try linear(normed, .{ .w = lw.in_w, .bias = lw.in_b }, s);
        defer _ = mlx.mlx_array_free(qkv);
        const q = try headsOf(qkv, 0, self.heads, hd, s);
        defer _ = mlx.mlx_array_free(q);
        const kk = try headsOf(qkv, 1, self.heads, hd, s);
        defer _ = mlx.mlx_array_free(kk);
        const v = try headsOf(qkv, 2, self.heads, hd, s);
        defer _ = mlx.mlx_array_free(v);
        var mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_reshape(&mask, self.attn_bias, &[_]c_int{ 1, @intCast(self.heads), nodes, nodes }, 4, s));
        var mask_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_dt);
        try mlx.check(mlx.mlx_astype(&mask_dt, mask, mlx.mlx_array_dtype(q), s));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const none: mlx.mlx_array = .{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, kk, v, scale, "array", mask_dt, none, false, s));
        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, attn_t, &[_]c_int{ 1, nodes, @intCast(self.hidden) }, 3, s));
        const attn_y = try linear(flat, .{ .w = lw.out_w, .bias = lw.out_b }, s);
        defer _ = mlx.mlx_array_free(attn_y);
        var h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_add(&h, x, attn_y, s));
        const mlp_n = try rms(h, lw.mlp_norm, self.rms_eps, s);
        defer _ = mlx.mlx_array_free(mlp_n);
        const up = try linear(mlp_n, .{ .w = lw.up_w, .bias = lw.up_b }, s);
        defer _ = mlx.mlx_array_free(up);
        const act = try silu(up, s);
        defer _ = mlx.mlx_array_free(act);
        const down = try linear(act, .{ .w = lw.down_w, .bias = lw.down_b }, s);
        defer _ = mlx.mlx_array_free(down);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, h, down, s));
        return out;
    }
};

pub const Embed = struct {
    ctx: *const anyopaque,
    call: *const fn (ctx: *const anyopaque, ids: mlx.mlx_array) anyerror!mlx.mlx_array,
};

const Candidates = struct {
    ids: []i32,
    features: []f32,

    fn deinit(self: *Candidates, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.features);
    }
};

fn candidateLattice(allocator: std.mem.Allocator, draft_logits: mlx.mlx_array, slots: usize, k: usize, s: mlx.mlx_stream) !Candidates {
    const lg = mlx.getShape(draft_logits);
    if (lg.len != 3 or lg[1] != @as(c_int, @intCast(slots))) return error.InvalidLilicorrDraft;
    const vocab: c_int = lg[2];
    if (vocab < @as(c_int, @intCast(k))) return error.InvalidLilicorrDraft;

    var logits_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f);
    try mlx.check(mlx.mlx_astype(&logits_f, draft_logits, .float32, s));

    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, logits_f, vocab - @as(c_int, @intCast(k)), 2, s));
    var cand_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_raw);
    const sl_start = [_]c_int{ 0, 0, vocab - @as(c_int, @intCast(k)) };
    const sl_stop = [_]c_int{ 1, @intCast(slots), vocab };
    const sl_step = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&cand_raw, part, &sl_start, 3, &sl_stop, 3, &sl_step, 3, s));
    var cands = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cands);
    try mlx.check(mlx.mlx_astype(&cands, cand_raw, .int32, s));

    var picked_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked_logits);
    try mlx.check(mlx.mlx_take_along_axis(&picked_logits, logits_f, cands, 2, s));
    var log_norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_norm);
    try mlx.check(mlx.mlx_logsumexp_axis(&log_norm, logits_f, 2, true, s));
    var picked_log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked_log_probs);
    try mlx.check(mlx.mlx_subtract(&picked_log_probs, picked_logits, log_norm, s));
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, cands);
        _ = mlx.mlx_vector_array_append_value(ev, picked_logits);
        _ = mlx.mlx_vector_array_append_value(ev, picked_log_probs);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const cand_data = mlx.mlx_array_data_int32(cands) orelse return error.MlxArrayDataNull;
    const logit_data = mlx.mlx_array_data_float32(picked_logits) orelse return error.MlxArrayDataNull;
    const log_prob_data = mlx.mlx_array_data_float32(picked_log_probs) orelse return error.MlxArrayDataNull;

    const nodes = slots * k;
    const sorted_ids = try allocator.alloc(i32, nodes);
    errdefer allocator.free(sorted_ids);
    const features = try allocator.alloc(f32, nodes * 5);
    errdefer allocator.free(features);
    var order: [64]usize = undefined;
    if (k > order.len) return error.InvalidLilicorrDraft;
    for (0..slots) |slot| {
        const base = slot * k;
        for (0..k) |j| order[j] = j;
        std.sort.insertion(usize, order[0..k], SortCtx{ .logits = logit_data[base .. base + k], .ids = cand_data[base .. base + k] }, SortCtx.less);
        const top_log = log_prob_data[base + order[0]];
        for (0..k) |rank| {
            const src = order[rank];
            const dst = base + rank;
            sorted_ids[dst] = cand_data[base + src];
            const lp = log_prob_data[base + src];
            const p = @exp(lp);
            features[dst * 5 + 0] = lp;
            features[dst * 5 + 1] = p;
            features[dst * 5 + 2] = lp - top_log;
            features[dst * 5 + 3] = if (k == 1) 0 else @as(f32, @floatFromInt(rank)) / @as(f32, @floatFromInt(k - 1));
            features[dst * 5 + 4] = if (rank == 0) 1 else 0;
        }
    }

    return .{ .ids = sorted_ids, .features = features };
}

const SortCtx = struct {
    logits: []const f32,
    ids: []const i32,

    fn less(ctx: SortCtx, a: usize, b: usize) bool {
        if (ctx.logits[a] == ctx.logits[b]) return ctx.ids[a] < ctx.ids[b];
        return ctx.logits[a] > ctx.logits[b];
    }
};

/// `cands` is `[slots, k]` already sorted by logit, descending, within each row.
/// `first` is `[k]`. `pair` is `[slots-1, k, k]` with `pair[t, prev, next]`, or empty when `slots < 2`.
pub fn greedyPick(cands: []const i32, first: []const f32, pair: []const f32, slots: usize, k: usize, n_out: usize, out: []u32) void {
    var pred: usize = 0;
    for (0..n_out) |slot| {
        const scores = if (slot == 0) first else pair[(slot - 1) * k * k + pred * k ..][0..k];
        var best: usize = 0;
        for (scores, 0..) |v, j| if (v > scores[best]) {
            best = j;
        };
        pred = best;
        out[slot] = @intCast(cands[slot * k + best]);
    }
    _ = slots;
}

pub fn load(allocator: std.mem.Allocator, weights: *const Weights, s: mlx.mlx_stream, cfg: LoadCfg) !Head {
    if (cfg.hidden == 0 or cfg.heads == 0 or cfg.hidden % cfg.heads != 0) return error.InvalidLilicorrWeight;
    if (cfg.slots == 0 or cfg.top_k == 0 or cfg.layers == 0) return error.InvalidLilicorrWeight;
    var head = Head{
        .s = s,
        .allocator = allocator,
        .top_k = cfg.top_k,
        .hidden = cfg.hidden,
        .trunk = cfg.trunk,
        .heads = cfg.heads,
        .slots = cfg.slots,
        .factor = cfg.factor,
        .vector_eps = cfg.vector_eps,
        .logit_scale = cfg.logit_scale,
        .rms_eps = cfg.rms_eps,
    };
    errdefer head.deinit();
    head.context_w = try load2(weights, "lilicorr.context_proj.weight");
    head.context_b = try loadKeep(weights, "lilicorr.context_proj.bias");
    head.token_w = try load2(weights, "lilicorr.token_proj.weight");
    head.token_b = try loadKeep(weights, "lilicorr.token_proj.bias");
    head.pass_w = try load2(weights, "lilicorr.pass_hidden_proj.weight");
    head.pass_b = try loadKeep(weights, "lilicorr.pass_hidden_proj.bias");
    head.feature_norm_w = try loadKeep(weights, "lilicorr.feature_norm.weight");
    head.feature_norm_b = try loadKeep(weights, "lilicorr.feature_norm.bias");
    head.feature_up_w = try load2(weights, "lilicorr.feature_mlp.up_proj.weight");
    head.feature_up_b = try loadKeep(weights, "lilicorr.feature_mlp.up_proj.bias");
    head.feature_down_w = try load2(weights, "lilicorr.feature_mlp.down_proj.weight");
    head.feature_down_b = try loadKeep(weights, "lilicorr.feature_mlp.down_proj.bias");
    head.slot_emb = try load2(weights, "lilicorr.slot_embedding");
    head.rank_emb = try load2(weights, "lilicorr.rank_embedding");
    head.attn_bias = try loadKeep(weights, "lilicorr.attention_bias");
    head.output_norm = try loadKeep(weights, "lilicorr.output_norm.weight");
    head.anchor_norm = try loadKeep(weights, "lilicorr.anchor_norm.weight");
    head.factor_w = try load2(weights, "lilicorr.factor_input_proj.weight");
    head.factor_b = try loadKeep(weights, "lilicorr.factor_input_proj.bias");
    head.in_w = try load2(weights, "lilicorr.in_head.weight");
    head.in_b = try loadKeep(weights, "lilicorr.in_head.bias");
    head.out_w = try load2(weights, "lilicorr.out_head.weight");
    head.out_b = try loadKeep(weights, "lilicorr.out_head.bias");
    head.anchor_out_w = try load2(weights, "lilicorr.anchor_out_head.weight");
    head.anchor_out_b = try loadKeep(weights, "lilicorr.anchor_out_head.bias");
    head.layers = try allocator.alloc(Layer, cfg.layers);
    for (head.layers) |*lw| lw.* = .{};
    var n: usize = 0;
    var buf: [160]u8 = undefined;
    while (n < cfg.layers) : (n += 1) {
        const lw = &head.layers[n];
        lw.attn_norm = try loadKeep(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.attn_norm.weight", .{n}));
        lw.mlp_norm = try loadKeep(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.mlp_norm.weight", .{n}));
        lw.in_w = try load2(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.attn.in_proj.weight", .{n}));
        lw.in_b = try loadKeep(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.attn.in_proj.bias", .{n}));
        lw.out_w = try load2(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.attn.out_proj.weight", .{n}));
        lw.out_b = try loadKeep(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.attn.out_proj.bias", .{n}));
        lw.up_w = try load2(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.mlp.up_proj.weight", .{n}));
        lw.up_b = try loadKeep(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.mlp.up_proj.bias", .{n}));
        lw.down_w = try load2(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.mlp.down_proj.weight", .{n}));
        lw.down_b = try loadKeep(weights, try std.fmt.bufPrint(&buf, "lilicorr.layers.{d}.mlp.down_proj.bias", .{n}));
    }
    const nodes: usize = @as(usize, cfg.slots) * cfg.top_k;
    const bias_shape = mlx.getShape(head.attn_bias);
    if (bias_shape.len != 3 or bias_shape[0] != @as(c_int, @intCast(cfg.heads)) or bias_shape[1] != @as(c_int, @intCast(nodes)) or bias_shape[2] != @as(c_int, @intCast(nodes))) {
        log.err("[lilicorr] attention_bias is {any}, expected [{d}, {d}, {d}]\n", .{ bias_shape, cfg.heads, nodes, nodes });
        return error.InvalidLilicorrWeight;
    }
    log.info("[lilicorr] head hidden={d} layers={d} heads={d} top_k={d} slots={d}\n", .{
        cfg.hidden, cfg.layers, cfg.heads, cfg.top_k, cfg.slots,
    });
    return head;
}

fn loadKeep(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[lilicorr] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingLilicorrWeight;
    };
    var owned = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(owned);
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

fn load2(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const raw = try loadKeep(w, key);
    defer _ = mlx.mlx_array_free(raw);
    if (mlx.mlx_array_ndim(raw) != 2) {
        log.err("[lilicorr] {s} is rank {d}, expected 2\n", .{ key, mlx.mlx_array_ndim(raw) });
        return error.InvalidLilicorrWeight;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_set(&out, raw));
    return out;
}

fn silu(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, x, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, x, sg, s));
    return out;
}

fn rms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

fn l2(x: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, x, s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_sum_axis(&sum, sq, -1, true, s));
    const eps_a = try scalarOf(eps, mlx.mlx_array_dtype(x), s);
    defer _ = mlx.mlx_array_free(eps_a);
    var denom_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(denom_in);
    try mlx.check(mlx.mlx_sqrt(&denom_in, sum, s));
    var denom = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(denom);
    try mlx.check(mlx.mlx_maximum(&denom, denom_in, eps_a, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&out, x, denom, s));
    return out;
}

fn add4(a: mlx.mlx_array, b: mlx.mlx_array, c: mlx.mlx_array, d: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_add(&ab, a, b, s));
    var abc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(abc);
    try mlx.check(mlx.mlx_add(&abc, ab, c, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, abc, d, s));
    return out;
}

fn repeatSlots(rows: mlx.mlx_array, slots: c_int, k: c_int, hidden: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, rows, &[_]c_int{ 1, slots, 1, hidden }, 4, s));
    var broad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(broad);
    try mlx.check(mlx.mlx_broadcast_to(&broad, shaped, &[_]c_int{ 1, slots, k, hidden }, 4, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, broad, &[_]c_int{ 1, slots * k, hidden }, 3, s));
    return out;
}

fn slotRank(slot: mlx.mlx_array, rank: mlx.mlx_array, slots: c_int, k: c_int, hidden: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var slot_s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slot_s);
    try mlx.check(mlx.mlx_reshape(&slot_s, slot, &[_]c_int{ 1, slots, 1, hidden }, 4, s));
    var rank_s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rank_s);
    try mlx.check(mlx.mlx_reshape(&rank_s, rank, &[_]c_int{ 1, 1, k, hidden }, 4, s));
    var slot_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slot_b);
    try mlx.check(mlx.mlx_broadcast_to(&slot_b, slot_s, &[_]c_int{ 1, slots, k, hidden }, 4, s));
    var rank_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rank_b);
    try mlx.check(mlx.mlx_broadcast_to(&rank_b, rank_s, &[_]c_int{ 1, slots, k, hidden }, 4, s));
    var added = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(added);
    try mlx.check(mlx.mlx_add(&added, slot_b, rank_b, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, added, &[_]c_int{ 1, slots * k, hidden }, 3, s));
    return out;
}

fn projectAnchor(anchor: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(anchor);
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    if (sh.len == 1) {
        try mlx.check(mlx.mlx_reshape(&shaped, anchor, &[_]c_int{ 1, 1, sh[0] }, 3, s));
    } else if (sh.len == 2) {
        try mlx.check(mlx.mlx_reshape(&shaped, anchor, &[_]c_int{ 1, 1, sh[1] }, 3, s));
    } else if (sh.len == 3) {
        try mlx.check(mlx.mlx_array_set(&shaped, anchor));
    } else return error.InvalidLilicorrDraft;
    return linear(shaped, .{ .w = w, .bias = b }, s);
}

/// Split fused QKV `[1, nodes, 3H]` into `[1, heads, nodes, head_dim]`. `which` is 0, 1, or 2.
fn headsOf(qkv: mlx.mlx_array, which: c_int, heads: u32, head_dim: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(qkv);
    const width = head_dim * @as(c_int, @intCast(heads));
    var chunk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk);
    const start = [_]c_int{ 0, 0, which * width };
    const stop = [_]c_int{ sh[0], sh[1], (which + 1) * width };
    try mlx.check(mlx.mlx_slice(&chunk, qkv, &start, 3, &stop, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, chunk, &[_]c_int{ sh[0], sh[1], @intCast(heads), head_dim }, 4, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, reshaped, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    return out;
}

test "lilicorr greedy walk follows the correlation scores" {
    const cands = [_]i32{ 10, 11, 20, 21 };
    const first = [_]f32{ 0.1, 0.9 };
    const pair = [_]f32{ 5, 0, 0, 7 };
    var out: [2]u32 = undefined;
    greedyPick(&cands, &first, &pair, 2, 2, 2, &out);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 11, 21 }, &out);
}

test "lilicorr L2 matches reference clamp and preserves the activation dtype" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const raw = [_]f32{ 0.00003, 0.00004, 0.3, 0.4 };
    const x = mlx.mlx_array_new_data(&raw, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    const y = try l2(x, 0.0001, s);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    const got = mlx.mlx_array_data_float32(y).?;
    for ([_]f32{ 0.3, 0.4, 0.6, 0.8 }, 0..) |v, i| try std.testing.expectApproxEqAbs(v, got[i], 1e-6);
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, x, .bfloat16, s));
    const y_bf = try l2(bf, 0.0001, s);
    defer _ = mlx.mlx_array_free(y_bf);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y_bf));
}

test "lilicorr head scores match NVIDIA reference in fp32" {
    if (mlx.noGpuBackend()) return;
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "fixture.safetensors", .data = @embedFile("../../../../fixtures/lilicorr_fp32.safetensors") });
    var dir_buf: [512]u8 = undefined;
    const dir_len = try tmp.dir.realPath(io, &dir_buf);
    const path = try std.fmt.allocPrint(a, "{s}/fixture.safetensors", .{dir_buf[0..dir_len]});
    defer a.free(path);
    var fx = try model_mod.loadWeightsSingleFile(a, path);
    defer fx.deinit();
    var head = try load(a, &fx, s, .{ .top_k = 3, .hidden = 8, .trunk = 12, .heads = 2, .slots = 3, .layers = 2, .factor = 6, .vector_eps = 0.0001, .logit_scale = 8, .rms_eps = 1e-6 });
    defer head.deinit();
    const embed = Embed{ .ctx = &fx, .call = struct {
        fn call(ctx: *const anyopaque, ids: mlx.mlx_array) !mlx.mlx_array {
            const w: *const Weights = @ptrCast(@alignCast(ctx));
            var out = mlx.mlx_array_new();
            const cs = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cs);
            try mlx.check(mlx.mlx_take_axis(&out, w.get("fixture.embedding").?, ids, 0, cs));
            return out;
        }
    }.call };
    for ([_][]const u8{ "fixture", "fixture.confident" }) |prefix| {
        var kb: [96]u8 = undefined;
        const logits = fx.get(try std.fmt.bufPrint(&kb, "{s}.logits", .{prefix})).?;
        const want_ids_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}.ids", .{prefix})).?;
        const want_feat_arr = fx.get(try std.fmt.bufPrint(&kb, "{s}.features", .{prefix})).?;
        try mlx.check(mlx.mlx_array_eval(want_ids_arr));
        try mlx.check(mlx.mlx_array_eval(want_feat_arr));
        const want_ids = mlx.mlx_array_data_int32(want_ids_arr).?[0..9];
        const want_features = mlx.mlx_array_data_float32(want_feat_arr).?[0..45];
        var candidates = try candidateLattice(a, logits, 3, 3, s);
        defer candidates.deinit(a);
        try std.testing.expectEqualSlices(i32, want_ids, candidates.ids);
        for (candidates.features, want_features) |x, y| try std.testing.expectApproxEqAbs(y, x, 1e-5);
        var scores = try head.score(embed, fx.get("fixture.anchor").?, fx.get("fixture.pass_hidden").?, candidates.ids, candidates.features);
        defer scores.deinit();
        for ([_]mlx.mlx_array{ scores.first, scores.pair }, [_][]const u8{ "first", "pair" }) |got, suffix| {
            const want = fx.get(try std.fmt.bufPrint(&kb, "{s}.{s}", .{ prefix, suffix })).?;
            try mlx.check(mlx.mlx_array_eval(got));
            try mlx.check(mlx.mlx_array_eval(want));
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(got));
            const n = mlx.mlx_array_size(want);
            const actual = mlx.mlx_array_data_float32(got).?[0..n];
            const expected = mlx.mlx_array_data_float32(want).?[0..n];
            for (actual, expected) |x, y| try std.testing.expectApproxEqAbs(y, x, 2e-5);
        }
        var expected_ids: [3]u32 = undefined;
        const first = fx.get(try std.fmt.bufPrint(&kb, "{s}.first", .{prefix})).?;
        const pair = fx.get(try std.fmt.bufPrint(&kb, "{s}.pair", .{prefix})).?;
        greedyPick(want_ids, mlx.mlx_array_data_float32(first).?[0..3], mlx.mlx_array_data_float32(pair).?[0..18], 3, 3, 3, &expected_ids);
        const drafted = try head.draft(a, embed, fx.get("fixture.anchor").?, fx.get("fixture.pass_hidden").?, logits, 3);
        defer a.free(drafted);
        try std.testing.expectEqualSlices(u32, &expected_ids, drafted);
    }
    const ids_arr = fx.get("fixture.ids").?;
    const features_arr = fx.get("fixture.features").?;
    const ids = mlx.mlx_array_data_int32(ids_arr).?[0..9];
    const features = mlx.mlx_array_data_float32(features_arr).?[0..45];

    var it = fx.map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, key, "lilicorr.") and !std.mem.eql(u8, key, "fixture.embedding") and !std.mem.eql(u8, key, "fixture.anchor") and !std.mem.eql(u8, key, "fixture.pass_hidden")) continue;
        var narrowed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&narrowed, entry.value_ptr.*, .bfloat16, s));
        _ = mlx.mlx_array_free(entry.value_ptr.*);
        entry.value_ptr.* = narrowed;
    }
    var bf_head = try load(a, &fx, s, .{ .top_k = 3, .hidden = 8, .trunk = 12, .heads = 2, .slots = 3, .layers = 2, .factor = 6, .vector_eps = 0.0001, .logit_scale = 8, .rms_eps = 1e-6 });
    defer bf_head.deinit();
    var bf_scores = try bf_head.score(embed, fx.get("fixture.anchor").?, fx.get("fixture.pass_hidden").?, ids, features);
    defer bf_scores.deinit();
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(bf_scores.first));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(bf_scores.pair));
    try mlx.check(mlx.mlx_array_eval(bf_scores.first));
    try mlx.check(mlx.mlx_array_eval(bf_scores.pair));

    fx.remove("lilicorr.layers.1.mlp.down_proj.bias");
    try std.testing.expectError(error.MissingLilicorrWeight, load(a, &fx, s, .{ .top_k = 3, .hidden = 8, .trunk = 12, .heads = 2, .slots = 3, .layers = 2, .factor = 6, .vector_eps = 0.0001, .logit_scale = 8, .rms_eps = 1e-6 }));
}

test "lilicorr candidate log probabilities retain confident tails" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const raw = [_]f32{ 100, 0, -1, -3, 0, 90, -5, -4 };
    const logits = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 2, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var candidates = try candidateLattice(a, logits, 2, 3, s);
    defer candidates.deinit(a);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2, 1, 0, 3 }, candidates.ids);
    for ([_]f32{ 0, -100, -101, 0, -90, -94 }, 0..) |lp, i| {
        try std.testing.expectApproxEqAbs(lp, candidates.features[i * 5], 1e-5);
        try std.testing.expectApproxEqAbs(lp, candidates.features[i * 5 + 2], 1e-5);
    }
    try std.testing.expect(candidates.features[6] < 1e-40);
}
