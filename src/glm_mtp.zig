//! GLM-5.3-Flash multi-token prediction: the checkpoint's own `mtp.0` layer (DeepSeek NextN).
//!
//! Row p pairs the trunk's FINAL-NORMED hidden at position p with token p+1:
//! `eh_proj(cat[enorm(embed(x_{p+1})), hnorm(h_p)])` through one PLAIN pre-norm block
//! (DSA attention + routed MoE, no hyper-connection) and `norm`, then the trunk's lm_head
//! predicts x_{p+2}. A chain feeds the normed output back as the next depth's hidden and
//! appends to the head's own one-layer latent + indexer cache (mlx-vlm's `glm5_next_mtp`
//! drafter, which `tests/dump_glm5_next_fixtures.py` records as the oracle).
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const mtp_mod = @import("mtp.zig");

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

pub const Head = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    /// Arrays this head made (transposes, f32 tables); the checkpoint's own are the trunk's.
    owned: std.ArrayList(mlx.mlx_array) = .empty,
    eps: f32,
    enorm: mlx.mlx_array,
    hnorm: mlx.mlx_array,
    norm: mlx.mlx_array,
    input_norm: mlx.mlx_array,
    post_norm: mlx.mlx_array,
    eh_proj: transformer_mod.QW,
    dsa: transformer_mod.DsaWeights,
    moe: transformer_mod.MoeMlpWeights,
    rerank_coarse: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,

    /// The head under `<prefix minus .model>.mtp.0`, or null when the pack ships none.
    pub fn load(allocator: std.mem.Allocator, s: mlx.mlx_stream, config: *const ModelConfig, weights: *const Weights) !?Head {
        var base_buf: [96]u8 = undefined;
        const base = mtpBase(&base_buf, config.weight_prefix);
        var name_buf: [256]u8 = undefined;
        if (weights.get(std.fmt.bufPrint(&name_buf, "{s}.eh_proj.weight", .{base}) catch unreachable) == null) return null;
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        errdefer {
            for (owned.items) |a| _ = mlx.mlx_array_free(a);
            owned.deinit(allocator);
        }
        var block_buf: [104]u8 = undefined;
        const block = std.fmt.bufPrint(&block_buf, "{s}.block", .{base}) catch unreachable;
        var head = Head{
            .s = s,
            .allocator = allocator,
            .eps = config.rms_norm_eps,
            .enorm = try transformer_mod.getWeightAt(weights, &name_buf, base, "enorm.weight"),
            .hnorm = try transformer_mod.getWeightAt(weights, &name_buf, base, "hnorm.weight"),
            .norm = try transformer_mod.getWeightAt(weights, &name_buf, base, "norm.weight"),
            .input_norm = try transformer_mod.getWeightAt(weights, &name_buf, block, "input_layernorm.weight"),
            .post_norm = try transformer_mod.getWeightAt(weights, &name_buf, block, "post_attention_layernorm.weight"),
            .eh_proj = try transformer_mod.loadQWAt(weights, &name_buf, base, "eh_proj"),
            .dsa = try transformer_mod.loadDsaWeightsAt(weights, &name_buf, block, &owned, allocator, s),
            .moe = try transformer_mod.loadRoutedMoeAt(weights, &name_buf, block, true, config, &owned, allocator, s),
        };
        try transformer_mod.maybeTransposeForBf16(&head.eh_proj.w, head.eh_proj.s, &owned, allocator, s);
        head.owned = owned;
        return head;
    }

    pub fn deinit(self: *Head) void {
        if (self.rerank_coarse) |*rc| rc.deinit();
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.allocator);
    }

    pub fn makeCache(self: *const Head, allocator: std.mem.Allocator) !KVCache {
        _ = self;
        return KVCache.init(allocator, 1);
    }

    /// Rows `ids` `[L]` int32 (token p+1 of each row) with `hidden` `[1, L, H]` (the trunk's
    /// final-normed row p at depth 1, this head's normed output past it) appended to `cache` at
    /// its current length: the last row's normed output (the next depth's hidden) and, under
    /// `.logits`, its lm_head logits; under `.mixed` the normed row rides `rerank_x`.
    pub fn forward(self: *const Head, target: *Transformer, cache: *KVCache, ids: mlx.mlx_array, hidden: mlx.mlx_array, want: mtp_mod.StepWant) !mtp_mod.StepOut {
        const s = self.s;
        const out = try self.rows(target, cache, ids, hidden);
        defer _ = mlx.mlx_array_free(out);
        const sh = mlx.getShape(out);
        var last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(last);
        if (sh[1] > 1) {
            try mlx.check(mlx.mlx_slice(&last, out, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        } else _ = mlx.mlx_array_set(&last, out);
        const normed = try self.rmsNorm(last, self.norm);
        errdefer _ = mlx.mlx_array_free(normed);
        return switch (want) {
            .none => .{ .logits = .{ .ctx = null }, .hidden_next = normed },
            .mixed => blk: {
                var copy = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&copy, normed));
                break :blk .{ .logits = .{ .ctx = null }, .hidden_next = normed, .rerank_x = copy };
            },
            .logits => .{ .logits = try target.lmHeadForDraft(normed), .hidden_next = normed },
        };
    }

    /// Committed rows: every (hidden p, token p+1) pair the trunk has verified, in order.
    pub fn appendHistory(self: *const Head, target: *Transformer, cache: *KVCache, token_ids: []const u32, hidden: mlx.mlx_array) !void {
        if (token_ids.len == 0) return;
        const ids_i32 = try self.allocator.alloc(i32, token_ids.len);
        defer self.allocator.free(ids_i32);
        for (token_ids, 0..) |t, i| ids_i32[i] = @intCast(t);
        const shape = [_]c_int{@intCast(token_ids.len)};
        const ids = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
        defer _ = mlx.mlx_array_free(ids);
        const out = try self.forward(target, cache, ids, hidden, .none);
        _ = mlx.mlx_array_free(out.hidden_next);
    }

    /// The block's output rows `[1, L, H]` (pre-`norm`) for `ids` `[L]` and `hidden` `[1, L, H]`,
    /// appended to `cache`.
    pub fn rows(self: *const Head, target: *Transformer, cache: *KVCache, ids: mlx.mlx_array, hidden: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const n: c_int = mlx.getShape(ids)[0];
        var ids2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids2);
        try mlx.check(mlx.mlx_reshape(&ids2, ids, &[_]c_int{ 1, n }, 2, s));
        const emb = try target.embedding(ids2);
        defer _ = mlx.mlx_array_free(emb);
        const en = try self.rmsNorm(emb, self.enorm);
        defer _ = mlx.mlx_array_free(en);
        var hid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hid);
        try mlx.check(mlx.mlx_astype(&hid, hidden, mlx.mlx_array_dtype(emb), s));
        const hn = try self.rmsNorm(hid, self.hnorm);
        defer _ = mlx.mlx_array_free(hn);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        {
            const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ en, hn }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
        }
        const x = try target.qmatmul(cat, self.eh_proj.w, self.eh_proj.s, self.eh_proj.b);
        defer _ = mlx.mlx_array_free(x);

        const a_in = try self.rmsNorm(x, self.input_norm);
        defer _ = mlx.mlx_array_free(a_in);
        var seq_offset: usize = 0;
        var ctx = transformer_mod.ForwardCtx{
            .cache = cache,
            .moe_seq_offset = &seq_offset,
            .ssm_entries = null,
            .capture_hidden = null,
            .vision_embeddings = null,
        };
        const attn = try target.glmDsaAttn(&ctx, a_in, &self.dsa, 0, cache.seqLen(0), 1, n);
        defer _ = mlx.mlx_array_free(attn);
        var x1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x1);
        try mlx.check(mlx.mlx_add(&x1, x, attn, s));
        const m_in = try self.rmsNorm(x1, self.post_norm);
        defer _ = mlx.mlx_array_free(m_in);
        const mlp = try target.moeMLP(m_in, &self.moe);
        defer _ = mlx.mlx_array_free(mlp);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_add(&out, x1, mlp, s));
        return out;
    }

    fn rmsNorm(self: *const Head, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, self.eps, self.s));
        return out;
    }

    pub fn bindRerank(self: *Head, target: *Transformer) void {
        if (mtp_mod.MtpModel.draftRerankMode() == .off or target.lm_head_s.ctx == null) return;
        self.rerank_coarse = mtp_mod.buildRerankCoarse(self.s, target, mtp_mod.rerankCoarseBits());
    }

    pub fn canRerankDrafts(self: *const Head) bool {
        return self.rerank_coarse != null;
    }

    pub fn draftSelect(self: *Head, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
        if (try mtp_mod.rerankSelect(self.s, target, &self.rerank_coarse, &self.rerank_logged, x, suppress_mask)) |tok| return tok;
        return mtp_mod.fullReadoutArgmax(self.s, target, x, suppress_mask);
    }

    pub fn draftShortlist(self: *Head, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mtp_mod.Shortlist {
        return mtp_mod.rerankShortlist(self.s, target, &self.rerank_coarse, &self.rerank_logged, x, suppress_mask);
    }
};

/// `language_model.model` -> `language_model.mtp.0`, `model` -> `model.mtp.0`: the MTP layer
/// sits beside the trunk's `.model`, not under it.
fn mtpBase(buf: *[96]u8, prefix: []const u8) []const u8 {
    const root = if (std.mem.endsWith(u8, prefix, ".model")) prefix[0 .. prefix.len - ".model".len] else prefix;
    return std.fmt.bufPrint(buf, "{s}.mtp.0", .{root}) catch unreachable;
}

const testing = std.testing;

test "glm mtp: the head's name base sits beside the trunk's .model prefix" {
    var buf: [96]u8 = undefined;
    try testing.expectEqualStrings("language_model.mtp.0", mtpBase(&buf, "language_model.model"));
    try testing.expectEqualStrings("model.mtp.0", mtpBase(&buf, "model"));
}
