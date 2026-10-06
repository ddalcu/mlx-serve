//! MiMo-V2.6 multi-token prediction: the checkpoint's own `model.mtp.layers.{k}` heads.
//!
//! Each head is one sliding-window decoder layer with sinks and a dense SwiGLU, fed
//! `eh_proj(cat[enorm(embed(token)), hnorm(target_hidden)])` and read through the
//! trunk's shared lm_head after its own `final_layernorm` (vLLM / SGLang MiMo-V2
//! MTP, XiaomiMiMo, Apache-2.0). Head k's row p pairs the trunk's final-normed
//! hidden at position p with token p+k+1 at rope position p and predicts token
//! p+k+2: every head reads the TARGET's hidden, only the token shifts (SGLang's
//! multi-layer EAGLE for MiMo). A round drafts d1..dm by running head i at the
//! round's last committed position q; head i's rows past q-i carry drafts and
//! are rolled back before the next round.
//!
//! Positions are the generator's head-relative positions (0 = the first row its
//! history ever appended), so only differences reach the rope.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const mtp_mod = @import("mtp.zig");

const Transformer = transformer_mod.Transformer;
const Weights = model_mod.Weights;
const ModelConfig = model_mod.ModelConfig;
const QLinear = mtp_mod.QLinear;

pub const MAX_HEADS: usize = 3;

/// Committed rows (hiddens and their next tokens) a state keeps: every head's
/// window plus the rows a lagging head catches up and a round's drafts.
const RING_ROWS: usize = 256;

const Layer = struct {
    enorm: mlx.mlx_array,
    hnorm: mlx.mlx_array,
    input_norm: mlx.mlx_array,
    pre_mlp_norm: mlx.mlx_array,
    final_norm: mlx.mlx_array,
    sinks: mlx.mlx_array,
    eh_proj: QLinear,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
};

/// Rows `[base, base + len)` of one head's keys and values, the newest last.
const RowCache = struct {
    k: mlx.mlx_array = .{ .ctx = null },
    v: mlx.mlx_array = .{ .ctx = null },
    base: usize = 0,
    len: usize = 0,

    fn end(self: *const RowCache) usize {
        return self.base + self.len;
    }

    fn reset(self: *RowCache, at: usize) void {
        if (self.k.ctx != null) _ = mlx.mlx_array_free(self.k);
        if (self.v.ctx != null) _ = mlx.mlx_array_free(self.v);
        self.* = .{ .base = at };
    }

    /// Drop every row at or past `at`; a cut below the kept rows starts over at `at`.
    fn truncate(self: *RowCache, s: mlx.mlx_stream, at: usize) !void {
        if (at >= self.end()) return;
        if (at <= self.base) return self.reset(at);
        const keep: c_int = @intCast(at - self.base);
        try sliceRows(s, &self.k, 0, keep);
        try sliceRows(s, &self.v, 0, keep);
        self.len = at - self.base;
    }

    /// Append rows starting at `pos0 == end()`, keeping the `window - 1` rows
    /// before them that the first new row attends.
    fn append(self: *RowCache, s: mlx.mlx_stream, k: mlx.mlx_array, v: mlx.mlx_array, pos0: usize, window: usize) !void {
        if (self.len == 0) self.base = pos0;
        if (pos0 != self.end()) return error.MimoMtpRowGap;
        const rows: usize = @intCast(mlx.getShape(k)[2]);
        const keep = @min(self.len, window - 1);
        if (keep < self.len) {
            try sliceRows(s, &self.k, @intCast(self.len - keep), @intCast(self.len));
            try sliceRows(s, &self.v, @intCast(self.len - keep), @intCast(self.len));
            self.base = pos0 - keep;
        }
        if (keep == 0) {
            if (self.k.ctx != null) _ = mlx.mlx_array_free(self.k);
            if (self.v.ctx != null) _ = mlx.mlx_array_free(self.v);
            self.k = mlx.mlx_array_new();
            self.v = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&self.k, k));
            try mlx.check(mlx.mlx_array_set(&self.v, v));
        } else {
            try concatRows(s, &self.k, k);
            try concatRows(s, &self.v, v);
        }
        self.len = keep + rows;
    }
};

fn sliceRows(s: mlx.mlx_stream, a: *mlx.mlx_array, start: c_int, stop: c_int) !void {
    const sh = mlx.getShape(a.*);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a.*, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], stop, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    _ = mlx.mlx_array_free(a.*);
    a.* = out;
}

fn concatRows(s: mlx.mlx_stream, a: *mlx.mlx_array, b: mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ a.*, b }, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    _ = mlx.mlx_array_free(a.*);
    a.* = out;
}

/// One request's head state: each head's rows, and the committed target hiddens
/// and tokens its catch-up rows are built from.
pub const State = struct {
    allocator: std.mem.Allocator,
    rows: [MAX_HEADS]RowCache = @splat(.{}),
    /// Rows past this carry a draft token (head k), so a round start drops them.
    committed_end: [MAX_HEADS]usize = @splat(0),
    /// Target hiddens `[1, n, H]` at positions `[hid_base, hid_base + n)`.
    hid: mlx.mlx_array = .{ .ctx = null },
    hid_base: usize = 0,
    hid_len: usize = 0,
    /// Committed tokens x_p for p in `[tok_base, tok_base + toks.len)`.
    toks: std.ArrayList(u32) = .empty,
    tok_base: usize = 0,
    /// This round: q = its last committed position, the drafts d1.. so far.
    round_q: usize = 0,
    drafts: [MAX_HEADS]mlx.mlx_array = @splat(.{ .ctx = null }),
    n_drafts: usize = 0,

    pub fn deinit(self: *State) void {
        for (&self.rows) |*r| r.reset(0);
        if (self.hid.ctx != null) _ = mlx.mlx_array_free(self.hid);
        self.clearDrafts();
        self.toks.deinit(self.allocator);
    }

    fn clearDrafts(self: *State) void {
        for (self.drafts[0..self.n_drafts]) |d| _ = mlx.mlx_array_free(d);
        self.n_drafts = 0;
    }

    /// Committed rows: head 0's history length, the generator's `step()`.
    pub fn step(self: *const State) usize {
        return self.hid_base + self.hid_len;
    }

    fn tokEnd(self: *const State) usize {
        return self.tok_base + self.toks.items.len;
    }

    /// Keep only what committed length `len` still vouches for: hiddens below
    /// `len`, tokens up to x_len, head k's rows whose token x_{p+k+1} is one of those.
    pub fn truncate(self: *State, s: mlx.mlx_stream, len: usize) !void {
        if (len < self.step()) {
            if (len <= self.hid_base) {
                if (self.hid.ctx != null) _ = mlx.mlx_array_free(self.hid);
                self.hid = .{ .ctx = null };
                self.hid_base = len;
                self.hid_len = 0;
            } else {
                var cut = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(cut);
                const sh = mlx.getShape(self.hid);
                try mlx.check(mlx.mlx_slice(&cut, self.hid, &.{ 0, 0, 0 }, 3, &.{ 1, @intCast(len - self.hid_base), sh[2] }, 3, &.{ 1, 1, 1 }, 3, s));
                _ = mlx.mlx_array_free(self.hid);
                self.hid = cut;
                self.hid_len = len - self.hid_base;
            }
        }
        if (len + 1 < self.tokEnd()) {
            if (len + 1 <= self.tok_base) {
                self.toks.clearRetainingCapacity();
                self.tok_base = len + 1;
            } else self.toks.shrinkRetainingCapacity(len + 1 - self.tok_base);
        }
        for (&self.rows, &self.committed_end, 0..) |*r, *ce, k| {
            ce.* = @min(ce.*, len -| k);
            try r.truncate(s, ce.*);
        }
        self.clearDrafts();
    }

    /// Record committed hiddens for positions `[p0, p0 + n)` and tokens x_{p0+1..p0+n}.
    fn record(self: *State, s: mlx.mlx_stream, p0: usize, hidden: mlx.mlx_array, ids: []const u32) !void {
        try self.truncate(s, p0);
        if (self.hid_len == 0) self.hid_base = p0;
        if (p0 != self.step()) return error.MimoMtpRowGap;
        if (self.hid.ctx == null) {
            self.hid = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&self.hid, hidden));
        } else {
            const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ self.hid, hidden }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
            _ = mlx.mlx_array_free(self.hid);
            self.hid = out;
        }
        self.hid_len += ids.len;
        if (self.toks.items.len == 0) self.tok_base = p0 + 1;
        if (self.tokEnd() != p0 + 1) return error.MimoMtpRowGap;
        try self.toks.appendSlice(self.allocator, ids);
        if (self.hid_len > RING_ROWS) {
            const drop = self.hid_len - RING_ROWS;
            var cut = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(cut);
            const sh = mlx.getShape(self.hid);
            try mlx.check(mlx.mlx_slice(&cut, self.hid, &.{ 0, @intCast(drop), 0 }, 3, &.{ 1, sh[1], sh[2] }, 3, &.{ 1, 1, 1 }, 3, s));
            // A slice pins the rows it views from: past a prefill chunk, the whole chunk's
            // hiddens instead of the ring. Copied and evaluated, it lets go.
            if (drop > RING_ROWS) {
                const owned = try transformer_mod.materializedOwnedCopy(s, cut);
                _ = mlx.mlx_array_free(cut);
                cut = owned;
                const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{cut}, 1);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_async_eval(vec));
            }
            _ = mlx.mlx_array_free(self.hid);
            self.hid = cut;
            self.hid_base += drop;
            self.hid_len = RING_ROWS;
        }
        if (self.toks.items.len > RING_ROWS + MAX_HEADS + 1) {
            const drop = self.toks.items.len - (RING_ROWS + MAX_HEADS + 1);
            std.mem.copyForwards(u32, self.toks.items[0 .. self.toks.items.len - drop], self.toks.items[drop..]);
            self.toks.shrinkRetainingCapacity(self.toks.items.len - drop);
            self.tok_base += drop;
        }
    }

    fn hiddenRows(self: *const State, s: mlx.mlx_stream, from: usize, to: usize) !mlx.mlx_array {
        if (from < self.hid_base or to > self.step()) return error.MimoMtpHiddenOutOfRing;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        const sh = mlx.getShape(self.hid);
        try mlx.check(mlx.mlx_slice(&out, self.hid, &.{ 0, @intCast(from - self.hid_base), 0 }, 3, &.{ 1, @intCast(to - self.hid_base), sh[2] }, 3, &.{ 1, 1, 1 }, 3, s));
        return out;
    }

    fn token(self: *const State, p: usize) !u32 {
        if (p < self.tok_base or p >= self.tokEnd()) return error.MimoMtpTokenOutOfRing;
        return self.toks.items[p - self.tok_base];
    }

    pub fn appendEvalArrays(self: *const State, vec: mlx.mlx_vector_array) void {
        for (&self.rows) |*r| {
            if (r.k.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, r.k);
            if (r.v.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, r.v);
        }
        if (self.hid.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, self.hid);
    }
};

/// The loaded heads; per-request state lives in `State`.
pub const Head = struct {
    s: mlx.mlx_stream,
    heads: usize,
    layers: [MAX_HEADS]Layer = undefined,
    /// Every array this head owns.
    owned: std.ArrayList(mlx.mlx_array) = .empty,
    allocator: std.mem.Allocator,
    hidden: c_int,
    n_heads: c_int,
    n_kv: c_int,
    head_dim: c_int,
    v_dim: c_int,
    rope_dims: c_int,
    rope_base: f32,
    value_scale: f32,
    eps: f32,
    window: usize,
    rerank_coarse: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,

    /// The heads present in `weights` (`model.mtp.layers.{0..}`), in the sliding
    /// layers' geometry. Null when the pack carries none.
    pub fn load(allocator: std.mem.Allocator, s: mlx.mlx_stream, config: *const ModelConfig, weights: *const Weights) !?Head {
        const swa = slidingLayer(config) orelse return error.MimoMtpNeedsSlidingGeometry;
        var head = Head{
            .s = s,
            .heads = 0,
            .allocator = allocator,
            .hidden = @intCast(config.hidden_size),
            .n_heads = @intCast(config.layerNumHeads(swa)),
            .n_kv = @intCast(config.layerKVHeads(swa)),
            .head_dim = @intCast(config.layerHeadDim(swa)),
            .v_dim = @intCast(config.layerVHeadDim(swa)),
            .rope_dims = @intCast(config.layerRopeDims(swa)),
            .rope_base = config.rope_local_base_freq,
            .value_scale = config.attention_value_scale,
            .eps = config.rms_norm_eps,
            .window = config.sliding_window,
        };
        errdefer head.deinit();
        var name: [128]u8 = undefined;
        while (head.heads < MAX_HEADS) : (head.heads += 1) {
            const p = try std.fmt.bufPrint(&name, "model.mtp.layers.{d}.eh_proj.weight", .{head.heads});
            if (weights.get(p) == null) break;
            head.layers[head.heads] = try head.loadLayer(weights, head.heads);
        }
        if (head.heads == 0) {
            head.deinit();
            return null;
        }
        return head;
    }

    fn slidingLayer(config: *const ModelConfig) ?u32 {
        for (0..config.num_hidden_layers) |i| {
            if (!config.isGlobalLayer(@intCast(i))) return @intCast(i);
        }
        return null;
    }

    fn get(self: *Head, weights: *const Weights, comptime fmt: []const u8, k: usize, required: bool) !mlx.mlx_array {
        var name: [128]u8 = undefined;
        const key = try std.fmt.bufPrint(&name, "model.mtp.layers.{d}." ++ fmt, .{k});
        const src = weights.get(key) orelse {
            if (!required) return .{ .ctx = null };
            @import("log.zig").err("[mimo-mtp] missing {s}\n", .{key});
            return error.MissingWeight;
        };
        var arr = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_array_set(&arr, src));
        try self.owned.append(self.allocator, arr);
        return arr;
    }

    /// A pack's projection: an affine triple, or bf16 `[out, in]` kept transposed
    /// for a plain matmul (`Transformer.qmatmul` reads either).
    fn linear(self: *Head, weights: *const Weights, comptime base: []const u8, k: usize) !QLinear {
        var w = try self.get(weights, base ++ ".weight", k, true);
        const sc = try self.get(weights, base ++ ".scales", k, false);
        const bi = try self.get(weights, base ++ ".biases", k, false);
        if (sc.ctx == null) {
            var t = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose(&t, w, self.s));
            var c = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_contiguous(&c, t, false, self.s));
            _ = mlx.mlx_array_free(t);
            try self.owned.append(self.allocator, c);
            w = c;
        }
        return .{ .w = w, .s = sc, .b = bi };
    }

    fn loadLayer(self: *Head, weights: *const Weights, k: usize) !Layer {
        return .{
            .enorm = try self.get(weights, "enorm.weight", k, true),
            .hnorm = try self.get(weights, "hnorm.weight", k, true),
            .input_norm = try self.get(weights, "input_layernorm.weight", k, true),
            .pre_mlp_norm = try self.get(weights, "pre_mlp_layernorm.weight", k, true),
            .final_norm = try self.get(weights, "final_layernorm.weight", k, true),
            .sinks = try self.get(weights, "self_attn.attention_sink_bias", k, false),
            .eh_proj = try self.linear(weights, "eh_proj", k),
            .q = try self.linear(weights, "self_attn.q_proj", k),
            .k = try self.linear(weights, "self_attn.k_proj", k),
            .v = try self.linear(weights, "self_attn.v_proj", k),
            .o = try self.linear(weights, "self_attn.o_proj", k),
            .gate = try self.linear(weights, "mlp.gate_proj", k),
            .up = try self.linear(weights, "mlp.up_proj", k),
            .down = try self.linear(weights, "mlp.down_proj", k),
        };
    }

    pub fn deinit(self: *Head) void {
        if (self.rerank_coarse) |*rc| rc.deinit();
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.allocator);
    }

    pub fn newState(self: *const Head, allocator: std.mem.Allocator) !*State {
        _ = self;
        const st = try allocator.create(State);
        st.* = .{ .allocator = allocator };
        return st;
    }

    fn rmsNorm(self: *const Head, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, self.eps, self.s));
        return out;
    }

    /// Head `li` over rows `[pos0, pos0 + R)`: `ids` `[1, R]` int32 (row p's token
    /// x_{p+li+1}), `hid` `[1, R, H]` the target hiddens. Appends the rows' keys and
    /// values and returns the final-normed `[1, R, H]` rows.
    fn layerForward(self: *const Head, target: *Transformer, li: usize, cache: *RowCache, ids: mlx.mlx_array, hid: mlx.mlx_array, pos0: usize) !mlx.mlx_array {
        const s = self.s;
        const lw = &self.layers[li];
        const rows: c_int = mlx.getShape(ids)[1];
        const emb = try target.embedding(ids);
        defer _ = mlx.mlx_array_free(emb);
        const en = try self.rmsNorm(emb, lw.enorm);
        defer _ = mlx.mlx_array_free(en);
        var hid_cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hid_cast);
        try mlx.check(mlx.mlx_astype(&hid_cast, hid, mlx.mlx_array_dtype(emb), s));
        const hn = try self.rmsNorm(hid_cast, lw.hnorm);
        defer _ = mlx.mlx_array_free(hn);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        {
            const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ en, hn }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
        }
        const x = try target.qmatmul(cat, lw.eh_proj.w, lw.eh_proj.s, lw.eh_proj.b);
        defer _ = mlx.mlx_array_free(x);

        const a_in = try self.rmsNorm(x, lw.input_norm);
        defer _ = mlx.mlx_array_free(a_in);
        const attn = try self.attention(target, lw, cache, a_in, rows, pos0);
        defer _ = mlx.mlx_array_free(attn);
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_add(&x2, x, attn, s));

        const m_in = try self.rmsNorm(x2, lw.pre_mlp_norm);
        defer _ = mlx.mlx_array_free(m_in);
        const g = try target.qmatmul(m_in, lw.gate.w, lw.gate.s, lw.gate.b);
        defer _ = mlx.mlx_array_free(g);
        const u = try target.qmatmul(m_in, lw.up.w, lw.up.s, lw.up.b);
        defer _ = mlx.mlx_array_free(u);
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_multiply(&act, g, sig, s));
        var gu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gu);
        try mlx.check(mlx.mlx_multiply(&gu, act, u, s));
        const mlp = try target.qmatmul(gu, lw.down.w, lw.down.s, lw.down.b);
        defer _ = mlx.mlx_array_free(mlp);
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_add(&x3, x2, mlp, s));
        return self.rmsNorm(x3, lw.final_norm);
    }

    fn attention(self: *const Head, target: *Transformer, lw: *const Layer, cache: *RowCache, x: mlx.mlx_array, rows: c_int, pos0: usize) !mlx.mlx_array {
        const s = self.s;
        var proj: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
        defer for (proj) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        for ([_]QLinear{ lw.q, lw.k, lw.v }, &proj) |l, *p| p.* = try target.qmatmul(x, l.w, l.s, l.b);
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const base = mlx.mlx_optional_float{ .value = self.rope_base, .has_value = true };
        const offset: c_int = @intCast(pos0);
        const q = try headsFirst(s, proj[0], rows, self.n_heads, self.head_dim, &perm);
        defer _ = mlx.mlx_array_free(q);
        var q_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_rope);
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q, self.rope_dims, false, base, 1.0, offset, .{ .ctx = null }, s));
        const k = try headsFirst(s, proj[1], rows, self.n_kv, self.head_dim, &perm);
        defer _ = mlx.mlx_array_free(k);
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k, self.rope_dims, false, base, 1.0, offset, .{ .ctx = null }, s));
        const v = try headsFirst(s, proj[2], rows, self.n_kv, self.v_dim, &perm);
        defer _ = mlx.mlx_array_free(v);
        var v_scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_scaled);
        {
            const scale = try transformer_mod.scalarOf(self.value_scale, mlx.mlx_array_dtype(v), s);
            defer _ = mlx.mlx_array_free(scale);
            try mlx.check(mlx.mlx_multiply(&v_scaled, v, scale, s));
        }
        try cache.append(s, k_rope, v_scaled, pos0, self.window);

        // Query i (position pos0 + i) sees keys in (its position - window, its position].
        const keys: c_int = @intCast(cache.len);
        const first_key: i64 = @intCast(cache.base);
        const mask_needed = rows > 1 or keys > @as(c_int, @intCast(self.window));
        var mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask);
        if (mask_needed) {
            const n: usize = @intCast(rows * keys);
            const host = try self.allocator.alloc(bool, n);
            defer self.allocator.free(host);
            for (0..@intCast(rows)) |i| {
                const qp: i64 = @as(i64, @intCast(pos0 + i));
                for (0..@intCast(keys)) |j| {
                    const kp = first_key + @as(i64, @intCast(j));
                    host[i * @as(usize, @intCast(keys)) + j] = kp <= qp and kp > qp - @as(i64, @intCast(self.window));
                }
            }
            _ = mlx.mlx_array_free(mask);
            mask = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ 1, 1, rows, keys }, 4, .bool_);
        }
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, cache.k, cache.v, attn_scale, if (mask_needed) "array" else "", mask, lw.sinks, false, s));
        var out_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out_t);
        try mlx.check(mlx.mlx_transpose_axes(&out_t, out, &perm, 4, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, out_t, &[_]c_int{ 1, rows, self.n_heads * self.v_dim }, 3, s));
        return target.qmatmul(flat, lw.o.w, lw.o.s, lw.o.b);
    }

    fn headsFirst(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, heads: c_int, dim: c_int, perm: *const [4]c_int) !mlx.mlx_array {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, x, &[_]c_int{ 1, rows, heads, dim }, 4, s));
        var t = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, r, perm, 4, s));
        return t;
    }

    /// Head `li`'s rows from wherever its rows end through `q`, the last `drafted`
    /// of them carrying this round's drafts; the rest carry committed tokens.
    /// Returns every forwarded row, final-normed.
    fn forwardRows(self: *const Head, target: *Transformer, st: *State, li: usize, q: usize, drafted: usize) !mlx.mlx_array {
        const s = self.s;
        const cache = &st.rows[li];
        // Rows before the last query's window never reach a later query either.
        const floor = (q + 1) -| self.window;
        const start = @max(@max(cache.end(), floor), st.hid_base);
        if (cache.end() < start) cache.reset(start);
        if (start > q) return error.MimoMtpNothingToForward;
        const rows = q + 1 - start;
        const tentative = @min(drafted, rows);
        const host = try self.allocator.alloc(i32, rows - tentative);
        defer self.allocator.free(host);
        for (host, 0..) |*t, i| t.* = @intCast(try st.token(start + i + li + 1));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        {
            var parts: [MAX_HEADS + 1]mlx.mlx_array = undefined;
            var n: usize = 0;
            const committed = mlx.mlx_array_new_data(host.ptr, &[_]c_int{@intCast(host.len)}, 1, .int32);
            defer _ = mlx.mlx_array_free(committed);
            if (host.len > 0) {
                parts[n] = committed;
                n += 1;
            }
            // Row p's token x_{p+li+1} is draft d_{p+li-q}.
            for (st.drafts[drafted - tentative .. drafted]) |d| {
                parts[n] = d;
                n += 1;
            }
            const vec = mlx.mlx_vector_array_new_data(&parts, n);
            defer _ = mlx.mlx_vector_array_free(vec);
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
            var flat32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat32);
            try mlx.check(mlx.mlx_astype(&flat32, flat, .int32, s));
            try mlx.check(mlx.mlx_reshape(&ids, flat32, &[_]c_int{ 1, @intCast(rows) }, 2, s));
        }
        const hid = try st.hiddenRows(s, start, q + 1);
        defer _ = mlx.mlx_array_free(hid);
        const out = try self.layerForward(target, li, cache, ids, hid, start);
        st.committed_end[li] = q + 1 - tentative;
        return out;
    }

    /// Committed history `(hidden_p, x_{p+1})` for p in `[p0, p0 + n)`: every head
    /// appends the rows whose own tokens are committed.
    pub fn appendHistory(self: *const Head, target: *Transformer, st: *State, token_ids: []const u32, hidden: mlx.mlx_array, p0: usize) !void {
        try st.record(self.s, p0, hidden, token_ids);
        const end = st.step();
        for (0..self.heads) |li| {
            // Head li's row p needs x_{p+li+1}, committed through x_end.
            if (end <= li or end - li <= st.rows[li].end()) continue;
            const out = try self.forwardRows(target, st, li, end - li - 1, 0);
            _ = mlx.mlx_array_free(out);
        }
    }

    fn lastRow(self: *const Head, rows: mlx.mlx_array) !mlx.mlx_array {
        defer _ = mlx.mlx_array_free(rows);
        var last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(last);
        const sh = mlx.getShape(rows);
        try mlx.check(mlx.mlx_slice(&last, rows, &.{ 0, sh[1] - 1, 0 }, 3, &.{ 1, sh[1], sh[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
        return last;
    }

    /// Draft step `i` of a round. Step 0 records the committed rows the round
    /// opens with (`token_ids` x_{p0+1..}, `hidden` h_{p0..}) and runs head 0 at
    /// their last position q; step i >= 1 (`hidden` null) runs head i at q on the
    /// drafts so far plus `draft` (d_i). Returns the lm_head's input for the next
    /// draft, `[1, 1, H]`.
    pub fn draftStep(self: *const Head, target: *Transformer, st: *State, step_i: usize, draft: ?mlx.mlx_array, token_ids: []const u32, hidden: ?mlx.mlx_array, p0: usize) !mlx.mlx_array {
        if (step_i == 0) {
            try st.record(self.s, p0, hidden orelse return error.MimoMtpStepZeroNeedsHidden, token_ids);
            st.round_q = st.step() - 1;
            return self.lastRow(try self.forwardRows(target, st, 0, st.round_q, 0));
        }
        // Past the last trained head the last one proposes again: a wasted row, never a wrong output.
        const li = @min(step_i, self.heads - 1);
        if (step_i < self.heads) {
            if (st.n_drafts != step_i - 1) return error.MimoMtpStepOutOfOrder;
            var d = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(d);
            try mlx.check(mlx.mlx_reshape(&d, draft orelse return error.MimoMtpStepNeedsDraft, &[_]c_int{1}, 1, self.s));
            st.drafts[st.n_drafts] = d;
            st.n_drafts += 1;
        }
        try st.rows[li].truncate(self.s, st.committed_end[li]);
        return self.lastRow(try self.forwardRows(target, st, li, st.round_q, @min(st.n_drafts, li)));
    }

    /// Drafts shortlist on a coarse low-bit copy of the trunk's lm_head (a 152k-row
    /// readout per draft step would cost more than the head layer itself).
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

// ── Tests ──

const testing = std.testing;

fn testReadF32(alloc: std.mem.Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const out = try alloc.alloc(f32, n);
    const src = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    @memcpy(out, src[0..n]);
    return out;
}

/// Rows `[from, to)` of a `[T, D]` fixture tensor as `[1, to - from, D]`.
fn testRows(fx: *const Weights, key: []const u8, from: usize, to: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const arr = fx.get(key) orelse return error.MissingFixtureTensor;
    const sh = mlx.getShape(arr);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &.{ @intCast(from), 0 }, 2, &.{ @intCast(to), sh[1] }, 2, &.{ 1, 1 }, 2, s));
    var r3 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r3);
    try mlx.check(mlx.mlx_reshape(&r3, out, &.{ 1, @intCast(to - from), sh[1] }, 3, s));
    return r3;
}

fn expectRowClose(label: []const u8, k: usize, q: usize, ours: mlx.mlx_array, reference: mlx.mlx_array, bar: f32, s: mlx.mlx_stream) !void {
    const a = try testReadF32(testing.allocator, ours, s);
    defer testing.allocator.free(a);
    const b = try testReadF32(testing.allocator, reference, s);
    defer testing.allocator.free(b);
    try testing.expectEqual(b.len, a.len);
    var worst: f32 = 0;
    var peak: f32 = 0;
    for (a, b) |x, y| {
        try testing.expect(std.math.isFinite(x));
        worst = @max(worst, @abs(x - y));
        peak = @max(peak, @abs(y));
    }
    // A bf16 pack against the f32 rendering: error relative to the row's peak.
    if (worst > bar * peak) {
        std.debug.print("mimo mtp {s}: head {d} row {d} max |d| {d} > {d} of peak {d}\n", .{ label, k, q, worst, bar, peak });
        return error.TestExpectedEqual;
    }
}

test "mimo mtp state keeps its hidden ring, not the prefill chunk it was cut from" {
    // The state holds RING_ROWS hiddens, never the prefill chunk a slice of them views.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const width: c_int = 1024;
    const rows: usize = 4096;
    const ids = try testing.allocator.alloc(u32, rows);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i);
    var st = State{ .allocator = testing.allocator };
    defer st.deinit();
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    {
        var hidden = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hidden);
        try mlx.check(mlx.mlx_ones(&hidden, &.{ 1, @intCast(rows), width }, 3, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(hidden));
        try st.record(s, 0, hidden, ids);
    }
    try mlx.check(mlx.mlx_array_eval(st.hid));
    try mlx.check(mlx.mlx_synchronize(s));
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    const ring: usize = RING_ROWS * @as(usize, @intCast(width)) * 2;
    std.testing.expect(after -| before <= ring + ring / 4) catch |err| {
        std.debug.print("mimo mtp hidden ring holds {d} B, ring is {d} B\n", .{ after -| before, ring });
        return err;
    };
}

test "mimo mtp heads track the torch rendering of the MiMo-V2 MTP layer across rounds, drafts and rollbacks (MIMO_V2_MODEL + MIMO_V2_MTP_FIXTURE)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("MIMO_V2_MTP_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| a.free(p);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var head = (try Head.load(a, s, &config, &weights)) orelse return error.NoMtpHeads;
    defer head.deinit();
    try testing.expectEqual(@as(usize, 3), head.heads);

    var fx = try model_mod.loadWeightsSingleFile(a, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const ids_src = mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable;
    const t_total = mlx.mlx_array_size(ids_arr);
    const x = try a.alloc(u32, t_total);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = @intCast(ids_src[i]);

    const st = try head.newState(a);
    defer {
        st.deinit();
        a.destroy(st);
    }
    // Prefill history in uneven chunks past the window (8): rows p in [0, 30) with x_{p+1}.
    var p: usize = 0;
    for ([_]usize{ 13, 11, 6 }) |n| {
        const hid = try testRows(&fx, "target_hidden", p, p + n, s);
        defer _ = mlx.mlx_array_free(hid);
        try head.appendHistory(&xfm, st, x[p + 1 .. p + n + 1], hid, p);
        p += n;
    }

    // Rounds as the generator runs them: each opens by truncating to the previous
    // round's q and recording the committed rows from there to its own q.
    const plan = [_]struct { wrong: bool, accept: usize }{
        .{ .wrong = false, .accept = 3 }, .{ .wrong = true, .accept = 0 }, .{ .wrong = false, .accept = 1 },
        .{ .wrong = false, .accept = 2 }, .{ .wrong = true, .accept = 0 }, .{ .wrong = false, .accept = 3 },
    };
    var stash_from: usize = p;
    var q = p;
    var rounds: usize = 0;
    for (plan) |round| {
        if (q + 4 >= t_total) break;
        try st.truncate(s, stash_from);
        const hid = try testRows(&fx, "target_hidden", stash_from, q + 1, s);
        defer _ = mlx.mlx_array_free(hid);
        const out0 = try head.draftStep(&xfm, st, 0, null, x[stash_from + 1 .. q + 2], hid, stash_from);
        defer _ = mlx.mlx_array_free(out0);
        {
            const ref = try testRows(&fx, "mtp0_out", q, q + 1, s);
            defer _ = mlx.mlx_array_free(ref);
            try expectRowClose("out", 0, q, out0, ref, 0.02, s);
            const logits = try xfm.lmHeadForDraft(out0);
            defer _ = mlx.mlx_array_free(logits);
            const ref_l = try testRows(&fx, "mtp0_logits", q, q + 1, s);
            defer _ = mlx.mlx_array_free(ref_l);
            try expectRowClose("logits", 0, q, logits, ref_l, 0.02, s);
        }
        for (1..3) |i| {
            const truth = x[q + 1 + i];
            const tok: i32 = @intCast(if (round.wrong) (truth + 7) % @as(u32, @intCast(config.vocab_size)) else truth);
            const d = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .int32);
            defer _ = mlx.mlx_array_free(d);
            const out = try head.draftStep(&xfm, st, i, d, &.{}, null, 0);
            defer _ = mlx.mlx_array_free(out);
            if (round.wrong) continue;
            var key: [16]u8 = undefined;
            const ref = try testRows(&fx, try std.fmt.bufPrint(&key, "mtp{d}_out", .{i}), q, q + 1, s);
            defer _ = mlx.mlx_array_free(ref);
            try expectRowClose("out", i, q, out, ref, 0.02, s);
        }
        stash_from = q;
        q += round.accept + 1;
        rounds += 1;
    }
    try testing.expect(rounds >= 5);
}
