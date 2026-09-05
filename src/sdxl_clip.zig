//! SDXL's CLIP text towers — ONE implementation over `sdxl.ClipTextConfig`,
//! serving both CLIP-L (768/12/12, quick_gelu) and OpenCLIP bigG
//! (1280/32/20, gelu).
//!
//! The two towers are the same architecture at different sizes, and the weight
//! names are byte-identical between them (verified against
//! `stabilityai/stable-diffusion-xl-base-1.0`), so forking this into two files
//! would be two places to get the same trap wrong.
//!
//! What SDXL takes from each tower, and why each choice is load-bearing:
//!
//!   - The PENULTIMATE hidden state (`hidden_states[-2]`), not the last, and
//!     NOT after `final_layer_norm`. Using the last is a silent quality
//!     regression: the embeddings are still well-formed, the images are just
//!     worse. `encode` returns the penultimate; the final norm is loaded
//!     because the POOLED path needs it.
//!   - The POOLED vector from bigG ALONE: the hidden state at the EOS token
//!     position, after `final_layer_norm`, projected by `text_projection`.
//!     CLIP-L has no `text_projection` tensor at all, which is the structural
//!     reason the choice is not arbitrary.
//!   - A CAUSAL mask. CLIP's text tower is autoregressively masked even though
//!     it is used as an encoder; running it bidirectional produces plausible
//!     embeddings and subtly wrong ones.
//!
//! ORACLE STATUS: shapes and weight binding are verified against the real
//! checkpoint (`sdxl checkpoint` test in sdxl.zig, and `sdxl clip forward`
//! below). NUMERICAL parity against transformers' CLIPTextModel is NOT
//! established — that needs a dumped fixture.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const sdxl = @import("sdxl.zig");

const nn = @import("sdxl_nn.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

// ── Small helpers (same shape as flux.zig's, kept local so this file stands
// alone as the port's second component) ──

inline fn matmul(x: mlx.mlx_array, w_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    return o;
}
inline fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}
inline fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}
inline fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}
inline fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    // CLIP's LayerNorm eps is 1e-5 in both towers' configs.
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, 1e-5, s));
    return o;
}

/// CLIP's activations. `quick_gelu` is `x * sigmoid(1.702x)` — OpenAI's
/// original approximation, NOT interchangeable with the erf form even though
/// both are "GELU". The two SDXL towers use one each.
fn activate(x: mlx.mlx_array, kind: sdxl.ClipActivation, s: S) !mlx.mlx_array {
    switch (kind) {
        .quick_gelu => {
            const c = mlx.mlx_array_new_float(1.702);
            defer _ = mlx.mlx_array_free(c);
            const scaled = try mulA(x, c, s);
            defer _ = mlx.mlx_array_free(scaled);
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, scaled, s));
            return mulA(x, sig, s);
        },
        .gelu => {
            // 0.5 * x * (1 + erf(x / sqrt(2)))
            const inv_sqrt2 = mlx.mlx_array_new_float(0.7071067811865476);
            defer _ = mlx.mlx_array_free(inv_sqrt2);
            const scaled = try mulA(x, inv_sqrt2, s);
            defer _ = mlx.mlx_array_free(scaled);
            var e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_erf(&e, scaled, s));
            const one = mlx.mlx_array_new_float(1.0);
            defer _ = mlx.mlx_array_free(one);
            const one_plus = try addA(e, one, s);
            defer _ = mlx.mlx_array_free(one_plus);
            const half = mlx.mlx_array_new_float(0.5);
            defer _ = mlx.mlx_array_free(half);
            const half_x = try mulA(x, half, s);
            defer _ = mlx.mlx_array_free(half_x);
            return mulA(half_x, one_plus, s);
        },
    }
}

/// A linear with a bias — every CLIP projection has one, unlike the flow
/// backends where bias-less linears are the norm.
const Linear = struct {
    /// DENSE: pre-transposed `[in, out]`. QUANTIZED: packed `[out, in·bits/32]`
    /// as stored (forward uses `mlx_quantized_matmul(transpose_w=true)`). The
    /// q4/q8 packs quantize every CLIP encoder linear (q/k/v/out_proj, fc1/fc2).
    w_t: mlx.mlx_array,
    b: mlx.mlx_array,
    scales: ?mlx.mlx_array = null,
    q_biases: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    fn deinit(self: *Linear) void {
        _ = mlx.mlx_array_free(self.w_t);
        _ = mlx.mlx_array_free(self.b);
        if (self.scales) |x| _ = mlx.mlx_array_free(x);
        if (self.q_biases) |x| _ = mlx.mlx_array_free(x);
    }

    fn forward(self: *const Linear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const y = if (self.scales) |sc| blk: {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w_t, sc, self.q_biases.?, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
            break :blk o;
        } else try matmul(x, self.w_t, s);
        errdefer _ = mlx.mlx_array_free(y);
        const out = try addA(y, self.b, s);
        _ = mlx.mlx_array_free(y);
        return out;
    }
};

const Layer = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    fc1: Linear,
    fc2: Linear,

    fn deinit(self: *Layer) void {
        _ = mlx.mlx_array_free(self.ln1_w);
        _ = mlx.mlx_array_free(self.ln1_b);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.ln2_w);
        _ = mlx.mlx_array_free(self.ln2_b);
        self.fc1.deinit();
        self.fc2.deinit();
    }
};

pub const TextTower = struct {
    cfg: sdxl.ClipTextConfig,
    allocator: std.mem.Allocator,
    s: S,
    token_embed: mlx.mlx_array, // [vocab, hidden]
    pos_embed: mlx.mlx_array, // [max_positions, hidden]
    layers: []Layer,
    final_ln_w: mlx.mlx_array,
    final_ln_b: mlx.mlx_array,
    /// bigG only; null on CLIP-L, which ships no such tensor.
    text_projection: ?mlx.mlx_array,

    pub fn deinit(self: *TextTower) void {
        _ = mlx.mlx_array_free(self.token_embed);
        _ = mlx.mlx_array_free(self.pos_embed);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.final_ln_w);
        _ = mlx.mlx_array_free(self.final_ln_b);
        if (self.text_projection) |p| _ = mlx.mlx_array_free(p);
    }

    /// What SDXL consumes from one tower.
    pub const Encoded = struct {
        /// `[1, seq, hidden]` — the PENULTIMATE hidden state, pre-final-norm.
        penultimate: mlx.mlx_array,
        /// `[1, projection_dim]` — bigG only, null on CLIP-L.
        pooled: ?mlx.mlx_array,

        pub fn deinit(self: *Encoded) void {
            _ = mlx.mlx_array_free(self.penultimate);
            if (self.pooled) |p| _ = mlx.mlx_array_free(p);
        }
    };

    /// Run the tower over `ids` (length <= max_positions).
    ///
    /// `eos_index` is where the pooled vector is read from. CLIP pools at the
    /// EOS token's POSITION, not at the last position of the padded window —
    /// with a padded prompt those differ, and pooling at the end reads padding.
    ///
    /// `final_norm` picks which hidden state `Encoded.penultimate` returns —
    /// SDXL's convention (`false`: `hidden_states[-2]`, penultimate, never
    /// final-normed) or SD 1.x's default (`true`: `hidden_states[-1]` AFTER
    /// `final_layer_norm` — `CLIPTextModel(ids)[0]`, diffusers'
    /// `StableDiffusionPipeline` reads with no `clip_skip`). The POOLED
    /// vector (bigG only) always reads the final-normed state regardless —
    /// that convention is SDXL-specific either way.
    pub fn encode(self: *TextTower, ids: []const i32, eos_index: usize, final_norm: bool) !Encoded {
        const s = self.s;
        const seq: c_int = @intCast(ids.len);
        const hidden: c_int = @intCast(self.cfg.hidden);
        const heads: c_int = @intCast(self.cfg.heads);
        const head_dim: c_int = @intCast(self.cfg.headDim());

        // ── Embeddings: token table gather + learned absolute positions.
        const id_shape = [_]c_int{seq};
        const ids_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(ids_arr);

        var tok = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tok);
        // AXIS 0, never the axis-less `mlx_take` — that flattens the table and
        // returns [seq] instead of [seq, hidden]. Same class as the mlx_topk
        // axis bug in the root rules: correct-looking, silently one-dimensional.
        try mlx.check(mlx.mlx_take_axis(&tok, self.token_embed, ids_arr, 0, s));

        // Positions 0..seq — a slice of the learned table, never a formula.
        var pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos);
        {
            const start = [_]c_int{ 0, 0 };
            const stop = [_]c_int{ seq, hidden };
            const stride = [_]c_int{ 1, 1 };
            try mlx.check(mlx.mlx_slice(&pos, self.pos_embed, &start, 2, &stop, 2, &stride, 2, s));
        }

        var h = try addA(tok, pos, s);
        errdefer _ = mlx.mlx_array_free(h);
        {
            const shape = [_]c_int{ 1, seq, hidden };
            const r = try reshape(h, &shape, s);
            _ = mlx.mlx_array_free(h);
            h = r;
        }

        // ── Layers. The penultimate output is what SDXL wants, so it is
        // captured one layer before the end rather than recomputed.
        var penultimate: ?mlx.mlx_array = null;
        errdefer if (penultimate) |p| {
            _ = mlx.mlx_array_free(p);
        };

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.headDim())));
        for (self.layers, 0..) |*layer, i| {
            if (!final_norm and i + 1 == self.layers.len) {
                var captured = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_copy(&captured, h, s));
                penultimate = captured;
            }

            // Self-attention, pre-norm.
            const normed = try layerNorm(h, layer.ln1_w, layer.ln1_b, s);
            defer _ = mlx.mlx_array_free(normed);

            const q = try layer.q.forward(normed, s);
            defer _ = mlx.mlx_array_free(q);
            const k = try layer.k.forward(normed, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try layer.v.forward(normed, s);
            defer _ = mlx.mlx_array_free(v);

            // [1, seq, heads*hd] → [1, heads, seq, hd]
            const hs = [_]c_int{ 1, seq, heads, head_dim };
            const perm = [_]c_int{ 0, 2, 1, 3 };
            const qh = blk: {
                const r = try reshape(q, &hs, s);
                defer _ = mlx.mlx_array_free(r);
                break :blk try transpose(r, &perm, s);
            };
            defer _ = mlx.mlx_array_free(qh);
            const kh = blk: {
                const r = try reshape(k, &hs, s);
                defer _ = mlx.mlx_array_free(r);
                break :blk try transpose(r, &perm, s);
            };
            defer _ = mlx.mlx_array_free(kh);
            const vh = blk: {
                const r = try reshape(v, &hs, s);
                defer _ = mlx.mlx_array_free(r);
                break :blk try transpose(r, &perm, s);
            };
            defer _ = mlx.mlx_array_free(vh);

            // CAUSAL — CLIP's text tower is autoregressively masked even
            // though SDXL uses it as an encoder.
            var attn = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
                &attn,
                qh,
                kh,
                vh,
                scale,
                "causal",
                mlx.mlx_array_new(),
                mlx.mlx_array_new(),
                false,
                s,
            ));

            // back to [1, seq, hidden]
            const merged = blk: {
                const t = try transpose(attn, &perm, s);
                defer _ = mlx.mlx_array_free(t);
                const shape = [_]c_int{ 1, seq, hidden };
                break :blk try reshape(t, &shape, s);
            };
            defer _ = mlx.mlx_array_free(merged);

            const proj = try layer.o.forward(merged, s);
            defer _ = mlx.mlx_array_free(proj);
            {
                const r = try addA(h, proj, s);
                _ = mlx.mlx_array_free(h);
                h = r;
            }

            // MLP, pre-norm.
            const normed2 = try layerNorm(h, layer.ln2_w, layer.ln2_b, s);
            defer _ = mlx.mlx_array_free(normed2);
            const up = try layer.fc1.forward(normed2, s);
            defer _ = mlx.mlx_array_free(up);
            const act = try activate(up, self.cfg.activation, s);
            defer _ = mlx.mlx_array_free(act);
            const down = try layer.fc2.forward(act, s);
            defer _ = mlx.mlx_array_free(down);
            {
                const r = try addA(h, down, s);
                _ = mlx.mlx_array_free(h);
                h = r;
            }
        }

        // Final-normed state: SD 1.x's primary stream when `final_norm`, and
        // ALWAYS what bigG's pooled vector reads from (`EOS row of the
        // FINAL-NORMED state, projected` — an SDXL-specific convention
        // independent of `final_norm`).
        const final_normed = try layerNorm(h, self.final_ln_w, self.final_ln_b, s);
        errdefer _ = mlx.mlx_array_free(final_normed);
        _ = mlx.mlx_array_free(h);

        var pooled: ?mlx.mlx_array = null;
        if (self.text_projection) |proj_w| {
            const idx: c_int = @intCast(@min(eos_index, ids.len - 1));
            var eos_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eos_row);
            {
                const start = [_]c_int{ 0, idx, 0 };
                const stop = [_]c_int{ 1, idx + 1, hidden };
                const stride = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(&eos_row, final_normed, &start, 3, &stop, 3, &stride, 3, s));
            }
            const flat = blk: {
                const shape = [_]c_int{ 1, hidden };
                break :blk try reshape(eos_row, &shape, s);
            };
            defer _ = mlx.mlx_array_free(flat);
            pooled = try matmul(flat, proj_w, s);
        }

        const primary = if (final_norm) final_normed else blk: {
            _ = mlx.mlx_array_free(final_normed);
            break :blk penultimate.?;
        };
        return .{ .penultimate = primary, .pooled = pooled };
    }
};

/// Load one tower from `<model_dir>/<sub>` (`text_encoder` / `text_encoder_2`).
///
/// The weight names are identical between towers, so `cfg` is what selects the
/// geometry — it is NOT inferred from shapes here, because both towers share a
/// head_dim of 64 and a vocab of 49408, so a swapped config would load cleanly
/// and produce garbage. `sdxl.CLIP_L_CONFIG` / `CLIP_BIG_G_CONFIG` are the two
/// legitimate values.
/// The width the towers are served at. The checkpoint ships fp16 and that is
/// the default, but the dtype is a PARAMETER because it is measurable at the
/// pipeline level: the towers' ~0.3% fp16 RMS error is multiplied by the
/// guidance scale and compounded over every denoising step, so widening them
/// is how an end-to-end parity gap gets attributed rather than guessed at.
pub const DEFAULT_DTYPE: mlx.mlx_dtype = .float16;

/// `SDXL_CLIP_F32=1` widens the text towers. Shared by the pipeline and the
/// parity tests so both arms of an attribution run agree on what they mean.
pub fn towerDtype() mlx.mlx_dtype {
    return if (std.c.getenv("SDXL_CLIP_F32") != null) .float32 else DEFAULT_DTYPE;
}

pub fn loadTower(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []const u8,
    sub: []const u8,
    cfg: sdxl.ClipTextConfig,
    dtype: mlx.mlx_dtype,
) !TextTower {
    const dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, sub });
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    return loadTowerFromWeights(allocator, s, &w, sub, cfg, dtype);
}

/// The weight-binding half, split out so a single-file checkpoint drives it
/// from an in-memory `Weights` map — the diffusers CLIP keys the converter
/// produces are exactly what a `text_encoder/` folder holds. `sub` is only a
/// log label here.
pub fn loadTowerFromWeights(
    allocator: std.mem.Allocator,
    s: S,
    w: *Weights,
    sub: []const u8,
    cfg: sdxl.ClipTextConfig,
    dtype: mlx.mlx_dtype,
) !TextTower {
    var tower: TextTower = undefined;
    tower.cfg = cfg;
    tower.allocator = allocator;
    tower.s = s;

    tower.token_embed = try dup(w, "text_model.embeddings.token_embedding.weight", dtype, s);
    tower.pos_embed = try dup(w, "text_model.embeddings.position_embedding.weight", dtype, s);
    tower.final_ln_w = try dup(w, "text_model.final_layer_norm.weight", dtype, s);
    tower.final_ln_b = try dup(w, "text_model.final_layer_norm.bias", dtype, s);

    // Present in bigG, absent in CLIP-L — an optional, never a required get.
    // The pooled projection is a PLAIN matmul (`pooled @ text_projection`), so a
    // quantized `text_projection.weight` (the q4/q8 packs quantize it) is
    // DEQUANTIZED here to a dense `[proj, hidden]` before the transpose, rather
    // than routed through a quantized_matmul like the encoder linears.
    tower.text_projection = if (w.get(sdxl.CLIP_PROJECTION_TENSOR)) |proj_src| blk: {
        var cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cast);
        if (w.get("text_projection.scales")) |proj_scales| {
            const proj_biases = w.get("text_projection.biases") orelse return error.MissingWeight;
            const geo = nn.inferQuantGeometry(proj_src, proj_scales);
            const empty = mlx.mlx_array{ .ctx = null };
            var deq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq);
            try mlx.check(mlx.mlx_dequantize(&deq, proj_src, proj_scales, proj_biases, mlx.mlx_optional_int.some(@intCast(geo.group_size)), mlx.mlx_optional_int.some(@intCast(geo.bits)), "affine", empty, mlx.mlx_optional_dtype{}, s));
            try mlx.check(mlx.mlx_astype(&cast, deq, dtype, s));
        } else {
            try mlx.check(mlx.mlx_astype(&cast, proj_src, dtype, s));
        }
        var t = mlx.mlx_array_new();
        const axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&t, cast, &axes, 2, s));
        break :blk t;
    } else null;

    tower.layers = try allocator.alloc(Layer, cfg.layers);
    errdefer allocator.free(tower.layers);
    for (tower.layers, 0..) |*layer, i| {
        layer.ln1_w = try dupIdx(w, allocator, i, "layer_norm1.weight", dtype, s);
        layer.ln1_b = try dupIdx(w, allocator, i, "layer_norm1.bias", dtype, s);
        layer.ln2_w = try dupIdx(w, allocator, i, "layer_norm2.weight", dtype, s);
        layer.ln2_b = try dupIdx(w, allocator, i, "layer_norm2.bias", dtype, s);
        layer.q = try linearIdx(w, allocator, s, i, "self_attn.q_proj", dtype);
        layer.k = try linearIdx(w, allocator, s, i, "self_attn.k_proj", dtype);
        layer.v = try linearIdx(w, allocator, s, i, "self_attn.v_proj", dtype);
        layer.o = try linearIdx(w, allocator, s, i, "self_attn.out_proj", dtype);
        layer.fc1 = try linearIdx(w, allocator, s, i, "mlp.fc1", dtype);
        layer.fc2 = try linearIdx(w, allocator, s, i, "mlp.fc2", dtype);
    }

    log.info("[sdxl] loaded {s}: hidden={d} layers={d} heads={d} act={s} projection={}\n", .{
        sub, cfg.hidden, cfg.layers, cfg.heads, @tagName(cfg.activation), tower.text_projection != null,
    });
    return tower;
}

fn dup(w: *Weights, name: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const src = w.get(name) orelse {
        log.err("[sdxl] missing weight {s}\n", .{name});
        return error.MissingWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, src, dt, s));
    return o;
}

fn dupIdx(w: *Weights, a: std.mem.Allocator, i: usize, suffix: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}.{s}", .{ i, suffix });
    defer a.free(name);
    return dup(w, name, dt, s);
}

/// Weights are stored `[out, in]`; the forward wants `[in, out]`, so the
/// transpose happens ONCE at load rather than per token.
fn linearIdx(w: *Weights, a: std.mem.Allocator, s: S, i: usize, prefix: []const u8, dt: mlx.mlx_dtype) !Linear {
    const wname = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}.{s}.weight", .{ i, prefix });
    defer a.free(wname);
    const bname = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}.{s}.bias", .{ i, prefix });
    defer a.free(bname);

    const src = w.get(wname) orelse return error.MissingWeight;
    const b = try dup(w, bname, dt, s);
    errdefer _ = mlx.mlx_array_free(b);

    // Affine-quantized when a sibling `.scales` exists (q4/q8 packs). Keep the
    // packed weight AS-IS; forward routes through `mlx_quantized_matmul`.
    const sname = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}.{s}.scales", .{ i, prefix });
    defer a.free(sname);
    if (w.get(sname)) |scales_src| {
        const qbname = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}.{s}.biases", .{ i, prefix });
        defer a.free(qbname);
        const qbiases_src = w.get(qbname) orelse return error.MissingWeight;
        var packed_w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&packed_w, src, mlx.mlx_array_dtype(src), s)); // own the uint32
        errdefer _ = mlx.mlx_array_free(packed_w);
        var scales = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&scales, scales_src, dt, s));
        errdefer _ = mlx.mlx_array_free(scales);
        var qbiases = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&qbiases, qbiases_src, dt, s));
        errdefer _ = mlx.mlx_array_free(qbiases);
        const geo = nn.inferQuantGeometry(packed_w, scales);
        return .{ .w_t = packed_w, .b = b, .scales = scales, .q_biases = qbiases, .bits = geo.bits, .group_size = geo.group_size };
    }

    var cast = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&cast, src, dt, s));
    defer _ = mlx.mlx_array_free(cast);
    var w_t = mlx.mlx_array_new();
    const axes = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&w_t, cast, &axes, 2, s));
    return .{ .w_t = w_t, .b = b };
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "the activation choice is per tower, not per module" {
    // Guards the trap the checkpoint surfaced: one activation for both towers
    // produces plausible embeddings and a plausible image.
    try testing.expect(sdxl.CLIP_L_CONFIG.activation != sdxl.CLIP_BIG_G_CONFIG.activation);
}

// Live structural check. Loads BOTH real towers, runs a forward, and asserts
// the shapes SDXL will consume. Not parity — it proves the thing binds, runs
// and produces finite numbers of the right shape.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//     zig build test -Dtest-filter="sdxl clip forward"
test "sdxl clip forward: both towers load and produce the shapes SDXL consumes" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    // A short padded window; ids are arbitrary but in-vocab, and the EOS index
    // is deliberately NOT the last position so pooling-at-the-end would show up.
    const seq = 8;
    var ids: [seq]i32 = .{ 49406, 320, 1125, 539, 49407, 0, 0, 0 };
    const eos_index: usize = 4;

    const towers = [_]struct { sub: []const u8, cfg: sdxl.ClipTextConfig, pooled: bool }{
        .{ .sub = "text_encoder", .cfg = sdxl.CLIP_L_CONFIG, .pooled = false },
        .{ .sub = "text_encoder_2", .cfg = sdxl.CLIP_BIG_G_CONFIG, .pooled = true },
    };

    for (towers) |t| {
        var tower = loadTower(io, a, s, dir, t.sub, t.cfg, DEFAULT_DTYPE) catch |e| {
            log.err("[sdxl-test] {s} load failed: {}\n", .{ t.sub, e });
            return e;
        };
        defer tower.deinit();

        var enc = try tower.encode(&ids, eos_index, false);
        defer enc.deinit();
        _ = mlx.mlx_array_eval(enc.penultimate);

        // [1, seq, hidden]
        try testing.expectEqual(@as(usize, 3), mlx.mlx_array_ndim(enc.penultimate));
        const shp = mlx.mlx_array_shape(enc.penultimate);
        try testing.expectEqual(@as(c_int, 1), shp[0]);
        try testing.expectEqual(@as(c_int, seq), shp[1]);
        try testing.expectEqual(@as(c_int, @intCast(t.cfg.hidden)), shp[2]);

        // Finiteness before anything else — an all-NaN tensor passes every
        // shape assertion there is.
        var absv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(absv);
        try mlx.check(mlx.mlx_abs(&absv, enc.penultimate, s));
        var total = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(total);
        try mlx.check(mlx.mlx_sum(&total, absv, false, s));
        _ = mlx.mlx_array_eval(total);
        var mv: f32 = 0;
        _ = mlx.mlx_array_item_float32(&mv, total);
        // A NaN anywhere poisons the sum, and an all-zero output (a dead
        // forward that still has the right shape) is caught by the > 0 half.
        try testing.expect(std.math.isFinite(mv));
        try testing.expect(mv > 0);

        // Only bigG produces a pooled vector, at its projection width.
        try testing.expectEqual(t.pooled, enc.pooled != null);
        if (enc.pooled) |p| {
            _ = mlx.mlx_array_eval(p);
            const ps = mlx.mlx_array_shape(p);
            try testing.expectEqual(@as(c_int, 1), ps[0]);
            try testing.expectEqual(@as(c_int, @intCast(t.cfg.projection_dim)), ps[1]);
        }
    }
}

// Numerical PARITY against transformers' own CLIPTextModel. This is the real
// oracle: the fixture is generated by `tests/dump_sdxl_clip_fixtures.py` over
// the SAME token ids this test uses, in float32 on CPU.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//   SDXL_CLIP_FIXTURE=~/.mlx-serve/staging/sdxl_clip_fixture.safetensors \
//     zig build test -Dtest-filter="sdxl clip parity"
//
// Asserts rms_ratio BESIDE the cosine. A cosine test cannot see a scale error,
// and these embeddings are concatenated into a stream the UNet cross-attends
// to — a uniformly scaled tower would score a perfect cosine and unbalance
// everything downstream.
test "sdxl clip parity: both towers match transformers" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("SDXL_CLIP_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    // The ids come FROM the fixture, so the two sides cannot describe
    // different prompts.
    const ids_arr = fx.get("ids") orelse return error.MissingFixtureIds;
    _ = mlx.mlx_array_eval(ids_arr);
    const n = mlx.mlx_array_size(ids_arr);
    const id_ptr = mlx.mlx_array_data_int32(ids_arr).?;
    const ids = try a.alloc(i32, n);
    defer a.free(ids);
    for (ids, 0..) |*v, i| v.* = id_ptr[i];

    const eos_arr = fx.get("eos_index") orelse return error.MissingFixtureEos;
    _ = mlx.mlx_array_eval(eos_arr);
    const eos_index: usize = @intCast(mlx.mlx_array_data_int32(eos_arr).?[0]);

    const towers = [_]struct {
        sub: []const u8,
        cfg: sdxl.ClipTextConfig,
        pen: []const u8,
        pooled: ?[]const u8,
    }{
        .{ .sub = "text_encoder", .cfg = sdxl.CLIP_L_CONFIG, .pen = "clip_l.penultimate", .pooled = null },
        .{ .sub = "text_encoder_2", .cfg = sdxl.CLIP_BIG_G_CONFIG, .pen = "big_g.penultimate", .pooled = "big_g.pooled" },
    };

    for (towers) |t| {
        var tower = try loadTower(io, a, s, dir, t.sub, t.cfg, towerDtype());
        defer tower.deinit();
        var enc = try tower.encode(ids, eos_index, false);
        defer enc.deinit();

        const ref = fx.get(t.pen) orelse return error.MissingFixturePenultimate;
        try expectMatches(t.sub, "penultimate", enc.penultimate, ref, s);

        if (t.pooled) |key| {
            const pref = fx.get(key) orelse return error.MissingFixturePooled;
            try expectMatches(t.sub, "pooled", enc.pooled.?, pref, s);
        }
    }
}

/// Cosine + rms_ratio between a computed array and its reference.
fn expectMatches(who: []const u8, what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const f32got = try asF32Flat(got, s);
    defer _ = mlx.mlx_array_free(f32got);
    const f32ref = try asF32Flat(ref, s);
    defer _ = mlx.mlx_array_free(f32ref);

    const n = mlx.mlx_array_size(f32got);
    try testing.expectEqual(n, mlx.mlx_array_size(f32ref));
    const g = mlx.mlx_array_data_float32(f32got).?;
    const r = mlx.mlx_array_data_float32(f32ref).?;

    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    for (0..n) |i| {
        const gv: f64 = g[i];
        const rv: f64 = r[i];
        try testing.expect(std.math.isFinite(gv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    std.debug.print("[sdxl-parity] {s} {s}: cos={d:.6} rms_ratio={d:.6}\n", .{ who, what, cos, rms_ratio });

    // Weights are fp16 on our side and the reference runs fp32, so the bar is
    // fp16-appropriate rather than exact.
    try testing.expect(cos > 0.999);
    try testing.expect(rms_ratio > 0.99 and rms_ratio < 1.01);
}

fn asF32Flat(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var f = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    const n: c_int = @intCast(mlx.mlx_array_size(x));
    const shape = [_]c_int{n};
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, f, &shape, 1, s));
    _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_array_eval(flat);
    return flat;
}
