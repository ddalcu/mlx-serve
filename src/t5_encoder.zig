//! T5-XXL's ENCODER, as SD 3.5 uses it (`text_encoder_3`).
//!
//! transformers' `T5EncoderModel`: 24 blocks of {self-attention, gated-GELU
//! feed-forward} around a 4096-wide residual stream, no decoder, no LM head.
//! Nothing in this repo had it — every other text tower here is either CLIP
//! (absolute positions, causal, LayerNorm) or a modern decoder LM (RoPE) — and
//! T5 differs from both in four ways that are each individually silent:
//!
//!   RMSNORM WITH NO MEAN SUBTRACTION AND NO BIAS, IN f32. `T5LayerNorm` is
//!   `w * x * rsqrt(mean(x^2) + 1e-6)` with the variance AND the scaling done
//!   in float32 and the result cast back — transformers does that cast
//!   explicitly, for this model, because the tower is served in fp16. Doing the
//!   reduction at the activation width is invisible for a few layers and drifts
//!   by layer 24.
//!
//!   RELATIVE POSITION BIAS, NOT RoPE AND NOT ABSOLUTE. Only block 0's
//!   self-attention owns `relative_attention_bias`; every later block REUSES
//!   the `[1, heads, T, T]` bias computed there. A port that gives each layer
//!   its own bias finds the tensor missing and a port that recomputes per layer
//!   just wastes time, but a port that DROPS the bias produces a position-blind
//!   encoder that still returns well-formed embeddings.
//!
//!   NO ATTENTION SCALING. T5 deliberately omits `1/sqrt(d_kv)` — it is folded
//!   into the initialisation — so the SDPA scale here is exactly 1.0. Passing
//!   the usual `1/sqrt(64)` is an 8x error on every logit, which softmax turns
//!   into a near-uniform attention rather than a crash.
//!
//!   `gelu_new`, THE TANH APPROXIMATION. `feed_forward_proj: gated-gelu` with
//!   `dense_act_fn: gelu_new`, i.e. `wo(gelu_new(wi_0(x)) * wi_1(x))`.
//!   `sdxl_nn.gelu` is the EXACT erf form (SDXL's GEGLU wants that one), so
//!   this file carries its own `geluNew` rather than widening a shared helper
//!   that another oracle-validated port depends on. The repo has been bitten
//!   by two GELUs in one checkpoint before (LFM2-VL).
//!
//! The encoder is BIDIRECTIONAL — no causal mask — and SD 3 pads to a fixed
//! length (256) with pad id 0 and passes NO attention mask, so the pad
//! positions are attended to like any other token. `forward` therefore takes a
//! plain id buffer and applies no mask of its own: adding one would be
//! "correct" and would not be what the checkpoint was conditioned on.
//!
//! ORACLE STATUS
//!
//!   VERIFIED against transformers. `tests/dump_t5_fixtures.py build` runs a
//!   tiny random-weight `T5EncoderModel` of the real class on CPU in float32
//!   and dumps the embedding, layer 0's relative bias, every block's raw
//!   output and the final hidden state; the `t5 encoder parity` test below
//!   checks all of them, so a drift is attributed to a layer rather than
//!   hunted for. Measured at the tiny geometry in float32: rel_bias exact
//!   (cos 1.000000, rms_ratio 1.000000), block.0 cos 0.999989 rms 1.000461,
//!   block.1 cos 0.999928 rms 1.000060, last_hidden cos 0.999905 rms 1.000126.
//!
//!   The fixture's `ids` are read through `fixtureIds`, which REFUSES a
//!   non-int32 tensor instead of reinterpreting its bits. The dump script once
//!   wrote them through a blanket `.to(torch.float32)` while printing the
//!   pre-cast dtype, so a float tensor arrived claiming to be int32; every
//!   nonzero id decoded to garbage and only id 0 survived (0.0f is all-zero
//!   bits), which presented as embedding cos = sqrt(pads/T) = 0.699 and read
//!   exactly like a broken encoder. The encoder was correct the whole time.
//!
//!   VERIFIED against the real checkpoint's own metadata. The weight names and
//!   every config value below were read out of
//!   `adamo1139/stable-diffusion-3.5-large-ungated`'s `text_encoder_3/`
//!   (`config.json` and `model.safetensors.index.json`), not from a doc.
//!
//!   NOT VERIFIED. Parity against the REAL 4.7B tower — that is what
//!   `dump_t5_fixtures.py real` exists for, and it needs the 9.5 GB download.
//!   fp16 serving error is likewise unmeasured; `SD3_T5_F32=1` widens the
//!   tower so a pipeline-level gap can be attributed rather than guessed at.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const S = mlx.mlx_stream;
pub const Weights = model_mod.Weights;

/// Geometry, defaulted to SD 3.5's `text_encoder_3/config.json`. `load` reads
/// the checkpoint's own `config.json` over these — the defaults exist so a
/// weights-only test can construct the struct, not so the file can be skipped.
pub const T5Config = struct {
    num_layers: u32 = 24,
    d_model: u32 = 4096,
    d_ff: u32 = 10240,
    num_heads: u32 = 64,
    d_kv: u32 = 64,
    eps: f32 = 1e-6,
    rel_buckets: u32 = 32,
    rel_max_distance: u32 = 128,
    vocab_size: u32 = 32128,

    /// `num_heads * d_kv`. NOT necessarily `d_model`: T5-XXL is 64 x 64 = 4096
    /// and happens to match, but `d_kv` is an independent config field and the
    /// projections are `[inner, d_model]`, so the reshape reads this.
    pub fn inner(self: T5Config) u32 {
        return self.num_heads * self.d_kv;
    }
};

/// The width the tower is served at. The checkpoint ships fp16 (`torch_dtype`)
/// and diffusers runs SD 3's T5 in fp16, so that is the default; the dtype is a
/// PARAMETER because a 24-layer tower's fp16 error is measurable end to end.
pub const DEFAULT_DTYPE: mlx.mlx_dtype = .float16;

/// `SD3_T5_F32=1` widens the tower. Shared by the pipeline and the parity test
/// so both arms of an attribution run agree on what they mean.
pub fn towerDtype() mlx.mlx_dtype {
    return if (std.c.getenv("SD3_T5_F32") != null) .float32 else DEFAULT_DTYPE;
}

// ── Small array helpers, kept local so this file stands alone (the same shape
// as `sdxl_clip.zig`'s) ──

inline fn matmul(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, a, b, s));
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
inline fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn contiguous(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn replace(slot: *mlx.mlx_array, next: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(slot.*);
    slot.* = next;
}

/// `gelu_new` — the TANH approximation:
///
///     0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
///
/// NOT `sdxl_nn.gelu`, which is the exact erf form. See the file header.
fn geluNew(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const three = try scalarOf(3.0, dt);
    defer _ = mlx.mlx_array_free(three);
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_power(&x3, x, three, s));

    const coef = try scalarOf(0.044715, dt);
    defer _ = mlx.mlx_array_free(coef);
    const cx3 = try mulA(x3, coef, s);
    defer _ = mlx.mlx_array_free(cx3);
    const inner_sum = try addA(x, cx3, s);
    defer _ = mlx.mlx_array_free(inner_sum);

    const sqrt_2_over_pi = try scalarOf(0.7978845608028654, dt);
    defer _ = mlx.mlx_array_free(sqrt_2_over_pi);
    const scaled = try mulA(inner_sum, sqrt_2_over_pi, s);
    defer _ = mlx.mlx_array_free(scaled);
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, scaled, s));

    const one = try scalarOf(1.0, dt);
    defer _ = mlx.mlx_array_free(one);
    const one_plus = try addA(th, one, s);
    defer _ = mlx.mlx_array_free(one_plus);
    const half = try scalarOf(0.5, dt);
    defer _ = mlx.mlx_array_free(half);
    const half_x = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(half_x);
    return mulA(half_x, one_plus, s);
}

/// A scalar in the OPERAND's dtype. `mlx_array_new_float` is f32 and an f32
/// scalar promotes every fp16 operand it touches — the `[dtype-trace] residual
/// widened` class from the root rules, here inside the feed-forward's hot path.
fn scalarOf(v: f32, dt: mlx.mlx_dtype) !mlx.mlx_array {
    const f = mlx.mlx_array_new_float(v);
    if (dt == .float32) return f;
    defer _ = mlx.mlx_array_free(f);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    return astype(f, dt, s);
}

/// `T5LayerNorm`: RMS only — no mean subtraction, no bias — with the variance
/// AND the rescale computed in float32 and cast back to the activation dtype
/// before the gain is applied. That cast placement is transformers' own, and it
/// is why this is hand-rolled rather than `mlx_fast_rms_norm`.
fn rmsNorm(x: mlx.mlx_array, weight: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const out_dt = mlx.mlx_array_dtype(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const sq = try mulA(xf, xf, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    const ve = try addA(v, epsa, s);
    defer _ = mlx.mlx_array_free(ve);
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, ve, s));
    const normed = try mulA(xf, rsq, s);
    defer _ = mlx.mlx_array_free(normed);
    const back = try astype(normed, out_dt, s);
    defer _ = mlx.mlx_array_free(back);
    return mulA(weight, back, s);
}

// ── Relative position buckets ───────────────────────────────────────────

/// transformers' `T5Attention._relative_position_bucket` with
/// `bidirectional=True` (the encoder's branch), transcribed exactly.
///
/// `relative_position` is `key_pos - query_pos`. The bidirectional branch
/// halves the bucket count and gives the whole upper half to POSITIVE offsets,
/// so a bucket table read with the decoder's branch is off by 16 for every
/// forward-looking pair — plausible, and wrong everywhere.
///
/// The large-distance half is the transcription trap: buckets `max_exact
/// .. num_buckets-1` are LOG-spaced over `[max_exact, max_distance)`, the
/// float-to-int conversion TRUNCATES, and the result is clamped to the last
/// bucket rather than allowed to run off the table.
pub fn relativePositionBucket(relative_position: i32, num_buckets: u32, max_distance: u32) u32 {
    var buckets: u32 = 0;
    const half = num_buckets / 2;
    var rp = relative_position;
    if (rp > 0) buckets += half else rp = -rp;
    const n: u32 = @intCast(rp); // now |relative_position|

    const max_exact = half / 2;
    if (n < max_exact) return buckets + n;

    const ratio = @log(@as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(max_exact))) /
        @log(@as(f64, @floatFromInt(max_distance)) / @as(f64, @floatFromInt(max_exact)));
    const scaled = ratio * @as(f64, @floatFromInt(half - max_exact));
    // `.to(torch.long)` truncates toward zero; `n >= max_exact` keeps `scaled`
    // non-negative, so truncation is a floor here.
    const large = max_exact + @as(u32, @intFromFloat(@max(0.0, scaled)));
    return buckets + @min(large, half - 1);
}

// ── Layers ──────────────────────────────────────────────────────────────

/// A bias-less linear holding its weight PRE-TRANSPOSED to `[in, out]`. Every
/// T5 projection is bias-free — `q`, `k`, `v`, `o`, `wi_0`, `wi_1`, `wo` — which
/// is why there is no optional bias here at all.
const Linear = struct {
    w_t: mlx.mlx_array,

    fn deinit(self: *Linear) void {
        _ = mlx.mlx_array_free(self.w_t);
    }

    fn forward(self: *const Linear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        return matmul(x, self.w_t, s);
    }
};

const Block = struct {
    attn_norm: mlx.mlx_array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ff_norm: mlx.mlx_array,
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,

    fn deinit(self: *Block) void {
        _ = mlx.mlx_array_free(self.attn_norm);
        _ = mlx.mlx_array_free(self.ff_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.wi_0.deinit();
        self.wi_1.deinit();
        self.wo.deinit();
    }
};

pub const T5Encoder = struct {
    cfg: T5Config,
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    /// `shared.weight` — `[vocab, d_model]`. The real checkpoint ships NO
    /// `encoder.embed_tokens.weight`; the encoder reads `shared` directly
    /// (verified against the checkpoint's `model.safetensors.index.json`).
    shared: mlx.mlx_array,
    blocks: []Block,
    final_norm: mlx.mlx_array,
    /// `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight`
    /// — `[rel_buckets, num_heads]`. Owned by block 0 and reused by every
    /// later block, which is why it lives on the tower and not on a `Block`.
    rel_bias_table: mlx.mlx_array,

    pub fn deinit(self: *T5Encoder) void {
        _ = mlx.mlx_array_free(self.shared);
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        _ = mlx.mlx_array_free(self.final_norm);
        _ = mlx.mlx_array_free(self.rel_bias_table);
    }

    /// `[1, heads, T, T]` — `compute_bias(T, T)`.
    ///
    /// The bucket indices are computed on the HOST and gathered once. T is 256
    /// at most here, so the 65k-entry index table is nothing next to a single
    /// 4096-wide matmul, and the alternative — reproducing a log-scaled,
    /// truncating, clamped integer function in MLX ops — is the same
    /// transcription risk a second time.
    pub fn relativeBias(self: *T5Encoder, seq: usize) !mlx.mlx_array {
        const s = self.s;
        const idx = try self.allocator.alloc(i32, seq * seq);
        defer self.allocator.free(idx);
        for (0..seq) |q| {
            for (0..seq) |k| {
                const rp = @as(i32, @intCast(k)) - @as(i32, @intCast(q));
                idx[q * seq + k] = @intCast(relativePositionBucket(rp, self.cfg.rel_buckets, self.cfg.rel_max_distance));
            }
        }
        const flat_shape = [_]c_int{@intCast(seq * seq)};
        const idx_arr = mlx.mlx_array_new_data(idx.ptr, &flat_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(idx_arr);

        // AXIS 0. The bare `mlx_take` flattens the table and would return
        // [T*T] instead of [T*T, heads] — right-looking and one-dimensional.
        var rows = mlx.mlx_array_new(); // [T*T, heads]
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_take_axis(&rows, self.rel_bias_table, idx_arr, 0, s));

        const heads: c_int = @intCast(self.cfg.num_heads);
        const t: c_int = @intCast(seq);
        const cube = try reshape(rows, &[_]c_int{ t, t, heads }, s);
        defer _ = mlx.mlx_array_free(cube);
        // [q, k, heads] -> [heads, q, k], transformers' `permute([2, 0, 1])`.
        const perm = try transpose(cube, &[_]c_int{ 2, 0, 1 }, s);
        defer _ = mlx.mlx_array_free(perm);
        const batched = try reshape(perm, &[_]c_int{ 1, heads, t, t }, s);
        defer _ = mlx.mlx_array_free(batched);
        return contiguous(batched, s);
    }

    /// `ids` `[T]` -> hidden `[1, T, d_model]`. No pooling, no LM head.
    ///
    /// Padding is the CALLER's: SD 3 hands over a full-length buffer padded
    /// with pad id 0 and passes no attention mask, so there is none here.
    pub fn forward(self: *T5Encoder, ids: []const i32, s: S) !mlx.mlx_array {
        return self.forwardCapture(ids, s, null);
    }

    /// The same forward, optionally publishing each block's RAW output into
    /// `blocks_out` (one owned handle per block, the caller frees).
    ///
    /// This exists for the oracle, not for the pipeline: a 24-layer tower that
    /// agrees at block 0 and disagrees at the end is a needle in a haystack,
    /// and the fixture dumps every block precisely so the parity test can name
    /// the layer that drifted. `forward` passes null and pays nothing.
    pub fn forwardCapture(self: *T5Encoder, ids: []const i32, s: S, blocks_out: ?[]mlx.mlx_array) !mlx.mlx_array {
        if (ids.len == 0) return error.EmptyT5Input;
        if (blocks_out) |b| {
            if (b.len != self.blocks.len) return error.BadCaptureLength;
        }
        const seq: c_int = @intCast(ids.len);
        const d_model: c_int = @intCast(self.cfg.d_model);
        const inner: c_int = @intCast(self.cfg.inner());
        const heads: c_int = @intCast(self.cfg.num_heads);
        const d_kv: c_int = @intCast(self.cfg.d_kv);

        const id_shape = [_]c_int{seq};
        const ids_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(ids_arr);

        var tok = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tok);
        try mlx.check(mlx.mlx_take_axis(&tok, self.shared, ids_arr, 0, s));

        var h = try reshape(tok, &[_]c_int{ 1, seq, d_model }, s);
        errdefer _ = mlx.mlx_array_free(h);

        // ONE bias for the whole tower — computed at block 0's expense and
        // reused, exactly as transformers threads `position_bias` through.
        const bias_f32 = try self.relativeBias(ids.len);
        defer _ = mlx.mlx_array_free(bias_f32);
        const bias = try astype(bias_f32, self.dtype, s);
        defer _ = mlx.mlx_array_free(bias);

        const qkv_shape = [_]c_int{ 1, seq, heads, d_kv };
        const to_heads = [_]c_int{ 0, 2, 1, 3 };

        for (self.blocks, 0..) |*b, bi| {
            // ── Self-attention, pre-norm.
            const normed = try rmsNorm(h, b.attn_norm, self.cfg.eps, s);
            defer _ = mlx.mlx_array_free(normed);

            const q = try b.q.forward(normed, s);
            defer _ = mlx.mlx_array_free(q);
            const k = try b.k.forward(normed, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try b.v.forward(normed, s);
            defer _ = mlx.mlx_array_free(v);

            const qh = try toHeads(q, &qkv_shape, &to_heads, s);
            defer _ = mlx.mlx_array_free(qh);
            const kh = try toHeads(k, &qkv_shape, &to_heads, s);
            defer _ = mlx.mlx_array_free(kh);
            const vh = try toHeads(v, &qkv_shape, &to_heads, s);
            defer _ = mlx.mlx_array_free(vh);

            // SCALE 1.0 — T5 folds `1/sqrt(d_kv)` into initialisation. The
            // relative bias rides as an ADDITIVE array mask; the encoder is
            // bidirectional, so there is no causal component at all.
            var attn = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(attn);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
                &attn,
                qh,
                kh,
                vh,
                1.0,
                "array",
                bias,
                .{ .ctx = null },
                false,
                s,
            ));

            const merged = blk: {
                const t = try transpose(attn, &to_heads, s);
                defer _ = mlx.mlx_array_free(t);
                break :blk try reshape(t, &[_]c_int{ 1, seq, inner }, s);
            };
            defer _ = mlx.mlx_array_free(merged);
            const proj = try b.o.forward(merged, s);
            defer _ = mlx.mlx_array_free(proj);
            replace(&h, try addA(h, proj, s));

            // ── Gated-GELU feed-forward, pre-norm.
            const normed2 = try rmsNorm(h, b.ff_norm, self.cfg.eps, s);
            defer _ = mlx.mlx_array_free(normed2);
            const gate_in = try b.wi_0.forward(normed2, s);
            defer _ = mlx.mlx_array_free(gate_in);
            const gate = try geluNew(gate_in, s);
            defer _ = mlx.mlx_array_free(gate);
            const up = try b.wi_1.forward(normed2, s);
            defer _ = mlx.mlx_array_free(up);
            const gated = try mulA(gate, up, s);
            defer _ = mlx.mlx_array_free(gated);
            const down = try b.wo.forward(gated, s);
            defer _ = mlx.mlx_array_free(down);
            replace(&h, try addA(h, down, s));

            if (blocks_out) |cap| {
                var copy = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_copy(&copy, h, s));
                cap[bi] = copy;
            }
        }

        const out = try rmsNorm(h, self.final_norm, self.cfg.eps, s);
        _ = mlx.mlx_array_free(h);
        return out;
    }
};

fn toHeads(x: mlx.mlx_array, shape: []const c_int, perm: []const c_int, s: S) !mlx.mlx_array {
    const r = try reshape(x, shape, s);
    defer _ = mlx.mlx_array_free(r);
    return transpose(r, perm, s);
}

// ── Loading ─────────────────────────────────────────────────────────────

/// Load the tower from `<model_dir>/<sub>` (`text_encoder_3`).
///
/// The component is SHARDED (`model.safetensors.index.json` names two files on
/// SD 3.5 Large), which `model.loadWeights` already handles by sweeping the
/// directory — the same call `sdxl_clip.loadTower` makes.
pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []const u8,
    sub: []const u8,
    dtype: mlx.mlx_dtype,
) !T5Encoder {
    const dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, sub });
    defer allocator.free(dir);

    const cfg = try readConfig(io, allocator, dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    return loadFromWeights(allocator, s, &w, cfg, dtype);
}

/// Read `<dir>/config.json`. Every field is read PER FIELD with the default as
/// the fallback rather than demanding the whole object parse into a struct — a
/// checkpoint that omits one key is a config this port can still serve, and
/// `std.json` mapping straight onto a struct turns that into a load failure.
pub fn readConfig(io: std.Io, allocator: std.mem.Allocator, dir: []const u8) !T5Config {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
    defer allocator.free(path);
    const bytes = try readFile(io, allocator, path, 1 << 20);
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadT5Config;
    return configFromJson(parsed.value.object);
}

/// Split from `readConfig` so a test can drive it from a literal — a config
/// reader with no test is a set of field names nobody has checked.
pub fn configFromJson(obj: std.json.ObjectMap) T5Config {
    var cfg = T5Config{};
    const U = struct {
        fn u32At(o: std.json.ObjectMap, key: []const u8, cur: u32) u32 {
            const v = o.get(key) orelse return cur;
            return switch (v) {
                .integer => |i| if (i > 0) @intCast(i) else cur,
                else => cur,
            };
        }
    };
    cfg.num_layers = U.u32At(obj, "num_layers", cfg.num_layers);
    cfg.d_model = U.u32At(obj, "d_model", cfg.d_model);
    cfg.d_ff = U.u32At(obj, "d_ff", cfg.d_ff);
    cfg.num_heads = U.u32At(obj, "num_heads", cfg.num_heads);
    cfg.d_kv = U.u32At(obj, "d_kv", cfg.d_kv);
    cfg.rel_buckets = U.u32At(obj, "relative_attention_num_buckets", cfg.rel_buckets);
    cfg.rel_max_distance = U.u32At(obj, "relative_attention_max_distance", cfg.rel_max_distance);
    cfg.vocab_size = U.u32At(obj, "vocab_size", cfg.vocab_size);
    if (obj.get("layer_norm_epsilon")) |v| {
        cfg.eps = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => cfg.eps,
        };
    }
    return cfg;
}

/// The weight-binding half, so a test drives the tower from an in-memory map.
pub fn loadFromWeights(
    allocator: std.mem.Allocator,
    s: S,
    w: *const Weights,
    cfg: T5Config,
    dtype: mlx.mlx_dtype,
) !T5Encoder {
    var enc: T5Encoder = undefined;
    enc.cfg = cfg;
    enc.allocator = allocator;
    enc.s = s;
    enc.dtype = dtype;

    enc.shared = try dup(w, "shared.weight", dtype, s);
    errdefer _ = mlx.mlx_array_free(enc.shared);
    enc.final_norm = try dup(w, "encoder.final_layer_norm.weight", dtype, s);
    errdefer _ = mlx.mlx_array_free(enc.final_norm);
    // The bias table stays f32: it is [32, 64] and it is ADDED to attention
    // logits, so rounding it to fp16 costs nothing to keep and is free to skip.
    enc.rel_bias_table = try dup(w, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", .float32, s);
    errdefer _ = mlx.mlx_array_free(enc.rel_bias_table);

    enc.blocks = try allocator.alloc(Block, cfg.num_layers);
    var built: usize = 0;
    errdefer {
        for (enc.blocks[0..built]) |*b| b.deinit();
        allocator.free(enc.blocks);
    }
    for (enc.blocks, 0..) |*b, i| {
        b.attn_norm = try dupIdx(w, allocator, i, "layer.0.layer_norm.weight", dtype, s);
        b.ff_norm = try dupIdx(w, allocator, i, "layer.1.layer_norm.weight", dtype, s);
        b.q = try linearIdx(w, allocator, s, i, "layer.0.SelfAttention.q", dtype);
        b.k = try linearIdx(w, allocator, s, i, "layer.0.SelfAttention.k", dtype);
        b.v = try linearIdx(w, allocator, s, i, "layer.0.SelfAttention.v", dtype);
        b.o = try linearIdx(w, allocator, s, i, "layer.0.SelfAttention.o", dtype);
        b.wi_0 = try linearIdx(w, allocator, s, i, "layer.1.DenseReluDense.wi_0", dtype);
        b.wi_1 = try linearIdx(w, allocator, s, i, "layer.1.DenseReluDense.wi_1", dtype);
        b.wo = try linearIdx(w, allocator, s, i, "layer.1.DenseReluDense.wo", dtype);
        built = i + 1;
    }

    log.info("[sd3-t5] loaded encoder: layers={d} d_model={d} d_ff={d} heads={d}x{d} buckets={d}\n", .{
        cfg.num_layers, cfg.d_model, cfg.d_ff, cfg.num_heads, cfg.d_kv, cfg.rel_buckets,
    });
    return enc;
}

fn dup(w: *const Weights, name: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const src = w.get(name) orelse {
        log.err("[sd3-t5] missing weight {s}\n", .{name});
        return error.MissingWeight;
    };
    return astype(src, dt, s);
}

fn dupIdx(w: *const Weights, a: std.mem.Allocator, i: usize, suffix: []const u8, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, "encoder.block.{d}.{s}", .{ i, suffix });
    defer a.free(name);
    return dup(w, name, dt, s);
}

/// Weights are stored `[out, in]`; the forward wants `[in, out]`, so the
/// transpose happens ONCE at load rather than per token.
fn linearIdx(w: *const Weights, a: std.mem.Allocator, s: S, i: usize, prefix: []const u8, dt: mlx.mlx_dtype) !Linear {
    const name = try std.fmt.allocPrint(a, "encoder.block.{d}.{s}.weight", .{ i, prefix });
    defer a.free(name);
    const src = try dup(w, name, dt, s);
    defer _ = mlx.mlx_array_free(src);
    const t = try transpose(src, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    return .{ .w_t = try contiguous(t, s) };
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, limit: usize) ![]u8 {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return error.BadPath;
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit));
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "t5 encoder: relative position buckets follow the bidirectional branch" {
    // Reference values computed by transformers' own
    // `T5Attention._relative_position_bucket(torch.tensor(rp), bidirectional=True,
    // num_buckets=32, max_distance=128)`, not re-derived from the code above —
    // a reference that reuses the implementation's formula proves only that the
    // formula is stable, not that it is the right one.
    const cases = [_]struct { rp: i32, want: u32 }{
        .{ .rp = 0, .want = 0 },
        .{ .rp = 1, .want = 17 },
        .{ .rp = -1, .want = 1 },
        .{ .rp = 7, .want = 23 },
        .{ .rp = -7, .want = 7 },
        // At |rp| == max_exact the log half takes over and the two signs stay
        // 16 apart — the bidirectional split.
        .{ .rp = 8, .want = 24 },
        .{ .rp = -8, .want = 8 },
        .{ .rp = 15, .want = 25 },
        .{ .rp = -15, .want = 9 },
        .{ .rp = 127, .want = 31 },
        .{ .rp = -127, .want = 15 },
        // Clamped: anything at or past max_distance lands in the LAST bucket
        // of its half and stays there forever.
        .{ .rp = 128, .want = 31 },
        .{ .rp = -128, .want = 15 },
        .{ .rp = 100000, .want = 31 },
        .{ .rp = -100000, .want = 15 },
    };
    for (cases) |c| {
        try testing.expectEqual(c.want, relativePositionBucket(c.rp, 32, 128));
    }
}

test "t5 encoder: the bidirectional branch is not the decoder's" {
    // The trap: reading the decoder branch (all buckets to one sign, offsets
    // clamped at 0) loads cleanly and mis-positions every forward-looking pair.
    // Positive and negative offsets of the same magnitude must DIFFER, and by
    // exactly half the table.
    for (1..40) |m| {
        const rp: i32 = @intCast(m);
        const pos = relativePositionBucket(rp, 32, 128);
        const neg = relativePositionBucket(-rp, 32, 128);
        try testing.expect(pos != neg);
        try testing.expectEqual(@as(u32, 16), pos - neg);
    }
}

test "t5 encoder: config reads the checkpoint's own fields" {
    const a = testing.allocator;
    // The real `text_encoder_3/config.json`, trimmed to the fields that decide
    // geometry. Values verified against
    // adamo1139/stable-diffusion-3.5-large-ungated.
    const json =
        \\{"d_ff": 10240, "d_kv": 64, "d_model": 4096, "num_heads": 64,
        \\ "num_layers": 24, "layer_norm_epsilon": 1e-06, "vocab_size": 32128,
        \\ "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer parsed.deinit();
    const cfg = configFromJson(parsed.value.object);
    try testing.expectEqual(@as(u32, 24), cfg.num_layers);
    try testing.expectEqual(@as(u32, 4096), cfg.d_model);
    try testing.expectEqual(@as(u32, 10240), cfg.d_ff);
    try testing.expectEqual(@as(u32, 64), cfg.num_heads);
    try testing.expectEqual(@as(u32, 64), cfg.d_kv);
    try testing.expectEqual(@as(u32, 4096), cfg.inner());
    try testing.expectApproxEqAbs(@as(f32, 1e-6), cfg.eps, 1e-12);

    // An EMPTY object keeps every default rather than failing — the defaults
    // are SD 3.5's, so a config missing a key still serves.
    var empty = try std.json.parseFromSlice(std.json.Value, a, "{}", .{});
    defer empty.deinit();
    try testing.expectEqual(@as(u32, 24), configFromJson(empty.value.object).num_layers);
}

test "t5 encoder: gelu_new is the tanh approximation, not the erf form" {
    const s = mlx.mlx_default_gpu_stream_new();
    const vals = [_]f32{ -3, -1, -0.5, 0, 0.5, 1, 3 };
    const shape = [_]c_int{7};
    const x = mlx.mlx_array_new_data(&vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(x);

    const y = try geluNew(x, s);
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);
    const got = mlx.mlx_array_data_float32(y).?;

    // Independently computed by CPython from the tanh formula. At x = -3 the
    // two GELU variants differ in the fourth decimal; the tolerance separates
    // them, and the erf-form values are listed so the difference is visible.
    const want_tanh = [_]f32{ -0.00363739, -0.15880801, -0.15428599, 0.0, 0.34571400, 0.84119199, 2.99636261 };
    const erf_form = [_]f32{ -0.00404969, -0.15865525, -0.15426877, 0.0, 0.34573123, 0.84134475, 2.99595031 };
    for (want_tanh, erf_form, 0..) |wt, ef, i| {
        try testing.expectApproxEqAbs(wt, got[i], 1e-5);
        if (wt != ef) try testing.expect(@abs(got[i] - ef) > 1e-6);
    }
}

test "t5 encoder: rmsNorm has no mean subtraction" {
    // The trap a LayerNorm would hide: T5's norm leaves a constant offset
    // ALONE. A mean-subtracting norm sends a constant vector to zero; the RMS
    // norm sends it to a nonzero constant.
    const s = mlx.mlx_default_gpu_stream_new();
    const vals = [_]f32{ 2, 2, 2, 2 };
    const shape = [_]c_int{ 1, 1, 4 };
    const x = mlx.mlx_array_new_data(&vals, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    const wsh = [_]c_int{4};
    const ones = [_]f32{ 1, 1, 1, 1 };
    const w = mlx.mlx_array_new_data(&ones, &wsh, 1, .float32);
    defer _ = mlx.mlx_array_free(w);

    const y = try rmsNorm(x, w, 1e-6, s);
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);
    const got = mlx.mlx_array_data_float32(y).?;
    for (0..4) |i| try testing.expectApproxEqAbs(@as(f32, 1.0), got[i], 1e-4);
}

// Numerical PARITY against transformers' own `T5EncoderModel`. The fixture is
// generated by `tests/dump_t5_fixtures.py` (see its docstring), on CPU in
// float32, over the SAME ids this test reads out of the fixture.
//
//   /tmp/claude-501/sd3venv/bin/python tests/dump_t5_fixtures.py build \
//       --dir ~/.mlx-serve/staging/t5-tiny \
//       --out ~/.mlx-serve/staging/t5_tiny_fixture.safetensors
//
//   T5_MODEL_DIR=~/.mlx-serve/staging/t5-tiny \
//   T5_FIXTURE=~/.mlx-serve/staging/t5_tiny_fixture.safetensors \
//     zig build test -Doptimize=ReleaseFast -Dtest-filter="t5 encoder parity"
//
// Points at the REAL `text_encoder_3` with a `real`-mode fixture unchanged.
//
// Asserts rms_ratio BESIDE the cosine at every stage. A cosine cannot see a
// scale error, and this tensor is CONCATENATED with the CLIP stream before the
// MMDiT cross-attends to it — a uniformly scaled tower would score a perfect
// cosine and unbalance everything downstream.
/// Read a fixture's `ids` row as int32, REFUSING a float tensor rather than
/// reinterpreting its bits.
///
/// The dump script once wrote `ids` through a blanket `.to(torch.float32)` while
/// its own log printed the pre-cast dtype, so a float tensor arrived claiming to
/// be int32. `mlx_array_data_int32` on that reinterprets the bit pattern: every
/// nonzero id becomes garbage and only id 0 survives (0.0f is all-zero bits),
/// which presents as embedding cos = sqrt(pads/T) — a number that looks like a
/// broken encoder and is actually a broken fixture. The repo's standing rule is
/// to dtype-gate every raw data-pointer read; this is that gate.
fn fixtureIds(a: std.mem.Allocator, ids_arr: mlx.mlx_array) ![]i32 {
    _ = mlx.mlx_array_eval(ids_arr);
    const dt = mlx.mlx_array_dtype(ids_arr);
    if (dt != .int32) {
        std.debug.print("[t5-parity] fixture `ids` is {s}, expected int32 — regenerate with tests/dump_t5_fixtures.py\n", .{@tagName(dt)});
        return error.FixtureIdsNotInt32;
    }
    const n = mlx.mlx_array_size(ids_arr);
    const p = mlx.mlx_array_data_int32(ids_arr) orelse return error.FixtureIdsUnreadable;
    const out = try a.alloc(i32, n);
    errdefer a.free(out);
    for (out, 0..) |*v, i| v.* = p[i];
    return out;
}

test "t5 encoder parity: matches transformers block by block" {
    const dir = std.mem.span(std.c.getenv("T5_MODEL_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("T5_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    // The ids come FROM the fixture, so the two sides cannot describe
    // different prompts.
    const ids = try fixtureIds(a, fx.get("ids") orelse return error.MissingFixtureIds);
    defer a.free(ids);

    const cfg = try readConfig(io, a, dir);
    var w = try model_mod.loadWeights(io, a, dir);
    defer w.deinit();
    var enc = try loadFromWeights(a, s, &w, cfg, towerDtype());
    defer enc.deinit();

    // ── The relative bias first. It is shared by every block, so a wrong
    // bucket table is a wrong tower and nothing after this point is diagnostic.
    {
        const ref = fx.get("rel_bias") orelse return error.MissingFixtureRelBias;
        const got = try enc.relativeBias(ids.len);
        defer _ = mlx.mlx_array_free(got);
        try expectMatches("rel_bias", got, ref, s);
    }

    const caps = try a.alloc(mlx.mlx_array, enc.blocks.len);
    defer a.free(caps);
    @memset(caps, .{ .ctx = null });
    defer for (caps) |c| {
        if (c.ctx != null) _ = mlx.mlx_array_free(c);
    };

    const out = try enc.forwardCapture(ids, s, caps);
    defer _ = mlx.mlx_array_free(out);

    // Block by block, so a drift is ATTRIBUTED rather than hunted for.
    for (caps, 0..) |c, i| {
        var key: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&key, "block.{d}.out", .{i});
        const bref = fx.get(name) orelse return error.MissingFixtureBlock;
        try expectMatches(name, c, bref, s);
    }

    const ref = fx.get("last_hidden") orelse return error.MissingFixtureHidden;
    try expectMatches("last_hidden", out, ref, s);
}

/// Cosine + rms_ratio between a computed array and its reference.
fn expectMatches(what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const g32 = try asF32Flat(got, s);
    defer _ = mlx.mlx_array_free(g32);
    const r32 = try asF32Flat(ref, s);
    defer _ = mlx.mlx_array_free(r32);

    const n = mlx.mlx_array_size(g32);
    try testing.expectEqual(n, mlx.mlx_array_size(r32));
    const g = mlx.mlx_array_data_float32(g32).?;
    const r = mlx.mlx_array_data_float32(r32).?;

    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    for (0..n) |i| {
        const gv: f64 = g[i];
        const rv: f64 = r[i];
        // Finiteness BEFORE the diff: an all-NaN tensor has no cosine at all,
        // and fp16 T5 is exactly the family that overflows.
        try testing.expect(std.math.isFinite(gv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    std.debug.print("[t5-parity] {s}: cos={d:.6} rms_ratio={d:.6}\n", .{ what, cos, rms_ratio });
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

test "t5 DEBUG stages" {
    const dir = std.mem.span(std.c.getenv("T5_MODEL_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("T5_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const ids = try fixtureIds(a, fx.get("ids").?);
    defer a.free(ids);

    const cfg = try readConfig(io, a, dir);
    var w = try model_mod.loadWeights(io, a, dir);
    defer w.deinit();
    var enc = try loadFromWeights(a, s, &w, cfg, .float32);
    defer enc.deinit();

    // embedding
    const seq: c_int = @intCast(ids.len);
    const idsh = [_]c_int{seq};
    const ia = mlx.mlx_array_new_data(ids.ptr, &idsh, 1, .int32);
    defer _ = mlx.mlx_array_free(ia);
    var tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok);
    try mlx.check(mlx.mlx_take_axis(&tok, enc.shared, ia, 0, s));
    try expectMatches("embedding", tok, fx.get("embedding").?, s);

    // attention only, block 0
    const h0 = try reshape(tok, &[_]c_int{ 1, seq, @intCast(cfg.d_model) }, s);
    defer _ = mlx.mlx_array_free(h0);
    const b = &enc.blocks[0];
    const normed = try rmsNorm(h0, b.attn_norm, cfg.eps, s);
    defer _ = mlx.mlx_array_free(normed);
    std.debug.print("normed shape ndim={d}\n", .{mlx.mlx_array_ndim(normed)});
    const q = try b.q.forward(normed, s);
    defer _ = mlx.mlx_array_free(q);
    _ = mlx.mlx_array_eval(q);
    const qsh = mlx.getShape(q);
    std.debug.print("q shape {any}\n", .{qsh});
    const qf = try asF32Flat(q, s);
    defer _ = mlx.mlx_array_free(qf);
    const qd = mlx.mlx_array_data_float32(qf).?;
    std.debug.print("q[0..4] {d:.5} {d:.5} {d:.5} {d:.5}\n", .{ qd[0], qd[1], qd[2], qd[3] });
    const nf = try asF32Flat(normed, s);
    defer _ = mlx.mlx_array_free(nf);
    const nd = mlx.mlx_array_data_float32(nf).?;
    std.debug.print("normed[0..4] {d:.5} {d:.5} {d:.5} {d:.5}\n", .{ nd[0], nd[1], nd[2], nd[3] });
}
