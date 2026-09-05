//! SD 3.5's MMDiT — diffusers' `SD3Transformer2DModel`, ported to MLX.
//!
//! ONE implementation serves all three shipped checkpoints, because the
//! difference between them is entirely in `transformer/config.json`:
//!
//!   Large / Large-Turbo  38 layers, 38 heads x 64 = 2432, `pos_embed_max_size`
//!                        192, NO `dual_attention_layers`.
//!   Medium               24 layers, 24 heads x 64 = 1536, `pos_embed_max_size`
//!                        384, `dual_attention_layers: [0..12]` — MMDiT-**X**,
//!                        where those blocks carry a SECOND self-attention on
//!                        the image stream only.
//!
//! Nothing here branches on a model NAME or a SIZE. The config is the
//! discriminator, which is this repo's standing rule about config fields that
//! silently decide numerics (`LtxVersion`, `has_micro_conditioning`, the v-pred
//! markers): read the field, never infer it from the folder.
//!
//! ── The shape, and the five places it is easy to get silently wrong ─────
//!
//!  1. THE POSITIONAL EMBEDDING IS CENTRE-CROPPED, NOT INTERPOLATED. It is
//!     stored for a `pos_embed_max_size` square of patches and sliced at
//!     `top = (max - h) // 2` (`sd3.posEmbedCropStart`). Top-left instead of
//!     centred, or off by one, mis-frames every generation with nothing to
//!     error on — the `add_time_ids` failure class.
//!
//!  2. THE CONCAT ORDER IS IMAGE FIRST. `JointAttnProcessor2_0` builds
//!     `cat([image, text], dim=seq)` and splits the result back at
//!     `residual.shape[1]`. FLUX's double-stream blocks in `flux.zig` put TEXT
//!     first; copying that order here is a permutation that a cosine test
//!     cannot see, which is why the parity test also checks a position-weighted
//!     checksum.
//!
//!  3. `AdaLayerNormZero` CHUNKS SHIFT-FIRST; `AdaLayerNormContinuous` CHUNKS
//!     SCALE-FIRST. The block modulations are
//!     `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`, but
//!     `norm_out` (and the last block's `norm1_context`) are `scale, shift`.
//!     Two adjacent halves of the same tensor, opposite meanings.
//!
//!  4. THE LAST BLOCK IS TRUNCATED (`context_pre_only`). Its `norm1_context` is
//!     an `AdaLayerNormContinuous` with a 2*dim `linear` — verified in the
//!     checkpoint: `transformer_blocks.37.norm1_context.linear.weight` is
//!     [4864, 2432] where every other block's is [14592, 2432] — and it has NO
//!     `attn.to_add_out`, NO `norm2_context` and NO `ff_context`, because the
//!     text stream's output is discarded. It still runs its half of the joint
//!     attention; it just never writes a text residual.
//!
//!  5. MMDiT-X's `norm1` PRODUCES NINE CHUNKS, not six. `SD35AdaLayerNormZeroX`
//!     appends `shift_msa2, scale_msa2, gate_msa2`, and the second modulation
//!     is applied to the SAME LayerNorm output as the first (the norm runs
//!     once), then `attn2`'s output is gated by `gate_msa2` and added into the
//!     image residual BEFORE the feed-forward. `attn2` has its own qk-norm and
//!     no added-stream projections at all.
//!
//! ── ORACLE STATUS ──────────────────────────────────────────────────────
//!
//!   VERIFIED against diffusers, both shapes. `tests/dump_sd3_mmdit_fixtures.py
//!   build` constructs a tiny random-weight `SD3Transformer2DModel` of the real
//!   class — once WITHOUT `dual_attention_layers` (the Large shape) and once
//!   WITH them (the Medium/MMDiT-X shape) — runs it on CPU in float32 and dumps
//!   its inputs, every structural intermediate and its output. The tests at the
//!   bottom of this file replay that forward and compare per capture. Measured
//!   at the time of writing, float32, batch 2:
//!
//!     Large shape   pos_embed cos 1.000000 rms 1.000000 · temb 1.000000/1.000000
//!                   context 1.000000/1.000000 · block0..2 img+txt 1.000000
//!                   norm_out 1.000000 · out.sample cos 1.000000 rms 1.000000
//!     Medium shape  identical, dual blocks included.
//!
//!   VERIFIED against the checkpoints' own metadata. Every weight NAME and
//!   every shape bound below was read out of
//!   `adamo1139/stable-diffusion-3.5-{large,medium}-ungated`'s safetensors
//!   headers (the stability repos are gated; those are exact mirrors), not from
//!   a description of them.
//!
//!   NOT VERIFIED. Numerical parity against a REAL SD 3.5 checkpoint — the
//!   `real` mode of the dump script exists for it and has not been run here (no
//!   20 GB download in this lane). Nothing has been measured at float16 either:
//!   the fixtures are float32 on both sides, so the serving dtype's error
//!   budget is unmeasured. Both are a download away and neither is a shape
//!   question, which is what a tiny fixture can and cannot settle.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const nn = @import("sdxl_nn.zig");
const sd3 = @import("sd3.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

/// diffusers' `Timesteps(num_channels=256, ...)` inside
/// `CombinedTimestepTextProjEmbeddings`. Not `inner_dim`: the sinusoid is
/// always 256 wide and `timestep_embedder.linear_1` is what widens it.
pub const TIME_PROJ_CHANNELS: usize = 256;

/// Every LayerNorm and RMSNorm in this tower. diffusers hardcodes 1e-6 at
/// construction (`Attention(..., eps=1e-6)`, `AdaLayerNormZero`'s
/// `nn.LayerNorm(..., eps=1e-6)`), so there is no config field for it.
const EPS: f32 = 1e-6;

// ════════════════════════════════════════════════════════════════════════
// Config
// ════════════════════════════════════════════════════════════════════════

/// `transformer/config.json`, as this module needs it: `sd3.MmditConfig` plus
/// the one thing that struct has no room for.
pub const Config = struct {
    base: sd3.MmditConfig,
    /// `dual_attention_layers` as a bitset — bit i set means layer i is
    /// MMDiT-X. A LIST in the config (Medium's is `[0..12]`), not a count and
    /// not a prefix length: the field is read literally so a checkpoint that
    /// ever ships a non-contiguous set still binds.
    dual_mask: u64 = 0,

    pub fn isDual(self: Config, layer: u32) bool {
        return layer < 64 and (self.dual_mask & (@as(u64, 1) << @intCast(layer))) != 0;
    }
};

pub fn parseConfig(allocator: std.mem.Allocator, json_bytes: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const int = struct {
        fn get(o: std.json.ObjectMap, key: []const u8, dflt: ?u32) !u32 {
            const v = o.get(key) orelse return dflt orelse error.MissingMmditField;
            return switch (v) {
                .integer => @intCast(v.integer),
                // A diffusers export writes `null` for an unset optional
                // rather than omitting it (SD-Turbo's `addition_embed_type`
                // is the precedent in this repo). Treat the two the same.
                .null => dflt orelse error.MissingMmditField,
                else => error.BadMmditField,
            };
        }
    };

    var cfg = Config{ .base = .{} };
    cfg.base.num_layers = try int.get(root, "num_layers", null);
    cfg.base.num_attention_heads = try int.get(root, "num_attention_heads", null);
    cfg.base.attention_head_dim = try int.get(root, "attention_head_dim", null);
    cfg.base.in_channels = try int.get(root, "in_channels", 16);
    cfg.base.out_channels = try int.get(root, "out_channels", cfg.base.in_channels);
    cfg.base.patch_size = try int.get(root, "patch_size", 2);
    cfg.base.pos_embed_max_size = try int.get(root, "pos_embed_max_size", null);
    cfg.base.joint_attention_dim = try int.get(root, "joint_attention_dim", 4096);
    cfg.base.pooled_projection_dim = try int.get(root, "pooled_projection_dim", 2048);
    cfg.base.sample_size = try int.get(root, "sample_size", 128);

    // `caption_projection_dim` is the width `context_embedder` projects the T5
    // sequence to, and from there it is a TOKEN STREAM inside blocks of width
    // `inner_dim`. diffusers never reconciles the two; every shipped SD 3.x
    // config sets them equal (2432 / 1536, both verified). A checkpoint that
    // ever separated them would need a second width threaded through every
    // block, so it is refused by NAME here rather than bound into a network
    // whose attention silently would not compose.
    const caption = try int.get(root, "caption_projection_dim", cfg.base.innerDim());
    if (caption != cfg.base.innerDim()) {
        log.err("[sd3] caption_projection_dim {d} != inner dim {d}\n", .{ caption, cfg.base.innerDim() });
        return error.UnsupportedCaptionWidth;
    }

    // `qk_norm` is a STRING or null. Present and "rms_norm" on every 3.5
    // checkpoint; SD 3.0 Medium shipped it absent, and that is a different
    // network, not a smaller one — so an unknown spelling is refused rather
    // than silently treated as "off".
    cfg.base.qk_norm = false;
    if (root.get("qk_norm")) |v| switch (v) {
        .null => {},
        .string => |sv| {
            if (!std.mem.eql(u8, sv, "rms_norm")) {
                log.err("[sd3] unsupported qk_norm: {s}\n", .{sv});
                return error.UnsupportedQkNorm;
            }
            cfg.base.qk_norm = true;
        },
        else => return error.BadMmditField,
    };

    if (root.get("dual_attention_layers")) |v| switch (v) {
        .null => {},
        .array => |arr| for (arr.items) |item| {
            const idx: u32 = switch (item) {
                .integer => @intCast(item.integer),
                else => return error.BadMmditField,
            };
            if (idx >= 64) return error.BadMmditField;
            cfg.dual_mask |= @as(u64, 1) << @intCast(idx);
        },
        else => return error.BadMmditField,
    };

    return cfg;
}

// ════════════════════════════════════════════════════════════════════════
// Small tensor helpers
// ════════════════════════════════════════════════════════════════════════

/// Slice `[lo, hi)` on one axis of an arbitrary-rank array.
fn sliceAxis(x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (axis >= sh.len or sh.len > 8) return error.BadSliceAxis;
    var start: [8]c_int = undefined;
    var stop: [8]c_int = undefined;
    var stride: [8]c_int = undefined;
    for (sh, 0..) |d, i| {
        start[i] = 0;
        stop[i] = d;
        stride[i] = 1;
    }
    start[axis] = lo;
    stop[axis] = hi;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &stride, sh.len, s));
    return o;
}

/// LayerNorm over the last axis with NO affine parameters — diffusers'
/// `nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)`, which is what every
/// norm in this tower is (the scale and shift arrive from the modulation
/// instead). Statistics in f32, result back in the input's dtype: that is
/// torch's own behaviour under autocast and the only way an fp16 forward keeps
/// its variance.
fn layerNormNA(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const out_dt = mlx.mlx_array_dtype(x);
    const xf = try nn.astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_mean_axis(&m, xf, -1, true, s));
    const xc = try nn.subA(xf, m, s);
    defer _ = mlx.mlx_array_free(xc);
    const sq = try nn.mulA(xc, xc, s);
    defer _ = mlx.mlx_array_free(sq);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_mean_axis(&v, sq, -1, true, s));
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    try mlx.check(mlx.mlx_add(&ve, v, epsa, s));
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, ve, s));
    const out = try nn.mulA(xc, rs, s);
    defer _ = mlx.mlx_array_free(out);
    return nn.astype(out, out_dt, s);
}

/// The TANH approximation, which is what `FeedForward(activation_fn=
/// "gelu-approximate")` builds. `sdxl_nn.gelu` is the EXACT erf form and is a
/// different function — the two differ in the third decimal, which is small
/// enough to look like noise and large enough to move an image, so they are
/// deliberately not shared.
fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const k_cube = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(k_cube);
    const k_sqrt = mlx.mlx_array_new_float(0.7978845608028654); // sqrt(2/pi)
    defer _ = mlx.mlx_array_free(k_sqrt);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);

    const xf = try nn.astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const x2 = try nn.mulA(xf, xf, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try nn.mulA(x2, xf, s);
    defer _ = mlx.mlx_array_free(x3);
    const cx3 = try nn.mulA(x3, k_cube, s);
    defer _ = mlx.mlx_array_free(cx3);
    const sum = try nn.addA(xf, cx3, s);
    defer _ = mlx.mlx_array_free(sum);
    const inner = try nn.mulA(sum, k_sqrt, s);
    defer _ = mlx.mlx_array_free(inner);
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, inner, s));
    const onep = try nn.addA(th, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const hx = try nn.mulA(xf, half, s);
    defer _ = mlx.mlx_array_free(hx);
    const out = try nn.mulA(hx, onep, s);
    defer _ = mlx.mlx_array_free(out);
    return nn.astype(out, dt, s);
}

/// diffusers' `get_timestep_embedding(t, 256, flip_sin_to_cos=True,
/// downscale_freq_shift=0)`.
///
/// Two conventions here are not the SDXL ones, and both are silent if wrong:
/// the frequency exponent divides by `half` and NOT `half - 1`
/// (`downscale_freq_shift` is 0 here, 1 in `sdxl_unet`), and `flip_sin_to_cos`
/// puts COSINE in the first half. Computed on the host into f32 because it is
/// 256 numbers once per forward, and a scalar read inside a graph build is a
/// GPU barrier.
pub fn timestepEmbedding(t: f32, out: []f32) void {
    const half = out.len / 2;
    const log_max: f64 = @log(@as(f64, 10000.0));
    const denom: f64 = @floatFromInt(half); // downscale_freq_shift = 0
    for (0..half) |i| {
        const exponent = -log_max * @as(f64, @floatFromInt(i)) / denom;
        const angle = @as(f64, t) * @exp(exponent);
        out[i] = @floatCast(@cos(angle));
        out[half + i] = @floatCast(@sin(angle));
    }
}

/// Per-head RMSNorm over the last axis, in f32 like diffusers' `RMSNorm`
/// (`hidden_states.to(float32).pow(2).mean(-1)`), result back in the input's
/// dtype.
fn rmsNormLast(x: mlx.mlx_array, w: mlx.mlx_array, s: S) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(x);
    const xf = try nn.astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const wf = try nn.astype(w, .float32, s);
    defer _ = mlx.mlx_array_free(wf);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_fast_rms_norm(&o, xf, wf, EPS, s));
    return nn.astype(o, dt, s);
}

/// `x * (1 + scale) + shift`, with `scale`/`shift` `[B, D]` broadcast over the
/// sequence axis — diffusers' `x * (1 + scale[:, None]) + shift[:, None]`.
fn modulate(x: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const sc = try nn.addA(scale, one, s);
    defer _ = mlx.mlx_array_free(sc);
    const m = try nn.mulA(x, sc, s);
    defer _ = mlx.mlx_array_free(m);
    return nn.addA(m, shift, s);
}

/// Chunk `[B, n*D]` into `n` pieces of `[B, 1, D]`, ready to broadcast over a
/// token sequence. `torch.chunk(n, dim=1)` order.
fn modChunks(comptime n: usize, mod: mlx.mlx_array, dim: c_int, s: S) ![n]mlx.mlx_array {
    var out: [n]mlx.mlx_array = undefined;
    var filled: usize = 0;
    errdefer for (out[0..filled]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const b = mlx.getShape(mod)[0];
    for (0..n) |i| {
        const piece = try sliceAxis(mod, 1, @intCast(@as(c_int, @intCast(i)) * dim), @intCast(@as(c_int, @intCast(i + 1)) * dim), s);
        defer _ = mlx.mlx_array_free(piece);
        out[i] = try nn.reshape(piece, &[_]c_int{ b, 1, dim }, s);
        filled = i + 1;
    }
    return out;
}

fn freeChunks(chunks: []const mlx.mlx_array) void {
    for (chunks) |c| _ = mlx.mlx_array_free(c);
}

// ════════════════════════════════════════════════════════════════════════
// Modules
// ════════════════════════════════════════════════════════════════════════

/// One `Attention`. The joint attention (`attn`) carries the added-stream
/// projections; MMDiT-X's `attn2` is the same struct with all of them null,
/// which is how one forward serves both.
const Attn = struct {
    q: nn.Linear,
    k: nn.Linear,
    v: nn.Linear,
    out: nn.Linear,
    nq: ?mlx.mlx_array = null,
    nk: ?mlx.mlx_array = null,
    add_q: ?nn.Linear = null,
    add_k: ?nn.Linear = null,
    add_v: ?nn.Linear = null,
    /// `to_add_out` — absent on the truncated last block and on `attn2`.
    add_out: ?nn.Linear = null,
    n_add_q: ?mlx.mlx_array = null,
    n_add_k: ?mlx.mlx_array = null,

    fn deinit(self: *Attn) void {
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.out.deinit();
        inline for (.{ "nq", "nk", "n_add_q", "n_add_k" }) |f| {
            if (@field(self, f)) |x| _ = mlx.mlx_array_free(x);
        }
        inline for (.{ "add_q", "add_k", "add_v", "add_out" }) |f| {
            if (@field(self, f)) |*l| @constCast(l).deinit();
        }
    }
};

const Block = struct {
    /// `norm1.linear` — 6*dim, or 9*dim on an MMDiT-X block.
    norm1: nn.Linear,
    /// `norm1_context.linear` — 6*dim, or 2*dim on the truncated last block.
    norm1_ctx: nn.Linear,
    attn: Attn,
    attn2: ?Attn,
    ff_in: nn.Linear,
    ff_out: nn.Linear,
    ffc_in: ?nn.Linear,
    ffc_out: ?nn.Linear,
    context_pre_only: bool,
    dual: bool,

    fn deinit(self: *Block) void {
        self.norm1.deinit();
        self.norm1_ctx.deinit();
        self.attn.deinit();
        if (self.attn2) |*a| a.deinit();
        self.ff_in.deinit();
        self.ff_out.deinit();
        if (self.ffc_in) |*l| l.deinit();
        if (self.ffc_out) |*l| l.deinit();
    }
};

pub const Mmdit = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    cfg: Config,

    /// `pos_embed.proj` — a conv2d with kernel = stride = `patch_size`.
    patch_w: mlx.mlx_array,
    patch_b: mlx.mlx_array,
    /// `pos_embed.pos_embed` — `[1, max*max, inner]`, a persistent BUFFER in
    /// the checkpoint (not a parameter), stored for the max grid and cropped.
    pos: mlx.mlx_array,

    t_lin1: nn.Linear,
    t_lin2: nn.Linear,
    txt_lin1: nn.Linear,
    txt_lin2: nn.Linear,
    context_embedder: nn.Linear,

    blocks: []Block,

    norm_out: nn.Linear,
    proj_out: nn.Linear,

    pub fn deinit(self: *Mmdit) void {
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.patch_b);
        _ = mlx.mlx_array_free(self.pos);
        self.t_lin1.deinit();
        self.t_lin2.deinit();
        self.txt_lin1.deinit();
        self.txt_lin2.deinit();
        self.context_embedder.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.norm_out.deinit();
        self.proj_out.deinit();
        const a = self.allocator;
        a.destroy(self);
    }

    pub fn config(self: *const Mmdit) sd3.MmditConfig {
        return self.cfg.base;
    }

    // ── Forward ──────────────────────────────────────────────────────────

    /// `latent [B,16,h,w]`, `enc [B,T,4096]`, `pooled [B,2048]`,
    /// `timestep = sigma*1000`. Returns the velocity `[B,16,h,w]`.
    ///
    /// B is 1 or 2 and nothing here loops over guidance: the caller stacks the
    /// conditional and unconditional halves and this runs them as one batch.
    pub fn forward(
        self: *Mmdit,
        latent: mlx.mlx_array,
        enc: mlx.mlx_array,
        pooled: mlx.mlx_array,
        timestep: f32,
        s: S,
    ) !mlx.mlx_array {
        const lsh = mlx.getShape(latent);
        if (lsh.len != 4) return error.BadLatentRank;
        const p: c_int = @intCast(self.cfg.base.patch_size);
        const grid_h = @divTrunc(lsh[2], p);
        const grid_w = @divTrunc(lsh[3], p);

        var img = try self.patchEmbed(latent, grid_h, grid_w, s);
        defer _ = mlx.mlx_array_free(img);

        const cond = try self.conditioning(timestep, pooled, s);
        defer _ = mlx.mlx_array_free(cond);

        const enc_c = try nn.astype(enc, self.dtype, s);
        defer _ = mlx.mlx_array_free(enc_c);
        var txt: ?mlx.mlx_array = try self.context_embedder.forward(enc_c, s);
        defer if (txt) |t| {
            _ = mlx.mlx_array_free(t);
        };

        for (self.blocks) |*b| {
            const r = try self.block(b, img, txt.?, cond, s);
            nn.replace(&img, r.img);
            // The truncated last block returns no text stream at all — the
            // diffusers forward sets `encoder_hidden_states = None` there. It
            // is the LAST block, so nothing reads it; a null is honest where a
            // stale copy would quietly become an input if a block were ever
            // appended after it.
            _ = mlx.mlx_array_free(txt.?);
            txt = r.txt;
        }

        // `norm_out` is an AdaLayerNormContinuous: SCALE first, then shift.
        {
            const act = try nn.silu(cond, s);
            defer _ = mlx.mlx_array_free(act);
            const mod = try self.norm_out.forward(act, s);
            defer _ = mlx.mlx_array_free(mod);
            const ch = try modChunks(2, mod, @intCast(self.cfg.base.innerDim()), s);
            defer freeChunks(&ch);
            const ln = try layerNormNA(img, EPS, s);
            defer _ = mlx.mlx_array_free(ln);
            nn.replace(&img, try modulate(ln, ch[0], ch[1], s));
        }
        nn.replace(&img, try self.proj_out.forward(img, s));

        return self.unpatchify(img, grid_h, grid_w, s);
    }

    /// Conv-patchify then ADD the centre-cropped positional embedding.
    fn patchEmbed(self: *Mmdit, latent: mlx.mlx_array, grid_h: c_int, grid_w: c_int, s: S) !mlx.mlx_array {
        const cast = try nn.astype(latent, self.dtype, s);
        defer _ = mlx.mlx_array_free(cast);
        const nhwc = try nn.nchwToNhwc(cast, s);
        defer _ = mlx.mlx_array_free(nhwc);

        const p: c_int = @intCast(self.cfg.base.patch_size);
        var conv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv);
        // kernel == stride == patch_size, no padding: the patches tile exactly.
        try mlx.check(mlx.mlx_conv2d(&conv, nhwc, self.patch_w, p, p, 0, 0, 1, 1, 1, s));
        const biased = try nn.addA(conv, self.patch_b, s);
        defer _ = mlx.mlx_array_free(biased);

        // `[B, gh, gw, inner]` is already row-major over the patch grid, which
        // is what torch's `flatten(2).transpose(1, 2)` of `[B, inner, gh, gw]`
        // produces. Same tokens, same order, one fewer permute.
        const b = mlx.getShape(biased)[0];
        const inner: c_int = @intCast(self.cfg.base.innerDim());
        const tokens = try nn.reshape(biased, &[_]c_int{ b, grid_h * grid_w, inner }, s);
        defer _ = mlx.mlx_array_free(tokens);

        const pe = try self.croppedPosEmbed(grid_h, grid_w, s);
        defer _ = mlx.mlx_array_free(pe);
        return nn.addA(tokens, pe, s);
    }

    /// The CENTRE crop. `sd3.posEmbedCropStart` owns the rule; this is the
    /// slice it names, taken out of the stored `max x max` grid and flattened
    /// row-major.
    fn croppedPosEmbed(self: *Mmdit, grid_h: c_int, grid_w: c_int, s: S) !mlx.mlx_array {
        const max: c_int = @intCast(self.cfg.base.pos_embed_max_size);
        if (grid_h > max or grid_w > max) return error.CanvasExceedsPosEmbed;
        const inner: c_int = @intCast(self.cfg.base.innerDim());
        const top: c_int = @intCast(sd3.posEmbedCropStart(self.cfg.base, @intCast(grid_h)));
        const left: c_int = @intCast(sd3.posEmbedCropStart(self.cfg.base, @intCast(grid_w)));

        const grid = try nn.reshape(self.pos, &[_]c_int{ 1, max, max, inner }, s);
        defer _ = mlx.mlx_array_free(grid);
        const rows = try sliceAxis(grid, 1, top, top + grid_h, s);
        defer _ = mlx.mlx_array_free(rows);
        const cols = try sliceAxis(rows, 2, left, left + grid_w, s);
        defer _ = mlx.mlx_array_free(cols);
        return nn.reshape(cols, &[_]c_int{ 1, grid_h * grid_w, inner }, s);
    }

    /// `CombinedTimestepTextProjEmbeddings`: the sinusoidal timestep through a
    /// silu MLP, PLUS the pooled CLIP vector through its own silu MLP, ADDED.
    ///
    /// The timestep half is `[1, inner]` and the pooled half is `[B, inner]`;
    /// the add broadcasts, which is exactly right — a CFG batch shares its
    /// timestep and differs only in its conditioning.
    fn conditioning(self: *Mmdit, timestep: f32, pooled: mlx.mlx_array, s: S) !mlx.mlx_array {
        var proj: [TIME_PROJ_CHANNELS]f32 = undefined;
        timestepEmbedding(timestep, &proj);
        const shape = [_]c_int{ 1, @intCast(TIME_PROJ_CHANNELS) };
        const t_arr = mlx.mlx_array_new_data(&proj, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(t_arr);
        const t_cast = try nn.astype(t_arr, self.dtype, s);
        defer _ = mlx.mlx_array_free(t_cast);

        var t_emb = try self.t_lin1.forward(t_cast, s);
        defer _ = mlx.mlx_array_free(t_emb);
        {
            const act = try nn.silu(t_emb, s);
            defer _ = mlx.mlx_array_free(act);
            nn.replace(&t_emb, try self.t_lin2.forward(act, s));
        }

        const pooled_c = try nn.astype(pooled, self.dtype, s);
        defer _ = mlx.mlx_array_free(pooled_c);
        var p_emb = try self.txt_lin1.forward(pooled_c, s);
        defer _ = mlx.mlx_array_free(p_emb);
        {
            const act = try nn.silu(p_emb, s);
            defer _ = mlx.mlx_array_free(act);
            nn.replace(&p_emb, try self.txt_lin2.forward(act, s));
        }

        return nn.addA(t_emb, p_emb, s);
    }

    /// `[B, N, heads*hd] -> [B, heads, N, hd]`, with the optional per-head
    /// RMSNorm folded in (queries and keys take it; values never do).
    fn splitHeads(self: *Mmdit, x: mlx.mlx_array, norm: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const heads: c_int = @intCast(self.cfg.base.num_attention_heads);
        const hd: c_int = @intCast(self.cfg.base.attention_head_dim);
        const r = try nn.reshape(x, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
        defer _ = mlx.mlx_array_free(r);
        const t = try nn.transpose(r, &[_]c_int{ 0, 2, 1, 3 }, s);
        if (norm) |w| {
            defer _ = mlx.mlx_array_free(t);
            return rmsNormLast(t, w, s);
        }
        return t;
    }

    /// `[B, heads, N, hd] -> [B, N, heads*hd]`.
    fn mergeHeads(self: *Mmdit, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x); // [B, heads, N, hd]
        const t = try nn.transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(t);
        return nn.reshape(t, &[_]c_int{ sh[0], sh[2], @intCast(self.cfg.base.innerDim()) }, s);
    }

    fn sdpa(self: *Mmdit, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, s: S) !mlx.mlx_array {
        const hd: f32 = @floatFromInt(self.cfg.base.attention_head_dim);
        const scale: f32 = 1.0 / std.math.sqrt(hd);
        const null_a = mlx.mlx_array{ .ctx = null };
        var o = mlx.mlx_array_new();
        // Full bidirectional attention over the joint sequence: no mask, and
        // no causality. SD 3 pads its text to a fixed length and attends to the
        // padding on purpose — that is the trained behaviour, not an oversight.
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", null_a, null_a, false, s));
        return o;
    }

    fn feedForward(self: *Mmdit, x: mlx.mlx_array, in: *const nn.Linear, out: *const nn.Linear, s: S) !mlx.mlx_array {
        _ = self;
        const h = try in.forward(x, s);
        defer _ = mlx.mlx_array_free(h);
        const a = try geluTanh(h, s);
        defer _ = mlx.mlx_array_free(a);
        return out.forward(a, s);
    }

    const BlockOut = struct { img: mlx.mlx_array, txt: ?mlx.mlx_array };

    fn block(self: *Mmdit, b: *const Block, img_in: mlx.mlx_array, txt_in: mlx.mlx_array, cond: mlx.mlx_array, s: S) !BlockOut {
        const dim: c_int = @intCast(self.cfg.base.innerDim());
        const act = try nn.silu(cond, s);
        defer _ = mlx.mlx_array_free(act);

        // ── Image modulation. Nine chunks on an MMDiT-X block, six otherwise;
        // the first six mean the same thing in both.
        const im_mod = try b.norm1.forward(act, s);
        defer _ = mlx.mlx_array_free(im_mod);
        var im: [9]mlx.mlx_array = undefined;
        const im_n: usize = if (b.dual) 9 else 6;
        if (b.dual) {
            im = try modChunks(9, im_mod, dim, s);
        } else {
            const six = try modChunks(6, im_mod, dim, s);
            @memcpy(im[0..6], &six);
        }
        defer freeChunks(im[0..im_n]);

        const img_ln = try layerNormNA(img_in, EPS, s);
        defer _ = mlx.mlx_array_free(img_ln);
        const img_norm = try modulate(img_ln, im[1], im[0], s);
        defer _ = mlx.mlx_array_free(img_norm);
        // MMDiT-X's second stream re-modulates the SAME LayerNorm output — the
        // norm runs once, `SD35AdaLayerNormZeroX` just scales and shifts it
        // twice. Re-normalizing here would be a different network.
        const img_norm2: ?mlx.mlx_array = if (b.dual) try modulate(img_ln, im[7], im[6], s) else null;
        defer if (img_norm2) |x| {
            _ = mlx.mlx_array_free(x);
        };

        // ── Text modulation. The truncated last block's `norm1_context` is an
        // AdaLayerNormContinuous: two chunks, SCALE FIRST.
        var ct: [6]mlx.mlx_array = undefined;
        var ct_n: usize = 0;
        const ct_mod = try b.norm1_ctx.forward(act, s);
        defer _ = mlx.mlx_array_free(ct_mod);
        const txt_ln = try layerNormNA(txt_in, EPS, s);
        defer _ = mlx.mlx_array_free(txt_ln);
        var txt_norm: mlx.mlx_array = undefined;
        if (b.context_pre_only) {
            const two = try modChunks(2, ct_mod, dim, s);
            defer freeChunks(&two);
            txt_norm = try modulate(txt_ln, two[0], two[1], s);
        } else {
            ct = try modChunks(6, ct_mod, dim, s);
            ct_n = 6;
            txt_norm = try modulate(txt_ln, ct[1], ct[0], s);
        }
        defer freeChunks(ct[0..ct_n]);
        defer _ = mlx.mlx_array_free(txt_norm);

        // ── Joint attention: both streams project separately, CONCATENATE
        // along the sequence with the IMAGE FIRST, attend once, split back.
        const n_img = mlx.getShape(img_norm)[1];
        const attn_img, const attn_txt = blk: {
            const a = &b.attn;
            const q = try a.q.forward(img_norm, s);
            defer _ = mlx.mlx_array_free(q);
            const k = try a.k.forward(img_norm, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try a.v.forward(img_norm, s);
            defer _ = mlx.mlx_array_free(v);
            const qh = try self.splitHeads(q, a.nq, s);
            defer _ = mlx.mlx_array_free(qh);
            const kh = try self.splitHeads(k, a.nk, s);
            defer _ = mlx.mlx_array_free(kh);
            const vh = try self.splitHeads(v, null, s);
            defer _ = mlx.mlx_array_free(vh);

            const eq = try a.add_q.?.forward(txt_norm, s);
            defer _ = mlx.mlx_array_free(eq);
            const ek = try a.add_k.?.forward(txt_norm, s);
            defer _ = mlx.mlx_array_free(ek);
            const ev = try a.add_v.?.forward(txt_norm, s);
            defer _ = mlx.mlx_array_free(ev);
            const eqh = try self.splitHeads(eq, a.n_add_q, s);
            defer _ = mlx.mlx_array_free(eqh);
            const ekh = try self.splitHeads(ek, a.n_add_k, s);
            defer _ = mlx.mlx_array_free(ekh);
            const evh = try self.splitHeads(ev, null, s);
            defer _ = mlx.mlx_array_free(evh);

            const qj = try nn.concat(&[_]mlx.mlx_array{ qh, eqh }, 2, s);
            defer _ = mlx.mlx_array_free(qj);
            const kj = try nn.concat(&[_]mlx.mlx_array{ kh, ekh }, 2, s);
            defer _ = mlx.mlx_array_free(kj);
            const vj = try nn.concat(&[_]mlx.mlx_array{ vh, evh }, 2, s);
            defer _ = mlx.mlx_array_free(vj);

            const raw = try self.sdpa(qj, kj, vj, s);
            defer _ = mlx.mlx_array_free(raw);
            const merged = try self.mergeHeads(raw, s);
            defer _ = mlx.mlx_array_free(merged);
            const n_all = mlx.getShape(merged)[1];

            const part_img = try sliceAxis(merged, 1, 0, n_img, s);
            defer _ = mlx.mlx_array_free(part_img);
            const part_txt = try sliceAxis(merged, 1, n_img, n_all, s);
            defer _ = mlx.mlx_array_free(part_txt);

            const oi = try a.out.forward(part_img, s);
            errdefer _ = mlx.mlx_array_free(oi);
            const ot: ?mlx.mlx_array = if (a.add_out) |*ao| try ao.forward(part_txt, s) else null;
            break :blk .{ oi, ot };
        };
        defer _ = mlx.mlx_array_free(attn_img);
        defer if (attn_txt) |x| {
            _ = mlx.mlx_array_free(x);
        };

        // ── Image residual: gated joint attention, then (MMDiT-X) gated dual
        // attention, then the gated feed-forward.
        var img = blk: {
            const g = try nn.mulA(im[2], attn_img, s);
            defer _ = mlx.mlx_array_free(g);
            break :blk try nn.addA(img_in, g, s);
        };
        errdefer _ = mlx.mlx_array_free(img);

        if (b.attn2) |*a2| {
            const x = img_norm2 orelse return error.MissingDualModulation;
            const q = try a2.q.forward(x, s);
            defer _ = mlx.mlx_array_free(q);
            const k = try a2.k.forward(x, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try a2.v.forward(x, s);
            defer _ = mlx.mlx_array_free(v);
            const qh = try self.splitHeads(q, a2.nq, s);
            defer _ = mlx.mlx_array_free(qh);
            const kh = try self.splitHeads(k, a2.nk, s);
            defer _ = mlx.mlx_array_free(kh);
            const vh = try self.splitHeads(v, null, s);
            defer _ = mlx.mlx_array_free(vh);
            const raw = try self.sdpa(qh, kh, vh, s);
            defer _ = mlx.mlx_array_free(raw);
            const merged = try self.mergeHeads(raw, s);
            defer _ = mlx.mlx_array_free(merged);
            const o = try a2.out.forward(merged, s);
            defer _ = mlx.mlx_array_free(o);
            const g = try nn.mulA(im[8], o, s);
            defer _ = mlx.mlx_array_free(g);
            nn.replace(&img, try nn.addA(img, g, s));
        }

        {
            const ln = try layerNormNA(img, EPS, s);
            defer _ = mlx.mlx_array_free(ln);
            const m = try modulate(ln, im[4], im[3], s);
            defer _ = mlx.mlx_array_free(m);
            const ff = try self.feedForward(m, &b.ff_in, &b.ff_out, s);
            defer _ = mlx.mlx_array_free(ff);
            const g = try nn.mulA(im[5], ff, s);
            defer _ = mlx.mlx_array_free(g);
            nn.replace(&img, try nn.addA(img, g, s));
        }

        // ── Text residual, unless this block throws the text stream away.
        if (b.context_pre_only) return .{ .img = img, .txt = null };

        var txt = blk: {
            const g = try nn.mulA(ct[2], attn_txt.?, s);
            defer _ = mlx.mlx_array_free(g);
            break :blk try nn.addA(txt_in, g, s);
        };
        errdefer _ = mlx.mlx_array_free(txt);
        {
            const ln = try layerNormNA(txt, EPS, s);
            defer _ = mlx.mlx_array_free(ln);
            const m = try modulate(ln, ct[4], ct[3], s);
            defer _ = mlx.mlx_array_free(m);
            const ff = try self.feedForward(m, &b.ffc_in.?, &b.ffc_out.?, s);
            defer _ = mlx.mlx_array_free(ff);
            const g = try nn.mulA(ct[5], ff, s);
            defer _ = mlx.mlx_array_free(g);
            nn.replace(&txt, try nn.addA(txt, g, s));
        }
        return .{ .img = img, .txt = txt };
    }

    /// `[B, N, p*p*C] -> [B, C, gh*p, gw*p]`, diffusers'
    /// `einsum("nhwpqc->nchpwq")`.
    fn unpatchify(self: *Mmdit, x: mlx.mlx_array, grid_h: c_int, grid_w: c_int, s: S) !mlx.mlx_array {
        const b = mlx.getShape(x)[0];
        const p: c_int = @intCast(self.cfg.base.patch_size);
        const c: c_int = @intCast(self.cfg.base.out_channels);
        const r = try nn.reshape(x, &[_]c_int{ b, grid_h, grid_w, p, p, c }, s);
        defer _ = mlx.mlx_array_free(r);
        // n h w p q c  ->  n c h p w q
        const t = try nn.transpose(r, &[_]c_int{ 0, 5, 1, 3, 2, 4 }, s);
        defer _ = mlx.mlx_array_free(t);
        return nn.reshape(t, &[_]c_int{ b, c, grid_h * p, grid_w * p }, s);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Loading
// ════════════════════════════════════════════════════════════════════════

/// Load the transformer from `<model_dir>/transformer`, reading its own
/// `config.json`.
pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []const u8,
    dtype: mlx.mlx_dtype,
) !*Mmdit {
    const dir = try nn.fmtKey(allocator, "{s}/transformer", .{model_dir});
    defer allocator.free(dir);
    const cfg_path = try nn.fmtKey(allocator, "{s}/config.json", .{dir});
    defer allocator.free(cfg_path);
    // `openFileAbsolute` ASSERTS its path is absolute, and a failed assert is
    // ReleaseFast UB that miscompiles the caller. `model_dir` reaches here from
    // a request, so guard before the call rather than after it.
    if (cfg_path.len == 0 or !std.fs.path.isAbsolute(cfg_path)) return error.MissingMmditConfig;
    const f = std.Io.Dir.openFileAbsolute(io, cfg_path, .{}) catch |e| {
        log.err("[sd3] cannot open {s}: {s}\n", .{ cfg_path, @errorName(e) });
        return error.MissingMmditConfig;
    };
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return error.MissingMmditConfig;
    defer allocator.free(bytes);
    const cfg = try parseConfig(allocator, bytes);

    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();

    return loadFromWeights(allocator, s, &w, cfg, dtype);
}

/// The weight-binding half, split out so a test can drive it from an already
/// loaded `Weights` without a second read of a multi-gigabyte file.
pub fn loadFromWeights(
    allocator: std.mem.Allocator,
    s: S,
    w: *Weights,
    cfg: Config,
    dtype: mlx.mlx_dtype,
) !*Mmdit {
    const m = try allocator.create(Mmdit);
    errdefer allocator.destroy(m);
    m.* = .{
        .allocator = allocator,
        .s = s,
        .dtype = dtype,
        .cfg = cfg,
        .patch_w = undefined,
        .patch_b = undefined,
        .pos = undefined,
        .t_lin1 = undefined,
        .t_lin2 = undefined,
        .txt_lin1 = undefined,
        .txt_lin2 = undefined,
        .context_embedder = undefined,
        .blocks = &[_]Block{},
        .norm_out = undefined,
        .proj_out = undefined,
    };

    // The patch conv is a real conv2d and its weight MUST go through
    // `loadConvWeight` — PyTorch stores `[out, in, kh, kw]`, MLX reads
    // `[out, kh, kw, in]`, and a mis-permuted conv weight is a valid tensor of
    // the right size, so nothing errors and the image is noise.
    m.patch_w = try nn.loadConvWeight(w, "pos_embed.proj.weight", dtype, s);
    errdefer _ = mlx.mlx_array_free(m.patch_w);
    m.patch_b = try nn.dupWeight(w, "pos_embed.proj.bias", dtype, s);
    errdefer _ = mlx.mlx_array_free(m.patch_b);
    // The positional table is added to the tokens BEFORE 38 residual blocks, so
    // it is kept at f32 whatever the tower runs at: it costs
    // `max*max*inner*4` bytes once and the add casts down for free.
    m.pos = try nn.dupWeight(w, "pos_embed.pos_embed", .float32, s);
    errdefer _ = mlx.mlx_array_free(m.pos);

    m.t_lin1 = try nn.loadLinear(w, "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias", dtype, s);
    errdefer m.t_lin1.deinit();
    m.t_lin2 = try nn.loadLinear(w, "time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias", dtype, s);
    errdefer m.t_lin2.deinit();
    m.txt_lin1 = try nn.loadLinear(w, "time_text_embed.text_embedder.linear_1.weight", "time_text_embed.text_embedder.linear_1.bias", dtype, s);
    errdefer m.txt_lin1.deinit();
    m.txt_lin2 = try nn.loadLinear(w, "time_text_embed.text_embedder.linear_2.weight", "time_text_embed.text_embedder.linear_2.bias", dtype, s);
    errdefer m.txt_lin2.deinit();
    m.context_embedder = try nn.loadLinear(w, "context_embedder.weight", "context_embedder.bias", dtype, s);
    errdefer m.context_embedder.deinit();

    const blocks = try allocator.alloc(Block, cfg.base.num_layers);
    errdefer allocator.free(blocks);
    var bound: usize = 0;
    errdefer for (blocks[0..bound]) |*b| b.deinit();
    for (blocks, 0..) |*b, i| {
        const pfx = try nn.fmtKey(allocator, "transformer_blocks.{d}", .{i});
        defer allocator.free(pfx);
        b.* = try loadBlock(w, allocator, pfx, cfg, @intCast(i), dtype, s);
        bound = i + 1;
    }
    m.blocks = blocks;

    m.norm_out = try nn.loadLinear(w, "norm_out.linear.weight", "norm_out.linear.bias", dtype, s);
    errdefer m.norm_out.deinit();
    m.proj_out = try nn.loadLinear(w, "proj_out.weight", "proj_out.bias", dtype, s);

    return m;
}

fn linAt(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8, dtype: mlx.mlx_dtype, s: S) !nn.Linear {
    const kw = try nn.fmtKey(a, "{s}.{s}.weight", .{ pfx, sub });
    defer a.free(kw);
    const kb = try nn.fmtKey(a, "{s}.{s}.bias", .{ pfx, sub });
    defer a.free(kb);
    return nn.loadLinear(w, kw, kb, dtype, s);
}

fn normAt(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, sub: []const u8, dtype: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const k = try nn.fmtKey(a, "{s}.{s}.weight", .{ pfx, sub });
    defer a.free(k);
    return nn.dupWeight(w, k, dtype, s);
}

fn loadBlock(
    w: *Weights,
    a: std.mem.Allocator,
    pfx: []const u8,
    cfg: Config,
    index: u32,
    dtype: mlx.mlx_dtype,
    s: S,
) !Block {
    const last = index + 1 == cfg.base.num_layers;
    const dual = cfg.isDual(index);

    var b = Block{
        .norm1 = undefined,
        .norm1_ctx = undefined,
        .attn = undefined,
        .attn2 = null,
        .ff_in = undefined,
        .ff_out = undefined,
        .ffc_in = null,
        .ffc_out = null,
        .context_pre_only = last,
        .dual = dual,
    };

    b.norm1 = try linAt(w, a, pfx, "norm1.linear", dtype, s);
    errdefer b.norm1.deinit();
    b.norm1_ctx = try linAt(w, a, pfx, "norm1_context.linear", dtype, s);
    errdefer b.norm1_ctx.deinit();

    // A shape check rather than a comment: `norm1.linear` is 9*dim exactly on
    // the blocks `dual_attention_layers` names and 6*dim everywhere else, and
    // `norm1_context.linear` is 2*dim exactly on the truncated last block. If
    // the config and the weights ever disagree about which block is which,
    // every chunk below lands on the wrong quantity and NOTHING errors — the
    // network still runs and the image is subtly wrong. So it is caught here.
    {
        const dim: c_int = @intCast(cfg.base.innerDim());
        const want_im: c_int = if (dual) 9 * dim else 6 * dim;
        const want_ct: c_int = if (last) 2 * dim else 6 * dim;
        // `w_t` is `[in, out]` (dense) or the packed `[out, ...]` (quantized);
        // the bias is `[out]` under both, so it is what the check reads.
        const got_im = mlx.getShape(b.norm1.b.?)[0];
        const got_ct = mlx.getShape(b.norm1_ctx.b.?)[0];
        if (got_im != want_im or got_ct != want_ct) {
            log.err("[sd3] block {d}: modulation widths {d}/{d}, expected {d}/{d} (dual={}, last={})\n", .{ index, got_im, got_ct, want_im, want_ct, dual, last });
            return error.MmditModulationMismatch;
        }
    }

    const ap = try nn.fmtKey(a, "{s}.attn", .{pfx});
    defer a.free(ap);
    b.attn = .{
        .q = try linAt(w, a, ap, "to_q", dtype, s),
        .k = undefined,
        .v = undefined,
        .out = undefined,
    };
    errdefer b.attn.deinit();
    b.attn.k = try linAt(w, a, ap, "to_k", dtype, s);
    b.attn.v = try linAt(w, a, ap, "to_v", dtype, s);
    b.attn.out = try linAt(w, a, ap, "to_out.0", dtype, s);
    b.attn.add_q = try linAt(w, a, ap, "add_q_proj", dtype, s);
    b.attn.add_k = try linAt(w, a, ap, "add_k_proj", dtype, s);
    b.attn.add_v = try linAt(w, a, ap, "add_v_proj", dtype, s);
    // `to_add_out` exists on every block BUT the last: diffusers' `Attention`
    // builds it only when `context_pre_only` is false, so demanding it on the
    // last block is a missing-weight load error.
    if (!last) b.attn.add_out = try linAt(w, a, ap, "to_add_out", dtype, s);
    if (cfg.base.qk_norm) {
        b.attn.nq = try normAt(w, a, ap, "norm_q", dtype, s);
        b.attn.nk = try normAt(w, a, ap, "norm_k", dtype, s);
        b.attn.n_add_q = try normAt(w, a, ap, "norm_added_q", dtype, s);
        b.attn.n_add_k = try normAt(w, a, ap, "norm_added_k", dtype, s);
    }

    if (dual) {
        const a2p = try nn.fmtKey(a, "{s}.attn2", .{pfx});
        defer a.free(a2p);
        var a2 = Attn{
            .q = try linAt(w, a, a2p, "to_q", dtype, s),
            .k = undefined,
            .v = undefined,
            .out = undefined,
        };
        errdefer a2.deinit();
        a2.k = try linAt(w, a, a2p, "to_k", dtype, s);
        a2.v = try linAt(w, a, a2p, "to_v", dtype, s);
        a2.out = try linAt(w, a, a2p, "to_out.0", dtype, s);
        // `attn2` is image-only self attention: it has `norm_q`/`norm_k` and no
        // added-stream projections and no `norm_added_*` at all.
        if (cfg.base.qk_norm) {
            a2.nq = try normAt(w, a, a2p, "norm_q", dtype, s);
            a2.nk = try normAt(w, a, a2p, "norm_k", dtype, s);
        }
        b.attn2 = a2;
    }
    errdefer if (b.attn2) |*x| x.deinit();

    b.ff_in = try linAt(w, a, pfx, "ff.net.0.proj", dtype, s);
    errdefer b.ff_in.deinit();
    b.ff_out = try linAt(w, a, pfx, "ff.net.2", dtype, s);
    errdefer b.ff_out.deinit();
    if (!last) {
        b.ffc_in = try linAt(w, a, pfx, "ff_context.net.0.proj", dtype, s);
        errdefer b.ffc_in.?.deinit();
        b.ffc_out = try linAt(w, a, pfx, "ff_context.net.2", dtype, s);
    }
    return b;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sd3 mmdit config: the SHAPE comes from the config, and dual_attention_layers is a LIST" {
    const a = testing.allocator;

    // Large / Large-Turbo, verbatim from
    // `stabilityai/stable-diffusion-3.5-large`'s transformer/config.json.
    const large =
        \\{"_class_name":"SD3Transformer2DModel","attention_head_dim":64,
        \\"caption_projection_dim":2432,"in_channels":16,"joint_attention_dim":4096,
        \\"num_attention_heads":38,"num_layers":38,"out_channels":16,"patch_size":2,
        \\"pooled_projection_dim":2048,"pos_embed_max_size":192,"qk_norm":"rms_norm",
        \\"sample_size":128}
    ;
    const lc = try parseConfig(a, large);
    try testing.expectEqual(@as(u32, 38), lc.base.num_layers);
    try testing.expectEqual(@as(u32, 2432), lc.base.innerDim());
    try testing.expectEqual(@as(u32, 192), lc.base.pos_embed_max_size);
    try testing.expect(lc.base.qk_norm);
    // No `dual_attention_layers` at all: NOT MMDiT-X.
    try testing.expectEqual(@as(u64, 0), lc.dual_mask);
    for (0..38) |i| try testing.expect(!lc.isDual(@intCast(i)));

    // Medium. Note `pos_embed_max_size` 384 and the dual list — same class,
    // materially different network.
    const medium =
        \\{"_class_name":"SD3Transformer2DModel","attention_head_dim":64,
        \\"caption_projection_dim":1536,"in_channels":16,"joint_attention_dim":4096,
        \\"num_attention_heads":24,"num_layers":24,"out_channels":16,"patch_size":2,
        \\"pooled_projection_dim":2048,"pos_embed_max_size":384,"qk_norm":"rms_norm",
        \\"sample_size":128,
        \\"dual_attention_layers":[0,1,2,3,4,5,6,7,8,9,10,11,12]}
    ;
    const mc = try parseConfig(a, medium);
    try testing.expectEqual(@as(u32, 24), mc.base.num_layers);
    try testing.expectEqual(@as(u32, 1536), mc.base.innerDim());
    try testing.expectEqual(@as(u32, 384), mc.base.pos_embed_max_size);
    for (0..13) |i| try testing.expect(mc.isDual(@intCast(i)));
    for (13..24) |i| try testing.expect(!mc.isDual(@intCast(i)));

    // A diffusers export writes the field as an EMPTY list rather than
    // omitting it, and that must read the same as absent.
    const empty =
        \\{"attention_head_dim":64,"caption_projection_dim":2432,"in_channels":16,
        \\"joint_attention_dim":4096,"num_attention_heads":38,"num_layers":38,
        \\"out_channels":16,"patch_size":2,"pooled_projection_dim":2048,
        \\"pos_embed_max_size":192,"qk_norm":"rms_norm","sample_size":128,
        \\"dual_attention_layers":[]}
    ;
    const ec = try parseConfig(a, empty);
    try testing.expectEqual(@as(u64, 0), ec.dual_mask);

    // An unknown `qk_norm` is refused by name rather than silently read as
    // "off" — SD 3.0's qk-norm-less blocks are a different network, not a
    // smaller one.
    const bad =
        \\{"attention_head_dim":64,"caption_projection_dim":2432,"in_channels":16,
        \\"joint_attention_dim":4096,"num_attention_heads":38,"num_layers":38,
        \\"out_channels":16,"patch_size":2,"pooled_projection_dim":2048,
        \\"pos_embed_max_size":192,"qk_norm":"layer_norm","sample_size":128}
    ;
    try testing.expectError(error.UnsupportedQkNorm, parseConfig(a, bad));

    // A caption width that does not equal the block width would need a second
    // width threaded through every block; refused rather than mis-bound.
    const split =
        \\{"attention_head_dim":64,"caption_projection_dim":1152,"in_channels":16,
        \\"joint_attention_dim":4096,"num_attention_heads":38,"num_layers":38,
        \\"out_channels":16,"patch_size":2,"pooled_projection_dim":2048,
        \\"pos_embed_max_size":192,"qk_norm":"rms_norm","sample_size":128}
    ;
    try testing.expectError(error.UnsupportedCaptionWidth, parseConfig(a, split));
}

test "sd3 mmdit: the timestep sinusoid is COS-first and divides by half, not half-1" {
    // `flip_sin_to_cos: True` and `downscale_freq_shift: 0` — and the second is
    // the one that differs from SDXL's identically-shaped embedding, which
    // divides by `half - 1`. Both are silent when wrong.
    var buf: [TIME_PROJ_CHANNELS]f32 = undefined;
    timestepEmbedding(628.5, &buf);
    const half = TIME_PROJ_CHANNELS / 2;
    // Frequency 0 is exp(0) = 1, so index 0 is cos(t) and index `half` is
    // sin(t). The unflipped order would swap them.
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(@as(f64, 628.5)))), buf[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(@as(f64, 628.5)))), buf[half], 1e-5);
    // The last frequency is exp(-ln(10000)*(half-1)/half), which is strictly
    // GREATER than 1/10000. Dividing by `half - 1` instead lands exactly on
    // 1/10000 — that is the whole observable difference, so pin it.
    const want_last = @exp(-@log(@as(f64, 10000.0)) * @as(f64, @floatFromInt(half - 1)) / @as(f64, @floatFromInt(half)));
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(628.5 * want_last))), buf[half - 1], 1e-5);
    try testing.expect(want_last > 1.0 / 10000.0);
    for (buf) |v| try testing.expect(v >= -1.0001 and v <= 1.0001);
}

test "sd3 mmdit: gelu-approximate is the TANH form, not the erf one" {
    // `FeedForward(activation_fn="gelu-approximate")`. `sdxl_nn.gelu` is the
    // exact erf GELU and is a different function; the two differ in the third
    // decimal, which is small enough to look like noise.
    const s = mlx.mlx_default_gpu_stream_new();
    const vals = [_]f32{ -2, -0.5, 0, 0.5, 2 };
    const shape = [_]c_int{5};
    const x = mlx.mlx_array_new_data(&vals, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(x);
    const got = try geluTanh(x, s);
    defer _ = mlx.mlx_array_free(got);
    _ = mlx.mlx_array_eval(got);
    const p = mlx.mlx_array_data_float32(got).?;
    // Independently computed from the tanh formula (CPython), NOT re-derived
    // from the implementation: a reference that reuses the code's own formula
    // proves only that the formula is stable.
    const want = [_]f32{ -0.04540229, -0.15428599, 0.0, 0.34571400, 1.95459771 };
    for (want, 0..) |wv, i| try testing.expectApproxEqAbs(wv, p[i], 1e-5);
    // And it is NOT the erf form at x = -2 (erf gives -0.04550026).
    const erf_at_m2: f32 = -0.04550026;
    try testing.expect(@abs(p[0] - erf_at_m2) > 1e-5);
}

// ── Numerical PARITY against diffusers' own SD3Transformer2DModel ────────
//
// The fixture directory is produced by `tests/dump_sd3_mmdit_fixtures.py`, in
// float32 on CPU, and holds a complete little checkpoint plus every structural
// intermediate. The two shapes are separate fixtures because they are separate
// networks:
//
//   .../sd3venv/bin/python tests/dump_sd3_mmdit_fixtures.py build \
//       --out ~/.mlx-serve/staging/sd3_mmdit_tiny_large
//   .../sd3venv/bin/python tests/dump_sd3_mmdit_fixtures.py build --dual \
//       --out ~/.mlx-serve/staging/sd3_mmdit_tiny_medium
//
//   SD3_MMDIT_FIXTURE_DIR=~/.mlx-serve/staging/sd3_mmdit_tiny_large \
//     zig build test -Doptimize=ReleaseFast -Dtest-filter="sd3 mmdit fixture"
//   SD3_MMDIT_DUAL_FIXTURE_DIR=~/.mlx-serve/staging/sd3_mmdit_tiny_medium \
//     zig build test -Doptimize=ReleaseFast -Dtest-filter="sd3 mmdit dual"
//
// Env-gated and skipped when unset, exactly like the `sdxl unet fixture` tests.
test "sd3 mmdit fixture: the Large shape matches diffusers, block by block" {
    const dir = std.mem.span(std.c.getenv("SD3_MMDIT_FIXTURE_DIR") orelse return error.SkipZigTest);
    try runFixture(dir, false);
}

test "sd3 mmdit dual fixture: the MMDiT-X (Medium) shape matches diffusers" {
    const dir = std.mem.span(std.c.getenv("SD3_MMDIT_DUAL_FIXTURE_DIR") orelse return error.SkipZigTest);
    try runFixture(dir, true);
}

fn runFixture(dir: []const u8, want_dual: bool) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const fx_path = try nn.fmtKey(a, "{s}/fixture.safetensors", .{dir});
    defer a.free(fx_path);
    var fx = try model_mod.loadWeightsSingleFile(a, fx_path);
    defer fx.deinit();

    var m = try load(io, a, s, dir, .float32);
    defer m.deinit();

    // The fixture's own shape must be the shape under test. A dual fixture run
    // against a non-dual load would silently be a weaker test, not a failure.
    try testing.expectEqual(want_dual, m.cfg.dual_mask != 0);

    // Every input comes FROM the fixture, so the two sides cannot be running
    // different forwards over inputs that merely look alike.
    const latent = fx.get("in.hidden_states") orelse return error.MissingFixtureLatent;
    const enc = fx.get("in.encoder_hidden_states") orelse return error.MissingFixtureEnc;
    const pooled = fx.get("in.pooled_projections") orelse return error.MissingFixturePooled;
    const ts_arr = fx.get("in.timestep") orelse return error.MissingFixtureTimestep;
    _ = mlx.mlx_array_eval(ts_arr);
    const timestep = (mlx.mlx_array_data_float32(ts_arr) orelse return error.BadFixtureTimestep)[0];

    // ── Intermediates first. A tower that disagrees only at the end is a
    // needle in a haystack; checking inward-out names the block.
    const lsh = mlx.getShape(latent);
    const p: c_int = @intCast(m.cfg.base.patch_size);
    const gh = @divTrunc(lsh[2], p);
    const gw = @divTrunc(lsh[3], p);

    var img = try m.patchEmbed(latent, gh, gw, s);
    defer _ = mlx.mlx_array_free(img);
    try expectClose("pos_embed", img, fx.get("cap.pos_embed").?, s);

    const cond = try m.conditioning(timestep, pooled, s);
    defer _ = mlx.mlx_array_free(cond);
    try expectClose("temb", cond, fx.get("cap.temb").?, s);

    var txt: ?mlx.mlx_array = try m.context_embedder.forward(enc, s);
    defer if (txt) |t| {
        _ = mlx.mlx_array_free(t);
    };
    try expectClose("context", txt.?, fx.get("cap.context").?, s);

    var name_buf: [64]u8 = undefined;
    for (m.blocks, 0..) |*b, i| {
        const r = try m.block(b, img, txt.?, cond, s);
        nn.replace(&img, r.img);
        _ = mlx.mlx_array_free(txt.?);
        txt = r.txt;

        const ik = try std.fmt.bufPrint(&name_buf, "cap.block{d}.img", .{i});
        try expectClose(ik, img, fx.get(ik) orelse return error.MissingFixtureBlock, s);
        if (txt) |t| {
            const tk = try std.fmt.bufPrint(&name_buf, "cap.block{d}.txt", .{i});
            try expectClose(tk, t, fx.get(tk) orelse return error.MissingFixtureBlock, s);
        } else {
            // The last block throws the text stream away, and the fixture
            // therefore has no `.txt` capture for it. If a `.txt` key IS there,
            // our `context_pre_only` landed on the wrong block.
            const tk = try std.fmt.bufPrint(&name_buf, "cap.block{d}.txt", .{i});
            try testing.expect(fx.get(tk) == null);
        }
    }

    // ── And the whole forward end to end, which also covers `norm_out`,
    // `proj_out` and the unpatchify the loop above stops short of.
    const out = try m.forward(latent, enc, pooled, timestep, s);
    defer _ = mlx.mlx_array_free(out);
    const ref = fx.get("out.sample") orelse return error.MissingFixtureOutput;
    try testing.expectEqualSlices(c_int, mlx.getShape(ref), mlx.getShape(out));
    try expectClose("out.sample", out, ref, s);
}

/// Cosine, rms_ratio AND a position-weighted checksum.
///
/// All three are load-bearing here and each is blind to what the others catch:
/// a cosine cannot see a SCALE error; an rms ratio cannot see a PERMUTATION
/// (swapping the joint-attention concat order to FLUX's text-first leaves every
/// norm identical); and neither notices a systematic offset the way max_abs
/// does. The position weighting is the repo's `pos_weighted` rule — a
/// permutation-invariant checksum cannot see a permutation.
fn expectClose(what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const g_arr = try flatF32(got, s);
    defer _ = mlx.mlx_array_free(g_arr);
    const r_arr = try flatF32(ref, s);
    defer _ = mlx.mlx_array_free(r_arr);

    const n = mlx.mlx_array_size(g_arr);
    try testing.expectEqual(n, mlx.mlx_array_size(r_arr));
    const g = mlx.mlx_array_data_float32(g_arr).?;
    const r = mlx.mlx_array_data_float32(r_arr).?;

    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    var max_abs: f64 = 0;
    var wg: f64 = 0;
    var wr: f64 = 0;
    for (0..n) |i| {
        const gv: f64 = g[i];
        const rv: f64 = r[i];
        // Finiteness FIRST: `NaN > bar` is false, so an all-NaN tensor scores
        // zero and reads as a mismatch rather than as the crash it is.
        try testing.expect(std.math.isFinite(gv));
        try testing.expect(std.math.isFinite(rv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
        max_abs = @max(max_abs, @abs(gv - rv));
        // An index-varying weight: any reordering of the elements moves this
        // even when every norm above is untouched.
        const wt: f64 = @floatFromInt(i % 1021);
        wg += wt * gv;
        wr += wt * rv;
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    const denom = @max(@abs(wr), 1e-6);
    const pos_rel = @abs(wg - wr) / denom;
    std.debug.print("[sd3-parity] {s}: cos={d:.6} rms_ratio={d:.6} max_abs={d:.6} pos_rel={d:.6}\n", .{ what, cos, rms_ratio, max_abs, pos_rel });

    try testing.expect(cos > 0.9999);
    try testing.expect(rms_ratio > 0.999 and rms_ratio < 1.001);
    try testing.expect(pos_rel < 1e-3);
}

fn flatF32(x: mlx.mlx_array, s: S) !mlx.mlx_array {
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
