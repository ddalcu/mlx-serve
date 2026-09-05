//! SDXL's VAE decoder — diffusers `AutoencoderKL`, latents to pixels.
//!
//! `flux.zig` already carries an oracle-validated `AutoencoderKL` decoder, and
//! an earlier note in `sdxl.zig` proposed reusing it. That note counted the
//! weight names (138 of 140 identical) and concluded the gap was three small
//! deltas. The names do match; the FORWARD does not, and the difference is
//! structural rather than parametric:
//!
//!   - FLUX.2's latent is PATCHIFIED. `flux.Vae.decode` opens with an
//!     `unpatchify` from `[1,128,h,w]` to `[1,32,2h,2w]`, hardcoded to those
//!     channel counts. SDXL's latent is a plain `[1,4,h,w]` — there is no
//!     patch structure to undo.
//!   - FLUX.2 denormalizes with `bn.running_mean`/`running_var`. SDXL ships no
//!     `bn.*` tensors and scales by a single `scaling_factor` instead.
//!   - FLUX.2's attention reads through `QLinear` (its packs are 4-bit); SDXL's
//!     VAE is dense fp16.
//!
//! Reusing it would mean threading three conditionals through the hot path of
//! a decoder whose correctness is the one thing in that file with an oracle
//! behind it. The topology below is genuinely shared — and it IS shared, via
//! `sdxl_nn.Resnet` — but the decode chain is its own.
//!
//! `force_upcast: true` in the checkpoint's config is load-bearing and is why
//! `DEFAULT_DTYPE` is float32: this VAE overflows fp16 on real latents. The
//! activations reach the tens of thousands in the mid block, and fp16 tops out
//! at 65504 — the failure is a black or banded image, not an error. Diffusers
//! carries the same flag for the same reason.
//!
//! ORACLE STATUS: pinned against diffusers' own `AutoencoderKL.decode` by
//! `tests/dump_sdxl_unet_fixtures.py`; see `sdxl vae parity` below.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const nn = @import("sdxl_nn.zig");
const sdxl = @import("sdxl.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;
const Resnet = nn.Resnet;

/// diffusers' `AutoencoderKL` resnets and norms use eps 1e-6 — NOT the UNet's
/// 1e-5. Same module class, different configured value.
const VAE_EPS: f32 = 1e-6;

/// The VAE runs float32 because the checkpoint says `force_upcast: true`.
/// See the file header: fp16 here is a silently black image.
pub const DEFAULT_DTYPE: mlx.mlx_dtype = .float32;

/// Single-head self-attention over the spatial grid, as the VAE's mid block
/// uses it.
const VaeAttn = struct {
    gnw: mlx.mlx_array,
    gnb: mlx.mlx_array,
    q: nn.Linear,
    k: nn.Linear,
    v: nn.Linear,
    o: nn.Linear,

    fn deinit(self: *VaeAttn) void {
        _ = mlx.mlx_array_free(self.gnw);
        _ = mlx.mlx_array_free(self.gnb);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
    }

    fn forward(self: *const VaeAttn, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        const h_ = sh[1];
        const w_ = sh[2];
        const c = sh[3];

        const normed = try nn.groupNorm(x, self.gnw, self.gnb, 32, VAE_EPS, s);
        defer _ = mlx.mlx_array_free(normed);
        // Flatten to tokens; NHWC makes this a pure reshape.
        const tokens = try nn.reshape(normed, &[_]c_int{ 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(tokens);

        const q = try self.q.forward(tokens, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.k.forward(tokens, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.v.forward(tokens, s);
        defer _ = mlx.mlx_array_free(v);

        // ONE head over the full channel width — this is `AttnProcessor` with
        // heads=1, not the UNet's multi-head attention.
        const qr = try nn.reshape(q, &[_]c_int{ 1, 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try nn.reshape(k, &[_]c_int{ 1, 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try nn.reshape(v, &[_]c_int{ 1, 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(vr);

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c)));
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        const null_a = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vr, scale, "", null_a, null_a, false, s));

        const flat = try nn.reshape(attn, &[_]c_int{ 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(flat);
        const proj = try self.o.forward(flat, s);
        defer _ = mlx.mlx_array_free(proj);
        const spatial = try nn.reshape(proj, &[_]c_int{ 1, h_, w_, c }, s);
        defer _ = mlx.mlx_array_free(spatial);
        return nn.addA(x, spatial, s);
    }
};

/// One decoder stage: `layers_per_block + 1` resnets, then an optional 2x
/// upsample. The last stage does not upsample.
const UpBlock = struct {
    resnets: []Resnet,
    up_w: ?mlx.mlx_array,
    up_b: ?mlx.mlx_array,
    allocator: std.mem.Allocator,

    fn deinit(self: *UpBlock) void {
        for (self.resnets) |*r| r.deinit();
        self.allocator.free(self.resnets);
        if (self.up_w) |x| _ = mlx.mlx_array_free(x);
        if (self.up_b) |x| _ = mlx.mlx_array_free(x);
    }
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    /// `scaling_factor` from the VAE's own config, 0.13025 on SDXL.
    scaling_factor: f32,

    pq_w: mlx.mlx_array,
    pq_b: mlx.mlx_array,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: VaeAttn,
    mid_r1: Resnet,
    up: []UpBlock,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Decoder) void {
        inline for (.{ "pq_w", "pq_b", "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b" }) |f| {
            _ = mlx.mlx_array_free(@field(self, f));
        }
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
        for (self.up) |*u| u.deinit();
        self.allocator.free(self.up);
    }

    /// Decode a RAW latent — one still scaled by `scaling_factor`, i.e. exactly
    /// what the sampler produces. The division happens here so no caller has to
    /// remember it.
    ///
    /// `latent` is NCHW `[1, 4, h, w]`; the result is NCHW `[1, 3, 8h, 8w]` in
    /// the model's own [-1, 1] range.
    pub fn decodeLatent(self: *const Decoder, latent: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const inv = mlx.mlx_array_new_float(1.0 / self.scaling_factor);
        defer _ = mlx.mlx_array_free(inv);
        const cast = try nn.astype(latent, self.dtype, s);
        defer _ = mlx.mlx_array_free(cast);
        const scaled = try nn.mulA(cast, inv, s);
        defer _ = mlx.mlx_array_free(scaled);
        return self.decode(scaled);
    }

    /// Decode an ALREADY-unscaled latent. Split from `decodeLatent` so the
    /// parity test can drive the exact tensor diffusers was handed.
    pub fn decode(self: *const Decoder, latent: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const cast = try nn.astype(latent, self.dtype, s);
        defer _ = mlx.mlx_array_free(cast);
        const nhwc = try nn.nchwToNhwc(cast, s);
        defer _ = mlx.mlx_array_free(nhwc);

        // post_quant_conv is 1x1 — padding 0, not 1.
        var h = try nn.conv2d(nhwc, self.pq_w, self.pq_b, 0, s);
        errdefer _ = mlx.mlx_array_free(h);
        nn.replace(&h, try nn.conv2d(h, self.conv_in_w, self.conv_in_b, 1, s));

        nn.replace(&h, try self.mid_r0.forward(h, null, s));
        nn.replace(&h, try self.mid_attn.forward(h, s));
        nn.replace(&h, try self.mid_r1.forward(h, null, s));

        for (self.up) |*blk| {
            for (blk.resnets) |*r| {
                nn.replace(&h, try r.forward(h, null, s));
            }
            if (blk.up_w) |uw| {
                nn.replace(&h, try nn.upsample2x(h, uw, blk.up_b.?, s));
            }
        }

        {
            const n = try nn.groupNorm(h, self.norm_out_w, self.norm_out_b, 32, VAE_EPS, s);
            nn.replace(&h, n);
        }
        nn.replace(&h, try nn.silu(h, s));
        nn.replace(&h, try nn.conv2d(h, self.conv_out_w, self.conv_out_b, 1, s));

        defer _ = mlx.mlx_array_free(h);
        return nn.nhwcToNchw(h, s);
    }
};

/// One down-block stage: two resnets, then an optional asymmetric-pad
/// stride-2 downsample. The last stage does not downsample.
const DownBlock = struct {
    resnets: []Resnet,
    down_w: ?mlx.mlx_array,
    down_b: ?mlx.mlx_array,
    allocator: std.mem.Allocator,

    fn deinit(self: *DownBlock) void {
        for (self.resnets) |*r| r.deinit();
        self.allocator.free(self.resnets);
        if (self.down_w) |x| _ = mlx.mlx_array_free(x);
        if (self.down_b) |x| _ = mlx.mlx_array_free(x);
    }
};

/// The encoder half of `AutoencoderKL` — pixels to latents, for img2img. A
/// structural mirror of `Decoder`: same resnet/attn/norm topology, down
/// instead of up. Unlike FLUX.2's encoder (`flux.VaeEncoder`), there is no
/// `bn.*` normalization to invert — SDXL's own `scaling_factor` is the whole
/// convention, matching what `Decoder.decodeLatent` divides back out.
pub const Encoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    scaling_factor: f32,

    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    down: []DownBlock,
    mid_r0: Resnet,
    mid_attn: VaeAttn,
    mid_r1: Resnet,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    quant_w: mlx.mlx_array, // top-level quant_conv, 1x1: 2*latent -> 2*latent
    quant_b: mlx.mlx_array,

    pub fn deinit(self: *Encoder) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b", "quant_w", "quant_b" }) |f| {
            _ = mlx.mlx_array_free(@field(self, f));
        }
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
        for (self.down) |*d| d.deinit();
        self.allocator.free(self.down);
    }

    /// image `[1,3,H,W]` f32 `[0,1]` (NCHW) -> raw latent `[1,4,H/8,W/8]`
    /// f32, scaled by `scaling_factor` — the SAME space `Decoder.decodeLatent`
    /// expects and the diffusion loop's own `latent` lives in. Deterministic:
    /// the distribution MEAN, never sampled (same convention as `flux.zig`'s
    /// and `krea.zig`'s encoders).
    pub fn encode(self: *const Encoder, img: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const two = mlx.mlx_array_new_float(2.0);
        defer _ = mlx.mlx_array_free(two);
        const one = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(one);
        const x2 = try nn.mulA(img, two, s);
        defer _ = mlx.mlx_array_free(x2);
        const xm = try nn.subA(x2, one, s);
        defer _ = mlx.mlx_array_free(xm);
        const nhwc = try nn.nchwToNhwc(xm, s);
        defer _ = mlx.mlx_array_free(nhwc);
        const xc = try nn.astype(nhwc, self.dtype, s);
        defer _ = mlx.mlx_array_free(xc);

        var h = try nn.conv2d(xc, self.conv_in_w, self.conv_in_b, 1, s);
        errdefer _ = mlx.mlx_array_free(h);

        for (self.down) |*blk| {
            for (blk.resnets) |*r| {
                nn.replace(&h, try r.forward(h, null, s));
            }
            if (blk.down_w) |dw| {
                nn.replace(&h, try conv2dDown(h, dw, blk.down_b.?, s));
            }
        }

        nn.replace(&h, try self.mid_r0.forward(h, null, s));
        nn.replace(&h, try self.mid_attn.forward(h, s));
        nn.replace(&h, try self.mid_r1.forward(h, null, s));

        {
            const n = try nn.groupNorm(h, self.norm_out_w, self.norm_out_b, 32, VAE_EPS, s);
            nn.replace(&h, n);
        }
        nn.replace(&h, try nn.silu(h, s));
        nn.replace(&h, try nn.conv2d(h, self.conv_out_w, self.conv_out_b, 1, s)); // [1,H,W,8]
        nn.replace(&h, try nn.conv2d(h, self.quant_w, self.quant_b, 0, s)); // [1,H,W,8]

        // mean = first LATENT_CHANNELS of the last axis; logvar discarded —
        // encode() is deterministic (the mode, never sampled).
        const hsh = mlx.getShape(h);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const stop = [_]c_int{ hsh[0], hsh[1], hsh[2], @intCast(sdxl.LATENT_CHANNELS) };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_slice(&mean, h, &start, 4, &stop, 4, &strides, 4, s));
        _ = mlx.mlx_array_free(h);
        const meanc = try nn.contiguous(mean, s);
        defer _ = mlx.mlx_array_free(meanc);
        const nchw = try nn.nhwcToNchw(meanc, s);
        defer _ = mlx.mlx_array_free(nchw);
        const nf = try nn.astype(nchw, .float32, s);
        defer _ = mlx.mlx_array_free(nf);

        const scale = mlx.mlx_array_new_float(self.scaling_factor);
        defer _ = mlx.mlx_array_free(scale);
        return nn.mulA(nf, scale, s);
    }
};

/// Asymmetric (0,1,0,1) zero-pad + 3x3 stride-2 valid conv on NHWC —
/// diffusers `Downsample2D` as the VAE's encoder configures it (padding 0,
/// not the UNet's symmetric-1 `nn.conv2dStride2`).
fn conv2dDown(x: mlx.mlx_array, w: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    const axes = [_]c_int{ 1, 2 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ 1, 1 };
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_pad(&p, x, &axes, 2, &low, 2, &high, 2, zero, "constant", s));
    const pc = try nn.contiguous(p, s);
    defer _ = mlx.mlx_array_free(pc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, pc, w, 2, 2, 0, 0, 1, 1, 1, s));
    const r = try nn.addA(o, bias, s);
    _ = mlx.mlx_array_free(o);
    return r;
}

/// Load the encoder half of `<model_dir>/vae` — img2img's source-image path.
/// A pack whose `vae/` carries no `encoder.*` tensors (unusual, but nothing
/// enforces they ship) fails cleanly; callers treat that as "no img2img".
pub fn loadEncoder(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []const u8,
    dtype: mlx.mlx_dtype,
) !Encoder {
    const dir = try nn.fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    const scaling = readScalingFactor(io, allocator, dir) orelse sdxl.VAE_SCALING_FACTOR;
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    return loadEncoderFromWeights(allocator, s, &w, scaling, dtype);
}

/// The weight-binding half, split out so a single-file checkpoint can drive
/// it from an in-memory `Weights` map — same reason `loadFromWeights` exists
/// beside `load` for the decoder.
pub fn loadEncoderFromWeights(
    allocator: std.mem.Allocator,
    s: S,
    w: *Weights,
    scaling: f32,
    dtype: mlx.mlx_dtype,
) !Encoder {
    const stages = countDownBlocks(w, allocator);
    if (stages == 0) return error.NoVaeDownBlocks;

    var e: Encoder = undefined;
    e.allocator = allocator;
    e.s = s;
    e.dtype = dtype;
    e.scaling_factor = scaling;

    e.conv_in_w = try nn.loadConvWeight(w, "encoder.conv_in.weight", dtype, s);
    e.conv_in_b = try nn.dupWeight(w, "encoder.conv_in.bias", dtype, s);

    e.down = try allocator.alloc(DownBlock, stages);
    errdefer allocator.free(e.down);
    for (e.down, 0..) |*blk, bi| {
        blk.allocator = allocator;
        blk.down_w = null;
        blk.down_b = null;
        const n_res = countEncoderResnets(w, allocator, bi);
        blk.resnets = try allocator.alloc(Resnet, n_res);
        for (blk.resnets, 0..) |*r, ri| {
            const pfx = try nn.fmtKey(allocator, "encoder.down_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            r.* = try nn.loadResnet(w, allocator, pfx, false, VAE_EPS, dtype, s);
        }
        const kw = try nn.fmtKey(allocator, "encoder.down_blocks.{d}.downsamplers.0.conv.weight", .{bi});
        defer allocator.free(kw);
        if (w.get(kw) != null) {
            const kb = try nn.fmtKey(allocator, "encoder.down_blocks.{d}.downsamplers.0.conv.bias", .{bi});
            defer allocator.free(kb);
            blk.down_w = try nn.loadConvWeight(w, kw, dtype, s);
            blk.down_b = try nn.dupWeight(w, kb, dtype, s);
        }
    }

    e.mid_r0 = try nn.loadResnet(w, allocator, "encoder.mid_block.resnets.0", false, VAE_EPS, dtype, s);
    e.mid_r1 = try nn.loadResnet(w, allocator, "encoder.mid_block.resnets.1", false, VAE_EPS, dtype, s);
    e.mid_attn = try loadVaeAttn(w, allocator, "encoder.mid_block.attentions.0", dtype, s);

    e.norm_out_w = try nn.dupWeight(w, "encoder.conv_norm_out.weight", dtype, s);
    e.norm_out_b = try nn.dupWeight(w, "encoder.conv_norm_out.bias", dtype, s);
    e.conv_out_w = try nn.loadConvWeight(w, "encoder.conv_out.weight", dtype, s);
    e.conv_out_b = try nn.dupWeight(w, "encoder.conv_out.bias", dtype, s);
    e.quant_w = try nn.loadConvWeight(w, "quant_conv.weight", dtype, s);
    e.quant_b = try nn.dupWeight(w, "quant_conv.bias", dtype, s);

    log.info("[sdxl] loaded vae encoder: stages={d} scaling={d:.5}\n", .{ stages, scaling });
    return e;
}

fn countDownBlocks(w: *const Weights, a: std.mem.Allocator) usize {
    var n: usize = 0;
    while (n < 16) : (n += 1) {
        const k = nn.fmtKey(a, "encoder.down_blocks.{d}.resnets.0.conv1.weight", .{n}) catch return n;
        defer a.free(k);
        if (w.get(k) == null) return n;
    }
    return n;
}

fn countEncoderResnets(w: *const Weights, a: std.mem.Allocator, block: usize) usize {
    var n: usize = 0;
    while (n < 16) : (n += 1) {
        const k = nn.fmtKey(a, "encoder.down_blocks.{d}.resnets.{d}.conv1.weight", .{ block, n }) catch return n;
        defer a.free(k);
        if (w.get(k) == null) return n;
    }
    return n;
}

/// Load the decoder half of `<model_dir>/vae`.
///
/// Only the decoder is bound by default; `loadEncoder`/`loadEncoderFromWeights`
/// above bind the `encoder.*` tensors in the same file for img2img.
pub fn load(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: S,
    model_dir: []const u8,
    dtype: mlx.mlx_dtype,
) !Decoder {
    const dir = try nn.fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);

    const scaling = readScalingFactor(io, allocator, dir) orelse sdxl.VAE_SCALING_FACTOR;

    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();

    return loadFromWeights(allocator, s, &w, scaling, dtype);
}

/// The weight-binding half, split out so a single-file checkpoint can drive it
/// from an in-memory `Weights` map (the diffusers keys are identical either
/// way — the converter produces exactly what `load` would have read from disk).
pub fn loadFromWeights(
    allocator: std.mem.Allocator,
    s: S,
    w: *Weights,
    scaling: f32,
    dtype: mlx.mlx_dtype,
) !Decoder {
    // Stage count comes from the checkpoint's own `up_blocks.N` numbering, so
    // a VAE with a different depth loads rather than being assumed to be 4.
    const stages = countUpBlocks(w, allocator);
    if (stages == 0) return error.NoVaeUpBlocks;

    var d: Decoder = undefined;
    d.allocator = allocator;
    d.s = s;
    d.dtype = dtype;
    d.scaling_factor = scaling;

    d.pq_w = try nn.loadConvWeight(w, "post_quant_conv.weight", dtype, s);
    d.pq_b = try nn.dupWeight(w, "post_quant_conv.bias", dtype, s);
    d.conv_in_w = try nn.loadConvWeight(w, "decoder.conv_in.weight", dtype, s);
    d.conv_in_b = try nn.dupWeight(w, "decoder.conv_in.bias", dtype, s);

    d.mid_r0 = try nn.loadResnet(w, allocator, "decoder.mid_block.resnets.0", false, VAE_EPS, dtype, s);
    d.mid_r1 = try nn.loadResnet(w, allocator, "decoder.mid_block.resnets.1", false, VAE_EPS, dtype, s);
    d.mid_attn = try loadVaeAttn(w, allocator, "decoder.mid_block.attentions.0", dtype, s);

    d.up = try allocator.alloc(UpBlock, stages);
    errdefer allocator.free(d.up);
    for (d.up, 0..) |*blk, bi| {
        blk.allocator = allocator;
        blk.up_w = null;
        blk.up_b = null;
        const n_res = countResnets(w, allocator, bi);
        blk.resnets = try allocator.alloc(Resnet, n_res);
        for (blk.resnets, 0..) |*r, ri| {
            const pfx = try nn.fmtKey(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            r.* = try nn.loadResnet(w, allocator, pfx, false, VAE_EPS, dtype, s);
        }
        // Every stage but the last carries an upsampler.
        const kw = try nn.fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi});
        defer allocator.free(kw);
        if (w.get(kw) != null) {
            const kb = try nn.fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi});
            defer allocator.free(kb);
            blk.up_w = try nn.loadConvWeight(w, kw, dtype, s);
            blk.up_b = try nn.dupWeight(w, kb, dtype, s);
        }
    }

    d.norm_out_w = try nn.dupWeight(w, "decoder.conv_norm_out.weight", dtype, s);
    d.norm_out_b = try nn.dupWeight(w, "decoder.conv_norm_out.bias", dtype, s);
    d.conv_out_w = try nn.loadConvWeight(w, "decoder.conv_out.weight", dtype, s);
    d.conv_out_b = try nn.dupWeight(w, "decoder.conv_out.bias", dtype, s);

    log.info("[sdxl] loaded vae decoder: stages={d} scaling={d:.5}\n", .{ stages, scaling });
    return d;
}

/// SDXL ships `to_out.0.*`; some conversions flatten it to `to_out.*`. Both
/// spellings are accepted because the difference is a converter's choice, not
/// a different module — `lora.zig` already normalizes this exact pair.
fn loadVaeAttn(
    w: *const Weights,
    a: std.mem.Allocator,
    prefix: []const u8,
    dt: mlx.mlx_dtype,
    s: S,
) !VaeAttn {
    const K = struct {
        fn lin(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8, d: mlx.mlx_dtype, st: S) !nn.Linear {
            const kw = try nn.fmtKey(aa, "{s}.{s}.weight", .{ p, sub });
            defer aa.free(kw);
            const kb = try nn.fmtKey(aa, "{s}.{s}.bias", .{ p, sub });
            defer aa.free(kb);
            return nn.loadLinear(ww, kw, kb, d, st);
        }
    };

    const gnw = try nn.fmtKey(a, "{s}.group_norm.weight", .{prefix});
    defer a.free(gnw);
    const gnb = try nn.fmtKey(a, "{s}.group_norm.bias", .{prefix});
    defer a.free(gnb);

    var at: VaeAttn = undefined;
    at.gnw = try nn.dupWeight(w, gnw, dt, s);
    errdefer _ = mlx.mlx_array_free(at.gnw);
    at.gnb = try nn.dupWeight(w, gnb, dt, s);
    errdefer _ = mlx.mlx_array_free(at.gnb);
    at.q = try K.lin(w, a, prefix, "to_q", dt, s);
    errdefer at.q.deinit();
    at.k = try K.lin(w, a, prefix, "to_k", dt, s);
    errdefer at.k.deinit();
    at.v = try K.lin(w, a, prefix, "to_v", dt, s);
    errdefer at.v.deinit();

    const indexed = try nn.fmtKey(a, "{s}.to_out.0.weight", .{prefix});
    defer a.free(indexed);
    at.o = if (w.get(indexed) != null)
        try K.lin(w, a, prefix, "to_out.0", dt, s)
    else
        try K.lin(w, a, prefix, "to_out", dt, s);
    return at;
}

fn countUpBlocks(w: *const Weights, a: std.mem.Allocator) usize {
    var n: usize = 0;
    while (n < 16) : (n += 1) {
        const k = nn.fmtKey(a, "decoder.up_blocks.{d}.resnets.0.conv1.weight", .{n}) catch return n;
        defer a.free(k);
        if (w.get(k) == null) return n;
    }
    return n;
}

fn countResnets(w: *const Weights, a: std.mem.Allocator, block: usize) usize {
    var n: usize = 0;
    while (n < 16) : (n += 1) {
        const k = nn.fmtKey(a, "decoder.up_blocks.{d}.resnets.{d}.conv1.weight", .{ block, n }) catch return n;
        defer a.free(k);
        if (w.get(k) == null) return n;
    }
    return n;
}

/// Read `scaling_factor` from the VAE's own `config.json`. A pack that omits
/// it falls back to SDXL's documented constant.
fn readScalingFactor(io: std.Io, allocator: std.mem.Allocator, vae_dir: []const u8) ?f32 {
    const path = std.fmt.allocPrint(allocator, "{s}/config.json", .{vae_dir}) catch return null;
    defer allocator.free(path);
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return null;
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 20)) catch return null;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get("scaling_factor") orelse return null;
    return switch (v) {
        .float => @floatCast(v.float),
        .integer => @floatFromInt(v.integer),
        else => null,
    };
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sdxl vae: the decoder's eps differs from the UNet's" {
    // Same module class upstream, different configured value. One eps for both
    // files is a small uniform error that no shape check can see.
    try testing.expect(VAE_EPS != 1e-5);
    try testing.expectEqual(@as(f32, 1e-6), VAE_EPS);
    // force_upcast: the VAE overflows fp16 on real latents.
    try testing.expectEqual(mlx.mlx_dtype.float32, DEFAULT_DTYPE);
}

// Numerical PARITY against diffusers' own AutoencoderKL.decode, over the
// fixture's latent.
//
//   SDXL_CHECKPOINT_DIR=~/.mlx-serve/staging/sdxl-base-1.0 \
//   SDXL_UNET_FIXTURE=~/.mlx-serve/staging/sdxl_unet_fixture.safetensors \
//     zig build test -Dtest-filter="sdxl vae parity"
test "sdxl vae parity: decode matches diffusers" {
    const dir = std.mem.span(std.c.getenv("SDXL_CHECKPOINT_DIR") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("SDXL_UNET_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    // The fixture stores the POST-division tensor diffusers was handed, so the
    // comparison isolates the decoder from the scaling convention.
    const latent = fx.get("in.vae_latent") orelse return error.MissingFixtureLatent;
    const ref = fx.get("out.vae_decoded") orelse return error.MissingFixtureDecoded;

    var dec = try load(io, a, s, dir, DEFAULT_DTYPE);
    defer dec.deinit();

    const out = try dec.decode(latent);
    defer _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_eval(out);

    try testing.expectEqualSlices(c_int, mlx.getShape(ref), mlx.getShape(out));
    try expectCloseVae("vae", "decoded", out, ref, s);

    // The scaling convention itself: decodeLatent must reproduce decode() on
    // the pre-divided latent. A decoder that silently scales twice still
    // produces an image, just a washed one.
    const scaled = blk: {
        const f = mlx.mlx_array_new_float(dec.scaling_factor);
        defer _ = mlx.mlx_array_free(f);
        break :blk try nn.mulA(latent, f, s);
    };
    defer _ = mlx.mlx_array_free(scaled);
    const out2 = try dec.decodeLatent(scaled);
    defer _ = mlx.mlx_array_free(out2);
    _ = mlx.mlx_array_eval(out2);
    try expectCloseVae("vae", "decodeLatent round trip", out2, ref, s);
}

fn expectCloseVae(who: []const u8, what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const g_arr = try flatF32Vae(got, s);
    defer _ = mlx.mlx_array_free(g_arr);
    const r_arr = try flatF32Vae(ref, s);
    defer _ = mlx.mlx_array_free(r_arr);
    const n = mlx.mlx_array_size(g_arr);
    try testing.expectEqual(n, mlx.mlx_array_size(r_arr));
    const g = mlx.mlx_array_data_float32(g_arr).?;
    const r = mlx.mlx_array_data_float32(r_arr).?;

    var dot: f64 = 0;
    var ng: f64 = 0;
    var nr: f64 = 0;
    var max_abs: f64 = 0;
    for (0..n) |i| {
        const gv: f64 = g[i];
        const rv: f64 = r[i];
        try testing.expect(std.math.isFinite(gv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
        max_abs = @max(max_abs, @abs(gv - rv));
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    std.debug.print("[sdxl-parity] {s} {s}: cos={d:.6} rms_ratio={d:.6} max_abs={d:.5}\n", .{ who, what, cos, rms_ratio, max_abs });
    try testing.expect(cos > 0.9999);
    try testing.expect(rms_ratio > 0.995 and rms_ratio < 1.005);
}

fn flatF32Vae(x: mlx.mlx_array, s: S) !mlx.mlx_array {
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
