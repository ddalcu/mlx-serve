//! SD 3.5's VAE — a 16-channel `AutoencoderKL` with no quant convs.
//!
//! Structurally this IS the VAE `sdxl_vae.zig` already serves: same
//! `AutoencoderKL` class, `block_out_channels` [128, 256, 512, 512],
//! `layers_per_block` 2, `norm_num_groups` 32, silu, mid-block attention on,
//! `force_upcast: true`. Three fields differ, and all three change numbers:
//!
//!   1. `latent_channels: 16`, not 4. Nothing here reads that as a constant —
//!      the decoder's `conv_in` and the encoder's `conv_out` are bound from
//!      the checkpoint's own shapes, so the width follows the weights. It
//!      matters in exactly one place: the encoder slicing the mean out of the
//!      `2 * latent` moments.
//!   2. `use_quant_conv: false` / `use_post_quant_conv: false`. SDXL runs a 1x1
//!      conv either side of the latent; SD 3.5 does not, and those tensors are
//!      simply absent from the checkpoint (verified: the real
//!      `vae/diffusion_pytorch_model.safetensors` has 244 keys and not one
//!      matches `*quant_conv*`). Binding one that is not there is a
//!      missing-weight error; skipping one that IS there silently changes the
//!      latent basis. So it is a CONFIG decision — `Config` is read from
//!      `vae/config.json` and defaults to diffusers' own default (true), which
//!      means a pack that drops the field fails LOUDLY on the missing tensor
//!      rather than quietly decoding in the wrong basis.
//!   3. The latent is SHIFTED as well as scaled (`z / scaling + shift`). That
//!      is `sd3.decodeScale` / `sd3.encodeScale` and it is the PIPELINE's
//!      arithmetic — `decode` here takes an already-unscaled latent and
//!      `encodeMean` returns an unscaled mean, so neither the constants nor
//!      the convention are restated in this file.
//!
//! WHY THE DECODER IS `sdxl_vae.Decoder` AND THE ENCODER IS NOT.
//!
//! `sdxl_vae`'s decode chain is the one piece of this family with a diffusers
//! oracle behind it, and `sd1_pipeline.zig` already established the pattern of
//! reusing that machinery at a second family's config rather than forking it.
//! It reuses cleanly here because everything that differs is read from the
//! weights — except `post_quant_conv`, which its `Decoder` binds
//! unconditionally. Rather than fork 200 lines to make one field optional,
//! `loadFromWeights` synthesizes an IDENTITY 1x1 conv when the config says the
//! checkpoint has none. That is a bit-exact no-op (each output channel is one
//! `1.0 *` term plus `latent-1` `0.0 *` terms plus a zero bias, all exact in
//! f32), it is asserted as one below, and it costs one 16-channel 1x1 conv per
//! decode. It is a WORKAROUND: the real fix is `Decoder.pq_w`/`pq_b` becoming
//! `?mlx.mlx_array` gated on the same config flag, and that lives in another
//! lane's file.
//!
//! The encoder could NOT be reused. `sdxl_vae.Encoder.encode` remaps its input
//! from [0,1] to [-1,1], slices the mean at `sdxl.LATENT_CHANNELS` (4 — a
//! silent 12-channel truncation here) and multiplies by `scaling_factor`. All
//! three are SDXL's conventions, not shared ones, so the encode chain below is
//! this file's own — built out of the same `sdxl_nn` pieces (`Resnet`,
//! `groupNorm`, `conv2d`), with local copies of the two helpers `sdxl_vae`
//! keeps private: the mid-block attention and the asymmetric-pad downsample.
//! Those two are the duplication this file would shed if `sdxl_vae` exported
//! them.
//!
//! `DEFAULT_DTYPE` is float32 because `force_upcast: true`. Same reason as
//! SDXL's: this VAE's mid-block activations exceed fp16's 65504, and the
//! failure is a black image, not an error.
//!
//! ORACLE STATUS:
//!
//!   VERIFIED against the checkpoint. The config fields and every weight name
//!   below were read out of `adamo1139/stable-diffusion-3.5-large-ungated`'s
//!   own `vae/config.json` and safetensors header (an exact ungated mirror of
//!   the gated stability repo), not from documentation.
//!
//!   VERIFIED against diffusers' NUMBERS, at the tiny scale. `sd3 vae parity`
//!   below runs against a fixture built by `tests/dump_sd3_vae_fixtures.py
//!   build` — a random-weight `AutoencoderKL` of the real class at SD 3.5's
//!   config shape (16 latent channels, no quant convs, 4 stages, 2 resnets per
//!   stage, mid-block attention on). Both directions clear cos > 0.9999 and
//!   rms_ratio within 0.5%. That pins the ARCHITECTURE and the weight binding.
//!
//!   NOT VERIFIED. The real checkpoint's own WIDTHS ([128, 256, 512, 512] and
//!   a 128px+ canvas) have not been run: the same test in `model` mode is the
//!   bar and needs the 20 GB pack on disk. The fixture's intermediates
//!   (`mid.dec.*`, `mid.enc.*`) are dumped for bisecting a mismatch but are
//!   not asserted here — neither chain returns per-stage activations.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const nn = @import("sdxl_nn.zig");
const sdxl_vae = @import("sdxl_vae.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;
const Resnet = nn.Resnet;

/// `AutoencoderKL`'s resnets and norms run eps 1e-6, not the UNet's 1e-5 —
/// same value SDXL's VAE uses, and the same trap if a shared helper picks one.
const VAE_EPS: f32 = 1e-6;

/// `norm_num_groups: 32` on every SD 3.5 VAE, exactly as on SDXL's.
const NORM_GROUPS: c_int = 32;

/// `force_upcast: true`. See the file header: fp16 here is a black image.
pub const DEFAULT_DTYPE: mlx.mlx_dtype = .float32;

/// The three `vae/config.json` fields that decide what gets bound.
///
/// The `use_*_quant_conv` defaults are DIFFUSERS' defaults, not SD 3.5's. A
/// config that declares them false (every SD 3.5 pack does) skips them; one
/// that omits them binds them, and a pack that has neither the field nor the
/// tensor fails at load by name. Defaulting the other way would decode an
/// SDXL-shaped VAE in the wrong basis with nothing to error on.
pub const Config = struct {
    latent_channels: u32 = 16,
    use_quant_conv: bool = true,
    use_post_quant_conv: bool = true,
};

/// Read `<vae_dir>/config.json`. A dir with no readable config takes the
/// defaults above, which is the loud arm.
pub fn readConfig(io: std.Io, allocator: std.mem.Allocator, vae_dir: []const u8) Config {
    const cfg = Config{};
    const path = std.fmt.allocPrint(allocator, "{s}/config.json", .{vae_dir}) catch return cfg;
    defer allocator.free(path);
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return cfg;
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return cfg;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 20)) catch return cfg;
    defer allocator.free(content);
    return parseConfig(allocator, content);
}

/// Split from `readConfig` so the field reading is testable without a file.
pub fn parseConfig(allocator: std.mem.Allocator, json_text: []const u8) Config {
    var cfg = Config{};
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_text, .{}) catch return cfg;
    defer parsed.deinit();
    if (parsed.value != .object) return cfg;
    const obj = parsed.value.object;
    if (obj.get("latent_channels")) |v| switch (v) {
        .integer => cfg.latent_channels = @intCast(v.integer),
        else => {},
    };
    inline for (.{ "use_quant_conv", "use_post_quant_conv" }) |field| {
        if (obj.get(field)) |v| switch (v) {
            .bool => @field(cfg, field) = v.bool,
            else => {},
        };
    }
    return cfg;
}

// ════════════════════════════════════════════════════════════════════════
// Decoder — sdxl_vae's chain, bound at SD 3.5's config
// ════════════════════════════════════════════════════════════════════════

pub const Vae = struct {
    dec: sdxl_vae.Decoder,
    cfg: Config,

    pub fn deinit(self: *Vae) void {
        self.dec.deinit();
    }

    /// Load `<model_dir>/vae`.
    pub fn load(
        io: std.Io,
        allocator: std.mem.Allocator,
        s: S,
        model_dir: []const u8,
        dtype: mlx.mlx_dtype,
    ) !Vae {
        const dir = try nn.fmtKey(allocator, "{s}/vae", .{model_dir});
        defer allocator.free(dir);
        const cfg = readConfig(io, allocator, dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        return loadFromWeights(allocator, s, &w, cfg, dtype);
    }

    /// The weight-binding half, so a converter or a test can drive it from an
    /// in-memory map — same split `sdxl_vae` keeps.
    ///
    /// `w` is MUTABLE because a checkpoint with `use_post_quant_conv: false`
    /// gets an identity stand-in put into the map (see the file header). The
    /// map owns it and frees it on `deinit`.
    pub fn loadFromWeights(
        allocator: std.mem.Allocator,
        s: S,
        w: *Weights,
        cfg: Config,
        dtype: mlx.mlx_dtype,
    ) !Vae {
        if (!cfg.use_post_quant_conv) {
            try putIdentityConv(w, "post_quant_conv", cfg.latent_channels, s);
        }
        // scaling 1.0: SD 3.5's scale AND shift are `sd3.decodeScale`, applied
        // by the caller. Handing the SDXL decoder its real `scaling_factor`
        // would arm `decodeLatent` with scale-only arithmetic that is wrong for
        // this family, so the field is neutralized and only `decode` is
        // exposed below.
        const dec = try sdxl_vae.loadFromWeights(allocator, s, w, 1.0, dtype);
        log.info(
            "[sd3] loaded vae decoder: latent_ch={d} post_quant_conv={}\n",
            .{ cfg.latent_channels, cfg.use_post_quant_conv },
        );
        return .{ .dec = dec, .cfg = cfg };
    }

    /// latent `[1, 16, h, w]`, ALREADY unscaled by the caller
    /// (`sd3.decodeScale`) -> pixels `[1, 3, 8h, 8w]` in the model's [-1, 1].
    pub fn decode(self: *Vae, z: mlx.mlx_array, s: S) !mlx.mlx_array {
        // `sdxl_vae.Decoder` carries its stream on the struct rather than
        // taking one per call; re-point it so a caller's stream still wins.
        self.dec.s = s;
        return self.dec.decode(z);
    }
};

/// Put a `channels x channels` identity 1x1 conv (and a zero bias) into `w`
/// under `<name>.weight` / `<name>.bias`, in PyTorch `[out, in, kh, kw]`
/// layout so `nn.loadConvWeight` permutes it like any other conv weight — an
/// identity stays an identity under that permute.
///
/// This exists ONLY to stand in for a tensor SD 3.5 does not ship, for a
/// loader that binds it unconditionally. It is arithmetically exact, and
/// `sd3 vae: the identity quant conv is a no-op` pins that.
fn putIdentityConv(w: *Weights, name: []const u8, channels: u32, s: S) !void {
    const a = w.allocator;
    const c: usize = channels;

    const buf = try a.alloc(f32, c * c);
    defer a.free(buf);
    @memset(buf, 0);
    for (0..c) |i| buf[i * c + i] = 1.0;

    const ci: c_int = @intCast(c);
    const shape = [_]c_int{ ci, ci, 1, 1 };
    // `mlx_array_new_data` copies shape-worth of bytes, so `buf` may die here.
    const eye = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    errdefer _ = mlx.mlx_array_free(eye);

    const bshape = [_]c_int{ci};
    var zero = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &bshape, 1, .float32, s));

    const kw = try nn.fmtKey(a, "{s}.weight", .{name});
    defer a.free(kw);
    const kb = try nn.fmtKey(a, "{s}.bias", .{name});
    defer a.free(kb);
    try w.put(kw, eye);
    try w.put(kb, zero);
}

// ════════════════════════════════════════════════════════════════════════
// Encoder — this file's own chain (see the header for why)
// ════════════════════════════════════════════════════════════════════════

/// Single-head self-attention over the spatial grid — the VAE mid block's
/// `Attention` with `heads=1`. A local copy of `sdxl_vae`'s private
/// `VaeAttn`; it would be an import if that file exported it.
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

        const normed = try nn.groupNorm(x, self.gnw, self.gnb, NORM_GROUPS, VAE_EPS, s);
        defer _ = mlx.mlx_array_free(normed);
        // NHWC makes the flatten to tokens a pure reshape.
        const tokens = try nn.reshape(normed, &[_]c_int{ 1, h_ * w_, c }, s);
        defer _ = mlx.mlx_array_free(tokens);

        const q = try self.q.forward(tokens, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.k.forward(tokens, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try self.v.forward(tokens, s);
        defer _ = mlx.mlx_array_free(v);

        // ONE head over the full channel width.
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

/// One encoder stage: `layers_per_block` resnets, then an optional
/// asymmetric-pad stride-2 downsample. The last stage does not downsample.
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

/// Asymmetric (0,1,0,1) zero-pad + 3x3 stride-2 valid conv on NHWC —
/// diffusers `Downsample2D` as the VAE configures it, NOT the UNet's
/// symmetric-1 `nn.conv2dStride2`. A local copy of `sdxl_vae`'s private
/// `conv2dDown`, for the same reason as `VaeAttn`.
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

/// The encoder half of SD 3.5's `AutoencoderKL` — pixels to a latent MEAN, for
/// img2img. Deterministic: the distribution's mode, never sampled.
pub const Encoder = struct {
    allocator: std.mem.Allocator,
    dtype: mlx.mlx_dtype,
    /// From the config, not a constant: it is what the `2 * latent` moments
    /// are sliced at, and slicing at SDXL's 4 would silently drop 12 channels.
    latent_channels: u32,

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
    /// `quant_conv`, present only when the config says so. Null on every SD
    /// 3.5 pack.
    quant_w: ?mlx.mlx_array,
    quant_b: ?mlx.mlx_array,

    pub fn deinit(self: *Encoder) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b" }) |f| {
            _ = mlx.mlx_array_free(@field(self, f));
        }
        if (self.quant_w) |x| _ = mlx.mlx_array_free(x);
        if (self.quant_b) |x| _ = mlx.mlx_array_free(x);
        self.mid_r0.deinit();
        self.mid_attn.deinit();
        self.mid_r1.deinit();
        for (self.down) |*d| d.deinit();
        self.allocator.free(self.down);
    }

    /// Load `<model_dir>/vae`'s `encoder.*` tensors. A pack that ships no
    /// encoder (nothing enforces one) fails cleanly; callers read that as "no
    /// img2img".
    pub fn load(
        io: std.Io,
        allocator: std.mem.Allocator,
        s: S,
        model_dir: []const u8,
        dtype: mlx.mlx_dtype,
    ) !Encoder {
        const dir = try nn.fmtKey(allocator, "{s}/vae", .{model_dir});
        defer allocator.free(dir);
        const cfg = readConfig(io, allocator, dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        return loadFromWeights(allocator, s, &w, cfg, dtype);
    }

    pub fn loadFromWeights(
        allocator: std.mem.Allocator,
        s: S,
        w: *const Weights,
        cfg: Config,
        dtype: mlx.mlx_dtype,
    ) !Encoder {
        const stages = countStages(w, allocator, "encoder.down_blocks");
        if (stages == 0) return error.NoVaeDownBlocks;

        var e: Encoder = undefined;
        e.allocator = allocator;
        e.dtype = dtype;
        e.latent_channels = cfg.latent_channels;
        e.quant_w = null;
        e.quant_b = null;

        e.conv_in_w = try nn.loadConvWeight(w, "encoder.conv_in.weight", dtype, s);
        e.conv_in_b = try nn.dupWeight(w, "encoder.conv_in.bias", dtype, s);

        e.down = try allocator.alloc(DownBlock, stages);
        errdefer allocator.free(e.down);
        for (e.down, 0..) |*blk, bi| {
            blk.allocator = allocator;
            blk.down_w = null;
            blk.down_b = null;
            const pfx = try nn.fmtKey(allocator, "encoder.down_blocks.{d}", .{bi});
            defer allocator.free(pfx);
            const n_res = countResnets(w, allocator, pfx);
            blk.resnets = try allocator.alloc(Resnet, n_res);
            for (blk.resnets, 0..) |*r, ri| {
                const rp = try nn.fmtKey(allocator, "{s}.resnets.{d}", .{ pfx, ri });
                defer allocator.free(rp);
                r.* = try nn.loadResnet(w, allocator, rp, false, VAE_EPS, dtype, s);
            }
            const kw = try nn.fmtKey(allocator, "{s}.downsamplers.0.conv.weight", .{pfx});
            defer allocator.free(kw);
            if (w.get(kw) != null) {
                const kb = try nn.fmtKey(allocator, "{s}.downsamplers.0.conv.bias", .{pfx});
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

        if (cfg.use_quant_conv) {
            e.quant_w = try nn.loadConvWeight(w, "quant_conv.weight", dtype, s);
            e.quant_b = try nn.dupWeight(w, "quant_conv.bias", dtype, s);
        }

        log.info(
            "[sd3] loaded vae encoder: stages={d} latent_ch={d} quant_conv={}\n",
            .{ stages, cfg.latent_channels, cfg.use_quant_conv },
        );
        return e;
    }

    /// pixels `[1, 3, H, W]` in the model's [-1, 1] -> latent MEAN
    /// `[1, 16, H/8, W/8]`, UNSCALED. The `(z - shift) * scaling` that turns
    /// this into a latent is `sd3.encodeScale`, the caller's job.
    ///
    /// The [-1, 1] convention is the CALLER's too — unlike
    /// `sdxl_vae.Encoder.encode`, nothing here remaps from [0, 1]. Handing it
    /// a [0, 1] image produces a plausible, systematically washed latent.
    pub fn encodeMean(self: *const Encoder, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const nhwc = try nn.nchwToNhwc(x, s);
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
            const n = try nn.groupNorm(h, self.norm_out_w, self.norm_out_b, NORM_GROUPS, VAE_EPS, s);
            nn.replace(&h, n);
        }
        nn.replace(&h, try nn.silu(h, s));
        // [1, H/8, W/8, 2*latent] — the moments.
        nn.replace(&h, try nn.conv2d(h, self.conv_out_w, self.conv_out_b, 1, s));
        if (self.quant_w) |qw| {
            nn.replace(&h, try nn.conv2d(h, qw, self.quant_b.?, 0, s));
        }

        // The mean is the FIRST `latent_channels` of the moments; the logvar
        // is the rest and is dropped — `encodeMean` is the mode, never a draw.
        const hsh = mlx.getShape(h);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const stop = [_]c_int{ hsh[0], hsh[1], hsh[2], @intCast(self.latent_channels) };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var mean = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mean);
        try mlx.check(mlx.mlx_slice(&mean, h, &start, 4, &stop, 4, &strides, 4, s));
        _ = mlx.mlx_array_free(h);
        const meanc = try nn.contiguous(mean, s);
        defer _ = mlx.mlx_array_free(meanc);
        return nn.nhwcToNchw(meanc, s);
    }
};

/// SDXL ships `to_out.0.*`; some conversions flatten it to `to_out.*`. Both
/// are accepted for the same reason `sdxl_vae` accepts both — a converter's
/// choice, not a different module.
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

/// Stage and resnet counts come from the checkpoint's own numbering, so a VAE
/// of a different depth loads rather than being assumed to be 4x2.
fn countStages(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) usize {
    var n: usize = 0;
    while (n < 16) : (n += 1) {
        const k = nn.fmtKey(a, "{s}.{d}.resnets.0.conv1.weight", .{ prefix, n }) catch return n;
        defer a.free(k);
        if (w.get(k) == null) return n;
    }
    return n;
}

fn countResnets(w: *const Weights, a: std.mem.Allocator, block_prefix: []const u8) usize {
    var n: usize = 0;
    while (n < 16) : (n += 1) {
        const k = nn.fmtKey(a, "{s}.resnets.{d}.conv1.weight", .{ block_prefix, n }) catch return n;
        defer a.free(k);
        if (w.get(k) == null) return n;
    }
    return n;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "sd3 vae: the quant convs are a CONFIG decision, defaulting to diffusers'" {
    const a = testing.allocator;
    // SD 3.5's own vae/config.json, trimmed to the fields that decide binding.
    const sd3_cfg =
        \\{"_class_name":"AutoencoderKL","latent_channels":16,
        \\ "use_quant_conv":false,"use_post_quant_conv":false,
        \\ "scaling_factor":1.5305,"shift_factor":0.0609}
    ;
    const c = parseConfig(a, sd3_cfg);
    try testing.expectEqual(@as(u32, 16), c.latent_channels);
    try testing.expect(!c.use_quant_conv);
    try testing.expect(!c.use_post_quant_conv);

    // A config that omits the flags takes DIFFUSERS' default (true), so the
    // load fails loudly on the absent tensor instead of quietly decoding in
    // the wrong latent basis. SDXL's own config is exactly this shape.
    const sdxl_cfg = "{\"_class_name\":\"AutoencoderKL\",\"latent_channels\":4}";
    const c2 = parseConfig(a, sdxl_cfg);
    try testing.expectEqual(@as(u32, 4), c2.latent_channels);
    try testing.expect(c2.use_quant_conv);
    try testing.expect(c2.use_post_quant_conv);

    // Junk is not a silent 0-channel VAE.
    const c3 = parseConfig(a, "not json");
    try testing.expectEqual(@as(u32, 16), c3.latent_channels);

    // force_upcast: this VAE overflows fp16, and the failure is a black image.
    try testing.expectEqual(mlx.mlx_dtype.float32, DEFAULT_DTYPE);
    try testing.expectEqual(@as(f32, 1e-6), VAE_EPS);
}

test "sd3 vae: the identity quant conv is a no-op" {
    // The decoder reuse in this file stands an identity 1x1 conv in for a
    // tensor SD 3.5 does not ship. If that is not EXACT, every SD 3.5 image is
    // subtly wrong with nothing to error on — so it is asserted, not assumed.
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();

    var w = Weights.init(a);
    defer w.deinit();
    try putIdentityConv(&w, "post_quant_conv", 16, s);

    const conv_w = try nn.loadConvWeight(&w, "post_quant_conv.weight", .float32, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const conv_b = try nn.dupWeight(&w, "post_quant_conv.bias", .float32, s);
    defer _ = mlx.mlx_array_free(conv_b);

    // A [1, 3, 5, 16] NHWC block of varied values, run through the 1x1 conv.
    var buf: [3 * 5 * 16]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 120)) * 0.37;
    const shape = [_]c_int{ 1, 3, 5, 16 };
    const x = mlx.mlx_array_new_data(&buf, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    const y = try nn.conv2d(x, conv_w, conv_b, 0, s);
    defer _ = mlx.mlx_array_free(y);
    _ = mlx.mlx_array_eval(y);

    try testing.expectEqualSlices(c_int, &shape, mlx.getShape(y));
    const got = mlx.mlx_array_data_float32(y).?;
    for (buf, 0..) |want, i| try testing.expectEqual(want, got[i]);
}

// Numerical PARITY against diffusers' own `AutoencoderKL`, both directions.
//
//   /tmp/claude-501/sd3venv/bin/python tests/dump_sd3_vae_fixtures.py build \
//       --out ~/.mlx-serve/staging/sd3-vae-tiny
//   SD3_VAE_FIXTURE_DIR=~/.mlx-serve/staging/sd3-vae-tiny \
//       zig build test -Doptimize=ReleaseFast -Dtest-filter="sd3 vae parity"
//
// `SD3_VAE_MODEL_DIR` points the weights at a real checkpoint (whose `vae/`
// the fixture was generated from in `model` mode); it defaults to the fixture
// dir, which `build` mode makes a complete checkpoint.
test "sd3 vae parity: decode and encodeMean match diffusers" {
    const fixture_dir = std.mem.span(std.c.getenv("SD3_VAE_FIXTURE_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const model_dir = if (std.c.getenv("SD3_VAE_MODEL_DIR")) |m| std.mem.span(m) else fixture_dir;
    const fixture = try std.fmt.allocPrint(a, "{s}/fixture.safetensors", .{fixture_dir});
    defer a.free(fixture);

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();

    // The fixture stores the tensor diffusers' `decode` was handed — already
    // `z / scaling + shift`ed — so this isolates the VAE from the pipeline's
    // scale/shift convention.
    const vae_in = fx.get("in.vae_in") orelse return error.MissingFixtureLatent;
    const ref_decoded = fx.get("out.decoded") orelse return error.MissingFixtureDecoded;
    const image = fx.get("in.image") orelse return error.MissingFixtureImage;
    const ref_mean = fx.get("out.encoded_mean") orelse return error.MissingFixtureMean;

    // 16 latent channels is the claim the shapes have to back.
    try testing.expectEqual(@as(c_int, 16), mlx.getShape(vae_in)[1]);

    var vae = try Vae.load(io, a, s, model_dir, DEFAULT_DTYPE);
    defer vae.deinit();
    try testing.expect(!vae.cfg.use_post_quant_conv);

    const decoded = try vae.decode(vae_in, s);
    defer _ = mlx.mlx_array_free(decoded);
    _ = mlx.mlx_array_eval(decoded);
    try testing.expectEqualSlices(c_int, mlx.getShape(ref_decoded), mlx.getShape(decoded));
    try expectCloseSd3("decode", decoded, ref_decoded, s);

    var enc = try Encoder.load(io, a, s, model_dir, DEFAULT_DTYPE);
    defer enc.deinit();
    try testing.expect(enc.quant_w == null);

    const mean = try enc.encodeMean(image, s);
    defer _ = mlx.mlx_array_free(mean);
    _ = mlx.mlx_array_eval(mean);
    try testing.expectEqualSlices(c_int, mlx.getShape(ref_mean), mlx.getShape(mean));
    try expectCloseSd3("encodeMean", mean, ref_mean, s);
}

/// A cosine test cannot see a SCALE error, so the rms ratio is asserted too —
/// the repo's standing rule for every media-parity bar.
fn expectCloseSd3(what: []const u8, got: mlx.mlx_array, ref: mlx.mlx_array, s: S) !void {
    const g_arr = try flatF32Sd3(got, s);
    defer _ = mlx.mlx_array_free(g_arr);
    const r_arr = try flatF32Sd3(ref, s);
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
        // Finiteness before the diff: a NaN makes every ratio meaningless.
        try testing.expect(std.math.isFinite(gv));
        dot += gv * rv;
        ng += gv * gv;
        nr += rv * rv;
        max_abs = @max(max_abs, @abs(gv - rv));
    }
    const cos = dot / (@sqrt(ng) * @sqrt(nr));
    const rms_ratio = @sqrt(ng) / @sqrt(nr);
    std.debug.print("[sd3-parity] {s}: cos={d:.8} rms_ratio={d:.8} max_abs={d:.8}\n", .{ what, cos, rms_ratio, max_abs });
    try testing.expect(cos > 0.9999);
    try testing.expect(rms_ratio > 0.995 and rms_ratio < 1.005);
}

fn flatF32Sd3(x: mlx.mlx_array, s: S) !mlx.mlx_array {
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
