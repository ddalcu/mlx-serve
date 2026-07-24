//! Native MageFlow text→image backend (Microsoft Mage-Flow family), ported from
//! the pure-MLX reference `ivanfioravanti/mflux@mage-flow-mlx` to mlx-c FFI.
//!
//! MageFlow is a three-component pipeline, none of which FLUX/Krea already have:
//!   • a Qwen3-VL text encoder (produces the DiT conditioning AND runs a
//!     fail-closed content-policy screen on prompts / source images),
//!   • a native-resolution double-stream flow transformer (its own MSRoPE),
//!   • its own AdaLN VAE (16× downsample, 128-channel latent),
//! plus a FlowMatchEuler scheduler and a Gaussian-Shading watermark baked into
//! the initial noise. See `docs/reference.md` (once ported) for the full map.
//!
//! Self-contained sibling of `flux.zig`/`krea.zig`: hosted by the image modality
//! slot in `gen.zig` (the `ImageEngine` backend union). Neither of those files is
//! touched. This is the PORT-IN-PROGRESS scaffold: it parses the released
//! checkpoint's component configs and validates the weight manifest, but the
//! forward passes are not implemented yet — `generateImage` returns
//! `error.MageFlowNotImplemented` (the HTTP layer maps it to an honest 501).
//! The component ports (VAE → DiT → Qwen3-VL encoder → scheduler) land behind
//! parity fixtures in later phases.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// The port has parsed the checkpoint and loaded config, but the forward pass
/// is not wired yet. Distinct from a generic failure so the HTTP layer can 501.
pub const Error = error{MageFlowNotImplemented};

// ── Weight-manifest expectations (mflux `MageFlowWeightMapping`) ──
// Post-mapping tensor counts per component. The VAE ships either "unfolded"
// (separate AdaLN modulation tensors) or "folded" (one modulation cache per
// block) — both are legal, hence two accepted counts.
pub const EXPECTED_TRANSFORMER_WEIGHTS: usize = 397;
pub const EXPECTED_TEXT_ENCODER_WEIGHTS: usize = 713;
pub const EXPECTED_VAE_WEIGHTS: usize = 728;
pub const EXPECTED_VAE_WEIGHTS_FOLDED: usize = 686;

pub const Component = enum { transformer, text_encoder, vae };

/// Reject a checkpoint whose mapped tensor count doesn't match the port — the
/// early "does this release still match?" guard mflux runs before allocation.
/// Pure so it's unit-testable now and reused once real weights load.
pub fn validateWeightCount(component: Component, actual: usize) !void {
    const ok = switch (component) {
        .transformer => actual == EXPECTED_TRANSFORMER_WEIGHTS,
        .text_encoder => actual == EXPECTED_TEXT_ENCODER_WEIGHTS,
        // Either the unfolded or the folded-AdaLN layout is accepted.
        .vae => actual == EXPECTED_VAE_WEIGHTS or actual == EXPECTED_VAE_WEIGHTS_FOLDED,
    };
    if (!ok) return error.MageFlowWeightCountMismatch;
}

// ── Checkpoint key mapping (mflux `MageFlowWeightMapping`) ────────────────
// Released HF tensor names → our module names. A null result means the tensor
// is intentionally dropped (tied/legacy). Conv weights additionally need an
// OIHW→OHWI transpose at load (handled where the tensors are materialized, not
// here); this layer is names only.

const VAE_LEGACY_ENC_PREFIX = "pipeline.y_embedder.encoder.";
const VAE_ENC_PREFIX = "student.dconv_encoder.";
const VAE_DEC_PREFIX = "pipeline.";

/// VAE: `student.dconv_encoder.*`→`encoder.*`, `pipeline.*`→`decoder_model.*`,
/// legacy VAE-encoder tensors dropped. Returns an owned dupe (caller frees) or
/// null to drop; errors on an unrecognized name (a converter mismatch guard).
pub fn mapVaeKey(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    if (std.mem.startsWith(u8, key, VAE_LEGACY_ENC_PREFIX)) return null;
    if (std.mem.startsWith(u8, key, VAE_ENC_PREFIX))
        return try std.fmt.allocPrint(a, "encoder.{s}", .{key[VAE_ENC_PREFIX.len..]});
    if (std.mem.startsWith(u8, key, VAE_DEC_PREFIX))
        return try std.fmt.allocPrint(a, "decoder_model.{s}", .{key[VAE_DEC_PREFIX.len..]});
    return error.UnexpectedMageFlowVaeKey;
}

/// Text encoder (Qwen3-VL): drop the tied `lm_head.weight` and the visual
/// rotary `inv_freq` buffer; strip the `model.` prefix off language_model/visual
/// tensors. Owned dupe or null (drop); errors on an unrecognized name.
pub fn mapTextEncoderKey(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    if (std.mem.eql(u8, key, "lm_head.weight")) return null;
    if (std.mem.eql(u8, key, "model.visual.rotary_pos_emb.inv_freq")) return null;
    if (std.mem.startsWith(u8, key, "model.language_model.") or
        std.mem.startsWith(u8, key, "model.visual."))
        return try a.dupe(u8, key["model.".len..]);
    return error.UnexpectedMageFlowTextEncoderKey;
}

// ── Config ──────────────────────────────────────────────────────────────
// Parsed from the released diffusers-style repo: transformer/config.json,
// vae/config.json, text_encoder/config.json. Defaults mirror the Turbo release
// so a minimal config still yields a usable struct; the three files must exist.

pub const Config = struct {
    // Transformer (double-stream flow DiT).
    dit_in_channels: u32 = 128,
    dit_context_dim: u32 = 2560, // matches the text encoder's hidden size
    dit_hidden: u32 = 3072,
    dit_heads: u32 = 24,
    dit_depth: u32 = 12, // double-stream blocks
    dit_mlp_ratio: f32 = 4.0,
    dit_axes_dim: [3]u32 = .{ 16, 56, 56 }, // MSRoPE, sums to head_dim (128)
    dit_theta: f32 = 10000,
    dit_max_seq_len: u32 = 2048,
    dit_static_shift: f32 = 6.0,

    // VAE (AdaLN, 16× downsample, 128-channel latent).
    vae_latent_channels: u32 = 128,
    vae_downsample: u32 = 16,

    // Text encoder — Qwen3-VL (Qwen3VLForConditionalGeneration).
    te_hidden: u32 = 2560,
    te_layers: u32 = 36,
    te_heads: u32 = 32,
    te_kv_heads: u32 = 8,
    te_head_dim: u32 = 128,
    te_intermediate: u32 = 9728,
    te_vocab: u32 = 151936,
    te_rope_theta: f32 = 5000000,
    te_mrope_section: [3]u32 = .{ 24, 20, 20 },
    // Vision tower.
    vit_depth: u32 = 24,
    vit_hidden: u32 = 1024,
    vit_heads: u32 = 16,
    vit_patch: u32 = 16,
    vit_spatial_merge: u32 = 2,
    vit_out_hidden: u32 = 2560,
    image_token_id: u32 = 151655,

    // Scheduler (FlowMatchEulerDiscrete).
    sched_shift: f32 = 6.0,
    sched_train_timesteps: u32 = 1000,

    /// head_dim derived from the DiT geometry (hidden / heads).
    pub fn ditHeadDim(self: Config) u32 {
        return self.dit_hidden / self.dit_heads;
    }
};

// ── JSON helpers (file-local; mirror gen.peekModelType's read pattern) ──

fn readJson(io: std.Io, a: std.mem.Allocator, path: []const u8) !std.json.Parsed(std.json.Value) {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [8192]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
    defer a.free(content);
    return std.json.parseFromSlice(std.json.Value, a, content, .{});
}

fn objGet(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}

fn getU32(v: std.json.Value, key: []const u8, default: u32) u32 {
    const x = objGet(v, key) orelse return default;
    return switch (x) {
        .integer => |i| if (i >= 0) @intCast(i) else default,
        .float => |f| @intFromFloat(f),
        else => default,
    };
}

fn getF32(v: std.json.Value, key: []const u8, default: f32) f32 {
    const x = objGet(v, key) orelse return default;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => default,
    };
}

fn getU32x3(v: std.json.Value, key: []const u8, default: [3]u32) [3]u32 {
    const x = objGet(v, key) orelse return default;
    if (x != .array or x.array.items.len < 3) return default;
    var out = default;
    for (0..3) |i| {
        out[i] = switch (x.array.items[i]) {
            .integer => |n| if (n >= 0) @intCast(n) else default[i],
            .float => |f| @intFromFloat(f),
            else => default[i],
        };
    }
    return out;
}

/// Parse the three component configs from a released MageFlow repo. The
/// transformer/vae/text_encoder configs are REQUIRED (this doubles as the
/// manifest presence check); missing scalars fall back to the release defaults.
pub fn parseConfig(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Config {
    var cfg = Config{};

    {
        const path = try std.fmt.allocPrint(a, "{s}/transformer/config.json", .{model_dir});
        defer a.free(path);
        var p = try readJson(io, a, path);
        defer p.deinit();
        const o = p.value;
        cfg.dit_in_channels = getU32(o, "in_channels", cfg.dit_in_channels);
        cfg.dit_context_dim = getU32(o, "context_in_dim", cfg.dit_context_dim);
        cfg.dit_hidden = getU32(o, "hidden_size", cfg.dit_hidden);
        cfg.dit_heads = getU32(o, "num_heads", cfg.dit_heads);
        cfg.dit_depth = getU32(o, "depth", cfg.dit_depth);
        cfg.dit_mlp_ratio = getF32(o, "mlp_ratio", cfg.dit_mlp_ratio);
        cfg.dit_axes_dim = getU32x3(o, "axes_dim", cfg.dit_axes_dim);
        cfg.dit_theta = getF32(o, "theta", cfg.dit_theta);
        cfg.dit_max_seq_len = getU32(o, "max_sequence_length", cfg.dit_max_seq_len);
        cfg.dit_static_shift = getF32(o, "static_shift", cfg.dit_static_shift);
    }

    {
        const path = try std.fmt.allocPrint(a, "{s}/vae/config.json", .{model_dir});
        defer a.free(path);
        var p = try readJson(io, a, path);
        defer p.deinit();
        const o = p.value;
        cfg.vae_latent_channels = getU32(o, "latent_channels", cfg.vae_latent_channels);
        cfg.vae_downsample = getU32(o, "downsample_factor", cfg.vae_downsample);
    }

    {
        const path = try std.fmt.allocPrint(a, "{s}/text_encoder/config.json", .{model_dir});
        defer a.free(path);
        var p = try readJson(io, a, path);
        defer p.deinit();
        const root = p.value;
        cfg.image_token_id = getU32(root, "image_token_id", cfg.image_token_id);
        // text_config carries the LM geometry; fall back to root for flat configs.
        const tc = objGet(root, "text_config") orelse root;
        cfg.te_hidden = getU32(tc, "hidden_size", cfg.te_hidden);
        cfg.te_layers = getU32(tc, "num_hidden_layers", cfg.te_layers);
        cfg.te_heads = getU32(tc, "num_attention_heads", cfg.te_heads);
        cfg.te_kv_heads = getU32(tc, "num_key_value_heads", cfg.te_kv_heads);
        cfg.te_head_dim = getU32(tc, "head_dim", cfg.te_head_dim);
        cfg.te_intermediate = getU32(tc, "intermediate_size", cfg.te_intermediate);
        cfg.te_vocab = getU32(tc, "vocab_size", cfg.te_vocab);
        cfg.te_rope_theta = getF32(tc, "rope_theta", cfg.te_rope_theta);
        if (objGet(tc, "rope_scaling")) |rs| {
            cfg.te_mrope_section = getU32x3(rs, "mrope_section", cfg.te_mrope_section);
        }
        if (objGet(root, "vision_config")) |vc| {
            cfg.vit_depth = getU32(vc, "depth", cfg.vit_depth);
            cfg.vit_hidden = getU32(vc, "hidden_size", cfg.vit_hidden);
            cfg.vit_heads = getU32(vc, "num_heads", cfg.vit_heads);
            cfg.vit_patch = getU32(vc, "patch_size", cfg.vit_patch);
            cfg.vit_spatial_merge = getU32(vc, "spatial_merge_size", cfg.vit_spatial_merge);
            cfg.vit_out_hidden = getU32(vc, "out_hidden_size", cfg.vit_out_hidden);
        }
    }

    return cfg;
}

// ══════════════════════════════════════════════════════════════════════════
// MageVAE decoder (DiCo generative decode). Ported from mflux
// `mage_flow_vae.py`. The released decode is deterministic: it runs the flow
// denoiser once at t=0 over zero noise, conditioned on the VAE-decoded latent.
// Only the `pipeline.*` (decoder) subtree is needed for text→image; the encoder
// (`student.dconv_encoder.*`) is loaded lazily elsewhere for img2img.
// ══════════════════════════════════════════════════════════════════════════

const model_mod = @import("model.zig");
const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

// ── mlx primitives (file-local; mirror krea.zig's helper style) ──
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
inline fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
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
inline fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
inline fn sigmoidA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    return o;
}
fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sig = try sigmoidA(x, s);
    defer _ = mlx.mlx_array_free(sig);
    return mulA(x, sig, s);
}
fn scalarF(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}
/// Exact GELU: 0.5*x*(1+erf(x/√2)) — the reference uses `nn.gelu` (erf form).
fn gelu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = scalarF(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const scaled = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(scaled);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, scaled, s));
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opl = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(opl);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opl, s);
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Split into `n` equal parts on `axis`, returning owned arrays (caller frees).
fn splitEqual(x: mlx.mlx_array, n: usize, axis: c_int, out: []mlx.mlx_array, s: S) !void {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_split(&vec, x, @intCast(n), axis, s));
    for (0..n) |i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, vec, i));
        out[i] = o;
    }
}
inline fn meanAxes(x: mlx.mlx_array, axes: []const c_int, keepdims: bool, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean_axes(&o, x, axes.ptr, axes.len, keepdims, s));
    return o;
}
/// conv2d on NHWC; weight OHWI f32; optional bias [O]; `groups` for depthwise.
fn conv2d(x: mlx.mlx_array, w: mlx.mlx_array, bias: ?mlx.mlx_array, stride: c_int, pad: c_int, groups: c_int, s: S) !mlx.mlx_array {
    const xc = try contig(x, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv2d(&o, xc, w, stride, stride, pad, pad, 1, 1, groups, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
    return o;
}
/// Linear: x[..,in] @ w_t[in,out] (+ bias). `w_t` is pre-transposed at load.
fn linearT(x: mlx.mlx_array, w_t: mlx.mlx_array, bias: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, w_t, s));
    if (bias) |b| {
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }
    return o;
}

// ── Norms (computed in f32, matching the reference) ──
/// GroupNorm over NHWC, `num_groups` groups, affine, pytorch-compatible
/// (normalizes over the group's channels × spatial). weight/bias are [C] f32.
fn groupNorm(x: mlx.mlx_array, weight: mlx.mlx_array, bias: mlx.mlx_array, num_groups: c_int, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,H,W,C]
    const B = sh[0];
    const H = sh[1];
    const Wd = sh[2];
    const C = sh[3];
    const cg = @divExact(C, num_groups);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const grouped = try reshape(xf, &[_]c_int{ B, H, Wd, num_groups, cg }, s);
    defer _ = mlx.mlx_array_free(grouped);
    const red = [_]c_int{ 1, 2, 4 };
    const mean = try meanAxes(grouped, &red, true, s);
    defer _ = mlx.mlx_array_free(mean);
    const centered = try subA(grouped, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const sq = try mulA(centered, centered, s);
    defer _ = mlx.mlx_array_free(sq);
    const variance = try meanAxes(sq, &red, true, s);
    defer _ = mlx.mlx_array_free(variance);
    const epsA = scalarF(eps);
    defer _ = mlx.mlx_array_free(epsA);
    const vpe = try addA(variance, epsA, s);
    defer _ = mlx.mlx_array_free(vpe);
    var rstd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rstd);
    try mlx.check(mlx.mlx_rsqrt(&rstd, vpe, s));
    const normed = try mulA(centered, rstd, s);
    defer _ = mlx.mlx_array_free(normed);
    const back = try reshape(normed, &[_]c_int{ B, H, Wd, C }, s);
    defer _ = mlx.mlx_array_free(back);
    const scaled = try mulA(back, weight, s);
    defer _ = mlx.mlx_array_free(scaled);
    return addA(scaled, bias, s);
}
/// LayerNorm over the LAST axis, computed in f32. weight/bias optional (affine).
fn layerNormLast(x: mlx.mlx_array, weight: ?mlx.mlx_array, bias: ?mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const nd = mlx.getShape(x).len;
    const last = [_]c_int{@intCast(nd - 1)};
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const mean = try meanAxes(xf, &last, true, s);
    defer _ = mlx.mlx_array_free(mean);
    const centered = try subA(xf, mean, s);
    defer _ = mlx.mlx_array_free(centered);
    const sq = try mulA(centered, centered, s);
    defer _ = mlx.mlx_array_free(sq);
    const variance = try meanAxes(sq, &last, true, s);
    defer _ = mlx.mlx_array_free(variance);
    const epsA = scalarF(eps);
    defer _ = mlx.mlx_array_free(epsA);
    const vpe = try addA(variance, epsA, s);
    defer _ = mlx.mlx_array_free(vpe);
    var rstd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rstd);
    try mlx.check(mlx.mlx_rsqrt(&rstd, vpe, s));
    const normed = try mulA(centered, rstd, s);
    if (weight == null) return normed;
    defer _ = mlx.mlx_array_free(normed);
    const scaled = try mulA(normed, weight.?, s);
    if (bias == null) return scaled;
    defer _ = mlx.mlx_array_free(scaled);
    return addA(scaled, bias.?, s);
}

// ── Weight loaders (reference the physical `pipeline.*` checkpoint keys) ──
fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[mageflow] MISSING VAE WEIGHT: {s}\n", .{key});
        return error.MissingMageFlowWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
/// Conv weight OIHW → OHWI, f32 (matches mflux `transform_vae_weight`).
fn loadConvW(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, .float32, s);
}
/// Linear weight [out,in] → pre-transposed [in,out], f32.
fn loadLinT(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, .float32, s);
}
/// A `.bias`/`.weight` vector, f32.
fn loadVec(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, s: S) !mlx.mlx_array {
    const k = try std.fmt.allocPrint(a, "{s}." ++ suffix, .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}

/// Precompute the NeRF DCT position embedding [1, patch², max_freqs²] on the
/// host (deterministic; the reference lru_caches it). patch=16, max_freqs=8.
fn buildNerfPosEmbedding(a: std.mem.Allocator, patch: usize, max_freqs: usize, s: S) !mlx.mlx_array {
    const area = patch * patch;
    const nf2 = max_freqs * max_freqs;
    const buf = try a.alloc(f32, area * nf2);
    defer a.free(buf);
    var position = try a.alloc(f64, patch);
    defer a.free(position);
    for (0..patch) |k| position[k] = if (patch == 1) 0 else @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(patch - 1));
    var freqs = try a.alloc(f64, max_freqs);
    defer a.free(freqs);
    for (0..max_freqs) |f| freqs[f] = if (max_freqs == 1) 0 else @as(f64, @floatFromInt(f)) * @as(f64, @floatFromInt(max_freqs)) / @as(f64, @floatFromInt(max_freqs - 1));
    const pi = std.math.pi;
    for (0..area) |idx| {
        const i = idx / patch; // pos_y row
        const j = idx % patch; // pos_x col
        for (0..max_freqs) |fx| {
            const dctx = @cos(position[j] * freqs[fx] * pi);
            for (0..max_freqs) |fy| {
                const dcty = @cos(position[i] * freqs[fy] * pi);
                const coef = 1.0 / (1.0 + freqs[fx] * freqs[fy]);
                buf[idx * nf2 + fx * max_freqs + fy] = @floatCast(dctx * dcty * coef);
            }
        }
    }
    const shape = [_]c_int{ 1, @intCast(area), @intCast(nf2) };
    const raw = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(raw);
    return contig(raw, s); // detach from the freed host buffer
}

// ── Engine ──────────────────────────────────────────────────────────────

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []u8,
    cfg: Config,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        // Parse + validate the manifest (component configs must be present).
        self.cfg = try parseConfig(io, allocator, model_dir);
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        log.info(
            "[image] MageFlow config parsed (DiT {d}×{d} heads {d}, ctx {d}; VAE {d}ch /{d}; TE Qwen3-VL {d}L) — forward pass not yet implemented\n",
            .{ self.cfg.dit_depth, self.cfg.dit_hidden, self.cfg.dit_heads, self.cfg.dit_context_dim, self.cfg.vae_latent_channels, self.cfg.vae_downsample, self.cfg.te_layers },
        );
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    /// Not implemented yet — the forward passes land in later phases. Returns a
    /// typed error the HTTP layer surfaces as an honest 501.
    pub fn generateImage(
        self: *Engine,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        width: u32,
        height: u32,
        seed: u64,
        steps: u32,
    ) !mlx.mlx_array {
        _ = self;
        _ = allocator;
        _ = prompt;
        _ = width;
        _ = height;
        _ = seed;
        _ = steps;
        return Error.MageFlowNotImplemented;
    }
};

// ── VAE decoder weight groups ──
const ResnetW = struct { n1w: mlx.mlx_array, n1b: mlx.mlx_array, c1w: mlx.mlx_array, c1b: mlx.mlx_array, n2w: mlx.mlx_array, n2b: mlx.mlx_array, c2w: mlx.mlx_array, c2b: mlx.mlx_array };
const AttnW = struct { nw: mlx.mlx_array, nb: mlx.mlx_array, qw: mlx.mlx_array, qb: mlx.mlx_array, kw: mlx.mlx_array, kb: mlx.mlx_array, vw: mlx.mlx_array, vb: mlx.mlx_array, pw: mlx.mlx_array, pb: mlx.mlx_array };
const DecBlock = union(enum) { resnet: ResnetW, attn: AttnW };
const DiCoW = struct {
    c1w: mlx.mlx_array,
    c1b: mlx.mlx_array,
    c2w: mlx.mlx_array,
    c2b: mlx.mlx_array,
    c3w: mlx.mlx_array,
    c3b: mlx.mlx_array,
    caw: mlx.mlx_array,
    cab: mlx.mlx_array,
    c4w: mlx.mlx_array,
    c4b: mlx.mlx_array,
    c5w: mlx.mlx_array,
    c5b: mlx.mlx_array,
    adaw: mlx.mlx_array,
    adab: mlx.mlx_array,
};
const MlpResW = struct { lnw: mlx.mlx_array, lnb: mlx.mlx_array, m0w: mlx.mlx_array, m0b: mlx.mlx_array, m2w: mlx.mlx_array, m2b: mlx.mlx_array, adaw: mlx.mlx_array, adab: mlx.mlx_array };

const NUM_DEC_BLOCKS = 5; // _Decoder: resnet, attn, resnet, attn, resnet
const NUM_DICO_BLOCKS = 21; // denoiser conditioning blocks (num_cond_blocks)
const NUM_MLP_RES = 3; // dec_net res blocks (num_blocks - num_cond_blocks)
const VAE_PATCH: c_int = 16;
const VAE_HIDDEN: c_int = 384;
const VAE_HIDDEN_X: c_int = 32;
const VAE_GROUPS: c_int = 32;
const VAE_ATTN_PATCH: c_int = 32;

/// MageVAE decoder — the deterministic `decode(z)` path (t=0, zero noise).
/// Loads only the `pipeline.*` (decoder) subtree of the released VAE.
pub const VaeDecoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    // _Decoder (pipeline.y_embedder.decoder.*)
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    dec_blocks: [NUM_DEC_BLOCKS]DecBlock,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    // denoiser
    denoiser_t: mlx.mlx_array, // [1,384] precomputed t=0 timestep embedding
    s_proj2_w: mlx.mlx_array,
    s_proj2_b: mlx.mlx_array,
    blocks: [NUM_DICO_BLOCKS]DiCoW,
    yx_w: mlx.mlx_array,
    yx_b: mlx.mlx_array,
    xemb_w: mlx.mlx_array, // NeRF linear [99,32] pre-transposed
    xemb_b: mlx.mlx_array,
    nerf_pe: mlx.mlx_array, // [1,256,64]
    cond_embed_w: mlx.mlx_array,
    cond_embed_b: mlx.mlx_array,
    input_proj_w: mlx.mlx_array,
    input_proj_b: mlx.mlx_array,
    res_blocks: [NUM_MLP_RES]MlpResW,
    final_norm_w: mlx.mlx_array,
    final_lin_w: mlx.mlx_array,
    final_lin_b: mlx.mlx_array,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !VaeDecoder {
        const dir = try std.fmt.allocPrint(allocator, "{s}/vae", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        const a = allocator;
        var self: VaeDecoder = undefined;
        self.allocator = allocator;
        self.s = s;

        // _Decoder.
        self.conv_in_w = try loadConvW(&w, a, "pipeline.y_embedder.decoder.conv_in", s);
        self.conv_in_b = try loadVec(&w, a, "pipeline.y_embedder.decoder.conv_in", "bias", s);
        // _Decoder.block order is fixed: resnet, attn, resnet, attn, resnet.
        for (0..NUM_DEC_BLOCKS) |i| {
            const pfx = try std.fmt.allocPrint(a, "pipeline.y_embedder.decoder.block.{d}", .{i});
            defer a.free(pfx);
            self.dec_blocks[i] = if (i == 1 or i == 3)
                .{ .attn = try loadAttn(&w, a, pfx, s) }
            else
                .{ .resnet = try loadResnet(&w, a, pfx, s) };
        }
        self.norm_out_w = try loadVec(&w, a, "pipeline.y_embedder.decoder.norm_out", "weight", s);
        self.norm_out_b = try loadVec(&w, a, "pipeline.y_embedder.decoder.norm_out", "bias", s);
        self.conv_out_w = try loadConvW(&w, a, "pipeline.y_embedder.decoder.conv_out", s);
        self.conv_out_b = try loadVec(&w, a, "pipeline.y_embedder.decoder.conv_out", "bias", s);

        // Denoiser: precompute the t=0 timestep embedding through t_embedder.mlp.
        self.denoiser_t = try buildDenoiserT(&w, a, s);
        self.s_proj2_w = try loadConvW(&w, a, "pipeline.s_embedder.proj2", s);
        self.s_proj2_b = try loadVec(&w, a, "pipeline.s_embedder.proj2", "bias", s);
        for (0..NUM_DICO_BLOCKS) |i| {
            const pfx = try std.fmt.allocPrint(a, "pipeline.blocks.{d}", .{i});
            defer a.free(pfx);
            self.blocks[i] = try loadDiCo(&w, a, pfx, s);
        }
        self.yx_w = try loadConvW(&w, a, "pipeline.y_embedder_x", s);
        self.yx_b = try loadVec(&w, a, "pipeline.y_embedder_x", "bias", s);
        self.xemb_w = try loadLinT(&w, a, "pipeline.x_embedder.embedder.0", s);
        self.xemb_b = try loadVec(&w, a, "pipeline.x_embedder.embedder.0", "bias", s);
        self.nerf_pe = try buildNerfPosEmbedding(a, @intCast(VAE_PATCH), 8, s);
        self.cond_embed_w = try loadLinT(&w, a, "pipeline.dec_net.cond_embed", s);
        self.cond_embed_b = try loadVec(&w, a, "pipeline.dec_net.cond_embed", "bias", s);
        self.input_proj_w = try loadLinT(&w, a, "pipeline.dec_net.input_proj", s);
        self.input_proj_b = try loadVec(&w, a, "pipeline.dec_net.input_proj", "bias", s);
        for (0..NUM_MLP_RES) |i| {
            const pfx = try std.fmt.allocPrint(a, "pipeline.dec_net.res_blocks.{d}", .{i});
            defer a.free(pfx);
            self.res_blocks[i] = try loadMlpRes(&w, a, pfx, s);
        }
        self.final_norm_w = try loadVec(&w, a, "pipeline.final_layer.norm", "weight", s);
        self.final_lin_w = try loadLinT(&w, a, "pipeline.final_layer.linear", s);
        self.final_lin_b = try loadVec(&w, a, "pipeline.final_layer.linear", "bias", s);
        return self;
    }

    pub fn deinit(self: *VaeDecoder) void {
        const frees = [_]mlx.mlx_array{
            self.conv_in_w,      self.conv_in_b,    self.norm_out_w,  self.norm_out_b,
            self.conv_out_w,     self.conv_out_b,   self.denoiser_t,  self.s_proj2_w,
            self.s_proj2_b,      self.yx_w,         self.yx_b,        self.xemb_w,
            self.xemb_b,         self.nerf_pe,      self.cond_embed_w, self.cond_embed_b,
            self.input_proj_w,   self.input_proj_b, self.final_norm_w, self.final_lin_w,
            self.final_lin_b,
        };
        for (frees) |f| _ = mlx.mlx_array_free(f);
        for (&self.dec_blocks) |*b| switch (b.*) {
            .resnet => |*r| freeResnet(r),
            .attn => |*at| freeAttn(at),
        };
        for (&self.blocks) |*b| freeDiCo(b);
        for (&self.res_blocks) |*b| freeMlpRes(b);
    }

    /// Decode z [1,128,lh,lw] NCHW → image [1,3,H,W] f32, H=16·lh, W=16·lw.
    pub fn decode(self: *const VaeDecoder, z_nchw: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        // NCHW → NHWC, f32.
        const zt = try transpose(z_nchw, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(zt);
        const latent = try astype(zt, .float32, s);
        defer _ = mlx.mlx_array_free(latent);
        const lsh = mlx.getShape(latent);
        const B = lsh[0];
        const gh = lsh[1];
        const gw = lsh[2];
        const L = gh * gw;
        const H = gh * VAE_PATCH;
        const Wd = gw * VAE_PATCH;

        // cond = _Decoder(latent)  [B,gh,gw,384]
        const cond = try self.decoderForward(latent);
        defer _ = mlx.mlx_array_free(cond);

        // conditioning = s_embedder(0, cond) = proj2(concat([0(128), cond]))
        const zeros128 = try zerosLike(B, gh, gw, 128, s);
        defer _ = mlx.mlx_array_free(zeros128);
        const s_in = try concat(&.{ zeros128, cond }, 3, s);
        defer _ = mlx.mlx_array_free(s_in);
        var conditioning = try conv2d(s_in, self.s_proj2_w, self.s_proj2_b, 1, 0, 1, s);
        for (&self.blocks) |*blk| {
            const nxt = try self.dicoForward(blk, conditioning);
            _ = mlx.mlx_array_free(conditioning);
            conditioning = nxt;
        }
        defer _ = mlx.mlx_array_free(conditioning);
        const cond_flat = try reshape(conditioning, &[_]c_int{ B * L, VAE_HIDDEN }, s); // [B*L,384]
        defer _ = mlx.mlx_array_free(cond_flat);

        // embedded_cond = y_embedder_x(cond) → [B,L,256,32]
        const yx = try conv2d(cond, self.yx_w, self.yx_b, 1, 0, 1, s); // [B,gh,gw,8192]
        defer _ = mlx.mlx_array_free(yx);
        const yx_r = try reshape(yx, &[_]c_int{ B, L, VAE_HIDDEN_X, VAE_PATCH * VAE_PATCH }, s); // [B,L,32,256]
        defer _ = mlx.mlx_array_free(yx_r);
        const embedded_cond = try transpose(yx_r, &[_]c_int{ 0, 1, 3, 2 }, s); // [B,L,256,32]
        defer _ = mlx.mlx_array_free(embedded_cond);

        // image_patches = 0 (noise=0) → pixel_features = concat([0(3), embedded_cond(32)])
        const zeros_ip = try zerosLike(B, L, VAE_PATCH * VAE_PATCH, 3, s); // [B,L,256,3]
        defer _ = mlx.mlx_array_free(zeros_ip);
        const pf_cat = try concat(&.{ zeros_ip, embedded_cond }, 3, s); // [B,L,256,35]
        defer _ = mlx.mlx_array_free(pf_cat);
        const pf_flat = try reshape(pf_cat, &[_]c_int{ B * L, VAE_PATCH * VAE_PATCH, 3 + VAE_HIDDEN_X }, s); // [B*L,256,35]
        defer _ = mlx.mlx_array_free(pf_flat);

        // x_embedder (NeRF): concat DCT pos-emb, linear → [B*L,256,32]
        const nbc = try broadcastPe(self.nerf_pe, B * L, VAE_PATCH * VAE_PATCH, s); // [B*L,256,64]
        defer _ = mlx.mlx_array_free(nbc);
        const nerf_in = try concat(&.{ pf_flat, nbc }, 2, s); // [B*L,256,99]
        defer _ = mlx.mlx_array_free(nerf_in);
        const x_emb = try linearT(nerf_in, self.xemb_w, self.xemb_b, s); // [B*L,256,32]
        defer _ = mlx.mlx_array_free(x_emb);

        // dec_net (SimpleMLPAdaLN): input_proj + cond_embed + 3 res blocks
        var pf = try linearT(x_emb, self.input_proj_w, self.input_proj_b, s); // [B*L,256,32]
        const c_emb = try linearT(cond_flat, self.cond_embed_w, self.cond_embed_b, s); // [B*L,8192]
        defer _ = mlx.mlx_array_free(c_emb);
        const c_r = try reshape(c_emb, &[_]c_int{ B * L, VAE_PATCH * VAE_PATCH, VAE_HIDDEN_X }, s); // [B*L,256,32]
        defer _ = mlx.mlx_array_free(c_r);
        for (&self.res_blocks) |*rb| {
            const nxt = try mlpResForward(rb, pf, c_r, s);
            _ = mlx.mlx_array_free(pf);
            pf = nxt;
        }
        defer _ = mlx.mlx_array_free(pf);

        // final_layer: linear(rms_norm(pf)) → [B*L,256,3]
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, pf, self.final_norm_w, 1e-6, s));
        const pixels = try linearT(normed, self.final_lin_w, self.final_lin_b, s); // [B*L,256,3]
        defer _ = mlx.mlx_array_free(pixels);

        // Unpatchify → [B,H,W,3] → NCHW.
        const p6 = try reshape(pixels, &[_]c_int{ B, gh, gw, VAE_PATCH, VAE_PATCH, 3 }, s);
        defer _ = mlx.mlx_array_free(p6);
        const pt = try transpose(p6, &[_]c_int{ 0, 1, 3, 2, 4, 5 }, s); // [B,gh,16,gw,16,3]
        defer _ = mlx.mlx_array_free(pt);
        const img_nhwc = try reshape(pt, &[_]c_int{ B, H, Wd, 3 }, s);
        defer _ = mlx.mlx_array_free(img_nhwc);
        return transpose(img_nhwc, &[_]c_int{ 0, 3, 1, 2 }, s); // NCHW
    }

    /// Just the `_Decoder` (y_embedder) output cond [B,gh,gw,384] — a bisection
    /// hook so the parity test can isolate the decoder from the denoiser.
    pub fn decoderCond(self: *const VaeDecoder, z_nchw: mlx.mlx_array) !mlx.mlx_array {
        const zt = try transpose(z_nchw, &[_]c_int{ 0, 2, 3, 1 }, self.s);
        defer _ = mlx.mlx_array_free(zt);
        const latent = try astype(zt, .float32, self.s);
        defer _ = mlx.mlx_array_free(latent);
        return self.decoderForward(latent);
    }

    // ── forward sub-blocks ──
    fn decoderForward(self: *const VaeDecoder, latent: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        var x = try conv2d(latent, self.conv_in_w, self.conv_in_b, 1, 1, 1, s);
        for (&self.dec_blocks) |*b| {
            const nxt = switch (b.*) {
                .resnet => |*r| try resnetForward(r, x, s),
                .attn => |*at| try attnForward(at, x, s),
            };
            _ = mlx.mlx_array_free(x);
            x = nxt;
        }
        defer _ = mlx.mlx_array_free(x);
        const gn = try groupNorm(x, self.norm_out_w, self.norm_out_b, VAE_GROUPS, 1e-6, s);
        defer _ = mlx.mlx_array_free(gn);
        const act = try silu(gn, s);
        defer _ = mlx.mlx_array_free(act);
        return conv2d(act, self.conv_out_w, self.conv_out_b, 1, 1, 1, s);
    }

    fn dicoForward(self: *const VaeDecoder, dw: *const DiCoW, inp: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        // AdaLN modulation from the (constant, t=0) timestep vector.
        const st = try silu(self.denoiser_t, s);
        defer _ = mlx.mlx_array_free(st);
        const mod = try linearT(st, dw.adaw, dw.adab, s); // [1,2304]
        defer _ = mlx.mlx_array_free(mod);
        var m: [6]mlx.mlx_array = undefined;
        try splitEqual(mod, 6, 1, &m, s);
        defer for (m) |mm| {
        _ = mlx.mlx_array_free(mm);
    };
        // shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp

        const n1 = try layerNormLast(inp, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(n1);
        const x0 = try modulate4d(n1, m[0], m[1], s);
        defer _ = mlx.mlx_array_free(x0);
        const c1 = try conv2d(x0, dw.c1w, dw.c1b, 1, 0, 1, s);
        defer _ = mlx.mlx_array_free(c1);
        const c2 = try conv2d(c1, dw.c2w, dw.c2b, 1, 1, VAE_HIDDEN, s); // depthwise 3x3
        defer _ = mlx.mlx_array_free(c2);
        const g1 = try gelu(c2, s);
        defer _ = mlx.mlx_array_free(g1);
        // channel attention: sigmoid(conv(avgpool(g1)))
        const red = [_]c_int{ 1, 2 };
        const pooled = try meanAxes(g1, &red, true, s); // [B,1,1,C]
        defer _ = mlx.mlx_array_free(pooled);
        const ca_c = try conv2d(pooled, dw.caw, dw.cab, 1, 0, 1, s);
        defer _ = mlx.mlx_array_free(ca_c);
        const ca_s = try sigmoidA(ca_c, s);
        defer _ = mlx.mlx_array_free(ca_s);
        const attd = try mulA(g1, ca_s, s);
        defer _ = mlx.mlx_array_free(attd);
        const c3 = try conv2d(attd, dw.c3w, dw.c3b, 1, 0, 1, s);
        defer _ = mlx.mlx_array_free(c3);
        // x = inp + gate_msa * c3
        const gate_msa = try reshape4(m[2], s);
        defer _ = mlx.mlx_array_free(gate_msa);
        const gated1 = try mulA(gate_msa, c3, s);
        defer _ = mlx.mlx_array_free(gated1);
        const x1 = try addA(inp, gated1, s);
        defer _ = mlx.mlx_array_free(x1);
        // MLP half
        const n2 = try layerNormLast(x1, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(n2);
        const xm = try modulate4d(n2, m[3], m[4], s);
        defer _ = mlx.mlx_array_free(xm);
        const c4 = try conv2d(xm, dw.c4w, dw.c4b, 1, 0, 1, s);
        defer _ = mlx.mlx_array_free(c4);
        const g2 = try gelu(c4, s);
        defer _ = mlx.mlx_array_free(g2);
        const c5 = try conv2d(g2, dw.c5w, dw.c5b, 1, 0, 1, s);
        defer _ = mlx.mlx_array_free(c5);
        const gate_mlp = try reshape4(m[5], s);
        defer _ = mlx.mlx_array_free(gate_mlp);
        const gated2 = try mulA(gate_mlp, c5, s);
        defer _ = mlx.mlx_array_free(gated2);
        return addA(x1, gated2, s);
    }
};

// ── VAE block forwards (free functions over weight groups) ──
/// Reshape a [1,C] modulation vector to [1,1,1,C] for NHWC broadcast.
fn reshape4(v: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c = mlx.getShape(v)[1];
    return reshape(v, &[_]c_int{ 1, 1, 1, c }, s);
}
/// _modulate (4D): x*(1+scale)+shift with scale/shift [1,C] → [1,1,1,C].
fn modulate4d(x: mlx.mlx_array, shift: mlx.mlx_array, scale: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh4 = try reshape4(shift, s);
    defer _ = mlx.mlx_array_free(sh4);
    const sc4 = try reshape4(scale, s);
    defer _ = mlx.mlx_array_free(sc4);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const one_p = try addA(sc4, one, s);
    defer _ = mlx.mlx_array_free(one_p);
    const xs = try mulA(x, one_p, s);
    defer _ = mlx.mlx_array_free(xs);
    return addA(xs, sh4, s);
}
fn zerosLike(d0: c_int, d1: c_int, d2: c_int, d3: c_int, s: S) !mlx.mlx_array {
    const shape = [_]c_int{ d0, d1, d2, d3 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&o, &shape, 4, .float32, s));
    return o;
}
fn broadcastPe(pe: mlx.mlx_array, n: c_int, area: c_int, s: S) !mlx.mlx_array {
    const nf2 = mlx.getShape(pe)[2];
    const shape = [_]c_int{ n, area, nf2 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_broadcast_to(&o, pe, &shape, 3, s));
    return o;
}
fn resnetForward(rw: *const ResnetW, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const gn1 = try groupNorm(x, rw.n1w, rw.n1b, VAE_GROUPS, 1e-6, s);
    defer _ = mlx.mlx_array_free(gn1);
    const a1 = try silu(gn1, s);
    defer _ = mlx.mlx_array_free(a1);
    const h1 = try conv2d(a1, rw.c1w, rw.c1b, 1, 1, 1, s);
    defer _ = mlx.mlx_array_free(h1);
    const gn2 = try groupNorm(h1, rw.n2w, rw.n2b, VAE_GROUPS, 1e-6, s);
    defer _ = mlx.mlx_array_free(gn2);
    const a2 = try silu(gn2, s);
    defer _ = mlx.mlx_array_free(a2);
    const h2 = try conv2d(a2, rw.c2w, rw.c2b, 1, 1, 1, s);
    defer _ = mlx.mlx_array_free(h2);
    return addA(x, h2, s);
}
fn attnForward(aw: *const AttnW, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const h = try groupNorm(x, aw.nw, aw.nb, VAE_GROUPS, 1e-6, s);
    defer _ = mlx.mlx_array_free(h);
    const q = try conv2d(h, aw.qw, aw.qb, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try conv2d(h, aw.kw, aw.kb, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try conv2d(h, aw.vw, aw.vb, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(v);
    const sh = mlx.getShape(q);
    const B = sh[0];
    const Hh = sh[1];
    const Ww = sh[2];
    const C = sh[3];
    const p = VAE_ATTN_PATCH;
    const pad_h = @mod(p - @mod(Hh, p), p);
    const pad_w = @mod(p - @mod(Ww, p), p);
    var qp = q;
    var kp = k;
    var vp = v;
    var to_free_pad = false;
    if (pad_h != 0 or pad_w != 0) {
        qp = try padEdgeHW(q, pad_h, pad_w, s);
        kp = try padEdgeHW(k, pad_h, pad_w, s);
        vp = try padEdgeHW(v, pad_h, pad_w, s);
        to_free_pad = true;
    }
    defer if (to_free_pad) {
        _ = mlx.mlx_array_free(qp);
        _ = mlx.mlx_array_free(kp);
        _ = mlx.mlx_array_free(vp);
    };
    const nph = @divExact(Hh + pad_h, p);
    const npw = @divExact(Ww + pad_w, p);
    const qpat = try toPatches(qp, p, nph, npw, s); // [B*nph*npw, p*p, C]
    defer _ = mlx.mlx_array_free(qpat);
    const kpat = try toPatches(kp, p, nph, npw, s);
    defer _ = mlx.mlx_array_free(kpat);
    const vpat = try toPatches(vp, p, nph, npw, s);
    defer _ = mlx.mlx_array_free(vpat);
    // SDPA (single head): expand to [N,1,pp,C]
    const q4 = try expandDim1(qpat, s);
    defer _ = mlx.mlx_array_free(q4);
    const k4 = try expandDim1(kpat, s);
    defer _ = mlx.mlx_array_free(k4);
    const v4 = try expandDim1(vpat, s);
    defer _ = mlx.mlx_array_free(v4);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(C)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q4, k4, v4, scale, "", null_a, null_a, s));
    const at3 = try squeezeDim1(attn, s); // [N,pp,C]
    defer _ = mlx.mlx_array_free(at3);
    const back = try fromPatches(at3, B, nph, npw, p, C, s); // [B,padH,padW,C]
    defer _ = mlx.mlx_array_free(back);
    const cropped = if (pad_h != 0 or pad_w != 0) try cropHW(back, Hh, Ww, s) else try contig(back, s);
    defer _ = mlx.mlx_array_free(cropped);
    const proj = try conv2d(cropped, aw.pw, aw.pb, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(proj);
    return addA(x, proj, s);
}
fn mlpResForward(mw: *const MlpResW, x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sy = try silu(y, s);
    defer _ = mlx.mlx_array_free(sy);
    const mod = try linearT(sy, mw.adaw, mw.adab, s); // [B*L,256,96]
    defer _ = mlx.mlx_array_free(mod);
    var m: [3]mlx.mlx_array = undefined;
    try splitEqual(mod, 3, 2, &m, s);
    defer for (m) |mm| {
        _ = mlx.mlx_array_free(mm);
    };
    const ln = try layerNormLast(x, mw.lnw, mw.lnb, 1e-5, s); // nn.LayerNorm default eps 1e-5
    defer _ = mlx.mlx_array_free(ln);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const one_p = try addA(m[1], one, s);
    defer _ = mlx.mlx_array_free(one_p);
    const scaled = try mulA(ln, one_p, s);
    defer _ = mlx.mlx_array_free(scaled);
    const h = try addA(scaled, m[0], s);
    defer _ = mlx.mlx_array_free(h);
    const l0 = try linearT(h, mw.m0w, mw.m0b, s);
    defer _ = mlx.mlx_array_free(l0);
    const a0 = try silu(l0, s);
    defer _ = mlx.mlx_array_free(a0);
    const l2 = try linearT(a0, mw.m2w, mw.m2b, s);
    defer _ = mlx.mlx_array_free(l2);
    const gated = try mulA(m[2], l2, s);
    defer _ = mlx.mlx_array_free(gated);
    return addA(x, gated, s);
}

// ── patch helpers for AttnBlock ──
fn padEdgeHW(x: mlx.mlx_array, pad_h: c_int, pad_w: c_int, s: S) !mlx.mlx_array {
    // Edge-pad H and W only; the "edge" mode implementation wants ALL axes
    // specified (partial-axes form drives slice_update with too few indices).
    const axes = [_]c_int{ 0, 1, 2, 3 };
    const low = [_]c_int{ 0, 0, 0, 0 };
    const high = [_]c_int{ 0, pad_h, pad_w, 0 };
    const zero = scalarF(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(&o, x, &axes, 4, &low, 4, &high, 4, zero, "edge", s));
    return o;
}
fn toPatches(x: mlx.mlx_array, p: c_int, nph: c_int, npw: c_int, s: S) !mlx.mlx_array {
    const C = mlx.getShape(x)[3];
    const B = mlx.getShape(x)[0];
    const r = try reshape(x, &[_]c_int{ B, nph, p, npw, p, C }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 1, 3, 2, 4, 5 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ B * nph * npw, p * p, C }, s);
}
fn fromPatches(x: mlx.mlx_array, B: c_int, nph: c_int, npw: c_int, p: c_int, C: c_int, s: S) !mlx.mlx_array {
    const r = try reshape(x, &[_]c_int{ B, nph, npw, p, p, C }, s);
    defer _ = mlx.mlx_array_free(r);
    const t = try transpose(r, &[_]c_int{ 0, 1, 3, 2, 4, 5 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ B, nph * p, npw * p, C }, s);
}
fn cropHW(x: mlx.mlx_array, h: c_int, wd: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const lo = [_]c_int{ 0, 0, 0, 0 };
    const hi = [_]c_int{ sh[0], h, wd, sh[3] };
    const st = [_]c_int{ 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 4, &hi, 4, &st, 4, s));
    return contig(o, s);
}
fn expandDim1(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&o, x, 1, s));
    return o;
}
fn squeezeDim1(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [N,1,pp,C]
    return reshape(x, &[_]c_int{ sh[0], sh[2], sh[3] }, s);
}

// ── VAE weight-group loaders ──
fn buildDenoiserT(w: *const Weights, a: std.mem.Allocator, s: S) !mlx.mlx_array {
    // t=0 sinusoidal embedding: [ones(128), zeros(128)] → mlp.0 → silu → mlp.2
    var emb0: [256]f32 = undefined;
    for (0..128) |i| emb0[i] = 1.0;
    for (128..256) |i| emb0[i] = 0.0;
    const shape = [_]c_int{ 1, 256 };
    const raw = mlx.mlx_array_new_data(&emb0, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const m0w = try loadLinT(w, a, "pipeline.t_embedder.mlp.0", s);
    defer _ = mlx.mlx_array_free(m0w);
    const m0b = try loadVec(w, a, "pipeline.t_embedder.mlp.0", "bias", s);
    defer _ = mlx.mlx_array_free(m0b);
    const m2w = try loadLinT(w, a, "pipeline.t_embedder.mlp.2", s);
    defer _ = mlx.mlx_array_free(m2w);
    const m2b = try loadVec(w, a, "pipeline.t_embedder.mlp.2", "bias", s);
    defer _ = mlx.mlx_array_free(m2b);
    const l0 = try linearT(raw, m0w, m0b, s);
    defer _ = mlx.mlx_array_free(l0);
    const act = try silu(l0, s);
    defer _ = mlx.mlx_array_free(act);
    const l2 = try linearT(act, m2w, m2b, s);
    defer _ = mlx.mlx_array_free(l2);
    return contig(l2, s);
}
fn loadResnet(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !ResnetW {
    const n1 = try std.fmt.allocPrint(a, "{s}.norm1", .{pfx});
    defer a.free(n1);
    const c1 = try std.fmt.allocPrint(a, "{s}.conv1", .{pfx});
    defer a.free(c1);
    const n2 = try std.fmt.allocPrint(a, "{s}.norm2", .{pfx});
    defer a.free(n2);
    const c2 = try std.fmt.allocPrint(a, "{s}.conv2", .{pfx});
    defer a.free(c2);
    return .{
        .n1w = try loadVec(w, a, n1, "weight", s),
        .n1b = try loadVec(w, a, n1, "bias", s),
        .c1w = try loadConvW(w, a, c1, s),
        .c1b = try loadVec(w, a, c1, "bias", s),
        .n2w = try loadVec(w, a, n2, "weight", s),
        .n2b = try loadVec(w, a, n2, "bias", s),
        .c2w = try loadConvW(w, a, c2, s),
        .c2b = try loadVec(w, a, c2, "bias", s),
    };
}
fn freeResnet(r: *ResnetW) void {
    inline for (.{ r.n1w, r.n1b, r.c1w, r.c1b, r.n2w, r.n2b, r.c2w, r.c2b }) |f| _ = mlx.mlx_array_free(f);
}
fn loadAttn(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !AttnW {
    const nn_ = try std.fmt.allocPrint(a, "{s}.norm", .{pfx});
    defer a.free(nn_);
    const qk = try std.fmt.allocPrint(a, "{s}.q", .{pfx});
    defer a.free(qk);
    const kk = try std.fmt.allocPrint(a, "{s}.k", .{pfx});
    defer a.free(kk);
    const vk = try std.fmt.allocPrint(a, "{s}.v", .{pfx});
    defer a.free(vk);
    const pk = try std.fmt.allocPrint(a, "{s}.proj_out", .{pfx});
    defer a.free(pk);
    return .{
        .nw = try loadVec(w, a, nn_, "weight", s),
        .nb = try loadVec(w, a, nn_, "bias", s),
        .qw = try loadConvW(w, a, qk, s),
        .qb = try loadVec(w, a, qk, "bias", s),
        .kw = try loadConvW(w, a, kk, s),
        .kb = try loadVec(w, a, kk, "bias", s),
        .vw = try loadConvW(w, a, vk, s),
        .vb = try loadVec(w, a, vk, "bias", s),
        .pw = try loadConvW(w, a, pk, s),
        .pb = try loadVec(w, a, pk, "bias", s),
    };
}
fn freeAttn(at: *AttnW) void {
    inline for (.{ at.nw, at.nb, at.qw, at.qb, at.kw, at.kb, at.vw, at.vb, at.pw, at.pb }) |f| _ = mlx.mlx_array_free(f);
}
fn loadDiCo(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !DiCoW {
    const c1 = try std.fmt.allocPrint(a, "{s}.conv1", .{pfx});
    defer a.free(c1);
    const c2 = try std.fmt.allocPrint(a, "{s}.conv2", .{pfx});
    defer a.free(c2);
    const c3 = try std.fmt.allocPrint(a, "{s}.conv3", .{pfx});
    defer a.free(c3);
    const ca = try std.fmt.allocPrint(a, "{s}.ca.1", .{pfx});
    defer a.free(ca);
    const c4 = try std.fmt.allocPrint(a, "{s}.conv4", .{pfx});
    defer a.free(c4);
    const c5 = try std.fmt.allocPrint(a, "{s}.conv5", .{pfx});
    defer a.free(c5);
    const ada = try std.fmt.allocPrint(a, "{s}.adaLN_modulation.1", .{pfx});
    defer a.free(ada);
    return .{
        .c1w = try loadConvW(w, a, c1, s),
        .c1b = try loadVec(w, a, c1, "bias", s),
        .c2w = try loadConvW(w, a, c2, s),
        .c2b = try loadVec(w, a, c2, "bias", s),
        .c3w = try loadConvW(w, a, c3, s),
        .c3b = try loadVec(w, a, c3, "bias", s),
        .caw = try loadConvW(w, a, ca, s),
        .cab = try loadVec(w, a, ca, "bias", s),
        .c4w = try loadConvW(w, a, c4, s),
        .c4b = try loadVec(w, a, c4, "bias", s),
        .c5w = try loadConvW(w, a, c5, s),
        .c5b = try loadVec(w, a, c5, "bias", s),
        .adaw = try loadLinT(w, a, ada, s),
        .adab = try loadVec(w, a, ada, "bias", s),
    };
}
fn freeDiCo(d: *DiCoW) void {
    inline for (.{ d.c1w, d.c1b, d.c2w, d.c2b, d.c3w, d.c3b, d.caw, d.cab, d.c4w, d.c4b, d.c5w, d.c5b, d.adaw, d.adab }) |f| _ = mlx.mlx_array_free(f);
}
fn loadMlpRes(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !MlpResW {
    const ln = try std.fmt.allocPrint(a, "{s}.in_ln", .{pfx});
    defer a.free(ln);
    const m0 = try std.fmt.allocPrint(a, "{s}.mlp.0", .{pfx});
    defer a.free(m0);
    const m2 = try std.fmt.allocPrint(a, "{s}.mlp.2", .{pfx});
    defer a.free(m2);
    const ada = try std.fmt.allocPrint(a, "{s}.adaLN_modulation.1", .{pfx});
    defer a.free(ada);
    return .{
        .lnw = try loadVec(w, a, ln, "weight", s),
        .lnb = try loadVec(w, a, ln, "bias", s),
        .m0w = try loadLinT(w, a, m0, s),
        .m0b = try loadVec(w, a, m0, "bias", s),
        .m2w = try loadLinT(w, a, m2, s),
        .m2b = try loadVec(w, a, m2, "bias", s),
        .adaw = try loadLinT(w, a, ada, s),
        .adab = try loadVec(w, a, ada, "bias", s),
    };
}
fn freeMlpRes(m: *MlpResW) void {
    inline for (.{ m.lnw, m.lnb, m.m0w, m.m0b, m.m2w, m.m2b, m.adaw, m.adab }) |f| _ = mlx.mlx_array_free(f);
}

// ══════════════════════════════════════════════════════════════════════════
// MageFlow DiT — native-resolution double-stream flow transformer (NR-MMDiT).
// Ported from mflux `mage_flow_transformer/*`. 12 joint-attention double-stream
// blocks (image + text streams), centered 3-axis image RoPE, param-free
// LayerNorms, RMSNorm q/k, GELU-tanh feed-forward. Runs in f32 (parity path).
// ══════════════════════════════════════════════════════════════════════════

const DIT_HEADS: c_int = 24;
const DIT_HEAD_DIM: c_int = 128;
const DIT_HIDDEN: c_int = 3072;
const DIT_AXES = [3]c_int{ 16, 56, 56 };
const DIT_THETA: f64 = 10000.0;
const DIT_DEPTH = 12;

/// GELU tanh approximation (nn.gelu_approx): 0.5x(1+tanh(√(2/π)(x+0.044715x³))).
fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c: f32 = 0.7978845608028654;
    const k = scalarF(0.044715);
    defer _ = mlx.mlx_array_free(k);
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const kx3 = try mulA(x3, k, s);
    defer _ = mlx.mlx_array_free(kx3);
    const inner = try addA(x, kx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const ca = scalarF(c);
    defer _ = mlx.mlx_array_free(ca);
    const cin = try mulA(inner, ca, s);
    defer _ = mlx.mlx_array_free(cin);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, cin, s));
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}
fn rmsNormLast(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}

const DitAttnW = struct {
    qw: mlx.mlx_array,
    qb: mlx.mlx_array,
    kw: mlx.mlx_array,
    kb: mlx.mlx_array,
    vw: mlx.mlx_array,
    vb: mlx.mlx_array,
    aqw: mlx.mlx_array,
    aqb: mlx.mlx_array,
    akw: mlx.mlx_array,
    akb: mlx.mlx_array,
    avw: mlx.mlx_array,
    avb: mlx.mlx_array,
    nq: mlx.mlx_array,
    nk: mlx.mlx_array,
    naq: mlx.mlx_array,
    nak: mlx.mlx_array,
    ow: mlx.mlx_array,
    ob: mlx.mlx_array,
    aow: mlx.mlx_array,
    aob: mlx.mlx_array,
};
const DitBlockW = struct {
    img_mod_w: mlx.mlx_array,
    img_mod_b: mlx.mlx_array,
    txt_mod_w: mlx.mlx_array,
    txt_mod_b: mlx.mlx_array,
    attn: DitAttnW,
    img0w: mlx.mlx_array,
    img0b: mlx.mlx_array,
    img2w: mlx.mlx_array,
    img2b: mlx.mlx_array,
    txt0w: mlx.mlx_array,
    txt0b: mlx.mlx_array,
    txt2w: mlx.mlx_array,
    txt2b: mlx.mlx_array,
};

pub const Dit = struct {
    allocator: std.mem.Allocator,
    s: S,
    img_in_w: mlx.mlx_array,
    img_in_b: mlx.mlx_array,
    txt_norm_w: mlx.mlx_array,
    txt_in_w: mlx.mlx_array,
    txt_in_b: mlx.mlx_array,
    t1w: mlx.mlx_array,
    t1b: mlx.mlx_array,
    t2w: mlx.mlx_array,
    t2b: mlx.mlx_array,
    blocks: [DIT_DEPTH]DitBlockW,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    proj_out_w: mlx.mlx_array,
    proj_out_b: mlx.mlx_array,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Dit {
        const dir = try std.fmt.allocPrint(allocator, "{s}/transformer", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        const a = allocator;
        var self: Dit = undefined;
        self.allocator = allocator;
        self.s = s;
        self.img_in_w = try loadLinT(&w, a, "img_in", s);
        self.img_in_b = try loadVec(&w, a, "img_in", "bias", s);
        self.txt_norm_w = try loadVec(&w, a, "txt_norm", "weight", s);
        self.txt_in_w = try loadLinT(&w, a, "txt_in", s);
        self.txt_in_b = try loadVec(&w, a, "txt_in", "bias", s);
        self.t1w = try loadLinT(&w, a, "time_text_embed.timestep_embedder.linear_1", s);
        self.t1b = try loadVec(&w, a, "time_text_embed.timestep_embedder.linear_1", "bias", s);
        self.t2w = try loadLinT(&w, a, "time_text_embed.timestep_embedder.linear_2", s);
        self.t2b = try loadVec(&w, a, "time_text_embed.timestep_embedder.linear_2", "bias", s);
        for (0..DIT_DEPTH) |i| {
            self.blocks[i] = try loadDitBlock(&w, a, i, s);
        }
        self.norm_out_w = try loadLinT(&w, a, "norm_out.linear", s);
        self.norm_out_b = try loadVec(&w, a, "norm_out.linear", "bias", s);
        self.proj_out_w = try loadLinT(&w, a, "proj_out", s);
        self.proj_out_b = try loadVec(&w, a, "proj_out", "bias", s);
        return self;
    }

    pub fn deinit(self: *Dit) void {
        const top = [_]mlx.mlx_array{
            self.img_in_w,  self.img_in_b, self.txt_norm_w, self.txt_in_w,
            self.txt_in_b,  self.t1w,      self.t1b,        self.t2w,
            self.t2b,       self.norm_out_w, self.norm_out_b, self.proj_out_w,
            self.proj_out_b,
        };
        for (top) |f| _ = mlx.mlx_array_free(f);
        for (&self.blocks) |*b| freeDitBlock(b);
    }

    /// Predict velocity for one flow step. img [B,Limg,128], txt [B,Ltxt,2560],
    /// t scalar. img_shape = (frames, lh, lw). txt_mask [B,Ltxt] (1=keep) or
    /// null. Returns [B,Limg,128] f32.
    pub fn forward(
        self: *const Dit,
        img_in: mlx.mlx_array,
        txt_in: mlx.mlx_array,
        t: f32,
        frames: c_int,
        lh: c_int,
        lw: c_int,
        txt_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        const a = self.allocator;
        const s = self.s;
        const Limg = mlx.getShape(img_in)[1];
        const B = mlx.getShape(img_in)[0];
        const Ltxt = mlx.getShape(txt_in)[1];

        // RoPE cos/sin [Limg, sum(axes)/2].
        const rope = try buildDitRope(a, frames, lh, lw, s);
        const cos = rope[0];
        defer _ = mlx.mlx_array_free(cos);
        const sin = rope[1];
        defer _ = mlx.mlx_array_free(sin);

        const img_f = try astype(img_in, .float32, s);
        defer _ = mlx.mlx_array_free(img_f);
        var img = try linearT(img_f, self.img_in_w, self.img_in_b, s); // [B,Limg,3072]
        const txt_f = try astype(txt_in, .float32, s);
        defer _ = mlx.mlx_array_free(txt_f);
        const txt_n = try rmsNormLast(txt_f, self.txt_norm_w, 1e-6, s);
        defer _ = mlx.mlx_array_free(txt_n);
        var txt = try linearT(txt_n, self.txt_in_w, self.txt_in_b, s); // [B,Ltxt,3072]

        const temb = try self.timeTextEmbed(t, B); // [B,3072]
        defer _ = mlx.mlx_array_free(temb);

        var mask: ?mlx.mlx_array = null;
        if (txt_mask) |tm| mask = try buildDitMask(tm, Limg, s);
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };

        for (&self.blocks) |*blk| {
            const res = try self.ditBlock(blk, img, txt, temb, cos, sin, mask, Ltxt);
            _ = mlx.mlx_array_free(img);
            _ = mlx.mlx_array_free(txt);
            txt = res[0];
            img = res[1];
        }
        defer _ = mlx.mlx_array_free(txt);
        defer _ = mlx.mlx_array_free(img);

        // norm_out (AdaLayerNormContinuous) + proj_out
        const st = try silu(temb, s);
        defer _ = mlx.mlx_array_free(st);
        const nmod = try linearT(st, self.norm_out_w, self.norm_out_b, s); // [B,6144]
        defer _ = mlx.mlx_array_free(nmod);
        var sc2: [2]mlx.mlx_array = undefined;
        try splitEqual(nmod, 2, 1, &sc2, s); // scale, shift
        defer for (sc2) |x| {
            _ = mlx.mlx_array_free(x);
        };
        const ln = try layerNormLast(img, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(ln);
        const modded = try modulateSeqNoGate(ln, sc2[0], sc2[1], s);
        defer _ = mlx.mlx_array_free(modded);
        return linearT(modded, self.proj_out_w, self.proj_out_b, s); // [B,Limg,128]
    }

    fn timeTextEmbed(self: *const Dit, t: f32, B: c_int) !mlx.mlx_array {
        const s = self.s;
        // Sinusoidal proj [B,256]: freqs bf16-free (f32 parity path).
        var buf: [256]f32 = undefined;
        const half = 128;
        const scale: f64 = 1000.0;
        const max_period: f64 = 10000.0;
        for (0..half) |j| {
            const exponent = -std.math.log(f64, std.math.e, max_period) * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(half));
            const freq = @exp(exponent);
            const ang = @as(f64, t) * freq * scale;
            buf[j] = @floatCast(@cos(ang)); // flip_sin_to_cos → cos first
            buf[half + j] = @floatCast(@sin(ang));
        }
        const shape = [_]c_int{ 1, 256 };
        const raw = mlx.mlx_array_new_data(&buf, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(raw);
        var proj = try contig(raw, s);
        if (B > 1) {
            const bshape = [_]c_int{ B, 256 };
            var bo = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_broadcast_to(&bo, proj, &bshape, 2, s));
            _ = mlx.mlx_array_free(proj);
            proj = bo;
        }
        defer _ = mlx.mlx_array_free(proj);
        const l1 = try linearT(proj, self.t1w, self.t1b, s);
        defer _ = mlx.mlx_array_free(l1);
        const a1 = try silu(l1, s);
        defer _ = mlx.mlx_array_free(a1);
        return linearT(a1, self.t2w, self.t2b, s);
    }

    fn ditBlock(
        self: *const Dit,
        bw: *const DitBlockW,
        img: mlx.mlx_array,
        txt: mlx.mlx_array,
        temb: mlx.mlx_array,
        cos: mlx.mlx_array,
        sin: mlx.mlx_array,
        mask: ?mlx.mlx_array,
        Ltxt: c_int,
    ) ![2]mlx.mlx_array {
        const s = self.s;
        const st = try silu(temb, s);
        defer _ = mlx.mlx_array_free(st);
        const img_mod = try linearT(st, bw.img_mod_w, bw.img_mod_b, s); // [B,18432]
        defer _ = mlx.mlx_array_free(img_mod);
        const txt_mod = try linearT(st, bw.txt_mod_w, bw.txt_mod_b, s);
        defer _ = mlx.mlx_array_free(txt_mod);
        var im2: [2]mlx.mlx_array = undefined;
        try splitEqual(img_mod, 2, 1, &im2, s);
        defer for (im2) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var tm2: [2]mlx.mlx_array = undefined;
        try splitEqual(txt_mod, 2, 1, &tm2, s);
        defer for (tm2) |x| {
            _ = mlx.mlx_array_free(x);
        };

        // Attention half.
        const in1 = try layerNormLast(img, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(in1);
        const imod1 = try modulateSeq(in1, im2[0], s);
        defer _ = mlx.mlx_array_free(imod1.hidden);
        defer _ = mlx.mlx_array_free(imod1.gate);
        const tn1 = try layerNormLast(txt, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(tn1);
        const tmod1 = try modulateSeq(tn1, tm2[0], s);
        defer _ = mlx.mlx_array_free(tmod1.hidden);
        defer _ = mlx.mlx_array_free(tmod1.gate);

        const attn = try jointAttn(&bw.attn, imod1.hidden, tmod1.hidden, cos, sin, mask, Ltxt, s);
        const img_attn = attn[0];
        defer _ = mlx.mlx_array_free(img_attn);
        const txt_attn = attn[1];
        defer _ = mlx.mlx_array_free(txt_attn);

        const img_g1 = try gateAdd(img, imod1.gate, img_attn, s);
        defer _ = mlx.mlx_array_free(img_g1);
        const txt_g1 = try gateAdd(txt, tmod1.gate, txt_attn, s);
        defer _ = mlx.mlx_array_free(txt_g1);

        // MLP half.
        const in2 = try layerNormLast(img_g1, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(in2);
        const imod2 = try modulateSeq(in2, im2[1], s);
        defer _ = mlx.mlx_array_free(imod2.hidden);
        defer _ = mlx.mlx_array_free(imod2.gate);
        const img_ff = try feedForward(bw.img0w, bw.img0b, bw.img2w, bw.img2b, imod2.hidden, s);
        defer _ = mlx.mlx_array_free(img_ff);
        const img_out = try gateAdd(img_g1, imod2.gate, img_ff, s);

        const tn2 = try layerNormLast(txt_g1, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(tn2);
        const tmod2 = try modulateSeq(tn2, tm2[1], s);
        defer _ = mlx.mlx_array_free(tmod2.hidden);
        defer _ = mlx.mlx_array_free(tmod2.gate);
        const txt_ff = try feedForward(bw.txt0w, bw.txt0b, bw.txt2w, bw.txt2b, tmod2.hidden, s);
        defer _ = mlx.mlx_array_free(txt_ff);
        const txt_out = try gateAdd(txt_g1, tmod2.gate, txt_ff, s);

        return .{ txt_out, img_out }; // (encoder_hidden_states, hidden_states)
    }
};

const ModOut = struct { hidden: mlx.mlx_array, gate: mlx.mlx_array };
/// _modulate: split mod[B,3C] → shift,scale,gate; return (h*(1+scale)+shift, gate).
fn modulateSeq(hidden: mlx.mlx_array, mod: mlx.mlx_array, s: S) !ModOut {
    var parts: [3]mlx.mlx_array = undefined;
    try splitEqual(mod, 3, 1, &parts, s); // shift, scale, gate
    defer {
        _ = mlx.mlx_array_free(parts[0]);
        _ = mlx.mlx_array_free(parts[1]);
    }
    const h = try modulateSeqNoGate(hidden, parts[1], parts[0], s);
    return .{ .hidden = h, .gate = parts[2] };
}
/// hidden*(1+scale[:,None,:]) + shift[:,None,:] — scale/shift [B,C].
fn modulateSeqNoGate(hidden: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const C = mlx.getShape(scale)[1];
    const B = mlx.getShape(scale)[0];
    const sc3 = try reshape(scale, &[_]c_int{ B, 1, C }, s);
    defer _ = mlx.mlx_array_free(sc3);
    const sh3 = try reshape(shift, &[_]c_int{ B, 1, C }, s);
    defer _ = mlx.mlx_array_free(sh3);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const onep = try addA(sc3, one, s);
    defer _ = mlx.mlx_array_free(onep);
    const scaled = try mulA(hidden, onep, s);
    defer _ = mlx.mlx_array_free(scaled);
    return addA(scaled, sh3, s);
}
/// residual + gate[:,None,:] * delta.
fn gateAdd(residual: mlx.mlx_array, gate: mlx.mlx_array, delta: mlx.mlx_array, s: S) !mlx.mlx_array {
    const C = mlx.getShape(gate)[1];
    const B = mlx.getShape(gate)[0];
    const g3 = try reshape(gate, &[_]c_int{ B, 1, C }, s);
    defer _ = mlx.mlx_array_free(g3);
    const gd = try mulA(g3, delta, s);
    defer _ = mlx.mlx_array_free(gd);
    return addA(residual, gd, s);
}
fn feedForward(w0: mlx.mlx_array, b0: mlx.mlx_array, w2: mlx.mlx_array, b2: mlx.mlx_array, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const p = try linearT(x, w0, b0, s);
    defer _ = mlx.mlx_array_free(p);
    const g = try geluTanh(p, s);
    defer _ = mlx.mlx_array_free(g);
    return linearT(g, w2, b2, s);
}
fn splitHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,L,3072]
    return reshape(x, &[_]c_int{ sh[0], sh[1], DIT_HEADS, DIT_HEAD_DIM }, s);
}
/// Adjacent-pair complex RoPE (f32). x [B,L,H,128]; cos/sin [L,64].
fn applyRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,L,H,128]
    const B = sh[0];
    const L = sh[1];
    const Hh = sh[2];
    const D = sh[3];
    const pairs = try reshape(x, &[_]c_int{ B, L, Hh, @divExact(D, 2), 2 }, s);
    defer _ = mlx.mlx_array_free(pairs);
    var ri: [2]mlx.mlx_array = undefined;
    try splitEqual(pairs, 2, 4, &ri, s); // real, imag [B,L,H,64,1]
    defer for (ri) |x2| {
        _ = mlx.mlx_array_free(x2);
    };
    const real = try reshape(ri[0], &[_]c_int{ B, L, Hh, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(real);
    const imag = try reshape(ri[1], &[_]c_int{ B, L, Hh, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(imag);
    const cos4 = try reshape(cos, &[_]c_int{ 1, L, 1, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(cos4);
    const sin4 = try reshape(sin, &[_]c_int{ 1, L, 1, @divExact(D, 2) }, s);
    defer _ = mlx.mlx_array_free(sin4);
    const rc = try mulA(real, cos4, s);
    defer _ = mlx.mlx_array_free(rc);
    const is_ = try mulA(imag, sin4, s);
    defer _ = mlx.mlx_array_free(is_);
    const rot_r = try subA(rc, is_, s);
    defer _ = mlx.mlx_array_free(rot_r);
    const rs = try mulA(real, sin4, s);
    defer _ = mlx.mlx_array_free(rs);
    const ic = try mulA(imag, cos4, s);
    defer _ = mlx.mlx_array_free(ic);
    const rot_i = try addA(rs, ic, s);
    defer _ = mlx.mlx_array_free(rot_i);
    // stack([rot_r, rot_i], -1) → [B,L,H,64,2] → reshape [B,L,H,128]
    const rr5 = try reshape(rot_r, &[_]c_int{ B, L, Hh, @divExact(D, 2), 1 }, s);
    defer _ = mlx.mlx_array_free(rr5);
    const ri5 = try reshape(rot_i, &[_]c_int{ B, L, Hh, @divExact(D, 2), 1 }, s);
    defer _ = mlx.mlx_array_free(ri5);
    const stacked = try concat(&.{ rr5, ri5 }, 4, s);
    defer _ = mlx.mlx_array_free(stacked);
    return reshape(stacked, &[_]c_int{ B, L, Hh, D }, s);
}
fn jointAttn(aw: *const DitAttnW, img_in: mlx.mlx_array, txt_in: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, mask: ?mlx.mlx_array, Ltxt: c_int, s: S) ![2]mlx.mlx_array {
    // Image q/k/v.
    const iq0 = try linearT(img_in, aw.qw, aw.qb, s);
    defer _ = mlx.mlx_array_free(iq0);
    const ik0 = try linearT(img_in, aw.kw, aw.kb, s);
    defer _ = mlx.mlx_array_free(ik0);
    const iv0 = try linearT(img_in, aw.vw, aw.vb, s);
    defer _ = mlx.mlx_array_free(iv0);
    const iqh = try splitHeads(iq0, s);
    defer _ = mlx.mlx_array_free(iqh);
    const ikh = try splitHeads(ik0, s);
    defer _ = mlx.mlx_array_free(ikh);
    const ivh = try splitHeads(iv0, s);
    defer _ = mlx.mlx_array_free(ivh);
    const iqn = try rmsNormLast(iqh, aw.nq, 1e-6, s);
    defer _ = mlx.mlx_array_free(iqn);
    const ikn = try rmsNormLast(ikh, aw.nk, 1e-6, s);
    defer _ = mlx.mlx_array_free(ikn);
    const iqr = try applyRope(iqn, cos, sin, s);
    defer _ = mlx.mlx_array_free(iqr);
    const ikr = try applyRope(ikn, cos, sin, s);
    defer _ = mlx.mlx_array_free(ikr);
    // Text q/k/v.
    const tq0 = try linearT(txt_in, aw.aqw, aw.aqb, s);
    defer _ = mlx.mlx_array_free(tq0);
    const tk0 = try linearT(txt_in, aw.akw, aw.akb, s);
    defer _ = mlx.mlx_array_free(tk0);
    const tv0 = try linearT(txt_in, aw.avw, aw.avb, s);
    defer _ = mlx.mlx_array_free(tv0);
    const tqh = try splitHeads(tq0, s);
    defer _ = mlx.mlx_array_free(tqh);
    const tkh = try splitHeads(tk0, s);
    defer _ = mlx.mlx_array_free(tkh);
    const tvh = try splitHeads(tv0, s);
    defer _ = mlx.mlx_array_free(tvh);
    const tqn = try rmsNormLast(tqh, aw.naq, 1e-6, s);
    defer _ = mlx.mlx_array_free(tqn);
    const tkn = try rmsNormLast(tkh, aw.nak, 1e-6, s);
    defer _ = mlx.mlx_array_free(tkn);
    // Concat [text, image] on sequence, then heads-first for SDPA.
    const q = try concatHeadsFirst(tqn, iqr, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try concatHeadsFirst(tkn, ikr, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try concatHeadsFirst(tvh, ivh, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(DIT_HEAD_DIM)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_a = mlx.mlx_array{ .ctx = null };
    if (mask) |m| {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, scale, "array", m, null_a, s));
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q, k, v, scale, "", null_a, null_a, s));
    }
    // [B,H,seq,128] → [B,seq,3072]
    const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(at);
    const ash = mlx.getShape(at);
    const flat = try reshape(at, &[_]c_int{ ash[0], ash[1], DIT_HIDDEN }, s);
    defer _ = mlx.mlx_array_free(flat);
    const txt_slice = try sliceSeq(flat, 0, Ltxt, s);
    defer _ = mlx.mlx_array_free(txt_slice);
    const img_slice = try sliceSeq(flat, Ltxt, ash[1], s);
    defer _ = mlx.mlx_array_free(img_slice);
    const txt_out = try linearT(txt_slice, aw.aow, aw.aob, s);
    const img_out = try linearT(img_slice, aw.ow, aw.ob, s);
    return .{ img_out, txt_out };
}
/// concat([txt, img], axis=1) then transpose to [B, H, seq, D].
fn concatHeadsFirst(txt: mlx.mlx_array, img: mlx.mlx_array, s: S) !mlx.mlx_array {
    const cat = try concat(&.{ txt, img }, 1, s); // [B, seq, H, D]
    defer _ = mlx.mlx_array_free(cat);
    return transpose(cat, &[_]c_int{ 0, 2, 1, 3 }, s);
}
fn sliceSeq(x: mlx.mlx_array, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,seq,C]
    const lo = [_]c_int{ 0, start, 0 };
    const hi = [_]c_int{ sh[0], stop, sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &lo, 3, &hi, 3, &st, 3, s));
    return contig(o, s);
}
/// Additive attention mask [B,1,1,Ltxt+Limg] from a [B,Ltxt] keep-mask.
fn buildDitMask(txt_mask: mlx.mlx_array, Limg: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(txt_mask); // [B,Ltxt]
    const B = sh[0];
    const tmf = try astype(txt_mask, .float32, s);
    defer _ = mlx.mlx_array_free(tmf);
    const ones = try onesRow(B, Limg, s);
    defer _ = mlx.mlx_array_free(ones);
    const valid = try concat(&.{ tmf, ones }, 1, s); // [B, Ltxt+Limg]
    defer _ = mlx.mlx_array_free(valid);
    // (valid - 1) * 1e9 → 0 where valid, -1e9 where padded.
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const vm1 = try subA(valid, one, s);
    defer _ = mlx.mlx_array_free(vm1);
    const big = scalarF(1e9);
    defer _ = mlx.mlx_array_free(big);
    const add = try mulA(vm1, big, s);
    defer _ = mlx.mlx_array_free(add);
    const seq = mlx.getShape(add)[1];
    return reshape(add, &[_]c_int{ B, 1, 1, seq }, s);
}
fn onesRow(B: c_int, n: c_int, s: S) !mlx.mlx_array {
    const shape = [_]c_int{ B, n };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&o, &shape, 2, .float32, s));
    return o;
}
/// Centered 3-axis image RoPE angles → (cos, sin) [frames·lh·lw, sum(axes)/2].
fn buildDitRope(a: std.mem.Allocator, frames: c_int, lh: c_int, lw: c_int, s: S) ![2]mlx.mlx_array {
    const f: usize = @intCast(frames);
    const h: usize = @intCast(lh);
    const wdt: usize = @intCast(lw);
    const nf: usize = @intCast(@divExact(DIT_AXES[0], 2)); // 8
    const nh: usize = @intCast(@divExact(DIT_AXES[1], 2)); // 28
    const nw: usize = @intCast(@divExact(DIT_AXES[2], 2)); // 28
    const total = nf + nh + nw; // 64
    const L = f * h * wdt;
    const ang = try a.alloc(f32, L * total);
    defer a.free(ang);
    // frequencies per axis: 1/theta^(2j/dim).
    var ff = try a.alloc(f64, nf);
    defer a.free(ff);
    for (0..nf) |j| ff[j] = 1.0 / std.math.pow(f64, DIT_THETA, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(DIT_AXES[0])));
    var fh = try a.alloc(f64, nh);
    defer a.free(fh);
    for (0..nh) |j| fh[j] = 1.0 / std.math.pow(f64, DIT_THETA, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(DIT_AXES[1])));
    var fw = try a.alloc(f64, nw);
    defer a.free(fw);
    for (0..nw) |j| fw[j] = 1.0 / std.math.pow(f64, DIT_THETA, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(DIT_AXES[2])));
    // positions (centered for h/w; frame starts at image_index 0).
    const h_start: i64 = -@as(i64, @intCast(h - h / 2));
    const w_start: i64 = -@as(i64, @intCast(wdt - wdt / 2));
    for (0..f) |fi| {
        const fpos: f64 = @floatFromInt(fi);
        for (0..h) |ri| {
            const hpos: f64 = @floatFromInt(h_start + @as(i64, @intCast(ri)));
            for (0..wdt) |ci| {
                const wpos: f64 = @floatFromInt(w_start + @as(i64, @intCast(ci)));
                const idx = ((fi * h) + ri) * wdt + ci;
                const base = idx * total;
                for (0..nf) |j| ang[base + j] = @floatCast(fpos * ff[j]);
                for (0..nh) |j| ang[base + nf + j] = @floatCast(hpos * fh[j]);
                for (0..nw) |j| ang[base + nf + nh + j] = @floatCast(wpos * fw[j]);
            }
        }
    }
    const shape = [_]c_int{ @intCast(L), @intCast(total) };
    const raw = mlx.mlx_array_new_data(ang.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const rc = try contig(raw, s);
    defer _ = mlx.mlx_array_free(rc);
    var cosA = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_cos(&cosA, rc, s));
    var sinA = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sin(&sinA, rc, s));
    return .{ cosA, sinA };
}

fn loadDitBlock(w: *const Weights, a: std.mem.Allocator, i: usize, s: S) !DitBlockW {
    const p = try std.fmt.allocPrint(a, "transformer_blocks.{d}", .{i});
    defer a.free(p);
    const im = try std.fmt.allocPrint(a, "{s}.img_mod.1", .{p});
    defer a.free(im);
    const tmo = try std.fmt.allocPrint(a, "{s}.txt_mod.1", .{p});
    defer a.free(tmo);
    const imlp0 = try std.fmt.allocPrint(a, "{s}.img_mlp.net.0.proj", .{p});
    defer a.free(imlp0);
    const imlp2 = try std.fmt.allocPrint(a, "{s}.img_mlp.net.2", .{p});
    defer a.free(imlp2);
    const tmlp0 = try std.fmt.allocPrint(a, "{s}.txt_mlp.net.0.proj", .{p});
    defer a.free(tmlp0);
    const tmlp2 = try std.fmt.allocPrint(a, "{s}.txt_mlp.net.2", .{p});
    defer a.free(tmlp2);
    return .{
        .img_mod_w = try loadLinT(w, a, im, s),
        .img_mod_b = try loadVec(w, a, im, "bias", s),
        .txt_mod_w = try loadLinT(w, a, tmo, s),
        .txt_mod_b = try loadVec(w, a, tmo, "bias", s),
        .attn = try loadDitAttn(w, a, p, s),
        .img0w = try loadLinT(w, a, imlp0, s),
        .img0b = try loadVec(w, a, imlp0, "bias", s),
        .img2w = try loadLinT(w, a, imlp2, s),
        .img2b = try loadVec(w, a, imlp2, "bias", s),
        .txt0w = try loadLinT(w, a, tmlp0, s),
        .txt0b = try loadVec(w, a, tmlp0, "bias", s),
        .txt2w = try loadLinT(w, a, tmlp2, s),
        .txt2b = try loadVec(w, a, tmlp2, "bias", s),
    };
}
fn loadDitAttn(w: *const Weights, a: std.mem.Allocator, p: []const u8, s: S) !DitAttnW {
    const j = struct {
        fn k(al: std.mem.Allocator, pre: []const u8, suf: []const u8) ![]u8 {
            return std.fmt.allocPrint(al, "{s}.{s}", .{ pre, suf });
        }
    }.k;
    const qk = try j(a, p, "attn.to_q");
    defer a.free(qk);
    const kk = try j(a, p, "attn.to_k");
    defer a.free(kk);
    const vk = try j(a, p, "attn.to_v");
    defer a.free(vk);
    const aqk = try j(a, p, "attn.add_q_proj");
    defer a.free(aqk);
    const akk = try j(a, p, "attn.add_k_proj");
    defer a.free(akk);
    const avk = try j(a, p, "attn.add_v_proj");
    defer a.free(avk);
    const nqk = try j(a, p, "attn.norm_q");
    defer a.free(nqk);
    const nkk = try j(a, p, "attn.norm_k");
    defer a.free(nkk);
    const naqk = try j(a, p, "attn.norm_added_q");
    defer a.free(naqk);
    const nakk = try j(a, p, "attn.norm_added_k");
    defer a.free(nakk);
    const ok = try j(a, p, "attn.to_out.0");
    defer a.free(ok);
    const aok = try j(a, p, "attn.to_add_out");
    defer a.free(aok);
    return .{
        .qw = try loadLinT(w, a, qk, s),
        .qb = try loadVec(w, a, qk, "bias", s),
        .kw = try loadLinT(w, a, kk, s),
        .kb = try loadVec(w, a, kk, "bias", s),
        .vw = try loadLinT(w, a, vk, s),
        .vb = try loadVec(w, a, vk, "bias", s),
        .aqw = try loadLinT(w, a, aqk, s),
        .aqb = try loadVec(w, a, aqk, "bias", s),
        .akw = try loadLinT(w, a, akk, s),
        .akb = try loadVec(w, a, akk, "bias", s),
        .avw = try loadLinT(w, a, avk, s),
        .avb = try loadVec(w, a, avk, "bias", s),
        .nq = try loadVec(w, a, nqk, "weight", s),
        .nk = try loadVec(w, a, nkk, "weight", s),
        .naq = try loadVec(w, a, naqk, "weight", s),
        .nak = try loadVec(w, a, nakk, "weight", s),
        .ow = try loadLinT(w, a, ok, s),
        .ob = try loadVec(w, a, ok, "bias", s),
        .aow = try loadLinT(w, a, aok, s),
        .aob = try loadVec(w, a, aok, "bias", s),
    };
}
fn freeDitAttn(x: *DitAttnW) void {
    inline for (.{ x.qw, x.qb, x.kw, x.kb, x.vw, x.vb, x.aqw, x.aqb, x.akw, x.akb, x.avw, x.avb, x.nq, x.nk, x.naq, x.nak, x.ow, x.ob, x.aow, x.aob }) |f| _ = mlx.mlx_array_free(f);
}
fn freeDitBlock(b: *DitBlockW) void {
    inline for (.{ b.img_mod_w, b.img_mod_b, b.txt_mod_w, b.txt_mod_b, b.img0w, b.img0b, b.img2w, b.img2b, b.txt0w, b.txt0b, b.txt2w, b.txt2b }) |f| _ = mlx.mlx_array_free(f);
    freeDitAttn(&b.attn);
}

// ── Tests ──

const testing = std.testing;

/// Cosine similarity between two arrays (flattened), read back as a scalar.
fn cosineSim(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !f32 {
    var na: c_int = 1;
    for (mlx.getShape(a)) |d| na *= d;
    const a32 = try astype(a, .float32, s);
    defer _ = mlx.mlx_array_free(a32);
    const af = try reshape(a32, &[_]c_int{na}, s);
    defer _ = mlx.mlx_array_free(af);
    const b32 = try astype(b, .float32, s);
    defer _ = mlx.mlx_array_free(b32);
    const bf = try reshape(b32, &[_]c_int{na}, s);
    defer _ = mlx.mlx_array_free(bf);
    const ab = try mulA(af, bf, s);
    defer _ = mlx.mlx_array_free(ab);
    const aa = try mulA(af, af, s);
    defer _ = mlx.mlx_array_free(aa);
    const bb = try mulA(bf, bf, s);
    defer _ = mlx.mlx_array_free(bb);
    const sumFn = struct {
        fn f(x: mlx.mlx_array, st: S) !f32 {
            var o = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_sum_axis(&o, x, 0, false, st));
            var v: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&v, o));
            return v;
        }
    }.f;
    const dot = try sumFn(ab, s);
    const nrm = @sqrt(try sumFn(aa, s)) * @sqrt(try sumFn(bb, s));
    return if (nrm == 0) 0 else dot / nrm;
}

// Live parity vs the mflux reference. Gated on MAGEFLOW_TEST_MODEL (checkpoint
// dir) + MAGEFLOW_VAE_FIXTURE (safetensors with z/cond/decoded, f32). Skips when
// unset so CI stays hermetic.
test "MageFlow VAE decode parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_VAE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var dec = try VaeDecoder.load(io, a, s, model_dir);
    defer dec.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const z = fx.get("z") orelse return error.MissingFixtureZ;
    const ref_cond = fx.get("cond") orelse return error.MissingFixtureCond;
    const ref_decoded = fx.get("decoded") orelse return error.MissingFixtureDecoded;

    // Bisection: _Decoder cond first, then the full decode.
    const my_cond = try dec.decoderCond(z);
    defer _ = mlx.mlx_array_free(my_cond);
    const cos_cond = try cosineSim(my_cond, ref_cond, s);
    std.debug.print("[mageflow-vae] cond cosine = {d:.6}\n", .{cos_cond});

    const my_dec = try dec.decode(z);
    defer _ = mlx.mlx_array_free(my_dec);
    const cos_dec = try cosineSim(my_dec, ref_decoded, s);
    std.debug.print("[mageflow-vae] decode cosine = {d:.6}\n", .{cos_dec});

    try testing.expect(cos_cond > 0.999);
    try testing.expect(cos_dec > 0.999);
}

// DiT parity vs the mflux reference. Gated on MAGEFLOW_TEST_MODEL +
// MAGEFLOW_DIT_FIXTURE (safetensors with img/txt/out, f32, lh=lw=8, Ltxt=16,
// t=0.7, no mask). Skips when unset.
test "MageFlow DiT forward parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_DIT_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var dit = try Dit.load(io, a, s, model_dir);
    defer dit.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const img = fx.get("img") orelse return error.MissingFixtureImg;
    const txt = fx.get("txt") orelse return error.MissingFixtureTxt;
    const ref_out = fx.get("out") orelse return error.MissingFixtureOut;

    const my_out = try dit.forward(img, txt, 0.7, 1, 8, 8, null);
    defer _ = mlx.mlx_array_free(my_out);
    const cos = try cosineSim(my_out, ref_out, s);
    std.debug.print("[mageflow-dit] velocity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

// DiT parity WITH an attention mask + a larger sequence (lh=lw=16, Ltxt=24 with
// 6 padded, t=0.3). Gated on MAGEFLOW_TEST_MODEL + MAGEFLOW_DIT_MASKED_FIXTURE.
test "MageFlow DiT masked parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_DIT_MASKED_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var dit = try Dit.load(io, a, s, model_dir);
    defer dit.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const img = fx.get("img") orelse return error.MissingFixtureImg;
    const txt = fx.get("txt") orelse return error.MissingFixtureTxt;
    const mask = fx.get("mask") orelse return error.MissingFixtureMask;
    const ref_out = fx.get("out") orelse return error.MissingFixtureOut;

    const my_out = try dit.forward(img, txt, 0.3, 1, 16, 16, mask);
    defer _ = mlx.mlx_array_free(my_out);
    const cos = try cosineSim(my_out, ref_out, s);
    std.debug.print("[mageflow-dit-masked] velocity cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

// Real config bytes from microsoft/Mage-Flow-Turbo (trimmed to the parsed keys).
const TEST_TRANSFORMER_CFG =
    \\{"in_channels":128,"out_channels":128,"context_in_dim":2560,"hidden_size":3072,
    \\"mlp_ratio":4.0,"num_heads":24,"depth":12,"axes_dim":[16,56,56],"theta":10000,
    \\"max_sequence_length":2048,"static_shift":6.0,"schedule_mode":"z-image"}
;
const TEST_VAE_CFG =
    \\{"_class_name":"MageVAE","latent_channels":128,"downsample_factor":16}
;
const TEST_TE_CFG =
    \\{"architectures":["Qwen3VLForConditionalGeneration"],"image_token_id":151655,
    \\"model_type":"qwen3_vl","text_config":{"hidden_size":2560,"num_hidden_layers":36,
    \\"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"intermediate_size":9728,
    \\"vocab_size":151936,"rope_theta":5000000,"rope_scaling":{"mrope_section":[24,20,20]}},
    \\"vision_config":{"depth":24,"hidden_size":1024,"num_heads":16,"patch_size":16,
    \\"spatial_merge_size":2,"out_hidden_size":2560}}
;

fn writeTestCheckpoint(io: std.Io, tmp: *std.testing.TmpDir) !void {
    try tmp.dir.createDirPath(io, "transformer");
    try tmp.dir.createDirPath(io, "vae");
    try tmp.dir.createDirPath(io, "text_encoder");
    try tmp.dir.writeFile(io, .{ .sub_path = "transformer/config.json", .data = TEST_TRANSFORMER_CFG });
    try tmp.dir.writeFile(io, .{ .sub_path = "vae/config.json", .data = TEST_VAE_CFG });
    try tmp.dir.writeFile(io, .{ .sub_path = "text_encoder/config.json", .data = TEST_TE_CFG });
}

test "parseConfig reads the MageFlow component configs" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try writeTestCheckpoint(io, &tmp);

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const model_dir = root_buf[0..root_len];

    const cfg = try parseConfig(io, a, model_dir);
    // Transformer.
    try testing.expectEqual(@as(u32, 12), cfg.dit_depth);
    try testing.expectEqual(@as(u32, 3072), cfg.dit_hidden);
    try testing.expectEqual(@as(u32, 24), cfg.dit_heads);
    try testing.expectEqual(@as(u32, 2560), cfg.dit_context_dim);
    try testing.expectEqual(@as(u32, 128), cfg.ditHeadDim());
    try testing.expectEqual([3]u32{ 16, 56, 56 }, cfg.dit_axes_dim);
    try testing.expectEqual(@as(f32, 6.0), cfg.dit_static_shift);
    // VAE.
    try testing.expectEqual(@as(u32, 128), cfg.vae_latent_channels);
    try testing.expectEqual(@as(u32, 16), cfg.vae_downsample);
    // Text encoder (Qwen3-VL).
    try testing.expectEqual(@as(u32, 2560), cfg.te_hidden);
    try testing.expectEqual(@as(u32, 36), cfg.te_layers);
    try testing.expectEqual(@as(u32, 8), cfg.te_kv_heads);
    try testing.expectEqual([3]u32{ 24, 20, 20 }, cfg.te_mrope_section);
    try testing.expectEqual(@as(u32, 24), cfg.vit_depth);
    try testing.expectEqual(@as(u32, 151655), cfg.image_token_id);
}

test "parseConfig requires the component configs (manifest presence)" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    // Only the VAE config present — transformer/text_encoder missing.
    try tmp.dir.createDirPath(io, "vae");
    try tmp.dir.writeFile(io, .{ .sub_path = "vae/config.json", .data = TEST_VAE_CFG });
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    try testing.expectError(error.FileNotFound, parseConfig(io, a, root_buf[0..root_len]));
}

test "mapVaeKey rewrites encoder/decoder prefixes and drops legacy tensors" {
    const a = testing.allocator;
    // student.dconv_encoder.* → encoder.*
    const enc = (try mapVaeKey(a, "student.dconv_encoder.blocks.0.conv1.weight")).?;
    defer a.free(enc);
    try testing.expectEqualStrings("encoder.blocks.0.conv1.weight", enc);
    // pipeline.* → decoder_model.*
    const dec = (try mapVaeKey(a, "pipeline.t_embedder.mlp.0.weight")).?;
    defer a.free(dec);
    try testing.expectEqualStrings("decoder_model.t_embedder.mlp.0.weight", dec);
    // legacy VAE encoder tensors are dropped (checked BEFORE the pipeline. arm)
    try testing.expect((try mapVaeKey(a, "pipeline.y_embedder.encoder.foo")) == null);
    // an unrecognized name is a converter-mismatch guard
    try testing.expectError(error.UnexpectedMageFlowVaeKey, mapVaeKey(a, "mystery.weight"));
}

test "mapTextEncoderKey strips model. prefix and drops tied/rotary tensors" {
    const a = testing.allocator;
    const lm = (try mapTextEncoderKey(a, "model.language_model.layers.0.self_attn.q_proj.weight")).?;
    defer a.free(lm);
    try testing.expectEqualStrings("language_model.layers.0.self_attn.q_proj.weight", lm);
    const vis = (try mapTextEncoderKey(a, "model.visual.patch_embed.proj.weight")).?;
    defer a.free(vis);
    try testing.expectEqualStrings("visual.patch_embed.proj.weight", vis);
    try testing.expect((try mapTextEncoderKey(a, "lm_head.weight")) == null);
    try testing.expect((try mapTextEncoderKey(a, "model.visual.rotary_pos_emb.inv_freq")) == null);
    try testing.expectError(error.UnexpectedMageFlowTextEncoderKey, mapTextEncoderKey(a, "extra.weight"));
}

test "validateWeightCount matches mflux expected counts (vae folded or unfolded)" {
    try validateWeightCount(.transformer, EXPECTED_TRANSFORMER_WEIGHTS);
    try validateWeightCount(.text_encoder, EXPECTED_TEXT_ENCODER_WEIGHTS);
    try validateWeightCount(.vae, EXPECTED_VAE_WEIGHTS);
    try validateWeightCount(.vae, EXPECTED_VAE_WEIGHTS_FOLDED); // folded AdaLN
    try testing.expectError(error.MageFlowWeightCountMismatch, validateWeightCount(.transformer, 396));
    try testing.expectError(error.MageFlowWeightCountMismatch, validateWeightCount(.vae, 700));
}

test "Engine.load parses config and generateImage honestly reports not-implemented" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try writeTestCheckpoint(io, &tmp);
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);

    const eng = try Engine.load(io, a, root_buf[0..root_len]);
    defer eng.deinit();
    try testing.expectEqual(@as(u32, 12), eng.cfg.dit_depth);
    try testing.expectError(
        Error.MageFlowNotImplemented,
        eng.generateImage(a, "a cat", 1024, 1024, 42, 4),
    );
}
