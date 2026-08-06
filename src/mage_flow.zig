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
//! touched. Turbo text→image works end-to-end (`Engine.generateImage`): VAE
//! (`VaeDecoder`) + DiT (`Dit`) + Qwen3-VL encoder (`TextEncoder`) + a static-shift
//! FlowMatchEuler scheduler, each validated behind env-gated parity fixtures.
//! Multi-reference in-context EDITING is implemented too (`Engine.editImage`,
//! Edit checkpoint only): a VAE encoder (`VaeEncoder`) for clean reference
//! latents + a Qwen3-VL vision tower with DeepStack (`VisionTower`) that
//! image-conditions the prompt (`TextEncoder.encodeEdit`), then the same DiT
//! denoises the target across the target+reference token stream. The
//! Gaussian-Shading watermark and the content-policy AR screen are later phases.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const stb = @import("stb");
const qvis = @import("qwen_vision.zig");
const sse = @import("gen_sse.zig");

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
    // Vision preprocessing (`text_encoder/preprocessor_config.json`) — the
    // smart-resize pixel budget for a reference image. Defaults are the Turbo
    // release's values; the file is read when present so a checkpoint that ships
    // different limits doesn't silently condition on a different grid than the
    // reference pipeline would.
    vlm_min_pixels: u32 = 65_536,
    vlm_max_pixels: u32 = 16_777_216,

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

    // Optional: the image processor's pixel budget (edit path only). Absent on a
    // text-only checkpoint, so a missing/unparseable file keeps the defaults.
    {
        const path = try std.fmt.allocPrint(a, "{s}/text_encoder/preprocessor_config.json", .{model_dir});
        defer a.free(path);
        if (readJson(io, a, path)) |parsed| {
            var p = parsed;
            defer p.deinit();
            cfg.vlm_min_pixels = getU32(p.value, "min_pixels", cfg.vlm_min_pixels);
            cfg.vlm_max_pixels = getU32(p.value, "max_pixels", cfg.vlm_max_pixels);
        } else |_| {}
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
const tok_mod = @import("tokenizer.zig");
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
/// Round an f32 to bfloat16 precision (round-to-nearest-even), kept in f32
/// storage. Used to replicate the model's bf16-rounded timestep frequency table.
fn roundBf16(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    const bits: u32 = @bitCast(x);
    const rounding_bias: u32 = 0x7FFF + ((bits >> 16) & 1);
    const rounded: u32 = (bits +% rounding_bias) & 0xFFFF0000;
    return @bitCast(rounded);
}
/// A scalar constant in `ref`'s dtype. mlx promotes bf16⊕f32-scalar to f32, so a
/// bare `scalarF` silently upcasts a bf16 activation chain to f32; matching the
/// operand's dtype keeps the chain in bf16 (and is a no-op on the f32 path).
fn scalarLike(v: f32, ref: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sc = scalarF(v);
    if (mlx.mlx_array_dtype(ref) == .float32) return sc;
    defer _ = mlx.mlx_array_free(sc);
    return astype(sc, mlx.mlx_array_dtype(ref), s);
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
/// `x[lo:hi:st]`, materialized. Every slicing helper in this file routes here so
/// the intermediate handle exists in exactly ONE place: a slice left alive holds
/// its PARENT's buffer, so wrapping it in `contiguous` and forgetting to free it
/// retains the whole source — correct output, ~47 MB lost per DiT block.
fn sliceContig(x: mlx.mlx_array, lo: []const c_int, hi: []const c_int, st: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, x, lo.ptr, lo.len, hi.ptr, hi.len, st.ptr, st.len, s));
    return contig(o, s);
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
/// LayerNorm over the LAST axis, computed in f32, RESULT cast back to the input's
/// dtype (f32 in → f32 out for the VAE parity path; bf16 in → bf16 out for the DiT
/// so the modulation chain stays bf16). weight/bias optional (affine).
fn layerNormLast(x: mlx.mlx_array, weight: ?mlx.mlx_array, bias: ?mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
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
    var out = try mulA(centered, rstd, s); // f32
    if (weight) |wgt| {
        const scaled = try mulA(out, wgt, s);
        _ = mlx.mlx_array_free(out);
        out = scaled;
        if (bias) |b| {
            const shifted = try addA(out, b, s);
            _ = mlx.mlx_array_free(out);
            out = shifted;
        }
    }
    if (in_dt == .float32) return out;
    defer _ = mlx.mlx_array_free(out);
    return astype(out, in_dt, s);
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
/// A `.bias`/`.weight` vector, f32 (norms stay f32; the reference upcasts them).
fn loadVec(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, s: S) !mlx.mlx_array {
    const k = try std.fmt.allocPrint(a, "{s}." ++ suffix, .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}
/// A `.bias`/`.weight` vector in a chosen compute dtype (biases must match the
/// bf16 activation chain — an f32 bias would upcast the sum).
fn loadVecDt(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, dtype: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const k = try std.fmt.allocPrint(a, "{s}." ++ suffix, .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, dtype, s);
}
/// An optional weight — `null` when the key is absent (no error, no log).
fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}

/// A DiT / text-encoder / ViT linear, dense bf16 OR affine-quantized, decided
/// per tensor by the presence of a `.scales` sibling. The upstream Microsoft
/// repos are dense; our 8-bit mirrors quantize most of the same tensors and
/// hold a few back, so a checkpoint carries no format flag and a MIXED one
/// works by construction. `(bits, group_size)` come from the packed geometry
/// for the same reason — nothing global to keep in sync with the converter.
///
/// Same primitive as `krea.zig`'s `MixedLinear`, minus the LoRA arm (MageFlow
/// has no adapter path yet) and with the compute dtype chosen by the caller:
/// production runs bf16, the component parity fixtures run f32.
///
/// The VAE deliberately does NOT use this — it stays dense on its proven f32
/// path (`loadLinT`/`linearT`), and the converter never quantizes it.
/// Dense-or-quantized linear whose packed geometry is solved per tensor from
/// the shapes alone. Shared with `minimax_h3.zig`: both backends must load the
/// bf16 releases AND our affine-quantized mirrors through ONE path, and a
/// second copy of this is a second place for the bits/group-size solve to drift.
/// Wide-M route for quantized MfLinear: dequantize to the compute dtype and run
/// the steel GEMM instead of `quantized_matmul`. Media DiTs run EVERY forward at
/// prefill-like widths (H3 at 480p is ~9K rows), where stock qmm_t sits in the
/// same tile dead zone `prefillDqGemm` exists for on the text side. Opt-in per
/// backend A/B (the prefill-perf-kernel rule): `MLX_SERVE_MF_DQ_GEMM=1` enables
/// at the 2048-row floor, `=N` sets an explicit floor, unset/`0` = off.
const OptFloor = ?usize;
/// Test seam: `some(null)` forces off, `some(f)` forces floor `f`.
pub var mf_dq_gemm_override: ?OptFloor = null;
var mf_dq_gemm_env_done: bool = false;
var mf_dq_gemm_env_val: OptFloor = null;
/// Engagement is COUNTED, never inferred from output equality (a rejected
/// guard is a silent no-op that is output-identical to the fallback).
pub var mf_dq_gemm_engaged: u64 = 0;
var mf_dq_gemm_logged: bool = false;

const MF_DQ_GEMM_DEFAULT_MIN_M: usize = 2048;

fn mfDqGemmFloorFrom(raw: ?[]const u8) ?usize {
    const v = raw orelse return null;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return null;
    if (std.mem.eql(u8, v, "1")) return MF_DQ_GEMM_DEFAULT_MIN_M;
    return std.fmt.parseInt(usize, v, 10) catch null;
}

fn mfDqGemmFloor() ?usize {
    if (mf_dq_gemm_override) |v| return v;
    if (!mf_dq_gemm_env_done) {
        const raw = std.c.getenv("MLX_SERVE_MF_DQ_GEMM");
        mf_dq_gemm_env_val = mfDqGemmFloorFrom(if (raw) |r| std.mem.sliceTo(r, 0) else null);
        mf_dq_gemm_env_done = true;
    }
    return mf_dq_gemm_env_val;
}

pub const MfLinear = struct {
    quantized: bool,
    /// quantized: packed u32 [out, in*bits/32] (`transpose_w=true` at use).
    /// dense: pre-transposed [in, out], matching the old `loadLinDt` layout.
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    dtype: mlx.mlx_dtype,
    bits: u32 = 0,
    group_size: u32 = 0,

    /// `in_features` is the module's input dim, known at every call site from
    /// `Config`; it is what makes the packed geometry solvable.
    pub fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, dtype: mlx.mlx_dtype, s: S) !MfLinear {
        const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
        defer a.free(sk);

        if (ownOpt(w, sk)) |raw_scales| {
            defer _ = mlx.mlx_array_free(raw_scales);
            const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
            defer a.free(bk);
            const weight = try ownWeight(w, wk);
            errdefer _ = mlx.mlx_array_free(weight);
            const raw_biases = try ownWeight(w, bk);
            defer _ = mlx.mlx_array_free(raw_biases);

            const w_cols: u32 = @intCast(mlx.getShape(weight)[1]); // in*bits/32
            const s_cols: u32 = @intCast(mlx.getShape(raw_scales)[1]); // in/group_size
            const bits: u32 = @intCast(@divExact(32 * w_cols, in_features));
            const gs: u32 = @intCast(@divExact(in_features, s_cols));

            // Scales/biases carry the quantized matmul's arithmetic dtype, so
            // they follow the compute dtype rather than whatever the file used.
            const scales = try astype(raw_scales, dtype, s);
            errdefer _ = mlx.mlx_array_free(scales);
            const biases = try astype(raw_biases, dtype, s);
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = biases,
                .dtype = dtype,
                .bits = bits,
                .group_size = gs,
            };
        }

        // Dense: pre-transpose [out,in] → [in,out], materialize, cast. Byte for
        // byte what `loadLinDt` did before this struct existed.
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        return .{ .quantized = false, .w = try astype(tc, dtype, s), .dtype = dtype };
    }

    pub fn deinit(self: *MfLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
    }

    /// x[.., in] @ W (+ bias). `bias` stays a caller-owned argument so this is a
    /// drop-in for `linearT`, which is how the dense path stays unchanged.
    pub fn forward(self: *const MfLinear, x: mlx.mlx_array, bias: ?mlx.mlx_array, s: S) !mlx.mlx_array {
        // No-op when x already matches (mlx returns the input unchanged), so the
        // dense path keeps the exact arithmetic the bf16 fixtures were pinned on.
        const xc = try astype(x, self.dtype, s);
        defer _ = mlx.mlx_array_free(xc);
        if (!self.quantized) return linearT(xc, self.w, bias, s);
        if (try self.dqGemmWide(xc, bias, s)) |y| return y;
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &o,
            xc,
            self.w,
            self.scales,
            self.biases,
            true,
            mlx.mlx_optional_int.some(@intCast(self.group_size)),
            mlx.mlx_optional_int.some(@intCast(self.bits)),
            "affine",
            s,
        ));
        if (bias) |b| {
            defer _ = mlx.mlx_array_free(o);
            return addA(o, b, s);
        }
        return o;
    }

    /// The wide-M dequant+GEMM route; null when the call must stay on qmm.
    fn dqGemmWide(self: *const MfLinear, xc: mlx.mlx_array, bias: ?mlx.mlx_array, s: S) !?mlx.mlx_array {
        const min_m = mfDqGemmFloor() orelse return null;
        switch (self.bits) {
            2, 3, 4, 5, 6, 8 => {},
            else => return null,
        }
        const shp = mlx.getShape(xc);
        if (shp.len == 0) return null;
        const last: usize = @intCast(shp[shp.len - 1]);
        if (last == 0) return null;
        const rows = mlx.mlx_array_size(xc) / last;
        if (rows < min_m) return null;

        var dq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dq);
        try mlx.check(mlx.mlx_dequantize(&dq, self.w, self.scales, self.biases, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = self.dtype, .has_value = true }, s));
        var dq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dq_t);
        try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_matmul(&o, xc, dq_t, s));
        mf_dq_gemm_engaged += 1;
        if (!mf_dq_gemm_logged) {
            mf_dq_gemm_logged = true;
            log.info("[mf-linear] dq-gemm engaged (rows={d}, floor={d})\n", .{ rows, min_m });
        }
        if (bias) |b| {
            defer _ = mlx.mlx_array_free(o);
            return try addA(o, b, s);
        }
        return o;
    }
};

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

/// The DiT + text encoder run in bf16 — REQUIRED for correctness, not just
/// memory: the Turbo checkpoint was distilled in bf16 and its 4-step denoise
/// washes out in f32 (validated: f32 DiT → grey blob, bf16 DiT → crisp). The VAE
/// stays on its proven f32 parity path (bf16-DiT + f32-VAE = crisp; it upcasts
/// its latent input internally). f32 remains available for the component parity
/// tests. Norms/rope compute in f32 internally either way.
const MF_COMPUTE: mlx.mlx_dtype = .bfloat16;

/// VAE downsample × nothing (native-resolution) — dimensions must be /16.
const MF_DOWNSAMPLE: u32 = 16;
const MF_DEFAULT_STEPS: u32 = 4; // Turbo default
/// Reference-image cap for one edit (mirrors `gen.MAX_EDIT_IMAGES` — the HTTP
/// layer rejects earlier with a 400; this is the engine's own backstop, since
/// every extra reference adds a FULL image's tokens to the DiT stream).
const MF_MAX_EDIT_REFS: usize = 4;

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []u8,
    cfg: Config,
    dtype: mlx.mlx_dtype,
    te: TextEncoder,
    dit: Dit,
    vae: VaeDecoder,
    tok: tok_mod.Tokenizer,
    // Edit-only components (the multi-reference in-context editor). Loaded when
    // the checkpoint is the Edit variant (`is_edit`); null on a txt2img model.
    is_edit: bool,
    vae_enc: ?VaeEncoder,
    vit: ?VisionTower,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        // Parse + validate the manifest (component configs must be present).
        self.cfg = try parseConfig(io, allocator, model_dir);
        self.model_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(self.model_dir);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.dtype = MF_COMPUTE;
        // Both repos carry the same `_class_name` and the `visual.*` weights, so
        // the edit capability is gated on the repo/dir name (the plan's fallback).
        self.is_edit = dirIsEdit(model_dir);

        self.te = try TextEncoder.load(io, allocator, self.s, model_dir, self.dtype);
        errdefer self.te.deinit();
        self.dit = try Dit.load(io, allocator, self.s, model_dir, self.dtype);
        errdefer self.dit.deinit();
        self.vae = try VaeDecoder.load(io, allocator, self.s, model_dir); // VAE: f32
        errdefer self.vae.deinit();

        // Edit checkpoints add the VAE encoder (clean reference latents) and the
        // Qwen3-VL vision tower (image-conditioned prompt). f32 VAE encoder (its
        // proven precision); bf16 vision tower (matches the DiT/LM compute path).
        self.vae_enc = null;
        self.vit = null;
        // Function-scope errdefers (a block-scoped one stops covering these the
        // moment the `if` exits — a later tokenizer failure would leak them).
        errdefer if (self.vae_enc) |*e| e.deinit();
        errdefer if (self.vit) |*v| v.deinit();
        if (self.is_edit) {
            self.vae_enc = try VaeEncoder.load(io, allocator, self.s, model_dir);
            self.vit = try VisionTower.load(io, allocator, self.s, model_dir, self.dtype);
        }

        const te_dir = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{model_dir});
        defer allocator.free(te_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, te_dir);

        log.info(
            "[image] MageFlow ready (DiT {d}×{d} heads {d}; VAE {d}ch /{d}; TE Qwen3-VL {d}L{s})\n",
            .{ self.cfg.dit_depth, self.cfg.dit_hidden, self.cfg.dit_heads, self.cfg.vae_latent_channels, self.cfg.vae_downsample, self.cfg.te_layers, if (self.is_edit)
                "; EDIT (multi-reference in-context editor)"
            else
                "; text-to-image only — editing needs the Mage-Flow-Edit checkpoint" },
        );
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.tok.deinit();
        if (self.vit) |*v| v.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.vae.deinit();
        self.dit.deinit();
        self.te.deinit();
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    /// True when this checkpoint can run the multi-reference in-context editor.
    pub fn supportsEdit(self: *const Engine) bool {
        return self.is_edit and self.vae_enc != null and self.vit != null;
    }

    /// Build the templated + tokenized prompt ids/mask (one unpadded prompt, so
    /// mask is all-ones). Truncated to `TE_MAX_COND + TE_DROP_TOKENS` up front so
    /// the post-drop context never exceeds the conditioning budget.
    fn buildPromptIds(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) !struct { ids: []i32, mask: []i32 } {
        const formatted = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ TE_PREFIX, prompt, TE_SUFFIX });
        defer allocator.free(formatted);
        const enc = try self.tok.encode(allocator, formatted);
        defer allocator.free(enc);
        const n = @min(enc.len, TE_MAX_COND + TE_DROP_TOKENS);
        const ids = try allocator.alloc(i32, n);
        const mask = try allocator.alloc(i32, n);
        for (0..n) |i| {
            ids[i] = @intCast(enc[i]);
            mask[i] = 1;
        }
        return .{ .ids = ids, .mask = mask };
    }

    /// Text→image. Returns the image [1,3,H,W] f32 in [0,1] (owned mlx array;
    /// caller frees). Turbo defaults: 4 steps, guidance 1.0 (no CFG).
    pub fn generateImage(
        self: *Engine,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        width: u32,
        height: u32,
        seed: u64,
        steps: u32,
        progress: ?sse.Progress,
    ) !mlx.mlx_array {
        const s = self.s;
        const n_steps: u32 = if (steps == 0) MF_DEFAULT_STEPS else steps;
        const W = normalizeDim(width);
        const H = normalizeDim(height);
        const lat_h: c_int = @intCast(H / MF_DOWNSAMPLE);
        const lat_w: c_int = @intCast(W / MF_DOWNSAMPLE);

        // 1. Conditioning: templated encode → [1, n, 2560] (no CFG at guidance 1).
        const pr = try self.buildPromptIds(allocator, prompt);
        defer allocator.free(pr.ids);
        defer allocator.free(pr.mask);
        const cond = try self.te.encodeTextToImage(pr.ids, pr.mask);
        const txt = cond.embeddings;
        defer _ = mlx.mlx_array_free(txt);

        // 2. Initial noise [1,128,lat_h,lat_w] → packed [1, HW, 128] (no watermark).
        var img = try initNoisePacked(seed, lat_h, lat_w, self.dtype, s);
        defer _ = mlx.mlx_array_free(img);

        // 3. Scheduler sigmas [N+1] (static shift, appended 0).
        const sigmas = try computeSigmas(allocator, n_steps, self.cfg.sched_shift);
        defer allocator.free(sigmas);

        // 4. Euler denoise. text mask is all-valid ⇒ null (no additive mask).
        // Update in f32, store back in the latent (bf16) dtype — `_mage_flow_euler_step`.
        for (0..n_steps) |i| {
            const v = try self.dit.forward(img, txt, sigmas[i], 1, lat_h, lat_w, null);
            defer _ = mlx.mlx_array_free(v);
            const img_f = try astype(img, .float32, s);
            defer _ = mlx.mlx_array_free(img_f);
            const v_f = try astype(v, .float32, s);
            defer _ = mlx.mlx_array_free(v_f);
            const dt = scalarF(sigmas[i + 1] - sigmas[i]);
            defer _ = mlx.mlx_array_free(dt);
            const stepv = try mulA(v_f, dt, s);
            defer _ = mlx.mlx_array_free(stepv);
            const ni_f = try addA(img_f, stepv, s);
            defer _ = mlx.mlx_array_free(ni_f);
            const ni = try astype(ni_f, self.dtype, s);
            _ = mlx.mlx_array_free(img);
            img = ni;
            _ = mlx.mlx_array_eval(img);
            if (progress) |p| p.emit("Generating", @intCast(i + 1), n_steps);
        }

        // 5. Unpack → [1,128,lat_h,lat_w] → VAE decode → [-1,1] → clip to [0,1].
        if (progress) |p| p.emit("Decoding image", n_steps, n_steps);
        const latent = try unpackLatents(img, lat_h, lat_w, s);
        defer _ = mlx.mlx_array_free(latent);
        const decoded = try self.vae.decode(latent);
        defer _ = mlx.mlx_array_free(decoded);
        return denormImage(decoded, s);
    }

    /// Multi-reference in-context EDIT (E7.5). `edit_bytes` are the raw PNG/JPEG
    /// bytes of the reference image(s) (primary first). Each ref is VAE-encoded
    /// CLEAN at the target size and concatenated into the DiT image stream, while
    /// the same refs (resized for the VLM) condition the prompt via the vision
    /// tower. Only the target denoises; the reference tokens are constant.
    /// Returns the image [1,3,H,W] f32 in [0,1] (owned; caller frees).
    pub fn editImage(
        self: *Engine,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        edit_bytes: []const []const u8,
        width: u32,
        height: u32,
        seed: u64,
        steps: u32,
        progress: ?sse.Progress,
    ) !mlx.mlx_array {
        if (edit_bytes.len == 0) return error.NoReferenceImages;
        if (edit_bytes.len > MF_MAX_EDIT_REFS) return error.TooManyReferenceImages;
        const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
        const vt = if (self.vit) |*v| v else return error.NoVisionTower;
        const s = self.s;
        const n_steps: u32 = if (steps == 0) MF_DEFAULT_STEPS else steps;
        const W = normalizeDim(width);
        const H = normalizeDim(height);
        const lat_h: c_int = @intCast(H / MF_DOWNSAMPLE);
        const lat_w: c_int = @intCast(W / MF_DOWNSAMPLE);
        const nrefs = edit_bytes.len;
        const n_images: c_int = @intCast(1 + nrefs);

        // 1. Per-ref: clean VAE latent (target size) + VLM pixel_values + grid.
        var ref_packs: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (ref_packs.items) |p| _ = mlx.mlx_array_free(p);
            ref_packs.deinit(allocator);
        }
        var pv_chunks: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (pv_chunks.items) |p| _ = mlx.mlx_array_free(p);
            pv_chunks.deinit(allocator);
        }
        var grids: std.ArrayList([3]i64) = .empty;
        defer grids.deinit(allocator);
        var ntoks: std.ArrayList(u32) = .empty;
        defer ntoks.deinit(allocator);

        for (edit_bytes) |b| {
            const vae_in = try decodeRefForVae(allocator, b, W, H, s); // [1,3,H,W] [-1,1]
            defer _ = mlx.mlx_array_free(vae_in);
            const lat = try ve.encode(vae_in, seed); // [1,128,lat_h,lat_w]
            defer _ = mlx.mlx_array_free(lat);
            try ref_packs.append(allocator, try packLatents(lat, s)); // [1,HW,128]

            const vlm = try decodeRefForVlm(allocator, b, self.cfg.vlm_min_pixels, self.cfg.vlm_max_pixels, s);
            try pv_chunks.append(allocator, vlm.pv);
            try grids.append(allocator, .{ 1, @intCast(vlm.gh), @intCast(vlm.gw) });
            try ntoks.append(allocator, vlm.ntok);
        }
        const ref_latents = try concat(ref_packs.items, 1, s); // [1, nrefs*HW, 128]
        defer _ = mlx.mlx_array_free(ref_latents);
        const pixel_values = try concat(pv_chunks.items, 0, s); // [total_patches, 1536]
        defer _ = mlx.mlx_array_free(pixel_values);

        // 2. Templated edit prompt (image placeholders expanded to Ntok each).
        const pr = try buildEditPromptIds(&self.tok, allocator, prompt, ntoks.items);
        defer allocator.free(pr.ids);
        defer allocator.free(pr.mask);
        const cond = try self.te.encodeEdit(vt, pr.ids, pr.mask, pixel_values, grids.items);
        const txt = cond.embeddings;
        defer _ = mlx.mlx_array_free(txt);

        // 3. Target noise [1,HW,128] + scheduler sigmas.
        const noise = try initNoisePacked(seed, lat_h, lat_w, self.dtype, s);
        defer _ = mlx.mlx_array_free(noise);
        const sigmas = try computeSigmas(allocator, n_steps, self.cfg.sched_shift);
        defer allocator.free(sigmas);

        // 4. Euler denoise over the concatenated [target, refs] stream.
        const target = try denoiseEditLoop(&self.dit, txt, noise, ref_latents, n_images, lat_h, lat_w, sigmas, self.dtype, s, progress);
        defer _ = mlx.mlx_array_free(target);

        // 5. Unpack target → VAE decode → [0,1].
        if (progress) |p| p.emit("Decoding image", n_steps, n_steps);
        const latent = try unpackLatents(target, lat_h, lat_w, s);
        defer _ = mlx.mlx_array_free(latent);
        const decoded = try self.vae.decode(latent);
        defer _ = mlx.mlx_array_free(decoded);
        return denormImage(decoded, s);
    }
};

/// Build the templated + tokenized edit prompt. Reproduces the reference
/// `format_edit` + processor placeholder expansion: each image's `<|image_pad|>`
/// is repeated `ntoks[k]` times (its merged vision-token count). Returns owned
/// ids/mask (all-ones — single unpadded prompt). Free function so the tokenizer
/// templating can be parity-tested without a full Engine.
fn buildEditPromptIds(tok: *const tok_mod.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8, ntoks: []const u32) !struct { ids: []i32, mask: []i32 } {
    var sb: std.ArrayList(u8) = .empty;
    defer sb.deinit(allocator);
    try sb.appendSlice(allocator, EDIT_PREFIX);
    for (ntoks, 0..) |ntok, k| {
        const hdr = try std.fmt.allocPrint(allocator, "Image {d}: <|vision_start|>", .{k + 1});
        defer allocator.free(hdr);
        try sb.appendSlice(allocator, hdr);
        for (0..ntok) |_| try sb.appendSlice(allocator, "<|image_pad|>");
        try sb.appendSlice(allocator, "<|vision_end|>");
    }
    try sb.appendSlice(allocator, prompt);
    try sb.appendSlice(allocator, EDIT_SUFFIX);

    const enc = try tok.encode(allocator, sb.items);
    defer allocator.free(enc);
    // Cap like the txt2img path: everything past the conditioning budget is
    // sliced off after the drop anyway, and the LM's causal mask is built dense
    // on the HOST — an uncapped client prompt is an O(L²) allocation (a 50k-token
    // prompt would ask for ~10 GB before a single matmul).
    const n = @min(enc.len, EDIT_DROP_TOKENS + TE_MAX_COND);
    const ids = try allocator.alloc(i32, n);
    const mask = try allocator.alloc(i32, n);
    for (enc[0..n], 0..) |t, i| {
        ids[i] = @intCast(t);
        mask[i] = 1;
    }
    return .{ .ids = ids, .mask = mask };
}

/// Round a dimension to the /16 floor (minimum 16), matching the reference
/// `normalize_image_dimension`.
fn normalizeDim(size: u32) u32 {
    return @max(MF_DOWNSAMPLE, MF_DOWNSAMPLE * (size / MF_DOWNSAMPLE));
}

/// The edit Euler denoise loop (E7.5). Each step concatenates the CONSTANT
/// reference latents after the target, runs the DiT over the full stream (the
/// multi-image RoPE is `buildDitRope(frames=1+nrefs)` — bit-identical to the
/// reference's per-image temporal offsets since all images share lh/lw), slices
/// the velocity back to the target tokens, and Euler-steps in f32. `noise` and
/// the result are packed target latents [1, HW, 128] (compute dtype). Factored
/// out so the loop math is parity-testable with any transformer. Caller frees.
fn denoiseEditLoop(dit: *const Dit, txt: mlx.mlx_array, noise: mlx.mlx_array, ref_latents: mlx.mlx_array, n_images: c_int, lat_h: c_int, lat_w: c_int, sigmas: []const f32, dtype: mlx.mlx_dtype, s: S, progress: ?sse.Progress) !mlx.mlx_array {
    const HW = lat_h * lat_w;
    var target = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&target, noise));
    for (0..sigmas.len - 1) |i| {
        const model_input = try concat(&.{ target, ref_latents }, 1, s); // [1, n_images*HW, 128]
        defer _ = mlx.mlx_array_free(model_input);
        const v_full = try dit.forward(model_input, txt, sigmas[i], n_images, lat_h, lat_w, null);
        defer _ = mlx.mlx_array_free(v_full);
        const v = try sliceTeSeq(v_full, 0, HW, s); // keep the target part
        defer _ = mlx.mlx_array_free(v);
        const target_f = try astype(target, .float32, s);
        defer _ = mlx.mlx_array_free(target_f);
        const v_f = try astype(v, .float32, s);
        defer _ = mlx.mlx_array_free(v_f);
        const dt = scalarF(sigmas[i + 1] - sigmas[i]);
        defer _ = mlx.mlx_array_free(dt);
        const stepv = try mulA(v_f, dt, s);
        defer _ = mlx.mlx_array_free(stepv);
        const nt_f = try addA(target_f, stepv, s);
        defer _ = mlx.mlx_array_free(nt_f);
        const nt = try astype(nt_f, dtype, s);
        _ = mlx.mlx_array_free(target);
        target = nt;
        _ = mlx.mlx_array_eval(target);
        if (progress) |p| p.emit("Generating", @intCast(i + 1), @intCast(sigmas.len - 1));
    }
    return target;
}

// Edit prompt template (`MageFlowPromptProcessor.EDIT_TEMPLATE`). The system
// prologue is load-bearing: the drop-64 in `encodeEdit` assumes exactly these
// template tokens ahead of the user content.
const EDIT_PREFIX =
    "<|im_start|>system\n" ++
    "Describe the key features of the input image (color, shape, size, texture," ++
    " objects, background), then explain how the user's text instruction should alter or modify the image. " ++
    "Generate a new image that meets the user's requirements while maintaining consistency with the original " ++
    "input where appropriate.<|im_end|>\n<|im_start|>user\n";
const EDIT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";

// VLM preprocessing constants. The long edge is the reference pipeline's own
// `resize_long_edge(384)` (util.py, NOT the processor config); the pixel budget
// comes from `text_encoder/preprocessor_config.json` via `Config` — these are
// only the fallbacks when the file is absent.
const VLM_LONG_EDGE: u32 = 384;
const VLM_MIN_PIXELS: u32 = 65_536;
const VLM_MAX_PIXELS: u32 = 16_777_216;
const VLM_FACTOR: u32 = 32; // patch(16) × merge(2)

/// The edit capability is gated on the repo/dir NAME. Verified against both
/// released repos side by side (2026-07-24): `model_index.json`,
/// `transformer/config.json` and `text_encoder/config.json` are BYTE-IDENTICAL
/// between Mage-Flow-Turbo and Mage-Flow-Edit-Turbo, and both ship the same
/// `visual.*` weights — nothing inside the checkpoint distinguishes them, so
/// there is no metadata gate to prefer. Rename the directory and you change the
/// capability; the load banner says which mode it came up in.
fn dirIsEdit(model_dir: []const u8) bool {
    return containsIgnoreCase(model_dir, "mage-flow-edit") or
        containsIgnoreCase(model_dir, "mageflow-edit");
}

/// Case-insensitive substring test (std.ascii has no indexOfIgnoreCase here).
fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    outer: while (i + needle.len <= haystack.len) : (i += 1) {
        for (needle, 0..) |c, j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(c)) continue :outer;
        }
        return true;
    }
    return false;
}

/// Decode a reference image and resize to the TARGET (W,H) for the VAE encoder,
/// normalized to [-1,1] NCHW (matches the reference `image.resize((W,H))` +
/// 0.5/0.5 normalize). Returns [1,3,H,W] f32 (owned).
fn decodeRefForVae(allocator: std.mem.Allocator, bytes: []const u8, W: u32, H: u32, s: S) !mlx.mlx_array {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src_ptr = stb.stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &w, &h, &ch, 3) orelse return error.ImageDecodeFailed;
    defer stb.stbi_image_free(src_ptr);
    const sw: u32 = @intCast(w);
    const sh: u32 = @intCast(h);
    if (sw == 0 or sh == 0) return error.ImageDecodeFailed;
    const rgb = src_ptr[0 .. @as(usize, sw) * sh * 3];
    const dst = try allocator.alloc(f32, @as(usize, W) * H * 3);
    defer allocator.free(dst);
    try qvis.resizeRgbBicubicNormalizedChw(allocator, dst, rgb, sh, sw, H, W);
    const shape = [_]c_int{ 1, 3, @intCast(H), @intCast(W) };
    const raw = mlx.mlx_array_new_data(dst.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(raw);
    return contig(raw, s); // detach from the freed host buffer
}

/// Decode a reference image and build the VLM `pixel_values` [Npatch,1536] the
/// vision tower expects: resize_long_edge(384) → smart-resize (/32 grid within
/// min/max pixels) → patchify (merge-block order, [C,tps,py,px] features). Also
/// returns the patch grid (gh,gw) and the merged-token count Ntok.
fn decodeRefForVlm(allocator: std.mem.Allocator, bytes: []const u8, min_px: u32, max_px: u32, s: S) !VlmInput {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src_ptr = stb.stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &w, &h, &ch, 3) orelse return error.ImageDecodeFailed;
    defer stb.stbi_image_free(src_ptr);
    const sw: u32 = @intCast(w);
    const sh: u32 = @intCast(h);
    if (sw == 0 or sh == 0) return error.ImageDecodeFailed;
    return vlmPixelValues(allocator, src_ptr[0 .. @as(usize, sw) * sh * 3], sh, sw, min_px, max_px, s);
}

const VlmInput = struct { pv: mlx.mlx_array, gh: u32, gw: u32, ntok: u32 };

/// The VLM pixel path over already-decoded RGB (pure but for the mlx array it
/// returns) — the half of `decodeRefForVlm` the preprocessing parity test drives
/// directly against the reference processor's `pixel_values`.
fn vlmPixelValues(allocator: std.mem.Allocator, rgb: []const u8, sh: u32, sw: u32, min_px: u32, max_px: u32, s: S) !VlmInput {
    // resize_long_edge(384): downscale so the long edge ≤ 384 (aspect kept).
    var le_h = sh;
    var le_w = sw;
    const long = @max(sw, sh);
    if (long > VLM_LONG_EDGE) {
        const scale = @as(f64, @floatFromInt(VLM_LONG_EDGE)) / @as(f64, @floatFromInt(long));
        le_w = @max(1, @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(sw)) * scale))));
        le_h = @max(1, @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(sh)) * scale))));
    }
    const resized = qvis.smartResizeImage(le_h, le_w, VLM_FACTOR, min_px, max_px);
    const rh = resized.h;
    const rw = resized.w;
    const chw = try allocator.alloc(f32, @as(usize, rh) * rw * 3);
    defer allocator.free(chw);
    // KNOWN DEVIATION: the reference resamples TWICE (source → long-edge 384 →
    // smart-resized grid); we resample once, straight to the final grid. The two
    // sizes differ by at most the /32 rounding, so this is a hair less blur on a
    // 384px conditioning thumbnail — measure before "fixing" it.
    try qvis.resizeRgbBicubicNormalizedChw(allocator, chw, rgb, sh, sw, rh, rw);

    const gh = rh / 16;
    const gw = rw / 16;
    const pv_buf = try allocator.alloc(f32, @as(usize, gh) * gw * @as(usize, @intCast(VIT_PATCH_IN)));
    defer allocator.free(pv_buf);
    qvis.buildPixelValues(pv_buf, chw, 3, rh, rw, 16, 2, 2);
    const shape = [_]c_int{ @intCast(gh * gw), VIT_PATCH_IN };
    const raw = mlx.mlx_array_new_data(pv_buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const pv = try contig(raw, s);
    return .{ .pv = pv, .gh = gh, .gw = gw, .ntok = (gh / 2) * (gw / 2) };
}

/// FlowMatchEuler static-shift sigmas [N+1]: base = linspace(1, 1/N, N);
/// sigma = shift·base / (1 + (shift−1)·base); then append 0. All f32.
fn computeSigmas(allocator: std.mem.Allocator, steps: u32, shift: f32) ![]f32 {
    const n: usize = steps;
    const out = try allocator.alloc(f32, n + 1);
    const sh: f64 = shift;
    for (0..n) |i| {
        const base: f64 = if (n == 1)
            1.0
        else
            1.0 + (1.0 / @as(f64, @floatFromInt(n)) - 1.0) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        out[i] = @floatCast(sh * base / (1.0 + (sh - 1.0) * base));
    }
    out[n] = 0.0;
    return out;
}

/// Initial noise [1,128,lat_h,lat_w] ~ N(0,1) (plain seed; NO Gaussian-Shading
/// watermark — the reference's torch-Philox+ndtri path is skipped), packed to
/// [1, lat_h·lat_w, 128] (transpose(0,2,3,1) → reshape).
fn initNoisePacked(seed: u64, lat_h: c_int, lat_w: c_int, dtype: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const nsh = [_]c_int{ 1, 128, lat_h, lat_w };
    var noise = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(noise);
    try mlx.check(mlx.mlx_random_normal(&noise, &nsh, 4, .float32, 0.0, 1.0, key, s));
    const nd = try astype(noise, dtype, s);
    defer _ = mlx.mlx_array_free(nd);
    return packLatents(nd, s);
}

/// pack_latents: NCHW [1,128,h,w] → transpose(0,2,3,1) → reshape [1, h·w, 128].
fn packLatents(latents: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(latents); // [1,C,h,w]
    const c = sh[1];
    const h = sh[2];
    const wd = sh[3];
    const t = try transpose(latents, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ 1, h * wd, c }, s);
}

/// unpack_latents: [1, h·w, 128] → reshape [1,h,w,128] → transpose(0,3,1,2).
fn unpackLatents(latents: mlx.mlx_array, lat_h: c_int, lat_w: c_int, s: S) !mlx.mlx_array {
    const c = mlx.getShape(latents)[2];
    const r = try reshape(latents, &[_]c_int{ 1, lat_h, lat_w, c }, s);
    defer _ = mlx.mlx_array_free(r);
    return transpose(r, &[_]c_int{ 0, 3, 1, 2 }, s);
}

/// Denormalize decoded pixels [1,3,H,W] in [-1,1] → clip(x·0.5 + 0.5, 0, 1) f32.
fn denormImage(decoded: mlx.mlx_array, s: S) !mlx.mlx_array {
    const df = try astype(decoded, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const sc = try mulA(df, half, s);
    defer _ = mlx.mlx_array_free(sc);
    const shifted = try addA(sc, half, s);
    defer _ = mlx.mlx_array_free(shifted);
    const lo = scalarF(0.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = scalarF(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var clo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(clo);
    try mlx.check(mlx.mlx_maximum(&clo, shifted, lo, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_minimum(&out, clo, hi, s));
    return out;
}

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
            self.conv_in_w,    self.conv_in_b,    self.norm_out_w,   self.norm_out_b,
            self.conv_out_w,   self.conv_out_b,   self.denoiser_t,   self.s_proj2_w,
            self.s_proj2_b,    self.yx_w,         self.yx_b,         self.xemb_w,
            self.xemb_b,       self.nerf_pe,      self.cond_embed_w, self.cond_embed_b,
            self.input_proj_w, self.input_proj_b, self.final_norm_w, self.final_lin_w,
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
        return dicoBlockForward(dw, inp, self.denoiser_t, self.s);
    }
};

/// One DiCoBlock forward with a precomputed (constant, t=0) timestep embedding.
/// Shared by the VAE decoder denoiser and the VAE encoder (same block, distinct
/// t_embedder weights ⇒ distinct `denoiser_t`).
fn dicoBlockForward(dw: *const DiCoW, inp: mlx.mlx_array, denoiser_t: mlx.mlx_array, s: S) !mlx.mlx_array {
    // AdaLN modulation from the (constant, t=0) timestep vector.
    const st = try silu(denoiser_t, s);
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
    return sliceContig(x, &lo, &hi, &st, s);
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
    return buildTimestepT0(w, a, "pipeline.t_embedder", s);
}
/// The t=0 timestep embedding through a `{prefix}.mlp.{0,2}` TimestepEmbedder:
/// sinusoidal [ones(128), zeros(128)] → mlp.0 → silu → mlp.2 → [1,384]. Shared by
/// the VAE decoder denoiser (`pipeline.t_embedder`) and the VAE encoder
/// (`student.dconv_encoder.t_embedder`) — both fold their DiCo AdaLN at t=0.
fn buildTimestepT0(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    // t=0 sinusoidal embedding: [ones(128), zeros(128)] → mlp.0 → silu → mlp.2
    var emb0: [256]f32 = undefined;
    for (0..128) |i| emb0[i] = 1.0;
    for (128..256) |i| emb0[i] = 0.0;
    const shape = [_]c_int{ 1, 256 };
    const raw = mlx.mlx_array_new_data(&emb0, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const m0p = try std.fmt.allocPrint(a, "{s}.mlp.0", .{prefix});
    defer a.free(m0p);
    const m2p = try std.fmt.allocPrint(a, "{s}.mlp.2", .{prefix});
    defer a.free(m2p);
    const m0w = try loadLinT(w, a, m0p, s);
    defer _ = mlx.mlx_array_free(m0w);
    const m0b = try loadVec(w, a, m0p, "bias", s);
    defer _ = mlx.mlx_array_free(m0b);
    const m2w = try loadLinT(w, a, m2p, s);
    defer _ = mlx.mlx_array_free(m2w);
    const m2b = try loadVec(w, a, m2p, "bias", s);
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
// MageVAE encoder (DiCo `_DConvEncoder`). Ported from mflux `mage_flow_vae.py`.
// Produces the CLEAN reference latents the edit pipeline concatenates into the
// DiT image stream. Loads the `student.dconv_encoder.*` subtree; runs f32 (the
// proven VAE precision — the encoder feeds the DiT which normalizes). The 21
// conditioning DiCo blocks are the SAME block as the decoder denoiser (shared
// `dicoBlockForward`), folded at the encoder's own t=0 timestep; the 2 head
// blocks are the AdaLN-free `_EncoderDiCoBlock` variant.
// ══════════════════════════════════════════════════════════════════════════

const VAE_ENC_HEAD: c_int = 768; // _DConvEncoder head_size
const NUM_ENC_HEAD_BLOCKS = 2; // num_head_blocks
const NUM_ENC_DICO_BLOCKS = 21; // num_blocks
const VAE_LATENT: c_int = 128; // z_ch

/// `_EncoderDiCoBlock` weights (AdaLN-free; affine norm1/norm2, no gate).
const EncDiCoW = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
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
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    fn deinit(self: *EncDiCoW) void {
        inline for (.{ self.n1w, self.n1b, self.c1w, self.c1b, self.c2w, self.c2b, self.c3w, self.c3b, self.caw, self.cab, self.c4w, self.c4b, self.c5w, self.c5b, self.n2w, self.n2b }) |f|
            _ = mlx.mlx_array_free(f);
    }
};

fn loadEncDiCo(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !EncDiCoW {
    const n1 = try std.fmt.allocPrint(a, "{s}.norm1", .{pfx});
    defer a.free(n1);
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
    const n2 = try std.fmt.allocPrint(a, "{s}.norm2", .{pfx});
    defer a.free(n2);
    return .{
        .n1w = try loadVec(w, a, n1, "weight", s),
        .n1b = try loadVec(w, a, n1, "bias", s),
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
        .n2w = try loadVec(w, a, n2, "weight", s),
        .n2b = try loadVec(w, a, n2, "bias", s),
    };
}

/// `_EncoderDiCoBlock.__call__`: two residual halves, no AdaLN, no gate.
fn encDiCoForward(ew: *const EncDiCoW, inp: mlx.mlx_array, s: S) !mlx.mlx_array {
    const x = try layerNormLast(inp, ew.n1w, ew.n1b, 1e-6, s);
    defer _ = mlx.mlx_array_free(x);
    const c1 = try conv2d(x, ew.c1w, ew.c1b, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(c1);
    const c2 = try conv2d(c1, ew.c2w, ew.c2b, 1, 1, VAE_ENC_HEAD, s); // depthwise 3x3
    defer _ = mlx.mlx_array_free(c2);
    const g1 = try gelu(c2, s);
    defer _ = mlx.mlx_array_free(g1);
    const red = [_]c_int{ 1, 2 };
    const pooled = try meanAxes(g1, &red, true, s);
    defer _ = mlx.mlx_array_free(pooled);
    const ca_c = try conv2d(pooled, ew.caw, ew.cab, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(ca_c);
    const ca_s = try sigmoidA(ca_c, s);
    defer _ = mlx.mlx_array_free(ca_s);
    const attd = try mulA(g1, ca_s, s);
    defer _ = mlx.mlx_array_free(attd);
    const c3 = try conv2d(attd, ew.c3w, ew.c3b, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(c3);
    const x1 = try addA(inp, c3, s);
    defer _ = mlx.mlx_array_free(x1);
    const n2 = try layerNormLast(x1, ew.n2w, ew.n2b, 1e-6, s);
    defer _ = mlx.mlx_array_free(n2);
    const c4 = try conv2d(n2, ew.c4w, ew.c4b, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(c4);
    const g2 = try gelu(c4, s);
    defer _ = mlx.mlx_array_free(g2);
    const c5 = try conv2d(g2, ew.c5w, ew.c5b, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(c5);
    return addA(x1, c5, s);
}

/// MageVAE encoder — the deterministic `encode_moments` network plus the seeded
/// posterior sample. Loads only the `student.dconv_encoder.*` subtree (f32).
pub const VaeEncoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    pce_w: mlx.mlx_array, // patch_cond_embed Conv2d(3→768, k16 s16)
    pce_b: mlx.mlx_array,
    head_blocks: [NUM_ENC_HEAD_BLOCKS]EncDiCoW,
    proj_down_w: mlx.mlx_array, // 768→384 k1
    proj_down_b: mlx.mlx_array,
    z_proj_w: mlx.mlx_array, // 128→384 k1
    z_proj_b: mlx.mlx_array,
    fuse_proj_w: mlx.mlx_array, // 768→384 k1
    fuse_proj_b: mlx.mlx_array,
    enc_t: mlx.mlx_array, // t=0 timestep embedding [1,384]
    blocks: [NUM_ENC_DICO_BLOCKS]DiCoW,
    norm_out_w: mlx.mlx_array, // LayerNorm2d(384) affine
    norm_out_b: mlx.mlx_array,
    proj_out_w: mlx.mlx_array, // 384→256 k1
    proj_out_b: mlx.mlx_array,

    const P = "student.dconv_encoder";

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !VaeEncoder {
        const dir = try std.fmt.allocPrint(allocator, "{s}/vae", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        const a = allocator;
        var self: VaeEncoder = undefined;
        self.allocator = allocator;
        self.s = s;

        self.pce_w = try loadConvW(&w, a, P ++ ".patch_cond_embed", s);
        self.pce_b = try loadVec(&w, a, P ++ ".patch_cond_embed", "bias", s);
        for (0..NUM_ENC_HEAD_BLOCKS) |i| {
            const pfx = try std.fmt.allocPrint(a, P ++ ".head_blocks.{d}", .{i});
            defer a.free(pfx);
            self.head_blocks[i] = try loadEncDiCo(&w, a, pfx, s);
        }
        self.proj_down_w = try loadConvW(&w, a, P ++ ".proj_down", s);
        self.proj_down_b = try loadVec(&w, a, P ++ ".proj_down", "bias", s);
        self.z_proj_w = try loadConvW(&w, a, P ++ ".z_proj", s);
        self.z_proj_b = try loadVec(&w, a, P ++ ".z_proj", "bias", s);
        self.fuse_proj_w = try loadConvW(&w, a, P ++ ".fuse_proj", s);
        self.fuse_proj_b = try loadVec(&w, a, P ++ ".fuse_proj", "bias", s);
        self.enc_t = try buildTimestepT0(&w, a, P ++ ".t_embedder", s);
        for (0..NUM_ENC_DICO_BLOCKS) |i| {
            const pfx = try std.fmt.allocPrint(a, P ++ ".blocks.{d}", .{i});
            defer a.free(pfx);
            self.blocks[i] = try loadDiCo(&w, a, pfx, s);
        }
        self.norm_out_w = try loadVec(&w, a, P ++ ".norm_out", "weight", s);
        self.norm_out_b = try loadVec(&w, a, P ++ ".norm_out", "bias", s);
        self.proj_out_w = try loadConvW(&w, a, P ++ ".proj_out", s);
        self.proj_out_b = try loadVec(&w, a, P ++ ".proj_out", "bias", s);
        return self;
    }

    pub fn deinit(self: *VaeEncoder) void {
        const frees = [_]mlx.mlx_array{
            self.pce_w,      self.pce_b,      self.proj_down_w, self.proj_down_b,
            self.z_proj_w,   self.z_proj_b,   self.fuse_proj_w, self.fuse_proj_b,
            self.enc_t,      self.norm_out_w, self.norm_out_b,  self.proj_out_w,
            self.proj_out_b,
        };
        for (frees) |f| _ = mlx.mlx_array_free(f);
        for (&self.head_blocks) |*b| b.deinit();
        for (&self.blocks) |*b| freeDiCo(b);
    }

    /// The deterministic moments network. `image_nchw` is [-1,1] f32 NCHW; returns
    /// (mean, logvar) both NCHW [B,128,H/16,W/16] (logvar clipped to [-20,10]).
    /// Caller frees both.
    pub fn encodeMoments(self: *const VaeEncoder, image_nchw: mlx.mlx_array) !struct { mean: mlx.mlx_array, logvar: mlx.mlx_array } {
        const s = self.s;
        const y = try transpose(image_nchw, &[_]c_int{ 0, 2, 3, 1 }, s); // NHWC
        defer _ = mlx.mlx_array_free(y);
        const yf = try astype(y, .float32, s);
        defer _ = mlx.mlx_array_free(yf);

        // patch_cond_embed + head_blocks + proj_down.
        var cond = try conv2d(yf, self.pce_w, self.pce_b, VAE_PATCH, 0, 1, s); // [B,h',w',768]
        for (&self.head_blocks) |*hb| {
            const nxt = try encDiCoForward(hb, cond, s);
            _ = mlx.mlx_array_free(cond);
            cond = nxt;
        }
        const cd = try conv2d(cond, self.proj_down_w, self.proj_down_b, 1, 0, 1, s); // [B,h',w',384]
        _ = mlx.mlx_array_free(cond);
        defer _ = mlx.mlx_array_free(cd);

        const csh = mlx.getShape(cd);
        const B = csh[0];
        const gh = csh[1];
        const gw = csh[2];

        // fuse_proj(concat([cond, z_proj(zeros)])) — z_t is the zero latent state.
        const z_state = try zerosLike(B, gh, gw, VAE_LATENT, s);
        defer _ = mlx.mlx_array_free(z_state);
        const zp = try conv2d(z_state, self.z_proj_w, self.z_proj_b, 1, 0, 1, s); // [B,h',w',384]
        defer _ = mlx.mlx_array_free(zp);
        const fused_in = try concat(&.{ cd, zp }, 3, s); // [B,h',w',768]
        defer _ = mlx.mlx_array_free(fused_in);
        var h = try conv2d(fused_in, self.fuse_proj_w, self.fuse_proj_b, 1, 0, 1, s); // [B,h',w',384]
        for (&self.blocks) |*blk| {
            const nxt = try dicoBlockForward(blk, h, self.enc_t, s);
            _ = mlx.mlx_array_free(h);
            h = nxt;
        }
        const no = try layerNormLast(h, self.norm_out_w, self.norm_out_b, 1e-6, s);
        _ = mlx.mlx_array_free(h);
        defer _ = mlx.mlx_array_free(no);
        const moments = try conv2d(no, self.proj_out_w, self.proj_out_b, 1, 0, 1, s); // [B,h',w',256]
        defer _ = mlx.mlx_array_free(moments);

        var parts: [2]mlx.mlx_array = undefined;
        try splitEqual(moments, 2, 3, &parts, s); // mean, logvar (NHWC)
        const mean_nhwc = parts[0];
        defer _ = mlx.mlx_array_free(mean_nhwc);
        const logvar_raw = parts[1];
        defer _ = mlx.mlx_array_free(logvar_raw);
        // clip logvar to [-20, 10].
        const lo = scalarF(-20.0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = scalarF(10.0);
        defer _ = mlx.mlx_array_free(hi);
        var clamped_lo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clamped_lo);
        try mlx.check(mlx.mlx_maximum(&clamped_lo, logvar_raw, lo, s));
        var logvar_nhwc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logvar_nhwc);
        try mlx.check(mlx.mlx_minimum(&logvar_nhwc, clamped_lo, hi, s));

        // NHWC → NCHW.
        const mean = try transpose(mean_nhwc, &[_]c_int{ 0, 3, 1, 2 }, s);
        const logvar = try transpose(logvar_nhwc, &[_]c_int{ 0, 3, 1, 2 }, s);
        return .{ .mean = mean, .logvar = logvar };
    }

    /// Full posterior encode: mean + exp(0.5·logvar)·N(0,1;seed). Returns the
    /// clean reference latent NCHW [B,128,H/16,W/16] (caller frees).
    pub fn encode(self: *const VaeEncoder, image_nchw: mlx.mlx_array, seed: u64) !mlx.mlx_array {
        const s = self.s;
        const m = try self.encodeMoments(image_nchw);
        defer _ = mlx.mlx_array_free(m.mean);
        defer _ = mlx.mlx_array_free(m.logvar);
        const msh = mlx.getShape(m.mean);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        var noise = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(noise);
        try mlx.check(mlx.mlx_random_normal(&noise, msh.ptr, @intCast(msh.len), .float32, 0.0, 1.0, key, s));
        const half = scalarF(0.5);
        defer _ = mlx.mlx_array_free(half);
        const hv = try mulA(m.logvar, half, s);
        defer _ = mlx.mlx_array_free(hv);
        var std_dev = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(std_dev);
        try mlx.check(mlx.mlx_exp(&std_dev, hv, s));
        const scaled = try mulA(std_dev, noise, s);
        defer _ = mlx.mlx_array_free(scaled);
        return addA(m.mean, scaled, s);
    }
};

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
// Input dims of the DiT's linears — needed by `MfLinear.load` to solve a packed
// weight's (bits, group_size) from its geometry, so they are load-bearing on the
// quantized path rather than documentation.
const DIT_IN_CH: c_int = 128; // img_in: packed latent channels
const DIT_CONTEXT: c_int = 2560; // txt_in: the text encoder's hidden size
const DIT_TIME_FREQ: c_int = 256; // timestep sinusoidal embedding width
const DIT_MLP: c_int = DIT_HIDDEN * 4; // mlp_ratio 4.0

/// GELU tanh approximation (nn.gelu_approx): 0.5x(1+tanh(√(2/π)(x+0.044715x³))).
/// Scalar constants match x's dtype so the bf16 chain stays bf16.
fn geluTanh(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c: f32 = 0.7978845608028654;
    const k = try scalarLike(0.044715, x, s);
    defer _ = mlx.mlx_array_free(k);
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
    defer _ = mlx.mlx_array_free(x3);
    const kx3 = try mulA(x3, k, s);
    defer _ = mlx.mlx_array_free(kx3);
    const inner = try addA(x, kx3, s);
    defer _ = mlx.mlx_array_free(inner);
    const ca = try scalarLike(c, x, s);
    defer _ = mlx.mlx_array_free(ca);
    const cin = try mulA(inner, ca, s);
    defer _ = mlx.mlx_array_free(cin);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, cin, s));
    const one = try scalarLike(1.0, x, s);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(t, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = try scalarLike(0.5, x, s);
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
    qw: MfLinear,
    qb: mlx.mlx_array,
    kw: MfLinear,
    kb: mlx.mlx_array,
    vw: MfLinear,
    vb: mlx.mlx_array,
    aqw: MfLinear,
    aqb: mlx.mlx_array,
    akw: MfLinear,
    akb: mlx.mlx_array,
    avw: MfLinear,
    avb: mlx.mlx_array,
    nq: mlx.mlx_array,
    nk: mlx.mlx_array,
    naq: mlx.mlx_array,
    nak: mlx.mlx_array,
    ow: MfLinear,
    ob: mlx.mlx_array,
    aow: MfLinear,
    aob: mlx.mlx_array,
};
const DitBlockW = struct {
    img_mod_w: MfLinear,
    img_mod_b: mlx.mlx_array,
    txt_mod_w: MfLinear,
    txt_mod_b: mlx.mlx_array,
    attn: DitAttnW,
    img0w: MfLinear,
    img0b: mlx.mlx_array,
    img2w: MfLinear,
    img2b: mlx.mlx_array,
    txt0w: MfLinear,
    txt0b: mlx.mlx_array,
    txt2w: MfLinear,
    txt2b: mlx.mlx_array,
};

pub const Dit = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype, // compute dtype (bf16 live, f32 parity)
    img_in_w: MfLinear,
    img_in_b: mlx.mlx_array,
    txt_norm_w: mlx.mlx_array,
    txt_in_w: MfLinear,
    txt_in_b: mlx.mlx_array,
    t1w: MfLinear,
    t1b: mlx.mlx_array,
    t2w: MfLinear,
    t2b: mlx.mlx_array,
    blocks: [DIT_DEPTH]DitBlockW,
    norm_out_w: MfLinear,
    norm_out_b: mlx.mlx_array,
    proj_out_w: MfLinear,
    proj_out_b: mlx.mlx_array,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !Dit {
        const dir = try std.fmt.allocPrint(allocator, "{s}/transformer", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        const a = allocator;
        var self: Dit = undefined;
        self.allocator = allocator;
        self.s = s;
        self.dtype = dtype;
        self.img_in_w = try MfLinear.load(&w, a, "img_in", DIT_IN_CH, dtype, s);
        self.img_in_b = try loadVecDt(&w, a, "img_in", "bias", dtype, s);
        self.txt_norm_w = try loadVec(&w, a, "txt_norm", "weight", s); // norm: f32
        self.txt_in_w = try MfLinear.load(&w, a, "txt_in", DIT_CONTEXT, dtype, s);
        self.txt_in_b = try loadVecDt(&w, a, "txt_in", "bias", dtype, s);
        self.t1w = try MfLinear.load(&w, a, "time_text_embed.timestep_embedder.linear_1", DIT_TIME_FREQ, dtype, s);
        self.t1b = try loadVecDt(&w, a, "time_text_embed.timestep_embedder.linear_1", "bias", dtype, s);
        self.t2w = try MfLinear.load(&w, a, "time_text_embed.timestep_embedder.linear_2", DIT_HIDDEN, dtype, s);
        self.t2b = try loadVecDt(&w, a, "time_text_embed.timestep_embedder.linear_2", "bias", dtype, s);
        for (0..DIT_DEPTH) |i| {
            self.blocks[i] = try loadDitBlock(&w, a, i, dtype, s);
        }
        self.norm_out_w = try MfLinear.load(&w, a, "norm_out.linear", DIT_HIDDEN, dtype, s);
        self.norm_out_b = try loadVecDt(&w, a, "norm_out.linear", "bias", dtype, s);
        self.proj_out_w = try MfLinear.load(&w, a, "proj_out", DIT_HIDDEN, dtype, s);
        self.proj_out_b = try loadVecDt(&w, a, "proj_out", "bias", dtype, s);
        return self;
    }

    pub fn deinit(self: *Dit) void {
        inline for (.{ &self.img_in_w, &self.txt_in_w, &self.t1w, &self.t2w, &self.norm_out_w, &self.proj_out_w }) |f| f.deinit();
        const top = [_]mlx.mlx_array{
            self.img_in_b, self.txt_norm_w,  self.txt_in_b,
            self.t1b,      self.t2b,         self.norm_out_b,
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

        const img_f = try astype(img_in, self.dtype, s);
        defer _ = mlx.mlx_array_free(img_f);
        var img = try self.img_in_w.forward(img_f, self.img_in_b, s); // [B,Limg,3072]
        const txt_f = try astype(txt_in, self.dtype, s);
        defer _ = mlx.mlx_array_free(txt_f);
        const txt_n = try rmsNormLast(txt_f, self.txt_norm_w, 1e-6, s);
        defer _ = mlx.mlx_array_free(txt_n);
        var txt = try self.txt_in_w.forward(txt_n, self.txt_in_b, s); // [B,Ltxt,3072]

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
            img = res.img;
            txt = res.txt;
        }
        defer _ = mlx.mlx_array_free(txt);
        defer _ = mlx.mlx_array_free(img);

        // norm_out (AdaLayerNormContinuous) + proj_out
        const st = try silu(temb, s);
        defer _ = mlx.mlx_array_free(st);
        const nmod = try self.norm_out_w.forward(st, self.norm_out_b, s); // [B,6144]
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
        return self.proj_out_w.forward(modded, self.proj_out_b, s); // [B,Limg,128]
    }

    fn timeTextEmbed(self: *const Dit, t: f32, B: c_int) !mlx.mlx_array {
        const s = self.s;
        // Sinusoidal proj [B,256]. The released model rounds its frequency table
        // to the timestep dtype (bf16 in the live engine) — a training-baked
        // detail we replicate on the bf16 path (f32 path leaves freqs full).
        const round_bf16 = self.dtype != .float32;
        // The reference casts the timestep to the model dtype (bf16) before the
        // embedding; a distilled 4-step model memorizes that exact conditioning,
        // so an f32 timestep washes the output. Round both the value and freqs.
        const tv: f64 = if (round_bf16) roundBf16(t) else t;
        var buf: [256]f32 = undefined;
        const half = 128;
        const scale: f64 = 1000.0;
        const max_period: f64 = 10000.0;
        for (0..half) |j| {
            const exponent = -std.math.log(f64, std.math.e, max_period) * @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(half));
            const freq: f64 = if (round_bf16) roundBf16(@floatCast(@exp(exponent))) else @exp(exponent);
            const ang = tv * freq * scale;
            buf[j] = @floatCast(@cos(ang)); // flip_sin_to_cos → cos first
            buf[half + j] = @floatCast(@sin(ang));
        }
        const shape = [_]c_int{ 1, 256 };
        const raw = mlx.mlx_array_new_data(&buf, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(raw);
        // Cast the projection to the compute dtype before the (dtype) linears.
        var proj = try astype(raw, self.dtype, s);
        if (B > 1) {
            const bshape = [_]c_int{ B, 256 };
            var bo = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_broadcast_to(&bo, proj, &bshape, 2, s));
            _ = mlx.mlx_array_free(proj);
            proj = bo;
        }
        defer _ = mlx.mlx_array_free(proj);
        const l1 = try self.t1w.forward(proj, self.t1b, s);
        defer _ = mlx.mlx_array_free(l1);
        const a1 = try silu(l1, s);
        defer _ = mlx.mlx_array_free(a1);
        return self.t2w.forward(a1, self.t2b, s);
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
    ) !StreamPair {
        const s = self.s;
        const st = try silu(temb, s);
        defer _ = mlx.mlx_array_free(st);
        const img_mod = try bw.img_mod_w.forward(st, bw.img_mod_b, s); // [B,18432]
        defer _ = mlx.mlx_array_free(img_mod);
        const txt_mod = try bw.txt_mod_w.forward(st, bw.txt_mod_b, s);
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
        const img_attn = attn.img;
        defer _ = mlx.mlx_array_free(img_attn);
        const txt_attn = attn.txt;
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
        const img_ff = try feedForward(&bw.img0w, bw.img0b, &bw.img2w, bw.img2b, imod2.hidden, s);
        defer _ = mlx.mlx_array_free(img_ff);
        const img_out = try gateAdd(img_g1, imod2.gate, img_ff, s);

        const tn2 = try layerNormLast(txt_g1, null, null, 1e-6, s);
        defer _ = mlx.mlx_array_free(tn2);
        const tmod2 = try modulateSeq(tn2, tm2[1], s);
        defer _ = mlx.mlx_array_free(tmod2.hidden);
        defer _ = mlx.mlx_array_free(tmod2.gate);
        const txt_ff = try feedForward(&bw.txt0w, bw.txt0b, &bw.txt2w, bw.txt2b, tmod2.hidden, s);
        defer _ = mlx.mlx_array_free(txt_ff);
        const txt_out = try gateAdd(txt_g1, tmod2.gate, txt_ff, s);

        return .{ .img = img_out, .txt = txt_out };
    }
};

/// The two streams a double-stream DiT carries. Named rather than a `[2]` tuple:
/// `jointAttn` and `ditBlock` used to return the same pair in OPPOSITE orders,
/// which is where a caller-owns mistake hides in plain sight.
const StreamPair = struct { img: mlx.mlx_array, txt: mlx.mlx_array };

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
    const one = try scalarLike(1.0, scale, s);
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
fn feedForward(w0: *const MfLinear, b0: mlx.mlx_array, w2: *const MfLinear, b2: mlx.mlx_array, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const p = try w0.forward(x, b0, s);
    defer _ = mlx.mlx_array_free(p);
    const g = try geluTanh(p, s);
    defer _ = mlx.mlx_array_free(g);
    return w2.forward(g, b2, s);
}
fn splitHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,L,3072]
    return reshape(x, &[_]c_int{ sh[0], sh[1], DIT_HEADS, DIT_HEAD_DIM }, s);
}
/// Adjacent-pair complex RoPE. x [B,L,H,128]; cos/sin [L,64]. The rotation runs
/// in f32 (cos/sin stay f32, x upcast), then the result is cast back to x's dtype
/// — matching the reference `apply_rotary_emb` (a bf16 rotation drifts over the
/// 12 blocks × N steps and washes the output).
fn applyRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const sh = mlx.getShape(xf); // [B,L,H,128]
    const B = sh[0];
    const L = sh[1];
    const Hh = sh[2];
    const D = sh[3];
    const pairs = try reshape(xf, &[_]c_int{ B, L, Hh, @divExact(D, 2), 2 }, s);
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
    const flat = try reshape(stacked, &[_]c_int{ B, L, Hh, D }, s); // f32
    if (in_dt == .float32) return flat;
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, in_dt, s);
}
fn jointAttn(aw: *const DitAttnW, img_in: mlx.mlx_array, txt_in: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, mask: ?mlx.mlx_array, Ltxt: c_int, s: S) !StreamPair {
    // Image q/k/v.
    const iq0 = try aw.qw.forward(img_in, aw.qb, s);
    defer _ = mlx.mlx_array_free(iq0);
    const ik0 = try aw.kw.forward(img_in, aw.kb, s);
    defer _ = mlx.mlx_array_free(ik0);
    const iv0 = try aw.vw.forward(img_in, aw.vb, s);
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
    const tq0 = try aw.aqw.forward(txt_in, aw.aqb, s);
    defer _ = mlx.mlx_array_free(tq0);
    const tk0 = try aw.akw.forward(txt_in, aw.akb, s);
    defer _ = mlx.mlx_array_free(tk0);
    const tv0 = try aw.avw.forward(txt_in, aw.avb, s);
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
    const txt_out = try aw.aow.forward(txt_slice, aw.aob, s);
    const img_out = try aw.ow.forward(img_slice, aw.ob, s);
    return .{ .img = img_out, .txt = txt_out };
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
    return sliceContig(x, &lo, &hi, &st, s);
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

fn loadDitBlock(w: *const Weights, a: std.mem.Allocator, i: usize, dtype: mlx.mlx_dtype, s: S) !DitBlockW {
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
        .img_mod_w = try MfLinear.load(w, a, im, DIT_HIDDEN, dtype, s),
        .img_mod_b = try loadVecDt(w, a, im, "bias", dtype, s),
        .txt_mod_w = try MfLinear.load(w, a, tmo, DIT_HIDDEN, dtype, s),
        .txt_mod_b = try loadVecDt(w, a, tmo, "bias", dtype, s),
        .attn = try loadDitAttn(w, a, p, dtype, s),
        .img0w = try MfLinear.load(w, a, imlp0, DIT_HIDDEN, dtype, s),
        .img0b = try loadVecDt(w, a, imlp0, "bias", dtype, s),
        .img2w = try MfLinear.load(w, a, imlp2, DIT_MLP, dtype, s),
        .img2b = try loadVecDt(w, a, imlp2, "bias", dtype, s),
        .txt0w = try MfLinear.load(w, a, tmlp0, DIT_HIDDEN, dtype, s),
        .txt0b = try loadVecDt(w, a, tmlp0, "bias", dtype, s),
        .txt2w = try MfLinear.load(w, a, tmlp2, DIT_MLP, dtype, s),
        .txt2b = try loadVecDt(w, a, tmlp2, "bias", dtype, s),
    };
}
fn loadDitAttn(w: *const Weights, a: std.mem.Allocator, p: []const u8, dtype: mlx.mlx_dtype, s: S) !DitAttnW {
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
        .qw = try MfLinear.load(w, a, qk, DIT_HIDDEN, dtype, s),
        .qb = try loadVecDt(w, a, qk, "bias", dtype, s),
        .kw = try MfLinear.load(w, a, kk, DIT_HIDDEN, dtype, s),
        .kb = try loadVecDt(w, a, kk, "bias", dtype, s),
        .vw = try MfLinear.load(w, a, vk, DIT_HIDDEN, dtype, s),
        .vb = try loadVecDt(w, a, vk, "bias", dtype, s),
        .aqw = try MfLinear.load(w, a, aqk, DIT_HIDDEN, dtype, s),
        .aqb = try loadVecDt(w, a, aqk, "bias", dtype, s),
        .akw = try MfLinear.load(w, a, akk, DIT_HIDDEN, dtype, s),
        .akb = try loadVecDt(w, a, akk, "bias", dtype, s),
        .avw = try MfLinear.load(w, a, avk, DIT_HIDDEN, dtype, s),
        .avb = try loadVecDt(w, a, avk, "bias", dtype, s),
        .nq = try loadVec(w, a, nqk, "weight", s), // q/k norms: f32
        .nk = try loadVec(w, a, nkk, "weight", s),
        .naq = try loadVec(w, a, naqk, "weight", s),
        .nak = try loadVec(w, a, nakk, "weight", s),
        // to_out / to_add_out consume the concatenated heads: 24×128 == 3072.
        .ow = try MfLinear.load(w, a, ok, DIT_HEADS * DIT_HEAD_DIM, dtype, s),
        .ob = try loadVecDt(w, a, ok, "bias", dtype, s),
        .aow = try MfLinear.load(w, a, aok, DIT_HEADS * DIT_HEAD_DIM, dtype, s),
        .aob = try loadVecDt(w, a, aok, "bias", dtype, s),
    };
}
fn freeDitAttn(x: *DitAttnW) void {
    inline for (.{ &x.qw, &x.kw, &x.vw, &x.aqw, &x.akw, &x.avw, &x.ow, &x.aow }) |f| f.deinit();
    inline for (.{ x.qb, x.kb, x.vb, x.aqb, x.akb, x.avb, x.nq, x.nk, x.naq, x.nak, x.ob, x.aob }) |f| _ = mlx.mlx_array_free(f);
}
fn freeDitBlock(b: *DitBlockW) void {
    inline for (.{ &b.img_mod_w, &b.txt_mod_w, &b.img0w, &b.img2w, &b.txt0w, &b.txt2w }) |f| f.deinit();
    inline for (.{ b.img_mod_b, b.txt_mod_b, b.img0b, b.img2b, b.txt0b, b.txt2b }) |f| _ = mlx.mlx_array_free(f);
    freeDitAttn(&b.attn);
}

// ══════════════════════════════════════════════════════════════════════════
// MageFlow vision tower — Qwen3-VL ViT with DeepStack (edit conditioning, E7.2).
// Ported from mflux `mage_flow_text_encoder/vision_model.py`. Conv3d patch embed
// (run as a linear over the flattened patch), 48²-grid learned pos-embed bilinear
// interpolated to the patch grid, 24 full-attention blocks with a 2D rotary
// (rotate-half, computed f32), a final 2×2 patch merger to 2560, plus 3 DeepStack
// mergers (post-shuffle norm) tapped after blocks [5,11,17]. Pixel_values arrive
// in Qwen spatial-merge order (from the processor / E7.3); pos-embed + rotary are
// built in the SAME merge order so the merger's 4-token grouping aligns.
// ══════════════════════════════════════════════════════════════════════════

// These four are shared by every Qwen3-VL tower (`QWEN3VL_VISION_COMMON`), so
// they stay file constants; everything that varies with the LM size lives in
// VitConfig below.
const VIT_MERGE: c_int = 2;
const VIT_GRID_SIDE: usize = 48; // sqrt(num_position_embeddings 2304)
const VIT_PATCH_IN: c_int = 1536; // in_ch(3)*temporal(2)*patch(16)*patch(16)
const VIT_THETA: f64 = 10000.0;

/// Per-checkpoint tower geometry. Qwen3-VL ships ONE vision design across LM
/// sizes and changes only these numbers, so a second consumer (MiniMax H3's
/// 32B conditioner) is a config, not a second implementation. Getting `depth`
/// or `deepstack` wrong is silent — the tower loads and conditions on features
/// tapped at the wrong layers.
pub const VitConfig = struct {
    hidden: c_int,
    heads: c_int,
    inter: c_int,
    depth: usize,
    /// Merged-token width, i.e. the LM's hidden size.
    out: c_int,
    /// Layers whose output feeds the LM's first three blocks (DeepStack).
    deepstack: [3]usize,
    /// Weight-name prefix, INCLUDING the trailing dot's parent
    /// ("model.visual" for MageFlow's repack, "visual" for H3's).
    prefix: []const u8,

    pub fn headDim(self: VitConfig) c_int {
        return @divExact(self.hidden, self.heads);
    }
    /// Rotary inv_freq length: (head_dim/2)/2, because the 2-D rotary lays out
    /// [h, w, h, w] over the head.
    pub fn rotHalf(self: VitConfig) usize {
        return @intCast(@divExact(self.headDim(), 4));
    }
    pub fn mergeHid(self: VitConfig) c_int {
        return self.hidden * VIT_MERGE * VIT_MERGE;
    }
};

/// Mage-Flow's Qwen3-VL-4B-shaped tower.
pub const MAGEFLOW_VIT = VitConfig{
    .hidden = 1024,
    .heads = 16,
    .inter = 4096,
    .depth = 24,
    .out = 2560,
    .deepstack = .{ 5, 11, 17 },
    .prefix = "model.visual",
};

/// MiniMax H3's conditioner is Qwen3-VL-32B, whose tower is the 8B/32B shape
/// (hidden 1152, depth 27, DeepStack at 8/16/24) merging to the LM's 5120.
pub const H3_VIT = VitConfig{
    .hidden = 1152,
    .heads = 16,
    .inter = 4304,
    .depth = 27,
    .out = 5120,
    .deepstack = .{ 8, 16, 24 },
    .prefix = "visual",
};

const VitBlockW = struct {
    n1w: mlx.mlx_array,
    n1b: mlx.mlx_array,
    n2w: mlx.mlx_array,
    n2b: mlx.mlx_array,
    qkv_w: MfLinear,
    qkv_b: mlx.mlx_array,
    proj_w: MfLinear,
    proj_b: mlx.mlx_array,
    fc1_w: MfLinear,
    fc1_b: mlx.mlx_array,
    fc2_w: MfLinear,
    fc2_b: mlx.mlx_array,
    fn deinit(self: *VitBlockW) void {
        inline for (.{ &self.qkv_w, &self.proj_w, &self.fc1_w, &self.fc2_w }) |f| f.deinit();
        inline for (.{ self.n1w, self.n1b, self.n2w, self.n2b, self.qkv_b, self.proj_b, self.fc1_b, self.fc2_b }) |f|
            _ = mlx.mlx_array_free(f);
    }
};

const VitMergerW = struct {
    norm_w: mlx.mlx_array,
    norm_b: mlx.mlx_array,
    fc1_w: MfLinear,
    fc1_b: mlx.mlx_array,
    fc2_w: MfLinear,
    fc2_b: mlx.mlx_array,
    fn deinit(self: *VitMergerW) void {
        inline for (.{ &self.fc1_w, &self.fc2_w }) |f| f.deinit();
        inline for (.{ self.norm_w, self.norm_b, self.fc1_b, self.fc2_b }) |f|
            _ = mlx.mlx_array_free(f);
    }
};

fn loadVitBlock(w: *const Weights, a: std.mem.Allocator, cfg: VitConfig, i: usize, dtype: mlx.mlx_dtype, s: S) !VitBlockW {
    const n1 = try std.fmt.allocPrint(a, "{s}.blocks.{d}.norm1", .{ cfg.prefix, i });
    defer a.free(n1);
    const n2 = try std.fmt.allocPrint(a, "{s}.blocks.{d}.norm2", .{ cfg.prefix, i });
    defer a.free(n2);
    const qkv = try std.fmt.allocPrint(a, "{s}.blocks.{d}.attn.qkv", .{ cfg.prefix, i });
    defer a.free(qkv);
    const proj = try std.fmt.allocPrint(a, "{s}.blocks.{d}.attn.proj", .{ cfg.prefix, i });
    defer a.free(proj);
    const fc1 = try std.fmt.allocPrint(a, "{s}.blocks.{d}.mlp.linear_fc1", .{ cfg.prefix, i });
    defer a.free(fc1);
    const fc2 = try std.fmt.allocPrint(a, "{s}.blocks.{d}.mlp.linear_fc2", .{ cfg.prefix, i });
    defer a.free(fc2);
    return .{
        .n1w = try loadVec(w, a, n1, "weight", s),
        .n1b = try loadVec(w, a, n1, "bias", s),
        .n2w = try loadVec(w, a, n2, "weight", s),
        .n2b = try loadVec(w, a, n2, "bias", s),
        .qkv_w = try MfLinear.load(w, a, qkv, @intCast(cfg.hidden), dtype, s),
        .qkv_b = try loadVecDt(w, a, qkv, "bias", dtype, s),
        .proj_w = try MfLinear.load(w, a, proj, @intCast(cfg.hidden), dtype, s),
        .proj_b = try loadVecDt(w, a, proj, "bias", dtype, s),
        .fc1_w = try MfLinear.load(w, a, fc1, @intCast(cfg.hidden), dtype, s),
        .fc1_b = try loadVecDt(w, a, fc1, "bias", dtype, s),
        .fc2_w = try MfLinear.load(w, a, fc2, @intCast(cfg.inter), dtype, s),
        .fc2_b = try loadVecDt(w, a, fc2, "bias", dtype, s),
    };
}

fn loadVitMerger(w: *const Weights, a: std.mem.Allocator, cfg: VitConfig, prefix: []const u8, dtype: mlx.mlx_dtype, s: S) !VitMergerW {
    const nm = try std.fmt.allocPrint(a, "{s}.norm", .{prefix});
    defer a.free(nm);
    const fc1 = try std.fmt.allocPrint(a, "{s}.linear_fc1", .{prefix});
    defer a.free(fc1);
    const fc2 = try std.fmt.allocPrint(a, "{s}.linear_fc2", .{prefix});
    defer a.free(fc2);
    return .{
        .norm_w = try loadVec(w, a, nm, "weight", s),
        .norm_b = try loadVec(w, a, nm, "bias", s),
        // Both merger linears consume the 2×2-shuffled patch block: hidden×4.
        .fc1_w = try MfLinear.load(w, a, fc1, @intCast(cfg.mergeHid()), dtype, s),
        .fc1_b = try loadVecDt(w, a, fc1, "bias", dtype, s),
        .fc2_w = try MfLinear.load(w, a, fc2, @intCast(cfg.mergeHid()), dtype, s),
        .fc2_b = try loadVecDt(w, a, fc2, "bias", dtype, s),
    };
}

/// The Conv3d patch embed as a pre-transposed linear [1536, hidden]. The raw
/// `visual.patch_embed.proj.weight` is OITHW [hidden,3,2,16,16]; transpose to
/// OHWI-3d [hidden,2,16,16,3] (T,H,W,I order — matches the reference reshape of
/// the pixel patch), flatten kernel dims, then transpose to [1536, hidden].
fn loadVitPatchEmbed(w: *const Weights, a: std.mem.Allocator, cfg: VitConfig, dtype: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const key = try std.fmt.allocPrint(a, "{s}.patch_embed.proj.weight", .{cfg.prefix});
    defer a.free(key);
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 4, 1 }, s); // [H,2,16,16,3]
    defer _ = mlx.mlx_array_free(t);
    const flat = try reshape(t, &[_]c_int{ cfg.hidden, VIT_PATCH_IN }, s);
    defer _ = mlx.mlx_array_free(flat);
    const tt = try transpose(flat, &[_]c_int{ 1, 0 }, s); // [1536, hidden]
    defer _ = mlx.mlx_array_free(tt);
    const tc = try contig(tt, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, dtype, s);
}

pub const VisionTower = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    cfg: VitConfig,
    patch_w: mlx.mlx_array, // [1536, hidden]
    patch_b: mlx.mlx_array, // [hidden]
    pos_embed: mlx.mlx_array, // [2304, hidden] embedding table
    blocks: []VitBlockW,
    merger: VitMergerW, // use_postshuffle_norm=false
    deepstack: [3]VitMergerW, // use_postshuffle_norm=true, at cfg.deepstack

    /// Mage-Flow's layout: the tower lives in the `text_encoder/` component dir.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !VisionTower {
        const dir = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        return loadFrom(allocator, s, &w, MAGEFLOW_VIT, dtype);
    }

    /// Load from an ALREADY-OPEN weight map. H3 keeps its tower in the same
    /// single `text_encoder.safetensors` as the LM, and re-opening a 28 GB file
    /// to read 529 tensors out of it would double the staged-residency peak the
    /// whole backend is built around.
    pub fn loadFrom(allocator: std.mem.Allocator, s: S, w: *const Weights, cfg: VitConfig, dtype: mlx.mlx_dtype) !VisionTower {
        const a = allocator;
        var self: VisionTower = undefined;
        self.allocator = allocator;
        self.s = s;
        self.dtype = dtype;
        self.cfg = cfg;

        self.patch_w = try loadVitPatchEmbed(w, a, cfg, dtype, s);
        const pe_pfx = try std.fmt.allocPrint(a, "{s}.patch_embed.proj", .{cfg.prefix});
        defer a.free(pe_pfx);
        self.patch_b = try loadVecDt(w, a, pe_pfx, "bias", dtype, s);
        const pos_key = try std.fmt.allocPrint(a, "{s}.pos_embed.weight", .{cfg.prefix});
        defer a.free(pos_key);
        const pe = try ownWeight(w, pos_key);
        defer _ = mlx.mlx_array_free(pe);
        self.pos_embed = try astype(pe, dtype, s);
        self.blocks = try a.alloc(VitBlockW, cfg.depth);
        for (0..cfg.depth) |i| self.blocks[i] = try loadVitBlock(w, a, cfg, i, dtype, s);
        const mg = try std.fmt.allocPrint(a, "{s}.merger", .{cfg.prefix});
        defer a.free(mg);
        self.merger = try loadVitMerger(w, a, cfg, mg, dtype, s);
        for (0..3) |i| {
            const pfx = try std.fmt.allocPrint(a, "{s}.deepstack_merger_list.{d}", .{ cfg.prefix, i });
            defer a.free(pfx);
            self.deepstack[i] = try loadVitMerger(w, a, cfg, pfx, dtype, s);
        }
        return self;
    }

    pub fn deinit(self: *VisionTower) void {
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.patch_b);
        _ = mlx.mlx_array_free(self.pos_embed);
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.merger.deinit();
        for (&self.deepstack) |*m| m.deinit();
    }

    /// Run the tower over `pixel_values` [Npatch, 1536] with per-image grids
    /// (t, gh, gw). Returns merged features [Ntok, 2560] and the 3 DeepStack
    /// feature sets (each [Ntok, 2560]). Caller frees all four arrays.
    pub fn forward(self: *const VisionTower, pixel_values: mlx.mlx_array, grids: []const [3]i64) !struct { merged: mlx.mlx_array, deepstack: [3]mlx.mlx_array } {
        const s = self.s;
        const a = self.allocator;

        // 1. Patch embed: reorder [Np,I,T,H,W]→[Np,T,H,W,I], flatten, linear.
        const pv = try astype(pixel_values, self.dtype, s);
        defer _ = mlx.mlx_array_free(pv);
        const np = mlx.getShape(pv)[0];
        const pv5 = try reshape(pv, &[_]c_int{ np, 3, 2, 16, 16 }, s);
        defer _ = mlx.mlx_array_free(pv5);
        const pvt = try transpose(pv5, &[_]c_int{ 0, 2, 3, 4, 1 }, s); // [Np,T,H,W,I]
        defer _ = mlx.mlx_array_free(pvt);
        const pvf = try reshape(pvt, &[_]c_int{ np, VIT_PATCH_IN }, s);
        defer _ = mlx.mlx_array_free(pvf);
        const cfg = self.cfg;
        var hidden = try linearT(pvf, self.patch_w, self.patch_b, s); // [Np, hidden]

        // 2. Interpolated pos-embed (merge order) + add.
        const pos = try self.buildPosEmbeds(grids);
        {
            const summed = try addA(hidden, pos, s);
            _ = mlx.mlx_array_free(hidden);
            _ = mlx.mlx_array_free(pos);
            hidden = summed;
        }

        // 3. 2D rotary cos/sin [Np, head_dim] (f32).
        const rope = try buildVisionRope(a, cfg, grids, s);
        defer _ = mlx.mlx_array_free(rope.cos);
        defer _ = mlx.mlx_array_free(rope.sin);

        // 4. cu_seqlens: one segment per (image × frame).
        var segs: std.ArrayList(c_int) = .empty;
        defer segs.deinit(a);
        try segs.append(a, 0);
        for (grids) |g| {
            const frame_len: c_int = @intCast(g[1] * g[2]);
            var f: usize = 0;
            while (f < @as(usize, @intCast(g[0]))) : (f += 1)
                try segs.append(a, segs.items[segs.items.len - 1] + frame_len);
        }

        // 5. Blocks (+ DeepStack taps).
        var deepstack_out: [3]mlx.mlx_array = undefined;
        for (self.blocks, 0..) |*blk, i| {
            const nxt = try vitBlockForward(blk, cfg, hidden, rope.cos, rope.sin, segs.items, s);
            _ = mlx.mlx_array_free(hidden);
            hidden = nxt;
            for (cfg.deepstack, 0..) |di, k| {
                if (i == di) deepstack_out[k] = try mergerForward(&self.deepstack[k], cfg, hidden, true, s);
            }
        }
        defer _ = mlx.mlx_array_free(hidden);

        // 6. Merger.
        const merged = try mergerForward(&self.merger, cfg, hidden, false, s);
        return .{ .merged = merged, .deepstack = deepstack_out };
    }

    /// Bilinear-interpolated learned position embeddings in spatial-merge order.
    fn buildPosEmbeds(self: *const VisionTower, grids: []const [3]i64) !mlx.mlx_array {
        const s = self.s;
        const a = self.allocator;
        const cfg = self.cfg;
        var outputs: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (outputs.items) |o| _ = mlx.mlx_array_free(o);
            outputs.deinit(a);
        }
        for (grids) |g| {
            const t: usize = @intCast(g[0]);
            const gh: usize = @intCast(g[1]);
            const gw: usize = @intCast(g[2]);
            const n = gh * gw;
            const idx = try a.alloc(i32, 4 * n);
            defer a.free(idx);
            const wt = try a.alloc(f32, 4 * n);
            defer a.free(wt);
            for (0..gh) |i| {
                const hidx: f64 = if (gh == 1) 0 else @as(f64, @floatFromInt(i)) * @as(f64, @floatFromInt(VIT_GRID_SIDE - 1)) / @as(f64, @floatFromInt(gh - 1));
                const hf = @floor(hidx);
                const hfi: i64 = @intFromFloat(hf);
                const hci: i64 = @min(hfi + 1, @as(i64, @intCast(VIT_GRID_SIDE - 1)));
                const dh = hidx - hf;
                for (0..gw) |j| {
                    const widx: f64 = if (gw == 1) 0 else @as(f64, @floatFromInt(j)) * @as(f64, @floatFromInt(VIT_GRID_SIDE - 1)) / @as(f64, @floatFromInt(gw - 1));
                    const wf = @floor(widx);
                    const wfi: i64 = @intFromFloat(wf);
                    const wci: i64 = @min(wfi + 1, @as(i64, @intCast(VIT_GRID_SIDE - 1)));
                    const dw = widx - wf;
                    const nn_ = i * gw + j;
                    const bh = hfi * @as(i64, @intCast(VIT_GRID_SIDE));
                    const bhc = hci * @as(i64, @intCast(VIT_GRID_SIDE));
                    idx[0 * n + nn_] = @intCast(bh + wfi);
                    idx[1 * n + nn_] = @intCast(bh + wci);
                    idx[2 * n + nn_] = @intCast(bhc + wfi);
                    idx[3 * n + nn_] = @intCast(bhc + wci);
                    wt[0 * n + nn_] = @floatCast((1 - dh) * (1 - dw));
                    wt[1 * n + nn_] = @floatCast((1 - dh) * dw);
                    wt[2 * n + nn_] = @floatCast(dh * (1 - dw));
                    wt[3 * n + nn_] = @floatCast(dh * dw);
                }
            }
            const idx_shape = [_]c_int{@intCast(4 * n)};
            const idx_arr = mlx.mlx_array_new_data(idx.ptr, &idx_shape, 1, .int32);
            defer _ = mlx.mlx_array_free(idx_arr);
            var gathered = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gathered);
            try mlx.check(mlx.mlx_take_axis(&gathered, self.pos_embed, idx_arr, 0, s)); // [4n, hidden]
            const g3 = try reshape(gathered, &[_]c_int{ 4, @intCast(n), cfg.hidden }, s);
            defer _ = mlx.mlx_array_free(g3);
            const wt_shape = [_]c_int{ 4, @intCast(n), 1 };
            const wt_raw = mlx.mlx_array_new_data(wt.ptr, &wt_shape, 3, .float32);
            defer _ = mlx.mlx_array_free(wt_raw);
            const wt_dt = try astype(wt_raw, self.dtype, s);
            defer _ = mlx.mlx_array_free(wt_dt);
            const weighted = try mulA(g3, wt_dt, s);
            defer _ = mlx.mlx_array_free(weighted);
            var interp = mlx.mlx_array_new(); // [n, hidden]
            defer _ = mlx.mlx_array_free(interp);
            try mlx.check(mlx.mlx_sum_axis(&interp, weighted, 0, false, s));
            // tile t frames, reshape to merge order.
            var tiled = interp;
            var free_tiled = false;
            if (t > 1) {
                tiled = try tileFrames(interp, t, s);
                free_tiled = true;
            }
            defer if (free_tiled) {
                _ = mlx.mlx_array_free(tiled);
            };
            const r = try reshape(tiled, &[_]c_int{ @intCast(t), @intCast(gh / 2), VIT_MERGE, @intCast(gw / 2), VIT_MERGE, cfg.hidden }, s);
            defer _ = mlx.mlx_array_free(r);
            const rt = try transpose(r, &[_]c_int{ 0, 1, 3, 2, 4, 5 }, s);
            defer _ = mlx.mlx_array_free(rt);
            const merged_order = try reshape(rt, &[_]c_int{ @intCast(t * n), cfg.hidden }, s);
            try outputs.append(a, merged_order);
        }
        if (outputs.items.len == 1) {
            const out = outputs.items[0];
            outputs.items.len = 0; // transfer ownership
            return out;
        }
        return concat(outputs.items, 0, s);
    }
};

/// Tile [n, C] along a new frame axis → [t*n, C] (repeat the whole block t times).
fn tileFrames(x: mlx.mlx_array, t: usize, s: S) !mlx.mlx_array {
    var arrs = try std.heap.page_allocator.alloc(mlx.mlx_array, t);
    defer std.heap.page_allocator.free(arrs);
    for (0..t) |i| arrs[i] = x;
    return concat(arrs, 0, s);
}

/// Vision 2D rotary cos/sin [Ntok, head_dim] (f32), built in spatial-merge order.
/// Per token (h,w): angles interleave [h·f, w·f, h·f, w·f] over the 16-freq table.
fn buildVisionRope(a: std.mem.Allocator, cfg: VitConfig, grids: []const [3]i64, s: S) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
    var ntok: usize = 0;
    for (grids) |g| ntok += @as(usize, @intCast(g[0])) * @as(usize, @intCast(g[1])) * @as(usize, @intCast(g[2]));
    const hd: usize = @intCast(cfg.headDim());
    const rot_half = cfg.rotHalf();
    const cosb = try a.alloc(f32, ntok * hd);
    defer a.free(cosb);
    const sinb = try a.alloc(f32, ntok * hd);
    defer a.free(sinb);
    // head_dim/4 frequencies; the 2-D rotary lays them out [h, w, h, w].
    var inv_freq_buf: [64]f64 = undefined;
    const inv_freq = inv_freq_buf[0..rot_half];
    const rot_dim: f64 = @floatFromInt(rot_half * 2);
    for (0..rot_half) |j| inv_freq[j] = 1.0 / std.math.pow(f64, VIT_THETA, @as(f64, @floatFromInt(2 * j)) / rot_dim);
    var tok: usize = 0;
    for (grids) |g| {
        const t: usize = @intCast(g[0]);
        const gh: usize = @intCast(g[1]);
        const gw: usize = @intCast(g[2]);
        const m: usize = 2;
        var frame: usize = 0;
        while (frame < t) : (frame += 1) {
            var ai: usize = 0;
            while (ai < gh / m) : (ai += 1) {
                var bi: usize = 0;
                while (bi < gw / m) : (bi += 1) {
                    var ci: usize = 0;
                    while (ci < m) : (ci += 1) {
                        var d: usize = 0;
                        while (d < m) : (d += 1) {
                            const hpos: f64 = @floatFromInt(ai * m + ci);
                            const wpos: f64 = @floatFromInt(bi * m + d);
                            const base = tok * hd;
                            for (0..rot_half) |j| {
                                const ah = hpos * inv_freq[j];
                                const aw = wpos * inv_freq[j];
                                cosb[base + j] = @floatCast(@cos(ah));
                                cosb[base + rot_half + j] = @floatCast(@cos(aw));
                                cosb[base + 2 * rot_half + j] = @floatCast(@cos(ah));
                                cosb[base + 3 * rot_half + j] = @floatCast(@cos(aw));
                                sinb[base + j] = @floatCast(@sin(ah));
                                sinb[base + rot_half + j] = @floatCast(@sin(aw));
                                sinb[base + 2 * rot_half + j] = @floatCast(@sin(ah));
                                sinb[base + 3 * rot_half + j] = @floatCast(@sin(aw));
                            }
                            tok += 1;
                        }
                    }
                }
            }
        }
    }
    const sh = [_]c_int{ @intCast(ntok), @intCast(hd) };
    const cf = mlx.mlx_array_new_data(cosb.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const sf = mlx.mlx_array_new_data(sinb.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(sf);
    return .{ .cos = try contig(cf, s), .sin = try contig(sf, s) };
}

/// rotate-half rope on x [Ntok, heads, head_dim] with cos/sin [Ntok, head_dim]
/// (f32), computed in f32 then cast back to x's dtype (HF vision rule).
fn applyVisionRope(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const in_dt = mlx.mlx_array_dtype(x);
    const sh = mlx.getShape(x); // [Ntok, heads, hd]
    const ntok = sh[0];
    const hd = sh[2];
    const half = @divExact(hd, 2);
    const xf = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(xf);
    const cos_b = try reshape(cos, &[_]c_int{ ntok, 1, hd }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ ntok, 1, hd }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const xc = try mulA(xf, cos_b, s);
    defer _ = mlx.mlx_array_free(xc);
    const x1 = try sliceLast3(xf, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceLast3(xf, half, hd, s);
    defer _ = mlx.mlx_array_free(x2);
    const neg1 = scalarF(-1.0);
    defer _ = mlx.mlx_array_free(neg1);
    const nx2 = try mulA(x2, neg1, s);
    defer _ = mlx.mlx_array_free(nx2);
    const rh = try concat(&.{ nx2, x1 }, 2, s);
    defer _ = mlx.mlx_array_free(rh);
    const rs = try mulA(rh, sin_b, s);
    defer _ = mlx.mlx_array_free(rs);
    const out = try addA(xc, rs, s);
    if (in_dt == .float32) return out;
    defer _ = mlx.mlx_array_free(out);
    return astype(out, in_dt, s);
}

/// Slice [start,end) on the last axis of a 3-D array.
fn sliceLast3(x: mlx.mlx_array, start: c_int, end: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const lo = [_]c_int{ 0, 0, start };
    const hi = [_]c_int{ sh[0], sh[1], end };
    const st = [_]c_int{ 1, 1, 1 };
    return sliceContig(x, &lo, &hi, &st, s);
}

/// One ViT block: h += attn(norm1(h)); h += mlp(norm2(h)) (gelu-tanh MLP).
fn vitBlockForward(bw: *const VitBlockW, cfg: VitConfig, hidden: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, segs: []const c_int, s: S) !mlx.mlx_array {
    const n1 = try layerNormLast(hidden, bw.n1w, bw.n1b, 1e-6, s);
    defer _ = mlx.mlx_array_free(n1);
    const attn = try visionAttn(bw, cfg, n1, cos, sin, segs, s);
    defer _ = mlx.mlx_array_free(attn);
    const h1 = try addA(hidden, attn, s);
    defer _ = mlx.mlx_array_free(h1);
    const n2 = try layerNormLast(h1, bw.n2w, bw.n2b, 1e-6, s);
    defer _ = mlx.mlx_array_free(n2);
    const fc1 = try bw.fc1_w.forward(n2, bw.fc1_b, s);
    defer _ = mlx.mlx_array_free(fc1);
    const g = try geluTanh(fc1, s);
    defer _ = mlx.mlx_array_free(g);
    const fc2 = try bw.fc2_w.forward(g, bw.fc2_b, s);
    defer _ = mlx.mlx_array_free(fc2);
    return addA(h1, fc2, s);
}

/// Full self-attention per cu_seqlens segment, 2D rotary on q/k.
fn visionAttn(bw: *const VitBlockW, cfg: VitConfig, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, segs: []const c_int, s: S) !mlx.mlx_array {
    const np = mlx.getShape(x)[0];
    const qkv = try bw.qkv_w.forward(x, bw.qkv_b, s); // [Np, 3*hidden]
    defer _ = mlx.mlx_array_free(qkv);
    const qkv5 = try reshape(qkv, &[_]c_int{ np, 3, cfg.heads, cfg.headDim() }, s);
    defer _ = mlx.mlx_array_free(qkv5);
    var parts: [3]mlx.mlx_array = undefined;
    try splitEqual(qkv5, 3, 1, &parts, s); // each [Np,1,heads,hd]
    var q = try squeezeDim1(parts[0], s);
    var k = try squeezeDim1(parts[1], s);
    const v = try squeezeDim1(parts[2], s);
    for (parts) |p| _ = mlx.mlx_array_free(p);
    defer _ = mlx.mlx_array_free(v);
    {
        const qr = try applyVisionRope(q, cos, sin, s);
        _ = mlx.mlx_array_free(q);
        q = qr;
        const kr = try applyVisionRope(k, cos, sin, s);
        _ = mlx.mlx_array_free(k);
        k = kr;
    }
    defer _ = mlx.mlx_array_free(q);
    defer _ = mlx.mlx_array_free(k);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.headDim())));
    const a = std.heap.page_allocator;
    var outs: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (outs.items) |o| _ = mlx.mlx_array_free(o);
        outs.deinit(a);
    }
    for (0..segs.len - 1) |i| {
        const start = segs[i];
        const end = segs[i + 1];
        const o = try visionSegAttn(q, k, v, cfg, start, end, scale, s);
        try outs.append(a, o);
    }
    const cat = if (outs.items.len == 1) try contig(outs.items[0], s) else try concat(outs.items, 0, s);
    defer _ = mlx.mlx_array_free(cat);
    const flat = try reshape(cat, &[_]c_int{ np, cfg.hidden }, s);
    defer _ = mlx.mlx_array_free(flat);
    return bw.proj_w.forward(flat, bw.proj_b, s);
}

/// SDPA over one segment q/k/v [Np,heads,hd] sliced to [start,end) → [seg,heads,hd].
fn visionSegAttn(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, cfg: VitConfig, start: c_int, end: c_int, scale: f32, s: S) !mlx.mlx_array {
    const qs = try sliceSeq3(q, start, end, s);
    defer _ = mlx.mlx_array_free(qs);
    const ks = try sliceSeq3(k, start, end, s);
    defer _ = mlx.mlx_array_free(ks);
    const vs = try sliceSeq3(v, start, end, s);
    defer _ = mlx.mlx_array_free(vs);
    const seg = end - start;
    // [seg,heads,hd] → [1,heads,seg,hd]
    const qt = try transpose(qs, &[_]c_int{ 1, 0, 2 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const kt = try transpose(ks, &[_]c_int{ 1, 0, 2 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const vt = try transpose(vs, &[_]c_int{ 1, 0, 2 }, s);
    defer _ = mlx.mlx_array_free(vt);
    const q4 = try reshape(qt, &[_]c_int{ 1, cfg.heads, seg, cfg.headDim() }, s);
    defer _ = mlx.mlx_array_free(q4);
    const k4 = try reshape(kt, &[_]c_int{ 1, cfg.heads, seg, cfg.headDim() }, s);
    defer _ = mlx.mlx_array_free(k4);
    const v4 = try reshape(vt, &[_]c_int{ 1, cfg.heads, seg, cfg.headDim() }, s);
    defer _ = mlx.mlx_array_free(v4);
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, q4, k4, v4, scale, "", null_a, null_a, s));
    // [1,heads,seg,hd] → [seg,heads,hd]
    const a3 = try reshape(attn, &[_]c_int{ cfg.heads, seg, cfg.headDim() }, s);
    defer _ = mlx.mlx_array_free(a3);
    return transpose(a3, &[_]c_int{ 1, 0, 2 }, s);
}

/// Slice the first axis of a 3-D array [start,end).
fn sliceSeq3(x: mlx.mlx_array, start: c_int, end: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const lo = [_]c_int{ start, 0, 0 };
    const hi = [_]c_int{ end, sh[1], sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    return sliceContig(x, &lo, &hi, &st, s);
}

/// Patch merger: (optional post-shuffle) LayerNorm → reshape 4-token group →
/// linear_fc1 → gelu-tanh → linear_fc2. Input [N, hidden] → [N/4, 2560].
fn mergerForward(mw: *const VitMergerW, cfg: VitConfig, hidden: mlx.mlx_array, postshuffle: bool, s: S) !mlx.mlx_array {
    const n = mlx.getShape(hidden)[0];
    const grouped = @divExact(n, VIT_MERGE * VIT_MERGE);
    var x: mlx.mlx_array = undefined;
    if (postshuffle) {
        const r = try reshape(hidden, &[_]c_int{ grouped, cfg.mergeHid() }, s);
        defer _ = mlx.mlx_array_free(r);
        x = try layerNormLast(r, mw.norm_w, mw.norm_b, 1e-6, s);
    } else {
        const nrm = try layerNormLast(hidden, mw.norm_w, mw.norm_b, 1e-6, s);
        defer _ = mlx.mlx_array_free(nrm);
        x = try reshape(nrm, &[_]c_int{ grouped, cfg.mergeHid() }, s);
    }
    defer _ = mlx.mlx_array_free(x);
    const fc1 = try mw.fc1_w.forward(x, mw.fc1_b, s);
    defer _ = mlx.mlx_array_free(fc1);
    const g = try geluTanh(fc1, s);
    defer _ = mlx.mlx_array_free(g);
    return mw.fc2_w.forward(g, mw.fc2_b, s);
}

// ══════════════════════════════════════════════════════════════════════════
// MageFlow text encoder — Qwen3-VL language backbone as a conditioner. Ported
// from mflux `mage_flow_text_encoder/*`. 36 GQA decoder layers (32 Q / 8 KV
// heads, head_dim 128), SwiGLU MLP, RMSNorm q/k + per-layer norms, standard
// rotate-half RoPE (θ=5e6), then a FINAL RMSNorm — the single hidden-state tap
// used as the DiT context (`[1, seq, 2560]`). NOT quantized (bf16 checkpoint).
//
// For text→image the Qwen3-VL M-RoPE (mrope_section [24,20,20]) collapses: all
// three axes share the sequential position, so it reduces to 1D RoPE. Vision +
// true 3-axis M-RoPE (edit) land in a later phase. Compute dtype is a load-time
// parameter — bf16 for the live engine, f32 for the parity fixtures; norm
// weights stay f32 either way (mlx_fast_rms_norm upcasts activations), matching
// the reference's Qwen3RMSNorm(weight.astype(f32)) and closing the late-layer
// outlier divergence a bf16 norm weight introduces.
// ══════════════════════════════════════════════════════════════════════════

const TE_HEADS: c_int = 32;
const TE_KV: c_int = 8;
const TE_HEAD_DIM: c_int = 128;
const TE_HIDDEN: c_int = 2560;
const TE_INTER: c_int = 9728;
const TE_LAYERS = 36;
const TE_THETA: f64 = 5_000_000.0;
const TE_EPS: f32 = 1e-6;
/// Template tokens dropped from the front of the conditioning sequence
/// (`MageFlowPromptProcessor.TEXT_TO_IMAGE_DROP_TOKENS`).
pub const TE_DROP_TOKENS: usize = 34;
/// Max conditioning tokens kept after the drop (`MAX_CONDITION_TOKENS`).
pub const TE_MAX_COND: usize = 2048;
/// Template tokens dropped from the front of an EDIT conditioning sequence
/// (`MageFlowPromptProcessor.EDIT_DROP_TOKENS` — 64, not the txt2img 34).
pub const EDIT_DROP_TOKENS: usize = 64;
/// Qwen3-VL image placeholder token id (`<|image_pad|>`).
const TE_IMAGE_TOKEN: i32 = 151655;

// Prompt template (`MageFlowPromptProcessor.TEXT_TO_IMAGE_TEMPLATE`); the whole
// formatted string is tokenized in one pass (matches the reference).
const TE_PREFIX =
    "<|im_start|>system\n" ++
    "Describe the image by detailing the color, shape, size, texture, quantity, " ++
    "text, spatial relationships of the objects and background:" ++
    "<|im_end|>\n<|im_start|>user\n";
const TE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";


const TeLayerW = struct {
    input_ln: mlx.mlx_array, // f32
    post_ln: mlx.mlx_array, // f32
    qw: MfLinear,
    kw: MfLinear,
    vw: MfLinear,
    ow: MfLinear,
    q_norm: mlx.mlx_array, // f32
    k_norm: mlx.mlx_array, // f32
    gate_w: MfLinear,
    up_w: MfLinear,
    down_w: MfLinear,
    fn deinit(self: *TeLayerW) void {
        inline for (.{ &self.qw, &self.kw, &self.vw, &self.ow, &self.gate_w, &self.up_w, &self.down_w }) |f| f.deinit();
        inline for (.{ self.input_ln, self.post_ln, self.q_norm, self.k_norm }) |f|
            _ = mlx.mlx_array_free(f);
    }
};

pub const TextEncoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    dtype: mlx.mlx_dtype,
    embed_table: mlx.mlx_array, // [vocab, hidden] compute dtype
    layers: [TE_LAYERS]TeLayerW,
    final_norm: mlx.mlx_array, // f32

    pub fn load(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !TextEncoder {
        const dir = try std.fmt.allocPrint(allocator, "{s}/text_encoder", .{model_dir});
        defer allocator.free(dir);
        var w = try model_mod.loadWeights(io, allocator, dir);
        defer w.deinit();
        const a = allocator;
        var self: TextEncoder = undefined;
        self.allocator = allocator;
        self.s = s;
        self.dtype = dtype;

        // Raw checkpoint keys carry the full `model.language_model.` prefix (the
        // reference strips `model.` at map time; loadWeights keeps names verbatim).
        const pfx = "model.language_model.";
        const ek = try std.fmt.allocPrint(a, "{s}embed_tokens.weight", .{pfx});
        defer a.free(ek);
        const raw_emb = try ownWeight(&w, ek);
        defer _ = mlx.mlx_array_free(raw_emb);
        self.embed_table = try astype(raw_emb, dtype, s);

        for (&self.layers, 0..) |*layer, i| {
            const p_in = try std.fmt.allocPrint(a, "{s}layers.{d}.input_layernorm", .{ pfx, i });
            defer a.free(p_in);
            const p_post = try std.fmt.allocPrint(a, "{s}layers.{d}.post_attention_layernorm", .{ pfx, i });
            defer a.free(p_post);
            const qn = try std.fmt.allocPrint(a, "{s}layers.{d}.self_attn.q_norm", .{ pfx, i });
            defer a.free(qn);
            const kn = try std.fmt.allocPrint(a, "{s}layers.{d}.self_attn.k_norm", .{ pfx, i });
            defer a.free(kn);
            const qp = try std.fmt.allocPrint(a, "{s}layers.{d}.self_attn.q_proj", .{ pfx, i });
            defer a.free(qp);
            const kp = try std.fmt.allocPrint(a, "{s}layers.{d}.self_attn.k_proj", .{ pfx, i });
            defer a.free(kp);
            const vp = try std.fmt.allocPrint(a, "{s}layers.{d}.self_attn.v_proj", .{ pfx, i });
            defer a.free(vp);
            const op = try std.fmt.allocPrint(a, "{s}layers.{d}.self_attn.o_proj", .{ pfx, i });
            defer a.free(op);
            const gp = try std.fmt.allocPrint(a, "{s}layers.{d}.mlp.gate_proj", .{ pfx, i });
            defer a.free(gp);
            const upp = try std.fmt.allocPrint(a, "{s}layers.{d}.mlp.up_proj", .{ pfx, i });
            defer a.free(upp);
            const dp = try std.fmt.allocPrint(a, "{s}layers.{d}.mlp.down_proj", .{ pfx, i });
            defer a.free(dp);
            layer.* = .{
                .input_ln = try loadVec(&w, a, p_in, "weight", s),
                .post_ln = try loadVec(&w, a, p_post, "weight", s),
                .qw = try MfLinear.load(&w, a, qp, TE_HIDDEN, dtype, s),
                .kw = try MfLinear.load(&w, a, kp, TE_HIDDEN, dtype, s),
                .vw = try MfLinear.load(&w, a, vp, TE_HIDDEN, dtype, s),
                .ow = try MfLinear.load(&w, a, op, TE_HEADS * TE_HEAD_DIM, dtype, s),
                .q_norm = try loadVec(&w, a, qn, "weight", s),
                .k_norm = try loadVec(&w, a, kn, "weight", s),
                .gate_w = try MfLinear.load(&w, a, gp, TE_HIDDEN, dtype, s),
                .up_w = try MfLinear.load(&w, a, upp, TE_HIDDEN, dtype, s),
                .down_w = try MfLinear.load(&w, a, dp, TE_INTER, dtype, s),
            };
        }
        self.final_norm = try loadVec(&w, a, pfx ++ "norm", "weight", s);
        return self;
    }

    pub fn deinit(self: *TextEncoder) void {
        _ = mlx.mlx_array_free(self.embed_table);
        _ = mlx.mlx_array_free(self.final_norm);
        for (&self.layers) |*l| l.deinit();
    }

    /// Encode a templated token sequence → final-norm hidden states [1, L, 2560]
    /// (compute dtype). `ids`/`mask` are the full templated sequence; `mask` gates
    /// padded keys (all-ones for a single unpadded prompt). Positions are the
    /// running index (text-only M-RoPE collapse). Caller owns the result.
    pub fn encode(self: *const TextEncoder, ids: []const i32, mask: []const i32) !mlx.mlx_array {
        const s = self.s;
        const seq: c_int = @intCast(ids.len);

        const id_shape = [_]c_int{seq};
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, self.embed_table, id_arr, 0, s));
        var x = try reshape(taken, &[_]c_int{ 1, seq, TE_HIDDEN }, s);

        const attn_mask = try buildTeMask(self.allocator, mask, seq, self.dtype, s);
        defer _ = mlx.mlx_array_free(attn_mask);
        const rope = try buildTeRope(self.allocator, @intCast(seq), self.dtype, s);
        defer {
            _ = mlx.mlx_array_free(rope.cos);
            _ = mlx.mlx_array_free(rope.sin);
        }

        for (&self.layers) |*layer| {
            const nx = try teLayerForward(layer, x, attn_mask, rope.cos, rope.sin, seq, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        const normed = try rmsNormLast(x, self.final_norm, TE_EPS, s);
        _ = mlx.mlx_array_free(x);
        return normed;
    }

    /// Text→image conditioning: encode, then drop the first `TE_DROP_TOKENS`
    /// active tokens and keep ≤`TE_MAX_COND` (`process_text_to_image_hidden_states`
    /// for one unpadded prompt — the padding step is a no-op at batch 1). Returns
    /// (embeddings [1, n, 2560], keep count n). Caller frees the array.
    pub fn encodeTextToImage(self: *const TextEncoder, ids: []const i32, mask: []const i32) !struct { embeddings: mlx.mlx_array, keep: c_int } {
        const s = self.s;
        const hidden = try self.encode(ids, mask);
        defer _ = mlx.mlx_array_free(hidden);
        const seq: c_int = @intCast(ids.len);
        const drop: c_int = @intCast(TE_DROP_TOKENS);
        const start = @min(drop, seq);
        const end = @min(seq, start + @as(c_int, @intCast(TE_MAX_COND)));
        const emb = try sliceTeSeq(hidden, start, end, s);
        return .{ .embeddings = emb, .keep = end - start };
    }

    /// Multi-reference EDIT conditioning (E7.4). `ids`/`mask` are the templated
    /// edit prompt with `<|image_pad|>` runs; `pixel_values`/`grids` are the ViT
    /// inputs for the reference images. Embeds tokens, runs the vision tower,
    /// REPLACES placeholder embeddings with the merged vision features, runs the
    /// 36-layer LM scatter-ADDing DeepStack features at layers 0/1/2, final-norms,
    /// then drops the first `EDIT_DROP_TOKENS`. Returns (embeddings [1,n,2560], n).
    /// The LM RoPE is sequential (1D) — encode_edit passes explicit positions, so
    /// no 3-axis M-RoPE. Caller frees the array.
    pub fn encodeEdit(self: *const TextEncoder, vit: *const VisionTower, ids: []const i32, mask: []const i32, pixel_values: mlx.mlx_array, grids: []const [3]i64) !struct { embeddings: mlx.mlx_array, keep: c_int } {
        const s = self.s;
        const seq: c_int = @intCast(ids.len);

        // Token embeddings [seq, 2560].
        const id_shape = [_]c_int{seq};
        const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_axis(&taken, self.embed_table, id_arr, 0, s));

        // Vision tower → merged + DeepStack features.
        const vout = try vit.forward(pixel_values, grids);
        defer _ = mlx.mlx_array_free(vout.merged);
        defer for (vout.deepstack) |d| {
            _ = mlx.mlx_array_free(d);
        };

        // Visual-token positions (host scan of the placeholder run).
        var poslist: std.ArrayList(i32) = .empty;
        defer poslist.deinit(self.allocator);
        for (ids, 0..) |id, i| if (id == TE_IMAGE_TOKEN) try poslist.append(self.allocator, @intCast(i));
        const nvis: c_int = @intCast(poslist.items.len);
        // The placeholder run and the merged vision features are produced by two
        // independent paths (prompt templating vs the ViT grid). A mismatch is a
        // preprocessing bug, and `put_along_axis` would die on the shape inside
        // mlx (an uncatchable kill) — fail honestly instead.
        if (nvis != mlx.getShape(vout.merged)[0]) {
            log.err("[mageflow] edit: {d} <|image_pad|> tokens but {d} vision features\n", .{ nvis, mlx.getShape(vout.merged)[0] });
            return error.MageFlowVisionTokenMismatch;
        }
        const pos_shape = [_]c_int{nvis};
        const pos_arr = mlx.mlx_array_new_data(poslist.items.ptr, &pos_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(pos_arr);

        // Replace placeholder embeddings with the merged vision features.
        const merged_dt = try astype(vout.merged, self.dtype, s);
        defer _ = mlx.mlx_array_free(merged_dt);
        const replaced = try scatterRows(taken, pos_arr, merged_dt, false, s); // [seq,2560]
        defer _ = mlx.mlx_array_free(replaced);
        var x = try reshape(replaced, &[_]c_int{ 1, seq, TE_HIDDEN }, s);

        const attn_mask = try buildTeMask(self.allocator, mask, seq, self.dtype, s);
        defer _ = mlx.mlx_array_free(attn_mask);
        const rope = try buildTeRope(self.allocator, @intCast(seq), self.dtype, s);
        defer {
            _ = mlx.mlx_array_free(rope.cos);
            _ = mlx.mlx_array_free(rope.sin);
        }

        for (&self.layers, 0..) |*layer, i| {
            const nx = try teLayerForward(layer, x, attn_mask, rope.cos, rope.sin, seq, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
            if (i < 3) { // DeepStack scatter-add at LM layers 0/1/2.
                const ds_dt = try astype(vout.deepstack[i], self.dtype, s);
                defer _ = mlx.mlx_array_free(ds_dt);
                const xf = try reshape(x, &[_]c_int{ seq, TE_HIDDEN }, s);
                defer _ = mlx.mlx_array_free(xf);
                const scat = try scatterRows(xf, pos_arr, ds_dt, true, s);
                defer _ = mlx.mlx_array_free(scat);
                const nx2 = try reshape(scat, &[_]c_int{ 1, seq, TE_HIDDEN }, s);
                _ = mlx.mlx_array_free(x);
                x = nx2;
            }
        }
        const normed = try rmsNormLast(x, self.final_norm, TE_EPS, s);
        _ = mlx.mlx_array_free(x);
        defer _ = mlx.mlx_array_free(normed);
        const drop: c_int = @intCast(EDIT_DROP_TOKENS);
        const start = @min(drop, seq);
        const end = @min(seq, start + @as(c_int, @intCast(TE_MAX_COND)));
        const emb = try sliceTeSeq(normed, start, end, s);
        return .{ .embeddings = emb, .keep = end - start };
    }
};

/// Scatter `values` [Nvis, H] into rows `positions` of `flat` [L, H]; when `add`,
/// accumulate onto the existing rows (DeepStack). Returns a new [L, H] (caller
/// frees). Positions must be distinct (they are — a placeholder run has no dups).
fn scatterRows(flat: mlx.mlx_array, positions: mlx.mlx_array, values: mlx.mlx_array, add: bool, s: S) !mlx.mlx_array {
    const H = mlx.getShape(flat)[1];
    const nvis = mlx.getShape(values)[0];
    const idx1 = try reshape(positions, &[_]c_int{ nvis, 1 }, s);
    defer _ = mlx.mlx_array_free(idx1);
    const idx2sh = [_]c_int{ nvis, H };
    var idx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx2);
    try mlx.check(mlx.mlx_broadcast_to(&idx2, idx1, &idx2sh, 2, s));
    var vals = values;
    var free_vals = false;
    if (add) {
        var gathered = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gathered);
        try mlx.check(mlx.mlx_take_axis(&gathered, flat, positions, 0, s)); // [Nvis,H]
        vals = try addA(gathered, values, s);
        free_vals = true;
    }
    defer if (free_vals) {
        _ = mlx.mlx_array_free(vals);
    };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_put_along_axis(&out, flat, idx2, vals, 0, s));
    return out;
}

/// One Qwen3-VL decoder layer: h += attn(input_ln(h)); h += mlp(post_ln(h)).
fn teLayerForward(layer: *const TeLayerW, x: mlx.mlx_array, mask: mlx.mlx_array, rope_cos: mlx.mlx_array, rope_sin: mlx.mlx_array, seq: c_int, s: S) !mlx.mlx_array {
    const xn = try rmsNormLast(x, layer.input_ln, TE_EPS, s);
    defer _ = mlx.mlx_array_free(xn);
    const q = try layer.qw.forward(xn, null, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try layer.kw.forward(xn, null, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try layer.vw.forward(xn, null, s);
    defer _ = mlx.mlx_array_free(v);
    const q4 = try reshape(q, &[_]c_int{ 1, seq, TE_HEADS, TE_HEAD_DIM }, s);
    defer _ = mlx.mlx_array_free(q4);
    const qn = try rmsNormLast(q4, layer.q_norm, TE_EPS, s);
    defer _ = mlx.mlx_array_free(qn);
    const qt = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const k4 = try reshape(k, &[_]c_int{ 1, seq, TE_KV, TE_HEAD_DIM }, s);
    defer _ = mlx.mlx_array_free(k4);
    const kn = try rmsNormLast(k4, layer.k_norm, TE_EPS, s);
    defer _ = mlx.mlx_array_free(kn);
    const kt = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const v4 = try reshape(v, &[_]c_int{ 1, seq, TE_KV, TE_HEAD_DIM }, s);
    defer _ = mlx.mlx_array_free(v4);
    const vt = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(vt);
    const qr = try applyRopeHalf(qt, rope_cos, rope_sin, s);
    defer _ = mlx.mlx_array_free(qr);
    const kr = try applyRopeHalf(kt, rope_cos, rope_sin, s);
    defer _ = mlx.mlx_array_free(kr);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(TE_HEAD_DIM)));
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_sink = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, "array", mask, null_sink, s));
    const at = try transpose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(at);
    const af = try reshape(at, &[_]c_int{ 1, seq, TE_HEADS * TE_HEAD_DIM }, s);
    defer _ = mlx.mlx_array_free(af);
    const o = try layer.ow.forward(af, null, s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(x, o, s);
    defer _ = mlx.mlx_array_free(h1);
    // SwiGLU MLP.
    const hn = try rmsNormLast(h1, layer.post_ln, TE_EPS, s);
    defer _ = mlx.mlx_array_free(hn);
    const g = try layer.gate_w.forward(hn, null, s);
    defer _ = mlx.mlx_array_free(g);
    const sg = try silu(g, s);
    defer _ = mlx.mlx_array_free(sg);
    const u = try layer.up_w.forward(hn, null, s);
    defer _ = mlx.mlx_array_free(u);
    const gu = try mulA(sg, u, s);
    defer _ = mlx.mlx_array_free(gu);
    const dn = try layer.down_w.forward(gu, null, s);
    defer _ = mlx.mlx_array_free(dn);
    return addA(h1, dn, s);
}

/// Causal + padding additive mask [1,1,seq,seq] (0 keep, -1e9 blocked) in the
/// compute dtype. `-1e9` (not -inf) avoids bf16 NaN; proven parity-equivalent.
fn buildTeMask(allocator: std.mem.Allocator, mask: []const i32, seq: c_int, dtype: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    const n: usize = @intCast(seq);
    const buf = try allocator.alloc(f32, n * n);
    defer allocator.free(buf);
    const neg: f32 = -1e9;
    for (0..n) |i| {
        for (0..n) |j| {
            const blocked = (j > i) or (mask[j] == 0);
            buf[i * n + j] = if (blocked) neg else 0.0;
        }
    }
    const shape = [_]c_int{ 1, 1, seq, seq };
    const f = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    return astype(f, dtype, s);
}

/// Standard rotate-half RoPE cos/sin [L, head_dim]: emb = concat([freqs,freqs]),
/// computed in f32 then cast to the compute dtype (the reference casts cos/sin to
/// hidden_states.dtype before applying).
fn buildTeRope(allocator: std.mem.Allocator, L: usize, dtype: mlx.mlx_dtype, s: S) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
    const hd: usize = @intCast(TE_HEAD_DIM);
    const cosb = try allocator.alloc(f32, L * hd);
    defer allocator.free(cosb);
    const sinb = try allocator.alloc(f32, L * hd);
    defer allocator.free(sinb);
    const half = hd / 2;
    for (0..L) |p| {
        for (0..half) |i| {
            const inv = std.math.pow(f64, TE_THETA, -@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd)));
            const ang = @as(f64, @floatFromInt(p)) * inv;
            const cv: f32 = @floatCast(@cos(ang));
            const sv: f32 = @floatCast(@sin(ang));
            cosb[p * hd + i] = cv;
            cosb[p * hd + i + half] = cv;
            sinb[p * hd + i] = sv;
            sinb[p * hd + i + half] = sv;
        }
    }
    const sh = [_]c_int{ @intCast(L), @intCast(hd) };
    const cf = mlx.mlx_array_new_data(cosb.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const sf = mlx.mlx_array_new_data(sinb.ptr, &sh, 2, .float32);
    defer _ = mlx.mlx_array_free(sf);
    return .{ .cos = try astype(cf, dtype, s), .sin = try astype(sf, dtype, s) };
}

/// rotate-half RoPE on x [1,H,L,hd]: x*cos + rotate_half(x)*sin;
/// rotate_half(x) = concat([-x[...,hd/2:], x[...,:hd/2]]).
fn applyRopeHalf(x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,H,L,hd]
    const L = sh[2];
    const hd = sh[3];
    const half = @divExact(hd, 2);
    const cos_b = try reshape(cos, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(cos_b);
    const sin_b = try reshape(sin, &[_]c_int{ 1, 1, L, hd }, s);
    defer _ = mlx.mlx_array_free(sin_b);
    const xc = try mulA(x, cos_b, s);
    defer _ = mlx.mlx_array_free(xc);
    const x1 = try sliceLastAxis(x, 0, half, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try sliceLastAxis(x, half, hd, s);
    defer _ = mlx.mlx_array_free(x2);
    const neg1 = scalarF(-1.0);
    defer _ = mlx.mlx_array_free(neg1);
    const nx2 = try mulA(x2, neg1, s);
    defer _ = mlx.mlx_array_free(nx2);
    const rh = try concat(&.{ nx2, x1 }, 3, s);
    defer _ = mlx.mlx_array_free(rh);
    const rs = try mulA(rh, sin_b, s);
    defer _ = mlx.mlx_array_free(rs);
    return addA(xc, rs, s);
}

/// Slice the last axis of a 4-D array [start, end).
fn sliceLastAxis(x: mlx.mlx_array, start: c_int, end: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,H,L,hd]
    const lo = [_]c_int{ 0, 0, 0, start };
    const hi = [_]c_int{ sh[0], sh[1], sh[2], end };
    const st = [_]c_int{ 1, 1, 1, 1 };
    return sliceContig(x, &lo, &hi, &st, s);
}

/// Slice the sequence axis of hidden states [1, seq, C] → [1, end-start, C].
fn sliceTeSeq(x: mlx.mlx_array, start: c_int, end: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [1,seq,C]
    const lo = [_]c_int{ 0, start, 0 };
    const hi = [_]c_int{ sh[0], end, sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    return sliceContig(x, &lo, &hi, &st, s);
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

// VAE ENCODER parity vs the mflux reference (E7.1). Gated on MAGEFLOW_TEST_MODEL
// + MAGEFLOW_VAE_ENC_FIXTURE (from tests/dump_mageflow_vae_encoder_fixture.py:
// pixels/mean/logvar/latent/packed, f32). The encoder network is fully exercised
// by the deterministic `mean` (no RNG), so that is the hard assertion; `latent`
// is checked for finiteness/shape (the seeded posterior draw is valid either way
// — the live edit feeds the reference's exact ref-latents at E7.5).
test "MageFlow VAE encode parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_VAE_ENC_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var enc = try VaeEncoder.load(io, a, s, model_dir);
    defer enc.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const pixels = fx.get("pixels") orelse return error.MissingFixturePixels;
    const ref_mean = fx.get("mean") orelse return error.MissingFixtureMean;
    const ref_logvar = fx.get("logvar") orelse return error.MissingFixtureLogvar;

    const m = try enc.encodeMoments(pixels);
    defer _ = mlx.mlx_array_free(m.mean);
    defer _ = mlx.mlx_array_free(m.logvar);
    const cos_mean = try cosineSim(m.mean, ref_mean, s);
    const cos_logvar = try cosineSim(m.logvar, ref_logvar, s);
    std.debug.print("[mageflow-vae-enc] mean cosine = {d:.6}  logvar cosine = {d:.6}\n", .{ cos_mean, cos_logvar });
    try testing.expect(cos_mean > 0.999);
    try testing.expect(cos_logvar > 0.999);
}

// VISION TOWER parity vs the mflux reference (E7.2). Gated on MAGEFLOW_TEST_MODEL
// + MAGEFLOW_VIT_FIXTURE (from tests/dump_mageflow_vit_fixture.py:
// pixel_values/grid_thw/merged/deepstack0..2, f32). Feeds the reference's exact
// pixel_values (preprocessing decoupled — E7.3).
test "MageFlow ViT parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_VIT_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var vit = try VisionTower.load(io, a, s, model_dir, .float32);
    defer vit.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const pv = fx.get("pixel_values") orelse return error.MissingFixturePixels;
    const grid = fx.get("grid_thw") orelse return error.MissingFixtureGrid;
    const ref_merged = fx.get("merged") orelse return error.MissingFixtureMerged;

    const gids = try readFixtureIds(a, grid, s);
    defer a.free(gids);
    const nimg = gids.len / 3;
    const grids = try a.alloc([3]i64, nimg);
    defer a.free(grids);
    for (0..nimg) |i| grids[i] = .{ gids[i * 3], gids[i * 3 + 1], gids[i * 3 + 2] };

    const out = try vit.forward(pv, grids);
    defer _ = mlx.mlx_array_free(out.merged);
    defer for (out.deepstack) |d| {
        _ = mlx.mlx_array_free(d);
    };
    const cos_m = try cosineSim(out.merged, ref_merged, s);
    std.debug.print("[mageflow-vit] merged cosine = {d:.6}\n", .{cos_m});
    try testing.expect(cos_m > 0.999);
    inline for (.{ "deepstack0", "deepstack1", "deepstack2" }, 0..) |name, k| {
        const ref_d = fx.get(name) orelse return error.MissingFixtureDeepstack;
        const cos_d = try cosineSim(out.deepstack[k], ref_d, s);
        std.debug.print("[mageflow-vit] {s} cosine = {d:.6}\n", .{ name, cos_d });
        try testing.expect(cos_d > 0.999);
    }
}

// TEXT-ENCODER EDIT-PATH parity vs the mflux reference (E7.4). Gated on
// MAGEFLOW_TEST_MODEL + MAGEFLOW_TE_EDIT_FIXTURE (from
// tests/dump_mageflow_te_edit_fixture.py: input_ids/attention_mask/pixel_values/
// image_grid_thw/embeddings/out_mask, f32). Feeds the reference's exact input_ids
// + pixel_values (tokenization/preprocessing decoupled — E7.3).
test "MageFlow TE edit parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_TE_EDIT_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var te = try TextEncoder.load(io, a, s, model_dir, .float32);
    defer te.deinit();
    var vit = try VisionTower.load(io, a, s, model_dir, .float32);
    defer vit.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureIds;
    const mask_arr = fx.get("attention_mask") orelse return error.MissingFixtureMask;
    const pv = fx.get("pixel_values") orelse return error.MissingFixturePixels;
    const grid = fx.get("image_grid_thw") orelse return error.MissingFixtureGrid;
    const ref_emb = fx.get("embeddings") orelse return error.MissingFixtureEmb;

    const ids = try readFixtureIds(a, ids_arr, s);
    defer a.free(ids);
    const mask = try readFixtureIds(a, mask_arr, s);
    defer a.free(mask);
    const gids = try readFixtureIds(a, grid, s);
    defer a.free(gids);
    const nimg = gids.len / 3;
    const grids = try a.alloc([3]i64, nimg);
    defer a.free(grids);
    for (0..nimg) |i| grids[i] = .{ gids[i * 3], gids[i * 3 + 1], gids[i * 3 + 2] };

    const out = try te.encodeEdit(&vit, ids, mask, pv, grids);
    defer _ = mlx.mlx_array_free(out.embeddings);
    const cos = try cosineSim(out.embeddings, ref_emb, s);
    std.debug.print("[mageflow-te-edit] embeddings cosine = {d:.6}  keep = {d}\n", .{ cos, out.keep });
    try testing.expect(cos > 0.999);
}

// EDIT PROMPT TOKENIZATION parity (E7.3). The Zig-side templating +
// placeholder expansion + tokenization must reproduce the reference processor's
// input_ids EXACTLY, or the placeholder count won't match the vision features.
// Gated on MAGEFLOW_TEST_MODEL + MAGEFLOW_TE_EDIT_FIXTURE (256×256 image → grid
// 16×16 → 64 image tokens; PROMPT pinned to the dump script).
test "MageFlow edit tokenization parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_TE_EDIT_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const te_dir = try std.fmt.allocPrint(a, "{s}/text_encoder", .{model_dir});
    defer a.free(te_dir);
    var tok = try tok_mod.loadTokenizerAny(io, a, te_dir);
    defer tok.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const ref_ids = try readFixtureIds(a, fx.get("input_ids") orelse return error.MissingFixtureIds, s);
    defer a.free(ref_ids);

    // Must match tests/dump_mageflow_te_edit_fixture.py PROMPT + grid (64 tokens).
    const PROMPT = "make it a snowy winter scene at golden hour";
    const pr = try buildEditPromptIds(&tok, a, PROMPT, &.{64});
    defer a.free(pr.ids);
    defer a.free(pr.mask);
    std.debug.print("[mageflow-edit-tok] mine={d} ref={d}\n", .{ pr.ids.len, ref_ids.len });
    try testing.expectEqual(ref_ids.len, pr.ids.len);
    for (ref_ids, pr.ids) |r, m| try testing.expectEqual(r, m);

    // MULTI-reference (composition) uses the same templating with one header +
    // placeholder run per image — pinned rather than extrapolated.
    if (fx.get("input_ids_2img")) |ref2_arr| {
        const ref2 = try readFixtureIds(a, ref2_arr, s);
        defer a.free(ref2);
        const pr2 = try buildEditPromptIds(&tok, a, PROMPT, &.{ 64, 64 });
        defer a.free(pr2.ids);
        defer a.free(pr2.mask);
        std.debug.print("[mageflow-edit-tok] 2-image mine={d} ref={d}\n", .{ pr2.ids.len, ref2.len });
        try testing.expectEqual(ref2.len, pr2.ids.len);
        for (ref2, pr2.ids) |r, m| try testing.expectEqual(r, m);
    }
}

// VLM PREPROCESSING parity (E7.3). Every other edit fixture feeds the
// REFERENCE's `pixel_values` straight in, which deliberately decouples the
// tower from preprocessing — and leaves `decodeRefForVlm` (resize → patchify →
// normalize) as the one link in the edit chain nothing checks. A wrong patch
// order or normalization here is invisible: the tower still runs, the token
// count still matches, the output is just quietly wrong. This re-derives
// pixel_values from the processor's own input image and compares element-wise.
// Gated on MAGEFLOW_TE_EDIT_FIXTURE carrying `source_rgb` (skips on older ones).
test "MageFlow VLM preprocessing parity (env-gated)" {
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_TE_EDIT_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const src = fx.get("source_rgb") orelse return error.SkipZigTest; // pre-`source_rgb` fixture
    const ref_pv = fx.get("pixel_values") orelse return error.MissingFixturePixels;

    // source_rgb is [H,W,3] f32 in 0..255 — back to bytes for the real entry point.
    const src_f = try astype(src, .float32, s);
    defer _ = mlx.mlx_array_free(src_f);
    _ = mlx.mlx_array_eval(src_f);
    const ssh = mlx.getShape(src_f);
    const sh: u32 = @intCast(ssh[0]);
    const sw: u32 = @intCast(ssh[1]);
    const sd = mlx.mlx_array_data_float32(src_f) orelse return error.NoData;
    const rgb = try a.alloc(u8, @as(usize, sh) * sw * 3);
    defer a.free(rgb);
    for (rgb, 0..) |*p, i| p.* = @intFromFloat(std.math.clamp(@round(sd[i]), 0, 255));

    const mine = try vlmPixelValues(a, rgb, sh, sw, VLM_MIN_PIXELS, VLM_MAX_PIXELS, s);
    defer _ = mlx.mlx_array_free(mine.pv);
    const rsh = mlx.getShape(ref_pv);
    try testing.expectEqual(rsh[0], mlx.getShape(mine.pv)[0]);
    try testing.expectEqual(rsh[1], mlx.getShape(mine.pv)[1]);
    // The merged-token count the prompt templating repeats <|image_pad|> for.
    try testing.expectEqual(@as(u32, @intCast(@divExact(rsh[0], 4))), mine.ntok);

    const diff = try subA(mine.pv, ref_pv, s);
    defer _ = mlx.mlx_array_free(diff);
    var absd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(absd);
    try mlx.check(mlx.mlx_abs(&absd, diff, s));
    var mx_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx_v);
    try mlx.check(mlx.mlx_max(&mx_v, absd, false, s));
    _ = mlx.mlx_array_eval(mx_v);
    var v: f32 = 1;
    try mlx.check(mlx.mlx_array_item_float32(&v, mx_v));
    std.debug.print("[mageflow-vlm-prep] grid={d}x{d} ntok={d} max|Δpixel_values| = {d:.8}\n", .{ mine.gh, mine.gw, mine.ntok, v });
    try testing.expect(v < 1e-5);
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

    var dit = try Dit.load(io, a, s, model_dir, .float32);
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

    var dit = try Dit.load(io, a, s, model_dir, .float32);
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

/// Read an int32 mlx array (from a fixture) into an owned []i32 (caller frees).
fn readFixtureIds(a: std.mem.Allocator, arr: mlx.mlx_array, s: S) ![]i32 {
    const c = try astype(arr, .int32, s);
    defer _ = mlx.mlx_array_free(c);
    _ = mlx.mlx_array_eval(c);
    const n = mlx.mlx_array_size(c);
    const d = mlx.mlx_array_data_int32(c) orelse return error.NoData;
    const out = try a.alloc(i32, n);
    @memcpy(out, d[0..n]);
    return out;
}

// End-to-end DiT-loop parity vs the mflux reference (bf16 — the live engine
// path). Feeds the reference's exact bf16 noise + context, runs the scheduler +
// Euler loop, and checks every step's latents track the reference. Gated on
// MAGEFLOW_TEST_MODEL + MAGEFLOW_E2E_BF16_FIXTURE (from tests/dump_mageflow_e2e_fixture.py:
// noise/txt/mask/sigmas/v0/lat1..4/final, native bf16). The lat 16×16 fixture
// pins the un-rounded-timestep washing regression (f32 timestep → cosine 0.55).
test "MageFlow end-to-end DiT-loop parity bf16 (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_E2E_BF16_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    var dit = try Dit.load(io, a, s, model_dir, .bfloat16);
    defer dit.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const noise = fx.get("noise") orelse return error.MissingFixtureNoise;
    const txt = fx.get("txt") orelse return error.MissingFixtureTxt;
    const sig_arr = try astype(fx.get("sigmas") orelse return error.MissingFixtureSigmas, .float32, s);
    defer _ = mlx.mlx_array_free(sig_arr);
    _ = mlx.mlx_array_eval(sig_arr);
    const sd = mlx.mlx_array_data_float32(sig_arr).?;

    var img = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&img, noise));
    var last: f32 = 0;
    for (0..4) |i| {
        const v = try dit.forward(img, txt, sd[i], 1, 16, 16, null);
        defer _ = mlx.mlx_array_free(v);
        const img_f = try astype(img, .float32, s);
        defer _ = mlx.mlx_array_free(img_f);
        const v_f = try astype(v, .float32, s);
        defer _ = mlx.mlx_array_free(v_f);
        const dt = scalarF(sd[i + 1] - sd[i]);
        defer _ = mlx.mlx_array_free(dt);
        const stepv = try mulA(v_f, dt, s);
        defer _ = mlx.mlx_array_free(stepv);
        const ni_f = try addA(img_f, stepv, s);
        defer _ = mlx.mlx_array_free(ni_f);
        const ni = try astype(ni_f, .bfloat16, s);
        _ = mlx.mlx_array_free(img);
        img = ni;
        const key = try std.fmt.allocPrint(a, "lat{d}", .{i + 1});
        defer a.free(key);
        last = try cosineSim(img, fx.get(key) orelse return error.MissingFixtureLat, s);
        std.debug.print("[mageflow-e2e] step {d} cosine = {d:.6}\n", .{ i + 1, last });
    }
    _ = mlx.mlx_array_free(img);
    try testing.expect(last > 0.99); // bf16 threshold; the f32-timestep bug drops this to ~0.55
}

// EDIT-loop parity vs the mflux reference (E7.5). Validates the edit denoise
// ASSEMBLY — concat([target, refs]) → multi-image RoPE DiT forward → target-slice
// → Euler — through the SAME `denoiseEditLoop` the engine runs, using the LOCAL
// txt2img transformer (decoupled from the Edit checkpoint's weights). Gated on
// MAGEFLOW_TEST_MODEL + MAGEFLOW_EDIT_E2E_FIXTURE (bf16;
// tests/dump_mageflow_edit_e2e_fixture.py). LH=LW=8, 1 ref → frames=2.
test "MageFlow edit loop parity bf16 (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_EDIT_E2E_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    var dit = try Dit.load(io, a, s, model_dir, .bfloat16);
    defer dit.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const noise = fx.get("noise") orelse return error.MissingFixtureNoise;
    const refs = fx.get("refs") orelse return error.MissingFixtureRefs;
    const txt = fx.get("txt") orelse return error.MissingFixtureTxt;
    const ref_final = fx.get("final") orelse return error.MissingFixtureFinal;

    const sig_arr = try astype(fx.get("sigmas") orelse return error.MissingFixtureSigmas, .float32, s);
    defer _ = mlx.mlx_array_free(sig_arr);
    _ = mlx.mlx_array_eval(sig_arr);
    const sn = mlx.mlx_array_size(sig_arr);
    const sd = mlx.mlx_array_data_float32(sig_arr).?;
    const sigmas = try a.alloc(f32, sn);
    defer a.free(sigmas);
    @memcpy(sigmas, sd[0..sn]);

    const target = try denoiseEditLoop(&dit, txt, noise, refs, 2, 8, 8, sigmas, .bfloat16, s, null);
    defer _ = mlx.mlx_array_free(target);
    const cos = try cosineSim(target, ref_final, s);
    std.debug.print("[mageflow-edit-loop] final cosine = {d:.6}\n", .{cos});
    try testing.expect(cos > 0.99);
}

// Text-encoder parity vs the mflux reference. Gated on MAGEFLOW_TEST_MODEL +
// MAGEFLOW_TE_FIXTURE (safetensors with input_ids/attention_mask/hidden_full/
// embeddings, f32; from tests/dump_mageflow_te_fixture.py). Runs the encoder in
// f32. Skips when unset.
test "MageFlow TE conditioning parity (env-gated)" {
    const model_dir = std.mem.span(std.c.getenv("MAGEFLOW_TEST_MODEL") orelse return error.SkipZigTest);
    const fixture = std.mem.span(std.c.getenv("MAGEFLOW_TE_FIXTURE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var te = try TextEncoder.load(io, a, s, model_dir, .float32);
    defer te.deinit();

    var fx = try model_mod.loadWeightsSingleFile(a, fixture);
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureIds;
    const mask_arr = fx.get("attention_mask") orelse return error.MissingFixtureMask;
    const ref_hidden = fx.get("hidden_full") orelse return error.MissingFixtureHidden;
    const ref_emb = fx.get("embeddings") orelse return error.MissingFixtureEmb;

    const ids = try readFixtureIds(a, ids_arr, s);
    defer a.free(ids);
    const mask = try readFixtureIds(a, mask_arr, s);
    defer a.free(mask);

    const hidden = try te.encode(ids, mask);
    defer _ = mlx.mlx_array_free(hidden);
    const cos_h = try cosineSim(hidden, ref_hidden, s);
    std.debug.print("[mageflow-te] hidden cosine = {d:.6} (L={d})\n", .{ cos_h, ids.len });
    try testing.expect(cos_h > 0.999);

    const cond = try te.encodeTextToImage(ids, mask);
    defer _ = mlx.mlx_array_free(cond.embeddings);
    const cos_e = try cosineSim(cond.embeddings, ref_emb, s);
    std.debug.print("[mageflow-te] embeddings cosine = {d:.6} (keep={d})\n", .{ cos_e, cond.keep });
    try testing.expect(cos_e > 0.999);
    try testing.expectEqual(@as(c_int, @intCast(ids.len - TE_DROP_TOKENS)), cond.keep);
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

test "parseConfig reads the optional vision preprocessor budget" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try writeTestCheckpoint(io, &tmp);
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const model_dir = root_buf[0..root_len];

    // Absent (text-only checkpoints don't ship one) → the release defaults.
    const base = try parseConfig(io, a, model_dir);
    try testing.expectEqual(@as(u32, 65_536), base.vlm_min_pixels);
    try testing.expectEqual(@as(u32, 16_777_216), base.vlm_max_pixels);

    // Present → the checkpoint's own budget wins. Hardcoding it would make a
    // checkpoint with different limits condition on a DIFFERENT reference grid
    // than the pipeline it was trained with, silently.
    try tmp.dir.writeFile(io, .{
        .sub_path = "text_encoder/preprocessor_config.json",
        .data = "{\"min_pixels\":200704,\"max_pixels\":1048576,\"patch_size\":16}",
    });
    const cfg = try parseConfig(io, a, model_dir);
    try testing.expectEqual(@as(u32, 200_704), cfg.vlm_min_pixels);
    try testing.expectEqual(@as(u32, 1_048_576), cfg.vlm_max_pixels);
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

test "Engine.load errors cleanly on a config-only checkpoint (no weights)" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try writeTestCheckpoint(io, &tmp);
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    // Config parses; the component weight files are absent → a clean error, no crash.
    if (Engine.load(io, a, root_buf[0..root_len])) |eng| {
        eng.deinit();
        try testing.expect(false); // should not have loaded
    } else |_| {}
}

test "computeSigmas: static-shift FlowMatchEuler schedule" {
    const a = testing.allocator;
    // N=4, shift=6: base = linspace(1, 0.25, 4) = [1, 0.75, 0.5, 0.25];
    // sigma = 6b/(1+5b). sigma[0]=1; then append 0.
    const sig = try computeSigmas(a, 4, 6.0);
    defer a.free(sig);
    try testing.expectEqual(@as(usize, 5), sig.len);
    const expect = [_]f32{ 1.0, 0.9473684, 0.85714287, 0.6666667, 0.0 };
    for (expect, 0..) |e, i| try testing.expect(@abs(sig[i] - e) < 1e-5);
    // N=1 edge: base=1 → sigma[0]=1, sigma[1]=0.
    const one = try computeSigmas(a, 1, 6.0);
    defer a.free(one);
    try testing.expectEqual(@as(usize, 2), one.len);
    try testing.expect(@abs(one[0] - 1.0) < 1e-6 and one[1] == 0.0);
}

test "normalizeDim floors to /16 with a 16px minimum" {
    try testing.expectEqual(@as(u32, 1024), normalizeDim(1024));
    try testing.expectEqual(@as(u32, 1024), normalizeDim(1030));
    try testing.expectEqual(@as(u32, 16), normalizeDim(0));
    try testing.expectEqual(@as(u32, 16), normalizeDim(15));
    try testing.expectEqual(@as(u32, 512), normalizeDim(519));
}

/// Live MLX bytes a helper failed to give back over `iters` calls.
///
/// The class this measures is a materializing helper that wraps an intermediate
/// handle (`mlx_slice` into `o`, then `contiguous(o)`) and never frees `o`: the
/// output is correct, every parity fixture passes, and the retained bytes only
/// show up as a footprint that climbs per generation. No output-equality test
/// can see it — only the accounting can.
///
/// The input is rebuilt and freed INSIDE each iteration, because a retained
/// slice handle keeps its PARENT's buffer alive — a caller-owned source that
/// outlives the call hides the whole class.
fn retainedBytes(
    comptime call: anytype,
    data: []const f32,
    shape: []const c_int,
    tail_args: anytype,
    iters: usize,
) !usize {
    const once = struct {
        fn run(d: []const f32, sh: []const c_int, tail: anytype) !void {
            const parent = mlx.mlx_array_new_data(d.ptr, sh.ptr, @intCast(sh.len), .float32);
            defer _ = mlx.mlx_array_free(parent);
            const r = try @call(.auto, call, .{parent} ++ tail);
            defer _ = mlx.mlx_array_free(r);
            try mlx.check(mlx.mlx_array_eval(r)); // force the buffers to exist
        }
    }.run;

    // `mlx_array_free` drops OUR reference, but the buffer stays alive until
    // the command buffer that touched it retires — so a counter read while the
    // stream is still in flight bills work that is already on its way out.
    // Without this the test failed for exactly ONE buffer (16384 B, the whole
    // source array) at random — measured 5 failures in 25 runs locally, and
    // intermittently on CI — which reads as a leak that comes and goes. With
    // the drain: 25/25. Verified the guard still bites by deleting a free in
    // `once` and watching it go red. Drain before every reading.
    const drain = struct {
        fn run() !void {
            const st = mlx.mlx_default_gpu_stream_new();
            defer _ = mlx.mlx_stream_free(st);
            try mlx.check(mlx.mlx_synchronize(st));
        }
    }.run;

    // Warm-up: the first calls pay one-time kernel/allocator setup, which is
    // not retention.
    for (0..2) |_| try once(data, shape, tail_args);
    try drain();
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    for (0..iters) |_| try once(data, shape, tail_args);
    try drain();
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    return if (after > before) after - before else 0;
}

test "MfLinear dq-gemm: floor parsing is opt-in with a 2048 default" {
    // Unset and "0" mean OFF — the wide-M route is opt-in until each backend
    // has its own A/B (the prefill-perf-kernel rule). "1" opts in at the
    // shared 2048 floor; an explicit integer sets its own floor; junk is off.
    try testing.expectEqual(@as(?usize, null), mfDqGemmFloorFrom(null));
    try testing.expectEqual(@as(?usize, null), mfDqGemmFloorFrom("0"));
    try testing.expectEqual(@as(?usize, 2048), mfDqGemmFloorFrom("1"));
    try testing.expectEqual(@as(?usize, 4096), mfDqGemmFloorFrom("4096"));
    try testing.expectEqual(@as(?usize, null), mfDqGemmFloorFrom("junk"));
}

test "MfLinear wide-M dq-gemm: engages and is no worse than qmm vs fp32 ground truth" {
    const s = mlx.mlx_default_gpu_stream_new();
    const rows: c_int = 2048;
    const in_f: c_int = 1024;
    const out_f: c_int = 384;

    // Random fp32 weight + bf16 activations at a contraction dim near the real
    // one (the fused-QKV lesson: a 256-wide contraction can agree by luck).
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_random_normal(&wf, &[_]c_int{ out_f, in_f }, 2, .float32, 0.0, 1.0, key, s));
    var key2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key2);
    try mlx.check(mlx.mlx_random_key(&key2, 11));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_random_normal(&xf, &[_]c_int{ rows, in_f }, 2, .float32, 0.0, 1.0, key2, s));
    const xb = try astype(xf, .bfloat16, s);
    defer _ = mlx.mlx_array_free(xb);

    // Quantize 8-bit / gs64 (the shipped H3 pack's geometry).
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
    const scb = try astype(sc, .bfloat16, s);
    const bib = try astype(bi, .bfloat16, s);

    var lin = MfLinear{ .quantized = true, .w = wq, .scales = scb, .biases = bib, .dtype = .bfloat16, .bits = 8, .group_size = 64 };
    defer {
        // lin.w aliases wq which the outer defers free; null it out so
        // deinit only releases the casts it owns.
        lin.w = mlx.mlx_array_new();
        lin.deinit();
    }

    // Ground truth: fp32 dequant matmul (the kernel-testing rule — never
    // kernel-vs-kernel agreement).
    var dqf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dqf);
    try mlx.check(mlx.mlx_dequantize(&dqf, wq, sc, bi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true }, s));
    var dqf_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dqf_t);
    try mlx.check(mlx.mlx_transpose(&dqf_t, dqf, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, xf, dqf_t, s));

    // Arm A: stock qmm (force the route off).
    mf_dq_gemm_override = @as(?usize, null);
    defer mf_dq_gemm_override = null;
    const y_qmm = try lin.forward(xb, null, s);
    defer _ = mlx.mlx_array_free(y_qmm);

    // Arm B: dq-gemm forced on for any M — engagement is COUNTED, never
    // inferred from output equality (the silent-fallback class).
    const engaged_before = mf_dq_gemm_engaged;
    mf_dq_gemm_override = @as(?usize, 1);
    const y_dq = try lin.forward(xb, null, s);
    defer _ = mlx.mlx_array_free(y_dq);
    try testing.expect(mf_dq_gemm_engaged > engaged_before);

    const err_qmm = try maxAbsDiff(y_qmm, ref, s);
    const err_dq = try maxAbsDiff(y_dq, ref, s);
    // Both are bf16 roundings of the same product; dq-gemm must not be
    // meaningfully worse than the kernel it replaces.
    try testing.expect(std.math.isFinite(err_dq));
    try testing.expect(err_dq <= err_qmm * 1.25 + 0.05);
}

test "materializing helpers hand back every array they take" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    // 16 KB of source, big enough that a retained slice is unmistakable
    // against allocator noise, reshaped 3-D or 4-D per helper.
    const v = try a.alloc(f32, 2 * 64 * 32);
    defer a.free(v);
    for (v, 0..) |*x, i| x.* = @floatFromInt(i % 13);
    const sh3 = [_]c_int{ 2, 64, 32 };
    const sh4 = [_]c_int{ 2, 4, 16, 32 };

    const n: usize = 8;
    // Every helper that slices and materializes, across the DiT, text-encoder,
    // ViT and VAE paths — one leaking site is one leak per block per step.
    try testing.expectEqual(@as(usize, 0), try retainedBytes(sliceSeq, v, &sh3, .{ 8, 56, s }, n));
    try testing.expectEqual(@as(usize, 0), try retainedBytes(sliceSeq3, v, &sh3, .{ 0, 1, s }, n));
    try testing.expectEqual(@as(usize, 0), try retainedBytes(sliceLast3, v, &sh3, .{ 4, 28, s }, n));
    try testing.expectEqual(@as(usize, 0), try retainedBytes(sliceTeSeq, v, &sh3, .{ 8, 56, s }, n));
    try testing.expectEqual(@as(usize, 0), try retainedBytes(sliceLastAxis, v, &sh4, .{ 4, 28, s }, n));
    try testing.expectEqual(@as(usize, 0), try retainedBytes(cropHW, v, &sh4, .{ 3, 12, s }, n));
    // Controls on the same shapes: a harness that reported retention for
    // everything would fail here too.
    try testing.expectEqual(@as(usize, 0), try retainedBytes(contig, v, &sh3, .{s}, n));
    try testing.expectEqual(@as(usize, 0), try retainedBytes(astype, v, &sh3, .{ mlx.mlx_dtype.bfloat16, s }, n));
}

test "packLatents/unpackLatents round-trip is identity" {
    const s = mlx.mlx_default_gpu_stream_new();
    // A tiny NCHW latent [1,3,2,2] with distinct values.
    var buf: [12]f32 = undefined;
    for (0..12) |i| buf[i] = @floatFromInt(i);
    const sh = [_]c_int{ 1, 3, 2, 2 };
    const nchw = mlx.mlx_array_new_data(&buf, &sh, 4, .float32);
    defer _ = mlx.mlx_array_free(nchw);
    const packed_lat = try packLatents(nchw, s);
    defer _ = mlx.mlx_array_free(packed_lat);
    // packed shape is [1, 4, 3].
    const psh = mlx.getShape(packed_lat);
    try testing.expectEqual(@as(c_int, 4), psh[1]);
    try testing.expectEqual(@as(c_int, 3), psh[2]);
    const back = try unpackLatents(packed_lat, 2, 2, s);
    defer _ = mlx.mlx_array_free(back);
    const diff = try subA(back, nchw, s);
    defer _ = mlx.mlx_array_free(diff);
    var mx_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx_abs);
    try mlx.check(mlx.mlx_abs(&mx_abs, diff, s));
    var mval = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mval);
    try mlx.check(mlx.mlx_max(&mval, mx_abs, false, s));
    _ = mlx.mlx_array_eval(mval);
    var v: f32 = 1;
    try mlx.check(mlx.mlx_array_item_float32(&v, mval));
    try testing.expect(v == 0.0);
}

// ── MfLinear (bf16 ⟷ affine-quantized weights) ──
//
// The 8-bit mirrors (`ddalcu/Mage-Flow-*-MLX-Serve-8bit`) ship affine-quantized
// DiT/TE linears; the upstream bf16 repos ship dense ones. `MfLinear` serves
// both from the same field, so a checkpoint needs no format flag anywhere and a
// MIXED checkpoint (some tensors held back at bf16) works by construction.

/// Max |a-b| between two arrays, as f32. Test-only.
fn maxAbsDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !f32 {
    const af = try astype(a, .float32, s);
    defer _ = mlx.mlx_array_free(af);
    const bf = try astype(b, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const d = try subA(af, bf, s);
    defer _ = mlx.mlx_array_free(d);
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, d, s));
    var mv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mv);
    try mlx.check(mlx.mlx_max(&mv, ab, false, s));
    _ = mlx.mlx_array_eval(mv);
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, mv));
    return out;
}

test "MfLinear: quantized load infers bits/group_size and matches dequant+matmul" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;

    // A real affine-quantized weight built by mlx_quantize, so the packed
    // geometry the loader reasons about is genuine (out=8, in=128).
    const in: c_int = 128;
    const out: c_int = 8;
    const raw = try a.alloc(f32, @intCast(out * in));
    defer a.free(raw);
    for (raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 11)) * 0.05 - 0.25;
    const rsh = [_]c_int{ out, in };
    const rw = mlx.mlx_array_new_data(raw.ptr, &rsh, 2, .float32);
    defer _ = mlx.mlx_array_free(rw);
    const rwb = try astype(rw, .bfloat16, s);
    defer _ = mlx.mlx_array_free(rwb);

    const xv = try a.alloc(f32, @intCast(in));
    defer a.free(xv);
    for (xv, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.2 - 0.4;
    const xsh = [_]c_int{ 1, in };
    const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);

    // Both the production recipe (8/64) and a narrower one, to prove the bits
    // and group size are read PER TENSOR from geometry rather than assumed.
    inline for (.{ .{ 8, 64 }, .{ 4, 32 } }) |cfg| {
        const bits: c_int = cfg[0];
        const gs: c_int = cfg[1];
        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        const null_gscale = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_quantize(&qv, rwb, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", null_gscale, s));
        var qw = mlx.mlx_array_new();
        var qs = mlx.mlx_array_new();
        var qb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&qw, qv, 0));
        try mlx.check(mlx.mlx_vector_array_get(&qs, qv, 1));
        try mlx.check(mlx.mlx_vector_array_get(&qb, qv, 2));

        var ww = model_mod.Weights.init(a);
        defer ww.deinit();
        try ww.map.put(try a.dupe(u8, "q.weight"), qw);
        try ww.map.put(try a.dupe(u8, "q.scales"), qs);
        try ww.map.put(try a.dupe(u8, "q.biases"), qb);

        var ml = try MfLinear.load(&ww, a, "q", @intCast(in), .bfloat16, s);
        defer ml.deinit();
        try testing.expect(ml.quantized);
        try testing.expectEqual(@as(u32, @intCast(bits)), ml.bits);
        try testing.expectEqual(@as(u32, @intCast(gs)), ml.group_size);

        // Ground truth: dequantize the SAME triple and matmul by hand.
        var deq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deq);
        try mlx.check(mlx.mlx_dequantize(&deq, qw, qs, qb, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", null_gscale, .{}, s));
        const deq_t = try transpose(deq, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(deq_t);
        const xb = try astype(xa, .bfloat16, s);
        defer _ = mlx.mlx_array_free(xb);
        const want = try linearT(xb, deq_t, null, s);
        defer _ = mlx.mlx_array_free(want);

        const got = try ml.forward(xa, null, s);
        defer _ = mlx.mlx_array_free(got);
        try testing.expectEqual(@as(c_int, out), mlx.getShape(got)[1]);
        // Same weights, same arithmetic — only kernel reduction order differs.
        try testing.expect(try maxAbsDiff(got, want, s) < 0.02);
    }
}

test "MfLinear: dense load is byte-for-byte the old pre-transposed path" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;

    const in: c_int = 32;
    const out: c_int = 6;
    const raw = try a.alloc(f32, @intCast(out * in));
    defer a.free(raw);
    for (raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 13)) * 0.1 - 0.6;
    const rsh = [_]c_int{ out, in };
    const rw = mlx.mlx_array_new_data(raw.ptr, &rsh, 2, .float32);
    defer _ = mlx.mlx_array_free(rw);
    const rwb = try astype(rw, .bfloat16, s);
    defer _ = mlx.mlx_array_free(rwb);

    var ww = model_mod.Weights.init(a);
    defer ww.deinit();
    var held = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&held, rwb));
    try ww.map.put(try a.dupe(u8, "d.weight"), held);

    var ml = try MfLinear.load(&ww, a, "d", @intCast(in), .bfloat16, s);
    defer ml.deinit();
    try testing.expect(!ml.quantized);

    const xv = try a.alloc(f32, @intCast(in));
    defer a.free(xv);
    for (xv, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.3 - 0.9;
    const xsh = [_]c_int{ 1, in };
    const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);

    // The pre-MfLinear code path, verbatim: pre-transpose at load, linearT at
    // use. A dense checkpoint must still take exactly this arithmetic — the
    // six bf16 parity fixtures depend on it.
    const t = try transpose(rwb, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    const wt = try astype(tc, .bfloat16, s);
    defer _ = mlx.mlx_array_free(wt);
    const xb = try astype(xa, .bfloat16, s);
    defer _ = mlx.mlx_array_free(xb);
    const want = try linearT(xb, wt, null, s);
    defer _ = mlx.mlx_array_free(want);

    const got = try ml.forward(xa, null, s);
    defer _ = mlx.mlx_array_free(got);
    try testing.expect(try maxAbsDiff(got, want, s) == 0.0);
}

test "MfLinear: bias is applied in the compute dtype on both paths" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = testing.allocator;

    const in: c_int = 64;
    const out: c_int = 4;
    const raw = try a.alloc(f32, @intCast(out * in));
    defer a.free(raw);
    for (raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 9)) * 0.05;
    const rsh = [_]c_int{ out, in };
    const rw = mlx.mlx_array_new_data(raw.ptr, &rsh, 2, .float32);
    defer _ = mlx.mlx_array_free(rw);
    const rwb = try astype(rw, .bfloat16, s);
    defer _ = mlx.mlx_array_free(rwb);

    var bvals = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    const bsh = [_]c_int{out};
    const bias32 = mlx.mlx_array_new_data(&bvals, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias32);
    const bias = try astype(bias32, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bias);

    const xv = try a.alloc(f32, @intCast(in));
    defer a.free(xv);
    for (xv, 0..) |*x, i| x.* = if (i == 0) 1.0 else 0.0; // pick row 0 of W
    const xsh = [_]c_int{ 1, in };
    const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);

    inline for (.{ true, false }) |quantize| {
        var ww = model_mod.Weights.init(a);
        defer ww.deinit();
        if (quantize) {
            var qv = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qv);
            const null_gscale = mlx.mlx_array{ .ctx = null };
            try mlx.check(mlx.mlx_quantize(&qv, rwb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", null_gscale, s));
            var qw = mlx.mlx_array_new();
            var qs = mlx.mlx_array_new();
            var qb = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&qw, qv, 0));
            try mlx.check(mlx.mlx_vector_array_get(&qs, qv, 1));
            try mlx.check(mlx.mlx_vector_array_get(&qb, qv, 2));
            try ww.map.put(try a.dupe(u8, "b.weight"), qw);
            try ww.map.put(try a.dupe(u8, "b.scales"), qs);
            try ww.map.put(try a.dupe(u8, "b.biases"), qb);
        } else {
            var held = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&held, rwb));
            try ww.map.put(try a.dupe(u8, "b.weight"), held);
        }
        var ml = try MfLinear.load(&ww, a, "b", @intCast(in), .bfloat16, s);
        defer ml.deinit();

        const no_bias = try ml.forward(xa, null, s);
        defer _ = mlx.mlx_array_free(no_bias);
        const with_bias = try ml.forward(xa, bias, s);
        defer _ = mlx.mlx_array_free(with_bias);
        const delta = try subA(with_bias, no_bias, s);
        defer _ = mlx.mlx_array_free(delta);
        try testing.expect(try maxAbsDiff(delta, bias, s) < 1e-2);
    }
}
