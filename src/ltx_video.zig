//! Native LTX-Video 2.3 (one-stage text→video) — Zig + mlx-c port scaffold.
//! See docs/native-mediagen/03-video-ltx.md for the full plan.
//!
//! STATUS (video fork, foundation slice): this file establishes the validated
//! FOUNDATION for the LTX port — the real config, a single-component q4/bf16
//! weight loader, and the conv3d/group-norm primitives — and pins them against
//! the actual `dgrauet/ltx-2.3-mlx-q4` checkpoint with a loader-validation test.
//! The full forward (Gemma-49-layer capture → connector → 48-block joint DiT →
//! guided Euler loop → 3D-conv VAE decode) is the large remaining work; this
//! file documents the exact architecture (real tensor keys/shapes) in code so
//! the next implementer can build each stage against per-stage oracle taps.
//!
//! KEY FINDINGS (from dumping the real checkpoint):
//!  - VAE conv weights are ALREADY in MLX layout `[C_out, kD, kH, kW, C_in]`
//!    (e.g. `vae_decoder.conv_in.conv.weight [1024,3,3,3,128]`) — NO transpose
//!    at load, unlike the PyTorch-layout audio convs. mlx_conv3d consumes this
//!    directly with NDHWC input.
//!  - DiT: 48 blocks, each 154 tensors. Six attention sub-modules per block:
//!    attn1 (video self), attn2 (video→text cross), audio_attn1, audio_attn2,
//!    audio_to_video_attn, video_to_audio_attn — all q4 (U32 weight + bf16
//!    scales/biases, group_size 64). adaLN tables are F32: scale_shift_table
//!    [9,4096] / audio_scale_shift_table [9,2048] (shift,scale,gate ×3),
//!    prompt_scale_shift_table [2,*], scale_shift_table_a2v_ca_video [5,4096] /
//!    _audio [5,2048] (AV-cross, SCALE-FIRST ordering — a porting trap).
//!  - Connector: bf16. Two `Embeddings1DConnector`s (video dim 4096 / audio
//!    2048), 8 `transformer_1d_blocks` each, `learnable_registers [128, dim]`,
//!    gated attention (`to_gate_logits [num_heads, dim]`), GEGLU ff, q/k RMSNorm.
//!    `text_embedding_projection.{video,audio}_aggregate_embed.weight
//!    [4096|2048, 188160]` (= 49×3840 Gemma-stack projection).
//!  - Quant: bits 4, group_size 64, only_transformer_blocks=true. A tensor is
//!    q4 iff a sibling `<name>.scales` exists; else it's stored at its dtype.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");

const S = mlx.mlx_stream;

// ── Config (from config.json) ──

pub const LtxConfig = struct {
    num_heads: u32 = 32,
    head_dim: u32 = 128,
    in_channels: u32 = 128,
    out_channels: u32 = 128,
    num_layers: u32 = 48,
    cross_attention_dim: u32 = 4096, // video_dim
    audio_num_heads: u32 = 32,
    audio_head_dim: u32 = 64,
    audio_in_channels: u32 = 128,
    audio_cross_attention_dim: u32 = 2048, // audio_dim
    pos_emb_theta: f32 = 10000.0,
    pos_emb_max_pos: [3]u32 = .{ 20, 2048, 2048 },
    audio_pos_emb_max_pos: u32 = 20,
    timestep_scale: f32 = 1000.0,
    norm_eps: f32 = 1e-6,
    connector_max_pos: u32 = 4096,
    connector_num_blocks: u32 = 8,
    connector_num_registers: u32 = 128,
    gemma_layers: u32 = 49, // embedding + 48 layers
    gemma_hidden: u32 = 3840,
    aggregate_in: u32 = 188160, // 49 * 3840
    quant_bits: u32 = 4,
    quant_group_size: u32 = 64,

    pub fn videoDim(self: LtxConfig) u32 {
        return self.num_heads * self.head_dim; // 4096
    }
    pub fn audioDim(self: LtxConfig) u32 {
        return self.audio_num_heads * self.audio_head_dim; // 2048
    }
};

// ── Single-component weight loader ──
//
// The LTX checkpoint stores each sub-model as a separate `*.safetensors` file in
// ONE directory (transformer-dev 11GB, connector 5.9GB, vae_decoder 777MB, ...).
// `model.loadWeights` scans a whole dir → it would load all 40GB+ at once (OOM).
// We load a single component file instead and classify each tensor q4-vs-bf16 by
// the presence of a sibling `.scales` entry.

pub const Component = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Component) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            self.allocator.free(e.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Component, key: []const u8) ?mlx.mlx_array {
        return self.map.get(key);
    }

    /// A weight is quantized (q4) iff a sibling `<base>.scales` exists. `weight`
    /// keys end in `.weight`; the quant siblings are `.scales`/`.biases`.
    pub fn isQuantized(self: *const Component, weight_key: []const u8) bool {
        if (!std.mem.endsWith(u8, weight_key, ".weight")) return false;
        const base = weight_key[0 .. weight_key.len - ".weight".len];
        var buf: [512]u8 = undefined;
        const scales_key = std.fmt.bufPrint(&buf, "{s}.scales", .{base}) catch return false;
        return self.map.contains(scales_key);
    }

    pub fn count(self: *const Component) u32 {
        return @intCast(self.map.count());
    }
};

/// Load one `*.safetensors` file (absolute path) into a Component map.
pub fn loadComponent(allocator: std.mem.Allocator, path: [:0]const u8, s: S) !Component {
    var comp = Component{ .map = std.StringHashMap(mlx.mlx_array).init(allocator), .allocator = allocator };
    errdefer comp.deinit();

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0) break;
        const k = key orelse break;
        const key_slice = std.mem.span(k);
        const owned_key = try allocator.dupe(u8, key_slice);
        errdefer allocator.free(owned_key);
        var owned_val = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&owned_val, value));
        try comp.map.put(owned_key, owned_val);
    }
    log.info("[ltx] loaded {d} tensors from {s}\n", .{ comp.count(), path });
    return comp;
}

// ── Primitives the DiT/VAE need beyond the LLM stack ──

/// 3D convolution on NDHWC input with MLX-layout weight `[C_out, kD, kH, kW, C_in]`.
/// `pad` is symmetric per spatial axis (caller does causal/replicate padding
/// separately via mlx_pad for the LTX causal-temporal scheme).
pub fn conv3d(input: mlx.mlx_array, weight: mlx.mlx_array, stride: [3]c_int, pad: [3]c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv3d(&out, input, weight, stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1, 1, s));
    return out;
}

/// Slice frames `[start, stop)` along the depth axis (axis 1) of an NDHWC array.
fn sliceAxis1(x: mlx.mlx_array, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const st = [_]c_int{ 0, start, 0, 0, 0 };
    const sp = [_]c_int{ sh[0], stop, sh[2], sh[3], sh[4] };
    const stride = [_]c_int{ 1, 1, 1, 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &st, 5, &sp, 5, &stride, 5, s));
    return out;
}

/// Decoder-side (causal=False) Conv3dBlock forward: temporal SYMMETRIC replicate
/// padding ((k-1)/2 frames front AND back), spatial zero-pad, then conv3d(pad=0),
/// + bias. `x` is NDHWC `[B,D,H,W,C]`; `weight` is MLX `[O,kD,kH,kW,I]`; `bias` `[O]`.
/// This is the reusable VAE-decoder conv (matches ltx_core_mlx Conv3dBlock, causal=False).
pub fn decoderConv3d(x: mlx.mlx_array, weight: mlx.mlx_array, bias: ?mlx.mlx_array, k: u32, spatial_pad: u32, s: S) !mlx.mlx_array {
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&t, x));

    const ps: c_int = @intCast((k - 1) / 2);
    if (ps > 0) {
        const D: c_int = mlx.getShape(x)[1];
        const first = try sliceAxis1(x, 0, 1, s);
        defer _ = mlx.mlx_array_free(first);
        const last = try sliceAxis1(x, D - 1, D, s);
        defer _ = mlx.mlx_array_free(last);
        var fp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fp);
        try mlx.check(mlx.mlx_repeat_axis(&fp, first, ps, 1, s));
        var lp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lp);
        try mlx.check(mlx.mlx_repeat_axis(&lp, last, ps, 1, s));
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, fp);
        _ = mlx.mlx_vector_array_append_value(vec, x);
        _ = mlx.mlx_vector_array_append_value(vec, lp);
        var nt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&nt, vec, 1, s));
        _ = mlx.mlx_array_free(t);
        t = nt;
    }
    if (spatial_pad > 0) {
        const sp: c_int = @intCast(spatial_pad);
        const axes = [_]c_int{ 2, 3 };
        const lo = [_]c_int{ sp, sp };
        const hi = [_]c_int{ sp, sp };
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        var nt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_pad(&nt, t, &axes, 2, &lo, 2, &hi, 2, zero, "constant", s));
        _ = mlx.mlx_array_free(t);
        t = nt;
    }
    // mlx_conv miscomputes on strided/lazy input — materialize first (image-fork gotcha).
    var tc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tc);
    try mlx.check(mlx.mlx_contiguous(&tc, t, false, s));
    _ = mlx.mlx_array_free(t);

    const out = try conv3d(tc, weight, .{ 1, 1, 1 }, .{ 0, 0, 0 }, s);
    if (bias) |b| {
        var wb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&wb, out, b, s));
        _ = mlx.mlx_array_free(out);
        return wb;
    }
    return out;
}

/// Null-weight RMS / "PixelNorm" over the channel (last) axis: x / sqrt(mean(x^2)+eps).
/// LTX VAE uses parameterless pixel norm; mlx_fast_rms_norm crashes on null weight,
/// so compute it explicitly.
pub fn pixelNorm(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_multiply(&sq, x, x, s));
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, sq, -1, true, s));
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    var denom = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(denom);
    try mlx.check(mlx.mlx_add(&denom, mean, eps_a, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    try mlx.check(mlx.mlx_rsqrt(&rsq, denom, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, x, rsq, s));
    return out;
}

// ── 3D VAE decoder (atop decoderConv3d) ──

/// PixelNorm matching the reference `mx.fast.rms_norm(x, weight=None, eps)`:
/// the FUSED kernel upcasts to fp32 internally, which the explicit bf16
/// `pixelNorm` does not — over the VAE's ~20 sequential norms that divergence
/// compounds. mlx_fast_rms_norm crashes on a null weight, so pass ones([C]).
fn pixelNormFast(x: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const c = sh[sh.len - 1];
    const ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    const one_val = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_val);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    try mlx.check(mlx.mlx_full(&ones_w, &[_]c_int{c}, 1, one_val, .bfloat16, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, ones_w, eps, s));
    return out;
}

fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, x, sig, s));
    return out;
}

/// Look up `<base>.weight`/`.bias` in the component and run the decoder conv
/// (k=3, spatial_pad=1). `base` e.g. "vae_decoder.conv_in.conv".
fn convByKey(comp: *const Component, base: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var wbuf: [256]u8 = undefined;
    var bbuf: [256]u8 = undefined;
    const wk = try std.fmt.bufPrint(&wbuf, "{s}.weight", .{base});
    const bk = try std.fmt.bufPrint(&bbuf, "{s}.bias", .{base});
    const w = comp.get(wk) orelse return error.MissingVaeWeight;
    const b = comp.get(bk);
    return decoderConv3d(x, w, b, 3, 1, s);
}

/// Pre-activation residual block: x = conv2(silu(pn(conv1(silu(pn(x)))))) + x.
fn resBlock3d(comp: *const Component, x: mlx.mlx_array, up_idx: u32, blk: u32, s: S) !mlx.mlx_array {
    var b1: [256]u8 = undefined;
    var b2: [256]u8 = undefined;
    const k1 = try std.fmt.bufPrint(&b1, "vae_decoder.up_blocks.{d}.res_blocks.{d}.conv1.conv", .{ up_idx, blk });
    const pn1 = try pixelNormFast(x, 1e-8, s);
    defer _ = mlx.mlx_array_free(pn1);
    const a1 = try silu(pn1, s);
    defer _ = mlx.mlx_array_free(a1);
    const c1 = try convByKey(comp, k1, a1, s);
    defer _ = mlx.mlx_array_free(c1);
    const k2 = try std.fmt.bufPrint(&b2, "vae_decoder.up_blocks.{d}.res_blocks.{d}.conv2.conv", .{ up_idx, blk });
    const pn2 = try pixelNormFast(c1, 1e-8, s);
    defer _ = mlx.mlx_array_free(pn2);
    const a2 = try silu(pn2, s);
    defer _ = mlx.mlx_array_free(a2);
    const c2 = try convByKey(comp, k2, a2, s);
    defer _ = mlx.mlx_array_free(c2);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, c2, x, s));
    return out;
}

fn reshapeTo(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}
fn transposeTo(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&out, x, axes.ptr, axes.len, s));
    return out;
}

/// depth-to-space: (B,D,H,W, C·tf·sf·sf) → (B, D·tf, H·sf, W·sf, C).
/// Channel split order (c, p1=temporal, p2=H, p3=W), c outermost.
fn pixelShuffle3d(x: mlx.mlx_array, sf: u32, tf: u32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B = sh[0];
    const D = sh[1];
    const H = sh[2];
    const W = sh[3];
    const Ct = sh[4];
    const sfc: c_int = @intCast(sf);
    const tfc: c_int = @intCast(tf);
    const C = @divExact(Ct, sfc * sfc * tfc);
    const r1 = try reshapeTo(x, &[_]c_int{ B, D, H, W, C, tfc, sfc, sfc }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t = try transposeTo(r1, &[_]c_int{ 0, 1, 5, 2, 6, 3, 7, 4 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshapeTo(t, &[_]c_int{ B, D * tfc, H * sfc, W * sfc, C }, s);
}

/// final spatial unpatchify (B,F,H,W, C·ps·ps) → (B,F, H·ps, W·ps, C).
/// Channel split (c, p=1, r=W, q=H) — width before height.
fn unpatchifySpatial(x: mlx.mlx_array, ps: u32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B = sh[0];
    const F = sh[1];
    const H = sh[2];
    const W = sh[3];
    const Ct = sh[4];
    const psc: c_int = @intCast(ps);
    const C = @divExact(Ct, psc * psc);
    const r1 = try reshapeTo(x, &[_]c_int{ B, F, H, W, C, psc, psc }, s);
    defer _ = mlx.mlx_array_free(r1);
    const t = try transposeTo(r1, &[_]c_int{ 0, 1, 2, 6, 3, 5, 4 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshapeTo(t, &[_]c_int{ B, F, H * psc, W * psc, C }, s);
}

const ResStageSpec = struct { up_idx: u32, num_blocks: u32 };
const UpSpec = struct { up_idx: u32, sf: u32, tf: u32 };

/// Decode an LTX latent `[B,128,F,H,W]` (BCFHW) → pixels `[B,3,8F-7,32H,32W]`
/// (BCFHW). Weights live in `comp` (the vae_decoder component). Runs on stream `s`.
pub fn vaeDecode(comp: *const Component, latent_bcfhw: mlx.mlx_array, s: S) !mlx.mlx_array {
    // BCFHW → BFHWC.
    var x = try transposeTo(latent_bcfhw, &[_]c_int{ 0, 2, 3, 4, 1 }, s);

    // denormalize: x*std + mean, stats [128] → [1,1,1,1,128].
    {
        const mean = comp.get("vae_decoder.per_channel_statistics.mean").?;
        const std_ = comp.get("vae_decoder.per_channel_statistics.std").?;
        const mr = try reshapeTo(mean, &[_]c_int{ 1, 1, 1, 1, 128 }, s);
        defer _ = mlx.mlx_array_free(mr);
        const sr = try reshapeTo(std_, &[_]c_int{ 1, 1, 1, 1, 128 }, s);
        defer _ = mlx.mlx_array_free(sr);
        var xs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&xs, x, sr, s));
        _ = mlx.mlx_array_free(x);
        var xm = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&xm, xs, mr, s));
        _ = mlx.mlx_array_free(xs);
        x = xm;
    }

    // conv_in (128→1024).
    {
        const nx = try convByKey(comp, "vae_decoder.conv_in.conv", x, s);
        _ = mlx.mlx_array_free(x);
        x = nx;
    }

    // up_blocks: even = ResStage, odd = DepthToSpace conv + pixel_shuffle.
    const res_stages = [_]ResStageSpec{
        .{ .up_idx = 0, .num_blocks = 2 }, .{ .up_idx = 2, .num_blocks = 2 },
        .{ .up_idx = 4, .num_blocks = 4 }, .{ .up_idx = 6, .num_blocks = 6 },
        .{ .up_idx = 8, .num_blocks = 4 },
    };
    const ups = [_]UpSpec{
        .{ .up_idx = 1, .sf = 2, .tf = 2 }, .{ .up_idx = 3, .sf = 2, .tf = 2 },
        .{ .up_idx = 5, .sf = 1, .tf = 2 }, .{ .up_idx = 7, .sf = 2, .tf = 1 },
    };
    var i: u32 = 0;
    while (i <= 8) : (i += 1) {
        if (i % 2 == 0) {
            // ResStage.
            const spec = for (res_stages) |r| {
                if (r.up_idx == i) break r;
            } else unreachable;
            var b: u32 = 0;
            while (b < spec.num_blocks) : (b += 1) {
                const nx = try resBlock3d(comp, x, i, b, s);
                _ = mlx.mlx_array_free(x);
                x = nx;
            }
        } else {
            // DepthToSpace conv then pixel_shuffle.
            const spec = for (ups) |u| {
                if (u.up_idx == i) break u;
            } else unreachable;
            var kbuf: [256]u8 = undefined;
            const key = try std.fmt.bufPrint(&kbuf, "vae_decoder.up_blocks.{d}.conv.conv", .{i});
            const cv = try convByKey(comp, key, x, s);
            _ = mlx.mlx_array_free(x);
            const ps = try pixelShuffle3d(cv, spec.sf, spec.tf, s);
            _ = mlx.mlx_array_free(cv);
            var dts = ps;
            if (spec.tf > 1) {
                // drop first frame after temporal upsample.
                const D: c_int = mlx.getShape(ps)[1];
                dts = try sliceAxis1(ps, 1, D, s);
                _ = mlx.mlx_array_free(ps);
            }
            // pixel_shuffle/slice produce STRIDED views; the next stage's norm/
            // conv read them wrong unless materialized (strided-array gotcha).
            x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_contiguous(&x, dts, false, s));
            _ = mlx.mlx_array_free(dts);
        }
        _ = mlx.mlx_array_eval(x); // keep the lazy graph bounded across stages
    }

    // conv_out(silu(pixel_norm(x))) (128→48).
    {
        const pn = try pixelNormFast(x, 1e-8, s);
        _ = mlx.mlx_array_free(x);
        const a = try silu(pn, s);
        _ = mlx.mlx_array_free(pn);
        const co = try convByKey(comp, "vae_decoder.conv_out.conv", a, s);
        _ = mlx.mlx_array_free(a);
        x = co;
    }

    // final spatial unpatchify 48→3, 4× spatial.
    {
        const up = try unpatchifySpatial(x, 4, s);
        _ = mlx.mlx_array_free(x);
        x = up;
    }

    // BFHWC → BCFHW. Materialize: a lazy transpose view reads back in SOURCE
    // memory order via mlx_array_data_* (strided-array gotcha) — contiguize so
    // callers get a real BCFHW array.
    const t = try transposeTo(x, &[_]c_int{ 0, 4, 1, 2, 3 }, s);
    _ = mlx.mlx_array_free(x);
    defer _ = mlx.mlx_array_free(t);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&out, t, false, s));
    return out;
}

// ── Connector: TextEmbeddingProjection (stage 2, front half) ──
// 49 Gemma hidden states [1,T,3840] → video[1,T,4096] + audio[1,T,2048].
// Per-token RMS over the 49-layer stack (axis 3840), reshape to [1,T,188160],
// then two √(target/3840)-rescaled biased Linears. Matches the reference
// connector.text_embedding_projection.* (the Embeddings1DConnector transformers
// that follow are the remaining back half).

pub const ProjOut = struct {
    video: mlx.mlx_array,
    audio: mlx.mlx_array,
    pub fn deinit(self: *ProjOut) void {
        _ = mlx.mlx_array_free(self.video);
        _ = mlx.mlx_array_free(self.audio);
    }
};

fn projOne(comp: *const Component, allocator: std.mem.Allocator, stacked: mlx.mlx_array, name: []const u8, dim: u32, s: S) !mlx.mlx_array {
    const scale: f32 = @sqrt(@as(f32, @floatFromInt(dim)) / 3840.0);
    const sa = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(sa);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, stacked, sa, s));

    const wk = try std.fmt.allocPrint(allocator, "connector.text_embedding_projection.{s}_aggregate_embed.weight", .{name});
    defer allocator.free(wk);
    const bk = try std.fmt.allocPrint(allocator, "connector.text_embedding_projection.{s}_aggregate_embed.bias", .{name});
    defer allocator.free(bk);
    const w = comp.get(wk) orelse return error.MissingConnectorWeight; // [dim, 188160]
    const b = comp.get(bk) orelse return error.MissingConnectorWeight; // [dim]

    // Transpose W → [188160, dim] and materialize (lazy transpose into matmul is
    // the pinned strided-array trap).
    var wt_lazy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt_lazy);
    try mlx.check(mlx.mlx_transpose_axes(&wt_lazy, w, &[_]c_int{ 1, 0 }, 2, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_contiguous(&wt, wt_lazy, false, s));

    var mm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mm);
    try mlx.check(mlx.mlx_matmul(&mm, scaled, wt, s)); // [1,T,dim]
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, mm, b, s));
    return out;
}

/// Run the projection front-half of the connector. `stack` is 49 arrays [1,T,3840].
pub fn connectorProject(comp: *const Component, allocator: std.mem.Allocator, stack: []const mlx.mlx_array, s: S) !ProjOut {
    const eps: f32 = 1e-6;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (stack) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var enc = mlx.mlx_array_new(); // [1,T,3840,49]
    defer _ = mlx.mlx_array_free(enc);
    try mlx.check(mlx.mlx_stack_axis(&enc, vec, 3, s));

    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_multiply(&sq, enc, enc, s));
    var meanv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(meanv);
    try mlx.check(mlx.mlx_mean_axis(&meanv, sq, 2, true, s)); // [1,T,1,49]
    const epsa = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(epsa);
    var vplus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vplus);
    try mlx.check(mlx.mlx_add(&vplus, meanv, epsa, s));
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, vplus, s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_multiply(&normed, enc, rs, s)); // [1,T,3840,49]

    const T: c_int = mlx.getShape(enc)[1];
    var stacked = mlx.mlx_array_new(); // [1,T,188160]
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_reshape(&stacked, normed, &[_]c_int{ 1, T, 188160 }, 3, s));

    const video = try projOne(comp, allocator, stacked, "video", 4096, s);
    errdefer _ = mlx.mlx_array_free(video);
    const audio = try projOne(comp, allocator, stacked, "audio", 2048, s);
    return .{ .video = video, .audio = audio };
}

// ════════════════════════════════════════════════════════════════════════
// Connector transformer (Embeddings1DConnector): the back half of the connector.
// Projected video/audio embeds [1,T,dim] + valid-token count → 128 learnable
// registers (replace pads) → 8 transformer blocks (gated attention, split-RoPE,
// GELU FF, affine-free RMSNorm) → DiT conditioning [1,T,dim].
// ════════════════════════════════════════════════════════════════════════

/// Linear with bias: matmul(x, Wᵀ)+b. W stored [out,in]; transposed+materialized
/// (a lazy transpose into matmul is the strided-array trap).
fn linBias(comp: *const Component, allocator: std.mem.Allocator, x: mlx.mlx_array, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(allocator, fmt ++ ".weight", args);
    defer allocator.free(wk);
    const bk = try std.fmt.allocPrint(allocator, fmt ++ ".bias", args);
    defer allocator.free(bk);
    const w = comp.get(wk) orelse return error.MissingConnectorWeight;
    const b = comp.get(bk) orelse return error.MissingConnectorWeight;
    var wt_lazy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt_lazy);
    try mlx.check(mlx.mlx_transpose_axes(&wt_lazy, w, &[_]c_int{ 1, 0 }, 2, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_contiguous(&wt, wt_lazy, false, s));
    var mm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mm);
    try mlx.check(mlx.mlx_matmul(&mm, x, wt, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, mm, b, s));
    return out;
}

/// Affine-free RMS norm over the last axis (mx.fast.rms_norm(x, weight=None));
/// mlx_fast_rms_norm crashes on null weight, so pass ones([dim]).
fn rmsAF(x: mlx.mlx_array, dim: u32, eps: f32, s: S) !mlx.mlx_array {
    const ov = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(ov);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    try mlx.check(mlx.mlx_broadcast_to(&ones_w, ov, &[_]c_int{@intCast(dim)}, 1, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, ones_w, eps, s));
    return out;
}

/// Weighted RMS norm (nn.RMSNorm with a learnable weight).
fn rmsW(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

/// gelu_approx (tanh): 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³))).
fn geluApprox(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const c0 = mlx.mlx_array_new_float(0.7978845608028654);
    defer _ = mlx.mlx_array_free(c0);
    const c1 = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(c1);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_multiply(&x2, x, x, s));
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_multiply(&x3, x2, x, s));
    var t1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t1);
    try mlx.check(mlx.mlx_multiply(&t1, x3, c1, s));
    var inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner);
    try mlx.check(mlx.mlx_add(&inner, x, t1, s));
    var inner_s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner_s);
    try mlx.check(mlx.mlx_multiply(&inner_s, inner, c0, s));
    var th = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(th);
    try mlx.check(mlx.mlx_tanh(&th, inner_s, s));
    var thp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(thp);
    try mlx.check(mlx.mlx_add(&thp, th, one, s));
    var hx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hx);
    try mlx.check(mlx.mlx_multiply(&hx, x, half, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, hx, thp, s));
    return out;
}

/// apply_rope_split: x [1,H,N,hd], cos/sin [1,H,N,hd/2] →
/// concat([x1*cos - x2*sin, x1*sin + x2*cos]).
fn applyRopeSplit(x: mlx.mlx_array, cos_f: mlx.mlx_array, sin_f: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const hd: c_int = sh[3];
    const half = @divExact(hd, 2);
    const st = [_]c_int{ 0, 0, 0, 0 };
    const sp1 = [_]c_int{ sh[0], sh[1], sh[2], half };
    const sp2 = [_]c_int{ sh[0], sh[1], sh[2], hd };
    const st2 = [_]c_int{ 0, 0, 0, half };
    const str = [_]c_int{ 1, 1, 1, 1 };
    var x1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1);
    try mlx.check(mlx.mlx_slice(&x1, x, &st, 4, &sp1, 4, &str, 4, s));
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_slice(&x2, x, &st2, 4, &sp2, 4, &str, 4, s));
    var x1c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1c);
    try mlx.check(mlx.mlx_multiply(&x1c, x1, cos_f, s));
    var x2s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2s);
    try mlx.check(mlx.mlx_multiply(&x2s, x2, sin_f, s));
    var lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo);
    try mlx.check(mlx.mlx_subtract(&lo, x1c, x2s, s));
    var x1s = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x1s);
    try mlx.check(mlx.mlx_multiply(&x1s, x1, sin_f, s));
    var x2c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2c);
    try mlx.check(mlx.mlx_multiply(&x2c, x2, cos_f, s));
    var hi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hi);
    try mlx.check(mlx.mlx_add(&hi, x1s, x2c, s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, lo);
    _ = mlx.mlx_vector_array_append_value(vec, hi);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 3, s));
    return out;
}

const RopeCS = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

/// Precompute split-RoPE cos/sin for the connector → [1, heads, T, head_dim/2].
/// freq_indices[j] = theta^(j/(num_freqs-1)) * pi/2 (log-spaced, fractional pos).
fn connectorRope(allocator: std.mem.Allocator, T: u32, dim: u32, head_dim: u32, s: S) !RopeCS {
    const theta: f32 = 10000.0;
    const max_pos: f32 = 4096.0;
    const num_heads: u32 = dim / head_dim;
    const num_freqs: u32 = dim / 2; // inner_dim//2 (inner_dim == dim)
    const hd_half: u32 = head_dim / 2;

    const fi = try allocator.alloc(f32, num_freqs);
    defer allocator.free(fi);
    const denom: f32 = @floatFromInt(num_freqs - 1);
    for (0..num_freqs) |j| {
        const e: f32 = @as(f32, @floatFromInt(j)) / denom; // linspace 0..1
        fi[j] = std.math.pow(f32, theta, e) * (std.math.pi / 2.0);
    }
    const freqs_buf = try allocator.alloc(f32, T * num_freqs);
    defer allocator.free(freqs_buf);
    for (0..T) |t| {
        const frac: f32 = @as(f32, @floatFromInt(t)) / max_pos;
        const sc = frac * 2.0 - 1.0;
        for (0..num_freqs) |j| freqs_buf[t * num_freqs + j] = fi[j] * sc;
    }
    const fshape = [_]c_int{ 1, @intCast(T), @intCast(num_freqs) };
    const freqs = mlx.mlx_array_new_data(freqs_buf.ptr, &fshape, 3, .float32);
    defer _ = mlx.mlx_array_free(freqs);

    var cos_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_raw);
    try mlx.check(mlx.mlx_cos(&cos_raw, freqs, s));
    var sin_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin_raw);
    try mlx.check(mlx.mlx_sin(&sin_raw, freqs, s));
    const rshape = [_]c_int{ 1, @intCast(T), @intCast(num_heads), @intCast(hd_half) };
    var cos_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_r);
    try mlx.check(mlx.mlx_reshape(&cos_r, cos_raw, &rshape, 4, s));
    var sin_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin_r);
    try mlx.check(mlx.mlx_reshape(&sin_r, sin_raw, &rshape, 4, s));
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var cos_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&cos_t, cos_r, &perm, 4, s));
    var sin_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&sin_t, sin_r, &perm, 4, s));
    return .{ .cos = cos_t, .sin = sin_t };
}

/// Run one Embeddings1DConnector (8 blocks) on `input` [1,T,dim]. `n_valid` =
/// number of valid (non-pad) tokens (left-padded → the last n_valid). `prefix` is
/// e.g. "connector.video_embeddings_connector". Returns [1,T,dim].
pub fn connectorTransform(comp: *const Component, allocator: std.mem.Allocator, input: mlx.mlx_array, n_valid: u32, dim: u32, head_dim: u32, prefix: []const u8, s: S) !mlx.mlx_array {
    const T: u32 = @intCast(mlx.getShape(input)[1]);
    const Ti: c_int = @intCast(T);
    const di: c_int = @intCast(dim);
    const num_heads: u32 = dim / head_dim;
    const nh: c_int = @intCast(num_heads);
    const hdi: c_int = @intCast(head_dim);
    const nv: c_int = @intCast(n_valid);

    // ── Register replacement (left-padded → valid first, registers fill rest) ──
    const reg_key = try std.fmt.allocPrint(allocator, "{s}.learnable_registers", .{prefix});
    defer allocator.free(reg_key);
    const registers = comp.get(reg_key) orelse return error.MissingConnectorWeight; // [128, dim]
    const num_reg: u32 = @intCast(mlx.getShape(registers)[0]);
    const num_tiles: u32 = T / num_reg;
    var reg3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reg3);
    try mlx.check(mlx.mlx_reshape(&reg3, registers, &[_]c_int{ 1, @intCast(num_reg), di }, 3, s));
    var reg_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reg_b);
    try mlx.check(mlx.mlx_broadcast_to(&reg_b, reg3, &[_]c_int{ @intCast(num_tiles), @intCast(num_reg), di }, 3, s));
    var tiled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tiled);
    try mlx.check(mlx.mlx_reshape(&tiled, reg_b, &[_]c_int{ 1, Ti, di }, 3, s)); // bf16 (registers bf16)
    // Cast input to bf16 so the slice matches the bf16 registers for concat.
    var input_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(input_bf);
    try mlx.check(mlx.mlx_astype(&input_bf, input, .bfloat16, s));
    const str3 = [_]c_int{ 1, 1, 1 };
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_slice(&valid, input_bf, &[_]c_int{ 0, Ti - nv, 0 }, 3, &[_]c_int{ 1, Ti, di }, 3, &str3, 3, s));
    var reg_part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reg_part);
    try mlx.check(mlx.mlx_slice(&reg_part, tiled, &[_]c_int{ 0, nv, 0 }, 3, &[_]c_int{ 1, Ti, di }, 3, &str3, 3, s));
    var x = mlx.mlx_array_new();
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, valid);
        _ = mlx.mlx_vector_array_append_value(vec, reg_part);
        try mlx.check(mlx.mlx_concatenate_axis(&x, vec, 1, s)); // [1,T,dim] bf16
    }

    // ── RoPE (cast to bf16 to match x) ──
    const rope = try connectorRope(allocator, T, dim, head_dim, s);
    defer _ = mlx.mlx_array_free(rope.cos);
    defer _ = mlx.mlx_array_free(rope.sin);
    var cosb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cosb);
    try mlx.check(mlx.mlx_astype(&cosb, rope.cos, .bfloat16, s));
    var sinb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinb);
    try mlx.check(mlx.mlx_astype(&sinb, rope.sin, .bfloat16, s));

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const null_arr = mlx.mlx_array{ .ctx = null };

    var blk: u32 = 0;
    while (blk < 8) : (blk += 1) {
        // ── Attention ──
        const normed = try rmsAF(x, dim, 1e-6, s);
        defer _ = mlx.mlx_array_free(normed);
        var q = try linBias(comp, allocator, normed, "{s}.transformer_1d_blocks.{d}.attn1.to_q", .{ prefix, blk }, s);
        defer _ = mlx.mlx_array_free(q);
        var k = try linBias(comp, allocator, normed, "{s}.transformer_1d_blocks.{d}.attn1.to_k", .{ prefix, blk }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try linBias(comp, allocator, normed, "{s}.transformer_1d_blocks.{d}.attn1.to_v", .{ prefix, blk }, s);
        defer _ = mlx.mlx_array_free(v);
        {
            const qnk = try std.fmt.allocPrint(allocator, "{s}.transformer_1d_blocks.{d}.attn1.q_norm.weight", .{ prefix, blk });
            defer allocator.free(qnk);
            const knk = try std.fmt.allocPrint(allocator, "{s}.transformer_1d_blocks.{d}.attn1.k_norm.weight", .{ prefix, blk });
            defer allocator.free(knk);
            const qn = try rmsW(q, comp.get(qnk).?, 1e-5, s);
            _ = mlx.mlx_array_free(q);
            q = qn;
            const kn = try rmsW(k, comp.get(knk).?, 1e-5, s);
            _ = mlx.mlx_array_free(k);
            k = kn;
        }
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var qh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qh);
        {
            var qr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qr);
            try mlx.check(mlx.mlx_reshape(&qr, q, &[_]c_int{ 1, Ti, nh, hdi }, 4, s));
            try mlx.check(mlx.mlx_transpose_axes(&qh, qr, &perm, 4, s));
        }
        var kh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kh);
        {
            var kr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kr);
            try mlx.check(mlx.mlx_reshape(&kr, k, &[_]c_int{ 1, Ti, nh, hdi }, 4, s));
            try mlx.check(mlx.mlx_transpose_axes(&kh, kr, &perm, 4, s));
        }
        var vh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vh);
        {
            var vr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vr);
            try mlx.check(mlx.mlx_reshape(&vr, v, &[_]c_int{ 1, Ti, nh, hdi }, 4, s));
            try mlx.check(mlx.mlx_transpose_axes(&vh, vr, &perm, 4, s));
        }
        const qrp = try applyRopeSplit(qh, cosb, sinb, s);
        defer _ = mlx.mlx_array_free(qrp);
        const krp = try applyRopeSplit(kh, cosb, sinb, s);
        defer _ = mlx.mlx_array_free(krp);
        var attn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qrp, krp, vh, scale, "", null_arr, null_arr, s));
        { // per-head gate: 2*sigmoid(to_gate_logits(normed)) [1,T,heads] → [1,heads,T,1]
            const gl = try linBias(comp, allocator, normed, "{s}.transformer_1d_blocks.{d}.attn1.to_gate_logits", .{ prefix, blk }, s);
            defer _ = mlx.mlx_array_free(gl);
            var sg = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sg);
            try mlx.check(mlx.mlx_sigmoid(&sg, gl, s));
            const two = mlx.mlx_array_new_float(2.0);
            defer _ = mlx.mlx_array_free(two);
            var gate = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate);
            try mlx.check(mlx.mlx_multiply(&gate, sg, two, s));
            var gate_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_t);
            try mlx.check(mlx.mlx_transpose_axes(&gate_t, gate, &[_]c_int{ 0, 2, 1 }, 3, s));
            var gate_4 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gate_4);
            try mlx.check(mlx.mlx_reshape(&gate_4, gate_t, &[_]c_int{ 1, nh, Ti, 1 }, 4, s));
            var gated = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&gated, attn, gate_4, s));
            _ = mlx.mlx_array_free(attn);
            attn = gated;
        }
        var ao = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ao);
        {
            var at = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(at);
            try mlx.check(mlx.mlx_transpose_axes(&at, attn, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
            try mlx.check(mlx.mlx_reshape(&ao, at, &[_]c_int{ 1, Ti, di }, 3, s));
        }
        const proj = try linBias(comp, allocator, ao, "{s}.transformer_1d_blocks.{d}.attn1.to_out.0", .{ prefix, blk }, s);
        defer _ = mlx.mlx_array_free(proj);
        var x1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&x1, x, proj, s));
        _ = mlx.mlx_array_free(x);
        x = x1;

        // ── Feed-forward ──
        const normed2 = try rmsAF(x, dim, 1e-6, s);
        defer _ = mlx.mlx_array_free(normed2);
        const ff0 = try linBias(comp, allocator, normed2, "{s}.transformer_1d_blocks.{d}.ff.net.0.proj", .{ prefix, blk }, s);
        defer _ = mlx.mlx_array_free(ff0);
        const ffg = try geluApprox(ff0, s);
        defer _ = mlx.mlx_array_free(ffg);
        const ff2 = try linBias(comp, allocator, ffg, "{s}.transformer_1d_blocks.{d}.ff.net.2", .{ prefix, blk }, s);
        defer _ = mlx.mlx_array_free(ff2);
        var x2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&x2, x, ff2, s));
        _ = mlx.mlx_array_free(x);
        x = x2;
        _ = mlx.mlx_array_eval(x);
    }

    const outn = try rmsAF(x, dim, 1e-6, s);
    _ = mlx.mlx_array_free(x);
    return outn;
}

// ════════════════════════════════════════════════════════════════════════
// Gemma-3-12B 49-layer hidden-state capture — the text encoder for the DiT
// conditioning (feeds connectorProject). Reproduces the reference
// get_all_hidden_states: state 0 = embed(ids)*sqrt(3840); states 1..48 = each
// decoder-layer residual output, with NO final model.norm. The reference passes
// ONE combined causal+pad mask to every layer, so no sliding-window masking
// applies — sliding vs full layers differ ONLY in the RoPE base (local 1e4 /
// global 1e6, full when (i+1)%6==0). Weights q4 g64; key prefix
// `language_model.model.*`.
// ════════════════════════════════════════════════════════════════════════

const GemmaCfg = struct {
    hidden: u32 = 3840,
    layers: u32 = 48,
    n_heads: u32 = 16,
    n_kv: u32 = 8,
    head_dim: u32 = 256,
    eps: f32 = 1e-6,
    qk_scale: f32 = 0.0625, // query_pre_attn_scalar(256)^-0.5
    theta_local: f32 = 10000.0,
    theta_global: f32 = 1000000.0,
};

fn gGet(w: *const model_mod.Weights, key: []const u8) !mlx.mlx_array {
    return w.get(key) orelse {
        log.err("[ltx-gemma] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingGemmaWeight;
    };
}

/// q4 linear: y = x @ dequant(<base>.weight).T  (affine g64 b4).
fn gQLin(w: *const model_mod.Weights, a: std.mem.Allocator, x: mlx.mlx_array, base: []const u8, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{base});
    defer a.free(wk);
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{base});
    defer a.free(sk);
    const bk = try std.fmt.allocPrint(a, "{s}.biases", .{base});
    defer a.free(bk);
    const wq = try gGet(w, wk);
    const sc = try gGet(w, sk);
    const bi = try gGet(w, bk);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(&o, x, wq, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
    return o;
}

/// Gemma RMSNorm: x_normed * (1 + w).
fn gRms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var wp1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wp1);
    try mlx.check(mlx.mlx_add(&wp1, w, one, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, wp1, eps, s));
    return out;
}

fn gReshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
fn gTranspose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}

/// One Gemma-3 decoder layer over [1,T,3840] with a precomputed combined mask.
fn gLayer(w: *const model_mod.Weights, a: std.mem.Allocator, idx: u32, h: mlx.mlx_array, mask: mlx.mlx_array, base: f32, cfg: GemmaCfg, s: S) !mlx.mlx_array {
    const pfx = try std.fmt.allocPrint(a, "language_model.model.layers.{d}", .{idx});
    defer a.free(pfx);
    const T: c_int = mlx.getShape(h)[1];
    const nh: c_int = @intCast(cfg.n_heads);
    const nkv: c_int = @intCast(cfg.n_kv);
    const hd: c_int = @intCast(cfg.head_dim);
    const eps = cfg.eps;

    // ── Attention ──
    const in_ln = try gGet(w, try fmtKey(a, pfx, "input_layernorm.weight"));
    const x = try gRms(h, in_ln, eps, s);
    defer _ = mlx.mlx_array_free(x);

    const q = try gQLin(w, a, x, try fmtKey(a, pfx, "self_attn.q_proj"), s);
    defer _ = mlx.mlx_array_free(q);
    const k = try gQLin(w, a, x, try fmtKey(a, pfx, "self_attn.k_proj"), s);
    defer _ = mlx.mlx_array_free(k);
    const v = try gQLin(w, a, x, try fmtKey(a, pfx, "self_attn.v_proj"), s);
    defer _ = mlx.mlx_array_free(v);

    const q4 = try gReshape(q, &[_]c_int{ 1, T, nh, hd }, s);
    defer _ = mlx.mlx_array_free(q4);
    const qt = try gTranspose(q4, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(qt);
    const k4 = try gReshape(k, &[_]c_int{ 1, T, nkv, hd }, s);
    defer _ = mlx.mlx_array_free(k4);
    const kt = try gTranspose(k4, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(kt);
    const v4 = try gReshape(v, &[_]c_int{ 1, T, nkv, hd }, s);
    defer _ = mlx.mlx_array_free(v4);
    const vt = try gTranspose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(vt);

    // q/k norm over head_dim (RMSNorm with 1+w), then RoPE.
    const qn_w = try gGet(w, try fmtKey(a, pfx, "self_attn.q_norm.weight"));
    const qn = try gRms(qt, qn_w, eps, s);
    defer _ = mlx.mlx_array_free(qn);
    const kn_w = try gGet(w, try fmtKey(a, pfx, "self_attn.k_norm.weight"));
    const kn = try gRms(kt, kn_w, eps, s);
    defer _ = mlx.mlx_array_free(kn);

    const base_opt = mlx.mlx_optional_float.some(base);
    var qr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qr);
    try mlx.check(mlx.mlx_fast_rope(&qr, qn, hd, false, base_opt, 1.0, 0, .{ .ctx = null }, s));
    var kr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kr);
    try mlx.check(mlx.mlx_fast_rope(&kr, kn, hd, false, base_opt, 1.0, 0, .{ .ctx = null }, s));

    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    const null_arr = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, cfg.qk_scale, "array", mask, null_arr, s));

    const at = try gTranspose(attn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(at);
    const af = try gReshape(at, &[_]c_int{ 1, T, nh * hd }, s);
    defer _ = mlx.mlx_array_free(af);
    const o = try gQLin(w, a, af, try fmtKey(a, pfx, "self_attn.o_proj"), s);
    defer _ = mlx.mlx_array_free(o);
    const pa_ln = try gGet(w, try fmtKey(a, pfx, "post_attention_layernorm.weight"));
    const on = try gRms(o, pa_ln, eps, s);
    defer _ = mlx.mlx_array_free(on);
    const h1 = try clipResidual(h, on, s);
    defer _ = mlx.mlx_array_free(h1);

    // ── MLP (gated GELU-approx) ──
    const pf_ln = try gGet(w, try fmtKey(a, pfx, "pre_feedforward_layernorm.weight"));
    const xm = try gRms(h1, pf_ln, eps, s);
    defer _ = mlx.mlx_array_free(xm);
    const gate = try gQLin(w, a, xm, try fmtKey(a, pfx, "mlp.gate_proj"), s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try gQLin(w, a, xm, try fmtKey(a, pfx, "mlp.up_proj"), s);
    defer _ = mlx.mlx_array_free(up);
    const gact = try geluApprox(gate, s);
    defer _ = mlx.mlx_array_free(gact);
    const ga = try mulArr(gact, up, s);
    defer _ = mlx.mlx_array_free(ga);
    const down = try gQLin(w, a, ga, try fmtKey(a, pfx, "mlp.down_proj"), s);
    defer _ = mlx.mlx_array_free(down);
    const po_ln = try gGet(w, try fmtKey(a, pfx, "post_feedforward_layernorm.weight"));
    const dn = try gRms(down, po_ln, eps, s);
    defer _ = mlx.mlx_array_free(dn);
    return clipResidual(h1, dn, s);
}

/// Format `<prefix>.<suffix>` into an arena-allocated key (caller's arena frees).
fn fmtKey(a: std.mem.Allocator, prefix: []const u8, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}.{s}", .{ prefix, suffix });
}

fn addArr(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, x, y, s));
    return o;
}
fn mulArr(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, y, s));
    return o;
}

/// Gemma-3 clip_residual: add in float32 then cast back to bf16 (clip to the
/// bf16 range is a no-op for normal activations but matches the reference's
/// rounding — the f32 intermediate, NOT a plain bf16 add, is what compounds
/// over 96 residual adds across 48 layers).
fn clipResidual(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var yf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yf);
    try mlx.check(mlx.mlx_astype(&yf, y, .float32, s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, xf, yf, s));
    const bound = mlx.mlx_array_new_float(3.3895314e38); // bf16 max
    defer _ = mlx.mlx_array_free(bound);
    const nbound = mlx.mlx_array_new_float(-3.3895314e38);
    defer _ = mlx.mlx_array_free(nbound);
    var lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo);
    try mlx.check(mlx.mlx_minimum(&lo, sum, bound, s));
    var hi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hi);
    try mlx.check(mlx.mlx_maximum(&hi, lo, nbound, s));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, hi, .bfloat16, s));
    return o;
}

/// Capture the 49 Gemma-3 hidden states for `ids` (length T, left-padded with
/// `pad_id`). Returns an owned slice of 49 arrays [1,T,3840]; caller frees each
/// + the slice. Weights loaded from `gemma_dir` (multi-shard, CPU stream).
pub fn gemmaCapture(io: std.Io, allocator: std.mem.Allocator, gemma_dir: []const u8, ids: []const i32, pad_id: i32, s: S) ![]mlx.mlx_array {
    const cfg = GemmaCfg{};
    const T: c_int = @intCast(ids.len);
    var w = try model_mod.loadWeights(io, allocator, gemma_dir);
    defer w.deinit();

    // Arena for the per-call key strings.
    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    var states = try allocator.alloc(mlx.mlx_array, cfg.layers + 1);
    var done: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < done) : (i += 1) _ = mlx.mlx_array_free(states[i]);
        allocator.free(states);
    }

    // ── Embedding (q4 table lookup → dequant → ×sqrt(3840)) ──
    const id_shape = [_]c_int{T};
    const ids_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const emb_w = try gGet(&w, "language_model.model.embed_tokens.weight");
    const emb_s = try gGet(&w, "language_model.model.embed_tokens.scales");
    const emb_b = try gGet(&w, "language_model.model.embed_tokens.biases");
    var rw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rw);
    try mlx.check(mlx.mlx_take_axis(&rw, emb_w, ids_arr, 0, s));
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_take_axis(&rs, emb_s, ids_arr, 0, s));
    var rb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rb);
    try mlx.check(mlx.mlx_take_axis(&rb, emb_b, ids_arr, 0, s));
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    const null_gs = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_dequantize(&deq, rw, rs, rb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", null_gs, .{ .value = .bfloat16, .has_value = true }, s));
    const hd: c_int = @intCast(cfg.hidden);
    const emb3 = try gReshape(deq, &[_]c_int{ 1, T, hd }, s);
    defer _ = mlx.mlx_array_free(emb3);
    const scale = mlx.mlx_array_new_float(@sqrt(@as(f32, @floatFromInt(cfg.hidden))));
    defer _ = mlx.mlx_array_free(scale);
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&h, emb3, scale, s));
    states[0] = h;
    done = 1;

    // ── Combined causal + left-pad mask [1,1,T,T] bf16 ──
    const tn: usize = ids.len;
    const mbuf = try allocator.alloc(f32, tn * tn);
    defer allocator.free(mbuf);
    const neg: f32 = -1e9;
    for (0..tn) |i| {
        for (0..tn) |j| {
            var m: f32 = if (j <= i) 0.0 else neg;
            if (ids[j] == pad_id) m += neg;
            mbuf[i * tn + j] = m;
        }
    }
    const mshape = [_]c_int{ 1, 1, T, T };
    const mask_f32 = mlx.mlx_array_new_data(mbuf.ptr, &mshape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

    // ── 48 decoder layers ──
    var li: u32 = 0;
    while (li < cfg.layers) : (li += 1) {
        const base = if ((li + 1) % 6 == 0) cfg.theta_global else cfg.theta_local;
        const nh = try gLayer(&w, a, li, states[li], mask, base, cfg, s);
        states[li + 1] = nh;
        done += 1;
        _ = mlx.mlx_array_eval(nh); // bound the lazy graph (GPU watchdog)
        _ = arena_inst.reset(.retain_capacity);
    }
    return states;
}

// ════════════════════════════════════════════════════════════════════════
// DiT conditioning: timestep sinusoidal embedding + AdaLayerNormSingle.
// Every transformer block + the output head consume these modulation params.
// (The 48-block joint DiT forward that uses them is the remaining work.)
// ════════════════════════════════════════════════════════════════════════

/// get_timestep_embedding(t_scaled, dim): sinusoidal, flip_sin_to_cos=True
/// (cos first), downscale_freq_shift=0, max_period=10000. Returns [1, dim].
fn ditTimestepSinusoid(t_scaled: f32, dim: u32, s: S) !mlx.mlx_array {
    const half: c_int = @intCast(dim / 2);
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0.0, @floatFromInt(dim / 2), 1.0, .float32, s));
    const coef: f32 = -@log(@as(f32, 10000.0)) / @as(f32, @floatFromInt(dim / 2));
    const cscal = mlx.mlx_array_new_float(coef);
    defer _ = mlx.mlx_array_free(cscal);
    var expo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expo);
    try mlx.check(mlx.mlx_multiply(&expo, ar, cscal, s));
    var freqs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(freqs);
    try mlx.check(mlx.mlx_exp(&freqs, expo, s));
    const tscal = mlx.mlx_array_new_float(t_scaled);
    defer _ = mlx.mlx_array_free(tscal);
    var args = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(args);
    try mlx.check(mlx.mlx_multiply(&args, freqs, tscal, s));
    var args2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(args2);
    try mlx.check(mlx.mlx_reshape(&args2, args, &[_]c_int{ 1, half }, 2, s));
    var cs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cs);
    try mlx.check(mlx.mlx_cos(&cs, args2, s));
    var sn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sn);
    try mlx.check(mlx.mlx_sin(&sn, args2, s));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, cs);
    _ = mlx.mlx_vector_array_append_value(vec, sn);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

const AdaLNOut = struct { params: mlx.mlx_array, embedded: mlx.mlx_array };

/// AdaLayerNormSingle: embedded = linear2(silu(linear1(t_sin))) [the timestep
/// MLP]; params = linear(silu(embedded)) → (B, num_params*dim). `prefix` is the
/// full key prefix, e.g. "transformer.adaln_single".
fn ditAdaLNSingle(comp: *const Component, alloc: std.mem.Allocator, t_sin: mlx.mlx_array, prefix: []const u8, s: S) !AdaLNOut {
    const e1 = try linBias(comp, alloc, t_sin, "{s}.emb.timestep_embedder.linear1", .{prefix}, s);
    defer _ = mlx.mlx_array_free(e1);
    const e1s = try silu(e1, s);
    defer _ = mlx.mlx_array_free(e1s);
    const embedded = try linBias(comp, alloc, e1s, "{s}.emb.timestep_embedder.linear2", .{prefix}, s);
    const es = try silu(embedded, s);
    defer _ = mlx.mlx_array_free(es);
    const params = try linBias(comp, alloc, es, "{s}.linear", .{prefix}, s);
    return .{ .params = params, .embedded = embedded };
}

// ════════════════════════════════════════════════════════════════════════
// The 48-block DiT (BasicAVTransformerBlock) — the last big video piece.
// Joint audio+video transformer: per block, 8 sublayers (video/audio self-attn,
// video/audio text cross-attn, a2v/v2a cross-modal attn, video/audio FF), each
// AdaLN-modulated. Weights q4 g64/b4 (to_q/k/v/out, ff, gate) + a bf16 linear
// `.bias`; adaLN scale_shift tables F32. Reference: transformer.py / attention.py.
// ════════════════════════════════════════════════════════════════════════

/// q4 linear over a Component: y = x @ dequant(<base>.weight).T + <base>.bias.
/// Mirrors `gQLin` (affine g64 b4) but reads from a Component and adds the
/// quantized Linear's separate bf16 `.bias` (present on every DiT projection).
fn dQLin(comp: *const Component, alloc: std.mem.Allocator, x: mlx.mlx_array, base: []const u8, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(alloc, "{s}.weight", .{base});
    defer alloc.free(wk);
    const sk = try std.fmt.allocPrint(alloc, "{s}.scales", .{base});
    defer alloc.free(sk);
    const bk = try std.fmt.allocPrint(alloc, "{s}.biases", .{base});
    defer alloc.free(bk);
    const lbk = try std.fmt.allocPrint(alloc, "{s}.bias", .{base});
    defer alloc.free(lbk);
    const wq = comp.get(wk) orelse return error.MissingDitWeight;
    const sc = comp.get(sk) orelse return error.MissingDitWeight;
    const bi = comp.get(bk) orelse return error.MissingDitWeight;
    var mm = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(&mm, x, wq, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
    if (comp.get(lbk)) |lb| {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&out, mm, lb, s));
        _ = mlx.mlx_array_free(mm);
        return out;
    }
    return mm;
}

/// AdaLN scalar unpack: `reshape(params,[1,P,dim]) + table[None]`. `params` is
/// (1, P*dim) bf16 from the global adaln_single; `table_key` is the per-block
/// (P, dim) F32 scale_shift table. Returns the combined [1, P, dim] (f32).
fn adalnCombine(comp: *const Component, alloc: std.mem.Allocator, params: mlx.mlx_array, table_key: []const u8, P: u32, dim: u32, s: S) !mlx.mlx_array {
    _ = alloc;
    const table = comp.get(table_key) orelse return error.MissingDitWeight; // [P_table, dim] f32
    const Pi: c_int = @intCast(P);
    const di: c_int = @intCast(dim);
    var p3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p3);
    try mlx.check(mlx.mlx_reshape(&p3, params, &[_]c_int{ 1, Pi, di }, 3, s));
    // Slice the table to its first P rows (the a2v tables carry a 5th gate row
    // unpacked separately): table[:P, :] → [1,P,dim] (reference table[None,:P,:]).
    var trows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(trows);
    try mlx.check(mlx.mlx_slice(&trows, table, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ Pi, di }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var t3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t3);
    try mlx.check(mlx.mlx_reshape(&t3, trows, &[_]c_int{ 1, Pi, di }, 3, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, p3, t3, s)); // bf16 + f32 → f32 (broadcast over batch=1)
    return out;
}

/// Slice row `i` of a combined adaLN array [1,P,dim] → [1,1,dim].
fn adalnRow(combined: mlx.mlx_array, i: u32, dim: u32, s: S) !mlx.mlx_array {
    const di: c_int = @intCast(dim);
    const ii: c_int = @intCast(i);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, combined, &[_]c_int{ 0, ii, 0 }, 3, &[_]c_int{ 1, ii + 1, di }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// `x * (1 + scale) + shift` (AdaLN modulation). scale/shift broadcast [1,1,dim].
fn modulate(x: mlx.mlx_array, scale: mlx.mlx_array, shift: mlx.mlx_array, s: S) !mlx.mlx_array {
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var sp1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp1);
    try mlx.check(mlx.mlx_add(&sp1, scale, one, s));
    var xs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xs);
    try mlx.check(mlx.mlx_multiply(&xs, x, sp1, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, xs, shift, s));
    return out;
}

/// compute_video_normed_sa: `rmsAF(video_hidden) * (1+scale_sa) + shift_sa`
/// where (shift_sa, scale_sa) = unpack rows 0,1 of (params + scale_shift_table).
/// `table_key` = e.g. "transformer.transformer_blocks.0.scale_shift_table".
fn ditNormedSA(comp: *const Component, alloc: std.mem.Allocator, video_hidden: mlx.mlx_array, params: mlx.mlx_array, table_key: []const u8, dim: u32, eps: f32, s: S) !mlx.mlx_array {
    const comb = try adalnCombine(comp, alloc, params, table_key, 9, dim, s);
    defer _ = mlx.mlx_array_free(comb);
    const shift = try adalnRow(comb, 0, dim, s);
    defer _ = mlx.mlx_array_free(shift);
    const scale = try adalnRow(comb, 1, dim, s);
    defer _ = mlx.mlx_array_free(scale);
    const normed = try rmsAF(video_hidden, dim, eps, s);
    defer _ = mlx.mlx_array_free(normed);
    return modulate(normed, scale, shift, s);
}

/// Port of `precompute_rope_freqs` (SPLIT): log-spaced fractional-position grid.
/// `pos` is a flat [N*A] f32 buffer (A position axes per token), `max_pos` is
/// [A]. Returns per-head cos/sin [1, num_heads, N, head_dim/2] in **f32** (the
/// DiT runs attention in f32, so cos/sin stay f32 to match the reference cast).
fn ditRope(alloc: std.mem.Allocator, pos: []const f32, N: u32, A: u32, num_heads: u32, head_dim: u32, max_pos: []const f32, s: S) !RopeCS {
    const theta: f32 = 10000.0;
    const inner_dim: u32 = num_heads * head_dim;
    const num_freqs: u32 = inner_dim / (2 * A);
    const expected: u32 = inner_dim / 2; // = num_heads * (head_dim/2)
    const pad: u32 = expected - num_freqs * A;
    const hd_half: u32 = head_dim / 2;

    const fi = try alloc.alloc(f32, num_freqs);
    defer alloc.free(fi);
    const denom: f32 = @floatFromInt(num_freqs - 1);
    for (0..num_freqs) |j| {
        const e: f32 = @as(f32, @floatFromInt(j)) / denom; // linspace 0..1
        fi[j] = std.math.pow(f32, theta, e) * (std.math.pi / 2.0);
    }

    const cos_buf = try alloc.alloc(f32, N * expected);
    defer alloc.free(cos_buf);
    const sin_buf = try alloc.alloc(f32, N * expected);
    defer alloc.free(sin_buf);
    for (0..N) |n| {
        const base = n * expected;
        for (0..pad) |c| {
            cos_buf[base + c] = 1.0; // cos(0)=1
            sin_buf[base + c] = 0.0; // sin(0)=0
        }
        for (0..num_freqs) |j| {
            for (0..A) |i| {
                const frac = pos[n * A + i] / max_pos[i];
                const sc = frac * 2.0 - 1.0;
                const ang = fi[j] * sc;
                const idx = base + pad + j * A + i;
                cos_buf[idx] = @cos(ang);
                sin_buf[idx] = @sin(ang);
            }
        }
    }
    const rshape = [_]c_int{ 1, @intCast(N), @intCast(num_heads), @intCast(hd_half) };
    const cos_d = mlx.mlx_array_new_data(cos_buf.ptr, &rshape, 4, .float32);
    defer _ = mlx.mlx_array_free(cos_d);
    const sin_d = mlx.mlx_array_new_data(sin_buf.ptr, &rshape, 4, .float32);
    defer _ = mlx.mlx_array_free(sin_d);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var cos_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos_t);
    try mlx.check(mlx.mlx_transpose_axes(&cos_t, cos_d, &perm, 4, s));
    var sin_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sin_t);
    try mlx.check(mlx.mlx_transpose_axes(&sin_t, sin_d, &perm, 4, s));
    // Materialize contiguous — the transpose is a lazy strided view; downstream
    // data-readback (and any non-stride-aware consumer) would otherwise see
    // source memory order.
    var cos_c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&cos_c, cos_t, false, s));
    var sin_c = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&sin_c, sin_t, false, s));
    return .{ .cos = cos_c, .sin = sin_c };
}

/// DiT q4 Attention (attention.py). q/k/v via dQLin; affine QK-norm over the full
/// inner_dim; optional split-RoPE (separate cos/sin for q vs k); SDPA(scale=hd^-.5,
/// no mask); per-head gate 2*sigmoid(to_gate_logits(x)); to_out. `cq/sq` and
/// `ck/sk` may be null (text cross-attn: use_rope=false). `x` is the query input,
/// `kv` the key/value input (== x for self-attn).
fn ditAttention(comp: *const Component, alloc: std.mem.Allocator, x: mlx.mlx_array, kv: mlx.mlx_array, prefix: []const u8, num_heads: u32, head_dim: u32, cq: ?mlx.mlx_array, sq: ?mlx.mlx_array, ck: ?mlx.mlx_array, sk: ?mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const nh: c_int = @intCast(num_heads);
    const hdi: c_int = @intCast(head_dim);
    const inner: c_int = @intCast(num_heads * head_dim);
    const Nq: c_int = mlx.getShape(x)[1];
    const Nk: c_int = mlx.getShape(kv)[1];
    const perm = [_]c_int{ 0, 2, 1, 3 };

    const qb = try std.fmt.allocPrint(alloc, "{s}.to_q", .{prefix});
    defer alloc.free(qb);
    const kb = try std.fmt.allocPrint(alloc, "{s}.to_k", .{prefix});
    defer alloc.free(kb);
    const vb = try std.fmt.allocPrint(alloc, "{s}.to_v", .{prefix});
    defer alloc.free(vb);
    const qnk = try std.fmt.allocPrint(alloc, "{s}.q_norm.weight", .{prefix});
    defer alloc.free(qnk);
    const knk = try std.fmt.allocPrint(alloc, "{s}.k_norm.weight", .{prefix});
    defer alloc.free(knk);

    var q = try dQLin(comp, alloc, x, qb, s);
    defer _ = mlx.mlx_array_free(q);
    var k = try dQLin(comp, alloc, kv, kb, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try dQLin(comp, alloc, kv, vb, s);
    defer _ = mlx.mlx_array_free(v);
    { // QK RMSNorm over full inner_dim
        const qn = try rmsW(q, comp.get(qnk) orelse return error.MissingDitWeight, eps, s);
        _ = mlx.mlx_array_free(q);
        q = qn;
        const kn = try rmsW(k, comp.get(knk) orelse return error.MissingDitWeight, eps, s);
        _ = mlx.mlx_array_free(k);
        k = kn;
    }
    var qh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qh);
    {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, q, &[_]c_int{ 1, Nq, nh, hdi }, 4, s));
        try mlx.check(mlx.mlx_transpose_axes(&qh, r, &perm, 4, s));
    }
    var kh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kh);
    {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, k, &[_]c_int{ 1, Nk, nh, hdi }, 4, s));
        try mlx.check(mlx.mlx_transpose_axes(&kh, r, &perm, 4, s));
    }
    var vh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vh);
    {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, v, &[_]c_int{ 1, Nk, nh, hdi }, 4, s));
        try mlx.check(mlx.mlx_transpose_axes(&vh, r, &perm, 4, s));
    }
    if (cq) |cqa| {
        const qr = try applyRopeSplit(qh, cqa, sq.?, s);
        _ = mlx.mlx_array_free(qh);
        qh = qr;
        const kr = try applyRopeSplit(kh, ck.?, sk.?, s);
        _ = mlx.mlx_array_free(kh);
        kh = kr;
    }
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const null_arr = mlx.mlx_array{ .ctx = null };
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn, qh, kh, vh, scale, "", null_arr, null_arr, s));
    { // per-head gate: 2*sigmoid(to_gate_logits(x)) [1,Nq,heads] → [1,heads,Nq,1]
        const gb = try std.fmt.allocPrint(alloc, "{s}.to_gate_logits", .{prefix});
        defer alloc.free(gb);
        const gl = try dQLin(comp, alloc, x, gb, s);
        defer _ = mlx.mlx_array_free(gl);
        var sg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sg);
        try mlx.check(mlx.mlx_sigmoid(&sg, gl, s));
        const two = mlx.mlx_array_new_float(2.0);
        defer _ = mlx.mlx_array_free(two);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_multiply(&gate, sg, two, s));
        var gt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gt);
        try mlx.check(mlx.mlx_transpose_axes(&gt, gate, &[_]c_int{ 0, 2, 1 }, 3, s));
        var g4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g4);
        try mlx.check(mlx.mlx_reshape(&g4, gt, &[_]c_int{ 1, nh, Nq, 1 }, 4, s));
        var gated = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&gated, attn, g4, s));
        _ = mlx.mlx_array_free(attn);
        attn = gated;
    }
    var ao = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ao);
    {
        var at = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(at);
        try mlx.check(mlx.mlx_transpose_axes(&at, attn, &perm, 4, s));
        try mlx.check(mlx.mlx_reshape(&ao, at, &[_]c_int{ 1, Nq, inner }, 3, s));
    }
    const ob = try std.fmt.allocPrint(alloc, "{s}.to_out", .{prefix});
    defer alloc.free(ob);
    return dQLin(comp, alloc, ao, ob, s);
}

/// DiT FeedForward: proj_out(gelu_approx(proj_in(x))). Both q4.
fn ditFF(comp: *const Component, alloc: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, s: S) !mlx.mlx_array {
    const ib = try std.fmt.allocPrint(alloc, "{s}.proj_in", .{prefix});
    defer alloc.free(ib);
    const ob = try std.fmt.allocPrint(alloc, "{s}.proj_out", .{prefix});
    defer alloc.free(ob);
    const h = try dQLin(comp, alloc, x, ib, s);
    defer _ = mlx.mlx_array_free(h);
    const g = try geluApprox(h, s);
    defer _ = mlx.mlx_array_free(g);
    return dQLin(comp, alloc, g, ob, s);
}

/// AV-cross gate: `(gate_params + table[4,:])[:, None, :]` → [1,1,dim] (f32).
fn avGate(comp: *const Component, gate_params: mlx.mlx_array, table_key: []const u8, dim: u32, s: S) !mlx.mlx_array {
    const di: c_int = @intCast(dim);
    const table = comp.get(table_key) orelse return error.MissingDitWeight; // [5,dim]
    var row4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row4);
    try mlx.check(mlx.mlx_slice(&row4, table, &[_]c_int{ 4, 0 }, 2, &[_]c_int{ 5, di }, 2, &[_]c_int{ 1, 1 }, 2, s)); // [1,dim]
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_add(&g, gate_params, row4, s)); // [1,dim] f32
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, g, &[_]c_int{ 1, 1, di }, 3, s));
    return out;
}

/// `x + delta * gate` (residual with AdaLN gate). Frees `delta`; returns new array.
fn residGate(x: mlx.mlx_array, delta: mlx.mlx_array, gate: mlx.mlx_array, s: S) !mlx.mlx_array {
    defer _ = mlx.mlx_array_free(delta);
    var dg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dg);
    try mlx.check(mlx.mlx_multiply(&dg, delta, gate, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, x, dg, s));
    return out;
}

/// One BasicAVTransformerBlock (transformer.py). `vh_in`/`ah_in` are borrowed
/// (caller frees); returns freshly-owned (video_hidden, audio_hidden). All adaLN
/// params are (1, P*dim) bf16; rope cos/sin are f32 [1,heads,N,hd/2].
const BlockRope = struct { vcos: mlx.mlx_array, vsin: mlx.mlx_array, acos: mlx.mlx_array, asin: mlx.mlx_array, vxcos: mlx.mlx_array, vxsin: mlx.mlx_array, axcos: mlx.mlx_array, axsin: mlx.mlx_array };
const BlockOut = struct { v: mlx.mlx_array, a: mlx.mlx_array };

fn ditBlock(
    comp: *const Component,
    alloc: std.mem.Allocator,
    cfg: LtxConfig,
    blk: u32,
    vh_in: mlx.mlx_array,
    ah_in: mlx.mlx_array,
    video_adaln_params: mlx.mlx_array,
    audio_adaln_params: mlx.mlx_array,
    video_prompt_params: mlx.mlx_array,
    audio_prompt_params: mlx.mlx_array,
    av_ca_video_params: mlx.mlx_array,
    av_ca_audio_params: mlx.mlx_array,
    av_a2v_gate_params: mlx.mlx_array,
    av_v2a_gate_params: mlx.mlx_array,
    video_text: mlx.mlx_array,
    audio_text: mlx.mlx_array,
    rope: BlockRope,
    s: S,
) !BlockOut {
    const vdim = cfg.videoDim(); // 4096
    const adim = cfg.audioDim(); // 2048
    const eps = cfg.norm_eps;
    const P = "transformer.transformer_blocks";

    const pfx = struct {
        fn k(a: std.mem.Allocator, b: u32, name: []const u8) ![]u8 {
            return std.fmt.allocPrint(a, "{s}.{d}.{s}", .{ P, b, name });
        }
    };

    // ── adaLN unpack (scalar mode) ──
    const sst = try pfx.k(alloc, blk, "scale_shift_table");
    defer alloc.free(sst);
    const vcomb = try adalnCombine(comp, alloc, video_adaln_params, sst, 9, vdim, s);
    defer _ = mlx.mlx_array_free(vcomb);
    const asst = try pfx.k(alloc, blk, "audio_scale_shift_table");
    defer alloc.free(asst);
    const acomb = try adalnCombine(comp, alloc, audio_adaln_params, asst, 9, adim, s);
    defer _ = mlx.mlx_array_free(acomb);
    const avv_k = try pfx.k(alloc, blk, "scale_shift_table_a2v_ca_video");
    defer alloc.free(avv_k);
    const avv = try adalnCombine(comp, alloc, av_ca_video_params, avv_k, 4, vdim, s);
    defer _ = mlx.mlx_array_free(avv);
    const ava_k = try pfx.k(alloc, blk, "scale_shift_table_a2v_ca_audio");
    defer alloc.free(ava_k);
    const ava = try adalnCombine(comp, alloc, av_ca_audio_params, ava_k, 4, adim, s);
    defer _ = mlx.mlx_array_free(ava);

    // video 9 rows: shift_sa,scale_sa,gate_sa, shift_ff,scale_ff,gate_ff, shift_ca,scale_ca,gate_ca
    const v_shift_sa = try adalnRow(vcomb, 0, vdim, s);
    defer _ = mlx.mlx_array_free(v_shift_sa);
    const v_scale_sa = try adalnRow(vcomb, 1, vdim, s);
    defer _ = mlx.mlx_array_free(v_scale_sa);
    const v_gate_sa = try adalnRow(vcomb, 2, vdim, s);
    defer _ = mlx.mlx_array_free(v_gate_sa);
    const v_shift_ff = try adalnRow(vcomb, 3, vdim, s);
    defer _ = mlx.mlx_array_free(v_shift_ff);
    const v_scale_ff = try adalnRow(vcomb, 4, vdim, s);
    defer _ = mlx.mlx_array_free(v_scale_ff);
    const v_gate_ff = try adalnRow(vcomb, 5, vdim, s);
    defer _ = mlx.mlx_array_free(v_gate_ff);
    const v_shift_ca = try adalnRow(vcomb, 6, vdim, s);
    defer _ = mlx.mlx_array_free(v_shift_ca);
    const v_scale_ca = try adalnRow(vcomb, 7, vdim, s);
    defer _ = mlx.mlx_array_free(v_scale_ca);
    const v_gate_ca = try adalnRow(vcomb, 8, vdim, s);
    defer _ = mlx.mlx_array_free(v_gate_ca);
    // audio 9 rows
    const a_shift_sa = try adalnRow(acomb, 0, adim, s);
    defer _ = mlx.mlx_array_free(a_shift_sa);
    const a_scale_sa = try adalnRow(acomb, 1, adim, s);
    defer _ = mlx.mlx_array_free(a_scale_sa);
    const a_gate_sa = try adalnRow(acomb, 2, adim, s);
    defer _ = mlx.mlx_array_free(a_gate_sa);
    const a_shift_ff = try adalnRow(acomb, 3, adim, s);
    defer _ = mlx.mlx_array_free(a_shift_ff);
    const a_scale_ff = try adalnRow(acomb, 4, adim, s);
    defer _ = mlx.mlx_array_free(a_scale_ff);
    const a_gate_ff = try adalnRow(acomb, 5, adim, s);
    defer _ = mlx.mlx_array_free(a_gate_ff);
    const a_shift_ca = try adalnRow(acomb, 6, adim, s);
    defer _ = mlx.mlx_array_free(a_shift_ca);
    const a_scale_ca = try adalnRow(acomb, 7, adim, s);
    defer _ = mlx.mlx_array_free(a_scale_ca);
    const a_gate_ca = try adalnRow(acomb, 8, adim, s);
    defer _ = mlx.mlx_array_free(a_gate_ca);
    // av cross video 4 rows (SCALE-first): scale_a2v,shift_a2v,scale_v2a,shift_v2a
    const av_v_scale_a2v = try adalnRow(avv, 0, vdim, s);
    defer _ = mlx.mlx_array_free(av_v_scale_a2v);
    const av_v_shift_a2v = try adalnRow(avv, 1, vdim, s);
    defer _ = mlx.mlx_array_free(av_v_shift_a2v);
    const av_v_scale_v2a = try adalnRow(avv, 2, vdim, s);
    defer _ = mlx.mlx_array_free(av_v_scale_v2a);
    const av_v_shift_v2a = try adalnRow(avv, 3, vdim, s);
    defer _ = mlx.mlx_array_free(av_v_shift_v2a);
    const av_a_scale_a2v = try adalnRow(ava, 0, adim, s);
    defer _ = mlx.mlx_array_free(av_a_scale_a2v);
    const av_a_shift_a2v = try adalnRow(ava, 1, adim, s);
    defer _ = mlx.mlx_array_free(av_a_shift_a2v);
    const av_a_scale_v2a = try adalnRow(ava, 2, adim, s);
    defer _ = mlx.mlx_array_free(av_a_scale_v2a);
    const av_a_shift_v2a = try adalnRow(ava, 3, adim, s);
    defer _ = mlx.mlx_array_free(av_a_shift_v2a);
    // av gates
    const av_v_gate_a2v = try avGate(comp, av_a2v_gate_params, avv_k, vdim, s);
    defer _ = mlx.mlx_array_free(av_v_gate_a2v);
    const av_a_gate_v2a = try avGate(comp, av_v2a_gate_params, ava_k, adim, s);
    defer _ = mlx.mlx_array_free(av_a_gate_v2a);

    var vh = vh_in;
    var vh_owned = false;
    var ah = ah_in;
    var ah_owned = false;

    // ── 1. video self-attn ──
    {
        const pn = try rmsAF(vh, vdim, eps, s);
        defer _ = mlx.mlx_array_free(pn);
        const normed = try modulate(pn, v_scale_sa, v_shift_sa, s);
        defer _ = mlx.mlx_array_free(normed);
        const aprefix = try pfx.k(alloc, blk, "attn1");
        defer alloc.free(aprefix);
        const out = try ditAttention(comp, alloc, normed, normed, aprefix, cfg.num_heads, cfg.head_dim, rope.vcos, rope.vsin, rope.vcos, rope.vsin, eps, s);
        const nv = try residGate(vh, out, v_gate_sa, s);
        if (vh_owned) _ = mlx.mlx_array_free(vh);
        vh = nv;
        vh_owned = true;
    }
    // ── 2. audio self-attn ──
    {
        const pn = try rmsAF(ah, adim, eps, s);
        defer _ = mlx.mlx_array_free(pn);
        const normed = try modulate(pn, a_scale_sa, a_shift_sa, s);
        defer _ = mlx.mlx_array_free(normed);
        const aprefix = try pfx.k(alloc, blk, "audio_attn1");
        defer alloc.free(aprefix);
        const out = try ditAttention(comp, alloc, normed, normed, aprefix, cfg.audio_num_heads, cfg.audio_head_dim, rope.acos, rope.asin, rope.acos, rope.asin, eps, s);
        const na = try residGate(ah, out, a_gate_sa, s);
        if (ah_owned) _ = mlx.mlx_array_free(ah);
        ah = na;
        ah_owned = true;
    }
    // ── 3. video text cross-attn ──
    {
        const pn = try rmsAF(vh, vdim, eps, s);
        defer _ = mlx.mlx_array_free(pn);
        const normed = try modulate(pn, v_scale_ca, v_shift_ca, s);
        defer _ = mlx.mlx_array_free(normed);
        const pk = try pfx.k(alloc, blk, "prompt_scale_shift_table");
        defer alloc.free(pk);
        const vp = try adalnCombine(comp, alloc, video_prompt_params, pk, 2, vdim, s);
        defer _ = mlx.mlx_array_free(vp);
        const vp_shift = try adalnRow(vp, 0, vdim, s);
        defer _ = mlx.mlx_array_free(vp_shift);
        const vp_scale = try adalnRow(vp, 1, vdim, s);
        defer _ = mlx.mlx_array_free(vp_scale);
        const text = try modulate(video_text, vp_scale, vp_shift, s);
        defer _ = mlx.mlx_array_free(text);
        const aprefix = try pfx.k(alloc, blk, "attn2");
        defer alloc.free(aprefix);
        const out = try ditAttention(comp, alloc, normed, text, aprefix, cfg.num_heads, cfg.head_dim, null, null, null, null, eps, s);
        const nv = try residGate(vh, out, v_gate_ca, s);
        _ = mlx.mlx_array_free(vh);
        vh = nv;
    }
    // ── 4. audio text cross-attn ──
    {
        const pn = try rmsAF(ah, adim, eps, s);
        defer _ = mlx.mlx_array_free(pn);
        const normed = try modulate(pn, a_scale_ca, a_shift_ca, s);
        defer _ = mlx.mlx_array_free(normed);
        const pk = try pfx.k(alloc, blk, "audio_prompt_scale_shift_table");
        defer alloc.free(pk);
        const ap = try adalnCombine(comp, alloc, audio_prompt_params, pk, 2, adim, s);
        defer _ = mlx.mlx_array_free(ap);
        const ap_shift = try adalnRow(ap, 0, adim, s);
        defer _ = mlx.mlx_array_free(ap_shift);
        const ap_scale = try adalnRow(ap, 1, adim, s);
        defer _ = mlx.mlx_array_free(ap_scale);
        const text = try modulate(audio_text, ap_scale, ap_shift, s);
        defer _ = mlx.mlx_array_free(text);
        const aprefix = try pfx.k(alloc, blk, "audio_attn2");
        defer alloc.free(aprefix);
        const out = try ditAttention(comp, alloc, normed, text, aprefix, cfg.audio_num_heads, cfg.audio_head_dim, null, null, null, null, eps, s);
        const na = try residGate(ah, out, a_gate_ca, s);
        _ = mlx.mlx_array_free(ah);
        ah = na;
    }
    // ── 5-6. AV cross-modal (shared norms) ──
    {
        const vn3 = try rmsAF(vh, vdim, eps, s);
        defer _ = mlx.mlx_array_free(vn3);
        const an3 = try rmsAF(ah, adim, eps, s);
        defer _ = mlx.mlx_array_free(an3);
        // A2V: q from video, kv from audio
        const vq = try modulate(vn3, av_v_scale_a2v, av_v_shift_a2v, s);
        defer _ = mlx.mlx_array_free(vq);
        const akv = try modulate(an3, av_a_scale_a2v, av_a_shift_a2v, s);
        defer _ = mlx.mlx_array_free(akv);
        const a2v_k = try pfx.k(alloc, blk, "audio_to_video_attn");
        defer alloc.free(a2v_k);
        const a2v = try ditAttention(comp, alloc, vq, akv, a2v_k, cfg.audio_num_heads, cfg.audio_head_dim, rope.vxcos, rope.vxsin, rope.axcos, rope.axsin, eps, s);
        const nv = try residGate(vh, a2v, av_v_gate_a2v, s);
        _ = mlx.mlx_array_free(vh);
        vh = nv;
        // V2A: q from audio, kv from video (pre-A2V norms)
        const aq = try modulate(an3, av_a_scale_v2a, av_a_shift_v2a, s);
        defer _ = mlx.mlx_array_free(aq);
        const vkv = try modulate(vn3, av_v_scale_v2a, av_v_shift_v2a, s);
        defer _ = mlx.mlx_array_free(vkv);
        const v2a_k = try pfx.k(alloc, blk, "video_to_audio_attn");
        defer alloc.free(v2a_k);
        const v2a = try ditAttention(comp, alloc, aq, vkv, v2a_k, cfg.audio_num_heads, cfg.audio_head_dim, rope.axcos, rope.axsin, rope.vxcos, rope.vxsin, eps, s);
        const na = try residGate(ah, v2a, av_a_gate_v2a, s);
        _ = mlx.mlx_array_free(ah);
        ah = na;
    }
    // ── 7. video FF ──
    {
        const pn = try rmsAF(vh, vdim, eps, s);
        defer _ = mlx.mlx_array_free(pn);
        const normed = try modulate(pn, v_scale_ff, v_shift_ff, s);
        defer _ = mlx.mlx_array_free(normed);
        const ffk = try pfx.k(alloc, blk, "ff");
        defer alloc.free(ffk);
        const out = try ditFF(comp, alloc, normed, ffk, s);
        const nv = try residGate(vh, out, v_gate_ff, s);
        _ = mlx.mlx_array_free(vh);
        vh = nv;
    }
    // ── 8. audio FF ──
    {
        const pn = try rmsAF(ah, adim, eps, s);
        defer _ = mlx.mlx_array_free(pn);
        const normed = try modulate(pn, a_scale_ff, a_shift_ff, s);
        defer _ = mlx.mlx_array_free(normed);
        const ffk = try pfx.k(alloc, blk, "audio_ff");
        defer alloc.free(ffk);
        const out = try ditFF(comp, alloc, normed, ffk, s);
        const na = try residGate(ah, out, a_gate_ff, s);
        _ = mlx.mlx_array_free(ah);
        ah = na;
    }
    return .{ .v = vh, .a = ah };
}

/// Affine-free LayerNorm over the last axis (mx.fast.layer_norm(weight=None,
/// bias=None)); pass ones/zeros since mlx-c crashes on null weight/bias.
fn layerNormAF(x: mlx.mlx_array, dim: u32, eps: f32, s: S) !mlx.mlx_array {
    const ov = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(ov);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_broadcast_to(&w, ov, &[_]c_int{@intCast(dim)}, 1, s));
    const zv = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zv);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_broadcast_to(&b, zv, &[_]c_int{@intCast(dim)}, 1, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, eps, s));
    return out;
}

/// Output head (_output_block): affine-free LayerNorm modulated by
/// `(scale_shift_table[2,dim] + embedded_timestep)`, then proj (bf16) → 128 ch.
fn outputBlock(comp: *const Component, alloc: std.mem.Allocator, x: mlx.mlx_array, embedded_ts: mlx.mlx_array, ss_key: []const u8, proj_key: []const u8, dim: u32, eps: f32, s: S) !mlx.mlx_array {
    const di: c_int = @intCast(dim);
    const table = comp.get(ss_key) orelse return error.MissingDitWeight; // [2,dim] f32
    var emb3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb3);
    try mlx.check(mlx.mlx_reshape(&emb3, embedded_ts, &[_]c_int{ 1, 1, di }, 3, s)); // [1,1,dim]
    var shift_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shift_row);
    try mlx.check(mlx.mlx_slice(&shift_row, table, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, di }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var scale_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scale_row);
    try mlx.check(mlx.mlx_slice(&scale_row, table, &[_]c_int{ 1, 0 }, 2, &[_]c_int{ 2, di }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shift);
    try mlx.check(mlx.mlx_add(&shift, shift_row, emb3, s)); // [1,1,dim]
    var scale = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scale);
    try mlx.check(mlx.mlx_add(&scale, scale_row, emb3, s));
    const ln = try layerNormAF(x, dim, eps, s);
    defer _ = mlx.mlx_array_free(ln);
    const mod = try modulate(ln, scale, shift, s);
    defer _ = mlx.mlx_array_free(mod);
    return linBias(comp, alloc, mod, "{s}", .{proj_key}, s);
}

/// Full DiT forward (LTXModel.__call__): patchify → adaLN sets → rope → 48
/// blocks (mx.eval every 8) → output head. Returns the VELOCITY (v) prediction;
/// the sampler converts to x0. `video_pos` is [Nv*3] f32, `audio_pos` is [Na] f32.
pub fn ditForward(
    comp: *const Component,
    alloc: std.mem.Allocator,
    cfg: LtxConfig,
    video_latent: mlx.mlx_array,
    audio_latent: mlx.mlx_array,
    sigma: f32,
    video_text: mlx.mlx_array,
    audio_text: mlx.mlx_array,
    video_pos: []const f32,
    audio_pos: []const f32,
    s: S,
) !BlockOut {
    const vdim = cfg.videoDim();
    const adim = cfg.audioDim();
    const eps = cfg.norm_eps;
    const Nv: u32 = @intCast(video_pos.len / 3);
    const Na: u32 = @intCast(audio_pos.len);

    // ── patchify ──
    var vh = try linBias(comp, alloc, video_latent, "{s}", .{"transformer.patchify_proj"}, s);
    var ah = try linBias(comp, alloc, audio_latent, "{s}", .{"transformer.audio_patchify_proj"}, s);

    // ── timestep embeddings → adaLN param sets ──
    const t_sin = try ditTimestepSinusoid(sigma * cfg.timestep_scale, 256, s); // sigma*1000
    defer _ = mlx.mlx_array_free(t_sin);
    const t_sin_gate = try ditTimestepSinusoid(sigma * 1.0, 256, s); // av_ca gate scale (×1)
    defer _ = mlx.mlx_array_free(t_sin_gate);

    const v_ada = try ditAdaLNSingle(comp, alloc, t_sin, "transformer.adaln_single", s);
    defer _ = mlx.mlx_array_free(v_ada.params);
    defer _ = mlx.mlx_array_free(v_ada.embedded);
    const a_ada = try ditAdaLNSingle(comp, alloc, t_sin, "transformer.audio_adaln_single", s);
    defer _ = mlx.mlx_array_free(a_ada.params);
    defer _ = mlx.mlx_array_free(a_ada.embedded);
    const v_avca = try ditAdaLNSingle(comp, alloc, t_sin, "transformer.av_ca_video_scale_shift_adaln_single", s);
    defer _ = mlx.mlx_array_free(v_avca.params);
    defer _ = mlx.mlx_array_free(v_avca.embedded);
    const a_avca = try ditAdaLNSingle(comp, alloc, t_sin, "transformer.av_ca_audio_scale_shift_adaln_single", s);
    defer _ = mlx.mlx_array_free(a_avca.params);
    defer _ = mlx.mlx_array_free(a_avca.embedded);
    const a2v_gate = try ditAdaLNSingle(comp, alloc, t_sin_gate, "transformer.av_ca_a2v_gate_adaln_single", s);
    defer _ = mlx.mlx_array_free(a2v_gate.params);
    defer _ = mlx.mlx_array_free(a2v_gate.embedded);
    const v2a_gate = try ditAdaLNSingle(comp, alloc, t_sin_gate, "transformer.av_ca_v2a_gate_adaln_single", s);
    defer _ = mlx.mlx_array_free(v2a_gate.params);
    defer _ = mlx.mlx_array_free(v2a_gate.embedded);
    const v_prompt = try ditAdaLNSingle(comp, alloc, t_sin, "transformer.prompt_adaln_single", s);
    defer _ = mlx.mlx_array_free(v_prompt.params);
    defer _ = mlx.mlx_array_free(v_prompt.embedded);
    const a_prompt = try ditAdaLNSingle(comp, alloc, t_sin, "transformer.audio_prompt_adaln_single", s);
    defer _ = mlx.mlx_array_free(a_prompt.params);
    defer _ = mlx.mlx_array_free(a_prompt.embedded);

    // ── rope (temporal-only axis for cross) ──
    const vtmp = try alloc.alloc(f32, Nv);
    defer alloc.free(vtmp);
    for (0..Nv) |n| vtmp[n] = video_pos[n * 3 + 0];
    const max_v = [_]f32{ 20.0, 2048.0, 2048.0 };
    const max_1 = [_]f32{20.0}; // cross_pe_max_pos = max(video[0], audio[0]) = 20
    const rv = try ditRope(alloc, video_pos, Nv, 3, cfg.num_heads, cfg.head_dim, &max_v, s);
    defer _ = mlx.mlx_array_free(rv.cos);
    defer _ = mlx.mlx_array_free(rv.sin);
    const ra = try ditRope(alloc, audio_pos, Na, 1, cfg.audio_num_heads, cfg.audio_head_dim, &max_1, s);
    defer _ = mlx.mlx_array_free(ra.cos);
    defer _ = mlx.mlx_array_free(ra.sin);
    const rvx = try ditRope(alloc, vtmp, Nv, 1, cfg.audio_num_heads, cfg.audio_head_dim, &max_1, s);
    defer _ = mlx.mlx_array_free(rvx.cos);
    defer _ = mlx.mlx_array_free(rvx.sin);
    const rax = try ditRope(alloc, audio_pos, Na, 1, cfg.audio_num_heads, cfg.audio_head_dim, &max_1, s);
    defer _ = mlx.mlx_array_free(rax.cos);
    defer _ = mlx.mlx_array_free(rax.sin);
    const rope = BlockRope{ .vcos = rv.cos, .vsin = rv.sin, .acos = ra.cos, .asin = ra.sin, .vxcos = rvx.cos, .vxsin = rvx.sin, .axcos = rax.cos, .axsin = rax.sin };

    // ── 48 blocks (mx.eval every 8 to bound the Metal command buffer) ──
    var blk: u32 = 0;
    while (blk < cfg.num_layers) : (blk += 1) {
        const out = try ditBlock(comp, alloc, cfg, blk, vh, ah, v_ada.params, a_ada.params, v_prompt.params, a_prompt.params, v_avca.params, a_avca.params, a2v_gate.params, v2a_gate.params, video_text, audio_text, rope, s);
        _ = mlx.mlx_array_free(vh);
        _ = mlx.mlx_array_free(ah);
        vh = out.v;
        ah = out.a;
        if ((blk + 1) % 8 == 0) {
            _ = mlx.mlx_array_eval(vh);
            _ = mlx.mlx_array_eval(ah);
        }
    }

    // ── output head → velocity ──
    const vout = try outputBlock(comp, alloc, vh, v_ada.embedded, "transformer.scale_shift_table", "transformer.proj_out", vdim, eps, s);
    _ = mlx.mlx_array_free(vh);
    const aout = try outputBlock(comp, alloc, ah, a_ada.embedded, "transformer.audio_scale_shift_table", "transformer.audio_proj_out", adim, eps, s);
    _ = mlx.mlx_array_free(ah);
    return .{ .v = vout, .a = aout };
}

// ════════════════════════════════════════════════════════════════════════
// Guided Euler sampler (CFG-only slice). Reference: ltx_pipelines_mlx
// ti2vid_one_stage + utils/samplers.guided_denoise_loop + scheduler +
// guiders + mlx_arsenal.diffusion.{dynamic_shift_schedule, euler_step}.
// ════════════════════════════════════════════════════════════════════════

/// LTX-2 token-adaptive flow-matching sigma schedule (dynamic_shift_schedule).
/// Returns num_steps+1 sigmas ending at 0.0. Caller frees the slice.
pub fn dynamicShiftSchedule(alloc: std.mem.Allocator, num_steps: u32, num_tokens: u32, base_shift: f64, max_shift: f64, base_tokens: f64, max_tokens: f64, stretch: bool, terminal: f64) ![]f32 {
    const n = num_steps + 1;
    const out = try alloc.alloc(f32, n);
    const slope = (max_shift - base_shift) / (max_tokens - base_tokens);
    const intercept = base_shift - slope * base_tokens;
    const sigma_shift = @as(f64, @floatFromInt(num_tokens)) * slope + intercept;
    const es = @exp(sigma_shift);
    var sig = try alloc.alloc(f64, n);
    defer alloc.free(sig);
    for (0..n) |i| {
        const lin = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num_steps)); // linspace(1,0)
        if (lin == 0.0) {
            sig[i] = 0.0;
        } else {
            sig[i] = es / (es + (1.0 / lin - 1.0));
        }
    }
    if (stretch) {
        // last non-zero sigma is index num_steps-1 (index num_steps is 0)
        const last_nz = sig[num_steps - 1];
        const scale_factor = (1.0 - last_nz) / (1.0 - terminal);
        if (scale_factor != 0.0) {
            for (0..n) |i| {
                if (sig[i] != 0.0) sig[i] = 1.0 - (1.0 - sig[i]) / scale_factor;
            }
        }
    }
    for (0..n) |i| out[i] = @floatCast(sig[i]);
    return out;
}

/// Single Euler step on an x0-prediction model:
/// `x_{t-1} = x + (sigma_next - sigma)*(x - x0)/sigma` (sigma==0 → x0).
fn eulerStep(x: mlx.mlx_array, x0: mlx.mlx_array, sigma: f32, sigma_next: f32, s: S) !mlx.mlx_array {
    if (sigma == 0.0) {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&c, x0));
        return c;
    }
    var xm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xm);
    try mlx.check(mlx.mlx_subtract(&xm, x, x0, s));
    const inv = mlx.mlx_array_new_float(1.0 / sigma);
    defer _ = mlx.mlx_array_free(inv);
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_multiply(&d, xm, inv, s));
    const coef = mlx.mlx_array_new_float(sigma_next - sigma);
    defer _ = mlx.mlx_array_free(coef);
    var step = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(step);
    try mlx.check(mlx.mlx_multiply(&step, d, coef, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, x, step, s));
    return out;
}

/// Global (population) variance over all elements — mean((x - mean(x))^2).
fn varGlobal(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_mean(&m, x, false, s));
    var xm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xm);
    try mlx.check(mlx.mlx_subtract(&xm, x, m, s));
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, xm, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_mean(&out, sq, false, s));
    return out;
}

/// CFG guider with optional norm-preserving rescale (STG/modality dropped):
/// `pred = cond + (cfg-1)*(cond-uncond)`; if rescale≠0,
/// `pred *= rescale*sqrt(var(cond))/(sqrt(var(pred))+1e-8) + (1-rescale)`.
fn cfgGuide(cond: mlx.mlx_array, uncond: mlx.mlx_array, cfg_scale: f32, rescale: f32, s: S) !mlx.mlx_array {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, cond, uncond, s));
    const w = mlx.mlx_array_new_float(cfg_scale - 1.0);
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_multiply(&sc, diff, w, s));
    var pred = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&pred, cond, sc, s));
    if (rescale != 0.0) {
        const vc = try varGlobal(cond, s);
        defer _ = mlx.mlx_array_free(vc);
        const vp = try varGlobal(pred, s);
        defer _ = mlx.mlx_array_free(vp);
        var scd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scd);
        try mlx.check(mlx.mlx_sqrt(&scd, vc, s));
        var spd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(spd);
        try mlx.check(mlx.mlx_sqrt(&spd, vp, s));
        const eps_a = mlx.mlx_array_new_float(1e-8);
        defer _ = mlx.mlx_array_free(eps_a);
        var spe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(spe);
        try mlx.check(mlx.mlx_add(&spe, spd, eps_a, s));
        var factor = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(factor);
        try mlx.check(mlx.mlx_divide(&factor, scd, spe, s));
        const rs = mlx.mlx_array_new_float(rescale);
        defer _ = mlx.mlx_array_free(rs);
        var f2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f2);
        try mlx.check(mlx.mlx_multiply(&f2, factor, rs, s));
        const one_minus = mlx.mlx_array_new_float(1.0 - rescale);
        defer _ = mlx.mlx_array_free(one_minus);
        var f3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f3);
        try mlx.check(mlx.mlx_add(&f3, f2, one_minus, s));
        var pr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&pr, pred, f3, s));
        _ = mlx.mlx_array_free(pred);
        pred = pr;
    }
    return pred;
}

/// x0 prediction (X0Model): x0 = x_t - sigma*v, where v = ditForward(...).
/// Returns bf16 x0 (matches the reference cast back to latent dtype).
fn ditX0(comp: *const Component, alloc: std.mem.Allocator, cfg: LtxConfig, vx: mlx.mlx_array, ax: mlx.mlx_array, sigma: f32, vtext: mlx.mlx_array, atext: mlx.mlx_array, vpos: []const f32, apos: []const f32, s: S) !BlockOut {
    const vel = try ditForward(comp, alloc, cfg, vx, ax, sigma, vtext, atext, vpos, apos, s);
    defer _ = mlx.mlx_array_free(vel.v);
    defer _ = mlx.mlx_array_free(vel.a);
    const sig = mlx.mlx_array_new_float(sigma);
    defer _ = mlx.mlx_array_free(sig);
    var sv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sv);
    try mlx.check(mlx.mlx_multiply(&sv, vel.v, sig, s)); // bf16*f32 → f32
    var x0vf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x0vf);
    try mlx.check(mlx.mlx_subtract(&x0vf, vx, sv, s));
    var x0v = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x0v, x0vf, .bfloat16, s));
    var sa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa);
    try mlx.check(mlx.mlx_multiply(&sa, vel.a, sig, s));
    var x0af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x0af);
    try mlx.check(mlx.mlx_subtract(&x0af, ax, sa, s));
    var x0a = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x0a, x0af, .bfloat16, s));
    return .{ .v = x0v, .a = x0a };
}

/// CFG-only guided Euler denoise loop. `noise_v`/`noise_a` are the bf16 initial
/// latents; `sigmas` includes the terminal 0.0. Returns the final (bf16) latents.
pub fn ditSampleCfg(
    comp: *const Component,
    alloc: std.mem.Allocator,
    cfg: LtxConfig,
    noise_v: mlx.mlx_array,
    noise_a: mlx.mlx_array,
    cond_v: mlx.mlx_array,
    cond_a: mlx.mlx_array,
    neg_v: mlx.mlx_array,
    neg_a: mlx.mlx_array,
    video_pos: []const f32,
    audio_pos: []const f32,
    sigmas: []const f32,
    cfg_video: f32,
    cfg_audio: f32,
    rescale: f32,
    s: S,
) !BlockOut {
    var vx = noise_v;
    var ax = noise_a;
    var owned = false;
    var i: usize = 0;
    while (i + 1 < sigmas.len) : (i += 1) {
        const sigma = sigmas[i];
        const sigma_next = sigmas[i + 1];
        const cond = try ditX0(comp, alloc, cfg, vx, ax, sigma, cond_v, cond_a, video_pos, audio_pos, s);
        defer _ = mlx.mlx_array_free(cond.v);
        defer _ = mlx.mlx_array_free(cond.a);
        const neg = try ditX0(comp, alloc, cfg, vx, ax, sigma, neg_v, neg_a, video_pos, audio_pos, s);
        defer _ = mlx.mlx_array_free(neg.v);
        defer _ = mlx.mlx_array_free(neg.a);
        const v_x0 = try cfgGuide(cond.v, neg.v, cfg_video, rescale, s);
        defer _ = mlx.mlx_array_free(v_x0);
        const a_x0 = try cfgGuide(cond.a, neg.a, cfg_audio, rescale, s);
        defer _ = mlx.mlx_array_free(a_x0);
        const nvx = try eulerStep(vx, v_x0, sigma, sigma_next, s);
        const nax = try eulerStep(ax, a_x0, sigma, sigma_next, s);
        if (owned) {
            _ = mlx.mlx_array_free(vx);
            _ = mlx.mlx_array_free(ax);
        }
        vx = nvx;
        ax = nax;
        owned = true;
        _ = mlx.mlx_array_eval(vx);
        _ = mlx.mlx_array_eval(ax);
    }
    if (!owned) { // no steps → copy inputs so caller owns the result
        var cv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&cv, vx));
        var ca = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&ca, ax));
        return .{ .v = cv, .a = ca };
    }
    return .{ .v = vx, .a = ax };
}

// ════════════════════════════════════════════════════════════════════════
// Latent → frames: unpatchify the sampled video latent, VAE-decode, convert
// to RGB uint8 frames. Reference: VideoLatentPatchifier.unpatchify + VideoDecoder
// pixel conversion (clip[-1,1], (x+1)*127.5, uint8, CHW→HWC).
// ════════════════════════════════════════════════════════════════════════

/// Unpatchify: [1, F*H*W, C] → [1, C, F, H, W] (contiguous, ready for vaeDecode).
pub fn unpatchifyVideo(tokens: mlx.mlx_array, F: u32, H: u32, W: u32, s: S) !mlx.mlx_array {
    const C: c_int = mlx.getShape(tokens)[2];
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, tokens, &[_]c_int{ 1, @intCast(F), @intCast(H), @intCast(W), C }, 5, s));
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, r, &[_]c_int{ 0, 4, 1, 2, 3 }, 5, s)); // [1,C,F,H,W]
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&out, t, false, s));
    return out;
}

/// Convert decoded pixels [1,3,F,H,W] (≈[-1,1]) to RGB uint8 frames laid out as
/// [F, H, W, 3] (row-major). Matches the reference clip→(x+1)*127.5→uint8→HWC.
/// Returns a freshly-allocated `[]u8` of length F*H*W*3 (caller frees).
pub fn framesToU8(pixels: mlx.mlx_array, alloc: std.mem.Allocator, s: S) ![]u8 {
    const sh = mlx.getShape(pixels); // [1,3,F,H,W]
    const Fp: usize = @intCast(sh[2]);
    const Hp: usize = @intCast(sh[3]);
    const Wp: usize = @intCast(sh[4]);
    const lo = mlx.mlx_array_new_float(-1.0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(hi);
    var mx0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx0);
    try mlx.check(mlx.mlx_maximum(&mx0, pixels, lo, s));
    var cl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cl);
    try mlx.check(mlx.mlx_minimum(&cl, mx0, hi, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, cl, one, s));
    const c127 = mlx.mlx_array_new_float(127.5);
    defer _ = mlx.mlx_array_free(c127);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_multiply(&sc, p1, c127, s));
    var u8a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u8a);
    try mlx.check(mlx.mlx_astype(&u8a, sc, .uint8, s)); // truncate, matches reference
    // CHW→HWC: [1,3,F,H,W] → [1,F,H,W,3]
    var fhwc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fhwc);
    try mlx.check(mlx.mlx_transpose_axes(&fhwc, u8a, &[_]c_int{ 0, 2, 3, 4, 1 }, 5, s));
    var cont = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cont);
    try mlx.check(mlx.mlx_contiguous(&cont, fhwc, false, s));
    // No uint8 readback binding → read via exact f32 (values are integers 0..255).
    var f32v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32v);
    try mlx.check(mlx.mlx_astype(&f32v, cont, .float32, s));
    _ = mlx.mlx_array_eval(f32v);
    const n = Fp * Hp * Wp * 3;
    const data = mlx.mlx_array_data_float32(f32v).?;
    const out = try alloc.alloc(u8, n);
    for (0..n) |idx| out[idx] = @intFromFloat(data[idx]);
    return out;
}

// ── Tests ──

const testing = std.testing;

test "LtxConfig derived dims" {
    const c = LtxConfig{};
    try testing.expectEqual(@as(u32, 4096), c.videoDim());
    try testing.expectEqual(@as(u32, 2048), c.audioDim());
    try testing.expectEqual(@as(u32, 188160), c.gemma_layers * c.gemma_hidden);
}

// Loader validation against the REAL ltx-2.3-mlx-q4 checkpoint. Loads the small
// bf16 components (connector + vae_decoder — NOT the 11GB transformer, to avoid
// OOM) and pins key tensors' presence/shape + the q4/bf16 classifier.
//   LTX_TEST_MODEL = path to the snapshot dir
test "ltx loader: connector + vae_decoder key map" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    // Connector (bf16).
    {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{dir}, 0);
        defer allocator.free(path);
        var comp = try loadComponent(allocator, path, s);
        defer comp.deinit();
        std.debug.print("[ltx] connector tensors: {d}\n", .{comp.count()});
        try testing.expect(comp.get("connector.text_embedding_projection.video_aggregate_embed.weight") != null);
        const vproj = comp.get("connector.text_embedding_projection.video_aggregate_embed.weight").?;
        const sh = mlx.getShape(vproj);
        try testing.expectEqual(@as(c_int, 4096), sh[0]);
        try testing.expectEqual(@as(c_int, 188160), sh[1]);
        // connector is bf16 → not quantized
        try testing.expect(!comp.isQuantized("connector.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"));
    }

    // VAE decoder (bf16; conv weights already MLX [out,kD,kH,kW,in]).
    {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{dir}, 0);
        defer allocator.free(path);
        var comp = try loadComponent(allocator, path, s);
        defer comp.deinit();
        std.debug.print("[ltx] vae_decoder tensors: {d}\n", .{comp.count()});
        const ci = comp.get("vae_decoder.conv_in.conv.weight").?;
        const sh = mlx.getShape(ci);
        // [C_out=1024, kD=3, kH=3, kW=3, C_in=128]
        try testing.expectEqual(@as(usize, 5), sh.len);
        try testing.expectEqual(@as(c_int, 1024), sh[0]);
        try testing.expectEqual(@as(c_int, 128), sh[4]);
    }
}

// Stage-5 keystone: validate decoderConv3d (the VAE's causal=False Conv3dBlock)
// against the reference conv_in. Confirms the conv3d primitive + [O,kD,kH,kW,I]
// weight layout + replicate/zero padding produce bit-faithful numerics — the
// gate for the whole 3D VAE decoder. Gated on:
//   LTX_TEST_MODEL = ltx-2.3-mlx-q4 snapshot dir
//   LTX_CONVIN_X / LTX_CONVIN_Y = raw f32 dumps from the Python oracle
//     ([1,3,8,8,128] input / [1,3,8,8,1024] output of conv_in).
test "ltx decoderConv3d matches reference conv_in" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const xp = std.mem.span(std.c.getenv("LTX_CONVIN_X") orelse return error.SkipZigTest);
    const yp = std.mem.span(std.c.getenv("LTX_CONVIN_Y") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const xbuf = try readF32(io, allocator, xp);
    defer allocator.free(xbuf);
    const yref = try readF32(io, allocator, yp);
    defer allocator.free(yref);

    const xshape = [_]c_int{ 1, 3, 8, 8, 128 };
    const x = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 5, .float32);
    defer _ = mlx.mlx_array_free(x);

    // Load weights on a CPU stream — the safetensors Load op only evals on CPU;
    // loading on the GPU stream makes the downstream GPU conv graph fail with
    // "[Load::eval_gpu] Not implemented".
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);
    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();
    const w = comp.get("vae_decoder.conv_in.conv.weight").?;
    const b = comp.get("vae_decoder.conv_in.conv.bias").?;
    _ = mlx.mlx_array_eval(w);
    _ = mlx.mlx_array_eval(b);

    const out = try decoderConv3d(x, w, b, 3, 1, s);
    defer _ = mlx.mlx_array_free(out);
    var outf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(outf);
    try mlx.check(mlx.mlx_astype(&outf, out, .float32, s));
    _ = mlx.mlx_array_eval(outf);

    const n: usize = @intCast(mlx.mlx_array_size(outf));
    try testing.expectEqual(yref.len, n);
    const data = mlx.mlx_array_data_float32(outf).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var maxabs: f64 = 0;
    for (0..n) |i| {
        const a: f64 = data[i];
        const r: f64 = yref[i];
        dot += a * r;
        na += a * a;
        nb += r * r;
        maxabs = @max(maxabs, @abs(a - r));
    }
    const corr = dot / (@sqrt(na) * @sqrt(nb));
    std.debug.print("[ltx-conv3d] n={d} corr={d:.6} max_abs_err={d:.5}\n", .{ n, corr, maxabs });
    try testing.expect(corr > 0.9999);
}

// Full 3D VAE decoder validated against the reference VideoDecoder on a fixed
// latent (NO 41GB pipeline — just vae_decoder.safetensors). Gated on:
//   LTX_TEST_MODEL  = ltx-2.3-mlx-q4 snapshot dir
//   LTX_VAE_LATENT  = raw f32 [1,128,2,8,12] (BCFHW) reference latent
//   LTX_VAE_DECODED = raw f32 [1,3,9,256,384] (BCFHW) reference decoded pixels
test "ltx vaeDecode reproduces reference VideoDecoder" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const lp = std.mem.span(std.c.getenv("LTX_VAE_LATENT") orelse return error.SkipZigTest);
    const dp = std.mem.span(std.c.getenv("LTX_VAE_DECODED") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const latbuf = try readF32(io, allocator, lp);
    defer allocator.free(latbuf);
    const ref = try readF32(io, allocator, dp);
    defer allocator.free(ref);

    // Weights on a CPU stream (Load has no GPU eval), evaled before the GPU graph.
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);
    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();
    var it = comp.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*);

    const lshape = [_]c_int{ 1, 128, 2, 8, 12 };
    const lat_f32 = mlx.mlx_array_new_data(latbuf.ptr, &lshape, 5, .float32);
    defer _ = mlx.mlx_array_free(lat_f32);
    var lat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lat);
    try mlx.check(mlx.mlx_astype(&lat, lat_f32, .bfloat16, s));

    const out = try vaeDecode(&comp, lat, s);
    defer _ = mlx.mlx_array_free(out);
    var outf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(outf);
    try mlx.check(mlx.mlx_astype(&outf, out, .float32, s));
    _ = mlx.mlx_array_eval(outf);

    const n: usize = @intCast(mlx.mlx_array_size(outf));
    try testing.expectEqual(ref.len, n);
    const data = mlx.mlx_array_data_float32(outf).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var se: f64 = 0;
    for (0..n) |idx| {
        const a: f64 = data[idx];
        const r: f64 = ref[idx];
        dot += a * r;
        na += a * a;
        nb += r * r;
        se += (a - r) * (a - r);
    }
    const corr = dot / (@sqrt(na) * @sqrt(nb));
    const mse = se / @as(f64, @floatFromInt(n));
    const psnr = 10.0 * std.math.log10(4.0 / mse); // signal range ~[-1,1] → peak 2, peak^2=4
    std.debug.print("[ltx-vae] n={d} corr={d:.6} psnr={d:.2}dB mse={d:.6}\n", .{ n, corr, psnr, mse });
    try testing.expect(corr > 0.99);
}

fn corrOf(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

// Connector projection (stage 2 front-half): 49-stack → video/audio embeds.
//   LTX_TEST_MODEL = q4 model dir (for connector.safetensors)
//   LTX_CONN_STACK = [49,1,T,3840] f32; LTX_CONN_VIDEO/AUDIO = reference outputs
test "ltx connectorProject reproduces TextEmbeddingProjection" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const sp = std.mem.span(std.c.getenv("LTX_CONN_STACK") orelse return error.SkipZigTest);
    const vp = std.mem.span(std.c.getenv("LTX_CONN_VIDEO") orelse return error.SkipZigTest);
    const ap = std.mem.span(std.c.getenv("LTX_CONN_AUDIO") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const stackbuf = try readF32(io, allocator, sp);
    defer allocator.free(stackbuf);
    const refv = try readF32(io, allocator, vp);
    defer allocator.free(refv);
    const refa = try readF32(io, allocator, ap);
    defer allocator.free(refa);

    const T: usize = stackbuf.len / (49 * 3840); // [49,1,T,3840]
    const per: usize = T * 3840;

    // Build 49 bf16 arrays [1,T,3840] from the f32 dump.
    var arrs = try allocator.alloc(mlx.mlx_array, 49);
    defer {
        for (arrs) |a| _ = mlx.mlx_array_free(a);
        allocator.free(arrs);
    }
    const shp = [_]c_int{ 1, @intCast(T), 3840 };
    for (0..49) |i| {
        const f32a = mlx.mlx_array_new_data(stackbuf.ptr + i * per, &shp, 3, .float32);
        defer _ = mlx.mlx_array_free(f32a);
        var bf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&bf, f32a, .bfloat16, s));
        arrs[i] = bf;
    }

    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);
    const cp = try std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{dir}, 0);
    defer allocator.free(cp);
    var comp = try loadComponent(allocator, cp, cpu_s);
    defer comp.deinit();
    var it = comp.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*);

    var out = try connectorProject(&comp, allocator, arrs, s);
    defer out.deinit();

    var vf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vf);
    try mlx.check(mlx.mlx_astype(&vf, out.video, .float32, s));
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, out.audio, .float32, s));
    _ = mlx.mlx_array_eval(vf);
    _ = mlx.mlx_array_eval(af);

    const vn: usize = @intCast(mlx.mlx_array_size(vf));
    const an: usize = @intCast(mlx.mlx_array_size(af));
    try testing.expectEqual(refv.len, vn);
    try testing.expectEqual(refa.len, an);
    const vc = corrOf(mlx.mlx_array_data_float32(vf).?[0..vn], refv);
    const ac = corrOf(mlx.mlx_array_data_float32(af).?[0..an], refa);
    std.debug.print("[ltx-conn] video corr={d:.6} ({d}), audio corr={d:.6} ({d})\n", .{ vc, vn, ac, an });
    try testing.expect(vc > 0.999);
    try testing.expect(ac > 0.999);
}

// Stage A: the Embeddings1DConnector transformers reproduce the reference output
// for a fixed projected input + mask (T=256, n_valid=40). Env:
//   LTX_TEST_MODEL, LTX_CONNT_VIN/VOUT (video [1,256,4096]), LTX_CONNT_AIN/AOUT (audio [1,256,2048]).
test "ltx connectorTransform reproduces Embeddings1DConnector" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const vin_p = std.mem.span(std.c.getenv("LTX_CONNT_VIN") orelse return error.SkipZigTest);
    const vout_p = std.mem.span(std.c.getenv("LTX_CONNT_VOUT") orelse return error.SkipZigTest);
    const ain_p = std.mem.span(std.c.getenv("LTX_CONNT_AIN") orelse return error.SkipZigTest);
    const aout_p = std.mem.span(std.c.getenv("LTX_CONNT_AOUT") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);
    const cp = try std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{dir}, 0);
    defer allocator.free(cp);
    var comp = try loadComponent(allocator, cp, cpu_s);
    defer comp.deinit();
    var it = comp.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*);

    const T: u32 = 256;
    const nv: u32 = 40;
    const Run = struct {
        fn go(c: *Component, a: std.mem.Allocator, ii: std.Io, in_p: []const u8, out_p: []const u8, dim: u32, hd: u32, prefix: []const u8, st: S) !f64 {
            const inbuf = try readF32(ii, a, in_p);
            defer a.free(inbuf);
            const ref = try readF32(ii, a, out_p);
            defer a.free(ref);
            const ish = [_]c_int{ 1, @intCast(T), @intCast(dim) };
            const inarr = mlx.mlx_array_new_data(inbuf.ptr, &ish, 3, .float32);
            defer _ = mlx.mlx_array_free(inarr);
            const out = try connectorTransform(c, a, inarr, nv, dim, hd, prefix, st);
            defer _ = mlx.mlx_array_free(out);
            var outf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(outf);
            try mlx.check(mlx.mlx_astype(&outf, out, .float32, st));
            _ = mlx.mlx_array_eval(outf);
            const n: usize = @intCast(mlx.mlx_array_size(outf));
            try testing.expectEqual(ref.len, n);
            return corrOf(mlx.mlx_array_data_float32(outf).?[0..n], ref);
        }
    };
    const vc = try Run.go(&comp, allocator, io, vin_p, vout_p, 4096, 128, "connector.video_embeddings_connector", s);
    const ac = try Run.go(&comp, allocator, io, ain_p, aout_p, 2048, 64, "connector.audio_embeddings_connector", s);
    std.debug.print("[ltx-connT] video corr={d:.6}, audio corr={d:.6}\n", .{ vc, ac });
    try testing.expect(vc > 0.99);
    try testing.expect(ac > 0.99);
}

fn readF32(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(bytes);
    const cnt = bytes.len / 4;
    const out = try allocator.alloc(f32, cnt);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. cnt * 4]);
    return out;
}

fn readI32(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]i32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(bytes);
    const cnt = bytes.len / 4;
    const out = try allocator.alloc(i32, cnt);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. cnt * 4]);
    return out;
}

fn readU8(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
}

fn corrF32(data: [*]const f32, ref: []const f32, start: usize) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var i: usize = start;
    while (i < ref.len) : (i += 1) {
        const aa: f64 = data[i];
        const bb: f64 = ref[i];
        dot += aa * bb;
        na += aa * aa;
        nb += bb * bb;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

// Gemma 49-layer capture vs the reference get_all_hidden_states. Gated on:
//   LTX_GEMMA_MODEL = gemma-3-12b-it-4bit dir
//   LTX_GEMMA_IDS   = int32 raw of the left-padded 1024 token ids
//   LTX_GEMMA_S0/S1/S24/S48 = f32 raw of reference states 0/1/24/48
test "ltx gemmaCapture reproduces reference hidden states" {
    const dir = std.mem.span(std.c.getenv("LTX_GEMMA_MODEL") orelse return error.SkipZigTest);
    const idp = std.mem.span(std.c.getenv("LTX_GEMMA_IDS") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const ids = try readI32(io, allocator, idp);
    defer allocator.free(ids);

    const states = try gemmaCapture(io, allocator, dir, ids, 0, s);
    defer {
        for (states) |st| _ = mlx.mlx_array_free(st);
        allocator.free(states);
    }
    try testing.expectEqual(@as(usize, 49), states.len);

    // Only the VALID (non-pad) token rows feed the LTX connector (it zeros
    // padding positions). Left-padded query positions attend an all-masked row
    // (softmax over all -1e9 → undefined garbage that diverges across layers in
    // both implementations), so compare only the valid tail.
    var first_valid: usize = 0;
    while (first_valid < ids.len and ids[first_valid] == 0) : (first_valid += 1) {}
    const valid_start = first_valid * 3840;

    const checks = [_]struct { li: usize, env: []const u8 }{
        .{ .li = 0, .env = "LTX_GEMMA_S0" },
        .{ .li = 1, .env = "LTX_GEMMA_S1" },
        .{ .li = 24, .env = "LTX_GEMMA_S24" },
        .{ .li = 48, .env = "LTX_GEMMA_S48" },
    };
    for (checks) |c| {
        const p = std.c.getenv(@ptrCast(c.env.ptr)) orelse continue;
        const ref = try readF32(io, allocator, std.mem.span(p));
        defer allocator.free(ref);
        var f32a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32a);
        try mlx.check(mlx.mlx_astype(&f32a, states[c.li], .float32, s));
        _ = mlx.mlx_array_eval(f32a);
        const n: usize = @intCast(mlx.mlx_array_size(f32a));
        try testing.expectEqual(ref.len, n);
        const data = mlx.mlx_array_data_float32(f32a).?;
        const corr_full = corrF32(data, ref, 0);
        const corr_valid = corrF32(data, ref, valid_start);
        std.debug.print("[ltx-gemma] layer {d} corr_valid={d:.6} (full={d:.4})\n", .{ c.li, corr_valid, corr_full });
        try testing.expect(corr_valid > 0.999);
    }
}

// DiT conditioning oracle: video adaln_single (9-param) for a fixed timestep.
//   LTX_TEST_MODEL=<ltx-2.3-mlx-q4 snapshot>, LTX_ADA_PARAMS=<params.raw>, LTX_ADA_EMB=<emb.raw>
test "ltx DiT adaLN conditioning reproduces AdaLayerNormSingle" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const pp = std.mem.span(std.c.getenv("LTX_ADA_PARAMS") orelse return error.SkipZigTest);
    const ep = std.mem.span(std.c.getenv("LTX_ADA_EMB") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();

    const refp = try readF32(io, allocator, pp);
    defer allocator.free(refp);
    const refe = try readF32(io, allocator, ep);
    defer allocator.free(refe);

    const t_sin = try ditTimestepSinusoid(700.0, 256, s);
    defer _ = mlx.mlx_array_free(t_sin);
    const out = try ditAdaLNSingle(&comp, allocator, t_sin, "transformer.adaln_single", s);
    defer _ = mlx.mlx_array_free(out.params);
    defer _ = mlx.mlx_array_free(out.embedded);

    var pf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pf);
    try mlx.check(mlx.mlx_astype(&pf, out.params, .float32, s));
    _ = mlx.mlx_array_eval(pf);
    var ef = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ef);
    try mlx.check(mlx.mlx_astype(&ef, out.embedded, .float32, s));
    _ = mlx.mlx_array_eval(ef);

    const np: usize = @intCast(mlx.mlx_array_size(pf));
    try testing.expectEqual(refp.len, np);
    const cp = corrF32(mlx.mlx_array_data_float32(pf).?, refp, 0);
    const ce = corrF32(mlx.mlx_array_data_float32(ef).?, refe, 0);
    std.debug.print("[ltx-ada] params corr={d:.6} embedded corr={d:.6}\n", .{ cp, ce });
    try testing.expect(cp > 0.999);
    try testing.expect(ce > 0.999);
}

// Stage 1: compute_video_normed_sa (the adaLN-table unpack + affine-free RMS
// modulation) reproduces the reference block-0 SA input. Catches the
// shift/scale row-order trap cheaply. Gated on:
//   LTX_TEST_MODEL=<q4 snapshot>, LTX_NSA_VH=<video_hidden.raw [1,192,4096]>,
//   LTX_NSA_PARAMS=<video_adaln_params.raw [1,36864]>, LTX_NSA_REF=<normed_sa.raw [1,192,4096]>
test "ltx DiT ditNormedSA reproduces compute_video_normed_sa" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const vhp = std.mem.span(std.c.getenv("LTX_NSA_VH") orelse return error.SkipZigTest);
    const pp = std.mem.span(std.c.getenv("LTX_NSA_PARAMS") orelse return error.SkipZigTest);
    const rp = std.mem.span(std.c.getenv("LTX_NSA_REF") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();

    const vhbuf = try readF32(io, allocator, vhp);
    defer allocator.free(vhbuf);
    const pbuf = try readF32(io, allocator, pp);
    defer allocator.free(pbuf);
    const ref = try readF32(io, allocator, rp);
    defer allocator.free(ref);

    const Nv: c_int = @intCast(vhbuf.len / 4096);
    const vh_f32 = mlx.mlx_array_new_data(vhbuf.ptr, &[_]c_int{ 1, Nv, 4096 }, 3, .float32);
    defer _ = mlx.mlx_array_free(vh_f32);
    var vh = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vh);
    try mlx.check(mlx.mlx_astype(&vh, vh_f32, .bfloat16, s));
    const p_f32 = mlx.mlx_array_new_data(pbuf.ptr, &[_]c_int{ 1, 36864 }, 2, .float32);
    defer _ = mlx.mlx_array_free(p_f32);
    var params = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(params);
    try mlx.check(mlx.mlx_astype(&params, p_f32, .bfloat16, s));

    const out = try ditNormedSA(&comp, allocator, vh, params, "transformer.transformer_blocks.0.scale_shift_table", 4096, 1e-6, s);
    defer _ = mlx.mlx_array_free(out);
    var of = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(of);
    try mlx.check(mlx.mlx_astype(&of, out, .float32, s));
    _ = mlx.mlx_array_eval(of);

    const n: usize = @intCast(mlx.mlx_array_size(of));
    try testing.expectEqual(ref.len, n);
    const corr = corrF32(mlx.mlx_array_data_float32(of).?, ref, 0);
    std.debug.print("[ltx-nsa] normed_sa corr={d:.6} (n={d})\n", .{ corr, n });
    try testing.expect(corr > 0.99);
}

// Stage 2a: ditRope (precompute_rope_freqs SPLIT) reproduces reference cos/sin.
// Cheap, no model load. Gated on LTX_TEST_MODEL not needed; needs the position +
// rope .raw dumps. Env: LTX_POS_V=[1,Nv,3], LTX_POS_A=[1,Na,1],
// LTX_ROPE_V{COS,SIN}=[1,32,Nv,64], LTX_ROPE_A{COS,SIN}=[1,32,Na,32],
// LTX_ROPE_VX{COS,SIN}=[1,32,Nv,32], LTX_ROPE_AX{COS,SIN}=[1,32,Na,32].
test "ltx DiT ditRope matches reference precompute_rope_freqs" {
    const pvp = std.c.getenv("LTX_POS_V") orelse return error.SkipZigTest;
    const pap = std.c.getenv("LTX_POS_A") orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const vpos = try readF32(io, allocator, std.mem.span(pvp));
    defer allocator.free(vpos);
    const apos = try readF32(io, allocator, std.mem.span(pap));
    defer allocator.free(apos);
    const Nv: u32 = @intCast(vpos.len / 3);
    const Na: u32 = @intCast(apos.len);

    // temporal-only axis for cross rope
    const vtmp = try allocator.alloc(f32, Nv);
    defer allocator.free(vtmp);
    for (0..Nv) |n| vtmp[n] = vpos[n * 3 + 0];

    const Check = struct {
        fn cmp(a: std.mem.Allocator, ii: std.Io, rope: RopeCS, env_cos: []const u8, env_sin: []const u8, st: S) !struct { c: f64, sgn: f64 } {
            const cp = std.c.getenv(@ptrCast(env_cos.ptr)) orelse return error.SkipZigTest;
            const sp = std.c.getenv(@ptrCast(env_sin.ptr)) orelse return error.SkipZigTest;
            const rc = try readF32(ii, a, std.mem.span(cp));
            defer a.free(rc);
            const rs = try readF32(ii, a, std.mem.span(sp));
            defer a.free(rs);
            var cf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cf);
            try mlx.check(mlx.mlx_astype(&cf, rope.cos, .float32, st));
            _ = mlx.mlx_array_eval(cf);
            var sf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sf);
            try mlx.check(mlx.mlx_astype(&sf, rope.sin, .float32, st));
            _ = mlx.mlx_array_eval(sf);
            const nc: usize = @intCast(mlx.mlx_array_size(cf));
            try testing.expectEqual(rc.len, nc);
            const cc = corrF32(mlx.mlx_array_data_float32(cf).?, rc, 0);
            const sc = corrF32(mlx.mlx_array_data_float32(sf).?, rs, 0);
            return .{ .c = cc, .sgn = sc };
        }
    };

    const max_v = [_]f32{ 20.0, 2048.0, 2048.0 };
    const max_1 = [_]f32{20.0};

    const rv = try ditRope(allocator, vpos, Nv, 3, 32, 128, &max_v, s);
    defer _ = mlx.mlx_array_free(rv.cos);
    defer _ = mlx.mlx_array_free(rv.sin);
    const rrv = try Check.cmp(allocator, io, rv, "LTX_ROPE_VCOS", "LTX_ROPE_VSIN", s);

    const ra = try ditRope(allocator, apos, Na, 1, 32, 64, &max_1, s);
    defer _ = mlx.mlx_array_free(ra.cos);
    defer _ = mlx.mlx_array_free(ra.sin);
    const rra = try Check.cmp(allocator, io, ra, "LTX_ROPE_ACOS", "LTX_ROPE_ASIN", s);

    const rvx = try ditRope(allocator, vtmp, Nv, 1, 32, 64, &max_1, s);
    defer _ = mlx.mlx_array_free(rvx.cos);
    defer _ = mlx.mlx_array_free(rvx.sin);
    const rrvx = try Check.cmp(allocator, io, rvx, "LTX_ROPE_VXCOS", "LTX_ROPE_VXSIN", s);

    const rax = try ditRope(allocator, apos, Na, 1, 32, 64, &max_1, s);
    defer _ = mlx.mlx_array_free(rax.cos);
    defer _ = mlx.mlx_array_free(rax.sin);
    const rrax = try Check.cmp(allocator, io, rax, "LTX_ROPE_AXCOS", "LTX_ROPE_AXSIN", s);

    std.debug.print("[ltx-rope] video cos={d:.6}/sin={d:.6} audio={d:.6}/{d:.6} vcross={d:.6}/{d:.6} across={d:.6}/{d:.6}\n", .{ rrv.c, rrv.sgn, rra.c, rra.sgn, rrvx.c, rrvx.sgn, rrax.c, rrax.sgn });
    try testing.expect(rrv.c > 0.9999 and rrv.sgn > 0.9999);
    try testing.expect(rra.c > 0.9999 and rra.sgn > 0.9999);
    try testing.expect(rrvx.c > 0.9999 and rrvx.sgn > 0.9999);
    try testing.expect(rrax.c > 0.9999 and rrax.sgn > 0.9999);
}

// Stage 2b: full BasicAVTransformerBlock block-0 forward reproduces out_v/out_a.
// Gated on LTX_TEST_MODEL + the block-0 IO .raw dumps (LTX_B0_*).
test "ltx DiT block-0 forward reproduces BasicAVTransformerBlock" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();

    // helper: load env .raw → bf16 mlx array of given shape (f32 if keep_f32)
    const L = struct {
        fn arr(a: std.mem.Allocator, ii: std.Io, env: []const u8, shape: []const c_int, keep_f32: bool, st: S) !struct { x: mlx.mlx_array, buf: []f32 } {
            const p = std.c.getenv(@ptrCast(env.ptr)) orelse return error.SkipZigTest;
            const buf = try readF32(ii, a, std.mem.span(p));
            const d = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            if (keep_f32) return .{ .x = d, .buf = buf };
            defer _ = mlx.mlx_array_free(d);
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&b, d, .bfloat16, st));
            return .{ .x = b, .buf = buf };
        }
    };

    const Nv: c_int = 192;
    const Na: c_int = 9;
    const Nt: c_int = 128;

    const vh = try L.arr(allocator, io, "LTX_B0_VH", &[_]c_int{ 1, Nv, 4096 }, false, s);
    defer allocator.free(vh.buf);
    defer _ = mlx.mlx_array_free(vh.x);
    const ah = try L.arr(allocator, io, "LTX_B0_AH", &[_]c_int{ 1, Na, 2048 }, false, s);
    defer allocator.free(ah.buf);
    defer _ = mlx.mlx_array_free(ah.x);
    const vap = try L.arr(allocator, io, "LTX_B0_VAP", &[_]c_int{ 1, 9 * 4096 }, false, s);
    defer allocator.free(vap.buf);
    defer _ = mlx.mlx_array_free(vap.x);
    const aap = try L.arr(allocator, io, "LTX_B0_AAP", &[_]c_int{ 1, 9 * 2048 }, false, s);
    defer allocator.free(aap.buf);
    defer _ = mlx.mlx_array_free(aap.x);
    const vpp = try L.arr(allocator, io, "LTX_B0_VPP", &[_]c_int{ 1, 2 * 4096 }, false, s);
    defer allocator.free(vpp.buf);
    defer _ = mlx.mlx_array_free(vpp.x);
    const app = try L.arr(allocator, io, "LTX_B0_APP", &[_]c_int{ 1, 2 * 2048 }, false, s);
    defer allocator.free(app.buf);
    defer _ = mlx.mlx_array_free(app.x);
    const avv = try L.arr(allocator, io, "LTX_B0_AVV", &[_]c_int{ 1, 4 * 4096 }, false, s);
    defer allocator.free(avv.buf);
    defer _ = mlx.mlx_array_free(avv.x);
    const ava = try L.arr(allocator, io, "LTX_B0_AVA", &[_]c_int{ 1, 4 * 2048 }, false, s);
    defer allocator.free(ava.buf);
    defer _ = mlx.mlx_array_free(ava.x);
    const a2vg = try L.arr(allocator, io, "LTX_B0_A2VG", &[_]c_int{ 1, 4096 }, false, s);
    defer allocator.free(a2vg.buf);
    defer _ = mlx.mlx_array_free(a2vg.x);
    const v2ag = try L.arr(allocator, io, "LTX_B0_V2AG", &[_]c_int{ 1, 2048 }, false, s);
    defer allocator.free(v2ag.buf);
    defer _ = mlx.mlx_array_free(v2ag.x);
    const vtext = try L.arr(allocator, io, "LTX_B0_VTEXT", &[_]c_int{ 1, Nt, 4096 }, false, s);
    defer allocator.free(vtext.buf);
    defer _ = mlx.mlx_array_free(vtext.x);
    const atext = try L.arr(allocator, io, "LTX_B0_ATEXT", &[_]c_int{ 1, Nt, 2048 }, false, s);
    defer allocator.free(atext.buf);
    defer _ = mlx.mlx_array_free(atext.x);

    // positions → rope
    const vpos = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_POS_V") orelse return error.SkipZigTest));
    defer allocator.free(vpos);
    const apos = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_POS_A") orelse return error.SkipZigTest));
    defer allocator.free(apos);
    const NvU: u32 = @intCast(vpos.len / 3);
    const NaU: u32 = @intCast(apos.len);
    const vtmp = try allocator.alloc(f32, NvU);
    defer allocator.free(vtmp);
    for (0..NvU) |n| vtmp[n] = vpos[n * 3 + 0];
    const max_v = [_]f32{ 20.0, 2048.0, 2048.0 };
    const max_1 = [_]f32{20.0};
    const rv = try ditRope(allocator, vpos, NvU, 3, 32, 128, &max_v, s);
    defer _ = mlx.mlx_array_free(rv.cos);
    defer _ = mlx.mlx_array_free(rv.sin);
    const ra = try ditRope(allocator, apos, NaU, 1, 32, 64, &max_1, s);
    defer _ = mlx.mlx_array_free(ra.cos);
    defer _ = mlx.mlx_array_free(ra.sin);
    const rvx = try ditRope(allocator, vtmp, NvU, 1, 32, 64, &max_1, s);
    defer _ = mlx.mlx_array_free(rvx.cos);
    defer _ = mlx.mlx_array_free(rvx.sin);
    const rax = try ditRope(allocator, apos, NaU, 1, 32, 64, &max_1, s);
    defer _ = mlx.mlx_array_free(rax.cos);
    defer _ = mlx.mlx_array_free(rax.sin);
    const rope = BlockRope{ .vcos = rv.cos, .vsin = rv.sin, .acos = ra.cos, .asin = ra.sin, .vxcos = rvx.cos, .vxsin = rvx.sin, .axcos = rax.cos, .axsin = rax.sin };

    const cfg = LtxConfig{};
    const out = try ditBlock(&comp, allocator, cfg, 0, vh.x, ah.x, vap.x, aap.x, vpp.x, app.x, avv.x, ava.x, a2vg.x, v2ag.x, vtext.x, atext.x, rope, s);
    defer _ = mlx.mlx_array_free(out.v);
    defer _ = mlx.mlx_array_free(out.a);

    const refv = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_B0_OUTV") orelse return error.SkipZigTest));
    defer allocator.free(refv);
    const refa = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_B0_OUTA") orelse return error.SkipZigTest));
    defer allocator.free(refa);
    var vf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vf);
    try mlx.check(mlx.mlx_astype(&vf, out.v, .float32, s));
    _ = mlx.mlx_array_eval(vf);
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, out.a, .float32, s));
    _ = mlx.mlx_array_eval(af);
    const vn: usize = @intCast(mlx.mlx_array_size(vf));
    const an: usize = @intCast(mlx.mlx_array_size(af));
    try testing.expectEqual(refv.len, vn);
    try testing.expectEqual(refa.len, an);
    const vc = corrF32(mlx.mlx_array_data_float32(vf).?, refv, 0);
    const ac = corrF32(mlx.mlx_array_data_float32(af).?, refa, 0);
    std.debug.print("[ltx-b0] out_v corr={d:.6} ({d}) out_a corr={d:.6} ({d})\n", .{ vc, vn, ac, an });
    try testing.expect(vc > 0.99);
    try testing.expect(ac > 0.99);
}

// Stage 3: full ditForward (48 blocks + head) reproduces the reference velocity.
// Gated on LTX_TEST_MODEL + latent/text/position/velocity .raw dumps (LTX_FWD_*).
test "ltx DiT ditForward reproduces LTXModel velocity" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();

    const Lf = struct {
        fn bf(a: std.mem.Allocator, ii: std.Io, env: []const u8, shape: []const c_int, st: S) !struct { x: mlx.mlx_array, buf: []f32 } {
            const p = std.c.getenv(@ptrCast(env.ptr)) orelse return error.SkipZigTest;
            const buf = try readF32(ii, a, std.mem.span(p));
            const d = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(d);
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&b, d, .bfloat16, st));
            return .{ .x = b, .buf = buf };
        }
    };

    const vlat = try Lf.bf(allocator, io, "LTX_FWD_VLAT", &[_]c_int{ 1, 192, 128 }, s);
    defer allocator.free(vlat.buf);
    defer _ = mlx.mlx_array_free(vlat.x);
    const alat = try Lf.bf(allocator, io, "LTX_FWD_ALAT", &[_]c_int{ 1, 9, 128 }, s);
    defer allocator.free(alat.buf);
    defer _ = mlx.mlx_array_free(alat.x);
    const vtext = try Lf.bf(allocator, io, "LTX_FWD_VTEXT", &[_]c_int{ 1, 128, 4096 }, s);
    defer allocator.free(vtext.buf);
    defer _ = mlx.mlx_array_free(vtext.x);
    const atext = try Lf.bf(allocator, io, "LTX_FWD_ATEXT", &[_]c_int{ 1, 128, 2048 }, s);
    defer allocator.free(atext.buf);
    defer _ = mlx.mlx_array_free(atext.x);

    const vpos = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_POS_V") orelse return error.SkipZigTest));
    defer allocator.free(vpos);
    const apos = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_POS_A") orelse return error.SkipZigTest));
    defer allocator.free(apos);

    const cfg = LtxConfig{};
    const out = try ditForward(&comp, allocator, cfg, vlat.x, alat.x, 0.7, vtext.x, atext.x, vpos, apos, s);
    defer _ = mlx.mlx_array_free(out.v);
    defer _ = mlx.mlx_array_free(out.a);

    const refv = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_FWD_VVEL") orelse return error.SkipZigTest));
    defer allocator.free(refv);
    const refa = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_FWD_AVEL") orelse return error.SkipZigTest));
    defer allocator.free(refa);
    var vf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vf);
    try mlx.check(mlx.mlx_astype(&vf, out.v, .float32, s));
    _ = mlx.mlx_array_eval(vf);
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, out.a, .float32, s));
    _ = mlx.mlx_array_eval(af);
    const vn: usize = @intCast(mlx.mlx_array_size(vf));
    const an: usize = @intCast(mlx.mlx_array_size(af));
    try testing.expectEqual(refv.len, vn);
    try testing.expectEqual(refa.len, an);
    const vc = corrF32(mlx.mlx_array_data_float32(vf).?, refv, 0);
    const ac = corrF32(mlx.mlx_array_data_float32(af).?, refa, 0);
    std.debug.print("[ltx-fwd] velocity_v corr={d:.6} ({d}) velocity_a corr={d:.6} ({d})\n", .{ vc, vn, ac, an });
    try testing.expect(vc > 0.99);
    try testing.expect(ac > 0.99);
}

// Stage 4a: dynamicShiftSchedule matches the reference ltx2_schedule. Cheap,
// no model load. Gated on LTX_SAMP_SIGMAS=<samp_sigmas.raw [num_steps+1]>.
test "ltx DiT dynamicShiftSchedule matches reference" {
    const sp = std.c.getenv("LTX_SAMP_SIGMAS") orelse return error.SkipZigTest;
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ref = try readF32(io, allocator, std.mem.span(sp));
    defer allocator.free(ref);
    const num_steps: u32 = @intCast(ref.len - 1);
    const sig = try dynamicShiftSchedule(allocator, num_steps, 192, 0.95, 2.05, 1024, 4096, true, 0.1);
    defer allocator.free(sig);
    try testing.expectEqual(ref.len, sig.len);
    var maxerr: f64 = 0;
    for (sig, ref) |a, b| maxerr = @max(maxerr, @abs(@as(f64, a) - @as(f64, b)));
    std.debug.print("[ltx-sched] num_steps={d} max_abs_err={e}\n", .{ num_steps, maxerr });
    try testing.expect(maxerr < 1e-4);
}

// Stage 4b: ditSampleCfg (CFG-only Euler loop) reproduces the reference final
// denoised latents. Gated on LTX_TEST_MODEL + the sampler .raw dumps (LTX_SAMP_*).
test "ltx DiT ditSampleCfg reproduces guided denoise loop" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{dir}, 0);
    defer allocator.free(vp);
    var comp = try loadComponent(allocator, vp, cpu_s);
    defer comp.deinit();

    const Lf = struct {
        fn bf(a: std.mem.Allocator, ii: std.Io, env: []const u8, shape: []const c_int, st: S) !struct { x: mlx.mlx_array, buf: []f32 } {
            const p = std.c.getenv(@ptrCast(env.ptr)) orelse return error.SkipZigTest;
            const buf = try readF32(ii, a, std.mem.span(p));
            const d = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(d);
            var b = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&b, d, .bfloat16, st));
            return .{ .x = b, .buf = buf };
        }
    };
    const nv = try Lf.bf(allocator, io, "LTX_SAMP_NOISE_V", &[_]c_int{ 1, 192, 128 }, s);
    defer allocator.free(nv.buf);
    defer _ = mlx.mlx_array_free(nv.x);
    const na = try Lf.bf(allocator, io, "LTX_SAMP_NOISE_A", &[_]c_int{ 1, 9, 128 }, s);
    defer allocator.free(na.buf);
    defer _ = mlx.mlx_array_free(na.x);
    const cv = try Lf.bf(allocator, io, "LTX_SAMP_COND_V", &[_]c_int{ 1, 128, 4096 }, s);
    defer allocator.free(cv.buf);
    defer _ = mlx.mlx_array_free(cv.x);
    const ca = try Lf.bf(allocator, io, "LTX_SAMP_COND_A", &[_]c_int{ 1, 128, 2048 }, s);
    defer allocator.free(ca.buf);
    defer _ = mlx.mlx_array_free(ca.x);
    const ngv = try Lf.bf(allocator, io, "LTX_SAMP_NEG_V", &[_]c_int{ 1, 128, 4096 }, s);
    defer allocator.free(ngv.buf);
    defer _ = mlx.mlx_array_free(ngv.x);
    const nga = try Lf.bf(allocator, io, "LTX_SAMP_NEG_A", &[_]c_int{ 1, 128, 2048 }, s);
    defer allocator.free(nga.buf);
    defer _ = mlx.mlx_array_free(nga.x);

    const vpos = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_POS_V") orelse return error.SkipZigTest));
    defer allocator.free(vpos);
    const apos = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_POS_A") orelse return error.SkipZigTest));
    defer allocator.free(apos);
    const sigmas = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_SAMP_SIGMAS") orelse return error.SkipZigTest));
    defer allocator.free(sigmas);

    const cfg = LtxConfig{};
    const out = try ditSampleCfg(&comp, allocator, cfg, nv.x, na.x, cv.x, ca.x, ngv.x, nga.x, vpos, apos, sigmas, 3.0, 7.0, 0.7, s);
    defer _ = mlx.mlx_array_free(out.v);
    defer _ = mlx.mlx_array_free(out.a);

    const refv = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_SAMP_FINAL_V") orelse return error.SkipZigTest));
    defer allocator.free(refv);
    const refa = try readF32(io, allocator, std.mem.span(std.c.getenv("LTX_SAMP_FINAL_A") orelse return error.SkipZigTest));
    defer allocator.free(refa);
    var vf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vf);
    try mlx.check(mlx.mlx_astype(&vf, out.v, .float32, s));
    _ = mlx.mlx_array_eval(vf);
    var af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(af);
    try mlx.check(mlx.mlx_astype(&af, out.a, .float32, s));
    _ = mlx.mlx_array_eval(af);
    const vc = corrF32(mlx.mlx_array_data_float32(vf).?, refv, 0);
    const ac = corrF32(mlx.mlx_array_data_float32(af).?, refa, 0);
    std.debug.print("[ltx-samp] final_v corr={d:.6} final_a corr={d:.6}\n", .{ vc, ac });
    try testing.expect(vc > 0.99);
    try testing.expect(ac > 0.99);
}

// Stage 5: latent → frames chain (unpatchifyVideo + vaeDecode + framesToU8)
// reproduces the reference decoded uint8 frames. Gated on LTX_TEST_MODEL +
// LTX_FRAMES_LATENT=<samp_final_v.raw [1,192,128]>, LTX_FRAMES_REF=<frames_u8.raw>.
test "ltx DiT latent-to-frames reproduces reference decode" {
    const dir = std.mem.span(std.c.getenv("LTX_TEST_MODEL") orelse return error.SkipZigTest);
    const lp = std.mem.span(std.c.getenv("LTX_FRAMES_LATENT") orelse return error.SkipZigTest);
    const rp = std.mem.span(std.c.getenv("LTX_FRAMES_REF") orelse return error.SkipZigTest);
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu_s);

    const vpth = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{dir}, 0);
    defer allocator.free(vpth);
    var comp = try loadComponent(allocator, vpth, cpu_s);
    defer comp.deinit();
    var it = comp.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*);

    const latbuf = try readF32(io, allocator, lp);
    defer allocator.free(latbuf);
    const lat_f32 = mlx.mlx_array_new_data(latbuf.ptr, &[_]c_int{ 1, 192, 128 }, 3, .float32);
    defer _ = mlx.mlx_array_free(lat_f32);
    var tokens = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tokens);
    try mlx.check(mlx.mlx_astype(&tokens, lat_f32, .bfloat16, s));

    const latent = try unpatchifyVideo(tokens, 2, 8, 12, s);
    defer _ = mlx.mlx_array_free(latent);
    const pixels = try vaeDecode(&comp, latent, s);
    defer _ = mlx.mlx_array_free(pixels);
    const frames = try framesToU8(pixels, allocator, s);
    defer allocator.free(frames);

    const ref = try readU8(io, allocator, rp);
    defer allocator.free(ref);
    try testing.expectEqual(ref.len, frames.len);
    var se: f64 = 0;
    var maxd: u32 = 0;
    var exact: usize = 0;
    for (frames, ref) |a, b| {
        const d: i32 = @as(i32, a) - @as(i32, b);
        se += @as(f64, @floatFromInt(d * d));
        const ad: u32 = @abs(d);
        maxd = @max(maxd, ad);
        if (d == 0) exact += 1;
    }
    const mse = se / @as(f64, @floatFromInt(frames.len));
    const psnr = if (mse > 0) 10.0 * std.math.log10(255.0 * 255.0 / mse) else 99.0;
    const exact_frac = @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(frames.len));
    std.debug.print("[ltx-frames] n={d} psnr={d:.2}dB max_diff={d} exact={d:.3}\n", .{ frames.len, psnr, maxd, exact_frac });
    try testing.expect(psnr > 45.0);
    try testing.expect(maxd <= 2);
}
