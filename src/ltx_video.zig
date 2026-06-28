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
