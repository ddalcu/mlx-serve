//! Native Hunyuan3D-2.1 SHAPE engine (image → untextured 3D mesh → GLB bytes),
//! ported from the Tencent reference (hy3dshape) to mlx-c FFI.
//!
//! Pipeline: DINOv2-Large conditioner → 3.3B flow-match DiT (U-ViT skips,
//! timestep-as-token-0, 6 MoE layers) → ShapeVAE decoder (16 self-attn blocks
//! + geo-decoder cross-attn SDF queries) → chunked volume decode → marching
//! cubes → GLB. Texture/paint is a later phase; `generateMeshRaw` is the seam.
//!
//! Self-contained sibling of krea.zig/flux.zig, hosted by the mesh modality
//! slot in gen.zig. Weights arrive PRE-CONVERTED by tests/convert_hunyuan3d_weights.py
//! per the binding tensor-name contract (dit/conditioner/vae.safetensors +
//! synthesized config.json {"model_type":"hunyuan3d_2_1"}).
//!
//! Precision: the DENSE dtype of this engine is FLOAT16, not bf16 (the source
//! ckpts are fp16; converting to bf16 would throw away 3 mantissa bits).
//! `MixedLinear` infers (bits, group_size) from tensor geometry so one engine
//! loads both the 8-bit (default ship) and fp16 (parity-debug) builds.
//! f32 discipline: timestep sincos, scheduler step, SDF grid accumulation and
//! marching-cubes interpolation all run in f32.
//!
//! Parity traps honored here (verified against the reference sources):
//! - NO AdaLN: the timestep embedding rides as TOKEN 0 (cat([t_emb, x]) → 4097).
//! - U-ViT skips: blocks 0..9 push their OUTPUT, blocks 11..20 pop LIFO
//!   (11↔9 … 20↔0); skip fuse = LayerNorm(Linear(cat([skip, x], -1))).
//! - MoE (blocks 15..20): softmax over 8 experts BEFORE top-2, NO renorm,
//!   plus an always-on shared expert.
//! - Two different qk-norms: DiT = per-head RMSNorm(128); VAE = per-head
//!   AFFINE LayerNorm(64); both eps 1e-6.
//! - Reversed flow-match schedule: sigmas ascend linspace(0,1,steps) with an
//!   appended 1.0 (final Δσ=0 step skipped, bit-identical); DiT timestep = σ.
//! - CFG: cond-first batch, uncond context = zeros, v = v_u + g·(v_c − v_u).
//! - VAE decode: latents ÷ scale_factor FIRST; Fourier embed (x, sin, cos)
//!   coordinate-major, no π, dim 51; geo cross-attn K/V precomputed per mesh.
//! - Volume grid: R+1 inclusive samples per axis, x-major; MC vertex world
//!   transform uses the reference's /(R+1) off-by-one (NOT /R).

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const sse = @import("gen_sse.zig");
const mc = @import("marching_cubes.zig");
const glb = @import("glb.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

const LN_EPS: f32 = 1e-6;

// ── Low-level mlx helpers (file-local primitives, mirror krea.zig) ──

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
inline fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, LN_EPS, s));
    return o;
}
inline fn layerNorm(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&o, x, w, b, LN_EPS, s));
    return o;
}
fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}
/// Slice [start,stop) on `axis` of an N-D array (N ≤ 8).
fn sliceAxis(x: mlx.mlx_array, axis: usize, start: c_int, stop: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const nd = sh.len;
    var lo: [8]c_int = undefined;
    var hi: [8]c_int = undefined;
    var st: [8]c_int = undefined;
    for (0..nd) |i| {
        lo[i] = 0;
        hi[i] = sh[i];
        st[i] = 1;
    }
    lo[axis] = start;
    hi[axis] = stop;
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, lo[0..nd].ptr, nd, hi[0..nd].ptr, nd, st[0..nd].ptr, nd, s));
    return o;
}
fn scalarF(v: f32) mlx.mlx_array {
    return mlx.mlx_array_new_float(v);
}
/// Exact (erf) GELU — every GELU in this model (t_embedder, DiT FFN experts,
/// DINO MLP, VAE MLP) is torch's default approximate="none".
fn geluErf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const inv_sqrt2 = scalarF(0.7071067811865476);
    defer _ = mlx.mlx_array_free(inv_sqrt2);
    const xs = try mulA(x, inv_sqrt2, s);
    defer _ = mlx.mlx_array_free(xs);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_erf(&e, xs, s));
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const opt = try addA(e, one, s);
    defer _ = mlx.mlx_array_free(opt);
    const half = scalarF(0.5);
    defer _ = mlx.mlx_array_free(half);
    const hx = try mulA(x, half, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, opt, s);
}
/// SDPA without mask: q/k/v [B,H,L,hd].
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", null_a, null_a, s));
    return o;
}
/// [B,L,H*hd] → [B,H,L,hd]
fn splitHeads(x: mlx.mlx_array, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const x4 = try reshape(x, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(x4);
    return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
}
/// [B,H,L,hd] → [B,L,H*hd]
fn mergeHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x); // [B,H,L,hd]
    const t = try transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, s);
}

pub fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[hy3d] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingHy3dWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
pub fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}
pub fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}
/// Load a norm vector (weight or bias), cast f32 (mlx fast norms upcast the
/// activations internally; f32 params close bf16/f16 rounding drift for free).
pub fn normVec(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}

/// Load ONE safetensors file into a Weights map. The three component files all
/// use `blocks.N.*` namespaces that would collide in a whole-dir load, so each
/// is loaded separately. Safetensors load runs on a CPU stream (Load::eval_gpu
/// is Not Implemented — the GPU-stream path kills the whole server). The
/// iterator hands a +1 reference in `value`; transfer it straight into the map
/// (the model.zig pattern) — copying and dropping it leaks every tensor.
pub fn loadFileWeights(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, cpu_s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = try allocator.dupe(u8, std.mem.span(key.?));
        errdefer allocator.free(owned_key);
        try w.map.put(owned_key, value);
    }
    log.info("[hy3d] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// MixedLinear — fp16 OR affine-quantized, bits/group_size inferred from
// tensor geometry (the krea.zig primitive, dense dtype float16 here).
// ════════════════════════════════════════════════════════════════════════

pub const MixedLinear = struct {
    quantized: bool,
    w: mlx.mlx_array, // quantized: packed u32 [out, in*bits/32]; dense: pre-transposed [in,out] f16
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    add_bias: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,

    /// `in_features` is the module's input dim (known per call); used only on
    /// the quantized path to solve (bits, group_size) from packed geometry.
    pub fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !MixedLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);

        if (ownOpt(w, sk)) |scales| {
            const weight = try ownWeight(w, wk);
            const biases = try ownWeight(w, bk);
            const w_cols: u32 = @intCast(mlx.getShape(weight)[1]); // in*bits/32
            const s_cols: u32 = @intCast(mlx.getShape(scales)[1]); // in/group_size
            const bits: u32 = @intCast(@divExact(32 * w_cols, in_features));
            const gs: u32 = @intCast(@divExact(in_features, s_cols));
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = biases,
                .add_bias = ownOpt(w, ak),
                .bits = bits,
                .group_size = gs,
            };
        }
        // Dense: pre-transpose [out,in] → [in,out], materialize, cast f16.
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        const wt = try astype(tc, .float16, s);
        return .{ .quantized = false, .w = wt, .add_bias = ownOpt(w, ak) };
    }

    pub fn deinit(self: *MixedLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }

    pub fn forward(self: *const MixedLinear, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const xh = try astype(x, .float16, s); // engine computes in f16
        defer _ = mlx.mlx_array_free(xh);
        var o = mlx.mlx_array_new();
        if (self.quantized) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, xh, self.w, self.scales, self.biases, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            try mlx.check(mlx.mlx_matmul(&o, xh, self.w, s));
        }
        if (self.add_bias) |b| {
            const r = try addA(o, b, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
};

/// Stacked per-expert linear [E,out,in] for gather-dispatch MoE. Quantized:
/// gather_qmm with transpose_w=true (weights stay [E,out,in_packed]); dense:
/// pre-transposed to [E,in,out] at load for gather_mm — which must be called
/// with sorted_indices=false (the mlx 0.31.2 dense kernel returns wrong
/// results with the flag set; it is only a perf hint, see transformer.zig).
const ExpertLinear = struct {
    quantized: bool,
    w: mlx.mlx_array,
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array, // per-expert additive bias [E,out] f16
    bits: u32 = 0,
    group_size: u32 = 0,

    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !ExpertLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);
        const add_bias = blk: {
            const raw = try ownWeight(w, ak);
            defer _ = mlx.mlx_array_free(raw);
            break :blk try astype(raw, .float16, s);
        };
        errdefer _ = mlx.mlx_array_free(add_bias);

        if (ownOpt(w, sk)) |scales| {
            const weight = try ownWeight(w, wk);
            const biases = try ownWeight(w, bk);
            const w_cols: u32 = @intCast(mlx.getShape(weight)[2]); // in*bits/32
            const s_cols: u32 = @intCast(mlx.getShape(scales)[2]); // in/group_size
            const bits: u32 = @intCast(@divExact(32 * w_cols, in_features));
            const gs: u32 = @intCast(@divExact(in_features, s_cols));
            return .{
                .quantized = true,
                .w = weight,
                .scales = scales,
                .biases = biases,
                .bias = add_bias,
                .bits = bits,
                .group_size = gs,
            };
        }
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 0, 2, 1 }, s);
        defer _ = mlx.mlx_array_free(t);
        // gather_mm reads expert blocks row-contiguous — a lazy transpose view
        // silently computes garbage at checkpoint scale. Materialize.
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        const wt = try astype(tc, .float16, s);
        return .{ .quantized = false, .w = wt, .bias = add_bias };
    }

    fn deinit(self: *ExpertLinear) void {
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        _ = mlx.mlx_array_free(self.bias);
    }

    /// x_rep [N,1,in] pre-gathered rows; sorted_inds [N] expert ids (ascending).
    /// Returns [N,out] with the per-expert bias added.
    fn gatherForward(self: *const ExpertLinear, x_rep: mlx.mlx_array, sorted_inds: mlx.mlx_array, s: S) !mlx.mlx_array {
        const no_idx = mlx.mlx_array{ .ctx = null };
        var o3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o3);
        if (self.quantized) {
            try mlx.check(mlx.mlx_gather_qmm(&o3, x_rep, self.w, self.scales, self.biases, no_idx, sorted_inds, true, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", true, s));
        } else {
            try mlx.check(mlx.mlx_gather_mm(&o3, x_rep, self.w, no_idx, sorted_inds, false, s));
        }
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_squeeze(&o, o3, s)); // [N,out]
        var bsel = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bsel);
        try mlx.check(mlx.mlx_take_axis(&bsel, self.bias, sorted_inds, 0, s)); // [N,out]
        return addA(o, bsel, s);
    }
};

// ── Config ──

pub const Hy3dConfig = struct {
    hidden: u32 = 2048,
    depth: u32 = 21,
    heads: u32 = 16,
    context_dim: u32 = 1024,
    num_latents: u32 = 4096,
    embed_dim: u32 = 64,
    vae_width: u32 = 1024,
    vae_heads: u32 = 16,
    vae_decoder_layers: u32 = 16,
    num_freqs: u32 = 8,
    scale_factor: f32 = 1.0039506158752403,
    num_moe_layers: u32 = 6,
    num_experts: u32 = 8,
    moe_top_k: u32 = 2,
    dino_hidden: u32 = 1024,
    dino_layers: u32 = 24,
    dino_heads: u32 = 16,
    dino_patch: u32 = 14,
    dino_image_size: u32 = 518,

    pub fn headDim(self: Hy3dConfig) u32 {
        return self.hidden / self.heads; // 128
    }
    pub fn vaeHeadDim(self: Hy3dConfig) u32 {
        return self.vae_width / self.vae_heads; // 64
    }
    /// Fourier embed output dim: 3·(2·num_freqs + 1) = 51.
    pub fn fourierDim(self: Hy3dConfig) u32 {
        return 3 * (2 * self.num_freqs + 1);
    }
    pub fn dinoTokens(self: Hy3dConfig) u32 {
        const side = self.dino_image_size / self.dino_patch; // 37
        return side * side + 1; // + CLS = 1370
    }
};

/// Parse the synthesized config.json text (pure — hermetically tested).
pub fn parseConfigText(allocator: std.mem.Allocator, text: []const u8) !Hy3dConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadHy3dConfig;
    const obj = parsed.value.object;
    const mt = obj.get("model_type") orelse return error.BadHy3dConfig;
    if (mt != .string or !std.mem.startsWith(u8, mt.string, "hunyuan3d")) return error.BadHy3dConfig;
    var cfg = Hy3dConfig{};
    inline for (@typeInfo(Hy3dConfig).@"struct".fields) |f| {
        if (obj.get(f.name)) |v| {
            switch (f.type) {
                u32 => {
                    if (v == .integer) @field(cfg, f.name) = @intCast(v.integer);
                },
                f32 => {
                    if (v == .float) @field(cfg, f.name) = @floatCast(v.float);
                    if (v == .integer) @field(cfg, f.name) = @floatFromInt(v.integer);
                },
                else => {},
            }
        }
    }
    // Contract spellings that differ from the struct field names.
    if (obj.get("hidden_size")) |v| {
        if (v == .integer) cfg.hidden = @intCast(v.integer);
    }
    if (obj.get("num_heads")) |v| {
        if (v == .integer) cfg.heads = @intCast(v.integer);
    }
    return cfg;
}

fn readConfigFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Hy3dConfig {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.BadHy3dConfig;
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(allocator, .limited(1024 * 1024));
    defer allocator.free(content);
    return parseConfigText(allocator, content);
}

// ════════════════════════════════════════════════════════════════════════
// Image preprocessing (ImageProcessorV2: recenter border_ratio 0.15,
// composite-on-white, 512², then the DINO transform → 518² ImageNet-normed).
// All CPU f32 — the oracle taps AFTER this stage to decouple resize kernels.
// ════════════════════════════════════════════════════════════════════════

const IMAGENET_MEAN = [3]f32{ 0.485, 0.456, 0.406 };
const IMAGENET_STD = [3]f32{ 0.229, 0.224, 0.225 };
const RECENTER_SIZE: usize = 512;
const BORDER_RATIO: f32 = 0.15;

/// Straight-alpha RGBA8 → CHW f32 [3·518·518], ImageNet-normalized. The alpha
/// bbox is recentered onto a white 512² canvas (longest side scaled to
/// (1−border_ratio)·512), composited over white, bilinear-resized to 518, then
/// normalized. Fully opaque images (server fallback: no cutout) recenter the
/// whole frame. Caller frees the returned slice.
pub fn preprocessImage(allocator: std.mem.Allocator, rgba: []const u8, w: u32, h: u32, out_size: u32) ![]f32 {
    if (rgba.len < @as(usize, w) * h * 4 or w == 0 or h == 0) return error.BadImage;
    const sw: usize = w;
    const sh: usize = h;

    // Alpha bbox (alpha > 0). All-transparent → whole image.
    var min_x: usize = sw;
    var min_y: usize = sh;
    var max_x: usize = 0;
    var max_y: usize = 0;
    var any = false;
    for (0..sh) |y| {
        for (0..sw) |x| {
            if (rgba[(y * sw + x) * 4 + 3] > 0) {
                any = true;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }
    if (!any) {
        min_x = 0;
        min_y = 0;
        max_x = sw - 1;
        max_y = sh - 1;
    }
    const bw: f32 = @floatFromInt(max_x - min_x + 1);
    const bh: f32 = @floatFromInt(max_y - min_y + 1);

    // Recenter onto a white 512² canvas.
    const cs: usize = RECENTER_SIZE;
    const canvas = try allocator.alloc(f32, 3 * cs * cs);
    defer allocator.free(canvas);
    @memset(canvas, 1.0);
    const desired: f32 = (1.0 - BORDER_RATIO) * @as(f32, @floatFromInt(cs));
    const scale: f32 = desired / @max(bw, bh);
    const nw: usize = @intFromFloat(@round(bw * scale));
    const nh: usize = @intFromFloat(@round(bh * scale));
    const x0 = (cs - @min(nw, cs)) / 2;
    const y0 = (cs - @min(nh, cs)) / 2;

    const clampIdx = struct {
        fn f(v: isize, lo: usize, hi: usize) usize {
            if (v < @as(isize, @intCast(lo))) return lo;
            const uv: usize = @intCast(v);
            return if (uv > hi) hi else uv;
        }
    }.f;

    // Bilinear sample the bbox region (composited over white) into the paste rect.
    var oy: usize = 0;
    while (oy < @min(nh, cs)) : (oy += 1) {
        const fy = @as(f32, @floatFromInt(min_y)) + (@as(f32, @floatFromInt(oy)) + 0.5) * bh / @as(f32, @floatFromInt(nh)) - 0.5;
        const fy0 = @floor(fy);
        const wy = fy - fy0;
        const ya = clampIdx(@intFromFloat(fy0), min_y, max_y);
        const yb = clampIdx(@as(isize, @intFromFloat(fy0)) + 1, min_y, max_y);
        var ox: usize = 0;
        while (ox < @min(nw, cs)) : (ox += 1) {
            const fx = @as(f32, @floatFromInt(min_x)) + (@as(f32, @floatFromInt(ox)) + 0.5) * bw / @as(f32, @floatFromInt(nw)) - 0.5;
            const fx0 = @floor(fx);
            const wx = fx - fx0;
            const xa = clampIdx(@intFromFloat(fx0), min_x, max_x);
            const xb = clampIdx(@as(isize, @intFromFloat(fx0)) + 1, min_x, max_x);
            var c: usize = 0;
            while (c < 3) : (c += 1) {
                const sample = struct {
                    fn f(px: []const u8, stride: usize, xx: usize, yy: usize, ch: usize) f32 {
                        const p = (yy * stride + xx) * 4;
                        const a_: f32 = @as(f32, @floatFromInt(px[p + 3])) / 255.0;
                        const cv: f32 = @as(f32, @floatFromInt(px[p + ch])) / 255.0;
                        return cv * a_ + (1.0 - a_); // composite over white
                    }
                }.f;
                const p00 = sample(rgba, sw, xa, ya, c);
                const p10 = sample(rgba, sw, xb, ya, c);
                const p01 = sample(rgba, sw, xa, yb, c);
                const p11 = sample(rgba, sw, xb, yb, c);
                const top = p00 * (1.0 - wx) + p10 * wx;
                const bot = p01 * (1.0 - wx) + p11 * wx;
                canvas[c * cs * cs + (y0 + oy) * cs + (x0 + ox)] = top * (1.0 - wy) + bot * wy;
            }
        }
    }

    // The reference maps to [−1,1] then back to [0,1] ((x+1)/2) — identity.
    // Bilinear 512 → out_size (518, an upsample: antialias is a no-op), then
    // ImageNet normalize, CHW.
    const os: usize = out_size;
    const out = try allocator.alloc(f32, 3 * os * os);
    errdefer allocator.free(out);
    var ty: usize = 0;
    while (ty < os) : (ty += 1) {
        const fy = (@as(f32, @floatFromInt(ty)) + 0.5) * @as(f32, @floatFromInt(cs)) / @as(f32, @floatFromInt(os)) - 0.5;
        const fy0 = @floor(fy);
        const wy = fy - fy0;
        const ya = clampIdx(@intFromFloat(fy0), 0, cs - 1);
        const yb = clampIdx(@as(isize, @intFromFloat(fy0)) + 1, 0, cs - 1);
        var tx: usize = 0;
        while (tx < os) : (tx += 1) {
            const fx = (@as(f32, @floatFromInt(tx)) + 0.5) * @as(f32, @floatFromInt(cs)) / @as(f32, @floatFromInt(os)) - 0.5;
            const fx0 = @floor(fx);
            const wx = fx - fx0;
            const xa = clampIdx(@intFromFloat(fx0), 0, cs - 1);
            const xb = clampIdx(@as(isize, @intFromFloat(fx0)) + 1, 0, cs - 1);
            var c: usize = 0;
            while (c < 3) : (c += 1) {
                const base = c * cs * cs;
                const p00 = canvas[base + ya * cs + xa];
                const p10 = canvas[base + ya * cs + xb];
                const p01 = canvas[base + yb * cs + xa];
                const p11 = canvas[base + yb * cs + xb];
                const top = p00 * (1.0 - wx) + p10 * wx;
                const bot = p01 * (1.0 - wx) + p11 * wx;
                const v = top * (1.0 - wy) + bot * wy;
                out[c * os * os + ty * os + tx] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Conditioner — DINOv2-Large ViT (24 layers, layerscale, CLS+patch tokens).
// ════════════════════════════════════════════════════════════════════════

const DinoLayer = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    ls1: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    fc1: MixedLinear,
    fc2: MixedLinear,
    ls2: mlx.mlx_array,
    fn deinit(self: *DinoLayer) void {
        _ = mlx.mlx_array_free(self.ln1_w);
        _ = mlx.mlx_array_free(self.ln1_b);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.ls1);
        _ = mlx.mlx_array_free(self.ln2_w);
        _ = mlx.mlx_array_free(self.ln2_b);
        self.fc1.deinit();
        self.fc2.deinit();
        _ = mlx.mlx_array_free(self.ls2);
    }
};

pub const DinoEncoder = struct {
    cfg: Hy3dConfig,
    allocator: std.mem.Allocator,
    s: S,
    cls_token: mlx.mlx_array, // [1,1,1024] f16
    pos_embed: mlx.mlx_array, // [1,1370,1024] f16
    patch_w: mlx.mlx_array, // [1024,14,14,3] f16 (MLX conv layout)
    patch_b: mlx.mlx_array, // [1024] f16
    layers: []DinoLayer,
    norm_w: mlx.mlx_array,
    norm_b: mlx.mlx_array,

    pub fn deinit(self: *DinoEncoder) void {
        _ = mlx.mlx_array_free(self.cls_token);
        _ = mlx.mlx_array_free(self.pos_embed);
        _ = mlx.mlx_array_free(self.patch_w);
        _ = mlx.mlx_array_free(self.patch_b);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.norm_b);
    }

    /// pixels [1,3,S,S] f32 (post-preprocess) → features [1,1370,1024] f16.
    pub fn encode(self: *DinoEncoder, pixels: mlx.mlx_array) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const heads: c_int = @intCast(c.dino_heads);
        const hd: c_int = @intCast(c.dino_hidden / c.dino_heads);
        const side: c_int = @intCast(c.dino_image_size / c.dino_patch);
        const H: c_int = @intCast(c.dino_hidden);

        // NCHW → NHWC f16, patch conv (stride = patch), → [1, side², H].
        const nhwc0 = try transpose(pixels, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(nhwc0);
        const nhwc1 = try contig(nhwc0, s);
        defer _ = mlx.mlx_array_free(nhwc1);
        const nhwc = try astype(nhwc1, .float16, s);
        defer _ = mlx.mlx_array_free(nhwc);
        var convd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(convd);
        const p: c_int = @intCast(c.dino_patch);
        try mlx.check(mlx.mlx_conv2d(&convd, nhwc, self.patch_w, p, p, 0, 0, 1, 1, 1, s));
        const flat = try reshape(convd, &[_]c_int{ 1, side * side, H }, s);
        defer _ = mlx.mlx_array_free(flat);
        const patched = try addA(flat, self.patch_b, s);
        defer _ = mlx.mlx_array_free(patched);

        // CLS first, + learned pos embed.
        const with_cls = try concat(&[_]mlx.mlx_array{ self.cls_token, patched }, 1, s);
        defer _ = mlx.mlx_array_free(with_cls);
        var x = try addA(with_cls, self.pos_embed, s);

        for (self.layers) |*layer| {
            const nx = try self.layerForward(x, layer, heads, hd, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        defer _ = mlx.mlx_array_free(x);
        return layerNorm(x, self.norm_w, self.norm_b, s);
    }

    fn layerForward(self: *DinoEncoder, x: mlx.mlx_array, layer: *const DinoLayer, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        _ = self;
        const n1 = try layerNorm(x, layer.ln1_w, layer.ln1_b, s);
        defer _ = mlx.mlx_array_free(n1);
        const q0 = try layer.q.forward(n1, s);
        defer _ = mlx.mlx_array_free(q0);
        const k0 = try layer.k.forward(n1, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try layer.v.forward(n1, s);
        defer _ = mlx.mlx_array_free(v0);
        const q = try splitHeads(q0, heads, hd, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try splitHeads(k0, heads, hd, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try splitHeads(v0, heads, hd, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(@as(u32, @intCast(hd)))));
        const attn = try sdpa(q, k, v, scale, s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try layer.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const o_ls = try mulA(o, layer.ls1, s);
        defer _ = mlx.mlx_array_free(o_ls);
        const h1 = try addA(x, o_ls, s);
        defer _ = mlx.mlx_array_free(h1);
        const n2 = try layerNorm(h1, layer.ln2_w, layer.ln2_b, s);
        defer _ = mlx.mlx_array_free(n2);
        const f1 = try layer.fc1.forward(n2, s);
        defer _ = mlx.mlx_array_free(f1);
        const g = try geluErf(f1, s);
        defer _ = mlx.mlx_array_free(g);
        const f2 = try layer.fc2.forward(g, s);
        defer _ = mlx.mlx_array_free(f2);
        const f_ls = try mulA(f2, layer.ls2, s);
        defer _ = mlx.mlx_array_free(f_ls);
        return addA(h1, f_ls, s);
    }
};

pub fn loadDino(allocator: std.mem.Allocator, cfg: Hy3dConfig, model_dir: []const u8, s: S) !DinoEncoder {
    var w = try loadFileWeights(allocator, model_dir, "conditioner.safetensors");
    defer w.deinit();
    var d: DinoEncoder = undefined;
    d.cfg = cfg;
    d.allocator = allocator;
    d.s = s;
    d.cls_token = blk: {
        const raw = try ownWeight(&w, "cls_token");
        defer _ = mlx.mlx_array_free(raw);
        break :blk try astype(raw, .float16, s);
    };
    d.pos_embed = blk: {
        const raw = try ownWeight(&w, "pos_embed");
        defer _ = mlx.mlx_array_free(raw);
        break :blk try astype(raw, .float16, s);
    };
    if (mlx.getShape(d.pos_embed)[1] != @as(c_int, @intCast(cfg.dinoTokens()))) {
        log.err("[hy3d] pos_embed token count {d} != expected {d}\n", .{ mlx.getShape(d.pos_embed)[1], cfg.dinoTokens() });
        return error.BadHy3dWeights;
    }
    d.patch_w = blk: {
        // torch [O,I,kH,kW] → MLX conv2d [O,kH,kW,I], materialized.
        const raw = try ownWeight(&w, "patch_embed.weight");
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        break :blk try astype(tc, .float16, s);
    };
    d.patch_b = blk: {
        const raw = try ownWeight(&w, "patch_embed.bias");
        defer _ = mlx.mlx_array_free(raw);
        break :blk try astype(raw, .float16, s);
    };
    d.layers = try allocator.alloc(DinoLayer, cfg.dino_layers);
    const H = cfg.dino_hidden;
    for (d.layers, 0..) |*layer, i| {
        const kq = try fmtKey(allocator, "layers.{d}.attn.q", .{i});
        defer allocator.free(kq);
        const kk = try fmtKey(allocator, "layers.{d}.attn.k", .{i});
        defer allocator.free(kk);
        const kv = try fmtKey(allocator, "layers.{d}.attn.v", .{i});
        defer allocator.free(kv);
        const ko = try fmtKey(allocator, "layers.{d}.attn.out", .{i});
        defer allocator.free(ko);
        const kf1 = try fmtKey(allocator, "layers.{d}.mlp.fc1", .{i});
        defer allocator.free(kf1);
        const kf2 = try fmtKey(allocator, "layers.{d}.mlp.fc2", .{i});
        defer allocator.free(kf2);
        const n1w = try fmtKey(allocator, "layers.{d}.norm1.weight", .{i});
        defer allocator.free(n1w);
        const n1b = try fmtKey(allocator, "layers.{d}.norm1.bias", .{i});
        defer allocator.free(n1b);
        const n2w = try fmtKey(allocator, "layers.{d}.norm2.weight", .{i});
        defer allocator.free(n2w);
        const n2b = try fmtKey(allocator, "layers.{d}.norm2.bias", .{i});
        defer allocator.free(n2b);
        const l1 = try fmtKey(allocator, "layers.{d}.ls1", .{i});
        defer allocator.free(l1);
        const l2 = try fmtKey(allocator, "layers.{d}.ls2", .{i});
        defer allocator.free(l2);
        layer.* = .{
            .ln1_w = try normVec(&w, n1w, s),
            .ln1_b = try normVec(&w, n1b, s),
            .q = try MixedLinear.load(&w, allocator, kq, H, s),
            .k = try MixedLinear.load(&w, allocator, kk, H, s),
            .v = try MixedLinear.load(&w, allocator, kv, H, s),
            .o = try MixedLinear.load(&w, allocator, ko, H, s),
            .ls1 = blk: {
                const raw = try ownWeight(&w, l1);
                defer _ = mlx.mlx_array_free(raw);
                break :blk try astype(raw, .float16, s);
            },
            .ln2_w = try normVec(&w, n2w, s),
            .ln2_b = try normVec(&w, n2b, s),
            .fc1 = try MixedLinear.load(&w, allocator, kf1, H, s),
            .fc2 = try MixedLinear.load(&w, allocator, kf2, H * 4, s),
            .ls2 = blk: {
                const raw = try ownWeight(&w, l2);
                defer _ = mlx.mlx_array_free(raw);
                break :blk try astype(raw, .float16, s);
            },
        };
    }
    d.norm_w = try normVec(&w, "norm.weight", s);
    d.norm_b = try normVec(&w, "norm.bias", s);
    log.info("[hy3d] DINOv2-L conditioner ready ({d} layers)\n", .{cfg.dino_layers});
    return d;
}

// ════════════════════════════════════════════════════════════════════════
// Denoiser — HunYuanDiTPlain (depth 21, hidden 2048, U-ViT skips, timestep
// token, per-head RMSNorm(128) qk-norm, MoE on the last 6 blocks).
// ════════════════════════════════════════════════════════════════════════

const DitAttn = struct {
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    out: MixedLinear,
    q_norm: mlx.mlx_array, // RMSNorm weight [128] f32
    k_norm: mlx.mlx_array,
    fn deinit(self: *DitAttn) void {
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.out.deinit();
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
    }
    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, q_in: u32, kv_in: u32, s: S) !DitAttn {
        const kq = try fmtKey(a, "{s}.q", .{pfx});
        defer a.free(kq);
        const kk = try fmtKey(a, "{s}.k", .{pfx});
        defer a.free(kk);
        const kv = try fmtKey(a, "{s}.v", .{pfx});
        defer a.free(kv);
        const ko = try fmtKey(a, "{s}.out", .{pfx});
        defer a.free(ko);
        const kqn = try fmtKey(a, "{s}.q_norm.weight", .{pfx});
        defer a.free(kqn);
        const kkn = try fmtKey(a, "{s}.k_norm.weight", .{pfx});
        defer a.free(kkn);
        return .{
            .q = try MixedLinear.load(w, a, kq, q_in, s),
            .k = try MixedLinear.load(w, a, kk, kv_in, s),
            .v = try MixedLinear.load(w, a, kv, kv_in, s),
            .out = try MixedLinear.load(w, a, ko, q_in, s),
            .q_norm = try normVec(w, kqn, s),
            .k_norm = try normVec(w, kkn, s),
        };
    }
    /// q from `xq` [B,Lq,2048]; k/v from `xkv` [B,Lkv,·]; per-head RMS on q,k.
    /// NOTE: the reference forward RE-FUSES q/k/v with a per-head interleave;
    /// the convert script BAKES that permutation into the emitted q/k/v
    /// tensors, so this reshape must stay completely STANDARD — adding any
    /// interleave here would double-scramble.
    fn forward(self: *const DitAttn, xq: mlx.mlx_array, xkv: mlx.mlx_array, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
        const q0 = try self.q.forward(xq, s);
        defer _ = mlx.mlx_array_free(q0);
        const k0 = try self.k.forward(xkv, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try self.v.forward(xkv, s);
        defer _ = mlx.mlx_array_free(v0);
        const qsh = mlx.getShape(q0);
        const ksh = mlx.getShape(k0);
        // [B,L,H,hd]: per-head RMSNorm over the last axis, THEN head transpose.
        const q4 = try reshape(q0, &[_]c_int{ qsh[0], qsh[1], heads, hd }, s);
        defer _ = mlx.mlx_array_free(q4);
        const qn = try rmsNorm(q4, self.q_norm, s);
        defer _ = mlx.mlx_array_free(qn);
        const q = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k4 = try reshape(k0, &[_]c_int{ ksh[0], ksh[1], heads, hd }, s);
        defer _ = mlx.mlx_array_free(k4);
        const kn = try rmsNorm(k4, self.k_norm, s);
        defer _ = mlx.mlx_array_free(kn);
        const k = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(k);
        const v4 = try reshape(v0, &[_]c_int{ ksh[0], ksh[1], heads, hd }, s);
        defer _ = mlx.mlx_array_free(v4);
        const v = try transpose(v4, &[_]c_int{ 0, 2, 1, 3 }, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(@as(u32, @intCast(hd)))));
        const attn = try sdpa(q, k, v, scale, s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        return self.out.forward(merged, s);
    }
};

const Mlp = struct {
    fc1: MixedLinear,
    fc2: MixedLinear,
    fn deinit(self: *Mlp) void {
        self.fc1.deinit();
        self.fc2.deinit();
    }
    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, in_dim: u32, inter: u32, s: S) !Mlp {
        const k1 = try fmtKey(a, "{s}.fc1", .{pfx});
        defer a.free(k1);
        const k2 = try fmtKey(a, "{s}.fc2", .{pfx});
        defer a.free(k2);
        return .{
            .fc1 = try MixedLinear.load(w, a, k1, in_dim, s),
            .fc2 = try MixedLinear.load(w, a, k2, inter, s),
        };
    }
    fn forward(self: *const Mlp, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const f1 = try self.fc1.forward(x, s);
        defer _ = mlx.mlx_array_free(f1);
        const g = try geluErf(f1, s);
        defer _ = mlx.mlx_array_free(g);
        return self.fc2.forward(g, s);
    }
};

const Moe = struct {
    gate_wt: mlx.mlx_array, // pre-transposed [2048, E] f16
    experts_fc1: ExpertLinear,
    experts_fc2: ExpertLinear,
    shared: Mlp,
    top_k: u32,
    fn deinit(self: *Moe) void {
        _ = mlx.mlx_array_free(self.gate_wt);
        self.experts_fc1.deinit();
        self.experts_fc2.deinit();
        self.shared.deinit();
    }
};

/// MoE routing: softmax over ALL experts FIRST, then top-k on the
/// probabilities, NO renormalization (norm_topk_prob=False). Returns owned
/// `inds` (int32/uint32 [..,k]) and `weights` (f32 [..,k]).
pub fn moeRoute(logits: mlx.mlx_array, k: c_int, s: S) !struct { inds: mlx.mlx_array, weights: mlx.mlx_array } {
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, logits, -1, true, s));
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, probs, s));
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, neg, k - 1, -1, s));
    const psh = mlx.getShape(part);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var lo: [8]c_int = undefined;
        var hi: [8]c_int = undefined;
        var st: [8]c_int = undefined;
        for (0..psh.len) |d| {
            lo[d] = 0;
            hi[d] = if (d == psh.len - 1) k else psh[d];
            st[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, part, lo[0..psh.len].ptr, psh.len, hi[0..psh.len].ptr, psh.len, st[0..psh.len].ptr, psh.len, s));
    }
    var pf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pf);
    try mlx.check(mlx.mlx_astype(&pf, probs, .float32, s));
    var weights = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(weights);
    try mlx.check(mlx.mlx_take_along_axis(&weights, pf, inds, -1, s));
    return .{ .inds = inds, .weights = weights };
}

/// Full MoE FFN on x [B,L,2048]: routed experts (sorted gather dispatch, the
/// transformer.zig prefill recipe) + the always-on shared expert.
fn moeForward(moe: *const Moe, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B = sh[0];
    const L = sh[1];
    const D = sh[2];
    const K: c_int = @intCast(moe.top_k);
    const N: c_int = B * L;

    const xh = try astype(x, .float16, s);
    defer _ = mlx.mlx_array_free(xh);
    const logits = try matmul(xh, moe.gate_wt, s); // [B,L,E]
    defer _ = mlx.mlx_array_free(logits);
    const routed = try moeRoute(logits, K, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.weights);

    // Sorted gather dispatch.
    const flat_shape = [_]c_int{N * K};
    var flat_inds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_inds);
    try mlx.check(mlx.mlx_reshape(&flat_inds, routed.inds, &flat_shape, 1, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, s));
    var inv_order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv_order);
    try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, s));
    var sorted_inds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_inds);
    try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, s));
    const k_arr = mlx.mlx_array_new_int(K);
    defer _ = mlx.mlx_array_free(k_arr);
    var lhs_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lhs_idx);
    try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, s));

    const flat_x_shape = [_]c_int{ N, D };
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_reshape(&x_flat, xh, &flat_x_shape, 2, s));
    var x_gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_gathered);
    try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, s));
    const rep_shape = [_]c_int{ N * K, 1, D };
    var x_rep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_rep);
    try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &rep_shape, 3, s));

    const h1 = try moe.experts_fc1.gatherForward(x_rep, sorted_inds, s); // [NK, inter]
    defer _ = mlx.mlx_array_free(h1);
    const g = try geluErf(h1, s);
    defer _ = mlx.mlx_array_free(g);
    var g3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g3);
    try mlx.check(mlx.mlx_expand_dims(&g3, g, -2, s)); // [NK,1,inter]
    const h2 = try moe.experts_fc2.gatherForward(g3, sorted_inds, s); // [NK, D]
    defer _ = mlx.mlx_array_free(h2);

    // Inverse permutation → original (token-major) order → [B,L,K,D].
    var unsorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unsorted);
    try mlx.check(mlx.mlx_take_axis(&unsorted, h2, inv_order, 0, s));
    const blkd = try reshape(unsorted, &[_]c_int{ B, L, K, D }, s);
    defer _ = mlx.mlx_array_free(blkd);

    // Weighted sum over K — weights stay UNNORMALIZED (softmax probs).
    const wh = try astype(routed.weights, .float16, s);
    defer _ = mlx.mlx_array_free(wh);
    var w4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w4);
    try mlx.check(mlx.mlx_expand_dims(&w4, wh, -1, s)); // [B,L,K,1]
    const weighted = try mulA(blkd, w4, s);
    defer _ = mlx.mlx_array_free(weighted);
    var routed_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed_sum);
    try mlx.check(mlx.mlx_sum_axis(&routed_sum, weighted, 2, false, s)); // [B,L,D]

    // Always-on shared expert.
    const shared_out = try moe.shared.forward(xh, s);
    defer _ = mlx.mlx_array_free(shared_out);
    return addA(routed_sum, shared_out, s);
}

const Ffn = union(enum) {
    mlp: Mlp,
    moe: Moe,
    fn deinit(self: *Ffn) void {
        switch (self.*) {
            .mlp => |*m| m.deinit(),
            .moe => |*m| m.deinit(),
        }
    }
};

const Skip = struct {
    lin: MixedLinear, // [2048, 4096]
    norm_w: mlx.mlx_array,
    norm_b: mlx.mlx_array,
    fn deinit(self: *Skip) void {
        self.lin.deinit();
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.norm_b);
    }
};

const DitBlock = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    attn1: DitAttn,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    attn2: DitAttn,
    norm3_w: mlx.mlx_array,
    norm3_b: mlx.mlx_array,
    ffn: Ffn,
    skip: ?Skip,
    fn deinit(self: *DitBlock) void {
        _ = mlx.mlx_array_free(self.norm1_w);
        _ = mlx.mlx_array_free(self.norm1_b);
        self.attn1.deinit();
        _ = mlx.mlx_array_free(self.norm2_w);
        _ = mlx.mlx_array_free(self.norm2_b);
        self.attn2.deinit();
        _ = mlx.mlx_array_free(self.norm3_w);
        _ = mlx.mlx_array_free(self.norm3_b);
        self.ffn.deinit();
        if (self.skip) |*sk| sk.deinit();
    }
};

pub const Dit = struct {
    cfg: Hy3dConfig,
    allocator: std.mem.Allocator,
    s: S,
    x_emb: MixedLinear, // 64 → 2048
    t_mlp1: MixedLinear,
    t_mlp2: MixedLinear,
    blocks: []DitBlock,
    final_norm_w: mlx.mlx_array,
    final_norm_b: mlx.mlx_array,
    final_lin: MixedLinear, // 2048 → 64

    pub fn deinit(self: *Dit) void {
        self.x_emb.deinit();
        self.t_mlp1.deinit();
        self.t_mlp2.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        _ = mlx.mlx_array_free(self.final_norm_w);
        _ = mlx.mlx_array_free(self.final_norm_b);
        self.final_lin.deinit();
    }

    /// One denoiser forward: x [B,4096,64] f16, cond [B,1370,1024] f16,
    /// sigma ∈ [0,1] (the DiT timestep IS σ) → velocity [B,4096,64] f16.
    pub fn forward(self: *Dit, x_in: mlx.mlx_array, cond: mlx.mlx_array, sigma: f32) !mlx.mlx_array {
        const s = self.s;
        const c = self.cfg;
        const B = mlx.getShape(x_in)[0];
        const heads: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.headDim());
        const half_push: usize = (c.depth - 1) / 2; // 10: blocks 0..9 push
        const pop_from: usize = half_push + 1; // 11: blocks 11..20 pop

        // Timestep token (f32 sincos → MLP → f16), broadcast over the batch.
        const t_vec = try timestepEmbed(self.allocator, sigma, c.hidden); // [1,dim] f32
        defer _ = mlx.mlx_array_free(t_vec);
        const t1 = try self.t_mlp1.forward(t_vec, s);
        defer _ = mlx.mlx_array_free(t1);
        const tg = try geluErf(t1, s);
        defer _ = mlx.mlx_array_free(tg);
        const t2 = try self.t_mlp2.forward(tg, s);
        defer _ = mlx.mlx_array_free(t2);
        const t3 = try reshape(t2, &[_]c_int{ 1, 1, @intCast(c.hidden) }, s);
        defer _ = mlx.mlx_array_free(t3);
        var t_tok = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t_tok);
        if (B == 1) {
            try mlx.check(mlx.mlx_array_set(&t_tok, t3));
        } else {
            var parts: [8]mlx.mlx_array = undefined;
            for (0..@intCast(B)) |i| parts[i] = t3;
            const cc = try concat(parts[0..@intCast(B)], 0, s);
            _ = mlx.mlx_array_free(t_tok);
            t_tok = cc;
        }

        const x_e = try self.x_emb.forward(x_in, s); // [B,4096,2048]
        defer _ = mlx.mlx_array_free(x_e);
        var h = try concat(&[_]mlx.mlx_array{ t_tok, x_e }, 1, s); // [B,4097,2048]

        var skips: [16]mlx.mlx_array = undefined;
        var n_skips: usize = 0;
        defer for (0..n_skips) |i| {
            _ = mlx.mlx_array_free(skips[i]);
        };

        for (self.blocks, 0..) |*blk, i| {
            if (i >= pop_from) {
                // Pop LIFO; fuse: LayerNorm(Linear(cat([skip, h], -1))).
                n_skips -= 1;
                const skip = skips[n_skips];
                const fused_in = try concat(&[_]mlx.mlx_array{ skip, h }, 2, s);
                _ = mlx.mlx_array_free(skip);
                defer _ = mlx.mlx_array_free(fused_in);
                const lin = try blk.skip.?.lin.forward(fused_in, s);
                defer _ = mlx.mlx_array_free(lin);
                const fused = try layerNorm(lin, blk.skip.?.norm_w, blk.skip.?.norm_b, s);
                _ = mlx.mlx_array_free(h);
                h = fused;
            }
            // Self-attn.
            {
                const n1 = try layerNorm(h, blk.norm1_w, blk.norm1_b, s);
                defer _ = mlx.mlx_array_free(n1);
                const a = try blk.attn1.forward(n1, n1, heads, hd, s);
                defer _ = mlx.mlx_array_free(a);
                const nh = try addA(h, a, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            // Cross-attn into the image tokens.
            {
                const n2 = try layerNorm(h, blk.norm2_w, blk.norm2_b, s);
                defer _ = mlx.mlx_array_free(n2);
                const a = try blk.attn2.forward(n2, cond, heads, hd, s);
                defer _ = mlx.mlx_array_free(a);
                const nh = try addA(h, a, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            // FFN (dense or MoE).
            {
                const n3 = try layerNorm(h, blk.norm3_w, blk.norm3_b, s);
                defer _ = mlx.mlx_array_free(n3);
                const f = switch (blk.ffn) {
                    .mlp => |*m| try m.forward(n3, s),
                    .moe => |*m| try moeForward(m, n3, s),
                };
                defer _ = mlx.mlx_array_free(f);
                const nh = try addA(h, f, s);
                _ = mlx.mlx_array_free(h);
                h = nh;
            }
            if (i < half_push) {
                var cp = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&cp, h));
                skips[n_skips] = cp;
                n_skips += 1;
            }
        }
        defer _ = mlx.mlx_array_free(h);

        const nf = try layerNorm(h, self.final_norm_w, self.final_norm_b, s);
        defer _ = mlx.mlx_array_free(nf);
        // Drop token 0 AFTER the final norm, then project to velocity.
        const body = try sliceAxis(nf, 1, 1, mlx.getShape(nf)[1], s);
        defer _ = mlx.mlx_array_free(body);
        return self.final_lin.forward(body, s);
    }
};

/// Sincos timestep embedding [1,dim] f32: half=dim/2, f_i = exp(−ln(1e4)·i/half),
/// emb = [sin(t·f_0..), cos(t·f_0..)] — SIN FIRST, computed in f32 (bf16/f16
/// sincos of tiny σ args is a parity killer).
pub fn timestepEmbed(allocator: std.mem.Allocator, t: f32, dim: u32) !mlx.mlx_array {
    const half = dim / 2;
    const buf = try allocator.alloc(f32, dim);
    defer allocator.free(buf);
    const ln1e4: f64 = std.math.log(f64, std.math.e, 10000.0);
    for (0..half) |i| {
        const f = std.math.exp(-ln1e4 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half)));
        const ang = @as(f64, t) * f;
        buf[i] = @floatCast(@sin(ang));
        buf[half + i] = @floatCast(@cos(ang));
    }
    const sh = [_]c_int{ 1, @intCast(dim) };
    return mlx.mlx_array_new_data(buf.ptr, &sh, 2, .float32);
}

/// Reversed flow-match sigma schedule: linspace(0,1,steps) ASCENDING with an
/// appended 1.0 (the reference's trailing σ; its Δσ=0 final step is skipped in
/// the loop, bit-identically). len = steps+1. Caller frees.
pub fn buildSigmas(allocator: std.mem.Allocator, steps: u32) ![]f32 {
    if (steps < 2) return error.BadSteps;
    const out = try allocator.alloc(f32, steps + 1);
    for (0..steps) |i| {
        out[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps - 1));
    }
    out[steps] = 1.0;
    return out;
}

pub fn loadDit(allocator: std.mem.Allocator, cfg: Hy3dConfig, model_dir: []const u8, s: S) !Dit {
    var w = try loadFileWeights(allocator, model_dir, "dit.safetensors");
    defer w.deinit();
    var d: Dit = undefined;
    d.cfg = cfg;
    d.allocator = allocator;
    d.s = s;
    d.x_emb = try MixedLinear.load(&w, allocator, "x_embedder", cfg.embed_dim, s);
    d.t_mlp1 = try MixedLinear.load(&w, allocator, "t_embedder.mlp1", cfg.hidden, s);
    const t_inter: u32 = @intCast(mlx.getShape(d.t_mlp1.w)[if (d.t_mlp1.quantized) 0 else 1]);
    d.t_mlp2 = try MixedLinear.load(&w, allocator, "t_embedder.mlp2", t_inter, s);
    d.blocks = try allocator.alloc(DitBlock, cfg.depth);
    const H = cfg.hidden;
    const moe_start: usize = cfg.depth - cfg.num_moe_layers; // 15
    const pop_from: usize = (cfg.depth - 1) / 2 + 1; // 11
    for (d.blocks, 0..) |*blk, i| {
        const n1w = try fmtKey(allocator, "blocks.{d}.norm1.weight", .{i});
        defer allocator.free(n1w);
        const n1b = try fmtKey(allocator, "blocks.{d}.norm1.bias", .{i});
        defer allocator.free(n1b);
        const n2w = try fmtKey(allocator, "blocks.{d}.norm2.weight", .{i});
        defer allocator.free(n2w);
        const n2b = try fmtKey(allocator, "blocks.{d}.norm2.bias", .{i});
        defer allocator.free(n2b);
        const n3w = try fmtKey(allocator, "blocks.{d}.norm3.weight", .{i});
        defer allocator.free(n3w);
        const n3b = try fmtKey(allocator, "blocks.{d}.norm3.bias", .{i});
        defer allocator.free(n3b);
        const ka1 = try fmtKey(allocator, "blocks.{d}.attn1", .{i});
        defer allocator.free(ka1);
        const ka2 = try fmtKey(allocator, "blocks.{d}.attn2", .{i});
        defer allocator.free(ka2);
        blk.* = .{
            .norm1_w = try normVec(&w, n1w, s),
            .norm1_b = try normVec(&w, n1b, s),
            .attn1 = try DitAttn.load(&w, allocator, ka1, H, H, s),
            .norm2_w = try normVec(&w, n2w, s),
            .norm2_b = try normVec(&w, n2b, s),
            .attn2 = try DitAttn.load(&w, allocator, ka2, H, cfg.context_dim, s),
            .norm3_w = try normVec(&w, n3w, s),
            .norm3_b = try normVec(&w, n3b, s),
            .ffn = undefined,
            .skip = null,
        };
        if (i >= moe_start) {
            const kg = try fmtKey(allocator, "blocks.{d}.moe.gate.weight", .{i});
            defer allocator.free(kg);
            const ke1 = try fmtKey(allocator, "blocks.{d}.moe.experts.fc1", .{i});
            defer allocator.free(ke1);
            const ke2 = try fmtKey(allocator, "blocks.{d}.moe.experts.fc2", .{i});
            defer allocator.free(ke2);
            const ksh = try fmtKey(allocator, "blocks.{d}.moe.shared", .{i});
            defer allocator.free(ksh);
            const gate_wt = blk2: {
                const raw = try ownWeight(&w, kg);
                defer _ = mlx.mlx_array_free(raw);
                const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
                defer _ = mlx.mlx_array_free(t);
                const tc = try contig(t, s);
                defer _ = mlx.mlx_array_free(tc);
                break :blk2 try astype(tc, .float16, s);
            };
            const e1 = try ExpertLinear.load(&w, allocator, ke1, H, s);
            const inter: u32 = @intCast(mlx.getShape(e1.bias)[1]);
            blk.ffn = .{ .moe = .{
                .gate_wt = gate_wt,
                .experts_fc1 = e1,
                .experts_fc2 = try ExpertLinear.load(&w, allocator, ke2, inter, s),
                .shared = try Mlp.load(&w, allocator, ksh, H, inter, s),
                .top_k = cfg.moe_top_k,
            } };
        } else {
            const km = try fmtKey(allocator, "blocks.{d}.mlp", .{i});
            defer allocator.free(km);
            var m = try Mlp.load(&w, allocator, km, H, H * 4, s);
            // fc2 in_features = the real intermediate dim (read from fc1's bias).
            if (m.fc1.add_bias) |b| {
                const inter: u32 = @intCast(mlx.getShape(b)[0]);
                if (inter != H * 4) {
                    m.fc2.deinit();
                    const k2 = try fmtKey(allocator, "blocks.{d}.mlp.fc2", .{i});
                    defer allocator.free(k2);
                    m.fc2 = try MixedLinear.load(&w, allocator, k2, inter, s);
                }
            }
            blk.ffn = .{ .mlp = m };
        }
        if (i >= pop_from) {
            const kl = try fmtKey(allocator, "blocks.{d}.skip.linear", .{i});
            defer allocator.free(kl);
            const knw = try fmtKey(allocator, "blocks.{d}.skip.norm.weight", .{i});
            defer allocator.free(knw);
            const knb = try fmtKey(allocator, "blocks.{d}.skip.norm.bias", .{i});
            defer allocator.free(knb);
            blk.skip = .{
                .lin = try MixedLinear.load(&w, allocator, kl, H * 2, s),
                .norm_w = try normVec(&w, knw, s),
                .norm_b = try normVec(&w, knb, s),
            };
        }
    }
    // FinalLayer.norm_final is affine LayerNorm in the reference — the
    // converted weights ALWAYS carry it; a missing tensor is a broken convert.
    d.final_norm_w = try normVec(&w, "final.norm.weight", s);
    d.final_norm_b = try normVec(&w, "final.norm.bias", s);
    d.final_lin = try MixedLinear.load(&w, allocator, "final.linear", H, s);
    log.info("[hy3d] DiT ready (depth {d}, {d} MoE blocks)\n", .{ cfg.depth, cfg.num_moe_layers });
    return d;
}

/// Flow-match Euler denoise with CFG (cond-first batch, zeros uncond context).
/// `init_noise` (optional, [1,4096,64] f32) replaces the seeded gaussian for
/// oracle parity. Returns the final latent [1,4096,64] f32 (owned).
pub fn denoise(dit: *Dit, allocator: std.mem.Allocator, cond: mlx.mlx_array, steps: u32, guidance: f32, seed: u64, init_noise: ?mlx.mlx_array, progress: ?sse.Progress) !mlx.mlx_array {
    const s = dit.s;
    const c = dit.cfg;
    const sigmas = try buildSigmas(allocator, steps);
    defer allocator.free(sigmas);

    var x = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x);
    if (init_noise) |n| {
        try mlx.check(mlx.mlx_array_set(&x, n));
    } else {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        const sh = [_]c_int{ 1, @intCast(c.num_latents), @intCast(c.embed_dim) };
        try mlx.check(mlx.mlx_random_normal(&x, &sh, 3, .float32, 0.0, 1.0, key, s));
    }

    // ctx2 = [cond; zeros] — uncond is output-level zeros, not a black image.
    var zeros_ctx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros_ctx);
    {
        const csh = mlx.getShape(cond);
        const zsh = [_]c_int{ csh[0], csh[1], csh[2] };
        try mlx.check(mlx.mlx_zeros(&zeros_ctx, &zsh, 3, .float16, s));
    }
    const cond_h = try astype(cond, .float16, s);
    defer _ = mlx.mlx_array_free(cond_h);
    const ctx2 = try concat(&[_]mlx.mlx_array{ cond_h, zeros_ctx }, 0, s);
    defer _ = mlx.mlx_array_free(ctx2);

    const g = scalarF(guidance);
    defer _ = mlx.mlx_array_free(g);

    for (0..steps) |i| {
        const ds = sigmas[i + 1] - sigmas[i];
        if (ds == 0.0) continue; // the appended trailing σ — a no-op step
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("denoise", @intCast(i + 1), steps);
        }
        const xh = try astype(x, .float16, s);
        defer _ = mlx.mlx_array_free(xh);
        const x2 = try concat(&[_]mlx.mlx_array{ xh, xh }, 0, s);
        defer _ = mlx.mlx_array_free(x2);
        const v = try dit.forward(x2, ctx2, sigmas[i]);
        defer _ = mlx.mlx_array_free(v);
        const v_c0 = try sliceAxis(v, 0, 0, 1, s);
        defer _ = mlx.mlx_array_free(v_c0);
        const v_u0 = try sliceAxis(v, 0, 1, 2, s);
        defer _ = mlx.mlx_array_free(v_u0);
        const v_c = try astype(v_c0, .float32, s);
        defer _ = mlx.mlx_array_free(v_c);
        const v_u = try astype(v_u0, .float32, s);
        defer _ = mlx.mlx_array_free(v_u);
        // v = v_u + g·(v_c − v_u), scheduler step in f32.
        const diff = try subA(v_c, v_u, s);
        defer _ = mlx.mlx_array_free(diff);
        const gd = try mulA(diff, g, s);
        defer _ = mlx.mlx_array_free(gd);
        const guided = try addA(v_u, gd, s);
        defer _ = mlx.mlx_array_free(guided);
        const dsa = scalarF(ds);
        defer _ = mlx.mlx_array_free(dsa);
        const step_v = try mulA(guided, dsa, s);
        defer _ = mlx.mlx_array_free(step_v);
        const nx = try addA(x, step_v, s);
        _ = mlx.mlx_array_free(x);
        x = nx;
        _ = mlx.mlx_array_eval(x);
    }
    return x;
}

// ════════════════════════════════════════════════════════════════════════
// ShapeVAE decoder — post_kl + 16 self-attn blocks + geo cross-attn SDF head.
// VAE qk-norm is per-head AFFINE LayerNorm(64) — NOT the DiT's RMSNorm.
// ════════════════════════════════════════════════════════════════════════

const VaeBlock = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    out: MixedLinear,
    qn_w: mlx.mlx_array,
    qn_b: mlx.mlx_array,
    kn_w: mlx.mlx_array,
    kn_b: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    mlp: Mlp,
    fn deinit(self: *VaeBlock) void {
        _ = mlx.mlx_array_free(self.ln1_w);
        _ = mlx.mlx_array_free(self.ln1_b);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.out.deinit();
        _ = mlx.mlx_array_free(self.qn_w);
        _ = mlx.mlx_array_free(self.qn_b);
        _ = mlx.mlx_array_free(self.kn_w);
        _ = mlx.mlx_array_free(self.kn_b);
        _ = mlx.mlx_array_free(self.ln2_w);
        _ = mlx.mlx_array_free(self.ln2_b);
        self.mlp.deinit();
    }
};

pub const GeoDecoder = struct {
    query_proj: MixedLinear, // 51 → 1024 (fp16 dense)
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    ln3_w: mlx.mlx_array,
    ln3_b: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    out: MixedLinear,
    qn_w: mlx.mlx_array,
    qn_b: mlx.mlx_array,
    kn_w: mlx.mlx_array,
    kn_b: mlx.mlx_array,
    mlp: Mlp,
    ln_post_w: mlx.mlx_array,
    ln_post_b: mlx.mlx_array,
    out_proj: MixedLinear, // 1024 → 1 (fp16 dense)
    fn deinit(self: *GeoDecoder) void {
        self.query_proj.deinit();
        _ = mlx.mlx_array_free(self.ln1_w);
        _ = mlx.mlx_array_free(self.ln1_b);
        _ = mlx.mlx_array_free(self.ln2_w);
        _ = mlx.mlx_array_free(self.ln2_b);
        _ = mlx.mlx_array_free(self.ln3_w);
        _ = mlx.mlx_array_free(self.ln3_b);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.out.deinit();
        _ = mlx.mlx_array_free(self.qn_w);
        _ = mlx.mlx_array_free(self.qn_b);
        _ = mlx.mlx_array_free(self.kn_w);
        _ = mlx.mlx_array_free(self.kn_b);
        self.mlp.deinit();
        _ = mlx.mlx_array_free(self.ln_post_w);
        _ = mlx.mlx_array_free(self.ln_post_b);
        self.out_proj.deinit();
    }
};

pub const VaeDecoder = struct {
    cfg: Hy3dConfig,
    allocator: std.mem.Allocator,
    s: S,
    post_kl: MixedLinear, // 64 → 1024
    blocks: []VaeBlock,
    geo: GeoDecoder,

    pub fn deinit(self: *VaeDecoder) void {
        self.post_kl.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.geo.deinit();
    }
};

/// Latents [1,4096,64] (f32 or f16) → the 1024-wide latent set [1,4096,1024]
/// f16. Applies ÷ scale_factor FIRST (the reference `_export` order).
pub fn vaeDecodeLatentSet(vae: *VaeDecoder, latents: mlx.mlx_array) !mlx.mlx_array {
    const s = vae.s;
    const c = vae.cfg;
    const heads: c_int = @intCast(c.vae_heads);
    const hd: c_int = @intCast(c.vaeHeadDim());
    const inv_sf = scalarF(1.0 / c.scale_factor);
    defer _ = mlx.mlx_array_free(inv_sf);
    const scaled = try mulA(latents, inv_sf, s);
    defer _ = mlx.mlx_array_free(scaled);
    var x = try vae.post_kl.forward(scaled, s); // [1,4096,1024] f16

    for (vae.blocks) |*blk| {
        // Self-attn with per-head affine LayerNorm(64) on q,k.
        {
            const n1 = try layerNorm(x, blk.ln1_w, blk.ln1_b, s);
            defer _ = mlx.mlx_array_free(n1);
            const q0 = try blk.q.forward(n1, s);
            defer _ = mlx.mlx_array_free(q0);
            const k0 = try blk.k.forward(n1, s);
            defer _ = mlx.mlx_array_free(k0);
            const v0 = try blk.v.forward(n1, s);
            defer _ = mlx.mlx_array_free(v0);
            const sh = mlx.getShape(q0);
            const q4 = try reshape(q0, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
            defer _ = mlx.mlx_array_free(q4);
            const qn = try layerNorm(q4, blk.qn_w, blk.qn_b, s);
            defer _ = mlx.mlx_array_free(qn);
            const q = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
            defer _ = mlx.mlx_array_free(q);
            const k4 = try reshape(k0, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
            defer _ = mlx.mlx_array_free(k4);
            const kn = try layerNorm(k4, blk.kn_w, blk.kn_b, s);
            defer _ = mlx.mlx_array_free(kn);
            const k = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try splitHeads(v0, heads, hd, s);
            defer _ = mlx.mlx_array_free(v);
            const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.vaeHeadDim())));
            const attn = try sdpa(q, k, v, scale, s);
            defer _ = mlx.mlx_array_free(attn);
            const merged = try mergeHeads(attn, s);
            defer _ = mlx.mlx_array_free(merged);
            const o = try blk.out.forward(merged, s);
            defer _ = mlx.mlx_array_free(o);
            const nx = try addA(x, o, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        // MLP.
        {
            const n2 = try layerNorm(x, blk.ln2_w, blk.ln2_b, s);
            defer _ = mlx.mlx_array_free(n2);
            const m = try blk.mlp.forward(n2, s);
            defer _ = mlx.mlx_array_free(m);
            const nx = try addA(x, m, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
    }
    _ = mlx.mlx_array_eval(x);
    return x;
}

/// Fourier positional embed for query points [1,P,3] f32: freqs = 2^[0..nf),
/// NO π, output = cat(x, sin(x⊗f), cos(x⊗f)) with the sin/cos blocks flattened
/// COORDINATE-MAJOR ([x·f0..x·f7, y·f0.., z·f0..]) → [1,P,3·(2nf+1)] f32.
pub fn fourierEmbed(queries: mlx.mlx_array, num_freqs: u32, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(queries); // [1,P,3]
    const nf: c_int = @intCast(num_freqs);
    var freqs_buf: [32]f32 = undefined;
    for (0..num_freqs) |i| freqs_buf[i] = std.math.pow(f32, 2.0, @floatFromInt(i));
    const fsh = [_]c_int{nf};
    const freqs = mlx.mlx_array_new_data(&freqs_buf, &fsh, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);

    var xe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xe);
    try mlx.check(mlx.mlx_expand_dims(&xe, queries, -1, s)); // [1,P,3,1]
    const xf = try mulA(xe, freqs, s); // [1,P,3,nf]
    defer _ = mlx.mlx_array_free(xf);
    var sinv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinv);
    try mlx.check(mlx.mlx_sin(&sinv, xf, s));
    var cosv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cosv);
    try mlx.check(mlx.mlx_cos(&cosv, xf, s));
    // Row-major reshape [1,P,3,nf] → [1,P,3·nf] IS the coordinate-major layout.
    const flat_sh = [_]c_int{ sh[0], sh[1], 3 * nf };
    const sflat = try reshape(sinv, &flat_sh, s);
    defer _ = mlx.mlx_array_free(sflat);
    const cflat = try reshape(cosv, &flat_sh, s);
    defer _ = mlx.mlx_array_free(cflat);
    return concat(&[_]mlx.mlx_array{ queries, sflat, cflat }, 2, s);
}

/// Query-independent geo cross-attn K/V, computed ONCE per mesh (the
/// reference's kv_cache seam): k/v projections of ln_2(latent_set), k-normed,
/// head-split [1,H,4096,64].
pub const GeoKv = struct {
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    pub fn deinit(self: *GeoKv) void {
        _ = mlx.mlx_array_free(self.k);
        _ = mlx.mlx_array_free(self.v);
    }
};

pub fn geoDecodePrepare(vae: *VaeDecoder, latent_set: mlx.mlx_array) !GeoKv {
    const s = vae.s;
    const c = vae.cfg;
    const geo = &vae.geo;
    const heads: c_int = @intCast(c.vae_heads);
    const hd: c_int = @intCast(c.vaeHeadDim());
    const n2 = try layerNorm(latent_set, geo.ln2_w, geo.ln2_b, s);
    defer _ = mlx.mlx_array_free(n2);
    const k0 = try geo.k.forward(n2, s);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try geo.v.forward(n2, s);
    defer _ = mlx.mlx_array_free(v0);
    const sh = mlx.getShape(k0);
    const k4 = try reshape(k0, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(k4);
    const kn = try layerNorm(k4, geo.kn_w, geo.kn_b, s);
    defer _ = mlx.mlx_array_free(kn);
    var kv = GeoKv{
        .k = try transpose(kn, &[_]c_int{ 0, 2, 1, 3 }, s),
        .v = undefined,
    };
    errdefer _ = mlx.mlx_array_free(kv.k);
    kv.v = try splitHeads(v0, heads, hd, s);
    _ = mlx.mlx_array_eval(kv.k);
    _ = mlx.mlx_array_eval(kv.v);
    return kv;
}

/// One chunk of SDF queries [1,P,3] f32 → logits [P] f32 (inside = positive).
pub fn geoDecodeChunk(vae: *VaeDecoder, kv: *const GeoKv, queries: mlx.mlx_array) !mlx.mlx_array {
    const s = vae.s;
    const c = vae.cfg;
    const geo = &vae.geo;
    const heads: c_int = @intCast(c.vae_heads);
    const hd: c_int = @intCast(c.vaeHeadDim());

    const emb = try fourierEmbed(queries, c.num_freqs, s); // [1,P,51] f32
    defer _ = mlx.mlx_array_free(emb);
    const q_emb = try geo.query_proj.forward(emb, s); // [1,P,1024] f16
    defer _ = mlx.mlx_array_free(q_emb);

    // Cross-attn: x = q_emb + attn(ln_1(q_emb), kv)
    const n1 = try layerNorm(q_emb, geo.ln1_w, geo.ln1_b, s);
    defer _ = mlx.mlx_array_free(n1);
    const q0 = try geo.q.forward(n1, s);
    defer _ = mlx.mlx_array_free(q0);
    const qsh = mlx.getShape(q0);
    const q4 = try reshape(q0, &[_]c_int{ qsh[0], qsh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(q4);
    const qn = try layerNorm(q4, geo.qn_w, geo.qn_b, s);
    defer _ = mlx.mlx_array_free(qn);
    const q = try transpose(qn, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(c.vaeHeadDim())));
    const attn = try sdpa(q, kv.k, kv.v, scale, s);
    defer _ = mlx.mlx_array_free(attn);
    const merged = try mergeHeads(attn, s);
    defer _ = mlx.mlx_array_free(merged);
    const o = try geo.out.forward(merged, s);
    defer _ = mlx.mlx_array_free(o);
    const h1 = try addA(q_emb, o, s);
    defer _ = mlx.mlx_array_free(h1);

    // x += mlp(ln_3(x))
    const n3 = try layerNorm(h1, geo.ln3_w, geo.ln3_b, s);
    defer _ = mlx.mlx_array_free(n3);
    const m = try geo.mlp.forward(n3, s);
    defer _ = mlx.mlx_array_free(m);
    const h2 = try addA(h1, m, s);
    defer _ = mlx.mlx_array_free(h2);

    const np = try layerNorm(h2, geo.ln_post_w, geo.ln_post_b, s);
    defer _ = mlx.mlx_array_free(np);
    const logits = try geo.out_proj.forward(np, s); // [1,P,1] f16
    defer _ = mlx.mlx_array_free(logits);
    // Cast to f32 BEFORE the caller stores it (fp16 grid = stair-step artifacts).
    const f = try astype(logits, .float32, s);
    _ = mlx.mlx_array_eval(f);
    return f;
}

pub fn loadVae(allocator: std.mem.Allocator, cfg: Hy3dConfig, model_dir: []const u8, s: S) !VaeDecoder {
    var w = try loadFileWeights(allocator, model_dir, "vae.safetensors");
    defer w.deinit();
    var v: VaeDecoder = undefined;
    v.cfg = cfg;
    v.allocator = allocator;
    v.s = s;
    v.post_kl = try MixedLinear.load(&w, allocator, "post_kl", cfg.embed_dim, s);
    v.blocks = try allocator.alloc(VaeBlock, cfg.vae_decoder_layers);
    const W = cfg.vae_width;
    for (v.blocks, 0..) |*blk, i| {
        const n1w = try fmtKey(allocator, "blocks.{d}.ln1.weight", .{i});
        defer allocator.free(n1w);
        const n1b = try fmtKey(allocator, "blocks.{d}.ln1.bias", .{i});
        defer allocator.free(n1b);
        const n2w = try fmtKey(allocator, "blocks.{d}.ln2.weight", .{i});
        defer allocator.free(n2w);
        const n2b = try fmtKey(allocator, "blocks.{d}.ln2.bias", .{i});
        defer allocator.free(n2b);
        const kq = try fmtKey(allocator, "blocks.{d}.attn.q", .{i});
        defer allocator.free(kq);
        const kk = try fmtKey(allocator, "blocks.{d}.attn.k", .{i});
        defer allocator.free(kk);
        const kv = try fmtKey(allocator, "blocks.{d}.attn.v", .{i});
        defer allocator.free(kv);
        const ko = try fmtKey(allocator, "blocks.{d}.attn.out", .{i});
        defer allocator.free(ko);
        const kqnw = try fmtKey(allocator, "blocks.{d}.attn.q_norm.weight", .{i});
        defer allocator.free(kqnw);
        const kqnb = try fmtKey(allocator, "blocks.{d}.attn.q_norm.bias", .{i});
        defer allocator.free(kqnb);
        const kknw = try fmtKey(allocator, "blocks.{d}.attn.k_norm.weight", .{i});
        defer allocator.free(kknw);
        const kknb = try fmtKey(allocator, "blocks.{d}.attn.k_norm.bias", .{i});
        defer allocator.free(kknb);
        const km = try fmtKey(allocator, "blocks.{d}.mlp", .{i});
        defer allocator.free(km);
        blk.* = .{
            .ln1_w = try normVec(&w, n1w, s),
            .ln1_b = try normVec(&w, n1b, s),
            .q = try MixedLinear.load(&w, allocator, kq, W, s),
            .k = try MixedLinear.load(&w, allocator, kk, W, s),
            .v = try MixedLinear.load(&w, allocator, kv, W, s),
            .out = try MixedLinear.load(&w, allocator, ko, W, s),
            .qn_w = try normVec(&w, kqnw, s),
            .qn_b = try normVec(&w, kqnb, s),
            .kn_w = try normVec(&w, kknw, s),
            .kn_b = try normVec(&w, kknb, s),
            .ln2_w = try normVec(&w, n2w, s),
            .ln2_b = try normVec(&w, n2b, s),
            .mlp = try Mlp.load(&w, allocator, km, W, W * 4, s),
        };
    }
    v.geo = .{
        .query_proj = try MixedLinear.load(&w, allocator, "geo.query_proj", cfg.fourierDim(), s),
        .ln1_w = try normVec(&w, "geo.ln1.weight", s),
        .ln1_b = try normVec(&w, "geo.ln1.bias", s),
        .ln2_w = try normVec(&w, "geo.ln2.weight", s),
        .ln2_b = try normVec(&w, "geo.ln2.bias", s),
        .ln3_w = try normVec(&w, "geo.ln3.weight", s),
        .ln3_b = try normVec(&w, "geo.ln3.bias", s),
        .q = try MixedLinear.load(&w, allocator, "geo.attn.q", W, s),
        .k = try MixedLinear.load(&w, allocator, "geo.attn.k", W, s),
        .v = try MixedLinear.load(&w, allocator, "geo.attn.v", W, s),
        .out = try MixedLinear.load(&w, allocator, "geo.attn.out", W, s),
        .qn_w = try normVec(&w, "geo.attn.q_norm.weight", s),
        .qn_b = try normVec(&w, "geo.attn.q_norm.bias", s),
        .kn_w = try normVec(&w, "geo.attn.k_norm.weight", s),
        .kn_b = try normVec(&w, "geo.attn.k_norm.bias", s),
        .mlp = try Mlp.load(&w, allocator, "geo.mlp", W, W * 4, s),
        .ln_post_w = try normVec(&w, "geo.ln_post.weight", s),
        .ln_post_b = try normVec(&w, "geo.ln_post.bias", s),
        .out_proj = try MixedLinear.load(&w, allocator, "geo.out_proj", W, s),
    };
    log.info("[hy3d] ShapeVAE decoder ready ({d} blocks + geo)\n", .{cfg.vae_decoder_layers});
    return v;
}

pub const VOLUME_BOUND: f32 = 1.01;

/// Sample the SDF on an (R+1)³ inclusive grid over [−bound, bound]³, x-major
/// (`idx = (ix·N + iy)·N + iz`), in GPU chunks with CPU-generated coords.
/// Returns the f32 grid (caller frees — and should free it BEFORE GLB
/// assembly; 385³ is 228 MB).
pub fn decodeVolume(vae: *VaeDecoder, allocator: std.mem.Allocator, latent_set: mlx.mlx_array, res: u32, bound: f32, chunk_size: u32, progress: ?sse.Progress) ![]f32 {
    const n: usize = res + 1;
    const total: usize = n * n * n;
    const grid = try allocator.alloc(f32, total);
    errdefer allocator.free(grid);

    var kv = try geoDecodePrepare(vae, latent_set);
    defer kv.deinit();

    const chunk: usize = @max(1024, chunk_size);
    const coords = try allocator.alloc(f32, chunk * 3);
    defer allocator.free(coords);
    const step: f32 = (2.0 * bound) / @as(f32, @floatFromInt(res)); // inclusive linspace spacing
    const n_chunks: u32 = @intCast((total + chunk - 1) / chunk);

    var start: usize = 0;
    var ci: u32 = 0;
    while (start < total) : (ci += 1) {
        const count = @min(chunk, total - start);
        for (0..count) |j| {
            const idx = start + j;
            const ix = idx / (n * n);
            const rem = idx % (n * n);
            const iy = rem / n;
            const iz = rem % n;
            coords[j * 3 + 0] = -bound + step * @as(f32, @floatFromInt(ix));
            coords[j * 3 + 1] = -bound + step * @as(f32, @floatFromInt(iy));
            coords[j * 3 + 2] = -bound + step * @as(f32, @floatFromInt(iz));
        }
        const qsh = [_]c_int{ 1, @intCast(count), 3 };
        const q = mlx.mlx_array_new_data(coords.ptr, &qsh, 3, .float32);
        defer _ = mlx.mlx_array_free(q);
        const logits = try geoDecodeChunk(vae, &kv, q); // [1,count,1] f32, evaluated
        defer _ = mlx.mlx_array_free(logits);
        const data = mlx.mlx_array_data_float32(logits) orelse return error.NoData;
        @memcpy(grid[start .. start + count], data[0..count]);
        start += count;
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("volume", ci + 1, n_chunks);
        }
    }
    return grid;
}

/// Coarse stride of the FlashVDM-class hierarchical decode: the coarse pass
/// samples every 4th fine point per axis (256 → 65³ ≈ 1.6% of the dense grid).
pub const HIER_FACTOR: u32 = 4;

/// FlashVDM-class coarse-to-fine grid fill. Evaluates a stride-`factor`
/// coarse grid, flags coarse cells whose corners straddle `level` (dilated by
/// one cell so marching cubes + its central-difference normals only ever read
/// EXACT values near the surface), trilinearly interpolates the far field
/// (same-sign corners → interpolation cannot cross the level, so MC emits no
/// spurious geometry there), then evaluates only the flagged cells' fine
/// points. `evaluator.evalChunk(coords, out)` scores packed xyz triples.
/// Returns the number of points evaluated exactly.
pub fn fillGridHierarchical(
    allocator: std.mem.Allocator,
    grid: []f32,
    res: u32,
    bound: f32,
    level: f32,
    factor: u32,
    chunk_size: u32,
    evaluator: anytype,
    progress: ?sse.Progress,
) !usize {
    std.debug.assert(factor >= 2 and res % factor == 0);
    const n: usize = res + 1;
    std.debug.assert(grid.len == n * n * n);
    const rc: usize = res / factor; // coarse cells per axis
    const nc: usize = rc + 1; // coarse points per axis
    const step: f32 = (2.0 * bound) / @as(f32, @floatFromInt(res));
    const chunk: usize = @max(1024, chunk_size);
    const coords = try allocator.alloc(f32, chunk * 3);
    defer allocator.free(coords);

    // 1. Coarse pass (points coincide with fine points at stride `factor`).
    const coarse = try allocator.alloc(f32, nc * nc * nc);
    defer allocator.free(coarse);
    const ctotal = nc * nc * nc;
    const cchunks: u32 = @intCast((ctotal + chunk - 1) / chunk);
    var evaluated: usize = 0;
    {
        var start: usize = 0;
        var ci: u32 = 0;
        while (start < ctotal) : (ci += 1) {
            const count = @min(chunk, ctotal - start);
            for (0..count) |j| {
                const idx = start + j;
                const ix = idx / (nc * nc);
                const rem = idx % (nc * nc);
                const iy = rem / nc;
                const iz = rem % nc;
                coords[j * 3 + 0] = -bound + step * @as(f32, @floatFromInt(ix * factor));
                coords[j * 3 + 1] = -bound + step * @as(f32, @floatFromInt(iy * factor));
                coords[j * 3 + 2] = -bound + step * @as(f32, @floatFromInt(iz * factor));
            }
            try evaluator.evalChunk(coords[0 .. count * 3], coarse[start .. start + count]);
            evaluated += count;
            start += count;
            if (progress) |p| {
                if (p.cancelled()) return error.Cancelled;
                p.emit("volume", ci + 1, cchunks);
            }
        }
    }

    // 2. Flag sign-change coarse cells, then dilate by one cell in all
    //    directions (26-neighborhood) so the exact region includes the ring
    //    MC's central-difference normals read from.
    const cells = rc * rc * rc;
    const flags = try allocator.alloc(bool, cells);
    defer allocator.free(flags);
    @memset(flags, false);
    for (0..rc) |cx| for (0..rc) |cy| for (0..rc) |cz| {
        var pos = false;
        var neg = false;
        for (0..2) |dx| for (0..2) |dy| for (0..2) |dz| {
            const v = coarse[((cx + dx) * nc + (cy + dy)) * nc + (cz + dz)];
            if (v > level) pos = true else neg = true;
        };
        if (pos and neg) flags[(cx * rc + cy) * rc + cz] = true;
    };
    const dilated = try allocator.alloc(bool, cells);
    defer allocator.free(dilated);
    @memcpy(dilated, flags);
    for (0..rc) |cx| for (0..rc) |cy| for (0..rc) |cz| {
        if (!flags[(cx * rc + cy) * rc + cz]) continue;
        const x0 = if (cx == 0) cx else cx - 1;
        const x1 = @min(cx + 1, rc - 1);
        const y0 = if (cy == 0) cy else cy - 1;
        const y1 = @min(cy + 1, rc - 1);
        const z0 = if (cz == 0) cz else cz - 1;
        const z1 = @min(cz + 1, rc - 1);
        var x = x0;
        while (x <= x1) : (x += 1) {
            var y = y0;
            while (y <= y1) : (y += 1) {
                var z = z0;
                while (z <= z1) : (z += 1) dilated[(x * rc + y) * rc + z] = true;
            }
        }
    };

    // 3. Fill the whole fine grid by trilinear interpolation of the coarse
    //    grid (exact at coarse nodes; flagged regions overwritten in step 4).
    const inv_f: f32 = 1.0 / @as(f32, @floatFromInt(factor));
    for (0..n) |ix| {
        const cx = @min(ix / factor, rc - 1);
        const fx = (@as(f32, @floatFromInt(ix)) - @as(f32, @floatFromInt(cx * factor))) * inv_f;
        for (0..n) |iy| {
            const cy = @min(iy / factor, rc - 1);
            const fy = (@as(f32, @floatFromInt(iy)) - @as(f32, @floatFromInt(cy * factor))) * inv_f;
            for (0..n) |iz| {
                const cz = @min(iz / factor, rc - 1);
                const fz = (@as(f32, @floatFromInt(iz)) - @as(f32, @floatFromInt(cz * factor))) * inv_f;
                const c000 = coarse[((cx + 0) * nc + (cy + 0)) * nc + (cz + 0)];
                const c001 = coarse[((cx + 0) * nc + (cy + 0)) * nc + (cz + 1)];
                const c010 = coarse[((cx + 0) * nc + (cy + 1)) * nc + (cz + 0)];
                const c011 = coarse[((cx + 0) * nc + (cy + 1)) * nc + (cz + 1)];
                const c100 = coarse[((cx + 1) * nc + (cy + 0)) * nc + (cz + 0)];
                const c101 = coarse[((cx + 1) * nc + (cy + 0)) * nc + (cz + 1)];
                const c110 = coarse[((cx + 1) * nc + (cy + 1)) * nc + (cz + 0)];
                const c111 = coarse[((cx + 1) * nc + (cy + 1)) * nc + (cz + 1)];
                const c00 = c000 + (c001 - c000) * fz;
                const c01 = c010 + (c011 - c010) * fz;
                const c10 = c100 + (c101 - c100) * fz;
                const c11 = c110 + (c111 - c110) * fz;
                const c0 = c00 + (c01 - c00) * fy;
                const c1 = c10 + (c11 - c10) * fy;
                grid[(ix * n + iy) * n + iz] = c0 + (c1 - c0) * fx;
            }
        }
    }
    // Coarse nodes: write the exact evaluated value (interp is exact there in
    // real arithmetic, but keep it bit-exact against float rounding).
    for (0..nc) |cx| for (0..nc) |cy| for (0..nc) |cz| {
        grid[((cx * factor) * n + cy * factor) * n + cz * factor] = coarse[(cx * nc + cy) * nc + cz];
    };

    // 4. Exact refine of flagged cells' fine points (deduped across shared
    //    faces; coarse nodes already exact).
    var refine: std.ArrayList(u32) = .empty;
    defer refine.deinit(allocator);
    const visited = try allocator.alloc(bool, n * n * n);
    defer allocator.free(visited);
    @memset(visited, false);
    for (0..rc) |cx| for (0..rc) |cy| for (0..rc) |cz| {
        if (!dilated[(cx * rc + cy) * rc + cz]) continue;
        var ix = cx * factor;
        while (ix <= (cx + 1) * factor) : (ix += 1) {
            var iy = cy * factor;
            while (iy <= (cy + 1) * factor) : (iy += 1) {
                var iz = cz * factor;
                while (iz <= (cz + 1) * factor) : (iz += 1) {
                    if (ix % factor == 0 and iy % factor == 0 and iz % factor == 0) continue;
                    const idx = (ix * n + iy) * n + iz;
                    if (visited[idx]) continue;
                    visited[idx] = true;
                    try refine.append(allocator, @intCast(idx));
                }
            }
        }
    };

    const rtotal = refine.items.len;
    const rchunks: u32 = @intCast((rtotal + chunk - 1) / chunk);
    const vals = try allocator.alloc(f32, chunk);
    defer allocator.free(vals);
    var start: usize = 0;
    var ri: u32 = 0;
    while (start < rtotal) : (ri += 1) {
        const count = @min(chunk, rtotal - start);
        for (0..count) |j| {
            const idx: usize = refine.items[start + j];
            const ix = idx / (n * n);
            const rem = idx % (n * n);
            const iy = rem / n;
            const iz = rem % n;
            coords[j * 3 + 0] = -bound + step * @as(f32, @floatFromInt(ix));
            coords[j * 3 + 1] = -bound + step * @as(f32, @floatFromInt(iy));
            coords[j * 3 + 2] = -bound + step * @as(f32, @floatFromInt(iz));
        }
        try evaluator.evalChunk(coords[0 .. count * 3], vals[0..count]);
        for (0..count) |j| grid[refine.items[start + j]] = vals[j];
        evaluated += count;
        start += count;
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("volume", cchunks + ri + 1, cchunks + rchunks);
        }
    }
    return evaluated;
}

/// Hierarchical (FlashVDM-class) volume decode: same contract as
/// `decodeVolume`, 4-20× fewer geo-decoder queries. Falls back to the dense
/// sweep when `res` isn't a multiple of the coarse stride. `level` must match
/// the marching-cubes level the caller extracts at.
pub fn decodeVolumeHierarchical(vae: *VaeDecoder, allocator: std.mem.Allocator, latent_set: mlx.mlx_array, res: u32, bound: f32, level: f32, chunk_size: u32, progress: ?sse.Progress) ![]f32 {
    if (res % HIER_FACTOR != 0 or res < HIER_FACTOR * 8)
        return decodeVolume(vae, allocator, latent_set, res, bound, chunk_size, progress);

    const n: usize = res + 1;
    const grid = try allocator.alloc(f32, n * n * n);
    errdefer allocator.free(grid);

    var kv = try geoDecodePrepare(vae, latent_set);
    defer kv.deinit();

    const GpuEval = struct {
        vae: *VaeDecoder,
        kv: *const GeoKv,
        pub fn evalChunk(self: *@This(), coords: []const f32, out: []f32) !void {
            const qsh = [_]c_int{ 1, @intCast(out.len), 3 };
            const q = mlx.mlx_array_new_data(coords.ptr, &qsh, 3, .float32);
            defer _ = mlx.mlx_array_free(q);
            const logits = try geoDecodeChunk(self.vae, self.kv, q);
            defer _ = mlx.mlx_array_free(logits);
            const data = mlx.mlx_array_data_float32(logits) orelse return error.NoData;
            @memcpy(out, data[0..out.len]);
        }
    };
    var ev = GpuEval{ .vae = vae, .kv = &kv };
    const evaluated = try fillGridHierarchical(allocator, grid, res, bound, level, HIER_FACTOR, chunk_size, &ev, progress);
    log.info("[hy3d] hierarchical volume decode: {d}/{d} points evaluated ({d:.1}x fewer)\n", .{ evaluated, n * n * n, @as(f64, @floatFromInt(n * n * n)) / @as(f64, @floatFromInt(@max(1, evaluated))) });
    return grid;
}

// ════════════════════════════════════════════════════════════════════════
// Engine — owns the three sub-models; composes the full pipeline.
// ════════════════════════════════════════════════════════════════════════

pub const MeshOpts = struct {
    steps: u32 = 50,
    guidance: f32 = 5.0,
    seed: u64 = 0,
    octree_resolution: u32 = 384,
    chunk: u32 = 32768,
    mc_level: f32 = 0.0,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: Hy3dConfig,
    dino: DinoEncoder,
    dit: Dit,
    vae: VaeDecoder,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = mlx.mlx_default_gpu_stream_new();
        self.cfg = try readConfigFile(io, allocator, model_dir);
        self.dino = try loadDino(allocator, self.cfg, model_dir, self.s);
        errdefer self.dino.deinit();
        self.dit = try loadDit(allocator, self.cfg, model_dir, self.s);
        errdefer self.dit.deinit();
        self.vae = try loadVae(allocator, self.cfg, model_dir, self.s);
        log.info("[hy3d] Hunyuan3D-2.1 shape engine ready\n", .{});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.dino.deinit();
        self.dit.deinit();
        self.vae.deinit();
        self.allocator.destroy(self);
    }

    /// image_rgba: straight-alpha RGBA8 (caller decodes PNG/JPEG); returns the
    /// extracted mesh in world space. Texture-phase seam: the paint stage will
    /// consume this Mesh rather than the GLB bytes.
    pub fn generateMeshRaw(self: *Engine, alloc: std.mem.Allocator, image_rgba: []const u8, w: u32, h: u32, opts: MeshOpts, progress: ?sse.Progress) !mc.Mesh {
        const c = self.cfg;
        if (progress) |p| p.emit("encode", 0, 1);

        // Preprocess + DINO conditioning.
        const pixels = try preprocessImage(alloc, image_rgba, w, h, c.dino_image_size);
        defer alloc.free(pixels);
        const side: c_int = @intCast(c.dino_image_size);
        const psh = [_]c_int{ 1, 3, side, side };
        const parr = mlx.mlx_array_new_data(pixels.ptr, &psh, 4, .float32);
        defer _ = mlx.mlx_array_free(parr);
        const cond = try self.dino.encode(parr);
        defer _ = mlx.mlx_array_free(cond);
        _ = mlx.mlx_array_eval(cond);
        if (progress) |p| {
            if (p.cancelled()) return error.Cancelled;
            p.emit("encode", 1, 1);
        }

        // Denoise → latent → 1024-wide latent set.
        const latent = try denoise(&self.dit, alloc, cond, opts.steps, opts.guidance, opts.seed, null, progress);
        defer _ = mlx.mlx_array_free(latent);
        const latent_set = try vaeDecodeLatentSet(&self.vae, latent);
        defer _ = mlx.mlx_array_free(latent_set);

        // Volume decode → marching cubes. The MC vertex world transform copies
        // the reference's /(R+1) off-by-one (NOT the /R sampling spacing).
        const res = opts.octree_resolution;
        const grid = try decodeVolumeHierarchical(&self.vae, alloc, latent_set, res, VOLUME_BOUND, opts.mc_level, opts.chunk, progress);
        defer alloc.free(grid);
        if (progress) |p| p.emit("mesh", 0, 1);
        const n: usize = res + 1;
        const mc_scale = 2.0 * VOLUME_BOUND / @as(f32, @floatFromInt(res + 1));
        const mesh = try mc.extract(alloc, grid, .{ n, n, n }, opts.mc_level, .{ mc_scale, mc_scale, mc_scale }, .{ -VOLUME_BOUND, -VOLUME_BOUND, -VOLUME_BOUND });
        if (progress) |p| p.emit("mesh", 1, 1);
        return mesh;
    }

    /// Full pipeline → GLB bytes (caller frees).
    pub fn generateGlb(self: *Engine, alloc: std.mem.Allocator, image_rgba: []const u8, w: u32, h: u32, opts: MeshOpts, progress: ?sse.Progress) ![]u8 {
        var mesh = try self.generateMeshRaw(alloc, image_rgba, w, h, opts, progress);
        defer mesh.deinit(alloc);
        return glb.writeGlb(alloc, &mesh);
    }
};

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first (no weights), then env-gated cos oracles fed by
// tests/dump_hunyuan3d_fixtures.py (HY3D_*, mirrors KREA_*).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "hy3d config parse reads the synthesized config.json" {
    const a = testing.allocator;
    const cfg = try parseConfigText(a,
        \\{"model_type":"hunyuan3d_2_1","quant":"8bit","hidden_size":2048,"depth":21,
        \\ "num_heads":16,"context_dim":1024,"num_latents":4096,"embed_dim":64,
        \\ "vae_width":1024,"vae_heads":16,"vae_decoder_layers":16,"num_freqs":8,
        \\ "scale_factor":1.0039506158752403,"num_moe_layers":6,"num_experts":8,
        \\ "moe_top_k":2,"dino_hidden":1024,"dino_layers":24,"dino_heads":16,
        \\ "dino_patch":14,"dino_image_size":518}
    );
    // Struct field names double as config keys; the contract spellings
    // hidden_size/num_heads are mapped explicitly in parseConfigText.
    try testing.expectEqual(@as(u32, 2048), cfg.hidden);
    try testing.expectEqual(@as(u32, 16), cfg.heads);
    try testing.expectEqual(@as(u32, 21), cfg.depth);
    try testing.expectEqual(@as(u32, 4096), cfg.num_latents);
    try testing.expectEqual(@as(u32, 8), cfg.num_freqs);
    try testing.expectEqual(@as(u32, 2), cfg.moe_top_k);
    try testing.expectEqual(@as(u32, 1370), cfg.dinoTokens());
    try testing.expectEqual(@as(u32, 51), cfg.fourierDim());
    try testing.expectEqual(@as(u32, 128), cfg.headDim());
    try testing.expectEqual(@as(u32, 64), cfg.vaeHeadDim());
    try testing.expectApproxEqAbs(@as(f32, 1.0039506), cfg.scale_factor, 1e-6);
    // Wrong model_type must be rejected.
    try testing.expectError(error.BadHy3dConfig, parseConfigText(a,
        \\{"model_type":"flux2"}
    ));
}

test "hy3d sigmas schedule is ascending linspace with an appended 1.0" {
    const a = testing.allocator;
    const s5 = try buildSigmas(a, 5);
    defer a.free(s5);
    const expect = [_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0, 1.0 };
    try testing.expectEqual(expect.len, s5.len);
    for (expect, 0..) |e, i| try testing.expectApproxEqAbs(e, s5[i], 1e-7);
    // Final pair is the Δσ=0 no-op the loop skips.
    try testing.expectEqual(s5[s5.len - 1], s5[s5.len - 2]);
    try testing.expectError(error.BadSteps, buildSigmas(a, 1));
}

test "hy3d fourier embed is coordinate-major (x, sin, cos), no pi" {
    const s = mlx.mlx_default_cpu_stream_new();
    const pts = [_]f32{ 0.5, -1.0, 0.25 };
    const psh = [_]c_int{ 1, 1, 3 };
    const q = mlx.mlx_array_new_data(&pts, &psh, 3, .float32);
    defer _ = mlx.mlx_array_free(q);
    const emb = try fourierEmbed(q, 8, s);
    defer _ = mlx.mlx_array_free(emb);
    _ = mlx.mlx_array_eval(emb);
    const sh = mlx.getShape(emb);
    try testing.expectEqual(@as(c_int, 51), sh[2]);
    const d = mlx.mlx_array_data_float32(emb) orelse return error.NoData;
    // Layout: [x,y,z | sin(x·1),sin(x·2)..sin(x·128), sin(y·1).., sin(z·1).. | cos(...)]
    try testing.expectApproxEqAbs(@as(f32, 0.5), d[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), d[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.25), d[2], 1e-6);
    // sin block, coordinate-major: d[3+i] = sin(x·2^i); d[3+8+i] = sin(y·2^i).
    for (0..8) |i| {
        const f = std.math.pow(f32, 2.0, @floatFromInt(i));
        try testing.expectApproxEqAbs(@sin(0.5 * f), d[3 + i], 2e-5);
        try testing.expectApproxEqAbs(@sin(-1.0 * f), d[3 + 8 + i], 2e-5);
        try testing.expectApproxEqAbs(@sin(0.25 * f), d[3 + 16 + i], 2e-5);
        try testing.expectApproxEqAbs(@cos(0.5 * f), d[27 + i], 2e-5);
        try testing.expectApproxEqAbs(@cos(0.25 * f), d[27 + 16 + i], 2e-5);
    }
    // NO π anywhere: sin(x·1) at freq0, not sin(π·x).
    try testing.expect(@abs(d[3] - @sin(@as(f32, 0.5))) < 1e-5);
}

test "hy3d MoE gate is softmax-before-top-k with NO renormalization" {
    const s = mlx.mlx_default_cpu_stream_new();
    // One token, 4 experts, k=2. logits chosen so softmax is easy to verify.
    const logits_v = [_]f32{ 1.0, 3.0, 2.0, 0.0 };
    const lsh = [_]c_int{ 1, 1, 4 };
    const logits = mlx.mlx_array_new_data(&logits_v, &lsh, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const r = try moeRoute(logits, 2, s);
    defer _ = mlx.mlx_array_free(r.inds);
    defer _ = mlx.mlx_array_free(r.weights);
    _ = mlx.mlx_array_eval(r.weights);
    // softmax([1,3,2,0]) = e^x / Σ; Σ = e+e³+e²+1
    const z: f64 = std.math.exp(1.0) + std.math.exp(3.0) + std.math.exp(2.0) + 1.0;
    const p1: f32 = @floatCast(std.math.exp(3.0) / z);
    const p2: f32 = @floatCast(std.math.exp(2.0) / z);
    const wd = mlx.mlx_array_data_float32(r.weights) orelse return error.NoData;
    // Top-2 experts are 1 and 2; weights are the RAW softmax probs (sum < 1 —
    // renormalizing them is the classic wrong-router bug this test pins).
    var got = [2]f32{ wd[0], wd[1] };
    std.mem.sort(f32, &got, {}, std.sort.desc(f32));
    try testing.expectApproxEqAbs(p1, got[0], 1e-5);
    try testing.expectApproxEqAbs(p2, got[1], 1e-5);
    try testing.expect(got[0] + got[1] < 0.99); // NOT renormalized to 1
    var inds_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds_f);
    try mlx.check(mlx.mlx_astype(&inds_f, r.inds, .float32, s));
    _ = mlx.mlx_array_eval(inds_f);
    const idm = mlx.mlx_array_data_float32(inds_f) orelse return error.NoData;
    var ids = [2]f32{ idm[0], idm[1] };
    std.mem.sort(f32, &ids, {}, std.sort.asc(f32));
    try testing.expectApproxEqAbs(@as(f32, 1), ids[0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 2), ids[1], 0.01);
}

test "hy3d timestep embedding is sin-first f32" {
    const a = testing.allocator;
    const emb = try timestepEmbed(a, 0.5, 2048);
    defer _ = mlx.mlx_array_free(emb);
    _ = mlx.mlx_array_eval(emb);
    try testing.expectEqual(@as(usize, 2048), @as(usize, @intCast(mlx.mlx_array_size(emb))));
    const d = mlx.mlx_array_data_float32(emb) orelse return error.NoData;
    // i=0: f=1 → sin(0.5) first, cos(0.5) at index half.
    try testing.expectApproxEqAbs(@sin(@as(f32, 0.5)), d[0], 1e-6);
    try testing.expectApproxEqAbs(@cos(@as(f32, 0.5)), d[1024], 1e-6);
    // i=1024/2: f = exp(−ln(1e4)·0.5) = 0.01
    const f_mid: f32 = @floatCast(std.math.exp(-std.math.log(f64, std.math.e, 10000.0) * 0.5));
    try testing.expectApproxEqAbs(@sin(0.5 * f_mid), d[512], 1e-6);
}

test "hy3d MixedLinear infers 8-bit group-64 geometry on the f16 engine" {
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    const in: c_int = 128;
    const out: c_int = 64;
    const wv = try a.alloc(f32, @intCast(in * out));
    defer a.free(wv);
    var prng = std.Random.DefaultPrng.init(7);
    for (wv) |*x| x.* = prng.random().float(f32) - 0.5;
    const wsh = [_]c_int{ out, in };
    const wf = mlx.mlx_array_new_data(wv.ptr, &wsh, 2, .float32);
    defer _ = mlx.mlx_array_free(wf);
    const wh = try astype(wf, .float16, s);
    defer _ = mlx.mlx_array_free(wh);

    var packed_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(packed_vec);
    const null_gscale = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_quantize(&packed_vec, wh, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", null_gscale, s));
    var qw = mlx.mlx_array_new();
    var qs = mlx.mlx_array_new();
    var qb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&qw, packed_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qs, packed_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&qb, packed_vec, 2));

    var ww = model_mod.Weights.init(a);
    defer ww.deinit();
    try ww.map.put(try a.dupe(u8, "l.weight"), qw);
    try ww.map.put(try a.dupe(u8, "l.scales"), qs);
    try ww.map.put(try a.dupe(u8, "l.biases"), qb);
    var ml = try MixedLinear.load(&ww, a, "l", @intCast(in), s);
    defer ml.deinit();
    try testing.expect(ml.quantized);
    try testing.expectEqual(@as(u32, 8), ml.bits);
    try testing.expectEqual(@as(u32, 64), ml.group_size);

    const xv = try a.alloc(f32, @intCast(in));
    defer a.free(xv);
    for (xv, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
    const xsh = [_]c_int{ 1, in };
    const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const o = try ml.forward(xa, s);
    defer _ = mlx.mlx_array_free(o);
    const of = try astype(o, .float32, s);
    defer _ = mlx.mlx_array_free(of);
    _ = mlx.mlx_array_eval(of);
    try testing.expectEqual(@as(usize, @intCast(out)), @as(usize, @intCast(mlx.mlx_array_size(of))));
    // 8-bit quant of a [-0.5,0.5] matrix reproduces the dense product closely.
    const od = mlx.mlx_array_data_float32(of) orelse return error.NoData;
    var manual: f32 = 0;
    for (0..@intCast(in)) |i| manual += xv[i] * wv[i]; // row 0
    try testing.expectApproxEqAbs(manual, od[0], 0.05);
}

// ── Oracle tests (env-gated; fixtures from tests/dump_hunyuan3d_fixtures.py) ──

fn readF32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

test "hy3d hierarchical volume decode matches dense mesh with far fewer evals" {
    const a = testing.allocator;
    const res: u32 = 96;
    const bound: f32 = 1.01;
    const level: f32 = 0.0;
    const n: usize = res + 1;
    const total: usize = n * n * n;
    const step: f32 = (2.0 * bound) / @as(f32, @floatFromInt(res));

    // Analytic off-center sphere SDF, inside = positive (engine convention).
    const Sphere = struct {
        count: usize = 0,
        fn sdf(x: f32, y: f32, z: f32) f32 {
            const dx = x - 0.11;
            const dy = y + 0.07;
            const dz = z - 0.035;
            return 0.55 - @sqrt(dx * dx + dy * dy + dz * dz);
        }
        pub fn evalChunk(self: *@This(), coords: []const f32, out: []f32) !void {
            for (0..out.len) |i| out[i] = sdf(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
            self.count += out.len;
        }
    };

    // Dense reference grid (same x-major layout as decodeVolume).
    const dense = try a.alloc(f32, total);
    defer a.free(dense);
    for (0..n) |ix| for (0..n) |iy| for (0..n) |iz| {
        const x = -bound + step * @as(f32, @floatFromInt(ix));
        const y = -bound + step * @as(f32, @floatFromInt(iy));
        const z = -bound + step * @as(f32, @floatFromInt(iz));
        dense[(ix * n + iy) * n + iz] = Sphere.sdf(x, y, z);
    };

    // Hierarchical fill through the generic seam.
    const grid = try a.alloc(f32, total);
    defer a.free(grid);
    var ev = Sphere{};
    const evaluated = try fillGridHierarchical(a, grid, res, bound, level, HIER_FACTOR, 32768, &ev, null);
    try testing.expectEqual(evaluated, ev.count);
    try testing.expect(evaluated < total / 4); // the whole point: far fewer queries

    // Extracted meshes must be IDENTICAL (exact values in every surface cell
    // + its dilated ring; interpolated far-field never crosses the level).
    const k = 2.0 * bound / @as(f32, @floatFromInt(res + 1));
    var m_dense = try mc.extract(a, dense, .{ n, n, n }, level, .{ k, k, k }, .{ -bound, -bound, -bound });
    defer m_dense.deinit(a);
    var m_hier = try mc.extract(a, grid, .{ n, n, n }, level, .{ k, k, k }, .{ -bound, -bound, -bound });
    defer m_hier.deinit(a);
    try testing.expect(m_dense.vertices.len > 0);
    try testing.expectEqualSlices(f32, m_dense.vertices, m_hier.vertices);
    try testing.expectEqualSlices(u32, m_dense.indices, m_hier.indices);
    try testing.expectEqualSlices(f32, m_dense.normals, m_hier.normals);
}

test "hy3d hierarchical fill flags a surface that crosses a coarse-cell face without corner sign change" {
    // A thin off-axis slab whose boundary bulges through coarse faces: the
    // dilation ring must still hand marching cubes exact values everywhere a
    // triangle is produced. Guard asserts hier mesh == dense mesh again on a
    // geometry with high-curvature features relative to the coarse grid.
    const a = testing.allocator;
    const res: u32 = 64;
    const bound: f32 = 1.0;
    const n: usize = res + 1;
    const total: usize = n * n * n;
    const step: f32 = (2.0 * bound) / @as(f32, @floatFromInt(res));

    const Blob = struct {
        count: usize = 0,
        fn sdf(x: f32, y: f32, z: f32) f32 {
            // Two overlapping spheres → high-curvature neck.
            const d1 = 0.42 - @sqrt((x - 0.28) * (x - 0.28) + y * y + z * z);
            const d2 = 0.42 - @sqrt((x + 0.28) * (x + 0.28) + y * y + z * z);
            return @max(d1, d2);
        }
        pub fn evalChunk(self: *@This(), coords: []const f32, out: []f32) !void {
            for (0..out.len) |i| out[i] = sdf(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
            self.count += out.len;
        }
    };

    const dense = try a.alloc(f32, total);
    defer a.free(dense);
    for (0..n) |ix| for (0..n) |iy| for (0..n) |iz| {
        const x = -bound + step * @as(f32, @floatFromInt(ix));
        const y = -bound + step * @as(f32, @floatFromInt(iy));
        const z = -bound + step * @as(f32, @floatFromInt(iz));
        dense[(ix * n + iy) * n + iz] = Blob.sdf(x, y, z);
    };

    const grid = try a.alloc(f32, total);
    defer a.free(grid);
    var ev = Blob{};
    _ = try fillGridHierarchical(a, grid, res, bound, 0.0, HIER_FACTOR, 8192, &ev, null);

    const k = 2.0 * bound / @as(f32, @floatFromInt(res + 1));
    var m_dense = try mc.extract(a, dense, .{ n, n, n }, 0.0, .{ k, k, k }, .{ -bound, -bound, -bound });
    defer m_dense.deinit(a);
    var m_hier = try mc.extract(a, grid, .{ n, n, n }, 0.0, .{ k, k, k }, .{ -bound, -bound, -bound });
    defer m_hier.deinit(a);
    try testing.expect(m_dense.vertices.len > 0);
    try testing.expectEqualSlices(f32, m_dense.vertices, m_hier.vertices);
    try testing.expectEqualSlices(u32, m_dense.indices, m_hier.indices);
}

fn cosine(data: []const f32, ref: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..data.len) |i| {
        dot += @as(f64, data[i]) * ref[i];
        na += @as(f64, data[i]) * data[i];
        nb += @as(f64, ref[i]) * ref[i];
    }
    return dot / (std.math.sqrt(na) * std.math.sqrt(nb));
}
fn arrayCosine(a: std.mem.Allocator, arr: mlx.mlx_array, ref: []const f32, s: S) !f64 {
    _ = a;
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_array_eval(f);
    const n: usize = @intCast(mlx.mlx_array_size(f));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return cosine(d[0..n], ref);
}

// Oracle 1: DINO features. HY3D_TEST_MODEL, HY3D_DINO_IN ([1,3,518,518] f32,
// post-preprocess — isolates resize-kernel drift), HY3D_DINO_OUT ([1,1370,1024]).
test "hy3d oracle: DINO features match reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3D_TEST_MODEL") orelse return error.SkipZigTest);
    const in_p = std.mem.span(std.c.getenv("HY3D_DINO_IN") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("HY3D_DINO_OUT") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const in_d = try readF32(io, a, in_p);
    defer a.free(in_d);
    const ref = try readF32(io, a, out_p);
    defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var dino = try loadDino(a, cfg, model_dir, s);
    defer dino.deinit();
    const side: c_int = @intCast(cfg.dino_image_size);
    const psh = [_]c_int{ 1, 3, side, side };
    const parr = mlx.mlx_array_new_data(in_d.ptr, &psh, 4, .float32);
    defer _ = mlx.mlx_array_free(parr);
    const feats = try dino.encode(parr);
    defer _ = mlx.mlx_array_free(feats);
    const corr = try arrayCosine(a, feats, ref, s);
    std.debug.print("[hy3d-dino] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.999);
}

// Oracle 2: DiT one-step velocity. HY3D_VEL_LAT [2,4096,64], HY3D_VEL_CTX
// [2,1370,1024], HY3D_VEL_SIGMA (float string), HY3D_VEL [2,4096,64].
test "hy3d oracle: DiT velocity matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3D_TEST_MODEL") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("HY3D_VEL_LAT") orelse return error.SkipZigTest);
    const ctx_p = std.mem.span(std.c.getenv("HY3D_VEL_CTX") orelse return error.SkipZigTest);
    const sig_s = std.mem.span(std.c.getenv("HY3D_VEL_SIGMA") orelse return error.SkipZigTest);
    const vel_p = std.mem.span(std.c.getenv("HY3D_VEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat = try readF32(io, a, lat_p);
    defer a.free(lat);
    const ctx = try readF32(io, a, ctx_p);
    defer a.free(ctx);
    const ref = try readF32(io, a, vel_p);
    defer a.free(ref);
    const sigma = try std.fmt.parseFloat(f32, sig_s);
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var dit = try loadDit(a, cfg, model_dir, s);
    defer dit.deinit();
    const B: c_int = @intCast(lat.len / (@as(usize, cfg.num_latents) * cfg.embed_dim));
    const lsh = [_]c_int{ B, @intCast(cfg.num_latents), @intCast(cfg.embed_dim) };
    const lf = mlx.mlx_array_new_data(lat.ptr, &lsh, 3, .float32);
    defer _ = mlx.mlx_array_free(lf);
    const lh = try astype(lf, .float16, s);
    defer _ = mlx.mlx_array_free(lh);
    const csh = [_]c_int{ B, @intCast(cfg.dinoTokens()), @intCast(cfg.context_dim) };
    const cf = mlx.mlx_array_new_data(ctx.ptr, &csh, 3, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const ch = try astype(cf, .float16, s);
    defer _ = mlx.mlx_array_free(ch);
    const vel = try dit.forward(lh, ch, sigma);
    defer _ = mlx.mlx_array_free(vel);
    const corr = try arrayCosine(a, vel, ref, s);
    std.debug.print("[hy3d-dit] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.995);
}

// Oracle 3: latent→SDF on fixed points. HY3D_SDF_LAT [1,4096,64] (final
// denoised, PRE-scale-factor), HY3D_SDF_PTS [1,P,3], HY3D_SDF [P] — exercises
// ÷scale_factor, post_kl, all 16 blocks and the geo decoder (incl. the fused
// per-head qkv de-interleave — a plain-concat conversion fails HERE).
test "hy3d oracle: latent to SDF matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3D_TEST_MODEL") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("HY3D_SDF_LAT") orelse return error.SkipZigTest);
    const pts_p = std.mem.span(std.c.getenv("HY3D_SDF_PTS") orelse return error.SkipZigTest);
    const sdf_p = std.mem.span(std.c.getenv("HY3D_SDF") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat = try readF32(io, a, lat_p);
    defer a.free(lat);
    const pts = try readF32(io, a, pts_p);
    defer a.free(pts);
    const ref = try readF32(io, a, sdf_p);
    defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var vae = try loadVae(a, cfg, model_dir, s);
    defer vae.deinit();
    const lsh = [_]c_int{ 1, @intCast(cfg.num_latents), @intCast(cfg.embed_dim) };
    const lf = mlx.mlx_array_new_data(lat.ptr, &lsh, 3, .float32);
    defer _ = mlx.mlx_array_free(lf);
    const latent_set = try vaeDecodeLatentSet(&vae, lf);
    defer _ = mlx.mlx_array_free(latent_set);
    var kv = try geoDecodePrepare(&vae, latent_set);
    defer kv.deinit();
    const P: c_int = @intCast(pts.len / 3);
    const psh = [_]c_int{ 1, P, 3 };
    const pf = mlx.mlx_array_new_data(pts.ptr, &psh, 3, .float32);
    defer _ = mlx.mlx_array_free(pf);
    const logits = try geoDecodeChunk(&vae, &kv, pf);
    defer _ = mlx.mlx_array_free(logits);
    const n: usize = @intCast(mlx.mlx_array_size(logits));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(logits) orelse return error.NoData;
    const corr = cosine(d[0..n], ref);
    std.debug.print("[hy3d-sdf] corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.995);
}

// Oracle 4: full denoise from injected noise (torch vs mlx RNG never match).
// HY3D_DENOISE_INIT [1,4096,64], HY3D_DENOISE_CTX [1,1370,1024],
// HY3D_DENOISE_STEPS, HY3D_DENOISE_OUT [1,4096,64]; guidance 5.0 (override
// via HY3D_DENOISE_GUIDANCE).
test "hy3d oracle: denoise loop matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3D_TEST_MODEL") orelse return error.SkipZigTest);
    const init_p = std.mem.span(std.c.getenv("HY3D_DENOISE_INIT") orelse return error.SkipZigTest);
    const ctx_p = std.mem.span(std.c.getenv("HY3D_DENOISE_CTX") orelse return error.SkipZigTest);
    const steps_s = std.mem.span(std.c.getenv("HY3D_DENOISE_STEPS") orelse return error.SkipZigTest);
    const out_p = std.mem.span(std.c.getenv("HY3D_DENOISE_OUT") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const init_d = try readF32(io, a, init_p);
    defer a.free(init_d);
    const ctx = try readF32(io, a, ctx_p);
    defer a.free(ctx);
    const ref = try readF32(io, a, out_p);
    defer a.free(ref);
    const steps = try std.fmt.parseInt(u32, steps_s, 10);
    const guidance: f32 = if (std.c.getenv("HY3D_DENOISE_GUIDANCE")) |v| try std.fmt.parseFloat(f32, std.mem.span(v)) else 5.0;
    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var dit = try loadDit(a, cfg, model_dir, s);
    defer dit.deinit();
    const ish = [_]c_int{ 1, @intCast(cfg.num_latents), @intCast(cfg.embed_dim) };
    const init_f = mlx.mlx_array_new_data(init_d.ptr, &ish, 3, .float32);
    defer _ = mlx.mlx_array_free(init_f);
    const csh = [_]c_int{ 1, @intCast(cfg.dinoTokens()), @intCast(cfg.context_dim) };
    const cf = mlx.mlx_array_new_data(ctx.ptr, &csh, 3, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const out = try denoise(&dit, a, cf, steps, guidance, 0, init_f, null);
    defer _ = mlx.mlx_array_free(out);
    const corr = try arrayCosine(a, out, ref, s);
    std.debug.print("[hy3d-denoise] steps={d} corr={d:.6}\n", .{ steps, corr });
    try testing.expect(corr > 0.99);
}

// Oracle 5: e2e SDF grid at reduced resolution, from the SAME injected
// noise/ctx as oracle 4 (grids compare robustly; MC vertex arrays don't align
// 1:1 across implementations). HY3D_E2E_GRID ((R+1)³ f32), HY3D_E2E_RES,
// optional HY3D_E2E_NVERT (±10% mesh sanity), HY3D_E2E_MIN (default 0.98;
// set 0.97 for the 8-bit build).
test "hy3d oracle: e2e SDF grid matches reference" {
    const model_dir = std.mem.span(std.c.getenv("HY3D_TEST_MODEL") orelse return error.SkipZigTest);
    const init_p = std.mem.span(std.c.getenv("HY3D_DENOISE_INIT") orelse return error.SkipZigTest);
    const ctx_p = std.mem.span(std.c.getenv("HY3D_DENOISE_CTX") orelse return error.SkipZigTest);
    const steps_s = std.mem.span(std.c.getenv("HY3D_DENOISE_STEPS") orelse return error.SkipZigTest);
    const grid_p = std.mem.span(std.c.getenv("HY3D_E2E_GRID") orelse return error.SkipZigTest);
    const res_s = std.mem.span(std.c.getenv("HY3D_E2E_RES") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const init_d = try readF32(io, a, init_p);
    defer a.free(init_d);
    const ctx = try readF32(io, a, ctx_p);
    defer a.free(ctx);
    const ref_grid = try readF32(io, a, grid_p);
    defer a.free(ref_grid);
    const steps = try std.fmt.parseInt(u32, steps_s, 10);
    const res = try std.fmt.parseInt(u32, res_s, 10);
    const guidance: f32 = if (std.c.getenv("HY3D_DENOISE_GUIDANCE")) |v| try std.fmt.parseFloat(f32, std.mem.span(v)) else 5.0;
    const min_corr: f64 = if (std.c.getenv("HY3D_E2E_MIN")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else 0.98;

    const s = mlx.mlx_default_gpu_stream_new();
    const cfg = try readConfigFile(io, a, model_dir);
    var dit = try loadDit(a, cfg, model_dir, s);
    defer dit.deinit();
    var vae = try loadVae(a, cfg, model_dir, s);
    defer vae.deinit();

    const ish = [_]c_int{ 1, @intCast(cfg.num_latents), @intCast(cfg.embed_dim) };
    const init_f = mlx.mlx_array_new_data(init_d.ptr, &ish, 3, .float32);
    defer _ = mlx.mlx_array_free(init_f);
    const csh = [_]c_int{ 1, @intCast(cfg.dinoTokens()), @intCast(cfg.context_dim) };
    const cf = mlx.mlx_array_new_data(ctx.ptr, &csh, 3, .float32);
    defer _ = mlx.mlx_array_free(cf);
    const latent = try denoise(&dit, a, cf, steps, guidance, 0, init_f, null);
    defer _ = mlx.mlx_array_free(latent);
    const latent_set = try vaeDecodeLatentSet(&vae, latent);
    defer _ = mlx.mlx_array_free(latent_set);
    const grid = try decodeVolume(&vae, a, latent_set, res, VOLUME_BOUND, 32768, null);
    defer a.free(grid);
    try testing.expectEqual(ref_grid.len, grid.len);
    const corr = cosine(grid, ref_grid);
    std.debug.print("[hy3d-e2e] res={d} corr={d:.6}\n", .{ res, corr });
    try testing.expect(corr > min_corr);

    // Mesh sanity: extract, non-empty, bbox within the volume, and (if the
    // reference count is provided) vertex count within ±10%.
    const n: usize = res + 1;
    const k = 2.0 * VOLUME_BOUND / @as(f32, @floatFromInt(res + 1));
    var mesh = try mc.extract(a, grid, .{ n, n, n }, 0.0, .{ k, k, k }, .{ -VOLUME_BOUND, -VOLUME_BOUND, -VOLUME_BOUND });
    defer mesh.deinit(a);
    const nvert = mesh.vertices.len / 3;
    std.debug.print("[hy3d-e2e] vertices={d} tris={d}\n", .{ nvert, mesh.indices.len / 3 });
    try testing.expect(nvert > 0);
    var i: usize = 0;
    while (i < mesh.vertices.len) : (i += 1) {
        try testing.expect(mesh.vertices[i] >= -VOLUME_BOUND - 0.01 and mesh.vertices[i] <= VOLUME_BOUND + 0.01);
    }
    if (std.c.getenv("HY3D_E2E_NVERT")) |v| {
        const ref_n = try std.fmt.parseInt(usize, std.mem.span(v), 10);
        const lo = ref_n * 9 / 10;
        const hi = ref_n * 11 / 10;
        try testing.expect(nvert >= lo and nvert <= hi);
    }
}
