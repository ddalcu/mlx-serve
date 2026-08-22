//! DeepSeek-V4-Flash (deepseek_v4) native architecture: weight structs +
//! loading. Forward pass lands next (see the plan at the bottom of this
//! header).
//!
//! Loaded from OUR converted mirror (tests/convert_dsv4_weights.py): bare
//! inference-style tensor names (`embed.weight`, `layers.N.attn.wkv.weight`),
//! stacked `[E, out, in]` expert banks
//! (`layers.N.ffn.experts.{w1,w2,w3}.{weight,scales,biases}`), mixed affine
//! quant (experts 2/3-bit, spine 8-bit, compressor/router bf16) resolved PER
//! WEIGHT from packed geometry — never from a global config mode.
//!
//! Reference: the release's own inference/{model,kernel}.py (torch); the
//! faithful python oracle is tests/dsv4_mlx_ref.py (start_pos==0 path);
//! full arch notes in the dsv4-port memory file.
//!
//! Forward plan (task 5): MQA over ONE head_dim latent — low-rank Q with
//! unweighted per-head RMS, rope on the last rope_head_dim dims, kv fp8-sim
//! (e4m3 gs64 ue8m0) on non-rope dims, sliding-window-128 raw + per-layer
//! compressed history (ratio-4 overlap + top-512 indexer / plain all-visible),
//! per-head sink in the softmax denominator only, INVERSE rope on the output,
//! grouped low-rank O; Sinkhorn hyper-connections around both sublayers;
//! sqrt(softplus) MoE with hash routing on the first num_hash_layers and
//! clipped SwiGLU (shared expert included). Serial v1: no spec, no prefix
//! cache, no batched decode.

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const transformer = @import("transformer.zig");
const log = @import("log.zig");

const ModelConfig = model.ModelConfig;
const QuantParams = transformer.QuantParams;

/// Quantized linear: non-owning handles into the Weights map + solved params.
pub const Q = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
    qp: QuantParams,
};

/// A Q reshaped to per-group slabs [og, ol, ·] for the grouped low-rank O
/// batched quantized_matmul. OWNING view handles (freed in deinit); qp is
/// the parent Q's.
const Q3 = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
};

/// int8-g32 side copy of a layer's combined compressor-input operand
/// (`comp_in_t`), served at decode/verify widths under the user-facing
/// `--decode-attn-quant` config flag. OWNING handles (freed in deinit).
const CompInQ = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,
};

/// Learned gated-pooling KV compressor (bf16 weights — the compression path
/// is fp32-sensitive, the converter never quantizes it).
pub const CompressorW = struct {
    wkv: mlx.mlx_array,
    wgate: mlx.mlx_array,
    ape: mlx.mlx_array,
    norm: mlx.mlx_array,
};

pub const IndexerW = struct {
    wq_b: Q,
    weights_proj: mlx.mlx_array,
    comp: CompressorW,
};

pub const Dsv4Layer = struct {
    attn_norm: mlx.mlx_array,
    ffn_norm: mlx.mlx_array,
    hc_attn_fn: mlx.mlx_array,
    hc_attn_base: mlx.mlx_array,
    hc_attn_scale: mlx.mlx_array,
    hc_ffn_fn: mlx.mlx_array,
    hc_ffn_base: mlx.mlx_array,
    hc_ffn_scale: mlx.mlx_array,

    wq_a: Q,
    q_norm: mlx.mlx_array,
    wq_b: Q,
    wkv: Q,
    kv_norm: mlx.mlx_array,
    wo_a: Q,
    wo_b: Q,
    attn_sink: mlx.mlx_array,
    compress_ratio: u8, // 0 = pure sliding window
    comp: ?CompressorW,
    idx: ?IndexerW, // ratio-4 layers only

    gate_w: mlx.mlx_array,
    gate_bias: ?mlx.mlx_array, // score-routed layers
    tid2eid: ?mlx.mlx_array, // hash-routed layers (first num_hash_layers)
    experts_w1: Q,
    experts_w2: Q,
    experts_w3: Q,
    shared_w1: Q,
    shared_w2: Q,
    shared_w3: Q,
};

/// DSpark extras beyond the trunk-shaped stage layers: stage-0 conditioning
/// projection and the last stage's own head machinery. embed/head are SHARED
/// with the trunk (the reference assigns the same modules).
pub const DsparkW = struct {
    main_proj: Q, // [dim, dim * n_targets]
    main_norm: mlx.mlx_array,
    last_norm: mlx.mlx_array, // mtp.{last}.norm
    markov_w1: mlx.mlx_array, // [V, rank] bf16 (bigram embed)
    markov_w2: mlx.mlx_array, // [V, rank] bf16 (bigram head)
    conf_proj: mlx.mlx_array, // [1, dim + rank]
    hc_head_fn: mlx.mlx_array, // f32 — last stage's OWN collapse params
    hc_head_base: mlx.mlx_array,
    hc_head_scale: mlx.mlx_array,
    n_stages: u32,
};

pub const Dsv4Weights = struct {
    embed: Q,
    head: Q,
    final_norm: mlx.mlx_array,
    hc_head_fn: mlx.mlx_array,
    hc_head_base: mlx.mlx_array,
    hc_head_scale: mlx.mlx_array,
    /// Trunk layers [0, num_hidden_layers) then DSpark stages — one slice so
    /// the GPU decode helpers address a stage by plain layer index.
    layers: []Dsv4Layer,
    dspark: ?DsparkW,
    allocator: std.mem.Allocator,

    /// Handles are NON-owning: the Weights map stays alive for the model's
    /// lifetime (house convention — LayerWeights do the same).
    pub fn deinit(self: *Dsv4Weights) void {
        self.allocator.free(self.layers);
    }
};

const NameBuf = [192]u8;

fn getReq(w: *const model.Weights, buf: *NameBuf, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, args) catch return error.NameTooLong;
    return w.get(name) orelse {
        log.err("dsv4 MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

/// Load a quantized triple `<base>.{weight,scales,biases}` and solve its
/// quant params from geometry (mixed-mix checkpoints resolve per weight).
fn getQ(cfg: *const ModelConfig, w: *const model.Weights, buf: *NameBuf, in_dim: u32, comptime base: []const u8, args: anytype) !Q {
    const wt = try getReq(w, buf, base ++ ".weight", args);
    const sc = try getReq(w, buf, base ++ ".scales", args);
    const bs = try getReq(w, buf, base ++ ".biases", args);
    return .{ .w = wt, .s = sc, .b = bs, .qp = transformer.computeQuantParams(cfg, wt, sc, in_dim) };
}

fn getCompressor(w: *const model.Weights, buf: *NameBuf, comptime pfx: []const u8, args: anytype) !CompressorW {
    return .{
        .wkv = try getReq(w, buf, pfx ++ ".wkv.weight", args),
        .wgate = try getReq(w, buf, pfx ++ ".wgate.weight", args),
        .ape = try getReq(w, buf, pfx ++ ".ape", args),
        .norm = try getReq(w, buf, pfx ++ ".norm.weight", args),
    };
}

/// One trunk-shaped layer under `pfx ++ ".{d}"` — the trunk uses "layers",
/// DSpark stages use "mtp" (ratio 0, never hash-routed, otherwise identical).
fn loadDsv4Layer(cfg: *const ModelConfig, w: *const model.Weights, buf: *NameBuf, li: usize, comptime pfx: []const u8, ratio: u8, is_hash: bool) !Dsv4Layer {
    const hidden = cfg.hidden_size;
    const q_lora = cfg.dsv4_q_lora_rank;
    const moe_inter = cfg.moe_intermediate_size;
    const o_in = cfg.num_attention_heads * cfg.head_dim / cfg.dsv4_o_groups;
    const o_all = cfg.dsv4_o_groups * cfg.dsv4_o_lora_rank;
    return .{
        .attn_norm = try getReq(w, buf, pfx ++ ".{d}.attn_norm.weight", .{li}),
        .ffn_norm = try getReq(w, buf, pfx ++ ".{d}.ffn_norm.weight", .{li}),
        .hc_attn_fn = try getReq(w, buf, pfx ++ ".{d}.hc_attn_fn", .{li}),
        .hc_attn_base = try getReq(w, buf, pfx ++ ".{d}.hc_attn_base", .{li}),
        .hc_attn_scale = try getReq(w, buf, pfx ++ ".{d}.hc_attn_scale", .{li}),
        .hc_ffn_fn = try getReq(w, buf, pfx ++ ".{d}.hc_ffn_fn", .{li}),
        .hc_ffn_base = try getReq(w, buf, pfx ++ ".{d}.hc_ffn_base", .{li}),
        .hc_ffn_scale = try getReq(w, buf, pfx ++ ".{d}.hc_ffn_scale", .{li}),
        .wq_a = try getQ(cfg, w, buf, hidden, pfx ++ ".{d}.attn.wq_a", .{li}),
        .q_norm = try getReq(w, buf, pfx ++ ".{d}.attn.q_norm.weight", .{li}),
        .wq_b = try getQ(cfg, w, buf, q_lora, pfx ++ ".{d}.attn.wq_b", .{li}),
        .wkv = try getQ(cfg, w, buf, hidden, pfx ++ ".{d}.attn.wkv", .{li}),
        .kv_norm = try getReq(w, buf, pfx ++ ".{d}.attn.kv_norm.weight", .{li}),
        .wo_a = try getQ(cfg, w, buf, o_in, pfx ++ ".{d}.attn.wo_a", .{li}),
        .wo_b = try getQ(cfg, w, buf, o_all, pfx ++ ".{d}.attn.wo_b", .{li}),
        .attn_sink = try getReq(w, buf, pfx ++ ".{d}.attn.attn_sink", .{li}),
        .compress_ratio = ratio,
        .comp = if (ratio != 0)
            try getCompressor(w, buf, pfx ++ ".{d}.attn.compressor", .{li})
        else
            null,
        .idx = if (ratio == 4) .{
            .wq_b = try getQ(cfg, w, buf, q_lora, pfx ++ ".{d}.attn.indexer.wq_b", .{li}),
            .weights_proj = try getReq(w, buf, pfx ++ ".{d}.attn.indexer.weights_proj.weight", .{li}),
            .comp = try getCompressor(w, buf, pfx ++ ".{d}.attn.indexer.compressor", .{li}),
        } else null,
        .gate_w = try getReq(w, buf, pfx ++ ".{d}.ffn.gate.weight", .{li}),
        .gate_bias = if (is_hash) null else try getReq(w, buf, pfx ++ ".{d}.ffn.gate.bias", .{li}),
        .tid2eid = if (is_hash) try getReq(w, buf, pfx ++ ".{d}.ffn.gate.tid2eid", .{li}) else null,
        .experts_w1 = try getQ(cfg, w, buf, hidden, pfx ++ ".{d}.ffn.experts.w1", .{li}),
        .experts_w2 = try getQ(cfg, w, buf, moe_inter, pfx ++ ".{d}.ffn.experts.w2", .{li}),
        .experts_w3 = try getQ(cfg, w, buf, hidden, pfx ++ ".{d}.ffn.experts.w3", .{li}),
        .shared_w1 = try getQ(cfg, w, buf, hidden, pfx ++ ".{d}.ffn.shared_experts.w1", .{li}),
        .shared_w2 = try getQ(cfg, w, buf, moe_inter, pfx ++ ".{d}.ffn.shared_experts.w2", .{li}),
        .shared_w3 = try getQ(cfg, w, buf, hidden, pfx ++ ".{d}.ffn.shared_experts.w3", .{li}),
    };
}

/// DSpark stage count: ratio-table entries past the trunk (the release
/// carries no n_mtp_layers key and a stale num_nextn_predict_layers=1 against
/// 3 shipped stages), gated on dspark_block_size. A config that declares
/// DSpark whose mtp.* weights are missing is a hard MissingWeight load error
/// — only our converter's layout is supported, and it always ships them.
fn dsparkStageCount(cfg: *const ModelConfig) u32 {
    if (cfg.dsv4_dspark_block_size == 0) return 0;
    if (cfg.dsv4_n_compress_ratios <= cfg.num_hidden_layers) return 0;
    return cfg.dsv4_n_compress_ratios - cfg.num_hidden_layers;
}

pub fn loadDsv4Weights(allocator: std.mem.Allocator, cfg: *const ModelConfig, w: *const model.Weights) !Dsv4Weights {
    var buf: NameBuf = undefined;
    const n_layers = cfg.num_hidden_layers;
    const hidden = cfg.hidden_size;
    const n_mtp = dsparkStageCount(cfg);

    const layers = try allocator.alloc(Dsv4Layer, n_layers + n_mtp);
    errdefer allocator.free(layers);

    for (layers[0..n_layers], 0..) |*ly, li| {
        const ratio: u8 = if (li < cfg.dsv4_n_compress_ratios) cfg.dsv4_compress_ratios[li] else 0;
        ly.* = try loadDsv4Layer(cfg, w, &buf, li, "layers", ratio, li < cfg.dsv4_hash_layers);
    }
    for (layers[n_layers..], 0..) |*ly, st| {
        const ratio: u8 = cfg.dsv4_compress_ratios[n_layers + st];
        if (ratio != 0) {
            log.err("dsv4: DSpark stage {d} declares compress_ratio {d} (must be 0)\n", .{ st, ratio });
            return error.UnsupportedDsv4Config;
        }
        ly.* = try loadDsv4Layer(cfg, w, &buf, st, "mtp", 0, false);
    }

    const dspark: ?DsparkW = if (n_mtp > 0) .{
        .main_proj = try getQ(cfg, w, &buf, hidden * cfg.dsv4_n_dspark_target_layers, "mtp.0.main_proj", .{}),
        .main_norm = try getReq(w, &buf, "mtp.0.main_norm.weight", .{}),
        .last_norm = try getReq(w, &buf, "mtp.{d}.norm.weight", .{n_mtp - 1}),
        .markov_w1 = try getReq(w, &buf, "mtp.{d}.markov_head.markov_w1.weight", .{n_mtp - 1}),
        .markov_w2 = try getReq(w, &buf, "mtp.{d}.markov_head.markov_w2.weight", .{n_mtp - 1}),
        .conf_proj = try getReq(w, &buf, "mtp.{d}.confidence_head.proj.weight", .{n_mtp - 1}),
        .hc_head_fn = try getReq(w, &buf, "mtp.{d}.hc_head_fn", .{n_mtp - 1}),
        .hc_head_base = try getReq(w, &buf, "mtp.{d}.hc_head_base", .{n_mtp - 1}),
        .hc_head_scale = try getReq(w, &buf, "mtp.{d}.hc_head_scale", .{n_mtp - 1}),
        .n_stages = n_mtp,
    } else null;

    return .{
        .embed = try getQ(cfg, w, &buf, hidden, "embed", .{}),
        .head = try getQ(cfg, w, &buf, hidden, "head", .{}),
        .final_norm = try getReq(w, &buf, "norm.weight", .{}),
        .hc_head_fn = try getReq(w, &buf, "hc_head_fn", .{}),
        .hc_head_base = try getReq(w, &buf, "hc_head_base", .{}),
        .hc_head_scale = try getReq(w, &buf, "hc_head_scale", .{}),
        .layers = layers,
        .dspark = dspark,
        .allocator = allocator,
    };
}

// ── numeric core (pure Zig, host-side; kernel ports must match these) ───
//
// The QAT simulations are LOAD-BEARING: the checkpoint was trained with its
// KV cache fp8-simulated (e4m3 gs64) and the indexer path fp4-simulated
// (e2m1 gs32, after a Hadamard rotate), both with POWER-OF-2 (ue8m0) scales
// = 2^ceil(log2(amax/max_code)). Semantics transcribed from the release's
// inference/kernel.py; golden values in the tests below come from the python
// oracle (tests/dsv4_mlx_ref.py), so oracle and engine share one definition.

fn roundHalfEvenPos(v: f64) f64 {
    const f = @floor(v);
    if (v - f == 0.5) {
        return if (@mod(f, 2.0) == 0.0) f else f + 1.0;
    }
    return @round(v);
}

/// Round |y| onto a float grid with `mant_bits` mantissa bits, subnormal
/// floor 2^min_exp, saturating at max_val. Ties to even (matches np.round).
fn roundToGrid(y: f64, mant_bits: f64, min_exp: f64, max_val: f64) f64 {
    var a = @abs(y);
    a = @min(a, max_val);
    var e = @floor(@log2(@max(a, std.math.pow(f64, 2.0, min_exp))));
    e = @max(e, min_exp);
    const quantum = std.math.pow(f64, 2.0, e - mant_bits);
    const q = @min(roundHalfEvenPos(a / quantum) * quantum, max_val);
    return if (y < 0) -q else q;
}

fn simInPlace(x: []f32, group: usize, amax_floor: f64, code_max: f64, mant_bits: f64, min_exp: f64) void {
    var g: usize = 0;
    while (g < x.len) : (g += group) {
        const end = @min(g + group, x.len);
        var amax: f64 = amax_floor;
        for (x[g..end]) |v| amax = @max(amax, @abs(@as(f64, v)));
        const scale = std.math.pow(f64, 2.0, @ceil(@log2(amax / code_max)));
        for (x[g..end]) |*v| {
            const y = std.math.clamp(@as(f64, v.*) / scale, -code_max, code_max);
            v.* = @floatCast(roundToGrid(y, mant_bits, min_exp, code_max) * scale);
        }
    }
}

/// e4m3 quant-dequant with ue8m0 per-group scales (KV-cache QAT sim).
pub fn fp8SimInPlace(x: []f32, group: usize) void {
    simInPlace(x, group, 1e-4, 448.0, 3.0, -6.0);
}

/// e2m1 quant-dequant with ue8m0 per-group scales (indexer QAT sim).
pub fn fp4SimInPlace(x: []f32, group: usize) void {
    simInPlace(x, group, 6.0 * std.math.pow(f64, 2.0, -126.0), 6.0, 1.0, 0.0);
}

/// In-place fast Walsh–Hadamard transform scaled by n^-1/2 (the reference's
/// hadamard_transform(x, scale=n**-0.5)). n must be a power of two.
pub fn hadamardInPlace(x: []f32) void {
    const n = x.len;
    std.debug.assert(n != 0 and (n & (n - 1)) == 0);
    var h: usize = 1;
    while (h < n) : (h *= 2) {
        var i: usize = 0;
        while (i < n) : (i += 2 * h) {
            for (0..h) |j| {
                const a = x[i + j];
                const b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
    }
    const inv = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
    for (x) |*v| v.* *= inv;
}

fn sigmoidF32(v: f32) f32 {
    return 1.0 / (1.0 + @exp(-v));
}

pub const HcSplit = struct { pre: [8]f32, post: [8]f32, comb: [64]f32 };

/// hc_split_sinkhorn (kernel.py): pre = σ(m·s0+b)+eps; post = 2σ(m·s1+b);
/// comb = row-softmax(+eps) → colnorm → (iters-1)×(rownorm, colnorm).
/// hc ≤ 8. Returns fixed-size buffers; read the first hc / hc² entries.
pub fn hcSplitSinkhorn(mixes: []const f32, hc_scale: []const f32, hc_base: []const f32, hc: usize, iters: u32, eps: f32) HcSplit {
    std.debug.assert(hc <= 8 and mixes.len >= (2 + hc) * hc);
    var out: HcSplit = .{ .pre = @splat(0), .post = @splat(0), .comb = @splat(0) };
    for (0..hc) |j| {
        out.pre[j] = sigmoidF32(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
        out.post[j] = 2.0 * sigmoidF32(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
    }
    var comb = out.comb[0 .. hc * hc];
    for (0..hc) |j| {
        for (0..hc) |k| comb[j * hc + k] = mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    // row softmax + eps
    for (0..hc) |j| {
        var m: f32 = -std.math.inf(f32);
        for (comb[j * hc ..][0..hc]) |v| m = @max(m, v);
        var sum: f32 = 0;
        for (comb[j * hc ..][0..hc]) |*v| {
            v.* = @exp(v.* - m);
            sum += v.*;
        }
        for (comb[j * hc ..][0..hc]) |*v| v.* = v.* / sum + eps;
    }
    var it: u32 = 0;
    while (it < iters) : (it += 1) {
        if (it > 0) {
            // row normalize (skipped on the first pass — softmax already did)
            for (0..hc) |j| {
                var sum: f32 = 0;
                for (comb[j * hc ..][0..hc]) |v| sum += v;
                for (comb[j * hc ..][0..hc]) |*v| v.* /= (sum + eps);
            }
        }
        // column normalize
        for (0..hc) |k| {
            var sum: f32 = 0;
            for (0..hc) |j| sum += comb[j * hc + k];
            for (0..hc) |j| comb[j * hc + k] /= (sum + eps);
        }
    }
    return out;
}

// ── forward pass v1: host-centric correctness mode ─────────────────────
//
// Direct transcription of the validated python oracle (tests/dsv4_mlx_ref.py,
// itself transcribed from the release's inference/model.py, start_pos==0).
// Quantized matmuls run through MLX; everything exotic (rope, QAT sims,
// window/compressed index construction, sink softmax, Sinkhorn hc, gated
// pooling) runs in host f32 via the golden-tested helpers above. This locks
// the SEMANTICS in Zig behind a fixture-parity test; the graph/GPU migration
// happens afterwards WITH that test as the guard (never before it exists).

/// True when the first size() elements at data_ptr ARE the logical row-major
/// order (dims of extent 1 have don't-care strides). A broadcast or sliced
/// view fails this and must be materialized before a raw-pointer read.
fn isRowMajorContiguous(arr: mlx.mlx_array) bool {
    const ndim = mlx.mlx_array_ndim(arr);
    if (ndim == 0) return true;
    const shape = mlx.mlx_array_shape(arr);
    const strides = mlx.mlx_array_strides(arr);
    var expect: usize = 1;
    var i = ndim;
    while (i > 0) {
        i -= 1;
        const dim: usize = @intCast(shape[i]);
        if (dim != 1 and strides[i] != expect) return false;
        expect *= dim;
    }
    return true;
}

fn toHostF32(alloc: std.mem.Allocator, arr: mlx.mlx_array, len: usize, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    // astype to the SAME dtype is an MLX no-op VIEW: a strided/broadcast f32
    // input keeps its strides and its raw buffer is SMALLER than the logical
    // element count. Flat reshape is the one reliable materializer — a
    // non-row-major layout can never be viewed as 1-D, so MLX must copy
    // (add-scalar-zero does NOT work here: binary ops propagate a broadcast
    // input's strides to the output and compute only the unique elements).
    // Strides are only populated by eval, so the check runs post-eval.
    if (!isRowMajorContiguous(f)) {
        const flat_shape = [_]c_int{@intCast(mlx.mlx_array_size(f))};
        var mat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mat);
        try mlx.check(mlx.mlx_reshape(&mat, f, &flat_shape, 1, s));
        try mlx.check(mlx.mlx_array_set(&f, mat));
        try mlx.check(mlx.mlx_array_eval(f));
    }
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    const out = try alloc.alloc(f32, len);
    @memcpy(out, ptr[0..len]);
    return out;
}

fn toHostI64(alloc: std.mem.Allocator, arr: mlx.mlx_array, len: usize, s: mlx.mlx_stream) ![]i64 {
    // int tensors: go through f32 is lossy for big ids? tid2eid values < 256,
    // token ids < 2^17 — f32 exact below 2^24, safe here.
    const f = try toHostF32(alloc, arr, len, s);
    defer alloc.free(f);
    const out = try alloc.alloc(i64, len);
    for (out, f) |*o, v| o.* = @intFromFloat(v);
    return out;
}

fn hostToBf16(host: []const f32, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    return out;
}

/// y[m, out] = bf16(x[m, in]) @ q.wᵀ via MLX quantized matmul, back to host f32.
fn qmmHost(alloc: std.mem.Allocator, q: *const Q, x: []const f32, m: usize, in_dim: usize, out_dim: usize, s: mlx.mlx_stream) ![]f32 {
    const xb = try hostToBf16(x, &.{ @intCast(m), @intCast(in_dim) }, s);
    defer _ = mlx.mlx_array_free(xb);
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_quantized_matmul(&y, xb, q.w, q.s, q.b, true, mlx.mlx_optional_int.some(@intCast(q.qp.group_size)), mlx.mlx_optional_int.some(@intCast(q.qp.bits)), "affine", s));
    return toHostF32(alloc, y, m * out_dim, s);
}

/// Dequantize q to host f32 [out, in] (transient use only — wo_a einsum).
fn dequantHost(alloc: std.mem.Allocator, q: *const Q, len: usize, s: mlx.mlx_stream) ![]f32 {
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    const empty = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty);
    try mlx.check(mlx.mlx_dequantize(&d, q.w, q.s, q.b, mlx.mlx_optional_int.some(@intCast(q.qp.group_size)), mlx.mlx_optional_int.some(@intCast(q.qp.bits)), "affine", empty, mlx.mlx_optional_dtype{}, s));
    return toHostF32(alloc, d, len, s);
}

/// out[m, n] = x[m, k] @ w[n, k]ᵀ — naive host matmul (correctness mode).
fn matHost(out: []f32, x: []const f32, w: []const f32, m: usize, k: usize, n: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f64 = 0;
            const xr = x[i * k ..][0..k];
            const wr = w[j * k ..][0..k];
            for (xr, wr) |a, b| acc += @as(f64, a) * @as(f64, b);
            out[i * n + j] = @floatCast(acc);
        }
    }
}

fn rmsNormRow(x: []f32, weight: ?[]const f32, eps: f32) void {
    var ss: f64 = 0;
    for (x) |v| ss += @as(f64, v) * v;
    const r: f32 = @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps));
    if (weight) |w| {
        for (x, w) |*v, wv| v.* = v.* * r * wv;
    } else {
        for (x) |*v| v.* *= r;
    }
}

const Freqs = struct { cos: []f32, sin: []f32, half: usize };

/// YaRN-interpolated rope tables (inference/model.py precompute_freqs_cis).
fn precomputeFreqs(alloc: std.mem.Allocator, dim: usize, seqlen: usize, original_seq_len: u32, base: f64, factor: f64, beta_fast: f64, beta_slow: f64) !Freqs {
    const half = dim / 2;
    const freqs = try alloc.alloc(f64, half);
    defer alloc.free(freqs);
    for (freqs, 0..) |*f, i| f.* = 1.0 / std.math.pow(f64, base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim)));
    if (original_seq_len > 0) {
        const dimf: f64 = @floatFromInt(dim);
        const osl: f64 = @floatFromInt(original_seq_len);
        const corr = struct {
            fn dim_(nr: f64, d: f64, b: f64, msl: f64) f64 {
                return d * @log(msl / (nr * 2.0 * std.math.pi)) / (2.0 * @log(b));
            }
        };
        const low = @max(@floor(corr.dim_(beta_fast, dimf, base, osl)), 0.0);
        var high = @min(@ceil(corr.dim_(beta_slow, dimf, base, osl)), dimf - 1.0);
        if (low == high) high += 0.001;
        for (freqs, 0..) |*f, i| {
            const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0.0, 1.0);
            const smooth = 1.0 - ramp;
            f.* = f.* / factor * (1.0 - smooth) + f.* * smooth;
        }
    }
    const cos = try alloc.alloc(f32, seqlen * half);
    const sin = try alloc.alloc(f32, seqlen * half);
    for (0..seqlen) |t| {
        for (0..half) |i| {
            const ang = @as(f64, @floatFromInt(t)) * freqs[i];
            cos[t * half + i] = @floatCast(@cos(ang));
            sin[t * half + i] = @floatCast(@sin(ang));
        }
    }
    return .{ .cos = cos, .sin = sin, .half = half };
}

/// Interleaved-pair rope on the trailing rd dims of one row, at position pos.
fn ropeRow(x: []f32, fr: *const Freqs, pos: usize, inverse: bool) void {
    const half = fr.half;
    const cos = fr.cos[pos * half ..][0..half];
    const sin = fr.sin[pos * half ..][0..half];
    for (0..half) |i| {
        const a = x[2 * i];
        const b = x[2 * i + 1];
        const sn: f32 = if (inverse) -sin[i] else sin[i];
        x[2 * i] = a * cos[i] - b * sn;
        x[2 * i + 1] = a * sn + b * cos[i];
    }
}

const HostComp = struct {
    wkv: []f32,
    wgate: []f32,
    ape: []f32,
    norm: []f32,
    coff: usize,
    head_dim: usize,
    // transposed [dim, coff*d] bf16 views for GPU-side x @ w.T
    wkv_t: mlx.mlx_array,
    wgate_t: mlx.mlx_array,
    // GPU window-emission operands: norm weight (f32) + ape rows [ratio, cd]
    norm_g: mlx.mlx_array,
    ape_g: mlx.mlx_array,
};
const HostLayer = struct {
    // GPU-side transposed f32 operands for the per-token host matmuls
    hc_attn_fn_t: mlx.mlx_array,
    hc_ffn_fn_t: mlx.mlx_array,
    gate_w_t: mlx.mlx_array,
    sink_gpu: mlx.mlx_array, // [nh, 1] f32
    // f32 norm weights for the GPU decode chain (mlx_fast_rms_norm operands)
    attn_norm_g: mlx.mlx_array,
    ffn_norm_g: mlx.mlx_array,
    q_norm_g: mlx.mlx_array,
    kv_norm_g: mlx.mlx_array,
    // ONE [dim, W] f32 operand = [comp.wkv | comp.wgate | idx.wkv | idx.wgate]
    // — a single matmul + host sync per token feeds every compressor ring.
    comp_in_t: ?mlx.mlx_array,
    comp_in_w: usize,
    idx_wp_t: ?mlx.mlx_array, // [dim, ih] f32 (indexer weights_proj)
    attn_norm: []f32,
    ffn_norm: []f32,
    q_norm: []f32,
    kv_norm: []f32,
    hc_attn_fn: []f32,
    hc_attn_base: []f32,
    hc_attn_scale: []f32,
    hc_ffn_fn: []f32,
    hc_ffn_base: []f32,
    hc_ffn_scale: []f32,
    sink: []f32,
    gate_w: []f32,
    gate_bias: ?[]f32,
    tid2eid: ?[]i64,
    comp: ?HostComp,
    idx_wp: ?[]f32,
    idx_comp: ?HostComp,
};

pub const Dsv4Model = struct {
    dw: Dsv4Weights,
    hl: []HostLayer,
    final_norm: []f32,
    hc_head_fn: []f32,
    hc_head_base: []f32,
    hc_head_scale: []f32,
    embed_f32: []f32, // [V, D]
    arena: std.heap.ArenaAllocator,
    s: mlx.mlx_stream,
    // v0 serving: full-reforward decode keeps the request's token history
    // here (SERIAL-ONLY — the scheduler must never batch this arch v0).
    history: std.array_list.Managed(u32),
    // decode-path rope tables, lazily grown (arena-owned)
    dec_freqs_plain: ?Freqs = null,
    dec_freqs_yarn: ?Freqs = null,
    // per-request incremental decode state (serial-only; recreated when the
    // serving seam sees cache.step == 0)
    dec_state: ?Dsv4DecodeState = null,
    // per-layer wo_a dequantized as batched-matmul operands [og, gin, ol]
    // bf16 (built once at init — dequantizing 134 MB per layer per TOKEN was
    // the v0 decode's dominant cost). EMPTY when wo_a is served quantized
    // (the default): the bf16 slabs were 67 MB/layer = 2.9 GB of the ~7.8 GB
    // read per serial token, the single largest read in the forward.
    wo_a_deq: []mlx.mlx_array,
    // per-layer wo_a quantized triples reshaped to per-group slabs
    // [og, ol, ·] (pure views of the checkpoint arrays — batched
    // quantized_matmul reads the 8-bit weight in place). Empty when
    // `MLX_SERVE_DSV4_WO_QMM=0` restores the dequantized operands.
    wo_a_q3: []Q3,
    // per-layer int8-g32 side copies of comp_in_t served at decode/verify
    // widths (C ≤ 32) under the user-facing --decode-attn-quant flag; big
    // prefill chunks keep the dense bf16 operand (quality anchor). LOSSY by
    // design (the laguna decode-attn-quant contract). Empty when the flag is
    // off or on CPU streams (host-reference/test path stays dense).
    comp_in_q: []?CompInQ,
    // Lazy-decode GPU embed table [V, d] bf16 (GPU streams only; ~1 GB on
    // the real mirror — the RAM the retired wo_a_deq freed): the sampled
    // token id never touches the host, so decode logits stay a lazy graph
    // and generate.zig's pipelined next() overlaps build with GPU execution.
    embed_g: ?mlx.mlx_array = null,
    // GPU decode-chain constants
    ones_hd_g: mlx.mlx_array, // [hd] f32 ones (param-free per-head RMS weight)
    final_norm_g: mlx.mlx_array,
    hc_head_fn_t: mlx.mlx_array, // [hc*d, hc] f32
    hada_g: ?mlx.mlx_array, // [ihd, ihd] f32 Hadamard matrix (x @ H == FWHT)
    // fused Sinkhorn kernel (GPU streams; null → host-sync fallback)
    sink_k: ?SinkhornK = null,
    // fused Sinkhorn + y-collapse kernel (`MLX_SERVE_DSV4_SINKY=0` kills;
    // falls back to sink_k + the composed multiply/sum tail)
    sink_y_k: ?SinkhornK = null,
    sink_y_logged: bool = false,
    // fused hc_post kernel (`MLX_SERVE_DSV4_HCPOST=0` kills; needs sink_y_k
    // for the pack input — falls back to the composed matmul/multiply/add)
    hc_post_k: ?SinkhornK = null,
    hc_post_logged: bool = false,
    sink_consts: mlx.mlx_array, // [hd_full, rms_eps, hc_eps] f32
    sink_logged: bool = false,
    // geometry (copied from config)
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    rd: usize,
    window: usize,
    hc: usize,
    hc_iters: u32,
    hc_eps: f32,
    eps: f32,
    o_groups: usize,
    o_lora: usize,
    q_lora: usize,
    topk: usize,
    n_experts: usize,
    n_hash: usize,
    route_scale: f32,
    swiglu_limit: f32,
    moe_inter: usize,
    idx_heads: usize,
    idx_hd: usize,
    idx_topk: usize,
    vocab: usize,
    ratios: [128]u8,
    yarn_theta: f64,
    yarn_orig: u32,
    yarn_factor: f64,
    yarn_bf: f64,
    yarn_bs: f64,
    plain_theta: f64,
    // DSpark draft stages (0 = checkpoint ships none / disabled). Stage i
    // lives at dw.layers/hl/wo_a_deq index n_layers + i.
    n_mtp: usize,
    ds_block: usize,
    ds_noise: u32,
    ds_rank: usize,
    ds_targets: [8]u8,
    n_ds_targets: usize,
    ds_main_norm_g: ?mlx.mlx_array = null,
    ds_last_norm_g: ?mlx.mlx_array = null,
    // draft-head operands: last stage's own collapse + bigram/confidence heads
    ds_hc_head_fn_t: ?mlx.mlx_array = null, // [hc*d, hc] f32
    ds_markov_w2_t: ?mlx.mlx_array = null, // [rank, V] f32
    ds_conf_proj_t: ?mlx.mlx_array = null, // [d+rank, 1] f32
    ds_hc_head_base: []f32 = &.{}, // arena-owned host copies
    ds_hc_head_scale: []f32 = &.{},
    /// Sinkhorn kernel configs owned by the model, indexed by token count
    /// (slot 0 = the prefill sub-chunk). See `sinkhornCfgFor`.
    sink_cfg: [SINK_CFG_MAX + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    sink_y_cfg: [SINK_CFG_MAX + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    hc_post_cfg: [SINK_CFG_MAX + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    /// Confidence LOGIT below which a drafted position is not submitted for
    /// verification (see `dsparkConfThreshold`). Tests move it to ±inf to pin
    /// the open/shut ends.
    ds_conf_thr: f32 = 0,
    /// Per-round cost audit, armed by `MLX_SERVE_DSPARK_PROFILE` at load.
    ds_prof: ?DsparkProfile = null,
    /// Scratch laps written by the last `extendChunk` (profiling only): the
    /// trunk layer loop vs the vocab head, so a round can attribute its
    /// verify. The head is the M=B+1 lane — the shape MLX serves worst.
    ds_prof_layers_ns: u64 = 0,
    ds_prof_head_ns: u64 = 0,
    /// Of the layer loop: the blocking per-layer compressor-input reads.
    ds_prof_comp_sync_ns: u64 = 0,

    pub fn deinit(self: *Dsv4Model) void {
        for (self.hl) |*h| {
            _ = mlx.mlx_array_free(h.sink_gpu);
            _ = mlx.mlx_array_free(h.hc_attn_fn_t);
            _ = mlx.mlx_array_free(h.hc_ffn_fn_t);
            _ = mlx.mlx_array_free(h.gate_w_t);
            _ = mlx.mlx_array_free(h.attn_norm_g);
            _ = mlx.mlx_array_free(h.ffn_norm_g);
            _ = mlx.mlx_array_free(h.q_norm_g);
            _ = mlx.mlx_array_free(h.kv_norm_g);
            if (h.comp_in_t) |t| _ = mlx.mlx_array_free(t);
            if (h.idx_wp_t) |t| _ = mlx.mlx_array_free(t);
            if (h.comp) |*c| {
                _ = mlx.mlx_array_free(c.wkv_t);
                _ = mlx.mlx_array_free(c.wgate_t);
                _ = mlx.mlx_array_free(c.norm_g);
                _ = mlx.mlx_array_free(c.ape_g);
            }
            if (h.idx_comp) |*c| {
                _ = mlx.mlx_array_free(c.wkv_t);
                _ = mlx.mlx_array_free(c.wgate_t);
                _ = mlx.mlx_array_free(c.norm_g);
                _ = mlx.mlx_array_free(c.ape_g);
            }
        }
        for (self.wo_a_deq) |h| _ = mlx.mlx_array_free(h);
        for (self.wo_a_q3) |q3| {
            _ = mlx.mlx_array_free(q3.w);
            _ = mlx.mlx_array_free(q3.s);
            _ = mlx.mlx_array_free(q3.b);
        }
        for (self.comp_in_q) |maybe| if (maybe) |q| {
            _ = mlx.mlx_array_free(q.w);
            _ = mlx.mlx_array_free(q.s);
            _ = mlx.mlx_array_free(q.b);
        };
        if (self.embed_g) |h| _ = mlx.mlx_array_free(h);
        _ = mlx.mlx_array_free(self.ones_hd_g);
        _ = mlx.mlx_array_free(self.final_norm_g);
        _ = mlx.mlx_array_free(self.hc_head_fn_t);
        if (self.hada_g) |h| _ = mlx.mlx_array_free(h);
        _ = mlx.mlx_array_free(self.sink_consts);
        if (self.sink_k) |*sk| {
            _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
            _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
        }
        if (self.sink_y_k) |*sk| {
            _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
            _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
        }
        if (self.hc_post_k) |*sk| {
            _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
            _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
        }
        for (self.sink_cfg) |c| if (c) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        for (self.sink_y_cfg) |c| if (c) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        for (self.hc_post_cfg) |c| if (c) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        if (self.ds_main_norm_g) |h| _ = mlx.mlx_array_free(h);
        if (self.ds_last_norm_g) |h| _ = mlx.mlx_array_free(h);
        if (self.ds_hc_head_fn_t) |h| _ = mlx.mlx_array_free(h);
        if (self.ds_markov_w2_t) |h| _ = mlx.mlx_array_free(h);
        if (self.ds_conf_proj_t) |h| _ = mlx.mlx_array_free(h);
        if (self.dec_state) |*ds| deinitDecodeState(ds);
        self.history.deinit();
        self.arena.deinit();
        self.dw.deinit();
    }
};

fn transposedF32(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // The compression path is fp32-mandated by the reference ("compression
    // need fp32") — these operands stay f32, never bf16.
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, w, .float32, s));
    var tr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose(&tr, f, s));
    try mlx.check(mlx.mlx_array_eval(tr));
    return tr;
}

/// Evaluated f32 copy of a (bf16) weight — mlx_fast_rms_norm operand.
fn f32Handle(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var f = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f, w, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    return f;
}

fn uploadF32(data: []const f32, shape: []const c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

/// Hadamard matrix H[i,j] = (-1)^popcount(i&j) / sqrt(n): x @ H equals the
/// in-place FWHT `hadamardInPlace` (pinned by a unit test below).
fn buildHadamardF32(alloc: std.mem.Allocator, n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const host = try alloc.alloc(f32, n * n);
    defer alloc.free(host);
    const inv: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
    for (0..n) |i| {
        for (0..n) |j| {
            const neg = (@popCount(i & j) & 1) == 1;
            host[i * n + j] = if (neg) -inv else inv;
        }
    }
    const shape = [_]c_int{ @intCast(n), @intCast(n) };
    const raw = uploadF32(host, &shape);
    errdefer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_array_eval(raw));
    _ = s;
    return raw;
}

fn hostComp(alloc: std.mem.Allocator, c: *const CompressorW, dim: usize, head_dim: usize, ratio: usize, s: mlx.mlx_stream) !HostComp {
    const coff: usize = if (ratio == 4) 2 else 1;
    const ape_g = blk: {
        const f = try f32Handle(c.ape, s);
        defer _ = mlx.mlx_array_free(f);
        var r = mlx.mlx_array_new();
        const shp = [_]c_int{ @intCast(ratio), @intCast(coff * head_dim) };
        try mlx.check(mlx.mlx_reshape(&r, f, &shp, 2, s));
        try mlx.check(mlx.mlx_array_eval(r));
        break :blk r;
    };
    return .{
        .wkv = try toHostF32(alloc, c.wkv, coff * head_dim * dim, s),
        .wgate = try toHostF32(alloc, c.wgate, coff * head_dim * dim, s),
        .ape = try toHostF32(alloc, c.ape, ratio * coff * head_dim, s),
        .norm = try toHostF32(alloc, c.norm, head_dim, s),
        .coff = coff,
        .head_dim = head_dim,
        .wkv_t = try transposedF32(c.wkv, s),
        .wgate_t = try transposedF32(c.wgate, s),
        .norm_g = try f32Handle(c.norm, s),
        .ape_g = ape_g,
    };
}

/// y[m2, n] = x[m2, k] @ (pre-transposed bf16 weight [k, n]) on the GPU.
fn matMlx(alloc: std.mem.Allocator, x: []const f32, wt: mlx.mlx_array, m2: usize, k: usize, n: usize, s: mlx.mlx_stream) ![]f32 {
    const shape = [_]c_int{ @intCast(m2), @intCast(k) };
    const xb = mlx.mlx_array_new_data(x.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(xb);
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);
    try mlx.check(mlx.mlx_matmul(&res, xb, wt, s));
    return toHostF32(alloc, res, m2 * n, s);
}

/// Serving headroom the DSpark fit decision reserves on top of the model's
/// logical weight bytes: init-derived tensors (wo_a_deq, comp_in_t, GPU
/// norm handles ~1-2 GB on the real mirror), KV growth, decode/verify
/// transients. 6 GB covers the measured envelope with margin.
const DSPARK_MEM_HEADROOM: usize = 6 << 30;

/// Pure fit decision for the DSpark stage weights: admitted only when
/// trunk + stages + a serving headroom fit the device working-set budget.
/// Both weight terms are LOGICAL bytes (shape × itemsize) — at init time
/// trunk AND stages are still lazy (warmup materializes the trunk later),
/// so runtime active_memory sees neither and cannot feed this decision.
/// max_rec 0 = device query failed — admit rather than guess.
pub fn dsparkFitsBudget(stage_bytes: usize, trunk_bytes: usize, max_rec: usize, headroom_bytes: usize) bool {
    if (max_rec == 0) return true;
    return trunk_bytes +| stage_bytes +| headroom_bytes <= max_rec;
}

const WeightTally = struct {
    n: usize = 0,
    bytes: usize = 0,
    fn add(self: WeightTally, other: WeightTally) WeightTally {
        return .{ .n = self.n + other.n, .bytes = self.bytes + other.bytes };
    }
};

/// Walk every mlx_array reachable in `value` (comptime-reflective: Q
/// triples, optional sub-structs, plain handles; non-array fields ignored).
/// Appends to `vec` when given (tally-only otherwise) and returns count +
/// logical bytes (size × itemsize — known pre-eval, so the fit decision can
/// run BEFORE anything is materialized). Reflection makes the collection
/// structural — a future weight field cannot be left out.
fn appendWeightArrays(vec: ?mlx.mlx_vector_array, value: anytype) WeightTally {
    const T = @TypeOf(value);
    if (T == mlx.mlx_array) {
        if (value.ctx == null) return .{};
        if (vec) |v| _ = mlx.mlx_vector_array_append_value(v, value);
        return .{ .n = 1, .bytes = mlx.mlx_array_size(value) * mlx.mlx_array_itemsize(value) };
    }
    return switch (@typeInfo(T)) {
        .@"struct" => |st| blk: {
            var t = WeightTally{};
            inline for (st.field_names) |name| t = t.add(appendWeightArrays(vec, @field(value, name)));
            break :blk t;
        },
        .optional => if (value) |v| appendWeightArrays(vec, v) else .{},
        else => .{},
    };
}

pub fn initModel(gpa: std.mem.Allocator, cfg: *const ModelConfig, dw: Dsv4Weights, s: mlx.mlx_stream) !Dsv4Model {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const dim: usize = cfg.hidden_size;
    const hc: usize = cfg.dsv4_hc_mult;
    const mix = (2 + hc) * hc;
    const n_layers: usize = cfg.num_hidden_layers;
    // hl/wo_a_deq cover the DSpark stages too (dw.layers carries trunk THEN
    // stages) so the GPU decode helpers address a stage by plain index.
    const hl = try a.alloc(HostLayer, dw.layers.len);
    for (hl, dw.layers, 0..) |*h, *ly, li| {
        const ratio: usize = ly.compress_ratio;
        h.* = .{
            .sink_gpu = blk: {
                var f = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(f);
                try mlx.check(mlx.mlx_astype(&f, ly.attn_sink, .float32, s));
                var col = mlx.mlx_array_new();
                const shp = [_]c_int{ @intCast(cfg.num_attention_heads), 1 };
                try mlx.check(mlx.mlx_reshape(&col, f, &shp, 2, s));
                try mlx.check(mlx.mlx_array_eval(col));
                break :blk col;
            },
            .hc_attn_fn_t = try transposedF32(ly.hc_attn_fn, s),
            .hc_ffn_fn_t = try transposedF32(ly.hc_ffn_fn, s),
            .gate_w_t = try transposedF32(ly.gate_w, s),
            .attn_norm_g = try f32Handle(ly.attn_norm, s),
            .ffn_norm_g = try f32Handle(ly.ffn_norm, s),
            .q_norm_g = try f32Handle(ly.q_norm, s),
            .kv_norm_g = try f32Handle(ly.kv_norm, s),
            .comp_in_t = null,
            .comp_in_w = 0,
            .idx_wp_t = if (ly.idx) |*ix| try transposedF32(ix.weights_proj, s) else null,
            .attn_norm = try toHostF32(a, ly.attn_norm, dim, s),
            .ffn_norm = try toHostF32(a, ly.ffn_norm, dim, s),
            .q_norm = try toHostF32(a, ly.q_norm, cfg.dsv4_q_lora_rank, s),
            .kv_norm = try toHostF32(a, ly.kv_norm, cfg.head_dim, s),
            .hc_attn_fn = try toHostF32(a, ly.hc_attn_fn, mix * hc * dim, s),
            .hc_attn_base = try toHostF32(a, ly.hc_attn_base, mix, s),
            .hc_attn_scale = try toHostF32(a, ly.hc_attn_scale, 3, s),
            .hc_ffn_fn = try toHostF32(a, ly.hc_ffn_fn, mix * hc * dim, s),
            .hc_ffn_base = try toHostF32(a, ly.hc_ffn_base, mix, s),
            .hc_ffn_scale = try toHostF32(a, ly.hc_ffn_scale, 3, s),
            .sink = try toHostF32(a, ly.attn_sink, cfg.num_attention_heads, s),
            .gate_w = try toHostF32(a, ly.gate_w, @as(usize, cfg.num_experts) * dim, s),
            .gate_bias = if (ly.gate_bias) |gb| try toHostF32(a, gb, cfg.num_experts, s) else null,
            .tid2eid = if (ly.tid2eid) |t| try toHostI64(a, t, @as(usize, cfg.vocab_size) * cfg.num_experts_per_tok, s) else null,
            .comp = if (ly.comp) |*c| try hostComp(a, c, dim, cfg.head_dim, ratio, s) else null,
            .idx_wp = if (ly.idx) |*ix| try toHostF32(a, ix.weights_proj, @as(usize, cfg.dsv4_index_n_heads) * dim, s) else null,
            .idx_comp = if (ly.idx) |*ix| try hostComp(a, &ix.comp, dim, cfg.dsv4_index_head_dim, 4, s) else null,
        };
        // combined compressor-input operand (needs the transposed handles above)
        if (h.comp) |*c| {
            const parts = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(parts);
            _ = mlx.mlx_vector_array_append_value(parts, c.wkv_t);
            _ = mlx.mlx_vector_array_append_value(parts, c.wgate_t);
            var w2: usize = 2 * c.coff * c.head_dim;
            if (h.idx_comp) |*ic| {
                _ = mlx.mlx_vector_array_append_value(parts, ic.wkv_t);
                _ = mlx.mlx_vector_array_append_value(parts, ic.wgate_t);
                w2 += 2 * ic.coff * ic.head_dim;
            }
            var joined = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&joined, parts, 1, s));
            try mlx.check(mlx.mlx_array_eval(joined));
            h.comp_in_t = joined;
            h.comp_in_w = w2;
        }
        _ = li;
    }
    // wo_a: per-group slabs for the grouped low-rank O. Default = quantized
    // VIEWS [og, ol, ·] read in place by batched quantized_matmul (the bf16
    // wo_a_deq slabs were 2.9 GB of the ~7.8 GB read per serial token);
    // `MLX_SERVE_DSV4_WO_QMM=0` restores the dequantized [og, gin, ol]
    // operands (dense matmul reads strided views fine).
    const og: c_int = @intCast(cfg.dsv4_o_groups);
    const ol: c_int = @intCast(cfg.dsv4_o_lora_rank);
    const gin: c_int = @intCast(cfg.num_attention_heads * cfg.head_dim / cfg.dsv4_o_groups);
    var wo_a_deq: []mlx.mlx_array = &.{};
    var wo_a_q3: []Q3 = &.{};
    // GPU streams only: the CPU stream is the host-reference/test path, whose
    // numerics the python oracle models with dequantized operands (and whose
    // decode-equivalence gate is strict BECAUSE both sides hit one gemm).
    if (woAQmmEnabled() and mlx.streamIsGpu(s)) {
        wo_a_q3 = try a.alloc(Q3, dw.layers.len);
        for (wo_a_q3, dw.layers) |*q3, *ly| {
            q3.* = .{
                .w = try reshapeQ3(ly.wo_a.w, og, ol, s),
                .s = try reshapeQ3(ly.wo_a.s, og, ol, s),
                .b = try reshapeQ3(ly.wo_a.b, og, ol, s),
            };
        }
    } else {
        wo_a_deq = try a.alloc(mlx.mlx_array, dw.layers.len);
        for (wo_a_deq, dw.layers) |*hnd, *ly| {
            var dq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dq);
            const emp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(emp);
            try mlx.check(mlx.mlx_dequantize(&dq, ly.wo_a.w, ly.wo_a.s, ly.wo_a.b, mlx.mlx_optional_int.some(@intCast(ly.wo_a.qp.group_size)), mlx.mlx_optional_int.some(@intCast(ly.wo_a.qp.bits)), "affine", emp, mlx.mlx_optional_dtype{}, s));
            var rs2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rs2);
            const shp = [_]c_int{ og, ol, gin };
            try mlx.check(mlx.mlx_reshape(&rs2, dq, &shp, 3, s));
            var tr = mlx.mlx_array_new();
            const axes = [_]c_int{ 0, 2, 1 };
            try mlx.check(mlx.mlx_transpose_axes(&tr, rs2, &axes, 3, s));
            try mlx.check(mlx.mlx_array_eval(tr));
            hnd.* = tr;
        }
    }

    // comp_in int8-g32 side copies (decode/verify widths), gated on the
    // user-facing --decode-attn-quant flag in its EXPLICIT form: the
    // 2026-08-01 characterization moved one answer (a duplicated paragraph
    // on 1/7 greedy prompts) against +7-8% decode, so dsv4 requires the
    // user to actually ask — the flag's silent default stays dense here
    // while laguna keeps its clean-characterization default. Built EAGERLY
    // (evaled here) so the copies land inside the load-time budget, never
    // first-touch mid-request (the lazy-stage-weights class). GPU streams
    // only — the CPU stream is the host-reference/test path whose strict
    // gates assume the dense operand.
    var comp_in_q: []?CompInQ = &.{};
    if (transformer.decodeAttnQuantExplicit() and mlx.streamIsGpu(s)) {
        comp_in_q = try a.alloc(?CompInQ, dw.layers.len);
        var q_bytes: usize = 0;
        for (comp_in_q, hl) |*slot, *h| {
            slot.* = null;
            const cin = h.comp_in_t orelse continue;
            // eligibility is the weight's own shape/dtype (house rule).
            // comp_in_t is f32 BY CONSTRUCTION (transposedF32 upcasts the
            // bf16 checkpoint weights for the f32 decode stream) — f32 is
            // the expected dtype here, bf16/f16 accepted for robustness.
            const dt = mlx.mlx_array_dtype(cin);
            if (dt != .float32 and dt != .bfloat16 and dt != .float16) continue;
            if (dim % 32 != 0) continue;
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, cin, s)); // [W, dim] for transposed qmm
            var qv = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qv);
            const emp2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(emp2);
            try mlx.check(mlx.mlx_quantize(&qv, wt, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "affine", emp2, s));
            var qw = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(qw);
            try mlx.check(mlx.mlx_vector_array_get(&qw, qv, 0));
            var qs = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(qs);
            try mlx.check(mlx.mlx_vector_array_get(&qs, qv, 1));
            var qb = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(qb);
            try mlx.check(mlx.mlx_vector_array_get(&qb, qv, 2));
            try mlx.check(mlx.mlx_array_eval(qw));
            try mlx.check(mlx.mlx_array_eval(qs));
            try mlx.check(mlx.mlx_array_eval(qb));
            q_bytes += mlx.mlx_array_size(qw) * mlx.mlx_array_itemsize(qw) +
                mlx.mlx_array_size(qs) * mlx.mlx_array_itemsize(qs) +
                mlx.mlx_array_size(qb) * mlx.mlx_array_itemsize(qb);
            slot.* = .{ .w = qw, .s = qs, .b = qb };
        }
        if (q_bytes > 0)
            log.info("dsv4: comp_in int8 side copies built ({d} MB; --no-decode-attn-quant restores dense)\n", .{q_bytes / (1024 * 1024)});
    }

    var embed_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(embed_deq);
    const empty = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty);
    try mlx.check(mlx.mlx_dequantize(&embed_deq, dw.embed.w, dw.embed.s, dw.embed.b, mlx.mlx_optional_int.some(@intCast(dw.embed.qp.group_size)), mlx.mlx_optional_int.some(@intCast(dw.embed.qp.bits)), "affine", empty, mlx.mlx_optional_dtype{}, s));
    // lazy-decode GPU embed table (bf16; host embed_f32 = f32 of the same
    // bf16 values, so the two lookup paths feed bit-identical rows)
    var embed_g: ?mlx.mlx_array = null;
    if (mlx.streamIsGpu(s) and lazyDecodeEnabled()) {
        var eb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&eb, embed_deq, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(eb));
        embed_g = eb;
    }
    const ones_hd_g = blk: {
        var o = mlx.mlx_array_new();
        const shp = [_]c_int{@intCast(cfg.head_dim)};
        try mlx.check(mlx.mlx_ones(&o, &shp, 1, .float32, s));
        try mlx.check(mlx.mlx_array_eval(o));
        break :blk o;
    };
    const sink_consts = blk: {
        const vals = [_]f32{ @floatFromInt(hc * dim), cfg.rms_norm_eps, cfg.dsv4_hc_eps };
        const shp = [_]c_int{3};
        const arr = uploadF32(&vals, &shp);
        try mlx.check(mlx.mlx_array_eval(arr));
        break :blk arr;
    };
    const sink_env_off = if (std.c.getenv("MLX_SERVE_DSV4_SINKHORN")) |v| v[0] == '0' else false;
    const sink_k = if (mlx.streamIsGpu(s) and !sink_env_off)
        buildSinkhornKernel(hc, cfg.dsv4_hc_sinkhorn_iters)
    else
        null;
    const sink_y_env_off = if (std.c.getenv("MLX_SERVE_DSV4_SINKY")) |v| v[0] == '0' else false;
    const sink_y_k = if (sink_k != null and !sink_y_env_off)
        buildSinkhornYKernel(hc, cfg.dsv4_hc_sinkhorn_iters, dim)
    else
        null;
    const hc_post_env_off = if (std.c.getenv("MLX_SERVE_DSV4_HCPOST")) |v| v[0] == '0' else false;
    const hc_post_k = if (sink_y_k != null and !hc_post_env_off)
        buildHcPostKernel(hc, dim)
    else
        null;
    // DSpark stage weights sit OUTSIDE every warmup/serial forward — left
    // lazy they materialize MID-FIRST-DRAFT (~GBs of expert banks), after
    // every load-time memory budget (wired limit, preflight, auto-context)
    // was computed without them. On the real mirror that read as Metal
    // command buffers dying at the ceiling and MLX returning ZERO-filled
    // outputs (token-0 drafts verified by token-0 logits = fake 100%
    // acceptance). So DSpark is OPT-IN (`--dspark` sets
    // MLX_SERVE_DSV4_DSPARK=1; "force" additionally skips the fit gate):
    // opted in → collect the stage tensors, decide the fit BEFORE touching
    // them (logical bytes need no eval), and either pay the true footprint
    // here in one batched eval or disable with an honest log. Default OFF:
    // untouched lazy stages cost nothing and the model serves serial.
    const ds_env = std.c.getenv("MLX_SERVE_DSV4_DSPARK");
    const ds_opt_in = if (ds_env) |v| (v[0] == '1' or v[0] == 'f') else false;
    const ds_force = if (ds_env) |v| v[0] == 'f' else false;
    const has_stages = dw.dspark != null and dw.layers.len > n_layers;
    var dspark_on = ds_opt_in and has_stages;
    if (has_stages and !ds_opt_in) {
        log.info("dsv4: DSpark stages present but OFF by default (stages stay lazy at zero cost; pass --dspark to enable)\n", .{});
    }
    if (dspark_on) {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var stages = WeightTally{};
        for (dw.layers[n_layers..]) |*ly| stages = stages.add(appendWeightArrays(vec, ly.*));
        stages = stages.add(appendWeightArrays(vec, dw.dspark.?));
        // Trunk logical bytes, tally-only: embed/head/final norm/hc head +
        // the trunk layers (everything the warmup forward will materialize).
        var trunk = appendWeightArrays(null, dw.embed);
        trunk = trunk.add(appendWeightArrays(null, dw.head));
        trunk = trunk.add(appendWeightArrays(null, dw.final_norm));
        trunk = trunk.add(appendWeightArrays(null, dw.hc_head_fn));
        trunk = trunk.add(appendWeightArrays(null, dw.hc_head_base));
        trunk = trunk.add(appendWeightArrays(null, dw.hc_head_scale));
        for (dw.layers[0..n_layers]) |*ly| trunk = trunk.add(appendWeightArrays(null, ly.*));
        const max_rec = mlx.maxRecommendedWorkingSet();
        const mb = 1024 * 1024;
        if (ds_force or dsparkFitsBudget(stages.bytes, trunk.bytes, max_rec, DSPARK_MEM_HEADROOM)) {
            try mlx.check(mlx.mlx_eval(vec));
            log.info("dsv4: materialized {d} DSpark stage weight tensors ({d} MB) at load\n", .{ stages.n, stages.bytes / mb });
        } else {
            dspark_on = false;
            log.warn("dsv4: DSpark DISABLED — trunk {d} MB + stages {d} MB + headroom {d} MB exceed the {d} MB working-set budget; serving serial. MLX_SERVE_DSV4_DSPARK=force overrides\n", .{ trunk.bytes / mb, stages.bytes / mb, DSPARK_MEM_HEADROOM / mb, max_rec / mb });
        }
    }
    return .{
        .dw = dw,
        .hl = hl,
        .wo_a_deq = wo_a_deq,
        .wo_a_q3 = wo_a_q3,
        .comp_in_q = comp_in_q,
        .embed_g = embed_g,
        .ones_hd_g = ones_hd_g,
        .final_norm_g = try f32Handle(dw.final_norm, s),
        .hc_head_fn_t = try transposedF32(dw.hc_head_fn, s),
        .hada_g = if (cfg.dsv4_index_head_dim > 0) try buildHadamardF32(gpa, cfg.dsv4_index_head_dim, s) else null,
        .sink_k = sink_k,
        .sink_y_k = sink_y_k,
        .hc_post_k = hc_post_k,
        .sink_consts = sink_consts,
        .history = std.array_list.Managed(u32).init(gpa),
        .final_norm = try toHostF32(a, dw.final_norm, dim, s),
        .hc_head_fn = try toHostF32(a, dw.hc_head_fn, hc * hc * dim, s),
        .hc_head_base = try toHostF32(a, dw.hc_head_base, hc, s),
        .hc_head_scale = try toHostF32(a, dw.hc_head_scale, 1, s),
        .embed_f32 = try toHostF32(a, embed_deq, @as(usize, cfg.vocab_size) * dim, s),
        .arena = arena,
        .s = s,
        .dim = dim,
        .n_layers = n_layers,
        .n_heads = cfg.num_attention_heads,
        .head_dim = cfg.head_dim,
        .rd = cfg.dsv4_rope_head_dim,
        .window = cfg.sliding_window,
        .hc = hc,
        .hc_iters = cfg.dsv4_hc_sinkhorn_iters,
        .hc_eps = cfg.dsv4_hc_eps,
        .eps = cfg.rms_norm_eps,
        .o_groups = cfg.dsv4_o_groups,
        .o_lora = cfg.dsv4_o_lora_rank,
        .q_lora = cfg.dsv4_q_lora_rank,
        .topk = cfg.num_experts_per_tok,
        .n_experts = cfg.num_experts,
        .n_hash = cfg.dsv4_hash_layers,
        .route_scale = cfg.router_scaling_factor,
        .swiglu_limit = cfg.dsv4_swiglu_limit,
        .moe_inter = cfg.moe_intermediate_size,
        .idx_heads = cfg.dsv4_index_n_heads,
        .idx_hd = cfg.dsv4_index_head_dim,
        .idx_topk = cfg.dsv4_index_topk,
        .vocab = cfg.vocab_size,
        .ratios = cfg.dsv4_compress_ratios,
        .yarn_theta = cfg.dsv4_compress_rope_theta,
        .yarn_orig = cfg.yarn_orig_max_pos,
        .yarn_factor = cfg.yarn_factor,
        .yarn_bf = cfg.yarn_beta_fast,
        .yarn_bs = cfg.yarn_beta_slow,
        .plain_theta = cfg.rope_theta,
        .n_mtp = if (dspark_on) dw.layers.len - n_layers else 0,
        .ds_block = cfg.dsv4_dspark_block_size,
        .ds_noise = cfg.dsv4_dspark_noise_token_id,
        .ds_rank = cfg.dsv4_dspark_markov_rank,
        .ds_targets = cfg.dsv4_dspark_target_layers,
        .n_ds_targets = cfg.dsv4_n_dspark_target_layers,
        .ds_main_norm_g = if (dspark_on) try f32Handle(dw.dspark.?.main_norm, s) else null,
        .ds_last_norm_g = if (dspark_on) try f32Handle(dw.dspark.?.last_norm, s) else null,
        .ds_hc_head_fn_t = if (dspark_on) try transposedF32(dw.dspark.?.hc_head_fn, s) else null,
        .ds_markov_w2_t = if (dspark_on) try transposedF32(dw.dspark.?.markov_w2, s) else null,
        .ds_conf_proj_t = if (dspark_on) try transposedF32(dw.dspark.?.conf_proj, s) else null,
        .ds_hc_head_base = if (dspark_on) try toHostF32(a, dw.dspark.?.hc_head_base, hc, s) else &.{},
        .ds_hc_head_scale = if (dspark_on) try toHostF32(a, dw.dspark.?.hc_head_scale, 1, s) else &.{},
        .ds_conf_thr = dsparkConfThreshold(),
        .ds_prof = if (dspark_on and std.c.getenv("MLX_SERVE_DSPARK_PROFILE") != null) DsparkProfile{} else null,
    };
}

/// Confidence gate, in the reference implementation's own units: the env var
/// is a SIGMOID probability (antirez's `--dspark-confidence`), compared here
/// against the raw logit, so it converts once at load. 0 disables the gate
/// (every position submitted), 1 shuts it.
///
/// DEFAULT OFF — measured, not inherited. The head IS informative (position 0
/// scores +5.08 mean on accepted rounds vs +0.02 on rejected), but our
/// batched forward is FIXED-COST dominated: verify(C) ≈ 70 ms + ~12 ms·C, so
/// a narrower block pays almost the same and commits less. Live: no gate 30.7
/// tok/s, 0.5 → 29.9, 0.9 (the reference default) → 25.3. The reference's
/// engine has a far cheaper per-forward floor, which is what makes the same
/// threshold pay there. Port, measure, let the number pick the default.
fn dsparkConfThreshold() f32 {
    var p: f32 = 0;
    if (std.c.getenv("MLX_SERVE_DSV4_DSPARK_CONF")) |v| {
        p = std.fmt.parseFloat(f32, std.mem.span(v)) catch 0;
    }
    if (p <= 0) return -std.math.inf(f32);
    if (p >= 1) return std.math.inf(f32);
    return @log(p / (1 - p));
}

/// Gated-pooling compression (prefill): x [s, dim] -> [s/ratio, head_dim].
fn compressorForward(m: *const Dsv4Model, alloc: std.mem.Allocator, c: *const HostComp, x: []const f32, seq: usize, ratio: usize, rotate: bool, fr: *const Freqs, cs: ?*CompDecState) ![]f32 {
    const d = c.head_dim;
    const coff = c.coff;
    const cd = coff * d;
    const nb = seq / ratio;
    const cutoff = nb * ratio;
    // Rows are computed for ALL seq positions (not just cutoff) so the
    // decode-state capture can seed the pending-window rings with the
    // remainder + last-full-window raw rows (reference prefill lines 331-336).
    const kv = try alloc.alloc(f32, seq * cd);
    defer alloc.free(kv);
    const score = try alloc.alloc(f32, seq * cd);
    defer alloc.free(score);
    {
        const kv_g = try matMlx(alloc, x[0 .. seq * m.dim], c.wkv_t, seq, m.dim, cd, m.s);
        defer alloc.free(kv_g);
        @memcpy(kv, kv_g);
        const sc_g = try matMlx(alloc, x[0 .. seq * m.dim], c.wgate_t, seq, m.dim, cd, m.s);
        defer alloc.free(sc_g);
        @memcpy(score, sc_g);
    }
    // + ape per within-block position
    for (0..seq) |t| {
        const r = t % ratio;
        for (0..cd) |j| score[t * cd + j] += c.ape[r * cd + j];
    }
    if (cs) |state| {
        const overlap = coff == 2;
        if (overlap and cutoff >= ratio) {
            for (0..ratio) |r| {
                @memcpy(state.kv_pend[r * state.width ..][0..cd], kv[(cutoff - ratio + r) * cd ..][0..cd]);
                @memcpy(state.sc_pend[r * state.width ..][0..cd], score[(cutoff - ratio + r) * cd ..][0..cd]);
            }
        }
        const off: usize = if (overlap) ratio else 0;
        for (cutoff..seq) |t| {
            const slot = off + (t - cutoff);
            @memcpy(state.kv_pend[slot * state.width ..][0..cd], kv[t * cd ..][0..cd]);
            @memcpy(state.sc_pend[slot * state.width ..][0..cd], score[t * cd ..][0..cd]);
        }
    }
    if (nb == 0) return try alloc.alloc(f32, 0);
    const out = try alloc.alloc(f32, nb * d);
    const win = try alloc.alloc(f32, 2 * ratio * d); // overlap window staging
    defer alloc.free(win);
    const wsc = try alloc.alloc(f32, 2 * ratio * d);
    defer alloc.free(wsc);
    for (0..nb) |b| {
        var rows: usize = ratio;
        if (coff == 2) {
            rows = 2 * ratio;
            // first half rows: PREVIOUS block's first-half dims; -inf score on block 0
            for (0..ratio) |r| {
                for (0..d) |j| {
                    if (b == 0) {
                        win[r * d + j] = 0;
                        wsc[r * d + j] = -std.math.inf(f32);
                    } else {
                        win[r * d + j] = kv[((b - 1) * ratio + r) * cd + j];
                        wsc[r * d + j] = score[((b - 1) * ratio + r) * cd + j];
                    }
                }
            }
            for (0..ratio) |r| {
                for (0..d) |j| {
                    win[(ratio + r) * d + j] = kv[(b * ratio + r) * cd + d + j];
                    wsc[(ratio + r) * d + j] = score[(b * ratio + r) * cd + d + j];
                }
            }
        } else {
            for (0..ratio) |r| {
                for (0..d) |j| {
                    win[r * d + j] = kv[(b * ratio + r) * cd + j];
                    wsc[r * d + j] = score[(b * ratio + r) * cd + j];
                }
            }
        }
        // softmax over the window dim, per feature j
        const ob = out[b * d ..][0..d];
        for (0..d) |j| {
            var mx_: f32 = -std.math.inf(f32);
            for (0..rows) |r| mx_ = @max(mx_, wsc[r * d + j]);
            var sum: f64 = 0;
            var acc: f64 = 0;
            for (0..rows) |r| {
                const e = @exp(@as(f64, wsc[r * d + j] - mx_));
                sum += e;
                acc += e * win[r * d + j];
            }
            ob[j] = @floatCast(acc / sum);
        }
        rmsNormRow(ob, c.norm, m.eps);
        ropeRow(ob[d - m.rd ..], fr, b * ratio, false);
        if (rotate) {
            hadamardInPlace(ob);
            fp4SimInPlace(ob, 32);
        } else {
            fp8SimInPlace(ob[0 .. d - m.rd], 64);
        }
    }
    if (cs) |state| try state.cache.appendSlice(out);
    return out;
}

/// Indexer: top-k compressed-slot GLOBAL indices (offset +seq), -1 = invalid.
fn indexerForward(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x: []const f32, qr: []const f32, seq: usize, fr: *const Freqs, cs: ?*CompDecState) ![]i64 {
    const ratio: usize = 4;
    const n_slots = seq / ratio;
    const k = @min(m.idx_topk, n_slots);
    const out = try alloc.alloc(i64, seq * k);
    if (n_slots == 0) return out;
    const ih = m.idx_heads;
    const ihd = m.idx_hd;
    const ix = &m.dw.layers[li].idx.?;
    const q = try qmmHost(alloc, &ix.wq_b, qr, seq, m.q_lora, ih * ihd, m.s);
    defer alloc.free(q);
    for (0..seq) |t| {
        for (0..ih) |h| {
            const row = q[(t * ih + h) * ihd ..][0..ihd];
            ropeRow(row[ihd - m.rd ..], fr, t, false);
            hadamardInPlace(row);
            fp4SimInPlace(row, 32);
        }
    }
    const ck = try compressorForward(m, alloc, &m.hl[li].idx_comp.?, x, seq, ratio, true, fr, cs);
    defer alloc.free(ck);
    const wts = try alloc.alloc(f32, seq * ih);
    defer alloc.free(wts);
    matHost(wts, x, m.hl[li].idx_wp.?, seq, m.dim, ih);
    const wscale: f32 = @floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(ihd))) * @sqrt(@as(f64, @floatFromInt(ih)))));
    const scores = try alloc.alloc(f32, n_slots);
    defer alloc.free(scores);
    const Cand = struct { score: f32, slot: usize };
    const cands = try alloc.alloc(Cand, n_slots);
    defer alloc.free(cands);
    for (0..seq) |t| {
        const visible = (t + 1) / ratio;
        @memset(scores, 0);
        for (0..ih) |h| {
            const wv = wts[t * ih + h] * wscale;
            const qrow = q[(t * ih + h) * ihd ..][0..ihd];
            for (0..n_slots) |sl| {
                var dot: f64 = 0;
                const kr = ck[sl * ihd ..][0..ihd];
                for (qrow, kr) |a2, b2| dot += @as(f64, a2) * b2;
                if (dot > 0) scores[sl] += @as(f32, @floatCast(dot)) * wv;
            }
        }
        for (cands, 0..) |*cd2, sl| cd2.* = .{ .score = if (sl < visible) scores[sl] else -std.math.inf(f32), .slot = sl };
        std.mem.sort(Cand, cands, {}, struct {
            fn lt(_: void, a2: Cand, b2: Cand) bool {
                return a2.score > b2.score;
            }
        }.lt);
        for (0..k) |i| {
            out[t * k + i] = if (cands[i].slot < visible) @intCast(seq + cands[i].slot) else -1;
        }
    }
    return out;
}

/// Full attention sublayer on host: x [s, dim] (post attn_norm) -> [s, dim].
fn attentionForward(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x: []const f32, seq: usize, st_layer: ?*LayerDecState) ![]f32 {
    const ly = &m.dw.layers[li];
    const h = &m.hl[li];
    const ratio: usize = ly.compress_ratio;
    const hd = m.head_dim;
    const nh = m.n_heads;
    const rd = m.rd;
    const fr = blk: {
        if (ratio != 0) {
            break :blk try precomputeFreqs(alloc, rd, seq + 1, m.yarn_orig, m.yarn_theta, m.yarn_factor, m.yarn_bf, m.yarn_bs);
        }
        break :blk try precomputeFreqs(alloc, rd, seq + 1, 0, m.plain_theta, 1, 32, 1);
    };
    defer alloc.free(fr.cos);
    defer alloc.free(fr.sin);

    // q: wq_a -> q_norm -> wq_b -> per-head unweighted RMS -> rope
    const qr = try qmmHost(alloc, &ly.wq_a, x, seq, m.dim, m.q_lora, m.s);
    defer alloc.free(qr);
    for (0..seq) |t| rmsNormRow(qr[t * m.q_lora ..][0..m.q_lora], h.q_norm, m.eps);
    const q = try qmmHost(alloc, &ly.wq_b, qr, seq, m.q_lora, nh * hd, m.s);
    defer alloc.free(q);
    for (0..seq) |t| {
        for (0..nh) |hh| {
            const row = q[(t * nh + hh) * hd ..][0..hd];
            rmsNormRow(row, null, m.eps);
            ropeRow(row[hd - rd ..], &fr, t, false);
        }
    }
    // kv latent: wkv -> kv_norm -> rope -> fp8 sim
    const kv = try qmmHost(alloc, &ly.wkv, x, seq, m.dim, hd, m.s);
    defer alloc.free(kv);
    for (0..seq) |t| {
        const row = kv[t * hd ..][0..hd];
        rmsNormRow(row, h.kv_norm, m.eps);
        ropeRow(row[hd - rd ..], &fr, t, false);
        fp8SimInPlace(row[0 .. hd - rd], 64);
    }
    // compressed history + index sets
    var comp: []f32 = &.{};
    defer if (comp.len > 0) alloc.free(comp);
    var cidx: []i64 = &.{};
    defer if (cidx.len > 0) alloc.free(cidx);
    var ck: usize = 0;
    if (st_layer) |sl| try sl.kv.appendSlice(kv);
    if (ratio != 0) {
        comp = try compressorForward(m, alloc, &h.comp.?, x, seq, ratio, false, &fr, if (st_layer) |sl| &sl.comp.? else null);
        const n_slots = comp.len / hd;
        if (n_slots > 0) {
            if (ratio == 4) {
                const qr_normed = qr; // already q_norm'd in place
                cidx = try indexerForward(m, alloc, li, x, qr_normed, seq, &fr, if (st_layer) |sl| &sl.idx_comp.? else null);
                ck = cidx.len / seq;
            } else {
                ck = n_slots;
                cidx = try alloc.alloc(i64, seq * ck);
                for (0..seq) |t| {
                    const visible = (t + 1) / ratio;
                    for (0..ck) |sl| cidx[t * ck + sl] = if (sl < visible) @intCast(seq + sl) else -1;
                }
            }
        }
    }
    const wk = @min(seq, m.window);
    const tk = wk + ck;
    // gathered sink-softmax attention
    const o = try alloc.alloc(f32, seq * nh * hd);
    defer alloc.free(o);
    const idxs = try alloc.alloc(i64, tk);
    defer alloc.free(idxs);
    const sc = try alloc.alloc(f32, tk);
    defer alloc.free(sc);
    for (0..seq) |t| {
        // window: last `window` raw positions (clamped base + offset; future -> -1)
        for (0..wk) |i| {
            var pos = @max(@as(i64, @intCast(t)) - @as(i64, @intCast(m.window)) + 1, 0) + @as(i64, @intCast(i));
            if (pos > @as(i64, @intCast(t))) pos = -1;
            idxs[i] = pos;
        }
        for (0..ck) |i| idxs[wk + i] = cidx[t * ck + i];
        for (0..nh) |hh| {
            const qrow = q[(t * nh + hh) * hd ..][0..hd];
            var mx_: f32 = h.sink[hh];
            for (0..tk) |i| {
                if (idxs[i] < 0) {
                    sc[i] = -std.math.inf(f32);
                    continue;
                }
                const gi: usize = @intCast(idxs[i]);
                const krow = if (gi < seq) kv[gi * hd ..][0..hd] else comp[(gi - seq) * hd ..][0..hd];
                var dot: f64 = 0;
                for (qrow, krow) |a2, b2| dot += @as(f64, a2) * b2;
                sc[i] = @floatCast(dot / @sqrt(@as(f64, @floatFromInt(hd))));
                mx_ = @max(mx_, sc[i]);
            }
            var denom: f64 = @exp(@as(f64, h.sink[hh] - mx_));
            const orow = o[(t * nh + hh) * hd ..][0..hd];
            @memset(orow, 0);
            for (0..tk) |i| {
                if (idxs[i] < 0) continue;
                const e = @exp(@as(f64, sc[i] - mx_));
                denom += e;
                const gi: usize = @intCast(idxs[i]);
                const vrow = if (gi < seq) kv[gi * hd ..][0..hd] else comp[(gi - seq) * hd ..][0..hd];
                for (orow, vrow) |*ov, vv| ov.* += @floatCast(e * vv);
            }
            for (orow) |*ov| ov.* = @floatCast(@as(f64, ov.*) / denom);
            ropeRow(orow[hd - rd ..], &fr, t, true);
        }
    }
    // grouped low-rank O: wo_a [og*ol, (nh/og)*hd] block-diagonal einsum
    const ored = try woAApply(m, alloc, li, o, seq);
    defer alloc.free(ored);
    return try qmmHost(alloc, &ly.wo_b, ored, seq, m.o_groups * m.o_lora, m.dim, m.s);
}


/// Grouped low-rank O via the cached [og, gin, ol] operands: o comes in
/// token-major [s, og, gin]; returns [s, og*ol] host f32.
fn woAApply(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, o: []const f32, s_len: usize) ![]f32 {
    const og = m.o_groups;
    const ol = m.o_lora;
    const gin = m.n_heads * m.head_dim / og;
    // reorder to group-major [og, s, gin] for the batched matmul
    const gm = try alloc.alloc(f32, og * s_len * gin);
    defer alloc.free(gm);
    for (0..s_len) |t| {
        for (0..og) |g| {
            @memcpy(gm[(g * s_len + t) * gin ..][0..gin], o[(t * og + g) * gin ..][0..gin]);
        }
    }
    const ob = try hostToBf16(gm, &.{ @intCast(og), @intCast(s_len), @intCast(gin) }, m.s);
    defer _ = mlx.mlx_array_free(ob);
    const res = try woAMatmul(m, li, ob);
    defer _ = mlx.mlx_array_free(res);
    const rh = try toHostF32(alloc, res, og * s_len * ol, m.s);
    defer alloc.free(rh);
    const out = try alloc.alloc(f32, s_len * og * ol);
    for (0..s_len) |t| {
        for (0..og) |g| {
            @memcpy(out[(t * og + g) * ol ..][0..ol], rh[(g * s_len + t) * ol ..][0..ol]);
        }
    }
    return out;
}

fn sqrtSoftplus(x: f64) f64 {
    const sp = @max(x, 0) + std.math.log1p(@exp(-@abs(x)));
    return @sqrt(sp);
}

/// GPU MoE routing kill switch (`MLX_SERVE_DSV4_MOE_ROUTE_GPU=0` → the host
/// routing sync via routeToken). Cached: read once per process.
var moe_route_gpu_state: ?bool = null;
var moe_route_gpu_logged: bool = false;
fn moeRouteGpuEnabled() bool {
    if (moe_route_gpu_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_MOE_ROUTE_GPU")) |e| e[0] != '0' else true;
    moe_route_gpu_state = v;
    return v;
}

/// Deferred compressor-row sync kill switch
/// (`MLX_SERVE_DSV4_COMP_DEFER=0` → every position syncs in-layer, the
/// pre-deferral behavior). Cached: read once per process.
var comp_defer_state: ?bool = null;
fn compDeferEnabled() bool {
    if (comp_defer_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_COMP_DEFER")) |e| e[0] != '0' else true;
    comp_defer_state = v;
    return v;
}

/// wo_a served quantized (`MLX_SERVE_DSV4_WO_QMM=0` → the bf16 wo_a_deq
/// slabs, the pre-quantized behavior). LOAD-time decision: the enabled path
/// never builds the 2.9 GB of dequantized operands. Cached once per process.
var wo_qmm_state: ?bool = null;
var wo_qmm_logged: bool = false;
fn woAQmmEnabled() bool {
    if (wo_qmm_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_WO_QMM")) |e| e[0] != '0' else true;
    wo_qmm_state = v;
    return v;
}

/// Reshape one member of a wo_a quantized triple [og*ol, X] → [og, ol, X]
/// (contiguous row-major → pure view, no copy).
fn reshapeQ3(arr: mlx.mlx_array, og: c_int, ol: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const x: c_int = mlx.mlx_array_shape(arr)[1];
    var out = mlx.mlx_array_new();
    const shp = [_]c_int{ og, ol, x };
    try mlx.check(mlx.mlx_reshape(&out, arr, &shp, 3, s));
    return out;
}

/// Grouped low-rank O tail shared by decode/batch/DSpark-stage attention and
/// the host reference: [og, M, gin] bf16 @ wo_a → [og, M, ol]. Default reads
/// the checkpoint's quantized wo_a in place (batched quantized_matmul over
/// the og slabs); the kill switch restores the dequantized bf16 operands.
fn woAMatmul(m: *const Dsv4Model, li: usize, ob: mlx.mlx_array) !mlx.mlx_array {
    if (m.wo_a_q3.len > 0) {
        if (!wo_qmm_logged) {
            wo_qmm_logged = true;
            log.info("dsv4: wo_a batched-qmm path engaged\n", .{});
        }
        const q3 = &m.wo_a_q3[li];
        const qp = &m.dw.layers[li].wo_a.qp;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_quantized_matmul(&out, ob, q3.w, q3.s, q3.b, true, mlx.mlx_optional_int.some(@intCast(qp.group_size)), mlx.mlx_optional_int.some(@intCast(qp.bits)), "affine", m.s));
        return out;
    }
    return try gpuOp2(mlx.mlx_matmul, ob, m.wo_a_deq[li], m.s);
}

/// GPU window emission kill switch (`MLX_SERVE_DSV4_GPU_EMIT=0` → host
/// emission with the in-layer blocking sync, the pre-GPU behavior). Cached.
var gpu_emit_state: ?bool = null;
var gpu_emit_logged: bool = false;
fn gpuEmitEnabled() bool {
    if (gpu_emit_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_GPU_EMIT")) |e| e[0] != '0' else true;
    gpu_emit_state = v;
    return v;
}

/// GPU emission serves GPU streams only: the CPU stream is the host
/// reference/test path whose strict gates assume host emission.
fn gpuEmitActive(m: *const Dsv4Model) bool {
    return gpuEmitEnabled() and mlx.streamIsGpu(m.s);
}

/// Fused window-emission kernel kill switch (`MLX_SERVE_DSV4_EMIT_KERNEL=0`
/// → the composed ~60-op emission graph, the pre-kernel behavior). Cached.
var emit_kernel_state: ?bool = null;
var emit_kernel_logged: bool = false;
/// Engagement counter (tests assert the kernel actually ran — a silent
/// decline is output-identical to the fallback, house rule).
var emit_kernel_hits: usize = 0;
fn emitKernelEnabled() bool {
    if (emit_kernel_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_EMIT_KERNEL")) |e| e[0] != '0' else true;
    emit_kernel_state = v;
    return v;
}
/// Test hook: force the fused-emission arm on/off (null = re-read the env).
fn emitKernelSetForTest(v: ?bool) void {
    emit_kernel_state = v;
}

/// Fused decode-chain kernel kill switch (`MLX_SERVE_DSV4_DEC_CHAIN=0` → the
/// composed per-head RMS/rope/sim op chains, the pre-kernel behavior). Cached.
var dec_chain_state: ?bool = null;
var dec_chain_logged: bool = false;
var dec_chain_hits: usize = 0;
fn decChainEnabled() bool {
    if (dec_chain_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_DEC_CHAIN")) |e| e[0] != '0' else true;
    dec_chain_state = v;
    return v;
}
/// Test hook: force the decode-chain arm on/off (null = re-read the env).
fn decChainSetForTest(v: ?bool) void {
    dec_chain_state = v;
}

/// Fused MoE gate+up gather — OPT-IN (`MLX_SERVE_DSV4_MOE_GATEUP=1`): the
/// same-boot A/B on the real 2-bit gs64 trunk measured it ~2.5% SLOWER than
/// the two stock gather_qmm dispatches + clippedSwigluG (29.6 → 28.9 tok/s,
/// 2026-08-01) — the house "stock gather wins where O(bank) doesn't dominate
/// the expert math" outcome, re-measured here because MLX's 2/3-bit kernels
/// are its slowest. Kept behind the switch for future geometries. Cached.
var moe_gateup_state: ?bool = null;
var moe_gateup_logged: bool = false;
var moe_gateup_hits: usize = 0;
fn moeGateUpEnabled() bool {
    if (moe_gateup_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_MOE_GATEUP")) |e| e[0] == '1' else false;
    moe_gateup_state = v;
    return v;
}
/// Test hook: force the MoE gate+up arm on/off (null = re-read the env).
fn moeGateUpSetForTest(v: ?bool) void {
    moe_gateup_state = v;
}

/// Fused sink-softmax — OPT-IN (`MLX_SERVE_DSV4_SINK_SOFTMAX=1`): the
/// same-boot A/B measured it NEUTRAL-to-slightly-negative (~29.6 vs ~29.8
/// tok/s composed, 2026-08-01) — the composed 4-dispatch chain is already
/// overlapped by the GPU (the fusedAttnGate on-chain-but-overlapped class),
/// and the kernel is not bit-identical (softmax reduction tree). A
/// byte-changing lever with no measured win stays off. Cached.
var sink_softmax_state: ?bool = null;
var sink_softmax_logged: bool = false;
var sink_softmax_hits: usize = 0;
fn sinkSoftmaxEnabled() bool {
    if (sink_softmax_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_SINK_SOFTMAX")) |e| e[0] == '1' else false;
    sink_softmax_state = v;
    return v;
}
/// Test hook: force the sink-softmax arm on/off (null = re-read the env).
fn sinkSoftmaxSetForTest(v: ?bool) void {
    sink_softmax_state = v;
}

/// comp_in decode-side int8 requant engagement counter — the serving arm is
/// gated by the user-facing `--decode-attn-quant` config flag
/// (`transformer.decodeAttnQuantEnabled`), same contract as the laguna
/// attention side copies: decode/verify widths only, prefill keeps dense.
var comp_in_q_hits: usize = 0;
var comp_in_q_logged: bool = false;

/// comp_in projection: the int8-g32 side copy at decode/verify widths
/// (C ≤ 32 — serial decode, DSpark verify blocks, and tiny suffix prefills,
/// which must all see the SAME weights or spec-on/off diverges at temp 0,
/// the laguna rule), dense bf16 otherwise. The copies exist only when
/// --decode-attn-quant was on at load, so this is a null-check per call.
fn compInProj(m: *const Dsv4Model, h: *const HostLayer, li: usize, x_g: mlx.mlx_array, C: usize) !mlx.mlx_array {
    if (C <= 32 and m.comp_in_q.len > li) {
        if (m.comp_in_q[li]) |q| {
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_quantized_matmul(&out, x_g, q.w, q.s, q.b, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "affine", m.s));
            comp_in_q_hits += 1;
            if (!comp_in_q_logged) {
                comp_in_q_logged = true;
                log.info("dsv4: comp_in decode requant engaged (int8 g32, decode/verify widths; --no-decode-attn-quant restores dense)\n", .{});
            }
            return out;
        }
    }
    return try gpuOp2(mlx.mlx_matmul, x_g, h.comp_in_t.?, m.s);
}

/// Lazy pipelined decode kill switch (`MLX_SERVE_DSV4_LAZY_DECODE=0` → the
/// synchronous decodeStep with per-token host logits). Cached.
var lazy_decode_state: ?bool = null;
var lazy_decode_logged: bool = false;
fn lazyDecodeEnabled() bool {
    if (lazy_decode_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_LAZY_DECODE")) |e| e[0] != '0' else true;
    lazy_decode_state = v;
    return v;
}

/// Per-token/per-chunk phase trace (`MLX_SERVE_DSV4_TRACE=1`): decode logs
/// pos, build/head/defer/comp µs + the gap since the previous step returned;
/// extendChunk logs C, layers/comp/defer/head ms. Wall-clock only, no extra
/// evals — everything is lazy until the head sync, so "build" is honest CPU
/// graph-construction time (plus any in-layer compressor barrier, which is
/// exactly what the mod-4 pattern is meant to expose).
var dsv4_trace_state: ?bool = null;
fn dsv4TraceEnabled() bool {
    if (dsv4_trace_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_TRACE")) |e| e[0] == '1' else false;
    dsv4_trace_state = v;
    return v;
}
/// Compressor barrier time (sync + host pushes) accumulated inside the layer
/// loop; single inference thread, reset by the caller per token/chunk.
var trace_comp_ns: u64 = 0;
/// End timestamp of the previous decodeStep (gap = host sampling/dispatch
/// time between steps, GPU idle).
var trace_last_end: ?std.Io.Timestamp = null;

const RouteG = struct {
    ind: mlx.mlx_array, // [C, k] int32 expert indices
    w: mlx.mlx_array, // [C, k, 1, 1] f32 normalized route weights × route_scale
    pub fn deinit(self: *RouteG) void {
        _ = mlx.mlx_array_free(self.ind);
        _ = mlx.mlx_array_free(self.w);
    }
};

/// GPU MoE routing (no host sync): `scores_g` is the RAW gate matmul output
/// [C, E] f32 (pre-softplus). sp = sqrt(logaddexp(scores, 0)) — the same
/// numerically-stable form as the host `sqrtSoftplus`, in f32. Selection:
/// hash layers (`tid2eid != null`) look up the token-id row on host (pure
/// table, no score read — an UPLOAD, not a sync); scored layers take the
/// ascending-argpartition TAIL of sp + gate.bias (bias joins SELECTION only,
/// the routeToken rule). Weights: take_along(sp) at the selected experts,
/// normalized by their sum × route_scale (host routeToken accumulates the
/// k=6 sum in f64; the f32 device sum is the sanctioned last-ulp class —
/// pinned within tolerance by the unit test below).
fn routeGpu(
    s: mlx.mlx_stream,
    alloc: std.mem.Allocator,
    scores_g: mlx.mlx_array,
    gate_bias: ?mlx.mlx_array,
    tid2eid: ?[]const i64,
    tid2eid_g: ?mlx.mlx_array,
    id_arr: ?mlx.mlx_array,
    ids: []const u32,
    E: usize,
    k: usize,
    route_scale: f32,
) !RouteG {
    const seq = ids.len;
    const ind_shape = [_]c_int{ @intCast(seq), @intCast(k) };
    const wshape = [_]c_int{ @intCast(seq), @intCast(k), 1, 1 };
    var ind = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ind);
    var w_arr = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(w_arr);

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    const sp_raw = try gpuOp2(mlx.mlx_logaddexp, scores_g, zero, s);
    defer _ = mlx.mlx_array_free(sp_raw);
    var sp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp);
    try mlx.check(mlx.mlx_sqrt(&sp, sp_raw, s));

    if (tid2eid != null and id_arr != null) {
        // lazy single-token hash lookup: tid2eid row via take on device
        std.debug.assert(seq == 1);
        const t2 = tid2eid_g.?;
        const t2_shape = [_]c_int{ @intCast(mlx.mlx_array_size(t2) / k), @intCast(k) };
        var t2v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t2v);
        try mlx.check(mlx.mlx_reshape(&t2v, t2, &t2_shape, 2, s));
        const fshape = [_]c_int{1};
        var idf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idf);
        try mlx.check(mlx.mlx_reshape(&idf, id_arr.?, &fshape, 1, s));
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_take_axis(&row, t2v, idf, 0, s)); // [1, k]
        try mlx.check(mlx.mlx_astype(&ind, row, .int32, s));
    } else if (tid2eid) |tid| {
        const hi = try alloc.alloc(i32, seq * k);
        defer alloc.free(hi);
        for (ids, 0..) |id, t| {
            for (0..k) |i| hi[t * k + i] = @intCast(tid[@as(usize, id) * k + i]);
        }
        const up = mlx.mlx_array_new_data(hi.ptr, &ind_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_array_set(&ind, up));
    } else {
        const sel = try gpuOp2(mlx.mlx_add, sp, gate_bias.?, s);
        defer _ = mlx.mlx_array_free(sel);
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_argpartition_axis(&part, sel, @intCast(E - k), 1, s));
        var tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail);
        const t_start = [_]c_int{ 0, @intCast(E - k) };
        const t_stop = [_]c_int{ @intCast(seq), @intCast(E) };
        const t_str = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&tail, part, &t_start, 2, &t_stop, 2, &t_str, 2, s));
        try mlx.check(mlx.mlx_astype(&ind, tail, .int32, s));
    }

    var w_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_sel);
    try mlx.check(mlx.mlx_take_along_axis(&w_sel, sp, ind, 1, s)); // [C, k]
    var wsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsum);
    try mlx.check(mlx.mlx_sum_axis(&wsum, w_sel, 1, true, s)); // [C, 1]
    const wn = try gpuOp2(mlx.mlx_divide, w_sel, wsum, s);
    defer _ = mlx.mlx_array_free(wn);
    const rs = mlx.mlx_array_new_float(route_scale);
    defer _ = mlx.mlx_array_free(rs);
    const ws = try gpuOp2(mlx.mlx_multiply, wn, rs, s);
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_reshape(&w_arr, ws, &wshape, 4, s));

    return .{ .ind = ind, .w = w_arr };
}

/// Expert-bank gather_qmm (shared by prefill and decode paths).
fn gatherQmmE(q: *const Q, xin: mlx.mlx_array, indices_: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const empty = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_gather_qmm(&out, xin, q.w, q.s, q.b, empty, indices_, true, mlx.mlx_optional_int.some(@intCast(q.qp.group_size)), mlx.mlx_optional_int.some(@intCast(q.qp.bits)), "affine", false, s));
    return out;
}

/// gatherQmmE with PRE-SORTED expert indices (x already gathered into slot
/// order): the sorted hint lets consecutive slots stream one expert bank.
fn gatherQmmESorted(q: *const Q, xin: mlx.mlx_array, indices_: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const empty = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_gather_qmm(&out, xin, q.w, q.s, q.b, empty, indices_, true, mlx.mlx_optional_int.some(@intCast(q.qp.group_size)), mlx.mlx_optional_int.some(@intCast(q.qp.bits)), "affine", true, s));
    return out;
}

/// Clipped SwiGLU in f32 (reference computes it in float32), back to bf16.
fn clippedSwigluG(gate_in: mlx.mlx_array, up_in: mlx.mlx_array, limit: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const lim = mlx.mlx_array_new_float(limit);
    defer _ = mlx.mlx_array_free(lim);
    const neg_lim = mlx.mlx_array_new_float(-limit);
    defer _ = mlx.mlx_array_free(neg_lim);
    var gate32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate32);
    try mlx.check(mlx.mlx_astype(&gate32, gate_in, .float32, s));
    var up32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up32);
    try mlx.check(mlx.mlx_astype(&up32, up_in, .float32, s));
    var g_cl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_cl);
    try mlx.check(mlx.mlx_minimum(&g_cl, gate32, lim, s));
    var u_lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u_lo);
    try mlx.check(mlx.mlx_maximum(&u_lo, up32, neg_lim, s));
    var u_cl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u_cl);
    try mlx.check(mlx.mlx_minimum(&u_cl, u_lo, lim, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g_cl, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g_cl, sig, s));
    var act32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act32);
    try mlx.check(mlx.mlx_multiply(&act32, silu, u_cl, s));
    var act = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&act, act32, .bfloat16, s));
    return act;
}

/// The transformer.zig `gatherQmvGateUp` body with dsv4's clipped SwiGLU
/// tail: accumulators round to T exactly where the composed gather_qmm pair
/// wrote T(acc), then the f32 clip chain with the sigmoid read from an exact
/// bf16-indexed table of mlx_sigmoid values (the swigluSigTable rule — a
/// metal::exp transcendental is a rounding apart from the metallib).
const MOE_GATEUP_KERNEL_SOURCE =
    \\auto lane = thread_index_in_simdgroup;
    \\uint n = thread_position_in_grid.y;      // output row within the expert
    \\uint e = thread_position_in_grid.z;      // top-K slot
    \\
    \\int K = KS;
    \\int N = NS;
    \\int VPW = 32 / BITS;
    \\int K_by_p = K / VPW;
    \\int K_by_gs = K / GS;
    \\uint mask = (1u << BITS) - 1u;
    \\
    \\uint eid = uint(inds[e]);
    \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_p + (size_t)n * (size_t)K_by_p;
    \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
    \\
    \\float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    \\float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    \\for (int pack = int(lane); pack < K_by_p; pack += 32) {
    \\  uint32_t packed_g = wg_q[wbase + (size_t)pack];
    \\  uint32_t packed_u = wu_q[wbase + (size_t)pack];
    \\  int k_base = pack * VPW;
    \\  int gi = k_base / GS;
    \\  float sjg = float(g_scales[gbase + (size_t)gi]);
    \\  float bjg = float(g_biases[gbase + (size_t)gi]);
    \\  float sju = float(u_scales[gbase + (size_t)gi]);
    \\  float bju = float(u_biases[gbase + (size_t)gi]);
    \\  for (int ki = 0; ki < VPW; ki += 4) {
    \\    size_t xi = (size_t)(k_base + ki);
    \\    uint32_t qg = packed_g >> (ki * BITS);
    \\    uint32_t qu = packed_u >> (ki * BITS);
    \\    float x0 = float(x[xi + 0]);
    \\    float x1 = float(x[xi + 1]);
    \\    float x2 = float(x[xi + 2]);
    \\    float x3 = float(x[xi + 3]);
    \\    g0 += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
    \\    g1 += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
    \\    g2 += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
    \\    g3 += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
    \\    u0 += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
    \\    u1 += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
    \\    u2 += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
    \\    u3 += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
    \\  }
    \\}
    \\float acc_g = simd_sum((g0 + g1) + (g2 + g3));
    \\float acc_u = simd_sum((u0 + u1) + (u2 + u3));
    \\if (lane == 0) {
    \\  // Round exactly where the composed pair wrote T(acc), then the f32
    \\  // clip chain: gate clips HIGH only, up clips both sides (reference).
    \\  T gt = T(acc_g);
    \\  T ut = T(acc_u);
    \\  float g32 = float(gt);
    \\  float u32 = float(ut);
    \\  float gc = metal::min(g32, consts[0]);
    \\  float uc = metal::clamp(u32, -consts[0], consts[0]);
    \\  float sig = sigtab[as_type<ushort>(T(gc))];
    \\  y[(size_t)e * (size_t)N + (size_t)n] = T((gc * sig) * uc);
    \\}
;

var moe_gateup_obj: ?mlx.mlx_fast_metal_kernel = null;
var moe_gateup_build_failed: bool = false;
fn moeGateUpObj() ?mlx.mlx_fast_metal_kernel {
    if (moe_gateup_build_failed) return null;
    if (moe_gateup_obj) |kk| return kk;
    const input_names = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "consts" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_moe_gateup", in_vec, out_vec, MOE_GATEUP_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) {
        moe_gateup_build_failed = true;
        log.warn("dsv4: fused MoE gate+up kernel failed to build — composed gathers fallback\n", .{});
        return null;
    }
    moe_gateup_obj = kernel;
    return kernel;
}

/// f32 sigmoid over every bf16-representable input, computed ONCE by
/// mlx_sigmoid itself (256 KB) so the kernel's activation cannot disagree
/// with the composed chain's transcendental. Never freed: process-lifetime.
var moe_sigtab: ?mlx.mlx_array = null;
fn moeSigTableF32(s: mlx.mlx_stream) !mlx.mlx_array {
    if (moe_sigtab) |t| return t;
    const vals = try std.heap.c_allocator.alloc(f32, 65536);
    defer std.heap.c_allocator.free(vals);
    for (vals, 0..) |*v, i| {
        const bits: u32 = @as(u32, @intCast(i)) << 16; // bf16 → f32 is exact widening
        v.* = @bitCast(bits);
    }
    const shape = [_]c_int{65536};
    const raw = uploadF32(vals, &shape);
    defer _ = mlx.mlx_array_free(raw);
    var sig = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, raw, s));
    try mlx.check(mlx.mlx_array_eval(sig));
    moe_sigtab = sig;
    return sig;
}

const MoeGateUpKey = struct { topk: u32, n: u32, ks: u32, bits: u32, gs: u32 };
var moe_gateup_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var moe_gateup_cfg_key: MoeGateUpKey = std.mem.zeroes(MoeGateUpKey);

/// Fused decode gate+up gather + clipped SwiGLU: both expert dot products in
/// one simdgroup (x loaded once, dequant chains interleaved), activation
/// applied before the write — the transformer.zig `gatherQmvGateUp` pattern
/// with dsv4's clipped SwiGLU (f32 chain, sigmoid via an exact bf16-indexed
/// table). Returns [1, TOPK, 1, N] bf16 (the composed `act` shape), or null
/// to decline (kill switch, ineligible geometry, build failure).
fn moeGateUpFused(m: *const Dsv4Model, xe: mlx.mlx_array, q_gate: *const Q, q_up: *const Q, ind: mlx.mlx_array, k: usize) !?mlx.mlx_array {
    if (!moeGateUpEnabled() or !mlx.streamIsGpu(m.s)) return null;
    const bits = q_gate.qp.bits;
    const gs = q_gate.qp.group_size;
    // eligibility is the kernel's own conditions, never a model list
    if (bits != 2 and bits != 4 and bits != 8) return null;
    if (gs % (32 / bits) != 0) return null;
    if (q_up.qp.bits != bits or q_up.qp.group_size != gs) return null;
    if (q_gate.b.ctx == null or q_up.b.ctx == null) return null;
    if (mlx.mlx_array_dtype(xe) != .bfloat16) return null; // sigtab is bf16-indexed
    const idt = mlx.mlx_array_dtype(ind);
    if (idt != .int32 and idt != .uint32) return null;
    if (mlx.mlx_array_ndim(q_gate.w) != 3 or mlx.mlx_array_ndim(q_up.w) != 3) return null;
    const wsh = mlx.mlx_array_shape(q_gate.w);
    const ush = mlx.mlx_array_shape(q_up.w);
    for (0..3) |i| if (wsh[i] != ush[i]) return null;
    const N: usize = @intCast(wsh[1]);
    const K: usize = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (K % gs != 0 or N % 8 != 0) return null;
    const kernel = moeGateUpObj() orelse return null;
    const key = MoeGateUpKey{ .topk = @intCast(k), .n = @intCast(N), .ks = @intCast(K), .bits = bits, .gs = gs };
    if (moe_gateup_cfg == null or !std.meta.eql(moe_gateup_cfg_key, key)) {
        if (moe_gateup_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        moe_gateup_cfg = null;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ @intCast(k), @intCast(N) };
        if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, .bfloat16) != 0 or
            mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @intCast(N), @intCast(k)) != 0 or
            mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 8, 1) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(gs)) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KS", @intCast(K)) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NS", @intCast(N)) != 0)
        {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            return null;
        }
        moe_gateup_cfg = cfg;
        moe_gateup_cfg_key = key;
    }
    const sigtab = try moeSigTableF32(m.s);
    const lim_arr = [_]f32{m.swiglu_limit};
    const lshape = [_]c_int{1};
    const lim_g = uploadF32(&lim_arr, &lshape);
    defer _ = mlx.mlx_array_free(lim_g);
    const inputs_arr = [_]mlx.mlx_array{ xe, q_gate.w, q_gate.s, q_gate.b, q_up.w, q_up.s, q_up.b, ind, sigtab, lim_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, moe_gateup_cfg.?, m.s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    const act_shape = [_]c_int{ 1, @intCast(k), 1, @intCast(N) };
    var act = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_reshape(&act, y, &act_shape, 4, m.s));
    moe_gateup_hits += 1;
    if (!moe_gateup_logged) {
        moe_gateup_logged = true;
        log.info("dsv4: fused MoE gate+up kernel engaged (topk={d} inter={d} bits={d} gs={d})\n", .{ k, N, bits, gs });
    }
    return act;
}

fn qmmBf16(q: *const Q, xin: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var y2 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(&y2, xin, q.w, q.s, q.b, true, mlx.mlx_optional_int.some(@intCast(q.qp.group_size)), mlx.mlx_optional_int.some(@intCast(q.qp.bits)), "affine", s));
    return y2;
}

/// Route one token: scores must already be sqrtsoftplus'd (len E). Writes k
/// expert indices + normalized route weights (hash tid2eid or noaux_tc top-k).
fn routeToken(m: *const Dsv4Model, h: *const HostLayer, scores: []const f32, id: u32, indices: []i32, wts: []f32) void {
    const E = m.n_experts;
    const k = m.topk;
    if (h.tid2eid) |tid| {
        for (0..k) |i| indices[i] = @intCast(tid[@as(usize, id) * k + i]);
    } else {
        const bias = h.gate_bias.?;
        // top-k on scores+bias (selection only)
        var sel: [16]i32 = undefined;
        var selv: [16]f32 = undefined;
        for (0..k) |i| {
            sel[i] = -1;
            selv[i] = -std.math.inf(f32);
        }
        for (0..E) |e| {
            const v = scores[e] + bias[e];
            var i: usize = 0;
            while (i < k) : (i += 1) {
                if (v > selv[i]) {
                    var j: usize = k - 1;
                    while (j > i) : (j -= 1) {
                        selv[j] = selv[j - 1];
                        sel[j] = sel[j - 1];
                    }
                    selv[i] = v;
                    sel[i] = @intCast(e);
                    break;
                }
            }
        }
        for (0..k) |i| indices[i] = sel[i];
    }
    var sum: f64 = 0;
    for (0..k) |i| {
        wts[i] = scores[@intCast(indices[i])];
        sum += wts[i];
    }
    for (0..k) |i| wts[i] = @floatCast(@as(f64, wts[i]) / sum * m.route_scale);
}

/// MoE sublayer on host+mlx: x [s, dim] (post ffn_norm) -> [s, dim].
fn moeForward(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x: []const f32, ids: []const u32, seq: usize) ![]f32 {
    const ly = &m.dw.layers[li];
    const h = &m.hl[li];
    const E = m.n_experts;
    const k = m.topk;
    // gate scores in f32/f64
    const scores = try matMlx(alloc, x, h.gate_w_t, seq, m.dim, E, m.s);
    defer alloc.free(scores);
    for (scores) |*v| v.* = @floatCast(sqrtSoftplus(v.*));
    const indices = try alloc.alloc(i32, seq * k);
    defer alloc.free(indices);
    const wts = try alloc.alloc(f32, seq * k);
    defer alloc.free(wts);
    for (0..seq) |t| {
        routeToken(m, h, scores[t * E ..][0..E], ids[t], indices[t * k ..][0..k], wts[t * k ..][0..k]);
    }
    // routed experts via gather_qmm: xe [s,1,1,D], ind [s,k]
    const xe = try hostToBf16(x, &.{ @intCast(seq), 1, 1, @intCast(m.dim) }, m.s);
    defer _ = mlx.mlx_array_free(xe);
    const ind_shape = [_]c_int{ @intCast(seq), @intCast(k) };
    const ind = mlx.mlx_array_new_data(indices.ptr, &ind_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(ind);
    // Whole expert + shared chain in ONE lazy graph: clipped SwiGLU in f32
    // (the reference computes it in float32), router-weighted sum on GPU,
    // shared expert added, a single host sync at the end.
    const gate_arr = try gatherQmmE(&ly.experts_w1, xe, ind, m.s);
    defer _ = mlx.mlx_array_free(gate_arr);
    const up_arr = try gatherQmmE(&ly.experts_w3, xe, ind, m.s);
    defer _ = mlx.mlx_array_free(up_arr);
    const act = try clippedSwigluG(gate_arr, up_arr, m.swiglu_limit, m.s);
    defer _ = mlx.mlx_array_free(act);
    const down_arr = try gatherQmmE(&ly.experts_w2, act, ind, m.s);
    defer _ = mlx.mlx_array_free(down_arr);
    // router weights [s, k, 1, 1] × down [s, k, 1, d] summed over k
    const wshape2 = [_]c_int{ @intCast(seq), @intCast(k), 1, 1 };
    const w_arr2 = mlx.mlx_array_new_data(wts.ptr, &wshape2, 4, .float32);
    defer _ = mlx.mlx_array_free(w_arr2);
    var down32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down32);
    try mlx.check(mlx.mlx_astype(&down32, down_arr, .float32, m.s));
    var weighted2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted2);
    try mlx.check(mlx.mlx_multiply(&weighted2, down32, w_arr2, m.s));
    var routed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed);
    try mlx.check(mlx.mlx_sum_axis(&routed, weighted2, 1, false, m.s)); // [s, 1, d]
    // shared expert on GPU (clipped too)
    const xb2 = try hostToBf16(x, &.{ @intCast(seq), @intCast(m.dim) }, m.s);
    defer _ = mlx.mlx_array_free(xb2);
    const sg_arr = try qmmBf16(&ly.shared_w1, xb2, m.s);
    defer _ = mlx.mlx_array_free(sg_arr);
    const su_arr = try qmmBf16(&ly.shared_w3, xb2, m.s);
    defer _ = mlx.mlx_array_free(su_arr);
    const sact = try clippedSwigluG(sg_arr, su_arr, m.swiglu_limit, m.s);
    defer _ = mlx.mlx_array_free(sact);
    const sd_arr = try qmmBf16(&ly.shared_w2, sact, m.s);
    defer _ = mlx.mlx_array_free(sd_arr);
    var sd32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sd32);
    try mlx.check(mlx.mlx_astype(&sd32, sd_arr, .float32, m.s));
    var sd_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sd_r);
    const rshape = [_]c_int{ @intCast(seq), 1, @intCast(m.dim) };
    try mlx.check(mlx.mlx_reshape(&sd_r, sd32, &rshape, 3, m.s));
    var total = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(total);
    try mlx.check(mlx.mlx_add(&total, routed, sd_r, m.s));
    return try toHostF32(alloc, total, seq * m.dim, m.s);
}

/// hc_pre on the [s, hc, dim] stream: returns collapsed [s, dim] + per-token
/// post [s, hc] + comb [s, hc, hc].
fn hcPreForward(m: *const Dsv4Model, alloc: std.mem.Allocator, stream_: []const f32, seq: usize, fn_w_t: mlx.mlx_array, scale: []const f32, base: []const f32) !struct { y: []f32, post: []f32, comb: []f32 } {
    const hcm = m.hc;
    const hd = hcm * m.dim;
    const mix = (2 + hcm) * hcm;
    const y = try alloc.alloc(f32, seq * m.dim);
    const post = try alloc.alloc(f32, seq * hcm);
    const comb = try alloc.alloc(f32, seq * hcm * hcm);
    // mixes for all rows in one GPU matmul (f32 — the hc path is f32 by design)
    const mixes_all = try matMlx(alloc, stream_[0 .. seq * hd], fn_w_t, seq, hd, mix, m.s);
    defer alloc.free(mixes_all);
    var mixes: [96]f32 = undefined;
    for (0..seq) |t| {
        const flat = stream_[t * hd ..][0..hd];
        var ss: f64 = 0;
        for (flat) |v| ss += @as(f64, v) * v;
        const rsq: f32 = @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(hd)) + m.eps));
        for (0..mix) |j| mixes[j] = mixes_all[t * mix + j] * rsq;
        const split = hcSplitSinkhorn(mixes[0..mix], scale, base, hcm, m.hc_iters, m.hc_eps);
        const yr = y[t * m.dim ..][0..m.dim];
        @memset(yr, 0);
        for (0..hcm) |c| {
            const pre = split.pre[c];
            const src = stream_[(t * hcm + c) * m.dim ..][0..m.dim];
            for (yr, src) |*ov, sv| ov.* += pre * sv;
        }
        @memcpy(post[t * hcm ..][0..hcm], split.post[0..hcm]);
        @memcpy(comb[t * hcm * hcm ..][0 .. hcm * hcm], split.comb[0 .. hcm * hcm]);
    }
    return .{ .y = y, .post = post, .comb = comb };
}

/// hc_post: stream[k] = post[k]·out + Σ_j comb[j,k]·residual[j] (comb TRANSPOSED).
fn hcPostForward(m: *const Dsv4Model, stream_: []f32, out: []const f32, post: []const f32, comb: []const f32, seq: usize) void {
    const hcm = m.hc;
    const d = m.dim;
    var tmp: [8 * 8192]f32 = undefined; // residual copy for one token (hc*dim)
    for (0..seq) |t| {
        const res = tmp[0 .. hcm * d];
        @memcpy(res, stream_[t * hcm * d ..][0 .. hcm * d]);
        for (0..hcm) |k| {
            const dst = stream_[(t * hcm + k) * d ..][0..d];
            const pk = post[t * hcm + k];
            const orow = out[t * d ..][0..d];
            for (0..d) |j2| {
                var acc: f64 = @as(f64, pk) * orow[j2];
                for (0..hcm) |j| acc += @as(f64, comb[t * hcm * hcm + j * hcm + k]) * res[j * d + j2];
                dst[j2] = @floatCast(acc);
            }
        }
    }
}

/// Full prefill forward; returns last-position logits [vocab] (host f32).
pub fn forwardPrefill(m: *const Dsv4Model, gpa: std.mem.Allocator, ids: []const u32) ![]f32 {
    return forwardPrefillCapture(m, gpa, ids, null);
}

/// Prefill that seeds an incremental-decode state: the batched GPU chunk
/// path (extendState) from an empty state. Pinned vs the python-oracle
/// fixtures AND the stateless host re-forward by the DSV4_MINI gates.
pub fn prefillIntoState(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, ids: []const u32) ![]f32 {
    return extendState(m, gpa, st, ids);
}

fn forwardPrefillCapture(m: *const Dsv4Model, gpa: std.mem.Allocator, ids: []const u32, st: ?*Dsv4DecodeState) ![]f32 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const seq = ids.len;
    const d = m.dim;
    const hcm = m.hc;
    // stream [s, hc, d] = embed repeated
    const stream_ = try a.alloc(f32, seq * hcm * d);
    for (0..seq) |t| {
        const e = m.embed_f32[@as(usize, ids[t]) * d ..][0..d];
        for (0..hcm) |c| @memcpy(stream_[(t * hcm + c) * d ..][0..d], e);
    }
    const x_norm = try a.alloc(f32, seq * d);
    for (0..m.n_layers) |li| {
        const h = &m.hl[li];
        {
            const pre = try hcPreForward(m, a, stream_, seq, h.hc_attn_fn_t, h.hc_attn_scale, h.hc_attn_base);
            @memcpy(x_norm, pre.y);
            for (0..seq) |t| rmsNormRow(x_norm[t * d ..][0..d], h.attn_norm, m.eps);
            const attn_out = try attentionForward(m, a, li, x_norm, seq, if (st) |s2| &s2.layers[li] else null);
            hcPostForward(m, stream_, attn_out, pre.post, pre.comb, seq);
        }
        {
            const pre = try hcPreForward(m, a, stream_, seq, h.hc_ffn_fn_t, h.hc_ffn_scale, h.hc_ffn_base);
            @memcpy(x_norm, pre.y);
            for (0..seq) |t| rmsNormRow(x_norm[t * d ..][0..d], h.ffn_norm, m.eps);
            const ffn_out = try moeForward(m, a, li, x_norm, ids, seq);
            hcPostForward(m, stream_, ffn_out, pre.post, pre.comb, seq);
        }
    }
    // hyper-head collapse (sigmoid weights only) on the LAST position
    const t = seq - 1;
    const flat = stream_[t * hcm * d ..][0 .. hcm * d];
    var ss: f64 = 0;
    for (flat) |v| ss += @as(f64, v) * v;
    const rsq: f32 = @floatCast(1.0 / @sqrt(ss / @as(f64, @floatFromInt(hcm * d)) + m.eps));
    const hout = try gpa.alloc(f32, d);
    defer gpa.free(hout);
    @memset(hout, 0);
    for (0..hcm) |c| {
        var acc: f64 = 0;
        const wr = m.hc_head_fn[c * hcm * d ..][0 .. hcm * d];
        for (flat, wr) |a2, b2| acc += @as(f64, a2) * b2;
        const mixv = @as(f32, @floatCast(acc)) * rsq;
        const pre = sigmoidF32(mixv * m.hc_head_scale[0] + m.hc_head_base[c]) + m.hc_eps;
        const src = stream_[(t * hcm + c) * d ..][0..d];
        for (hout, src) |*ov, sv| ov.* += pre * sv;
    }
    rmsNormRow(hout, m.final_norm, m.eps);
    const logits = try qmmHost(gpa, &m.dw.head, hout, 1, d, m.vocab, m.s);
    return logits;
}

// ── tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// DSpark is opt-in at load; tests that exercise it must say so explicitly
/// (test order must not decide whether the stages arm).
fn testEnableDspark() void {
    _ = setenv("MLX_SERVE_DSV4_DSPARK", "1", 1);
}

test "dsv4: fp8/fp4 QAT sims match the python oracle goldens" {
    var x8 = [_]f32{ 0.001, -0.02, 0.3, -4.0, 55.0, -600.0, 0.125, 3.14159 };
    fp8SimInPlace(&x8, 8);
    const want8 = [_]f32{ 0, -0.01953125, 0.3125, -4, 56, -576, 0.125, 3.25 };
    for (x8, want8) |got, want| try testing.expectApproxEqAbs(want, got, 1e-7);

    var x4 = [_]f32{ 0.001, -0.02, 0.3, -4.0, 55.0, -600.0, 0.125, 3.14159 };
    fp4SimInPlace(&x4, 8);
    const want4 = [_]f32{ 0, 0, 0, 0, 64, -512, 0, 0 };
    for (x4, want4) |got, want| try testing.expectApproxEqAbs(want, got, 1e-7);
}

test "dsv4: hadamard transform matches the python oracle golden" {
    var v = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    hadamardInPlace(&v);
    const want = [_]f32{ 12.7279224, -1.41421366, -2.82842708, 0, -5.65685415, 0, 0, 0 };
    for (v, want) |got, w| try testing.expectApproxEqAbs(w, got, 1e-5);
}

test "dsv4: sinkhorn hyper-connection split matches the python oracle golden" {
    const mixes = [_]f32{ -0.561352015, -0.927051306, -0.173853129, 0.294311672, 0.795232594, 0.0767944828, -0.386853129, -0.549346268, 0.524122059, 1.14434814, 0.190938145, -0.863330066, -0.670785666, 1.12001336, 0.142017707, -1.21249437, -0.0585873351, -0.814258158, -0.44050166, -0.341604084, -0.499319375, 0.387364924, -0.0441601798, -0.412601888 };
    const scale = [_]f32{ 0.5, 0.8, 1.2 };
    const base = [_]f32{ 0.122891352, 0.248956591, -0.492907017, -0.0770190358, -0.294224203, -0.0519465692, -0.386825621, 0.00620711828, -0.0113657219, -0.091301322, -0.314377964, -0.118857101, -0.327398658, -0.406562626, 0.0674357191, -0.332804978, 0.351088822, 0.214976296, -0.599345028, 0.081638664, -0.330514997, 0.00991716608, 0.0130895982, -0.596528947 };
    const got = hcSplitSinkhorn(&mixes, &scale, &base, 4, 20, 1e-6);
    const want_pre = [_]f32{ 0.46063647, 0.446563598, 0.358971887, 0.517528016 };
    const want_post = [_]f32{ 1.16933402, 1.00474447, 0.665262542, 0.78669156 };
    const want_comb = [_]f32{ 0.368557931, 0.315652379, 0.175677823, 0.140110867, 0.105903384, 0.369821373, 0.401255678, 0.123018565, 0.383524451, 0.0595728191, 0.0902480571, 0.466653673, 0.142013235, 0.254952428, 0.332817443, 0.270215894 };
    for (got.pre[0..4], want_pre) |g, w| try testing.expectApproxEqAbs(w, g, 1e-4);
    for (got.post[0..4], want_post) |g, w| try testing.expectApproxEqAbs(w, g, 1e-4);
    for (got.comb[0..16], want_comb) |g, w| try testing.expectApproxEqAbs(w, g, 1e-4);
}

test "dsv4: loads the fabricated miniature checkpoint (DSV4_MINI)" {
    // Fabricate with:
    //   python3 tests/dsv4_mlx_ref.py --fabricate /tmp/dsv4-mini
    //   DSV4_MINI=/tmp/dsv4-mini zig build test -Dtest-filter=DSV4_MINI
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const cfg_json = try reader_state.interface.allocRemaining(allocator, .limited(1 << 20));
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);

    try testing.expectEqualStrings("deepseek_v4", cfg.model_type);
    try testing.expectEqual(@as(u32, 4), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u8, 0), cfg.dsv4_compress_ratios[0]);
    try testing.expectEqual(@as(u8, 4), cfg.dsv4_compress_ratios[1]);
    try testing.expectEqual(@as(u8, 16), cfg.dsv4_compress_ratios[2]);

    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    var dw = try loadDsv4Weights(allocator, &cfg, &weights);
    defer dw.deinit();

    // 4 trunk layers + 3 DSpark stages share one slice (stages appended so
    // the GPU decode helpers address them by layer index).
    try testing.expectEqual(@as(usize, 7), dw.layers.len);
    // Miniature is uniformly affine 8-bit gs32 — geometry solving must agree.
    try testing.expectEqual(@as(u32, 8), dw.embed.qp.bits);
    try testing.expectEqual(@as(u32, 32), dw.embed.qp.group_size);
    try testing.expectEqual(@as(u32, 8), dw.layers[0].experts_w1.qp.bits);
    // Layer roles: 0 = hash-routed sliding; 1 = ratio-4 with indexer;
    // 2 = ratio-16 compressor, no indexer; 3 = ratio-4 with indexer.
    try testing.expect(dw.layers[0].tid2eid != null);
    try testing.expect(dw.layers[0].gate_bias == null);
    try testing.expect(dw.layers[0].comp == null);
    try testing.expect(dw.layers[1].tid2eid == null);
    try testing.expect(dw.layers[1].gate_bias != null);
    try testing.expect(dw.layers[1].comp != null);
    try testing.expect(dw.layers[1].idx != null);
    try testing.expect(dw.layers[2].comp != null);
    try testing.expect(dw.layers[2].idx == null);
    try testing.expect(dw.layers[3].idx != null);
    // Expert banks are stacked [E, out, packed]: rank 3.
    try testing.expectEqual(@as(usize, 3), mlx.mlx_array_ndim(dw.layers[0].experts_w1.w));

    // DSpark: 3 trunk-shaped stages appended after the trunk layers (stage
    // count = ratio-table entries past num_hidden_layers — the release ships
    // no n_mtp_layers key), plus the stage-0/last-stage extras.
    try testing.expectEqual(@as(u32, 3), cfg.dsv4_dspark_block_size);
    try testing.expectEqual(@as(u32, 2), cfg.dsv4_n_dspark_target_layers);
    try testing.expectEqual(@as(u8, 1), cfg.dsv4_dspark_target_layers[0]);
    try testing.expectEqual(@as(u8, 3), cfg.dsv4_dspark_target_layers[1]);
    const ds = &dw.dspark.?;
    for (dw.layers[4..]) |*st| {
        try testing.expectEqual(@as(u8, 0), st.compress_ratio); // DSparkAttention asserts ratio 0
        try testing.expect(st.comp == null and st.idx == null);
        try testing.expect(st.tid2eid == null and st.gate_bias != null); // scored, never hash
    }
    // main_proj input = dim * n_targets (the concat must not hardcode 3)
    try testing.expectEqual(@as(u32, 8), ds.main_proj.qp.bits);
    try testing.expectEqual(@as(usize, 2), mlx.mlx_array_ndim(ds.markov_w1));

    // Stage weights are OUTSIDE every warmup/serial forward, so initModel
    // must batch-eval them at load (left lazy they materialize mid-first-
    // draft, GBs past every load-time memory budget). The collector is
    // comptime-reflective — a future stage field is picked up structurally —
    // and must see every Q triple + handle of all 3 stages + the extras.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var tally = WeightTally{};
        for (dw.layers[4..]) |*st| tally = tally.add(appendWeightArrays(vec, st.*));
        tally = tally.add(appendWeightArrays(vec, dw.dspark.?));
        try testing.expect(tally.n > 120); // 3 stages ≈ 45 arrays each + extras
        try testing.expectEqual(tally.n, mlx.mlx_vector_array_size(vec));
        try testing.expect(tally.bytes > 0); // logical bytes pre-eval feed the fit check
        try mlx.check(mlx.mlx_eval(vec));
    }
}

test "dsv4: forwardPrefill matches the python oracle fixtures (DSV4_MINI)" {
    // End-to-end parity on the fabricated miniature: same weights both sides,
    // fixtures dumped by `tests/dsv4_mlx_ref.py --model <mini> --dump-fixtures
    // <mini>/fixtures.json`. Compares last-position logits (cosine + argmax) —
    // one number downstream of EVERY component (attention incl. window +
    // compression + indexer + sims, Sinkhorn hc, hash + scored MoE, head).
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const readAll = struct {
        fn f(io_: std.Io, alloc: std.mem.Allocator, p: []const u8) ![]u8 {
            const file = try std.Io.Dir.openFileAbsolute(io_, p, .{});
            defer file.close(io_);
            var rb: [4096]u8 = undefined;
            var rs = file.reader(io_, &rb);
            return try rs.interface.allocRemaining(alloc, .limited(1 << 26));
        }
    }.f;

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const cfg_json = try readAll(io, allocator, cfg_path);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);

    // Read fixtures BEFORE loading weights: the skip path (`catch return`)
    // used to sit after `loadDsv4Weights`, leaking the layer table when a
    // mini had no fixtures.json dumped.
    const fx_path = try std.fmt.allocPrint(allocator, "{s}/fixtures.json", .{path});
    defer allocator.free(fx_path);
    const fx_json = readAll(io, allocator, fx_path) catch return; // fixtures not dumped -> skip
    defer allocator.free(fx_json);

    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();
    const dw = try loadDsv4Weights(allocator, &cfg, &weights);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, fx_json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const ids_v = root.get("input_ids").?.array.items;
    const ids = try allocator.alloc(u32, ids_v.len);
    defer allocator.free(ids);
    for (ids, ids_v) |*o, v| o.* = @intCast(v.integer);
    const want_v = root.get("logits_last").?.array.items;
    const want = try allocator.alloc(f32, want_v.len);
    defer allocator.free(want);
    for (want, want_v) |*o, v| o.* = @floatCast(switch (v) {
        .float => |f| f,
        .integer => |i| @as(f64, @floatFromInt(i)),
        else => unreachable,
    });

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var mdl = try initModel(allocator, &cfg, dw, s);
    defer mdl.deinit();

    const logits = try forwardPrefill(&mdl, allocator, ids);
    defer allocator.free(logits);

    try testing.expectEqual(want.len, logits.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var argmax_got: usize = 0;
    var argmax_want: usize = 0;
    for (logits, want, 0..) |g, w2, i| {
        try testing.expect(std.math.isFinite(g)); // NaN must FAIL, never score
        dot += @as(f64, g) * w2;
        na += @as(f64, g) * g;
        nb += @as(f64, w2) * w2;
        if (g > logits[argmax_got]) argmax_got = i;
        if (w2 > want[argmax_want]) argmax_want = i;
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
    std.debug.print("dsv4 parity: cos={d:.6} argmax got={d} want={d}\n", .{ cos, argmax_got, argmax_want });
    // The miniature has RANDOM weights (near-uniform logits), so this cosine
    // is deliberately loose — it amplifies benign dtype-path differences
    // (host f64 accum vs mlx bf16/f32 kernels). The sharp gates are the
    // argmax here + the decode-equivalence test + the real-mirror greedy test.
    try testing.expect(cos > 0.99);
    try testing.expectEqual(argmax_want, argmax_got);

    // The BATCHED GPU prefill (extendState — the serving path) must hit the
    // same oracle fixture independently.
    var st = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st);
    const logits_b = try prefillIntoState(&mdl, allocator, &st, ids);
    defer allocator.free(logits_b);
    var dot_b: f64 = 0;
    var na_b: f64 = 0;
    var am_b: usize = 0;
    for (logits_b, want, 0..) |g, w2, i| {
        try testing.expect(std.math.isFinite(g));
        dot_b += @as(f64, g) * w2;
        na_b += @as(f64, g) * g;
        if (g > logits_b[am_b]) am_b = i;
    }
    const cos_b = dot_b / (@sqrt(na_b) * @sqrt(nb) + 1e-30);
    std.debug.print("dsv4 parity (batched): cos={d:.6} argmax got={d} want={d}\n", .{ cos_b, am_b, argmax_want });
    try testing.expect(cos_b > 0.99);
    try testing.expectEqual(argmax_want, am_b);
}

test "dsv4: forwardPrefill on the REAL mirror continues 'The capital of France is' (DSV4_REAL)" {
    // Real-scale sanity: DSV4_REAL=<mirror dir> loads the converted ~118 GB
    // checkpoint and greedy-continues the oracle's exact prompt. The python
    // oracle produced token 11111 (' Paris') at this position — the Zig
    // forward must agree. Host-centric v1 is slow (~tens of seconds/token at
    // 43 real layers) — 2 tokens is the sanity bar, not a benchmark.
    const path_z = std.c.getenv("DSV4_REAL") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);

    var weights = try model.loadWeights(io, allocator, path);
    defer weights.deinit();
    const dw = try loadDsv4Weights(allocator, &cfg, &weights);

    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var mdl = try initModel(allocator, &cfg, dw, s);
    defer mdl.deinit();

    var ids = std.array_list.Managed(u32).init(allocator);
    defer ids.deinit();
    try ids.appendSlice(&.{ 671, 6102, 294, 8760, 344 }); // "The capital of France is"
    // Ground truth RE-DERIVED per MIRROR (tests/dsv4_mlx_ref.py) — the
    // expectation is weights-pinned, so a checkpoint OR quant-recipe bump
    // re-derives it from the oracle before anyone calls a diff a bug.
    // imx-2-3-8bit (imatrix gs128, 2026-08-01): 11111, 16 = ' Paris', '.'.
    // iQ-MLX-3.3bpw (greedy late-layer 3b plan, 2026-08-03): re-derived,
    // SAME 11111, 16. The superseded mixed-2-3-8bit (minmax gs64) continued
    // 11111, 66910 ('.",') — same first token, near-tie second.
    const want = [_]u32{ 11111, 16 };
    for (want) |w| {
        const logits = try forwardPrefill(&mdl, allocator, ids.items);
        defer allocator.free(logits);
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            try testing.expect(std.math.isFinite(v));
            if (v > logits[best]) best = i;
        }
        std.debug.print("dsv4 real: s={d} -> token {d} (want {d})\n", .{ ids.items.len, best, w });
        try testing.expectEqual(@as(usize, w), best);
        try ids.append(@intCast(best));
    }
    // The SERVING path on real weights: batched prefill + one decode step.
    var st = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st);
    {
        const logits = try prefillIntoState(&mdl, allocator, &st, ids.items[0..5]);
        defer allocator.free(logits);
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            try testing.expect(std.math.isFinite(v));
            if (v > logits[best]) best = i;
        }
        std.debug.print("dsv4 real batched prefill -> token {d} (want 11111)\n", .{best});
        try testing.expectEqual(@as(usize, 11111), best);
    }
    {
        const logits = try decodeStep(&mdl, allocator, &st, 11111);
        defer allocator.free(logits);
        var best: usize = 0;
        for (logits, 0..) |v, i| {
            try testing.expect(std.math.isFinite(v));
            if (v > logits[best]) best = i;
        }
        // Same weights-pinned second token as the prefill loop above.
        std.debug.print("dsv4 real decode after batched prefill -> token {d} (want {d})\n", .{ best, want[1] });
        try testing.expectEqual(@as(usize, want[1]), best);
    }
}

test "dsv4: incremental decode matches full re-forward (DSV4_MINI)" {
    // Decode-after-prefill must agree with the stateless full re-forward at
    // every step (the only tolerated divergence is mlx picking different
    // matmul kernels by batch shape — the qmv-vs-qmm class), across lengths
    // that cross the window (8), ratio-4 and ratio-16 boundaries.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    // Both streams: the CPU run pins the composed chain; the GPU run also
    // engages the fused Sinkhorn kernel in situ (Metal-only, exactly what
    // production serving uses).
    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        var rng = std.Random.DefaultPrng.init(9);
        var ids: [34]u32 = undefined;
        for (&ids) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

        // A ONE-token fresh prefill first: C==1 through the BATCH path is the
        // serving warmup's exact shape and produced a rank-1-vs-rank-2
        // sinkhorn-output slice kill on first live boot (2026-07-31) — the
        // 6-token prefill below never sees it.
        {
            var st1 = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st1);
            const p1 = try prefillIntoState(&mdl, allocator, &st1, ids[0..1]);
            defer allocator.free(p1);
            const f1 = try forwardPrefill(&mdl, allocator, ids[0..1]);
            defer allocator.free(f1);
            var am_p: usize = 0;
            var am_f1: usize = 0;
            for (p1, f1, 0..) |a2, b2, i| {
                try testing.expect(std.math.isFinite(a2));
                if (a2 > p1[am_p]) am_p = i;
                if (b2 > f1[am_f1]) am_f1 = i;
            }
            try testing.expectEqual(am_f1, am_p);
        }

        // prefill 6 tokens into state, EXTEND with a batched chunk of 11
        // (crossing ratio-4 boundaries + the window), then decode one-by-one
        // to 34 — the chunk stage pins base>0 chunked-prefill continuation.
        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        {
            const pl = try prefillIntoState(&mdl, allocator, &st, ids[0..6]);
            allocator.free(pl);
        }
        {
            const cl = try extendState(&mdl, allocator, &st, ids[6..17]);
            defer allocator.free(cl);
            const full = try forwardPrefill(&mdl, allocator, ids[0..17]);
            defer allocator.free(full);
            var am_c: usize = 0;
            var am_f2: usize = 0;
            for (cl, full, 0..) |a2, b2, i| {
                try testing.expect(std.math.isFinite(a2));
                if (a2 > cl[am_c]) am_c = i;
                if (b2 > full[am_f2]) am_f2 = i;
            }
            try testing.expectEqual(am_f2, am_c);
        }
        // The decode chain reduces in a different order than the prefill
        // reference, and the QAT sims + indexer top-k are DISCRETE: a value
        // within that drift of a code/rank boundary flips a whole quantum
        // (the sanctioned qmv-vs-qmm class). On the CPU stream both paths hit
        // the same gemm, so drift is ~1e-6 and the gate stays strict (cos ≥
        // 0.99 + argmax EVERY step). On the GPU stream prefill ([seq,·]) and
        // decode ([1,·]) matmuls pick DIFFERENT Metal kernels (bf16 drift
        // ~1e-3), which the sims/top-k quantize into occasional slot swaps on
        // the random-weight mini — so the GPU arm gates on: first 6 steps
        // EXACT (wiring bugs scream immediately), cos ≥ 0.9 every step, and
        // argmax agreement ≥ 80%. The real-mirror greedy test arbitrates
        // end-to-end quality.
        const strict = !use_gpu;
        var agree: usize = 0;
        var steps: usize = 0;
        var n: usize = 17;
        while (n < ids.len) : (n += 1) {
            const dec = try decodeStep(&mdl, allocator, &st, ids[n]);
            defer allocator.free(dec);
            // reference: stateless full re-forward over the same prefix + token
            const full = try forwardPrefill(&mdl, allocator, ids[0 .. n + 1]);
            defer allocator.free(full);
            var dot: f64 = 0;
            var na: f64 = 0;
            var nb: f64 = 0;
            var am_d: usize = 0;
            var am_f: usize = 0;
            for (dec, full, 0..) |a2, b2, i| {
                try testing.expect(std.math.isFinite(a2));
                dot += @as(f64, a2) * b2;
                na += @as(f64, a2) * a2;
                nb += @as(f64, b2) * b2;
                if (a2 > dec[am_d]) am_d = i;
                if (b2 > full[am_f]) am_f = i;
            }
            const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
            steps += 1;
            if (am_d == am_f) agree += 1;
            const cos_floor: f64 = if (strict) 0.99 else 0.9;
            const argmax_must_match = strict or steps <= 6;
            if (cos < cos_floor or (argmax_must_match and am_d != am_f)) {
                std.debug.print("decode-equiv DIVERGED (gpu={}) at n={d}: cos={d:.6} argmax {d} vs {d}\n", .{ use_gpu, n + 1, cos, am_d, am_f });
                try testing.expect(false);
            }
        }
        try testing.expect(agree * 10 >= steps * 8);
        std.debug.print("dsv4 decode-equivalence (gpu={}): OK through n=34, argmax {d}/{d} (window 8, ratios 4/16 crossed)\n", .{ use_gpu, agree, steps });
    }
}

test "dsv4: wo_a batched qmm slabs match the dequantized operands (no worse than bf16 ref)" {
    // The grouped low-rank O tail's two servings of the SAME checkpoint
    // weight: dequantized bf16 slabs [og, gin, ol] (matmul) vs quantized
    // views [og, ol, ·] (batched quantized_matmul). Both compared against
    // f32 dequant ground truth — never kernel-vs-kernel (house rule): the
    // qmm arm must not be meaningfully worse than the bf16 arm it replaces.
    const allocator = testing.allocator;
    const og: usize = 4;
    const ol: usize = 8;
    const gin: usize = 128;
    const M: usize = 3;
    const s = if (!mlx.noGpuBackend()) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var rng = std.Random.DefaultPrng.init(7);
    const wf = try allocator.alloc(f32, og * ol * gin);
    defer allocator.free(wf);
    for (wf) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
    const xf = try allocator.alloc(f32, og * M * gin);
    defer allocator.free(xf);
    for (xf) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    const wshape = [_]c_int{ @intCast(og * ol), @intCast(gin) };
    const w32 = uploadF32(wf, &wshape);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s)); // bf16 scales, like the checkpoint
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
    var q_w = mlx.mlx_array_new();
    var q_s = mlx.mlx_array_new();
    var q_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    defer _ = mlx.mlx_array_free(q_s);
    defer _ = mlx.mlx_array_free(q_b);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q_b, triple, 2));

    const gs64 = mlx.mlx_optional_int.some(64);
    const b8 = mlx.mlx_optional_int.some(8);
    const emp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emp);
    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, q_w, q_s, q_b, gs64, b8, "affine", emp, mlx.mlx_optional_dtype{}, s));

    // f32 ground truth: x [og, M, gin] @ deq-f32 [og, gin, ol]
    const xshape = [_]c_int{ @intCast(og), @intCast(M), @intCast(gin) };
    const x32 = uploadF32(xf, &xshape);
    defer _ = mlx.mlx_array_free(x32);
    const ref = blk: {
        var dq32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dq32);
        try mlx.check(mlx.mlx_astype(&dq32, dq, .float32, s));
        var r3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r3);
        const shp3 = [_]c_int{ @intCast(og), @intCast(ol), @intCast(gin) };
        try mlx.check(mlx.mlx_reshape(&r3, dq32, &shp3, 3, s));
        var tr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tr);
        const axes = [_]c_int{ 0, 2, 1 };
        try mlx.check(mlx.mlx_transpose_axes(&tr, r3, &axes, 3, s));
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_matmul(&out, x32, tr, s));
        break :blk try toHostF32(allocator, out, og * M * ol, s);
    };
    defer allocator.free(ref);

    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_astype(&xb, x32, .bfloat16, s));

    // arm A: the bf16 dequantized-slab matmul (the wo_a_deq path)
    const arm_a = blk: {
        var r3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r3);
        const shp3 = [_]c_int{ @intCast(og), @intCast(ol), @intCast(gin) };
        try mlx.check(mlx.mlx_reshape(&r3, dq, &shp3, 3, s));
        var tr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tr);
        const axes = [_]c_int{ 0, 2, 1 };
        try mlx.check(mlx.mlx_transpose_axes(&tr, r3, &axes, 3, s));
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_matmul(&out, xb, tr, s));
        break :blk try toHostF32(allocator, out, og * M * ol, s);
    };
    defer allocator.free(arm_a);

    // arm B: quantized views + batched qmm (the wo_a_q3 path)
    const arm_b = blk: {
        const v_w = try reshapeQ3(q_w, @intCast(og), @intCast(ol), s);
        defer _ = mlx.mlx_array_free(v_w);
        const v_s = try reshapeQ3(q_s, @intCast(og), @intCast(ol), s);
        defer _ = mlx.mlx_array_free(v_s);
        const v_b = try reshapeQ3(q_b, @intCast(og), @intCast(ol), s);
        defer _ = mlx.mlx_array_free(v_b);
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_quantized_matmul(&out, xb, v_w, v_s, v_b, true, gs64, b8, "affine", s));
        break :blk try toHostF32(allocator, out, og * M * ol, s);
    };
    defer allocator.free(arm_b);

    var err_a: f64 = 0;
    var err_b: f64 = 0;
    for (ref, arm_a, arm_b) |r, va, vb| {
        try testing.expect(std.math.isFinite(vb));
        err_a += @abs(@as(f64, va) - r);
        err_b += @abs(@as(f64, vb) - r);
    }
    // qmm dequantizes in f32 on the fly — it should be no worse than the
    // bf16-rounded operands (1.5x slack absorbs reduction-order noise).
    try testing.expect(err_b <= err_a * 1.5 + 1e-3);
}

test "dsv4: lazy pipelined decode matches decodeStep at every position (DSV4_MINI)" {
    // Two states, same prefill, same fed ids: decodeStepLazy (GPU token id,
    // lazy logits, deferred ring drains) must produce the same argmax as the
    // synchronous decodeStep every step, crossing ratio-4/16 boundaries so
    // the drain-before-emission path runs. GPU stream only — the lazy path
    // requires it by contract.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const dw = try loadDsv4Weights(allocator, &cfg, &weights);
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    _ = unsetenv("MLX_SERVE_DSV4_DSPARK"); // test-order hygiene: lazy needs DSpark off
    var mdl = try initModel(allocator, &cfg, dw, s);
    defer mdl.deinit();
    try testing.expect(mdl.embed_g != null);

    var rng = std.Random.DefaultPrng.init(21);
    var ids: [33]u32 = undefined;
    for (&ids) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

    var st_sync = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st_sync);
    var st_lazy = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st_lazy);
    try testing.expect(lazyDecodeReady(&mdl, &st_lazy));
    {
        const p1 = try prefillIntoState(&mdl, allocator, &st_sync, ids[0..17]);
        allocator.free(p1);
        const p2 = try prefillIntoState(&mdl, allocator, &st_lazy, ids[0..17]);
        allocator.free(p2);
    }
    var agree: usize = 0;
    var steps: usize = 0;
    for (17..ids.len) |n| {
        const sync_logits = try decodeStep(&mdl, allocator, &st_sync, ids[n]);
        defer allocator.free(sync_logits);
        const idv: i32 = @intCast(ids[n]);
        const ishape = [_]c_int{ 1, 1 };
        const id_arr = mlx.mlx_array_new_data(&idv, &ishape, 2, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        const lazy_g = try decodeStepLazy(&mdl, allocator, &st_lazy, id_arr);
        defer _ = mlx.mlx_array_free(lazy_g);
        const lazy_logits = try toHostF32(allocator, lazy_g, @intCast(mdl.vocab), s);
        defer allocator.free(lazy_logits);
        var am_s: usize = 0;
        var am_l: usize = 0;
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (sync_logits, lazy_logits, 0..) |a2, b2, i| {
            try testing.expect(std.math.isFinite(b2));
            dot += @as(f64, a2) * b2;
            na += @as(f64, a2) * a2;
            nb += @as(f64, b2) * b2;
            if (a2 > sync_logits[am_s]) am_s = i;
            if (b2 > lazy_logits[am_l]) am_l = i;
        }
        const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
        steps += 1;
        if (am_s == am_l) agree += 1;
        try testing.expect(cos > 0.999);
    }
    // the graphs are op-identical (same inputs, same order) — argmax must
    // agree everywhere; cos gate above catches wiring bugs with a message
    try testing.expectEqual(steps, agree);
    // teardown-drain path: pending rows may remain after the last token
    try drainPending(&mdl, &st_lazy);
    try testing.expectEqual(@as(usize, 0), st_lazy.pending.items.len);
}

test "dsv4: GpuRows.truncate is offset-only and appends overwrite stale rows" {
    // DSpark rollback primitive: truncate rewinds `used` (capacity-agnostic,
    // KVCache.truncate convention); the next append lands where the rolled-
    // back rows were and readers only ever see [0, used).
    var rows = GpuRows.init(2);
    defer rows.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    try rows.append(&[_]f32{ 1, 2, 3, 4 }, s); // rows [1,2],[3,4]
    try rows.append(&[_]f32{ 5, 6 }, s);
    try testing.expectEqual(@as(usize, 3), rows.used);
    const cap_before = rows.cap;
    rows.truncate(1);
    try testing.expectEqual(@as(usize, 1), rows.used);
    try testing.expectEqual(cap_before, rows.cap); // offset-only, no realloc
    try rows.append(&[_]f32{ 7, 8, 9, 10 }, s);
    try testing.expectEqual(@as(usize, 3), rows.used);

    const v = try rows.sliceRows(0, 3, s);
    defer _ = mlx.mlx_array_free(v);
    const host = try toHostF32(testing.allocator, v, 6, s);
    defer testing.allocator.free(host);
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 7, 8, 9, 10 }, host);
}

test "dsv4: dsparkFitsBudget admits stages only inside the working-set budget" {
    const GB = 1024 * 1024 * 1024;
    // the real mirror on a 128 GB box: trunk 107 GB + stages 11 GB + 6 GB
    // headroom > 118 GB working-set recommendation → DISABLED (the live
    // failure this guards: stages materialized to a ~90 MB margin, Metal
    // command buffers died, MLX returned zero logits = fake 100% acceptance)
    try testing.expect(!dsparkFitsBudget(11 * GB, 107 * GB, 118 * GB, 6 * GB));
    // same stages over a roomier trunk quant: 100+11+6 <= 118 → admitted
    try testing.expect(dsparkFitsBudget(11 * GB, 100 * GB, 118 * GB, 6 * GB));
    // exact boundary is admitted
    try testing.expect(dsparkFitsBudget(10 * GB, 102 * GB, 118 * GB, 6 * GB));
    // dead device query (max_rec 0) declines the guess and admits
    try testing.expect(dsparkFitsBudget(11 * GB, 107 * GB, 0, 6 * GB));
}

test "dsv4: DSpark is OPT-IN — stages stay off (and lazy) without the flag (DSV4_MINI)" {
    // Default serve must not pay the ~11 GB stage footprint: no env/--dspark
    // → n_mtp 0, no dspark decode state, serial-only. Explicit opt-in arms it.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    _ = unsetenv("MLX_SERVE_DSV4_DSPARK");
    {
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();
        try testing.expectEqual(@as(usize, 0), mdl.n_mtp);
        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        try testing.expect(st.dspark == null);
    }
    testEnableDspark();
    {
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();
        try testing.expectEqual(@as(usize, 3), mdl.n_mtp);
    }
}

test "dsv4: toHostF32 reads non-contiguous f32 views in logical order" {
    // astype-to-same-dtype is an MLX no-op VIEW, so a broadcast/strided f32
    // input used to hand the raw (smaller) buffer to the memcpy — an overread
    // that read stale pool bytes (nondeterministic garbage) or a foreign
    // stack guard (SIGBUS, live 2026-07-31 in the dspark trace).
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // broadcast [1,3] -> [2,3]: buffer holds 3 floats, logical is 6
    const vals = [_]f32{ 1, 2, 3 };
    const rshape = [_]c_int{ 1, 3 };
    const row = mlx.mlx_array_new_data(&vals, &rshape, 2, .float32);
    defer _ = mlx.mlx_array_free(row);
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    const bshape = [_]c_int{ 2, 3 };
    try mlx.check(mlx.mlx_broadcast_to(&bc, row, &bshape, 2, s));
    const got = try toHostF32(testing.allocator, bc, 6, s);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 1, 2, 3 }, got);
    // column slice of a [2,4]: strided rows, logical {2,3,6,7}
    const m4 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const mshape = [_]c_int{ 2, 4 };
    const mat = mlx.mlx_array_new_data(&m4, &mshape, 2, .float32);
    defer _ = mlx.mlx_array_free(mat);
    const sl = try gpuSliceCols(mat, 2, 1, 3, s);
    defer _ = mlx.mlx_array_free(sl);
    const got2 = try toHostF32(testing.allocator, sl, 4, s);
    defer testing.allocator.free(got2);
    try testing.expectEqualSlices(f32, &[_]f32{ 2, 3, 6, 7 }, got2);
}

test "dsv4: decode-state snapshot/restore replays bit-identically (DSV4_MINI)" {
    testEnableDspark();
    // DSpark verify prerequisite: the draft/verify loop appends speculative
    // tokens into the module-owned rings/caches and must be able to roll
    // back a rejected tail. The bar is BIT identity — argmax agreement would
    // hide a ring off-by-one that only flips outputs near a window/ratio
    // boundary. Shape: prefill, decode N, snapshot, decode a DIVERGENT tail
    // (crossing ratio-4 boundaries so the pending rings + caches are
    // clobbered), restore, replay a control tail and compare against the
    // same tail decoded straight through — restoring TWICE also proves
    // restore never consumes the snapshot.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        var rng = std.Random.DefaultPrng.init(23);
        var ids: [20]u32 = undefined;
        for (&ids) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));
        var alt: [5]u32 = undefined;
        for (&alt) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        {
            const pl = try prefillIntoState(&mdl, allocator, &st, ids[0..6]);
            allocator.free(pl);
        }
        for (ids[6..10]) |id| {
            const l = try decodeStep(&mdl, allocator, &st, id);
            allocator.free(l);
        }

        var snap = try snapshotDecodeState(&st, allocator);
        defer snap.deinit();

        // Control: the un-diverged continuation, recorded per step.
        var control = std.ArrayList([]f32).empty;
        defer {
            for (control.items) |l| allocator.free(l);
            control.deinit(allocator);
        }
        for (ids[10..14]) |id| {
            try control.append(allocator, try decodeStep(&mdl, allocator, &st, id));
        }

        // Diverge: different tokens, crossing the n=12 ratio-4 boundary so
        // the pending rings, caches AND GpuRows all move past the snapshot.
        restoreDecodeState(&st, &snap);
        try testing.expectEqual(@as(usize, 10), st.n);
        for (alt) |id| {
            const l = try decodeStep(&mdl, allocator, &st, id);
            allocator.free(l);
        }

        // Restore again and replay the control tail: bit-identical.
        restoreDecodeState(&st, &snap);
        for (ids[10..14], 0..) |id, k| {
            const l = try decodeStep(&mdl, allocator, &st, id);
            defer allocator.free(l);
            for (l) |v| try testing.expect(std.math.isFinite(v));
            try testing.expectEqualSlices(f32, control.items[k], l);
        }
        std.debug.print("dsv4 snapshot/restore (gpu={}): bit-identical replay over 4 steps\n", .{use_gpu});
    }
}

test "dsv4: a chunk only stalls for layers whose window closes inside it" {
    // The batched path syncs each layer's compressor inputs to the host, and
    // that read is a GPU BARRIER — 41 of them per forward, measured at 128 ms
    // of a 143 ms verify. It is only owed when the chunk closes a window,
    // because only then is a slot emitted that this same chunk's later
    // tokens can see. Boundaries are pure position arithmetic (no data
    // dependency), so the decision is exact, not heuristic.
    try testing.expect(chunkCrossesBoundary(6, 6, 4)); // 7..12 closes 8 and 12
    try testing.expect(!chunkCrossesBoundary(6, 6, 128)); // nothing closes
    try testing.expect(chunkCrossesBoundary(127, 1, 128)); // pos 127 closes it
    try testing.expect(!chunkCrossesBoundary(128, 1, 128));
    try testing.expect(chunkCrossesBoundary(0, 4, 4));
    try testing.expect(!chunkCrossesBoundary(0, 3, 4));
    try testing.expect(!chunkCrossesBoundary(4, 3, 4)); // 5,6,7 -> closes at 8
    try testing.expect(chunkCrossesBoundary(4, 4, 4));
}

test "dsv4: deferring a chunk's compressor read changes nothing (DSV4_MINI)" {
    // The deferral is a PIPELINE decision, never a numeric one: the same host
    // pushes run, just after the layer loop instead of inside it. Logits and
    // the state both sides leave behind must be bit-identical — and this runs
    // a chunk that closes a window on the ratio-4 layers while closing none
    // on the ratio-16 ones, so both arms of the predicate fire in one chunk.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const saved = comp_defer_state;
    defer comp_defer_state = saved;

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        var rng = std.Random.DefaultPrng.init(31337);
        var ids: [16]u32 = undefined;
        for (&ids) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

        var out: [2][]f32 = undefined;
        for ([_]bool{ true, false }, 0..) |defer_on, arm| {
            comp_defer_state = defer_on;
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            {
                const pl = try prefillIntoState(&mdl, allocator, &st, ids[0..6]);
                allocator.free(pl);
            }
            {
                const cl = try extendState(&mdl, allocator, &st, ids[6..12]);
                allocator.free(cl);
            }
            // the tail decode proves the STATE both arms left, not just logits
            const l = try decodeStep(&mdl, allocator, &st, ids[12]);
            out[arm] = l;
        }
        defer for (out) |l| allocator.free(l);
        for (out[0]) |v| try testing.expect(std.math.isFinite(v));
        try testing.expectEqualSlices(f32, out[1], out[0]);
        std.debug.print("dsv4 compressor deferral (gpu={}): bit-identical vs in-layer\n", .{use_gpu});
    }
}

test "dsv4: the sinkhorn config cache keys on the token count it bakes" {
    // The config bakes an output SHAPE, a grid and a threadgroup, all keyed
    // by the token count — the house rule is that such a cache keys on the
    // full shape, never a product, so a [C=6] verify can never be handed the
    // [C=1] decode config. Small repeating widths (spec + decode) and the
    // prefill sub-chunk are cached; one-off remainders are not, so a long
    // session cannot accumulate configs.
    try testing.expect(sinkhornCfgCacheable(1));
    try testing.expect(sinkhornCfgCacheable(6));
    try testing.expect(sinkhornCfgCacheable(SINK_CFG_MAX));
    try testing.expect(sinkhornCfgCacheable(PREFILL_SUB));
    try testing.expect(!sinkhornCfgCacheable(SINK_CFG_MAX + 1));
    try testing.expect(!sinkhornCfgCacheable(PREFILL_SUB - 1));
    try testing.expect(!sinkhornCfgCacheable(0));
}

test "dsv4: only prefill-width chunks return their transients to the OS" {
    // The clear exists for shapes that NEVER repeat (a prompt's sub-chunks
    // vary with its length). A DSpark round's widths are ≤ block+1 and repeat
    // every round, so clearing after one makes the NEXT round re-allocate its
    // ~340 MB of gather transients from the OS — the pool doing its job is
    // the whole point. Boundary = the house "a multi-token forward is not a
    // prefill" line (seq ≥ 32).
    try testing.expect(extendChunkShouldClearCache(PREFILL_SUB));
    try testing.expect(extendChunkShouldClearCache(64));
    try testing.expect(extendChunkShouldClearCache(32));
    try testing.expect(!extendChunkShouldClearCache(31));
    try testing.expect(!extendChunkShouldClearCache(16)); // max verify width
    try testing.expect(!extendChunkShouldClearCache(6)); // block_size+1
    try testing.expect(!extendChunkShouldClearCache(1));
}

test "dsv4: DSpark profile attributes round cost per committed token" {
    // The cost audit's arbiter: a round pays only when its ms-per-COMMITTED
    // token beats the serial step. Two rounds, one partial (a rollback) and
    // one full accept (none), so the rollback share is a real average.
    const ms = std.time.ns_per_ms;
    var p = DsparkProfile{};
    p.observe(.{
        .draft_ns = 30 * ms,
        .markov_ns = 10 * ms,
        .snapshot_ns = 2 * ms,
        .verify_ns = 100 * ms,
        .rollback_ns = 68 * ms,
        .accepted = 2,
        .committed = 3,
    });
    p.observe(.{
        .draft_ns = 30 * ms,
        .markov_ns = 10 * ms,
        .snapshot_ns = 2 * ms,
        .verify_ns = 100 * ms,
        .rollback_ns = 0,
        .accepted = 5,
        .committed = 6,
    });
    const s = p.summary();
    try testing.expectApproxEqAbs(@as(f64, 166), s.round_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 30), s.draft_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 10), s.markov_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 34), s.rollback_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 3.5), s.accepts_per_round, 1e-6);
    // 332 ms of round cost bought 9 committed tokens.
    try testing.expectApproxEqAbs(@as(f64, 332.0 / 9.0), s.ms_per_token, 1e-6);
    // An empty profile reports zeros rather than dividing by zero.
    const empty = (DsparkProfile{}).summary();
    try testing.expectEqual(@as(f64, 0), empty.ms_per_token);
    try testing.expectEqual(@as(f64, 0), empty.round_ms);
}

test "dsv4: main_hidden capture matches the oracle across both prefill and decode (DSV4_MINI)" {
    testEnableDspark();
    // DSpark conditioning: the trunk must capture the hc-averaged stream at
    // the target layers, concatenated — through BOTH mutation paths (the
    // batched chunk prefill at i=6 and token-by-token decode at i=9), against
    // the oracle's dspark.{i}.main_hidden_last fixtures. Ring bookkeeping is
    // asserted alongside: each stage's main_kv ring holds one row per
    // position seen.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const readAll = struct {
        fn f(io_: std.Io, alloc: std.mem.Allocator, p: []const u8) ![]u8 {
            const file = try std.Io.Dir.openFileAbsolute(io_, p, .{});
            defer file.close(io_);
            var rb: [4096]u8 = undefined;
            var rs = file.reader(io_, &rb);
            return try rs.interface.allocRemaining(alloc, .limited(1 << 26));
        }
    }.f;
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const cfg_json = try readAll(io, allocator, cfg_path);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const fx_path = try std.fmt.allocPrint(allocator, "{s}/fixtures.json", .{path});
    defer allocator.free(fx_path);
    const fx_json = readAll(io, allocator, fx_path) catch return; // fixtures not dumped -> skip
    defer allocator.free(fx_json);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, fx_json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    if (root.get("dspark.6.main_hidden_last") == null) return; // pre-DSpark fixtures

    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const ids_v = root.get("input_ids").?.array.items;
    const ids = try allocator.alloc(u32, ids_v.len);
    defer allocator.free(ids);
    for (ids, ids_v) |*o, v| o.* = @intCast(v.integer);

    const checkMh = struct {
        fn f(m: *Dsv4Model, st_: *Dsv4DecodeState, root_: anytype, alloc: std.mem.Allocator, comptime key: []const u8, floor: f64) !void {
            const want_v = root_.get(key).?.array.items;
            const ds = &st_.dspark.?;
            try testing.expect(ds.has_mh);
            const got = try toHostF32(alloc, ds.mh_last, want_v.len, m.s);
            defer alloc.free(got);
            var dot: f64 = 0;
            var na: f64 = 0;
            var nb: f64 = 0;
            for (got, want_v) |g, wv| {
                const wf: f64 = switch (wv) {
                    .float => |x| x,
                    .integer => |x| @floatFromInt(x),
                    else => return error.BadFixture,
                };
                try testing.expect(std.math.isFinite(g));
                dot += @as(f64, g) * wf;
                na += @as(f64, g) * g;
                nb += wf * wf;
            }
            const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
            if (cos < floor) {
                // per-target cosines localize which capture layer is off
                const half = want_v.len / 2;
                var stats = [2][3]f64{ .{ 0, 0, 0 }, .{ 0, 0, 0 } };
                for (got, want_v, 0..) |g, wv, i| {
                    const wf: f64 = switch (wv) {
                        .float => |x| x,
                        .integer => |x| @floatFromInt(x),
                        else => 0,
                    };
                    const sl = &stats[i / half];
                    sl[0] += @as(f64, g) * wf;
                    sl[1] += @as(f64, g) * g;
                    sl[2] += wf * wf;
                }
                std.debug.print("main_hidden {s}: cos={d:.6} (floor {d:.2}) t0={d:.6} t1={d:.6}\n", .{
                    key,                                                cos,
                    floor,                                              stats[0][0] / (@sqrt(stats[0][1]) * @sqrt(stats[0][2]) + 1e-30),
                    stats[1][0] / (@sqrt(stats[1][1]) * @sqrt(stats[1][2]) + 1e-30),
                });
                for (0..6) |i| {
                    const wf: f64 = switch (want_v[i]) {
                        .float => |x| x,
                        .integer => |x| @floatFromInt(x),
                        else => 0,
                    };
                    std.debug.print("  [{d}] got={d:.6} want={d:.6}\n", .{ i, got[i], wf });
                }
                try testing.expect(false);
            }
        }
    }.f;

    // Same two-stream split as the decode-equivalence gate: on the CPU
    // stream prefill and decode hit the same gemm, so the capture must agree
    // with the stateless oracle STRICTLY (cos ≥ 0.99 — a wiring bug screams
    // here); the GPU stream's [C,·]-vs-[1,·] kernel-choice drift compounds
    // through the raw residual stream (the sanctioned qmv-vs-qmm class), so
    // the decode-path floor is 0.95 with the trunk logits arbitrated by the
    // sibling test.
    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();
        try testing.expectEqual(@as(usize, 3), mdl.n_mtp);

        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        try testing.expect(st.dspark != null);

        // i=6 via the batched chunk prefill
        {
            const pl = try prefillIntoState(&mdl, allocator, &st, ids[0..7]);
            allocator.free(pl);
        }
        try checkMh(&mdl, &st, root, allocator, "dspark.6.main_hidden_last", 0.99);
        for (st.dspark.?.main_kv) |*ring| try testing.expectEqual(@as(usize, 7), ring.used);

        // i=9 via token-by-token decode. Floor 0.95: past the window wrap the
        // raw residual stream drifts more against the python oracle than the
        // LOGITS do (the s=17 logits fixture gate itself reads ~0.9938 on
        // this random-weight mini — sims + indexer top-k are discrete and
        // near-ties flip) — measured identically through a fresh one-chunk
        // prefill of the same 10 ids, so it is engine-vs-oracle drift, not a
        // decode-path capture bug; engine decode logits vs the engine's own
        // stateless forward stay at cos ≈ 1.0 here. A wiring bug (wrong
        // axis/layer) reads far below any of this.
        for (ids[7..10]) |id| {
            const l = try decodeStep(&mdl, allocator, &st, id);
            allocator.free(l);
        }
        try checkMh(&mdl, &st, root, allocator, "dspark.9.main_hidden_last", 0.95);
        for (st.dspark.?.main_kv) |*ring| try testing.expectEqual(@as(usize, 10), ring.used);
        std.debug.print("dsv4 main_hidden capture (gpu={}): OK at i=6 (chunk) and i=9 (decode)\n", .{use_gpu});
    }
}

test "dsv4: DSpark draft matches the oracle (DSV4_MINI)" {
    testEnableDspark();
    // The whole draft path — stage-0 conditioning through the rings, 3
    // block-parallel stages, last-stage hc collapse, shared head, sequential
    // Markov bias + greedy sample, confidence — against the oracle's
    // dspark.{i}.{out_ids,logits,confidence} fixtures at ring-not-full (6),
    // just-past-window-wrap (9) and a deeper prefix (12). CPU strict (ids
    // exact, logits cos ≥ 0.99); GPU on the random-weight mini tolerates the
    // kernel-choice near-tie class (cos ≥ 0.9, first draft position's logits
    // argmax free to flip). The draft must never MUTATE the decode state.
    // The gate is held OPEN here: the reference drafts the whole block
    // unconditionally, so a truncated block would be comparing against a
    // different computation (the gate's own contract is pinned separately).
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const readAll = struct {
        fn f(io_: std.Io, alloc: std.mem.Allocator, p: []const u8) ![]u8 {
            const file = try std.Io.Dir.openFileAbsolute(io_, p, .{});
            defer file.close(io_);
            var rb: [4096]u8 = undefined;
            var rs = file.reader(io_, &rb);
            return try rs.interface.allocRemaining(alloc, .limited(1 << 26));
        }
    }.f;
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const cfg_json = try readAll(io, allocator, cfg_path);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const fx_path = try std.fmt.allocPrint(allocator, "{s}/fixtures.json", .{path});
    defer allocator.free(fx_path);
    const fx_json = readAll(io, allocator, fx_path) catch return;
    defer allocator.free(fx_json);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, fx_json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    if (root.get("dspark.6.out_ids") == null) return;

    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const ids_v = root.get("input_ids").?.array.items;
    const ids = try allocator.alloc(u32, ids_v.len);
    defer allocator.free(ids);
    for (ids, ids_v) |*o, v| o.* = @intCast(v.integer);

    const jf = struct {
        fn f(v: std.json.Value) f64 {
            return switch (v) {
                .float => |x| x,
                .integer => |x| @floatFromInt(x),
                else => 0,
            };
        }
    }.f;

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        inline for ([_]usize{ 6, 9, 12 }) |i| {
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            {
                const pl = try prefillIntoState(&mdl, allocator, &st, ids[0 .. i + 1]);
                allocator.free(pl);
            }
            const key_tok = std.fmt.comptimePrint("dspark.{d}.trunk_tok", .{i});
            const trunk_tok: u32 = @intCast(root.get(key_tok).?.integer);
            mdl.ds_conf_thr = -std.math.inf(f32); // gate open — see the header
            const n_before = st.n;
            const ring_before = st.dspark.?.main_kv[0].used;

            var draft = try dsparkDraft(&mdl, allocator, &st, trunk_tok);
            defer draft.deinit(allocator);

            // the draft READS the state, never mutates it
            try testing.expectEqual(n_before, st.n);
            try testing.expectEqual(ring_before, st.dspark.?.main_kv[0].used);

            const want_ids = root.get(std.fmt.comptimePrint("dspark.{d}.out_ids", .{i})).?.array.items;
            const want_log = root.get(std.fmt.comptimePrint("dspark.{d}.logits", .{i})).?.array.items;
            const want_conf = root.get(std.fmt.comptimePrint("dspark.{d}.confidence", .{i})).?.array.items;
            try testing.expectEqual(want_ids.len, draft.ids.len);
            try testing.expectEqual(trunk_tok, draft.ids[0]);

            // Strictness ladder: at i=6 (ring pre-window-wrap, minimal state
            // drift) the CPU arm pins the whole pipeline — cos ≥ 0.99, ids
            // EXACT, confidence within 0.1 (it reads the UN-normalized
            // hc-collapsed stream, so drift the rms-normed logits cancel
            // surfaces directly; 0.1 is a wiring gate, not a numerics pin).
            // Past the wrap (9/12) the draft INHERITS the trunk state's
            // engine-vs-oracle drift through the main_kv rings (the capture
            // gate measures that stream at cos ~0.97 there), so the floor
            // drops to 0.95 — still far above any wiring bug. GPU adds the
            // kernel-choice near-tie class on top: 0.9. The real-model
            // acceptance floor arbitrates end-to-end.
            const strict = (i == 6) and !use_gpu;
            const B = draft.ids.len - 1;
            var pos_ok: usize = 0;
            for (0..B) |b| {
                const row = draft.logits[b * mdl.vocab ..][0..mdl.vocab];
                const wrow = want_log[b].array.items;
                var dot: f64 = 0;
                var na: f64 = 0;
                var nb: f64 = 0;
                for (row, wrow) |g, wv| {
                    try testing.expect(std.math.isFinite(g));
                    const wf = jf(wv);
                    dot += @as(f64, g) * wf;
                    na += @as(f64, g) * g;
                    nb += wf * wf;
                }
                const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
                const floor: f64 = if (use_gpu) 0.9 else if (strict) 0.99 else 0.95;
                if (cos < floor) {
                    std.debug.print("dspark draft i={d} pos {d}: cos={d:.6}\n", .{ i, b, cos });
                    try testing.expect(false);
                }
                if (draft.ids[b + 1] == @as(u32, @intCast(want_ids[b + 1].integer))) pos_ok += 1;
            }
            if (strict) {
                try testing.expectEqual(B, pos_ok);
                for (0..B) |b| {
                    const wc = jf(want_conf[b]);
                    if (@abs(@as(f64, draft.confidence[b]) - wc) > 0.1) {
                        std.debug.print("dspark conf i={d} pos {d}: got={d:.5} want={d:.5}\n", .{ i, b, draft.confidence[b], wc });
                        try testing.expect(false);
                    }
                }
            }
            for (draft.confidence) |cv| try testing.expect(std.math.isFinite(cv));
        }
        std.debug.print("dsv4 DSpark draft parity (gpu={}): OK at i=6/9/12\n", .{use_gpu});
    }
}

test "dsv4: extendStateAllLogits rows match serial decode at every position (DSV4_MINI)" {
    testEnableDspark();
    // The verify primitive's PER-ROW contract: L[k] (logits after verify
    // tokens 0..k) must argmax-agree with the serial decodeStep at the same
    // position — this is exactly what the accept loop trusts, and a wrong
    // later row fakes 100% acceptance of garbage (the live 2026-07-31 bos
    // cascade). Feeds the SERIAL continuation as the verify block so every
    // row is on-distribution.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const argmaxOf = struct {
        fn f(row: []const f32) u32 {
            var am: usize = 0;
            for (row, 0..) |v, j| {
                if (v > row[am]) am = j;
            }
            return @intCast(am);
        }
    }.f;

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        var rng = std.Random.DefaultPrng.init(77);
        var prefix: [10]u32 = undefined;
        for (&prefix) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));
        const B = 6;

        // serial: record the greedy continuation + per-step argmaxes
        var chain: [B + 1]u32 = undefined; // t1 .. t_{B+1}
        {
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
            defer allocator.free(pl);
            chain[0] = argmaxOf(pl);
            for (0..B) |k| {
                const l = try decodeStep(&mdl, allocator, &st, chain[k]);
                defer allocator.free(l);
                chain[k + 1] = argmaxOf(l);
            }
        }

        // verify block = the serial chain's first B tokens: L[k] must argmax
        // to chain[k+1] for every k.
        {
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
            defer allocator.free(pl);
            const vl = try extendStateAllLogits(&mdl, allocator, &st, chain[0..B]);
            defer allocator.free(vl);
            var agree: usize = 0;
            for (0..B) |k| {
                const got = argmaxOf(vl[k * mdl.vocab ..][0..mdl.vocab]);
                if (got == chain[k + 1]) agree += 1 else if (!use_gpu) {
                    std.debug.print("allLogits row {d} (gpu={}): got {d} want {d}\n", .{ k, use_gpu, got, chain[k + 1] });
                }
            }
            if (!use_gpu) {
                try testing.expectEqual(@as(usize, B), agree);
            } else {
                try testing.expect(agree * 10 >= B * 8);
            }
            std.debug.print("dsv4 extendStateAllLogits rows (gpu={}): {d}/{d} argmax agree\n", .{ use_gpu, agree, B });
        }
    }
}

test "dsv4: extendChunk .all_gpu + host read matches .all_host bytes (DSV4_MINI)" {
    testEnableDspark();
    // Pins the mode-enum refactor directly: `.all_host` is `.all_gpu` +
    // toHostF32 by construction (headLogitsBatchGpu wraps headLogitsBatchG),
    // so the two must agree BYTE-for-byte from the same entry state — same
    // graph, same kernels, only who owns the sync differs. Runs both arms
    // from one snapshot so state bytes are identical by construction.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        var rng = std.Random.DefaultPrng.init(311);
        var prefix: [10]u32 = undefined;
        for (&prefix) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));
        var block: [5]u32 = undefined;
        for (&block) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
        defer allocator.free(pl);
        var snap = try snapshotDecodeState(&st, allocator);
        defer snap.deinit();

        const vl_host = try extendStateAllLogits(&mdl, allocator, &st, &block);
        defer allocator.free(vl_host);

        restoreDecodeState(&st, &snap);
        const vl_g = try extendChunk(&mdl, allocator, &st, &block, .all_gpu);
        defer _ = mlx.mlx_array_free(vl_g);
        const vl_read = try toHostF32(allocator, vl_g, block.len * mdl.vocab, mdl.s);
        defer allocator.free(vl_read);

        try testing.expectEqualSlices(f32, vl_host, vl_read);
        std.debug.print("dsv4 extendChunk all_gpu==all_host (gpu={}): {d} logits byte-equal\n", .{ use_gpu, vl_host.len });
    }
}

test "dsv4: dsparkRound greedy-equivalence with serial decode (DSV4_MINI)" {
    testEnableDspark();
    // The verify/accept loop end-to-end: from the same prefix, rounds of
    // draft→batch-verify→rollback must emit the SAME greedy sequence as
    // plain token-by-token decode. CPU strict (batch and single-token hit
    // the same gemm); the GPU arm tolerates the [C,·]-vs-[1,·] kernel-choice
    // near-tie class on the random mini (first divergence ≥ 4, the wiring
    // floor). Per-round invariants hold on both streams: st.n advances by
    // exactly tokens.len, accepted ≤ block, tokens[0] == the round's t1.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const argmaxOf = struct {
        fn f(row: []const f32) u32 {
            var am: usize = 0;
            for (row, 0..) |v, j| {
                if (v > row[am]) am = j;
            }
            return @intCast(am);
        }
    }.f;

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();

        var rng = std.Random.DefaultPrng.init(41);
        var prefix: [10]u32 = undefined;
        for (&prefix) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));
        const T = 12; // generated tokens compared

        // serial arm
        var serial: [T]u32 = undefined;
        {
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
            defer allocator.free(pl);
            var t: u32 = argmaxOf(pl);
            for (0..T) |k| {
                serial[k] = t;
                const l = try decodeStep(&mdl, allocator, &st, t);
                defer allocator.free(l);
                t = argmaxOf(l);
            }
        }

        // dspark arm
        var spec: [T + 16]u32 = undefined;
        var n_spec: usize = 0;
        var total_accepted: usize = 0;
        var rounds: usize = 0;
        {
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
            defer allocator.free(pl);
            var t: u32 = argmaxOf(pl);
            while (n_spec < T) {
                const n_before = st.n;
                var round = try dsparkRound(&mdl, allocator, &st, t, std.math.maxInt(usize));
                defer round.deinit(allocator);
                try testing.expectEqual(t, round.tokens[0]);
                try testing.expect(round.accepted <= mdl.ds_block);
                try testing.expectEqual(n_before + round.tokens.len, st.n);
                for (round.tokens) |tokv| {
                    if (n_spec < spec.len) {
                        spec[n_spec] = tokv;
                        n_spec += 1;
                    }
                }
                total_accepted += round.accepted;
                rounds += 1;
                t = round.next_token;
            }
        }

        var first_div: usize = T;
        for (0..T) |k| {
            if (serial[k] != spec[k]) {
                first_div = k;
                break;
            }
        }
        if (!use_gpu) {
            if (first_div != T) {
                std.debug.print("dsparkRound serial={any} spec={any}\n", .{ serial, spec[0..T] });
                try testing.expect(false);
            }
        } else {
            try testing.expect(first_div >= 4);
        }
        std.debug.print("dsv4 dsparkRound equivalence (gpu={}): first_div={d}/{d}, {d} rounds, {d} drafts accepted\n", .{ use_gpu, first_div, T, rounds, total_accepted });
    }
}

test "dsv4: the confidence gate truncates the block without changing tokens (DSV4_MINI)" {
    testEnableDspark();
    // The checkpoint's own confidence head says whether a drafted position is
    // worth VERIFYING — and verify width is ~85% of a round's cost, so a
    // block the model has no confidence in is pure waste. Gating may only
    // change how many drafts are SUBMITTED: every committed token still comes
    // from the trunk's own argmax, so the emitted sequence is invariant.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const argmaxOf = struct {
        fn f(row: []const f32) u32 {
            var am: usize = 0;
            for (row, 0..) |v, j| {
                if (v > row[am]) am = j;
            }
            return @intCast(am);
        }
    }.f;

    const dw = try loadDsv4Weights(allocator, &cfg, &weights);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var mdl = try initModel(allocator, &cfg, dw, s);
    defer mdl.deinit();
    const B: usize = mdl.ds_block;

    var rng = std.Random.DefaultPrng.init(4242);
    var prefix: [10]u32 = undefined;
    for (&prefix) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

    // serial reference chain
    const T = 6;
    var chain: [T + 2]u32 = undefined;
    {
        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
        defer allocator.free(pl);
        var t: u32 = argmaxOf(pl);
        for (0..T + 2) |k| {
            chain[k] = t;
            const l = try decodeStep(&mdl, allocator, &st, t);
            defer allocator.free(l);
            t = argmaxOf(l);
        }
    }

    // gate WIDE OPEN: the full block is drafted and submitted.
    {
        mdl.ds_conf_thr = -std.math.inf(f32);
        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
        allocator.free(pl);
        var draft = try dsparkDraft(&mdl, allocator, &st, chain[0]);
        defer draft.deinit(allocator);
        try testing.expectEqual(B, draft.len);
    }

    // gate SHUT: nothing is submitted, so a round costs one verified token —
    // and that token is still exactly the serial continuation.
    mdl.ds_conf_thr = std.math.inf(f32);
    var st = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st);
    const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
    allocator.free(pl);
    {
        var draft = try dsparkDraft(&mdl, allocator, &st, chain[0]);
        defer draft.deinit(allocator);
        try testing.expectEqual(@as(usize, 0), draft.len);
    }
    var t: u32 = chain[0];
    var emitted = std.ArrayList(u32).empty;
    defer emitted.deinit(allocator);
    var guard: usize = 0;
    while (emitted.items.len < T and guard < 32) : (guard += 1) {
        var round = try dsparkRound(&mdl, allocator, &st, t, std.math.maxInt(usize));
        defer round.deinit(allocator);
        try testing.expectEqual(@as(u32, 0), round.accepted); // gate shut
        try emitted.appendSlice(allocator, round.tokens);
        t = round.next_token;
    }
    try testing.expectEqualSlices(u32, chain[0..T], emitted.items[0..T]);
    mdl.ds_conf_thr = 0;
    std.debug.print("dsv4 confidence gate: shut -> {d} serial-identical tokens, open -> block {d}\n", .{ T, B });
}

test "dsv4: anchored rollback replays like snapshot + re-extend (DSV4_MINI)" {
    testEnableDspark();
    // A partial accept must leave EXACTLY the state that restoring the entry
    // snapshot and re-forwarding the accepted prefix leaves — and that
    // re-forward is the round's dominant waste (measured ~65 ms of a ~195 ms
    // partial round on the real mirror: it re-runs the whole trunk over
    // tokens the verify already ran). Per-position anchors replace it with a
    // truncate. Reference arm = the old strategy; candidate = the anchors;
    // both then decode a tail that must agree BIT-for-bit.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    for ([_]bool{ false, true }) |use_gpu| {
        if (use_gpu and mlx.noGpuBackend()) continue;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = if (use_gpu) mlx.gpuStream() else mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();
        const B: usize = mdl.ds_block;

        var rng = std.Random.DefaultPrng.init(1729);
        var ids: [11]u32 = undefined;
        for (&ids) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));
        var block: [8]u32 = undefined;
        for (block[0 .. B + 1]) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));
        const TAIL = 3;

        // Every acceptance count, so at least one crosses the ratio-4 window
        // boundary the pending rings shift on.
        for (0..B) |accepted| {
            var st = try initDecodeState(&mdl, allocator);
            defer deinitDecodeState(&st);
            {
                const pl = try prefillIntoState(&mdl, allocator, &st, ids[0..9]);
                allocator.free(pl);
            }
            {
                const l = try decodeStep(&mdl, allocator, &st, ids[9]);
                allocator.free(l);
            }
            var entry = try snapshotDecodeState(&st, allocator);
            defer entry.deinit();

            // reference: verify the block, then restore + re-extend the prefix
            {
                const vl = try extendStateAllLogits(&mdl, allocator, &st, block[0 .. B + 1]);
                allocator.free(vl);
                restoreDecodeState(&st, &entry);
                const rl = try extendState(&mdl, allocator, &st, block[0 .. accepted + 1]);
                allocator.free(rl);
            }
            try testing.expectEqual(entry.n + accepted + 1, st.n);
            var ref = std.ArrayList([]f32).empty;
            defer {
                for (ref.items) |l| allocator.free(l);
                ref.deinit(allocator);
            }
            for (0..TAIL) |k| try ref.append(allocator, try decodeStep(&mdl, allocator, &st, ids[k]));

            // candidate: verify with anchors armed, then truncate to `accepted`
            restoreDecodeState(&st, &entry);
            try armAnchors(&mdl, &st, B);
            {
                const vl = try extendStateAllLogits(&mdl, allocator, &st, block[0 .. B + 1]);
                allocator.free(vl);
                restoreToAnchor(&st, &entry, accepted);
            }
            try testing.expectEqual(entry.n + accepted + 1, st.n);
            for (0..TAIL) |k| {
                const l = try decodeStep(&mdl, allocator, &st, ids[k]);
                defer allocator.free(l);
                for (l) |v| try testing.expect(std.math.isFinite(v));
                try testing.expectEqualSlices(f32, ref.items[k], l);
            }
        }
        std.debug.print("dsv4 anchored rollback (gpu={}): {d} acceptance counts replay bit-identically\n", .{ use_gpu, B });
    }
}

test "dsv4: dsparkRound FULL-ACCEPT commits the block and leaves serial state (DSV4_MINI)" {
    testEnableDspark();
    // The random mini's drafts never match, so the equivalence test above
    // only ever exercises the partial-accept/rollback branch. This rigs the
    // draft to the TRUE serial continuation (dsparkRoundWith seam) so verify
    // accepts the whole block: no restore runs, the verify extendChunk's
    // appends ARE the state — and continuing serially afterwards must match
    // a pure-serial run bit-for-bit (CPU strict; the branch is stream-
    // independent Zig, and CPU is where batched==serial argmax is pinned).
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const argmaxOf = struct {
        fn f(row: []const f32) u32 {
            var am: usize = 0;
            for (row, 0..) |v, j| {
                if (v > row[am]) am = j;
            }
            return @intCast(am);
        }
    }.f;

    const dw = try loadDsv4Weights(allocator, &cfg, &weights);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var mdl = try initModel(allocator, &cfg, dw, s);
    defer mdl.deinit();
    const B: usize = mdl.ds_block;
    const TAIL = 3; // serial tokens decoded after the round

    var rng = std.Random.DefaultPrng.init(97);
    var prefix: [10]u32 = undefined;
    for (&prefix) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

    // serial arm: chain[0]=t1 then B+1+TAIL more greedy continuations
    var chain: [24]u32 = undefined;
    {
        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
        defer allocator.free(pl);
        var t: u32 = argmaxOf(pl);
        for (0..B + 2 + TAIL) |k| {
            chain[k] = t;
            const l = try decodeStep(&mdl, allocator, &st, t);
            defer allocator.free(l);
            t = argmaxOf(l);
        }
    }

    // dspark arm with the rigged draft = the serial continuation
    var st = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st);
    const pl = try prefillIntoState(&mdl, allocator, &st, &prefix);
    defer allocator.free(pl);
    const n_entry = st.n;
    var rigged = DsparkDraft{
        .ids = try allocator.dupe(u32, chain[0 .. B + 1]),
        .len = B,
        .logits = try allocator.alloc(f32, B * mdl.vocab),
        .confidence = try allocator.alloc(f32, B),
    };
    @memset(rigged.logits, 0);
    @memset(rigged.confidence, 0);
    defer rigged.deinit(allocator);
    var round = try dsparkRoundWith(&mdl, allocator, &st, chain[0], &rigged, B);
    defer round.deinit(allocator);
    try testing.expectEqual(@as(u32, @intCast(B)), round.accepted); // FULL accept
    try testing.expectEqual(@as(usize, B + 1), round.tokens.len);
    try testing.expectEqualSlices(u32, chain[0 .. B + 1], round.tokens);
    try testing.expectEqual(chain[B + 1], round.next_token); // the bonus token
    try testing.expectEqual(n_entry + B + 1, st.n);
    // state integrity: the committed block must continue exactly like serial
    var t: u32 = round.next_token;
    for (0..TAIL) |k| {
        try testing.expectEqual(chain[B + 1 + k], t);
        const l = try decodeStep(&mdl, allocator, &st, t);
        defer allocator.free(l);
        t = argmaxOf(l);
    }
    std.debug.print("dsv4 dsparkRound FULL-ACCEPT: {d}/{d} accepted, {d} serial tail tokens match\n", .{ round.accepted, B, TAIL });

    // Token-budget cap: the same fully-accepted proposal must roll module
    // state back to the capped prefix and choose its pending token at that
    // boundary. Slicing round.tokens after return would leave st.n overrun.
    var capped_st = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&capped_st);
    const capped_pl = try prefillIntoState(&mdl, allocator, &capped_st, &prefix);
    defer allocator.free(capped_pl);
    const capped_entry = capped_st.n;
    var capped = try dsparkRoundWith(&mdl, allocator, &capped_st, chain[0], &rigged, 2);
    defer capped.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), capped.accepted);
    try testing.expectEqualSlices(u32, chain[0..3], capped.tokens);
    try testing.expectEqual(chain[3], capped.next_token);
    try testing.expectEqual(capped_entry + 3, capped_st.n);
}

// ── incremental decode ─────────────────────────────────────────────────
//
// Transcribes the reference's start_pos>0 branches: per-layer raw-kv buffer
// (FULL, house sliding convention), append-only compressed caches, and the
// compressor pending-window rings (kv_state/score_state — overlap keeps the
// previous window's first-half dims in rows [0, ratio)). Equivalence with the
// stateless full re-forward is pinned by the DSV4_MINI test above.
//
// PERF NOTE (v0): still host-centric; wo_a is re-dequantized per step. The
// GPU-graph migration replaces the internals behind decodeStep's seam.

/// GPU-resident growing row buffer ([cap, width] f32; used rows tracked
/// host-side). Growth is proportional (+25%, KVCache policy) so appends don't
/// strand a full copy every row.
pub const GpuRows = struct {
    buf: mlx.mlx_array,
    used: usize,
    cap: usize,
    width: usize,

    pub fn init(width: usize) GpuRows {
        return .{ .buf = mlx.mlx_array_new(), .used = 0, .cap = 0, .width = width };
    }

    pub fn deinit(self: *GpuRows) void {
        _ = mlx.mlx_array_free(self.buf);
    }

    fn ensure(self: *GpuRows, need: usize, s: mlx.mlx_stream) !void {
        if (need <= self.cap) return;
        const new_cap = @max(need, @max(64, self.cap + self.cap / 4));
        const shape = [_]c_int{ @intCast(new_cap), @intCast(self.width) };
        var zeros = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zeros);
        try mlx.check(mlx.mlx_zeros(&zeros, &shape, 2, .float32, s));
        if (self.used > 0) {
            const start = [_]c_int{ 0, 0 };
            const stop = [_]c_int{ @intCast(self.used), @intCast(self.width) };
            const strides = [_]c_int{ 1, 1 };
            var old_view = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(old_view);
            try mlx.check(mlx.mlx_slice(&old_view, self.buf, &start, 2, &stop, 2, &strides, 2, s));
            var updated = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&updated, zeros, old_view, &start, 2, &stop, 2, &strides, 2, s));
            _ = mlx.mlx_array_free(self.buf);
            self.buf = updated;
        } else {
            var copy = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&copy, zeros, .float32, s));
            _ = mlx.mlx_array_free(self.buf);
            self.buf = copy;
        }
        self.cap = new_cap;
    }

    /// Append host rows [n, width].
    pub fn append(self: *GpuRows, rows: []const f32, s: mlx.mlx_stream) !void {
        const n = rows.len / self.width;
        const shape = [_]c_int{ @intCast(n), @intCast(self.width) };
        const up = mlx.mlx_array_new_data(rows.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(up);
        try self.appendGpu(up, n, s);
    }

    /// Append n GPU rows ([n, width] f32 array) — no host hop.
    pub fn appendGpu(self: *GpuRows, rows: mlx.mlx_array, n: usize, s: mlx.mlx_stream) !void {
        try self.ensure(self.used + n, s);
        const start = [_]c_int{ @intCast(self.used), 0 };
        const stop = [_]c_int{ @intCast(self.used + n), @intCast(self.width) };
        const strides = [_]c_int{ 1, 1 };
        var updated = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&updated, self.buf, rows, &start, 2, &stop, 2, &strides, 2, s));
        _ = mlx.mlx_array_free(self.buf);
        self.buf = updated;
        self.used += n;
    }

    /// Rollback to `n` used rows — offset-only and capacity-agnostic
    /// (KVCache.truncate convention). Stale rows past `n` stay in the buffer
    /// and are overwritten by the next append; every reader is bounded by
    /// `used`, never `cap`.
    pub fn truncate(self: *GpuRows, n: usize) void {
        std.debug.assert(n <= self.used);
        self.used = n;
    }

    /// Contiguous row range [lo, hi) as a view.
    pub fn sliceRows(self: *const GpuRows, lo: usize, hi: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const start = [_]c_int{ @intCast(lo), 0 };
        const stop = [_]c_int{ @intCast(hi), @intCast(self.width) };
        const strides = [_]c_int{ 1, 1 };
        var v = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&v, self.buf, &start, 2, &stop, 2, &strides, 2, s));
        return v;
    }
};

pub const CompDecState = struct {
    kv_pend: []f32, // [rows, coff*d] raw wkv outputs (full width)
    sc_pend: []f32, // [rows, coff*d] wgate outputs + ape (full width)
    cache: std.array_list.Managed(f32), // emitted slots [n_slots * d]
    rows: usize,
    width: usize,
};

pub const LayerDecState = struct {
    kv: std.array_list.Managed(f32), // [n * head_dim] post norm+rope+fp8sim (host mirror)
    kv_gpu: GpuRows,
    comp_gpu: GpuRows, // emitted attn-compressor slots
    idx_gpu: GpuRows, // emitted indexer-compressor slots (ratio-4 layers)
    comp: ?CompDecState,
    idx_comp: ?CompDecState,
};

/// DSpark draft-side per-request state. `main_kv[i]` is stage i's window-ring
/// source: one finalized row (kv_norm→rope→fp8sim of wkv(main_x)) per trunk
/// position, append-only FULL history (house sliding convention — the draft
/// attention slices to the last min(window, n) rows), so rollback is a plain
/// truncate. `mh_last` is the last position's main_hidden [1, n_targets*dim]
/// (parity seam — the draft itself reads only the rings; NOT snapshotted,
/// every trunk step overwrites it before any draft runs).
pub const DsparkDecState = struct {
    main_kv: []GpuRows,
    mh_last: mlx.mlx_array,
    has_mh: bool,
};

pub const Dsv4DecodeState = struct {
    layers: []LayerDecState,
    n: usize,
    alloc: std.mem.Allocator,
    dspark: ?DsparkDecState = null,
    /// Per-position rollback anchors, armed only for the duration of a DSpark
    /// verify chunk (null everywhere else — prefill pays one null check per
    /// token per layer). See `DsparkAnchors`.
    anchors: ?DsparkAnchors = null,
    /// Lazy-decode: compressor-input rows whose HOST push is still pending
    /// (position order). Drained by `drainPending` before any consumer of the
    /// host rings runs — the next window-boundary token's emission build, a
    /// prefill/verify chunk, or state teardown.
    pending: std.ArrayList(PendingComp) = .empty,
};

/// One deferred compressor-input row awaiting its host ring push (lazy
/// decode). `arr` is the [1, comp_in_w] GPU row for layer `li` at `pos`.
pub const PendingComp = struct { li: usize, arr: mlx.mlx_array, pos: usize };

/// A batched verify integrates B+1 tokens at once, but a partial accept must
/// keep only a prefix. Everything the chunk mutates rolls back by OFFSET
/// (GpuRows `used`, the append-only compressed caches, `st.n`) EXCEPT each
/// layer's compressor pending rings, which are overwritten in place per token
/// — so those, and only those, are copied per position while the verify runs.
/// The alternative (restore the entry snapshot, re-forward the accepted
/// prefix through the trunk) costs a second batched forward per partial
/// round; these copies cost ~12 MB of memcpy for the whole block.
pub const DsparkAnchors = struct {
    /// [width * n_layers], row-major by position. Position p holds the state
    /// after the first p+1 tokens of the chunk were integrated.
    layers: []AnchorLayer,
    width: usize,
    n_layers: usize,
    alloc: std.mem.Allocator,
    /// Capture runs only while a verify chunk is in flight.
    armed: bool = false,

    pub const AnchorLayer = struct {
        comp_kv: []f32 = &.{},
        comp_sc: []f32 = &.{},
        comp_cache_len: usize = 0,
        idx_kv: []f32 = &.{},
        idx_sc: []f32 = &.{},
        idx_cache_len: usize = 0,
    };

    fn at(self: *DsparkAnchors, pos: usize, li: usize) *AnchorLayer {
        return &self.layers[pos * self.n_layers + li];
    }

    /// Record layer `li`'s compressor rings after token `pos` of the chunk.
    /// Positions past `width` are the tail the round can never keep (a full
    /// accept needs no rollback), so they are skipped.
    fn captureComp(self: *DsparkAnchors, pos: usize, li: usize, cs: *const CompDecState, comptime indexer: bool) void {
        if (!self.armed or pos >= self.width) return;
        const al = self.at(pos, li);
        if (indexer) {
            @memcpy(al.idx_kv, cs.kv_pend);
            @memcpy(al.idx_sc, cs.sc_pend);
            al.idx_cache_len = cs.cache.items.len;
        } else {
            @memcpy(al.comp_kv, cs.kv_pend);
            @memcpy(al.comp_sc, cs.sc_pend);
            al.comp_cache_len = cs.cache.items.len;
        }
    }

    pub fn deinit(self: *DsparkAnchors) void {
        for (self.layers) |*al| {
            self.alloc.free(al.comp_kv);
            self.alloc.free(al.comp_sc);
            self.alloc.free(al.idx_kv);
            self.alloc.free(al.idx_sc);
        }
        self.alloc.free(self.layers);
    }
};

/// Size (once) and arm the anchor buffers for a `width`-position rollback —
/// i.e. the acceptance counts 1..width tokens. Reused across rounds: the
/// shapes are fixed by the model, so this allocates on the first round only.
pub fn armAnchors(m: *const Dsv4Model, st: *Dsv4DecodeState, width: usize) !void {
    if (st.anchors) |*an| {
        if (an.width >= width) {
            an.armed = true;
            return;
        }
        an.deinit();
        st.anchors = null;
    }
    const alloc = st.alloc;
    const layers = try alloc.alloc(DsparkAnchors.AnchorLayer, width * m.n_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*al| {
            alloc.free(al.comp_kv);
            alloc.free(al.comp_sc);
            alloc.free(al.idx_kv);
            alloc.free(al.idx_sc);
        }
        alloc.free(layers);
    }
    for (0..width) |p| {
        for (st.layers, 0..) |*ls, li| {
            const al = &layers[p * m.n_layers + li];
            al.* = .{};
            if (ls.comp) |*c| {
                al.comp_kv = try alloc.alloc(f32, c.kv_pend.len);
                al.comp_sc = try alloc.alloc(f32, c.sc_pend.len);
            }
            if (ls.idx_comp) |*c| {
                al.idx_kv = try alloc.alloc(f32, c.kv_pend.len);
                al.idx_sc = try alloc.alloc(f32, c.sc_pend.len);
            }
            built += 1;
        }
    }
    st.anchors = .{ .layers = layers, .width = width, .n_layers = m.n_layers, .alloc = alloc, .armed = true };
}

/// Roll back to the state after `accepted + 1` tokens of the verify chunk.
/// `entry` is the round's entry snapshot: it supplies the offsets the chunk
/// grew FROM (the anchors only carry the in-place ring contents).
pub fn restoreToAnchor(st: *Dsv4DecodeState, entry: *const Dsv4Snapshot, accepted: usize) void {
    const an = &st.anchors.?;
    std.debug.assert(accepted < an.width);
    std.debug.assert(entry.layers.len == st.layers.len);
    const kept = accepted + 1;
    st.n = entry.n + kept;
    for (st.layers, entry.layers, 0..) |*ls, *es, li| {
        const al = an.at(accepted, li);
        ls.kv_gpu.truncate(es.kv_gpu_used + kept);
        if (ls.comp) |*c| {
            @memcpy(c.kv_pend, al.comp_kv);
            @memcpy(c.sc_pend, al.comp_sc);
            c.cache.shrinkRetainingCapacity(al.comp_cache_len);
            const emitted = (al.comp_cache_len - es.comp.?.cache_len) / ls.comp_gpu.width;
            ls.comp_gpu.truncate(es.comp_gpu_used + emitted);
        }
        if (ls.idx_comp) |*c| {
            @memcpy(c.kv_pend, al.idx_kv);
            @memcpy(c.sc_pend, al.idx_sc);
            c.cache.shrinkRetainingCapacity(al.idx_cache_len);
            const emitted = (al.idx_cache_len - es.idx_comp.?.cache_len) / ls.idx_gpu.width;
            ls.idx_gpu.truncate(es.idx_gpu_used + emitted);
        }
    }
    if (st.dspark) |*ds| {
        for (ds.main_kv, entry.dspark_used) |*r, u| r.truncate(u + kept);
    }
    an.armed = false;
}

fn initCompDecState(alloc: std.mem.Allocator, ratio: usize, coff: usize, d: usize) !CompDecState {
    const rows = coff * ratio;
    const width = coff * d;
    const kvp = try alloc.alloc(f32, rows * width);
    @memset(kvp, 0);
    const scp = try alloc.alloc(f32, rows * width);
    @memset(scp, -std.math.inf(f32));
    return .{ .kv_pend = kvp, .sc_pend = scp, .cache = std.array_list.Managed(f32).init(alloc), .rows = rows, .width = width };
}

pub fn initDecodeState(m: *const Dsv4Model, alloc: std.mem.Allocator) !Dsv4DecodeState {
    const layers = try alloc.alloc(LayerDecState, m.n_layers);
    for (layers, 0..) |*ls, li| {
        const ratio: usize = m.ratios[li];
        const coff: usize = if (ratio == 4) 2 else 1;
        ls.* = .{
            .kv = std.array_list.Managed(f32).init(alloc),
            .kv_gpu = GpuRows.init(m.head_dim),
            .comp_gpu = GpuRows.init(m.head_dim),
            .idx_gpu = GpuRows.init(m.idx_hd),
            .comp = if (ratio != 0) try initCompDecState(alloc, ratio, coff, m.head_dim) else null,
            .idx_comp = if (ratio == 4) try initCompDecState(alloc, 4, 2, m.idx_hd) else null,
        };
    }
    var dspark: ?DsparkDecState = null;
    if (m.n_mtp > 0) {
        const rings = try alloc.alloc(GpuRows, m.n_mtp);
        for (rings) |*r| r.* = GpuRows.init(m.head_dim);
        dspark = .{ .main_kv = rings, .mh_last = mlx.mlx_array_new(), .has_mh = false };
    }
    return .{ .layers = layers, .n = 0, .alloc = alloc, .dspark = dspark };
}

pub fn deinitDecodeState(st: *Dsv4DecodeState) void {
    for (st.pending.items) |p| _ = mlx.mlx_array_free(p.arr);
    st.pending.deinit(st.alloc);
    for (st.layers) |*ls| {
        ls.kv.deinit();
        ls.kv_gpu.deinit();
        ls.comp_gpu.deinit();
        ls.idx_gpu.deinit();
        if (ls.comp) |*c| {
            st.alloc.free(c.kv_pend);
            st.alloc.free(c.sc_pend);
            c.cache.deinit();
        }
        if (ls.idx_comp) |*c| {
            st.alloc.free(c.kv_pend);
            st.alloc.free(c.sc_pend);
            c.cache.deinit();
        }
    }
    st.alloc.free(st.layers);
    if (st.anchors) |*an| an.deinit();
    if (st.dspark) |*ds| {
        for (ds.main_kv) |*r| r.deinit();
        st.alloc.free(ds.main_kv);
        _ = mlx.mlx_array_free(ds.mh_last);
    }
}

// ── snapshot / restore (DSpark verify rollback) ────────────────────────
//
// The draft/verify loop runs speculative tokens through `extendState` /
// `decodeStep` and must roll a rejected tail back. NO live handles onto the
// GPU buffers (the copy-on-write snapshot class): GpuRows roll back by
// OFFSET alone (`truncate` — appends overwrite the stale rows), the
// append-only caches shrink to their recorded length, and only the small
// fixed pending rings (kv_pend/sc_pend, overwritten in place every token)
// are byte-copied. The `ls.kv` HOST mirror is deliberately absent: only the
// host-centric initial prefill writes it and no decode path reads it, so a
// decode-window rollback never touches it.

const CompDecSnapshot = struct {
    kv_pend: []f32,
    sc_pend: []f32,
    cache_len: usize,
};

const LayerDecSnapshot = struct {
    kv_gpu_used: usize,
    comp_gpu_used: usize,
    idx_gpu_used: usize,
    comp: ?CompDecSnapshot,
    idx_comp: ?CompDecSnapshot,
};

pub const Dsv4Snapshot = struct {
    n: usize,
    layers: []LayerDecSnapshot,
    /// DSpark main_kv ring row counts, one per stage (empty when no stages).
    dspark_used: []usize,
    alloc: std.mem.Allocator,

    pub fn deinit(self: *Dsv4Snapshot) void {
        for (self.layers) |*ls| {
            if (ls.comp) |*c| {
                self.alloc.free(c.kv_pend);
                self.alloc.free(c.sc_pend);
            }
            if (ls.idx_comp) |*c| {
                self.alloc.free(c.kv_pend);
                self.alloc.free(c.sc_pend);
            }
        }
        self.alloc.free(self.layers);
        self.alloc.free(self.dspark_used);
    }
};

fn snapshotCompDec(alloc: std.mem.Allocator, cs: *const CompDecState) !CompDecSnapshot {
    const kvp = try alloc.dupe(f32, cs.kv_pend);
    errdefer alloc.free(kvp);
    const scp = try alloc.dupe(f32, cs.sc_pend);
    return .{ .kv_pend = kvp, .sc_pend = scp, .cache_len = cs.cache.items.len };
}

pub fn snapshotDecodeState(st: *const Dsv4DecodeState, alloc: std.mem.Allocator) !Dsv4Snapshot {
    const n_ds = if (st.dspark) |*ds| ds.main_kv.len else 0;
    const ds_used = try alloc.alloc(usize, n_ds);
    errdefer alloc.free(ds_used);
    if (st.dspark) |*ds| for (ds.main_kv, ds_used) |*r, *u| {
        u.* = r.used;
    };
    const layers = try alloc.alloc(LayerDecSnapshot, st.layers.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*ls| {
            if (ls.comp) |*c| {
                alloc.free(c.kv_pend);
                alloc.free(c.sc_pend);
            }
            if (ls.idx_comp) |*c| {
                alloc.free(c.kv_pend);
                alloc.free(c.sc_pend);
            }
        }
        alloc.free(layers);
    }
    for (st.layers, layers) |*ls, *out| {
        out.* = .{
            .kv_gpu_used = ls.kv_gpu.used,
            .comp_gpu_used = ls.comp_gpu.used,
            .idx_gpu_used = ls.idx_gpu.used,
            .comp = if (ls.comp) |*c| try snapshotCompDec(alloc, c) else null,
            .idx_comp = if (ls.idx_comp) |*c| try snapshotCompDec(alloc, c) else null,
        };
        built += 1;
    }
    return .{ .n = st.n, .layers = layers, .dspark_used = ds_used, .alloc = alloc };
}

fn restoreCompDec(cs: *CompDecState, snap: *const CompDecSnapshot) void {
    @memcpy(cs.kv_pend, snap.kv_pend);
    @memcpy(cs.sc_pend, snap.sc_pend);
    std.debug.assert(snap.cache_len <= cs.cache.items.len);
    cs.cache.shrinkRetainingCapacity(snap.cache_len);
}

/// Rollback-only (never forward): the state must be AT or PAST the snapshot.
/// Restoring does not consume the snapshot — a verify loop restores the same
/// anchor as many times as it rejects.
pub fn restoreDecodeState(st: *Dsv4DecodeState, snap: *const Dsv4Snapshot) void {
    std.debug.assert(snap.layers.len == st.layers.len);
    std.debug.assert(snap.n <= st.n);
    st.n = snap.n;
    for (st.layers, snap.layers) |*ls, *sl| {
        ls.kv_gpu.truncate(sl.kv_gpu_used);
        ls.comp_gpu.truncate(sl.comp_gpu_used);
        ls.idx_gpu.truncate(sl.idx_gpu_used);
        if (ls.comp) |*c| restoreCompDec(c, &sl.comp.?);
        if (ls.idx_comp) |*c| restoreCompDec(c, &sl.idx_comp.?);
    }
    if (st.dspark) |*ds| {
        std.debug.assert(snap.dspark_used.len == ds.main_kv.len);
        for (ds.main_kv, snap.dspark_used) |*r, u| r.truncate(u);
    }
}

/// Finalize + emit one compressed slot from combined rows (shared tail of the
/// prefill and decode branches): RMSNorm → rope at the block-start position →
/// QAT sim (hadamard+fp4 for the indexer path, fp8 otherwise).
fn emitCompSlot(m: *const Dsv4Model, c: *const HostComp, cs: *CompDecState, combined: []f32, block_start: usize, rotate: bool, fr: *const Freqs) !void {
    rmsNormRow(combined, c.norm, m.eps);
    ropeRow(combined[c.head_dim - m.rd ..], fr, block_start, false);
    if (rotate) {
        hadamardInPlace(combined);
        fp4SimInPlace(combined, 32);
    } else {
        fp8SimInPlace(combined[0 .. c.head_dim - m.rd], 64);
    }
    try cs.cache.appendSlice(combined);
}

/// Reference decode branch: push one token's compressor inputs (wkv/wgate
/// matmul rows come PRE-COMPUTED from the layer's single combined sync); on a
/// window boundary combine + emit into the cache.
fn compressorPush(m: *const Dsv4Model, c: *const HostComp, cs: *CompDecState, kv_in: []const f32, sc_in: []const f32, pos: usize, ratio: usize, rotate: bool, fr: *const Freqs, alloc: std.mem.Allocator) !void {
    const d = c.head_dim;
    const cd = c.coff * d;
    const kv_row = kv_in[0..cd];
    const sc_row = try alloc.alloc(f32, cd);
    defer alloc.free(sc_row);
    @memcpy(sc_row, sc_in[0..cd]);
    const r_in = pos % ratio;
    for (0..cd) |j| sc_row[j] += c.ape[r_in * cd + j];
    const overlap = c.coff == 2;
    const slot = if (overlap) ratio + r_in else r_in;
    @memcpy(cs.kv_pend[slot * cs.width ..][0..cd], kv_row);
    @memcpy(cs.sc_pend[slot * cs.width ..][0..cd], sc_row);
    if ((pos + 1) % ratio != 0) return;
    // boundary: combine
    const rows = if (overlap) 2 * ratio else ratio;
    const combined = try alloc.alloc(f32, d);
    defer alloc.free(combined);
    for (0..d) |j| {
        var mx_: f32 = -std.math.inf(f32);
        for (0..rows) |r| {
            const col = if (overlap and r >= ratio) d + j else j;
            mx_ = @max(mx_, cs.sc_pend[r * cs.width + col]);
        }
        var sum: f64 = 0;
        var acc: f64 = 0;
        for (0..rows) |r| {
            const col = if (overlap and r >= ratio) d + j else j;
            const e = @exp(@as(f64, cs.sc_pend[r * cs.width + col] - mx_));
            sum += e;
            acc += e * cs.kv_pend[r * cs.width + col];
        }
        combined[j] = @floatCast(acc / sum);
    }
    // overlap: previous-window rows <- current-window rows (full width)
    if (overlap) {
        for (0..ratio) |r| {
            @memcpy(cs.kv_pend[r * cs.width ..][0..cs.width], cs.kv_pend[(ratio + r) * cs.width ..][0..cs.width]);
            @memcpy(cs.sc_pend[r * cs.width ..][0..cs.width], cs.sc_pend[(ratio + r) * cs.width ..][0..cs.width]);
        }
    }
    try emitCompSlot(m, c, cs, combined, pos + 1 - ratio, rotate, fr);
}

/// Ring-only mirror of `compressorPush` for when the GPU emission owns the
/// emitted bytes: identical ring writes (memcpy + ape add + boundary shift),
/// but the emitted slot advances the host cache by LENGTH only (zeros).
/// Nothing on the GPU stream reads host cache CONTENT — the mirror appends
/// are suppressed, snapshots/anchors compare lengths, restore only
/// truncates — while the combine/norm/rope/QAT-sim host arithmetic this
/// skips measured as ~95% of a 512-token prefill chunk's wall time.
fn compressorPushLight(c: *const HostComp, cs: *CompDecState, kv_in: []const f32, sc_in: []const f32, pos: usize, ratio: usize) !void {
    const d = c.head_dim;
    const cd = c.coff * d;
    const r_in = pos % ratio;
    const overlap = c.coff == 2;
    const slot = if (overlap) ratio + r_in else r_in;
    @memcpy(cs.kv_pend[slot * cs.width ..][0..cd], kv_in[0..cd]);
    const dst = cs.sc_pend[slot * cs.width ..][0..cd];
    for (dst, sc_in[0..cd], c.ape[r_in * cd ..][0..cd]) |*o, sv, av| o.* = sv + av;
    if ((pos + 1) % ratio != 0) return;
    if (overlap) {
        for (0..ratio) |r2| {
            @memcpy(cs.kv_pend[r2 * cs.width ..][0..cs.width], cs.kv_pend[(ratio + r2) * cs.width ..][0..cs.width]);
            @memcpy(cs.sc_pend[r2 * cs.width ..][0..cs.width], cs.sc_pend[(ratio + r2) * cs.width ..][0..cs.width]);
        }
    }
    try cs.cache.appendNTimes(0, d);
}

/// Lazily grown rope tables for decode (per kind), stored on the model.
fn freqsFor(m: *Dsv4Model, comptime kind: enum { plain, yarn }, upto: usize, alloc: std.mem.Allocator) !*const Freqs {
    const slot = switch (kind) {
        .plain => &m.dec_freqs_plain,
        .yarn => &m.dec_freqs_yarn,
    };
    if (slot.*) |*f| {
        if (f.cos.len / f.half >= upto) return f;
        alloc.free(f.cos);
        alloc.free(f.sin);
        slot.* = null;
    }
    const cap = @max(upto * 2, 256);
    slot.* = switch (kind) {
        .plain => try precomputeFreqs(alloc, m.rd, cap, 0, m.plain_theta, 1, 32, 1),
        .yarn => try precomputeFreqs(alloc, m.rd, cap, m.yarn_orig, m.yarn_theta, m.yarn_factor, m.yarn_bf, m.yarn_bs),
    };
    return &(slot.*.?);
}

/// GPU cos/sin rows for ONE position (uploaded from the host Freqs tables —
/// bit-identical trig to ropeRow).
const RopeRows = struct { cos: mlx.mlx_array, sin: mlx.mlx_array };

fn gpuReshape(x: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, @intCast(shape.len), s));
    return out;
}

fn gpuConcat2(a2: mlx.mlx_array, b2: mlx.mlx_array, axis: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    _ = mlx.mlx_vector_array_append_value(parts, a2);
    _ = mlx.mlx_vector_array_append_value(parts, b2);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, axis, s));
    return out;
}

fn gpuSliceCols(x: mlx.mlx_array, rows: usize, c0: usize, c1: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ 0, @intCast(c0) };
    const stop = [_]c_int{ @intCast(rows), @intCast(c1) };
    const strides = [_]c_int{ 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, 2, &stop, 2, &strides, 2, s));
    return out;
}

fn gpuSlice1d(x: mlx.mlx_array, lo: usize, hi: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{@intCast(lo)};
    const stop = [_]c_int{@intCast(hi)};
    const strides = [_]c_int{1};
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, 1, &stop, 1, &strides, 1, s));
    return out;
}

/// Quantized matmul on a GPU f32 row: astype bf16 → qmm → back to f32 (the
/// same dtype hops as qmmHost, so parity with the host path holds).
fn gpuQmmB(q: *const Q, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_astype(&xb, x, .bfloat16, s));
    const y = try qmmBf16(q, xb, s);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, y, .float32, s));
    return out;
}

fn gpuRms(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

// ── fused Sinkhorn kernel (GPU streams only; host-sync fallback below) ──
//
// The composed hc_split_sinkhorn is 20 iterations of tiny [hc,hc] reductions
// — ~80 dispatches per sublayer if built from mlx ops, or a full host sync if
// computed on the CPU (2 syncs/layer, half the decode's sync floor). One
// single-lane Metal dispatch does the whole thing: the work is ~500 flops on
// a 4×4 matrix, so occupancy is irrelevant and one thread is the right shape.
// Reference = hcSplitSinkhorn (golden-tested); metal::exp differs from libm
// by ~1 ulp — the comb path is continuous (no quantizer downstream), and the
// decode-equivalence gate arbitrates. Kill: MLX_SERVE_DSV4_SINKHORN=0.
/// SINKHORN + y-collapse in one dispatch: the composed tail (pre slice →
/// reshape → multiply → sum) was 4 strictly-serial dispatches per sublayer,
/// ~350/token at hc-chain depth. Grid (S, D); each threadgroup's lane-0
/// thread recomputes the tiny per-token Sinkhorn into threadgroup memory
/// (HC=4: ~700 flops, negligible), then every thread emits one y channel:
/// y[t, j] = Σ_c pre[c]·stream[t, c, j] — ascending c, product rounded per
/// step, matching the composed multiply+sum reduction. `out` (pre|post|combT
/// pack) is written by the j==0 threadgroup only.
const SINKHORN_Y_KERNEL_SOURCE =
    \\int t = thread_position_in_grid.x;
    \\int j = thread_position_in_grid.y;
    \\if (t >= S || j >= D) return;
    \\const int PACK = 2 * HC + HC * HC;
    \\const int MIX = (2 + HC) * HC;
    \\threadgroup float tg_pre[HC];
    \\if (thread_position_in_threadgroup.y == 0) {
    \\  float rsq = metal::rsqrt(ssq[t] / consts[0] + consts[1]);
    \\  const float heps = consts[2];
    \\  float pre_v[HC];
    \\  float post_v[HC];
    \\  float comb[HC * HC];
    \\  for (int c = 0; c < HC; ++c) {
    \\    float mj = mixes[t * MIX + c] * rsq;
    \\    pre_v[c] = 1.0f / (1.0f + metal::exp(-(mj * scale[0] + base[c]))) + heps;
    \\    float mp = mixes[t * MIX + HC + c] * rsq;
    \\    post_v[c] = 2.0f / (1.0f + metal::exp(-(mp * scale[1] + base[HC + c])));
    \\  }
    \\  for (int c = 0; c < HC * HC; ++c) {
    \\    comb[c] = mixes[t * MIX + 2 * HC + c] * rsq * scale[2] + base[2 * HC + c];
    \\  }
    \\  for (int c = 0; c < HC; ++c) {
    \\    float mx = comb[c * HC];
    \\    for (int q = 1; q < HC; ++q) mx = metal::max(mx, comb[c * HC + q]);
    \\    float sum = 0.0f;
    \\    for (int q = 0; q < HC; ++q) {
    \\      comb[c * HC + q] = metal::exp(comb[c * HC + q] - mx);
    \\      sum += comb[c * HC + q];
    \\    }
    \\    for (int q = 0; q < HC; ++q) comb[c * HC + q] = comb[c * HC + q] / sum + heps;
    \\  }
    \\  for (int it = 0; it < ITERS; ++it) {
    \\    if (it > 0) {
    \\      for (int c = 0; c < HC; ++c) {
    \\        float sum = 0.0f;
    \\        for (int q = 0; q < HC; ++q) sum += comb[c * HC + q];
    \\        for (int q = 0; q < HC; ++q) comb[c * HC + q] /= (sum + heps);
    \\      }
    \\    }
    \\    for (int q = 0; q < HC; ++q) {
    \\      float sum = 0.0f;
    \\      for (int c = 0; c < HC; ++c) sum += comb[c * HC + q];
    \\      for (int c = 0; c < HC; ++c) comb[c * HC + q] /= (sum + heps);
    \\    }
    \\  }
    \\  for (int c = 0; c < HC; ++c) tg_pre[c] = pre_v[c];
    \\  if (j == 0) {
    \\    for (int c = 0; c < HC; ++c) {
    \\      out[t * PACK + c] = pre_v[c];
    \\      out[t * PACK + HC + c] = post_v[c];
    \\    }
    \\    for (int c = 0; c < HC; ++c) {
    \\      for (int q = 0; q < HC; ++q) out[t * PACK + 2 * HC + q * HC + c] = comb[c * HC + q];
    \\    }
    \\  }
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float acc = 0.0f;
    \\for (int c = 0; c < HC; ++c) {
    \\  float p = tg_pre[c] * stream_in[(t * HC + c) * D + j];
    \\  acc = acc + p;
    \\}
    \\y[t * D + j] = acc;
;

/// hc_post in one dispatch: ns[t,c,j] = Σ_q combT[t,c,q]·stream[t,q,j] +
/// post[t,c]·out[t,j], reading post/combT straight from the sinkhorn pack —
/// replaces the per-sublayer slice/reshape/matmul/multiply/add tail.
const HCPOST_KERNEL_SOURCE =
    \\int t = thread_position_in_grid.x;
    \\int c = thread_position_in_grid.y;
    \\int j = thread_position_in_grid.z;
    \\if (t >= S || c >= HC || j >= D) return;
    \\const int PACK = 2 * HC + HC * HC;
    \\float acc = 0.0f;
    \\for (int q = 0; q < HC; ++q) {
    \\  float p = pk[t * PACK + 2 * HC + c * HC + q] * stream_in[(t * HC + q) * D + j];
    \\  acc = acc + p;
    \\}
    \\acc = acc + pk[t * PACK + HC + c] * out_v[t * D + j];
    \\ns[(t * HC + c) * D + j] = acc;
;

const SINKHORN_KERNEL_SOURCE =
    \\int t = thread_position_in_grid.x;
    \\if (t >= S) return;
    \\const int PACK = 2 * HC + HC * HC;
    \\const int MIX = (2 + HC) * HC;
    \\float rsq = metal::rsqrt(ssq[t] / consts[0] + consts[1]);
    \\const float heps = consts[2];
    \\float pre_v[HC];
    \\float post_v[HC];
    \\float comb[HC * HC];
    \\for (int j = 0; j < HC; ++j) {
    \\  float mj = mixes[t * MIX + j] * rsq;
    \\  pre_v[j] = 1.0f / (1.0f + metal::exp(-(mj * scale[0] + base[j]))) + heps;
    \\  float mp = mixes[t * MIX + HC + j] * rsq;
    \\  post_v[j] = 2.0f / (1.0f + metal::exp(-(mp * scale[1] + base[HC + j])));
    \\}
    \\for (int j = 0; j < HC * HC; ++j) {
    \\  comb[j] = mixes[t * MIX + 2 * HC + j] * rsq * scale[2] + base[2 * HC + j];
    \\}
    \\for (int j = 0; j < HC; ++j) {
    \\  float mx = comb[j * HC];
    \\  for (int q = 1; q < HC; ++q) mx = metal::max(mx, comb[j * HC + q]);
    \\  float sum = 0.0f;
    \\  for (int q = 0; q < HC; ++q) {
    \\    comb[j * HC + q] = metal::exp(comb[j * HC + q] - mx);
    \\    sum += comb[j * HC + q];
    \\  }
    \\  for (int q = 0; q < HC; ++q) comb[j * HC + q] = comb[j * HC + q] / sum + heps;
    \\}
    \\for (int it = 0; it < ITERS; ++it) {
    \\  if (it > 0) {
    \\    for (int j = 0; j < HC; ++j) {
    \\      float sum = 0.0f;
    \\      for (int q = 0; q < HC; ++q) sum += comb[j * HC + q];
    \\      for (int q = 0; q < HC; ++q) comb[j * HC + q] /= (sum + heps);
    \\    }
    \\  }
    \\  for (int q = 0; q < HC; ++q) {
    \\    float sum = 0.0f;
    \\    for (int j = 0; j < HC; ++j) sum += comb[j * HC + q];
    \\    for (int j = 0; j < HC; ++j) comb[j * HC + q] /= (sum + heps);
    \\  }
    \\}
    \\for (int j = 0; j < HC; ++j) {
    \\  out[t * PACK + j] = pre_v[j];
    \\  out[t * PACK + HC + j] = post_v[j];
    \\}
    \\for (int j = 0; j < HC; ++j) {
    \\  for (int q = 0; q < HC; ++q) out[t * PACK + 2 * HC + q * HC + j] = comb[j * HC + q];
    \\}
;

const SinkhornK = struct { kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config };

/// Per-batch-size kernel config: output [tokens, pre|post|combT] — ALWAYS
/// rank 2, tokens == 1 included. A rank-1 special case for decode made the
/// batch path's rank-2 column slice an uncatchable MLX kill on a ONE-token
/// prefill chunk (the serving warmup's exact shape, live 2026-07-31).
/// Widths whose sinkhorn config is worth keeping: every decode/verify width
/// (1..32) plus the prefill sub-chunk. A prefill remainder is a one-off, so
/// caching it would grow the table without ever hitting.
const SINK_CFG_MAX: usize = 32;

fn sinkhornCfgCacheable(tokens: usize) bool {
    return (tokens >= 1 and tokens <= SINK_CFG_MAX) or tokens == prefillSub();
}

/// Cached `sinkhornCfg`, keyed by the token count it bakes into the output
/// shape/grid/threadgroup. hcPreBatch runs TWICE PER LAYER, so a rebuilt
/// config is ~86 allocations + ~520 FFI calls per batched forward — the
/// CPU-side tax that reads as kernel cost (house rule; measured here as the
/// bulk of a batched forward's ~70 ms fixed cost). Never freed by callers:
/// the model owns the table.
fn sinkhornCfgFor(m: *Dsv4Model, tokens: usize) !mlx.mlx_fast_metal_kernel_config {
    if (!sinkhornCfgCacheable(tokens))
        return sinkhornCfg(m.hc, m.hc_iters, tokens) orelse error.MetalKernelConfigFailed;
    const slot: usize = if (tokens == prefillSub()) 0 else tokens;
    if (m.sink_cfg[slot]) |c| return c;
    const c = sinkhornCfg(m.hc, m.hc_iters, tokens) orelse return error.MetalKernelConfigFailed;
    m.sink_cfg[slot] = c;
    return c;
}

fn sinkhornCfg(hc: usize, iters: u32, tokens: usize) ?mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const pack: c_int = @intCast(2 * hc + hc * hc);
    const out_shape = [_]c_int{ @intCast(tokens), pack };
    const tg: c_int = @intCast(@min(tokens, 32));
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float32) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(tokens), 1, 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", @intCast(hc)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", @intCast(tokens)) != 0)
    {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        return null;
    }
    return cfg;
}

fn sinkhornYCfg(hc: usize, iters: u32, tokens: usize, d: usize) ?mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const pack: c_int = @intCast(2 * hc + hc * hc);
    const out_shape = [_]c_int{ @intCast(tokens), pack };
    const y_shape = [_]c_int{ @intCast(tokens), @intCast(d) };
    const tgy: c_int = @intCast(@min(d, 256));
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float32) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, .float32) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(tokens), @intCast(d), 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1, tgy, 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", @intCast(hc)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", @intCast(tokens)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", @intCast(d)) != 0)
    {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        return null;
    }
    return cfg;
}

/// Cached `sinkhornYCfg` (same slot policy as `sinkhornCfgFor`).
fn sinkhornYCfgFor(m: *Dsv4Model, tokens: usize) !mlx.mlx_fast_metal_kernel_config {
    if (!sinkhornCfgCacheable(tokens))
        return sinkhornYCfg(m.hc, m.hc_iters, tokens, m.dim) orelse error.MetalKernelConfigFailed;
    const slot: usize = if (tokens == prefillSub()) 0 else tokens;
    if (m.sink_y_cfg[slot]) |c| return c;
    const c = sinkhornYCfg(m.hc, m.hc_iters, tokens, m.dim) orelse return error.MetalKernelConfigFailed;
    m.sink_y_cfg[slot] = c;
    return c;
}

fn hcPostCfg(hc: usize, tokens: usize, d: usize) ?mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const ns_shape = [_]c_int{ @intCast(tokens), @intCast(hc), @intCast(d) };
    const tgz: c_int = @intCast(@min(d, 256));
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &ns_shape, 3, .float32) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(tokens), @intCast(hc), @intCast(d)) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1, 1, tgz) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", @intCast(hc)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", @intCast(tokens)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", @intCast(d)) != 0)
    {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        return null;
    }
    return cfg;
}

/// Cached `hcPostCfg` (same slot policy as `sinkhornCfgFor`).
fn hcPostCfgFor(m: *Dsv4Model, tokens: usize) !mlx.mlx_fast_metal_kernel_config {
    if (!sinkhornCfgCacheable(tokens))
        return hcPostCfg(m.hc, tokens, m.dim) orelse error.MetalKernelConfigFailed;
    const slot: usize = if (tokens == prefillSub()) 0 else tokens;
    if (m.hc_post_cfg[slot]) |c| return c;
    const c = hcPostCfg(m.hc, tokens, m.dim) orelse return error.MetalKernelConfigFailed;
    m.hc_post_cfg[slot] = c;
    return c;
}

fn buildHcPostKernel(hc: usize, d: usize) ?SinkhornK {
    const input_names = [_][*:0]const u8{ "pk", "stream_in", "out_v" };
    const output_names = [_][*:0]const u8{"ns"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_hc_post", in_vec, out_vec, HCPOST_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) return null;
    const cfg = hcPostCfg(hc, 1, d) orelse {
        _ = mlx.mlx_fast_metal_kernel_free(kernel);
        return null;
    };
    return .{ .kernel = kernel, .cfg = cfg };
}

fn applyHcPost(sk: *const SinkhornK, cfg: mlx.mlx_fast_metal_kernel_config, pk: mlx.mlx_array, stream_g: mlx.mlx_array, out_g: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const inputs_arr = [_]mlx.mlx_array{ pk, stream_g, out_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, sk.kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

fn buildSinkhornYKernel(hc: usize, iters: u32, d: usize) ?SinkhornK {
    const input_names = [_][*:0]const u8{ "mixes", "ssq", "scale", "base", "consts", "stream_in" };
    const output_names = [_][*:0]const u8{ "out", "y" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_sinkhorn_y", in_vec, out_vec, SINKHORN_Y_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) return null;
    const cfg = sinkhornYCfg(hc, iters, 1, d) orelse {
        _ = mlx.mlx_fast_metal_kernel_free(kernel);
        return null;
    };
    return .{ .kernel = kernel, .cfg = cfg };
}

/// Apply the fused sinkhorn+collapse kernel: returns the packed
/// [pre|post|combT] rows AND y [tokens, d] (both owned).
fn applySinkhornY(sk: *const SinkhornK, cfg: mlx.mlx_fast_metal_kernel_config, mixes_g: mlx.mlx_array, ss_g: mlx.mlx_array, scale_g: mlx.mlx_array, base_g: mlx.mlx_array, consts_g: mlx.mlx_array, stream_g: mlx.mlx_array, s: mlx.mlx_stream) !struct { pk: mlx.mlx_array, y: mlx.mlx_array } {
    const inputs_arr = [_]mlx.mlx_array{ mixes_g, ss_g, scale_g, base_g, consts_g, stream_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, sk.kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var pk = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pk);
    try mlx.check(mlx.mlx_vector_array_get(&pk, outputs_vec, 0));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 1));
    return .{ .pk = pk, .y = y };
}

/// Build the fused Sinkhorn kernel + its decode (tokens=1) config. Owned by
/// the caller; null when the Metal source fails to construct.
fn buildSinkhornKernel(hc: usize, iters: u32) ?SinkhornK {
    const input_names = [_][*:0]const u8{ "mixes", "ssq", "scale", "base", "consts" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_sinkhorn", in_vec, out_vec, SINKHORN_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) return null;
    const cfg = sinkhornCfg(hc, iters, 1) orelse {
        _ = mlx.mlx_fast_metal_kernel_free(kernel);
        return null;
    };
    return .{ .kernel = kernel, .cfg = cfg };
}

/// Apply the fused kernel: returns packed [pre(hc) | post(hc) | combT(hc²)]
/// per token ([pack] for the decode config, [tokens, pack] for batch configs).
fn applySinkhorn(sk: *const SinkhornK, cfg: mlx.mlx_fast_metal_kernel_config, mixes_g: mlx.mlx_array, ss_g: mlx.mlx_array, scale_g: mlx.mlx_array, base_g: mlx.mlx_array, consts_g: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const inputs_arr = [_]mlx.mlx_array{ mixes_g, ss_g, scale_g, base_g, consts_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, sk.kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

const HcPreG = struct {
    y: mlx.mlx_array,
    post_g: mlx.mlx_array,
    combT_g: mlx.mlx_array,
    /// The raw sinkhorn pack when the fused hc_post kernel will consume it
    /// directly (post_g/combT_g stay EMPTY handles then).
    pk: ?mlx.mlx_array = null,
};

/// hc_pre for ONE token with the stream resident on GPU. With the fused
/// Sinkhorn kernel (GPU streams) there is NO host hop at all; the fallback
/// syncs [mix+1] floats (mixes ++ Σx²) and runs the host Sinkhorn. Outputs:
/// y [1,d], post [hc,1] and combT [hc,hc] (comb TRANSPOSED) — all GPU.
fn hcPreGpu(m: *Dsv4Model, alloc: std.mem.Allocator, stream_g: mlx.mlx_array, fn_w_t: mlx.mlx_array, scale: []const f32, base: []const f32, scale_g: mlx.mlx_array, base_g: mlx.mlx_array) !HcPreG {
    const hcm = m.hc;
    const hd_full = hcm * m.dim;
    const mix = (2 + hcm) * hcm;
    const fshape = [_]c_int{ 1, @intCast(hd_full) };
    const flat = try gpuReshape(stream_g, &fshape, m.s);
    defer _ = mlx.mlx_array_free(flat);
    const mixes_g = try gpuOp2(mlx.mlx_matmul, flat, fn_w_t, m.s);
    defer _ = mlx.mlx_array_free(mixes_g);
    const sq = try gpuOp1(mlx.mlx_square, flat, m.s);
    defer _ = mlx.mlx_array_free(sq);
    var ss = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ss);
    try mlx.check(mlx.mlx_sum_axis(&ss, sq, 1, true, m.s));
    var pre_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pre_col);
    var post_g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(post_g);
    var combT_g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(combT_g);
    const col_shape = [_]c_int{ @intCast(hcm), 1 };
    const comb_shape = [_]c_int{ @intCast(hcm), @intCast(hcm) };
    if (m.sink_y_k) |*sk| {
        // sinkhorn + y collapse in ONE dispatch — the composed 4-op tail
        // (pre slice/reshape/multiply/sum) sat serially on the hc chain
        if (!m.sink_y_logged) {
            m.sink_y_logged = true;
            log.info("dsv4: fused sinkhorn+collapse kernel engaged\n", .{});
        }
        const r = try applySinkhornY(sk, sk.cfg, mixes_g, ss, scale_g, base_g, m.sink_consts, stream_g, m.s);
        if (m.hc_post_k != null) {
            // the fused hc_post consumes the pack directly — no slices
            return .{ .y = r.y, .post_g = post_g, .combT_g = combT_g, .pk = r.pk };
        }
        defer _ = mlx.mlx_array_free(r.pk);
        const post_flat = try gpuSliceCols(r.pk, 1, hcm, 2 * hcm, m.s);
        defer _ = mlx.mlx_array_free(post_flat);
        try mlx.check(mlx.mlx_reshape(&post_g, post_flat, &col_shape, 2, m.s));
        const comb_flat = try gpuSliceCols(r.pk, 1, 2 * hcm, 2 * hcm + hcm * hcm, m.s);
        defer _ = mlx.mlx_array_free(comb_flat);
        try mlx.check(mlx.mlx_reshape(&combT_g, comb_flat, &comb_shape, 2, m.s));
        return .{ .y = r.y, .post_g = post_g, .combT_g = combT_g };
    } else if (m.sink_k) |*sk| {
        const pk = try applySinkhorn(sk, sk.cfg, mixes_g, ss, scale_g, base_g, m.sink_consts, m.s);
        defer _ = mlx.mlx_array_free(pk);
        if (!m.sink_logged) {
            m.sink_logged = true;
            log.info("dsv4: fused sinkhorn kernel engaged\n", .{});
        }
        // pk is [1, pack] (the config is always rank 2) — column slices.
        const pre_flat = try gpuSliceCols(pk, 1, 0, hcm, m.s);
        defer _ = mlx.mlx_array_free(pre_flat);
        try mlx.check(mlx.mlx_reshape(&pre_col, pre_flat, &col_shape, 2, m.s));
        const post_flat = try gpuSliceCols(pk, 1, hcm, 2 * hcm, m.s);
        defer _ = mlx.mlx_array_free(post_flat);
        try mlx.check(mlx.mlx_reshape(&post_g, post_flat, &col_shape, 2, m.s));
        const comb_flat = try gpuSliceCols(pk, 1, 2 * hcm, 2 * hcm + hcm * hcm, m.s);
        defer _ = mlx.mlx_array_free(comb_flat);
        try mlx.check(mlx.mlx_reshape(&combT_g, comb_flat, &comb_shape, 2, m.s));
    } else {
        const joint = try gpuConcat2(mixes_g, ss, 1, m.s);
        defer _ = mlx.mlx_array_free(joint);
        const row = try toHostF32(alloc, joint, mix + 1, m.s); // the sublayer's one sync
        defer alloc.free(row);
        const rsq: f32 = @floatCast(1.0 / @sqrt(@as(f64, row[mix]) / @as(f64, @floatFromInt(hd_full)) + m.eps));
        var mm: [96]f32 = undefined;
        for (0..mix) |j| mm[j] = row[j] * rsq;
        const split = hcSplitSinkhorn(mm[0..mix], scale, base, hcm, m.hc_iters, m.hc_eps);
        var combT: [64]f32 = undefined;
        for (0..hcm) |k| {
            for (0..hcm) |j| combT[k * hcm + j] = split.comb[j * hcm + k];
        }
        _ = mlx.mlx_array_free(pre_col);
        pre_col = uploadF32(split.pre[0..hcm], &col_shape);
        _ = mlx.mlx_array_free(post_g);
        post_g = uploadF32(split.post[0..hcm], &col_shape);
        _ = mlx.mlx_array_free(combT_g);
        combT_g = uploadF32(combT[0 .. hcm * hcm], &comb_shape);
    }
    const weighted = try gpuOp2(mlx.mlx_multiply, pre_col, stream_g, m.s);
    defer _ = mlx.mlx_array_free(weighted);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_sum_axis(&y, weighted, 0, true, m.s));
    return .{ .y = y, .post_g = post_g, .combT_g = combT_g };
}

fn freeHcPre(pre: *const HcPreG) void {
    _ = mlx.mlx_array_free(pre.y);
    _ = mlx.mlx_array_free(pre.post_g);
    _ = mlx.mlx_array_free(pre.combT_g);
    if (pre.pk) |p| _ = mlx.mlx_array_free(p);
}

/// hc_post on GPU: new_stream = combᵀ @ residual + post·out. With the fused
/// kernel (pre.pk set) the whole tail is ONE dispatch.
fn hcPostGpu(m: *Dsv4Model, stream_g: mlx.mlx_array, out_g: mlx.mlx_array, pre: *const HcPreG) !mlx.mlx_array {
    if (pre.pk) |pk| {
        const sk = &m.hc_post_k.?;
        if (!m.hc_post_logged) {
            m.hc_post_logged = true;
            log.info("dsv4: fused hc_post kernel engaged\n", .{});
        }
        const ns3 = try applyHcPost(sk, sk.cfg, pk, stream_g, out_g, m.s); // [1, hc, d]
        defer _ = mlx.mlx_array_free(ns3);
        const shape2 = [_]c_int{ @intCast(m.hc), @intCast(m.dim) };
        return try gpuReshape(ns3, &shape2, m.s);
    }
    const res = try gpuOp2(mlx.mlx_matmul, pre.combT_g, stream_g, m.s);
    defer _ = mlx.mlx_array_free(res);
    const po = try gpuOp2(mlx.mlx_multiply, pre.post_g, out_g, m.s);
    defer _ = mlx.mlx_array_free(po);
    return try gpuOp2(mlx.mlx_add, res, po, m.s);
}

/// MoE for a batch of tokens with x resident on GPU ([C, d] f32 in and out).
/// Routing runs on GPU too: sp = sqrt(logaddexp(scores, 0)) (the host
/// sqrtSoftplus's stable form), selection = ascending-argpartition tail of
/// sp + gate.bias (bias joins SELECTION only — the routeToken rule), weights
/// = take_along(sp) normalized × route_scale, and gather_qmm consumes the
/// GPU indices directly. The old path synced the [C, E] scores to host per
/// MoE call — at decode that was ~43 blocking round-trips per token. Hash
/// layers never read scores for selection (pure token-id table, host-known),
/// so their index rows are an UPLOAD, not a sync; their weights ride the
/// same GPU gather. `MLX_SERVE_DSV4_MOE_ROUTE_GPU=0` restores the host sync
/// (routeToken, f64 normalize) for A/B.
fn traceMemMb(comptime which: enum { active, cache, peak }) usize {
    var v: usize = 0;
    _ = switch (which) {
        .active => mlx.mlx_get_active_memory(&v),
        .cache => mlx.mlx_get_cache_memory(&v),
        .peak => mlx.mlx_get_peak_memory(&v),
    };
    return v / (1024 * 1024);
}

fn moeGpu(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x_g: mlx.mlx_array, ids: []const u32) !mlx.mlx_array {
    return moeGpuImpl(m, alloc, li, x_g, ids, null);
}

/// `moeGpu` with the token id as a LAZY GPU array (single token): hash-layer
/// routing looks the tid2eid row up via take on device, so the id never
/// touches the host (the lazy decode path's requirement).
fn moeGpuLazy(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x_g: mlx.mlx_array, id_arr: mlx.mlx_array) !mlx.mlx_array {
    return moeGpuImpl(m, alloc, li, x_g, &.{0}, id_arr);
}

fn moeGpuImpl(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x_g: mlx.mlx_array, ids: []const u32, id_arr: ?mlx.mlx_array) !mlx.mlx_array {
    const ly = &m.dw.layers[li];
    const h = &m.hl[li];
    const E = m.n_experts;
    const k = m.topk;
    const seq = ids.len;
    const scores_g = try gpuOp2(mlx.mlx_matmul, x_g, h.gate_w_t, m.s);
    defer _ = mlx.mlx_array_free(scores_g);

    const ind_shape = [_]c_int{ @intCast(seq), @intCast(k) };
    const wshape = [_]c_int{ @intCast(seq), @intCast(k), 1, 1 };
    var ind = mlx.mlx_array_new(); // [C, k] int32
    defer _ = mlx.mlx_array_free(ind);
    var w_arr = mlx.mlx_array_new(); // [C, k, 1, 1] f32 route weights
    defer _ = mlx.mlx_array_free(w_arr);

    if (moeRouteGpuEnabled() or id_arr != null) {
        if (!moe_route_gpu_logged) {
            moe_route_gpu_logged = true;
            log.info("dsv4: GPU MoE routing engaged\n", .{});
        }
        var rg = try routeGpu(m.s, alloc, scores_g, ly.gate_bias, h.tid2eid, ly.tid2eid, id_arr, ids, E, k, m.route_scale);
        defer rg.deinit();
        try mlx.check(mlx.mlx_array_set(&ind, rg.ind));
        try mlx.check(mlx.mlx_array_set(&w_arr, rg.w));
    } else {
        const scores = try toHostF32(alloc, scores_g, seq * E, m.s); // routing sync
        defer alloc.free(scores);
        for (scores) |*v| v.* = @floatCast(sqrtSoftplus(v.*));
        const indices = try alloc.alloc(i32, seq * k);
        defer alloc.free(indices);
        const wts = try alloc.alloc(f32, seq * k);
        defer alloc.free(wts);
        for (0..seq) |t| {
            routeToken(m, h, scores[t * E ..][0..E], ids[t], indices[t * k ..][0..k], wts[t * k ..][0..k]);
        }
        const up = mlx.mlx_array_new_data(indices.ptr, &ind_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_array_set(&ind, up));
        const wu = uploadF32(wts, &wshape);
        defer _ = mlx.mlx_array_free(wu);
        try mlx.check(mlx.mlx_array_set(&w_arr, wu));
    }

    var down32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down32);
    if (seq > 1) {
        // ── global-sort gather (mlx-lm's _gather_sort; the moeMLP2 /
        // inklingExpertsApply pattern): argsort the flattened (token, expert)
        // pairs so consecutive gather_qmm slots hit the SAME expert bank and
        // each routed expert's rows stream from DRAM once per chunk instead
        // of once per token — the unsorted path re-read ~12x at C=512.
        const total: c_int = @intCast(seq * k);
        const flat_shape = [_]c_int{total};
        var flat_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_inds);
        try mlx.check(mlx.mlx_reshape(&flat_inds, ind, &flat_shape, 1, m.s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, m.s));
        var inv_order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inv_order);
        try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, m.s));
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, m.s));
        const k_arr = mlx.mlx_array_new_int(@intCast(k));
        defer _ = mlx.mlx_array_free(k_arr);
        var lhs_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs_idx);
        try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, m.s));
        var xb2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xb2);
        try mlx.check(mlx.mlx_astype(&xb2, x_g, .bfloat16, m.s));
        var x_gathered = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_gathered);
        try mlx.check(mlx.mlx_take_axis(&x_gathered, xb2, lhs_idx, 0, m.s));
        const n1d = [_]c_int{ total, 1, @intCast(m.dim) };
        const x_rep = try gpuReshape(x_gathered, &n1d, m.s);
        defer _ = mlx.mlx_array_free(x_rep);
        const gate_arr = try gatherQmmESorted(&ly.experts_w1, x_rep, sorted_inds, m.s);
        defer _ = mlx.mlx_array_free(gate_arr);
        const up_arr = try gatherQmmESorted(&ly.experts_w3, x_rep, sorted_inds, m.s);
        defer _ = mlx.mlx_array_free(up_arr);
        const act = try clippedSwigluG(gate_arr, up_arr, m.swiglu_limit, m.s);
        defer _ = mlx.mlx_array_free(act);
        const down_arr = try gatherQmmESorted(&ly.experts_w2, act, sorted_inds, m.s);
        defer _ = mlx.mlx_array_free(down_arr);
        const nd_shape = [_]c_int{ total, @intCast(m.dim) };
        const down_flat = try gpuReshape(down_arr, &nd_shape, m.s);
        defer _ = mlx.mlx_array_free(down_flat);
        var unsorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(unsorted);
        try mlx.check(mlx.mlx_take_axis(&unsorted, down_flat, inv_order, 0, m.s));
        var un32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(un32);
        try mlx.check(mlx.mlx_astype(&un32, unsorted, .float32, m.s));
        const ckd = [_]c_int{ @intCast(seq), @intCast(k), 1, @intCast(m.dim) };
        var d4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d4);
        try mlx.check(mlx.mlx_reshape(&d4, un32, &ckd, 4, m.s));
        try mlx.check(mlx.mlx_array_set(&down32, d4));
    } else {
        const xshape = [_]c_int{ @intCast(seq), 1, 1, @intCast(m.dim) };
        const xr = try gpuReshape(x_g, &xshape, m.s);
        defer _ = mlx.mlx_array_free(xr);
        var xe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xe);
        try mlx.check(mlx.mlx_astype(&xe, xr, .bfloat16, m.s));
        const topk: usize = @intCast(mlx.mlx_array_shape(ind)[1]);
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        if (try moeGateUpFused(m, xe, &ly.experts_w1, &ly.experts_w3, ind, topk)) |fused| {
            defer _ = mlx.mlx_array_free(fused);
            try mlx.check(mlx.mlx_array_set(&act, fused));
        } else {
            const gate_arr = try gatherQmmE(&ly.experts_w1, xe, ind, m.s);
            defer _ = mlx.mlx_array_free(gate_arr);
            const up_arr = try gatherQmmE(&ly.experts_w3, xe, ind, m.s);
            defer _ = mlx.mlx_array_free(up_arr);
            const composed = try clippedSwigluG(gate_arr, up_arr, m.swiglu_limit, m.s);
            defer _ = mlx.mlx_array_free(composed);
            try mlx.check(mlx.mlx_array_set(&act, composed));
        }
        const down_arr = try gatherQmmE(&ly.experts_w2, act, ind, m.s);
        defer _ = mlx.mlx_array_free(down_arr);
        try mlx.check(mlx.mlx_astype(&down32, down_arr, .float32, m.s));
    }
    const weighted = try gpuOp2(mlx.mlx_multiply, down32, w_arr, m.s);
    defer _ = mlx.mlx_array_free(weighted);
    var routed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(routed);
    try mlx.check(mlx.mlx_sum_axis(&routed, weighted, 1, false, m.s)); // [C, 1, d]
    // shared expert (clipped too)
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_astype(&xb, x_g, .bfloat16, m.s));
    const sg_arr = try qmmBf16(&ly.shared_w1, xb, m.s);
    defer _ = mlx.mlx_array_free(sg_arr);
    const su_arr = try qmmBf16(&ly.shared_w3, xb, m.s);
    defer _ = mlx.mlx_array_free(su_arr);
    const sact = try clippedSwigluG(sg_arr, su_arr, m.swiglu_limit, m.s);
    defer _ = mlx.mlx_array_free(sact);
    const sd_arr = try qmmBf16(&ly.shared_w2, sact, m.s);
    defer _ = mlx.mlx_array_free(sd_arr);
    var sd32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sd32);
    try mlx.check(mlx.mlx_astype(&sd32, sd_arr, .float32, m.s));
    const sshape = [_]c_int{ @intCast(seq), 1, @intCast(m.dim) };
    const sd_r = try gpuReshape(sd32, &sshape, m.s);
    defer _ = mlx.mlx_array_free(sd_r);
    const total = try gpuOp2(mlx.mlx_add, routed, sd_r, m.s);
    defer _ = mlx.mlx_array_free(total);
    if (li >= m.n_layers and std.c.getenv("MLX_SERVE_DSPARK_TRACE") != null) {
        const pr = struct {
            fn f(al: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s2: mlx.mlx_stream) f64 {
                const hh = toHostF32(al, arr, n, s2) catch return -1;
                defer al.free(hh);
                var acc: f64 = 0;
                for (hh) |v| acc += @as(f64, v) * v;
                return @sqrt(acc);
            }
        }.f;
        log.info("[dspark-trace] moe li={d} mem(pre)={d}/{d}/{d}MB scores={d:.4} w={d:.4}\n", .{
            li,
            traceMemMb(.active),      traceMemMb(.cache),       traceMemMb(.peak),
            pr(alloc, scores_g, seq * E, m.s),
            pr(alloc, w_arr, seq * k, m.s),
        });
        log.info("[dspark-trace] moe li={d} mem(mid)={d}/{d}/{d}MB down={d:.4} routed={d:.4} shared={d:.4} mem(post)={d}/{d}/{d}MB\n", .{
            li,
            traceMemMb(.active),      traceMemMb(.cache),    traceMemMb(.peak),
            pr(alloc, down32, seq * k * m.dim, m.s),
            pr(alloc, routed, seq * m.dim, m.s),
            pr(alloc, sd32, seq * m.dim, m.s),
            traceMemMb(.active),      traceMemMb(.cache),    traceMemMb(.peak),
        });
    }
    const oshape = [_]c_int{ @intCast(seq), @intCast(m.dim) };
    return try gpuReshape(total, &oshape, m.s);
}

fn moeDecodeGpu(m: *const Dsv4Model, alloc: std.mem.Allocator, li: usize, x_g: mlx.mlx_array, id: u32) !mlx.mlx_array {
    return moeGpu(m, alloc, li, x_g, &.{id});
}

/// Single-token attention with the whole q/kv/indexer/output chain on GPU.
/// Host hops: ONE combined compressor-input sync (ratio != 0 layers only) —
/// the top-k compressed-slot select runs on GPU via argpartition.
/// A compressor-input row whose host read was deferred to the end-of-token
/// batched eval. Only NON-boundary positions defer (their push lands in the
/// pending rings and nothing this token reads it); boundary positions keep
/// the in-layer sync because the emitted slot is same-token-visible to the
/// compressed-slot selection.
const DeferredCompRow = struct { li: usize, arr: mlx.mlx_array };

/// Does the chunk [base, base+c_tokens) close a compression window? A window
/// closes at p where (p+1) % ratio == 0, so this counts the multiples of
/// `ratio` in (base, base+c_tokens]. Pure position arithmetic — the answer is
/// known before any GPU work, which is what makes the deferral exact.
fn chunkCrossesBoundary(base: usize, c_tokens: usize, ratio: usize) bool {
    if (ratio == 0) return false;
    return (base + c_tokens) / ratio != base / ratio;
}

fn attentionDecodeGpu(m: *const Dsv4Model, alloc: std.mem.Allocator, st: *Dsv4DecodeState, li: usize, x_g: mlx.mlx_array, pos: usize, rr: *const RopeRows, fr: *const Freqs, deferred: *std.ArrayList(DeferredCompRow)) !mlx.mlx_array {
    const ly = &m.dw.layers[li];
    const h = &m.hl[li];
    const ls = &st.layers[li];
    const ratio: usize = ly.compress_ratio;
    const hd = m.head_dim;
    const nh = m.n_heads;
    const rd = m.rd;

    // q chain: wq_a → q_norm → wq_b → per-head RMS → rope (zero host hops)
    const qr_n = blk: {
        const qr = try gpuQmmB(&ly.wq_a, x_g, m.s);
        defer _ = mlx.mlx_array_free(qr);
        break :blk try gpuRms(qr, h.q_norm_g, m.eps, m.s);
    };
    defer _ = mlx.mlx_array_free(qr_n);
    const q_rot = blk: {
        const q_flat = try gpuQmmB(&ly.wq_b, qr_n, m.s);
        defer _ = mlx.mlx_array_free(q_flat);
        if (try decChainKernel(m, q_flat, nh, hd, rd, m.ones_hd_g, 0, false, rr)) |fused| break :blk fused;
        const qshape = [_]c_int{ @intCast(nh), @intCast(hd) };
        const q_r = try gpuReshape(q_flat, &qshape, m.s);
        defer _ = mlx.mlx_array_free(q_r);
        const q_rms = try gpuRms(q_r, m.ones_hd_g, m.eps, m.s); // unweighted per-head RMS
        defer _ = mlx.mlx_array_free(q_rms);
        break :blk try gpuRopeTail(q_rms, rd, rr.cos, rr.sin, false, m.s);
    };
    defer _ = mlx.mlx_array_free(q_rot);

    // kv chain: wkv → kv_norm → rope → fp8 sim on the non-rope dims → append
    {
        const kv0 = try gpuQmmB(&ly.wkv, x_g, m.s);
        defer _ = mlx.mlx_array_free(kv0);
        if (try decChainKernel(m, kv0, 1, hd, rd, h.kv_norm_g, 1, false, rr)) |kv_fused| {
            defer _ = mlx.mlx_array_free(kv_fused);
            try ls.kv_gpu.appendGpu(kv_fused, 1, m.s);
        } else {
        const kv_n = try gpuRms(kv0, h.kv_norm_g, m.eps, m.s);
        defer _ = mlx.mlx_array_free(kv_n);
        const kv_rot = try gpuRopeTail(kv_n, rd, rr.cos, rr.sin, false, m.s);
        defer _ = mlx.mlx_array_free(kv_rot);
        const head0 = try gpuSliceCols(kv_rot, 1, 0, hd - rd, m.s);
        defer _ = mlx.mlx_array_free(head0);
        const head_sim = try gpuFp8Sim(head0, m.s);
        defer _ = mlx.mlx_array_free(head_sim);
        const tail = try gpuSliceCols(kv_rot, 1, hd - rd, hd, m.s);
        defer _ = mlx.mlx_array_free(tail);
        const kv_fin = try gpuConcat2(head_sim, tail, 1, m.s);
        defer _ = mlx.mlx_array_free(kv_fin);
        try ls.kv_gpu.appendGpu(kv_fin, 1, m.s);
        }
    }

    // compressor rings: ONE sync feeds the attn AND indexer wkv/wgate rows —
    // and ONLY on boundary positions, where the emitted slot is visible to
    // this token's own compressed-slot selection below. Non-boundary rows
    // land solely in the pending rings, which nothing reads until a later
    // token, so their host read joins the end-of-token batched eval
    // (`processDeferredComp`) instead of stalling the pipeline per layer
    // (~41 blocking round-trips/token at steady state before this).
    if (ratio != 0) {
        const comp_in = try compInProj(m, h, li, x_g, 1);
        const boundary_attn = (pos + 1) % ratio == 0;
        const boundary_idx = ls.idx_comp != null and (pos + 1) % 4 == 0;
        if (gpuEmitActive(m)) {
            // window emission stays pure GPU graph (the emitted slot is
            // same-token-visible to the selection below via the mirror);
            // the host pushes defer to end-of-token like every other row.
            const cdd = h.comp.?.coff * h.comp.?.head_dim;
            if (boundary_attn)
                try emitWindowsGpu(m, &h.comp.?, &ls.comp.?, &ls.comp_gpu, comp_in, 0, pos, 1, ratio, false, fr, alloc);
            if (boundary_idx)
                try emitWindowsGpu(m, &h.idx_comp.?, &ls.idx_comp.?, &ls.idx_gpu, comp_in, 2 * cdd, pos, 1, 4, true, fr, alloc);
            errdefer _ = mlx.mlx_array_free(comp_in);
            try deferred.append(alloc, .{ .li = li, .arr = comp_in });
        } else if (boundary_attn or boundary_idx or !compDeferEnabled()) {
            defer _ = mlx.mlx_array_free(comp_in);
            var cclk: DsparkClock = if (dsv4TraceEnabled()) DsparkClock.init() else undefined;
            const row = try toHostF32(alloc, comp_in, h.comp_in_w, m.s);
            defer alloc.free(row);
            const c = &h.comp.?;
            const cd = c.coff * c.head_dim;
            {
                const csx = &ls.comp.?;
                const before = csx.cache.items.len;
                try compressorPush(m, c, csx, row[0..cd], row[cd .. 2 * cd], pos, ratio, false, fr, alloc);
                if (csx.cache.items.len > before)
                    try ls.comp_gpu.append(csx.cache.items[before..], m.s);
            }
            if (ls.idx_comp) |*csx| {
                const ic = &h.idx_comp.?;
                const icd = ic.coff * ic.head_dim;
                const before = csx.cache.items.len;
                try compressorPush(m, ic, csx, row[2 * cd ..][0..icd], row[2 * cd + icd ..][0..icd], pos, 4, true, fr, alloc);
                if (csx.cache.items.len > before)
                    try ls.idx_gpu.append(csx.cache.items[before..], m.s);
            }
            if (dsv4TraceEnabled()) trace_comp_ns += cclk.lap();
        } else {
            errdefer _ = mlx.mlx_array_free(comp_in);
            try deferred.append(alloc, .{ .li = li, .arr = comp_in });
        }
    }

    // compressed-slot selection (all-GPU; top-k via argpartition)
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    var n_sel: usize = 0;
    if (ratio == 4 and ls.comp_gpu.used > 0) {
        const avail = @min(ls.comp_gpu.used, ls.idx_gpu.used);
        const k = @min(m.idx_topk, avail);
        if (k > 0) {
            const ih = m.idx_heads;
            const ihd = m.idx_hd;
            // indexer q chain: wq_b(qr) → rope → Hadamard → fp4 sim
            const qi_sim = blk: {
                const qi = try gpuQmmB(&ly.idx.?.wq_b, qr_n, m.s);
                defer _ = mlx.mlx_array_free(qi);
                if (try decChainKernel(m, qi, ih, ihd, rd, null, 2, false, rr)) |fused| break :blk fused;
                const qsh = [_]c_int{ @intCast(ih), @intCast(ihd) };
                const qi_r = try gpuReshape(qi, &qsh, m.s);
                defer _ = mlx.mlx_array_free(qi_r);
                const qi_rot = try gpuRopeTail(qi_r, rd, rr.cos, rr.sin, false, m.s);
                defer _ = mlx.mlx_array_free(qi_rot);
                const qi_had = try gpuOp2(mlx.mlx_matmul, qi_rot, m.hada_g.?, m.s);
                defer _ = mlx.mlx_array_free(qi_had);
                break :blk try gpuFp4Sim(qi_had, m.s);
            };
            defer _ = mlx.mlx_array_free(qi_sim);
            const islots = try ls.idx_gpu.sliceRows(0, avail, m.s);
            defer _ = mlx.mlx_array_free(islots);
            const it_ = try gpuOp1(mlx.mlx_transpose, islots, m.s);
            defer _ = mlx.mlx_array_free(it_);
            const sc = try gpuOp2(mlx.mlx_matmul, qi_sim, it_, m.s);
            defer _ = mlx.mlx_array_free(sc);
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            const relu = try gpuOp2(mlx.mlx_maximum, sc, zero, m.s);
            defer _ = mlx.mlx_array_free(relu);
            // per-head weights: weights_proj(x)·(ihd·ih)^-1/2 — the head sum
            // is ONE [1, ih] @ [ih, avail] matmul (replaces the composed
            // transpose/broadcast-multiply/multiply/sum chain)
            const wts_row = try gpuOp2(mlx.mlx_matmul, x_g, h.idx_wp_t.?, m.s);
            defer _ = mlx.mlx_array_free(wts_row);
            const wscale = mlx.mlx_array_new_float(@floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(ihd))) * @sqrt(@as(f64, @floatFromInt(ih))))));
            defer _ = mlx.mlx_array_free(wscale);
            const wts_s = try gpuOp2(mlx.mlx_multiply, wts_row, wscale, m.s);
            defer _ = mlx.mlx_array_free(wts_s);
            const summed2 = try gpuOp2(mlx.mlx_matmul, wts_s, relu, m.s); // [1, avail]
            defer _ = mlx.mlx_array_free(summed2);
            const sshape = [_]c_int{@intCast(avail)};
            var summed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(summed);
            try mlx.check(mlx.mlx_reshape(&summed, summed2, &sshape, 1, m.s));
            const all_slots = try ls.comp_gpu.sliceRows(0, avail, m.s);
            defer _ = mlx.mlx_array_free(all_slots);
            if (avail > k) {
                var part = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(part);
                try mlx.check(mlx.mlx_argpartition_axis(&part, summed, @intCast(avail - k), 0, m.s));
                const sel_idx = try gpuSlice1d(part, avail - k, avail, m.s);
                defer _ = mlx.mlx_array_free(sel_idx);
                try mlx.check(mlx.mlx_take_axis(&picked, all_slots, sel_idx, 0, m.s));
            } else {
                try mlx.check(mlx.mlx_astype(&picked, all_slots, .float32, m.s));
            }
            n_sel = k;
        }
    } else if (ratio != 0 and ls.comp_gpu.used > 0) {
        const n_slots = ls.comp_gpu.used;
        const all_slots = try ls.comp_gpu.sliceRows(0, n_slots, m.s);
        defer _ = mlx.mlx_array_free(all_slots);
        try mlx.check(mlx.mlx_astype(&picked, all_slots, .float32, m.s));
        n_sel = n_slots;
    }

    // sink-softmax attention: K = [window rows ++ selected slots]; scores =
    // qK^T·scale with the sink appended as one extra column — the softmax
    // normalizes real slots by (denom + e^sink) exactly like the reference
    // kernel, and the sink column is dropped before the value mix.
    const seq_now = pos + 1;
    const wk = @min(seq_now, m.window);
    const win = try ls.kv_gpu.sliceRows(seq_now - wk, seq_now, m.s);
    defer _ = mlx.mlx_array_free(win);
    var kmat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kmat);
    if (n_sel > 0) {
        const parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        _ = mlx.mlx_vector_array_append_value(parts, win);
        _ = mlx.mlx_vector_array_append_value(parts, picked);
        try mlx.check(mlx.mlx_concatenate_axis(&kmat, parts, 0, m.s));
    } else {
        try mlx.check(mlx.mlx_astype(&kmat, win, .float32, m.s));
    }
    const tk = wk + n_sel;
    const kt = try gpuOp1(mlx.mlx_transpose, kmat, m.s);
    defer _ = mlx.mlx_array_free(kt);
    const scores0 = try gpuOp2(mlx.mlx_matmul, q_rot, kt, m.s);
    defer _ = mlx.mlx_array_free(scores0);
    const scale_f: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
    var probs_real = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs_real);
    if (try sinkSoftmaxKernel(m, scores0, h.sink_gpu, nh, tk, scale_f)) |fused| {
        defer _ = mlx.mlx_array_free(fused);
        try mlx.check(mlx.mlx_array_set(&probs_real, fused));
    } else {
        const scale_arr = mlx.mlx_array_new_float(scale_f);
        defer _ = mlx.mlx_array_free(scale_arr);
        const scaled = try gpuOp2(mlx.mlx_multiply, scores0, scale_arr, m.s);
        defer _ = mlx.mlx_array_free(scaled);
        const with_sink = try gpuConcat2(scaled, h.sink_gpu, 1, m.s);
        defer _ = mlx.mlx_array_free(with_sink);
        var probs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(probs);
        try mlx.check(mlx.mlx_softmax_axis(&probs, with_sink, -1, true, m.s));
        const pr = try gpuSliceCols(probs, nh, 0, tk, m.s);
        defer _ = mlx.mlx_array_free(pr);
        try mlx.check(mlx.mlx_array_set(&probs_real, pr));
    }
    const o_arr = try gpuOp2(mlx.mlx_matmul, probs_real, kmat, m.s);
    defer _ = mlx.mlx_array_free(o_arr);
    const o_inv = (try decChainKernel(m, o_arr, nh, hd, rd, null, 0, true, rr)) orelse
        try gpuRopeTail(o_arr, rd, rr.cos, rr.sin, true, m.s);
    defer _ = mlx.mlx_array_free(o_inv);
    // grouped low-rank O on GPU: [og, 1, gin] bf16 @ wo_a_deq [og, gin, ol]
    const og = m.o_groups;
    const ol = m.o_lora;
    const gin = nh * hd / og;
    const oshape = [_]c_int{ @intCast(og), 1, @intCast(gin) };
    const o_g = try gpuReshape(o_inv, &oshape, m.s);
    defer _ = mlx.mlx_array_free(o_g);
    var o_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_b);
    try mlx.check(mlx.mlx_astype(&o_b, o_g, .bfloat16, m.s));
    const ored = try woAMatmul(m, li, o_b);
    defer _ = mlx.mlx_array_free(ored);
    const rshape2 = [_]c_int{ 1, @intCast(og * ol) };
    const ored_r = try gpuReshape(ored, &rshape2, m.s);
    defer _ = mlx.mlx_array_free(ored_r);
    return try gpuQmmB(&ly.wo_b, ored_r, m.s); // [1, dim] f32
}

/// The 43-layer decode walk shared by the sync (`decodeStep`) and lazy
/// (`decodeStepLazy`) paths. Consumes `stream_in` (ownership transfers) and
/// returns the final hc stream (owned). `mh_parts` collects DSpark target
/// captures when armed (sync path only — the lazy path requires DSpark off).
fn decodeLayers(m: *Dsv4Model, a: std.mem.Allocator, st: *Dsv4DecodeState, stream_in: mlx.mlx_array, id: u32, id_arr: ?mlx.mlx_array, pos: usize, rr_plain: *const RopeRows, rr_yarn: *const RopeRows, fr_plain: *const Freqs, fr_yarn: *const Freqs, deferred: *std.ArrayList(DeferredCompRow), mh_parts: *std.ArrayList(mlx.mlx_array)) !mlx.mlx_array {
    var stream_g = stream_in;
    // Timing-only probes (GARBAGE OUTPUT): `MLX_SERVE_DSV4_LAYER_CAP=N`
    // truncates the walk, `MLX_SERVE_DSV4_SKIP_MOE=1` drops the ffn
    // sublayer — slope/intercept attribution of the serial forward.
    const cap = blk: {
        const e = std.c.getenv("MLX_SERVE_DSV4_LAYER_CAP") orelse break :blk m.n_layers;
        break :blk std.fmt.parseInt(usize, std.mem.span(e), 10) catch m.n_layers;
    };
    const skip_moe = std.c.getenv("MLX_SERVE_DSV4_SKIP_MOE") != null;
    for (0..@min(m.n_layers, cap)) |li| {
        const h = &m.hl[li];
        const ly = &m.dw.layers[li];
        const ratio: usize = ly.compress_ratio;
        const fr = if (ratio != 0) fr_yarn else fr_plain;
        const rr = if (ratio != 0) rr_yarn else rr_plain;
        {
            const pre = try hcPreGpu(m, a, stream_g, h.hc_attn_fn_t, h.hc_attn_scale, h.hc_attn_base, ly.hc_attn_scale, ly.hc_attn_base);
            defer freeHcPre(&pre);
            const x = try gpuRms(pre.y, h.attn_norm_g, m.eps, m.s);
            defer _ = mlx.mlx_array_free(x);
            const attn_out = try attentionDecodeGpu(m, a, st, li, x, pos, rr, fr, deferred);
            defer _ = mlx.mlx_array_free(attn_out);
            const ns = try hcPostGpu(m, stream_g, attn_out, &pre);
            _ = mlx.mlx_array_free(stream_g);
            stream_g = ns;
        }
        if (!skip_moe) {
            const pre = try hcPreGpu(m, a, stream_g, h.hc_ffn_fn_t, h.hc_ffn_scale, h.hc_ffn_base, ly.hc_ffn_scale, ly.hc_ffn_base);
            defer freeHcPre(&pre);
            const x = try gpuRms(pre.y, h.ffn_norm_g, m.eps, m.s);
            defer _ = mlx.mlx_array_free(x);
            const ffn_out = if (id_arr) |ia| try moeGpuLazy(m, a, li, x, ia) else try moeDecodeGpu(m, a, li, x, id);
            defer _ = mlx.mlx_array_free(ffn_out);
            const ns = try hcPostGpu(m, stream_g, ffn_out, &pre);
            _ = mlx.mlx_array_free(stream_g);
            stream_g = ns;
        }
        // DSpark conditioning: hc-averaged stream at the target layers
        // (reference forward's main_hiddens capture, [1, d] per target).
        if (st.dspark != null and dsIsTarget(m, li)) {
            var mean = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(mean);
            try mlx.check(mlx.mlx_mean_axis(&mean, stream_g, 0, true, m.s));
            try mh_parts.append(a, mean);
        }
    }
    return stream_g;
}

/// One incremental decode step: appends token `id`, returns [vocab] logits.
/// The hc stream stays RESIDENT on GPU across all layers; host hops per token
/// are the two hc-mix syncs per layer (Sinkhorn), the MoE routing sync, one
/// combined compressor-input sync on ratio≠0 layers, and the final logits.
pub fn decodeStep(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, id: u32) ![]f32 {
    const tracing = dsv4TraceEnabled();
    var clk: DsparkClock = if (tracing) DsparkClock.init() else undefined;
    const gap_us: u64 = if (tracing and trace_last_end != null)
        @as(u64, @intCast(trace_last_end.?.untilNow(std.Io.Threaded.global_single_threaded.io(), .boot).nanoseconds)) / 1000
    else
        0;
    if (tracing) trace_comp_ns = 0;
    if (st.pending.items.len > 0) try drainPending(m, st); // lazy→sync handoff
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pos = st.n;
    st.n += 1;
    const d = m.dim;
    const hcm = m.hc;
    // host rope tables (compressor emits read them) + this position's GPU rows
    const fr_plain = try freqsFor(m, .plain, pos + 2, m.arena.allocator());
    const fr_yarn = try freqsFor(m, .yarn, pos + 2, m.arena.allocator());
    const half = m.rd / 2;
    const rowshape = [_]c_int{@intCast(half)};
    const rr_plain = RopeRows{
        .cos = uploadF32(fr_plain.cos[pos * half ..][0..half], &rowshape),
        .sin = uploadF32(fr_plain.sin[pos * half ..][0..half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr_plain.cos);
    defer _ = mlx.mlx_array_free(rr_plain.sin);
    const rr_yarn = RopeRows{
        .cos = uploadF32(fr_yarn.cos[pos * half ..][0..half], &rowshape),
        .sin = uploadF32(fr_yarn.sin[pos * half ..][0..half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr_yarn.cos);
    defer _ = mlx.mlx_array_free(rr_yarn.sin);
    // stream [hc, d] = embed row broadcast, resident on GPU across all layers
    var stream_g = blk: {
        const e = m.embed_f32[@as(usize, id) * d ..][0..d];
        const eshape = [_]c_int{ 1, @intCast(d) };
        const e_row = uploadF32(e, &eshape);
        defer _ = mlx.mlx_array_free(e_row);
        const bshape = [_]c_int{ @intCast(hcm), @intCast(d) };
        var b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&b, e_row, &bshape, 2, m.s));
        break :blk b;
    };
    defer _ = mlx.mlx_array_free(stream_g);
    var deferred = std.ArrayList(DeferredCompRow).empty;
    defer {
        for (deferred.items) |r| _ = mlx.mlx_array_free(r.arr);
        deferred.deinit(a);
    }
    var mh_parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (mh_parts.items) |p| _ = mlx.mlx_array_free(p);
        mh_parts.deinit(a);
    }
    stream_g = try decodeLayers(m, a, st, stream_g, id, null, pos, &rr_plain, &rr_yarn, fr_plain, fr_yarn, &deferred, &mh_parts);
    // Schedule the deferred compressor side-branches WITH the head walk:
    // they share ~the whole cone with the logits, so the GPU computes them
    // for free during the head sync and `processDeferredComp`'s eval
    // becomes a wait-free formality (it was a serial ~2.7 ms mini-eval).
    if (deferred.items.len > 0) {
        const dvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(dvec);
        for (deferred.items) |r2| _ = mlx.mlx_vector_array_append_value(dvec, r2.arr);
        try mlx.check(mlx.mlx_async_eval(dvec));
    }
    const build_ns: u64 = if (tracing) clk.lap() else 0;
    const logits = try headLogitsGpu(m, gpa, a, stream_g);
    errdefer gpa.free(logits);
    const head_ns: u64 = if (tracing) clk.lap() else 0;
    try processDeferredComp(m, st, a, deferred.items, pos, fr_yarn);
    if (tracing) {
        const defer_ns = clk.lap();
        log.info("[dsv4-trace] pos={d} build={d}us head={d}us defer={d}us comp={d}us gap={d}us\n", .{
            pos, build_ns / 1000, head_ns / 1000, defer_ns / 1000, trace_comp_ns / 1000, gap_us,
        });
        trace_last_end = std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .boot);
    }
    if (st.dspark != null and mh_parts.items.len > 0) {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (mh_parts.items) |p| _ = mlx.mlx_vector_array_append_value(vec, p);
        var mh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mh);
        try mlx.check(mlx.mlx_concatenate_axis(&mh, vec, 1, m.s));
        try appendDsparkMainKv(m, st, mh, 1, &rr_plain);
    }
    return logits;
}

/// Lazy pipelined decode step: the same graph as `decodeStep`, but the token
/// id stays a GPU array (embed row via `embed_g` take, hash routing via a
/// device tid2eid lookup) and the logits return UNEVALUATED [1, vocab] —
/// generate.zig's pipelined next() samples on GPU and overlaps the next
/// build with execution, so the id never forces a host round trip. Host ring
/// pushes stash on `st.pending` (async-scheduled with the token's own flow)
/// and drain just before the next window-boundary token's emission build, a
/// prefill/verify chunk, or teardown. Requires `lazyDecodeReady`.
pub fn decodeStepLazy(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, id_arr: mlx.mlx_array) !mlx.mlx_array {
    if (!lazy_decode_logged) {
        lazy_decode_logged = true;
        log.info("dsv4: lazy pipelined decode engaged\n", .{});
    }
    const tracing = dsv4TraceEnabled();
    var clk: DsparkClock = if (tracing) DsparkClock.init() else undefined;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pos = st.n;
    // rings must be current through pos-1 before an emission build reads them
    if (st.pending.items.len > 0 and posClosesAnyWindow(m, pos)) try drainPending(m, st);
    const drain_ns: u64 = if (tracing) clk.lap() else 0;
    st.n += 1;
    const d = m.dim;
    const hcm = m.hc;
    const fr_plain = try freqsFor(m, .plain, pos + 2, m.arena.allocator());
    const fr_yarn = try freqsFor(m, .yarn, pos + 2, m.arena.allocator());
    const half = m.rd / 2;
    const rowshape = [_]c_int{@intCast(half)};
    const rr_plain = RopeRows{
        .cos = uploadF32(fr_plain.cos[pos * half ..][0..half], &rowshape),
        .sin = uploadF32(fr_plain.sin[pos * half ..][0..half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr_plain.cos);
    defer _ = mlx.mlx_array_free(rr_plain.sin);
    const rr_yarn = RopeRows{
        .cos = uploadF32(fr_yarn.cos[pos * half ..][0..half], &rowshape),
        .sin = uploadF32(fr_yarn.sin[pos * half ..][0..half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr_yarn.cos);
    defer _ = mlx.mlx_array_free(rr_yarn.sin);
    // stream [hc, d]: embed row via GPU take (bf16 → f32 — the host
    // embed_f32 is f32 of the same bf16 values, so the rows are identical)
    var stream_g = blk: {
        const fshape = [_]c_int{1};
        var idf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idf);
        try mlx.check(mlx.mlx_reshape(&idf, id_arr, &fshape, 1, m.s));
        var row_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_b);
        try mlx.check(mlx.mlx_take_axis(&row_b, m.embed_g.?, idf, 0, m.s)); // [1, d] bf16
        var row_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_f);
        try mlx.check(mlx.mlx_astype(&row_f, row_b, .float32, m.s));
        const bshape = [_]c_int{ @intCast(hcm), @intCast(d) };
        var b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&b, row_f, &bshape, 2, m.s));
        break :blk b;
    };
    defer _ = mlx.mlx_array_free(stream_g);
    var deferred = std.ArrayList(DeferredCompRow).empty;
    defer {
        for (deferred.items) |r| _ = mlx.mlx_array_free(r.arr);
        deferred.deinit(a);
    }
    var mh_parts = std.ArrayList(mlx.mlx_array).empty; // DSpark off by contract
    defer mh_parts.deinit(a);
    stream_g = try decodeLayers(m, a, st, stream_g, 0, id_arr, pos, &rr_plain, &rr_yarn, fr_plain, fr_yarn, &deferred, &mh_parts);
    // schedule the deferred rows with the token's own pipeline flow, then
    // move them to the pending stash (host pushes happen at the next drain)
    if (deferred.items.len > 0) {
        const dvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(dvec);
        for (deferred.items) |r| _ = mlx.mlx_vector_array_append_value(dvec, r.arr);
        try mlx.check(mlx.mlx_async_eval(dvec));
        while (deferred.pop()) |r| {
            st.pending.append(st.alloc, .{ .li = r.li, .arr = r.arr, .pos = pos }) catch |e| {
                _ = mlx.mlx_array_free(r.arr);
                return e;
            };
        }
    }
    // boundary prefetch: the NEXT token blocks on a full drain before its
    // emission build — drain the OLDER rows now, leaving only this token's.
    if (drainPrefetchEnabled() and st.pending.items.len > 0 and posClosesAnyWindow(m, pos + 1))
        try drainPendingBefore(m, st, pos);
    const logits = try headLogitsLazyGpu(m, stream_g);
    if (tracing) {
        const build_ns = clk.lap();
        log.info("[dsv4-trace] pos={d} lazy build={d}us drain={d}us pending={d}\n", .{
            pos, build_ns / 1000, drain_ns / 1000, st.pending.items.len,
        });
    }
    return logits;
}

/// Everything `decodeStepLazy` requires: GPU stream + GPU emission + GPU MoE
/// routing (device hash lookup), the GPU embed table, and DSpark off.
pub fn lazyDecodeReady(m: *const Dsv4Model, st: *const Dsv4DecodeState) bool {
    return lazyDecodeEnabled() and m.embed_g != null and st.dspark == null and
        moeRouteGpuEnabled() and gpuEmitActive(m);
}

/// Drain the non-boundary compressor rows deferred by `attentionDecodeGpu`:
/// ONE `mlx_eval` submission materializes every side-branch (their inputs
/// were already computed by the logits cone), then each host read is a plain
/// memcpy into the pending rings. Non-boundary pushes can never emit a
/// compressed slot by construction; the release-mode append below keeps the
/// GPU mirrors correct anyway if that invariant is ever broken.
fn processDeferredComp(m: *const Dsv4Model, st: *Dsv4DecodeState, a: std.mem.Allocator, rows: []const DeferredCompRow, pos: usize, fr: *const Freqs) !void {
    if (rows.len == 0) return;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (rows) |r| _ = mlx.mlx_vector_array_append_value(vec, r.arr);
    try mlx.check(mlx.mlx_eval(vec));
    // Under GPU emission the deferred set includes BOUNDARY rows: their host
    // push EMITS into the host cache (snapshots/anchors read it), but the
    // mirror append is suppressed — the in-layer GPU emission already
    // advanced `used` for those slots. Without GPU emission only
    // non-boundary rows defer, which can never emit by construction; the
    // release-mode append keeps the mirrors correct anyway if that
    // invariant is ever broken.
    const emit_gpu = gpuEmitActive(m);
    for (rows) |r| {
        const h = &m.hl[r.li];
        const ls = &st.layers[r.li];
        const ratio: usize = m.dw.layers[r.li].compress_ratio;
        const row = try toHostF32(a, r.arr, h.comp_in_w, m.s);
        defer a.free(row);
        const c = &h.comp.?;
        const cd = c.coff * c.head_dim;
        {
            const csx = &ls.comp.?;
            const before = csx.cache.items.len;
            if (emit_gpu) {
                try compressorPushLight(c, csx, row[0..cd], row[cd .. 2 * cd], pos, ratio);
            } else {
                try compressorPush(m, c, csx, row[0..cd], row[cd .. 2 * cd], pos, ratio, false, fr, a);
                std.debug.assert(csx.cache.items.len == before);
                if (csx.cache.items.len > before)
                    try ls.comp_gpu.append(csx.cache.items[before..], m.s);
            }
        }
        if (ls.idx_comp) |*csx| {
            const ic = &h.idx_comp.?;
            const icd = ic.coff * ic.head_dim;
            const before = csx.cache.items.len;
            if (emit_gpu) {
                try compressorPushLight(ic, csx, row[2 * cd ..][0..icd], row[2 * cd + icd ..][0..icd], pos, 4);
            } else {
                try compressorPush(m, ic, csx, row[2 * cd ..][0..icd], row[2 * cd + icd ..][0..icd], pos, 4, true, fr, a);
                std.debug.assert(csx.cache.items.len == before);
                if (csx.cache.items.len > before)
                    try ls.idx_gpu.append(csx.cache.items[before..], m.s);
            }
        }
    }
}

/// Does position `pos` close ANY compressor window (attn ratios; the ratio-4
/// indexer shares the ratio-4 boundary)? The lazy path must drain pending
/// host pushes before this token's emission build reads the rings.
fn posClosesAnyWindow(m: *const Dsv4Model, pos: usize) bool {
    for (m.ratios[0..m.n_layers]) |r| {
        if (r != 0 and (pos + 1) % @as(usize, r) == 0) return true;
    }
    return false;
}

/// One pending row's host pushes (ring updates only — lazy decode implies
/// GPU emission, the mirrors were appended in-graph). Shared by the full
/// drain and the boundary prefetch so a fix can't honor one path only.
fn pendingHostPush(m: *const Dsv4Model, st: *Dsv4DecodeState, p: PendingComp) !void {
    const h = &m.hl[p.li];
    const ls = &st.layers[p.li];
    const ratio: usize = m.dw.layers[p.li].compress_ratio;
    const row = try toHostF32(st.alloc, p.arr, h.comp_in_w, m.s);
    defer st.alloc.free(row);
    const c = &h.comp.?;
    const cd = c.coff * c.head_dim;
    try compressorPushLight(c, &ls.comp.?, row[0..cd], row[cd .. 2 * cd], p.pos, ratio);
    if (ls.idx_comp) |*csx| {
        const ic = &h.idx_comp.?;
        const icd = ic.coff * ic.head_dim;
        try compressorPushLight(ic, csx, row[2 * cd ..][0..icd], row[2 * cd + icd ..][0..icd], p.pos, 4);
    }
}

/// Drain the lazy-decode pending host pushes: ONE batched eval (the rows
/// were already async-scheduled with their token's pipeline flow, so this is
/// mostly a wait), then the light ring pushes in position order.
pub fn drainPending(m: *const Dsv4Model, st: *Dsv4DecodeState) !void {
    if (st.pending.items.len == 0) return;
    std.debug.assert(gpuEmitActive(m));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (st.pending.items) |p| _ = mlx.mlx_vector_array_append_value(vec, p.arr);
    try mlx.check(mlx.mlx_eval(vec));
    for (st.pending.items) |p| {
        try pendingHostPush(m, st, p);
        _ = mlx.mlx_array_free(p.arr);
    }
    st.pending.clearRetainingCapacity();
}

/// Boundary-prefetch kill switch (`MLX_SERVE_DSV4_DRAIN_PREFETCH=0` → the
/// boundary token drains everything itself, the pre-prefetch behavior).
var drain_prefetch_state: ?bool = null;
fn drainPrefetchEnabled() bool {
    if (drain_prefetch_state) |v| return v;
    const v = if (std.c.getenv("MLX_SERVE_DSV4_DRAIN_PREFETCH")) |e| e[0] != '0' else true;
    drain_prefetch_state = v;
    return v;
}

/// Partial drain of pending rows with pos < `before` (kept entries stay in
/// order). Called at the end of the step BEFORE a window-closing token:
/// those rows' async evals completed a token+ ago, so the wait is a memcpy
/// that overlaps this token's logits cone instead of sitting on the boundary
/// token's critical path.
fn drainPendingBefore(m: *const Dsv4Model, st: *Dsv4DecodeState, before: usize) !void {
    var n_old: usize = 0;
    for (st.pending.items) |p| {
        if (p.pos < before) n_old += 1;
    }
    if (n_old == 0) return;
    std.debug.assert(gpuEmitActive(m));
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (st.pending.items) |p| {
        if (p.pos < before) _ = mlx.mlx_vector_array_append_value(vec, p.arr);
    }
    try mlx.check(mlx.mlx_eval(vec));
    var w: usize = 0;
    for (st.pending.items) |p| {
        if (p.pos < before) {
            try pendingHostPush(m, st, p);
            _ = mlx.mlx_array_free(p.arr);
        } else {
            st.pending.items[w] = p;
            w += 1;
        }
    }
    st.pending.items.len = w;
}

fn dsIsTarget(m: *const Dsv4Model, li: usize) bool {
    for (m.ds_targets[0..m.n_ds_targets]) |t| if (t == li) return true;
    return false;
}

/// Owned [1, w] copy of the LAST row of a [n, w] GPU array — a slice is a
/// view and must never outlive its parent (house materializedOwnedCopy
/// class); add-zero materializes a fresh buffer.
fn ownedLastRow(x: mlx.mlx_array, n: usize, w: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ @intCast(n - 1), 0 };
    const stop = [_]c_int{ @intCast(n), @intCast(w) };
    const strides = [_]c_int{ 1, 1 };
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_slice(&v, x, &start, 2, &stop, 2, &strides, 2, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, v, zero, s));
    return out;
}

/// DSpark conditioning append: from main_hidden rows [n, n_targets*dim] f32
/// (GPU, positions base..base+n), compute main_x = main_norm(main_proj(mh))
/// once, then each stage's finalized ring row — wkv → kv_norm → rope at the
/// absolute positions → fp8 sim on the non-rope dims, the exact
/// DSparkAttention main_kv recipe — and append it to that stage's ring.
/// `rr` are PLAIN rope rows covering the same positions (stages are ratio-0).
fn appendDsparkMainKv(m: *Dsv4Model, st: *Dsv4DecodeState, mh: mlx.mlx_array, n: usize, rr: *const RopeRows) !void {
    const ds = &st.dspark.?;
    const dspw = &m.dw.dspark.?;
    const hd = m.head_dim;
    const rd = m.rd;
    const single = mlx.mlx_array_ndim(rr.cos) == 1;
    const main_x = blk: {
        const proj = try gpuQmmB(&dspw.main_proj, mh, m.s);
        defer _ = mlx.mlx_array_free(proj);
        break :blk try gpuRms(proj, m.ds_main_norm_g.?, m.eps, m.s);
    };
    defer _ = mlx.mlx_array_free(main_x);
    for (0..m.n_mtp) |sti| {
        const ly = &m.dw.layers[m.n_layers + sti];
        const h = &m.hl[m.n_layers + sti];
        const kv0 = try gpuQmmB(&ly.wkv, main_x, m.s);
        defer _ = mlx.mlx_array_free(kv0);
        const kv_n = try gpuRms(kv0, h.kv_norm_g, m.eps, m.s);
        defer _ = mlx.mlx_array_free(kv_n);
        const kv_rot = if (single)
            try gpuRopeTail(kv_n, rd, rr.cos, rr.sin, false, m.s)
        else
            try gpuRopeTailRows(kv_n, rd, rr.cos, rr.sin, false, m.s);
        defer _ = mlx.mlx_array_free(kv_rot);
        const head0 = try gpuSliceCols(kv_rot, n, 0, hd - rd, m.s);
        defer _ = mlx.mlx_array_free(head0);
        const head_sim = try gpuFp8Sim(head0, m.s);
        defer _ = mlx.mlx_array_free(head_sim);
        const tail = try gpuSliceCols(kv_rot, n, hd - rd, hd, m.s);
        defer _ = mlx.mlx_array_free(tail);
        const kv_fin = try gpuConcat2(head_sim, tail, 1, m.s);
        defer _ = mlx.mlx_array_free(kv_fin);
        try ds.main_kv[sti].appendGpu(kv_fin, n, m.s);
    }
    // parity seam: keep the last position's main_hidden (owned copy)
    const last = try ownedLastRow(mh, n, m.dim * m.n_ds_targets, m.s);
    _ = mlx.mlx_array_free(ds.mh_last);
    ds.mh_last = last;
    ds.has_mh = true;
}

/// Hyper-head collapse (sigmoid weights only) → final norm → lm head, from a
/// [hc, d] GPU stream row. One tiny sync (mixes ++ Σx²) + the logits sync.
/// hc-head sigmoid pre-weights, all-GPU: prew[t, c] = sigmoid(mix[t, c] ·
/// rsqrt(ss[t]/(hc·d) + eps) · scale + base[c]) + hc_eps. mixes [C, hc],
/// ssum [C, 1] → [C, hc]. Replaces the per-token mid-head host sync (the
/// sigmoid mix was the LAST remaining barrier before the logits read).
fn headPreWeightsGpu(m: *const Dsv4Model, mixes_g: mlx.mlx_array, ssum: mlx.mlx_array) !mlx.mlx_array {
    const hcm = m.hc;
    const denom_c = mlx.mlx_array_new_float(@floatFromInt(hcm * m.dim));
    defer _ = mlx.mlx_array_free(denom_c);
    const ss_n = try gpuOp2(mlx.mlx_divide, ssum, denom_c, m.s);
    defer _ = mlx.mlx_array_free(ss_n);
    const eps_c = mlx.mlx_array_new_float(m.eps);
    defer _ = mlx.mlx_array_free(eps_c);
    const ss_e = try gpuOp2(mlx.mlx_add, ss_n, eps_c, m.s);
    defer _ = mlx.mlx_array_free(ss_e);
    const rsq = try gpuOp1(mlx.mlx_rsqrt, ss_e, m.s);
    defer _ = mlx.mlx_array_free(rsq);
    const mixed = try gpuOp2(mlx.mlx_multiply, mixes_g, rsq, m.s);
    defer _ = mlx.mlx_array_free(mixed);
    const scale_c = mlx.mlx_array_new_float(m.hc_head_scale[0]);
    defer _ = mlx.mlx_array_free(scale_c);
    const z = try gpuOp2(mlx.mlx_multiply, mixed, scale_c, m.s);
    defer _ = mlx.mlx_array_free(z);
    const bshape = [_]c_int{ 1, @intCast(hcm) };
    const base_g = uploadF32(m.hc_head_base[0..hcm], &bshape);
    defer _ = mlx.mlx_array_free(base_g);
    const zb = try gpuOp2(mlx.mlx_add, z, base_g, m.s);
    defer _ = mlx.mlx_array_free(zb);
    const sg = try gpuOp1(mlx.mlx_sigmoid, zb, m.s);
    defer _ = mlx.mlx_array_free(sg);
    const heps_c = mlx.mlx_array_new_float(m.hc_eps);
    defer _ = mlx.mlx_array_free(heps_c);
    return try gpuOp2(mlx.mlx_add, sg, heps_c, m.s);
}

/// The whole trunk head as ONE lazy graph (GPU streams): hc collapse
/// (all-GPU sigmoid mix) → final norm → head qmm. Returns [1, vocab], owned,
/// NOT evaluated — the lazy decode path returns it to the generator's
/// pipelined sampler.
fn headLogitsLazyGpu(m: *const Dsv4Model, stream_g: mlx.mlx_array) !mlx.mlx_array {
    const hcm = m.hc;
    const d = m.dim;
    const fshape = [_]c_int{ 1, @intCast(hcm * d) };
    const flat = try gpuReshape(stream_g, &fshape, m.s);
    defer _ = mlx.mlx_array_free(flat);
    const mixes_g = try gpuOp2(mlx.mlx_matmul, flat, m.hc_head_fn_t, m.s);
    defer _ = mlx.mlx_array_free(mixes_g);
    const sq = try gpuOp1(mlx.mlx_square, flat, m.s);
    defer _ = mlx.mlx_array_free(sq);
    var ssum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ssum);
    try mlx.check(mlx.mlx_sum_axis(&ssum, sq, 1, true, m.s));
    const prew = try headPreWeightsGpu(m, mixes_g, ssum); // [1, hc]
    defer _ = mlx.mlx_array_free(prew);
    const pshape = [_]c_int{ @intCast(hcm), 1 };
    const pre_col = try gpuReshape(prew, &pshape, m.s);
    defer _ = mlx.mlx_array_free(pre_col);
    const weighted = try gpuOp2(mlx.mlx_multiply, pre_col, stream_g, m.s);
    defer _ = mlx.mlx_array_free(weighted);
    var hout = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hout);
    try mlx.check(mlx.mlx_sum_axis(&hout, weighted, 0, true, m.s));
    const hn = try gpuRms(hout, m.final_norm_g, m.eps, m.s);
    defer _ = mlx.mlx_array_free(hn);
    return try gpuQmmB(&m.dw.head, hn, m.s);
}

fn headLogitsGpu(m: *const Dsv4Model, gpa: std.mem.Allocator, alloc: std.mem.Allocator, stream_g: mlx.mlx_array) ![]f32 {
    const hcm = m.hc;
    const d = m.dim;
    if (mlx.streamIsGpu(m.s)) {
        const logits_g = try headLogitsLazyGpu(m, stream_g);
        defer _ = mlx.mlx_array_free(logits_g);
        return try toHostF32(gpa, logits_g, m.vocab, m.s);
    }
    // CPU stream = the strict-gated reference path: keep the host sigmoid
    // byte-stable with forwardPrefill's collapse
    const fshape = [_]c_int{ 1, @intCast(hcm * d) };
    const flat = try gpuReshape(stream_g, &fshape, m.s);
    defer _ = mlx.mlx_array_free(flat);
    const mixes_g = try gpuOp2(mlx.mlx_matmul, flat, m.hc_head_fn_t, m.s);
    defer _ = mlx.mlx_array_free(mixes_g);
    const sq = try gpuOp1(mlx.mlx_square, flat, m.s);
    defer _ = mlx.mlx_array_free(sq);
    var ssum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ssum);
    try mlx.check(mlx.mlx_sum_axis(&ssum, sq, 1, true, m.s));
    const joint = try gpuConcat2(mixes_g, ssum, 1, m.s);
    defer _ = mlx.mlx_array_free(joint);
    const jh = try toHostF32(alloc, joint, hcm + 1, m.s);
    defer alloc.free(jh);
    const rsq: f32 = @floatCast(1.0 / @sqrt(@as(f64, jh[hcm]) / @as(f64, @floatFromInt(hcm * d)) + m.eps));
    var prew: [8]f32 = undefined;
    for (0..hcm) |c| prew[c] = sigmoidF32(jh[c] * rsq * m.hc_head_scale[0] + m.hc_head_base[c]) + m.hc_eps;
    const pshape = [_]c_int{ @intCast(hcm), 1 };
    const pre_col = uploadF32(prew[0..hcm], &pshape);
    defer _ = mlx.mlx_array_free(pre_col);
    const weighted = try gpuOp2(mlx.mlx_multiply, pre_col, stream_g, m.s);
    defer _ = mlx.mlx_array_free(weighted);
    var hout = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hout);
    try mlx.check(mlx.mlx_sum_axis(&hout, weighted, 0, true, m.s));
    const hn = try gpuRms(hout, m.final_norm_g, m.eps, m.s);
    defer _ = mlx.mlx_array_free(hn);
    const logits_g = try gpuQmmB(&m.dw.head, hn, m.s);
    defer _ = mlx.mlx_array_free(logits_g);
    return try toHostF32(gpa, logits_g, m.vocab, m.s);
}

// ── batched GPU prefill (extendState) ──────────────────────────────────
//
// The batched mirror of the decode chain: a chunk of C tokens runs every
// layer with the SAME building blocks (gpuQmmB / fast_rms_norm / rope /
// sims / fused Sinkhorn / gather_qmm), attention gathers a per-token K set
// (window band + selected compressed slots) and masks causality explicitly.
// The compressor pending rings stay host (one [C, W] sync per layer feeds
// every ring; the per-token push code is the SAME decode-proven
// compressorPush). extendState is both the fresh prefill (base == 0) and the
// chunked-prefill continuation (base > 0); decode is its C == 1 sibling.

/// Batched rope on the trailing rd dims at PER-ROW positions: cos/sin arrive
/// as [C, rd/2] row tables (row t = token t's position). x is [C, hd] or
/// [C, H, hd]; math identical to gpuRopeTail.
fn gpuRopeTailRows(x: mlx.mlx_array, rd: usize, cos_rows: mlx.mlx_array, sin_rows: mlx.mlx_array, inverse: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const ndim = mlx.mlx_array_ndim(x);
    const sh = mlx.mlx_array_shape(x);
    var shape: [4]c_int = undefined;
    for (0..ndim) |i| shape[i] = sh[i];
    const d: c_int = shape[ndim - 1];
    const rdc: c_int = @intCast(rd);
    var starts: [4]c_int = @splat(0);
    var stops: [4]c_int = undefined;
    var strides: [4]c_int = @splat(1);
    for (0..ndim) |i| stops[i] = shape[i];
    stops[ndim - 1] = d - rdc;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, x, &starts, ndim, &stops, ndim, &strides, ndim, s));
    starts[ndim - 1] = d - rdc;
    stops[ndim - 1] = d;
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_slice(&tail, x, &starts, ndim, &stops, ndim, &strides, ndim, s));
    var pshape: [5]c_int = undefined;
    for (0..ndim - 1) |i| pshape[i] = shape[i];
    pshape[ndim - 1] = @divExact(rdc, 2);
    pshape[ndim] = 2;
    var pairs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pairs);
    try mlx.check(mlx.mlx_reshape(&pairs, tail, &pshape, ndim + 1, s));
    var pstart: [5]c_int = @splat(0);
    var pstop: [5]c_int = undefined;
    var pstr: [5]c_int = @splat(1);
    for (0..ndim + 1) |i| pstop[i] = pshape[i];
    pstop[ndim] = 1;
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_slice(&xr, pairs, &pstart, ndim + 1, &pstop, ndim + 1, &pstr, ndim + 1, s));
    pstart[ndim] = 1;
    pstop[ndim] = 2;
    var xi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xi);
    try mlx.check(mlx.mlx_slice(&xi, pairs, &pstart, ndim + 1, &pstop, ndim + 1, &pstr, ndim + 1, s));
    // cos/sin -> [C, (1,)*(ndim-2), rd/2, 1] for broadcast over heads + pairs
    var cshape: [5]c_int = undefined;
    cshape[0] = shape[0];
    for (1..ndim - 1) |i| cshape[i] = 1;
    cshape[ndim - 1] = @divExact(rdc, 2);
    cshape[ndim] = 1;
    var cosb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cosb);
    try mlx.check(mlx.mlx_reshape(&cosb, cos_rows, &cshape, ndim + 1, s));
    var sinb0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinb0);
    try mlx.check(mlx.mlx_reshape(&sinb0, sin_rows, &cshape, ndim + 1, s));
    var sinb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinb);
    if (inverse) {
        const neg1 = mlx.mlx_array_new_float(-1.0);
        defer _ = mlx.mlx_array_free(neg1);
        try mlx.check(mlx.mlx_multiply(&sinb, sinb0, neg1, s));
    } else {
        try mlx.check(mlx.mlx_astype(&sinb, sinb0, .float32, s));
    }
    const a1 = try gpuOp2(mlx.mlx_multiply, xr, cosb, s);
    defer _ = mlx.mlx_array_free(a1);
    const a2 = try gpuOp2(mlx.mlx_multiply, xi, sinb, s);
    defer _ = mlx.mlx_array_free(a2);
    const yr = try gpuOp2(mlx.mlx_subtract, a1, a2, s);
    defer _ = mlx.mlx_array_free(yr);
    const b1 = try gpuOp2(mlx.mlx_multiply, xr, sinb, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try gpuOp2(mlx.mlx_multiply, xi, cosb, s);
    defer _ = mlx.mlx_array_free(b2);
    const yi = try gpuOp2(mlx.mlx_add, b1, b2, s);
    defer _ = mlx.mlx_array_free(yi);
    const rot_pairs = try gpuConcat2(yr, yi, @intCast(ndim), s);
    defer _ = mlx.mlx_array_free(rot_pairs);
    var tshape: [4]c_int = undefined;
    for (0..ndim) |i| tshape[i] = shape[i];
    tshape[ndim - 1] = rdc;
    var rot_tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot_tail);
    try mlx.check(mlx.mlx_reshape(&rot_tail, rot_pairs, &tshape, ndim, s));
    return try gpuConcat2(head, rot_tail, @intCast(ndim - 1), s);
}

fn gpuSliceLast3(x: mlx.mlx_array, d0: usize, d1: usize, c0: usize, c1: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ 0, 0, @intCast(c0) };
    const stop = [_]c_int{ @intCast(d0), @intCast(d1), @intCast(c1) };
    const strides = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, 3, &stop, 3, &strides, 3, s));
    return out;
}

/// Batched hc_pre on a [C, hc, d] stream: y [C, d], post [C, hc, 1], combT
/// [C, hc, hc]. Fused-kernel arm (per-C config, grid C) or host-Sinkhorn
/// fallback (one [C, mix+1] sync + per-token loop).
fn hcPreBatch(m: *Dsv4Model, alloc: std.mem.Allocator, stream_g: mlx.mlx_array, c_tokens: usize, fn_w_t: mlx.mlx_array, scale: []const f32, base: []const f32, scale_g: mlx.mlx_array, base_g: mlx.mlx_array) !HcPreG {
    const hcm = m.hc;
    const hd_full = hcm * m.dim;
    const mix = (2 + hcm) * hcm;
    const cc: c_int = @intCast(c_tokens);
    const fshape = [_]c_int{ cc, @intCast(hd_full) };
    const flat = try gpuReshape(stream_g, &fshape, m.s);
    defer _ = mlx.mlx_array_free(flat);
    const mixes_g = try gpuOp2(mlx.mlx_matmul, flat, fn_w_t, m.s);
    defer _ = mlx.mlx_array_free(mixes_g);
    const sq = try gpuOp1(mlx.mlx_square, flat, m.s);
    defer _ = mlx.mlx_array_free(sq);
    var ss = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ss);
    try mlx.check(mlx.mlx_sum_axis(&ss, sq, 1, true, m.s));
    var pre3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pre3);
    var post_g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(post_g);
    var combT_g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(combT_g);
    const col3 = [_]c_int{ cc, @intCast(hcm), 1 };
    const comb3 = [_]c_int{ cc, @intCast(hcm), @intCast(hcm) };
    const pack = 2 * hcm + hcm * hcm;
    if (m.sink_y_k) |*sk| {
        const cfg = try sinkhornYCfgFor(m, c_tokens);
        const r = try applySinkhornY(sk, cfg, mixes_g, ss, scale_g, base_g, m.sink_consts, stream_g, m.s);
        if (m.hc_post_k != null) {
            return .{ .y = r.y, .post_g = post_g, .combT_g = combT_g, .pk = r.pk };
        }
        defer _ = mlx.mlx_array_free(r.pk);
        const post_flat = try gpuSliceCols(r.pk, c_tokens, hcm, 2 * hcm, m.s);
        defer _ = mlx.mlx_array_free(post_flat);
        try mlx.check(mlx.mlx_reshape(&post_g, post_flat, &col3, 3, m.s));
        const comb_flat = try gpuSliceCols(r.pk, c_tokens, 2 * hcm, pack, m.s);
        defer _ = mlx.mlx_array_free(comb_flat);
        try mlx.check(mlx.mlx_reshape(&combT_g, comb_flat, &comb3, 3, m.s));
        return .{ .y = r.y, .post_g = post_g, .combT_g = combT_g };
    } else if (m.sink_k) |*sk| {
        const cfg = try sinkhornCfgFor(m, c_tokens);
        const pk = try applySinkhorn(sk, cfg, mixes_g, ss, scale_g, base_g, m.sink_consts, m.s);
        defer _ = mlx.mlx_array_free(pk);
        const pre_flat = try gpuSliceCols(pk, c_tokens, 0, hcm, m.s);
        defer _ = mlx.mlx_array_free(pre_flat);
        try mlx.check(mlx.mlx_reshape(&pre3, pre_flat, &col3, 3, m.s));
        const post_flat = try gpuSliceCols(pk, c_tokens, hcm, 2 * hcm, m.s);
        defer _ = mlx.mlx_array_free(post_flat);
        try mlx.check(mlx.mlx_reshape(&post_g, post_flat, &col3, 3, m.s));
        const comb_flat = try gpuSliceCols(pk, c_tokens, 2 * hcm, pack, m.s);
        defer _ = mlx.mlx_array_free(comb_flat);
        try mlx.check(mlx.mlx_reshape(&combT_g, comb_flat, &comb3, 3, m.s));
    } else {
        const joint = try gpuConcat2(mixes_g, ss, 1, m.s);
        defer _ = mlx.mlx_array_free(joint);
        const rows = try toHostF32(alloc, joint, c_tokens * (mix + 1), m.s); // fallback sync
        defer alloc.free(rows);
        const pre_h = try alloc.alloc(f32, c_tokens * hcm);
        defer alloc.free(pre_h);
        const post_h = try alloc.alloc(f32, c_tokens * hcm);
        defer alloc.free(post_h);
        const combT_h = try alloc.alloc(f32, c_tokens * hcm * hcm);
        defer alloc.free(combT_h);
        for (0..c_tokens) |t| {
            const row = rows[t * (mix + 1) ..][0 .. mix + 1];
            const rsq: f32 = @floatCast(1.0 / @sqrt(@as(f64, row[mix]) / @as(f64, @floatFromInt(hd_full)) + m.eps));
            var mm: [96]f32 = undefined;
            for (0..mix) |j| mm[j] = row[j] * rsq;
            const split = hcSplitSinkhorn(mm[0..mix], scale, base, hcm, m.hc_iters, m.hc_eps);
            @memcpy(pre_h[t * hcm ..][0..hcm], split.pre[0..hcm]);
            @memcpy(post_h[t * hcm ..][0..hcm], split.post[0..hcm]);
            for (0..hcm) |k| {
                for (0..hcm) |j| combT_h[t * hcm * hcm + k * hcm + j] = split.comb[j * hcm + k];
            }
        }
        _ = mlx.mlx_array_free(pre3);
        pre3 = uploadF32(pre_h, &col3);
        _ = mlx.mlx_array_free(post_g);
        post_g = uploadF32(post_h, &col3);
        _ = mlx.mlx_array_free(combT_g);
        combT_g = uploadF32(combT_h, &comb3);
    }
    const weighted = try gpuOp2(mlx.mlx_multiply, pre3, stream_g, m.s);
    defer _ = mlx.mlx_array_free(weighted);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_sum_axis(&y, weighted, 1, false, m.s)); // [C, d]
    return .{ .y = y, .post_g = post_g, .combT_g = combT_g };
}

/// Batched hc_post: new_stream[t] = combᵀ[t] @ residual[t] + post[t]·out[t].
fn hcPostBatch(m: *Dsv4Model, stream_g: mlx.mlx_array, out_g: mlx.mlx_array, c_tokens: usize, pre: *const HcPreG) !mlx.mlx_array {
    if (pre.pk) |pk| {
        const sk = &m.hc_post_k.?;
        const cfg = try hcPostCfgFor(m, c_tokens);
        return try applyHcPost(sk, cfg, pk, stream_g, out_g, m.s); // [C, hc, d]
    }
    const res = try gpuOp2(mlx.mlx_matmul, pre.combT_g, stream_g, m.s);
    defer _ = mlx.mlx_array_free(res);
    const oshape = [_]c_int{ @intCast(c_tokens), 1, @intCast(m.dim) };
    const o3 = try gpuReshape(out_g, &oshape, m.s);
    defer _ = mlx.mlx_array_free(o3);
    const po = try gpuOp2(mlx.mlx_multiply, pre.post_g, o3, m.s);
    defer _ = mlx.mlx_array_free(po);
    return try gpuOp2(mlx.mlx_add, res, po, m.s);
}

/// Batched sparse attention for C tokens at global positions [base, base+C):
/// the batched mirror of attentionDecodeGpu. kv rows + compressed slots for
/// the WHOLE chunk are appended before scoring; causality is the window band
/// (never past a token's own position) + the per-token compressed-visibility
/// mask ((pos+1)/ratio — the reference prefill's rule).
/// Host half of one layer's compressor over a chunk: the per-token pushes
/// (and their rollback anchors), then ONE append of whatever slots the chunk
/// emitted. Shared by the in-layer and deferred paths so they cannot drift.
fn pushCompChunk(m: *Dsv4Model, st: *Dsv4DecodeState, li: usize, rows: []const f32, base: usize, C: usize, fr: *const Freqs, alloc: std.mem.Allocator) !void {
    const h = &m.hl[li];
    const ls = &st.layers[li];
    const w2 = h.comp_in_w;
    const ratio: usize = m.dw.layers[li].compress_ratio;
    const c = &h.comp.?;
    const cd = c.coff * c.head_dim;
    // Mirror appends are suppressed under GPU emission — the in-layer GPU
    // emission already appended these slots (host cache still fills for
    // snapshots/anchors).
    const emit_gpu = gpuEmitActive(m);
    {
        const csx = &ls.comp.?;
        const before = csx.cache.items.len;
        for (0..C) |t| {
            const row = rows[t * w2 ..][0..w2];
            if (emit_gpu)
                try compressorPushLight(c, csx, row[0..cd], row[cd .. 2 * cd], base + t, ratio)
            else
                try compressorPush(m, c, csx, row[0..cd], row[cd .. 2 * cd], base + t, ratio, false, fr, alloc);
            if (st.anchors) |*an| an.captureComp(t, li, csx, false);
        }
        if (!emit_gpu and csx.cache.items.len > before)
            try ls.comp_gpu.append(csx.cache.items[before..], m.s);
    }
    if (ls.idx_comp) |*csx| {
        const ic = &h.idx_comp.?;
        const icd = ic.coff * ic.head_dim;
        const before = csx.cache.items.len;
        for (0..C) |t| {
            const row = rows[t * w2 ..][0..w2];
            if (emit_gpu)
                try compressorPushLight(ic, csx, row[2 * cd ..][0..icd], row[2 * cd + icd ..][0..icd], base + t, 4)
            else
                try compressorPush(m, ic, csx, row[2 * cd ..][0..icd], row[2 * cd + icd ..][0..icd], base + t, 4, true, fr, alloc);
            if (st.anchors) |*an| an.captureComp(t, li, csx, true);
        }
        if (!emit_gpu and csx.cache.items.len > before)
            try ls.idx_gpu.append(csx.cache.items[before..], m.s);
    }
}

/// Drain the layers whose compressor read was deferred: ONE batched eval for
/// every layer's [C, W] rows, then the same host pushes they would have done
/// in-layer. The decode sibling is `processDeferredComp`.
fn processDeferredCompChunk(m: *Dsv4Model, st: *Dsv4DecodeState, alloc: std.mem.Allocator, rows: []const DeferredCompRow, base: usize, C: usize, fr: *const Freqs) !void {
    if (rows.len == 0) return;
    var sclk: DsparkClock = if (m.ds_prof != null) DsparkClock.init() else undefined;
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    for (rows) |r| _ = mlx.mlx_vector_array_append_value(ev, r.arr);
    try mlx.check(mlx.mlx_eval(ev));
    if (m.ds_prof != null) m.ds_prof_comp_sync_ns += sclk.lap();
    for (rows) |r| {
        const host = try toHostF32(alloc, r.arr, C * m.hl[r.li].comp_in_w, m.s);
        defer alloc.free(host);
        try pushCompChunk(m, st, r.li, host, base, C, fr, alloc);
    }
}

/// Slice rows [r0, r1) × cols [c0, c1) of a 2-D array.
fn gpuSliceRC(x: mlx.mlx_array, r0: usize, r1: usize, c0: usize, c1: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const start = [_]c_int{ @intCast(r0), @intCast(c0) };
    const stop = [_]c_int{ @intCast(r1), @intCast(c1) };
    const strides = [_]c_int{ 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, 2, &stop, 2, &strides, 2, s));
    return out;
}

/// Rows [off, off + n_win*r) of an extended row matrix, stacked per window:
/// [n_win, r, cd].
fn winStack(ext: mlx.mlx_array, off: usize, n_win: usize, r: usize, cd: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const sl = try gpuSliceRC(ext, off, off + n_win * r, 0, cd, s);
    defer _ = mlx.mlx_array_free(sl);
    const shp = [_]c_int{ @intCast(n_win), @intCast(r), @intCast(cd) };
    return gpuReshape(sl, &shp, s);
}

/// Fused window emission: one threadgroup per closed window runs the whole
/// tail (masked-softmax combine → RMSNorm → rope tail → [Hadamard →] QAT sim)
/// with the ext-row indexing (pre-ring rows vs comp_in chunk rows + ape)
/// resolved in-kernel. Softmax/RMS reduction trees are ascending-f32 like the
/// composed chain (drift snapped by the sims; decode-equivalence arbitrates,
/// sinkhorn-kernel precedent); the QAT sim's scale/grid exponents use
/// frexp/ldexp bit arithmetic — EXACT, matching the host f64 semantics rather
/// than the composed chain's f32 log2/ceil approximations.
const EMIT_WIN_KERNEL_SOURCE =
    \\int w = thread_position_in_grid.y;
    \\int lane = thread_position_in_threadgroup.x;
    \\if (w >= NWIN) return; // uniform per threadgroup (grid.y == NWIN)
    \\const int CD = (OVERLAP != 0 ? 2 : 1) * D;
    \\const int ROWS = (OVERLAP != 0 ? 2 : 1) * R;
    \\const int HALF = RD / 2;
    \\const int QW = (ROTATE != 0) ? D : (D - RD);
    \\const int NG = QW / GROUP;
    \\const float CODE_MAX = (ROTATE != 0) ? 6.0f : 448.0f;
    \\const float AMAX_FLOOR = (ROTATE != 0) ? metal::ldexp(6.0f, -126) : 1e-4f;
    \\const int MANT_BITS = (ROTATE != 0) ? 1 : 3;
    \\const int MIN_EXP = (ROTATE != 0) ? 0 : -6;
    \\const float MIN_SUB = metal::ldexp(1.0f, MIN_EXP);
    \\threadgroup float row[D];
    \\threadgroup float row2[(ROTATE != 0) ? D : 1];
    \\threadgroup float red[TG];
    \\threadgroup float gsc[(QW / GROUP) > 0 ? (QW / GROUP) : 1];
    \\// phase 1: per-feature masked-softmax combine over the window rows.
    \\// 3 passes (max / exp-sum / weighted acc) re-reading the sources — the
    \\// composed chain's softmax→multiply→sum rounding order, no register
    \\// caching so ratio-128 windows (ROWS 128) don't spill.
    \\for (int j = lane; j < D; j += TG) {
    \\  float mx = -INFINITY;
    \\  float sum = 0.0f;
    \\  float acc = 0.0f;
    \\  float inv = 0.0f;
    \\  for (int pass = 0; pass < 3; ++pass) {
    \\    if (pass == 2) inv = 1.0f / sum;
    \\    for (int i = 0; i < ROWS; ++i) {
    \\      int ext; int col;
    \\      if (OVERLAP != 0) {
    \\        if (i < R) { ext = w * R + i; col = j; }
    \\        else { ext = R + w * R + (i - R); col = D + j; }
    \\      } else { ext = w * R + i; col = j; }
    \\      float kv; float sc;
    \\      if (ext < PRE_N) {
    \\        kv = pre_kv[ext * CD + col];
    \\        sc = pre_sc[ext * CD + col];
    \\      } else {
    \\        int cr = ext - PRE_N;
    \\        kv = cin[cr * CIN_W + COL_OFF + col];
    \\        sc = cin[cr * CIN_W + COL_OFF + CD + col] + ape[((BASE_MOD + cr) % R) * CD + col];
    \\      }
    \\      if (pass == 0) mx = metal::max(mx, sc);
    \\      else if (pass == 1) sum += metal::exp(sc - mx);
    \\      else acc += (metal::exp(sc - mx) * inv) * kv;
    \\    }
    \\  }
    \\  row[j] = acc;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\// phase 2: RMSNorm (tree-reduced sum of squares, weight applied)
    \\float ss = 0.0f;
    \\for (int j = lane; j < D; j += TG) ss += row[j] * row[j];
    \\red[lane] = ss;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\for (int off = TG / 2; off > 0; off >>= 1) {
    \\  if (lane < off) red[lane] += red[lane + off];
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\float rinv = metal::rsqrt(red[0] / (float)D + consts[0]);
    \\for (int j = lane; j < D; j += TG) row[j] = row[j] * rinv * nw[j];
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\// phase 3: interleaved-pair rope on the tail RD dims at the block start
    \\for (int p = lane; p < HALF; p += TG) {
    \\  float a = row[D - RD + 2 * p];
    \\  float b = row[D - RD + 2 * p + 1];
    \\  float cv = cosr[w * HALF + p];
    \\  float sv = sinr[w * HALF + p];
    \\  row[D - RD + 2 * p] = a * cv - b * sv;
    \\  row[D - RD + 2 * p + 1] = a * sv + b * cv;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\// phase 4 (indexer only): Hadamard rotate = row @ hada
    \\if (ROTATE != 0) {
    \\  for (int j = lane; j < D; j += TG) {
    \\    float hacc = 0.0f;
    \\    for (int kk = 0; kk < D; ++kk) hacc += row[kk] * hada[kk * D + j];
    \\    row2[j] = hacc;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\// phase 5: ue8m0 per-group scales — 2^ceil(log2(amax/code_max)), exact
    \\for (int g = lane; g < NG; g += TG) {
    \\  float amax = AMAX_FLOOR;
    \\  for (int e = 0; e < GROUP; ++e) {
    \\    float v = (ROTATE != 0) ? row2[g * GROUP + e] : row[g * GROUP + e];
    \\    amax = metal::max(amax, metal::abs(v));
    \\  }
    \\  float q = amax / CODE_MAX;
    \\  int ee = 0;
    \\  float mm = metal::frexp(q, ee);
    \\  int ce = (mm == 0.5f) ? (ee - 1) : ee;
    \\  gsc[g] = metal::ldexp(1.0f, ce);
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\// phase 6: quant-dequant (rope tail stays raw on the fp8 path) + write
    \\for (int j = lane; j < D; j += TG) {
    \\  float v = (ROTATE != 0) ? row2[j] : row[j];
    \\  float res = v;
    \\  if (ROTATE != 0 || j < D - RD) {
    \\    float scale = gsc[j / GROUP];
    \\    float y = metal::clamp(v / scale, -CODE_MAX, CODE_MAX);
    \\    float a = metal::abs(y);
    \\    float af = metal::max(a, MIN_SUB);
    \\    int ee2 = 0;
    \\    float mm2 = metal::frexp(af, ee2);
    \\    int e2 = ee2 - 1; // floor(log2(af)): mm2 in [0.5, 1)
    \\    (void)mm2;
    \\    if (e2 < MIN_EXP) e2 = MIN_EXP;
    \\    float quantum = metal::ldexp(1.0f, e2 - MANT_BITS);
    \\    float qv = metal::min(metal::rint(a / quantum) * quantum, CODE_MAX);
    \\    float sq = (y < 0.0f) ? -qv : ((y > 0.0f) ? qv : 0.0f);
    \\    res = sq * scale;
    \\  }
    \\  outw[w * D + j] = res;
    \\}
;

var emit_kernel_obj: ?mlx.mlx_fast_metal_kernel = null;
var emit_kernel_build_failed: bool = false;
fn emitKernelObj() ?mlx.mlx_fast_metal_kernel {
    if (emit_kernel_build_failed) return null;
    if (emit_kernel_obj) |kk| return kk;
    const input_names = [_][*:0]const u8{ "cin", "pre_kv", "pre_sc", "ape", "nw", "cosr", "sinr", "hada", "consts" };
    const output_names = [_][*:0]const u8{"outw"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_emit_win", in_vec, out_vec, EMIT_WIN_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) {
        emit_kernel_build_failed = true;
        log.warn("dsv4: fused emission kernel failed to build — composed graph fallback\n", .{});
        return null;
    }
    emit_kernel_obj = kernel;
    return kernel;
}

/// Everything the emission kernel bakes into its config (output shape, grid,
/// template ints) — the cache key is the FULL tuple (ShapeKey rule).
const EmitCfgKey = struct {
    d: u32,
    rd: u32,
    r: u32,
    overlap: bool,
    rotate: bool,
    pre_n: u32,
    col_off: u32,
    cin_w: u32,
    base_mod: u32,
    n_win: u32,
    tg: u32,
};
const EmitCfgSlot = struct { key: EmitCfgKey, cfg: mlx.mlx_fast_metal_kernel_config };
/// Decode/verify emission geometries are a handful of stable tuples; prefill
/// chunks vary per chunk and build-and-free instead (bounded table, no LRU).
var emit_cfg_cache: [64]?EmitCfgSlot = @splat(null);

fn emitWinCfg(key: EmitCfgKey) ?mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{ @intCast(key.n_win), @intCast(key.d) };
    const group: c_int = if (key.rotate) 32 else 64;
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float32) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(key.tg), @intCast(key.n_win), 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(key.tg), 1, 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", @intCast(key.d)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RD", @intCast(key.rd)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", @intCast(key.r)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "OVERLAP", @intFromBool(key.overlap)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROTATE", @intFromBool(key.rotate)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GROUP", group) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PRE_N", @intCast(key.pre_n)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "COL_OFF", @intCast(key.col_off)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CIN_W", @intCast(key.cin_w)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BASE_MOD", @intCast(key.base_mod)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NWIN", @intCast(key.n_win)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", @intCast(key.tg)) != 0)
    {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        return null;
    }
    return cfg;
}

/// Largest power of two ≤ min(d, 256) — the tree reduction needs a pow-2
/// threadgroup; every house D (32/96/128/512) maps to a full-occupancy size.
fn emitTgFor(d: usize) usize {
    var tg: usize = 1;
    while (tg * 2 <= @min(d, 256)) tg *= 2;
    return tg;
}

/// Fused single-dispatch arm of `emitWindowsGpu`: replaces the ~60-op
/// composed emission graph (the +4.5 ms boundary-token tax at decode) with
/// one `dsv4_emit_win` dispatch per compressor. Returns true when it handled
/// the emission (mirror appended), false to decline to the composed path
/// (kill switch, ineligible geometry, build failure).
fn emitWindowsKernel(
    m: *const Dsv4Model,
    c: *const HostComp,
    cs: *const CompDecState,
    mirror: *GpuRows,
    comp_in: mlx.mlx_array,
    col_off: usize,
    base: usize,
    r: usize,
    rotate: bool,
    fr: *const Freqs,
    alloc: std.mem.Allocator,
    W: usize,
    n_win: usize,
    pre_n: usize,
    used_c: usize,
) !bool {
    if (!emitKernelEnabled() or !mlx.streamIsGpu(m.s)) return false; // metal_kernel is GPU-only (CPU stream = uncatchable kill)
    const d = c.head_dim;
    const cd = c.coff * d;
    // eligibility is the kernel's own conditions, never a model list
    if (d > 1024 or m.rd == 0 or m.rd % 2 != 0) return false;
    if (rotate) {
        if (d % 32 != 0) return false;
        const hg = m.hada_g orelse return false;
        if (mlx.mlx_array_shape(hg)[0] != @as(c_int, @intCast(d))) return false;
    } else {
        if (d <= m.rd or (d - m.rd) % 64 != 0) return false;
    }
    const kernel = emitKernelObj() orelse return false;
    const tg = emitTgFor(d);
    const key = EmitCfgKey{
        .d = @intCast(d),
        .rd = @intCast(m.rd),
        .r = @intCast(r),
        .overlap = c.coff == 2,
        .rotate = rotate,
        .pre_n = @intCast(pre_n),
        .col_off = @intCast(col_off),
        .cin_w = @intCast(mlx.mlx_array_shape(comp_in)[1]),
        .base_mod = @intCast(base % r),
        .n_win = @intCast(n_win),
        .tg = @intCast(tg),
    };
    _ = used_c;
    // decode/verify tuples (n_win ≤ 2) are stable — cache; chunk geometries
    // vary per chunk and are built fresh + freed after apply.
    var cfg: mlx.mlx_fast_metal_kernel_config = undefined;
    var cached = false;
    if (key.n_win <= 2) {
        for (&emit_cfg_cache) |*slot| {
            if (slot.*) |sl| {
                if (std.meta.eql(sl.key, key)) {
                    cfg = sl.cfg;
                    cached = true;
                    break;
                }
            } else {
                const built = emitWinCfg(key) orelse return false;
                slot.* = .{ .key = key, .cfg = built };
                cfg = built;
                cached = true;
                break;
            }
        }
    }
    if (!cached) cfg = emitWinCfg(key) orelse return false;
    defer if (!cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };

    // pre-ring staging (same layout as the composed path's ext build)
    const stage_n = @max(pre_n, 1);
    const stage_kv = try alloc.alloc(f32, stage_n * cd);
    defer alloc.free(stage_kv);
    const stage_sc = try alloc.alloc(f32, stage_n * cd);
    defer alloc.free(stage_sc);
    if (pre_n == 0) {
        stage_kv[0] = 0;
        stage_sc[0] = 0;
    } else {
        const overlap = c.coff == 2;
        const k = base % r;
        var row: usize = 0;
        if (overlap) {
            for (0..r) |i| {
                @memcpy(stage_kv[row * cd ..][0..cd], cs.kv_pend[i * cs.width ..][0..cd]);
                @memcpy(stage_sc[row * cd ..][0..cd], cs.sc_pend[i * cs.width ..][0..cd]);
                row += 1;
            }
        }
        const cur0: usize = if (overlap) r else 0;
        for (0..k) |i| {
            @memcpy(stage_kv[row * cd ..][0..cd], cs.kv_pend[(cur0 + i) * cs.width ..][0..cd]);
            @memcpy(stage_sc[row * cd ..][0..cd], cs.sc_pend[(cur0 + i) * cs.width ..][0..cd]);
            row += 1;
        }
        std.debug.assert(row == pre_n);
    }
    const pshape = [_]c_int{ @intCast(stage_n), @intCast(cd) };
    const pre_kv_g = uploadF32(stage_kv, &pshape);
    defer _ = mlx.mlx_array_free(pre_kv_g);
    const pre_sc_g = uploadF32(stage_sc, &pshape);
    defer _ = mlx.mlx_array_free(pre_sc_g);

    const half = m.rd / 2;
    const cos_h = try alloc.alloc(f32, n_win * half);
    defer alloc.free(cos_h);
    const sin_h = try alloc.alloc(f32, n_win * half);
    defer alloc.free(sin_h);
    for (0..n_win) |j| {
        const bs = (W + j) * r;
        @memcpy(cos_h[j * half ..][0..half], fr.cos[bs * half ..][0..half]);
        @memcpy(sin_h[j * half ..][0..half], fr.sin[bs * half ..][0..half]);
    }
    const rshape = [_]c_int{ @intCast(n_win), @intCast(half) };
    const cos_g = uploadF32(cos_h, &rshape);
    defer _ = mlx.mlx_array_free(cos_g);
    const sin_g = uploadF32(sin_h, &rshape);
    defer _ = mlx.mlx_array_free(sin_g);
    const eps_arr = [_]f32{m.eps};
    const eshape = [_]c_int{1};
    const eps_g = uploadF32(&eps_arr, &eshape);
    defer _ = mlx.mlx_array_free(eps_g);
    // non-rotate never reads hada — norm_g stands in to keep the input list fixed
    const hada_in = if (rotate) m.hada_g.? else c.norm_g;

    const inputs_arr = [_]mlx.mlx_array{ comp_in, pre_kv_g, pre_sc_g, c.ape_g, c.norm_g, cos_g, sin_g, hada_in, eps_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, m.s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    try mirror.appendGpu(out, n_win, m.s);
    emit_kernel_hits += 1;
    if (!emit_kernel_logged) {
        emit_kernel_logged = true;
        log.info("dsv4: fused emission kernel engaged\n", .{});
    }
    return true;
}

/// Fused decode attention-chain tail: per-head [RMS →] rope tail →
/// [fp8-head-sim | Hadamard+fp4] in ONE dispatch, replacing the composed
/// ~15-25-op chains around the decode qmvs (q / kv / indexer-q / o-inverse).
/// post: 0 = rope only, 1 = fp8 head sim + raw roped tail, 2 = Hadamard+fp4.
/// Returns null to decline (kill switch, ineligible geometry, build failure)
/// — the caller keeps the composed chain.
/// Row softmax with the sink folded into the max and the denominator, sink
/// column dropped at the write — one threadgroup per head, tree reductions.
/// exp/reduction-tree drift vs the composed mlx softmax is the sanctioned
/// sinkhorn-kernel class (continuous path, decode-equivalence arbitrates).
const SINK_SOFTMAX_KERNEL_SOURCE =
    \\int h = thread_position_in_grid.y;
    \\int lane = thread_position_in_threadgroup.x;
    \\if (h >= H) return; // uniform per threadgroup (grid.y == H)
    \\// TK is an INPUT, not a template arg: it changes every token during the
    \\// context ramp, and a template value would JIT a fresh Metal kernel per
    \\// tk (measured: first request 19 tok/s vs 29.7 warm).
    \\int TK = int(tk_size[0]);
    \\threadgroup float red[TG];
    \\float sk = sink[h];
    \\float mx = sk;
    \\for (int j = lane; j < TK; j += TG) mx = metal::max(mx, scores[h * TK + j] * consts[0]);
    \\red[lane] = mx;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\for (int off = TG / 2; off > 0; off >>= 1) {
    \\  if (lane < off) red[lane] = metal::max(red[lane], red[lane + off]);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\float m = red[0];
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float sum = 0.0f;
    \\for (int j = lane; j < TK; j += TG) sum += metal::exp(scores[h * TK + j] * consts[0] - m);
    \\red[lane] = sum;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\for (int off = TG / 2; off > 0; off >>= 1) {
    \\  if (lane < off) red[lane] += red[lane + off];
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\float inv = 1.0f / (red[0] + metal::exp(sk - m));
    \\for (int j = lane; j < TK; j += TG) {
    \\  y[h * TK + j] = metal::exp(scores[h * TK + j] * consts[0] - m) * inv;
    \\}
;

var sink_softmax_obj: ?mlx.mlx_fast_metal_kernel = null;
var sink_softmax_build_failed: bool = false;
fn sinkSoftmaxObj() ?mlx.mlx_fast_metal_kernel {
    if (sink_softmax_build_failed) return null;
    if (sink_softmax_obj) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "sink", "consts", "tk_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_sink_softmax", in_vec, out_vec, SINK_SOFTMAX_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) {
        sink_softmax_build_failed = true;
        log.warn("dsv4: fused sink-softmax kernel failed to build — composed chain fallback\n", .{});
        return null;
    }
    sink_softmax_obj = kernel;
    return kernel;
}

const SinkSmKey = struct { h: u32, tk: u32 };
const SinkSmSlot = struct { key: SinkSmKey, cfg: mlx.mlx_fast_metal_kernel_config };
/// tk differs per ratio class within one token (~4 live values at steady
/// state) and grows during the ramp — an 8-slot rotating table keeps the
/// steady-state configs cached without unbounded growth.
var sink_sm_cfg_cache: [8]?SinkSmSlot = @splat(null);
var sink_sm_cfg_next: usize = 0;

/// Fused decode sink-softmax: scale → sink column in the denominator →
/// row softmax → drop the sink, in ONE dispatch between the two attention
/// GEMMs (the composed chain is 4 strictly-serial dispatches per layer).
/// Returns probs [H, TK] f32, or null to decline.
fn sinkSoftmaxKernel(m: *const Dsv4Model, scores0: mlx.mlx_array, sink: mlx.mlx_array, H: usize, TK: usize, scale: f32) !?mlx.mlx_array {
    if (!sinkSoftmaxEnabled() or !mlx.streamIsGpu(m.s)) return null; // metal_kernel is GPU-only
    if (TK == 0 or H == 0 or H > 65535) return null;
    const kernel = sinkSoftmaxObj() orelse return null;
    const tg: usize = @min(256, std.math.ceilPowerOfTwoAssert(usize, @max(TK, 2)));
    const key = SinkSmKey{ .h = @intCast(H), .tk = @intCast(TK) };
    var cfg: mlx.mlx_fast_metal_kernel_config = undefined;
    var cached = false;
    for (&sink_sm_cfg_cache) |*slot| {
        if (slot.*) |sl| {
            if (std.meta.eql(sl.key, key)) {
                cfg = sl.cfg;
                cached = true;
                break;
            }
        }
    }
    if (!cached) {
        const built = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ @intCast(H), @intCast(TK) };
        if (mlx.mlx_fast_metal_kernel_config_add_output_arg(built, &y_shape, 2, .float32) != 0 or
            mlx.mlx_fast_metal_kernel_config_set_grid(built, @intCast(tg), @intCast(H), 1) != 0 or
            mlx.mlx_fast_metal_kernel_config_set_thread_group(built, @intCast(tg), 1, 1) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_int(built, "H", @intCast(H)) != 0 or
            mlx.mlx_fast_metal_kernel_config_add_template_arg_int(built, "TG", @intCast(tg)) != 0)
        {
            _ = mlx.mlx_fast_metal_kernel_config_free(built);
            return null;
        }
        // rotate a slot (frees the evicted config — never one in flight:
        // apply consumes the config synchronously at graph-build time)
        if (sink_sm_cfg_cache[sink_sm_cfg_next]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old.cfg);
        sink_sm_cfg_cache[sink_sm_cfg_next] = .{ .key = key, .cfg = built };
        sink_sm_cfg_next = (sink_sm_cfg_next + 1) % sink_sm_cfg_cache.len;
        cfg = built;
    }
    const consts = [_]f32{scale};
    const cshape = [_]c_int{1};
    const consts_g = uploadF32(&consts, &cshape);
    defer _ = mlx.mlx_array_free(consts_g);
    const tk_val = [_]i32{@intCast(TK)};
    const tk_g = mlx.mlx_array_new_data(&tk_val, &cshape, 1, .int32);
    defer _ = mlx.mlx_array_free(tk_g);
    const inputs_arr = [_]mlx.mlx_array{ scores0, sink, consts_g, tk_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, m.s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    sink_softmax_hits += 1;
    if (!sink_softmax_logged) {
        sink_softmax_logged = true;
        log.info("dsv4: fused sink-softmax kernel engaged\n", .{});
    }
    return out;
}

const DEC_CHAIN_KERNEL_SOURCE =
    \\int h = thread_position_in_grid.y;
    \\int lane = thread_position_in_threadgroup.x;
    \\if (h >= H) return; // uniform per threadgroup (grid.y == H)
    \\const int HALF = RD / 2;
    \\const int QW = (POST == 1) ? (D - RD) : ((POST == 2) ? D : 0);
    \\const int GROUP = (POST == 2) ? 32 : 64;
    \\const float CODE_MAX = (POST == 2) ? 6.0f : 448.0f;
    \\const float AMAX_FLOOR = (POST == 2) ? metal::ldexp(6.0f, -126) : 1e-4f;
    \\const int MANT_BITS = (POST == 2) ? 1 : 3;
    \\const int MIN_EXP = (POST == 2) ? 0 : -6;
    \\const float MIN_SUB = metal::ldexp(1.0f, MIN_EXP);
    \\threadgroup float row[D];
    \\threadgroup float row2[(POST == 2) ? D : 1];
    \\threadgroup float red[TG];
    \\threadgroup float gsc[(QW > 0) ? (QW / GROUP) : 1];
    \\// load (+ per-head RMSNorm)
    \\if (RMS != 0) {
    \\  float ss = 0.0f;
    \\  for (int j = lane; j < D; j += TG) { float v = x[h * D + j]; ss += v * v; }
    \\  red[lane] = ss;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int off = TG / 2; off > 0; off >>= 1) {
    \\    if (lane < off) red[lane] += red[lane + off];
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\  float rinv = metal::rsqrt(red[0] / (float)D + consts[0]);
    \\  for (int j = lane; j < D; j += TG) row[j] = x[h * D + j] * rinv * w[j];
    \\} else {
    \\  for (int j = lane; j < D; j += TG) row[j] = x[h * D + j];
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\// interleaved-pair rope on the tail RD dims (INV = conjugate rotation)
    \\for (int p = lane; p < HALF; p += TG) {
    \\  float a = row[D - RD + 2 * p];
    \\  float b = row[D - RD + 2 * p + 1];
    \\  float cv = cosr[p];
    \\  float sv = (INV != 0) ? -sinr[p] : sinr[p];
    \\  row[D - RD + 2 * p] = a * cv - b * sv;
    \\  row[D - RD + 2 * p + 1] = a * sv + b * cv;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (POST == 2) {
    \\  for (int j = lane; j < D; j += TG) {
    \\    float hacc = 0.0f;
    \\    for (int kk = 0; kk < D; ++kk) hacc += row[kk] * hada[kk * D + j];
    \\    row2[j] = hacc;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\if (QW > 0) {
    \\  const int NG = QW / GROUP;
    \\  for (int g = lane; g < NG; g += TG) {
    \\    float amax = AMAX_FLOOR;
    \\    for (int e = 0; e < GROUP; ++e) {
    \\      float v = (POST == 2) ? row2[g * GROUP + e] : row[g * GROUP + e];
    \\      amax = metal::max(amax, metal::abs(v));
    \\    }
    \\    float q = amax / CODE_MAX;
    \\    int ee = 0;
    \\    float mm = metal::frexp(q, ee);
    \\    int ce = (mm == 0.5f) ? (ee - 1) : ee;
    \\    gsc[g] = metal::ldexp(1.0f, ce);
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\for (int j = lane; j < D; j += TG) {
    \\  float v = (POST == 2) ? row2[j] : row[j];
    \\  float res = v;
    \\  if (POST == 2 || (POST == 1 && j < D - RD)) {
    \\    float scale = gsc[j / GROUP];
    \\    float y = metal::clamp(v / scale, -CODE_MAX, CODE_MAX);
    \\    float a = metal::abs(y);
    \\    float af = metal::max(a, MIN_SUB);
    \\    int ee2 = 0;
    \\    float mm2 = metal::frexp(af, ee2);
    \\    int e2 = ee2 - 1; // floor(log2(af)): mm2 in [0.5, 1)
    \\    (void)mm2;
    \\    if (e2 < MIN_EXP) e2 = MIN_EXP;
    \\    float quantum = metal::ldexp(1.0f, e2 - MANT_BITS);
    \\    float qv = metal::min(metal::rint(a / quantum) * quantum, CODE_MAX);
    \\    float sq = (y < 0.0f) ? -qv : ((y > 0.0f) ? qv : 0.0f);
    \\    res = sq * scale;
    \\  }
    \\  y_out[h * D + j] = res;
    \\}
;

var dec_chain_obj: ?mlx.mlx_fast_metal_kernel = null;
var dec_chain_build_failed: bool = false;
fn decChainObj() ?mlx.mlx_fast_metal_kernel {
    if (dec_chain_build_failed) return null;
    if (dec_chain_obj) |kk| return kk;
    const input_names = [_][*:0]const u8{ "x", "w", "cosr", "sinr", "hada", "consts" };
    const output_names = [_][*:0]const u8{"y_out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("dsv4_dec_chain", in_vec, out_vec, DEC_CHAIN_KERNEL_SOURCE, "", true, false);
    if (kernel.ctx == null) {
        dec_chain_build_failed = true;
        log.warn("dsv4: fused decode-chain kernel failed to build — composed chain fallback\n", .{});
        return null;
    }
    dec_chain_obj = kernel;
    return kernel;
}

const DecChainKey = struct { h: u32, d: u32, rd: u32, rms: bool, post: u8, inv: bool, tg: u32 };
const DecChainSlot = struct { key: DecChainKey, cfg: mlx.mlx_fast_metal_kernel_config };
/// Decode chains have a handful of per-model geometries — small fixed cache.
var dec_chain_cfg_cache: [16]?DecChainSlot = @splat(null);

fn decChainCfg(key: DecChainKey) ?mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    const out_shape = [_]c_int{ @intCast(key.h), @intCast(key.d) };
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .float32) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(key.tg), @intCast(key.h), 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(key.tg), 1, 1) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", @intCast(key.d)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", @intCast(key.h)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RD", @intCast(key.rd)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RMS", @intFromBool(key.rms)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "POST", key.post) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "INV", @intFromBool(key.inv)) != 0 or
        mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", @intCast(key.tg)) != 0)
    {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        return null;
    }
    return cfg;
}

/// Fused decode attention-chain tail: per-head [RMS →] rope tail →
/// [fp8-head-sim | Hadamard+fp4] in ONE dispatch, replacing the composed
/// ~15-25-op chains around the decode qmvs (q / kv / indexer-q / o-inverse).
/// post: 0 = rope only, 1 = fp8 head sim + raw roped tail, 2 = Hadamard+fp4.
/// Returns null to decline (kill switch, ineligible geometry, build failure)
/// — the caller keeps the composed chain.
fn decChainKernel(
    m: *const Dsv4Model,
    x: mlx.mlx_array,
    H: usize,
    D: usize,
    rd: usize,
    rms_w: ?mlx.mlx_array,
    post: u8,
    inverse: bool,
    rr: *const RopeRows,
) !?mlx.mlx_array {
    if (!decChainEnabled() or !mlx.streamIsGpu(m.s)) return null; // metal_kernel is GPU-only (CPU stream = uncatchable kill)
    // eligibility is the kernel's own conditions, never a model list
    if (D > 1024 or rd == 0 or rd % 2 != 0 or rd > D) return null;
    if (post == 1 and (D <= rd or (D - rd) % 64 != 0)) return null;
    var hada_in: mlx.mlx_array = undefined;
    if (post == 2) {
        if (D % 32 != 0) return null;
        const hg = m.hada_g orelse return null;
        if (mlx.mlx_array_shape(hg)[0] != @as(c_int, @intCast(D))) return null;
        hada_in = hg;
    }
    const kernel = decChainObj() orelse return null;
    const key = DecChainKey{
        .h = @intCast(H),
        .d = @intCast(D),
        .rd = @intCast(rd),
        .rms = rms_w != null,
        .post = post,
        .inv = inverse,
        .tg = @intCast(emitTgFor(D)),
    };
    var cfg: mlx.mlx_fast_metal_kernel_config = undefined;
    var cached = false;
    for (&dec_chain_cfg_cache) |*slot| {
        if (slot.*) |sl| {
            if (std.meta.eql(sl.key, key)) {
                cfg = sl.cfg;
                cached = true;
                break;
            }
        } else {
            const built = decChainCfg(key) orelse return null;
            slot.* = .{ .key = key, .cfg = built };
            cfg = built;
            cached = true;
            break;
        }
    }
    if (!cached) cfg = decChainCfg(key) orelse return null;
    defer if (!cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    const eps_arr = [_]f32{m.eps};
    const eshape = [_]c_int{1};
    const eps_g = uploadF32(&eps_arr, &eshape);
    defer _ = mlx.mlx_array_free(eps_g);
    // unused slots stand in with always-present arrays (never read in-kernel)
    const w_in = rms_w orelse rr.cos;
    if (post != 2) hada_in = rr.cos;
    const inputs_arr = [_]mlx.mlx_array{ x, w_in, rr.cos, rr.sin, hada_in, eps_g };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, m.s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    dec_chain_hits += 1;
    if (!dec_chain_logged) {
        dec_chain_logged = true;
        log.info("dsv4: fused decode-chain kernel engaged\n", .{});
    }
    return out;
}

/// GPU window emission for one compressor over a chunk (decode is C == 1):
/// every window the chunk closes gets the masked-softmax combine → RMSNorm →
/// rope(block start) → QAT sim, and the emitted slots append to the GPU
/// mirror — pure lazy graph, NO host round trip. Pre-chunk rows come from
/// the HOST pending rings, which hold bit-identical bytes to the GPU rows
/// that fed them (same matmul output, same elementwise ape add), so no GPU
/// ring state exists and snapshots/anchors/rollback stay host-owned. The
/// host pushes still run — fully deferred to end of token/chunk — with
/// their mirror append suppressed (`gpuEmitActive`) so `used` advances
/// exactly once per slot.
fn emitWindowsGpu(
    m: *const Dsv4Model,
    c: *const HostComp,
    cs: *const CompDecState,
    mirror: *GpuRows,
    comp_in: mlx.mlx_array, // [C, comp_in_w] f32
    col_off: usize, // kv cols [col_off, col_off+cd); sc cols follow
    base: usize,
    C: usize,
    ratio: usize,
    rotate: bool,
    fr: *const Freqs,
    alloc: std.mem.Allocator,
) !void {
    const r = ratio;
    const d = c.head_dim;
    const cd = c.coff * d;
    const overlap = c.coff == 2;
    const W = base / r;
    const n_win = (base + C) / r - W;
    if (n_win == 0) return;
    if (!gpu_emit_logged) {
        gpu_emit_logged = true;
        log.info("dsv4: GPU window emission engaged\n", .{});
    }
    const k = base % r; // current-window rows already in the host ring
    const pre_n = k + (if (overlap) r else @as(usize, 0));
    const used_c = (W + n_win) * r - base; // chunk rows the emission consumes
    std.debug.assert(used_c <= C);

    if (try emitWindowsKernel(m, c, cs, mirror, comp_in, col_off, base, r, rotate, fr, alloc, W, n_win, pre_n, used_c)) return;

    // ── extended row matrices [pre_n + used_c, cd]; row 0 sits at position
    // W*r - (overlap ? r : 0), so closed windows are CONTIGUOUS row ranges.
    const kv_chunk = try gpuSliceRC(comp_in, 0, used_c, col_off, col_off + cd, m.s);
    defer _ = mlx.mlx_array_free(kv_chunk);
    const sc_chunk = blk: {
        const raw = try gpuSliceRC(comp_in, 0, used_c, col_off + cd, col_off + 2 * cd, m.s);
        defer _ = mlx.mlx_array_free(raw);
        const idxs = try alloc.alloc(i32, used_c);
        defer alloc.free(idxs);
        for (idxs, 0..) |*v, t| v.* = @intCast((base + t) % r);
        const ishape = [_]c_int{@intCast(used_c)};
        const idx_g = mlx.mlx_array_new_data(idxs.ptr, &ishape, 1, .int32);
        defer _ = mlx.mlx_array_free(idx_g);
        var rows_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows_g);
        try mlx.check(mlx.mlx_take_axis(&rows_g, c.ape_g, idx_g, 0, m.s));
        break :blk try gpuOp2(mlx.mlx_add, raw, rows_g, m.s);
    };
    defer _ = mlx.mlx_array_free(sc_chunk);
    var ext_kv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ext_kv);
    var ext_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ext_sc);
    if (pre_n > 0) {
        const stage_kv = try alloc.alloc(f32, pre_n * cd);
        defer alloc.free(stage_kv);
        const stage_sc = try alloc.alloc(f32, pre_n * cd);
        defer alloc.free(stage_sc);
        var row: usize = 0;
        if (overlap) { // previous window: ring slots 0..r, in position order
            for (0..r) |i| {
                @memcpy(stage_kv[row * cd ..][0..cd], cs.kv_pend[i * cs.width ..][0..cd]);
                @memcpy(stage_sc[row * cd ..][0..cd], cs.sc_pend[i * cs.width ..][0..cd]);
                row += 1;
            }
        }
        const cur0: usize = if (overlap) r else 0;
        for (0..k) |i| { // current window: slots cur0..cur0+k
            @memcpy(stage_kv[row * cd ..][0..cd], cs.kv_pend[(cur0 + i) * cs.width ..][0..cd]);
            @memcpy(stage_sc[row * cd ..][0..cd], cs.sc_pend[(cur0 + i) * cs.width ..][0..cd]);
            row += 1;
        }
        const pshape = [_]c_int{ @intCast(pre_n), @intCast(cd) };
        const kv_pre = uploadF32(stage_kv, &pshape);
        defer _ = mlx.mlx_array_free(kv_pre);
        const sc_pre = uploadF32(stage_sc, &pshape);
        defer _ = mlx.mlx_array_free(sc_pre);
        const ekv = try gpuConcat2(kv_pre, kv_chunk, 0, m.s);
        defer _ = mlx.mlx_array_free(ekv);
        try mlx.check(mlx.mlx_array_set(&ext_kv, ekv));
        const esc = try gpuConcat2(sc_pre, sc_chunk, 0, m.s);
        defer _ = mlx.mlx_array_free(esc);
        try mlx.check(mlx.mlx_array_set(&ext_sc, esc));
    } else {
        try mlx.check(mlx.mlx_array_set(&ext_kv, kv_chunk));
        try mlx.check(mlx.mlx_array_set(&ext_sc, sc_chunk));
    }

    // ── per-window effective rows [n_win, rows_eff, d]: with overlap the
    // previous window's rows contribute their FIRST half and the current
    // window's their SECOND (the compressorPush col rule).
    const cur_off: usize = if (overlap) r else 0;
    var eff_kv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eff_kv);
    var eff_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eff_sc);
    if (overlap) {
        const prev_kv = try winStack(ext_kv, 0, n_win, r, cd, m.s);
        defer _ = mlx.mlx_array_free(prev_kv);
        const cur_kv = try winStack(ext_kv, cur_off, n_win, r, cd, m.s);
        defer _ = mlx.mlx_array_free(cur_kv);
        const pk = try gpuSliceLast3(prev_kv, n_win, r, 0, d, m.s);
        defer _ = mlx.mlx_array_free(pk);
        const ck = try gpuSliceLast3(cur_kv, n_win, r, d, 2 * d, m.s);
        defer _ = mlx.mlx_array_free(ck);
        const jk = try gpuConcat2(pk, ck, 1, m.s);
        defer _ = mlx.mlx_array_free(jk);
        try mlx.check(mlx.mlx_array_set(&eff_kv, jk));
        const prev_sc = try winStack(ext_sc, 0, n_win, r, cd, m.s);
        defer _ = mlx.mlx_array_free(prev_sc);
        const cur_sc = try winStack(ext_sc, cur_off, n_win, r, cd, m.s);
        defer _ = mlx.mlx_array_free(cur_sc);
        const ps = try gpuSliceLast3(prev_sc, n_win, r, 0, d, m.s);
        defer _ = mlx.mlx_array_free(ps);
        const csl = try gpuSliceLast3(cur_sc, n_win, r, d, 2 * d, m.s);
        defer _ = mlx.mlx_array_free(csl);
        const js = try gpuConcat2(ps, csl, 1, m.s);
        defer _ = mlx.mlx_array_free(js);
        try mlx.check(mlx.mlx_array_set(&eff_sc, js));
    } else {
        const ek = try winStack(ext_kv, 0, n_win, r, cd, m.s);
        defer _ = mlx.mlx_array_free(ek);
        try mlx.check(mlx.mlx_array_set(&eff_kv, ek));
        const es = try winStack(ext_sc, 0, n_win, r, cd, m.s);
        defer _ = mlx.mlx_array_free(es);
        try mlx.check(mlx.mlx_array_set(&eff_sc, es));
    }

    // ── masked-softmax combine over the row axis (precise softmax = the
    // host's max-sub exp/sum in f32; the sims downstream snap the drift)
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, eff_sc, 1, true, m.s));
    const prod = try gpuOp2(mlx.mlx_multiply, probs, eff_kv, m.s);
    defer _ = mlx.mlx_array_free(prod);
    var num = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(num);
    try mlx.check(mlx.mlx_sum_axis(&num, prod, 1, true, m.s));
    const cshape = [_]c_int{ @intCast(n_win), @intCast(d) };
    const combined = try gpuReshape(num, &cshape, m.s);
    defer _ = mlx.mlx_array_free(combined);

    // ── emit tail: norm → rope at the block starts → QAT sim → append
    const normd = try gpuRms(combined, c.norm_g, m.eps, m.s);
    defer _ = mlx.mlx_array_free(normd);
    const half = m.rd / 2;
    const cos_h = try alloc.alloc(f32, n_win * half);
    defer alloc.free(cos_h);
    const sin_h = try alloc.alloc(f32, n_win * half);
    defer alloc.free(sin_h);
    for (0..n_win) |j| {
        const bs = (W + j) * r;
        @memcpy(cos_h[j * half ..][0..half], fr.cos[bs * half ..][0..half]);
        @memcpy(sin_h[j * half ..][0..half], fr.sin[bs * half ..][0..half]);
    }
    const rshape = [_]c_int{ @intCast(n_win), @intCast(half) };
    const cos_g = uploadF32(cos_h, &rshape);
    defer _ = mlx.mlx_array_free(cos_g);
    const sin_g = uploadF32(sin_h, &rshape);
    defer _ = mlx.mlx_array_free(sin_g);
    const roped = try gpuRopeTailRows(normd, m.rd, cos_g, sin_g, false, m.s);
    defer _ = mlx.mlx_array_free(roped);
    var final = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(final);
    if (rotate) {
        const had = try gpuOp2(mlx.mlx_matmul, roped, m.hada_g.?, m.s);
        defer _ = mlx.mlx_array_free(had);
        const simd = try gpuFp4Sim(had, m.s);
        defer _ = mlx.mlx_array_free(simd);
        try mlx.check(mlx.mlx_array_set(&final, simd));
    } else {
        const head = try gpuSliceCols(roped, n_win, 0, d - m.rd, m.s);
        defer _ = mlx.mlx_array_free(head);
        const hs = try gpuFp8Sim(head, m.s);
        defer _ = mlx.mlx_array_free(hs);
        const tail = try gpuSliceCols(roped, n_win, d - m.rd, d, m.s);
        defer _ = mlx.mlx_array_free(tail);
        const fin = try gpuConcat2(hs, tail, 1, m.s);
        defer _ = mlx.mlx_array_free(fin);
        try mlx.check(mlx.mlx_array_set(&final, fin));
    }
    try mirror.appendGpu(final, n_win, m.s);
}

fn attentionBatch(m: *Dsv4Model, alloc: std.mem.Allocator, st: *Dsv4DecodeState, li: usize, x_g: mlx.mlx_array, c_tokens: usize, base: usize, rr: *const RopeRows, fr: *const Freqs, deferred: *std.ArrayList(DeferredCompRow)) !mlx.mlx_array {
    const ly = &m.dw.layers[li];
    const h = &m.hl[li];
    const ls = &st.layers[li];
    const ratio: usize = ly.compress_ratio;
    const hd = m.head_dim;
    const nh = m.n_heads;
    const rd = m.rd;
    const C = c_tokens;
    const cc: c_int = @intCast(C);

    // q chain [C, nh, hd]
    const qr_n = blk: {
        const qr = try gpuQmmB(&ly.wq_a, x_g, m.s);
        defer _ = mlx.mlx_array_free(qr);
        break :blk try gpuRms(qr, h.q_norm_g, m.eps, m.s);
    };
    defer _ = mlx.mlx_array_free(qr_n);
    const q3 = blk: {
        const q_flat = try gpuQmmB(&ly.wq_b, qr_n, m.s);
        defer _ = mlx.mlx_array_free(q_flat);
        const qshape = [_]c_int{ cc, @intCast(nh), @intCast(hd) };
        const q_r = try gpuReshape(q_flat, &qshape, m.s);
        defer _ = mlx.mlx_array_free(q_r);
        const q_rms = try gpuRms(q_r, m.ones_hd_g, m.eps, m.s);
        defer _ = mlx.mlx_array_free(q_rms);
        break :blk try gpuRopeTailRows(q_rms, rd, rr.cos, rr.sin, false, m.s);
    };
    defer _ = mlx.mlx_array_free(q3);

    // kv chain [C, hd] → append C rows
    {
        const kv0 = try gpuQmmB(&ly.wkv, x_g, m.s);
        defer _ = mlx.mlx_array_free(kv0);
        const kv_n = try gpuRms(kv0, h.kv_norm_g, m.eps, m.s);
        defer _ = mlx.mlx_array_free(kv_n);
        const kv_rot = try gpuRopeTailRows(kv_n, rd, rr.cos, rr.sin, false, m.s);
        defer _ = mlx.mlx_array_free(kv_rot);
        const head0 = try gpuSliceCols(kv_rot, C, 0, hd - rd, m.s);
        defer _ = mlx.mlx_array_free(head0);
        const head_sim = try gpuFp8Sim(head0, m.s);
        defer _ = mlx.mlx_array_free(head_sim);
        const tail = try gpuSliceCols(kv_rot, C, hd - rd, hd, m.s);
        defer _ = mlx.mlx_array_free(tail);
        const kv_fin = try gpuConcat2(head_sim, tail, 1, m.s);
        defer _ = mlx.mlx_array_free(kv_fin);
        try ls.kv_gpu.appendGpu(kv_fin, C, m.s);
    }

    // compressor rings: ONE [C, W] sync feeds every per-token push — but that
    // read is a GPU BARRIER, and a chunk that closes NO window emits nothing
    // this chunk's own attention can see, so its read joins one batched eval
    // after the layer loop (`processDeferredCompChunk`) exactly as the decode
    // path defers non-boundary positions.
    if (ratio != 0) {
        const comp_in = try compInProj(m, h, li, x_g, C);
        const closes_attn = chunkCrossesBoundary(base, C, ratio);
        const closes_idx = ls.idx_comp != null and chunkCrossesBoundary(base, C, 4);
        const closes = closes_attn or closes_idx;
        if (gpuEmitActive(m)) {
            // in-chunk emitted slots stay same-chunk-visible through the
            // mirror (the visibility masks below cover per-token scope);
            // the host pushes defer to ONE end-of-chunk batched read.
            const cdd = h.comp.?.coff * h.comp.?.head_dim;
            if (closes_attn)
                try emitWindowsGpu(m, &h.comp.?, &ls.comp.?, &ls.comp_gpu, comp_in, 0, base, C, ratio, false, fr, alloc);
            if (closes_idx)
                try emitWindowsGpu(m, &h.idx_comp.?, &ls.idx_comp.?, &ls.idx_gpu, comp_in, 2 * cdd, base, C, 4, true, fr, alloc);
            errdefer _ = mlx.mlx_array_free(comp_in);
            try deferred.append(alloc, .{ .li = li, .arr = comp_in });
        } else if (closes or !compDeferEnabled()) {
            defer _ = mlx.mlx_array_free(comp_in);
            var sclk: DsparkClock = if (m.ds_prof != null) DsparkClock.init() else undefined;
            var cclk: DsparkClock = if (dsv4TraceEnabled()) DsparkClock.init() else undefined;
            const rows = try toHostF32(alloc, comp_in, C * h.comp_in_w, m.s);
            if (m.ds_prof != null) m.ds_prof_comp_sync_ns += sclk.lap();
            defer alloc.free(rows);
            try pushCompChunk(m, st, li, rows, base, C, fr, alloc);
            if (dsv4TraceEnabled()) trace_comp_ns += cclk.lap();
        } else {
            errdefer _ = mlx.mlx_array_free(comp_in);
            try deferred.append(alloc, .{ .li = li, .arr = comp_in });
        }
    }

    // window band: w_idx[t, i] = base + t - wk + 1 + i, invalid < 0 masked
    const seq_total = base + C;
    const wk = @min(m.window, seq_total);
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    const zero_f = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_f);
    const neginf_f = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neginf_f);
    var t_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t_col);
    {
        const start_v: f64 = @floatFromInt(@as(i64, @intCast(base)) - @as(i64, @intCast(wk)) + 1);
        var t_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t_flat);
        try mlx.check(mlx.mlx_arange(&t_flat, start_v, start_v + @as(f64, @floatFromInt(C)), 1.0, .int32, m.s));
        const tshape = [_]c_int{ cc, 1 };
        try mlx.check(mlx.mlx_reshape(&t_col, t_flat, &tshape, 2, m.s));
    }
    var i_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(i_row);
    {
        var i_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(i_flat);
        try mlx.check(mlx.mlx_arange(&i_flat, 0.0, @floatFromInt(wk), 1.0, .int32, m.s));
        const ishape = [_]c_int{ 1, @intCast(wk) };
        try mlx.check(mlx.mlx_reshape(&i_row, i_flat, &ishape, 2, m.s));
    }
    const w_idx = try gpuOp2(mlx.mlx_add, t_col, i_row, m.s);
    defer _ = mlx.mlx_array_free(w_idx);
    const w_ok = try gpuOp2(mlx.mlx_greater_equal, w_idx, zero_i, m.s);
    defer _ = mlx.mlx_array_free(w_ok);
    var w_mask2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_mask2);
    try mlx.check(mlx.mlx_where(&w_mask2, w_ok, zero_f, neginf_f, m.s));
    const wm3 = [_]c_int{ cc, 1, @intCast(wk) };
    const w_mask = try gpuReshape(w_mask2, &wm3, m.s);
    defer _ = mlx.mlx_array_free(w_mask);
    const w_clamped = try gpuOp2(mlx.mlx_maximum, w_idx, zero_i, m.s);
    defer _ = mlx.mlx_array_free(w_clamped);
    const kv_all = try ls.kv_gpu.sliceRows(0, seq_total, m.s);
    defer _ = mlx.mlx_array_free(kv_all);
    var kw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kw);
    try mlx.check(mlx.mlx_take_axis(&kw, kv_all, w_clamped, 0, m.s)); // [C, wk, hd]

    // compressed arm: per-token gathered top-k (ratio 4) or shared all-slot
    // K with the visibility mask (other ratios)
    var kc: ?mlx.mlx_array = null;
    defer if (kc) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    var comp_shared: ?mlx.mlx_array = null;
    defer if (comp_shared) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    var c_mask: ?mlx.mlx_array = null;
    defer if (c_mask) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    var comp_cols: usize = 0;
    if (ratio == 4) {
        const S = @min(ls.comp_gpu.used, ls.idx_gpu.used);
        const k = @min(m.idx_topk, S);
        if (k > 0) {
            const ih = m.idx_heads;
            const ihd = m.idx_hd;
            const qi_sim = blk: {
                const qi = try gpuQmmB(&ly.idx.?.wq_b, qr_n, m.s);
                defer _ = mlx.mlx_array_free(qi);
                const qsh = [_]c_int{ cc, @intCast(ih), @intCast(ihd) };
                const qi_r = try gpuReshape(qi, &qsh, m.s);
                defer _ = mlx.mlx_array_free(qi_r);
                const qi_rot = try gpuRopeTailRows(qi_r, rd, rr.cos, rr.sin, false, m.s);
                defer _ = mlx.mlx_array_free(qi_rot);
                const qi_had = try gpuOp2(mlx.mlx_matmul, qi_rot, m.hada_g.?, m.s);
                defer _ = mlx.mlx_array_free(qi_had);
                break :blk try gpuFp4Sim(qi_had, m.s);
            };
            defer _ = mlx.mlx_array_free(qi_sim);
            const islots = try ls.idx_gpu.sliceRows(0, S, m.s);
            defer _ = mlx.mlx_array_free(islots);
            const it_ = try gpuOp1(mlx.mlx_transpose, islots, m.s);
            defer _ = mlx.mlx_array_free(it_);
            const sc = try gpuOp2(mlx.mlx_matmul, qi_sim, it_, m.s);
            defer _ = mlx.mlx_array_free(sc);
            const relu = try gpuOp2(mlx.mlx_maximum, sc, zero_f, m.s);
            defer _ = mlx.mlx_array_free(relu);
            const wts_r = try gpuOp2(mlx.mlx_matmul, x_g, h.idx_wp_t.?, m.s);
            defer _ = mlx.mlx_array_free(wts_r);
            const wr3 = [_]c_int{ cc, @intCast(m.idx_heads), 1 };
            const wts_c = try gpuReshape(wts_r, &wr3, m.s);
            defer _ = mlx.mlx_array_free(wts_c);
            const wscale = mlx.mlx_array_new_float(@floatCast(1.0 / (@sqrt(@as(f64, @floatFromInt(ihd))) * @sqrt(@as(f64, @floatFromInt(ih))))));
            defer _ = mlx.mlx_array_free(wscale);
            const wts_s = try gpuOp2(mlx.mlx_multiply, wts_c, wscale, m.s);
            defer _ = mlx.mlx_array_free(wts_s);
            const weighted = try gpuOp2(mlx.mlx_multiply, relu, wts_s, m.s);
            defer _ = mlx.mlx_array_free(weighted);
            var scores = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scores);
            try mlx.check(mlx.mlx_sum_axis(&scores, weighted, 1, false, m.s)); // [C, S]
            // visibility: token t sees the first (base+t+1)/ratio slots
            const vis_h = try alloc.alloc(f32, C);
            defer alloc.free(vis_h);
            for (0..C) |t| vis_h[t] = @floatFromInt((base + t + 1) / ratio);
            const vshape = [_]c_int{ cc, 1 };
            const vis_col = uploadF32(vis_h, &vshape);
            defer _ = mlx.mlx_array_free(vis_col);
            var slot_row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(slot_row);
            try mlx.check(mlx.mlx_arange(&slot_row, 0.0, @floatFromInt(S), 1.0, .float32, m.s));
            const srshape = [_]c_int{ 1, @intCast(S) };
            const slot_r2 = try gpuReshape(slot_row, &srshape, m.s);
            defer _ = mlx.mlx_array_free(slot_r2);
            const vis_ok = try gpuOp2(mlx.mlx_less, slot_r2, vis_col, m.s);
            defer _ = mlx.mlx_array_free(vis_ok);
            var vmask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vmask);
            try mlx.check(mlx.mlx_where(&vmask, vis_ok, zero_f, neginf_f, m.s));
            const masked = try gpuOp2(mlx.mlx_add, scores, vmask, m.s);
            defer _ = mlx.mlx_array_free(masked);
            var sel = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sel);
            if (S > k) {
                var part = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(part);
                try mlx.check(mlx.mlx_argpartition_axis(&part, masked, @intCast(S - k), 1, m.s));
                const sel_v = try gpuSliceCols(part, C, S - k, S, m.s);
                defer _ = mlx.mlx_array_free(sel_v);
                try mlx.check(mlx.mlx_astype(&sel, sel_v, .uint32, m.s));
            } else {
                var ar = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ar);
                try mlx.check(mlx.mlx_arange(&ar, 0.0, @floatFromInt(S), 1.0, .uint32, m.s));
                const arshape = [_]c_int{ 1, @intCast(S) };
                const ar2 = try gpuReshape(ar, &arshape, m.s);
                defer _ = mlx.mlx_array_free(ar2);
                const bshape = [_]c_int{ cc, @intCast(S) };
                try mlx.check(mlx.mlx_broadcast_to(&sel, ar2, &bshape, 2, m.s));
            }
            const comp_all = try ls.comp_gpu.sliceRows(0, S, m.s);
            defer _ = mlx.mlx_array_free(comp_all);
            var kc_v = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(&kc_v, comp_all, sel, 0, m.s)); // [C, k, hd]
            kc = kc_v;
            var sel_sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sel_sc);
            try mlx.check(mlx.mlx_take_along_axis(&sel_sc, masked, sel, 1, m.s)); // [C, k]
            const thresh = mlx.mlx_array_new_float(-1e30);
            defer _ = mlx.mlx_array_free(thresh);
            const sel_ok = try gpuOp2(mlx.mlx_greater, sel_sc, thresh, m.s);
            defer _ = mlx.mlx_array_free(sel_ok);
            var cm2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cm2);
            try mlx.check(mlx.mlx_where(&cm2, sel_ok, zero_f, neginf_f, m.s));
            const cm3 = [_]c_int{ cc, 1, @intCast(k) };
            c_mask = try gpuReshape(cm2, &cm3, m.s);
            comp_cols = k;
        }
    } else if (ratio != 0 and ls.comp_gpu.used > 0) {
        const S = ls.comp_gpu.used;
        comp_shared = try ls.comp_gpu.sliceRows(0, S, m.s);
        const vis_h = try alloc.alloc(f32, C);
        defer alloc.free(vis_h);
        for (0..C) |t| vis_h[t] = @floatFromInt((base + t + 1) / ratio);
        const vshape = [_]c_int{ cc, 1 };
        const vis_col = uploadF32(vis_h, &vshape);
        defer _ = mlx.mlx_array_free(vis_col);
        var slot_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slot_row);
        try mlx.check(mlx.mlx_arange(&slot_row, 0.0, @floatFromInt(S), 1.0, .float32, m.s));
        const srshape = [_]c_int{ 1, @intCast(S) };
        const slot_r2 = try gpuReshape(slot_row, &srshape, m.s);
        defer _ = mlx.mlx_array_free(slot_r2);
        const vis_ok = try gpuOp2(mlx.mlx_less, slot_r2, vis_col, m.s);
        defer _ = mlx.mlx_array_free(vis_ok);
        var vm2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vm2);
        try mlx.check(mlx.mlx_where(&vm2, vis_ok, zero_f, neginf_f, m.s));
        const vm3 = [_]c_int{ cc, 1, @intCast(S) };
        c_mask = try gpuReshape(vm2, &vm3, m.s);
        comp_cols = S;
    }

    // sink-softmax assembly over [window ++ compressed ++ sink]
    const scale_arr = mlx.mlx_array_new_float(@floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd)))));
    defer _ = mlx.mlx_array_free(scale_arr);
    const kwt_axes = [_]c_int{ 0, 2, 1 };
    var kwt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kwt);
    try mlx.check(mlx.mlx_transpose_axes(&kwt, kw, &kwt_axes, 3, m.s));
    const sw_raw = try gpuOp2(mlx.mlx_matmul, q3, kwt, m.s);
    defer _ = mlx.mlx_array_free(sw_raw);
    const sw_sc = try gpuOp2(mlx.mlx_multiply, sw_raw, scale_arr, m.s);
    defer _ = mlx.mlx_array_free(sw_sc);
    const sw = try gpuOp2(mlx.mlx_add, sw_sc, w_mask, m.s);
    defer _ = mlx.mlx_array_free(sw);
    var s_c: ?mlx.mlx_array = null;
    defer if (s_c) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    if (comp_cols > 0) {
        var raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw);
        if (kc) |kc_arr| {
            var kct = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kct);
            try mlx.check(mlx.mlx_transpose_axes(&kct, kc_arr, &kwt_axes, 3, m.s));
            try mlx.check(mlx.mlx_matmul(&raw, q3, kct, m.s));
        } else {
            var cst = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cst);
            try mlx.check(mlx.mlx_transpose(&cst, comp_shared.?, m.s));
            try mlx.check(mlx.mlx_matmul(&raw, q3, cst, m.s));
        }
        const sc_sc = try gpuOp2(mlx.mlx_multiply, raw, scale_arr, m.s);
        defer _ = mlx.mlx_array_free(sc_sc);
        s_c = try gpuOp2(mlx.mlx_add, sc_sc, c_mask.?, m.s);
    }
    var sink3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sink3);
    {
        const s1 = [_]c_int{ 1, @intCast(nh), 1 };
        const sink_r = try gpuReshape(h.sink_gpu, &s1, m.s);
        defer _ = mlx.mlx_array_free(sink_r);
        const sb = [_]c_int{ cc, @intCast(nh), 1 };
        try mlx.check(mlx.mlx_broadcast_to(&sink3, sink_r, &sb, 3, m.s));
    }
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    {
        const parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        _ = mlx.mlx_vector_array_append_value(parts, sw);
        if (s_c) |arr| _ = mlx.mlx_vector_array_append_value(parts, arr);
        _ = mlx.mlx_vector_array_append_value(parts, sink3);
        try mlx.check(mlx.mlx_concatenate_axis(&all, parts, 2, m.s));
    }
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, all, -1, true, m.s));
    const p_w = try gpuSliceLast3(probs, C, nh, 0, wk, m.s);
    defer _ = mlx.mlx_array_free(p_w);
    var o = try gpuOp2(mlx.mlx_matmul, p_w, kw, m.s); // [C, nh, hd]
    defer _ = mlx.mlx_array_free(o);
    if (comp_cols > 0) {
        const p_c = try gpuSliceLast3(probs, C, nh, wk, wk + comp_cols, m.s);
        defer _ = mlx.mlx_array_free(p_c);
        var oc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(oc);
        if (kc) |kc_arr| {
            try mlx.check(mlx.mlx_matmul(&oc, p_c, kc_arr, m.s));
        } else {
            try mlx.check(mlx.mlx_matmul(&oc, p_c, comp_shared.?, m.s));
        }
        const o_sum = try gpuOp2(mlx.mlx_add, o, oc, m.s);
        _ = mlx.mlx_array_free(o);
        o = o_sum;
    }
    const o_inv = try gpuRopeTailRows(o, rd, rr.cos, rr.sin, true, m.s);
    defer _ = mlx.mlx_array_free(o_inv);
    // grouped low-rank O: [og, C, gin] bf16 @ wo_a_deq [og, gin, ol]
    const og = m.o_groups;
    const ol = m.o_lora;
    const gin = nh * hd / og;
    const o2shape = [_]c_int{ cc, @intCast(og), @intCast(gin) };
    const o2 = try gpuReshape(o_inv, &o2shape, m.s);
    defer _ = mlx.mlx_array_free(o2);
    const gaxes = [_]c_int{ 1, 0, 2 };
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    try mlx.check(mlx.mlx_transpose_axes(&ot, o2, &gaxes, 3, m.s));
    var ob = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ob);
    try mlx.check(mlx.mlx_astype(&ob, ot, .bfloat16, m.s));
    const ored = try woAMatmul(m, li, ob); // [og, C, ol]
    defer _ = mlx.mlx_array_free(ored);
    var ort = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ort);
    try mlx.check(mlx.mlx_transpose_axes(&ort, ored, &gaxes, 3, m.s)); // [C, og, ol]
    const orshape = [_]c_int{ cc, @intCast(og * ol) };
    const orr = try gpuReshape(ort, &orshape, m.s);
    defer _ = mlx.mlx_array_free(orr);
    return try gpuQmmB(&ly.wo_b, orr, m.s); // [C, d]
}

/// Sub-chunk cap: bounds the attention gather transient
/// (C·(window+idx_topk)·head_dim f32 ≈ 670 MB at 512 on the real geometry).
const PREFILL_SUB: usize = 512;

/// Prefill sub-chunk size, env-tunable for A/B (`MLX_SERVE_DSV4_PREFILL_SUB`;
/// default `PREFILL_SUB`). Bigger chunks raise the per-expert M of the
/// sorted MoE gather (small-M qmm efficiency) at the cost of larger
/// attention-gather transients. Cached once per process.
var prefill_sub_state: ?usize = null;
/// Pub: server.zig's prefill memory guard bills the SUB-chunk this arch
/// actually forwards (dsv4PrefillMemoryNeeded), and it must read the same
/// env-overridable value the engine runs — billing a stale constant while
/// MLX_SERVE_DSV4_PREFILL_SUB raises the real width under-bills into an
/// uncatchable Metal OOM.
pub fn prefillSub() usize {
    if (prefill_sub_state) |v| return v;
    var v: usize = PREFILL_SUB;
    if (std.c.getenv("MLX_SERVE_DSV4_PREFILL_SUB")) |e| {
        const parsed = std.fmt.parseInt(usize, std.mem.span(e), 10) catch PREFILL_SUB;
        if (parsed > 0) v = parsed;
    }
    prefill_sub_state = v;
    return v;
}

/// Prefill transients (the [C, tk, hd] gathers) never repeat a shape across
/// prompt lengths, so a prefill sub-chunk returns them to the OS (the
/// allocator-cache growth class). A DSpark round is the OPPOSITE case: its
/// widths are ≤ block+1 and recur every round, so clearing after one makes
/// the next round re-allocate every transient from the OS — measured as a
/// ~110 ms tax on the C=6 verify that FOLLOWS a rollback. Boundary is the
/// house's "a multi-token forward is not a prefill" line (seq ≥ 32).
fn extendChunkShouldClearCache(c_tokens: usize) bool {
    return c_tokens >= 32;
}

/// Extend the decode state with a chunk of tokens (batched prefill AND
/// chunked-prefill continuation; ids.len ≥ 1). Returns the LAST position's
/// logits [vocab] (gpa-owned).
pub fn extendState(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, ids: []const u32) ![]f32 {
    var logits: ?[]f32 = null;
    errdefer if (logits) |l| gpa.free(l);
    var done: usize = 0;
    while (done < ids.len) {
        const c_tokens = @min(prefillSub(), ids.len - done);
        if (logits) |l| gpa.free(l);
        logits = try extendChunk(m, gpa, st, ids[done .. done + c_tokens], .last_host);
        done += c_tokens;
        if (extendChunkShouldClearCache(c_tokens)) _ = mlx.mlx_clear_cache();
    }
    return logits.?;
}

/// Verify primitive: extend with a SMALL block and return EVERY position's
/// logits ([ids.len * vocab] f32, row-major). One chunk only by design — the
/// DSpark verify block is ≤ block_size+1 tokens.
pub fn extendStateAllLogits(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, ids: []const u32) ![]f32 {
    std.debug.assert(ids.len <= prefillSub());
    return extendChunk(m, gpa, st, ids, .all_host);
}

/// What extendChunk hands back. `.last_host`/`.all_host` sync and return
/// host f32 (last row / every row); `.all_gpu` returns the LAZY `[C, vocab]`
/// logits array un-synced — the stochastic verify feeds it straight into the
/// filtered-probs + accept graph so the round pays ONE bounded sync at its
/// end. `.all_host` is `.all_gpu` + toHostF32 by construction
/// (headLogitsBatchGpu wraps headLogitsBatchG): same graph, same sync point.
const ExtendMode = enum { last_host, all_host, all_gpu };

fn ExtendRet(comptime mode: ExtendMode) type {
    return if (mode == .all_gpu) mlx.mlx_array else []f32;
}

fn extendChunk(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, ids: []const u32, comptime mode: ExtendMode) !ExtendRet(mode) {
    var chunk_clk: DsparkClock = if (m.ds_prof != null) DsparkClock.init() else undefined;
    if (m.ds_prof != null) m.ds_prof_comp_sync_ns = 0;
    const tracing = dsv4TraceEnabled();
    var tclk: DsparkClock = if (tracing) DsparkClock.init() else undefined;
    if (tracing) trace_comp_ns = 0;
    if (st.pending.items.len > 0) try drainPending(m, st); // lazy→chunk handoff
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const base = st.n;
    const C = ids.len;
    const cc: c_int = @intCast(C);
    st.n += C;
    const d = m.dim;
    const hcm = m.hc;
    const fr_plain = try freqsFor(m, .plain, base + C + 2, m.arena.allocator());
    const fr_yarn = try freqsFor(m, .yarn, base + C + 2, m.arena.allocator());
    const half = m.rd / 2;
    const rowshape = [_]c_int{ cc, @intCast(half) };
    const rr_plain = RopeRows{
        .cos = uploadF32(fr_plain.cos[base * half ..][0 .. C * half], &rowshape),
        .sin = uploadF32(fr_plain.sin[base * half ..][0 .. C * half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr_plain.cos);
    defer _ = mlx.mlx_array_free(rr_plain.sin);
    const rr_yarn = RopeRows{
        .cos = uploadF32(fr_yarn.cos[base * half ..][0 .. C * half], &rowshape),
        .sin = uploadF32(fr_yarn.sin[base * half ..][0 .. C * half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr_yarn.cos);
    defer _ = mlx.mlx_array_free(rr_yarn.sin);
    // stream [C, hc, d] = embed rows broadcast over the hc copies
    var stream_g = blk: {
        const eh = try a.alloc(f32, C * d);
        for (0..C) |t| @memcpy(eh[t * d ..][0..d], m.embed_f32[@as(usize, ids[t]) * d ..][0..d]);
        const eshape = [_]c_int{ cc, 1, @intCast(d) };
        const e_rows = uploadF32(eh, &eshape);
        defer _ = mlx.mlx_array_free(e_rows);
        const bshape = [_]c_int{ cc, @intCast(hcm), @intCast(d) };
        var b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&b, e_rows, &bshape, 3, m.s));
        break :blk b;
    };
    defer _ = mlx.mlx_array_free(stream_g);
    var mh_parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (mh_parts.items) |p| _ = mlx.mlx_array_free(p);
        mh_parts.deinit(a);
    }
    var deferred = std.ArrayList(DeferredCompRow).empty;
    defer {
        for (deferred.items) |r| _ = mlx.mlx_array_free(r.arr);
        deferred.deinit(a);
    }
    for (0..m.n_layers) |li| {
        const h = &m.hl[li];
        const ly = &m.dw.layers[li];
        const ratio: usize = ly.compress_ratio;
        const fr = if (ratio != 0) fr_yarn else fr_plain;
        const rr = if (ratio != 0) &rr_yarn else &rr_plain;
        {
            const pre = try hcPreBatch(m, a, stream_g, C, h.hc_attn_fn_t, h.hc_attn_scale, h.hc_attn_base, ly.hc_attn_scale, ly.hc_attn_base);
            defer freeHcPre(&pre);
            const x = try gpuRms(pre.y, h.attn_norm_g, m.eps, m.s);
            defer _ = mlx.mlx_array_free(x);
            const attn_out = try attentionBatch(m, a, st, li, x, C, base, rr, fr, &deferred);
            defer _ = mlx.mlx_array_free(attn_out);
            const ns = try hcPostBatch(m, stream_g, attn_out, C, &pre);
            _ = mlx.mlx_array_free(stream_g);
            stream_g = ns;
        }
        {
            const pre = try hcPreBatch(m, a, stream_g, C, h.hc_ffn_fn_t, h.hc_ffn_scale, h.hc_ffn_base, ly.hc_ffn_scale, ly.hc_ffn_base);
            defer freeHcPre(&pre);
            const x = try gpuRms(pre.y, h.ffn_norm_g, m.eps, m.s);
            defer _ = mlx.mlx_array_free(x);
            const ffn_out = try moeGpu(m, a, li, x, ids);
            defer _ = mlx.mlx_array_free(ffn_out);
            const ns = try hcPostBatch(m, stream_g, ffn_out, C, &pre);
            _ = mlx.mlx_array_free(stream_g);
            stream_g = ns;
        }
        // DSpark conditioning: hc-averaged stream [C, d] at target layers.
        if (st.dspark != null and dsIsTarget(m, li)) {
            var mean = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(mean);
            try mlx.check(mlx.mlx_mean_axis(&mean, stream_g, 1, false, m.s));
            try mh_parts.append(a, mean);
        }
    }
    const layers_ns: u64 = if (tracing) tclk.lap() else 0;
    try processDeferredCompChunk(m, st, a, deferred.items, base, C, fr_yarn);
    const defer_ns: u64 = if (tracing) tclk.lap() else 0;
    if (st.dspark != null and mh_parts.items.len > 0) {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (mh_parts.items) |p| _ = mlx.mlx_vector_array_append_value(vec, p);
        var mh = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mh);
        try mlx.check(mlx.mlx_concatenate_axis(&mh, vec, 1, m.s));
        try appendDsparkMainKv(m, st, mh, C, &rr_plain);
    }
    if (m.ds_prof != null) m.ds_prof_layers_ns = chunk_clk.lap();
    const out = if (comptime mode == .all_gpu)
        // Lazy: the caller owns the sync, so under profiling the head lap
        // below measures BUILD only — the eval lands in the caller's
        // verify lap (DsparkPending.lapVerify).
        try headLogitsBatchG(m, a, stream_g, C)
    else if (comptime mode == .all_host)
        try headLogitsBatchGpu(m, gpa, a, stream_g, C)
    else blk: {
        // head on the last position
        const lstart = [_]c_int{ @intCast(C - 1), 0, 0 };
        const lstop = [_]c_int{ cc, @intCast(hcm), @intCast(d) };
        const lstr = [_]c_int{ 1, 1, 1 };
        var last3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(last3);
        try mlx.check(mlx.mlx_slice(&last3, stream_g, &lstart, 3, &lstop, 3, &lstr, 3, m.s));
        const hshape = [_]c_int{ @intCast(hcm), @intCast(d) };
        const last2 = try gpuReshape(last3, &hshape, m.s);
        defer _ = mlx.mlx_array_free(last2);
        break :blk try headLogitsGpu(m, gpa, a, last2);
    };
    if (m.ds_prof != null) m.ds_prof_head_ns = chunk_clk.lap();
    if (tracing) {
        const head_ns = tclk.lap();
        log.info("[dsv4-trace] chunk base={d} C={d} layers={d}ms defer={d}ms head={d}ms comp={d}ms\n", .{
            base, C, layers_ns / 1_000_000, defer_ns / 1_000_000, head_ns / 1_000_000, trace_comp_ns / 1_000_000,
        });
    }
    return out;
}

/// Batched trunk head: hc collapse (sigmoid pre weights, one [C, hc+1] sync)
/// → final norm → shared head on EVERY position. Returns [C * vocab] f32.
/// The C==1 sibling is headLogitsGpu; the verify loop needs all positions.
fn headLogitsBatchGpu(m: *const Dsv4Model, gpa: std.mem.Allocator, alloc: std.mem.Allocator, stream_g: mlx.mlx_array, C: usize) ![]f32 {
    const logits_g = try headLogitsBatchG(m, alloc, stream_g, C);
    defer _ = mlx.mlx_array_free(logits_g);
    return try toHostF32(gpa, logits_g, C * m.vocab, m.s);
}

/// Lazy sibling: builds the SAME `[C, vocab]` graph and returns it un-synced
/// (caller owns the handle and the eventual eval). On a GPU stream nothing
/// here syncs; the CPU-stream arm still pays its host pre-weight loop.
fn headLogitsBatchG(m: *const Dsv4Model, alloc: std.mem.Allocator, stream_g: mlx.mlx_array, C: usize) !mlx.mlx_array {
    const hcm = m.hc;
    const d = m.dim;
    const cc: c_int = @intCast(C);
    const fshape = [_]c_int{ cc, @intCast(hcm * d) };
    const flat = try gpuReshape(stream_g, &fshape, m.s);
    defer _ = mlx.mlx_array_free(flat);
    const mixes_g = try gpuOp2(mlx.mlx_matmul, flat, m.hc_head_fn_t, m.s);
    defer _ = mlx.mlx_array_free(mixes_g);
    const sq = try gpuOp1(mlx.mlx_square, flat, m.s);
    defer _ = mlx.mlx_array_free(sq);
    var ssum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ssum);
    try mlx.check(mlx.mlx_sum_axis(&ssum, sq, 1, true, m.s));
    var pre_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pre_col);
    const pshape = [_]c_int{ cc, @intCast(hcm), 1 };
    if (mlx.streamIsGpu(m.s)) {
        const prew_g = try headPreWeightsGpu(m, mixes_g, ssum); // [C, hc]
        defer _ = mlx.mlx_array_free(prew_g);
        var pc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pc);
        try mlx.check(mlx.mlx_reshape(&pc, prew_g, &pshape, 3, m.s));
        try mlx.check(mlx.mlx_array_set(&pre_col, pc));
    } else {
        const joint = try gpuConcat2(mixes_g, ssum, 1, m.s);
        defer _ = mlx.mlx_array_free(joint);
        const jh = try toHostF32(alloc, joint, C * (hcm + 1), m.s);
        defer alloc.free(jh);
        const prew = try alloc.alloc(f32, C * hcm);
        defer alloc.free(prew);
        for (0..C) |t| {
            const row = jh[t * (hcm + 1) ..][0 .. hcm + 1];
            const rsq: f32 = @floatCast(1.0 / @sqrt(@as(f64, row[hcm]) / @as(f64, @floatFromInt(hcm * d)) + m.eps));
            for (0..hcm) |c| prew[t * hcm + c] = sigmoidF32(row[c] * rsq * m.hc_head_scale[0] + m.hc_head_base[c]) + m.hc_eps;
        }
        const up = uploadF32(prew, &pshape);
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_array_set(&pre_col, up));
    }
    const weighted = try gpuOp2(mlx.mlx_multiply, pre_col, stream_g, m.s);
    defer _ = mlx.mlx_array_free(weighted);
    var hout = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hout);
    try mlx.check(mlx.mlx_sum_axis(&hout, weighted, 1, false, m.s)); // [C, d]
    const hn = try gpuRms(hout, m.final_norm_g, m.eps, m.s);
    defer _ = mlx.mlx_array_free(hn);
    return try gpuQmmB(&m.dw.head, hn, m.s);
}

// ── DSpark draft (forward_spec at start_pos > 0, greedy) ───────────────
//
// Block-parallel: the whole draft block runs the 3 stages in ONE pass. The
// stage rings already hold every position ≤ start_pos (the trunk appends
// main_kv rows at decode/prefill time), so the draft only READS state —
// rollback needs nothing from it. Attention is the reference's sparse set
// made dense: every draft position attends ALL of [last min(win, n) ring
// rows ++ the whole draft block] — NO causal mask inside the block.

/// One DSpark stage's attention over the draft block: x [B, dim] f32
/// (attn-normed), draft positions n..n+B-1, PLAIN rope (stages are ratio-0).
fn dsparkAttentionG(m: *Dsv4Model, st: *Dsv4DecodeState, sti: usize, x_g: mlx.mlx_array, B: usize, rr: *const RopeRows) !mlx.mlx_array {
    const li = m.n_layers + sti;
    const ly = &m.dw.layers[li];
    const h = &m.hl[li];
    const ring = &st.dspark.?.main_kv[sti];
    const hd = m.head_dim;
    const nh = m.n_heads;
    const rd = m.rd;
    const bc: c_int = @intCast(B);

    // q chain [B, nh, hd] — the attentionBatch recipe
    const qr_n = blk: {
        const qr = try gpuQmmB(&ly.wq_a, x_g, m.s);
        defer _ = mlx.mlx_array_free(qr);
        break :blk try gpuRms(qr, h.q_norm_g, m.eps, m.s);
    };
    defer _ = mlx.mlx_array_free(qr_n);
    const q3 = blk: {
        const q_flat = try gpuQmmB(&ly.wq_b, qr_n, m.s);
        defer _ = mlx.mlx_array_free(q_flat);
        const qshape = [_]c_int{ bc, @intCast(nh), @intCast(hd) };
        const q_r = try gpuReshape(q_flat, &qshape, m.s);
        defer _ = mlx.mlx_array_free(q_r);
        const q_rms = try gpuRms(q_r, m.ones_hd_g, m.eps, m.s);
        defer _ = mlx.mlx_array_free(q_rms);
        break :blk try gpuRopeTailRows(q_rms, rd, rr.cos, rr.sin, false, m.s);
    };
    defer _ = mlx.mlx_array_free(q3);

    // draft kv [B, hd] (rope + fp8 sim) — read-only, never appended
    const kv_fin = blk: {
        const kv0 = try gpuQmmB(&ly.wkv, x_g, m.s);
        defer _ = mlx.mlx_array_free(kv0);
        const kv_n = try gpuRms(kv0, h.kv_norm_g, m.eps, m.s);
        defer _ = mlx.mlx_array_free(kv_n);
        const kv_rot = try gpuRopeTailRows(kv_n, rd, rr.cos, rr.sin, false, m.s);
        defer _ = mlx.mlx_array_free(kv_rot);
        const head0 = try gpuSliceCols(kv_rot, B, 0, hd - rd, m.s);
        defer _ = mlx.mlx_array_free(head0);
        const head_sim = try gpuFp8Sim(head0, m.s);
        defer _ = mlx.mlx_array_free(head_sim);
        const tail = try gpuSliceCols(kv_rot, B, hd - rd, hd, m.s);
        defer _ = mlx.mlx_array_free(tail);
        break :blk try gpuConcat2(head_sim, tail, 1, m.s);
    };
    defer _ = mlx.mlx_array_free(kv_fin);

    // keys = [ring window ++ draft block]
    const n = st.n;
    const wk = @min(n, m.window);
    const ring_win = try ring.sliceRows(n - wk, n, m.s);
    defer _ = mlx.mlx_array_free(ring_win);
    var kmat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kmat);
    {
        const parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        _ = mlx.mlx_vector_array_append_value(parts, ring_win);
        _ = mlx.mlx_vector_array_append_value(parts, kv_fin);
        try mlx.check(mlx.mlx_concatenate_axis(&kmat, parts, 0, m.s));
    }
    const tk = wk + B;
    const kt = try gpuOp1(mlx.mlx_transpose, kmat, m.s);
    defer _ = mlx.mlx_array_free(kt);
    const scores0 = try gpuOp2(mlx.mlx_matmul, q3, kt, m.s); // [B, nh, tk]
    defer _ = mlx.mlx_array_free(scores0);
    const scale_arr = mlx.mlx_array_new_float(@floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd)))));
    defer _ = mlx.mlx_array_free(scale_arr);
    const scaled = try gpuOp2(mlx.mlx_multiply, scores0, scale_arr, m.s);
    defer _ = mlx.mlx_array_free(scaled);
    var sink3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sink3);
    {
        const s1 = [_]c_int{ 1, @intCast(nh), 1 };
        const sink_r = try gpuReshape(h.sink_gpu, &s1, m.s);
        defer _ = mlx.mlx_array_free(sink_r);
        const sb = [_]c_int{ bc, @intCast(nh), 1 };
        try mlx.check(mlx.mlx_broadcast_to(&sink3, sink_r, &sb, 3, m.s));
    }
    const with_sink = try gpuConcat2(scaled, sink3, 2, m.s);
    defer _ = mlx.mlx_array_free(with_sink);
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, with_sink, -1, true, m.s));
    const p_real = try gpuSliceLast3(probs, B, nh, 0, tk, m.s);
    defer _ = mlx.mlx_array_free(p_real);
    const o = try gpuOp2(mlx.mlx_matmul, p_real, kmat, m.s); // [B, nh, hd]
    defer _ = mlx.mlx_array_free(o);
    const o_inv = try gpuRopeTailRows(o, rd, rr.cos, rr.sin, true, m.s);
    defer _ = mlx.mlx_array_free(o_inv);
    // grouped low-rank O (the attentionBatch tail)
    const og = m.o_groups;
    const ol = m.o_lora;
    const gin = nh * hd / og;
    const o2shape = [_]c_int{ bc, @intCast(og), @intCast(gin) };
    const o2 = try gpuReshape(o_inv, &o2shape, m.s);
    defer _ = mlx.mlx_array_free(o2);
    const gaxes = [_]c_int{ 1, 0, 2 };
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    try mlx.check(mlx.mlx_transpose_axes(&ot, o2, &gaxes, 3, m.s));
    var ob = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ob);
    try mlx.check(mlx.mlx_astype(&ob, ot, .bfloat16, m.s));
    const ored = try woAMatmul(m, li, ob); // [og, B, ol]
    defer _ = mlx.mlx_array_free(ored);
    var ort = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ort);
    try mlx.check(mlx.mlx_transpose_axes(&ort, ored, &gaxes, 3, m.s)); // [B, og, ol]
    const orshape = [_]c_int{ bc, @intCast(og * ol) };
    const orr = try gpuReshape(ort, &orshape, m.s);
    defer _ = mlx.mlx_array_free(orr);
    return try gpuQmmB(&ly.wo_b, orr, m.s); // [B, d]
}

pub const DsparkDraft = struct {
    ids: []u32, // [block+1]; ids[0] = the trunk token that seeded the draft
    /// Drafts actually SUBMITTED for verification: `ids[1..len+1]`. Below
    /// `block` when the confidence gate truncated the block (0 = draft
    /// nothing, i.e. this round verifies the trunk token alone).
    len: usize,
    logits: []f32, // [block * vocab] Markov-biased logits (row-major; live rows only)
    confidence: []f32, // [block] logits; positions past `len` are unevaluated
    /// Sequential Markov-loop time (ns) — 0 unless profiling is armed.
    markov_ns: u64 = 0,

    pub fn deinit(self: *DsparkDraft, alloc: std.mem.Allocator) void {
        alloc.free(self.ids);
        alloc.free(self.logits);
        alloc.free(self.confidence);
    }
};

// ── DSpark cost audit (`MLX_SERVE_DSPARK_PROFILE=1`) ───────────────────
//
// Unlike the decode profiler (whose per-phase evals kill pipelining and make
// it a lying sizing tool), every phase boundary measured here is ALREADY a
// host sync in the un-profiled path: the draft ends in the confidence read,
// both verify paths end in a logits read, and the snapshot is pure host work.
// So these laps add no evals and the numbers are the shipping path's.

/// One round's measured phases (ns) plus what it bought.
pub const DsparkPhases = struct {
    draft_ns: u64 = 0,
    /// Of `draft_ns`: the sequential Markov bigram loop (B host syncs).
    markov_ns: u64 = 0,
    snapshot_ns: u64 = 0,
    verify_ns: u64 = 0,
    /// Of `verify_ns`: the vocab head over all B+1 rows (the M=B+1 lane).
    verify_head_ns: u64 = 0,
    /// Of `verify_ns`: blocking per-layer compressor-input reads.
    verify_comp_sync_ns: u64 = 0,
    /// Restore + re-extend of the accepted prefix (0 on a full accept).
    rollback_ns: u64 = 0,
    accepted: u32 = 0,
    committed: u32 = 0,
    /// Drafts SUBMITTED this round (what the confidence gate let through).
    submitted: u32 = 0,
};

pub const DsparkProfile = struct {
    rounds: u64 = 0,
    draft_ns: u64 = 0,
    markov_ns: u64 = 0,
    snapshot_ns: u64 = 0,
    verify_ns: u64 = 0,
    verify_head_ns: u64 = 0,
    verify_comp_sync_ns: u64 = 0,
    rollback_ns: u64 = 0,
    accepted: u64 = 0,
    committed: u64 = 0,
    submitted: u64 = 0,

    pub fn observe(self: *DsparkProfile, r: DsparkPhases) void {
        self.rounds += 1;
        self.draft_ns += r.draft_ns;
        self.markov_ns += r.markov_ns;
        self.snapshot_ns += r.snapshot_ns;
        self.verify_ns += r.verify_ns;
        self.verify_head_ns += r.verify_head_ns;
        self.verify_comp_sync_ns += r.verify_comp_sync_ns;
        self.rollback_ns += r.rollback_ns;
        self.accepted += r.accepted;
        self.committed += r.committed;
        self.submitted += r.submitted;
    }

    pub const Summary = struct {
        round_ms: f64,
        draft_ms: f64,
        markov_ms: f64,
        snapshot_ms: f64,
        verify_ms: f64,
        verify_head_ms: f64,
        verify_comp_sync_ms: f64,
        rollback_ms: f64,
        accepts_per_round: f64,
        submitted_per_round: f64,
        /// THE arbiter: round cost per COMMITTED token. DSpark pays only
        /// while this sits below the serial step time on the same box.
        ms_per_token: f64,
    };

    pub fn summary(self: *const DsparkProfile) Summary {
        if (self.rounds == 0) return std.mem.zeroes(Summary);
        const r: f64 = @floatFromInt(self.rounds);
        const ms: f64 = @floatFromInt(std.time.ns_per_ms);
        const per = struct {
            fn f(ns: u64, rounds: f64, msn: f64) f64 {
                return @as(f64, @floatFromInt(ns)) / msn / rounds;
            }
        }.f;
        const total_ns = self.draft_ns + self.snapshot_ns + self.verify_ns + self.rollback_ns;
        const committed: f64 = @floatFromInt(self.committed);
        return .{
            .round_ms = per(total_ns, r, ms),
            .draft_ms = per(self.draft_ns, r, ms),
            .markov_ms = per(self.markov_ns, r, ms),
            .snapshot_ms = per(self.snapshot_ns, r, ms),
            .verify_ms = per(self.verify_ns, r, ms),
            .verify_head_ms = per(self.verify_head_ns, r, ms),
            .verify_comp_sync_ms = per(self.verify_comp_sync_ns, r, ms),
            .rollback_ms = per(self.rollback_ns, r, ms),
            .accepts_per_round = @as(f64, @floatFromInt(self.accepted)) / r,
            .submitted_per_round = @as(f64, @floatFromInt(self.submitted)) / r,
            .ms_per_token = if (self.committed == 0) 0 else @as(f64, @floatFromInt(total_ns)) / ms / committed,
        };
    }

    pub fn report(self: *const DsparkProfile) void {
        const s = self.summary();
        log.info(
            "[dspark-prof] rounds={d} sub/round={d:.2} accepts/round={d:.2} round={d:.1}ms (draft {d:.1} [markov {d:.1}] snap {d:.2} verify {d:.1} [head {d:.1} sync {d:.1}] rollback {d:.1}) -> {d:.1} ms/token\n",
            .{ self.rounds, s.submitted_per_round, s.accepts_per_round, s.round_ms, s.draft_ms, s.markov_ms, s.snapshot_ms, s.verify_ms, s.verify_head_ms, s.verify_comp_sync_ms, s.rollback_ms, s.ms_per_token },
        );
    }
};

/// Monotonic phase clock (this Zig nightly has no std.time.Timer — the
/// transformer.zig ProfClock pattern).
const DsparkClock = struct {
    io: std.Io,
    mark: std.Io.Timestamp,
    fn init() DsparkClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .mark = std.Io.Timestamp.now(io, .boot) };
    }
    fn lap(self: *DsparkClock) u64 {
        const now = std.Io.Timestamp.now(self.io, .boot);
        const d: u64 = @intCast(self.mark.untilNow(self.io, .boot).nanoseconds);
        self.mark = now;
        return d;
    }
};

/// The reference's forward_spec at start_pos>0, greedy: draft ids
/// [trunk_tok, noise…] → 3 stages (main_x conditioning already in the rings)
/// → last stage's OWN hc collapse → shared trunk head → sequential Markov
/// bigram bias + argmax per position → confidence. Never mutates `st`.
pub fn dsparkDraft(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, trunk_tok: u32) !DsparkDraft {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const B = m.ds_block;
    const bc: c_int = @intCast(B);
    const d = m.dim;
    const hcm = m.hc;
    const n = st.n; // start_pos + 1: the trunk token at n-1 is in the rings

    // plain rope rows for the draft positions n .. n+B-1
    const fr_plain = try freqsFor(m, .plain, n + B + 2, m.arena.allocator());
    const half = m.rd / 2;
    const rowshape = [_]c_int{ bc, @intCast(half) };
    const rr = RopeRows{
        .cos = uploadF32(fr_plain.cos[n * half ..][0 .. B * half], &rowshape),
        .sin = uploadF32(fr_plain.sin[n * half ..][0 .. B * half], &rowshape),
    };
    defer _ = mlx.mlx_array_free(rr.cos);
    defer _ = mlx.mlx_array_free(rr.sin);

    // draft ids [trunk_tok, noise…] → embed → [B, hc, d] stream
    const draft_ids = try a.alloc(u32, B);
    @memset(draft_ids, m.ds_noise);
    draft_ids[0] = trunk_tok;
    var stream_g = blk: {
        const eh = try a.alloc(f32, B * d);
        for (0..B) |t| @memcpy(eh[t * d ..][0..d], m.embed_f32[@as(usize, draft_ids[t]) * d ..][0..d]);
        const eshape = [_]c_int{ bc, 1, @intCast(d) };
        const e_rows = uploadF32(eh, &eshape);
        defer _ = mlx.mlx_array_free(e_rows);
        const bshape = [_]c_int{ bc, @intCast(hcm), @intCast(d) };
        var b = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&b, e_rows, &bshape, 3, m.s));
        break :blk b;
    };
    defer _ = mlx.mlx_array_free(stream_g);

    const ds_trace = std.c.getenv("MLX_SERVE_DSPARK_TRACE") != null;
    if (ds_trace) {
        const sh = try toHostF32(a, stream_g, B * hcm * d, m.s);
        defer a.free(sh);
        var nrm: f64 = 0;
        for (sh) |v| nrm += @as(f64, v) * v;
        log.info("[dspark-trace] embed stream norm={d:.4}\n", .{@sqrt(nrm)});
        // weight-handle probes: ones through each stage's shared_w1 + expert-0 w1
        for (0..m.n_mtp) |sti| {
            const ly = &m.dw.layers[m.n_layers + sti];
            const osh = [_]c_int{ 1, @intCast(d) };
            var ones_g = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones_g);
            try mlx.check(mlx.mlx_ones(&ones_g, &osh, 2, .bfloat16, m.s));
            const sg = try qmmBf16(&ly.shared_w1, ones_g, m.s);
            defer _ = mlx.mlx_array_free(sg);
            const sgh = try toHostF32(a, sg, m.moe_inter, m.s);
            defer a.free(sgh);
            var sn: f64 = 0;
            for (sgh) |v| sn += @as(f64, v) * v;
            const e4 = [_]c_int{ 1, 1, 1, @intCast(d) };
            var ones4 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones4);
            try mlx.check(mlx.mlx_ones(&ones4, &e4, 4, .bfloat16, m.s));
            const idx0 = [_]i32{0};
            const ish = [_]c_int{ 1, 1 };
            const ind0 = mlx.mlx_array_new_data(&idx0, &ish, 2, .int32);
            defer _ = mlx.mlx_array_free(ind0);
            const ge = try gatherQmmE(&ly.experts_w1, ones4, ind0, m.s);
            defer _ = mlx.mlx_array_free(ge);
            const geh = try toHostF32(a, ge, m.moe_inter, m.s);
            defer a.free(geh);
            var gn: f64 = 0;
            for (geh) |v| gn += @as(f64, v) * v;
            log.info("[dspark-trace] stage {d} probes: shared_w1(ones)={d:.4} expert0_w1(ones)={d:.4} qp bits={d} gs={d}\n", .{ sti, @sqrt(sn), @sqrt(gn), ly.experts_w1.qp.bits, ly.experts_w1.qp.group_size });
        }
    }
    for (0..m.n_mtp) |sti| {
        const li = m.n_layers + sti;
        const h = &m.hl[li];
        const ly = &m.dw.layers[li];
        {
            const pre = try hcPreBatch(m, a, stream_g, B, h.hc_attn_fn_t, h.hc_attn_scale, h.hc_attn_base, ly.hc_attn_scale, ly.hc_attn_base);
            defer freeHcPre(&pre);
            const x = try gpuRms(pre.y, h.attn_norm_g, m.eps, m.s);
            defer _ = mlx.mlx_array_free(x);
            const attn_out = try dsparkAttentionG(m, st, sti, x, B, &rr);
            defer _ = mlx.mlx_array_free(attn_out);
            if (ds_trace) {
                const ah = try toHostF32(a, attn_out, B * d, m.s);
                defer a.free(ah);
                var an: f64 = 0;
                for (ah) |v| an += @as(f64, v) * v;
                const ph = try toHostF32(a, pre.post_g, B * hcm, m.s);
                defer a.free(ph);
                var pn: f64 = 0;
                for (ph) |v| pn += @as(f64, v) * v;
                log.info("[dspark-trace] stage {d} attn_out={d:.4} post={d:.4} mem={d}/{d}/{d}MB\n", .{ sti, @sqrt(an), @sqrt(pn), traceMemMb(.active), traceMemMb(.cache), traceMemMb(.peak) });
            }
            const ns = try hcPostBatch(m, stream_g, attn_out, B, &pre);
            _ = mlx.mlx_array_free(stream_g);
            stream_g = ns;
        }
        {
            const pre = try hcPreBatch(m, a, stream_g, B, h.hc_ffn_fn_t, h.hc_ffn_scale, h.hc_ffn_base, ly.hc_ffn_scale, ly.hc_ffn_base);
            defer freeHcPre(&pre);
            const x = try gpuRms(pre.y, h.ffn_norm_g, m.eps, m.s);
            defer _ = mlx.mlx_array_free(x);
            const ffn_out = try moeGpu(m, a, li, x, draft_ids);
            defer _ = mlx.mlx_array_free(ffn_out);
            if (ds_trace) {
                const fh = try toHostF32(a, ffn_out, B * d, m.s);
                defer a.free(fh);
                var fnn: f64 = 0;
                for (fh) |v| fnn += @as(f64, v) * v;
                const yh = try toHostF32(a, pre.y, B * d, m.s);
                defer a.free(yh);
                var yn: f64 = 0;
                for (yh) |v| yn += @as(f64, v) * v;
                const xh = try toHostF32(a, x, B * d, m.s);
                defer a.free(xh);
                var xn: f64 = 0;
                for (xh) |v| xn += @as(f64, v) * v;
                log.info("[dspark-trace] stage {d} ffn_out={d:.4} pre.y={d:.4} x={d:.4}\n", .{ sti, @sqrt(fnn), @sqrt(yn), @sqrt(xn) });
            }
            const ns = try hcPostBatch(m, stream_g, ffn_out, B, &pre);
            _ = mlx.mlx_array_free(stream_g);
            stream_g = ns;
        }
        if (ds_trace) {
            const sh = try toHostF32(a, stream_g, B * hcm * d, m.s);
            defer a.free(sh);
            var nrm: f64 = 0;
            for (sh) |v| nrm += @as(f64, v) * v;
            log.info("[dspark-trace] stage {d} stream norm={d:.4}\n", .{ sti, @sqrt(nrm) });
        }
    }

    // last stage's OWN hc collapse (sigmoid pre weights only, batched
    // headLogitsGpu shape): one [B, hc+1] sync for the mixes ++ Σx².
    const hout = blk: {
        const fshape = [_]c_int{ bc, @intCast(hcm * d) };
        const flat = try gpuReshape(stream_g, &fshape, m.s);
        defer _ = mlx.mlx_array_free(flat);
        const mixes_g = try gpuOp2(mlx.mlx_matmul, flat, m.ds_hc_head_fn_t.?, m.s);
        defer _ = mlx.mlx_array_free(mixes_g);
        const sq = try gpuOp1(mlx.mlx_square, flat, m.s);
        defer _ = mlx.mlx_array_free(sq);
        var ssum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ssum);
        try mlx.check(mlx.mlx_sum_axis(&ssum, sq, 1, true, m.s));
        const joint = try gpuConcat2(mixes_g, ssum, 1, m.s);
        defer _ = mlx.mlx_array_free(joint);
        const jh = try toHostF32(a, joint, B * (hcm + 1), m.s);
        defer a.free(jh);
        const prew = try a.alloc(f32, B * hcm);
        for (0..B) |t| {
            const row = jh[t * (hcm + 1) ..][0 .. hcm + 1];
            const rsq: f32 = @floatCast(1.0 / @sqrt(@as(f64, row[hcm]) / @as(f64, @floatFromInt(hcm * d)) + m.eps));
            for (0..hcm) |c| prew[t * hcm + c] = sigmoidF32(row[c] * rsq * m.ds_hc_head_scale[0] + m.ds_hc_head_base[c]) + m.hc_eps;
        }
        const pshape = [_]c_int{ bc, @intCast(hcm), 1 };
        const pre_col = uploadF32(prew, &pshape);
        defer _ = mlx.mlx_array_free(pre_col);
        const weighted = try gpuOp2(mlx.mlx_multiply, pre_col, stream_g, m.s);
        defer _ = mlx.mlx_array_free(weighted);
        var ho = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&ho, weighted, 1, false, m.s)); // [B, d]
        break :blk ho;
    };
    defer _ = mlx.mlx_array_free(hout);

    var mclk: DsparkClock = if (m.ds_prof != null) DsparkClock.init() else undefined;
    const out_ids = try gpa.alloc(u32, B + 1);
    errdefer gpa.free(out_ids);
    @memset(out_ids, 0); // positions past `len` are never drafted
    out_ids[0] = trunk_tok;
    const logits = try gpa.alloc(f32, B * m.vocab);
    errdefer gpa.free(logits);
    @memset(logits, 0);
    const confidence = try gpa.alloc(f32, B);
    errdefer gpa.free(confidence);
    @memset(confidence, 0);

    // markov_w1 row for a token id, as f32 [1, rank] — the bigram state that
    // both the bias matvec and the confidence head read.
    const markovRow = struct {
        fn f(mm: *Dsv4Model, tok: u32) !mlx.mlx_array {
            const idx_shape = [_]c_int{1};
            const idx_val = [_]i32{@intCast(tok)};
            const idx = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
            defer _ = mlx.mlx_array_free(idx);
            var w1row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w1row);
            try mlx.check(mlx.mlx_take_axis(&w1row, mm.dw.dspark.?.markov_w1, idx, 0, mm.s)); // [1, rank] bf16
            var w1f = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&w1f, w1row, .float32, mm.s));
            return w1f;
        }
    }.f;

    // Confidence head on ONE position: proj([hout_i | markov_embed]) — the
    // checkpoint's own "is this position worth verifying" score. Evaluated
    // BEFORE position i's bias+argmax (and, for i=0, before the vocab head is
    // computed at all), so an unconfident block costs a dot product instead
    // of a [B, V] matmul plus B verify rows through the trunk.
    const confAt = struct {
        fn f(mm: *Dsv4Model, alloc: std.mem.Allocator, ho: mlx.mlx_array, w1f: mlx.mlx_array, i: usize) !f32 {
            const start = [_]c_int{ @intCast(i), 0 };
            const stop = [_]c_int{ @intCast(i + 1), @intCast(mm.dim) };
            const str = [_]c_int{ 1, 1 };
            var hrow = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hrow);
            try mlx.check(mlx.mlx_slice(&hrow, ho, &start, 2, &stop, 2, &str, 2, mm.s));
            const hidden = try gpuConcat2(hrow, w1f, 1, mm.s); // [1, d+rank]
            defer _ = mlx.mlx_array_free(hidden);
            const conf_g = try gpuOp2(mlx.mlx_matmul, hidden, mm.ds_conf_proj_t.?, mm.s); // [1, 1]
            defer _ = mlx.mlx_array_free(conf_g);
            const h = try toHostF32(alloc, conf_g, 1, mm.s);
            defer alloc.free(h);
            return h[0];
        }
    }.f;

    var submitted: usize = 0;
    var w1f = try markovRow(m, trunk_tok);
    defer _ = mlx.mlx_array_free(w1f);
    confidence[0] = try confAt(m, a, hout, w1f, 0);

    if (confidence[0] >= m.ds_conf_thr) {
        // norm → SHARED trunk head → [B, V] f32
        const logits_g = blk: {
            const hn = try gpuRms(hout, m.ds_last_norm_g.?, m.eps, m.s);
            defer _ = mlx.mlx_array_free(hn);
            break :blk try gpuQmmB(&m.dw.head, hn, m.s);
        };
        defer _ = mlx.mlx_array_free(logits_g);

        // sequential Markov bigram bias + greedy sample (host loop; each
        // step's bias matvec runs on GPU, the biased ROW syncs once and
        // argmax is host). Stops at the first position the confidence head
        // does not vouch for — everything after it would be verified at full
        // trunk cost on a draft the model itself doubts.
        for (0..B) |i| {
            if (confidence[i] < m.ds_conf_thr) break;
            const bias = try gpuOp2(mlx.mlx_matmul, w1f, m.ds_markov_w2_t.?, m.s); // [1, V]
            defer _ = mlx.mlx_array_free(bias);
            const row_start = [_]c_int{ @intCast(i), 0 };
            const row_stop = [_]c_int{ @intCast(i + 1), @intCast(m.vocab) };
            const row_str = [_]c_int{ 1, 1 };
            var lr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(lr);
            try mlx.check(mlx.mlx_slice(&lr, logits_g, &row_start, 2, &row_stop, 2, &row_str, 2, m.s));
            const biased = try gpuOp2(mlx.mlx_add, lr, bias, m.s);
            defer _ = mlx.mlx_array_free(biased);
            const brow = try toHostF32(a, biased, m.vocab, m.s);
            defer a.free(brow);
            @memcpy(logits[i * m.vocab ..][0..m.vocab], brow);
            var am: usize = 0;
            for (brow, 0..) |v, j| {
                if (v > brow[am]) am = j;
            }
            out_ids[i + 1] = @intCast(am);
            submitted = i + 1;
            if (i + 1 < B) {
                _ = mlx.mlx_array_free(w1f);
                w1f = try markovRow(m, out_ids[i + 1]);
                confidence[i + 1] = try confAt(m, a, hout, w1f, i + 1);
            }
        }
    }
    const markov_ns: u64 = if (m.ds_prof != null) mclk.lap() else 0;

    return .{ .ids = out_ids, .len = submitted, .logits = logits, .confidence = confidence, .markov_ns = markov_ns };
}

pub const DsparkRound = struct {
    /// Tokens COMMITTED this round, in order: [t1, accepted drafts…].
    tokens: []u32,
    /// The next round's trunk token — the correction (partial accept) or the
    /// bonus token (full accept), always from the ORIGINAL verify logits at
    /// the acceptance point (the house partial-accept invariant).
    next_token: u32,
    /// Drafted tokens accepted (excludes the always-committed t1).
    accepted: u32,
    /// Measured phases — populated only while `Dsv4Model.ds_prof` is armed
    /// (the draft half is filled in by `dsparkRound`, which owns that call).
    phases: DsparkPhases = .{},

    pub fn deinit(self: *DsparkRound, alloc: std.mem.Allocator) void {
        alloc.free(self.tokens);
    }
};

/// An in-flight round between `dsparkBegin`/`dsparkBeginWith` and
/// `dsparkFinish`: the draft ran, the state snapshot + anchors are armed,
/// the verify block was appended to `st`, and `vl_g` holds the LAZY
/// `[b+1, vocab]` verify logits — un-synced, so the caller can build its
/// accept decision (host argmax read, or the stochastic filtered-probs +
/// accept graph) on top and pay ONE bounded sync. The caller always
/// `deinit`s (finish borrows the snapshot for rollback, it does not free).
pub const DsparkPending = struct {
    snap: Dsv4Snapshot,
    /// Lazy `[b+1, vocab]` verify logits (every position, trunk head).
    vl_g: mlx.mlx_array,
    /// Drafts submitted this round (≤ ds_block; the confidence gate may
    /// have truncated the block, possibly to 0).
    b: usize,
    /// The verify ids `[t1, d1..db]` — everything an accept loop needs
    /// from the draft.
    verify: [16]u32,
    phases: DsparkPhases,
    clk: DsparkClock,
    prof_on: bool,

    /// Called by the consumer right after ITS host sync of `vl_g`'s graph —
    /// the honest place for the verify lap now that begin returns lazy.
    /// `verify_head_ns` is build-only under the split (the head eval lands
    /// in this lap); `verify_ns` still covers the whole verify as before.
    pub fn lapVerify(self: *DsparkPending, m: *const Dsv4Model) void {
        if (!self.prof_on) return;
        self.phases.verify_ns = self.clk.lap();
        self.phases.verify_head_ns = m.ds_prof_head_ns;
        self.phases.verify_comp_sync_ns = m.ds_prof_comp_sync_ns;
    }

    pub fn deinit(self: *DsparkPending) void {
        self.snap.deinit();
        _ = mlx.mlx_array_free(self.vl_g);
    }
};

/// Front half of a round: draft from the stages, then arm the verify
/// (snapshot + anchors + batched extend with LAZY all-position logits).
/// Entry invariant as `dsparkRound`: `t1` NOT in state, pending empty.
pub fn dsparkBegin(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, t1: u32) !DsparkPending {
    const prof_on = m.ds_prof != null;
    var clk: DsparkClock = if (prof_on) DsparkClock.init() else undefined;
    var draft = try dsparkDraft(m, gpa, st, t1);
    defer draft.deinit(gpa);
    const draft_ns: u64 = if (prof_on) clk.lap() else 0;
    var pending = try dsparkBeginWith(m, gpa, st, t1, &draft);
    pending.phases.draft_ns = draft_ns;
    pending.phases.markov_ns = draft.markov_ns;
    return pending;
}

/// The verify-arming half with the draft injected — the seam the FULL-ACCEPT
/// test (and `dsparkRound`'s own draft) drives. Borrows the draft: the ids
/// the accept loop needs are copied into `pending.verify`.
pub fn dsparkBeginWith(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, t1: u32, draft: *const DsparkDraft) !DsparkPending {
    // The block the confidence gate actually submitted (≤ ds_block).
    const B = draft.len;
    std.debug.assert(B <= m.ds_block and st.dspark != null);
    const prof_on = m.ds_prof != null;
    var clk: DsparkClock = if (prof_on) DsparkClock.init() else undefined;
    var ph = DsparkPhases{};

    var snap = try snapshotDecodeState(st, gpa);
    errdefer snap.deinit();
    // Per-position anchors: a rejected tail becomes a truncate instead of a
    // second batched forward over the accepted prefix. Sized to the FULL
    // block so the buffers are allocated once and reused whatever the
    // confidence gate submits this round.
    try armAnchors(m, st, m.ds_block);
    if (prof_on) ph.snapshot_ns = clk.lap();

    var verify: [16]u32 = undefined;
    verify[0] = t1;
    @memcpy(verify[1 .. B + 1], draft.ids[1 .. B + 1]);
    std.debug.assert(B + 1 <= prefillSub());
    const vl_g = try extendChunk(m, gpa, st, verify[0 .. B + 1], .all_gpu);
    return .{ .snap = snap, .vl_g = vl_g, .b = B, .verify = verify, .phases = ph, .clk = clk, .prof_on = prof_on };
}

/// Back half: commit/rollback against the accept decision. `accepted` drafts
/// (≤ pending.b) survive; a partial accept truncates to the per-position
/// anchor, a full accept disarms them (the verify appends ARE the state).
/// `next_token` is the caller's — correction or bonus, always derived from
/// the ORIGINAL verify logits at the acceptance point (the house
/// partial-accept invariant; sampled form on the stochastic arm).
pub fn dsparkFinish(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, pending: *DsparkPending, accepted: usize, next_token: u32) !DsparkRound {
    _ = m;
    std.debug.assert(accepted <= pending.b);
    if (accepted < pending.b) {
        if (pending.prof_on) _ = pending.clk.lap(); // the accept scan is host-only bookkeeping
        restoreToAnchor(st, &pending.snap, accepted);
        if (pending.prof_on) pending.phases.rollback_ns = pending.clk.lap();
    } else if (st.anchors) |*an| an.armed = false;

    const tokens = try gpa.alloc(u32, accepted + 1);
    @memcpy(tokens, pending.verify[0 .. accepted + 1]);
    pending.phases.accepted = @intCast(accepted);
    pending.phases.committed = @intCast(accepted + 1);
    pending.phases.submitted = @intCast(pending.b);
    return .{ .tokens = tokens, .next_token = next_token, .accepted = @intCast(accepted), .phases = pending.phases };
}

/// Feed a finished round's phases into the profile (armed via
/// MLX_SERVE_DSPARK_PROFILE=1) — shared by the greedy wrapper here and the
/// stochastic arm in generate.zig.
pub fn dsparkObserve(m: *Dsv4Model, ph: DsparkPhases) void {
    if (m.ds_prof) |*p| {
        p.observe(ph);
        if (p.rounds % 16 == 0) p.report();
    }
}

/// One greedy draft/verify round. Entry: st holds prompt+emitted positions
/// and `t1` (the trunk-sampled token) is NOT in the state. Draft B ids from
/// t1, batch-verify `[t1, d1..dB]` through the trunk, accept the longest
/// argmax-matching prefix, and roll the rejected tail back — FULL restore to
/// the snapshot plus a re-extend of the accepted prefix (the compressor
/// pending rings are overwritten in place, so partial-position rollback has
/// no anchor short of the snapshot). Exit: st = entry + tokens.len positions,
/// next_token not in state — the entry invariant again.
pub fn dsparkRound(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, t1: u32, accepted_cap: usize) !DsparkRound {
    const prof_on = m.ds_prof != null;
    var clk: DsparkClock = if (prof_on) DsparkClock.init() else undefined;
    var draft = try dsparkDraft(m, gpa, st, t1);
    defer draft.deinit(gpa);
    const draft_ns: u64 = if (prof_on) clk.lap() else 0;
    const round = try dsparkRoundWith(m, gpa, st, t1, &draft, accepted_cap);
    if (m.ds_prof != null) {
        var ph = round.phases;
        ph.draft_ns = draft_ns;
        ph.markov_ns = draft.markov_ns;
        dsparkObserve(m, ph);
    }
    return round;
}

/// The verify/accept half of a round, with the draft injected — the seam the
/// FULL-ACCEPT test drives (the random mini's own drafts never match, so
/// only this entry can exercise the no-rollback branch hermetically). Built
/// ON TOP of the begin/finish split so the greedy and stochastic arms share
/// one verify seam: begin → host read → argmax accept loop → finish.
fn dsparkRoundWith(m: *Dsv4Model, gpa: std.mem.Allocator, st: *Dsv4DecodeState, t1: u32, draft: *const DsparkDraft, accepted_cap: usize) !DsparkRound {
    var pending = try dsparkBeginWith(m, gpa, st, t1, draft);
    defer pending.deinit();
    const B = pending.b;
    const vl = try toHostF32(gpa, pending.vl_g, (B + 1) * m.vocab, m.s);
    defer gpa.free(vl);
    pending.lapVerify(m);

    var accepted: usize = 0;
    while (accepted < B) {
        const row = vl[accepted * m.vocab ..][0..m.vocab];
        var am: usize = 0;
        for (row, 0..) |v, j| {
            if (v > row[am]) am = j;
        }
        if (am != draft.ids[accepted + 1]) break;
        accepted += 1;
    }
    // The always-emitted t1 consumes one request-budget token. Cap the draft
    // prefix before choosing the pending token and before dsparkFinish commits
    // or rolls module-owned state back to the accepted boundary.
    accepted = @min(accepted, accepted_cap);
    const nrow = vl[accepted * m.vocab ..][0..m.vocab];
    var next_am: usize = 0;
    for (nrow, 0..) |v, j| {
        if (v > nrow[next_am]) next_am = j;
    }
    if (std.c.getenv("MLX_SERVE_DSPARK_TRACE") != null) {
        var vam: [16]u32 = undefined;
        for (0..B) |k| {
            const row = vl[k * m.vocab ..][0..m.vocab];
            var am2: usize = 0;
            for (row, 0..) |v, j| {
                if (v > row[am2]) am2 = j;
            }
            vam[k] = @intCast(am2);
        }
        var v_nan: usize = 0;
        for (vl) |v| {
            if (!std.math.isFinite(v)) v_nan += 1;
        }
        var d_nan: usize = 0;
        for (draft.logits) |v| {
            if (!std.math.isFinite(v)) d_nan += 1;
        }
        log.info("[dspark-trace] n={d} t1={d} draft={any} verify_am={any} accepted={d} next={d} conf={any} nan(verify)={d}/{d} nan(draft)={d}/{d}\n", .{
            st.n - (B + 1),  t1,         draft.ids[1 .. B + 1], vam[0..B], accepted, next_am, draft.confidence,
            v_nan,           vl.len,     d_nan,                 draft.logits.len,
        });
    }

    return dsparkFinish(m, gpa, st, &pending, accepted, @intCast(next_am));
}

// ── GPU-side QAT sims + rope (consolidation round) ─────────────────────
//
// Parity-pinned against the golden-tested host helpers (test below). The
// only tolerated deviation is round-to-nearest tie behavior (metal rint vs
// np.round — both RNE in practice; the parity test sweeps enough values to
// catch a real mismatch).

fn gpuOp1(comptime f: anytype, a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(f(&r, a, s));
    return r;
}

fn gpuOp2(comptime f: anytype, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(f(&r, a, b, s));
    return r;
}

/// e4m3/e2m1 quant-dequant on a GPU tensor whose LAST dim is a multiple of
/// `group`. Mirrors simInPlace exactly (ue8m0 scale = 2^ceil(log2(amax/max))).
fn gpuSim(x: mlx.mlx_array, group: usize, amax_floor: f32, code_max: f32, mant_bits: f32, min_exp: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const ndim = mlx.mlx_array_ndim(x);
    var shape: [8]c_int = undefined;
    var total: usize = 1;
    const sh = mlx.mlx_array_shape(x);
    for (0..ndim) |i| {
        shape[i] = sh[i];
        total *= @intCast(shape[i]);
    }
    const g: c_int = @intCast(group);
    const gshape = [_]c_int{ @intCast(total / group), g };
    var xg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xg);
    try mlx.check(mlx.mlx_reshape(&xg, x, &gshape, 2, s));
    const ax = try gpuOp1(mlx.mlx_abs, xg, s);
    defer _ = mlx.mlx_array_free(ax);
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_max_axis(&amax, ax, 1, true, s));
    const floor_arr = mlx.mlx_array_new_float(amax_floor);
    defer _ = mlx.mlx_array_free(floor_arr);
    const amax_f = try gpuOp2(mlx.mlx_maximum, amax, floor_arr, s);
    defer _ = mlx.mlx_array_free(amax_f);
    const cmax = mlx.mlx_array_new_float(code_max);
    defer _ = mlx.mlx_array_free(cmax);
    const ratio = try gpuOp2(mlx.mlx_divide, amax_f, cmax, s);
    defer _ = mlx.mlx_array_free(ratio);
    const lg = try gpuOp1(mlx.mlx_log2, ratio, s);
    defer _ = mlx.mlx_array_free(lg);
    const ce = try gpuOp1(mlx.mlx_ceil, lg, s);
    defer _ = mlx.mlx_array_free(ce);
    const two = mlx.mlx_array_new_float(2.0);
    defer _ = mlx.mlx_array_free(two);
    const scale = try gpuOp2(mlx.mlx_power, two, ce, s);
    defer _ = mlx.mlx_array_free(scale);
    const yd = try gpuOp2(mlx.mlx_divide, xg, scale, s);
    defer _ = mlx.mlx_array_free(yd);
    const ncmax = mlx.mlx_array_new_float(-code_max);
    defer _ = mlx.mlx_array_free(ncmax);
    const ylo = try gpuOp2(mlx.mlx_maximum, yd, ncmax, s);
    defer _ = mlx.mlx_array_free(ylo);
    const y = try gpuOp2(mlx.mlx_minimum, ylo, cmax, s);
    defer _ = mlx.mlx_array_free(y);
    // round |y| to the grid: e = max(floor(log2(max(|y|, 2^min_exp))), min_exp)
    const ya = try gpuOp1(mlx.mlx_abs, y, s);
    defer _ = mlx.mlx_array_free(ya);
    const min_sub = mlx.mlx_array_new_float(std.math.pow(f32, 2.0, min_exp));
    defer _ = mlx.mlx_array_free(min_sub);
    const ya_f = try gpuOp2(mlx.mlx_maximum, ya, min_sub, s);
    defer _ = mlx.mlx_array_free(ya_f);
    const lg2 = try gpuOp1(mlx.mlx_log2, ya_f, s);
    defer _ = mlx.mlx_array_free(lg2);
    const fl = try gpuOp1(mlx.mlx_floor, lg2, s);
    defer _ = mlx.mlx_array_free(fl);
    const mine = mlx.mlx_array_new_float(min_exp);
    defer _ = mlx.mlx_array_free(mine);
    const e = try gpuOp2(mlx.mlx_maximum, fl, mine, s);
    defer _ = mlx.mlx_array_free(e);
    const mant = mlx.mlx_array_new_float(mant_bits);
    defer _ = mlx.mlx_array_free(mant);
    const em = try gpuOp2(mlx.mlx_subtract, e, mant, s);
    defer _ = mlx.mlx_array_free(em);
    const quantum = try gpuOp2(mlx.mlx_power, two, em, s);
    defer _ = mlx.mlx_array_free(quantum);
    const yq = try gpuOp2(mlx.mlx_divide, ya, quantum, s);
    defer _ = mlx.mlx_array_free(yq);
    var yr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yr);
    try mlx.check(mlx.mlx_round(&yr, yq, 0, s));
    const yg = try gpuOp2(mlx.mlx_multiply, yr, quantum, s);
    defer _ = mlx.mlx_array_free(yg);
    const ycap = try gpuOp2(mlx.mlx_minimum, yg, cmax, s);
    defer _ = mlx.mlx_array_free(ycap);
    const sgn = try gpuOp1(mlx.mlx_sign, y, s);
    defer _ = mlx.mlx_array_free(sgn);
    const signed = try gpuOp2(mlx.mlx_multiply, ycap, sgn, s);
    defer _ = mlx.mlx_array_free(signed);
    const rescaled = try gpuOp2(mlx.mlx_multiply, signed, scale, s);
    defer _ = mlx.mlx_array_free(rescaled);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, rescaled, &shape, ndim, s));
    return out;
}

pub fn gpuFp8Sim(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return gpuSim(x, 64, 1e-4, 448.0, 3.0, -6.0, s);
}

pub fn gpuFp4Sim(x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return gpuSim(x, 32, 6.0 * std.math.pow(f32, 2.0, -126.0), 6.0, 1.0, 0.0, s);
}

test "dsv4: GPU QAT sims match the host golden helpers" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var rng = std.Random.DefaultPrng.init(3);
    var vals: [512]f32 = undefined;
    for (&vals, 0..) |*v, i| {
        // mix magnitudes across binades incl. tiny/huge/negatives/zeros
        const mag = std.math.pow(f32, 10.0, (rng.random().float(f32) - 0.5) * 8.0);
        v.* = if (i % 17 == 0) 0.0 else (rng.random().float(f32) - 0.5) * 2.0 * mag;
    }
    for ([_]enum { fp8, fp4 }{ .fp8, .fp4 }) |kind| {
        var host = vals;
        const shape = [_]c_int{ 8, 64 };
        const arr = mlx.mlx_array_new_data(&vals, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(arr);
        const got_arr = switch (kind) {
            .fp8 => blk: {
                fp8SimInPlace(&host, 64);
                break :blk try gpuFp8Sim(arr, s);
            },
            .fp4 => blk: {
                fp4SimInPlace(&host, 32);
                break :blk try gpuFp4Sim(arr, s);
            },
        };
        defer _ = mlx.mlx_array_free(got_arr);
        try mlx.check(mlx.mlx_array_eval(got_arr));
        const ptr = mlx.mlx_array_data_float32(got_arr).?;
        for (host, ptr[0..512], 0..) |want, got, i| {
            if (want != got) {
                std.debug.print("sim mismatch kind={s} i={d}: host={e} gpu={e} in={e}\n", .{ @tagName(kind), i, want, got, vals[i] });
                try testing.expect(false);
            }
        }
    }
}

test "dsv4: routeGpu matches host routeToken (scored + hash)" {
    // The GPU MoE routing (sqrt(logaddexp) scores, argpartition-tail
    // selection on sp + bias, take_along-normalized weights) replaced the
    // per-layer [C, E] host score sync. Pin it against the host reference
    // (sqrtSoftplus + routeToken, f64 normalize): index SETS must match
    // (argpartition's within-partition order is unspecified vs the host's
    // descending insertion sort — a set is the contract, the weighted sum is
    // order-invariant), each selected expert's weight within f32-vs-f64
    // normalize tolerance. Hash rows are a positional table lookup on BOTH
    // sides, so those compare exactly and in order.
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    const E: usize = 16;
    const k: usize = 4;
    const C: usize = 3;
    const route_scale: f32 = 1.5;

    var rng = std.Random.DefaultPrng.init(7);
    var raw: [C * E]f32 = undefined;
    for (&raw) |*v| v.* = (rng.random().float(f32) - 0.5) * 6.0;
    var bias: [E]f32 = undefined;
    for (&bias) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;

    // Host reference shells: routeToken reads only these fields.
    var m: Dsv4Model = undefined;
    m.n_experts = E;
    m.topk = k;
    m.route_scale = route_scale;

    var sp_host: [C * E]f32 = raw;
    for (&sp_host) |*v| v.* = @floatCast(sqrtSoftplus(v.*));

    const score_shape = [_]c_int{ @intCast(C), @intCast(E) };
    const scores_arr = mlx.mlx_array_new_data(&raw, &score_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(scores_arr);

    const readBack = struct {
        fn f(a: std.mem.Allocator, rg: *const RouteG, st: mlx.mlx_stream, n: usize) !struct { ind: []f32, w: []f32 } {
            var indf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(indf);
            try mlx.check(mlx.mlx_astype(&indf, rg.ind, .float32, st));
            const gi = try toHostF32(a, indf, n, st);
            errdefer a.free(gi);
            const gw = try toHostF32(a, rg.w, n, st);
            return .{ .ind = gi, .w = gw };
        }
    }.f;

    // ── Scored layer ──
    {
        var h: HostLayer = undefined;
        h.tid2eid = null;
        h.gate_bias = bias[0..];

        const bias_shape = [_]c_int{@intCast(E)};
        const bias_arr = mlx.mlx_array_new_data(&bias, &bias_shape, 1, .float32);
        defer _ = mlx.mlx_array_free(bias_arr);

        const ids = [_]u32{ 0, 0, 0 }; // unused for scored selection; len = C
        var rg = try routeGpu(s, alloc, scores_arr, bias_arr, null, null, null, &ids, E, k, route_scale);
        defer rg.deinit();
        const got = try readBack(alloc, &rg, s, C * k);
        defer alloc.free(got.ind);
        defer alloc.free(got.w);

        for (0..C) |t| {
            var hind: [k]i32 = undefined;
            var hw: [k]f32 = undefined;
            routeToken(&m, &h, sp_host[t * E ..][0..E], 0, &hind, &hw);
            for (hind, hw) |hi, hwv| {
                var found = false;
                for (0..k) |j| {
                    const gi: i32 = @intFromFloat(got.ind[t * k + j]);
                    if (gi == hi) {
                        found = true;
                        try testing.expect(@abs(got.w[t * k + j] - hwv) <= 1e-5);
                        break;
                    }
                }
                if (!found) {
                    std.debug.print("routeGpu scored: token {d} missing expert {d}\n", .{ t, hi });
                    try testing.expect(false);
                }
            }
        }
    }

    // ── Hash layer ──
    {
        const V: usize = 8;
        var table: [V * k]i64 = undefined;
        for (&table) |*v| v.* = rng.random().uintLessThan(u32, E);
        var h: HostLayer = undefined;
        h.tid2eid = table[0..];
        h.gate_bias = null;

        const ids = [_]u32{ 2, 7, 0 };
        var rg = try routeGpu(s, alloc, scores_arr, null, table[0..], null, null, &ids, E, k, route_scale);
        defer rg.deinit();
        const got = try readBack(alloc, &rg, s, C * k);
        defer alloc.free(got.ind);
        defer alloc.free(got.w);

        for (ids, 0..) |id, t| {
            var hind: [k]i32 = undefined;
            var hw: [k]f32 = undefined;
            routeToken(&m, &h, sp_host[t * E ..][0..E], id, &hind, &hw);
            for (0..k) |j| {
                const gi: i32 = @intFromFloat(got.ind[t * k + j]);
                try testing.expectEqual(hind[j], gi);
                try testing.expect(@abs(got.w[t * k + j] - hw[j]) <= 1e-5);
            }
        }
    }
}

test "dsv4: fused sinkhorn kernel matches the host reference (GPU)" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    // Same inputs as the host golden test; rsq forced to 1 (ssq=hd_full=1,
    // rms_eps=0) so the kernel sees the already-scaled mixes verbatim.
    const mixes = [_]f32{ -0.561352015, -0.927051306, -0.173853129, 0.294311672, 0.795232594, 0.0767944828, -0.386853129, -0.549346268, 0.524122059, 1.14434814, 0.190938145, -0.863330066, -0.670785666, 1.12001336, 0.142017707, -1.21249437, -0.0585873351, -0.814258158, -0.44050166, -0.341604084, -0.499319375, 0.387364924, -0.0441601798, -0.412601888 };
    const scale = [_]f32{ 0.5, 0.8, 1.2 };
    const base = [_]f32{ 0.122891352, 0.248956591, -0.492907017, -0.0770190358, -0.294224203, -0.0519465692, -0.386825621, 0.00620711828, -0.0113657219, -0.091301322, -0.314377964, -0.118857101, -0.327398658, -0.406562626, 0.0674357191, -0.332804978, 0.351088822, 0.214976296, -0.599345028, 0.081638664, -0.330514997, 0.00991716608, 0.0130895982, -0.596528947 };
    const hc: usize = 4;
    const want = hcSplitSinkhorn(&mixes, &scale, &base, hc, 20, 1e-6);
    var sk = buildSinkhornKernel(hc, 20) orelse return error.MetalKernelCompileFailed;
    defer _ = mlx.mlx_fast_metal_kernel_config_free(sk.cfg);
    defer _ = mlx.mlx_fast_metal_kernel_free(sk.kernel);
    const mshape = [_]c_int{ 1, 24 };
    const mixes_g = uploadF32(&mixes, &mshape);
    defer _ = mlx.mlx_array_free(mixes_g);
    const one = [_]f32{1.0};
    const sshape = [_]c_int{ 1, 1 };
    const ss_g = uploadF32(&one, &sshape);
    defer _ = mlx.mlx_array_free(ss_g);
    const scshape = [_]c_int{3};
    const scale_g = uploadF32(&scale, &scshape);
    defer _ = mlx.mlx_array_free(scale_g);
    const bshape = [_]c_int{24};
    const base_g = uploadF32(&base, &bshape);
    defer _ = mlx.mlx_array_free(base_g);
    const consts = [_]f32{ 1.0, 0.0, 1e-6 };
    const consts_g = uploadF32(&consts, &scshape);
    defer _ = mlx.mlx_array_free(consts_g);
    const pk = try applySinkhorn(&sk, sk.cfg, mixes_g, ss_g, scale_g, base_g, consts_g, s);
    defer _ = mlx.mlx_array_free(pk);
    try mlx.check(mlx.mlx_array_eval(pk));
    const ptr = mlx.mlx_array_data_float32(pk).?;
    for (0..hc) |j| {
        try testing.expectApproxEqAbs(want.pre[j], ptr[j], 1e-4);
        try testing.expectApproxEqAbs(want.post[j], ptr[hc + j], 1e-4);
    }
    for (0..hc) |j| {
        for (0..hc) |k| {
            // kernel output is TRANSPOSED: out[2hc + k*hc + j] == comb[j*hc+k]
            try testing.expectApproxEqAbs(want.comb[j * hc + k], ptr[2 * hc + k * hc + j], 1e-4);
        }
    }
}

test "dsv4: Hadamard matrix matmul matches hadamardInPlace" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const n: usize = 32;
    const hada = try buildHadamardF32(testing.allocator, n, s);
    defer _ = mlx.mlx_array_free(hada);
    var rng = std.Random.DefaultPrng.init(7);
    var host: [2 * 32]f32 = undefined;
    for (&host) |*v| v.* = (rng.random().float(f32) - 0.5) * 4.0;
    var want = host;
    hadamardInPlace(want[0..32]);
    hadamardInPlace(want[32..64]);
    const shape = [_]c_int{ 2, 32 };
    const arr = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);
    const got_arr = try gpuOp2(mlx.mlx_matmul, arr, hada, s);
    defer _ = mlx.mlx_array_free(got_arr);
    try mlx.check(mlx.mlx_array_eval(got_arr));
    const ptr = mlx.mlx_array_data_float32(got_arr).?;
    for (want, ptr[0..64]) |w, g| try testing.expectApproxEqAbs(w, g, 1e-4);
}

/// Rotate the trailing rd dims of x (last axis) at ONE position, using GPU
/// cos/sin row tensors of shape [rd/2] uploaded from the SAME host tables
/// ropeRow reads (bit-identical trig). inverse = conjugate rotation.
fn gpuRopeTail(x: mlx.mlx_array, rd: usize, cos_row: mlx.mlx_array, sin_row: mlx.mlx_array, inverse: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const ndim = mlx.mlx_array_ndim(x);
    const sh = mlx.mlx_array_shape(x);
    var shape: [8]c_int = undefined;
    for (0..ndim) |i| shape[i] = sh[i];
    const d: c_int = shape[ndim - 1];
    const rdc: c_int = @intCast(rd);
    var starts: [8]c_int = @splat(0);
    var stops: [8]c_int = undefined;
    var strides: [8]c_int = @splat(1);
    for (0..ndim) |i| stops[i] = shape[i];
    // head = x[..., 0 : d-rd]
    stops[ndim - 1] = d - rdc;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, x, &starts, ndim, &stops, ndim, &strides, ndim, s));
    // tail = x[..., d-rd : d] -> pairs [..., rd/2, 2]
    starts[ndim - 1] = d - rdc;
    stops[ndim - 1] = d;
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_slice(&tail, x, &starts, ndim, &stops, ndim, &strides, ndim, s));
    var pshape: [9]c_int = undefined;
    for (0..ndim - 1) |i| pshape[i] = shape[i];
    pshape[ndim - 1] = @divExact(rdc, 2);
    pshape[ndim] = 2;
    var pairs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pairs);
    try mlx.check(mlx.mlx_reshape(&pairs, tail, &pshape, ndim + 1, s));
    // xr = pairs[..., 0], xi = pairs[..., 1]
    var pstart: [9]c_int = @splat(0);
    var pstop: [9]c_int = undefined;
    var pstr: [9]c_int = @splat(1);
    for (0..ndim + 1) |i| pstop[i] = pshape[i];
    pstop[ndim] = 1;
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_slice(&xr, pairs, &pstart, ndim + 1, &pstop, ndim + 1, &pstr, ndim + 1, s));
    pstart[ndim] = 1;
    pstop[ndim] = 2;
    var xi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xi);
    try mlx.check(mlx.mlx_slice(&xi, pairs, &pstart, ndim + 1, &pstop, ndim + 1, &pstr, ndim + 1, s));
    // broadcast cos/sin as [rd/2, 1]
    const cshape = [_]c_int{ @divExact(rdc, 2), 1 };
    var cosb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cosb);
    try mlx.check(mlx.mlx_reshape(&cosb, cos_row, &cshape, 2, s));
    var sinb0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinb0);
    try mlx.check(mlx.mlx_reshape(&sinb0, sin_row, &cshape, 2, s));
    var sinb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinb);
    if (inverse) {
        const neg1 = mlx.mlx_array_new_float(-1.0);
        defer _ = mlx.mlx_array_free(neg1);
        try mlx.check(mlx.mlx_multiply(&sinb, sinb0, neg1, s));
    } else {
        try mlx.check(mlx.mlx_astype(&sinb, sinb0, .float32, s));
    }
    // yr = xr*cos - xi*sin; yi = xr*sin + xi*cos
    const a1 = try gpuOp2(mlx.mlx_multiply, xr, cosb, s);
    defer _ = mlx.mlx_array_free(a1);
    const a2 = try gpuOp2(mlx.mlx_multiply, xi, sinb, s);
    defer _ = mlx.mlx_array_free(a2);
    const yr = try gpuOp2(mlx.mlx_subtract, a1, a2, s);
    defer _ = mlx.mlx_array_free(yr);
    const b1 = try gpuOp2(mlx.mlx_multiply, xr, sinb, s);
    defer _ = mlx.mlx_array_free(b1);
    const b2 = try gpuOp2(mlx.mlx_multiply, xi, cosb, s);
    defer _ = mlx.mlx_array_free(b2);
    const yi = try gpuOp2(mlx.mlx_add, b1, b2, s);
    defer _ = mlx.mlx_array_free(yi);
    var rot_pairs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot_pairs);
    {
        const parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        _ = mlx.mlx_vector_array_append_value(parts, yr);
        _ = mlx.mlx_vector_array_append_value(parts, yi);
        try mlx.check(mlx.mlx_concatenate_axis(&rot_pairs, parts, @intCast(ndim), s));
    }
    var tshape: [8]c_int = undefined;
    for (0..ndim) |i| tshape[i] = shape[i];
    tshape[ndim - 1] = rdc;
    var rot_tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot_tail);
    try mlx.check(mlx.mlx_reshape(&rot_tail, rot_pairs, &tshape, ndim, s));
    var out = mlx.mlx_array_new();
    const parts2 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts2);
    _ = mlx.mlx_vector_array_append_value(parts2, head);
    _ = mlx.mlx_vector_array_append_value(parts2, rot_tail);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts2, @intCast(ndim - 1), s));
    return out;
}

test "dsv4: gpuRopeTail matches the host ropeRow" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    const rd: usize = 16;
    const fr = try precomputeFreqs(alloc, rd, 40, 64, 160000.0, 16, 32, 1);
    defer alloc.free(fr.cos);
    defer alloc.free(fr.sin);
    var rng = std.Random.DefaultPrng.init(4);
    for ([_]usize{ 0, 3, 17, 39 }) |pos| {
        for ([_]bool{ false, true }) |inverse| {
            // rows [3 heads, 48 dims] — rope on the last 16
            var host: [3 * 48]f32 = undefined;
            for (&host) |*v| v.* = (rng.random().float(f32) - 0.5) * 4.0;
            var want = host;
            for (0..3) |hh| ropeRow(want[hh * 48 ..][0..48][48 - rd ..], &fr, pos, inverse);
            const shape = [_]c_int{ 3, 48 };
            const arr = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
            defer _ = mlx.mlx_array_free(arr);
            const half = rd / 2;
            const cshape = [_]c_int{@intCast(half)};
            const cos_row = mlx.mlx_array_new_data(fr.cos[pos * half ..].ptr, &cshape, 1, .float32);
            defer _ = mlx.mlx_array_free(cos_row);
            const sin_row = mlx.mlx_array_new_data(fr.sin[pos * half ..].ptr, &cshape, 1, .float32);
            defer _ = mlx.mlx_array_free(sin_row);
            const got_arr = try gpuRopeTail(arr, rd, cos_row, sin_row, inverse, s);
            defer _ = mlx.mlx_array_free(got_arr);
            try mlx.check(mlx.mlx_array_eval(got_arr));
            const ptr = mlx.mlx_array_data_float32(got_arr).?;
            for (want, ptr[0 .. 3 * 48], 0..) |w, g, i| {
                if (@abs(w - g) > 1e-6) {
                    std.debug.print("rope mismatch pos={d} inv={} i={d}: {e} vs {e}\n", .{ pos, inverse, i, w, g });
                    try testing.expect(false);
                }
            }
        }
    }
}

test "dsv4: fused emission kernel matches the composed emission graph (GPU)" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    const hada = try buildHadamardF32(alloc, 128, s);
    defer _ = mlx.mlx_array_free(hada);
    const Case = struct { d: usize, rd: usize, r: usize, coff: usize, rotate: bool, base: usize, C: usize, col_off: usize, cin_w: usize, seed_neg_inf: bool };
    const cases = [_]Case{
        // real attn-compressor decode boundary (first window: -inf prev seed)
        .{ .d = 512, .rd = 64, .r = 4, .coff = 2, .rotate = false, .base = 3, .C = 1, .col_off = 0, .cin_w = 2560, .seed_neg_inf = true },
        // real indexer decode boundary: Hadamard + fp4 sim, cols after attn's
        .{ .d = 128, .rd = 64, .r = 4, .coff = 2, .rotate = true, .base = 3, .C = 1, .col_off = 2048, .cin_w = 2560, .seed_neg_inf = false },
        // non-overlap chunk with ring remainder (mini ratio-16 shape)
        .{ .d = 96, .rd = 32, .r = 16, .coff = 1, .rotate = false, .base = 8, .C = 40, .col_off = 0, .cin_w = 192, .seed_neg_inf = false },
        // overlap chunk crossing mid-window (ape wrap, multi-window)
        .{ .d = 128, .rd = 64, .r = 4, .coff = 2, .rotate = false, .base = 6, .C = 10, .col_off = 0, .cin_w = 512, .seed_neg_inf = false },
    };
    var rng = std.Random.DefaultPrng.init(11);
    for (cases) |tc| {
        const cd = tc.coff * tc.d;
        const half = tc.rd / 2;
        const npos = tc.base + tc.C + tc.r;
        var m: Dsv4Model = undefined;
        m.s = s;
        m.eps = 1e-6;
        m.rd = tc.rd;
        m.hada_g = hada;

        var c: HostComp = undefined;
        c.head_dim = tc.d;
        c.coff = tc.coff;
        const norm_h = try alloc.alloc(f32, tc.d);
        defer alloc.free(norm_h);
        for (norm_h) |*v| v.* = 0.5 + rng.random().float(f32);
        const nshape = [_]c_int{@intCast(tc.d)};
        c.norm_g = uploadF32(norm_h, &nshape);
        defer _ = mlx.mlx_array_free(c.norm_g);
        const ape_h = try alloc.alloc(f32, tc.r * cd);
        defer alloc.free(ape_h);
        for (ape_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        const ashape = [_]c_int{ @intCast(tc.r), @intCast(cd) };
        c.ape_g = uploadF32(ape_h, &ashape);
        defer _ = mlx.mlx_array_free(c.ape_g);

        var cstate: CompDecState = undefined;
        const ring_rows = (if (tc.coff == 2) 2 * tc.r else tc.r);
        const ring = try alloc.alloc(f32, ring_rows * cd);
        defer alloc.free(ring);
        const ring_sc = try alloc.alloc(f32, ring_rows * cd);
        defer alloc.free(ring_sc);
        for (ring) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        for (ring_sc) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        if (tc.seed_neg_inf) {
            // first window: previous-window half seeded 0 / -inf (host rule)
            for (ring[0 .. tc.r * cd]) |*v| v.* = 0;
            for (ring_sc[0 .. tc.r * cd]) |*v| v.* = -std.math.inf(f32);
        }
        cstate.kv_pend = ring;
        cstate.sc_pend = ring_sc;
        cstate.width = cd;

        const cin_h = try alloc.alloc(f32, tc.C * tc.cin_w);
        defer alloc.free(cin_h);
        for (cin_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        const cshape = [_]c_int{ @intCast(tc.C), @intCast(tc.cin_w) };
        const cin = uploadF32(cin_h, &cshape);
        defer _ = mlx.mlx_array_free(cin);

        var fr: Freqs = undefined;
        fr.half = half;
        const cos_h = try alloc.alloc(f32, npos * half);
        defer alloc.free(cos_h);
        const sin_h = try alloc.alloc(f32, npos * half);
        defer alloc.free(sin_h);
        for (cos_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        for (sin_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        fr.cos = cos_h;
        fr.sin = sin_h;

        const n_win = (tc.base + tc.C) / tc.r - tc.base / tc.r;
        try testing.expect(n_win > 0);

        emitKernelSetForTest(false);
        var mir_ref = GpuRows.init(tc.d);
        defer mir_ref.deinit();
        try emitWindowsGpu(&m, &c, &cstate, &mir_ref, cin, tc.col_off, tc.base, tc.C, tc.r, tc.rotate, &fr, alloc);
        emitKernelSetForTest(true);
        const hits0 = emit_kernel_hits;
        var mir_k = GpuRows.init(tc.d);
        defer mir_k.deinit();
        try emitWindowsGpu(&m, &c, &cstate, &mir_k, cin, tc.col_off, tc.base, tc.C, tc.r, tc.rotate, &fr, alloc);
        emitKernelSetForTest(null);
        try testing.expect(emit_kernel_hits > hits0); // ENGAGED, not silently declined
        try testing.expectEqual(n_win, mir_ref.used);
        try testing.expectEqual(n_win, mir_k.used);

        const ref_rows = try mir_ref.sliceRows(0, n_win, s);
        defer _ = mlx.mlx_array_free(ref_rows);
        const k_rows = try mir_k.sliceRows(0, n_win, s);
        defer _ = mlx.mlx_array_free(k_rows);
        const want = try toHostF32(alloc, ref_rows, n_win * tc.d, s);
        defer alloc.free(want);
        const got = try toHostF32(alloc, k_rows, n_win * tc.d, s);
        defer alloc.free(got);
        for (want, got, 0..) |w, g, i| {
            // approx-eq is NaN-safe: a NaN diff fails (finiteness rule)
            if (!(@abs(w - g) <= 2e-3)) {
                std.debug.print("emit-kernel mismatch d={d} rotate={} i={d}: composed={e} kernel={e}\n", .{ tc.d, tc.rotate, i, w, g });
                try testing.expect(false);
            }
        }
    }
}

/// Acquit a composed-vs-kernel fp4 mismatch ONLY when the pre-quant value
/// sits at the rounding midpoint between the two returned grid points: the
/// two arms sum the hadamard matmul in different orders, and a value a few
/// ulps from the midpoint legitimately rounds to ADJACENT fp4 points on
/// another machine (M5 measured composed -0.5 vs kernel -0.75, PR #223).
/// Anything not at a midpoint is a real mismatch and still fails.
fn fp4MidpointAcquits(composed: f32, kernel: f32, prequant: f32) bool {
    const gap = @abs(composed - kernel);
    if (gap == 0) return false;
    const mid = (composed + kernel) * 0.5;
    return @abs(prequant - mid) <= 0.05 * gap;
}

test "dsv4: fp4 midpoint acquittal accepts a rounding near-tie, rejects a real mismatch" {
    // The M5 reading: adjacent grid points, pre-quant at the midpoint.
    try testing.expect(fp4MidpointAcquits(-0.5, -0.75, -0.625));
    // A few reduction-order ulps off the midpoint still acquits.
    try testing.expect(fp4MidpointAcquits(-0.5, -0.75, -0.62501));
    // A pre-quant value that rounds cleanly to one grid point is a bug.
    try testing.expect(!fp4MidpointAcquits(-0.5, -0.75, -0.51));
    try testing.expect(!fp4MidpointAcquits(-0.5, -0.75, -0.72));
    // No gap: nothing to acquit.
    try testing.expect(!fp4MidpointAcquits(1.0, 1.0, 1.0));
}

test "dsv4: fused decode-chain kernel matches the composed q/kv/idx/o chains (GPU)" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    const hada = try buildHadamardF32(alloc, 128, s);
    defer _ = mlx.mlx_array_free(hada);
    // post: 0 = rope only, 1 = fp8 head sim + raw roped tail, 2 = hadamard+fp4
    const Case = struct { h: usize, d: usize, rd: usize, rms: bool, ones_w: bool, post: u8, inv: bool };
    const cases = [_]Case{
        .{ .h = 64, .d = 512, .rd = 64, .rms = true, .ones_w = true, .post = 0, .inv = false }, // q chain
        .{ .h = 1, .d = 512, .rd = 64, .rms = true, .ones_w = false, .post = 1, .inv = false }, // kv chain
        .{ .h = 64, .d = 128, .rd = 64, .rms = false, .ones_w = false, .post = 2, .inv = false }, // idx q chain
        .{ .h = 64, .d = 512, .rd = 64, .rms = false, .ones_w = false, .post = 0, .inv = true }, // o rope⁻¹
        .{ .h = 4, .d = 96, .rd = 32, .rms = true, .ones_w = false, .post = 1, .inv = false }, // mini kv shape
    };
    var rng = std.Random.DefaultPrng.init(23);
    for (cases) |tc| {
        var m: Dsv4Model = undefined;
        m.s = s;
        m.eps = 1e-6;
        const half = tc.rd / 2;
        const xs = try alloc.alloc(f32, tc.h * tc.d);
        defer alloc.free(xs);
        for (xs) |*v| v.* = (rng.random().float(f32) - 0.5) * 4.0;
        const xshape = [_]c_int{ 1, @intCast(tc.h * tc.d) };
        const x = uploadF32(xs, &xshape);
        defer _ = mlx.mlx_array_free(x);
        const ws = try alloc.alloc(f32, tc.d);
        defer alloc.free(ws);
        for (ws) |*v| v.* = if (tc.ones_w) 1.0 else 0.5 + rng.random().float(f32);
        const wshape = [_]c_int{@intCast(tc.d)};
        const w = uploadF32(ws, &wshape);
        defer _ = mlx.mlx_array_free(w);
        const cos_h = try alloc.alloc(f32, half);
        defer alloc.free(cos_h);
        const sin_h = try alloc.alloc(f32, half);
        defer alloc.free(sin_h);
        for (cos_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        for (sin_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        const rshape = [_]c_int{@intCast(half)};
        const rr = RopeRows{ .cos = uploadF32(cos_h, &rshape), .sin = uploadF32(sin_h, &rshape) };
        defer _ = mlx.mlx_array_free(rr.cos);
        defer _ = mlx.mlx_array_free(rr.sin);

        // composed reference: the exact attentionDecodeGpu op chain
        const hshape = [_]c_int{ @intCast(tc.h), @intCast(tc.d) };
        const xr = try gpuReshape(x, &hshape, s);
        defer _ = mlx.mlx_array_free(xr);
        var stage = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stage);
        if (tc.rms) {
            const n = try gpuRms(xr, w, m.eps, s);
            defer _ = mlx.mlx_array_free(n);
            try mlx.check(mlx.mlx_array_set(&stage, n));
        } else {
            try mlx.check(mlx.mlx_array_set(&stage, xr));
        }
        const roped = try gpuRopeTail(stage, tc.rd, rr.cos, rr.sin, tc.inv, s);
        defer _ = mlx.mlx_array_free(roped);
        var want_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want_arr);
        // Pre-quant hadamard product (post==2 only): the fp4 midpoint
        // acquittal below needs it to tell a rounding near-tie from a bug.
        var prequant: ?[]f32 = null;
        defer if (prequant) |p| alloc.free(p);
        switch (tc.post) {
            0 => try mlx.check(mlx.mlx_array_set(&want_arr, roped)),
            1 => {
                const head = try gpuSliceCols(roped, tc.h, 0, tc.d - tc.rd, s);
                defer _ = mlx.mlx_array_free(head);
                const hs = try gpuFp8Sim(head, s);
                defer _ = mlx.mlx_array_free(hs);
                const tail = try gpuSliceCols(roped, tc.h, tc.d - tc.rd, tc.d, s);
                defer _ = mlx.mlx_array_free(tail);
                const fin = try gpuConcat2(hs, tail, 1, s);
                defer _ = mlx.mlx_array_free(fin);
                try mlx.check(mlx.mlx_array_set(&want_arr, fin));
            },
            else => {
                const had = try gpuOp2(mlx.mlx_matmul, roped, hada, s);
                defer _ = mlx.mlx_array_free(had);
                prequant = try toHostF32(alloc, had, tc.h * tc.d, s);
                const simd = try gpuFp4Sim(had, s);
                defer _ = mlx.mlx_array_free(simd);
                try mlx.check(mlx.mlx_array_set(&want_arr, simd));
            },
        }

        decChainSetForTest(true);
        const hits0 = dec_chain_hits;
        m.hada_g = if (tc.post == 2) hada else null;
        const got_arr = (try decChainKernel(&m, x, tc.h, tc.d, tc.rd, if (tc.rms) w else null, tc.post, tc.inv, &rr)) orelse {
            decChainSetForTest(null);
            std.debug.print("dec-chain DECLINED h={d} d={d} post={d}\n", .{ tc.h, tc.d, tc.post });
            try testing.expect(false);
            unreachable;
        };
        defer _ = mlx.mlx_array_free(got_arr);
        decChainSetForTest(null);
        try testing.expect(dec_chain_hits > hits0);

        const want = try toHostF32(alloc, want_arr, tc.h * tc.d, s);
        defer alloc.free(want);
        const got = try toHostF32(alloc, got_arr, tc.h * tc.d, s);
        defer alloc.free(got);
        for (want, got, 0..) |wv, gv, i| {
            if (!(@abs(wv - gv) <= 2e-3)) {
                if (prequant) |pq| {
                    if (fp4MidpointAcquits(wv, gv, pq[i])) continue;
                }
                std.debug.print("dec-chain mismatch h={d} d={d} post={d} i={d}: composed={e} kernel={e}\n", .{ tc.h, tc.d, tc.post, i, wv, gv });
                try testing.expect(false);
            }
        }
    }
}

test "dsv4: fused MoE gate+up kernel is no worse than the composed gathers (GPU)" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    const E: usize = 8;
    const N: usize = 32;
    const K: usize = 128;
    const k: usize = 6;
    const Case = struct { bits: u32, gs: u32, limit: f32 };
    const cases = [_]Case{
        .{ .bits = 2, .gs = 64, .limit = 10.0 }, // real trunk experts
        .{ .bits = 8, .gs = 32, .limit = 1.0 }, // mini shape, clip engaged
        .{ .bits = 4, .gs = 32, .limit = 10.0 },
    };
    var rng = std.Random.DefaultPrng.init(31);
    for (cases) |tc| {
        var m: Dsv4Model = undefined;
        m.s = s;
        m.swiglu_limit = tc.limit;

        var banks: [2]Q = undefined;
        var deq_host: [2][]f32 = .{ &.{}, &.{} };
        defer for (deq_host) |dh| alloc.free(dh);
        for (0..2) |bi| {
            const wf = try alloc.alloc(f32, E * N * K);
            defer alloc.free(wf);
            for (wf) |*v| v.* = (rng.random().float(f32) - 0.5) * 0.6;
            const wshape = [_]c_int{ @intCast(E), @intCast(N), @intCast(K) };
            const w32 = uploadF32(wf, &wshape);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
            var qv = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qv);
            const empty = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(empty);
            try mlx.check(mlx.mlx_quantize(&qv, wb, mlx.mlx_optional_int.some(@intCast(tc.gs)), mlx.mlx_optional_int.some(@intCast(tc.bits)), "affine", empty, s));
            var wq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&wq, qv, 0));
            var sc = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&sc, qv, 1));
            var bs = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&bs, qv, 2));
            banks[bi] = .{ .w = wq, .s = sc, .b = bs, .qp = .{ .bits = tc.bits, .group_size = tc.gs } };
            // dequant ground-truth operand (widened to f32 for the host read)
            var deq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq);
            try mlx.check(mlx.mlx_dequantize(&deq, wq, sc, bs, mlx.mlx_optional_int.some(@intCast(tc.gs)), mlx.mlx_optional_int.some(@intCast(tc.bits)), "affine", empty, mlx.mlx_optional_dtype{}, s));
            var deq32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq32);
            try mlx.check(mlx.mlx_astype(&deq32, deq, .float32, s));
            deq_host[bi] = try toHostF32(alloc, deq32, E * N * K, s);
        }
        defer for (&banks) |*b| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.s);
            _ = mlx.mlx_array_free(b.b);
        };

        const xf = try alloc.alloc(f32, K);
        defer alloc.free(xf);
        for (xf) |*v| v.* = (rng.random().float(f32) - 0.5) * 3.0;
        const xshape = [_]c_int{ 1, 1, 1, @intCast(K) };
        const x32 = uploadF32(xf, &xshape);
        defer _ = mlx.mlx_array_free(x32);
        var xe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xe);
        try mlx.check(mlx.mlx_astype(&xe, x32, .bfloat16, s));
        // x as the kernel sees it (bf16-rounded), for the f64 ground truth
        const xh = try toHostF32(alloc, xe, K, s);
        defer alloc.free(xh);

        const inds = [_]i32{ 0, 3, 3, 7, 1, 0 }; // duplicates included
        const ishape = [_]c_int{ 1, @intCast(k) };
        const ind = mlx.mlx_array_new_data(&inds, &ishape, 2, .int32);
        defer _ = mlx.mlx_array_free(ind);

        // composed reference chain (the decode branch verbatim)
        const gate_arr = try gatherQmmE(&banks[0], xe, ind, s);
        defer _ = mlx.mlx_array_free(gate_arr);
        const up_arr = try gatherQmmE(&banks[1], xe, ind, s);
        defer _ = mlx.mlx_array_free(up_arr);
        const act_c = try clippedSwigluG(gate_arr, up_arr, tc.limit, s);
        defer _ = mlx.mlx_array_free(act_c);
        // the fused arm must reproduce the composed act SHAPE exactly — the
        // down gather consumes it as its x operand
        try mlx.check(mlx.mlx_array_eval(act_c));
        {
            const nd = mlx.mlx_array_ndim(act_c);
            const sh = mlx.mlx_array_shape(act_c);
            const want_sh = [_]c_int{ 1, @intCast(k), 1, @intCast(N) };
            try testing.expectEqual(@as(usize, 4), nd);
            for (0..4) |i| try testing.expectEqual(want_sh[i], sh[i]);
        }

        moeGateUpSetForTest(true);
        const hits0 = moe_gateup_hits;
        const act_f = (try moeGateUpFused(&m, xe, &banks[0], &banks[1], ind, k)) orelse {
            moeGateUpSetForTest(null);
            std.debug.print("moe gate+up DECLINED bits={d} gs={d}\n", .{ tc.bits, tc.gs });
            try testing.expect(false);
            unreachable;
        };
        defer _ = mlx.mlx_array_free(act_f);
        moeGateUpSetForTest(null);
        try testing.expect(moe_gateup_hits > hits0);

        const got_c = try toHostF32(alloc, act_c, k * N, s);
        defer alloc.free(got_c);
        const got_f = try toHostF32(alloc, act_f, k * N, s);
        defer alloc.free(got_f);

        // f64 ground truth from the dequantized banks + bf16-rounded x
        var err_c: f64 = 0;
        var err_f: f64 = 0;
        for (0..k) |e| {
            const eid: usize = @intCast(inds[e]);
            for (0..N) |n| {
                var g: f64 = 0;
                var u: f64 = 0;
                for (0..K) |j| {
                    g += @as(f64, xh[j]) * deq_host[0][(eid * N + n) * K + j];
                    u += @as(f64, xh[j]) * deq_host[1][(eid * N + n) * K + j];
                }
                const gc = @min(g, @as(f64, tc.limit));
                const uc = std.math.clamp(u, -@as(f64, tc.limit), @as(f64, tc.limit));
                const truth = (gc / (1.0 + @exp(-gc))) * uc;
                const i = e * N + n;
                try testing.expect(std.math.isFinite(got_c[i]));
                try testing.expect(std.math.isFinite(got_f[i]));
                err_c = @max(err_c, @abs(@as(f64, got_c[i]) - truth));
                err_f = @max(err_f, @abs(@as(f64, got_f[i]) - truth));
            }
        }
        std.debug.print("dsv4 moe gate+up bits={d}: err composed={e:.3} fused={e:.3}\n", .{ tc.bits, err_c, err_f });
        // no-worse-than-reference (house rule: never kernel-vs-kernel exact)
        try testing.expect(err_f <= err_c * 1.25 + 1e-3);
    }
}

test "dsv4: gs-128 expert pack resolves per-weight and qmm matches the dequant reference" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    // The imatrix mirror ships trunk experts 2b/3b at gs 128 over a config
    // whose top-level quantization block still says gs 64 — the loader owes
    // the exact per-weight solve (getQ threads in_dim into
    // computeQuantParams → affineParamsFromGeometry), and the qmm must serve
    // the g128 pack through the resolved params. This is the "no Zig changes
    // needed" proof for the g128 rebuild.
    var cfg = model.ModelConfig{};
    cfg.quant_bits = 8;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const K: usize = 256; // in_dim: 2 groups of 128
    const N: usize = 16;
    var rng = std.Random.DefaultPrng.init(41);
    for ([_]u32{ 2, 3 }) |bits| {
        const wf = try alloc.alloc(f32, N * K);
        defer alloc.free(wf);
        for (wf) |*v| v.* = (rng.random().float(f32) - 0.5) * 0.8;
        const wshape = [_]c_int{ @intCast(N), @intCast(K) };
        const w32 = uploadF32(wf, &wshape);
        defer _ = mlx.mlx_array_free(w32);
        var wb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wb);
        try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
        var qv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qv);
        const empty = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(empty);
        try mlx.check(mlx.mlx_quantize(&qv, wb, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(@intCast(bits)), "affine", empty, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        try mlx.check(mlx.mlx_vector_array_get(&wq, qv, 0));
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_vector_array_get(&sc, qv, 1));
        var bs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bs);
        try mlx.check(mlx.mlx_vector_array_get(&bs, qv, 2));

        const qp = transformer.computeQuantParams(&cfg, wq, sc, @intCast(K));
        try testing.expectEqual(bits, qp.bits);
        try testing.expectEqual(@as(u32, 128), qp.group_size);

        const q = Q{ .w = wq, .s = sc, .b = bs, .qp = qp };
        const xf = try alloc.alloc(f32, K);
        defer alloc.free(xf);
        for (xf) |*v| v.* = (rng.random().float(f32) - 0.5) * 2.0;
        const xshape = [_]c_int{ 1, @intCast(K) };
        const x32 = uploadF32(xf, &xshape);
        defer _ = mlx.mlx_array_free(x32);
        const y = try gpuQmmB(&q, x32, s);
        defer _ = mlx.mlx_array_free(y);
        const got = try toHostF32(alloc, y, N, s);
        defer alloc.free(got);

        // f32 dequant reference (no bf16 re-round — the qmm's in-kernel
        // dequant computes in float) against the bf16-rounded x it sees.
        var deq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deq);
        try mlx.check(mlx.mlx_dequantize(&deq, wq, sc, bs, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(@intCast(bits)), "affine", empty, .{ .value = .float32, .has_value = true }, s));
        const dh = try toHostF32(alloc, deq, N * K, s);
        defer alloc.free(dh);
        var xb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xb);
        try mlx.check(mlx.mlx_astype(&xb, x32, .bfloat16, s));
        const xh = try toHostF32(alloc, xb, K, s);
        defer alloc.free(xh);
        for (0..N) |n| {
            var t: f64 = 0;
            for (0..K) |j| t += @as(f64, xh[j]) * dh[n * K + j];
            try testing.expect(std.math.isFinite(got[n]));
            try testing.expectApproxEqAbs(@as(f32, @floatCast(t)), got[n], 0.05);
        }
    }
}

test "dsv4: boundary prefetch drains older pending rows a token early (DSV4_MINI)" {
    // At the end of the step BEFORE a window-closing token, older pending
    // compressor rows (async-scheduled ≥1 token ago) must drain so the
    // boundary token's blocking drain shrinks to the latest token's rows.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    if (mlx.noGpuBackend()) return;
    if (!drainPrefetchEnabled()) return; // fallback-arm runs assert nothing here
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    const dw = try loadDsv4Weights(allocator, &cfg, &weights);
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    _ = unsetenv("MLX_SERVE_DSV4_DSPARK");
    var mdl = try initModel(allocator, &cfg, dw, s);
    defer mdl.deinit();

    var rng = std.Random.DefaultPrng.init(43);
    var ids: [20]u32 = undefined;
    for (&ids) |*v| v.* = rng.random().uintLessThan(u32, @intCast(mdl.vocab));

    var st = try initDecodeState(&mdl, allocator);
    defer deinitDecodeState(&st);
    try testing.expect(lazyDecodeReady(&mdl, &st));
    {
        const p = try prefillIntoState(&mdl, allocator, &st, ids[0..17]);
        allocator.free(p);
    }
    // mini ratios [0,4,16,4]: pos 19 closes a ratio-4 window. Steps at pos
    // 17 and 18 accumulate pending rows; the pos-18 step must prefetch-drain
    // pos-17's rows because pos 19 is a boundary.
    for (17..19) |n| {
        const idv: i32 = @intCast(ids[n]);
        const ishape = [_]c_int{ 1, 1 };
        const id_arr = mlx.mlx_array_new_data(&idv, &ishape, 2, .int32);
        defer _ = mlx.mlx_array_free(id_arr);
        const lazy_g = try decodeStepLazy(&mdl, allocator, &st, id_arr);
        defer _ = mlx.mlx_array_free(lazy_g);
        const logits = try toHostF32(allocator, lazy_g, @intCast(mdl.vocab), s);
        defer allocator.free(logits);
        for (logits) |v| try testing.expect(std.math.isFinite(v));
    }
    try testing.expect(st.pending.items.len > 0);
    for (st.pending.items) |p| {
        if (p.pos != 18) {
            std.debug.print("prefetch missed: pending row li={d} pos={d} (want only pos 18)\n", .{ p.li, p.pos });
            try testing.expect(false);
        }
    }
    try drainPending(&mdl, &st);
    try testing.expectEqual(@as(usize, 0), st.pending.items.len);
}

test "dsv4: fused sink-softmax kernel matches the composed scale/concat/softmax/slice (GPU)" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const alloc = testing.allocator;
    const Case = struct { nh: usize, tk: usize };
    const cases = [_]Case{
        .{ .nh = 64, .tk = 640 }, // real steady state (window 128 + top-512)
        .{ .nh = 64, .tk = 128 }, // ratio-128 layer / pure sliding
        .{ .nh = 4, .tk = 7 }, // mini early-ramp shape
    };
    var rng = std.Random.DefaultPrng.init(53);
    for (cases) |tc| {
        var m: Dsv4Model = undefined;
        m.s = s;
        const hd: usize = 512;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))));
        const sc_h = try alloc.alloc(f32, tc.nh * tc.tk);
        defer alloc.free(sc_h);
        for (sc_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 40.0;
        const sshape = [_]c_int{ @intCast(tc.nh), @intCast(tc.tk) };
        const scores0 = uploadF32(sc_h, &sshape);
        defer _ = mlx.mlx_array_free(scores0);
        const sink_h = try alloc.alloc(f32, tc.nh);
        defer alloc.free(sink_h);
        for (sink_h) |*v| v.* = (rng.random().float(f32) - 0.5) * 6.0;
        const kshape = [_]c_int{ @intCast(tc.nh), 1 };
        const sink_g = uploadF32(sink_h, &kshape);
        defer _ = mlx.mlx_array_free(sink_g);

        // composed reference: the attentionDecodeGpu chain verbatim
        const scale_arr = mlx.mlx_array_new_float(scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        const scaled = try gpuOp2(mlx.mlx_multiply, scores0, scale_arr, s);
        defer _ = mlx.mlx_array_free(scaled);
        const with_sink = try gpuConcat2(scaled, sink_g, 1, s);
        defer _ = mlx.mlx_array_free(with_sink);
        var probs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(probs);
        try mlx.check(mlx.mlx_softmax_axis(&probs, with_sink, -1, true, s));
        const want_arr = try gpuSliceCols(probs, tc.nh, 0, tc.tk, s);
        defer _ = mlx.mlx_array_free(want_arr);

        sinkSoftmaxSetForTest(true);
        const hits0 = sink_softmax_hits;
        const got_arr = (try sinkSoftmaxKernel(&m, scores0, sink_g, tc.nh, tc.tk, scale)) orelse {
            sinkSoftmaxSetForTest(null);
            std.debug.print("sink-softmax DECLINED nh={d} tk={d}\n", .{ tc.nh, tc.tk });
            try testing.expect(false);
            unreachable;
        };
        defer _ = mlx.mlx_array_free(got_arr);
        sinkSoftmaxSetForTest(null);
        try testing.expect(sink_softmax_hits > hits0);

        const want = try toHostF32(alloc, want_arr, tc.nh * tc.tk, s);
        defer alloc.free(want);
        const got = try toHostF32(alloc, got_arr, tc.nh * tc.tk, s);
        defer alloc.free(got);
        for (want, got, 0..) |w, g, i| {
            if (!(@abs(w - g) <= 1e-5)) {
                std.debug.print("sink-softmax mismatch nh={d} tk={d} i={d}: composed={e} kernel={e}\n", .{ tc.nh, tc.tk, i, w, g });
                try testing.expect(false);
            }
        }
    }
}

test "dsv4: comp_in decode requant rides --decode-attn-quant (DSV4_MINI)" {
    // The compressor-input projection (dense bf16, ~610 MB/token on the real
    // trunk) gets an int8-g32 side copy served at decode/verify widths when
    // the user-facing --decode-attn-quant flag is on; big prefill chunks keep
    // the dense weight (quality anchor). LOSSY by design — this gate pins
    // engagement + closeness, the real-checkpoint characterization decides
    // the shipped default.
    const path_z = std.c.getenv("DSV4_MINI") orelse return;
    if (mlx.noGpuBackend()) return;
    const path = std.mem.span(path_z);
    const allocator = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{path});
    defer allocator.free(cfg_path);
    const file = try std.Io.Dir.openFileAbsolute(io, cfg_path, .{});
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    const cfg_json = try rs.interface.allocRemaining(allocator, .limited(1 << 20));
    file.close(io);
    defer allocator.free(cfg_json);
    const cfg = try model.parseConfigFromJson(allocator, cfg_json);
    const shard_path = try std.fmt.allocPrint(allocator, "{s}/model-mini.safetensors", .{path});
    defer allocator.free(shard_path);
    var weights = try model.loadWeightsSingleFile(allocator, shard_path);
    defer weights.deinit();

    var rng = std.Random.DefaultPrng.init(61);
    var ids: [26]u32 = undefined;
    const vocab_guess: u32 = 64;
    for (&ids) |*v| v.* = rng.random().uintLessThan(u32, vocab_guess);

    var dense_logits: [9][]f32 = undefined;
    var n_steps: usize = 0;
    defer for (dense_logits[0..n_steps]) |l| allocator.free(l);

    defer {
        transformer.decode_attn_quant_override = null;
    }
    for ([_]bool{ false, true }) |quant_arm| {
        transformer.decode_attn_quant_override = quant_arm;
        const dw = try loadDsv4Weights(allocator, &cfg, &weights);
        const s = mlx.gpuStream();
        defer _ = mlx.mlx_stream_free(s);
        _ = unsetenv("MLX_SERVE_DSV4_DSPARK");
        var mdl = try initModel(allocator, &cfg, dw, s);
        defer mdl.deinit();
        var st = try initDecodeState(&mdl, allocator);
        defer deinitDecodeState(&st);
        {
            const p = try prefillIntoState(&mdl, allocator, &st, ids[0..17]);
            allocator.free(p);
        }
        const hits0 = comp_in_q_hits;
        var agree: usize = 0;
        var step: usize = 0;
        for (17..ids.len) |n| {
            const dec = try decodeStep(&mdl, allocator, &st, ids[n]);
            if (!quant_arm) {
                dense_logits[step] = dec;
                n_steps += 1;
            } else {
                defer allocator.free(dec);
                const ref = dense_logits[step];
                var dot: f64 = 0;
                var na: f64 = 0;
                var nb: f64 = 0;
                var am_a: usize = 0;
                var am_b: usize = 0;
                for (ref, dec, 0..) |a2, b2, i| {
                    try testing.expect(std.math.isFinite(b2));
                    dot += @as(f64, a2) * b2;
                    na += @as(f64, a2) * a2;
                    nb += @as(f64, b2) * b2;
                    if (a2 > ref[am_a]) am_a = i;
                    if (b2 > dec[am_b]) am_b = i;
                }
                const cos = dot / (@sqrt(na) * @sqrt(nb) + 1e-30);
                if (am_a == am_b) agree += 1;
                // int8-g32 on the mini's random weights and tiny groups is
                // relatively coarser than the real geometry — 0.95 catches
                // wiring bugs (a wrong operand reads ~0); the real-checkpoint
                // characterization is the quality gate.
                if (cos < 0.95) {
                    std.debug.print("comp_in requant drifted at n={d}: cos={d:.6}\n", .{ n + 1, cos });
                    try testing.expect(false);
                }
            }
            step += 1;
        }
        if (quant_arm) {
            try testing.expect(comp_in_q_hits > hits0); // ENGAGED under the flag
            try testing.expect(agree * 10 >= step * 8);
            std.debug.print("dsv4 comp_in requant: argmax {d}/{d}, engaged\n", .{ agree, step });
        } else {
            try testing.expectEqual(hits0, comp_in_q_hits); // dense arm never engages
        }
    }
}
