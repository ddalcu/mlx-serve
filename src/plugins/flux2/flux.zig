//! FLUX.2 klein text to image, ported from mflux: the Qwen3 prompt encoder (layers 9/18/27 tapped), the
//! double/single-stream DiT on the shared skeleton, FlowMatchEuler, and the VAE with its encoder (variation and
//! edits). Weights are mflux's: affine-quantized linears (bits and group read off their geometry) and bf16 OHWI
//! convs.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const sse = @import("../../sdk/media/progress.zig");
const lora_mod = @import("../../sdk/weights/lora.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");
const int8_attn = @import("../../sdk/blocks/dit/int8_attn.zig");
const qwen3 = @import("../../sdk/blocks/text_encoders/qwen3.zig");
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");
const modulate_blk = @import("../../sdk/blocks/dit/modulate.zig");
const timestep_blk = @import("../../sdk/blocks/dit/timestep.zig");
const ln_blk = @import("../../sdk/blocks/norms/layer_norm_f32.zig");
const gn_blk = @import("../../sdk/blocks/norms/group_norm.zig");
const conv_blk = @import("../../sdk/blocks/vae/conv.zig");
const resnet_blk = @import("../../sdk/blocks/vae/resnet.zig");
const flow_match = @import("../../sdk/blocks/schedulers/flow_match.zig");
const dit_base = @import("../../sdk/bases/dit.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const Graph = ops.Graph;
const Array = ops.Array;
const Resnet = resnet_blk.Resnet;

/// The VAE's GroupNorm: 32 groups, f32 statistics, bf16 out.
const vae_norm: resnet_blk.Norm = .{ .flat = .{ .eps = 1e-6, .out = .bfloat16 } };

// ── Weight loading ──

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[flux] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingFluxWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}
fn ownOpt(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}
fn fmtKey(a: std.mem.Allocator, comptime f: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(a, f, args);
}

// ── Quantized Linear (affine; bits/group inferred per weight) ──

/// Infer (bits, group_size) of an affine-quantized weight from its packed
/// geometry. `w` packs `in` logical columns into u32s (w_cols = in·bits/32)
/// and scales have s_cols = in/gs, so w_cols·32/s_cols = bits·gs. The product
/// alone is ambiguous ((3,64) vs (6,32)…), so group sizes are tried in
/// convention order: mflux and mlx-community conversions use 64 (the mlx
/// default); 32 and 128 are fallbacks. Lets ONE loader serve the 3/4/6/8-bit
/// klein builds (the 3-bit one is what fits 8 GB iPhones).
const QuantGeometry = struct { bits: u32, group_size: u32 };

fn inferQuantGeometry(w_cols: usize, s_cols: usize) QuantGeometry {
    const fallback: QuantGeometry = .{ .bits = 4, .group_size = 64 };
    if (s_cols == 0 or (w_cols * 32) % s_cols != 0) return fallback;
    const product = w_cols * 32 / s_cols; // bits · group_size
    const valid_bits = [_]u32{ 2, 3, 4, 5, 6, 8 };
    for ([_]u32{ 64, 32, 128 }) |gs| {
        if (product % gs != 0) continue;
        const bits: u32 = @intCast(product / gs);
        for (valid_bits) |vb| {
            if (bits == vb) return .{ .bits = bits, .group_size = gs };
        }
    }
    return fallback;
}

/// Geometry of a loaded (weight, scales) pair → inferred quant params.
fn inferQuantGeometryOf(w: mlx.mlx_array, scales: mlx.mlx_array) QuantGeometry {
    const wsh = mlx.getShape(w);
    const ssh = mlx.getShape(scales);
    if (wsh.len == 0 or ssh.len == 0) return .{ .bits = 4, .group_size = 64 };
    const w_cols: usize = @intCast(wsh[wsh.len - 1]);
    const s_cols: usize = @intCast(ssh[ssh.len - 1]);
    return inferQuantGeometry(w_cols, s_cols);
}

const QLinear = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array, // quant zero-points (NOT additive bias)
    add_bias: ?mlx.mlx_array = null, // optional additive bias (VAE attn)
    bits: u32 = 4,
    group_size: u32 = 64,
    // Runtime adapters (non-owning; the plugin's lora.Stack owns the arrays).
    // Fixed-capacity so attach/forward never allocate.
    lora_refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
    lora_count: u8 = 0,
    /// The int8 copy that REPLACES `w`/`scales`/`biases` (`--w8a8`), built by `toW8a8`.
    w8: ?w8a8.Weight = null,

    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) !QLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);
        var ql: QLinear = .{
            .w = try ownWeight(w, wk),
            .scales = try ownWeight(w, sk),
            .biases = try ownWeight(w, bk),
            .add_bias = ownOpt(w, ak),
        };
        const geo = inferQuantGeometryOf(ql.w, ql.scales);
        ql.bits = geo.bits;
        ql.group_size = geo.group_size;
        return ql;
    }
    fn deinit(self: *QLinear) void {
        if (self.w8) |*q| q.deinit();
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }
    pub fn forward(self: *const QLinear, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        var o = if (self.w8) |*q|
            try g.adopt(try w8a8.forward(try g.astype(x, .bfloat16), q, g.s))
        else
            try g.qmm(x, self.w, self.scales, self.biases, @intCast(self.group_size), @intCast(self.bits));
        if (self.add_bias) |b| o = try g.add(o, b);
        if (self.lora_count > 0) o = try g.add(o, try g.adopt(try lora_mod.deltaSum(x, self.lora_refs[0..self.lora_count], g.s)));
        return parent.keep(o);
    }

    /// Replace the packed weights with their int8 copy when the route takes this shape.
    fn toW8a8(self: *QLinear, s: S) !void {
        self.w8 = try w8a8.takeLinear(&self.w, &self.scales, &self.biases, self.bits, self.group_size, s);
    }

    /// Install the stacked adapter Refs for this linear (from `Stack.findAll`).
    fn setLoraRefs(self: *QLinear, refs: []const lora_mod.Ref) void {
        self.lora_count = @intCast(refs.len);
        @memcpy(self.lora_refs[0..refs.len], refs);
    }

    fn clearLoraRefs(self: *QLinear) void {
        self.lora_count = 0;
    }
};

/// Dequantize a quantized embedding/weight table → bf16 [rows, cols].
fn dequantTable(w_q: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array, bits: u32, gs: u32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_gs = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_dequantize(&o, w_q, scales, biases, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", null_gs, .{ .value = .bfloat16, .has_value = true }, s));
    return o;
}

// ── Config ──

/// Defaults are the klein 4B build; `ditConfigFrom`/`teConfigFrom` overwrite
/// the six fields the 9B differs in, read off the checkpoint itself.
pub const FluxConfig = struct {
    // DiT
    inner_dim: u32 = 3072,
    heads: u32 = 24,
    head_dim: u32 = 128,
    double_layers: u32 = 5,
    single_layers: u32 = 20,
    joint_dim: u32 = 7680,
    in_channels: u32 = 128,
    mlp_ratio: f32 = 3.0,
    rope_theta_dit: f32 = 2000.0,
    // text encoder (Qwen3)
    te_hidden: u32 = 2560,
    te_layers: u32 = 36,
    te_heads: u32 = 32,
    te_kv: u32 = 8,
    te_head_dim: u32 = 128,
    te_inter: u32 = 9728,
    te_rope_theta: f32 = 1_000_000.0,
    te_rms_eps: f32 = 1e-6,
    te_vocab: u32 = 151936,
};

// ── Geometry, derived from the checkpoint ──
//
// FLUX.2 klein ships as 4B and 9B. Per mflux's own model configs the ENTIRE
// delta is six numbers — 5→8 double blocks, 20→24 single, 24→32 heads, joint
// 7680→12288, encoder hidden 2560→4096, encoder MLP 9728→12288 — while
// head_dim 128, in_channels 128, mlp_ratio 3, the RoPE theta, the 36 encoder
// layers and the 9/18/27 taps are shared.
//
// None of it is in the checkpoint's json: the mlx-community 9B repo carries no
// config.json at all, and the 4B's records only its quantization. So it comes
// from the WEIGHT SHAPES, which is also the only source that cannot disagree
// with the tensors about to be loaded — a hardcoded 5 against a 9B DiT loads 5
// of its 8 blocks without erroring and generates a plausible WRONG image.

/// The DiT shapes that vary, read off the loaded weights. Zero = unreadable,
/// which leaves the corresponding field at its base value (the subsequent
/// weight load names whatever is actually missing).
pub const DitShapes = struct {
    x_embedder_rows: u32 = 0, // → inner_dim
    x_embedder_in_dim: u32 = 0, // → in_channels (logical, pre-packing)
    context_in_dim: u32 = 0, // → joint_dim (logical, pre-packing)
    norm_q_len: u32 = 0, // → head_dim
    ff_linear_in_rows: u32 = 0, // → 2·inner·mlp_ratio (gate+up fused in one tensor)
    double_blocks: u32 = 0,
    single_blocks: u32 = 0,
};

pub fn ditConfigFrom(shapes: DitShapes, base: FluxConfig) FluxConfig {
    var c = base;
    if (shapes.x_embedder_rows != 0) c.inner_dim = shapes.x_embedder_rows;
    if (shapes.x_embedder_in_dim != 0) c.in_channels = shapes.x_embedder_in_dim;
    if (shapes.context_in_dim != 0) c.joint_dim = shapes.context_in_dim;
    if (shapes.norm_q_len != 0) c.head_dim = shapes.norm_q_len;
    if (c.head_dim != 0 and c.inner_dim % c.head_dim == 0) c.heads = c.inner_dim / c.head_dim;
    if (shapes.ff_linear_in_rows != 0 and c.inner_dim != 0) {
        const half: f32 = @floatFromInt(shapes.ff_linear_in_rows / 2);
        c.mlp_ratio = half / @as(f32, @floatFromInt(c.inner_dim));
    }
    if (shapes.double_blocks != 0) c.double_layers = shapes.double_blocks;
    if (shapes.single_blocks != 0) c.single_layers = shapes.single_blocks;
    return c;
}

/// The Qwen3 encoder shapes that vary. `q_proj_rows`/`k_proj_rows` decide the
/// head counts — NOT hidden/head_dim, which happens to agree on the 9B and is
/// wrong on the 4B (2560/128 = 20 against 32 real q heads).
pub const TeShapes = struct {
    hidden: u32 = 0, // input_layernorm length
    q_norm_len: u32 = 0, // → te_head_dim
    q_proj_rows: u32 = 0,
    k_proj_rows: u32 = 0,
    gate_proj_rows: u32 = 0, // → te_inter
    layers: u32 = 0,
    vocab: u32 = 0,
};

pub fn teConfigFrom(shapes: TeShapes, base: FluxConfig) FluxConfig {
    var c = base;
    if (shapes.hidden != 0) c.te_hidden = shapes.hidden;
    if (shapes.q_norm_len != 0) c.te_head_dim = shapes.q_norm_len;
    if (c.te_head_dim != 0) {
        if (shapes.q_proj_rows != 0) c.te_heads = shapes.q_proj_rows / c.te_head_dim;
        if (shapes.k_proj_rows != 0) c.te_kv = shapes.k_proj_rows / c.te_head_dim;
    }
    if (shapes.gate_proj_rows != 0) c.te_inter = shapes.gate_proj_rows;
    if (shapes.layers != 0) c.te_layers = shapes.layers;
    if (shapes.vocab != 0) c.te_vocab = shapes.vocab;
    return c;
}

/// First-axis extent of a loaded weight, or 0 when absent/rankless. Doubles as
/// the length reader for the 1-D norms.
fn rowsOf(w: *const Weights, key: []const u8) u32 {
    const a = w.get(key) orelse return 0;
    const sh = mlx.getShape(a);
    if (sh.len == 0 or sh[0] < 0) return 0;
    return @intCast(sh[0]);
}

/// Logical (pre-packing) input width of an affine-quantized linear: the packed
/// u32 columns carry `in·bits/32` of them, and the bit width is itself inferred
/// from the scales' geometry.
fn logicalInDim(w: *const Weights, a: std.mem.Allocator, prefix: []const u8) u32 {
    const wk = fmtKey(a, "{s}.weight", .{prefix}) catch return 0;
    defer a.free(wk);
    const sk = fmtKey(a, "{s}.scales", .{prefix}) catch return 0;
    defer a.free(sk);
    const wa = w.get(wk) orelse return 0;
    const sa = w.get(sk) orelse return 0;
    const wsh = mlx.getShape(wa);
    if (wsh.len == 0) return 0;
    const geo = inferQuantGeometryOf(wa, sa);
    if (geo.bits == 0) return 0;
    const w_cols: u32 = @intCast(wsh[wsh.len - 1]);
    return w_cols * 32 / geo.bits;
}

/// Highest `i` for which `fmt`-with-`{d}` names a present weight, counting from
/// 0. Bounded so a malformed map cannot spin.
fn countIndexed(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8) u32 {
    const max_blocks = 256;
    var n: u32 = 0;
    while (n < max_blocks) : (n += 1) {
        const key = fmtKey(a, fmt, .{n}) catch return n;
        defer a.free(key);
        if (w.get(key) == null) return n;
    }
    return n;
}
// ── Qwen3 text encoder ──

const TeLayer = struct {
    input_ln: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    fn deinit(self: *TeLayer) void {
        _ = mlx.mlx_array_free(self.input_ln);
        _ = mlx.mlx_array_free(self.post_ln);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
    }
};

pub const TextEncoder = struct {
    cfg: FluxConfig,
    allocator: std.mem.Allocator,
    s: S,
    embed_table: mlx.mlx_array, // dequantized [vocab, hidden] bf16
    layers: []TeLayer,
    norm: mlx.mlx_array, // final norm (unused for layer-capture but loaded)

    pub fn deinit(self: *TextEncoder) void {
        _ = mlx.mlx_array_free(self.embed_table);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        _ = mlx.mlx_array_free(self.norm);
    }

    /// Token ids [seq] with their attention mask → the hidden states after layers 9, 18 and 27 side by side:
    /// prompt_embeds [1, seq, 3·hidden].
    pub fn encode(self: *TextEncoder, parent: *Graph, ids: []const i32, mask: []const i32) !Array {
        var g = parent.scope();
        defer g.deinit();
        const c = self.cfg;
        if (self.layers.len < 27) return error.TooFewEncoderLayers;
        const seq: c_int = @intCast(ids.len);
        const H: c_int = @intCast(c.te_hidden);
        var x = try g.reshape(try g.take(self.embed_table, try g.fromSlice(i32, ids, &.{seq}), 0), &.{ 1, seq, H });
        const mask_buf = try self.maskBuf(mask);
        defer self.allocator.free(mask_buf);
        const attn_mask = try g.fromSlice(f32, mask_buf, &.{ 1, 1, seq, seq });
        const cfg: qwen3.Cfg = .{ .heads = @intCast(c.te_heads), .kv_heads = @intCast(c.te_kv), .head_dim = @intCast(c.te_head_dim), .eps = c.te_rms_eps };
        const rope: qwen3.Rope = .{ .fast = .{ .theta = c.te_rope_theta, .attn_f32 = .bfloat16 } };
        // Hidden-state list index k is the output of layer k-1 (index 0 = the embeddings).
        const want = [_]usize{ 9, 18, 27 };
        var caps: [3]Array = undefined;
        var captured = false;
        for (self.layers, 1..) |*layer, idx| {
            const nx = try qwen3.layerForward(&g, layer, cfg, rope, x, attn_mask);
            if (!captured) g.drop(x);
            x = nx;
            captured = false;
            for (want, 0..) |wv, wi| if (wv == idx) {
                caps[wi] = x;
                captured = true;
            };
        }
        const stacked = try g.reshape(try g.concat(&caps, 0), &.{ 3, seq, H });
        return parent.keep(try g.reshape(try g.transpose(stacked, &.{ 1, 0, 2 }), &.{ 1, seq, 3 * H }));
    }

    /// The additive [seq, seq] mask: causal, and padding keys masked (caller frees).
    fn maskBuf(self: *TextEncoder, mask: []const i32) ![]f32 {
        const n = mask.len;
        const buf = try self.allocator.alloc(f32, n * n);
        const neg = -std.math.inf(f32);
        for (0..n) |i| for (0..n) |j| {
            buf[i * n + j] = if (j > i or mask[j] == 0) neg else 0.0;
        };
        return buf;
    }
};

pub fn loadTextEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !TextEncoder {
    const dir = try fmtKey(allocator, "{s}/text_encoder", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var te: TextEncoder = undefined;
    te.cfg = teConfigFrom(.{
        .hidden = rowsOf(&w, "layers.0.input_layernorm.weight"),
        .q_norm_len = rowsOf(&w, "layers.0.self_attn.q_norm.weight"),
        .q_proj_rows = rowsOf(&w, "layers.0.self_attn.q_proj.weight"),
        .k_proj_rows = rowsOf(&w, "layers.0.self_attn.k_proj.weight"),
        .gate_proj_rows = rowsOf(&w, "layers.0.mlp.gate_proj.weight"),
        .layers = countIndexed(&w, allocator, "layers.{d}.input_layernorm.weight"),
        .vocab = rowsOf(&w, "embed_tokens.weight"),
    }, .{});
    log.info("[flux] text encoder: hidden={d} layers={d} heads={d}/{d} inter={d}\n", .{
        te.cfg.te_hidden, te.cfg.te_layers, te.cfg.te_heads, te.cfg.te_kv, te.cfg.te_inter,
    });
    te.allocator = allocator;
    te.s = s;

    // dequantize embed table once
    const ew = try ownWeight(&w, "embed_tokens.weight");
    defer _ = mlx.mlx_array_free(ew);
    const es = try ownWeight(&w, "embed_tokens.scales");
    defer _ = mlx.mlx_array_free(es);
    const eb = try ownWeight(&w, "embed_tokens.biases");
    defer _ = mlx.mlx_array_free(eb);
    const embed_geo = inferQuantGeometryOf(ew, es);
    te.embed_table = try dequantTable(ew, es, eb, embed_geo.bits, embed_geo.group_size, s);
    te.norm = try ownWeight(&w, "norm.weight");

    te.layers = try allocator.alloc(TeLayer, te.cfg.te_layers);
    for (te.layers, 0..) |*layer, i| {
        const p_in = try fmtKey(allocator, "layers.{d}.input_layernorm.weight", .{i});
        defer allocator.free(p_in);
        const p_post = try fmtKey(allocator, "layers.{d}.post_attention_layernorm.weight", .{i});
        defer allocator.free(p_post);
        const qn = try fmtKey(allocator, "layers.{d}.self_attn.q_norm.weight", .{i});
        defer allocator.free(qn);
        const kn = try fmtKey(allocator, "layers.{d}.self_attn.k_norm.weight", .{i});
        defer allocator.free(kn);
        const qp = try fmtKey(allocator, "layers.{d}.self_attn.q_proj", .{i});
        defer allocator.free(qp);
        const kp = try fmtKey(allocator, "layers.{d}.self_attn.k_proj", .{i});
        defer allocator.free(kp);
        const vp = try fmtKey(allocator, "layers.{d}.self_attn.v_proj", .{i});
        defer allocator.free(vp);
        const op = try fmtKey(allocator, "layers.{d}.self_attn.o_proj", .{i});
        defer allocator.free(op);
        const gp = try fmtKey(allocator, "layers.{d}.mlp.gate_proj", .{i});
        defer allocator.free(gp);
        const upp = try fmtKey(allocator, "layers.{d}.mlp.up_proj", .{i});
        defer allocator.free(upp);
        const dp = try fmtKey(allocator, "layers.{d}.mlp.down_proj", .{i});
        defer allocator.free(dp);
        layer.* = .{
            .input_ln = try ownWeight(&w, p_in),
            .post_ln = try ownWeight(&w, p_post),
            .q = try QLinear.load(&w, allocator, qp),
            .k = try QLinear.load(&w, allocator, kp),
            .v = try QLinear.load(&w, allocator, vp),
            .o = try QLinear.load(&w, allocator, op),
            .q_norm = try ownWeight(&w, qn),
            .k_norm = try ownWeight(&w, kn),
            .gate = try QLinear.load(&w, allocator, gp),
            .up = try QLinear.load(&w, allocator, upp),
            .down = try QLinear.load(&w, allocator, dp),
        };
    }
    return te;
}

// ── DiT: double-stream blocks (text and image apart, joint attention), then single-stream blocks over the joint
// sequence [text, image]; one modulation per stream kind, shared by every block ──

const DoubleBlock = struct {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    add_q: QLinear,
    add_k: QLinear,
    add_v: QLinear,
    add_o: QLinear,
    nq: mlx.mlx_array,
    nk: mlx.mlx_array,
    naq: mlx.mlx_array,
    nak: mlx.mlx_array,
    ff_in: QLinear,
    ff_out: QLinear,
    ffc_in: QLinear,
    ffc_out: QLinear,
    fn deinit(self: *DoubleBlock) void {
        inline for (.{ "q", "k", "v", "o", "add_q", "add_k", "add_v", "add_o", "ff_in", "ff_out", "ffc_in", "ffc_out" }) |f| @field(self, f).deinit();
        inline for (.{ "nq", "nk", "naq", "nak" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
};
const SingleBlock = struct {
    qkv_mlp: QLinear,
    o: QLinear,
    nq: mlx.mlx_array,
    nk: mlx.mlx_array,
    fn deinit(self: *SingleBlock) void {
        self.qkv_mlp.deinit();
        self.o.deinit();
        _ = mlx.mlx_array_free(self.nq);
        _ = mlx.mlx_array_free(self.nk);
    }
};

pub const Dit = struct {
    cfg: FluxConfig,
    allocator: std.mem.Allocator,
    s: S,
    x_embedder: QLinear,
    context_embedder: QLinear,
    t_lin1: QLinear,
    t_lin2: QLinear,
    mod_img: QLinear,
    mod_txt: QLinear,
    mod_single: QLinear,
    doubles: []DoubleBlock,
    singles: []SingleBlock,
    norm_out_lin: QLinear,
    proj_out: QLinear,

    pub fn deinit(self: *Dit) void {
        self.x_embedder.deinit();
        self.context_embedder.deinit();
        self.t_lin1.deinit();
        self.t_lin2.deinit();
        self.mod_img.deinit();
        self.mod_txt.deinit();
        self.mod_single.deinit();
        for (self.doubles) |*b| b.deinit();
        self.allocator.free(self.doubles);
        for (self.singles) |*b| b.deinit();
        self.allocator.free(self.singles);
        self.norm_out_lin.deinit();
        self.proj_out.deinit();
    }

    /// Interleaved-RoPE cos/sin [seq, 64] from position ids [seq, 4] (t, h, w, l), 16 frequencies per axis.
    fn buildRope(self: *const Dit, g: *Graph, ids: []const i32, seq: usize) !struct { cos: Array, sin: Array } {
        const cosb = try self.allocator.alloc(f32, seq * 64);
        defer self.allocator.free(cosb);
        const sinb = try self.allocator.alloc(f32, seq * 64);
        defer self.allocator.free(sinb);
        const theta: f64 = self.cfg.rope_theta_dit;
        for (0..seq) |p| {
            for (0..4) |axis| {
                const pos: f64 = @floatFromInt(ids[p * 4 + axis]);
                for (0..16) |kk| {
                    const omega = std.math.pow(f64, theta, -@as(f64, @floatFromInt(2 * kk)) / 32.0);
                    const ang = pos * omega;
                    const idx = p * 64 + axis * 16 + kk;
                    cosb[idx] = @floatCast(@cos(ang));
                    sinb[idx] = @floatCast(@sin(ang));
                }
            }
        }
        const shape = [_]c_int{ @intCast(seq), 64 };
        return .{ .cos = try g.fromSlice(f32, cosb, &shape), .sin = try g.fromSlice(f32, sinb, &shape) };
    }

    /// The noise prediction [1, Ni, 128] for latents [1, Ni, 128] beside the prompt [1, Nt, joint] at `timestep`
    /// (×1000 scale), with the position ids of both.
    pub fn forward(self: *const Dit, parent: *Graph, latents: Array, enc: Array, timestep: f32, img_ids: []const i32, txt_ids: []const i32) !Array {
        var g = parent.scope();
        defer g.deinit();
        const c = self.cfg;
        const inner: c_int = @intCast(c.inner_dim);
        const nt = ops.shape(enc)[1];
        const nj = nt + ops.shape(latents)[1];
        const temb = try self.timeEmbed(&g, timestep);
        const tr = try self.buildRope(&g, txt_ids, @intCast(nt));
        const ir = try self.buildRope(&g, img_ids, @intCast(nj - nt));
        const rope_sh = [_]c_int{ 1, 1, nj, 64 };
        // Flux2Modulation applies silu(temb) before its linear.
        const stemb = try g.silu(temb);
        var ctx: StepCtx = .{
            .nt = nt,
            .nj = nj,
            .inner = inner,
            .heads = @intCast(c.heads),
            .hd = @intCast(c.head_dim),
            .mlp_h = @intFromFloat(@as(f32, @floatFromInt(inner)) * c.mlp_ratio),
            .cos = try g.reshape(try g.concat(&.{ tr.cos, ir.cos }, 0), &rope_sh),
            .sin = try g.reshape(try g.concat(&.{ tr.sin, ir.sin }, 0), &rope_sh),
            .temb = temb,
        };
        try chunks(&g, try self.mod_img.forward(&g, stemb), inner, &ctx.mi);
        try chunks(&g, try self.mod_txt.forward(&g, stemb), inner, &ctx.mt);
        try chunks(&g, try self.mod_single.forward(&g, stemb), inner, &ctx.ms);
        return parent.keep(try dit_base.Dit(Hooks).forward(&g, self, .{ .latents = latents, .enc = enc }, ctx, null));
    }

    /// The time MLP over sinusoidal features (cos half first) → [1, 1, inner].
    fn timeEmbed(self: *const Dit, parent: *Graph, t: f32) !Array {
        var g = parent.scope();
        defer g.deinit();
        var feats: [256]f32 = undefined;
        timestep_blk.sinusoid(f32, &feats, t, .cos_first);
        const te = try g.astype(try g.fromSlice(f32, &feats, &.{ 1, 256 }), .bfloat16);
        const t2 = try self.t_lin2.forward(&g, try g.silu(try self.t_lin1.forward(&g, te)));
        return parent.keep(try g.reshape(t2, &.{ 1, 1, @intCast(self.cfg.inner_dim) }));
    }
};

/// One step's conditioning: the rope tables over [text, image], the three modulations split into their rows.
const StepCtx = struct {
    nt: c_int,
    nj: c_int,
    inner: c_int,
    heads: c_int,
    hd: c_int,
    mlp_h: c_int,
    cos: Array,
    sin: Array,
    temb: Array,
    /// [shift, scale, gate] for attention then the MLP, each [1, 1, inner].
    mi: [6]Array = undefined,
    mt: [6]Array = undefined,
    ms: [3]Array = undefined,
};

/// `mod` [1, 1, n·inner] → its n rows.
fn chunks(g: *Graph, mod: Array, inner: c_int, out: []Array) !void {
    for (out, 0..) |*o, i| o.* = try g.sliceAxis(mod, 2, @as(c_int, @intCast(i)) * inner, @as(c_int, @intCast(i + 1)) * inner);
}

/// The skeleton's hooks: the activation between blocks is the joint sequence [text, image].
const Hooks = struct {
    pub const Model = Dit;
    pub const Ctx = StepCtx;
    pub const Input = struct { latents: Array, enc: Array };

    pub fn embed(parent: *Graph, m: *const Dit, in: Input, _: StepCtx) !Array {
        var g = parent.scope();
        defer g.deinit();
        const hs = try m.x_embedder.forward(&g, in.latents);
        const ehs = try m.context_embedder.forward(&g, in.enc);
        return parent.keep(try g.concat(&.{ ehs, hs }, 1));
    }

    pub fn layers(m: *const Dit) usize {
        return m.doubles.len + m.singles.len;
    }

    pub fn block(g: *Graph, m: *const Dit, i: usize, x: Array, ctx: StepCtx) !Array {
        if (i < m.doubles.len) return doubleBlock(g, &m.doubles[i], x, ctx);
        return singleBlock(g, &m.singles[i - m.doubles.len], x, ctx);
    }

    /// AdaLayerNormContinuous on the image span, then the projection out.
    pub fn final(parent: *Graph, m: *const Dit, x: Array, ctx: StepCtx) !Array {
        var g = parent.scope();
        defer g.deinit();
        const img = try g.sliceAxis(x, 1, ctx.nt, ctx.nj);
        const so = try m.norm_out_lin.forward(&g, try g.silu(ctx.temb));
        const scale = try g.sliceAxis(so, 2, 0, ctx.inner);
        const shift = try g.sliceAxis(so, 2, ctx.inner, 2 * ctx.inner);
        const modded = try modulated(&g, img, scale, shift);
        return parent.keep(try m.proj_out.forward(&g, modded));
    }
};

/// LayerNorm without affine (f32 statistics, bf16 out), then x·(scale + 1) + shift.
fn modulated(g: *Graph, x: Array, scale: Array, shift: Array) !Array {
    var c = g.scope();
    defer c.deinit();
    return g.keep(try modulate_blk.modulate(&c, try ln_blk.layerNormF32(&c, x, null, null, 1e-6, .bfloat16), scale, shift, .float32));
}

/// [1, seq, heads·hd] → [1, heads, seq, hd], RMS-normed over hd in f32, back to bf16.
fn headsNormed(parent: *Graph, x: Array, seq: c_int, ctx: StepCtx, norm: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const xt = try g.transpose(try g.reshape(x, &.{ 1, seq, ctx.heads, ctx.hd }), &.{ 0, 2, 1, 3 });
    return parent.keep(try g.astype(try g.rmsNorm(try g.astype(xt, .float32), norm, 1e-5), .bfloat16));
}

fn heads(g: *Graph, x: Array, seq: c_int, ctx: StepCtx) !Array {
    var c = g.scope();
    defer c.deinit();
    return g.keep(try c.transpose(try c.reshape(x, &.{ 1, seq, ctx.heads, ctx.hd }), &.{ 0, 2, 1, 3 }));
}

/// Joint attention over [text, image] after RoPE → [1, Nj, heads·hd].
fn attention(parent: *Graph, q: Array, k: Array, v: Array, ctx: StepCtx) !Array {
    var g = parent.scope();
    defer g.deinit();
    const qr = try rope_blk.ropePairs(&g, q, ctx.cos, ctx.sin, .bfloat16);
    const kr = try rope_blk.ropePairs(&g, k, ctx.cos, ctx.sin, .bfloat16);
    const out_sh = [_]c_int{ 1, ctx.nj, ctx.heads * ctx.hd };
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(ctx.hd)));
    if (w8a8.w8a8State().enabled) if (try int8_attn.attention(qr, kr, v, scale, g.s)) |o|
        return parent.keep(try g.reshape(try g.adopt(o), &out_sh));
    const at = try g.transpose(try g.sdpa(qr, kr, v, scale, .none), &.{ 0, 2, 1, 3 });
    return parent.keep(try g.reshape(at, &out_sh));
}

/// SwiGLU over the fused gate|up projection.
fn feedForward(parent: *Graph, x: Array, lin_in: *const QLinear, lin_out: *const QLinear) !Array {
    var g = parent.scope();
    defer g.deinit();
    const proj = try lin_in.forward(&g, x);
    const half = @divExact(ops.shape(proj)[2], 2);
    const act = try g.mul(try g.silu(try g.sliceAxis(proj, 2, 0, half)), try g.sliceAxis(proj, 2, half, 2 * half));
    return parent.keep(try lin_out.forward(&g, act));
}

fn doubleBlock(parent: *Graph, b: *const DoubleBlock, x: Array, ctx: StepCtx) !Array {
    var g = parent.scope();
    defer g.deinit();
    const ni = ctx.nj - ctx.nt;
    const im = ctx.mi;
    const tm = ctx.mt;
    const ehs = try g.sliceAxis(x, 1, 0, ctx.nt);
    const hs = try g.sliceAxis(x, 1, ctx.nt, ctx.nj);
    const nhm = try modulated(&g, hs, im[1], im[0]);
    const nem = try modulated(&g, ehs, tm[1], tm[0]);
    const qh = try headsNormed(&g, try b.q.forward(&g, nhm), ni, ctx, b.nq);
    const kh = try headsNormed(&g, try b.k.forward(&g, nhm), ni, ctx, b.nk);
    const vh = try heads(&g, try b.v.forward(&g, nhm), ni, ctx);
    const eqh = try headsNormed(&g, try b.add_q.forward(&g, nem), ctx.nt, ctx, b.naq);
    const ekh = try headsNormed(&g, try b.add_k.forward(&g, nem), ctx.nt, ctx, b.nak);
    const evh = try heads(&g, try b.add_v.forward(&g, nem), ctx.nt, ctx);
    const attn = try attention(&g, try g.concat(&.{ eqh, qh }, 2), try g.concat(&.{ ekh, kh }, 2), try g.concat(&.{ evh, vh }, 2), ctx);
    const ao_img = try b.o.forward(&g, try g.sliceAxis(attn, 1, ctx.nt, ctx.nj));
    const ao_txt = try b.add_o.forward(&g, try g.sliceAxis(attn, 1, 0, ctx.nt));
    const hs1 = try g.add(hs, try g.mul(im[2], ao_img));
    const ehs1 = try g.add(ehs, try g.mul(tm[2], ao_txt));
    const ff_img = try feedForward(&g, try modulated(&g, hs1, im[4], im[3]), &b.ff_in, &b.ff_out);
    const img_out = try g.add(hs1, try g.mul(im[5], ff_img));
    const ff_txt = try feedForward(&g, try modulated(&g, ehs1, tm[4], tm[3]), &b.ffc_in, &b.ffc_out);
    const txt_out = try g.add(ehs1, try g.mul(tm[5], ff_txt));
    return parent.keep(try g.concat(&.{ txt_out, img_out }, 1));
}

/// Parallel attention + SwiGLU from one fused projection.
fn singleBlock(parent: *Graph, b: *const SingleBlock, x: Array, ctx: StepCtx) !Array {
    var g = parent.scope();
    defer g.deinit();
    const m = ctx.ms;
    const inner = ctx.inner;
    const proj = try b.qkv_mlp.forward(&g, try modulated(&g, x, m[1], m[0]));
    const qkv = try g.sliceAxis(proj, 2, 0, 3 * inner);
    const mlp = try g.sliceAxis(proj, 2, 3 * inner, 3 * inner + 2 * ctx.mlp_h);
    const qh = try headsNormed(&g, try g.sliceAxis(qkv, 2, 0, inner), ctx.nj, ctx, b.nq);
    const kh = try headsNormed(&g, try g.sliceAxis(qkv, 2, inner, 2 * inner), ctx.nj, ctx, b.nk);
    const vh = try heads(&g, try g.sliceAxis(qkv, 2, 2 * inner, 3 * inner), ctx.nj, ctx);
    const attn = try attention(&g, qh, kh, vh, ctx);
    const mlp_act = try g.mul(try g.silu(try g.sliceAxis(mlp, 2, 0, ctx.mlp_h)), try g.sliceAxis(mlp, 2, ctx.mlp_h, 2 * ctx.mlp_h));
    const ao = try b.o.forward(&g, try g.concat(&.{ attn, mlp_act }, 2));
    return parent.keep(try g.add(x, try g.mul(m[2], ao)));
}

pub fn loadDit(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Dit {
    const dir = try fmtKey(allocator, "{s}/transformer", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var d: Dit = undefined;
    d.cfg = ditConfigFrom(.{
        .x_embedder_rows = rowsOf(&w, "x_embedder.weight"),
        .x_embedder_in_dim = logicalInDim(&w, allocator, "x_embedder"),
        .context_in_dim = logicalInDim(&w, allocator, "context_embedder"),
        .norm_q_len = rowsOf(&w, "transformer_blocks.0.attn.norm_q.weight"),
        .ff_linear_in_rows = rowsOf(&w, "transformer_blocks.0.ff.linear_in.weight"),
        .double_blocks = countIndexed(&w, allocator, "transformer_blocks.{d}.attn.to_q.weight"),
        .single_blocks = countIndexed(&w, allocator, "single_transformer_blocks.{d}.attn.to_out.weight"),
    }, .{});
    log.info("[flux] dit: inner={d} heads={d}x{d} double={d} single={d} joint={d}\n", .{
        d.cfg.inner_dim, d.cfg.heads, d.cfg.head_dim, d.cfg.double_layers, d.cfg.single_layers, d.cfg.joint_dim,
    });
    d.allocator = allocator;
    d.s = s;
    d.x_embedder = try QLinear.load(&w, allocator, "x_embedder");
    d.context_embedder = try QLinear.load(&w, allocator, "context_embedder");
    d.t_lin1 = try QLinear.load(&w, allocator, "time_guidance_embed.linear_1");
    d.t_lin2 = try QLinear.load(&w, allocator, "time_guidance_embed.linear_2");
    d.mod_img = try QLinear.load(&w, allocator, "double_stream_modulation_img.linear");
    d.mod_txt = try QLinear.load(&w, allocator, "double_stream_modulation_txt.linear");
    d.mod_single = try QLinear.load(&w, allocator, "single_stream_modulation.linear");
    d.norm_out_lin = try QLinear.load(&w, allocator, "norm_out.linear");
    d.proj_out = try QLinear.load(&w, allocator, "proj_out");
    d.doubles = try allocator.alloc(DoubleBlock, d.cfg.double_layers);
    for (d.doubles, 0..) |*b, i| {
        const pfx = try fmtKey(allocator, "transformer_blocks.{d}", .{i});
        defer allocator.free(pfx);
        b.* = try loadDouble(&w, allocator, pfx);
        // The token-GEMM linears go int8 (`--w8a8`); modulation and embedders see one row or run once.
        inline for (.{ "q", "k", "v", "o", "add_q", "add_k", "add_v", "add_o", "ff_in", "ff_out", "ffc_in", "ffc_out" }) |f| try @field(b, f).toW8a8(s);
        if (w8a8.w8a8State().enabled) w.dropBlock("transformer_blocks.{d}.", i);
    }
    d.singles = try allocator.alloc(SingleBlock, d.cfg.single_layers);
    for (d.singles, 0..) |*b, i| {
        const pfx = try fmtKey(allocator, "single_transformer_blocks.{d}", .{i});
        defer allocator.free(pfx);
        const qp = try fmtKey(allocator, "{s}.attn.to_qkv_mlp_proj", .{pfx});
        defer allocator.free(qp);
        const op = try fmtKey(allocator, "{s}.attn.to_out", .{pfx});
        defer allocator.free(op);
        const nq = try fmtKey(allocator, "{s}.attn.norm_q.weight", .{pfx});
        defer allocator.free(nq);
        const nk = try fmtKey(allocator, "{s}.attn.norm_k.weight", .{pfx});
        defer allocator.free(nk);
        b.* = .{ .qkv_mlp = try QLinear.load(&w, allocator, qp), .o = try QLinear.load(&w, allocator, op), .nq = try ownWeight(&w, nq), .nk = try ownWeight(&w, nk) };
        try b.qkv_mlp.toW8a8(s);
        try b.o.toW8a8(s);
        if (w8a8.w8a8State().enabled) w.dropBlock("single_transformer_blocks.{d}.", i);
    }
    return d;
}

/// Attach every adapter in `stack` to its matching DiT linear (summed at
/// forward time — see `lora.deltaSum`). Non-owning: `stack` must outlive
/// the attach. Returns the number of (module, matched-adapter) attachments
/// across the whole stack, i.e. a module hit by two stacked LoRAs counts
/// twice — useful as a "did anything match" / logging signal.
pub fn attachLora(dit: *Dit, stack: *const lora_mod.Stack) u32 {
    detachLora(dit);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;
    for (dit.doubles, 0..) |*b, i| {
        const mods = .{
            .{ "attn.to_q", &b.q },                 .{ "attn.to_k", &b.k },
            .{ "attn.to_v", &b.v },                 .{ "attn.to_out", &b.o },
            .{ "attn.add_q_proj", &b.add_q },       .{ "attn.add_k_proj", &b.add_k },
            .{ "attn.add_v_proj", &b.add_v },       .{ "attn.to_add_out", &b.add_o },
            .{ "ff.linear_in", &b.ff_in },          .{ "ff.linear_out", &b.ff_out },
            .{ "ff_context.linear_in", &b.ffc_in }, .{ "ff_context.linear_out", &b.ffc_out },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "transformer_blocks.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }
    for (dit.singles, 0..) |*b, i| {
        const mods = .{ .{ "attn.to_qkv_mlp_proj", &b.qkv_mlp }, .{ "attn.to_out", &b.o } };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "single_transformer_blocks.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }
    return matched;
}

pub fn detachLora(dit: *Dit) void {
    for (dit.doubles) |*b| {
        inline for (.{ &b.q, &b.k, &b.v, &b.o, &b.add_q, &b.add_k, &b.add_v, &b.add_o, &b.ff_in, &b.ff_out, &b.ffc_in, &b.ffc_out }) |ql| ql.clearLoraRefs();
    }
    for (dit.singles) |*b| {
        b.qkv_mlp.clearLoraRefs();
        b.o.clearLoraRefs();
    }
}

fn loadDouble(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !DoubleBlock {
    const ld = struct {
        fn q(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !QLinear {
            const k = try fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(k);
            return QLinear.load(ww, aa, k);
        }
        fn n(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
            const k = try fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(k);
            return ownWeight(ww, k);
        }
    };
    return .{
        .q = try ld.q(w, a, pfx, "attn.to_q"),
        .k = try ld.q(w, a, pfx, "attn.to_k"),
        .v = try ld.q(w, a, pfx, "attn.to_v"),
        .o = try ld.q(w, a, pfx, "attn.to_out"),
        .add_q = try ld.q(w, a, pfx, "attn.add_q_proj"),
        .add_k = try ld.q(w, a, pfx, "attn.add_k_proj"),
        .add_v = try ld.q(w, a, pfx, "attn.add_v_proj"),
        .add_o = try ld.q(w, a, pfx, "attn.to_add_out"),
        .nq = try ld.n(w, a, pfx, "attn.norm_q.weight"),
        .nk = try ld.n(w, a, pfx, "attn.norm_k.weight"),
        .naq = try ld.n(w, a, pfx, "attn.norm_added_q.weight"),
        .nak = try ld.n(w, a, pfx, "attn.norm_added_k.weight"),
        .ff_in = try ld.q(w, a, pfx, "ff.linear_in"),
        .ff_out = try ld.q(w, a, pfx, "ff.linear_out"),
        .ffc_in = try ld.q(w, a, pfx, "ff_context.linear_in"),
        .ffc_out = try ld.q(w, a, pfx, "ff_context.linear_out"),
    };
}

// ── VAE: packed latents [1, 128, h, w] ↔ pixels [1, 3, 16h, 16w]; channels-last inside, bf16 activations ──

const VaeAttn = struct {
    gnw: mlx.mlx_array,
    gnb: mlx.mlx_array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    fn deinit(self: *VaeAttn) void {
        _ = mlx.mlx_array_free(self.gnw);
        _ = mlx.mlx_array_free(self.gnb);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
    }
};

/// Frees the arrays a loaded resnet owns.
fn freeResnet(r: *Resnet) void {
    inline for (.{ "n1w", "c1w", "c1b", "n2w", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(r, f));
    inline for (.{ "n1b", "n2b", "sw", "sb" }) |f| if (@field(r, f)) |x| {
        _ = mlx.mlx_array_free(x);
    };
}

/// The batch-norm statistics the latents are standardized by: mean and sqrt(var + 1e-4), f32 [1, 128, 1, 1].
fn bnStats(parent: *Graph, mean: Array, variance: Array) !struct { mean: Array, std: Array } {
    var g = parent.scope();
    defer g.deinit();
    const vsh = [_]c_int{ 1, 128, 1, 1 };
    const sd = try g.sqrt(try g.add(try g.reshape(variance, &vsh), try g.scalar(1e-4, .float32)));
    return .{ .mean = try parent.keep(try g.astype(try g.reshape(mean, &vsh), .float32)), .std = try parent.keep(try g.astype(sd, .float32)) };
}

pub const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    bn_mean: mlx.mlx_array,
    bn_var: mlx.mlx_array,
    pq_w: mlx.mlx_array,
    pq_b: mlx.mlx_array, // post_quant_conv 1x1
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: VaeAttn,
    mid_r1: Resnet,
    up_resnets: [4][3]Resnet,
    up_conv_w: [3]mlx.mlx_array,
    up_conv_b: [3]mlx.mlx_array, // upsamplers for blocks 0,1,2
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Vae) void {
        inline for (.{ "bn_mean", "bn_var", "pq_w", "pq_b", "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        freeResnet(&self.mid_r0);
        self.mid_attn.deinit();
        freeResnet(&self.mid_r1);
        for (&self.up_resnets) |*blk| for (blk) |*r| freeResnet(r);
        for (0..3) |i| {
            _ = mlx.mlx_array_free(self.up_conv_w[i]);
            _ = mlx.mlx_array_free(self.up_conv_b[i]);
        }
    }

    /// Packed latents [1, 128, h, w] (bn-normalized) → image [1, 3, 16h, 16w] in [-1, 1].
    pub fn decode(self: *const Vae, parent: *Graph, packed_in: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const bn = try bnStats(&g, self.bn_mean, self.bn_var);
        const denorm = try g.add(try g.mul(try g.astype(packed_in, .float32), bn.std), bn.mean);
        const nhwc = try g.transpose(try g.astype(try unpatchify(&g, denorm), .bfloat16), &.{ 0, 2, 3, 1 });
        var h = try conv_blk.conv2d(&g, nhwc, self.pq_w, self.pq_b, 1, 0, 1);
        h = try conv_blk.conv2d(&g, h, self.conv_in_w, self.conv_in_b, 1, 1, 1);
        h = try self.mid_r0.forward(&g, vae_norm, h);
        h = try resnet_blk.midAttention(&g, vae_norm, h, self.mid_attn.gnw, self.mid_attn.gnb, &self.mid_attn);
        h = try self.mid_r1.forward(&g, vae_norm, h);
        for (0..4) |bi| {
            for (&self.up_resnets[bi]) |*r| h = try r.forward(&g, vae_norm, h);
            if (bi < 3) h = try conv_blk.upsample2x(&g, h, self.up_conv_w[bi], self.up_conv_b[bi]);
        }
        h = try g.silu(try gn_blk.groupNormFlat(&g, h, self.norm_out_w, self.norm_out_b, 32, 1e-6, .bfloat16));
        h = try conv_blk.conv2d(&g, h, self.conv_out_w, self.conv_out_b, 1, 1, 1);
        // Materialized: a lazy transpose reads back in source order.
        return parent.keep(try g.contiguous(try g.transpose(h, &.{ 0, 3, 1, 2 })));
    }
};

/// Packed [1, 128, h, w] → [1, 32, 2h, 2w] (packed channel = c·4 + ph·2 + pw).
fn unpatchify(parent: *Graph, x: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const r = try g.reshape(x, &.{ 1, 32, 2, 2, sh[2], sh[3] });
    return parent.keep(try g.reshape(try g.transpose(r, &.{ 0, 1, 4, 2, 5, 3 }), &.{ 1, 32, sh[2] * 2, sh[3] * 2 }));
}

/// [1, 32, H, W] → packed [1, 128, H/2, W/2]; the exact inverse of `unpatchify`.
fn patchify(parent: *Graph, x: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x);
    const gh = @divExact(sh[2], 2);
    const gw = @divExact(sh[3], 2);
    const r = try g.reshape(x, &.{ 1, 32, gh, 2, gw, 2 });
    return parent.keep(try g.reshape(try g.transpose(r, &.{ 0, 1, 3, 5, 2, 4 }), &.{ 1, 128, gh, gw }));
}

fn loadResnet(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) !Resnet {
    const g = struct {
        fn k(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) !mlx.mlx_array {
            const kk = try fmtKey(aa, "{s}.{s}", .{ p, sub });
            defer aa.free(kk);
            return ownWeight(ww, kk);
        }
        fn opt(ww: *const Weights, aa: std.mem.Allocator, p: []const u8, sub: []const u8) ?mlx.mlx_array {
            const kk = fmtKey(aa, "{s}.{s}", .{ p, sub }) catch return null;
            defer aa.free(kk);
            return ownOpt(ww, kk);
        }
    };
    return .{
        .n1w = try g.k(w, a, pfx, "norm1.weight"),
        .n1b = try g.k(w, a, pfx, "norm1.bias"),
        .c1w = try g.k(w, a, pfx, "conv1.weight"),
        .c1b = try g.k(w, a, pfx, "conv1.bias"),
        .n2w = try g.k(w, a, pfx, "norm2.weight"),
        .n2b = try g.k(w, a, pfx, "norm2.bias"),
        .c2w = try g.k(w, a, pfx, "conv2.weight"),
        .c2b = try g.k(w, a, pfx, "conv2.bias"),
        .sw = g.opt(w, a, pfx, "conv_shortcut.weight"),
        .sb = g.opt(w, a, pfx, "conv_shortcut.bias"),
    };
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: Vae = undefined;
    v.allocator = allocator;
    v.s = s;
    v.bn_mean = try ownWeight(&w, "bn.running_mean");
    v.bn_var = try ownWeight(&w, "bn.running_var");
    v.pq_w = try ownWeight(&w, "post_quant_conv.weight");
    v.pq_b = try ownWeight(&w, "post_quant_conv.bias");
    v.conv_in_w = try ownWeight(&w, "decoder.conv_in.weight");
    v.conv_in_b = try ownWeight(&w, "decoder.conv_in.bias");
    v.mid_r0 = try loadResnet(&w, allocator, "decoder.mid_block.resnets.0");
    v.mid_r1 = try loadResnet(&w, allocator, "decoder.mid_block.resnets.1");
    v.mid_attn = .{
        .gnw = try ownWeight(&w, "decoder.mid_block.attentions.0.group_norm.weight"),
        .gnb = try ownWeight(&w, "decoder.mid_block.attentions.0.group_norm.bias"),
        .q = try QLinear.load(&w, allocator, "decoder.mid_block.attentions.0.to_q"),
        .k = try QLinear.load(&w, allocator, "decoder.mid_block.attentions.0.to_k"),
        .v = try QLinear.load(&w, allocator, "decoder.mid_block.attentions.0.to_v"),
        .o = try QLinear.load(&w, allocator, "decoder.mid_block.attentions.0.to_out"),
    };
    for (0..4) |bi| {
        for (0..3) |ri| {
            const pfx = try fmtKey(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            v.up_resnets[bi][ri] = try loadResnet(&w, allocator, pfx);
        }
    }
    for (0..3) |bi| {
        const wk = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi});
        defer allocator.free(wk);
        const bk = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi});
        defer allocator.free(bk);
        v.up_conv_w[bi] = try ownWeight(&w, wk);
        v.up_conv_b[bi] = try ownWeight(&w, bk);
    }
    v.norm_out_w = try ownWeight(&w, "decoder.conv_norm_out.weight");
    v.norm_out_b = try ownWeight(&w, "decoder.conv_norm_out.bias");
    v.conv_out_w = try ownWeight(&w, "decoder.conv_out.weight");
    v.conv_out_b = try ownWeight(&w, "decoder.conv_out.bias");
    return v;
}

/// The VAE encoder (the decoder's mirror with stride-2 downsamplers): pixels → the bn-normalized packed latents
/// `Vae.decode` reads. Feeds variation and edits.
pub const VaeEncoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    bn_mean: mlx.mlx_array,
    bn_var: mlx.mlx_array,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    down_resnets: [4][2]Resnet,
    down_conv_w: [3]mlx.mlx_array, // downsamplers for blocks 0,1,2
    down_conv_b: [3]mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: VaeAttn,
    mid_r1: Resnet,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    quant_w: mlx.mlx_array, // quant_conv 1x1 (64→64: mean+logvar)
    quant_b: mlx.mlx_array,

    pub fn deinit(self: *VaeEncoder) void {
        inline for (.{ "bn_mean", "bn_var", "conv_in_w", "conv_in_b", "norm_out_w", "norm_out_b", "conv_out_w", "conv_out_b", "quant_w", "quant_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        for (&self.down_resnets) |*blk| for (blk) |*r| freeResnet(r);
        for (0..3) |i| {
            _ = mlx.mlx_array_free(self.down_conv_w[i]);
            _ = mlx.mlx_array_free(self.down_conv_b[i]);
        }
        freeResnet(&self.mid_r0);
        self.mid_attn.deinit();
        freeResnet(&self.mid_r1);
    }

    /// Image [1, 3, H, W] f32 in [0, 1] → packed latents [1, 128, H/16, W/16] f32, bn-normalized (the
    /// distribution's mean: deterministic, no sampling).
    pub fn encode(self: *const VaeEncoder, parent: *Graph, img: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const signed = try g.sub(try g.mul(img, try g.scalar(2.0, .float32)), try g.scalar(1.0, .float32));
        var h = try g.astype(try g.transpose(signed, &.{ 0, 2, 3, 1 }), .bfloat16);
        h = try conv_blk.conv2d(&g, h, self.conv_in_w, self.conv_in_b, 1, 1, 1);
        for (0..4) |bi| {
            for (&self.down_resnets[bi]) |*r| h = try r.forward(&g, vae_norm, h);
            if (bi < 3) h = try conv_blk.downsample2x(&g, h, self.down_conv_w[bi], self.down_conv_b[bi]);
        }
        h = try self.mid_r0.forward(&g, vae_norm, h);
        h = try resnet_blk.midAttention(&g, vae_norm, h, self.mid_attn.gnw, self.mid_attn.gnb, &self.mid_attn);
        h = try self.mid_r1.forward(&g, vae_norm, h);
        h = try g.silu(try gn_blk.groupNormFlat(&g, h, self.norm_out_w, self.norm_out_b, 32, 1e-6, .bfloat16));
        h = try conv_blk.conv2d(&g, h, self.conv_out_w, self.conv_out_b, 1, 1, 1);
        const qc = try conv_blk.conv2d(&g, h, self.quant_w, self.quant_b, 1, 0, 1);
        // The mean is the first 32 channels; the log-variance is discarded.
        const mean = try g.contiguous(try g.sliceAxis(qc, 3, 0, 32));
        const packed_lat = try patchify(&g, try g.astype(try g.transpose(mean, &.{ 0, 3, 1, 2 }), .float32));
        const bn = try bnStats(&g, self.bn_mean, self.bn_var);
        return parent.keep(try g.div(try g.sub(packed_lat, bn.mean), bn.std));
    }
};

pub fn loadVaeEncoder(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !VaeEncoder {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: VaeEncoder = undefined;
    v.allocator = allocator;
    v.s = s;
    v.bn_mean = try ownWeight(&w, "bn.running_mean");
    v.bn_var = try ownWeight(&w, "bn.running_var");
    v.conv_in_w = try ownWeight(&w, "encoder.conv_in.weight");
    v.conv_in_b = try ownWeight(&w, "encoder.conv_in.bias");
    for (0..4) |bi| {
        for (0..2) |ri| {
            const pfx = try fmtKey(allocator, "encoder.down_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            v.down_resnets[bi][ri] = try loadResnet(&w, allocator, pfx);
        }
    }
    for (0..3) |bi| {
        const wk = try fmtKey(allocator, "encoder.down_blocks.{d}.downsamplers.0.conv.weight", .{bi});
        defer allocator.free(wk);
        const bk = try fmtKey(allocator, "encoder.down_blocks.{d}.downsamplers.0.conv.bias", .{bi});
        defer allocator.free(bk);
        v.down_conv_w[bi] = try ownWeight(&w, wk);
        v.down_conv_b[bi] = try ownWeight(&w, bk);
    }
    v.mid_r0 = try loadResnet(&w, allocator, "encoder.mid_block.resnets.0");
    v.mid_r1 = try loadResnet(&w, allocator, "encoder.mid_block.resnets.1");
    v.mid_attn = .{
        .gnw = try ownWeight(&w, "encoder.mid_block.attentions.0.group_norm.weight"),
        .gnb = try ownWeight(&w, "encoder.mid_block.attentions.0.group_norm.bias"),
        .q = try QLinear.load(&w, allocator, "encoder.mid_block.attentions.0.to_q"),
        .k = try QLinear.load(&w, allocator, "encoder.mid_block.attentions.0.to_k"),
        .v = try QLinear.load(&w, allocator, "encoder.mid_block.attentions.0.to_v"),
        .o = try QLinear.load(&w, allocator, "encoder.mid_block.attentions.0.to_out"),
    };
    v.norm_out_w = try ownWeight(&w, "encoder.conv_norm_out.weight");
    v.norm_out_b = try ownWeight(&w, "encoder.conv_norm_out.bias");
    v.conv_out_w = try ownWeight(&w, "encoder.conv_out.weight");
    v.conv_out_b = try ownWeight(&w, "encoder.conv_out.bias");
    v.quant_w = try ownWeight(&w, "quant_conv.weight");
    v.quant_b = try ownWeight(&w, "quant_conv.bias");
    return v;
}

/// Conditioning rebalance: the prompt embeddings scaled by a global `gain` and per-tap `weights` (one per captured
/// encoder layer, concatenated on the feature axis); same dtype out.
pub fn applyCondRebalance(parent: *Graph, enc: Array, gain: f32, weights: ?[]const f32) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(enc);
    const dt = ops.dtype(enc);
    var cur = enc;
    if (weights) |w| {
        if (w.len != 3) return error.InvalidCondWeights;
        const r = try g.reshape(cur, &.{ sh[0], sh[1], 3, @divExact(sh[2], 3) });
        cur = try g.reshape(try g.mul(r, try g.fromSlice(f32, w, &.{ 1, 1, 3, 1 })), &.{ sh[0], sh[1], sh[2] });
    }
    if (gain != 1.0) cur = try g.mul(cur, try g.scalar(gain, .float32));
    if (ops.dtype(cur) != dt) cur = try g.astype(cur, dt);
    return parent.keep(cur);
}

/// Real CFG: uncond + scale · (cond − uncond), the scale an f32 scalar.
fn cfgBlend(parent: *Graph, cond: Array, uncond: Array, scale: f32) !Array {
    var g = parent.scope();
    defer g.deinit();
    return parent.keep(try g.add(uncond, try g.mul(try g.sub(cond, uncond), try g.scalar(scale, .float32))));
}

// ── The text → image pipeline ──

/// FLUX.2's FlowMatchEuler schedule: an empirical mu (linear in the step count below 4300 image tokens), the
/// exponential shift over linspace(1, 1/n, n); timesteps on the ×1000 scale, sigmas ending at 0.
fn computeSchedule(allocator: std.mem.Allocator, image_seq_len: u32, steps: u32) !struct { ts: []f32, sig: []f32 } {
    const ts = try allocator.alloc(f32, steps);
    errdefer allocator.free(ts);
    const sig = try allocator.alloc(f32, steps + 1);
    const isl: f64 = @floatFromInt(image_seq_len);
    const m_200 = 0.00016927 * isl + 0.45666666;
    const m_10 = 8.73809524e-05 * isl + 1.89833333;
    const mu = if (isl > 4300) m_200 else flow_match.muLinear(@floatFromInt(steps), 200, m_200, 10, m_10);
    const emu = std.math.exp(mu);
    for (0..steps) |i| {
        const shifted = flow_match.timeShiftExp(emu, flow_match.gridToInvN(i, steps));
        sig[i] = @floatCast(shifted);
        ts[i] = @floatCast(shifted * 1000.0);
    }
    sig[steps] = 0.0;
    return .{ .ts = ts, .sig = sig };
}

fn buildLatentIds(allocator: std.mem.Allocator, lh: u32, lw: u32) ![]i32 {
    return buildLatentIdsT(allocator, lh, lw, 0);
}

/// Position ids for an edit forward: the generated lh×lw grid at t=0
/// followed by each reference grid (`ref_dims[i] = {rh, rw}`) at
/// t=10·(i+1) — the official FLUX.2 multi-reference convention — each with
/// its own 0-based h/w grid.
fn buildEditIds(allocator: std.mem.Allocator, lh: u32, lw: u32, ref_dims: []const [2]u32) ![]i32 {
    var total: usize = lh * lw;
    for (ref_dims) |d| total += d[0] * d[1];
    const out = try allocator.alloc(i32, total * 4);
    errdefer allocator.free(out);
    const gen_ids = try buildLatentIdsT(allocator, lh, lw, 0);
    defer allocator.free(gen_ids);
    @memcpy(out[0..gen_ids.len], gen_ids);
    var off = gen_ids.len;
    for (ref_dims, 1..) |d, i| {
        const rids = try buildLatentIdsT(allocator, d[0], d[1], @intCast(10 * i));
        defer allocator.free(rids);
        @memcpy(out[off .. off + rids.len], rids);
        off += rids.len;
    }
    return out;
}

/// Latent position ids (t,h,w,l) with an explicit time coordinate. Generated
/// tokens use t=0; reference-image tokens use the official FLUX.2 edit
/// convention t=10·(ref_index+1) with their own 0-based h/w grid.
fn buildLatentIdsT(allocator: std.mem.Allocator, lh: u32, lw: u32, t: i32) ![]i32 {
    const n = lh * lw;
    const out = try allocator.alloc(i32, n * 4);
    var idx: usize = 0;
    for (0..lh) |h| for (0..lw) |wv| {
        out[idx * 4 + 0] = t;
        out[idx * 4 + 1] = @intCast(h);
        out[idx * 4 + 2] = @intCast(wv);
        out[idx * 4 + 3] = 0; // layer
        idx += 1;
    };
    return out;
}
fn buildTextIds(allocator: std.mem.Allocator, seq: u32) ![]i32 {
    const out = try allocator.alloc(i32, seq * 4);
    for (0..seq) |i| {
        out[i * 4 + 0] = 0;
        out[i * 4 + 1] = 0;
        out[i * 4 + 2] = 0;
        out[i * 4 + 3] = @intCast(i);
    }
    return out;
}

/// Generate an image. Returns [1,3,H,W] f32 in [0,1] (owned mlx array).
/// Per-request generation options beyond the core prompt/size/steps.
pub const GenOpts = struct {
    /// img2img: bn-normalized packed latents [1,128,64,64] f32 from
    /// `VaeEncoder.encode`. Mixed with noise at the start sigma.
    init_latents: ?mlx.mlx_array = null,
    /// First schedule index to run (img2img skip; 0 = full schedule).
    start_step: u32 = 0,
    /// Reference-image editing (FLUX.2 in-context conditioning): CLEAN
    /// bn-normalized packed latents [1,128,rh,rw] from `VaeEncoder.encode`,
    /// concatenated on the token sequence with t=10·(i+1) position ids
    /// (official multi-reference convention — "replace the face in image 1
    /// with the face from image 2"). Never noised; generation starts from
    /// pure noise and attends to them. Empty = no references.
    ref_latents: []const mlx.mlx_array = &.{},
    /// Conditioning rebalance: global gain + per-tap weights (len 3).
    cond_gain: f32 = 1.0,
    cond_weights: ?[]const f32 = null,
    /// Classifier-free guidance (the UNDISTILLED "base" klein checkpoints —
    /// distilled klein has guidance baked into the weights and takes neither
    /// field: `guidance_scale` 1.0 skips the unconditional forward entirely,
    /// byte-identical to the no-CFG path). `neg_enc` is the negative prompt's
    /// text-encoder output, same fixed [1,FLUX_SEQ_LEN,hidden] shape as the
    /// positive `enc` so both forwards share `img_ids`/`txt_ids`.
    guidance_scale: f32 = 1.0,
    neg_enc: ?mlx.mlx_array = null,
};

/// Stage 1 alone: the prompt encoded (and rebalanced), MATERIALIZED and owned by the caller. The eval matters: a
/// low-memory host frees the text encoder right after this returns, and an unevaluated graph would pin its weights.
pub fn encodePrompt(te: *TextEncoder, ids: []const i32, mask: []const i32, opts: GenOpts) !mlx.mlx_array {
    var g = Graph.init(te.allocator, te.s);
    defer g.deinit();
    var enc = try te.encode(&g, ids, mask);
    if (opts.cond_gain != 1.0 or opts.cond_weights != null) enc = try applyCondRebalance(&g, enc, opts.cond_gain, opts.cond_weights);
    try ops.eval(enc);
    return g.escape(enc);
}

/// One DiT forward → velocity [1, nlat, 128]; reference tokens ride after the latents and are sliced off.
fn ditVelocity(parent: *Graph, dit: *const Dit, latents: Array, enc: Array, t: f32, img_ids: []const i32, txt_ids: []const i32, ref_tokens: ?Array, all_ids: ?[]i32, nlat: c_int) !Array {
    var g = parent.scope();
    defer g.deinit();
    const r = ref_tokens orelse return parent.keep(try dit.forward(&g, latents, enc, t, img_ids, txt_ids));
    const full = try dit.forward(&g, try g.concat(&.{ latents, r }, 1), enc, t, all_ids.?, txt_ids);
    return parent.keep(try g.sliceAxis(full, 1, 0, nlat));
}

pub fn generateWithOpts(te: *TextEncoder, dit: *Dit, vae: *Vae, ids: []const i32, mask: []const i32, seed: u64, steps: u32, height: u32, width: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
    const enc = try encodePrompt(te, ids, mask, opts);
    return generateFromCondWithOpts(dit, vae, enc, ids.len, seed, steps, height, width, opts, progress);
}

/// Packed latents [1, 128, h, w] → tokens [1, h·w, 128] in bf16.
fn tokensOf(parent: *Graph, lat: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(lat);
    const t = try g.transpose(try g.reshape(lat, &.{ 1, 128, sh[2] * sh[3] }), &.{ 0, 2, 1 });
    return parent.keep(try g.astype(try g.contiguous(t), .bfloat16));
}

/// Stages 2+ (latents → denoise → VAE decode) from a computed prompt encoding: image [1, 3, H, W] f32 in [0, 1],
/// owned by the caller. Takes ownership of `enc_owned` AND `opts.neg_enc`. `prompt_len` drives the text position
/// ids; the negative encoding shares the prompt's fixed padded length.
pub fn generateFromCondWithOpts(dit: *Dit, vae: *Vae, enc_owned: mlx.mlx_array, prompt_len: usize, seed: u64, steps: u32, height: u32, width: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
    const a = dit.allocator;
    var g = Graph.init(a, dit.s);
    defer g.deinit();
    const neg_enc = if (opts.neg_enc) |ne| try g.adopt(ne) else null;
    const enc = try g.adopt(enc_owned);
    const lh = height / 16;
    const lw = width / 16;
    const nlat = lh * lw;

    // Noise [1, 128, lh, lw] from the seed, packed to tokens [1, nlat, 128].
    const key = try g.call(mlx.mlx_random_key, .{seed});
    const nsh = [_]c_int{ 1, 128, @intCast(lh), @intCast(lw) };
    const noise = try g.astype(try g.call(mlx.mlx_random_normal, .{ &nsh, nsh.len, .float32, 0.0, 1.0, key }), .bfloat16);
    var latents = try g.contiguous(try g.transpose(try g.reshape(noise, &.{ 1, 128, @intCast(nlat) }), &.{ 0, 2, 1 }));

    const img_ids = try buildLatentIds(a, lh, lw);
    defer a.free(img_ids);
    const txt_ids = try buildTextIds(a, @intCast(prompt_len));
    defer a.free(txt_ids);

    // Edits: the CLEAN reference latents as tokens at t = 10·(i+1) (the official multi-reference convention), in
    // every forward; only the generated span is stepped.
    var ref_tokens: ?Array = null;
    var all_ids: ?[]i32 = null;
    defer if (all_ids) |ai| a.free(ai);
    if (opts.ref_latents.len > 0) {
        const ref_dims = try a.alloc([2]u32, opts.ref_latents.len);
        defer a.free(ref_dims);
        for (opts.ref_latents, 0..) |rl, ri| {
            const rsh = ops.shape(rl);
            ref_dims[ri] = .{ @intCast(rsh[2]), @intCast(rsh[3]) };
            const rtok = try tokensOf(&g, rl);
            ref_tokens = if (ref_tokens) |rt| try g.concat(&.{ rt, rtok }, 1) else rtok;
        }
        all_ids = try buildEditIds(a, lh, lw, ref_dims);
    }

    const sched = try computeSchedule(a, nlat, steps);
    defer {
        a.free(sched.ts);
        a.free(sched.sig);
    }
    const start_step: u32 = @min(opts.start_step, steps - 1);

    // Variation: x = (1-σ)·z0 + σ·noise at the start sigma (flow-match scale_noise).
    if (opts.init_latents) |z0| {
        const z_bf = try g.astype(try g.transpose(try g.reshape(z0, &.{ 1, 128, @intCast(nlat) }), &.{ 0, 2, 1 }), .bfloat16);
        const sigma = sched.sig[start_step];
        const mixed = try g.add(try g.mul(z_bf, try g.scalar(1.0 - sigma, .float32)), try g.mul(latents, try g.scalar(sigma, .float32)));
        latents = try g.astype(mixed, .bfloat16);
    }

    const use_cfg = opts.guidance_scale != 1.0 and neg_enc != null;
    const run_steps = steps - start_step;
    for (start_step..steps) |t| {
        if (progress) |p| if (p.boundary()) return error.Cancelled;
        var sg = g.scope();
        defer sg.deinit();
        const cond = try ditVelocity(&sg, dit, latents, enc, sched.ts[t], img_ids, txt_ids, ref_tokens, all_ids, @intCast(nlat));
        const v = if (use_cfg)
            try cfgBlend(&sg, cond, try ditVelocity(&sg, dit, latents, neg_enc.?, sched.ts[t], img_ids, txt_ids, ref_tokens, all_ids, @intCast(nlat)), opts.guidance_scale)
        else
            cond;
        const next = try g.keep(try flow_match.eulerStep(&sg, latents, v, sched.sig[t + 1] - sched.sig[t], .float32));
        g.drop(latents);
        latents = next;
        try ops.eval(latents);
        if (progress) |p| p.emit("Generating", @intCast(t + 1 - start_step), run_steps);
    }
    if (progress) |p| p.emit("Decoding image", steps, steps);

    const packed_lat = try g.transpose(try g.reshape(latents, &.{ 1, @intCast(lh), @intCast(lw), 128 }), &.{ 0, 3, 1, 2 });
    const decoded = try g.astype(try vae.decode(&g, packed_lat), .float32);
    // clip(x/2 + 0.5, 0, 1)
    const half = try g.scalar(0.5, .float32);
    const shifted = try g.add(try g.mul(decoded, half), half);
    const out = try g.minimum(try g.maximum(shifted, try g.scalar(0.0, .float32)), try g.scalar(1.0, .float32));
    return g.escape(out);
}

// ── tests ──

const testing = std.testing;

fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

test "flux schedule: one step is the full sigma, finite" {
    const sched = try computeSchedule(testing.allocator, 4096, 1);
    defer testing.allocator.free(sched.ts);
    defer testing.allocator.free(sched.sig);
    try testing.expectEqual(@as(f32, 1.0), sched.sig[0]);
    try testing.expectEqual(@as(f32, 0.0), sched.sig[1]);
    try testing.expectEqual(@as(f32, 1000.0), sched.ts[0]);
}

fn readI32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]i32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(i32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}
fn readF32(io: std.Io, a: std.mem.Allocator, path: []const u8) ![]f32 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(512 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

// Stage 2 oracle: DiT one forward → noise_step0.
//   FLUX_TEST_MODEL, FLUX_LATENTS (latents_init f32), FLUX_PE, FLUX_IMGIDS, FLUX_TXTIDS, FLUX_NOISE0
test "flux DiT reproduces noise_step0" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("FLUX_LATENTS") orelse return error.SkipZigTest);
    const pe_p = std.mem.span(std.c.getenv("FLUX_PE") orelse return error.SkipZigTest);
    const ii_p = std.mem.span(std.c.getenv("FLUX_IMGIDS") orelse return error.SkipZigTest);
    const ti_p = std.mem.span(std.c.getenv("FLUX_TXTIDS") orelse return error.SkipZigTest);
    const n0_p = std.mem.span(std.c.getenv("FLUX_NOISE0") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat = try readF32(io, a, lat_p);
    defer a.free(lat);
    const pe = try readF32(io, a, pe_p);
    defer a.free(pe);
    const ii = try readI32(io, a, ii_p);
    defer a.free(ii);
    const ti = try readI32(io, a, ti_p);
    defer a.free(ti);
    const ref = try readF32(io, a, n0_p);
    defer a.free(ref);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();

    // upload latents [1,4096,128] and pe [1,512,7680] as bf16
    const lat_sh = [_]c_int{ 1, 4096, 128 };
    const lat_f = mlx.mlx_array_new_data(lat.ptr, &lat_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(lat_f);
    const lat_bf = try astype(lat_f, .bfloat16, s);
    defer _ = mlx.mlx_array_free(lat_bf);
    const pe_sh = [_]c_int{ 1, 512, 7680 };
    const pe_f = mlx.mlx_array_new_data(pe.ptr, &pe_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(pe_f);
    const pe_bf = try astype(pe_f, .bfloat16, s);
    defer _ = mlx.mlx_array_free(pe_bf);

    var tg = Graph.init(a, s);
    defer tg.deinit();
    const noise = tg.escape(try dit.forward(&tg, lat_bf, pe_bf, 1000.0, ii, ti));
    defer _ = mlx.mlx_array_free(noise);
    const nf = try astype(noise, .float32, s);
    defer _ = mlx.mlx_array_free(nf);
    _ = mlx.mlx_array_eval(nf);
    const n: usize = @intCast(mlx.mlx_array_size(nf));
    const data = mlx.mlx_array_data_float32(nf) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        dot += @as(f64, data[i]) * ref[i];
        na += @as(f64, data[i]) * data[i];
        nb += @as(f64, ref[i]) * ref[i];
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    std.debug.print("[flux-dit] n={d} corr={d:.6}\n", .{ n, corr });
    try testing.expect(corr > 0.99);
}

// Stage 3 oracle: VAE decode packed_pre_vae -> decoded.
//   FLUX_TEST_MODEL, FLUX_PACKED (packed_pre_vae f32 [1,128,64,64]), FLUX_DECODED
test "flux VAE reproduces decoded image" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const pk_p = std.mem.span(std.c.getenv("FLUX_PACKED") orelse return error.SkipZigTest);
    const dec_p = std.mem.span(std.c.getenv("FLUX_DECODED") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const pk = try readF32(io, a, pk_p);
    defer a.free(pk);
    const ref = try readF32(io, a, dec_p);
    defer a.free(ref);
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const pk_sh = [_]c_int{ 1, 128, 64, 64 };
    const pk_f = mlx.mlx_array_new_data(pk.ptr, &pk_sh, 4, .float32);
    defer _ = mlx.mlx_array_free(pk_f);
    var tg = Graph.init(a, s);
    defer tg.deinit();
    const dec = try vae.decode(&tg, pk_f);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var se: f64 = 0;
    for (0..n) |i| {
        const x: f64 = data[i];
        const y: f64 = ref[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        se += (x - y) * (x - y);
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    const rmse = std.math.sqrt(se / @as(f64, @floatFromInt(n)));
    std.debug.print("[flux-vae] n={d} corr={d:.6} rmse={d:.5}\n", .{ n, corr, rmse });
    try testing.expect(corr > 0.99);
}

// VAE encoder guard: encode a synthetic smooth image, decode it back — the
// pixels must round-trip through the oracle-validated decoder (pins the
// encoder without needing a Python fixture).  FLUX_TEST_MODEL
test "flux VAE encoder round-trips through the decoder" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var enc = try loadVaeEncoder(io, a, s, model_dir);
    defer enc.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const HW: usize = 1024;
    const img_buf = try a.alloc(f32, 3 * HW * HW);
    defer a.free(img_buf);
    for (0..HW) |y| for (0..HW) |x| {
        const fy: f32 = @as(f32, @floatFromInt(y)) / (HW - 1);
        const fx: f32 = @as(f32, @floatFromInt(x)) / (HW - 1);
        img_buf[0 * HW * HW + y * HW + x] = fx;
        img_buf[1 * HW * HW + y * HW + x] = fy;
        img_buf[2 * HW * HW + y * HW + x] = 0.5 * (fx + fy);
    };
    const ish = [_]c_int{ 1, 3, @intCast(HW), @intCast(HW) };
    const img = mlx.mlx_array_new_data(img_buf.ptr, &ish, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    var tg = Graph.init(a, s);
    defer tg.deinit();
    const lat = tg.escape(try enc.encode(&tg, img));
    defer _ = mlx.mlx_array_free(lat);
    const lsh = mlx.getShape(lat);
    try testing.expectEqual(@as(c_int, 128), lsh[1]);
    try testing.expectEqual(@as(c_int, 64), lsh[2]);
    const dec = tg.escape(try vae.decode(&tg, lat)); // [-1,1]
    defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    try testing.expectEqual(img_buf.len, n);
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var mae: f64 = 0;
    for (0..n) |i| {
        const x: f64 = data[i] * 0.5 + 0.5; // decode is [-1,1]
        const y: f64 = img_buf[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
        mae += @abs(x - y);
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    mae /= @floatFromInt(n);
    std.debug.print("[flux-vae-enc] roundtrip corr={d:.6} mae={d:.5}\n", .{ corr, mae });
    try testing.expect(corr > 0.99);
    try testing.expect(mae < 0.05);
}

// Non-square sizes: the reference image in edit mode keeps its own aspect
// ratio, so the VAE encode→decode pair must work off the 1024² grid.
//   FLUX_TEST_MODEL
test "flux VAE encoder round-trips at a non-square size" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var enc = try loadVaeEncoder(io, a, s, model_dir);
    defer enc.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const W: usize = 1152;
    const H: usize = 896;
    const img_buf = try a.alloc(f32, 3 * W * H);
    defer a.free(img_buf);
    for (0..H) |y| for (0..W) |x| {
        const fy: f32 = @as(f32, @floatFromInt(y)) / (H - 1);
        const fx: f32 = @as(f32, @floatFromInt(x)) / (W - 1);
        img_buf[0 * W * H + y * W + x] = fx;
        img_buf[1 * W * H + y * W + x] = fy;
        img_buf[2 * W * H + y * W + x] = 0.5 * (fx + fy);
    };
    const ish = [_]c_int{ 1, 3, @intCast(H), @intCast(W) };
    const img = mlx.mlx_array_new_data(img_buf.ptr, &ish, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    var tg = Graph.init(a, s);
    defer tg.deinit();
    const lat = tg.escape(try enc.encode(&tg, img));
    defer _ = mlx.mlx_array_free(lat);
    const lsh = mlx.getShape(lat);
    try testing.expectEqual(@as(c_int, 128), lsh[1]);
    try testing.expectEqual(@as(c_int, @intCast(H / 16)), lsh[2]);
    try testing.expectEqual(@as(c_int, @intCast(W / 16)), lsh[3]);
    const dec = tg.escape(try vae.decode(&tg, lat)); // [-1,1]
    defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    try testing.expectEqual(img_buf.len, n);
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const x: f64 = data[i] * 0.5 + 0.5;
        const y: f64 = img_buf[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    std.debug.print("[flux-vae-enc] non-square roundtrip corr={d:.6}\n", .{corr});
    try testing.expect(corr > 0.99);
}

test "buildLatentIdsT stamps the reference time coordinate (FLUX.2 edit convention)" {
    const a = testing.allocator;
    // Generated tokens: t=0 (the existing convention).
    const gen_ids = try buildLatentIdsT(a, 2, 3, 0);
    defer a.free(gen_ids);
    try testing.expectEqual(@as(usize, 2 * 3 * 4), gen_ids.len);
    try testing.expectEqual(@as(i32, 0), gen_ids[0]); // t
    try testing.expectEqual(@as(i32, 0), gen_ids[1]); // h
    try testing.expectEqual(@as(i32, 0), gen_ids[2]); // w
    try testing.expectEqual(@as(i32, 0), gen_ids[3]); // l
    // Reference tokens: t=10 (official sampler: t_off = 10·(i+1)), own 0-based grid.
    const ref_ids = try buildLatentIdsT(a, 2, 3, 10);
    defer a.free(ref_ids);
    const last = (2 * 3 - 1) * 4;
    try testing.expectEqual(@as(i32, 10), ref_ids[0]);
    try testing.expectEqual(@as(i32, 10), ref_ids[last + 0]);
    try testing.expectEqual(@as(i32, 1), ref_ids[last + 1]); // h max
    try testing.expectEqual(@as(i32, 2), ref_ids[last + 2]); // w max
    // The legacy builder is the t=0 case.
    const legacy = try buildLatentIds(a, 2, 3);
    defer a.free(legacy);
    try testing.expectEqualSlices(i32, gen_ids, legacy);
}

test "buildEditIds stacks each reference grid at t=10·(i+1) (multi-reference edit)" {
    const a = testing.allocator;
    // Generated 2x2 grid + two references of different grids.
    const dims = [_][2]u32{ .{ 2, 3 }, .{ 1, 2 } };
    const ids = try buildEditIds(a, 2, 2, &dims);
    defer a.free(ids);
    try testing.expectEqual(@as(usize, (4 + 6 + 2) * 4), ids.len);
    // Generated tokens: t=0.
    try testing.expectEqual(@as(i32, 0), ids[0]);
    try testing.expectEqual(@as(i32, 0), ids[(4 - 1) * 4]);
    // First reference: t=10, its OWN 0-based h/w grid.
    const r0 = 4 * 4;
    try testing.expectEqual(@as(i32, 10), ids[r0 + 0]);
    try testing.expectEqual(@as(i32, 0), ids[r0 + 1]);
    try testing.expectEqual(@as(i32, 0), ids[r0 + 2]);
    const r0_last = (4 + 6 - 1) * 4;
    try testing.expectEqual(@as(i32, 10), ids[r0_last + 0]);
    try testing.expectEqual(@as(i32, 1), ids[r0_last + 1]); // h max of 2x3
    try testing.expectEqual(@as(i32, 2), ids[r0_last + 2]); // w max of 2x3
    // Second reference: t=20 (the official multi-reference convention).
    const r1 = (4 + 6) * 4;
    try testing.expectEqual(@as(i32, 20), ids[r1 + 0]);
    try testing.expectEqual(@as(i32, 0), ids[r1 + 1]);
    // Single reference reproduces the legacy gen-ids + t=10 concat exactly.
    const single = try buildEditIds(a, 2, 3, &[_][2]u32{.{ 2, 3 }});
    defer a.free(single);
    const gen_ids = try buildLatentIdsT(a, 2, 3, 0);
    defer a.free(gen_ids);
    const ref_ids = try buildLatentIdsT(a, 2, 3, 10);
    defer a.free(ref_ids);
    try testing.expectEqualSlices(i32, gen_ids, single[0..gen_ids.len]);
    try testing.expectEqualSlices(i32, ref_ids, single[gen_ids.len..]);
}

// Reference-image editing: conditioning the DiT on clean reference tokens must
// change the output vs the same seed without a reference (and produce a valid
// full-size image).  FLUX_TEST_MODEL (steps=2 keeps it quick)
test "flux reference-image conditioning engages and changes the output" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var te = try loadTextEncoder(io, a, s, model_dir);
    defer te.deinit();
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    var venc = try loadVaeEncoder(io, a, s, model_dir);
    defer venc.deinit();

    // Trivial fixed prompt ids (pad token everywhere, short mask) — the test
    // only needs determinism, not a meaningful prompt.
    var ids: [512]i32 = @splat(151643);
    var mask: [512]i32 = @splat(0);
    for (0..8) |i| mask[i] = 1;

    // Reference: smooth gradient encoded to clean latents.
    const HW: usize = 1024;
    const buf = try a.alloc(f32, 3 * HW * HW);
    defer a.free(buf);
    for (0..HW) |y| for (0..HW) |x| {
        buf[0 * HW * HW + y * HW + x] = @as(f32, @floatFromInt(x)) / (HW - 1);
        buf[1 * HW * HW + y * HW + x] = @as(f32, @floatFromInt(y)) / (HW - 1);
        buf[2 * HW * HW + y * HW + x] = 0.5;
    };
    const ish = [_]c_int{ 1, 3, @intCast(HW), @intCast(HW) };
    const img = mlx.mlx_array_new_data(buf.ptr, &ish, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    var tg = Graph.init(a, s);
    defer tg.deinit();
    const ref_lat = tg.escape(try venc.encode(&tg, img));
    defer _ = mlx.mlx_array_free(ref_lat);

    const base = try generateWithOpts(&te, &dit, &vae, &ids, &mask, 42, 2, 1024, 1024, .{}, null);
    defer _ = mlx.mlx_array_free(base);
    const edited = try generateWithOpts(&te, &dit, &vae, &ids, &mask, 42, 2, 1024, 1024, .{ .ref_latents = &.{ref_lat} }, null);
    defer _ = mlx.mlx_array_free(edited);
    const esh = mlx.getShape(edited);
    try testing.expectEqual(@as(c_int, 1024), esh[2]);
    try testing.expectEqual(@as(c_int, 1024), esh[3]);
    // Same seed, same prompt — the reference must be the thing that changed it.
    const bf = try astype(base, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const ef = try astype(edited, .float32, s);
    defer _ = mlx.mlx_array_free(ef);
    _ = mlx.mlx_array_eval(bf);
    _ = mlx.mlx_array_eval(ef);
    const bd = mlx.mlx_array_data_float32(bf) orelse return error.NoData;
    const ed = mlx.mlx_array_data_float32(ef) orelse return error.NoData;
    var diff: f64 = 0;
    const n: usize = @intCast(mlx.mlx_array_size(bf));
    for (0..n) |i| diff += @abs(@as(f64, bd[i]) - ed[i]);
    diff /= @floatFromInt(n);
    std.debug.print("[flux-edit] mean|base-edited|={d:.4}\n", .{diff});
    try testing.expect(diff > 0.01);
}

test "flux applyCondRebalance scales tap thirds and global gain" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    // enc [1,2,6]: hidden H=2 per tap, 3 taps concatenated on the feature axis.
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60 };
    const enc = try g.fromSlice(f32, &vals, &.{ 1, 2, 6 });
    const w = [_]f32{ 2, 3, 4 };
    const out = try applyCondRebalance(&g, enc, 0.5, &w);
    try ops.eval(out);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    // Row 0: [1,2]*2*.5, [3,4]*3*.5, [5,6]*4*.5 = [1,2, 4.5,6, 10,12]
    const expect0 = [_]f32{ 1, 2, 4.5, 6, 10, 12 };
    for (expect0, 0..) |e, i| try testing.expectApproxEqAbs(e, d[i], 1e-5);
    // Gain-only path (weights null) multiplies everything.
    const gained = try applyCondRebalance(&g, enc, 2.0, null);
    try ops.eval(gained);
    const gd = mlx.mlx_array_data_float32(gained) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 2), gd[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 120), gd[11], 1e-5);
    // Wrong count is a hard error (3 taps for FLUX).
    const bad = [_]f32{ 1, 2 };
    try testing.expectError(error.InvalidCondWeights, applyCondRebalance(&g, enc, 1.0, &bad));
}

test "flux CFG blend is uncond + guidance_scale * (cond - uncond)" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const cond_v = [_]f32{ 1, 2, 3, 4 };
    const uncond_v = [_]f32{ 0, 0, 1, 2 };
    const cond = try g.fromSlice(f32, &cond_v, &.{ 1, 1, 4 });
    const uncond = try g.fromSlice(f32, &uncond_v, &.{ 1, 1, 4 });
    const blended = try cfgBlend(&g, cond, uncond, 3.5);
    try ops.eval(blended);
    const d = mlx.mlx_array_data_float32(blended) orelse return error.NoData;
    for (cond_v, uncond_v, 0..) |c, u, i| try testing.expectApproxEqAbs(u + 3.5 * (c - u), d[i], 1e-5);
    // guidance_scale 1.0 collapses the blend to `cond`: the identity that lets CFG-off skip the second forward.
    const one = try cfgBlend(&g, cond, uncond, 1.0);
    try ops.eval(one);
    const d1 = mlx.mlx_array_data_float32(one) orelse return error.NoData;
    for (cond_v, 0..) |c, i| try testing.expectApproxEqAbs(c, d1[i], 1e-5);
}

test "flux GenOpts defaults keep CFG off (distilled klein pays no extra forward)" {
    const opts = GenOpts{};
    try testing.expectEqual(@as(f32, 1.0), opts.guidance_scale);
    try testing.expectEqual(@as(?mlx.mlx_array, null), opts.neg_enc);
}

// End-to-end: full pipeline vs the reference image (and writes a PPM).
//   FLUX_TEST_MODEL, FLUX_IDS, FLUX_MASK, FLUX_DECODED(decoded.raw)
test "flux end-to-end pipeline matches reference image" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("FLUX_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("FLUX_MASK") orelse return error.SkipZigTest);
    const dec_p = std.mem.span(std.c.getenv("FLUX_DECODED") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p);
    defer a.free(ids);
    const mask = try readI32(io, a, mask_p);
    defer a.free(mask);
    const ref = try readF32(io, a, dec_p);
    defer a.free(ref); // [1,3,1024,1024] in [-1,1]

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var te = try loadTextEncoder(io, a, s, model_dir);
    defer te.deinit();
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();

    const img = try generateWithOpts(&te, &dit, &vae, ids, mask, 42, 4, 1024, 1024, .{}, null); // [0,1]
    defer _ = mlx.mlx_array_free(img);
    _ = mlx.mlx_array_eval(img);
    const n: usize = @intCast(mlx.mlx_array_size(img));
    const data = mlx.mlx_array_data_float32(img) orelse return error.NoData;
    // reference is in [-1,1]; convert to [0,1] for comparison
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const x: f64 = data[i];
        const y: f64 = std.math.clamp(@as(f64, ref[i]) / 2.0 + 0.5, 0, 1);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    std.debug.print("[flux-e2e] n={d} corr={d:.5}\n", .{ n, corr });
    try testing.expect(corr > 0.95);
}

// Stage 1 oracle: text encoder prompt_embeds.
//   FLUX_TEST_MODEL, FLUX_IDS (int32 .raw), FLUX_MASK, FLUX_PE (f32 .raw of [1,seq,7680])
test "flux text encoder reproduces prompt_embeds" {
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("FLUX_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("FLUX_MASK") orelse return error.SkipZigTest);
    const pe_p = std.mem.span(std.c.getenv("FLUX_PE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p);
    defer a.free(ids);
    const mask = try readI32(io, a, mask_p);
    defer a.free(mask);
    const ref = try readF32(io, a, pe_p);
    defer a.free(ref);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var te = try loadTextEncoder(io, a, s, model_dir);
    defer te.deinit();

    var tg = Graph.init(a, s);
    defer tg.deinit();
    const pe = tg.escape(try te.encode(&tg, ids, mask));
    defer _ = mlx.mlx_array_free(pe);
    const pe_f = try astype(pe, .float32, s);
    defer _ = mlx.mlx_array_free(pe_f);
    _ = mlx.mlx_array_eval(pe_f);
    const n: usize = @intCast(mlx.mlx_array_size(pe_f));
    const data = mlx.mlx_array_data_float32(pe_f) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    var se: f64 = 0;
    var maxabs: f64 = 0;
    for (0..n) |i| {
        const d = @abs(@as(f64, data[i]) - @as(f64, ref[i]));
        se += d * d;
        maxabs = @max(maxabs, d);
    }
    // Qwen hidden states have massive-activation outliers (|x| up to ~15000),
    // so use correlation, not absolute rmse.
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        const av: f64 = data[i];
        const bv: f64 = ref[i];
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    const corr = dot / (std.math.sqrt(na) * std.math.sqrt(nb));
    const rmse = std.math.sqrt(se / @as(f64, @floatFromInt(n)));
    const rel = rmse / std.math.sqrt(nb / @as(f64, @floatFromInt(n)));
    std.debug.print("[flux-te] n={d} corr={d:.6} rmse={d:.4} rel={d:.5} maxabs={d:.1}\n", .{ n, corr, rmse, rel, maxabs });
    try testing.expect(corr > 0.9995);
}

test "flux geometry is read off the checkpoint, not assumed (klein 4B vs 9B)" {
    // The shapes below are the real ones, read out of the two shipped
    // checkpoints' safetensors headers. mflux's own model configs say the whole
    // 4B→9B delta is six numbers; everything else is shared.
    const four = ditConfigFrom(.{
        .x_embedder_rows = 3072,
        .x_embedder_in_dim = 128,
        .context_in_dim = 7680,
        .norm_q_len = 128,
        .ff_linear_in_rows = 18432,
        .double_blocks = 5,
        .single_blocks = 20,
    }, .{});
    try testing.expectEqual(@as(u32, 3072), four.inner_dim);
    try testing.expectEqual(@as(u32, 24), four.heads);
    try testing.expectEqual(@as(u32, 128), four.head_dim);
    try testing.expectEqual(@as(u32, 5), four.double_layers);
    try testing.expectEqual(@as(u32, 20), four.single_layers);
    try testing.expectEqual(@as(u32, 7680), four.joint_dim);
    try testing.expectEqual(@as(u32, 128), four.in_channels);
    try testing.expectEqual(@as(f32, 3.0), four.mlp_ratio);

    const nine = ditConfigFrom(.{
        .x_embedder_rows = 4096,
        .x_embedder_in_dim = 128,
        .context_in_dim = 12288,
        .norm_q_len = 128,
        .ff_linear_in_rows = 24576,
        .double_blocks = 8,
        .single_blocks = 24,
    }, .{});
    try testing.expectEqual(@as(u32, 4096), nine.inner_dim);
    try testing.expectEqual(@as(u32, 32), nine.heads);
    try testing.expectEqual(@as(u32, 8), nine.double_layers);
    try testing.expectEqual(@as(u32, 24), nine.single_layers);
    try testing.expectEqual(@as(u32, 12288), nine.joint_dim);
    try testing.expectEqual(@as(f32, 3.0), nine.mlp_ratio);
    // Shared across both builds — a deriver that "fixes" these is wrong.
    try testing.expectEqual(four.head_dim, nine.head_dim);
    try testing.expectEqual(four.in_channels, nine.in_channels);
    try testing.expectEqual(four.rope_theta_dit, nine.rope_theta_dit);

    const te4 = teConfigFrom(.{
        .hidden = 2560,
        .q_norm_len = 128,
        .q_proj_rows = 4096,
        .k_proj_rows = 1024,
        .gate_proj_rows = 9728,
        .layers = 36,
        .vocab = 151936,
    }, .{});
    try testing.expectEqual(@as(u32, 2560), te4.te_hidden);
    try testing.expectEqual(@as(u32, 32), te4.te_heads);
    try testing.expectEqual(@as(u32, 8), te4.te_kv);
    try testing.expectEqual(@as(u32, 9728), te4.te_inter);
    try testing.expectEqual(@as(u32, 36), te4.te_layers);

    const te9 = teConfigFrom(.{
        .hidden = 4096,
        .q_norm_len = 128,
        .q_proj_rows = 4096,
        .k_proj_rows = 1024,
        .gate_proj_rows = 12288,
        .layers = 36,
        .vocab = 151936,
    }, .{});
    try testing.expectEqual(@as(u32, 4096), te9.te_hidden);
    try testing.expectEqual(@as(u32, 12288), te9.te_inter);
    // The 9B's Qwen3-8B encoder keeps 32 q heads / 8 kv heads — same as the
    // 4B's. The heads are NOT hidden/head_dim: that would read 32 here by luck
    // and 20 on the 4B, which is wrong.
    try testing.expectEqual(@as(u32, 32), te9.te_heads);
    try testing.expectEqual(@as(u32, 8), te9.te_kv);
    try testing.expectEqual(@as(u32, 36), te9.te_layers);
    // Both DiTs concatenate three tapped encoder layers.
    try testing.expectEqual(3 * te4.te_hidden, four.joint_dim);
    try testing.expectEqual(3 * te9.te_hidden, nine.joint_dim);

    // An unreadable probe (0) leaves the field at its base value rather than
    // writing a zero the loader would then allocate against.
    const empty = ditConfigFrom(.{}, .{});
    try testing.expectEqual(@as(u32, 3072), empty.inner_dim);
    try testing.expectEqual(@as(u32, 5), empty.double_layers);
    const empty_te = teConfigFrom(.{}, .{});
    try testing.expectEqual(@as(u32, 2560), empty_te.te_hidden);
    try testing.expectEqual(@as(u32, 36), empty_te.te_layers);
}

test "flux inferQuantGeometry resolves mflux/mlx-community bit widths" {
    // in = 3072 logical columns at gs = 64 → s_cols = 48; w_cols = in·bits/32.
    try testing.expectEqual(@as(u32, 3), inferQuantGeometry(288, 48).bits); // 3-bit (8 GB iPhones)
    try testing.expectEqual(@as(u32, 64), inferQuantGeometry(288, 48).group_size);
    try testing.expectEqual(@as(u32, 4), inferQuantGeometry(384, 48).bits);
    try testing.expectEqual(@as(u32, 5), inferQuantGeometry(480, 48).bits);
    try testing.expectEqual(@as(u32, 6), inferQuantGeometry(576, 48).bits);
    try testing.expectEqual(@as(u32, 8), inferQuantGeometry(768, 48).bits);
    // gs-128 conversion of the same 4-bit width: s_cols = 24 → product 512;
    // 512/64 = 8 valid → convention prefers 64 (correct for every catalog
    // repo; a real gs-128 build would need explicit metadata to win here).
    try testing.expectEqual(@as(u32, 8), inferQuantGeometry(384, 24).bits);
    // Degenerate/odd geometry falls back to the historical 4/64.
    try testing.expectEqual(@as(u32, 4), inferQuantGeometry(0, 0).bits);
    try testing.expectEqual(@as(u32, 4), inferQuantGeometry(383, 48).bits);
}

test "flux QLinear with inferred sub-4-bit geometry matches the dequantized reference" {
    // Quantize a synthetic weight at each shipped bit width, build a QLinear
    // through the same inference the loader uses, and require its forward to
    // match dense matmul against the dequantized weight. Catches both a wrong
    // (bits, gs) inference and any mlx kernel gap at that width.
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const out_f = 64;
    const in_f = 128;
    var wbuf: [out_f * in_f]f32 = undefined;
    var state: u64 = 0x2545F4914F6CDD1D;
    for (&wbuf) |*v| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        v.* = @as(f32, @floatFromInt(@as(i32, @bitCast(@as(u32, @truncate(state >> 33)))))) / 2.147e9;
    }
    const wsh = [_]c_int{ out_f, in_f };
    const wdense = mlx.mlx_array_new_data(&wbuf, &wsh, 2, .float32);
    defer _ = mlx.mlx_array_free(wdense);

    var xbuf: [in_f]f32 = undefined;
    for (&xbuf) |*v| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        v.* = @as(f32, @floatFromInt(@as(i32, @bitCast(@as(u32, @truncate(state >> 33)))))) / 2.147e9;
    }
    const xsh = [_]c_int{ 1, in_f };
    const x = mlx.mlx_array_new_data(&xbuf, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(x);

    for ([_]u8{ 3, 4, 6, 8 }) |bits| {
        var vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_quantize(
            &vec,
            wdense,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(bits),
            "affine",
            .{},
            s,
        ));
        var ql: QLinear = .{ .w = mlx.mlx_array_new(), .scales = mlx.mlx_array_new(), .biases = mlx.mlx_array_new() };
        defer ql.deinit();
        try mlx.check(mlx.mlx_vector_array_get(&ql.w, vec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&ql.scales, vec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&ql.biases, vec, 2));

        const geo = inferQuantGeometryOf(ql.w, ql.scales);
        try testing.expectEqual(@as(u32, bits), geo.bits);
        try testing.expectEqual(@as(u32, 64), geo.group_size);
        ql.bits = geo.bits;
        ql.group_size = geo.group_size;

        var tg = Graph.init(testing.allocator, s);
        defer tg.deinit();
        const got = tg.escape(try ql.forward(&tg, x));
        defer _ = mlx.mlx_array_free(got);

        // Reference: x @ dequantize(w)ᵀ.
        const wref = try dequantTable(ql.w, ql.scales, ql.biases, geo.bits, geo.group_size, s);
        defer _ = mlx.mlx_array_free(wref);
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        const axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wt, wref, &axes, 2, s));
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_matmul(&ref, x, wt, s));

        _ = mlx.mlx_array_eval(got);
        _ = mlx.mlx_array_eval(ref);
        var gf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gf);
        var rf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rf);
        try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
        try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
        _ = mlx.mlx_array_eval(gf);
        _ = mlx.mlx_array_eval(rf);
        const gd = mlx.mlx_array_data_float32(gf) orelse return error.NoData;
        const rd = mlx.mlx_array_data_float32(rf) orelse return error.NoData;
        var dot: f64 = 0;
        var ng: f64 = 0;
        var nr: f64 = 0;
        for (0..out_f) |i| {
            dot += @as(f64, gd[i]) * rd[i];
            ng += @as(f64, gd[i]) * gd[i];
            nr += @as(f64, rd[i]) * rd[i];
        }
        const cos = dot / (@max(std.math.sqrt(ng * nr), 1e-12));
        std.debug.print("[flux-qgeo] bits={d} cos={d:.6}\n", .{ bits, cos });
        try testing.expect(cos > 0.999);
    }
}

/// One DiT forward on seeded noise (32 x 32 latent tokens + 64 text rows), noise prediction as f32;
/// `resident` receives the MLX bytes the loaded DiT holds.
fn fluxProbe(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, resident: *usize) !mlx.mlx_array {
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();
    const img = try w8a8.randn(&.{ 1, 1024, @intCast(dit.cfg.in_channels) }, 1, .bfloat16, s);
    defer _ = mlx.mlx_array_free(img);
    const enc = try w8a8.randn(&.{ 1, 64, @intCast(dit.cfg.joint_dim) }, 2, .bfloat16, s);
    defer _ = mlx.mlx_array_free(enc);
    const img_ids = try buildLatentIds(a, 32, 32);
    defer a.free(img_ids);
    const txt_ids = try buildTextIds(a, 64);
    defer a.free(txt_ids);
    var tg = Graph.init(a, s);
    defer tg.deinit();
    const out = tg.escape(try dit.forward(&tg, img, enc, 0.5, img_ids, txt_ids));
    defer _ = mlx.mlx_array_free(out);
    const f = try astype(out, .float32, s);
    try mlx.check(mlx.mlx_array_eval(f));
    // Weights are lazy until the first forward has read them.
    w8a8.dropMemo(); // job-end state: the last activation is not weights
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    resident.* = after -| before;
    return f;
}

// The lossy int8 route against the same pack's packed forward. FLUX_TEST_MODEL.
test "FLUX DiT: the w8a8 route stays within its bar of the packed forward" {
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const model_dir = std.mem.span(std.c.getenv("FLUX_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const prev = w8a8.w8a8State().enabled;
    defer w8a8.w8a8State().enabled = prev;
    w8a8.w8a8State().enabled = false;
    var packed_bytes: usize = 0;
    const want = try fluxProbe(io, a, s, model_dir, &packed_bytes);
    defer _ = mlx.mlx_array_free(want);
    w8a8.w8a8State().enabled = true;
    var int8_bytes: usize = 0;
    const got = try fluxProbe(io, a, s, model_dir, &int8_bytes);
    defer _ = mlx.mlx_array_free(got);
    const g = try w8a8.agreement(got, want, s);
    std.debug.print("[flux dit w8a8] cos={d:.6} rms_ratio={d:.6} resident packed={d} int8={d}\n", .{ g.cos, g.rms_ratio, packed_bytes, int8_bytes });
    try testing.expect(g.cos < 0.99999); // the route engaged: a packed-numerics output would be identical
    try testing.expect(g.cos > 0.998);
    try testing.expectApproxEqAbs(@as(f64, 1.0), g.rms_ratio, 0.02);
    try w8a8.expectBilled(packed_bytes, int8_bytes, try w8a8.billedDelta(io, model_dir, &.{ "transformer/0.safetensors", "transformer/1.safetensors" }));
}
