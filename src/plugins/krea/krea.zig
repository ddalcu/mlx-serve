//! Native Krea-2-Turbo text→image backend (single-stream MMDiT + Qwen3-VL-4B
//! text encoder + Qwen-Image 3D causal VAE + flow-match Euler sampler), ported
//! from the pure-MLX reference `avlp12/krea2_alis_mlx`.
//!
//! Mixed precision: the reference quantizes ONLY the 28 main blocks' attn+mlp
//! linears (affine, group_size 64); everything else stays bf16. There is no
//! per-tensor bit map in any config — `MixedLinear` infers (bits, group_size)
//! from tensor geometry, so ONE engine loads all three builds (8bit / mixed-4-8
//! / bf16). The VAE runs in f16; encoder + DiT run in bf16.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const sse = @import("../../sdk/media/progress.zig");
const png = @import("../../sdk/media/png.zig");
const tok_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const lora_mod = @import("../../sdk/weights/lora.zig");
const ane = @import("../../sdk/ane/offload.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");
const dit_kernels = @import("../../sdk/blocks/dit/dit_kernels.zig");
const int8_attn = @import("../../sdk/blocks/dit/int8_attn.zig");
const qwen3 = @import("../../sdk/blocks/text_encoders/qwen3.zig");
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");
const modulate_blk = @import("../../sdk/blocks/dit/modulate.zig");
const timestep_blk = @import("../../sdk/blocks/dit/timestep.zig");
const gelu = @import("../../sdk/blocks/mlp/gelu.zig");
const conv_blk = @import("../../sdk/blocks/vae/conv.zig");
const resnet_blk = @import("../../sdk/blocks/vae/resnet.zig");
const flow_match = @import("../../sdk/blocks/schedulers/flow_match.zig");
const dit_base = @import("../../sdk/bases/dit.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const Graph = ops.Graph;
const Array = ops.Array;
const Resnet = resnet_blk.Resnet;

// ── Load-time weight preparation (the forward runs on `ops.Graph`) ──

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

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[krea] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingKreaWeight;
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

// ════════════════════════════════════════════════════════════════════════
// B0. MixedLinear — bf16 OR affine-quantized, bits/group_size inferred from
//     tensor geometry. The single primitive that loads all three builds.
// ════════════════════════════════════════════════════════════════════════

const MixedLinear = struct {
    quantized: bool,
    w: mlx.mlx_array, // quantized: packed u32 [out, in*bits/32]; bf16: pre-transposed [in,out]
    scales: mlx.mlx_array = .{ .ctx = null },
    biases: mlx.mlx_array = .{ .ctx = null },
    add_bias: ?mlx.mlx_array = null,
    bits: u32 = 0,
    group_size: u32 = 0,
    // Runtime adapters (non-owning; the plugin's lora.Stack owns the arrays).
    // Fixed-capacity so attach/forward never allocate.
    lora_refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
    lora_count: u8 = 0,
    /// The int8 copy that REPLACES `w`/`scales`/`biases` (`--w8a8`), built by `toW8a8`.
    w8: ?w8a8.Weight = null,

    /// `in_features` is the module's input dim (known per call); used only on
    /// the quantized path to solve (bits, group_size) from packed geometry.
    fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !MixedLinear {
        const wk = try fmtKey(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try fmtKey(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try fmtKey(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const ak = try fmtKey(a, "{s}.bias", .{prefix});
        defer a.free(ak);

        if (ownOpt(w, sk)) |scales| {
            // Quantized: infer bits & group_size from geometry.
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
        // bf16: pre-transpose [out,in] → [in,out], materialize, cast bf16.
        const raw = try ownWeight(w, wk);
        defer _ = mlx.mlx_array_free(raw);
        const t = try transpose(raw, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(t);
        const tc = try contig(t, s);
        defer _ = mlx.mlx_array_free(tc);
        const wt = try astype(tc, .bfloat16, s);
        return .{ .quantized = false, .w = wt, .add_bias = ownOpt(w, ak) };
    }

    fn deinit(self: *MixedLinear) void {
        if (self.w8) |*q| q.deinit();
        _ = mlx.mlx_array_free(self.w);
        if (self.quantized) {
            _ = mlx.mlx_array_free(self.scales);
            _ = mlx.mlx_array_free(self.biases);
        }
        if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
    }

    pub fn forward(self: *const MixedLinear, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const xb = try g.astype(x, .bfloat16); // reference computes in bf16
        var o = if (self.w8) |*q| try g.adopt(try w8a8.forward(xb, q, g.s)) else try self.matmul(&g, xb);
        if (self.add_bias) |b| o = try g.add(o, b);
        return parent.keep(try self.addLora(&g, o, xb));
    }

    /// `o` plus every attached adapter's delta on `xb`.
    fn addLora(self: *const MixedLinear, parent: *Graph, o: Array, xb: Array) !Array {
        if (self.lora_count == 0) return o;
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(try g.add(o, try g.adopt(try lora_mod.deltaSum(xb, self.lora_refs[0..self.lora_count], g.s))));
    }

    fn matmul(self: *const MixedLinear, g: *Graph, xb: Array) !Array {
        if (self.w.ctx == null) return error.WeightReleased;
        if (self.quantized) return g.qmm(xb, self.w, self.scales, self.biases, @intCast(self.group_size), @intCast(self.bits));
        return g.matmul(xb, self.w);
    }

    fn outFeatures(self: *const MixedLinear) u32 {
        if (self.w8) |q| return @intCast(q.n);
        return @intCast(if (self.quantized) mlx.getShape(self.w)[0] else mlx.getShape(self.w)[1]);
    }

    /// Replace the weights with their int8 copy when the route takes this shape.
    fn toW8a8(self: *MixedLinear, s: S) !void {
        self.w8 = try w8a8.takeLinear(&self.w, &self.scales, &self.biases, self.bits, self.group_size, s);
    }

    /// Install the stacked adapter Refs for this linear (from `Stack.findAll`).
    fn setLoraRefs(self: *MixedLinear, refs: []const lora_mod.Ref) void {
        self.lora_count = @intCast(refs.len);
        @memcpy(self.lora_refs[0..refs.len], refs);
    }

    fn clearLoraRefs(self: *MixedLinear) void {
        self.lora_count = 0;
    }
};

// ── Config ──

/// Defaults are the Krea-2-Turbo release; the widths and counts are read off the checkpoint at load
/// (`fromWeights`), the sinusoid width, patch, latent channels, RoPE theta and the 12 encoder taps are fixed.
pub const KreaConfig = struct {
    // DiT
    features: u32 = 6144,
    tdim: u32 = 256,
    txtdim: u32 = 2560,
    heads: u32 = 48,
    kvheads: u32 = 12,
    /// SwiGLU hidden widths: roundup(int(2·dim/3)·4, 128) in the release.
    mlpdim: u32 = 16384,
    txt_mlpdim: u32 = 6912,
    layers: u32 = 28,
    patch: u32 = 2,
    channels: u32 = 16,
    theta: f32 = 1000.0,
    txtheads: u32 = 20,
    txtlayers: u32 = 12,
    // text encoder (Qwen3-VL-4B)
    te_hidden: u32 = 2560,
    te_layers: u32 = 36,
    te_heads: u32 = 32,
    te_kv: u32 = 8,
    te_head_dim: u32 = 128,
    te_inter: u32 = 9728,
    te_theta: f32 = 5_000_000.0,
    te_eps: f32 = 1e-6,

    fn headDim(self: KreaConfig) u32 {
        return self.features / self.heads; // 128
    }

    /// The DiT's widths from its weights (output rows, quantized or not); an unreadable one keeps its default.
    fn ditFrom(w: *const Weights, a: std.mem.Allocator) KreaConfig {
        var c: KreaConfig = .{};
        set(&c.features, rowsOf(w, "first.weight"));
        const hd = rowsOf(w, "blocks.0.attn.qknorm.qnorm.scale");
        if (hd != 0) {
            c.heads = c.features / hd;
            set(&c.kvheads, rowsOf(w, "blocks.0.attn.wk.weight") / hd);
        }
        set(&c.mlpdim, rowsOf(w, "blocks.0.mlp.gate.weight"));
        set(&c.layers, countIndexed(w, a, "blocks.{d}.mod.lin"));
        set(&c.txtdim, rowsOf(w, "txtfusion.layerwise_blocks.0.prenorm.scale"));
        const thd = rowsOf(w, "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale");
        if (thd != 0) c.txtheads = c.txtdim / thd;
        set(&c.txt_mlpdim, rowsOf(w, "txtfusion.layerwise_blocks.0.mlp.gate.weight"));
        return c;
    }

    /// The encoder's widths; heads from `q_proj` rows over the q-norm width.
    fn encoderFrom(w: *const Weights, a: std.mem.Allocator, pfx: []const u8) KreaConfig {
        var c: KreaConfig = .{};
        var buf: [128]u8 = undefined;
        set(&c.te_hidden, rowsOf(w, std.fmt.bufPrint(&buf, "{s}layers.0.input_layernorm.weight", .{pfx}) catch ""));
        set(&c.te_head_dim, rowsOf(w, std.fmt.bufPrint(&buf, "{s}layers.0.self_attn.q_norm.weight", .{pfx}) catch ""));
        set(&c.te_heads, rowsOf(w, std.fmt.bufPrint(&buf, "{s}layers.0.self_attn.q_proj.weight", .{pfx}) catch "") / c.te_head_dim);
        set(&c.te_kv, rowsOf(w, std.fmt.bufPrint(&buf, "{s}layers.0.self_attn.k_proj.weight", .{pfx}) catch "") / c.te_head_dim);
        set(&c.te_inter, rowsOf(w, std.fmt.bufPrint(&buf, "{s}layers.0.mlp.gate_proj.weight", .{pfx}) catch ""));
        set(&c.te_layers, if (pfx.len == 0) countIndexed(w, a, "layers.{d}.input_layernorm.weight") else countIndexed(w, a, "language_model.layers.{d}.input_layernorm.weight"));
        return c;
    }

    fn set(field: *u32, v: u32) void {
        if (v != 0) field.* = v;
    }
};

/// First-axis extent of a weight, or 0 when absent.
fn rowsOf(w: *const Weights, key: []const u8) u32 {
    const x = w.get(key) orelse return 0;
    const sh = mlx.getShape(x);
    if (sh.len == 0 or sh[0] < 0) return 0;
    return @intCast(sh[0]);
}

/// How many consecutive indices from 0 name a present weight (bounded).
fn countIndexed(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8) u32 {
    var n: u32 = 0;
    while (n < 256) : (n += 1) {
        const key = fmtKey(a, fmt, .{n}) catch return n;
        defer a.free(key);
        if (w.get(key) == null) return n;
    }
    return n;
}

// Prompt template (must match text_encoder.py exactly).
const PREFIX =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, " ++
    "texture, quantity, text, spatial relationships of the objects and background:" ++
    "<|im_end|>\n<|im_start|>user\n";
const SUFFIX = "<|im_end|>\n<|im_start|>assistant\n";
const PREFIX_START_IDX: usize = 34;
const SUFFIX_START_IDX: usize = 5;
const MAX_LENGTH: usize = 512; // output text tokens
const PAD_TOKEN: i32 = 151643; // Qwen <|endoftext|>
// HF hidden_states tapped (input-to-layer i; i<36 so never the final norm).
const SELECT_LAYERS = [_]usize{ 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 };

// ════════════════════════════════════════════════════════════════════════
// B1. Conditioner — Qwen3-VL-4B text encoder (12-layer capture), bf16.
// ════════════════════════════════════════════════════════════════════════

const TeLayer = struct {
    input_ln: mlx.mlx_array,
    post_ln: mlx.mlx_array,
    q: MixedLinear,
    k: MixedLinear,
    v: MixedLinear,
    o: MixedLinear,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    gate: MixedLinear,
    up: MixedLinear,
    down: MixedLinear,
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

pub const Conditioner = struct {
    cfg: KreaConfig,
    allocator: std.mem.Allocator,
    s: S,
    embed_table: mlx.mlx_array, // [vocab, hidden] bf16
    layers: []TeLayer,

    pub fn deinit(self: *Conditioner) void {
        _ = mlx.mlx_array_free(self.embed_table);
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
    }

    /// ids/mask are the FULL templated sequence (PREFIX+prompt padded + SUFFIX,
    /// length L0 = MAX_LENGTH + PREFIX_START_IDX). Returns the post-prefix slice
    /// stacked over 12 tapped layers: embeds [1, MAX_LENGTH, 12, hidden] bf16.
    pub fn encode(self: *Conditioner, parent: *Graph, ids: []const i32, mask: []const i32) !Array {
        var g = parent.scope();
        defer g.deinit();
        const c = self.cfg;
        const seq: c_int = @intCast(ids.len);
        const H: c_int = @intCast(c.te_hidden);
        var x = try g.reshape(try g.take(self.embed_table, try g.fromSlice(i32, ids, &.{seq}), 0), &.{ 1, seq, H });
        const attn_mask = try causalMask(&g, self.allocator, mask);
        // Standard rope cos/sin [seq, head_dim] in f32 ROUNDED to bf16 (the reference casts them to the
        // activations' dtype; mlx_fast_rope's f32-internal rotation misses parity).
        const rope = try encoderRope(&g, self.allocator, @intCast(seq), c.te_head_dim, c.te_theta);
        const cfg: qwen3.Cfg = .{ .heads = @intCast(c.te_heads), .kv_heads = @intCast(c.te_kv), .head_dim = @intCast(c.te_head_dim), .eps = c.te_eps };
        const table: qwen3.Rope = .{ .table = .{ .cos = rope.cos, .sin = rope.sin, .neg = .bfloat16 } };

        // HF output_hidden_states: each tap is the input of its layer, [1, seq, 1, H].
        var caps: [SELECT_LAYERS.len]Array = undefined;
        var ncap: usize = 0;
        for (self.layers, 0..) |*layer, li| {
            const tapped = std.mem.indexOfScalar(usize, &SELECT_LAYERS, li) != null;
            if (tapped) {
                caps[ncap] = try g.expandDims(x, 2);
                ncap += 1;
            }
            const nx = try qwen3.layerForward(&g, layer, cfg, table, x, attn_mask);
            if (!tapped) g.drop(x);
            x = nx;
        }
        const stacked = try g.concat(caps[0..ncap], 2);
        // drop the prefix tokens
        return parent.keep(try g.sliceAxis(stacked, 1, @intCast(PREFIX_START_IDX), seq));
    }
};

/// Causal + padding additive mask [1,1,seq,seq] bf16 (-1e9 on blocked).
fn causalMask(g: *Graph, allocator: std.mem.Allocator, mask: []const i32) !Array {
    const n = mask.len;
    const buf = try allocator.alloc(f32, n * n);
    defer allocator.free(buf);
    for (0..n) |i| for (0..n) |j| {
        buf[i * n + j] = if (j > i or mask[j] == 0) -1e9 else 0.0;
    };
    const seq: c_int = @intCast(n);
    var c = g.scope();
    defer c.deinit();
    return g.keep(try c.astype(try c.fromSlice(f32, buf, &.{ 1, 1, seq, seq }), .bfloat16));
}

/// Standard HF rope cos/sin [1, 1, L, head_dim] for the encoder: emb = concat([freqs, freqs]), computed in f32
/// then ROUNDED to bf16 (the reference casts cos/sin to h.dtype before applying).
fn encoderRope(g: *Graph, allocator: std.mem.Allocator, L: usize, hd: usize, theta: f32) !struct { cos: Array, sin: Array } {
    const cosb = try allocator.alloc(f32, L * hd);
    defer allocator.free(cosb);
    const sinb = try allocator.alloc(f32, L * hd);
    defer allocator.free(sinb);
    const half = hd / 2;
    const th: f64 = theta;
    for (0..L) |p| {
        for (0..half) |i| {
            const inv = std.math.pow(f64, th, -@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(hd)));
            const ang = @as(f64, @floatFromInt(p)) * inv;
            const cv: f32 = @floatCast(@cos(ang));
            const sv: f32 = @floatCast(@sin(ang));
            cosb[p * hd + i] = cv;
            cosb[p * hd + i + half] = cv;
            sinb[p * hd + i] = sv;
            sinb[p * hd + i + half] = sv;
        }
    }
    const sh = [_]c_int{ 1, 1, @intCast(L), @intCast(hd) };
    var c = g.scope();
    defer c.deinit();
    return .{
        .cos = try g.keep(try c.astype(try c.fromSlice(f32, cosb, &sh), .bfloat16)),
        .sin = try g.keep(try c.astype(try c.fromSlice(f32, sinb, &sh), .bfloat16)),
    };
}

pub fn loadConditioner(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Conditioner {
    const dir = try fmtKey(allocator, "{s}/text_encoder", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var te: Conditioner = undefined;
    te.allocator = allocator;
    te.s = s;

    // Qwen3-VL text weights live under a `language_model.` prefix in the
    // checkpoint (the reference strips it at load); detect it and prefix every
    // key. Some flat exports omit it — fall back to no prefix.
    const pfx: []const u8 = if (w.get("language_model.embed_tokens.weight") != null) "language_model." else "";
    te.cfg = KreaConfig.encoderFrom(&w, allocator, pfx);

    const ek = try fmtKey(allocator, "{s}embed_tokens.weight", .{pfx});
    defer allocator.free(ek);
    te.embed_table = try ownWeight(&w, ek);
    {
        const bf = try astype(te.embed_table, .bfloat16, s);
        _ = mlx.mlx_array_free(te.embed_table);
        te.embed_table = bf;
    }

    te.layers = try allocator.alloc(TeLayer, te.cfg.te_layers);
    const H = te.cfg.te_hidden;
    const inter = te.cfg.te_inter;
    for (te.layers, 0..) |*layer, i| {
        const p_in = try fmtKey(allocator, "{s}layers.{d}.input_layernorm.weight", .{ pfx, i });
        defer allocator.free(p_in);
        const p_post = try fmtKey(allocator, "{s}layers.{d}.post_attention_layernorm.weight", .{ pfx, i });
        defer allocator.free(p_post);
        const qn = try fmtKey(allocator, "{s}layers.{d}.self_attn.q_norm.weight", .{ pfx, i });
        defer allocator.free(qn);
        const kn = try fmtKey(allocator, "{s}layers.{d}.self_attn.k_norm.weight", .{ pfx, i });
        defer allocator.free(kn);
        const qp = try fmtKey(allocator, "{s}layers.{d}.self_attn.q_proj", .{ pfx, i });
        defer allocator.free(qp);
        const kp = try fmtKey(allocator, "{s}layers.{d}.self_attn.k_proj", .{ pfx, i });
        defer allocator.free(kp);
        const vp = try fmtKey(allocator, "{s}layers.{d}.self_attn.v_proj", .{ pfx, i });
        defer allocator.free(vp);
        const op = try fmtKey(allocator, "{s}layers.{d}.self_attn.o_proj", .{ pfx, i });
        defer allocator.free(op);
        const gp = try fmtKey(allocator, "{s}layers.{d}.mlp.gate_proj", .{ pfx, i });
        defer allocator.free(gp);
        const upp = try fmtKey(allocator, "{s}layers.{d}.mlp.up_proj", .{ pfx, i });
        defer allocator.free(upp);
        const dp = try fmtKey(allocator, "{s}layers.{d}.mlp.down_proj", .{ pfx, i });
        defer allocator.free(dp);
        layer.* = .{
            .input_ln = try normF32(&w, p_in, s),
            .post_ln = try normF32(&w, p_post, s),
            .q = try MixedLinear.load(&w, allocator, qp, H, s),
            .k = try MixedLinear.load(&w, allocator, kp, H, s),
            .v = try MixedLinear.load(&w, allocator, vp, H, s),
            .o = try MixedLinear.load(&w, allocator, op, te.cfg.te_heads * te.cfg.te_head_dim, s),
            .q_norm = try normF32(&w, qn, s),
            .k_norm = try normF32(&w, kn, s),
            .gate = try MixedLinear.load(&w, allocator, gp, H, s),
            .up = try MixedLinear.load(&w, allocator, upp, H, s),
            .down = try MixedLinear.load(&w, allocator, dp, inter, s),
        };
    }
    return te;
}

/// Load a `.weight`-convention RMSNorm vector (no +1), cast bf16.
fn normBf16(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .bfloat16, s);
}

/// Load a `.weight`-convention RMSNorm vector as f32. The reference Qwen3RMSNorm
/// multiplies by `weight.astype(f32)`; keeping the encoder norm weights in f32
/// (mlx_fast_rms_norm upcasts the bf16 activations internally) matches that and
/// closes the late-layer outlier divergence a bf16 weight introduces.
fn normF32(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, .float32, s);
}

// ════════════════════════════════════════════════════════════════════════
// B2. Denoiser — SingleStreamDiT (28-block single-stream MMDiT).
// ════════════════════════════════════════════════════════════════════════

/// Build interleaved 3-axis RoPE cos/sin [L,64] (f32) from pos [L*3] (i32).
/// axes = [32,48,48] → half-counts [16,24,24]; theta=1000.
fn buildRope(g: *Graph, allocator: std.mem.Allocator, pos: []const i32, L: usize, theta: f32) !struct { cos: Array, sin: Array } {
    const cosb = try allocator.alloc(f32, L * 64);
    defer allocator.free(cosb);
    const sinb = try allocator.alloc(f32, L * 64);
    defer allocator.free(sinb);
    const axes = [_]usize{ 32, 48, 48 };
    const th: f64 = theta;
    for (0..L) |p| {
        var kk: usize = 0;
        for (0..3) |ai| {
            const d = axes[ai];
            const pv: f64 = @floatFromInt(pos[p * 3 + ai]);
            var j: usize = 0;
            while (j < d / 2) : (j += 1) {
                const scale = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d));
                const omega = std.math.pow(f64, th, -scale);
                const ang = pv * omega;
                cosb[p * 64 + kk] = @floatCast(@cos(ang));
                sinb[p * 64 + kk] = @floatCast(@sin(ang));
                kk += 1;
            }
        }
    }
    const shape = [_]c_int{ @intCast(L), 64 };
    return .{ .cos = try g.fromSlice(f32, cosb, &shape), .sin = try g.fromSlice(f32, sinb, &shape) };
}

const Attention = struct {
    wq: MixedLinear,
    wk: MixedLinear,
    wv: MixedLinear,
    gate: MixedLinear,
    wo: MixedLinear,
    qnorm: mlx.mlx_array, // (1+scale) bf16
    knorm: mlx.mlx_array,
    heads: c_int,
    kvheads: c_int,
    head_dim: c_int,
    /// q, k, v and gate rows in one int8 weight (`--w8a8`): they read one input,
    /// and the four linears are views of it.
    qkvg: ?w8a8.Weight = null,

    fn deinit(self: *Attention) void {
        if (self.qkvg) |*q| q.deinit();
        self.wq.deinit();
        self.wk.deinit();
        self.wv.deinit();
        self.gate.deinit();
        self.wo.deinit();
        _ = mlx.mlx_array_free(self.qnorm);
        _ = mlx.mlx_array_free(self.knorm);
    }

    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, dim: u32, heads: u32, kvheads: u32, head_dim: u32, s: S) !Attention {
        const kq = try fmtKey(a, "{s}.wq", .{pfx});
        defer a.free(kq);
        const kk = try fmtKey(a, "{s}.wk", .{pfx});
        defer a.free(kk);
        const kv = try fmtKey(a, "{s}.wv", .{pfx});
        defer a.free(kv);
        const kg = try fmtKey(a, "{s}.gate", .{pfx});
        defer a.free(kg);
        const ko = try fmtKey(a, "{s}.wo", .{pfx});
        defer a.free(ko);
        const qn = try fmtKey(a, "{s}.qknorm.qnorm.scale", .{pfx});
        defer a.free(qn);
        const kn = try fmtKey(a, "{s}.qknorm.knorm.scale", .{pfx});
        defer a.free(kn);
        return .{
            .wq = try MixedLinear.load(w, a, kq, dim, s),
            .wk = try MixedLinear.load(w, a, kk, dim, s),
            .wv = try MixedLinear.load(w, a, kv, dim, s),
            .gate = try MixedLinear.load(w, a, kg, dim, s),
            .wo = try MixedLinear.load(w, a, ko, dim, s),
            .qnorm = try kreaNorm(w, qn, s),
            .knorm = try kreaNorm(w, kn, s),
            .heads = @intCast(heads),
            .kvheads = @intCast(kvheads),
            .head_dim = @intCast(head_dim),
        };
    }

    /// Fuse the four int8 projections into one. A projection with a bias keeps its own.
    fn fuseW8a8(self: *Attention, s: S) !void {
        if (self.wq.add_bias != null or self.wk.add_bias != null or self.wv.add_bias != null or self.gate.add_bias != null) return;
        self.qkvg = try w8a8.fuse(&.{ &self.wq.w8, &self.wk.w8, &self.wv.w8, &self.gate.w8 }, .stack, s);
    }

    const Heads = struct { q: Array, k: Array, v: Array, gate: Array };

    /// One int8 GEMM for q, k, v and gate plus one norm+RoPE kernel; null when
    /// the fused weight is absent or the call is outside it. `cos`/`sin` [L, 64].
    fn fusedHeads(self: *const Attention, g: *Graph, x: Array, cos: ?Array, sin: ?Array) !?Heads {
        const f = if (self.qkvg) |*q| q else return null;
        // An adapter at a nonzero scale needs the four projections apart; a strength-0 stack keeps the fused bytes.
        const lora = lora_mod.anyActive(self.wq.lora_refs[0..self.wq.lora_count]) or lora_mod.anyActive(self.wk.lora_refs[0..self.wk.lora_count]) or
            lora_mod.anyActive(self.wv.lora_refs[0..self.wv.lora_count]) or lora_mod.anyActive(self.gate.lora_refs[0..self.gate.lora_count]);
        if (cos == null or lora or ops.shape(x)[0] != 1) return null;
        const xb = try g.astype(x, .bfloat16); // the projections compute in bf16
        const y = try g.adopt(try w8a8.forward(xb, f, g.s));
        const sh = ops.shape(y);
        const y2 = try g.reshape(y, &.{ sh[1], sh[2] });
        const qo = self.heads * self.head_dim;
        const ko = qo + self.kvheads * self.head_dim;
        const go = ko + self.kvheads * self.head_dim;
        const h = try dit_kernels.qkvHeads(y2, .{ .heads = self.heads, .kv_heads = self.kvheads, .q_off = 0, .k_off = qo, .v_off = ko }, self.qnorm, self.knorm, cos.?, sin.?, 1e-5, true, g.s);
        return .{ .q = try g.adopt(h.q), .k = try g.adopt(h.k), .v = try g.adopt(h.v), .gate = try g.sliceAxis(y, 2, go, go + qo) };
    }

    /// q, k, v heads-major (q and k normed, roped when `cos` is given) and the
    /// output gate. With the fused int8 weight one GEMM and one kernel produce
    /// all of it; otherwise four projections and the op chain.
    fn projectHeads(self: *const Attention, g: *Graph, x: Array, cos: ?Array, sin: ?Array) !Heads {
        const B: c_int = ops.shape(x)[0];
        const L: c_int = ops.shape(x)[1];
        const hd = self.head_dim;
        if (try self.fusedHeads(g, x, cos, sin)) |h| return h;
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const q = try g.transpose(try g.reshape(try self.wq.forward(g, x), &.{ B, L, self.heads, hd }), &perm);
        const k = try g.transpose(try g.reshape(try self.wk.forward(g, x), &.{ B, L, self.kvheads, hd }), &perm);
        const v = try g.transpose(try g.reshape(try self.wv.forward(g, x), &.{ B, L, self.kvheads, hd }), &perm);
        // QK-RMSNorm over head_dim
        var out: Heads = .{ .q = try g.rmsNorm(q, self.qnorm, 1e-5), .k = try g.rmsNorm(k, self.knorm, 1e-5), .v = v, .gate = try self.gate.forward(g, x) };
        if (cos) |cc| {
            const tab = [_]c_int{ 1, 1, L, 64 };
            const cb = try g.reshape(cc, &tab);
            const sb = try g.reshape(sin.?, &tab);
            out.q = try rope_blk.ropePairs(g, out.q, cb, sb, .bfloat16);
            out.k = try rope_blk.ropePairs(g, out.k, cb, sb, .bfloat16);
        }
        return out;
    }

    /// qkv [B,L,dim]. cos/sin null → no rope. mask null → no mask. Batch B is
    /// read from the input — the text-fusion layerwise pass runs with B=B·L,L=12.
    fn forward(self: *const Attention, parent: *Graph, qkv: Array, cos: ?Array, sin: ?Array, mask: ?Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const B: c_int = ops.shape(qkv)[0];
        const L: c_int = ops.shape(qkv)[1];
        const h = try self.projectHeads(&g, qkv, cos, sin);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const attn8 = if (w8a8.w8a8State().enabled and mask == null) try int8_attn.attention(h.q, h.k, h.v, scale, g.s) else null;
        const merged = if (attn8) |a8|
            try g.adopt(a8)
        else
            try g.transpose(try g.sdpa(h.q, h.k, h.v, scale, if (mask) |m| .{ .array = m } else .none), &.{ 0, 2, 1, 3 });
        const af = try g.reshape(merged, &.{ B, L, self.heads * self.head_dim });
        // output gate: out * sigmoid(gate), then wo
        return parent.keep(try self.wo.forward(&g, try g.mul(af, try g.sigmoid(h.gate))));
    }
};

/// Load a krea RMSNorm: stored `.scale`, effective weight = (scale+1), bf16.
fn kreaNorm(w: *const Weights, key: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    const one = scalarF(1.0);
    defer _ = mlx.mlx_array_free(one);
    const w1 = try addA(raw, one, s);
    defer _ = mlx.mlx_array_free(w1);
    return astype(w1, .bfloat16, s);
}

const SwiGLU = struct {
    gate: MixedLinear,
    up: MixedLinear,
    down: MixedLinear,
    /// gate and up interleaved in one int8 weight, SwiGLU in its epilogue; it
    /// REPLACES both linears (`--w8a8`).
    gu: ?w8a8.Weight = null,
    fn deinit(self: *SwiGLU) void {
        if (self.gu) |*q| q.deinit();
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
    }
    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, features: u32, mlpdim: u32, s: S) !SwiGLU {
        const kg = try fmtKey(a, "{s}.gate", .{pfx});
        defer a.free(kg);
        const ku = try fmtKey(a, "{s}.up", .{pfx});
        defer a.free(ku);
        const kd = try fmtKey(a, "{s}.down", .{pfx});
        defer a.free(kd);
        return .{
            .gate = try MixedLinear.load(w, a, kg, features, s),
            .up = try MixedLinear.load(w, a, ku, features, s),
            .down = try MixedLinear.load(w, a, kd, mlpdim, s),
        };
    }
    /// Gate and up fuse only when neither has a bias; the interleaved GEMM drops it.
    fn fuseW8a8(self: *SwiGLU, s: S) !void {
        if (self.gate.add_bias != null or self.up.add_bias != null) return;
        self.gu = try w8a8.fuse(&.{ &self.gate.w8, &self.up.w8 }, .interleave, s);
    }

    fn forward(self: *const SwiGLU, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(try self.down.forward(&g, try self.gateUp(&g, x)));
    }

    /// `silu(gate(x)) * up(x)`. With the interleaved weight that is one GEMM; an
    /// adapter at a nonzero scale needs the two projections apart, so it takes
    /// the plain GEMM and splits it (a strength-0 stack keeps the fused bytes).
    fn gateUp(self: *const SwiGLU, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        var gt: Array = undefined;
        var u: Array = undefined;
        if (self.gu) |*f| {
            const xb = try g.astype(x, .bfloat16);
            if (!lora_mod.anyActive(self.gate.lora_refs[0..self.gate.lora_count]) and !lora_mod.anyActive(self.up.lora_refs[0..self.up.lora_count]))
                return parent.keep(try g.adopt(try w8a8.forwardSwiglu(xb, f, g.s)));
            const pair = try w8a8.forwardPair(xb, f, g.s);
            const pg = g.adopt(pair.gate) catch |e| {
                _ = mlx.mlx_array_free(pair.up);
                return e;
            };
            gt = try self.gate.addLora(&g, pg, xb);
            u = try self.up.addLora(&g, try g.adopt(pair.up), xb);
        } else {
            gt = try self.gate.forward(&g, x);
            u = try self.up.forward(&g, x);
        }
        return parent.keep(try g.mul(try g.silu(gt), u));
    }

    /// The raw-array forward the ANE seam calls (it owns no graph).
    fn forwardRaw(self: *const SwiGLU, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var g = Graph.init(std.heap.c_allocator, s);
        defer g.deinit();
        return g.escape(try self.forward(&g, x));
    }
};

// ── ANE MLP offload (`--ane-image`, opt-in, LOSSY) ──
//
// A denoising step is a batch job at the COMPUTE roofline, which is the case
// the LM prefill seam was never able to be: no live decode waits on it, the
// token count is constant for a run so one compiled tile covers every step,
// and the seam cost amortizes over a whole DiT block. Geometry, split and
// parity bar are the channel-mode seam's (transformer.zig) — the ANE holds
// output channels [0..k) of gate/up plus the matching down K-slabs (a PARTIAL
// down sum), the GPU the complement, and the two partials ADD.

/// Whether one DiT block's SwiGLU may ride the ANE this run. Rows are the
/// compiled tile (`ane.mediaTileRows`), baked into the program. A
/// LoRA-attached block DECLINES: the int8 copy was quantized before the
/// adapter existed, so an offloaded slice would drop it silently.
pub fn aneBlockEligible(rows: u32, quantized: bool, lora_count: u8) bool {
    if (!quantized or lora_count > 0) return false;
    if (rows < ane.ANE_MIN_ROWS or rows % 32 != 0) return false;
    return true;
}

fn mlpLoraCount(m: *const SwiGLU) u8 {
    return @max(@max(m.gate.lora_count, m.up.lora_count), m.down.lora_count);
}

/// What `ane.planMediaOffload` calibrates on: block 0 built alone.
const AneCalib = struct {
    d: *Dit,
    rows: u32,
    /// The DiT stream the MLP sees is f32 (each linear casts it to bf16).
    dtype: mlx.mlx_dtype = .float32,
    feat: u32,
    mlpdim: u32,
    s: S,
    pub fn buildBlock0(self: AneCalib, k: u32, units: u32, share: f32) !*ane.AnePrefill {
        try self.d.aneBuild(self.rows, k, units, self.feat, self.mlpdim, share, self.s, 1);
        return self.d.ane_eng.?;
    }
    pub fn complement(self: AneCalib, x: mlx.mlx_array) !mlx.mlx_array {
        return self.d.ane_rest[0].forwardRaw(x, self.s);
    }
    pub fn teardown(self: AneCalib) void {
        self.d.aneDeinit();
    }
};

/// What `ane.mediaMlp` runs on the GPU: the complement channels, or the whole
/// MLP for the tail and the fallback.
const AneSeam = struct {
    mlp: *const SwiGLU,
    rest: *const SwiGLU,
    s: S,
    pub fn complement(self: AneSeam, head: mlx.mlx_array) !mlx.mlx_array {
        return self.rest.forwardRaw(head, self.s);
    }
    pub fn full(self: AneSeam, x: mlx.mlx_array) !mlx.mlx_array {
        return self.mlp.forwardRaw(x, self.s);
    }
};

/// Dequantize one MixedLinear to host f32 [out_dim, in_dim] (`ane` owns the op).
fn aneDequantHostF32(a: std.mem.Allocator, ml: *const MixedLinear, in_dim: u32, out_dim: u32, s: S) ![]f32 {
    if (!ml.quantized) return error.AneDenseBf16Unsupported;
    return ane.dequantToHostF32(a, s, ml.w, ml.scales, ml.biases, ml.bits, ml.group_size, in_dim, out_dim);
}

/// Non-owning row view [from..) of a packed axis-0 tensor (gate/up rest).
fn aneRowsFrom(x: mlx.mlx_array, from: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    return sliceAxis(x, 0, from, sh[0], s);
}

/// Materialized column slice [from..) of an axis-1 tensor (down rest): a view
/// into a packed weight computes wrong at scale, so it is copied.
fn aneColsFrom(x: mlx.mlx_array, from: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const v = try sliceAxis(x, 1, from, sh[1], s);
    defer _ = mlx.mlx_array_free(v);
    return contig(v, s);
}

/// One rest-channel MixedLinear: an axis-0 row VIEW [from..) of a packed
/// weight (gate/up — the ANE holds rows [0..from)).
fn aneRestRows(ml: *const MixedLinear, from: c_int, s: S) !MixedLinear {
    const w = try aneRowsFrom(ml.w, from, s);
    errdefer _ = mlx.mlx_array_free(w);
    const sc = try aneRowsFrom(ml.scales, from, s);
    errdefer _ = mlx.mlx_array_free(sc);
    const bi = try aneRowsFrom(ml.biases, from, s);
    return .{ .quantized = true, .w = w, .scales = sc, .biases = bi, .bits = ml.bits, .group_size = ml.group_size };
}

/// The down rest: an axis-1 K-slice, MATERIALIZED — a lazy view into a packed
/// weight computes wrong at scale (`splitPackedGateUp`'s rule).
fn aneRestCols(ml: *const MixedLinear, k: u32, s: S) !MixedLinear {
    const w = try aneColsFrom(ml.w, @intCast(k * ml.bits / 32), s);
    errdefer _ = mlx.mlx_array_free(w);
    const g: c_int = @intCast(k / ml.group_size);
    const sc = try aneColsFrom(ml.scales, g, s);
    errdefer _ = mlx.mlx_array_free(sc);
    const bi = try aneColsFrom(ml.biases, g, s);
    errdefer _ = mlx.mlx_array_free(bi);
    var add: ?mlx.mlx_array = null;
    if (ml.add_bias) |b| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, b));
        add = o;
    }
    return .{ .quantized = true, .w = w, .scales = sc, .biases = bi, .add_bias = add, .bits = ml.bits, .group_size = ml.group_size };
}

/// The GPU complement of one block. `k` is the TOTAL width the ANE units hold:
/// gate/up keep rows [k..), down keeps K-columns [k..), and the down partials
/// sum at the seam.
fn aneBuildMlpRest(m: *const SwiGLU, k: u32, s: S) !SwiGLU {
    const d = &m.down;
    if (d.group_size == 0 or k % d.group_size != 0 or (k * d.bits) % 32 != 0) return error.AneChanAlignment;
    // The ANE program carries no bias term, so a biased gate/up cannot split.
    if (m.gate.add_bias != null or m.up.add_bias != null) return error.AneBiasedProjection;
    const kk: c_int = @intCast(k);
    var rest: SwiGLU = undefined;
    rest.gate = try aneRestRows(&m.gate, kk, s);
    errdefer rest.gate.deinit();
    rest.up = try aneRestRows(&m.up, kk, s);
    errdefer rest.up.deinit();
    rest.down = try aneRestCols(d, k, s);
    return rest;
}

const Block = struct {
    mod_lin: mlx.mlx_array, // [6*features] bf16
    prenorm: mlx.mlx_array, // (1+scale)
    postnorm: mlx.mlx_array,
    attn: Attention,
    mlp: SwiGLU,
    fn deinit(self: *Block) void {
        _ = mlx.mlx_array_free(self.mod_lin);
        _ = mlx.mlx_array_free(self.prenorm);
        _ = mlx.mlx_array_free(self.postnorm);
        self.attn.deinit();
        self.mlp.deinit();
    }
};

const TextFusionBlock = struct {
    prenorm: mlx.mlx_array,
    postnorm: mlx.mlx_array,
    attn: Attention,
    mlp: SwiGLU,
    fn deinit(self: *TextFusionBlock) void {
        _ = mlx.mlx_array_free(self.prenorm);
        _ = mlx.mlx_array_free(self.postnorm);
        self.attn.deinit();
        self.mlp.deinit();
    }
    fn load(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, cfg: KreaConfig, s: S) !TextFusionBlock {
        const pre = try fmtKey(a, "{s}.prenorm.scale", .{pfx});
        defer a.free(pre);
        const post = try fmtKey(a, "{s}.postnorm.scale", .{pfx});
        defer a.free(post);
        const ka = try fmtKey(a, "{s}.attn", .{pfx});
        defer a.free(ka);
        const km = try fmtKey(a, "{s}.mlp", .{pfx});
        defer a.free(km);
        const hd = cfg.txtdim / cfg.txtheads;
        return .{
            .prenorm = try kreaNorm(w, pre, s),
            .postnorm = try kreaNorm(w, post, s),
            .attn = try Attention.load(w, a, ka, cfg.txtdim, cfg.txtheads, cfg.txtheads, hd, s),
            .mlp = try SwiGLU.load(w, a, km, cfg.txtdim, cfg.txt_mlpdim, s),
        };
    }
    /// x [.,L,txtdim]; no rope; mask optional. Used for both layerwise (over the
    /// 12-layer axis) and refiner (over tokens) passes.
    /// x [.,L,txtdim]; no rope; mask optional. Used for both layerwise (over the
    /// 12-layer axis) and refiner (over tokens) passes.
    fn forward(self: *const TextFusionBlock, parent: *Graph, x: Array, mask: ?Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const h1 = try g.add(x, try self.attn.forward(&g, try g.rmsNorm(x, self.prenorm, 1e-5), null, null, mask));
        return parent.keep(try g.add(h1, try self.mlp.forward(&g, try g.rmsNorm(h1, self.postnorm, 1e-5))));
    }
};

pub const Dit = struct {
    cfg: KreaConfig,
    allocator: std.mem.Allocator,
    s: S,
    first: MixedLinear, // 64 → features (+bias)
    blocks: []Block,
    tmlp0: MixedLinear, // tdim → features (+bias)
    tmlp2: MixedLinear, // features → features (+bias)
    tproj1: MixedLinear, // features → 6*features (+bias)
    layerwise: [2]TextFusionBlock,
    projector: MixedLinear, // 12 → 1 (no bias)
    refiner: [2]TextFusionBlock,
    txtmlp0: mlx.mlx_array, // RMSNorm (1+scale)
    txtmlp1: MixedLinear, // txtdim → features (+bias)
    txtmlp3: MixedLinear, // features → features (+bias)
    last_norm: mlx.mlx_array, // (1+scale)
    last_lin: MixedLinear, // features → 64 (+bias)
    last_mod: mlx.mlx_array, // [2, features] bf16
    io: std.Io,
    /// ANE MLP offload, built lazily at the first forward (the token count is
    /// a per-request property and the compiled tile is fixed).
    ane_eng: ?*ane.AnePrefill,
    ane_rest: []SwiGLU,
    ane_tried: bool,

    pub fn deinit(self: *Dit) void {
        w8a8.dropMemo();
        self.aneDeinit();
        self.first.deinit();
        for (self.blocks) |*b| b.deinit();
        self.allocator.free(self.blocks);
        self.tmlp0.deinit();
        self.tmlp2.deinit();
        self.tproj1.deinit();
        for (&self.layerwise) |*b| b.deinit();
        self.projector.deinit();
        for (&self.refiner) |*b| b.deinit();
        _ = mlx.mlx_array_free(self.txtmlp0);
        self.txtmlp1.deinit();
        self.txtmlp3.deinit();
        _ = mlx.mlx_array_free(self.last_norm);
        self.last_lin.deinit();
        _ = mlx.mlx_array_free(self.last_mod);
    }

    fn aneDeinit(self: *Dit) void {
        if (self.ane_eng) |e| e.deinit();
        self.ane_eng = null;
        for (self.ane_rest) |*r| r.deinit();
        if (self.ane_rest.len > 0) self.allocator.free(self.ane_rest);
        self.ane_rest = &.{};
    }

    /// timestep embed (B=1): [cos(args), sin(args)] of dim tdim, period 1e4,
    /// tfactor 1e3, then tmlp → t_emb [1,1,features]; tproj → tvec [1,1,6*features].
    fn timeVecs(self: *const Dit, parent: *Graph, t: f32) !struct { t_emb: Array, tvec: Array } {
        var g = parent.scope();
        defer g.deinit();
        const dim = self.cfg.tdim;
        const buf = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(buf);
        timestep_blk.sinusoid(f32, buf, t * 1e3, .cos_first);
        const emb = try g.astype(try g.fromSlice(f32, buf, &.{ 1, 1, @intCast(dim) }), .bfloat16);
        // tmlp: lin0 → gelu → lin2; tproj: gelu → lin1
        const t_emb = try self.tmlp2.forward(&g, try gelu.geluTanh(&g, try self.tmlp0.forward(&g, emb), .float32, gelu.SQRT_2_PI));
        const tvec = try self.tproj1.forward(&g, try gelu.geluTanh(&g, t_emb, .float32, gelu.SQRT_2_PI));
        return .{ .t_emb = try parent.keep(t_emb), .tvec = try parent.keep(tvec) };
    }

    /// One DiT forward. img [1,Limg,64] bf16, context [1,512,12,txtdim] bf16,
    /// t scalar, pos [L*3] i32, valid [L] i32 (L = 512 + Limg). → velocity
    /// [1,Limg,64] bf16.
    pub fn forward(self: *Dit, parent: *Graph, img: Array, context: Array, t: f32, pos: []const i32, valid: []const i32) !Array {
        var g = parent.scope();
        defer g.deinit();
        const txtlen: usize = @intCast(ops.shape(context)[1]);
        const rope = try buildRope(&g, self.allocator, pos, valid.len, self.cfg.theta);
        const full_mask = try outerMask(&g, valid);
        const fused = try self.textFusion(&g, context, try outerMask(&g, valid[0..txtlen]));
        return parent.keep(try self.forwardPrebuilt(&g, img, fused, t, rope.cos, rope.sin, full_mask));
    }

    /// `fused` is `textFusion`'s output, which does not depend on `t`: `generate`
    /// builds it, the rope and the mask once per request. A null mask = every row valid.
    pub fn forwardPrebuilt(self: *Dit, parent: *Graph, img: Array, fused: Array, t: f32, cos: Array, sin: Array, full_mask: ?Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const txtlen: c_int = ops.shape(fused)[1];
        const img_len: c_int = ops.shape(img)[1];
        const tv = try self.timeVecs(&g, t);
        self.aneEnsure(@intCast(txtlen + img_len), g.s);
        const feat: c_int = @intCast(self.cfg.features);
        const mods = try self.allocator.alloc(dit_base.Mods, self.blocks.len);
        defer self.allocator.free(mods);
        // mod = tvec + lin; split 6 → prescale, preshift, pregate, postscale, postshift, postgate
        for (self.blocks, mods) |*b, *m| {
            const mod = try g.add(tv.tvec, b.mod_lin);
            m.* = .{
                .scale_a = try g.sliceAxis(mod, 2, 0, feat),
                .shift_a = try g.sliceAxis(mod, 2, feat, 2 * feat),
                .gate_a = try g.sliceAxis(mod, 2, 2 * feat, 3 * feat),
                .scale_f = try g.sliceAxis(mod, 2, 3 * feat, 4 * feat),
                .shift_f = try g.sliceAxis(mod, 2, 4 * feat, 5 * feat),
                .gate_f = try g.sliceAxis(mod, 2, 5 * feat, 6 * feat),
            };
        }
        const ctx: StepCtx = .{ .dit = self, .cos = cos, .sin = sin, .mask = full_mask, .t_emb = tv.t_emb, .mods = mods, .txtlen = txtlen, .img_len = img_len };
        return parent.keep(try dit_base.Dit(Hooks).forward(&g, self, .{ .img = img, .fused = fused }, ctx, null));
    }

    fn textFusion(self: *const Dit, parent: *Graph, context: Array, txt_mask: ?Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const sh = ops.shape(context); // [1,L,12,txtdim]
        const L = sh[1];
        const n = sh[2];
        const d = sh[3];
        // (1,L,12,d) → (L,12,d) for layerwise attention over the layer axis.
        var x = try g.reshape(context, &.{ L * 1, n, d });
        for (&self.layerwise) |*b| x = try b.forward(&g, x, null);
        // (L,12,d) → (1,L,d,12) → projector(12→1) → (1,L,d)
        const tr = try g.transpose(try g.reshape(x, &.{ 1, L, n, d }), &.{ 0, 1, 3, 2 });
        var ctx = try g.reshape(try self.projector.forward(&g, tr), &.{ 1, L, d });
        for (&self.refiner) |*b| ctx = try b.forward(&g, ctx, txt_mask);
        // txtmlp: RMSNorm → lin1 → gelu → lin3
        const l1 = try self.txtmlp1.forward(&g, try g.rmsNorm(ctx, self.txtmlp0, 1e-5));
        return parent.keep(try self.txtmlp3.forward(&g, try gelu.geluTanh(&g, l1, .float32, gelu.SQRT_2_PI)));
    }

    /// Build the per-block ANE programs once the run's token count is known.
    /// Never fails the request: every refusal is a NAMED `[ane]` line and the
    /// DiT runs GPU-only.
    fn aneEnsure(self: *Dit, seq: u32, s: S) void {
        if (self.ane_tried) return;
        if (!ane.aneState().media_offload.image) return;
        if (!ane.available()) {
            self.ane_tried = true;
            log.warn("[ane] image offload: AppleNeuralEngine framework not present — GPU only\n", .{});
            return;
        }
        const rows = ane.mediaTileRows(seq);
        const b0 = &self.blocks[0].mlp;
        const loras = mlpLoraCount(b0);
        if (!aneBlockEligible(rows, b0.gate.quantized, loras)) {
            // A LoRA or a short request declines this request only; a bf16 pack never qualifies.
            self.ane_tried = !b0.gate.quantized;
            log.warn("[ane] image offload declined: {d} rows, quantized={}, loras={d} — GPU only\n", .{ rows, b0.gate.quantized, loras });
            return;
        }
        self.ane_tried = true;
        const feat = self.cfg.features;
        const mlpdim = self.cfg.mlpdim;
        const probe: ane.QuantWeight = .{ .w = b0.gate.w, .scales = b0.gate.scales, .biases = b0.gate.biases, .bits = b0.gate.bits, .group_size = b0.gate.group_size, .in_dim = feat, .out_dim = mlpdim };
        const plan = ane.planMediaOffload(self.io, s, "image", self.blocks.len, feat, mlpdim, rows, probe, AneCalib{ .d = self, .rows = rows, .feat = feat, .mlpdim = mlpdim, .s = s }) orelse return;
        self.aneBuild(rows, plan.k, plan.units, feat, mlpdim, plan.share, s, self.blocks.len) catch |err| {
            log.warn("[ane] image offload build failed ({s}) — GPU only\n", .{@errorName(err)});
            self.aneDeinit();
        };
    }

    /// Builds blocks [0, n); fewer than all is the calibration build.
    fn aneBuild(self: *Dit, rows: u32, k: u32, units: u32, feat: u32, mlpdim: u32, share: f32, s: S, n: usize) !void {
        const a = self.allocator;
        const eng = try ane.AnePrefill.init(a, self.io, self.blocks.len, feat, k, rows, rows, 0, 0, units);
        errdefer eng.deinit();
        const rests = try a.alloc(SwiGLU, n);
        errdefer a.free(rests);
        var built_rests: usize = 0;
        errdefer for (rests[0..built_rests]) |*r| r.deinit();

        const start = std.Io.Timestamp.now(self.io, .awake);
        const kh: usize = @as(usize, k) * feat;
        const down_slice = try a.alloc(f32, @as(usize, feat) * k);
        defer a.free(down_slice);
        const up_slice = try a.alloc(f32, kh);
        defer a.free(up_slice);
        for (self.blocks[0..n], 0..) |*b, i| {
            rests[i] = try aneBuildMlpRest(&b.mlp, k * units, s);
            built_rests = i + 1;
            const gate = try aneDequantHostF32(a, &b.mlp.gate, feat, mlpdim, s);
            defer a.free(gate);
            const up = try aneDequantHostF32(a, &b.mlp.up, feat, mlpdim, s);
            defer a.free(up);
            const down = try aneDequantHostF32(a, &b.mlp.down, mlpdim, feat, s);
            defer a.free(down);
            for (0..units) |u| {
                const c0: usize = u * @as(usize, k);
                for (0..feat) |r| @memcpy(down_slice[r * k ..][0..k], down[r * mlpdim + c0 ..][0..k]);
                // `up` carries the fp16 range scale for the whole program.
                for (up_slice, up[u * kh ..][0..kh]) |*d, v| d.* = v / ane.OUT_PLANE_SCALE;
                eng.addMlpLayer(u, i, gate[u * kh ..][0..kh], up_slice, down_slice) catch |err| {
                    log.warn("[ane] block {d} unit {d} enqueue failed ({s})\n", .{ i, u, @errorName(err) });
                    break;
                };
            }
        }
        eng.finishPending();
        const built = eng.coveredLayers();
        if (built == 0) return error.AneNoProgram;
        const secs: f64 = @as(f64, @floatFromInt(@as(u64, @intCast(start.untilNow(self.io, .awake).nanoseconds)))) / 1e9;
        const int8_bytes: u64 = @as(u64, built) * 3 * feat * k * units;
        eng.publishLive(share, int8_bytes);
        self.ane_eng = eng;
        self.ane_rest = rests;
        if (n == self.blocks.len) log.info("[ane] image offload ready: units={d} {d}/{d} blocks in {d} banks, rows={d}, k={d}/{d} (share {d:.2}), int8 ~{d} MB, built in {d:.1}s\n", .{ units, built, self.blocks.len, eng.compiledBanks(), rows, k * units, mlpdim, share, int8_bytes / (1024 * 1024), secs });
    }

    fn mlpMaybeAne(self: *const Dit, g: *Graph, x: Array, b: *const Block, idx: usize) !Array {
        const eng = self.ane_eng orelse return b.mlp.forward(g, x);
        if (!eng.mlpReady(idx) or idx >= self.ane_rest.len) return b.mlp.forward(g, x);
        if (mlpLoraCount(&b.mlp) > 0) return b.mlp.forward(g, x);
        const sh = ops.shape(x);
        if (sh.len != 3 or sh[0] != 1 or sh[2] != @as(c_int, @intCast(eng.hidden))) return b.mlp.forward(g, x);
        const out = ane.mediaMlp(g.s, eng, idx, "image", x, AneSeam{ .mlp = &b.mlp, .rest = &self.ane_rest[idx], .s = g.s }) catch |err| {
            // An ANE failure is never a request failure — the block runs on GPU.
            log.warn("[ane] image block {d} split failed ({s}) — GPU fallback\n", .{ idx, @errorName(err) });
            return b.mlp.forward(g, x);
        };
        return g.adopt(out);
    }
};

/// One step's conditioning for the skeleton.
const StepCtx = struct {
    dit: *const Dit,
    cos: Array,
    sin: Array,
    mask: ?Array,
    t_emb: Array,
    mods: []const dit_base.Mods,
    txtlen: c_int,
    img_len: c_int,
};

/// The skeleton's hooks: pre-norm adaLN blocks over [text, image], RMS norms, x·(scale + 1) + shift in f32.
const Hooks = struct {
    pub const Model = Dit;
    pub const Ctx = StepCtx;
    pub const Input = struct { img: Array, fused: Array };
    pub const Layer = Block;
    pub const mod: dit_base.Modulation = .{ .plus_one = .float32 };

    pub fn embed(parent: *Graph, m: *const Dit, in: Input, _: StepCtx) !Array {
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(try g.concat(&.{ in.fused, try m.first.forward(&g, in.img) }, 1));
    }
    pub fn layers(m: *const Dit) usize {
        return m.blocks.len;
    }
    pub fn layer(m: *const Dit, i: usize) *const Block {
        return &m.blocks[i];
    }
    pub fn mods(ctx: StepCtx, i: usize) dit_base.Mods {
        return ctx.mods[i];
    }
    pub fn norm1(g: *Graph, w: *const Block, x: Array) !Array {
        return g.rmsNorm(x, w.prenorm, 1e-5);
    }
    pub fn norm2(g: *Graph, w: *const Block, x: Array) !Array {
        return g.rmsNorm(x, w.postnorm, 1e-5);
    }
    pub fn attention(g: *Graph, w: *const Block, x: Array, ctx: StepCtx) !Array {
        return w.attn.forward(g, x, ctx.cos, ctx.sin, ctx.mask);
    }
    pub fn ffn(g: *Graph, w: *const Block, x: Array, ctx: StepCtx) !Array {
        const idx = (@intFromPtr(w) - @intFromPtr(ctx.dit.blocks.ptr)) / @sizeOf(Block);
        return ctx.dit.mlpMaybeAne(g, x, w, idx);
    }

    /// LastLayer: (1+scale)·norm(x)+shift from SimpleModulation(t_emb), then the image span.
    pub fn final(parent: *Graph, m: *const Dit, x: Array, ctx: StepCtx) !Array {
        var g = parent.scope();
        defer g.deinit();
        const feat: c_int = @intCast(m.cfg.features);
        const lin = try g.reshape(m.last_mod, &.{ 1, 2, feat });
        const md = try g.add(ctx.t_emb, lin); // (1,1,f)+(1,2,f) → (1,2,f)
        const xm = try modulate_blk.modulate(&g, try g.rmsNorm(x, m.last_norm, 1e-5), try g.sliceAxis(md, 1, 0, 1), try g.sliceAxis(md, 1, 1, 2), .float32);
        const out = try m.last_lin.forward(&g, xm);
        return parent.keep(try g.sliceAxis(out, 1, ctx.txtlen, ctx.txtlen + ctx.img_len));
    }
};

/// Additive attention mask from a validity vector valid [L] (i32 0/1):
/// [1,1,L,L] bf16, (1 - valid_i*valid_j) * -1e9.
fn outerMask(parent: *Graph, valid: []const i32) !Array {
    var g = parent.scope();
    defer g.deinit();
    const L: c_int = @intCast(valid.len);
    const buf = try g.a.alloc(f32, valid.len);
    defer g.a.free(buf);
    for (valid, 0..) |vv, i| buf[i] = @floatFromInt(vv);
    const va = try g.fromSlice(f32, buf, &.{ 1, L });
    const full = try g.mul(try g.reshape(va, &.{ 1, L, 1 }), try g.reshape(va, &.{ 1, 1, L }));
    const add = try g.mul(try g.sub(try g.scalar(1.0, .float32), full), try g.scalar(-1e9, .float32));
    return parent.keep(try g.astype(try g.reshape(add, &.{ 1, 1, L, L }), .bfloat16));
}

pub fn loadDit(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Dit {
    // The transformer weights live in a single safetensors file whose precision
    // is auto-detected by presence (mixed-4-8 → 8bit → bf16). loadWeights merges
    // every *.safetensors in the dir, so any single present file is read.
    var w = try model_mod.loadWeights(io, allocator, model_dir);
    defer w.deinit();
    var d: Dit = undefined;
    d.cfg = KreaConfig.ditFrom(&w, allocator);
    d.allocator = allocator;
    d.s = s;
    const cfg = d.cfg;
    const feat = cfg.features;
    const hd = cfg.headDim();

    d.first = try MixedLinear.load(&w, allocator, "first", cfg.channels * cfg.patch * cfg.patch, s);
    d.tmlp0 = try MixedLinear.load(&w, allocator, "tmlp.0", cfg.tdim, s);
    d.tmlp2 = try MixedLinear.load(&w, allocator, "tmlp.2", feat, s);
    d.tproj1 = try MixedLinear.load(&w, allocator, "tproj.1", feat, s);

    d.blocks = try allocator.alloc(Block, cfg.layers);
    const mlpdim = cfg.mlpdim;
    for (d.blocks, 0..) |*b, i| {
        const pfx = try fmtKey(allocator, "blocks.{d}", .{i});
        defer allocator.free(pfx);
        const modk = try fmtKey(allocator, "blocks.{d}.mod.lin", .{i});
        defer allocator.free(modk);
        const prek = try fmtKey(allocator, "blocks.{d}.prenorm.scale", .{i});
        defer allocator.free(prek);
        const postk = try fmtKey(allocator, "blocks.{d}.postnorm.scale", .{i});
        defer allocator.free(postk);
        const attnk = try fmtKey(allocator, "blocks.{d}.attn", .{i});
        defer allocator.free(attnk);
        const mlpk = try fmtKey(allocator, "blocks.{d}.mlp", .{i});
        defer allocator.free(mlpk);
        b.* = .{
            .mod_lin = try normBf16(&w, modk, s),
            .prenorm = try kreaNorm(&w, prek, s),
            .postnorm = try kreaNorm(&w, postk, s),
            .attn = try Attention.load(&w, allocator, attnk, feat, cfg.heads, cfg.kvheads, hd, s),
            .mlp = try SwiGLU.load(&w, allocator, mlpk, feat, mlpdim, s),
        };
        if (!ane.aneState().media_offload.image) { // the ANE seam reads the packed weights
            inline for (.{ &b.attn.wq, &b.attn.wk, &b.attn.wv, &b.attn.gate, &b.attn.wo, &b.mlp.gate, &b.mlp.up }) |l| try l.toW8a8(s);
            try b.mlp.down.toW8a8(s);
            try b.attn.fuseW8a8(s);
            try b.mlp.fuseW8a8(s);
            if (w8a8.w8a8State().enabled) w.dropBlock("blocks.{d}.", i);
        }
    }

    inline for (.{ 0, 1 }) |i| {
        const pfx = try fmtKey(allocator, "txtfusion.layerwise_blocks.{d}", .{i});
        defer allocator.free(pfx);
        d.layerwise[i] = try TextFusionBlock.load(&w, allocator, pfx, cfg, s);
    }
    d.projector = try MixedLinear.load(&w, allocator, "txtfusion.projector", cfg.txtlayers, s);
    inline for (.{ 0, 1 }) |i| {
        const pfx = try fmtKey(allocator, "txtfusion.refiner_blocks.{d}", .{i});
        defer allocator.free(pfx);
        d.refiner[i] = try TextFusionBlock.load(&w, allocator, pfx, cfg, s);
    }

    d.txtmlp0 = try kreaNorm(&w, "txtmlp.0.scale", s);
    d.txtmlp1 = try MixedLinear.load(&w, allocator, "txtmlp.1", cfg.txtdim, s);
    d.txtmlp3 = try MixedLinear.load(&w, allocator, "txtmlp.3", feat, s);

    d.last_norm = try kreaNorm(&w, "last.norm.scale", s);
    d.last_lin = try MixedLinear.load(&w, allocator, "last.linear", feat, s);
    d.last_mod = try normBf16(&w, "last.modulation.lin", s);
    d.io = io;
    d.ane_eng = null;
    d.ane_rest = &.{};
    d.ane_tried = false;
    return d;
}

/// Attach every adapter in `stack` to its matching DiT linear (summed at
/// forward time — see `lora.deltaSum`). Non-owning: `stack` must outlive
/// the attach. Returns the number of (module, matched-adapter) attachments
/// across the whole stack (a module hit by two stacked LoRAs counts twice).
pub fn attachLora(dit: *Dit, stack_arg: *const lora_mod.Stack) u32 {
    detachLora(dit);
    var matched: u32 = 0;
    var kbuf: [128]u8 = undefined;
    var rbuf: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined;

    // Main 28 blocks
    for (dit.blocks, 0..) |*b, i| {
        const mods = .{
            .{ "attn.wq", &b.attn.wq }, .{ "attn.wk", &b.attn.wk },
            .{ "attn.wv", &b.attn.wv }, .{ "attn.gate", &b.attn.gate },
            .{ "attn.wo", &b.attn.wo }, .{ "mlp.gate", &b.mlp.gate },
            .{ "mlp.up", &b.mlp.up },   .{ "mlp.down", &b.mlp.down },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "blocks.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack_arg.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }

    // Text fusion: layerwise blocks (2 layers)
    for (&dit.layerwise, 0..) |*lb, i| {
        const mods = .{
            .{ "attn.wq", &lb.attn.wq }, .{ "attn.wk", &lb.attn.wk },
            .{ "attn.wv", &lb.attn.wv }, .{ "attn.gate", &lb.attn.gate },
            .{ "attn.wo", &lb.attn.wo }, .{ "mlp.gate", &lb.mlp.gate },
            .{ "mlp.up", &lb.mlp.up },   .{ "mlp.down", &lb.mlp.down },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "txtfusion.layerwise_blocks.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack_arg.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }

    // Text fusion: refiner blocks (2 layers)
    for (&dit.refiner, 0..) |*rb, i| {
        const mods = .{
            .{ "attn.wq", &rb.attn.wq }, .{ "attn.wk", &rb.attn.wk },
            .{ "attn.wv", &rb.attn.wv }, .{ "attn.gate", &rb.attn.gate },
            .{ "attn.wo", &rb.attn.wo }, .{ "mlp.gate", &rb.mlp.gate },
            .{ "mlp.up", &rb.mlp.up },   .{ "mlp.down", &rb.mlp.down },
        };
        inline for (mods) |m| {
            const key = std.fmt.bufPrint(&kbuf, "txtfusion.refiner_blocks.{d}.{s}", .{ i, m[0] }) catch "";
            const refs = stack_arg.findAll(key, &rbuf);
            if (refs.len > 0) {
                m[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }

    // Projector (text fusion)
    {
        const key = "txtfusion.projector";
        const refs = stack_arg.findAll(key, &rbuf);
        if (refs.len > 0) {
            dit.projector.setLoraRefs(refs);
            matched += @intCast(refs.len);
        }
    }

    // Other DiT linears
    {
        const others = .{
            .{ "first", &dit.first },
            .{ "tmlp.0", &dit.tmlp0 },
            .{ "tmlp.2", &dit.tmlp2 },
            .{ "tproj.1", &dit.tproj1 },
            .{ "txtmlp.1", &dit.txtmlp1 },
            .{ "txtmlp.3", &dit.txtmlp3 },
            .{ "last.linear", &dit.last_lin },
        };
        inline for (others) |o| {
            const refs = stack_arg.findAll(o[0], &rbuf);
            if (refs.len > 0) {
                o[1].setLoraRefs(refs);
                matched += @intCast(refs.len);
            }
        }
    }

    return matched;
}

pub fn detachLora(dit: *Dit) void {
    // Main blocks
    for (dit.blocks) |*b| {
        inline for (.{ &b.attn.wq, &b.attn.wk, &b.attn.wv, &b.attn.gate, &b.attn.wo, &b.mlp.gate, &b.mlp.up, &b.mlp.down }) |ml| ml.clearLoraRefs();
    }
    // Text fusion blocks
    for (&dit.layerwise) |*lb| {
        inline for (.{ &lb.attn.wq, &lb.attn.wk, &lb.attn.wv, &lb.attn.gate, &lb.attn.wo, &lb.mlp.gate, &lb.mlp.up, &lb.mlp.down }) |ml| ml.clearLoraRefs();
    }
    for (&dit.refiner) |*rb| {
        inline for (.{ &rb.attn.wq, &rb.attn.wk, &rb.attn.wv, &rb.attn.gate, &rb.attn.wo, &rb.mlp.gate, &rb.mlp.up, &rb.mlp.down }) |ml| ml.clearLoraRefs();
    }
    // Other linears
    inline for (.{ &dit.projector, &dit.first, &dit.tmlp0, &dit.tmlp2, &dit.tproj1, &dit.txtmlp1, &dit.txtmlp3, &dit.last_lin }) |ml| ml.clearLoraRefs();
}

// ════════════════════════════════════════════════════════════════════════
// B3. Autoencoder — Qwen-Image 3D causal VAE (decode path only, T=1, f16).
//     Raw diffusers tensor names; conv weights transposed to MLX layout at load.
// ════════════════════════════════════════════════════════════════════════

const LATENTS_MEAN = [16]f32{ -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921 };
const LATENTS_STD = [16]f32{ 2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916 };

/// The VAE's compute dtype: f16 decodes within ~56 dB of the reference's f32 and
/// lets every conv run on the half-precision matrix path.
const VAE_DT: mlx.mlx_dtype = .float16;

/// Qwen-Image's RMSNorm over channels (x / max(‖x‖₂, eps) · √C · γ = rms_norm over the last axis of NHWC).
const vae_norm: resnet_blk.Norm = .{ .rms = 1e-12 };

/// Frees the arrays a loaded resnet owns.
fn freeResnet(r: *Resnet) void {
    inline for (.{ "n1w", "c1w", "c1b", "n2w", "c2w", "c2b" }) |f| _ = mlx.mlx_array_free(@field(r, f));
    inline for (.{ "sw", "sb" }) |f| if (@field(r, f)) |x| {
        _ = mlx.mlx_array_free(x);
    };
}

const AttnBlock3D = struct {
    norm: mlx.mlx_array,
    qkv_w: mlx.mlx_array, // conv2d 1x1 [3C,1,1,C]
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    fn deinit(self: *AttnBlock3D) void {
        inline for (.{ "norm", "qkv_w", "qkv_b", "proj_w", "proj_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
    }
    fn forward(self: *const AttnBlock3D, g: *Graph, x: Array) !Array {
        return resnet_blk.midAttentionQkv(g, vae_norm, x, self.norm, null, self.qkv_w, self.qkv_b, self.proj_w, self.proj_b);
    }
};

const UpBlock3D = struct {
    resnets: []Resnet,
    up_w: ?mlx.mlx_array = null, // resample_conv (conv2d) [out,kh,kw,in]
    up_b: ?mlx.mlx_array = null,
    allocator: std.mem.Allocator,
    fn deinit(self: *UpBlock3D) void {
        for (self.resnets) |*r| freeResnet(r);
        self.allocator.free(self.resnets);
        if (self.up_w) |x| _ = mlx.mlx_array_free(x);
        if (self.up_b) |x| _ = mlx.mlx_array_free(x);
    }
    fn forward(self: *const UpBlock3D, parent: *Graph, x: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        var h = x;
        for (self.resnets) |*r| h = try r.forward(&g, vae_norm, h);
        if (self.up_w) |uw| h = try conv_blk.upsample2x(&g, h, uw, self.up_b.?);
        return parent.keep(h);
    }
};

/// Per-channel latent statistics [16] in `dt`.
fn latentStat(g: *Graph, vals: *const [16]f32, dt: ops.Dtype) !Array {
    var c = g.scope();
    defer c.deinit();
    return g.keep(try c.astype(try c.fromSlice(f32, vals, &.{16}), dt));
}

pub const Vae = struct {
    allocator: std.mem.Allocator,
    s: S,
    pq_w: mlx.mlx_array, // post_quant_conv (1x1x1)
    pq_b: mlx.mlx_array,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: AttnBlock3D,
    mid_r1: Resnet,
    up_blocks: [4]UpBlock3D,
    norm_out: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(self: *Vae) void {
        inline for (.{ "pq_w", "pq_b", "conv_in_w", "conv_in_b", "norm_out", "conv_out_w", "conv_out_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        freeResnet(&self.mid_r0);
        self.mid_attn.deinit();
        freeResnet(&self.mid_r1);
        for (&self.up_blocks) |*b| b.deinit();
    }

    /// latent [1,16,lat_h,lat_w] f32 → image [1,3,H,W] f32 in [-1,1].
    pub fn decode(self: *const Vae, parent: *Graph, latent: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        const l = try g.astype(try g.transpose(latent, &.{ 0, 2, 3, 1 }), VAE_DT); // NHWC
        // denorm: latent * STD + MEAN (per channel)
        const dn = try g.add(try g.mul(l, try latentStat(&g, &LATENTS_STD, VAE_DT)), try latentStat(&g, &LATENTS_MEAN, VAE_DT));
        var h = try conv_blk.conv2d(&g, dn, self.pq_w, self.pq_b, 1, 0, 1);
        h = try conv_blk.conv2d(&g, h, self.conv_in_w, self.conv_in_b, 1, 1, 1);
        h = try self.mid_r0.forward(&g, vae_norm, h);
        h = try self.mid_attn.forward(&g, h);
        h = try self.mid_r1.forward(&g, vae_norm, h);
        for (&self.up_blocks) |*b| h = try b.forward(&g, h);
        h = try g.silu(try g.rmsNorm(h, self.norm_out, 1e-12));
        h = try conv_blk.conv2d(&g, h, self.conv_out_w, self.conv_out_b, 1, 1, 1);
        return parent.keep(try g.contiguous(try g.astype(try g.transpose(h, &.{ 0, 3, 1, 2 }), .float32)));
    }
};

/// Load a causal conv3d weight as the conv2d it is over one frame: raw PyTorch
/// [out,in,kt,kh,kw] → its last temporal tap as MLX [out,kh,kw,in].
fn loadConv3dW(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try fmtKey(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const kt = mlx.getShape(raw)[2];
    const tap = try sliceAxis(raw, 2, kt - 1, kt, s);
    defer _ = mlx.mlx_array_free(tap);
    const t = try transpose(tap, &[_]c_int{ 0, 2, 3, 4, 1 }, s); // [out,1,kh,kw,in]
    defer _ = mlx.mlx_array_free(t);
    const sh = mlx.getShape(t);
    const t4 = try reshape(t, &[_]c_int{ sh[0], sh[2], sh[3], sh[4] }, s);
    defer _ = mlx.mlx_array_free(t4);
    const tc = try contig(t4, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, VAE_DT, s);
}
/// Load a conv2d weight: raw [out,in,kh,kw] → MLX [out,kh,kw,in].
fn loadConv2dW(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try fmtKey(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const raw = try ownWeight(w, wk);
    defer _ = mlx.mlx_array_free(raw);
    const t = try transpose(raw, &[_]c_int{ 0, 2, 3, 1 }, s);
    defer _ = mlx.mlx_array_free(t);
    const tc = try contig(t, s);
    defer _ = mlx.mlx_array_free(tc);
    return astype(tc, VAE_DT, s);
}
fn loadVec(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}." ++ suffix, .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, VAE_DT, s);
}
/// Load a `.gamma` RMSNorm vector, flattened to [C].
fn loadGamma(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, s: S) !mlx.mlx_array {
    const k = try fmtKey(a, "{s}.gamma", .{prefix});
    defer a.free(k);
    const raw = try ownWeight(w, k);
    defer _ = mlx.mlx_array_free(raw);
    const C: c_int = mlx.getShape(raw)[0];
    const flat = try reshape(raw, &[_]c_int{C}, s);
    defer _ = mlx.mlx_array_free(flat);
    return astype(flat, VAE_DT, s);
}

fn loadResBlock(w: *const Weights, a: std.mem.Allocator, pfx: []const u8, s: S) !Resnet {
    const c1 = try fmtKey(a, "{s}.conv1", .{pfx});
    defer a.free(c1);
    const c2 = try fmtKey(a, "{s}.conv2", .{pfx});
    defer a.free(c2);
    const n1 = try fmtKey(a, "{s}.norm1", .{pfx});
    defer a.free(n1);
    const n2 = try fmtKey(a, "{s}.norm2", .{pfx});
    defer a.free(n2);
    const sk = try fmtKey(a, "{s}.conv_shortcut", .{pfx});
    defer a.free(sk);
    const skwk = try fmtKey(a, "{s}.conv_shortcut.weight", .{pfx});
    defer a.free(skwk);
    var r: Resnet = .{
        .n1w = try loadGamma(w, a, n1, s),
        .c1w = try loadConv3dW(w, a, c1, s),
        .c1b = try loadVec(w, a, c1, "bias", s),
        .n2w = try loadGamma(w, a, n2, s),
        .c2w = try loadConv3dW(w, a, c2, s),
        .c2b = try loadVec(w, a, c2, "bias", s),
    };
    if (w.get(skwk) != null) {
        r.sw = try loadConv3dW(w, a, sk, s);
        r.sb = try loadVec(w, a, sk, "bias", s);
    }
    return r;
}

pub fn loadVae(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8) !Vae {
    const dir = try fmtKey(allocator, "{s}/vae", .{model_dir});
    defer allocator.free(dir);
    var w = try model_mod.loadWeights(io, allocator, dir);
    defer w.deinit();
    var v: Vae = undefined;
    v.allocator = allocator;
    v.s = s;
    v.pq_w = try loadConv3dW(&w, allocator, "post_quant_conv", s);
    v.pq_b = try loadVec(&w, allocator, "post_quant_conv", "bias", s);
    v.conv_in_w = try loadConv3dW(&w, allocator, "decoder.conv_in", s);
    v.conv_in_b = try loadVec(&w, allocator, "decoder.conv_in", "bias", s);
    v.mid_r0 = try loadResBlock(&w, allocator, "decoder.mid_block.resnets.0", s);
    v.mid_r1 = try loadResBlock(&w, allocator, "decoder.mid_block.resnets.1", s);
    v.mid_attn = .{
        .norm = try loadGamma(&w, allocator, "decoder.mid_block.attentions.0.norm", s),
        .qkv_w = try loadConv2dW(&w, allocator, "decoder.mid_block.attentions.0.to_qkv", s),
        .qkv_b = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.to_qkv", "bias", s),
        .proj_w = try loadConv2dW(&w, allocator, "decoder.mid_block.attentions.0.proj", s),
        .proj_b = try loadVec(&w, allocator, "decoder.mid_block.attentions.0.proj", "bias", s),
    };
    // up_blocks: 0,1,2 upsample (resample.1 conv); 3 has no upsampler.
    const nres = [_]usize{ 3, 3, 3, 3 };
    for (0..4) |bi| {
        var ub: UpBlock3D = .{ .resnets = try allocator.alloc(Resnet, nres[bi]), .allocator = allocator };
        for (0..nres[bi]) |ri| {
            const pfx = try fmtKey(allocator, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer allocator.free(pfx);
            ub.resnets[ri] = try loadResBlock(&w, allocator, pfx, s);
        }
        if (bi < 3) {
            const up = try fmtKey(allocator, "decoder.up_blocks.{d}.upsamplers.0.resample.1", .{bi});
            defer allocator.free(up);
            ub.up_w = try loadConv2dW(&w, allocator, up, s);
            ub.up_b = try loadVec(&w, allocator, up, "bias", s);
        }
        v.up_blocks[bi] = ub;
    }
    v.norm_out = try loadGamma(&w, allocator, "decoder.norm_out", s);
    v.conv_out_w = try loadConv3dW(&w, allocator, "decoder.conv_out", s);
    v.conv_out_b = try loadVec(&w, allocator, "decoder.conv_out", "bias", s);
    return v;
}

/// One entry of the encoder's flat down_blocks list (resnets at 0,1,3,4,6,7,9,10;
/// spatial resamples at 2,5,8).
const EncBlock = union(enum) {
    res: Resnet,
    down: struct { w: mlx.mlx_array, b: mlx.mlx_array },
};

/// Qwen-Image 3D VAE encoder (T=1) — structural mirror of the decoder. Feeds
/// img2img: pixels → per-channel-normalized latents, the space `Vae.decode` reads.
pub const VaeEncoder = struct {
    allocator: std.mem.Allocator,
    s: S,
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    blocks: [11]EncBlock,
    mid_r0: Resnet,
    mid_attn: AttnBlock3D,
    mid_r1: Resnet,
    norm_out: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,
    quant_w: mlx.mlx_array,
    quant_b: mlx.mlx_array,

    pub fn deinit(self: *VaeEncoder) void {
        inline for (.{ "conv_in_w", "conv_in_b", "norm_out", "conv_out_w", "conv_out_b", "quant_w", "quant_b" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        for (&self.blocks) |*blk| switch (blk.*) {
            .res => |*r| freeResnet(r),
            .down => |*d| {
                _ = mlx.mlx_array_free(d.w);
                _ = mlx.mlx_array_free(d.b);
            },
        };
        freeResnet(&self.mid_r0);
        self.mid_attn.deinit();
        freeResnet(&self.mid_r1);
    }

    /// image [1,3,H,W] f32 [0,1] (NCHW, H/W multiples of 16) → latents
    /// [1,16,H/8,W/8] f32, per-channel NORMALIZED ((z-mean)/std; distribution
    /// mean — deterministic, no sampling).
    pub fn encode(self: *const VaeEncoder, parent: *Graph, img: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        // [0,1] → [-1,1], NHWC (T=1)
        const signed = try g.sub(try g.mul(img, try g.scalar(2.0, .float32)), try g.scalar(1.0, .float32));
        var h = try g.astype(try g.transpose(signed, &.{ 0, 2, 3, 1 }), VAE_DT);
        h = try conv_blk.conv2d(&g, h, self.conv_in_w, self.conv_in_b, 1, 1, 1);
        for (&self.blocks) |*blk| h = switch (blk.*) {
            .res => |*rb| try rb.forward(&g, vae_norm, h),
            .down => |*d| try conv_blk.downsample2x(&g, h, d.w, d.b),
        };
        h = try self.mid_r0.forward(&g, vae_norm, h);
        h = try self.mid_attn.forward(&g, h);
        h = try self.mid_r1.forward(&g, vae_norm, h);
        h = try g.silu(try g.rmsNorm(h, self.norm_out, 1e-12));
        const co = try conv_blk.conv2d(&g, h, self.conv_out_w, self.conv_out_b, 1, 1, 1); // [1,h,w,32]
        const qc = try conv_blk.conv2d(&g, co, self.quant_w, self.quant_b, 1, 0, 1);
        // mean = first 16 channels; logvar discarded. normalize: (z - MEAN) / STD (inverse of decode's denorm)
        const mf = try g.astype(try g.sliceAxis(qc, 3, 0, 16), .float32);
        const zn = try g.div(try g.sub(mf, try latentStat(&g, &LATENTS_MEAN, .float32)), try latentStat(&g, &LATENTS_STD, .float32));
        return parent.keep(try g.contiguous(try g.transpose(zn, &.{ 0, 3, 1, 2 })));
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
    v.conv_in_w = try loadConv3dW(&w, allocator, "encoder.conv_in", s);
    v.conv_in_b = try loadVec(&w, allocator, "encoder.conv_in", "bias", s);
    for (0..11) |bi| {
        const pfx = try fmtKey(allocator, "encoder.down_blocks.{d}", .{bi});
        defer allocator.free(pfx);
        if (bi == 2 or bi == 5 or bi == 8) {
            const rp = try fmtKey(allocator, "{s}.resample.1", .{pfx});
            defer allocator.free(rp);
            v.blocks[bi] = .{ .down = .{
                .w = try loadConv2dW(&w, allocator, rp, s),
                .b = try loadVec(&w, allocator, rp, "bias", s),
            } };
        } else {
            v.blocks[bi] = .{ .res = try loadResBlock(&w, allocator, pfx, s) };
        }
    }
    v.mid_r0 = try loadResBlock(&w, allocator, "encoder.mid_block.resnets.0", s);
    v.mid_r1 = try loadResBlock(&w, allocator, "encoder.mid_block.resnets.1", s);
    v.mid_attn = .{
        .norm = try loadGamma(&w, allocator, "encoder.mid_block.attentions.0.norm", s),
        .qkv_w = try loadConv2dW(&w, allocator, "encoder.mid_block.attentions.0.to_qkv", s),
        .qkv_b = try loadVec(&w, allocator, "encoder.mid_block.attentions.0.to_qkv", "bias", s),
        .proj_w = try loadConv2dW(&w, allocator, "encoder.mid_block.attentions.0.proj", s),
        .proj_b = try loadVec(&w, allocator, "encoder.mid_block.attentions.0.proj", "bias", s),
    };
    v.norm_out = try loadGamma(&w, allocator, "encoder.norm_out", s);
    v.conv_out_w = try loadConv3dW(&w, allocator, "encoder.conv_out", s);
    v.conv_out_b = try loadVec(&w, allocator, "encoder.conv_out", "bias", s);
    v.quant_w = try loadConv3dW(&w, allocator, "quant_conv", s);
    v.quant_b = try loadVec(&w, allocator, "quant_conv", "bias", s);
    return v;
}

/// Conditioning rebalance: scale the stacked context [1,L,taps,d] by a global
/// `gain` and optional per-tapped-layer `weights` (len must equal the tap axis
/// — 12 for Krea). Same dtype out.
pub fn applyCondRebalance(parent: *Graph, ctx: Array, gain: f32, weights: ?[]const f32) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(ctx); // [1,L,n,d]
    const dt = ops.dtype(ctx);
    var cur = ctx;
    if (weights) |w| {
        if (w.len != @as(usize, @intCast(sh[2]))) return error.InvalidCondWeights;
        cur = try g.mul(cur, try g.fromSlice(f32, w, &.{ 1, 1, sh[2], 1 }));
    }
    if (gain != 1.0) cur = try g.mul(cur, try g.scalar(gain, .float32));
    if (ops.dtype(cur) != dt) cur = try g.astype(cur, dt);
    return parent.keep(cur);
}

// ════════════════════════════════════════════════════════════════════════
// B4. Sampler (flow-match Euler, resolution-shifted schedule) + pipeline.
// ════════════════════════════════════════════════════════════════════════

const SPATIAL_SCALE: u32 = 8;
const MIN_RES: u32 = 256;
const MAX_RES: u32 = 1280;

fn roundup(value: u32, multiple: u32) u32 {
    return ((value + multiple - 1) / multiple) * multiple;
}

/// patchify NCHW [n,16,lat_h,lat_w] (p=2, [c ph pw]) → [n, h_*w_, 64].
fn patchify(parent: *Graph, x: Array, p: c_int) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(x); // [n,c,H,W]
    const n = sh[0];
    const c = sh[1];
    const h = @divExact(sh[2], p);
    const wq = @divExact(sh[3], p);
    const t = try g.transpose(try g.reshape(x, &.{ n, c, h, p, wq, p }), &.{ 0, 2, 4, 1, 3, 5 }); // (n,h,w,c,p,p)
    return parent.keep(try g.reshape(t, &.{ n, h * wq, c * p * p }));
}

/// unpatchify [n,h_*w_,64] → NCHW [n,16,lat_h,lat_w].
fn unpatchify(parent: *Graph, x: Array, p: c_int, h: c_int, wq: c_int, c: c_int) !Array {
    var g = parent.scope();
    defer g.deinit();
    const n = ops.shape(x)[0];
    const t = try g.transpose(try g.reshape(x, &.{ n, h, wq, c, p, p }), &.{ 0, 3, 1, 4, 2, 5 }); // (n,c,h,p,w,p)
    return parent.keep(try g.reshape(t, &.{ n, c, h * p, wq * p }));
}

/// Resolution-shifted flow-match schedule (steps+1 values, 1→0): mu on the line through the 256² and 1280² token
/// counts, the exponential shift over linspace(1, 0, steps+1).
fn computeTimesteps(allocator: std.mem.Allocator, seq_len: u32, steps: u32) ![]f64 {
    const ts = try allocator.alloc(f64, steps + 1);
    const align_ = SPATIAL_SCALE * 2; // spatial_scale * patch = 16
    const x1: f64 = @floatFromInt((MIN_RES / align_) * (MIN_RES / align_));
    const x2: f64 = @floatFromInt((MAX_RES / align_) * (MAX_RES / align_));
    const emu = std.math.exp(flow_match.muLinear(@floatFromInt(seq_len), x1, 0.5, x2, 1.15));
    for (ts, 0..) |*t, i| {
        const lin = flow_match.gridToZero(f64, i, steps);
        t.* = if (lin <= 0.0) 0.0 else flow_match.timeShiftExp(emu, lin);
    }
    return ts;
}

fn stageMs(io: std.Io, t0: *std.Io.Timestamp) u64 {
    const ns = t0.untilNow(io, .awake).nanoseconds;
    t0.* = std.Io.Timestamp.now(io, .awake);
    return @intCast(@divTrunc(ns, 1_000_000));
}

/// Indices of the rows whose validity is non-zero, in order.
fn keptRows(allocator: std.mem.Allocator, valid: []const i32) ![]i32 {
    var n: usize = 0;
    for (valid) |v| n += @intFromBool(v != 0);
    const out = try allocator.alloc(i32, n);
    var j: usize = 0;
    for (valid, 0..) |v, i| if (v != 0) {
        out[j] = @intCast(i);
        j += 1;
    };
    return out;
}

fn buildPositions(allocator: std.mem.Allocator, txtlen: usize, h_: usize, w_: usize) ![]i32 {
    const L = txtlen + h_ * w_;
    const out = try allocator.alloc(i32, L * 3);
    @memset(out, 0);
    var idx = txtlen;
    for (0..h_) |row| {
        for (0..w_) |col| {
            out[idx * 3 + 1] = @intCast(row);
            out[idx * 3 + 2] = @intCast(col);
            idx += 1;
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Engine — owns the three sub-models + tokenizer. load/deinit/generate(Png).
// ════════════════════════════════════════════════════════════════════════

/// Per-request generation options beyond the core prompt/size/steps.
pub const GenOpts = struct {
    /// img2img: source pixels [1,3,H,W] f32 [0,1], pre-resized to the target
    /// size. VAE-encoded internally and mixed with noise at the start timestep.
    init_image: ?mlx.mlx_array = null,
    /// First schedule index to run (img2img skip; 0 = full schedule).
    start_step: u32 = 0,
    /// Conditioning rebalance: global gain + per-tapped-layer weights (len 12).
    cond_gain: f32 = 1.0,
    cond_weights: ?[]const f32 = null,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    te: Conditioner,
    dit: Dit,
    vae: Vae,
    vae_enc: ?VaeEncoder,
    tok: tok_mod.Tokenizer,

    /// Every array of the engine lives and runs on `s`.
    pub fn loadOn(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;
        self.te = try loadConditioner(io, allocator, self.s, model_dir);
        errdefer self.te.deinit();
        self.dit = try loadDit(io, allocator, self.s, model_dir);
        errdefer self.dit.deinit();
        self.vae = try loadVae(io, allocator, self.s, model_dir);
        errdefer self.vae.deinit();
        self.vae_enc = loadVaeEncoder(io, allocator, self.s, model_dir) catch |e| blk: {
            log.warn("[image] krea VAE encoder load failed ({}) — image-to-image disabled\n", .{e});
            break :blk null;
        };
        errdefer if (self.vae_enc) |*e| e.deinit();
        const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
        defer allocator.free(tok_dir);
        self.tok = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
        log.info("[image] Krea-2 models + tokenizer ready\n", .{});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.te.deinit();
        self.dit.deinit();
        self.vae.deinit();
        if (self.vae_enc) |*e| e.deinit();
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    /// Build the templated ids/mask (PREFIX+prompt padded to 541 + SUFFIX 5 →
    /// 546). The encoder drops the first 34 (PREFIX) tokens.
    fn buildPromptIds(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8) !struct { ids: []i32, mask: []i32 } {
        const pre_max = MAX_LENGTH + PREFIX_START_IDX - SUFFIX_START_IDX; // 541
        const total = pre_max + SUFFIX_START_IDX; // 546
        const ids = try allocator.alloc(i32, total);
        const mask = try allocator.alloc(i32, total);

        const pre_text = try std.fmt.allocPrint(allocator, "{s}{s}", .{ PREFIX, prompt });
        defer allocator.free(pre_text);
        const enc = try self.tok.encode(allocator, pre_text);
        defer allocator.free(enc);
        const suf = try self.tok.encode(allocator, SUFFIX);
        defer allocator.free(suf);

        const real = @min(enc.len, pre_max);
        for (0..pre_max) |i| {
            if (i < real) {
                ids[i] = @intCast(enc[i]);
                mask[i] = 1;
            } else {
                ids[i] = PAD_TOKEN;
                mask[i] = 0;
            }
        }
        for (0..SUFFIX_START_IDX) |i| {
            ids[pre_max + i] = if (i < suf.len) @intCast(suf[i]) else PAD_TOKEN;
            mask[pre_max + i] = 1;
        }
        return .{ .ids = ids, .mask = mask };
    }

    /// Full pipeline. ids/mask are the templated 546-length sequence. Returns
    /// [1,3,H,W] f32 in [0,1] (owned mlx array; caller frees).
    pub fn generate(self: *Engine, allocator: std.mem.Allocator, ids: []const i32, mask: []const i32, seed: u64, steps: u32, width: u32, height: u32, progress: ?sse.Progress) !mlx.mlx_array {
        return self.generateWithOpts(allocator, ids, mask, seed, steps, width, height, .{}, progress);
    }

    pub fn generateWithOpts(self: *Engine, allocator: std.mem.Allocator, ids: []const i32, mask: []const i32, seed: u64, steps: u32, width: u32, height: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
        var g = Graph.init(allocator, self.s);
        defer g.deinit();
        const cfg = self.dit.cfg;
        const align_ = SPATIAL_SCALE * cfg.patch; // 16
        const W = roundup(width, align_);
        const H = roundup(height, align_);
        const lat_h = H / SPATIAL_SCALE;
        const lat_w = W / SPATIAL_SCALE;
        const h_ = lat_h / cfg.patch;
        const w_ = lat_w / cfg.patch;
        const img_len = h_ * w_;

        // 1. text encode → context [1,512,12,txtdim], validity [512]
        var stage_t0 = std.Io.Timestamp.now(self.dit.io, .awake);
        var ctx = try self.te.encode(&g, ids, mask);
        if (opts.cond_gain != 1.0 or opts.cond_weights != null) ctx = try applyCondRebalance(&g, ctx, opts.cond_gain, opts.cond_weights);
        // Pad text rows only ever meet a -1e9 mask, so dropping them is exact and
        // leaves every row valid (no mask); an all-pad prompt keeps the masked layout.
        const txt_valid = mask[PREFIX_START_IDX..][0..@intCast(ops.shape(ctx)[1])];
        const kept = try keptRows(allocator, txt_valid);
        defer allocator.free(kept);
        const trim = kept.len > 0;
        if (trim) ctx = try g.take(ctx, try g.fromSlice(i32, kept, &.{@intCast(kept.len)}), 1);
        const txtlen: usize = @intCast(ops.shape(ctx)[1]);

        // 2. init noise [1,16,lat_h,lat_w] → patchify [1,img_len,64]
        const key = try g.call(mlx.mlx_random_key, .{seed});
        const nsh = [_]c_int{ 1, @intCast(cfg.channels), @intCast(lat_h), @intCast(lat_w) };
        const noise = try g.call(mlx.mlx_random_normal, .{ &nsh, nsh.len, .float32, 0.0, 1.0, key });
        var img = try patchify(&g, try g.astype(noise, .bfloat16), @intCast(cfg.patch));

        // 3. positions, validity, rope, masks (constant across steps)
        const pos = try buildPositions(allocator, txtlen, h_, w_);
        defer allocator.free(pos);
        const valid = try allocator.alloc(i32, txtlen + img_len);
        defer allocator.free(valid);
        @memset(valid, 1);
        if (!trim) @memcpy(valid[0..txtlen], txt_valid);

        const rope = try buildRope(&g, allocator, pos, valid.len, cfg.theta);
        const full_mask: ?Array = if (trim) null else try outerMask(&g, valid);
        const txt_mask: ?Array = if (trim) null else try outerMask(&g, valid[0..txtlen]);
        const fused = try self.dit.textFusion(&g, ctx, txt_mask);
        try ops.eval(fused);
        const prompt_ms = stageMs(self.dit.io, &stage_t0);

        // 4. schedule + denoise
        const ts = try computeTimesteps(allocator, img_len, steps);
        defer allocator.free(ts);
        const start_step: u32 = @min(opts.start_step, steps - 1);

        // img2img: VAE-encode the source, mix x = (1-t)·z0 + t·noise at the
        // start timestep (flow-match scale_noise).
        if (opts.init_image) |pix| {
            const ve = if (self.vae_enc) |*e| e else return error.NoVaeEncoder;
            const z_tok = try patchify(&g, try g.astype(try ve.encode(&g, pix), .bfloat16), @intCast(cfg.patch));
            const t0: f32 = @floatCast(ts[start_step]);
            const mixed = try g.add(try g.mul(z_tok, try g.scalar(1.0 - t0, .float32)), try g.mul(img, try g.scalar(t0, .float32)));
            img = try g.astype(mixed, .bfloat16);
        }

        const run_steps = steps - start_step;
        for (start_step..steps) |i| {
            if (progress) |p| if (p.boundary()) return error.Cancelled;
            var sg = g.scope();
            defer sg.deinit();
            const tc: f32 = @floatCast(ts[i]);
            const tp: f32 = @floatCast(ts[i + 1]);
            const v = try self.dit.forwardPrebuilt(&sg, img, fused, tc, rope.cos, rope.sin, full_mask);
            const next = try g.keep(try flow_match.eulerStep(&sg, img, v, tp - tc, .float32));
            g.drop(img);
            img = next;
            try ops.eval(img);
            if (progress) |p| p.emit("Generating", @intCast(i + 1 - start_step), run_steps);
        }
        const denoise_ms = stageMs(self.dit.io, &stage_t0);
        if (progress) |p| p.emit("Decoding image", steps, steps);

        // 5. unpatchify → [1,16,lat_h,lat_w] → VAE decode → [-1,1] → [0,1]
        const latent = try unpatchify(&g, img, @intCast(cfg.patch), @intCast(h_), @intCast(w_), @intCast(cfg.channels));
        const decoded = try self.vae.decode(&g, try g.astype(latent, .float32));
        try ops.eval(decoded);
        log.info("[krea] stages: prompt {d} ms, denoise {d} ms ({d} steps), vae decode {d} ms\n", .{ prompt_ms, denoise_ms, run_steps, stageMs(self.dit.io, &stage_t0) });
        // clip(decoded,-1,1)*0.5+0.5
        const clipped = try g.minimum(try g.maximum(decoded, try g.scalar(-1.0, .float32)), try g.scalar(1.0, .float32));
        const half = try g.scalar(0.5, .float32);
        return g.escape(try g.add(try g.mul(clipped, half), half));
    }

    /// Tokenize + run the pipeline → image [1,3,H,W] f32 [0,1] (owned mlx array;
    /// caller frees). Lets the caller run the content filter before PNG-encoding.
    pub fn generateImage(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, progress: ?sse.Progress) !mlx.mlx_array {
        return self.generateImageOpts(allocator, prompt, width, height, seed, steps, .{}, progress);
    }

    /// `generateImage` with per-request options (img2img + conditioning rebalance).
    pub fn generateImageOpts(self: *Engine, allocator: std.mem.Allocator, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32, opts: GenOpts, progress: ?sse.Progress) !mlx.mlx_array {
        const pr = try self.buildPromptIds(allocator, prompt);
        defer allocator.free(pr.ids);
        defer allocator.free(pr.mask);
        return self.generateWithOpts(allocator, pr.ids, pr.mask, seed, steps, width, height, opts, progress);
    }
};

/// [1,3/4,H,W] f32 [0,1] → RGB/RGBA8 PNG bytes (caller frees).
pub const imageToPng = @import("../../sdk/media/image_pixels.zig").toPng;

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

// ── Hermetic: MixedLinear bits/group_size inference + forward ──

test "MixedLinear bf16 forward (no scales)" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    const a = testing.allocator;
    // weight [out=2, in=3] = [[1,2,3],[4,5,6]]; bias [2] = [10,20]
    var wbuf = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const wsh = [_]c_int{ 2, 3 };
    const wa = mlx.mlx_array_new_data(&wbuf, &wsh, 2, .float32);
    var bbuf = [_]f32{ 10, 20 };
    const bsh = [_]c_int{2};
    const ba = mlx.mlx_array_new_data(&bbuf, &bsh, 1, .float32);
    var ww = model_mod.Weights.init(a);
    defer ww.deinit();
    try ww.map.put(try a.dupe(u8, "lin.weight"), wa);
    try ww.map.put(try a.dupe(u8, "lin.bias"), ba);

    var ml = try MixedLinear.load(&ww, a, "lin", 3, s);
    defer ml.deinit();
    try testing.expect(!ml.quantized);

    var xb = [_]f32{ 1, 1, 1 };
    const xsh = [_]c_int{ 1, 3 };
    const xa = mlx.mlx_array_new_data(&xb, &xsh, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const o = tg.escape(try ml.forward(&tg, xa));
    defer _ = mlx.mlx_array_free(o);
    const of = try astype(o, .float32, s);
    defer _ = mlx.mlx_array_free(of);
    _ = mlx.mlx_array_eval(of);
    const d = mlx.mlx_array_data_float32(of).?;
    // [1,1,1]·[1,2,3]+10 = 16 ; ·[4,5,6]+20 = 35
    try testing.expect(@abs(d[0] - 16.0) < 0.2);
    try testing.expect(@abs(d[1] - 35.0) < 0.2);
}

test "MixedLinear infers bits/group_size from quantized geometry" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    const a = testing.allocator;
    // Build a real affine-quantized weight via mlx_quantize so geometry is valid.
    // out=4, in=64, bits=4, gs=32 → packed cols = 64*4/32 = 8 ; scales cols = 64/32 = 2.
    const in: c_int = 64;
    const out: c_int = 4;
    const raw = try a.alloc(f32, @intCast(out * in));
    defer a.free(raw);
    for (raw, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
    const rsh = [_]c_int{ out, in };
    const rw = mlx.mlx_array_new_data(raw.ptr, &rsh, 2, .float32);
    defer _ = mlx.mlx_array_free(rw);
    const rwb = try astype(rw, .bfloat16, s);
    defer _ = mlx.mlx_array_free(rwb);

    inline for (.{ .{ 4, 32 }, .{ 8, 64 } }) |cfg| {
        const bits: c_int = cfg[0];
        const gs: c_int = cfg[1];
        var packed_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(packed_vec);
        const null_gscale = mlx.mlx_array{ .ctx = null };
        try mlx.check(mlx.mlx_quantize(&packed_vec, rwb, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", null_gscale, s));
        var qw = mlx.mlx_array_new();
        var qs = mlx.mlx_array_new();
        var qb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&qw, packed_vec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&qs, packed_vec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&qb, packed_vec, 2));

        var ww = model_mod.Weights.init(a);
        defer ww.deinit();
        try ww.map.put(try a.dupe(u8, "q.weight"), qw);
        try ww.map.put(try a.dupe(u8, "q.scales"), qs);
        try ww.map.put(try a.dupe(u8, "q.biases"), qb);
        var ml = try MixedLinear.load(&ww, a, "q", @intCast(in), s);
        defer ml.deinit();
        try testing.expect(ml.quantized);
        try testing.expectEqual(@as(u32, @intCast(bits)), ml.bits);
        try testing.expectEqual(@as(u32, @intCast(gs)), ml.group_size);

        // forward sanity vs a manual matmul against the dequantized weight.
        const xv = try a.alloc(f32, @intCast(in));
        defer a.free(xv);
        for (xv, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.2;
        const xsh = [_]c_int{ 1, in };
        const xa = mlx.mlx_array_new_data(xv.ptr, &xsh, 2, .float32);
        defer _ = mlx.mlx_array_free(xa);
        const o = tg.escape(try ml.forward(&tg, xa));
        defer _ = mlx.mlx_array_free(o);
        const of = try astype(o, .float32, s);
        defer _ = mlx.mlx_array_free(of);
        _ = mlx.mlx_array_eval(of);
        try testing.expectEqual(@as(usize, @intCast(out)), @as(usize, @intCast(mlx.mlx_array_size(of))));
    }
}

// ── Oracle tests (env-gated; fixtures from tests/dump_krea_fixtures.py) ──

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
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
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

test "MixedLinear applies an attached LoRA delta (bf16 path)" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    // Base: identity [in=2,out=2] (pre-transposed layout). x = [1,2].
    const wv = [_]f32{ 1, 0, 0, 1 };
    const wsh = [_]c_int{ 2, 2 };
    const wf = mlx.mlx_array_new_data(&wv, &wsh, 2, .float32);
    defer _ = mlx.mlx_array_free(wf);
    const wbf = try astype(wf, .bfloat16, s);
    var ml = MixedLinear{ .quantized = false, .w = wbf };
    defer ml.deinit();
    const xv = [_]f32{ 1, 2 };
    const xs = [_]c_int{ 1, 2 };
    const x = mlx.mlx_array_new_data(&xv, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    // Without LoRA: y = x.
    const y0 = tg.escape(try ml.forward(&tg, x));
    defer _ = mlx.mlx_array_free(y0);
    const y0f = try astype(y0, .float32, s);
    defer _ = mlx.mlx_array_free(y0f);
    _ = mlx.mlx_array_eval(y0f);
    const d0 = mlx.mlx_array_data_float32(y0f) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 1), d0[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 2), d0[1], 1e-3);
    // LoRA: at [2,1]=[1;1], bt [1,2]=[2,4], scale .5 → delta = [3,6]; y = [4,8].
    const atv = [_]f32{ 1, 1 };
    const ats = [_]c_int{ 2, 1 };
    const atf = mlx.mlx_array_new_data(&atv, &ats, 2, .float32);
    defer _ = mlx.mlx_array_free(atf);
    const at = try astype(atf, .bfloat16, s);
    defer _ = mlx.mlx_array_free(at);
    const btv = [_]f32{ 2, 4 };
    const bts = [_]c_int{ 1, 2 };
    const btf = mlx.mlx_array_new_data(&btv, &bts, 2, .float32);
    defer _ = mlx.mlx_array_free(btf);
    const bt = try astype(btf, .bfloat16, s);
    defer _ = mlx.mlx_array_free(bt);
    ml.setLoraRefs(&.{.{ .at = at, .bt = bt, .scale = 0.5 }});
    const y1 = tg.escape(try ml.forward(&tg, x));
    defer _ = mlx.mlx_array_free(y1);
    const y1f = try astype(y1, .float32, s);
    defer _ = mlx.mlx_array_free(y1f);
    _ = mlx.mlx_array_eval(y1f);
    const d1 = mlx.mlx_array_data_float32(y1f) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 4), d1[0], 1e-2);
    try testing.expectApproxEqAbs(@as(f32, 8), d1[1], 1e-2);
    // Detached again: back to base.
    ml.clearLoraRefs();
    const y2 = tg.escape(try ml.forward(&tg, x));
    defer _ = mlx.mlx_array_free(y2);
    const y2f = try astype(y2, .float32, s);
    defer _ = mlx.mlx_array_free(y2f);
    _ = mlx.mlx_array_eval(y2f);
    const d2 = mlx.mlx_array_data_float32(y2f) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 1), d2[0], 1e-3);
}

test "krea applyCondRebalance scales tapped layers and global gain" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    // ctx [1,2,3,2]: 2 tokens × 3 tapped layers × hidden 2.
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60 };
    const ctx = try g.fromSlice(f32, &vals, &.{ 1, 2, 3, 2 });
    const w = [_]f32{ 2, 3, 4 };
    const out = try applyCondRebalance(&g, ctx, 0.5, &w);
    try ops.eval(out);
    const d = mlx.mlx_array_data_float32(out) orelse return error.NoData;
    // Token 0: layer0 [1,2]*2*.5, layer1 [3,4]*3*.5, layer2 [5,6]*4*.5
    const expect0 = [_]f32{ 1, 2, 4.5, 6, 10, 12 };
    for (expect0, 0..) |e, i| try testing.expectApproxEqAbs(e, d[i], 1e-5);
    // Gain-only path.
    const gained = try applyCondRebalance(&g, ctx, 2.0, null);
    try ops.eval(gained);
    const gd = mlx.mlx_array_data_float32(gained) orelse return error.NoData;
    try testing.expectApproxEqAbs(@as(f32, 2), gd[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 120), gd[11], 1e-5);
    // Count must match the tap axis.
    const bad = [_]f32{ 1, 2 };
    try testing.expectError(error.InvalidCondWeights, applyCondRebalance(&g, ctx, 1.0, &bad));
}

// VAE encoder guard: encode a synthetic smooth image, decode it back — the
// pixels must round-trip through the oracle-validated decoder.  KREA_TEST_MODEL
test "krea VAE encoder round-trips through the decoder" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    var enc = try loadVaeEncoder(io, a, s, model_dir);
    defer enc.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const HW: usize = 512;
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
    const lat = tg.escape(try enc.encode(&tg, img));
    defer _ = mlx.mlx_array_free(lat);
    const lsh = mlx.getShape(lat);
    try testing.expectEqual(@as(c_int, 16), lsh[1]);
    try testing.expectEqual(@as(c_int, @intCast(HW / 8)), lsh[2]);
    const lat_f = try astype(lat, .float32, s);
    defer _ = mlx.mlx_array_free(lat_f);
    const dec = tg.escape(try vae.decode(&tg, lat_f)); // [-1,1]
    defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    try testing.expectEqual(img_buf.len, n);
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    var mapped = try a.alloc(f32, n);
    defer a.free(mapped);
    var mae: f64 = 0;
    for (0..n) |i| {
        mapped[i] = data[i] * 0.5 + 0.5;
        mae += @abs(@as(f64, mapped[i]) - img_buf[i]);
    }
    mae /= @floatFromInt(n);
    const corr = cosine(mapped, img_buf);
    std.debug.print("[krea-vae-enc] roundtrip corr={d:.6} mae={d:.5}\n", .{ corr, mae });
    try testing.expect(corr > 0.99);
    try testing.expect(mae < 0.05);
}

// Stage 1: text encoder. KREA_TEST_MODEL, KREA_IDS, KREA_MASK, KREA_PE.
test "krea text encoder reproduces hidden states" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("KREA_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("KREA_MASK") orelse return error.SkipZigTest);
    const pe_p = std.mem.span(std.c.getenv("KREA_PE") orelse return error.SkipZigTest);
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
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    var te = try loadConditioner(io, a, s, model_dir);
    defer te.deinit();
    const pe = tg.escape(try te.encode(&tg, ids, mask));
    defer _ = mlx.mlx_array_free(pe);
    const pf = try astype(pe, .float32, s);
    defer _ = mlx.mlx_array_free(pf);
    _ = mlx.mlx_array_eval(pf);
    const n: usize = @intCast(mlx.mlx_array_size(pf));
    const data = mlx.mlx_array_data_float32(pf) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-te] n={d} corr={d:.6}\n", .{ n, corr });
    // Default bar 0.997 is for a BF16 encoder (engine-correctness vs the bf16
    // reference): this compares the RAW Qwen3-VL hidden states of the 12 tapped
    // layers (up to layer 35), whose massive late-layer outlier activations (|x|
    // in the thousands) make the cosine hypersensitive to irreducible bf16
    // op-ordering noise. Measured 0.99806 bf16; the f32 norm-weight match lifts it
    // from 0.9963 (reverting `normF32` trips this bar — a real regression guard).
    // An 8-BIT-quantized encoder (the public bundle) lands ~0.9927 here — the
    // quant error on the outliers is large in this metric but washes out by the
    // pipeline output, so set KREA_TE_MIN=0.99 when validating an 8-bit dir. The
    // DEFINITIVE proofs are downstream: DiT 0.99995, VAE 1.0, e2e 0.9996 (bf16) /
    // 0.99964 (8-bit encoder).
    const min_corr: f64 = if (std.c.getenv("KREA_TE_MIN")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.997) else 0.997;
    try testing.expect(corr > min_corr);
}

/// A bf16 SwiGLU of 256-wide linears whose weights follow `seed`, int8-converted.
fn toySwiglu(seed: u64, s: S) !SwiGLU {
    var g: SwiGLU = undefined;
    inline for (.{ "gate", "up", "down" }, 0..) |name, i| {
        const w = try w8a8.randn(&.{ 256, 256 }, seed + i, .bfloat16, s);
        defer _ = mlx.mlx_array_free(w);
        const small = try mulA(w, scalarF(0.05), s);
        @field(g, name) = .{ .quantized = false, .w = small };
        try @field(g, name).toW8a8(s);
    }
    g.gu = null;
    return g;
}

test "krea SwiGLU: an adapter on the fused int8 gate/up equals the same adapter on the separate projections" {
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const prev = w8a8.w8a8State().enabled;
    defer w8a8.w8a8State().enabled = prev;
    w8a8.w8a8State().enabled = true;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    var fused = try toySwiglu(10, s);
    defer fused.deinit();
    var apart = try toySwiglu(10, s);
    defer apart.deinit();
    try fused.fuseW8a8(s);
    try testing.expect(fused.gu != null and fused.gate.w8 == null and apart.gu == null);

    const x = try w8a8.randn(&.{ 1, 300, 256 }, 20, .bfloat16, s);
    defer _ = mlx.mlx_array_free(x);
    const plain_f = tg.escape(try fused.forward(&tg, x));
    defer _ = mlx.mlx_array_free(plain_f);
    const plain_a = tg.escape(try apart.forward(&tg, x));
    defer _ = mlx.mlx_array_free(plain_a);
    // The epilogue rounds once where the separate chain rounds three times.
    try testing.expect(try w8a8.relRms(plain_f, plain_a, s) < 0.02);

    var refs: [2]lora_mod.Ref = undefined;
    for (&refs, 0..) |*r, i| {
        const at = try w8a8.randn(&.{ 256, 4 }, 30 + i, .bfloat16, s);
        const bt = try w8a8.randn(&.{ 4, 256 }, 40 + i, .bfloat16, s);
        r.* = .{ .at = at, .bt = bt, .scale = 0.05 };
    }
    defer for (refs) |r| {
        _ = mlx.mlx_array_free(r.at);
        _ = mlx.mlx_array_free(r.bt);
    };
    inline for (.{ &fused, &apart }) |m| {
        m.gate.setLoraRefs(refs[0..1]);
        m.up.setLoraRefs(refs[1..2]);
    }
    const lora_f = tg.escape(try fused.forward(&tg, x));
    defer _ = mlx.mlx_array_free(lora_f);
    const lora_a = tg.escape(try apart.forward(&tg, x));
    defer _ = mlx.mlx_array_free(lora_a);
    // Same int8 weights and activations on both sides: only the gate/up rounding differs.
    try testing.expect(try w8a8.relRms(lora_f, lora_a, s) < 0.02);
    try testing.expect(try w8a8.relRms(lora_f, plain_f, s) > 0.01); // the adapter moved the output
}

// Stage 2: DiT one forward → velocity at t=1.0.
// KREA_TEST_MODEL, KREA_VEL_IMG, KREA_VEL_CTX, KREA_VEL_POS, KREA_VEL_VALID, KREA_VEL.
test "krea DiT reproduces velocity" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const img_p = std.mem.span(std.c.getenv("KREA_VEL_IMG") orelse return error.SkipZigTest);
    const ctx_p = std.mem.span(std.c.getenv("KREA_VEL_CTX") orelse return error.SkipZigTest);
    const pos_p = std.mem.span(std.c.getenv("KREA_VEL_POS") orelse return error.SkipZigTest);
    const valid_p = std.mem.span(std.c.getenv("KREA_VEL_VALID") orelse return error.SkipZigTest);
    const vel_p = std.mem.span(std.c.getenv("KREA_VEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const imgd = try readF32(io, a, img_p);
    defer a.free(imgd);
    const ctxd = try readF32(io, a, ctx_p);
    defer a.free(ctxd);
    const pos = try readI32(io, a, pos_p);
    defer a.free(pos);
    const valid = try readI32(io, a, valid_p);
    defer a.free(valid);
    const ref = try readF32(io, a, vel_p);
    defer a.free(ref);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();

    const img_len: c_int = @intCast(imgd.len / 64);
    const L = valid.len;
    const txtlen: c_int = @intCast(L - @as(usize, @intCast(img_len)));
    const img_sh = [_]c_int{ 1, img_len, 64 };
    const img_f = mlx.mlx_array_new_data(imgd.ptr, &img_sh, 3, .float32);
    defer _ = mlx.mlx_array_free(img_f);
    const img_bf = try astype(img_f, .bfloat16, s);
    defer _ = mlx.mlx_array_free(img_bf);
    const ctx_sh = [_]c_int{ 1, txtlen, 12, 2560 };
    const ctx_f = mlx.mlx_array_new_data(ctxd.ptr, &ctx_sh, 4, .float32);
    defer _ = mlx.mlx_array_free(ctx_f);
    const ctx_bf = try astype(ctx_f, .bfloat16, s);
    defer _ = mlx.mlx_array_free(ctx_bf);

    const vel = tg.escape(try dit.forward(&tg, img_bf, ctx_bf, 1.0, pos, valid));
    defer _ = mlx.mlx_array_free(vel);
    const vf = try astype(vel, .float32, s);
    defer _ = mlx.mlx_array_free(vf);
    _ = mlx.mlx_array_eval(vf);
    const n: usize = @intCast(mlx.mlx_array_size(vf));
    const data = mlx.mlx_array_data_float32(vf) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-dit] n={d} corr={d:.6}\n", .{ n, corr });
    try testing.expect(corr > 0.99);
}

/// One DiT forward on seeded noise (16 x 16 latent tokens + 512 text rows), velocity as f32;
/// `resident` receives the MLX bytes the loaded DiT holds.
fn kreaProbe(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, resident: *usize) !mlx.mlx_array {
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    var dit = try loadDit(io, a, s, model_dir);
    defer dit.deinit();
    var tg = Graph.init(a, s);
    defer tg.deinit();
    const h_: usize = 16;
    const txtlen: usize = 512;
    const img_len: c_int = @intCast(h_ * h_);
    const img = try w8a8.randn(&.{ 1, img_len, 64 }, 1, .bfloat16, s);
    defer _ = mlx.mlx_array_free(img);
    const ctx = try w8a8.randn(&.{ 1, @intCast(txtlen), 12, 2560 }, 2, .bfloat16, s);
    defer _ = mlx.mlx_array_free(ctx);
    const pos = try buildPositions(a, txtlen, h_, h_);
    defer a.free(pos);
    const valid = try a.alloc(i32, txtlen + h_ * h_);
    defer a.free(valid);
    @memset(valid, 1);
    const vel = tg.escape(try dit.forward(&tg, img, ctx, 1.0, pos, valid));
    defer _ = mlx.mlx_array_free(vel);
    const out = try astype(vel, .float32, s);
    try mlx.check(mlx.mlx_array_eval(out));
    // Weights are lazy until the first forward has read them.
    w8a8.dropMemo(); // job-end state: the last activation is not weights
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    resident.* = after -| before;
    return out;
}

// The lossy int8 route against the same pack's packed forward (no reference fixture
// needed). KREA_TEST_MODEL.
test "krea DiT: the w8a8 route stays within its bar of the packed forward" {
    if (!@import("../../mlx/device.zig").naxAvailable()) return error.SkipZigTest;
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const prev = w8a8.w8a8State().enabled;
    defer w8a8.w8a8State().enabled = prev;
    w8a8.w8a8State().enabled = false;
    var packed_bytes: usize = 0;
    const want = try kreaProbe(io, a, s, model_dir, &packed_bytes);
    defer _ = mlx.mlx_array_free(want);
    w8a8.w8a8State().enabled = true;
    var int8_bytes: usize = 0;
    const got = try kreaProbe(io, a, s, model_dir, &int8_bytes);
    defer _ = mlx.mlx_array_free(got);
    const g = try w8a8.agreement(got, want, s);
    std.debug.print("[krea-dit w8a8] cos={d:.6} rms_ratio={d:.6} resident packed={d} int8={d}\n", .{ g.cos, g.rms_ratio, packed_bytes, int8_bytes });
    try w8a8.expectBilled(packed_bytes, int8_bytes, try w8a8.billedDelta(io, model_dir, &.{"transformer_mixed_4_8.safetensors"}));
    try testing.expect(g.cos < 0.99999); // the route engaged: a packed-numerics output would be identical
    try testing.expect(g.cos > 0.995);
    try testing.expectApproxEqAbs(@as(f64, 1.0), g.rms_ratio, 0.03);
}

// Stage 3: VAE decode. KREA_TEST_MODEL, KREA_VAE_LATENT, KREA_DECODED.
test "krea VAE reproduces decoded image" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const lat_p = std.mem.span(std.c.getenv("KREA_VAE_LATENT") orelse return error.SkipZigTest);
    const dec_p = std.mem.span(std.c.getenv("KREA_DECODED") orelse return error.SkipZigTest);
    const lath_s = std.mem.span(std.c.getenv("KREA_VAE_LATH") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const lat = try readF32(io, a, lat_p);
    defer a.free(lat);
    const ref = try readF32(io, a, dec_p);
    defer a.free(ref);
    const lath = try std.fmt.parseInt(c_int, lath_s, 10);
    const latw: c_int = @intCast(@divExact(lat.len, @as(usize, @intCast(16 * lath))));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var tg = Graph.init(testing.allocator, s);
    defer tg.deinit();
    var vae = try loadVae(io, a, s, model_dir);
    defer vae.deinit();
    const lat_sh = [_]c_int{ 1, 16, lath, latw };
    const lat_f = mlx.mlx_array_new_data(lat.ptr, &lat_sh, 4, .float32);
    defer _ = mlx.mlx_array_free(lat_f);
    const dec = tg.escape(try vae.decode(&tg, lat_f));
    defer _ = mlx.mlx_array_free(dec);
    const df = try astype(dec, .float32, s);
    defer _ = mlx.mlx_array_free(df);
    _ = mlx.mlx_array_eval(df);
    const n: usize = @intCast(mlx.mlx_array_size(df));
    const data = mlx.mlx_array_data_float32(df) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-vae] n={d} corr={d:.6}\n", .{ n, corr });
    try testing.expect(corr > 0.99);
}

// Stage 4: end-to-end. KREA_TEST_MODEL, KREA_IDS, KREA_MASK, KREA_E2E (final
// image [0,1]), KREA_E2E_W, KREA_E2E_H, KREA_E2E_SEED, KREA_E2E_STEPS.
test "krea end-to-end pipeline matches reference image" {
    const model_dir = std.mem.span(std.c.getenv("KREA_TEST_MODEL") orelse return error.SkipZigTest);
    const ids_p = std.mem.span(std.c.getenv("KREA_IDS") orelse return error.SkipZigTest);
    const mask_p = std.mem.span(std.c.getenv("KREA_MASK") orelse return error.SkipZigTest);
    const e2e_p = std.mem.span(std.c.getenv("KREA_E2E") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try readI32(io, a, ids_p);
    defer a.free(ids);
    const mask = try readI32(io, a, mask_p);
    defer a.free(mask);
    const ref = try readF32(io, a, e2e_p);
    defer a.free(ref);
    const width: u32 = if (std.c.getenv("KREA_E2E_W")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 512;
    const height: u32 = if (std.c.getenv("KREA_E2E_H")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 512;
    const seed: u64 = if (std.c.getenv("KREA_E2E_SEED")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else 0;
    const steps: u32 = if (std.c.getenv("KREA_E2E_STEPS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 8;

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var eng: Engine = undefined;
    eng.allocator = a;
    eng.s = s;
    eng.te = try loadConditioner(io, a, s, model_dir);
    defer eng.te.deinit();
    eng.dit = try loadDit(io, a, s, model_dir);
    defer eng.dit.deinit();
    eng.vae = try loadVae(io, a, s, model_dir);
    defer eng.vae.deinit();

    const img = try eng.generate(a, ids, mask, seed, steps, width, height, null);
    defer _ = mlx.mlx_array_free(img);
    _ = mlx.mlx_array_eval(img);
    const n: usize = @intCast(mlx.mlx_array_size(img));
    const data = mlx.mlx_array_data_float32(img) orelse return error.NoData;
    try testing.expectEqual(ref.len, n);
    const corr = cosine(data[0..n], ref);
    std.debug.print("[krea-e2e] n={d} corr={d:.5}\n", .{ n, corr });
    try testing.expect(corr > 0.95);
}

test "krea geometry: an empty map keeps the release's widths" {
    var w = model_mod.Weights.init(testing.allocator);
    defer w.deinit();
    const d = KreaConfig.ditFrom(&w, testing.allocator);
    try testing.expectEqual(KreaConfig{}, d);
    try testing.expectEqual(KreaConfig{}, KreaConfig.encoderFrom(&w, testing.allocator, ""));
}

test "aneBlockEligible: fp16 plane grid, quantized only, never under a LoRA" {
    try testing.expect(aneBlockEligible(4608, true, 0)); // 512 text + 4096 img
    // A LoRA-attached block stays on the GPU: the int8 ANE copy was
    // quantized before the adapter existed, so an offload drops it silently.
    try testing.expect(!aneBlockEligible(4608, true, 1));
    try testing.expect(!aneBlockEligible(4608, false, 0)); // bf16 pack
    try testing.expect(!aneBlockEligible(4600, true, 0)); // off the 32-row pitch
    try testing.expect(!aneBlockEligible(128, true, 0)); // under the engagement floor
}

test "RGBA tensor PNG roundtrip preserves transparent and half-transparent pixels" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const planes = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5, 1 };
    const shape = [_]c_int{ 1, 4, 1, 3 };
    const img = mlx.mlx_array_new_data(&planes, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    const bytes = try imageToPng(a, img, s);
    defer a.free(bytes);
    try testing.expectEqual(@as(u8, 6), bytes[25]); // PNG RGBA color type.
    const stb = @import("stb");
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;
    const decoded = stb.stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &width, &height, &channels, 4);
    try testing.expect(decoded != null);
    defer stb.stbi_image_free(decoded);
    try testing.expectEqual(@as(c_int, 4), channels);
    try testing.expectEqual(@as(c_int, 3), width);
    try testing.expectEqual(@as(c_int, 1), height);
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0, 0, 0, 255, 0, 127, 0, 0, 255, 255 }, decoded[0..12]);
}

test "RGB tensor PNG roundtrip keeps the default three-channel output" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const planes = [_]f32{ 1, 0, 0, 1, 0.5, 0 };
    const shape = [_]c_int{ 1, 3, 1, 2 };
    const img = mlx.mlx_array_new_data(&planes, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(img);
    const bytes = try imageToPng(a, img, s);
    defer a.free(bytes);
    try testing.expectEqual(@as(u8, 2), bytes[25]);
    const expected = try png.encodeRgb(a, &.{ 255, 0, 127, 0, 255, 0 }, 2, 1);
    defer a.free(expected);
    try testing.expectEqualSlices(u8, expected, bytes);
}

test "keptRows keeps the valid text rows in order, none for an all-pad prompt" {
    const a = testing.allocator;
    const k = try keptRows(a, &.{ 1, 1, 0, 1, 0, 0 });
    defer a.free(k);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 3 }, k);
    const none = try keptRows(a, &.{ 0, 0 });
    defer a.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);
}
